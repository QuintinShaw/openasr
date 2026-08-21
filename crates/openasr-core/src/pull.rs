use std::{
    collections::{BTreeSet, VecDeque},
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::models::pack_verifier::{
    AdmittedPack, PackCandidate, PackRoute, PackVerificationError, PackVerifier,
};
use crate::{
    CatalogBackendFile, CatalogBackendFileRole, CatalogBackendVendor, CatalogModel,
    CatalogPullRequest, CatalogQuant, ModelCatalog, OPENASR_RUNTIME_PACK_EXTENSION,
    ResolvedCatalogBackendPull, ResolvedCatalogPull, atomic_file, canonical_quant_tag,
    catalog_series::family_aliases_match,
    content_store,
    download_source::{self, DownloadSource},
    has_openasr_runtime_pack_extension, http, parse_model_ref, resolve_catalog_pull,
    safety::{validate_safe_relative_path, validate_sha256},
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
/// Window length for the per-segment low-speed guard's rolling check (see
/// `SegmentLowSpeedWindow`). Shorter than the whole-file single-stream guard
/// (60s): abandoning a segment only discards that segment's in-flight bytes,
/// not the whole download, so it can afford to reevaluate more often.
const SEGMENT_LOW_SPEED_TIMEOUT: Duration = Duration::from_secs(15);
/// Outlier threshold for the per-segment low-speed guard: a segment is only
/// a candidate for abandonment once a window reads under this fraction of
/// the download session's own current reference throughput (see
/// `SegmentThroughputReference`). This is a *relative* judgment, not a fixed
/// floor -- a session where every connection tops out around, say, 200 KB/s
/// (a real, working, if modest network) must never trip this guard just
/// because 200 KB/s is a small number in absolute terms: no segment in that
/// session is an outlier relative to the others, so the ratio test alone
/// already never fires. Picked from a defensible "meaningfully behind its
/// siblings" range (roughly a seventh to a tenth of the reference) and
/// biased toward the lenient end to keep false positives rare.
const SEGMENT_LOW_SPEED_RELATIVE_RATIO: f64 = 0.15;
/// Second half of the low-speed AND, in absolute terms: bytes expected
/// within one `SEGMENT_LOW_SPEED_TIMEOUT` window, roughly a 273 KB/s floor.
/// Without this, the relative test alone could abandon a segment that's
/// merely somewhat slower than an *unusually fast* reference (say 300-400
/// KB/s in a session whose other segments hit several MB/s) even though
/// that speed is still a perfectly normal, working connection -- just not
/// this session's best. Requiring both conditions means only a segment that
/// is genuinely slow in real terms *and* a clear outlier among its own
/// siblings gets abandoned; this is what actually catches the reported
/// failure mode (a lone tail segment at ~90 KB/s while the rest of the
/// download ran at several MB/s) without ever penalizing a uniformly slow
/// session.
const SEGMENT_LOW_SPEED_ABSOLUTE_FLOOR_BYTES: u64 = 4 * 1024 * 1024;
/// Cooldown after a segment is abandoned for low speed, before it's eligible
/// to be judged low-speed again: twice the window length, so a freshly
/// reconnected attempt gets at least two full observation windows before
/// being re-evaluated. Without this, a segment sitting right at the ratio
/// boundary could thrash -- reconnect, immediately look slow again next
/// window, reconnect again -- burning through `SEGMENT_MAX_RETRIES` on
/// connection churn instead of giving each fresh connection a fair chance.
const SEGMENT_LOW_SPEED_COOLDOWN: Duration = Duration::from_secs(30);
/// Minimum recorded windows before the session reference is trusted enough
/// to judge outliers against. Below this, a single (possibly unlucky) early
/// sample would effectively become "the reference", which can never
/// correctly identify an outlier -- it would just compare a segment against
/// itself. `SegmentThroughputReference::median` returns `None` (never judge
/// low-speed yet) until this many samples exist, which is also what keeps a
/// download's very first segments from ever being penalized cold.
const SEGMENT_LOW_SPEED_MIN_REFERENCE_SAMPLES: usize = 3;
/// Bounds how many recent per-window byte counts the session reference
/// keeps: large enough for a stable median, small enough that the reference
/// tracks *current* conditions on a long, many-segment download rather than
/// staying anchored to however the first few segments happened to perform.
const SEGMENT_LOW_SPEED_REFERENCE_CAPACITY: usize = 64;
/// Discriminator stamped into the segmented-download partial-meta file so a
/// resume never misreads a legacy (pre-chunking) `PartialMeta` -- or a future
/// incompatible format -- as a valid segment bitmap. Bumping the segment size
/// or the bitmap's shape must also bump this string.
const PARALLEL_META_FORMAT: &str = "segmented-v1";
const BACKEND_STORE_SCHEMA_VERSION: u32 = 1;
const DEFAULT_BACKEND_GC_MIN_AGE: Duration = Duration::from_secs(7 * 24 * 60 * 60);

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
    #[error("Unexpected file in installed backend pack '{path}'")]
    UnexpectedInstalledBackendFile { path: PathBuf },
    #[error("Downloaded pack failed runtime path validation for '{path}': {reason}")]
    RuntimeValidation { path: PathBuf, reason: String },
    #[error("Installed model pack not found: {reference}")]
    NotInstalled { reference: String },
    #[error("Model pack pull was canceled: {reference}")]
    Canceled { reference: String },
    #[error("Model pack pull was paused: {reference}")]
    Paused { reference: String },
    #[error(transparent)]
    ContentStore(#[from] content_store::ContentStoreError),
    #[error(transparent)]
    Config(#[from] crate::config::ConfigError),
}

#[derive(Clone, Debug)]
struct PullTarget {
    model_id: String,
    /// Present only for targets resolved from a signed catalog. Explicit
    /// legacy-store migration has no catalog assertion to invent; it trusts
    /// the route already proven from its admitted bytes.
    expected_catalog_family_id: Option<String>,
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
    /// Relative per-segment low-speed guard for the concurrent
    /// chunked-download path; see `SEGMENT_LOW_SPEED_TIMEOUT` and friends for
    /// the production defaults and the rationale for judging a segment
    /// against this download session's own throughput rather than a fixed
    /// floor. All overridable (like the whole-file pair above) so tests can
    /// force a deterministic trip without waiting out a real window.
    segment_low_speed_timeout: Duration,
    segment_low_speed_relative_ratio: f64,
    segment_low_speed_absolute_floor_bytes: u64,
    segment_low_speed_cooldown: Duration,
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
            segment_low_speed_timeout: SEGMENT_LOW_SPEED_TIMEOUT,
            segment_low_speed_relative_ratio: SEGMENT_LOW_SPEED_RELATIVE_RATIO,
            segment_low_speed_absolute_floor_bytes: SEGMENT_LOW_SPEED_ABSOLUTE_FLOOR_BYTES,
            segment_low_speed_cooldown: SEGMENT_LOW_SPEED_COOLDOWN,
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

#[derive(Debug)]
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
    execution_services: Option<&'a crate::NativeExecutionServices>,
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
            execution_services: None,
            should_cancel: None,
            should_pause: None,
        }
    }

    /// Attach the execution-service root whose resident runtime caches should
    /// be reclaimed when this install replaces an already-loaded pack.
    ///
    /// Standalone pack-management tools that cannot have resident native
    /// runtimes may omit this. Long-lived hosts must pass the same service
    /// root they use for offline and streaming execution.
    pub fn execution_services(
        mut self,
        execution_services: &'a crate::NativeExecutionServices,
    ) -> Self {
        self.execution_services = Some(execution_services);
        self
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
            execution_services,
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
            execution_services,
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
    install_model_pack_from_path_with_execution_services(
        resolved,
        source_path,
        home,
        None,
        progress,
    )
}

/// Installs a local pack and reclaims any superseded resident runtime from the
/// explicitly supplied execution-service root.
pub fn install_model_pack_from_path_with_execution_services(
    resolved: &ResolvedCatalogPull,
    source_path: impl AsRef<Path>,
    home: impl AsRef<Path>,
    execution_services: Option<&crate::NativeExecutionServices>,
    progress: impl FnMut(PullProgress),
) -> Result<InstalledPack, PullError> {
    let target = PullTarget::from_resolved(resolved)?.with_source("local");
    install_model_pack_from_path_with_target(
        &target,
        source_path,
        home,
        execution_services,
        progress,
    )
}

pub fn install_catalog_model_pack_from_path(
    catalog: &ModelCatalog,
    source_path: impl AsRef<Path>,
    home: impl AsRef<Path>,
    progress: impl FnMut(PullProgress),
) -> Result<InstalledPack, PullError> {
    install_catalog_model_pack_from_path_with_execution_services(
        catalog,
        source_path,
        home,
        None,
        progress,
    )
}

/// Catalog-verified local install with explicit resident-runtime reclamation.
pub fn install_catalog_model_pack_from_path_with_execution_services(
    catalog: &ModelCatalog,
    source_path: impl AsRef<Path>,
    home: impl AsRef<Path>,
    execution_services: Option<&crate::NativeExecutionServices>,
    progress: impl FnMut(PullProgress),
) -> Result<InstalledPack, PullError> {
    let source_path = source_path.as_ref();
    let resolved = resolve_catalog_model_pack_from_path(catalog, source_path)?;
    install_model_pack_from_path_with_execution_services(
        &resolved,
        source_path,
        home,
        execution_services,
        progress,
    )
}

/// Resolve a local `.oasr` pack to the immutable signed-catalog entry whose
/// size and digest it matches, without installing it.
///
/// Installation frontends use this preflight to obtain the catalog-owned
/// license metadata before asking the shared install-license policy for an
/// admission decision. [`install_model_pack_from_path`] verifies the bytes
/// again while admitting them to the content store, so replacing the source
/// path between this preflight and installation cannot install different
/// content under the admitted identity.
pub fn resolve_catalog_model_pack_from_path(
    catalog: &ModelCatalog,
    source_path: impl AsRef<Path>,
) -> Result<ResolvedCatalogPull, PullError> {
    let source_path = source_path.as_ref();
    if !has_openasr_runtime_pack_extension(source_path) {
        return Err(PullError::InvalidTarget {
            field: "path",
            reason: format!("local imports must use .{OPENASR_RUNTIME_PACK_EXTENSION} model packs"),
        });
    }
    // Fail-closed admission control: a local pack is only installable when its
    // sha256/size matches exactly one quant of the signed public catalog. The
    // digest computed here is not trusted as the final authority -- the content
    // store re-hashes the bytes it actually copies and
    // `install_admitted_model_pack` rejects any drift against this target.
    let (size_bytes, sha256) = file_size_and_sha256(source_path)?;
    resolve_catalog_pull_by_file_digest(catalog, size_bytes, &sha256)
}

fn admit_model_content(
    source_path: &Path,
    home: &Path,
    expected_catalog_family_id: Option<&str>,
) -> Result<AdmittedPack, PullError> {
    admit_model_content_into_root(source_path, &models_root(home), expected_catalog_family_id)
}

fn admit_model_content_into_root(
    source_path: &Path,
    root: &Path,
    expected_catalog_family_id: Option<&str>,
) -> Result<AdmittedPack, PullError> {
    ensure_safe_directory_under_root(root, root)?;
    ensure_safe_directory_under_root(root, &root.join("staging"))?;
    ensure_safe_directory_under_root(root, &content_store::objects_root(root))?;
    let content = content_store::admit_file(source_path, root, |lease| {
        let verified = PackVerifier
            .verify_admission_lease(lease)
            .map_err(pack_verification_to_pull_error)?;
        ensure_catalog_family_matches(
            expected_catalog_family_id,
            verified.catalog_family_id(),
            lease.path(),
        )?;
        Ok(verified)
    })
    .map_err(|error| match error {
        content_store::ContentStoreError::Preflight(error) => *error,
        other => PullError::ContentStore(other),
    })?;
    AdmittedPack::from_content(content).map_err(|reason| PullError::RuntimeValidation {
        path: source_path.to_path_buf(),
        reason,
    })
}

fn install_model_pack_from_path_with_target(
    target: &PullTarget,
    source_path: impl AsRef<Path>,
    home: impl AsRef<Path>,
    execution_services: Option<&crate::NativeExecutionServices>,
    progress: impl FnMut(PullProgress),
) -> Result<InstalledPack, PullError> {
    // Reject an unusable target before copying potentially gigabytes of pack.
    let paths = pull_paths(home.as_ref(), target)?;
    ensure_storage_dir_within_root(home.as_ref(), &paths)?;
    let source_path = source_path.as_ref();
    if !has_openasr_runtime_pack_extension(source_path) {
        return Err(PullError::InvalidTarget {
            field: "from",
            reason: format!("local imports must use .{OPENASR_RUNTIME_PACK_EXTENSION} model packs"),
        });
    }

    // Admission needs no exclusion: the staging name is process-unique and an
    // object is keyed by its own digest. The pull lock is taken once, by
    // `install_admitted_model_pack`, around publishing the ref.
    let admitted = admit_model_content(
        source_path,
        home.as_ref(),
        target.expected_catalog_family_id.as_deref(),
    )?;
    install_admitted_model_pack(
        target,
        home.as_ref(),
        admitted,
        execution_services,
        progress,
    )
}

fn install_admitted_model_pack(
    target: &PullTarget,
    home: &Path,
    admitted: AdmittedPack,
    execution_services: Option<&crate::NativeExecutionServices>,
    mut progress: impl FnMut(PullProgress),
) -> Result<InstalledPack, PullError> {
    if admitted.size_bytes() != target.size_bytes {
        return Err(PullError::SizeMismatch {
            path: admitted.object_path().to_path_buf(),
            expected: target.size_bytes,
            actual: admitted.size_bytes(),
        });
    }
    if admitted.digest() != target.sha256 {
        return Err(PullError::ShaMismatch {
            path: admitted.object_path().to_path_buf(),
            expected: target.sha256.clone(),
            actual: admitted.digest().to_string(),
        });
    }
    ensure_catalog_family_matches_target(
        target,
        admitted.catalog_family_id(),
        admitted.object_path(),
    )?;
    let paths = pull_paths(home, target)?;
    ensure_storage_dir_within_root(home, &paths)?;
    let _lock = PullLock::acquire(&paths.lock_path)?;
    // The ref about to be rewritten may still name an older object whose
    // runtime state is resident. Resolve that identity before the new ref is
    // visible, and evict it after -- a memory-reclaim step only: the new
    // object's content id misses every content-addressed cache on its own.
    let previous_pack_content_id = existing_pack_content_id_for_eviction(&paths);
    // Keep the descriptor that was hashed and preflighted alive until its
    // durable logical reference is visible; do not switch validation authority
    // back to a display path during commit.
    let (_validated_lease, _verified_pack, _, _, _) = admitted.into_parts();
    let pack = write_installed_record(target, &paths)?;
    if let Some(old_content_id) = previous_pack_content_id {
        evict_resident_runtime_caches_for_content_id(execution_services, &old_content_id);
    }
    progress(PullProgress::Installed {
        path: pack.path.clone(),
    });
    Ok(pack)
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
    // `InstalledModelStore` reads exactly one layout and never writes. Converting
    // a legacy store is the separate, explicit `migrate_legacy_model_store`
    // operation, which startup runs once -- a read path must not move gigabytes
    // as a side effect, and an I/O failure there must not be able to empty a
    // listing that would otherwise have succeeded.
    crate::InstalledModelStore::read(home.as_ref()).map(crate::InstalledModelStore::into_packs)
}

/// Bring the model store up to date once per process start.
///
/// Callers own reporting: this returns the report rather than logging, so the
/// CLI, the server, and the desktop sidecar each surface migration failures in
/// their own voice. A failed record keeps its legacy bytes on disk untouched, so
/// the worst case is a pack that is temporarily not listed and loudly reported
/// -- never a pack that is gone.
pub fn migrate_model_store_at_startup(
    home: impl AsRef<Path>,
) -> Result<LegacyMigrationReport, PullError> {
    migrate_legacy_model_store(home.as_ref())
}

/// One legacy record that could not be converted. The bytes behind it are
/// always left untouched, so a failure costs visibility, never data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyMigrationFailure {
    pub path: PathBuf,
    pub reason: String,
}

/// Outcome of one [`migrate_legacy_model_store`] pass.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LegacyMigrationReport {
    /// `<id>:<quant>` references now served from the content store.
    pub migrated: Vec<String>,
    /// Bytes released by dropping legacy copies whose content is already an
    /// object. This is the duplicate-copy leak the old converter left behind.
    pub reclaimed_bytes: u64,
    pub failures: Vec<LegacyMigrationFailure>,
}

impl LegacyMigrationReport {
    pub fn is_empty(&self) -> bool {
        self.migrated.is_empty() && self.reclaimed_bytes == 0 && self.failures.is_empty()
    }
}

/// Convert every `<models>/<model>/<quant>/installed.json` record into the
/// content-addressed layout, in place, and drop the legacy copy afterwards.
///
/// Runs once per process start ([`crate::migrate_model_store_at_startup`]) and
/// is idempotent: on a converted store it is a directory scan that finds
/// nothing. Three properties make it safe to run unattended:
///
/// * **In place.** Every path derives from [`models_root`], so a user who
///   redirected storage with `OPENASR_MODELS_DIR` or `config.models_dir` gets
///   the conversion inside *their* directory. Nothing is ever relocated.
/// * **Content before ref.** Bytes become a durable object before any ref names
///   them, and the legacy record is revoked only after the ref is durable. A
///   crash at any point leaves at worst an unreferenced object (collectable)
///   and never a ref pointing at nothing.
/// * **Move, do not copy.** Landing is a rename, so within one filesystem no
///   pack bytes are copied; only the verification pass reads them. A models
///   root that spans devices falls back to a copying admission.
pub fn migrate_legacy_model_store(home: &Path) -> Result<LegacyMigrationReport, PullError> {
    let root = models_root(home);
    let mut report = LegacyMigrationReport::default();
    let Ok(model_dirs) = fs::read_dir(&root) else {
        return Ok(report);
    };
    for model_dir in model_dirs {
        let model_dir = model_dir.map_err(|source| PullError::Io {
            path: root.clone(),
            source,
        })?;
        if matches!(
            model_dir.file_name().to_str(),
            Some("objects" | "refs" | "staging" | "locks")
        ) {
            continue;
        }
        let Ok(quant_dirs) = fs::read_dir(model_dir.path()) else {
            continue;
        };
        for quant_dir in quant_dirs {
            let quant_dir = quant_dir.map_err(|source| PullError::Io {
                path: model_dir.path(),
                source,
            })?;
            let quant_path = quant_dir.path();
            let metadata_path = quant_path.join("installed.json");
            let Ok(contents) = fs::read_to_string(&metadata_path) else {
                continue;
            };
            let Ok(legacy) = serde_json::from_str::<InstalledPack>(&contents) else {
                continue;
            };
            if let Err(reason) =
                crate::installed_model_store::validate_legacy_record(&legacy, &quant_path)
            {
                report.failures.push(LegacyMigrationFailure {
                    path: metadata_path,
                    reason,
                });
                continue;
            }
            match migrate_one_legacy_record(home, &root, &legacy, &quant_path) {
                Ok(outcome) => {
                    report.reclaimed_bytes += outcome.reclaimed_bytes;
                    if outcome.published_ref {
                        report.migrated.push(legacy.pull.clone());
                    }
                }
                Err(error) => report.failures.push(LegacyMigrationFailure {
                    path: metadata_path,
                    reason: error.to_string(),
                }),
            }
        }
    }
    report.migrated.sort();
    Ok(report)
}

struct LegacyRecordOutcome {
    published_ref: bool,
    reclaimed_bytes: u64,
}

fn migrate_one_legacy_record(
    home: &Path,
    root: &Path,
    legacy: &InstalledPack,
    quant_path: &Path,
) -> Result<LegacyRecordOutcome, PullError> {
    let ref_path = root
        .join("refs")
        .join(&legacy.model_id)
        .join(format!("{}.json", legacy.quant));
    // A durable ref is the migration commit point. If a previous run died after
    // publishing it, the legacy tree is pure duplication: finish the cleanup
    // rather than re-admitting mutable legacy bytes. Leaving it is exactly the
    // leak that made a converted store carry two copies of every pack.
    if ref_path.is_file() {
        return Ok(LegacyRecordOutcome {
            published_ref: false,
            reclaimed_bytes: discard_legacy_quant_dir(root, quant_path)?,
        });
    }

    // Legacy migration uses the same lease-based admission seam as every new
    // install. The old zero-copy branch preflighted one path generation, then
    // hashed and renamed that pathname later; a replacement in between could
    // therefore put runtime-invalid bytes under a trusted digest. Admission
    // copies, hashes and verifies one held descriptor generation before any
    // object becomes visible. Migration is a one-time cold operation, so that
    // correctness boundary is worth the copy and removes the special writer.
    let admitted = admit_model_content(&legacy.path, home, None)?;
    let size_bytes = admitted.size_bytes();
    let digest = admitted.digest().to_string();

    let target = PullTarget {
        model_id: legacy.model_id.clone(),
        expected_catalog_family_id: None,
        display_name: legacy.display_name.clone(),
        quant: legacy.quant.clone(),
        suffix: legacy.suffix.clone(),
        pull: legacy.pull.clone(),
        filename: legacy.filename.clone(),
        url: legacy.url.clone(),
        hf_revision: legacy.hf_revision.clone(),
        sha256: digest.clone(),
        size_bytes,
        source: legacy.source.clone(),
    };
    let paths = pull_paths(home, &target)?;
    ensure_storage_dir_within_root(home, &paths)?;
    ensure_safe_directory_under_root(root, &content_store::objects_root(root))?;
    let _lock = PullLock::acquire(&paths.lock_path)?;
    // Re-check under the lock: a concurrent install of the same variant may have
    // published the ref while this record was being hashed.
    if ref_path.is_file() {
        return Ok(LegacyRecordOutcome {
            published_ref: false,
            reclaimed_bytes: discard_legacy_quant_dir(root, quant_path)?,
        });
    }

    let mut reclaimed_bytes = remove_file_reporting_size(&legacy.path)?;

    // Content is durable before the ref that names it exists.
    write_installed_record(&target, &paths)?;
    // Keep the exact verified admission lease alive until the durable logical
    // reference is visible. No later path-based validation is substituted.
    let (_validated_lease, _verified_pack, _, _, _) = admitted.into_parts();
    reclaimed_bytes += discard_legacy_quant_dir(root, quant_path)?;
    Ok(LegacyRecordOutcome {
        published_ref: true,
        reclaimed_bytes,
    })
}

fn remove_file_reporting_size(path: &Path) -> Result<u64, PullError> {
    let size = fs::symlink_metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    match fs::remove_file(path) {
        Ok(()) => {
            atomic_file::sync_parent_dir_best_effort(path);
            Ok(size)
        }
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(0),
        Err(source) => Err(PullError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

/// Drop a fully superseded `<models>/<model>/<quant>/` tree and report the bytes
/// it was holding. Only reached once the content-addressed ref for the same
/// variant is durable, so nothing here is the last copy of anything.
fn discard_legacy_quant_dir(root: &Path, quant_path: &Path) -> Result<u64, PullError> {
    ensure_safe_directory_under_root(root, quant_path)?;
    let mut reclaimed = 0;
    if let Ok(entries) = fs::read_dir(quant_path) {
        for entry in entries.flatten() {
            reclaimed += entry
                .metadata()
                .map(|metadata| {
                    if metadata.is_file() {
                        metadata.len()
                    } else {
                        0
                    }
                })
                .unwrap_or(0);
        }
    }
    match fs::remove_dir_all(quant_path) {
        Ok(()) => atomic_file::sync_parent_dir_best_effort(quant_path),
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(source) => {
            return Err(PullError::Io {
                path: quant_path.to_path_buf(),
                source,
            });
        }
    }
    if let Some(model_dir) = quant_path.parent() {
        // Only ever removes an already-empty directory, so a sibling quant that
        // has not migrated yet is never touched.
        let _ = fs::remove_dir(model_dir);
    }
    Ok(reclaimed)
}

pub fn default_pack_pointer_path(home: impl AsRef<Path>) -> PathBuf {
    home.as_ref().join("default.json")
}

fn content_addressed_ref_path(home: &Path, pack: &InstalledPack) -> Option<PathBuf> {
    let root = models_root(home);
    let object_path = root
        .join("objects")
        .join("sha256")
        .join(&pack.sha256)
        .join("content");
    if pack.path != object_path {
        return None;
    }
    let ref_path = root
        .join("refs")
        .join(&pack.model_id)
        .join(format!("{}.json", pack.quant));
    ref_path.is_file().then_some(ref_path)
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

pub fn remove_model_pack(
    home: impl AsRef<Path>,
    reference: &str,
) -> Result<Option<InstalledPack>, PullError> {
    remove_model_pack_with_execution_services(home, reference, None)
}

/// Removes a pack and reclaims its resident runtime from the explicitly
/// supplied execution-service root.
pub fn remove_model_pack_with_execution_services(
    home: impl AsRef<Path>,
    reference: &str,
    execution_services: Option<&crate::NativeExecutionServices>,
) -> Result<Option<InstalledPack>, PullError> {
    let home = home.as_ref();
    let Some(pack) = find_installed_pack(home, reference)? else {
        return Ok(None);
    };
    if let Some(ref_path) = content_addressed_ref_path(home, &pack) {
        // Content objects are immutable and may be referenced by another
        // model/quant. Removing a model deletes its ref first, and only then
        // collects the object once no surviving ref names the same digest.
        let root = models_root(home);
        let ref_parent = ref_path.parent().expect("ref has parent");
        ensure_safe_directory_under_root(&root, ref_parent)?;
        reject_symlink(&ref_path)?;
        match fs::remove_file(&ref_path) {
            Ok(()) => {}
            Err(source) if source.kind() == io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(PullError::Io {
                    path: ref_path,
                    source,
                });
            }
        }
        atomic_file::sync_parent_dir_best_effort(&ref_path);
        let _ = fs::remove_dir(ref_parent);
        // Installs still create the legacy `<models>/<model>/<quant>/` skeleton;
        // prune it here while it is empty so uninstall leaves nothing behind.
        let legacy_quant_dir = root.join(&pack.model_id).join(&pack.quant);
        let _ = fs::remove_dir(&legacy_quant_dir);
        let _ = fs::remove_dir(legacy_quant_dir.parent().expect("legacy quant has parent"));
        let still_referenced = list_installed_packs_without_gc(home)?
            .iter()
            .any(|candidate| candidate.sha256 == pack.sha256);
        content_store::remove_object_if_unreferenced(&root, &pack.sha256, still_referenced)?;
        evict_resident_runtime_caches_for_content_id(
            execution_services,
            &crate::models::runtime_cache_coordinator::content_id_from_sha256_hex(&pack.sha256),
        );
        return Ok(Some(pack));
    }
    // Legacy per-quant layout: the pack file lives inside its own directory, so
    // the whole directory is the unit of removal.
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
    evict_resident_runtime_caches_for_content_id(
        execution_services,
        &crate::models::runtime_cache_coordinator::content_id_from_sha256_hex(&pack.sha256),
    );
    Ok(Some(pack))
}

/// Packs still visible on disk, read without the legacy migration that
/// `list_installed_packs` performs. Object collection must never re-admit bytes
/// as a side effect of counting the refs that survive a removal.
fn list_installed_packs_without_gc(home: &Path) -> Result<Vec<InstalledPack>, PullError> {
    crate::InstalledModelStore::read(home).map(crate::InstalledModelStore::into_packs)
}

/// Open an installed pack for use.
///
/// This is the hot path -- every model load and every desktop model switch comes
/// through here -- so it does not re-hash the object. The digest was established
/// when the pack was admitted and the object has been read-only since; see
/// `content_store`'s integrity chain. Use `verify_model_store` to re-check
/// digests on demand.
pub fn open_installed_content_lease(
    home: impl AsRef<Path>,
    reference: &str,
) -> Result<Option<crate::ContentLease>, PullError> {
    let home = home.as_ref();
    let Some(pack) = find_installed_pack(home, reference)? else {
        return Ok(None);
    };
    Ok(Some(content_store::open_declared_lease(
        &models_root(home),
        &pack.sha256,
        pack.size_bytes,
    )?))
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
        None,
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
    execution_services: Option<&crate::NativeExecutionServices>,
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
                execution_services,
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
#[derive(Debug)]
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

    // Shared low-speed guard state for this attempt, used by both the probe
    // (below) and every worker spawned later: one session-wide throughput
    // reference (see `SegmentThroughputReference`), one low-speed abandon
    // counter per segment index (a segment can be requeued and picked up by
    // a *different* worker than the one that abandoned it, so this can't
    // live in any single worker's local state), and one cooldown timestamp
    // per segment index (see `SEGMENT_LOW_SPEED_COOLDOWN`).
    let throughput_reference = Arc::new(SegmentThroughputReference::new());
    let low_speed_abandon_counts: Arc<Vec<AtomicUsize>> =
        Arc::new((0..total_segments).map(|_| AtomicUsize::new(0)).collect());
    let low_speed_cooldowns: Arc<Vec<Mutex<Option<Instant>>>> =
        Arc::new((0..total_segments).map(|_| Mutex::new(None)).collect());

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
        // The probe segment downloads synchronously on this orchestrating
        // thread, before any worker thread exists to poll the controls, so it
        // must honor cancel/pause itself -- otherwise a cancel issued while the
        // probe is in flight would not take effect until the whole (up to
        // 64 MiB) probe segment finished, stranding the pull in `Downloading`
        // for seconds. `write_segment_body` stops early on the predicate above,
        // leaving a short segment, so these checks must come before the
        // size-mismatch check below (an intentional stop is not a mismatch).
        //
        // The probe also carries its own low-speed guard: it runs before any
        // worker exists and can just as easily land on a degraded connection
        // as a worker-fetched segment can. Unlike the worker path there's no
        // shared queue to requeue into here (there's exactly one prober), so
        // a low-speed trip just re-opens a fresh request for the same range
        // and retries in place, bounded by the same `SEGMENT_MAX_RETRIES` cap
        // every other segment failure mode uses -- past that cap, evaluation
        // is disabled and the probe simply finishes on its current
        // connection (see `SegmentLowSpeedWindow::disabled`), never a hard
        // failure.
        let mut probe_reader = probe_response.reader;
        let mut probe_abandon_count = 0_usize;
        let probe_cooldown = &low_speed_cooldowns[probe_index];
        let written = loop {
            let disabled = probe_abandon_count >= SEGMENT_MAX_RETRIES;
            let mut low_speed = SegmentLowSpeedWindow::new(
                options,
                &throughput_reference,
                probe_cooldown,
                disabled,
            );
            let outcome = write_segment_body(
                &mut file,
                &paths.partial_path,
                probe_start,
                probe_end,
                probe_reader,
                |delta| {
                    bytes_done = bytes_done.saturating_add(delta);
                    progress(PullProgress::Downloading {
                        bytes_done,
                        bytes_total: target.size_bytes,
                    });
                },
                &|| should_cancel() || should_pause(),
                &mut low_speed,
            )?;
            match outcome {
                SegmentWriteOutcome::Completed(written) => break written,
                SegmentWriteOutcome::AbortedByControl => {
                    if should_cancel() {
                        cleanup_partial(paths);
                        return Err(PullError::Canceled {
                            reference: target.pull.clone(),
                        });
                    }
                    // Keep the partial file and segment bitmap (segment not
                    // marked done) so a later resume re-probes and refetches
                    // this segment cleanly, exactly like a pause caught by
                    // the worker loop below.
                    return Err(PullError::Paused {
                        reference: target.pull.clone(),
                    });
                }
                SegmentWriteOutcome::LowSpeed(partial_written) => {
                    // Roll back the progress already reported for this
                    // abandoned attempt so bytes_done never double-counts
                    // once the retry below re-downloads the same range from
                    // scratch.
                    bytes_done = bytes_done.saturating_sub(partial_written);
                    progress(PullProgress::Downloading {
                        bytes_done,
                        bytes_total: target.size_bytes,
                    });
                    probe_abandon_count += 1;
                    if probe_abandon_count == SEGMENT_MAX_RETRIES {
                        eprintln!(
                            "openasr: warning: probe segment [{probe_start}-{probe_end}] for \
                             '{}' stayed a relative low-speed outlier after {probe_abandon_count} \
                             reconnect attempts; accepting it on its current connection",
                            target.url
                        );
                    }
                    let retry_response = probe_client.open(
                        &target.url,
                        Some(ByteRange::bounded(probe_start, probe_end)),
                    )?;
                    if retry_response.status != 206 {
                        return Err(PullError::UnexpectedStatus {
                            url: target.url.clone(),
                            status: retry_response.status,
                        });
                    }
                    probe_reader = retry_response.reader;
                }
            }
        };
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
        let worker_options = options.clone();
        let worker_low_speed_abandon_counts = low_speed_abandon_counts.clone();
        let worker_throughput_reference = throughput_reference.clone();
        let worker_low_speed_cooldowns = low_speed_cooldowns.clone();
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
                worker_options,
                worker_low_speed_abandon_counts,
                worker_throughput_reference,
                worker_low_speed_cooldowns,
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
            Ok(SegmentEvent::ProgressRollback(delta)) => {
                // A worker abandoned a low-speed attempt after already
                // reporting some of its bytes via `Progress` above; undo
                // that now so the retry (which re-downloads the same range
                // from scratch) doesn't double-count them.
                bytes_done = bytes_done.saturating_sub(delta);
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

/// How [`write_segment_body`] stopped before reaching `expected_len`. The
/// `Completed` case is the only one where the caller may mark the segment
/// done; `LowSpeed` carries the partial byte count so the caller can roll
/// back whatever progress it already reported for this (now-discarded)
/// attempt before retrying. `AbortedByControl` needs no such rollback: a
/// cancel discards the whole partial download and a pause preserves it
/// on-disk exactly as-is (no retry follows), so there's nothing to correct.
enum SegmentWriteOutcome {
    /// Read exactly `expected_len` bytes (the size check against
    /// `end_inclusive - start + 1` still happens at the call site, matching
    /// the pre-existing contract).
    Completed(u64),
    /// `should_abort` tripped (pause/cancel): stop silently, same as before.
    AbortedByControl,
    /// The per-segment low-speed window tripped: this attempt's connection
    /// is judged too slow to keep riding out; the caller should abandon it
    /// and get a fresh connection rather than treat this as a hard error.
    LowSpeed(u64),
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
/// both paths write segments identically and stop identically. Also shared:
/// the per-segment low-speed guard (`low_speed`), so a lone tail segment
/// stuck on a degraded connection is caught the same way regardless of which
/// of the two call sites is currently downloading it (see
/// `SEGMENT_LOW_SPEED_TIMEOUT`).
fn write_segment_body(
    file: &mut File,
    path_for_errors: &Path,
    start: u64,
    end_inclusive: u64,
    reader: Box<dyn Read>,
    mut on_progress: impl FnMut(u64),
    should_abort: &dyn Fn() -> bool,
    low_speed: &mut SegmentLowSpeedWindow,
) -> Result<SegmentWriteOutcome, PullError> {
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
            return Ok(SegmentWriteOutcome::AbortedByControl);
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
        if low_speed.observe(read as u64) {
            return Ok(SegmentWriteOutcome::LowSpeed(written));
        }
    }
    Ok(SegmentWriteOutcome::Completed(written))
}

/// Events a segment worker thread reports back to the orchestrating thread
/// over the `mpsc` channel. Kept intentionally minimal: only the
/// orchestrating thread touches `should_cancel`/`should_pause`, the segment
/// bitmap, and the `progress` callback, so workers never need anything more
/// than "here is a byte delta" / "undo this many previously-reported bytes"
/// / "this segment index is done" / "this segment failed".
enum SegmentEvent {
    Progress(u64),
    /// A worker abandoned a low-speed attempt after already reporting
    /// `Progress` for some of its bytes; the orchestrator subtracts this
    /// from `bytes_done` so the (from-scratch) retry doesn't double-count
    /// them. See `run_segment_worker`'s `SegmentFetchOutcome::LowSpeed` arm.
    ProgressRollback(u64),
    Done(usize),
    Failed(PullError),
}

/// Outcome of one segment fetch attempt (one `client.open` + body read).
/// Distinguished from a hard `Err` because a low-speed trip is not a fatal
/// condition -- it means "this connection is bad, get a new one" -- so it is
/// handled by `run_segment_worker` requeuing the segment instead of by
/// `fetch_segment_with_retries`'s same-connection retry loop.
enum SegmentFetchOutcome {
    Done,
    /// Aborted mid-segment by cancel/pause; no event needed, the
    /// orchestrator already knows.
    AbortedByControl,
    /// The per-segment low-speed window tripped; carries the bytes written
    /// (and already reported via `SegmentEvent::Progress`) so the caller can
    /// roll that progress back before abandoning the connection.
    LowSpeed(u64),
}

/// One worker thread's loop: pop segment indices off the shared `queue`
/// until it's empty or `abort` is set, fetching and writing each with
/// `fetch_segment_with_retries`. Never panics on I/O failure -- every error
/// path reports a `SegmentEvent::Failed` and returns instead.
///
/// A segment that trips the per-segment low-speed guard is not retried
/// in place: it's pushed back onto the tail of the shared `queue` so the
/// next `client.open()` call for it (by this worker or another, once its
/// current segment finishes) starts a genuinely fresh connection rather than
/// riding out the same degraded one. `low_speed_abandon_counts` bounds how
/// many times any single segment index can be abandoned this way; past
/// `SEGMENT_MAX_RETRIES`, the *next* attempt on that index is constructed
/// with its low-speed guard disabled (see `SegmentLowSpeedWindow::disabled`),
/// so it can no longer trip -- the worst case degrades to "just let it
/// finish on whatever connection it's on", never a hard failure.
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
    options: PullOptions,
    low_speed_abandon_counts: Arc<Vec<AtomicUsize>>,
    throughput_reference: Arc<SegmentThroughputReference>,
    low_speed_cooldowns: Arc<Vec<Mutex<Option<Instant>>>>,
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
        // SeqCst: read-then-later-write of this same counter happens across
        // worker threads, so ordering must be total, not just
        // per-worker-relative.
        let disabled =
            low_speed_abandon_counts[index].load(Ordering::SeqCst) >= SEGMENT_MAX_RETRIES;
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
            &options,
            &throughput_reference,
            &low_speed_cooldowns[index],
            disabled,
        ) {
            Ok(SegmentFetchOutcome::Done) => {
                if sender.send(SegmentEvent::Done(index)).is_err() {
                    return;
                }
            }
            Ok(SegmentFetchOutcome::AbortedByControl) => return,
            Ok(SegmentFetchOutcome::LowSpeed(partial_written)) => {
                if sender
                    .send(SegmentEvent::ProgressRollback(partial_written))
                    .is_err()
                {
                    return;
                }
                let attempts = low_speed_abandon_counts[index].fetch_add(1, Ordering::SeqCst) + 1;
                if attempts == SEGMENT_MAX_RETRIES {
                    eprintln!(
                        "openasr: warning: segment [{start}-{end}] for '{url}' stayed a \
                         relative low-speed outlier after {attempts} reconnect attempts; \
                         accepting it on its current connection"
                    );
                }
                // Back of the queue, not the front: give any segments still
                // untouched a chance first, so a persistently bad source
                // doesn't get to monopolize every worker retrying the same
                // handful of unlucky indices in a tight loop.
                queue.lock().unwrap().push_back(index);
            }
            Err(error) => {
                let _ = sender.send(SegmentEvent::Failed(error));
                return;
            }
        }
    }
}

/// Retry one segment fetch up to `SEGMENT_MAX_RETRIES` times, backing off
/// between attempts exactly like the single-stream path's outer retry loop.
/// This loop only ever retries a hard `Err` (I/O, unexpected status, ...) on
/// the *same* connection -- a low-speed trip is a `SegmentFetchOutcome::
/// LowSpeed`, not an `Err`, so it passes straight back to the caller
/// (`run_segment_worker`) untouched, which is what routes it to the
/// requeue-for-a-new-connection handling instead.
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
    options: &PullOptions,
    throughput_reference: &SegmentThroughputReference,
    cooldown_slot: &Mutex<Option<Instant>>,
    low_speed_disabled: bool,
) -> Result<SegmentFetchOutcome, PullError> {
    let mut attempt = 0_usize;
    loop {
        if abort.load(Ordering::SeqCst) {
            return Ok(SegmentFetchOutcome::AbortedByControl);
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
            options,
            throughput_reference,
            cooldown_slot,
            low_speed_disabled,
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
    options: &PullOptions,
    throughput_reference: &SegmentThroughputReference,
    cooldown_slot: &Mutex<Option<Instant>>,
    low_speed_disabled: bool,
) -> Result<SegmentFetchOutcome, PullError> {
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
    let mut low_speed = SegmentLowSpeedWindow::new(
        options,
        throughput_reference,
        cooldown_slot,
        low_speed_disabled,
    );
    let write_outcome = write_segment_body(
        file,
        path,
        start,
        end,
        response.reader,
        |delta| {
            let _ = sender.send(SegmentEvent::Progress(delta));
        },
        &|| abort.load(Ordering::SeqCst),
        &mut low_speed,
    )?;
    let written = match write_outcome {
        SegmentWriteOutcome::Completed(written) => written,
        SegmentWriteOutcome::AbortedByControl => return Ok(SegmentFetchOutcome::AbortedByControl),
        SegmentWriteOutcome::LowSpeed(partial_written) => {
            return Ok(SegmentFetchOutcome::LowSpeed(partial_written));
        }
    };
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
    Ok(SegmentFetchOutcome::Done)
}

fn verify_partial_and_install(
    target: &PullTarget,
    paths: &PullPaths,
    downloaded: Option<DownloadedPartial>,
    execution_services: Option<&crate::NativeExecutionServices>,
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
    cancel_before_commit(target, paths, should_cancel)?;
    // Resolve the pack this install supersedes (if any) *before* the reference
    // moves, so its content id is still hashable from the old bytes. The new
    // bytes land in a different immutable object and so resolve to a different
    // content id, missing every content-addressed runtime cache on their own --
    // no invalidation needed for that. This id is purely so the *old*, now
    // unreferenced identity's resident state can be evicted promptly after
    // install to release memory, rather than waiting for the next idle unload.
    let previous_pack_content_id = existing_pack_content_id_for_eviction(paths);
    // The verified staging file is copied into an immutable content-addressed
    // object before the logical reference becomes visible. Existing objects are
    // never replaced, which removes the Windows same-path mmap failure mode.
    let admitted = match admit_model_content_into_root(
        &paths.partial_path,
        &models_root_for_paths(paths),
        target.expected_catalog_family_id.as_deref(),
    ) {
        Ok(admitted) => admitted,
        Err(error) => {
            cleanup_partial(paths);
            return Err(error);
        }
    };
    if admitted.digest() != target.sha256 || admitted.size_bytes() != target.size_bytes {
        return Err(PullError::ShaMismatch {
            path: admitted.object_path().to_path_buf(),
            expected: target.sha256.clone(),
            actual: admitted.digest().to_string(),
        });
    }
    ensure_catalog_family_matches_target(
        target,
        admitted.catalog_family_id(),
        admitted.object_path(),
    )?;
    let _ = fs::remove_file(&paths.partial_path);
    let _ = fs::remove_file(&paths.partial_meta_path);
    // A resume can switch from the chunked/parallel path (which persists
    // `partial_segments_meta_path`) to this single-stream success path once
    // the remaining bytes drop below the parallel-eligibility threshold; clean
    // it up here too so it cannot outlive the `.partial` file it describes.
    let _ = fs::remove_file(&paths.partial_segments_meta_path);
    let pack = write_installed_record(target, paths)?;
    if let Some(old_content_id) = previous_pack_content_id {
        evict_resident_runtime_caches_for_content_id(execution_services, &old_content_id);
    }
    progress(PullProgress::Installed {
        path: pack.path.clone(),
    });
    Ok(pack)
}

/// Resolves the content id of whatever pack currently sits at `paths.final_path`,
/// if any -- called before it is overwritten by an install/replace. Returns
/// `None` when there is nothing there yet (first install) or no identity can
/// be resolved (nothing meaningful to evict either way). `final_path` is
/// always a content-addressed object path *under this pull's own models
/// root*, so a sealed existing object answers from the digest in its path and
/// a re-install of an already installed pack pays no read of its bytes.
fn existing_pack_content_id_for_eviction(paths: &PullPaths) -> Option<String> {
    if !paths.final_path.exists() {
        return None;
    }
    let content_id =
        crate::models::runtime_cache_coordinator::pack_content_id_for_path_before_replace(
            &paths.final_path,
            &models_root_for_paths(paths),
        );
    crate::models::runtime_cache_coordinator::is_cacheable_pack_content_id(&content_id)
        .then_some(content_id)
}

/// Evicts `pack_content_id`'s resident state from the explicitly supplied
/// execution-service root.
///
/// This is a memory-reclaim step, not a correctness requirement: every one of
/// these caches is already content-addressed, so a request against the newly
/// installed bytes at this path naturally misses (a different content id) and
/// rebuilds on its own even if this eviction never ran. Called unconditionally
/// across every builtin family -- a `HashMap` removal for a content id that
/// family never cached is just a no-op lookup miss, so this does not need to
/// know which architecture the replaced pack belonged to.
fn evict_resident_runtime_caches_for_content_id(
    execution_services: Option<&crate::NativeExecutionServices>,
    pack_content_id: &str,
) {
    if let Some(execution_services) = execution_services {
        execution_services.evict_prepared_runtime_content_id(pack_content_id);
    }
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

/// Whether the object this target would download is already in the store,
/// and is exactly the bytes the catalog names.
///
/// `final_path` is built from `target.sha256` under this pull's own models
/// root, so a *sealed* object found there has its digest pinned by the path,
/// its immutability pinned by the seal, and its provenance pinned by the
/// anchor (see `content_store`'s integrity chain and
/// `trusted_object_digest`'s `models_root` parameter): the match is a stat
/// for the size plus the seal-gated, anchor-gated path digest, with no read
/// of the bytes. Re-pulling an installed gigabyte pack must not cost a
/// gigabyte of hash. Anything the gate declines -- a lost seal, a layout the
/// parser does not recognise -- falls back to hashing the whole file, the
/// only way to prove identity for bytes nothing has pinned.
///
/// Either way the exact install-time verifier still runs before the object
/// stands in for a download: skipping the download on the strength of a
/// digest must not skip route selection or the family runtime contract. This
/// also upgrades objects admitted by older clients instead of reporting an
/// unusable legacy object as successfully installed.
fn installed_matches(target: &PullTarget, paths: &PullPaths) -> Result<bool, PullError> {
    let Ok(metadata) = fs::metadata(&paths.final_path) else {
        return Ok(false);
    };
    if metadata.len() != target.size_bytes {
        return Ok(false);
    }
    let matched = match content_store::trusted_object_digest(
        &paths.final_path,
        metadata.permissions().readonly(),
        &models_root_for_paths(paths),
    ) {
        Some(digest) => digest == target.sha256,
        None => {
            let (size, sha) = file_size_and_sha256(&paths.final_path)?;
            size == target.size_bytes && sha == target.sha256
        }
    };
    if !matched {
        return Ok(false);
    }
    let verified = PackVerifier
        .verify_candidate(PackCandidate::new(&paths.final_path))
        .map_err(pack_verification_to_pull_error)?;
    ensure_catalog_family_matches_target(target, verified.catalog_family_id(), &paths.final_path)?;
    Ok(true)
}

fn ensure_catalog_family_matches_target(
    target: &PullTarget,
    actual_catalog_family_id: Option<&str>,
    path: &Path,
) -> Result<(), PullError> {
    ensure_catalog_family_matches(
        target.expected_catalog_family_id.as_deref(),
        actual_catalog_family_id,
        path,
    )
}

fn ensure_catalog_family_matches(
    expected_catalog_family_id: Option<&str>,
    actual_catalog_family_id: Option<&str>,
    path: &Path,
) -> Result<(), PullError> {
    let Some(expected) = expected_catalog_family_id else {
        return Ok(());
    };
    match actual_catalog_family_id {
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => Err(PullError::RuntimeValidation {
            path: path.to_path_buf(),
            reason: format!(
                "verified pack catalog family '{actual}' does not match signed catalog target family '{expected}'"
            ),
        }),
        None => Err(PullError::RuntimeValidation {
            path: path.to_path_buf(),
            reason: format!(
                "verified pack route has no canonical catalog family; expected '{expected}'"
            ),
        }),
    }
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
    let legacy_model_dir = root.join(
        paths
            .installed_meta_path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .unwrap_or_default(),
    );
    let legacy_quant_dir = legacy_model_dir.join(
        paths
            .installed_meta_path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or_default(),
    );
    for path in [
        &root,
        legacy_model_dir.as_path(),
        legacy_quant_dir.as_path(),
        paths.dir.as_path(),
        paths.partial_path.parent().expect("partial has parent"),
        paths.installed_meta_path.parent().expect("ref has parent"),
        paths.lock_path.parent().expect("lock has parent"),
        paths.final_path.parent().expect("object has parent"),
    ] {
        ensure_safe_directory_under_root(&root, path)?;
    }
    Ok(())
}

/// Create and walk storage one component at a time. A leaf-only symlink check
/// is not sufficient: `refs`, `objects`, or a model ancestor can be swapped for
/// a link after its child path is derived. Each existing component is rejected
/// when it is a symlink, and each canonicalized component must remain beneath
/// the canonical storage root.
fn ensure_safe_directory_under_root(root: &Path, path: &Path) -> Result<(), PullError> {
    if !path.starts_with(root) {
        return Err(PullError::UnsafeStoragePath {
            path: path.to_path_buf(),
        });
    }
    fs::create_dir_all(root).map_err(|source| PullError::CreateDir {
        path: root.to_path_buf(),
        source,
    })?;
    reject_symlink(root)?;
    let canonical_root = fs::canonicalize(root).map_err(|source| PullError::Io {
        path: root.to_path_buf(),
        source,
    })?;
    let relative = path
        .strip_prefix(root)
        .map_err(|_| PullError::UnsafeStoragePath {
            path: path.to_path_buf(),
        })?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let std::path::Component::Normal(component) = component else {
            return Err(PullError::UnsafeStoragePath {
                path: path.to_path_buf(),
            });
        };
        current.push(component);
        if current.exists() {
            reject_symlink(&current)?;
        } else {
            fs::create_dir(&current).map_err(|source| PullError::CreateDir {
                path: current.clone(),
                source,
            })?;
        }
        let canonical = fs::canonicalize(&current).map_err(|source| PullError::Io {
            path: current.clone(),
            source,
        })?;
        if !canonical.starts_with(&canonical_root) {
            return Err(PullError::UnsafeStoragePath { path: current });
        }
    }
    Ok(())
}
/// Recover the models root from already-derived paths. `dir` is the staging
/// directory (`<models>/staging`), which stays one level below the root
/// regardless of how deep the object layout nests.
fn models_root_for_paths(paths: &PullPaths) -> PathBuf {
    paths
        .dir
        .parent()
        .expect("staging layout has models root")
        .to_path_buf()
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
    let root = models_root(home);
    let dir = root.join("staging");
    let staging_dir = dir.clone();
    let final_path = content_store::object_path(&root, &target.sha256)?;
    let ref_dir = root.join("refs").join(&target.model_id);
    Ok(PullPaths {
        partial_path: staging_dir.join(format!("{}-{}.partial", target.sha256, target.filename)),
        partial_meta_path: staging_dir.join(format!(
            "{}-{}.partial.meta.json",
            target.sha256, target.filename
        )),
        partial_segments_meta_path: dir.join(format!(
            "{}-{}.partial.segments.json",
            target.sha256, target.filename
        )),
        installed_meta_path: ref_dir.join(format!("{}.json", target.quant)),
        lock_path: dir.join(format!("{}-{}.lock", target.model_id, target.quant)),
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

/// Per-segment counterpart to [`LowSpeedWindow`], used by the concurrent
/// chunked-download path (see `SEGMENT_LOW_SPEED_TIMEOUT`,
/// `SEGMENT_LOW_SPEED_RELATIVE_RATIO`, and
/// `SEGMENT_LOW_SPEED_ABSOLUTE_FLOOR_BYTES` for the relative-judgment
/// rationale). Deliberately a separate, smaller type rather than a
/// generalization of `LowSpeedWindow`:
/// the whole-file guard's job is "fail the pull with a typed error", while
/// this one's job is "tell the caller whether to give up on *this attempt*"
/// -- the caller (`write_segment_body`) then decides whether that means
/// requeuing the segment for a fresh connection. Unlike the whole-file guard,
/// abandoning a segment is never a hard failure: past `SEGMENT_MAX_RETRIES`,
/// further evaluation is simply disabled for that segment's final attempt
/// (see `disabled`), so the worst case is identical to not having this guard
/// at all -- the segment just finishes on whatever connection it's on.
///
/// Session-wide shared state, cloned/threaded into every worker and the
/// probe:
/// - [`SegmentThroughputReference`] -- the rolling median every segment is
///   judged against (see its doc comment for why this is relative, not an
///   absolute floor).
/// - `cooldown_slot` -- this specific segment index's last-trip timestamp,
///   damping requeue churn for a segment sitting right at the outlier
///   boundary (see `SEGMENT_LOW_SPEED_COOLDOWN`).
struct SegmentLowSpeedWindow<'a> {
    started_at: Instant,
    bytes_read: u64,
    timeout: Duration,
    reference: &'a SegmentThroughputReference,
    cooldown_slot: &'a Mutex<Option<Instant>>,
    relative_ratio: f64,
    absolute_floor_bytes: u64,
    cooldown: Duration,
    /// Set once this segment index has already been abandoned
    /// `SEGMENT_MAX_RETRIES` times: this attempt is its last chance, so
    /// evaluation is skipped entirely and it's simply allowed to finish
    /// (see `download_parallel_attempt`'s degrade-and-log handling).
    disabled: bool,
}

impl<'a> SegmentLowSpeedWindow<'a> {
    fn new(
        options: &PullOptions,
        reference: &'a SegmentThroughputReference,
        cooldown_slot: &'a Mutex<Option<Instant>>,
        disabled: bool,
    ) -> Self {
        Self {
            started_at: Instant::now(),
            bytes_read: 0,
            timeout: options.segment_low_speed_timeout,
            reference,
            cooldown_slot,
            relative_ratio: options.segment_low_speed_relative_ratio,
            absolute_floor_bytes: options.segment_low_speed_absolute_floor_bytes,
            cooldown: options.segment_low_speed_cooldown,
            disabled,
        }
    }

    /// Rolling-window check: once a full `timeout` window elapses, records
    /// its byte count into the shared reference and decides whether this
    /// window was a low-speed outlier (see `SegmentThroughputReference` and
    /// the module-level constants for the two-part AND condition). Resets on
    /// every window regardless of the outcome, so a segment that starts fast
    /// and later degrades is still caught (or one that recovers stops being
    /// flagged). `disabled` skips everything -- this segment already used up
    /// its reconnect budget, so its last attempt is allowed to run to
    /// completion unconditionally.
    ///
    /// Note this compares raw per-window *byte counts*, never a computed
    /// bytes/sec rate: every window shares the same configured `timeout`, so
    /// byte counts are already directly comparable as a throughput proxy,
    /// without ever dividing by a measured elapsed time (which would be
    /// unstable for very short or test-forced windows).
    fn observe(&mut self, bytes_read: u64) -> bool {
        if self.disabled {
            return false;
        }
        self.bytes_read = self.bytes_read.saturating_add(bytes_read);
        if self.started_at.elapsed() < self.timeout {
            return false;
        }
        let window_bytes = self.bytes_read;
        self.started_at = Instant::now();
        self.bytes_read = 0;
        // Compare against the reference as it stood *before* this window is
        // folded in, so a slow window never gets to (even slightly) pull
        // down the baseline it is itself being judged against.
        let median = self.reference.median();
        self.reference.record(window_bytes);
        let Some(median) = median else {
            return false; // cold start: no reference yet, never judge
        };
        let relative_floor = (median as f64 * self.relative_ratio) as u64;
        let is_outlier = window_bytes < relative_floor && window_bytes < self.absolute_floor_bytes;
        if !is_outlier {
            return false;
        }
        let mut cooldown_slot = self.cooldown_slot.lock().unwrap();
        let now = Instant::now();
        if let Some(last_trip) = *cooldown_slot
            && now.duration_since(last_trip) < self.cooldown
        {
            return false; // still cooling down from the last trip
        }
        *cooldown_slot = Some(now);
        true
    }
}

/// Session-wide reference for the concurrent chunked-download path's
/// relative low-speed guard: a bounded rolling window of per-segment-window
/// byte counts (see [`SegmentLowSpeedWindow`]), shared across every worker
/// and the probe so a segment is judged against what *this* download
/// session is actually achieving right now -- never a fixed absolute number,
/// which would misjudge a uniformly slow-but-working network as broken (see
/// `SEGMENT_LOW_SPEED_RELATIVE_RATIO`'s doc comment for the full rationale).
struct SegmentThroughputReference {
    samples: Mutex<VecDeque<u64>>,
}

impl SegmentThroughputReference {
    fn new() -> Self {
        Self {
            samples: Mutex::new(VecDeque::new()),
        }
    }

    /// The median of every recorded window's byte count. Requires at least
    /// `SEGMENT_LOW_SPEED_MIN_REFERENCE_SAMPLES` samples; `None` (fewer than
    /// that) means "cold start: never judge a segment low-speed yet" -- see
    /// that constant's doc comment for why.
    fn median(&self) -> Option<u64> {
        let samples = self.samples.lock().unwrap();
        if samples.len() < SEGMENT_LOW_SPEED_MIN_REFERENCE_SAMPLES {
            return None;
        }
        let mut sorted: Vec<u64> = samples.iter().copied().collect();
        sorted.sort_unstable();
        Some(sorted[sorted.len() / 2])
    }

    /// Records one window's byte count, bounding the reference to the most
    /// recent `SEGMENT_LOW_SPEED_REFERENCE_CAPACITY` samples (see that
    /// constant's doc comment).
    fn record(&self, bytes_in_window: u64) {
        let mut samples = self.samples.lock().unwrap();
        samples.push_back(bytes_in_window);
        while samples.len() > SEGMENT_LOW_SPEED_REFERENCE_CAPACITY {
            samples.pop_front();
        }
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
        if resolved.catalog_family_id.trim().is_empty() {
            return Err(PullError::InvalidTarget {
                field: "catalog_family_id",
                reason: "catalog family id must not be empty".to_string(),
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
            expected_catalog_family_id: Some(resolved.catalog_family_id.clone()),
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

fn write_backend_partial_meta(
    path: &Path,
    file: &CatalogBackendFile,
    etag: Option<String>,
    bytes_done: u64,
) -> Result<(), PullError> {
    let meta = BackendPartialMeta::for_file(file, etag, bytes_done);
    let json = serde_json::to_string_pretty(&meta).map_err(|source| PullError::SerializeMeta {
        path: path.to_path_buf(),
        source,
    })?;
    write_json_atomic(path, &format!("{json}\n"))
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

fn lock_owner_is_gone(path: &Path) -> bool {
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
    process_is_gone(pid)
}

/// Whether `pid` has certainly exited. Shared by stale-lock recovery and by
/// model-store garbage collection, which uses it to decide that a staging entry
/// no process can still finish is unconditionally garbage.
///
/// Always answers "still alive" when liveness cannot be established, so every
/// caller fails toward keeping state rather than deleting it.
#[cfg(unix)]
pub(crate) fn process_is_gone(pid: u32) -> bool {
    let Ok(pid) = libc::pid_t::try_from(pid) else {
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
pub(crate) fn process_is_gone(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    // STILL_ACTIVE (STATUS_PENDING): a process that has not exited reports this as
    // its "exit code". Any other value means it has terminated.
    const STILL_ACTIVE: u32 = 259;

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
        // queried == 0 -> status unreadable; be conservative and treat as live.
        queried != 0 && exit_code != STILL_ACTIVE
    }
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn process_is_gone(_pid: u32) -> bool {
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
        // `ShaMismatch` stays source-fallback eligible: a mirror can serve
        // corrupted bytes for an object whose canonical sha256 is fine
        // elsewhere, so the next source is a genuine remedy. The attempt
        // count is bounded by the finite source chain itself -- one pass
        // through the chain per pull, no unbounded retry loop.
        PullError::Http { .. }
        | PullError::Io { .. }
        | PullError::RestartedPartial { .. }
        | PullError::SizeMismatch { .. }
        | PullError::ShaMismatch { .. }
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
        //
        // Content-deterministic failures are deliberately NOT included either:
        // `GgufPreflight`, `BackendFilePreflight`, and `RuntimeValidation`
        // only run AFTER the downloaded bytes matched the catalog's sha256,
        // so every source in the chain serves byte-identical content -- the
        // pack itself is broken (or incompatible with this build), and
        // re-downloading multi-GB files from the remaining mirrors just
        // repeats the same verdict. These must fail the whole pull on the
        // first occurrence as a permanent error.
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
/// local import) and before it is committed into the local model store.
/// [`PackVerifier`] owns the single fail-closed gate: it checks the `.oasr` v1
/// package contract, validates the runtime source path, and then applies the
/// registry-owned runtime contract for the selected ASR or auxiliary family.
/// The individual checks remain exposed for callers that need one structural
/// or backend-specific preflight, but installation always uses this unified
/// verifier path.
///
/// Importer tests reuse this exact function so a pack a family importer can
/// build but `openasr pull` would reject can never ship.
pub fn preflight_model_pack_for_install(path: &Path) -> Result<(), PullError> {
    PackVerifier
        .verify_candidate(PackCandidate::new(path))
        .map(|_| ())
        .map_err(pack_verification_to_pull_error)
}

/// Stable, machine-readable evidence emitted by the exact install-time pack
/// verifier. Tooling may persist this receipt, but it is not itself an
/// execution capability: only the in-process `VerifiedPack` can authorize
/// install/runtime use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPackPreflightReceipt {
    pub schema: String,
    pub content_id: String,
    pub size_bytes: u64,
    pub route: String,
    pub catalog_family_id: String,
    pub model_family: Option<String>,
    pub model_architecture: String,
    pub build_commit: Option<String>,
}

/// Run the client verifier once and project its proof into a data-only receipt
/// for release tooling. The digest and size come from the exact open mapping
/// that passed the package and runtime contracts, never from a later path
/// reopen.
pub fn preflight_model_pack_with_receipt(
    path: &Path,
) -> Result<ModelPackPreflightReceipt, PullError> {
    let verified = PackVerifier
        .verify_candidate(PackCandidate::new(path))
        .map_err(pack_verification_to_pull_error)?;
    let (route, catalog_family_id, model_family, model_architecture) = match verified.route() {
        PackRoute::Asr {
            model_family,
            model_architecture,
        } => {
            let catalog_family_id = crate::arch::OpenAsrArchitectureRegistry::with_builtins()
                .find_by_model_architecture(model_architecture)
                .map(|descriptor| descriptor.identity.catalog_family_id)
                .ok_or_else(|| PullError::RuntimeValidation {
                    path: path.to_path_buf(),
                    reason: format!(
                        "verified ASR architecture '{model_architecture}' has no canonical inventory row"
                    ),
                })?;
            (
                "asr".to_string(),
                catalog_family_id.to_string(),
                Some((*model_family).to_string()),
                (*model_architecture).to_string(),
            )
        }
        PackRoute::Aux {
            model_architecture, ..
        } => {
            let catalog_family_id =
                crate::models::aux_pack_registry::auxiliary_catalog_family_id(model_architecture)
                    .ok_or_else(|| PullError::RuntimeValidation {
                        path: path.to_path_buf(),
                        reason: format!(
                            "verified auxiliary architecture '{model_architecture}' has no canonical catalog family"
                        ),
                    })?;
            (
                "aux".to_string(),
                catalog_family_id.to_string(),
                None,
                model_architecture.clone(),
            )
        }
    };
    Ok(ModelPackPreflightReceipt {
        schema: "openasr.model-pack-preflight.v1".to_string(),
        content_id: verified.content_id().to_string(),
        size_bytes: verified.preflight().runtime_source().byte_len(),
        route,
        catalog_family_id,
        model_family,
        model_architecture,
        build_commit: verified
            .preflight()
            .metadata()
            .get_string(crate::ggml_runtime::OASR_METADATA_KEY_BUILD_COMMIT)
            .map(str::to_string),
    })
}

fn pack_verification_to_pull_error(error: PackVerificationError) -> PullError {
    let path = match &error {
        PackVerificationError::RuntimeSource { path, .. } => path.clone(),
        PackVerificationError::PackageContract { path, .. }
        | PackVerificationError::RuntimePreflight { path, .. }
        | PackVerificationError::RuntimeContract { path, .. } => path.clone(),
    };
    match error {
        PackVerificationError::PackageContract { reason, .. } => {
            PullError::GgufPreflight { path, reason }
        }
        PackVerificationError::RuntimeSource {
            source: crate::GgmlRuntimeSourcePathError::Probe(source),
            ..
        } => PullError::GgufPreflight {
            path,
            reason: source.to_string(),
        },
        other => PullError::RuntimeValidation {
            path,
            reason: other.to_string(),
        },
    }
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
/// installed or loaded — the backend analogue of [`preflight_model_pack_for_install`]
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

pub const INSTALLED_BACKEND_SCHEMA_VERSION: u32 = 3;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledBackendMaterializedFile {
    pub relative_path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledBackendFile {
    pub filename: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub role: CatalogBackendFileRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extract_subdir: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extracted_tree_sha256: Option<String>,
    pub materialized_files: Vec<InstalledBackendMaterializedFile>,
}

const BACKEND_CONTENT_OBJECT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct BackendContentObject {
    schema_version: u32,
    source_filename: String,
    source_sha256: String,
    source_size_bytes: u64,
    role: CatalogBackendFileRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    extract_subdir: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    extracted_tree_sha256: Option<String>,
    files: Vec<InstalledBackendMaterializedFile>,
}

/// Cross-process backend install lock. Unlike [`PullLock`], this deliberately
/// keeps a stable lock file and lets the operating system own the lock
/// lifetime. A Windows installer terminating the old Desktop process releases
/// the file handle immediately, so the replacement process can resume the
/// exact same content-addressed partial instead of waiting for a stale-file
/// timeout or deleting another process's state.
struct BackendInstallLock {
    _file: File,
}

impl BackendInstallLock {
    fn acquire(path: &Path) -> Result<Self, PullError> {
        #[cfg(windows)]
        let mut file = {
            use std::os::windows::fs::OpenOptionsExt;

            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .share_mode(0)
                .open(path)
                .map_err(|source| {
                    if matches!(source.raw_os_error(), Some(32 | 33)) {
                        PullError::LockHeld {
                            path: path.to_path_buf(),
                        }
                    } else {
                        PullError::LockIo {
                            path: path.to_path_buf(),
                            source,
                        }
                    }
                })?
        };

        #[cfg(unix)]
        let mut file = {
            use std::os::fd::AsRawFd;

            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path)
                .map_err(|source| PullError::LockIo {
                    path: path.to_path_buf(),
                    source,
                })?;
            // SAFETY: `file` owns a live descriptor for the duration of this
            // guard. LOCK_NB makes a concurrent installer fail closed instead
            // of blocking an application thread indefinitely.
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                let source = io::Error::last_os_error();
                if matches!(
                    source.raw_os_error(),
                    Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN
                ) {
                    return Err(PullError::LockHeld {
                        path: path.to_path_buf(),
                    });
                }
                return Err(PullError::LockIo {
                    path: path.to_path_buf(),
                    source,
                });
            }
            file
        };

        #[cfg(not(any(unix, windows)))]
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|source| {
                if source.kind() == io::ErrorKind::AlreadyExists {
                    PullError::LockHeld {
                        path: path.to_path_buf(),
                    }
                } else {
                    PullError::LockIo {
                        path: path.to_path_buf(),
                        source,
                    }
                }
            })?;

        file.set_len(0).map_err(|source| PullError::LockIo {
            path: path.to_path_buf(),
            source,
        })?;
        writeln!(file, "pid={}", std::process::id()).map_err(|source| PullError::LockIo {
            path: path.to_path_buf(),
            source,
        })?;
        file.sync_all().map_err(|source| PullError::LockIo {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Self { _file: file })
    }
}

/// Process-replacement-safe serialization for every mutation of the backend
/// plugin store. Per-artifact locks still prevent duplicate downloads, while
/// this outer lock prevents install promotion, activation-pointer commits,
/// and garbage collection from observing or deleting each other's partial
/// state. The stable file is intentionally never deleted: the operating
/// system releases the handle if NSIS terminates the old process.
pub(crate) struct BackendStoreMutationLock {
    _inner: BackendInstallLock,
}

impl BackendStoreMutationLock {
    pub(crate) fn acquire(home: &Path) -> Result<Self, PullError> {
        let locks_root = home.join("backends").join(".locks");
        fs::create_dir_all(&locks_root).map_err(|source| PullError::Io {
            path: locks_root.clone(),
            source,
        })?;
        Ok(Self {
            _inner: BackendInstallLock::acquire(&locks_root.join("lifecycle-v1.lock"))?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendStoreGcReport {
    pub schema_version: u32,
    pub retained_backend_ids: Vec<String>,
    pub removed_pack_directories: u64,
    pub removed_staging_directories: u64,
    pub removed_content_objects: u64,
    pub reclaimed_bytes: u64,
    pub deferred_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct BackendPartialMeta {
    filename: String,
    sha256: String,
    size_bytes: u64,
    etag: Option<String>,
    bytes_done: u64,
    updated_at_unix_seconds: u64,
}

impl BackendPartialMeta {
    fn for_file(file: &CatalogBackendFile, etag: Option<String>, bytes_done: u64) -> Self {
        Self {
            filename: file.filename.clone(),
            sha256: file.sha256.clone(),
            size_bytes: file.size_bytes,
            etag,
            bytes_done,
            updated_at_unix_seconds: unix_seconds_now(),
        }
    }

    fn matches_file(&self, file: &CatalogBackendFile) -> bool {
        self.filename == file.filename
            && self.sha256 == file.sha256
            && self.size_bytes == file.size_bytes
    }
}

/// Tamper-evident record of an installed backend plugin pack. The record is
/// useful only together with the signed catalog entry it mirrors: load-time
/// verification compares the full identity and rehashes every declared file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledBackend {
    pub schema_version: u32,
    pub backend_id: String,
    pub vendor: String,
    pub version: String,
    pub host_abi: crate::backend_distribution::BackendHostAbi,
    pub artifact_fingerprint: String,
    #[serde(skip, default)]
    pub dir: PathBuf,
    pub plugin_filename: String,
    pub files: Vec<InstalledBackendFile>,
    pub installed_at_unix_seconds: u64,
}

/// Exact transfer accounting for one signed backend pack in the current
/// content-addressed store. `required_*` subtracts verified installed bytes,
/// shared runtime/archive objects, and resumable partials; callers can present
/// an honest maintenance prompt without duplicating store semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendPackDownloadPlan {
    pub total_bytes: u64,
    pub plugin_bytes: u64,
    pub vendor_bytes: u64,
    pub required_download_bytes: u64,
    pub required_plugin_bytes: u64,
    pub required_vendor_bytes: u64,
}

/// Fail-closed, not a panic: `filter_forward_compatible_catalog` already drops
/// a backend pack whose `vendor` this build does not recognize before it can
/// reach pull resolution, but this is trust-boundary code parsing signed-yet-
/// external data, so an `Unknown` reaching here (a filtering bug, or a caller
/// that built a `ResolvedCatalogBackendPull` some other way) must return a
/// typed error rather than panic or silently guess a directory.
pub(crate) fn backend_vendor_dirname(
    vendor: CatalogBackendVendor,
) -> Result<&'static str, PullError> {
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

pub fn backend_artifact_fingerprint(resolved: &ResolvedCatalogBackendPull) -> String {
    let mut hasher = Sha256::new();
    for value in [
        resolved.backend_id.as_str(),
        backend_vendor_name(resolved.vendor),
        resolved.version.as_str(),
        resolved.host_abi.fingerprint.as_str(),
        resolved.min_driver_api.as_deref().unwrap_or(""),
    ] {
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    for target in &resolved.targets {
        hasher.update((target.len() as u64).to_le_bytes());
        hasher.update(target.as_bytes());
    }
    for file in &resolved.files {
        for value in [
            file.filename.as_str(),
            file.sha256.as_str(),
            file.extract_subdir.as_deref().unwrap_or(""),
            file.extracted_tree_sha256.as_deref().unwrap_or(""),
        ] {
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        hasher.update(file.size_bytes.to_le_bytes());
        hasher.update([backend_file_role_tag(file.role)]);
    }
    format!("{:x}", hasher.finalize())
}

pub(crate) fn backend_pack_install_dir(
    home: &Path,
    resolved: &ResolvedCatalogBackendPull,
) -> Result<PathBuf, PullError> {
    validate_backend_pack_version(&resolved.version)?;
    let vendor = backend_vendor_dirname(resolved.vendor)?;
    Ok(home
        .join("backends")
        .join(vendor)
        .join(&resolved.version)
        .join(backend_artifact_fingerprint(resolved)))
}

fn validate_backend_pack_version(version: &str) -> Result<(), PullError> {
    if version.is_empty()
        || version
            .split(['/', '\\'])
            .any(|component| component.is_empty() || component == "..")
        || version.contains(':')
    {
        return Err(PullError::InvalidTarget {
            field: "backend.version",
            reason: format!("'{version}' is not a safe path segment"),
        });
    }
    Ok(())
}

fn backend_vendor_name(vendor: CatalogBackendVendor) -> &'static str {
    match vendor {
        CatalogBackendVendor::Cpu => "cpu",
        CatalogBackendVendor::Vulkan => "vulkan",
        CatalogBackendVendor::Hip => "hip",
        CatalogBackendVendor::Cuda => "cuda",
        CatalogBackendVendor::Unknown => "unknown",
    }
}

fn backend_file_role_tag(role: CatalogBackendFileRole) -> u8 {
    match role {
        CatalogBackendFileRole::Runtime => 0,
        CatalogBackendFileRole::Plugin => 1,
        CatalogBackendFileRole::Archive => 2,
        CatalogBackendFileRole::Unknown => u8::MAX,
    }
}

fn installed_backend_file(
    file: &CatalogBackendFile,
    materialized_files: Vec<InstalledBackendMaterializedFile>,
) -> InstalledBackendFile {
    InstalledBackendFile {
        filename: file.filename.clone(),
        sha256: file.sha256.to_ascii_lowercase(),
        size_bytes: file.size_bytes,
        role: file.role,
        extract_subdir: file.extract_subdir.clone(),
        extracted_tree_sha256: file.extracted_tree_sha256.clone(),
        materialized_files,
    }
}

fn installed_backend_file_identity_matches(
    installed: &InstalledBackendFile,
    expected: &CatalogBackendFile,
) -> bool {
    installed.filename == expected.filename
        && installed.sha256.eq_ignore_ascii_case(&expected.sha256)
        && installed.size_bytes == expected.size_bytes
        && installed.role == expected.role
        && installed.extract_subdir == expected.extract_subdir
        && installed.extracted_tree_sha256 == expected.extracted_tree_sha256
}

fn materialized_tree_sha256(files: &[InstalledBackendMaterializedFile]) -> String {
    let mut sorted = files.to_vec();
    sorted.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let mut hasher = Sha256::new();
    hasher.update(b"openasr-backend-tree-v1\0");
    for file in sorted {
        hasher.update((file.relative_path.len() as u64).to_le_bytes());
        hasher.update(file.relative_path.as_bytes());
        hasher.update(file.size_bytes.to_le_bytes());
        hasher.update(file.sha256.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn installed_backend_relative_path(root: &Path, path: &Path) -> Result<String, PullError> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| PullError::BackendFilePreflight {
            path: path.to_path_buf(),
            reason: "payload path escaped its content object".to_string(),
        })?;
    let relative_path = relative
        .components()
        .map(|component| component.as_os_str().to_str())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| PullError::BackendFilePreflight {
            path: path.to_path_buf(),
            reason: "payload path is not UTF-8".to_string(),
        })?
        .join("/");
    validate_safe_relative_path("backend payload path", &relative_path).map_err(|reason| {
        PullError::BackendFilePreflight {
            path: path.to_path_buf(),
            reason,
        }
    })?;
    Ok(relative_path)
}

fn walk_installed_backend_files(
    root: &Path,
    mut on_file: impl FnMut(&Path, &str) -> Result<(), PullError>,
) -> Result<(), PullError> {
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let entries = fs::read_dir(&dir).map_err(|source| PullError::Io {
            path: dir.clone(),
            source,
        })?;
        for entry in entries {
            let entry = entry.map_err(|source| PullError::Io {
                path: dir.clone(),
                source,
            })?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path).map_err(|source| PullError::Io {
                path: path.clone(),
                source,
            })?;
            if metadata.file_type().is_symlink() {
                return Err(PullError::BackendFilePreflight {
                    path,
                    reason: "content-addressed backend payload may not contain symlinks"
                        .to_string(),
                });
            }
            if metadata.is_dir() {
                pending.push(path);
                continue;
            }
            if !metadata.is_file() {
                return Err(PullError::BackendFilePreflight {
                    path,
                    reason: "content-addressed backend payload contains a non-file entry"
                        .to_string(),
                });
            }
            let relative_path = installed_backend_relative_path(root, &path)?;
            on_file(&path, &relative_path)?;
        }
    }
    Ok(())
}

fn collect_materialized_files(
    root: &Path,
) -> Result<Vec<InstalledBackendMaterializedFile>, PullError> {
    let mut files = Vec::new();
    walk_installed_backend_files(root, |path, relative_path| {
        let (size_bytes, sha256) = file_size_and_sha256(path)?;
        files.push(InstalledBackendMaterializedFile {
            relative_path: relative_path.to_string(),
            sha256,
            size_bytes,
        });
        Ok(())
    })?;
    files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    Ok(files)
}

fn is_backend_load_image_relative_path(relative_path: &str) -> bool {
    Path::new(relative_path)
        .extension()
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("dll")
                || extension.eq_ignore_ascii_case("so")
                || extension.eq_ignore_ascii_case("dylib")
        })
}

/// How thoroughly an installed backend pack is checked before use.
///
/// `Full` is the fail-closed install/repair gate: every materialized file is
/// size/hash verified and the install tree is a closed set.
///
/// `LoadImages` is the activation hot path. HIP vendor trees contain thousands
/// of Tensile/hsaco objects that DllMain never maps; hashing them twice at
/// process start dominated `ggml_backend` on hosts without SHA-NI. Activation
/// still authenticates `backend.json`, still checks the signed archive tree
/// digest from recorded materialized hashes, still hashes the plugin and every
/// native library image, and still rejects extra sibling libraries the loader
/// would map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstalledBackendVerifyScope {
    Full,
    LoadImages,
}

fn should_hash_installed_backend_file(
    relative_path: &str,
    plugin_filename: &str,
    scope: InstalledBackendVerifyScope,
) -> bool {
    match scope {
        InstalledBackendVerifyScope::Full => true,
        InstalledBackendVerifyScope::LoadImages => {
            relative_path == plugin_filename || is_backend_load_image_relative_path(relative_path)
        }
    }
}

fn backend_content_object_dir(home: &Path, file: &CatalogBackendFile) -> PathBuf {
    home.join("backends")
        .join("_objects")
        .join(file.sha256.to_ascii_lowercase())
}

fn backend_pack_staging_dir(
    home: &Path,
    resolved: &ResolvedCatalogBackendPull,
) -> Result<PathBuf, PullError> {
    validate_backend_pack_version(&resolved.version)?;
    let vendor = backend_vendor_dirname(resolved.vendor)?;
    Ok(home
        .join("backends")
        .join(".staging")
        .join(vendor)
        .join(&resolved.version)
        .join(backend_artifact_fingerprint(resolved)))
}

fn backend_object_staging_source(home: &Path, file: &CatalogBackendFile) -> PathBuf {
    home.join("backends")
        .join("_objects")
        .join(".staging")
        .join(file.sha256.to_ascii_lowercase())
        .join("source")
        .join(&file.filename)
}

fn backend_file_matches(path: &Path, file: &CatalogBackendFile) -> bool {
    file_size_and_sha256(path).is_ok_and(|(size, sha256)| {
        size == file.size_bytes && sha256.eq_ignore_ascii_case(&file.sha256)
    })
}

fn backend_partial_paths(dest: &Path) -> Result<(PathBuf, PathBuf), PullError> {
    let stem = dest
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| PullError::InvalidTarget {
            field: "backend.files.filename",
            reason: format!("'{}' has no UTF-8 file name", dest.display()),
        })?;
    Ok((
        dest.with_file_name(format!(".{stem}.partial")),
        dest.with_file_name(format!(".{stem}.partial.json")),
    ))
}

fn verified_backend_partial_len(file: &CatalogBackendFile, dest: &Path) -> u64 {
    let Ok((partial, partial_meta)) = backend_partial_paths(dest) else {
        return 0;
    };
    let Ok(meta_text) = fs::read_to_string(partial_meta) else {
        return 0;
    };
    let Ok(meta) = serde_json::from_str::<BackendPartialMeta>(&meta_text) else {
        return 0;
    };
    if !meta.matches_file(file) {
        return 0;
    }
    let len = fs::metadata(&partial)
        .ok()
        .map(|metadata| metadata.len())
        .filter(|len| *len <= file.size_bytes)
        .unwrap_or(0);
    if len == file.size_bytes && !backend_file_matches(&partial, file) {
        return 0;
    }
    len
}

fn checked_backend_byte_sum(left: u64, right: u64) -> Result<u64, PullError> {
    left.checked_add(right)
        .ok_or_else(|| PullError::InvalidTarget {
            field: "backend.files.size_bytes",
            reason: "backend pack byte total overflowed u64".to_string(),
        })
}

/// Computes the exact remaining network bytes for a resolved, signed backend
/// pack without mutating installation or activation state.
pub fn backend_pack_download_plan(
    home: impl AsRef<Path>,
    resolved: &ResolvedCatalogBackendPull,
) -> Result<BackendPackDownloadPlan, PullError> {
    let home = home.as_ref();
    let mut plan = BackendPackDownloadPlan {
        total_bytes: 0,
        plugin_bytes: 0,
        vendor_bytes: 0,
        required_download_bytes: 0,
        required_plugin_bytes: 0,
        required_vendor_bytes: 0,
    };
    for file in &resolved.files {
        plan.total_bytes = checked_backend_byte_sum(plan.total_bytes, file.size_bytes)?;
        match file.role {
            CatalogBackendFileRole::Plugin => {
                plan.plugin_bytes = checked_backend_byte_sum(plan.plugin_bytes, file.size_bytes)?;
            }
            CatalogBackendFileRole::Runtime | CatalogBackendFileRole::Archive => {
                plan.vendor_bytes = checked_backend_byte_sum(plan.vendor_bytes, file.size_bytes)?;
            }
            CatalogBackendFileRole::Unknown => {
                return Err(PullError::InvalidTarget {
                    field: "backend.files.role",
                    reason: "unknown backend file role".to_string(),
                });
            }
        }
    }

    let install_dir = backend_pack_install_dir(home, resolved)?;
    if read_and_verify_installed_backend(&install_dir, resolved).is_ok() {
        return Ok(plan);
    }
    let pack_staging = backend_pack_staging_dir(home, resolved)?;
    for file in &resolved.files {
        let dest = match file.role {
            CatalogBackendFileRole::Plugin => pack_staging.join(&file.filename),
            CatalogBackendFileRole::Runtime | CatalogBackendFileRole::Archive => {
                if verify_backend_content_object(&backend_content_object_dir(home, file), file)
                    .is_ok()
                {
                    continue;
                }
                backend_object_staging_source(home, file)
            }
            CatalogBackendFileRole::Unknown => unreachable!(),
        };
        let required = if backend_file_matches(&dest, file) {
            0
        } else {
            file.size_bytes
                .saturating_sub(verified_backend_partial_len(file, &dest))
        };
        plan.required_download_bytes =
            checked_backend_byte_sum(plan.required_download_bytes, required)?;
        match file.role {
            CatalogBackendFileRole::Plugin => {
                plan.required_plugin_bytes =
                    checked_backend_byte_sum(plan.required_plugin_bytes, required)?;
            }
            CatalogBackendFileRole::Runtime | CatalogBackendFileRole::Archive => {
                plan.required_vendor_bytes =
                    checked_backend_byte_sum(plan.required_vendor_bytes, required)?;
            }
            CatalogBackendFileRole::Unknown => unreachable!(),
        }
    }
    Ok(plan)
}

fn verify_backend_content_object(
    object_dir: &Path,
    file: &CatalogBackendFile,
) -> Result<BackendContentObject, PullError> {
    let marker_path = object_dir.join("object.json");
    let text = fs::read_to_string(&marker_path).map_err(|source| PullError::Io {
        path: marker_path.clone(),
        source,
    })?;
    let object = serde_json::from_str::<BackendContentObject>(&text).map_err(|source| {
        PullError::ParseMeta {
            path: marker_path,
            source,
        }
    })?;
    if object.schema_version != BACKEND_CONTENT_OBJECT_SCHEMA_VERSION
        || object.source_filename != file.filename
        || !object.source_sha256.eq_ignore_ascii_case(&file.sha256)
        || object.source_size_bytes != file.size_bytes
        || object.role != file.role
        || object.extract_subdir != file.extract_subdir
        || object.extracted_tree_sha256 != file.extracted_tree_sha256
    {
        return Err(PullError::InvalidTarget {
            field: "backend content object",
            reason: "object identity does not match the signed catalog file".to_string(),
        });
    }
    let actual_files = collect_materialized_files(&object_dir.join("payload"))?;
    if actual_files != object.files {
        return Err(PullError::InvalidTarget {
            field: "backend content object files",
            reason: "object payload differs from its marker".to_string(),
        });
    }
    match file.role {
        CatalogBackendFileRole::Runtime => {
            if object.files.len() != 1
                || object.files[0].relative_path != file.filename
                || object.files[0].size_bytes != file.size_bytes
                || !object.files[0].sha256.eq_ignore_ascii_case(&file.sha256)
            {
                return Err(PullError::InvalidTarget {
                    field: "backend content object files",
                    reason: "runtime object does not match the signed file".to_string(),
                });
            }
        }
        CatalogBackendFileRole::Archive => {
            let expected =
                file.extracted_tree_sha256
                    .as_deref()
                    .ok_or(PullError::InvalidTarget {
                        field: "backend.files.extracted_tree_sha256",
                        reason: "archive has no signed extracted tree digest".to_string(),
                    })?;
            let actual = materialized_tree_sha256(&object.files);
            if !actual.eq_ignore_ascii_case(expected) {
                return Err(PullError::ShaMismatch {
                    path: object_dir.join("payload"),
                    expected: expected.to_string(),
                    actual,
                });
            }
        }
        CatalogBackendFileRole::Plugin | CatalogBackendFileRole::Unknown => {
            return Err(PullError::InvalidTarget {
                field: "backend content object role",
                reason: "only runtime and archive files use shared content objects".to_string(),
            });
        }
    }
    Ok(object)
}

fn link_or_copy_backend_payload(source: &Path, dest: &Path) -> Result<(), PullError> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|source| PullError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    match fs::remove_file(dest) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(PullError::Io {
                path: dest.to_path_buf(),
                source,
            });
        }
    }
    if fs::hard_link(source, dest).is_err() {
        fs::copy(source, dest).map_err(|source| PullError::Io {
            path: dest.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}

fn ensure_backend_content_object<C: DownloadClient>(
    client: &mut C,
    file: &CatalogBackendFile,
    home: &Path,
    progress: &mut impl FnMut(PullProgress),
) -> Result<BackendContentObject, PullError> {
    let objects_root = home.join("backends").join("_objects");
    fs::create_dir_all(&objects_root).map_err(|source| PullError::Io {
        path: objects_root.clone(),
        source,
    })?;
    let object_dir = backend_content_object_dir(home, file);
    let lock_path = objects_root.join(format!("{}.lock", file.sha256.to_ascii_lowercase()));
    let _lock = BackendInstallLock::acquire(&lock_path)?;
    if let Ok(object) = verify_backend_content_object(&object_dir, file) {
        return Ok(object);
    }

    let work_dir = objects_root
        .join(".staging")
        .join(file.sha256.to_ascii_lowercase());
    fs::create_dir_all(work_dir.join("source")).map_err(|source| PullError::Io {
        path: work_dir.clone(),
        source,
    })?;
    let source_path = work_dir.join("source").join(&file.filename);
    let source_valid = file_size_and_sha256(&source_path)
        .is_ok_and(|(size, sha)| size == file.size_bytes && sha.eq_ignore_ascii_case(&file.sha256));
    if !source_valid {
        download_backend_file(client, file, &source_path, progress)?;
    }
    preflight_backend_file(&source_path, backend_file_format(file.role)?)?;

    let payload = work_dir.join("payload");
    match fs::remove_dir_all(&payload) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(PullError::Io {
                path: payload,
                source,
            });
        }
    }
    fs::create_dir_all(&payload).map_err(|source| PullError::Io {
        path: payload.clone(),
        source,
    })?;
    match file.role {
        CatalogBackendFileRole::Runtime => {
            link_or_copy_backend_payload(&source_path, &payload.join(&file.filename))?;
        }
        CatalogBackendFileRole::Archive => {
            extract_backend_archive(
                &source_path,
                &payload,
                file.extract_subdir.as_deref().unwrap_or(""),
            )?;
        }
        CatalogBackendFileRole::Plugin | CatalogBackendFileRole::Unknown => {
            return Err(PullError::InvalidTarget {
                field: "backend.files.role",
                reason: "plugin/unknown files cannot use a shared content object".to_string(),
            });
        }
    }
    let files = collect_materialized_files(&payload)?;
    let object = BackendContentObject {
        schema_version: BACKEND_CONTENT_OBJECT_SCHEMA_VERSION,
        source_filename: file.filename.clone(),
        source_sha256: file.sha256.to_ascii_lowercase(),
        source_size_bytes: file.size_bytes,
        role: file.role,
        extract_subdir: file.extract_subdir.clone(),
        extracted_tree_sha256: file.extracted_tree_sha256.clone(),
        files,
    };
    match file.role {
        CatalogBackendFileRole::Runtime => {
            if object.files.len() != 1 || !object.files[0].sha256.eq_ignore_ascii_case(&file.sha256)
            {
                return Err(PullError::ShaMismatch {
                    path: payload,
                    expected: file.sha256.clone(),
                    actual: object
                        .files
                        .first()
                        .map(|entry| entry.sha256.clone())
                        .unwrap_or_default(),
                });
            }
        }
        CatalogBackendFileRole::Archive => {
            let expected =
                file.extracted_tree_sha256
                    .as_deref()
                    .ok_or(PullError::InvalidTarget {
                        field: "backend.files.extracted_tree_sha256",
                        reason: "archive has no signed extracted tree digest".to_string(),
                    })?;
            let actual = materialized_tree_sha256(&object.files);
            if !actual.eq_ignore_ascii_case(expected) {
                return Err(PullError::ShaMismatch {
                    path: payload,
                    expected: expected.to_string(),
                    actual,
                });
            }
        }
        CatalogBackendFileRole::Plugin | CatalogBackendFileRole::Unknown => unreachable!(),
    }
    let marker_path = work_dir.join("object.json");
    let json =
        serde_json::to_string_pretty(&object).map_err(|source| PullError::SerializeMeta {
            path: marker_path.clone(),
            source,
        })?;
    write_json_atomic(&marker_path, &format!("{json}\n"))?;
    verify_backend_content_object(&work_dir, file)?;
    promote_backend_directory(&work_dir, &object_dir, &file.sha256.to_ascii_lowercase())?;
    verify_backend_content_object(&object_dir, file)
}

fn materialize_backend_content_object(
    home: &Path,
    pack_dir: &Path,
    file: &CatalogBackendFile,
    object: &BackendContentObject,
) -> Result<Vec<InstalledBackendMaterializedFile>, PullError> {
    let payload = backend_content_object_dir(home, file).join("payload");
    for materialized in &object.files {
        link_or_copy_backend_payload(
            &payload.join(Path::new(&materialized.relative_path)),
            &pack_dir.join(Path::new(&materialized.relative_path)),
        )?;
    }
    Ok(object.files.clone())
}

/// Verify an installed backend against the exact signed-catalog resolution.
/// Marker paths are never trusted: `dir` is supplied by the caller and every
/// declared artifact is size/hash/preflight checked before native code loads.
///
/// This is the install/repair gate ([`InstalledBackendVerifyScope::Full`]).
/// Runtime plugin activation must use
/// [`read_and_verify_installed_backend_for_activation`] so Tensile/hsaco
/// payloads are not re-hashed on the load path.
pub fn verify_installed_backend(
    dir: &Path,
    installed: &InstalledBackend,
    resolved: &ResolvedCatalogBackendPull,
) -> Result<(), PullError> {
    verify_installed_backend_with_scope(dir, installed, resolved, InstalledBackendVerifyScope::Full)
}

pub fn verify_installed_backend_with_scope(
    dir: &Path,
    installed: &InstalledBackend,
    resolved: &ResolvedCatalogBackendPull,
    scope: InstalledBackendVerifyScope,
) -> Result<(), PullError> {
    let expected_vendor = backend_vendor_dirname(resolved.vendor)?;
    if installed.schema_version != INSTALLED_BACKEND_SCHEMA_VERSION
        || installed.backend_id != resolved.backend_id
        || installed.vendor != expected_vendor
        || installed.version != resolved.version
        || installed.host_abi != resolved.host_abi
        || installed.artifact_fingerprint != backend_artifact_fingerprint(resolved)
        || installed.files.len() != resolved.files.len()
        || !installed
            .files
            .iter()
            .zip(&resolved.files)
            .all(|(actual, expected)| installed_backend_file_identity_matches(actual, expected))
    {
        return Err(PullError::InvalidTarget {
            field: "backend.json",
            reason: "installed backend identity does not match the signed catalog entry"
                .to_string(),
        });
    }
    let expected_plugin = resolved
        .files
        .iter()
        .find(|file| file.role == CatalogBackendFileRole::Plugin)
        .ok_or(PullError::InvalidTarget {
            field: "backend.files",
            reason: "pack declares no plugin file".to_string(),
        })?;
    if installed.plugin_filename != expected_plugin.filename {
        return Err(PullError::InvalidTarget {
            field: "backend.json.plugin_filename",
            reason: "installed plugin does not match the signed catalog entry".to_string(),
        });
    }
    let mut allowed_relative_paths = BTreeSet::from(["backend.json".to_string()]);
    for (installed_file, catalog_file) in installed.files.iter().zip(&resolved.files) {
        if installed_file.materialized_files.is_empty() {
            return Err(PullError::InvalidTarget {
                field: "backend.json.materialized_files",
                reason: format!("'{}' materialized no files", catalog_file.filename),
            });
        }
        let mut seen = std::collections::BTreeSet::new();
        for materialized in &installed_file.materialized_files {
            validate_safe_relative_path(
                "backend.json.materialized_files.relative_path",
                &materialized.relative_path,
            )
            .map_err(|reason| PullError::InvalidTarget {
                field: "backend.json.materialized_files.relative_path",
                reason,
            })?;
            if !seen.insert(materialized.relative_path.as_str()) {
                return Err(PullError::InvalidTarget {
                    field: "backend.json.materialized_files",
                    reason: format!("duplicate path '{}'", materialized.relative_path),
                });
            }
            allowed_relative_paths.insert(materialized.relative_path.clone());
            if !should_hash_installed_backend_file(
                &materialized.relative_path,
                &installed.plugin_filename,
                scope,
            ) {
                continue;
            }
            let path = dir.join(&materialized.relative_path);
            let (actual_size, actual_sha256) = file_size_and_sha256(&path)?;
            if actual_size != materialized.size_bytes {
                return Err(PullError::SizeMismatch {
                    path,
                    expected: materialized.size_bytes,
                    actual: actual_size,
                });
            }
            if !actual_sha256.eq_ignore_ascii_case(&materialized.sha256) {
                return Err(PullError::ShaMismatch {
                    path,
                    expected: materialized.sha256.clone(),
                    actual: actual_sha256,
                });
            }
        }
        match catalog_file.role {
            CatalogBackendFileRole::Plugin | CatalogBackendFileRole::Runtime => {
                if installed_file.materialized_files.len() != 1
                    || installed_file.materialized_files[0].relative_path != catalog_file.filename
                    || installed_file.materialized_files[0].size_bytes != catalog_file.size_bytes
                    || !installed_file.materialized_files[0]
                        .sha256
                        .eq_ignore_ascii_case(&catalog_file.sha256)
                {
                    return Err(PullError::InvalidTarget {
                        field: "backend.json.materialized_files",
                        reason: format!(
                            "'{}' does not match its signed file identity",
                            catalog_file.filename
                        ),
                    });
                }
                preflight_backend_file(
                    &dir.join(&catalog_file.filename),
                    backend_file_format(catalog_file.role)?,
                )?;
            }
            CatalogBackendFileRole::Archive => {
                let expected_tree = catalog_file.extracted_tree_sha256.as_deref().ok_or(
                    PullError::InvalidTarget {
                        field: "backend.files.extracted_tree_sha256",
                        reason: format!(
                            "archive '{}' has no signed extracted tree digest",
                            catalog_file.filename
                        ),
                    },
                )?;
                let actual_tree = materialized_tree_sha256(&installed_file.materialized_files);
                if !actual_tree.eq_ignore_ascii_case(expected_tree) {
                    return Err(PullError::ShaMismatch {
                        path: dir.join(catalog_file.extract_subdir.as_deref().unwrap_or("")),
                        expected: expected_tree.to_string(),
                        actual: actual_tree,
                    });
                }
            }
            CatalogBackendFileRole::Unknown => {
                return Err(PullError::InvalidTarget {
                    field: "backend.files.role",
                    reason: "unknown backend file role".to_string(),
                });
            }
        }
    }
    match scope {
        InstalledBackendVerifyScope::Full => {
            // Installed packs are a closed file set. Windows LoadLibraryEx with
            // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR will load unsigned sibling DLLs
            // that never appear in the signed materialized list.
            for actual in collect_materialized_files(dir)? {
                if !allowed_relative_paths.contains(&actual.relative_path) {
                    return Err(PullError::UnexpectedInstalledBackendFile {
                        path: dir.join(&actual.relative_path),
                    });
                }
            }
        }
        InstalledBackendVerifyScope::LoadImages => {
            // Activation only has to refuse extra mapped images. Extra Tensile
            // objects are not a DllMain TOCTOU; hashing or enumerating them as
            // unexpected files is the install-path job.
            walk_installed_backend_files(dir, |path, relative_path| {
                if is_backend_load_image_relative_path(relative_path)
                    && !allowed_relative_paths.contains(relative_path)
                {
                    return Err(PullError::UnexpectedInstalledBackendFile {
                        path: path.to_path_buf(),
                    });
                }
                Ok(())
            })?;
        }
    }
    Ok(())
}

fn read_and_verify_installed_backend_with_scope(
    dir: &Path,
    resolved: &ResolvedCatalogBackendPull,
    scope: InstalledBackendVerifyScope,
) -> Result<InstalledBackend, PullError> {
    let marker_path = dir.join("backend.json");
    let text = fs::read_to_string(&marker_path).map_err(|source| PullError::Io {
        path: marker_path.clone(),
        source,
    })?;
    let mut installed =
        serde_json::from_str::<InstalledBackend>(&text).map_err(|source| PullError::ParseMeta {
            path: marker_path,
            source,
        })?;
    verify_installed_backend_with_scope(dir, &installed, resolved, scope)?;
    installed.dir = dir.to_path_buf();
    Ok(installed)
}

pub fn read_and_verify_installed_backend(
    dir: &Path,
    resolved: &ResolvedCatalogBackendPull,
) -> Result<InstalledBackend, PullError> {
    read_and_verify_installed_backend_with_scope(dir, resolved, InstalledBackendVerifyScope::Full)
}

/// Activation-path verification: identity + mapped images only.
///
/// Install and other callers must keep using [`read_and_verify_installed_backend`],
/// which remains [`InstalledBackendVerifyScope::Full`].
pub fn read_and_verify_installed_backend_for_activation(
    dir: &Path,
    resolved: &ResolvedCatalogBackendPull,
) -> Result<InstalledBackend, PullError> {
    read_and_verify_installed_backend_with_scope(
        dir,
        resolved,
        InstalledBackendVerifyScope::LoadImages,
    )
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

/// Installs one resolved pack while the caller holds the backend-store
/// mutation lock. This is intentionally crate-private: the only caller that
/// needs to extend the lock past installation is the production
/// install-and-activate transaction. Public install-only callers must use
/// [`install_backend_pack`], which acquires the same lock itself.
pub(crate) fn install_backend_pack_locked(
    resolved: &ResolvedCatalogBackendPull,
    home: &Path,
    progress: impl FnMut(PullProgress),
) -> Result<InstalledBackend, PullError> {
    let mut client = HttpDownloadClient::new()?;
    install_backend_pack_with_client_locked(resolved, home, &mut client, progress)
}

/// Conservative logical bytes that must remain reachable while `resolved` is
/// installed.  The count is owned by open-core because only the content store
/// knows which shared runtime objects back a pack.  Directory entries are
/// walked fail-closed and symlinks are rejected by `safe_tree_stats`; hard-link
/// aliases are deliberately counted at each protected path so a product shell
/// never under-budgets a filesystem that had to fall back to copies.
pub fn installed_backend_protected_bytes(
    resolved: &ResolvedCatalogBackendPull,
    home: impl AsRef<Path>,
) -> Result<u64, PullError> {
    let home = home.as_ref();
    let mut roots = BTreeSet::new();
    roots.insert(backend_pack_install_dir(home, resolved)?);
    for file in &resolved.files {
        if matches!(
            file.role,
            CatalogBackendFileRole::Runtime | CatalogBackendFileRole::Archive
        ) {
            roots.insert(backend_content_object_dir(home, file));
        }
    }

    protected_backend_roots_bytes(roots)
}

fn protected_backend_roots_bytes(
    roots: impl IntoIterator<Item = PathBuf>,
) -> Result<u64, PullError> {
    roots.into_iter().try_fold(0_u64, |total, root| {
        let bytes = safe_tree_stats(&root)
            .ok_or_else(|| PullError::RuntimeValidation {
                path: root,
                reason: "backend protected-byte tree is missing, unreadable, or contains a symlink"
                    .to_string(),
            })?
            .bytes;
        total.checked_add(bytes).ok_or(PullError::InvalidTarget {
            field: "backend protected bytes",
            reason: "logical byte count overflow".to_string(),
        })
    })
}

/// Downloads and verifies only the provider runtime/archive objects shared by
/// every target-scoped pack, while the caller holds the global backend-store
/// mutation lock. This is the HIP discovery bootstrap: no target-specific
/// plugin is downloaded or made loadable until the trusted runtime reports an
/// exact `gfx` target. The returned directories are immutable content-object
/// payload roots suitable for restricted dynamic-library lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PreparedBackendRuntimeFile {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PreparedBackendRuntimeObjects {
    pub dependency_dirs: Vec<PathBuf>,
    pub files: Vec<PreparedBackendRuntimeFile>,
}

pub(crate) fn prepare_backend_runtime_objects_locked(
    resolved: &ResolvedCatalogBackendPull,
    home: &Path,
    mut progress: impl FnMut(PullProgress),
) -> Result<PreparedBackendRuntimeObjects, PullError> {
    let mut client = HttpDownloadClient::new()?;
    let mut dependency_dirs = BTreeSet::new();
    let mut verified_files = Vec::new();
    let mut saw_runtime = false;
    for file in &resolved.files {
        match file.role {
            CatalogBackendFileRole::Runtime | CatalogBackendFileRole::Archive => {
                saw_runtime = true;
                let object = ensure_backend_content_object(&mut client, file, home, &mut progress)?;
                let payload = backend_content_object_dir(home, file).join("payload");
                for materialized in &object.files {
                    let relative = Path::new(&materialized.relative_path);
                    let parent = relative.parent().unwrap_or_else(|| Path::new(""));
                    dependency_dirs.insert(payload.join(parent));
                    verified_files.push(PreparedBackendRuntimeFile {
                        path: payload.join(relative),
                        size_bytes: materialized.size_bytes,
                        sha256: materialized.sha256.clone(),
                    });
                }
            }
            CatalogBackendFileRole::Plugin => {}
            CatalogBackendFileRole::Unknown => {
                return Err(PullError::InvalidTarget {
                    field: "backend.files.role",
                    reason: "unknown backend file role".to_string(),
                });
            }
        }
    }
    if !saw_runtime {
        return Err(PullError::InvalidTarget {
            field: "backend.files",
            reason: "provider discovery requires a signed shared runtime".to_string(),
        });
    }
    verified_files.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(PreparedBackendRuntimeObjects {
        dependency_dirs: dependency_dirs.into_iter().collect(),
        files: verified_files,
    })
}

fn install_backend_pack_with_client<C: DownloadClient>(
    resolved: &ResolvedCatalogBackendPull,
    home: &Path,
    client: &mut C,
    progress: impl FnMut(PullProgress),
) -> Result<InstalledBackend, PullError> {
    let _store_lock = BackendStoreMutationLock::acquire(home)?;
    install_backend_pack_with_client_locked(resolved, home, client, progress)
}

fn install_backend_pack_with_client_locked<C: DownloadClient>(
    resolved: &ResolvedCatalogBackendPull,
    home: &Path,
    client: &mut C,
    mut progress: impl FnMut(PullProgress),
) -> Result<InstalledBackend, PullError> {
    let vendor = backend_vendor_dirname(resolved.vendor)?;
    validate_backend_pack_version(&resolved.version)?;
    let plugin_filename = resolved
        .files
        .iter()
        .find(|file| file.role == CatalogBackendFileRole::Plugin)
        .map(|file| file.filename.clone())
        .ok_or(PullError::InvalidTarget {
            field: "backend.files",
            reason: "pack declares no plugin file".to_string(),
        })?;

    let dir = backend_pack_install_dir(home, resolved)?;
    let fingerprint = backend_artifact_fingerprint(resolved);
    let locks_root = home.join("backends").join(".locks");
    fs::create_dir_all(&locks_root).map_err(|source| PullError::Io {
        path: locks_root.clone(),
        source,
    })?;
    let _lock = BackendInstallLock::acquire(&locks_root.join(format!("{fingerprint}.lock")))?;

    let marker_path = dir.join("backend.json");
    if marker_path.is_file()
        && let Ok(existing) = read_and_verify_installed_backend(&dir, resolved)
    {
        progress(PullProgress::UsingInstalled { path: dir.clone() });
        return Ok(existing);
    }

    // Never expose a partially materialized pack at its final path. The
    // stable content-keyed staging directory preserves resumable partials
    // across NSIS/process replacement; the final same-volume rename is the
    // only point at which runtime discovery can observe this artifact.
    let staging_dir = backend_pack_staging_dir(home, resolved)?;
    fs::create_dir_all(&staging_dir).map_err(|source| PullError::Io {
        path: staging_dir.clone(),
        source,
    })?;

    let mut installed_files = Vec::new();
    for file in &resolved.files {
        let materialized_files = match file.role {
            CatalogBackendFileRole::Plugin => {
                let dest = staging_dir.join(&file.filename);
                download_backend_file(client, file, &dest, &mut progress)?;
                preflight_backend_file(&dest, backend_file_format(file.role)?)?;
                vec![InstalledBackendMaterializedFile {
                    relative_path: file.filename.clone(),
                    sha256: file.sha256.to_ascii_lowercase(),
                    size_bytes: file.size_bytes,
                }]
            }
            CatalogBackendFileRole::Runtime | CatalogBackendFileRole::Archive => {
                let object = ensure_backend_content_object(client, file, home, &mut progress)?;
                materialize_backend_content_object(home, &staging_dir, file, &object)?
            }
            CatalogBackendFileRole::Unknown => {
                return Err(PullError::InvalidTarget {
                    field: "backend.files.role",
                    reason: "unknown backend file role".to_string(),
                });
            }
        };
        installed_files.push(installed_backend_file(file, materialized_files));
    }

    let record = InstalledBackend {
        schema_version: INSTALLED_BACKEND_SCHEMA_VERSION,
        backend_id: resolved.backend_id.clone(),
        vendor: vendor.to_string(),
        version: resolved.version.clone(),
        host_abi: resolved.host_abi.clone(),
        artifact_fingerprint: backend_artifact_fingerprint(resolved),
        dir: staging_dir.clone(),
        plugin_filename,
        files: installed_files,
        installed_at_unix_seconds: unix_seconds_now(),
    };
    verify_installed_backend(&staging_dir, &record, resolved)?;
    let staging_marker_path = staging_dir.join("backend.json");
    let json =
        serde_json::to_string_pretty(&record).map_err(|source| PullError::SerializeMeta {
            path: staging_marker_path.clone(),
            source,
        })?;
    write_json_atomic(&staging_marker_path, &format!("{json}\n"))?;
    read_and_verify_installed_backend(&staging_dir, resolved)?;
    promote_backend_directory(&staging_dir, &dir, &fingerprint)?;
    read_and_verify_installed_backend(&dir, resolved)
}

fn promote_backend_directory(
    staging_dir: &Path,
    final_dir: &Path,
    fingerprint: &str,
) -> Result<(), PullError> {
    let final_parent = final_dir.parent().ok_or_else(|| PullError::InvalidTarget {
        field: "backend install path",
        reason: "final backend directory has no parent".to_string(),
    })?;
    fs::create_dir_all(final_parent).map_err(|source| PullError::Io {
        path: final_parent.to_path_buf(),
        source,
    })?;
    let displaced = staging_dir
        .parent()
        .expect("content-keyed staging directory has a parent")
        .join(format!(".replaced-{fingerprint}-{}", unix_seconds_now()));
    let had_previous = final_dir.exists();
    if had_previous {
        fs::rename(final_dir, &displaced).map_err(|source| PullError::Io {
            path: final_dir.to_path_buf(),
            source,
        })?;
    }
    if let Err(source) = fs::rename(staging_dir, final_dir) {
        if had_previous {
            let _ = fs::rename(&displaced, final_dir);
        }
        return Err(PullError::Io {
            path: final_dir.to_path_buf(),
            source,
        });
    }
    if had_previous {
        let _ = fs::remove_dir_all(displaced);
    }
    Ok(())
}

/// Reclaims backend-pack generations and shared runtime objects that are no
/// longer selected or explicitly retained by a caller (for example Desktop's
/// future-core pending candidate). The mutation lock makes this safe against
/// a concurrent installer or activation commit. Young artifacts are retained
/// for a bounded rollback/resume window; corrupt metadata never broadens a
/// deletion target and causes shared objects to be retained conservatively.
pub fn gc_backend_store(
    home: impl AsRef<Path>,
    keep_backend_ids: impl IntoIterator<Item = String>,
    min_age: Option<Duration>,
) -> Result<BackendStoreGcReport, PullError> {
    let home = home.as_ref();
    let _store_lock = BackendStoreMutationLock::acquire(home)?;
    gc_backend_store_locked(
        home,
        keep_backend_ids.into_iter().collect(),
        min_age.unwrap_or(DEFAULT_BACKEND_GC_MIN_AGE),
        unix_seconds_now(),
    )
}

fn gc_backend_store_locked(
    home: &Path,
    mut keep_backend_ids: BTreeSet<String>,
    min_age: Duration,
    now: u64,
) -> Result<BackendStoreGcReport, PullError> {
    let backends_root = home.join("backends");
    fs::create_dir_all(&backends_root).map_err(|source| PullError::Io {
        path: backends_root.clone(),
        source,
    })?;
    let active = crate::backend_distribution::read_activated_backend(home).map_err(|error| {
        PullError::InvalidTarget {
            field: "backends/active.json",
            reason: error.to_string(),
        }
    })?;
    if let Some(active) = active.as_ref() {
        keep_backend_ids.insert(active.backend_id.clone());
    }
    let cutoff = now.saturating_sub(min_age.as_secs());
    let mut report = BackendStoreGcReport {
        schema_version: BACKEND_STORE_SCHEMA_VERSION,
        retained_backend_ids: keep_backend_ids.iter().cloned().collect(),
        removed_pack_directories: 0,
        removed_staging_directories: 0,
        removed_content_objects: 0,
        reclaimed_bytes: 0,
        deferred_paths: Vec::new(),
    };
    let mut referenced_objects = BTreeSet::new();
    let mut retain_all_objects = false;

    for vendor in ["cpu", "vulkan", "cuda", "hip"] {
        let vendor_dir = backends_root.join(vendor);
        for version_dir in safe_child_directories(&vendor_dir, &mut report.deferred_paths)? {
            for pack_dir in safe_child_directories(&version_dir, &mut report.deferred_paths)? {
                let marker_path = pack_dir.join("backend.json");
                let installed = fs::read_to_string(&marker_path)
                    .ok()
                    .and_then(|text| serde_json::from_str::<InstalledBackend>(&text).ok())
                    .filter(|installed| {
                        installed.schema_version == INSTALLED_BACKEND_SCHEMA_VERSION
                            && installed.vendor == vendor
                            && installed.version
                                == version_dir
                                    .file_name()
                                    .and_then(|name| name.to_str())
                                    .unwrap_or_default()
                            && installed.artifact_fingerprint
                                == pack_dir
                                    .file_name()
                                    .and_then(|name| name.to_str())
                                    .unwrap_or_default()
                    });
                let active_match = installed.as_ref().is_some_and(|installed| {
                    active.as_ref().is_some_and(|active| {
                        active.backend_id == installed.backend_id
                            && active.artifact_fingerprint == installed.artifact_fingerprint
                    })
                });
                let caller_kept = installed
                    .as_ref()
                    .is_some_and(|installed| keep_backend_ids.contains(&installed.backend_id));
                let modified = safe_tree_stats(&pack_dir)
                    .map(|stats| stats.newest_modified_unix_seconds)
                    .unwrap_or(now);
                let young = installed
                    .as_ref()
                    .map(|installed| installed.installed_at_unix_seconds.max(modified))
                    .unwrap_or(modified)
                    > cutoff;
                if active_match || caller_kept || young {
                    if let Some(installed) = installed {
                        for file in installed.files {
                            if matches!(
                                file.role,
                                CatalogBackendFileRole::Runtime | CatalogBackendFileRole::Archive
                            ) {
                                referenced_objects.insert(file.sha256.to_ascii_lowercase());
                            }
                        }
                    } else {
                        // A young/caller-visible but malformed pack is not
                        // loadable, yet deleting a potentially shared object
                        // based on metadata we cannot prove would be unsafe.
                        retain_all_objects = true;
                    }
                    continue;
                }
                if let Some(bytes) = remove_backend_gc_directory(&pack_dir, &mut report) {
                    report.removed_pack_directories =
                        report.removed_pack_directories.saturating_add(1);
                    report.reclaimed_bytes = report.reclaimed_bytes.saturating_add(bytes);
                }
            }
            remove_empty_directory(&version_dir);
        }
        remove_empty_directory(&vendor_dir);
    }

    let staging_root = backends_root.join(".staging");
    for candidate in safe_gc_leaf_directories(&staging_root, &mut report.deferred_paths)? {
        let modified = safe_tree_stats(&candidate)
            .map(|stats| stats.newest_modified_unix_seconds)
            .unwrap_or(now);
        if modified <= cutoff
            && let Some(bytes) = remove_backend_gc_directory(&candidate, &mut report)
        {
            report.removed_staging_directories =
                report.removed_staging_directories.saturating_add(1);
            report.reclaimed_bytes = report.reclaimed_bytes.saturating_add(bytes);
        }
    }
    prune_empty_directories(&staging_root);

    if !retain_all_objects {
        let objects_root = backends_root.join("_objects");
        let object_staging_root = objects_root.join(".staging");
        for candidate in safe_gc_leaf_directories(&object_staging_root, &mut report.deferred_paths)?
        {
            let modified = safe_tree_stats(&candidate)
                .map(|stats| stats.newest_modified_unix_seconds)
                .unwrap_or(now);
            if modified <= cutoff
                && let Some(bytes) = remove_backend_gc_directory(&candidate, &mut report)
            {
                report.removed_staging_directories =
                    report.removed_staging_directories.saturating_add(1);
                report.reclaimed_bytes = report.reclaimed_bytes.saturating_add(bytes);
            }
        }
        prune_empty_directories(&object_staging_root);
        for object_dir in safe_child_directories(&objects_root, &mut report.deferred_paths)? {
            let Some(name) = object_dir.file_name().and_then(|name| name.to_str()) else {
                report.deferred_paths.push(object_dir);
                continue;
            };
            if name == ".staging" || referenced_objects.contains(&name.to_ascii_lowercase()) {
                continue;
            }
            let modified = safe_tree_stats(&object_dir)
                .map(|stats| stats.newest_modified_unix_seconds)
                .unwrap_or(now);
            if modified <= cutoff
                && let Some(bytes) = remove_backend_gc_directory(&object_dir, &mut report)
            {
                report.removed_content_objects = report.removed_content_objects.saturating_add(1);
                report.reclaimed_bytes = report.reclaimed_bytes.saturating_add(bytes);
            }
        }
    }

    report.deferred_paths.sort();
    report.deferred_paths.dedup();
    Ok(report)
}

#[derive(Clone, Copy)]
struct BackendTreeStats {
    bytes: u64,
    newest_modified_unix_seconds: u64,
}

fn safe_tree_stats(root: &Path) -> Option<BackendTreeStats> {
    let mut pending = vec![root.to_path_buf()];
    let mut bytes = 0_u64;
    let mut newest = 0_u64;
    while let Some(path) = pending.pop() {
        let metadata = fs::symlink_metadata(&path).ok()?;
        if metadata.file_type().is_symlink() {
            return None;
        }
        newest = newest.max(
            metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_secs())
                .unwrap_or(0),
        );
        if metadata.is_file() {
            bytes = bytes.checked_add(metadata.len())?;
        } else if metadata.is_dir() {
            for entry in fs::read_dir(&path).ok()? {
                pending.push(entry.ok()?.path());
            }
        } else {
            return None;
        }
    }
    Some(BackendTreeStats {
        bytes,
        newest_modified_unix_seconds: newest,
    })
}

fn safe_child_directories(
    root: &Path,
    deferred: &mut Vec<PathBuf>,
) -> Result<Vec<PathBuf>, PullError> {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => {
            return Err(PullError::Io {
                path: root.to_path_buf(),
                source,
            });
        }
    };
    let mut directories = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| PullError::Io {
            path: root.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(|source| PullError::Io {
            path: path.clone(),
            source,
        })?;
        if metadata.is_dir() && !metadata.file_type().is_symlink() {
            directories.push(path);
        } else {
            deferred.push(path);
        }
    }
    directories.sort();
    Ok(directories)
}

fn safe_gc_leaf_directories(
    root: &Path,
    deferred: &mut Vec<PathBuf>,
) -> Result<Vec<PathBuf>, PullError> {
    let mut leaves = Vec::new();
    let mut pending = safe_child_directories(root, deferred)?;
    while let Some(path) = pending.pop() {
        let mut children = Vec::new();
        let entries = fs::read_dir(&path).map_err(|source| PullError::Io {
            path: path.clone(),
            source,
        })?;
        for entry in entries {
            let entry = entry.map_err(|source| PullError::Io {
                path: path.clone(),
                source,
            })?;
            let child = entry.path();
            let metadata = fs::symlink_metadata(&child).map_err(|source| PullError::Io {
                path: child.clone(),
                source,
            })?;
            if metadata.file_type().is_symlink() || (!metadata.is_dir() && !metadata.is_file()) {
                deferred.push(child);
            } else if metadata.is_dir() {
                children.push(child);
            }
        }
        if children.is_empty() {
            leaves.push(path);
        } else {
            pending.extend(children);
        }
    }
    leaves.sort();
    Ok(leaves)
}

fn remove_backend_gc_directory(path: &Path, report: &mut BackendStoreGcReport) -> Option<u64> {
    let Some(stats) = safe_tree_stats(path) else {
        report.deferred_paths.push(path.to_path_buf());
        return None;
    };
    match fs::remove_dir_all(path) {
        Ok(()) => Some(stats.bytes),
        Err(_) => {
            report.deferred_paths.push(path.to_path_buf());
            None
        }
    }
}

fn remove_empty_directory(path: &Path) {
    let _ = fs::remove_dir(path);
}

fn prune_empty_directories(root: &Path) {
    let mut directories = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(path) = pending.pop() {
        directories.push(path.clone());
        if let Ok(entries) = fs::read_dir(&path) {
            pending.extend(entries.flatten().filter_map(|entry| {
                let path = entry.path();
                fs::symlink_metadata(&path)
                    .ok()
                    .filter(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
                    .map(|_| path)
            }));
        }
    }
    directories.sort_by_key(|path| std::cmp::Reverse(path.components().count()));
    for directory in directories {
        remove_empty_directory(&directory);
    }
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
/// The partial and its metadata are keyed by the signed content identity, so
/// a replacement Desktop process can resume after NSIS terminates the old
/// process. The signed sha256 remains the final authority; ETag is only a
/// transport guard that prevents appending bytes from two representations.
fn download_backend_file<C: DownloadClient>(
    client: &mut C,
    file: &CatalogBackendFile,
    dest: &Path,
    progress: &mut impl FnMut(PullProgress),
) -> Result<(), PullError> {
    if backend_file_matches(dest, file) {
        return Ok(());
    }
    if let Some(local_path) = file.url.strip_prefix("file://") {
        return copy_local_backend_file(Path::new(local_path), dest, file, progress);
    }
    // The parent pack/object directory is already keyed by the full artifact
    // digest. Keep the leaf short enough for Windows MAX_PATH-era tools while
    // the metadata still repeats and verifies the complete sha256 identity.
    let (partial, partial_meta) = backend_partial_paths(dest)?;

    let mut attempt = 0_usize;
    let (partial_len, mut expected_etag) =
        prepare_backend_partial_for_resume(file, &partial, &partial_meta)?;
    if partial_len == file.size_bytes {
        let verified = file_size_and_sha256(&partial).is_ok_and(|(size, sha256)| {
            size == file.size_bytes && sha256.eq_ignore_ascii_case(&file.sha256)
        });
        if verified {
            atomic_file::replace_file_atomically(&partial, dest).map_err(|source| {
                PullError::Io {
                    path: dest.to_path_buf(),
                    source,
                }
            })?;
            let _ = fs::remove_file(partial_meta);
            return Ok(());
        }
        discard_backend_partial(&partial, &partial_meta);
        expected_etag = None;
    }
    loop {
        match download_backend_file_attempt(
            client,
            file,
            dest,
            &partial,
            &partial_meta,
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

fn copy_local_backend_file(
    source: &Path,
    dest: &Path,
    file: &CatalogBackendFile,
    progress: &mut impl FnMut(PullProgress),
) -> Result<(), PullError> {
    progress(PullProgress::DownloadStarted {
        bytes_total: file.size_bytes,
        resume_from: 0,
    });
    let (size, sha256) = file_size_and_sha256(source)?;
    if size != file.size_bytes || !sha256.eq_ignore_ascii_case(&file.sha256) {
        return Err(PullError::InvalidTarget {
            field: "backend.files",
            reason: format!(
                "local file '{}' does not match the signed size/sha256",
                source.display()
            ),
        });
    }
    progress(PullProgress::Downloading {
        bytes_done: size,
        bytes_total: size,
    });
    progress(PullProgress::Verifying { bytes_done: size });
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|source_error| PullError::Io {
            path: parent.to_path_buf(),
            source: source_error,
        })?;
    }
    let staging = dest.with_extension("local-copy");
    fs::copy(source, &staging).map_err(|source_error| PullError::Io {
        path: staging.clone(),
        source: source_error,
    })?;
    atomic_file::replace_file_atomically(&staging, dest).map_err(|source_error| PullError::Io {
        path: dest.to_path_buf(),
        source: source_error,
    })?;
    Ok(())
}

fn discard_backend_partial(partial: &Path, partial_meta: &Path) {
    let _ = fs::remove_file(partial);
    let _ = fs::remove_file(partial_meta);
}

fn prepare_backend_partial_for_resume(
    file: &CatalogBackendFile,
    partial: &Path,
    partial_meta: &Path,
) -> Result<(u64, Option<String>), PullError> {
    if !partial.is_file() || !partial_meta.is_file() {
        discard_backend_partial(partial, partial_meta);
        return Ok((0, None));
    }
    let meta_text = match fs::read_to_string(partial_meta) {
        Ok(text) => text,
        Err(_) => {
            discard_backend_partial(partial, partial_meta);
            return Ok((0, None));
        }
    };
    let meta = match serde_json::from_str::<BackendPartialMeta>(&meta_text) {
        Ok(meta) if meta.matches_file(file) => meta,
        _ => {
            discard_backend_partial(partial, partial_meta);
            return Ok((0, None));
        }
    };
    let partial_len = fs::metadata(partial)
        .map_err(|source| PullError::Io {
            path: partial.to_path_buf(),
            source,
        })?
        .len();
    if partial_len > file.size_bytes {
        discard_backend_partial(partial, partial_meta);
        return Ok((0, None));
    }
    if meta.bytes_done != partial_len {
        write_backend_partial_meta(partial_meta, file, meta.etag.clone(), partial_len)?;
    }
    Ok((partial_len, meta.etag))
}

fn download_backend_file_attempt<C: DownloadClient>(
    client: &mut C,
    file: &CatalogBackendFile,
    dest: &Path,
    partial: &Path,
    partial_meta: &Path,
    expected_etag: &mut Option<String>,
    progress: &mut impl FnMut(PullProgress),
) -> Result<(), PullError> {
    let (resume_from, persisted_etag) =
        prepare_backend_partial_for_resume(file, partial, partial_meta)?;
    if expected_etag.is_none() {
        *expected_etag = persisted_etag;
    }
    let response = client.open(
        &file.url,
        (resume_from > 0).then(|| ByteRange::from_start(resume_from)),
    )?;

    if resume_from > 0
        && let (Some(expected), Some(actual)) = (expected_etag.as_deref(), response.etag.as_deref())
        && expected != actual
    {
        discard_backend_partial(partial, partial_meta);
        *expected_etag = None;
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
        discard_backend_partial(partial, partial_meta);
        *expected_etag = None;
        return Err(PullError::RestartedPartial {
            url: file.url.clone(),
        });
    }
    let actual_resume = if append { resume_from } else { 0 };
    if resume_from > 0 && !append {
        discard_backend_partial(partial, partial_meta);
        *expected_etag = response.etag.clone();
    }
    if let Some(content_length) = response.content_length {
        let expected_body = file.size_bytes.saturating_sub(actual_resume);
        if content_length != expected_body {
            discard_backend_partial(partial, partial_meta);
            *expected_etag = None;
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
    write_backend_partial_meta(partial_meta, file, expected_etag.clone(), bytes_done)?;
    let mut last_meta_bytes = bytes_done;
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
        if bytes_done.saturating_sub(last_meta_bytes) >= METADATA_WRITE_INTERVAL_BYTES {
            out.flush().map_err(|source| PullError::Io {
                path: partial.to_path_buf(),
                source,
            })?;
            write_backend_partial_meta(partial_meta, file, expected_etag.clone(), bytes_done)?;
            last_meta_bytes = bytes_done;
        }
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
    write_backend_partial_meta(partial_meta, file, expected_etag.clone(), bytes_done)?;
    let actual = format!("{:x}", hasher.finalize());
    if actual != file.sha256 {
        discard_backend_partial(partial, partial_meta);
        *expected_etag = None;
        return Err(PullError::ShaMismatch {
            path: dest.to_path_buf(),
            expected: file.sha256.clone(),
            actual,
        });
    }
    atomic_file::replace_file_atomically(partial, dest).map_err(|source| PullError::Io {
        path: dest.to_path_buf(),
        source,
    })?;
    let _ = fs::remove_file(partial_meta);
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
