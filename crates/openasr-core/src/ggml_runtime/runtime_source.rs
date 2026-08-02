use std::{
    collections::HashMap,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    sync::{Arc, Mutex, OnceLock},
};

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

use memmap2::Mmap;
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::{GgmlPackageFormat, GgmlPackageProbe, GgmlPackageProbeError, probe_ggml_package_path};

/// Strong OS file identity: device, inode, length, and the *full*
/// nanosecond mtime of an already-opened file. Never truncated to whole
/// seconds -- see `models::runtime_cache_coordinator`'s module doc comment
/// for the audited bug (a `(len, whole-second mtime)` memo key) this type
/// replaces.
///
/// Always derived from an open file handle's `metadata()` (an `fstat` on the
/// held fd), never from a fresh `stat` on a path: a path-based stat can
/// observe a *different* file than the one actually opened if something
/// replaces the path in between, which is exactly the class of race this
/// contract exists to close.
///
/// `dev`/`ino` are only available through `std::os::unix::fs::MetadataExt`;
/// non-unix targets fall back to `(len, mtime)` alone, which is narrower
/// (two different files could theoretically share a length and mtime) but
/// still nanosecond-precise, unlike the bug being fixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StrongFileIdentity {
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
    len: u64,
    mtime_secs: u64,
    mtime_nanos: u32,
}

impl StrongFileIdentity {
    /// `None` when any needed metadata field cannot be read (including a
    /// pre-1970 mtime, which cannot be represented here) -- callers must fail
    /// closed rather than trust a partial identity.
    pub(crate) fn of(metadata: &std::fs::Metadata) -> Option<Self> {
        let modified = metadata.modified().ok()?;
        let since_epoch = modified.duration_since(std::time::UNIX_EPOCH).ok()?;
        Some(Self {
            #[cfg(unix)]
            dev: metadata.dev(),
            #[cfg(unix)]
            ino: metadata.ino(),
            len: metadata.len(),
            mtime_secs: since_epoch.as_secs(),
            mtime_nanos: since_epoch.subsec_nanos(),
        })
    }
}

/// Process-wide memo: canonical path -> (strong identity at last hash,
/// `sha256:<hex>` digest). Shared by every *hashing* content id resolver in
/// the crate -- a `GgmlRuntimeSource`'s fd-derived identity and
/// `models::runtime_cache_coordinator`'s narrow path-based pre-replace
/// snapshot both go through [`resolve_content_id`] -- so hashing a path once
/// through either warms the other's lookup too.
///
/// Sealed content-addressed objects anchored under the resolved model store
/// never arrive here: both entry points answer those from the digest in the
/// path via `content_store::trusted_object_digest` before any hashing is
/// considered. This memo is the slow path for everything that gate declines
/// -- including an object-shaped path outside the store, which the anchor
/// check routes here rather than trusting.
fn content_id_memo() -> &'static Mutex<HashMap<PathBuf, (StrongFileIdentity, String)>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, (StrongFileIdentity, String)>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Resolves a `sha256:<hex>` content id for `canonical_path`, memoized by
/// [`StrongFileIdentity`].
///
/// Warm path: a memo hit whose stored identity matches `identity` exactly
/// returns the cached digest without ever calling `hash_hex` -- this is what
/// keeps identity resolution from paying a full-file hash on every call.
/// Cold path (first call for this path, or the identity changed): calls
/// `hash_hex` once and memoizes the result. `hash_hex` returning `None`
/// (unreadable) is never memoized, so a transient read failure cannot poison
/// the cache for a path that becomes readable again with the same identity.
pub(crate) fn resolve_content_id(
    canonical_path: &Path,
    identity: StrongFileIdentity,
    hash_hex: impl FnOnce() -> Option<String>,
) -> String {
    let cache = content_id_memo();
    if let Ok(guard) = cache.lock()
        && let Some((cached_identity, content_id)) = guard.get(canonical_path)
        && *cached_identity == identity
    {
        return content_id.clone();
    }

    let content_id = match hash_hex() {
        Some(hex) => format!("sha256:{hex}"),
        None => unreadable_content_id(canonical_path),
    };
    if content_id.starts_with("sha256:")
        && let Ok(mut guard) = cache.lock()
    {
        guard.insert(canonical_path.to_path_buf(), (identity, content_id.clone()));
    }
    content_id
}

pub(crate) fn unreadable_content_id(path: &Path) -> String {
    static UNREADABLE_COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "unreadable:{}:{}",
        path.display(),
        UNREADABLE_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

/// A validated ggml runtime source: the file has been opened and mapped
/// exactly once. Its content id (full-file sha256) is derived from that same
/// mapping, lazily, the first time a caller actually asks for one.
///
/// This is the fix for a reopen TOCTOU that used to exist between building a
/// [`super::GgufTensorIndex`] (path-based) and mapping tensor *data*
/// (previously a fresh, separate `File::open` of the same path): metadata,
/// the tensor index, and the mapped weight bytes could come from different
/// file generations if the pack was replaced between the two opens. Holding
/// the open mapping here and threading it through
/// [`super::GgufTensorDataReader::from_runtime_source`] means the bytes a
/// caller hashes for identity are the exact same bytes later read for
/// weights -- there is no second open to race against.
///
/// # Content identity: hot/cold split
///
/// [`GgmlRuntimeSource::content_id`] resolves in two tiers:
///
/// * **Trusted (no hashing).** When the source's path has the content store's
///   object layout (`.../objects/sha256/<digest>/content`), falls under this
///   process's own resolved model-store root
///   (`content_store::default_models_root`), and the file was read-only at
///   open time, the digest is taken straight from the path via
///   `content_store::trusted_object_digest`. That trust is not a shortcut
///   around integrity -- it rests on `content_store`'s integrity chain: the
///   digest was established once over the bytes actually written (and checked
///   against the signed catalog above the store), the object has been sealed
///   read-only since, and `openasr model-pack verify` can re-prove the claim
///   on demand. The models-root anchor is what stops the shape alone from
///   being enough: a file elsewhere on disk laid out to look like an object
///   (a user-supplied pack, a dev fixture, an extracted archive) was never
///   admitted by this store and must not be trusted just because it parses.
///   Re-hashing a multi-gigabyte pack on every process start to re-derive
///   what admission already established is the exact per-request
///   full-file-sha256 cost this split exists to remove.
/// * **Hashed (lazy, memoized).** Any other path -- a user-supplied pack, an
///   unsealed object, a dev fixture, an object-shaped path outside the
///   resolved model store, or a process with no resolvable
///   `default_models_root` at all -- hashes the mapping this source already
///   holds open, once, and memoizes the result by [`StrongFileIdentity`]
///   through [`resolve_content_id`]. A seal lost to a permissions-stripping
///   backup restore therefore degrades gracefully: hashing slow path until
///   `verify` re-seals, never a wrong id.
///
/// The id is deliberately **not** computed at validation time on the hashed
/// tier: this constructor sits on the per-request admission path (see
/// `validate_local_native_runtime_source`), and only a caller that actually
/// calls [`GgmlRuntimeSource::content_id`] ever pays anything -- callers that
/// only need [`GgmlRuntimeSource::path`] / [`GgmlRuntimeSource::package_probe`]
/// (the common case) never do. The trusted tier costs a path-shape check and
/// nothing else.
///
/// `path()` is downgraded to an admission / diagnostics / GC / fixture-lookup
/// helper: it must never be re-derived into a content identity by a caller
/// that already holds a `GgmlRuntimeSource` (use [`GgmlRuntimeSource::content_id`]
/// instead).
pub struct GgmlRuntimeSource {
    path: PathBuf,
    package_probe: GgmlPackageProbe,
    /// The same file descriptor the mapping came from. Small auxiliary-model
    /// request plans use it to make an anonymous immutable copy only after
    /// audio validation and memory admission. A mutex serializes the shared
    /// file cursor across cloned plans; mmap readers never take this lock.
    file: Option<Arc<Mutex<File>>>,
    mmap: Arc<Mmap>,
    /// Captured from `file.metadata()` (an `fstat` on the fd this source
    /// opened) at validation time -- never from a later `stat` on `path`.
    /// This is the identity the hashed tier of
    /// [`GgmlRuntimeSource::content_id`] memoizes against, so the digest it
    /// returns is provably a digest of exactly the bytes in `mmap`, not of
    /// whatever happens to be at `path` right now.
    stat_identity: StrongFileIdentity,
    /// The seal observed on the same fd at open time: the file was read-only.
    /// Gates the trusted tier of [`GgmlRuntimeSource::content_id`] -- a
    /// content-addressed object answers with the digest in its path only
    /// while this holds; anything writable is hashed instead.
    opened_read_only: bool,
    /// `sha256:<hex>` content id of the full mapped file. Computed once,
    /// lazily, either trusted from the object path (no I/O) or hashed from
    /// `mmap` -- never by re-opening `path`.
    content_id: OnceLock<String>,
}

impl Clone for GgmlRuntimeSource {
    fn clone(&self) -> Self {
        let content_id = OnceLock::new();
        if let Some(existing) = self.content_id.get() {
            // Best-effort: propagate an already-computed id so cloning a
            // source that already paid the hash cost does not force the
            // clone to pay it again. Losing a race here just means the clone
            // lazily recomputes (same answer, same bytes) instead of reusing
            // the value -- never a correctness issue.
            let _ = content_id.set(existing.clone());
        }
        Self {
            path: self.path.clone(),
            package_probe: self.package_probe.clone(),
            file: self.file.as_ref().map(Arc::clone),
            mmap: Arc::clone(&self.mmap),
            stat_identity: self.stat_identity,
            opened_read_only: self.opened_read_only,
            content_id,
        }
    }
}

impl std::fmt::Debug for GgmlRuntimeSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgmlRuntimeSource")
            .field("path", &self.path)
            .field("package_probe", &self.package_probe)
            .field("content_id", &self.content_id.get())
            .field("opened_read_only", &self.opened_read_only)
            .field("mmap_len", &self.mmap.len())
            .finish()
    }
}

// Equality is defined on admission identity (path + probe), not on the
// mapping or the lazily-computed content id: `Mmap` has no `PartialEq`, and
// forcing the hash just to compare two sources would defeat the whole point
// of making it lazy. Nothing in this crate compares `GgmlRuntimeSource` for
// content equality; callers that need a content proof use `content_id()`
// directly.
impl PartialEq for GgmlRuntimeSource {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path && self.package_probe == other.package_probe
    }
}

impl Eq for GgmlRuntimeSource {}

impl GgmlRuntimeSource {
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Byte length of the exact open file generation pinned by this source.
    /// Prefer this over re-statting `path` when admission must describe the
    /// source a runtime is about to materialize. Auxiliary-model callers that
    /// require byte stability across deferred materialization additionally
    /// call [`Self::immutable_snapshot_matching_content_id`].
    pub fn byte_len(&self) -> u64 {
        self.mmap.len().try_into().unwrap_or(u64::MAX)
    }

    pub fn package_probe(&self) -> &GgmlPackageProbe {
        &self.package_probe
    }

    /// `sha256:<hex>` content id of the full mapped file. This is the
    /// identity authority for this source -- prefer it over re-deriving an id
    /// from [`GgmlRuntimeSource::path`]. Computed once, lazily, then cached
    /// on this instance; see the type-level "Content identity" docs for the
    /// two tiers.
    ///
    /// **Trusted tier (installed objects, no hashing):** a source opened
    /// read-only at the content store's object layout, under this process's
    /// own resolved model-store root, answers with the digest its own path
    /// names. Content addressing's premise is that the path *is* the
    /// checksum, and for a sealed object *inside the store this process
    /// resolves* that premise was proven once at admission and pinned in
    /// place by the read-only seal ever since -- so the digest is not being
    /// assumed here, it is being read off a verified, immutable fact. The
    /// models-root anchor (`content_store::default_models_root`) is what
    /// keeps a same-shaped path outside that store -- which was never
    /// admitted or hashed by anything -- from being trusted just because it
    /// parses. This is the same trust `open_declared_lease` takes on the load
    /// path; full re-verification remains `model-pack verify`'s job.
    ///
    /// **Hashed tier (everything else, at most once per process):** any
    /// source the trusted gate declines -- a non-object path, an object
    /// outside the resolved model store, an object whose seal was lost, or a
    /// process with no resolvable model-store root at all -- hashes the
    /// mapping this source already holds open (never a fresh `File::open` of
    /// `path`), memoized by `stat_identity`. A memo hit never touches `mmap`
    /// at all; a genuine cold miss hashes exactly once per `StrongFileIdentity`
    /// per process, not once per call.
    pub fn content_id(&self) -> &str {
        self.content_id.get_or_init(|| {
            if let Some(models_root) = crate::content_store::default_models_root()
                && let Some(digest) = crate::content_store::trusted_object_digest(
                    &self.path,
                    self.opened_read_only,
                    &models_root,
                )
            {
                return format!("sha256:{digest}");
            }
            resolve_content_id(&self.path, self.stat_identity, || {
                Some(format!("{:x}", Sha256::digest(&self.mmap[..])))
            })
        })
    }

    /// The open mapping backing this source. Sharing this `Arc` (rather than
    /// re-opening `path()`) is what lets metadata / tensor-index / weight
    /// readers agree on exactly the bytes this source's `content_id` was
    /// computed from.
    pub(crate) fn backing_mmap(&self) -> Arc<Mmap> {
        Arc::clone(&self.mmap)
    }

    /// Process-local identity of the already-open mapping. Clones of this
    /// runtime source retain the same `Arc<Mmap>` and therefore the same value;
    /// a separately admitted file gets a distinct value even at the same path.
    /// This is suitable for ephemeral weak caches that must not force the
    /// potentially expensive full-file `content_id()` hash.
    pub(crate) fn backing_mmap_identity(&self) -> usize {
        Arc::as_ptr(&self.mmap) as usize
    }

    /// Copy this source's currently held file generation into an anonymous,
    /// read-only mapping. This is intentionally opt-in: copying a multi-GiB
    /// ASR pack would defeat mmap's demand paging, while the small auxiliary
    /// Voice ID packs need a byte-stable request snapshot across the gap
    /// between preflight/admission and deferred graph materialization.
    ///
    /// The copy reads through the already-open descriptor, never `path()`. If
    /// an in-place rewrite raced preflight, callers compare the returned
    /// content id with their prepared key and fail closed before parsing or
    /// constructing a graph. Once returned, later truncation or rewriting of
    /// the original inode cannot affect this anonymous mapping.
    pub(crate) fn immutable_snapshot(&self) -> Result<Self, GgmlRuntimeSourcePathError> {
        let Some(source_file) = self.file.as_ref() else {
            return Ok(self.clone());
        };
        let mut file = source_file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        file.seek(SeekFrom::Start(0)).map_err(|source| {
            GgmlRuntimeSourcePathError::SnapshotFile {
                path: self.path.clone(),
                source,
            }
        })?;
        let mut bytes = Vec::with_capacity(self.mmap.len());
        file.read_to_end(&mut bytes).map_err(|source| {
            GgmlRuntimeSourcePathError::SnapshotFile {
                path: self.path.clone(),
                source,
            }
        })?;
        if bytes.len() != self.mmap.len() {
            return Err(GgmlRuntimeSourcePathError::SnapshotLengthChanged {
                path: self.path.clone(),
                expected: self.mmap.len(),
                actual: bytes.len(),
            });
        }
        drop(file);

        let mut owned = memmap2::MmapMut::map_anon(bytes.len()).map_err(|source| {
            GgmlRuntimeSourcePathError::MapFile {
                path: self.path.clone(),
                source,
            }
        })?;
        owned.copy_from_slice(&bytes);
        let mmap =
            owned
                .make_read_only()
                .map_err(|source| GgmlRuntimeSourcePathError::MapFile {
                    path: self.path.clone(),
                    source,
                })?;
        let content_id = OnceLock::new();
        content_id
            .set(format!("sha256:{:x}", Sha256::digest(&mmap[..])))
            .expect("fresh immutable runtime snapshot content id");
        Ok(Self {
            path: self.path.clone(),
            package_probe: self.package_probe.clone(),
            file: None,
            mmap: Arc::new(mmap),
            stat_identity: self.stat_identity,
            opened_read_only: true,
            content_id,
        })
    }

    /// Create an immutable auxiliary-model snapshot and prove it is still the
    /// content admitted during preflight. A rewrite between those phases is a
    /// typed fail-closed error, never permission to parse different bytes or
    /// fall back to another provider.
    pub(crate) fn immutable_snapshot_matching_content_id(
        &self,
        expected: &str,
    ) -> Result<Self, GgmlRuntimeSourcePathError> {
        let snapshot = self.immutable_snapshot()?;
        let actual = snapshot.content_id().to_string();
        if actual != expected {
            return Err(GgmlRuntimeSourcePathError::SnapshotContentChanged {
                path: self.path.clone(),
                expected: expected.to_string(),
                actual,
            });
        }
        Ok(snapshot)
    }
}

#[derive(Debug, Error)]
pub enum GgmlRuntimeSourcePathError {
    #[error("ggml runtime source path does not exist: {path}")]
    PathDoesNotExist { path: String },
    #[error("ggml runtime source path must be local; remote URL is not supported: {path}")]
    RemoteUrlNotSupported { path: String },
    #[error("could not inspect ggml runtime source path '{path}': {source}")]
    Metadata {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("ggml runtime source path must be a regular file: {path}")]
    NotARegularFile { path: String },
    #[error(
        "ggml runtime source path '{path}' uses reserved OASR container magic; this container is not supported yet"
    )]
    ReservedOpenAsrContainer { path: PathBuf },
    #[error(transparent)]
    Probe(#[from] GgmlPackageProbeError),
    #[error("could not open ggml runtime source '{path}' for content identity: {source}")]
    OpenFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not map ggml runtime source '{path}' for content identity: {source}")]
    MapFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "could not copy ggml runtime source '{path}' into an immutable request snapshot: {source}"
    )]
    SnapshotFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "ggml runtime source '{path}' changed length while creating an immutable request snapshot: expected {expected} bytes, got {actual}"
    )]
    SnapshotLengthChanged {
        path: PathBuf,
        expected: usize,
        actual: usize,
    },
    #[error(
        "ggml runtime source '{path}' changed after request preflight: expected {expected}, got {actual}"
    )]
    SnapshotContentChanged {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error(
        "ggml runtime source '{path}' has a file identity this platform cannot represent (e.g. a pre-1970 mtime)"
    )]
    UnsupportedFileIdentity { path: PathBuf },
}

/// Validate a path as a loadable ggml runtime source.
///
/// This is the low-level *container* primitive: it checks the path is a local,
/// regular, readable file whose magic is a supported GGUF container (rejecting
/// remote URLs and the reserved native-OASR magic). It is intentionally
/// **extension-agnostic** — it accepts a GGUF-magic file regardless of whether it
/// is named `.oasr`, `.gguf`, or anything else — because it is the reader shared
/// by metadata/tensor-index loading and by internal GGUF test fixtures.
///
/// The user-facing `.oasr`-only naming contract is a *boundary* concern, enforced
/// where packs are produced or consumed by users: the CLI run/import paths and
/// the `convert_local_*_to_runtime_pack` converters (all via
/// [`crate::has_openasr_runtime_pack_extension`]). Keeping the extension gate at
/// the boundaries and the magic check here is deliberate layering, not drift.
pub fn validate_ggml_runtime_source_path(
    path: impl AsRef<Path>,
) -> Result<GgmlRuntimeSource, GgmlRuntimeSourcePathError> {
    let path = path.as_ref();
    let rendered = path.as_os_str().to_string_lossy().to_string();
    if !path.exists() {
        return if looks_like_remote_path(&rendered) {
            Err(GgmlRuntimeSourcePathError::RemoteUrlNotSupported { path: rendered })
        } else {
            Err(GgmlRuntimeSourcePathError::PathDoesNotExist { path: rendered })
        };
    }

    let metadata = fs::metadata(path).map_err(|source| GgmlRuntimeSourcePathError::Metadata {
        path: rendered,
        source,
    })?;
    if !metadata.is_file() {
        return Err(GgmlRuntimeSourcePathError::NotARegularFile {
            path: path.display().to_string(),
        });
    }

    let package_probe = probe_ggml_package_path(path)?;
    if package_probe.format == GgmlPackageFormat::UnsupportedOpenAsrContainerReserved {
        return Err(GgmlRuntimeSourcePathError::ReservedOpenAsrContainer {
            path: path.to_path_buf(),
        });
    }

    // Open and map once. Every later reader of this source (metadata,
    // tensor-index cross-checks, tensor data, and a lazily-computed
    // `content_id`) shares this `Arc<Mmap>` instead of re-opening `path` --
    // that is what makes `content_id()` an honest proof of the bytes a caller
    // actually reads, not just of whatever happened to be at `path` at
    // validation time. Mapping is a cheap `mmap(2)` (no read); the expensive
    // full-file hash only happens if/when `content_id()` is called.
    let file = File::open(path).map_err(|source| GgmlRuntimeSourcePathError::OpenFile {
        path: path.to_path_buf(),
        source,
    })?;
    // fstat on the fd we just opened, not a second `stat` on `path`: a
    // path-based stat here would itself be a race against whatever this
    // source is actually about to map.
    let fd_metadata = file
        .metadata()
        .map_err(|source| GgmlRuntimeSourcePathError::Metadata {
            path: path.display().to_string(),
            source,
        })?;
    let stat_identity = StrongFileIdentity::of(&fd_metadata).ok_or_else(|| {
        GgmlRuntimeSourcePathError::UnsupportedFileIdentity {
            path: path.to_path_buf(),
        }
    })?;
    // The content store's seal, observed on the same fd: content-addressed
    // objects are admitted read-only, so this is what lets `content_id()`
    // trust the digest in an object's path instead of re-hashing its bytes.
    let opened_read_only = fd_metadata.permissions().readonly();
    let mmap =
        unsafe { Mmap::map(&file) }.map_err(|source| GgmlRuntimeSourcePathError::MapFile {
            path: path.to_path_buf(),
            source,
        })?;

    Ok(GgmlRuntimeSource {
        path: path.to_path_buf(),
        package_probe,
        file: Some(Arc::new(Mutex::new(file))),
        mmap: Arc::new(mmap),
        stat_identity,
        opened_read_only,
        content_id: OnceLock::new(),
    })
}

fn looks_like_remote_path(value: &str) -> bool {
    let Some((scheme, _)) = value.split_once("://") else {
        return false;
    };
    !scheme.is_empty()
        && scheme.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.')
        })
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        time::Instant,
    };

    use sha2::{Digest, Sha256};
    use tempfile::{NamedTempFile, tempdir};

    use super::{
        GgmlPackageProbeError, GgmlRuntimeSourcePathError, validate_ggml_runtime_source_path,
    };
    use crate::GgmlPackageExtensionHint;

    fn write_magic_file(path: &Path, bytes: &[u8]) {
        fs::write(path, bytes).expect("write probe fixture");
    }

    #[test]
    fn validates_gguf_runtime_source_with_gguf_extension() {
        let file = NamedTempFile::new().expect("temp file");
        let runtime_path = file.path().with_extension("gguf");
        write_magic_file(&runtime_path, b"GGUFpayload");

        let source =
            validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
        assert_eq!(source.path(), runtime_path.as_path());
    }

    #[test]
    fn validates_gguf_runtime_source_with_oasr_extension() {
        let file = NamedTempFile::new().expect("temp file");
        let runtime_path = file.path().with_extension("oasr");
        write_magic_file(&runtime_path, b"GGUFpayload");

        let source =
            validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
        assert_eq!(source.path(), runtime_path.as_path());
        assert_eq!(
            source.package_probe().extension_hint,
            GgmlPackageExtensionHint::Oasr
        );
    }

    #[test]
    fn rejects_reserved_oasr_container_magic() {
        let file = NamedTempFile::new().expect("temp file");
        write_magic_file(file.path(), b"OASRpayload");

        let error =
            validate_ggml_runtime_source_path(file.path()).expect_err("reserved magic must fail");
        match error {
            GgmlRuntimeSourcePathError::ReservedOpenAsrContainer { .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn rejects_unknown_magic() {
        let file = NamedTempFile::new().expect("temp file");
        write_magic_file(file.path(), b"ABCDpayload");

        let error =
            validate_ggml_runtime_source_path(file.path()).expect_err("unknown magic must fail");
        match error {
            GgmlRuntimeSourcePathError::Probe(GgmlPackageProbeError::UnknownMagic {
                magic,
                ..
            }) => assert_eq!(magic, *b"ABCD"),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn rejects_short_file() {
        let file = NamedTempFile::new().expect("temp file");
        write_magic_file(file.path(), b"GG");

        let error = validate_ggml_runtime_source_path(file.path()).expect_err("short file fails");
        match error {
            GgmlRuntimeSourcePathError::Probe(GgmlPackageProbeError::FileTooShort {
                expected,
                actual,
                ..
            }) => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 2);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn rejects_directory() {
        let directory = tempdir().expect("temp dir");
        let error = validate_ggml_runtime_source_path(directory.path())
            .expect_err("directory must be rejected");
        match error {
            GgmlRuntimeSourcePathError::NotARegularFile { .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn rejects_remote_url_paths() {
        let error = validate_ggml_runtime_source_path(Path::new("https://example.invalid/model"))
            .expect_err("remote URL must fail");
        match error {
            GgmlRuntimeSourcePathError::RemoteUrlNotSupported { .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn rejects_missing_path() {
        let file = NamedTempFile::new().expect("temp file");
        let missing_path = file.path().to_path_buf();
        drop(file);

        let error = validate_ggml_runtime_source_path(&missing_path)
            .expect_err("missing path should be rejected");
        match error {
            GgmlRuntimeSourcePathError::PathDoesNotExist { .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[cfg(unix)]
    fn set_mtime(path: &Path, secs: i64, nanos: i64) {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let c_path = CString::new(path.as_os_str().as_bytes()).expect("path cstring");
        let times = [
            libc::timespec {
                tv_sec: secs as libc::time_t,
                tv_nsec: libc::UTIME_OMIT,
            },
            libc::timespec {
                tv_sec: secs as libc::time_t,
                tv_nsec: nanos as _,
            },
        ];
        let rc = unsafe { libc::utimensat(libc::AT_FDCWD, c_path.as_ptr(), times.as_ptr(), 0) };
        assert_eq!(
            rc,
            0,
            "utimensat failed: {}",
            std::io::Error::last_os_error()
        );
    }

    #[test]
    fn content_id_misses_same_path_byte_replacement() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("same-path.gguf");
        write_magic_file(&path, b"GGUFcontent-a-bytes");
        let id_a = validate_ggml_runtime_source_path(&path)
            .expect("validate a")
            .content_id()
            .to_string();
        write_magic_file(&path, b"GGUFcontent-b-bytes-different");
        let id_b = validate_ggml_runtime_source_path(&path)
            .expect("validate b")
            .content_id()
            .to_string();
        assert!(id_a.starts_with("sha256:"), "got {id_a}");
        assert!(id_b.starts_with("sha256:"), "got {id_b}");
        assert_ne!(id_a, id_b);
    }

    /// Direct repro of the audited defect: the historical memo key truncated
    /// mtime to whole seconds, so an equal-length replacement whose mtime
    /// landed in the same wall-clock second as the file it replaced reused
    /// the stale memoized content id instead of re-hashing. This is the
    /// primary production identity entry point
    /// (`GgmlRuntimeSource::content_id`); two equal-length packs pinned to
    /// the *same whole second* (different nanoseconds) must still resolve to
    /// distinct ids.
    #[test]
    #[cfg(unix)]
    fn content_id_rehashes_equal_length_same_second_mtime_replacement() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("same-second.gguf");

        let pack_a = b"GGUFpack-a-equal-length-byte";
        let pack_b = b"GGUFpack-b-equal-length-bytz";
        assert_eq!(
            pack_a.len(),
            pack_b.len(),
            "fixture bytes must be equal length"
        );
        assert_ne!(pack_a, pack_b);

        const SAME_SECOND: i64 = 1_700_000_000;

        write_magic_file(&path, pack_a);
        set_mtime(&path, SAME_SECOND, 111_000_000);
        let source_a = validate_ggml_runtime_source_path(&path).expect("validate a");
        let id_a = source_a.content_id().to_string();

        // A second, independently-opened source for the *same unchanged*
        // file must hit the warm-path memo and agree (proves the memo
        // itself still works across distinct `GgmlRuntimeSource` instances,
        // not just repeated calls on one instance).
        let source_a_again = validate_ggml_runtime_source_path(&path).expect("validate a again");
        assert_eq!(
            id_a,
            source_a_again.content_id(),
            "unchanged file must not re-resolve to a new id"
        );

        write_magic_file(&path, pack_b);
        set_mtime(&path, SAME_SECOND, 222_000_000);
        let source_b = validate_ggml_runtime_source_path(&path).expect("validate b");
        let id_b = source_b.content_id().to_string();

        assert!(id_a.starts_with("sha256:"), "got {id_a}");
        assert!(id_b.starts_with("sha256:"), "got {id_b}");
        assert_ne!(
            id_a, id_b,
            "equal-length replacement landing in the same whole second as the \
             original must still be rehashed, not aliased by a second-truncated memo"
        );
    }

    /// Identity and bytes must come from the same open handle. Holds a
    /// `GgmlRuntimeSource` open, replaces the file at
    /// its path via a rename (the same swap-the-directory-entry pattern
    /// `pull` uses), and proves the held source's mapped bytes and
    /// `content_id()` are both unchanged -- while a *fresh* resolution of the
    /// same path (a new open, as every new request performs) yields a
    /// different id. A rename (not an in-place truncate+rewrite) is
    /// essential here: it swaps the directory entry to a genuinely different
    /// inode, which is what an already-open mmap is immune to; an in-place
    /// rewrite of the same inode would (correctly) be visible to the old
    /// mapping too and would not exercise this guarantee.
    #[test]
    fn held_source_bytes_and_content_id_survive_a_rename_based_replacement() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("pack.gguf");
        write_magic_file(&path, b"GGUForiginal-bytes-untouched");

        let held = validate_ggml_runtime_source_path(&path).expect("validate held source");
        let held_id_before = held.content_id().to_string();
        let held_bytes_before = held.backing_mmap()[..].to_vec();

        // Replace via rename into place: a genuinely different inode ends up
        // at `path`; `held` keeps its own fd/mapping to the old inode.
        let replacement_path = dir.path().join("pack-replacement.gguf");
        write_magic_file(&replacement_path, b"GGUFreplaced-bytes-different");
        assert_eq!(
            std::fs::metadata(&path).expect("stat original").len(),
            std::fs::metadata(&replacement_path)
                .expect("stat replacement")
                .len(),
            "fixture bytes must be equal length"
        );
        fs::rename(&replacement_path, &path).expect("rename replacement into place");

        assert_eq!(
            held.content_id(),
            held_id_before,
            "an already-open source's content id must not change after a replacement at its path"
        );
        assert_eq!(
            &held.backing_mmap()[..],
            held_bytes_before.as_slice(),
            "an already-open source's mapped bytes must not change after a replacement at its path"
        );

        let fresh = validate_ggml_runtime_source_path(&path).expect("validate fresh source");
        assert_ne!(
            fresh.content_id(),
            held_id_before,
            "a fresh resolution of the replaced path must yield a different id"
        );
    }

    #[test]
    fn immutable_snapshot_bytes_survive_a_later_in_place_rewrite() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("auxiliary-pack.gguf");
        let original = b"GGUFauxiliary-content-a";
        let replacement = b"GGUFauxiliary-content-b";
        assert_eq!(original.len(), replacement.len());
        write_magic_file(&path, original);

        let source = validate_ggml_runtime_source_path(&path).expect("validate source");
        let expected = source.content_id().to_string();
        let immutable = source
            .immutable_snapshot_matching_content_id(&expected)
            .expect("make immutable snapshot");
        write_magic_file(&path, replacement);

        assert_eq!(&immutable.backing_mmap()[..], original);
        assert_eq!(immutable.content_id(), expected);
    }

    #[test]
    fn immutable_snapshot_fails_closed_when_source_changed_after_preflight() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("auxiliary-pack.gguf");
        let original = b"GGUFauxiliary-content-a";
        let replacement = b"GGUFauxiliary-content-b";
        assert_eq!(original.len(), replacement.len());
        write_magic_file(&path, original);

        let source = validate_ggml_runtime_source_path(&path).expect("validate source");
        let expected = source.content_id().to_string();
        write_magic_file(&path, replacement);
        let error = source
            .immutable_snapshot_matching_content_id(&expected)
            .expect_err("content changed between preflight and materialization");
        assert!(matches!(
            error,
            super::GgmlRuntimeSourcePathError::SnapshotContentChanged { .. }
        ));
    }

    /// Performance guardrail: the whole point of family TLS caches switching
    /// to [`GgmlRuntimeSource::content_id`] (via
    /// `models::runtime_cache_coordinator::PackContentKey::for_runtime_source`)
    /// instead of a from-scratch path-based fingerprint is that repeated
    /// admissions of the *same unchanged* pack must not pay a full-file
    /// SHA-256 every time -- `resolve_content_id`'s process-wide memo, keyed
    /// by [`StrongFileIdentity`], must short-circuit every open after the
    /// first. This is a timing proof (the memo itself is a private
    /// process-wide static with no counter seam to instrument), so the
    /// bound is deliberately generous: many independent warm opens together
    /// must stay cheaper than a single additional cold hash of the same
    /// file, which is only possible if the warm opens are not each redoing
    /// the full-file digest.
    #[test]
    fn content_id_memo_keeps_repeated_admissions_of_an_unchanged_pack_cheap() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("warm-path-pack.gguf");
        // 32 MiB: large enough that a full SHA-256 pass is measurable
        // (single-digit-plus milliseconds on any reasonable host), so a
        // memo failure (re-hashing on every open) would make the warm loop
        // below obviously, not marginally, slower than one cold hash.
        let mut bytes = vec![0_u8; 32 * 1024 * 1024];
        bytes[..4].copy_from_slice(b"GGUF");
        for (index, byte) in bytes.iter_mut().enumerate().skip(4) {
            *byte = (index % 251) as u8;
        }
        fs::write(&path, &bytes).expect("write warm-path fixture");

        // Cold: first open of this file, first call to `content_id()` --
        // pays the one real full-file hash.
        let cold_start = Instant::now();
        let cold_source = validate_ggml_runtime_source_path(&path).expect("cold open");
        let cold_id = cold_source.content_id().to_string();
        let cold_elapsed = cold_start.elapsed();
        assert!(cold_id.starts_with("sha256:"), "got {cold_id}");

        // Warm: many independent fresh opens of the SAME unchanged file --
        // each is a real `File::open`/`mmap` (never reused across
        // iterations) but must hit the `StrongFileIdentity`-keyed memo on
        // `content_id()` instead of re-hashing 32 MiB again.
        const WARM_ITERATIONS: u32 = 20;
        let warm_start = Instant::now();
        for _ in 0..WARM_ITERATIONS {
            let warm_source = validate_ggml_runtime_source_path(&path).expect("warm open");
            let warm_id = warm_source.content_id();
            assert_eq!(
                warm_id, cold_id,
                "unchanged bytes must keep the same content id"
            );
        }
        let warm_elapsed = warm_start.elapsed();

        assert!(
            warm_elapsed < cold_elapsed,
            "{WARM_ITERATIONS} warm (memoized) admissions of an unchanged pack took \
             {warm_elapsed:?}, which is not cheaper than the {cold_elapsed:?} single cold \
             full-file hash it should have avoided paying {WARM_ITERATIONS} more times over -- \
             the content-id memo appears to be re-hashing on every open"
        );
    }

    #[test]
    fn unreadable_source_path_is_rejected_before_content_id_is_ever_asked_for() {
        let missing = Path::new("/tmp/openasr-definitely-missing-runtime-source.gguf");
        let error = validate_ggml_runtime_source_path(missing)
            .expect_err("missing path must fail validation");
        match error {
            GgmlRuntimeSourcePathError::PathDoesNotExist { .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    fn write_object_at_layout(root: &Path, digest: &str, bytes: &[u8]) -> PathBuf {
        let object = root
            .join("models")
            .join("objects")
            .join("sha256")
            .join(digest)
            .join("content");
        fs::create_dir_all(object.parent().expect("object path has parent"))
            .expect("create digest dir");
        write_magic_file(&object, bytes);
        object
    }

    fn set_mode(path: &Path, read_only: bool) {
        let mut permissions = fs::metadata(path).expect("stat fixture").permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            permissions.set_mode(if read_only { 0o444 } else { 0o644 });
        }
        #[cfg(not(unix))]
        permissions.set_readonly(read_only);
        fs::set_permissions(path, permissions).expect("set fixture mode");
    }

    /// The trusted tier, pinned by construction: an object whose bytes do
    /// *not* hash to the digest its path names can only resolve to that path
    /// digest if identity was taken from the path without hashing. This is
    /// the runtime-cache analogue of content_store's
    /// `declared_lease_does_not_rehash_the_object` -- it is what keeps a
    /// gigabyte-scale re-read from creeping back into every model load.
    ///
    /// `content_id()` anchors trust to `content_store::default_models_root()`,
    /// so this test points `OPENASR_HOME` at the fixture's own tempdir --
    /// nextest's per-test process isolation makes this safe (see AGENTS.md's
    /// note on why nextest, not `cargo test`, is required for this
    /// workspace).
    #[test]
    fn sealed_content_addressed_object_content_id_trusts_the_path_digest() {
        let dir = tempdir().expect("tempdir");
        unsafe { std::env::set_var("OPENASR_HOME", dir.path()) };
        // A digest that is structurally valid but cannot be the hash of the
        // fixture bytes, so any hashing resolution would disagree with it.
        let named_digest = "ab".repeat(32);
        let bytes = b"GGUFtrusted-path-digest-fixture";
        let object = write_object_at_layout(dir.path(), &named_digest, bytes);
        set_mode(&object, true);
        assert_ne!(
            format!("{:x}", Sha256::digest(bytes)),
            named_digest,
            "the fixture must not accidentally hash to the named digest"
        );

        let source = validate_ggml_runtime_source_path(&object).expect("validate object");
        assert_eq!(source.content_id(), format!("sha256:{named_digest}"));

        set_mode(&object, false); // let the temp dir clean up on all platforms
    }

    /// The fail-closed half of the trusted tier: the same mismatched object,
    /// unsealed, must go back through a full hash and resolve to the digest
    /// of its actual bytes -- never the one its path claims.
    #[test]
    fn unsealed_content_addressed_object_content_id_falls_back_to_hashing() {
        let dir = tempdir().expect("tempdir");
        let named_digest = "cd".repeat(32);
        let bytes = b"GGUFunsealed-fallback-fixture";
        let object = write_object_at_layout(dir.path(), &named_digest, bytes);
        set_mode(&object, false);

        let source = validate_ggml_runtime_source_path(&object).expect("validate object");
        assert_eq!(
            source.content_id(),
            format!("sha256:{:x}", Sha256::digest(bytes)),
            "an unsealed object must be hashed, not trusted"
        );
        assert_ne!(source.content_id(), format!("sha256:{named_digest}"));
    }

    /// A genuinely admitted object (hashed and sealed by the content store
    /// itself) resolves to exactly the store's own digest -- the identity
    /// `pull`'s cache eviction derives from the installed ref without
    /// hashing, so both sides of a model install must key the runtime caches
    /// identically. Defeating the seal afterwards must flip the next open
    /// back onto the hashing tier and expose the changed bytes.
    #[test]
    fn admitted_object_content_id_agrees_with_the_store_and_degrades_on_seal_loss() {
        let dir = tempdir().expect("tempdir");
        unsafe { std::env::set_var("OPENASR_HOME", dir.path()) };
        let source_file = dir.path().join("source.oasr");
        write_magic_file(&source_file, b"GGUFadmitted-identity-fixture");
        let models = dir.path().join("models");

        let admitted = crate::content_store::admit_file(&source_file, &models, |_| Ok(()))
            .expect("admit fixture");
        let object = admitted.object_path.clone();
        let digest = admitted.digest.clone();

        let source = validate_ggml_runtime_source_path(&object).expect("validate object");
        assert_eq!(source.content_id(), format!("sha256:{digest}"));

        // Defeat the seal and rewrite the bytes in place: the next open is no
        // longer sealed, so trust is withdrawn and the hash speaks.
        set_mode(&object, false);
        write_magic_file(&object, b"GGUFadmitted-identity-XXXXXXX");
        let tampered = validate_ggml_runtime_source_path(&object).expect("validate tampered");
        assert_eq!(
            tampered.content_id(),
            format!(
                "sha256:{:x}",
                Sha256::digest(b"GGUFadmitted-identity-XXXXXXX")
            )
        );
        assert_ne!(tampered.content_id(), format!("sha256:{digest}"));
    }

    /// The same adversarial shape `content_store`'s own regression test pins,
    /// exercised through the primary production entry point: a same-shaped,
    /// sealed, read-only path outside `OPENASR_HOME`'s resolved model store
    /// must never be trusted, even though a models root does resolve.
    #[test]
    fn content_id_rejects_a_same_shaped_object_outside_the_models_root() {
        let dir = tempdir().expect("tempdir");
        unsafe { std::env::set_var("OPENASR_HOME", dir.path()) };
        let attacker_digest = "99".repeat(32);
        let bytes = b"GGUFattacker-controlled-bytes";
        let object = dir
            .path()
            .join("totally-unrelated")
            .join("objects")
            .join("sha256")
            .join(&attacker_digest)
            .join("content");
        fs::create_dir_all(object.parent().expect("object path has parent"))
            .expect("create digest dir");
        write_magic_file(&object, bytes);
        set_mode(&object, true);

        let source = validate_ggml_runtime_source_path(&object).expect("validate object");
        assert_eq!(
            source.content_id(),
            format!("sha256:{:x}", Sha256::digest(bytes)),
            "a same-shaped sealed path outside the models root must be hashed, not trusted"
        );
        assert_ne!(source.content_id(), format!("sha256:{attacker_digest}"));
    }
}
