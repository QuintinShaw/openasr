//! Pack content identity + the TLS lazy-eviction generation.
//!
//! This module used to also coordinate a single process-wide "build
//! generation" shared by serve-batch engine keys, prepared-runtime caches,
//! and process pools. That counter was an audited bug: baking a generation
//! into a content-addressed cache key means one bump (idle unload, a
//! serve-batch owner shutdown, or a pack install/replace anywhere in the
//! process) invalidates *every* resident content identity at once, not just
//! the one that actually changed. Content ids already change when pack bytes
//! change, so nothing needed that counter for correctness -- the callers that
//! used to bump it now either rely on their own explicit registry/cache
//! `clear()` (idle unload, serve-batch owner shutdown) or on the new content
//! id naturally missing (pack install/replace). See `pull.rs`'s post-install
//! handling and each family's `unload_idle_state`.
//!
//! The one counter that remains here is [`current_unload_generation`] /
//! [`bump_unload_generation`], which is a different thing: a lazy-eviction
//! signal for thread-local runtime caches (see `thread_local_runtime_cache`).
//! TLS caches live on worker threads the idle-unload reaper cannot reach
//! directly, so each cache instead records the generation it last synced at
//! and drops its resident entries the next time its owning thread touches it
//! after the generation has moved on. This generation must never be mixed
//! into a content-identity cache key.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

/// TLS lazy-eviction generation. See the module doc comment: this is
/// intentionally the *only* process-wide counter left in this module, and it
/// must never be read by content-identity resolution.
static TLS_LAZY_EVICTION_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Current TLS lazy-eviction generation. `Relaxed` matches the historical
/// counter this replaces: a coarse "an idle unload happened since this
/// thread-local entry was filled" signal for the owning thread to observe on
/// its own next access, not a cross-thread synchronization fence.
pub(crate) fn current_unload_generation() -> u64 {
    TLS_LAZY_EVICTION_GENERATION.load(Ordering::Relaxed)
}

/// Marks one idle-unload sweep by advancing the TLS lazy-eviction generation.
pub(crate) fn bump_unload_generation() {
    TLS_LAZY_EVICTION_GENERATION.fetch_add(1, Ordering::Relaxed);
}

/// Formats a content id from a lowercase hex sha256 digest.
pub fn content_id_from_sha256_hex(sha256_hex: &str) -> String {
    format!("sha256:{sha256_hex}")
}

/// Returns true when `pack_content_id` is safe to use as a reusable cache key.
///
/// Production inserts require a real content proof (`sha256:`) or an explicit
/// test/verified token. `unreadable:*` tokens must never enter a shared slot:
/// they would either poison a path forever or collide incorrectly after a
/// later successful hash of the same path.
pub fn is_cacheable_pack_content_id(pack_content_id: &str) -> bool {
    pack_content_id.starts_with("sha256:")
        || pack_content_id.starts_with("test:")
        || pack_content_id.starts_with("verified:")
}

/// Resolves a stable pack content id for `runtime_path`.
///
/// Identity authority is a full-file sha256 read from an open handle -- never
/// the installed-pack registry's recorded sha (that is a claim about what was
/// *installed*, not a proof of what is currently on disk) and never truncated
/// file metadata. Path alone is never returned.
///
/// Cost contract (this is a deliberate, documented trade-off, not an
/// oversight -- hashing a multi-GB pack on every request is unacceptable):
///
/// - **Warm path** (the common per-request call): a single `stat` builds a
///   [`StrongFileIdentity`] (device, inode, length, full-nanosecond mtime)
///   and compares it against a memoized `(identity, content_id)` pair for the
///   canonical path. A match returns the memoized id without opening or
///   reading the file.
/// - **Cold path** (first call for a path, or the strong identity changed):
///   opens the file once and hashes it once, then memoizes the new
///   `(identity, content_id)` pair for next time.
///
/// The strong identity is deliberately **not** truncated to whole seconds --
/// that truncation was the audited bug: an equal-length replacement that
/// completed within the same wall-clock second as the file it replaced
/// aliased the previous content id without ever re-hashing. Only a
/// replacement that fakes the mtime's nanosecond field while preserving
/// device, inode, and length could still alias a stale id here; that is
/// outside this project's threat model (anyone able to hand-craft such a
/// replacement can already substitute an arbitrary model file), and every
/// real replacement path (a pull install, a user copying a new pack over an
/// old one) moves the mtime's nanosecond field and is caught.
pub fn pack_content_id_for_runtime_path(runtime_path: &Path) -> String {
    let canonical =
        std::fs::canonicalize(runtime_path).unwrap_or_else(|_| runtime_path.to_path_buf());
    let Ok(metadata) = std::fs::metadata(&canonical) else {
        return unreadable_pack_content_id(&canonical);
    };
    let Some(identity) = StrongFileIdentity::of(&metadata) else {
        return unreadable_pack_content_id(&canonical);
    };
    cached_or_hash_pack_content_id(&canonical, identity)
}

/// Content-addressed prepared/process-pool cache key: pack content id alone.
///
/// Carries no generation/epoch -- see the module doc comment for why that was
/// removed. Route / options stay out of this key when the cached object is
/// device- and adapter-neutral (prepared packs, Dolphin dequantized weight
/// tables); callers that need those to participate mix them in separately.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PackContentKey {
    pub pack_content_id: String,
}

impl PackContentKey {
    pub(crate) fn new(pack_content_id: impl Into<String>) -> Self {
        Self {
            pack_content_id: pack_content_id.into(),
        }
    }

    /// Resolve a cacheable key for `runtime_path`.
    ///
    /// Returns `None` when the pack cannot be content-hashed -- callers must
    /// skip the reusable cache (one-shot uncached execute) rather than key by
    /// path alone.
    pub(crate) fn try_for_runtime_path(runtime_path: &Path) -> Option<Self> {
        let pack_content_id = pack_content_id_for_runtime_path(runtime_path);
        if !is_cacheable_pack_content_id(&pack_content_id) {
            return None;
        }
        Some(Self::new(pack_content_id))
    }
}

/// Byte budget for each of the leading/trailing pack slices mixed into
/// [`pack_content_fingerprint`].
///
/// 64 KiB per edge keeps the per-lookup cost O(1) for multi-GB packs (unlike
/// the full-file sha256 behind [`pack_content_id_for_runtime_path`]) while
/// still covering both ends of a `.oasr` zip: the leading slice carries the
/// first stored entry's GGUF magic/header/metadata, and the trailing slice
/// carries the zip central directory, whose per-entry CRC32s change on any
/// content replacement -- so a same-size in-place swap whose head bytes are
/// identical is still caught through the tail.
const PACK_CONTENT_FINGERPRINT_EDGE_BYTES: usize = 64 * 1024;

/// Source of never-equal tokens for packs whose fingerprint cannot be read.
static PACK_FINGERPRINT_UNREADABLE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Lightweight pack content fingerprint for the path-keyed thread-local
/// runtime caches (whole-decoder / encoder / persistent-session / process
/// pool keys).
///
/// This is the cheap per-request sibling of the full content proof in
/// [`pack_content_id_for_runtime_path`]: it mixes the file length, the full
/// mtime (seconds + nanoseconds), and a sha256 over the first and last
/// [`PACK_CONTENT_FINGERPRINT_EDGE_BYTES`] bytes. A runtime cache key that
/// includes this fingerprint can never hand a runtime built from one pack's
/// bytes to a request against different bytes at the same path (an in-place
/// `.oasr` replacement): the replacement moves the mtime and/or size and/or
/// edge bytes, so the next lookup misses and rebuilds. The trade-offs:
///
/// - A rewrite with byte-identical content also moves the mtime and therefore
///   invalidates once. That conservative miss (one rebuild) is intentional --
///   mtime equality cannot be trusted as a same-content proof.
/// - Hashing the full file here would re-read multi-GB weights on every cache
///   lookup after each replacement; the edge slices cap that at 128 KiB. The
///   full-file proof stays where its one-shot cost is acceptable
///   ([`pack_content_id_for_runtime_path`], memoized by strong file identity,
///   for prepared/process-pool keys).
///
/// Unreadable packs fail closed: every call returns a fresh `unreadable:*`
/// token that never equals any previously stored fingerprint, so the lookup
/// always misses and the (almost certainly failing) build is retried rather
/// than a stale runtime reused.
pub(crate) fn pack_content_fingerprint(runtime_path: &Path) -> String {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Seek, SeekFrom};

    let Ok(metadata) = std::fs::metadata(runtime_path) else {
        return unreadable_pack_content_fingerprint();
    };
    let Ok(modified) = metadata.modified() else {
        return unreadable_pack_content_fingerprint();
    };
    let Ok(since_epoch) = modified.duration_since(std::time::UNIX_EPOCH) else {
        return unreadable_pack_content_fingerprint();
    };
    let Ok(mut file) = std::fs::File::open(runtime_path) else {
        return unreadable_pack_content_fingerprint();
    };

    let len = metadata.len();
    let edge = PACK_CONTENT_FINGERPRINT_EDGE_BYTES as u64;
    let head_len = len.min(edge);
    let mut hasher = Sha256::new();
    // Domain separation: a fingerprint digest must never collide with a raw
    // byte digest (e.g. the full-file `sha256:` content ids) for the same pack.
    hasher.update(b"openasr-pack-content-fingerprint-v1");
    hasher.update(len.to_le_bytes());
    hasher.update(since_epoch.as_secs().to_le_bytes());
    hasher.update(since_epoch.subsec_nanos().to_le_bytes());

    let mut head = vec![0_u8; head_len as usize];
    if file.read_exact(&mut head).is_err() {
        return unreadable_pack_content_fingerprint();
    }
    hasher.update(&head);

    // Trailing slice (zip central directory); skipped when the head already
    // covered the whole file.
    if len > head_len && file.seek(SeekFrom::End(-(edge as i64))).is_ok() {
        let mut tail = vec![0_u8; edge as usize];
        if file.read_exact(&mut tail).is_err() {
            return unreadable_pack_content_fingerprint();
        }
        hasher.update(&tail);
    }
    format!("fp1:{:x}", hasher.finalize())
}

fn unreadable_pack_content_fingerprint() -> String {
    format!(
        "unreadable:{}",
        PACK_FINGERPRINT_UNREADABLE_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

/// Canonical path + content fingerprint identity half of a path-keyed
/// thread-local runtime cache key.
///
/// The path alone only proves "same file name"; a runtime built from one
/// pack's bytes must not be reused after the file at that path is replaced in
/// place, so every runtime cache key carries the [`pack_content_fingerprint`]
/// observed when the key was built. Lookup against a replaced pack computes a
/// different fingerprint, misses, and rebuilds.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RuntimeCachePathIdentity {
    pub(crate) path: PathBuf,
    pub(crate) fingerprint: String,
}

/// Resolve the cache-key identity (canonical path + current content
/// fingerprint) for `runtime_path`.
pub(crate) fn runtime_cache_path_identity(runtime_path: &Path) -> RuntimeCachePathIdentity {
    let path = std::fs::canonicalize(runtime_path).unwrap_or_else(|_| runtime_path.to_path_buf());
    let fingerprint = pack_content_fingerprint(&path);
    RuntimeCachePathIdentity { path, fingerprint }
}

/// Strong OS file identity used as the warm-path cache-hit precheck for
/// [`pack_content_id_for_runtime_path`]: device, inode, length, and the
/// *full* nanosecond mtime.
///
/// The mtime is deliberately never truncated to whole seconds -- that
/// truncation was the audited defect this type replaces (the historical
/// memo key kept only length plus a whole-second mtime). Any single field
/// differing forces a fresh hash.
///
/// `dev`/`ino` are only available through `std::os::unix::fs::MetadataExt`;
/// non-unix targets fall back to `(len, mtime)` alone, which is narrower
/// (two different files could theoretically share a length and mtime) but
/// still nanosecond-precise, unlike the bug being fixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct StrongFileIdentity {
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
    /// closed to a fresh hash rather than trust a partial identity.
    fn of(metadata: &std::fs::Metadata) -> Option<Self> {
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

/// Cold path: hash `path` once (via an open handle) and memoize the result
/// against `identity`. A later call whose freshly-stated `StrongFileIdentity`
/// still matches the memoized one returns the cached id without opening or
/// reading the file again (the warm path).
fn cached_or_hash_pack_content_id(path: &Path, identity: StrongFileIdentity) -> String {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, (StrongFileIdentity, String)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = cache.lock()
        && let Some((cached_identity, content_id)) = guard.get(path)
        && *cached_identity == identity
    {
        return content_id.clone();
    }

    let content_id = match sha256_hex_file(path) {
        Ok(hex) => content_id_from_sha256_hex(&hex),
        Err(_) => unreadable_pack_content_id(path),
    };
    // Only memoize cacheable proofs. Caching an `unreadable:*` token would pin a
    // miss forever even after the file becomes readable with the same identity.
    if is_cacheable_pack_content_id(&content_id)
        && let Ok(mut guard) = cache.lock()
    {
        guard.insert(path.to_path_buf(), (identity, content_id.clone()));
    }
    content_id
}

fn unreadable_pack_content_id(path: &Path) -> String {
    static UNREADABLE_COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "unreadable:{}:{}",
        path.display(),
        UNREADABLE_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

fn sha256_hex_file(path: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn pack_content_id_misses_same_path_byte_replacement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("same-path.oasr");
        std::fs::write(&path, b"content-a-bytes").expect("write a");
        let id_a = pack_content_id_for_runtime_path(&path);
        std::fs::write(&path, b"content-b-bytes-different").expect("write b");
        let id_b = pack_content_id_for_runtime_path(&path);
        assert!(id_a.starts_with("sha256:"), "got {id_a}");
        assert!(id_b.starts_with("sha256:"), "got {id_b}");
        assert_ne!(id_a, id_b);
        assert!(is_cacheable_pack_content_id(&id_a));
        assert!(is_cacheable_pack_content_id(&id_b));
    }

    /// Direct repro of the audited defect: the historical memo key truncated
    /// mtime to whole seconds, so an equal-length replacement whose mtime
    /// landed in the same wall-clock second as the file it replaced reused
    /// the stale memoized content id instead of re-hashing.
    /// `StrongFileIdentity` carries the full nanosecond mtime specifically to
    /// catch this -- two equal-length packs pinned to the *same whole
    /// second* (different nanoseconds) must still resolve to distinct ids.
    #[test]
    #[cfg(unix)]
    fn pack_content_id_rehashes_equal_length_same_second_mtime_replacement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("same-second.oasr");

        let pack_a = b"pack-a-equal-length-bytes";
        let pack_b = b"pack-b-equal-length-bytz2";
        assert_eq!(
            pack_a.len(),
            pack_b.len(),
            "fixture bytes must be equal length"
        );
        assert_ne!(pack_a, pack_b);

        const SAME_SECOND: i64 = 1_700_000_000;

        std::fs::write(&path, pack_a).expect("write a");
        set_mtime(&path, SAME_SECOND, 111_000_000);
        let id_a = pack_content_id_for_runtime_path(&path);

        // Re-resolving without touching the file must hit the warm path and
        // return the same id (proves the memo itself still works).
        let id_a_again = pack_content_id_for_runtime_path(&path);
        assert_eq!(
            id_a, id_a_again,
            "unchanged file must not re-resolve to a new id"
        );

        std::fs::write(&path, pack_b).expect("write b (equal length)");
        set_mtime(&path, SAME_SECOND, 222_000_000);
        let id_b = pack_content_id_for_runtime_path(&path);

        assert!(id_a.starts_with("sha256:"), "got {id_a}");
        assert!(id_b.starts_with("sha256:"), "got {id_b}");
        assert_ne!(
            id_a, id_b,
            "equal-length replacement landing in the same whole second as the \
             original must still be rehashed, not aliased by a second-truncated memo"
        );
    }

    #[test]
    fn unreadable_pack_is_not_cacheable() {
        let missing = PathBuf::from("/tmp/openasr-definitely-missing-runtime-pack.oasr");
        let id = pack_content_id_for_runtime_path(&missing);
        assert!(
            id.starts_with("unreadable:"),
            "unreadable path must fail closed, got {id}"
        );
        assert!(!is_cacheable_pack_content_id(&id));
        assert!(PackContentKey::try_for_runtime_path(&missing).is_none());
    }

    #[test]
    fn pack_content_fingerprint_is_stable_while_the_pack_is_unchanged() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("pack.oasr");
        std::fs::write(&path, b"stable-pack-bytes").expect("write");
        let first = pack_content_fingerprint(&path);
        let second = pack_content_fingerprint(&path);
        assert!(first.starts_with("fp1:"), "got {first}");
        assert_eq!(first, second, "no file change between the two lookups");
    }

    #[test]
    fn pack_content_fingerprint_misses_in_place_byte_replacement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("same-path.oasr");
        std::fs::write(&path, b"content-a-bytes").expect("write a");
        let before = pack_content_fingerprint(&path);
        std::fs::write(&path, b"content-b-bytes-different").expect("write b");
        let after = pack_content_fingerprint(&path);
        assert!(before.starts_with("fp1:"), "got {before}");
        assert!(after.starts_with("fp1:"), "got {after}");
        assert_ne!(
            before, after,
            "an in-place replacement at the same path must not fingerprint equal"
        );
    }

    #[test]
    fn pack_content_fingerprint_misses_same_length_replacement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("same-length.oasr");
        // 192 KiB: bigger than both edge windows, so the changed byte sits in
        // the middle, covered by neither the head nor the tail slice -- the
        // size-stable replacement is still caught (by the mtime move).
        let edge = PACK_CONTENT_FINGERPRINT_EDGE_BYTES;
        let mut bytes = vec![7_u8; edge * 3];
        std::fs::write(&path, &bytes).expect("write v1");
        let before = pack_content_fingerprint(&path);
        bytes[edge + edge / 2] = 9;
        std::fs::write(&path, &bytes).expect("write v2");
        let after = pack_content_fingerprint(&path);
        assert_ne!(
            before, after,
            "a same-length in-place replacement must not fingerprint equal"
        );
    }

    #[test]
    fn pack_content_fingerprint_unreadable_pack_never_matches() {
        let missing = PathBuf::from("/tmp/openasr-definitely-missing-fingerprint-pack.oasr");
        let first = pack_content_fingerprint(&missing);
        let second = pack_content_fingerprint(&missing);
        assert!(first.starts_with("unreadable:"), "got {first}");
        assert!(second.starts_with("unreadable:"), "got {second}");
        assert_ne!(
            first, second,
            "unreadable tokens must be unique per call so no lookup can ever hit on them"
        );
    }

    #[test]
    fn runtime_cache_path_identity_changes_on_in_place_replacement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("pack.oasr");
        std::fs::write(&path, b"identity-content-v1").expect("write v1");
        let before = runtime_cache_path_identity(&path);
        assert_eq!(
            before.path,
            std::fs::canonicalize(&path).expect("canonicalize"),
            "identity carries the canonical path half"
        );
        assert!(before.fingerprint.starts_with("fp1:"));

        let unchanged = runtime_cache_path_identity(&path);
        assert_eq!(before, unchanged, "unchanged pack keeps the same identity");

        std::fs::write(&path, b"identity-content-v2").expect("write v2");
        let after = runtime_cache_path_identity(&path);
        assert_eq!(before.path, after.path, "same path half");
        assert_ne!(
            before, after,
            "replaced bytes must change the identity (via the fingerprint)"
        );
    }

    #[test]
    fn cacheable_content_id_prefixes() {
        assert!(is_cacheable_pack_content_id("sha256:abc"));
        assert!(is_cacheable_pack_content_id("test:fixture"));
        assert!(is_cacheable_pack_content_id("verified:fixture"));
        assert!(!is_cacheable_pack_content_id("unreadable:/tmp/x:0"));
        assert!(!is_cacheable_pack_content_id("/tmp/x.oasr"));
        assert!(!is_cacheable_pack_content_id(""));
    }
}
