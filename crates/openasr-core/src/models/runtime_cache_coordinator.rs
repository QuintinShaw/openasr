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

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::ggml_runtime::{StrongFileIdentity, resolve_content_id, unreadable_content_id};

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

/// Resolves the content id of whatever pack currently sits at `path`, purely
/// from a path -- **for `pull`'s pre-replace snapshot only**.
///
/// This exists because `pull` needs the id of the pack it is about to
/// overwrite, and by definition has no open [`crate::GgmlRuntimeSource`] for
/// a file it has not opened (and is not about to open for real use -- it is
/// being discarded). Every live production identity that actually feeds a
/// runtime build goes through [`crate::GgmlRuntimeSource::content_id`]
/// instead, which derives its warm-path precheck from the fd it already has
/// open rather than a fresh `stat` on the path. This function must not grow
/// another caller that feeds a runtime build -- if you need a content id to
/// key a cache or build a runtime, you should already be holding a
/// `GgmlRuntimeSource`; use its `content_id()`.
///
/// Shares [`resolve_content_id`]'s memo with `GgmlRuntimeSource::content_id`,
/// so hashing a path once through either entry point warms the other's
/// lookup too.
pub(crate) fn pack_content_id_for_path_before_replace(path: &Path) -> String {
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let Ok(metadata) = std::fs::metadata(&canonical) else {
        return unreadable_content_id(&canonical);
    };
    let Some(identity) = StrongFileIdentity::of(&metadata) else {
        return unreadable_content_id(&canonical);
    };
    resolve_content_id(&canonical, identity, || sha256_hex_file(&canonical).ok())
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

    /// Resolve a cacheable key from an already-open source's content id.
    ///
    /// Returns `None` when the pack cannot be content-hashed -- callers must
    /// skip the reusable cache (one-shot uncached execute) rather than key by
    /// path alone.
    pub(crate) fn try_for_runtime_source(source: &crate::GgmlRuntimeSource) -> Option<Self> {
        let pack_content_id = source.content_id();
        if !is_cacheable_pack_content_id(pack_content_id) {
            return None;
        }
        Some(Self::new(pack_content_id.to_string()))
    }
}

/// Byte budget for each of the leading/trailing pack slices mixed into
/// [`pack_content_fingerprint`].
///
/// 64 KiB per edge keeps the per-lookup cost O(1) for multi-GB packs (unlike
/// the full-file sha256 behind [`crate::GgmlRuntimeSource::content_id`]) while
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
/// [`crate::GgmlRuntimeSource::content_id`]: it mixes the file length, the full
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
///   ([`crate::GgmlRuntimeSource::content_id`], memoized by strong file
///   identity, for prepared/process-pool keys).
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

/// Opens and hashes `path` directly (no mmap, no identity memo) -- the
/// unavoidable cold-path I/O behind [`pack_content_id_for_path_before_replace`].
/// `GgmlRuntimeSource::content_id` never calls this: it hashes the mapping it
/// already holds open instead.
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

    /// `pack_content_id_for_path_before_replace` is a thin, narrow-purpose
    /// wrapper (see its doc comment); the core strong-identity algorithm
    /// (equal-length/same-second-mtime rehash, warm-path memo, unreadable
    /// fail-closed) is exercised directly against the primary production
    /// entry point, `GgmlRuntimeSource::content_id`, in
    /// `ggml_runtime::runtime_source`'s test module. These tests only cover
    /// this function's own narrow contract.
    #[test]
    fn pack_content_id_for_path_before_replace_misses_same_path_byte_replacement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("same-path.oasr");
        std::fs::write(&path, b"content-a-bytes").expect("write a");
        let id_a = pack_content_id_for_path_before_replace(&path);
        std::fs::write(&path, b"content-b-bytes-different").expect("write b");
        let id_b = pack_content_id_for_path_before_replace(&path);
        assert!(id_a.starts_with("sha256:"), "got {id_a}");
        assert!(id_b.starts_with("sha256:"), "got {id_b}");
        assert_ne!(id_a, id_b);
        assert!(is_cacheable_pack_content_id(&id_a));
        assert!(is_cacheable_pack_content_id(&id_b));
    }

    #[test]
    fn unreadable_path_before_replace_is_not_cacheable() {
        let missing = PathBuf::from("/tmp/openasr-definitely-missing-runtime-pack.oasr");
        let id = pack_content_id_for_path_before_replace(&missing);
        assert!(
            id.starts_with("unreadable:"),
            "unreadable path must fail closed, got {id}"
        );
        assert!(!is_cacheable_pack_content_id(&id));
    }

    #[test]
    fn pack_content_key_resolves_from_an_open_runtime_source() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("pack.gguf");
        std::fs::write(&path, b"GGUFpack-content-key-fixture").expect("write");
        let source = crate::validate_ggml_runtime_source_path(&path).expect("validate source");

        let key = PackContentKey::try_for_runtime_source(&source).expect("cacheable");
        assert_eq!(key.pack_content_id, source.content_id());
        assert!(is_cacheable_pack_content_id(&key.pack_content_id));
    }

    /// The pull-only path-based resolver and the primary
    /// `GgmlRuntimeSource::content_id` resolver share one memo: hashing a
    /// path through either warms the other's lookup for the same bytes.
    #[test]
    fn path_before_replace_and_runtime_source_content_id_share_the_memo() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("shared-memo.gguf");
        std::fs::write(&path, b"GGUFshared-memo-fixture-bytes").expect("write");

        let source = crate::validate_ggml_runtime_source_path(&path).expect("validate source");
        let via_source = source.content_id().to_string();
        let via_path_snapshot = pack_content_id_for_path_before_replace(&path);
        assert_eq!(
            via_source, via_path_snapshot,
            "both entry points must resolve to the same content id for the same bytes"
        );
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
