//! Process-wide runtime cache identity + epoch coordinator.
//!
//! Owns the single invalidation epoch shared by serve-batch engines, TLS
//! `BoundedRuntimeCache` / `UnloadGenerationGated` maps, prepared runtime
//! caches, and process pools (Dolphin / XASR). Family caches keep their own
//! typed storage; this module only supplies content identity, epoch, and the
//! thin alias surface used during migration from the historical dual counters
//! (`RUNTIME_BUILD_GENERATION` + `RUNTIME_CACHE_UNLOAD_GENERATION`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Process-wide epoch for all reusable native runtime state.
static RUNTIME_CACHE_EPOCH: AtomicU64 = AtomicU64::new(0);

/// Why the coordinator epoch was bumped. Call sites pass a reason for
/// diagnostics; behavior is identical for every variant in v1 (full epoch
/// advance). Scoped invalidation can hang off this later without changing
/// callers again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeCacheInvalidation {
    IdleUnload,
    ServeBatchOwnerShutdown,
    /// After a successful pull/import that may replace bytes at an existing path.
    PackInstallOrReplace,
    /// Explicit operator / test / legacy alias callers.
    Manual,
}

/// Process-wide coordinator for runtime-cache identity and invalidation.
#[derive(Debug, Default)]
pub struct RuntimeCacheCoordinator;

impl RuntimeCacheCoordinator {
    pub fn global() -> &'static Self {
        static GLOBAL: RuntimeCacheCoordinator = RuntimeCacheCoordinator;
        &GLOBAL
    }

    /// Current epoch. `Relaxed` matches the historical unload/build counters:
    /// this is a coarse "has invalidation happened since this entry was filled"
    /// signal, not a cross-thread synchronization fence.
    pub fn epoch(&self) -> u64 {
        RUNTIME_CACHE_EPOCH.load(Ordering::Relaxed)
    }

    /// Bump the process-wide epoch and return the new value.
    pub fn invalidate(&self, reason: RuntimeCacheInvalidation) -> u64 {
        let _ = reason;
        RUNTIME_CACHE_EPOCH.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Resolve a stable pack content id for `pack_path`.
    ///
    /// Preference order:
    /// 1. Installed-pack registry entry whose path matches (pull-verified sha256).
    /// 2. Full-file sha256 of the runtime pack bytes.
    ///
    /// Path alone is never returned. Results are memoized by canonical path +
    /// size + mtime so repeated requests against an unchanged pack do not
    /// re-hash multi-GB weights; different bytes at the same path always miss
    /// the memo and re-hash. Unreadable paths yield a unique `unreadable:...`
    /// token that never collides with a real `sha256:` id -- callers that
    /// insert into reusable caches must treat those as non-cacheable via
    /// [`is_cacheable_pack_content_id`].
    pub fn content_id_for_pack(&self, pack_path: &Path) -> String {
        pack_content_id_for_runtime_path_inner(pack_path)
    }
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

/// Current process-wide runtime-cache epoch (alias of the unified coordinator
/// epoch). Historical name: runtime-build generation observed by serve-batch
/// engine keys.
pub fn current_runtime_build_generation() -> u64 {
    RuntimeCacheCoordinator::global().epoch()
}

/// Bumps the process-wide epoch and returns the new value.
///
/// Idle unload / serve-batch owner shutdown / pack replace call this (directly
/// or via [`bump_unload_generation`]) so a later same-path request cannot
/// silently reuse drained owners or stale TLS entries.
pub fn bump_runtime_build_generation() -> u64 {
    RuntimeCacheCoordinator::global().invalidate(RuntimeCacheInvalidation::Manual)
}

/// Current idle-unload generation. Alias of the unified coordinator epoch so
/// TLS caches and serve-batch engines observe one clock.
pub(crate) fn current_unload_generation() -> u64 {
    RuntimeCacheCoordinator::global().epoch()
}

/// Marks one idle-unload sweep by advancing the unified epoch.
pub(crate) fn bump_unload_generation() {
    let _ = RuntimeCacheCoordinator::global().invalidate(RuntimeCacheInvalidation::IdleUnload);
}

/// Serve-batch owner shutdown bump (same epoch as idle unload).
pub(crate) fn bump_serve_batch_owner_shutdown_generation() -> u64 {
    RuntimeCacheCoordinator::global().invalidate(RuntimeCacheInvalidation::ServeBatchOwnerShutdown)
}

/// Bump the unified epoch after a successful pull/import that may replace pack
/// bytes at an existing path. Kept as a named entry point so call sites do not
/// invent a second invalidation clock.
pub fn invalidate_after_pack_install_or_replace() -> u64 {
    RuntimeCacheCoordinator::global().invalidate(RuntimeCacheInvalidation::PackInstallOrReplace)
}

/// Resolves a stable pack content id for `runtime_path`.
///
/// See [`RuntimeCacheCoordinator::content_id_for_pack`].
pub fn pack_content_id_for_runtime_path(runtime_path: &Path) -> String {
    RuntimeCacheCoordinator::global().content_id_for_pack(runtime_path)
}

fn pack_content_id_for_runtime_path_inner(runtime_path: &Path) -> String {
    let canonical =
        std::fs::canonicalize(runtime_path).unwrap_or_else(|_| runtime_path.to_path_buf());
    if let Some(installed_sha) = installed_pack_sha256_for_path(&canonical) {
        return content_id_from_sha256_hex(&installed_sha);
    }
    match file_metadata_key(&canonical) {
        Some(meta_key) => cached_or_hash_pack_content_id(&canonical, meta_key),
        None => match sha256_hex_file(&canonical) {
            Ok(hex) => content_id_from_sha256_hex(&hex),
            Err(_) => unreadable_pack_content_id(&canonical),
        },
    }
}

/// Content-addressed prepared/process-pool key half: pack content id + epoch.
///
/// Route / options stay out of this key when the cached object is device- and
/// adapter-neutral (prepared packs, Dolphin dequantized weight tables).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PackContentEpochKey {
    pub pack_content_id: String,
    pub generation: u64,
}

impl PackContentEpochKey {
    pub(crate) fn new(pack_content_id: impl Into<String>, generation: u64) -> Self {
        Self {
            pack_content_id: pack_content_id.into(),
            generation,
        }
    }

    /// Resolve a cacheable key for `runtime_path` at the current epoch.
    ///
    /// Returns `None` when the pack cannot be content-hashed -- callers must
    /// skip the reusable cache (one-shot uncached execute) rather than key by
    /// path alone.
    pub(crate) fn try_for_runtime_path(runtime_path: &Path) -> Option<Self> {
        let pack_content_id = pack_content_id_for_runtime_path(runtime_path);
        if !is_cacheable_pack_content_id(&pack_content_id) {
            return None;
        }
        Some(Self::new(
            pack_content_id,
            RuntimeCacheCoordinator::global().epoch(),
        ))
    }
}

fn installed_pack_sha256_for_path(canonical_path: &Path) -> Option<String> {
    let home = crate::openasr_home().ok()?;
    let packs = crate::list_installed_packs(home).ok()?;
    packs.into_iter().find_map(|pack| {
        let pack_path = std::fs::canonicalize(&pack.path).unwrap_or(pack.path);
        (pack_path == canonical_path && !pack.sha256.is_empty()).then_some(pack.sha256)
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PackMetaKey {
    len: u64,
    modified_secs: u64,
}

fn file_metadata_key(path: &Path) -> Option<PackMetaKey> {
    let meta = std::fs::metadata(path).ok()?;
    let modified_secs = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some(PackMetaKey {
        len: meta.len(),
        modified_secs,
    })
}

fn cached_or_hash_pack_content_id(path: &Path, meta_key: PackMetaKey) -> String {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, (PackMetaKey, String)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = cache.lock()
        && let Some((cached_meta, content_id)) = guard.get(path)
        && *cached_meta == meta_key
    {
        return content_id.clone();
    }

    let content_id = match sha256_hex_file(path) {
        Ok(hex) => content_id_from_sha256_hex(&hex),
        Err(_) => unreadable_pack_content_id(path),
    };
    // Only memoize cacheable proofs. Caching an `unreadable:*` token would pin a
    // miss forever even after the file becomes readable with the same mtime.
    if is_cacheable_pack_content_id(&content_id)
        && let Ok(mut guard) = cache.lock()
    {
        guard.insert(path.to_path_buf(), (meta_key, content_id.clone()));
    }
    content_id
}

fn unreadable_pack_content_id(path: &Path) -> String {
    format!(
        "unreadable:{}:{}",
        path.display(),
        RuntimeCacheCoordinator::global().epoch()
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

    #[test]
    fn epoch_aliases_share_one_counter() {
        let before = RuntimeCacheCoordinator::global().epoch();
        assert_eq!(current_runtime_build_generation(), before);
        assert_eq!(current_unload_generation(), before);

        let after_build = bump_runtime_build_generation();
        assert!(after_build > before);
        assert_eq!(current_unload_generation(), after_build);
        assert_eq!(RuntimeCacheCoordinator::global().epoch(), after_build);

        let before_unload = current_unload_generation();
        bump_unload_generation();
        let after_unload = current_unload_generation();
        assert!(after_unload > before_unload);
        assert_eq!(current_runtime_build_generation(), after_unload);

        let after_shutdown = bump_serve_batch_owner_shutdown_generation();
        assert!(after_shutdown > after_unload);
        assert_eq!(current_unload_generation(), after_shutdown);
        assert_eq!(current_runtime_build_generation(), after_shutdown);
    }

    #[test]
    fn invalidate_advances_epoch_for_every_reason() {
        let coordinator = RuntimeCacheCoordinator::global();
        let mut previous = coordinator.epoch();
        for reason in [
            RuntimeCacheInvalidation::IdleUnload,
            RuntimeCacheInvalidation::ServeBatchOwnerShutdown,
            RuntimeCacheInvalidation::PackInstallOrReplace,
            RuntimeCacheInvalidation::Manual,
        ] {
            let next = coordinator.invalidate(reason);
            assert!(next > previous, "reason={reason:?}");
            previous = next;
        }
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

    #[test]
    fn unreadable_pack_is_not_cacheable() {
        let missing = PathBuf::from("/tmp/openasr-definitely-missing-runtime-pack.oasr");
        let id = pack_content_id_for_runtime_path(&missing);
        assert!(
            id.starts_with("unreadable:"),
            "unreadable path must fail closed, got {id}"
        );
        assert!(!is_cacheable_pack_content_id(&id));
        assert!(PackContentEpochKey::try_for_runtime_path(&missing).is_none());
    }

    #[test]
    fn pack_content_epoch_key_includes_generation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("pack.oasr");
        std::fs::write(&path, b"stable-bytes").expect("write");
        let key_a = PackContentEpochKey::try_for_runtime_path(&path).expect("cacheable");
        let _ = bump_runtime_build_generation();
        let key_b = PackContentEpochKey::try_for_runtime_path(&path).expect("cacheable");
        assert_eq!(key_a.pack_content_id, key_b.pack_content_id);
        assert_ne!(key_a.generation, key_b.generation);
        assert_ne!(key_a, key_b);
    }

    #[test]
    fn content_id_for_pack_matches_free_function() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("pack.oasr");
        std::fs::write(&path, b"coord-api").expect("write");
        let via_api = RuntimeCacheCoordinator::global().content_id_for_pack(&path);
        let via_fn = pack_content_id_for_runtime_path(&path);
        assert_eq!(via_api, via_fn);
        assert!(via_api.starts_with("sha256:"));
    }

    #[test]
    fn invalidate_after_pack_install_or_replace_advances_epoch() {
        let before = RuntimeCacheCoordinator::global().epoch();
        let after = invalidate_after_pack_install_or_replace();
        assert!(after > before);
        assert_eq!(current_runtime_build_generation(), after);
        assert_eq!(current_unload_generation(), after);
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
