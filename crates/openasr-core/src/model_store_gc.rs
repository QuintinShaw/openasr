//! Reclamation and accounting for the content-addressed model store.
//!
//! The store has three writers (install, resumed download, legacy migration) and
//! until now no reader ever removed anything, so every interrupted or superseded
//! write stayed on disk forever. This module is the single exit.
//!
//! # What makes collection safe
//!
//! Objects are written *before* the refs that name them, which is what keeps a
//! crash from ever producing a ref pointing at missing bytes. The cost of that
//! ordering is a window in which a perfectly live object has no ref yet, so a
//! naive mark-and-sweep would delete a download that is still in progress.
//! Three independent guards close it, and an object must clear all of them:
//!
//! 1. **Grace period.** An unreferenced object younger than
//!    [`ORPHAN_OBJECT_GRACE`] is never collected.
//! 2. **Live-writer interlock.** If any pull lock in the store is held by a
//!    process that is still running, object collection is skipped entirely.
//! 3. **Complete root set.** The roots come from the raw `refs/` files, not from
//!    validated packs, so an object whose ref is momentarily unreadable still
//!    pins it. If any ref file cannot be parsed the sweep refuses to collect
//!    objects at all rather than act on a root set it knows is incomplete.
//!
//! Staging is different and needs no grace: an `admit-<pid>-<nonce>` entry can
//! only ever be finished by the process that created it, so once that pid is
//! gone the bytes are unconditionally garbage. Resumable download partials share
//! the directory and are deliberately left alone.
//!
//! Mark-and-sweep over a full listing is the intended design at this scale. A
//! store holds tens of objects, so the scan is trivial, and reference counts
//! would add a second piece of mutable state that concurrent installs and
//! removals could disagree with.

use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

use crate::{
    DefaultPackPointer, InstalledPack, PullError,
    content_store::{self, STAGING_DIR_NAME, admit_staging_owner_pid},
};

/// How long an unreferenced object is kept before it may be collected.
///
/// This covers the window between an object landing and the ref that names it
/// becoming durable -- in practice milliseconds, but a paused or throttled
/// download can hold it open far longer, and the cost of waiting is only
/// deferred disk space.
pub const ORPHAN_OBJECT_GRACE: Duration = Duration::from_secs(60 * 60);

/// One model/quant currently served from the content store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelStoreEntry {
    pub pull: String,
    pub model_id: String,
    pub quant: String,
    pub digest: String,
    pub size_bytes: u64,
}

/// Where the space in a model store has gone.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelStoreUsage {
    pub models_dir: PathBuf,
    /// Installed models, largest first.
    pub entries: Vec<ModelStoreEntry>,
    /// Every object present, referenced or not.
    pub objects_total_bytes: u64,
    pub objects_count: usize,
    /// Objects that no ref names. Not all of this is collectable yet -- see
    /// `reclaimable_bytes`.
    pub orphan_object_bytes: u64,
    pub orphan_object_count: usize,
    /// Staging bytes owned by processes that have exited.
    pub dead_staging_bytes: u64,
    pub dead_staging_count: usize,
    /// Legacy per-quant copies still on disk, pending migration.
    pub legacy_copy_bytes: u64,
    pub legacy_copy_count: usize,
    /// What a collection run right now would actually free.
    pub reclaimable_bytes: u64,
    /// Set when object collection is currently withheld, with the reason.
    pub collection_withheld: Option<String>,
}

/// What one collection pass removed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelStoreGcReport {
    pub removed_objects: Vec<String>,
    pub removed_staging: Vec<PathBuf>,
    pub freed_bytes: u64,
    /// Orphans deliberately kept because they are still inside the grace period.
    pub retained_young_orphans: usize,
    /// Set when object collection was skipped; staging collection still ran.
    pub collection_withheld: Option<String>,
}

impl ModelStoreGcReport {
    pub fn is_empty(&self) -> bool {
        self.removed_objects.is_empty() && self.removed_staging.is_empty()
    }
}

/// Report on the model store without changing anything.
pub fn model_store_usage(home: &Path) -> Result<ModelStoreUsage, PullError> {
    let root = models_root(home)?;
    let mut usage = ModelStoreUsage {
        models_dir: root.clone(),
        ..ModelStoreUsage::default()
    };

    let roots = referenced_digests(home, &root)?;
    let objects = content_store::stored_objects(&root)?;
    let now = SystemTime::now();

    let sizes: BTreeMap<&str, u64> = objects
        .iter()
        .map(|object| (object.digest.as_str(), object.size_bytes))
        .collect();
    for pack in installed_refs(&root)? {
        usage.entries.push(ModelStoreEntry {
            pull: pack.pull,
            model_id: pack.model_id,
            quant: pack.quant,
            size_bytes: sizes.get(pack.sha256.as_str()).copied().unwrap_or(0),
            digest: pack.sha256,
        });
    }
    usage.entries.sort_by(|left, right| {
        right
            .size_bytes
            .cmp(&left.size_bytes)
            .then_with(|| left.pull.cmp(&right.pull))
    });

    let withheld = collection_block_reason(&root, &roots);
    for object in &objects {
        usage.objects_total_bytes += object.size_bytes;
        usage.objects_count += 1;
        if roots.digests.contains(&object.digest) {
            continue;
        }
        usage.orphan_object_bytes += object.size_bytes;
        usage.orphan_object_count += 1;
        if withheld.is_none() && is_past_grace(object.modified, now) {
            usage.reclaimable_bytes += object.size_bytes;
        }
    }

    for entry in dead_staging_entries(&root)? {
        usage.dead_staging_bytes += entry.size_bytes;
        usage.dead_staging_count += 1;
        usage.reclaimable_bytes += entry.size_bytes;
    }

    let (legacy_bytes, legacy_count) = legacy_copy_totals(&root);
    usage.legacy_copy_bytes = legacy_bytes;
    usage.legacy_copy_count = legacy_count;
    usage.collection_withheld = withheld;
    Ok(usage)
}

/// Collect dead staging entries and unreferenced objects.
///
/// Staging collection always runs. Object collection is withheld whenever the
/// root set cannot be trusted or another process may be mid-install; the reason
/// is reported rather than silently swallowed.
pub fn collect_model_store_garbage(home: &Path) -> Result<ModelStoreGcReport, PullError> {
    let root = models_root(home)?;
    let mut report = ModelStoreGcReport::default();

    for entry in dead_staging_entries(&root)? {
        match remove_staging_entry(&entry.path) {
            Ok(()) => {
                report.freed_bytes += entry.size_bytes;
                report.removed_staging.push(entry.path);
            }
            // A racing collector or the owner's own cleanup got there first.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(PullError::Io {
                    path: entry.path,
                    source,
                });
            }
        }
    }

    let roots = referenced_digests(home, &root)?;
    if let Some(reason) = collection_block_reason(&root, &roots) {
        report.collection_withheld = Some(reason);
        return Ok(report);
    }

    let now = SystemTime::now();
    for object in content_store::stored_objects(&root)? {
        if roots.digests.contains(&object.digest) {
            continue;
        }
        if !is_past_grace(object.modified, now) {
            report.retained_young_orphans += 1;
            continue;
        }
        report.freed_bytes += content_store::remove_object(&root, &object.digest)?;
        report.removed_objects.push(object.digest);
    }
    Ok(report)
}

/// One ref checked by [`verify_model_store`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelStoreRefVerification {
    pub pull: String,
    pub digest: String,
    /// `None` when the object is present and its bytes hash to `digest`.
    pub failure: Option<String>,
}

/// Result of a full store verification.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelStoreVerification {
    pub models_dir: PathBuf,
    pub checked: Vec<ModelStoreRefVerification>,
}

impl ModelStoreVerification {
    pub fn failures(&self) -> impl Iterator<Item = &ModelStoreRefVerification> {
        self.checked.iter().filter(|check| check.failure.is_some())
    }

    pub fn is_ok(&self) -> bool {
        self.failures().next().is_none()
    }
}

/// Re-hash every object a ref names and confirm it is present and intact.
///
/// This is the deliberate counterpart to *not* verifying on every load: packs
/// are hashed once at admission and then only on demand, so this command is the
/// thing that turns "the path is the checksum" back into a checked claim after
/// hardware faults, restores from backup, or a bug that defeated the seal.
///
/// An object that passes is re-sealed read-only afterwards. That closes the
/// loop with the seal-gated hot paths (declared leases and path-trusted
/// content ids): a store whose seals were stripped -- a backup restore without
/// permission bits, a stray chmod -- runs on the hashing slow path until this
/// command both proves the bytes and restores the invariant they trust. A
/// failed object is left exactly as found.
pub fn verify_model_store(home: &Path) -> Result<ModelStoreVerification, PullError> {
    let root = models_root(home)?;
    let mut verification = ModelStoreVerification {
        models_dir: root.clone(),
        ..ModelStoreVerification::default()
    };
    let mut packs = installed_refs(&root)?;
    packs.sort_by(|left, right| left.pull.cmp(&right.pull));
    for pack in packs {
        // The verifying variant maps the object and compares its bytes against
        // the digest in its own path, which is exactly the invariant under test.
        // This is the one place that deliberately pays a full read per pack.
        let failure = match content_store::open_verified_lease(&root, &pack.sha256) {
            Ok(lease) => {
                let failure = (lease.bytes().len() as u64 != pack.size_bytes).then(|| {
                    format!(
                        "object is {} bytes, ref records {}",
                        lease.bytes().len(),
                        pack.size_bytes
                    )
                });
                if failure.is_none() {
                    // Verified intact: restore the seal if it was lost. Best
                    // effort on purpose -- the report above is this command's
                    // authority, and a chmod failure only means this object
                    // stays on the hash-verifying slow path (safe, never
                    // wrong). Never seal an object that failed verification.
                    if let Ok(path) = content_store::object_path(&root, &pack.sha256) {
                        let _ = content_store::seal_object(&path);
                    }
                }
                failure
            }
            Err(error) => Some(error.to_string()),
        };
        verification.checked.push(ModelStoreRefVerification {
            pull: pack.pull,
            digest: pack.sha256,
            failure,
        });
    }
    Ok(verification)
}

/// Every digest a ref or the default pointer names, plus whether the scan was
/// complete enough to act on.
#[derive(Debug, Default)]
struct RootSet {
    digests: HashSet<String>,
    /// Ref files that exist but could not be read or parsed. Any of these means
    /// the root set may be missing a live digest.
    unreadable: Vec<PathBuf>,
}

/// Build the mark phase's root set from the raw `refs/` tree.
///
/// Deliberately *not* built from `InstalledModelStore`, which drops a ref whose
/// object currently fails validation. Such a ref is exactly the case where the
/// object must be preserved: dropping it from the roots would let the sweep
/// delete the bytes a repairable ref still points at.
fn referenced_digests(home: &Path, root: &Path) -> Result<RootSet, PullError> {
    let mut set = RootSet::default();
    let refs_root = root.join("refs");
    match fs::read_dir(&refs_root) {
        Ok(model_dirs) => {
            for model_dir in model_dirs {
                let Ok(model_dir) = model_dir else {
                    // A per-entry read failure (e.g. the OS raced a delete) is
                    // as much a hole in the root set as an unreadable ref file.
                    set.unreadable.push(refs_root.clone());
                    continue;
                };
                let Ok(ref_files) = fs::read_dir(model_dir.path()) else {
                    set.unreadable.push(model_dir.path());
                    continue;
                };
                for ref_file in ref_files {
                    let Ok(ref_file) = ref_file else {
                        set.unreadable.push(model_dir.path());
                        continue;
                    };
                    let path = ref_file.path();
                    if path.extension().and_then(|value| value.to_str()) != Some("json") {
                        continue;
                    }
                    match fs::read_to_string(&path)
                        .ok()
                        .and_then(|contents| serde_json::from_str::<InstalledPack>(&contents).ok())
                    {
                        Some(pack) => {
                            set.digests.insert(pack.sha256);
                        }
                        None => set.unreadable.push(path),
                    }
                }
            }
        }
        // No refs/ directory at all is a legitimate empty store (a fresh
        // install before anything has ever landed).
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        // Any other error -- permission denied, I/O failure -- means the root
        // set cannot be trusted. Recording it as unreadable, not silently
        // treating the store as empty, is what makes `collection_block_reason`
        // withhold collection instead of authorizing it against a blind spot.
        Err(_) => set.unreadable.push(refs_root.clone()),
    }

    // The default pointer names a pack independently of the refs tree; treat it
    // as a root so a store whose ref was removed out from under it still keeps
    // the bytes the pointer advertises.
    let pointer_path = crate::pull::default_pack_pointer_path(home);
    match fs::read_to_string(&pointer_path) {
        Ok(contents) => match serde_json::from_str::<DefaultPackPointer>(&contents) {
            Ok(pointer) => {
                set.digests.insert(pointer.sha256);
            }
            Err(_) => set.unreadable.push(pointer_path),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => set.unreadable.push(pointer_path),
    }
    Ok(set)
}

/// Packs currently served by refs, for usage accounting.
fn installed_refs(root: &Path) -> Result<Vec<InstalledPack>, PullError> {
    let mut packs = Vec::new();
    let refs_root = root.join("refs");
    let model_dirs = match fs::read_dir(&refs_root) {
        Ok(model_dirs) => model_dirs,
        // No refs/ directory at all is a legitimate empty store.
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(packs),
        // Any other error must surface as a failure, not report an empty
        // store: usage/verify accounting would otherwise silently show zero
        // installed packs while the directory it could not read still holds
        // the real answer.
        Err(source) => {
            return Err(PullError::Io {
                path: refs_root,
                source,
            });
        }
    };
    for model_dir in model_dirs.flatten() {
        let Ok(ref_files) = fs::read_dir(model_dir.path()) else {
            continue;
        };
        for ref_file in ref_files.flatten() {
            let path = ref_file.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            if let Some(pack) = fs::read_to_string(&path)
                .ok()
                .and_then(|contents| serde_json::from_str::<InstalledPack>(&contents).ok())
            {
                packs.push(pack);
            }
        }
    }
    Ok(packs)
}

/// Why object collection must not run right now, if it must not.
fn collection_block_reason(root: &Path, roots: &RootSet) -> Option<String> {
    if let Some(path) = roots.unreadable.first() {
        return Some(format!(
            "{} model ref(s) could not be read, starting with '{}'; \
             refusing to collect objects against an incomplete root set",
            roots.unreadable.len(),
            path.display()
        ));
    }
    if let Some(path) = live_pull_lock(root) {
        return Some(format!(
            "another process holds the pull lock '{}'; \
             its content may not have a ref yet",
            path.display()
        ));
    }
    None
}

/// A pull lock whose owning process is still running, if any.
///
/// An install holds this from before its object lands until after its ref is
/// durable, so a live lock is direct evidence that an object without a ref may
/// still be legitimate.
fn live_pull_lock(root: &Path) -> Option<PathBuf> {
    let staging = root.join(STAGING_DIR_NAME);
    let entries = fs::read_dir(staging).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("lock") {
            continue;
        }
        let Ok(contents) = fs::read_to_string(&path) else {
            // Unreadable lock: assume it is held rather than assume it is not.
            return Some(path);
        };
        let owner = contents
            .lines()
            .find_map(|line| line.strip_prefix("pid="))
            .and_then(|value| value.trim().parse::<u32>().ok());
        match owner {
            Some(pid) if crate::pull::process_is_gone(pid) => {}
            _ => return Some(path),
        }
    }
    None
}

struct StagingEntry {
    path: PathBuf,
    size_bytes: u64,
}

/// Staging entries whose owning process has exited.
///
/// Only `admit-<pid>-<nonce>` entries qualify. Download partials in the same
/// directory are resumable by any later process and are never touched here;
/// deleting them would silently throw away a mostly-finished multi-gigabyte
/// download.
fn dead_staging_entries(root: &Path) -> Result<Vec<StagingEntry>, PullError> {
    let staging = root.join(STAGING_DIR_NAME);
    let Ok(entries) = fs::read_dir(&staging) else {
        return Ok(Vec::new());
    };
    let mut dead = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        let Some(pid) = admit_staging_owner_pid(name) else {
            continue;
        };
        if !crate::pull::process_is_gone(pid) {
            continue;
        }
        let Ok(metadata) = fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.file_type().is_symlink() {
            continue;
        }
        dead.push(StagingEntry {
            size_bytes: metadata.len(),
            path,
        });
    }
    dead.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(dead)
}

fn remove_staging_entry(path: &Path) -> std::io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        Err(error) => Err(error),
    }
}

/// Bytes still held by unmigrated `<models>/<model>/<quant>/` trees.
fn legacy_copy_totals(root: &Path) -> (u64, usize) {
    let mut bytes = 0;
    let mut count = 0;
    let Ok(model_dirs) = fs::read_dir(root) else {
        return (bytes, count);
    };
    for model_dir in model_dirs.flatten() {
        if matches!(
            model_dir.file_name().to_str(),
            Some("objects" | "refs" | STAGING_DIR_NAME | "locks")
        ) {
            continue;
        }
        let Ok(quant_dirs) = fs::read_dir(model_dir.path()) else {
            continue;
        };
        for quant_dir in quant_dirs.flatten() {
            if !quant_dir.path().join("installed.json").is_file() {
                continue;
            }
            count += 1;
            if let Ok(entries) = fs::read_dir(quant_dir.path()) {
                for entry in entries.flatten() {
                    bytes += entry
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
        }
    }
    (bytes, count)
}

fn is_past_grace(modified: Option<SystemTime>, now: SystemTime) -> bool {
    let Some(modified) = modified else {
        // An object whose age cannot be established is treated as brand new.
        return false;
    };
    now.duration_since(modified)
        .is_ok_and(|age| age >= ORPHAN_OBJECT_GRACE)
}

fn models_root(home: &Path) -> Result<PathBuf, PullError> {
    // A corrupt config.json must fail closed here, not fall back to the
    // default `<home>/models` -- these callers run destructive commands (GC,
    // verify), and silently acting against the wrong directory because the
    // real one could not be resolved is worse than refusing to run.
    let config = crate::config::load_config(home)?;
    Ok(crate::config::models_dir(home, &config))
}

#[cfg(test)]
mod tests;
