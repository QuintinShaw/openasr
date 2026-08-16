//! The immutable object half of the model store: `<models>/objects/sha256/`.
//!
//! An object is written once, named by the SHA-256 of its own bytes, and never
//! modified. Refs under `<models>/refs/` point at objects; this module owns
//! everything about the objects themselves.
//!
//! # Integrity chain
//!
//! Content addressing rests on "the path is the checksum". Three things keep
//! that true, and they are meant to be read together -- weakening any one of
//! them shifts load onto the others:
//!
//! 1. **The digest is established once, on the bytes actually written.**
//!    [`admit_file`] hashes its own private staging copy while holding the sole
//!    descriptor to it, and the caller's preflight runs against that same
//!    descriptor. A source file mutated mid-admission cannot change what was
//!    admitted. Above this module, install additionally checks the digest
//!    against the signed catalog before anything is admitted at all.
//! 2. **The object is sealed read-only once it lands** ([`seal_object`]).
//!    This is what makes step 1's answer keep holding: the bytes behind a digest
//!    cannot be edited in place afterwards, so a later reader does not have to
//!    re-derive what is already known. **If this seal is ever removed, the
//!    per-load check in [`open_declared_lease`] is no longer sufficient and the
//!    load path has to go back to re-hashing.** The same seal also gates every
//!    identity consumer that skips hashing: [`trusted_object_digest`] hands the
//!    digest out only while the seal is observably intact *and* the path is
//!    anchored under the caller's own models root, so a defeated seal or a
//!    same-shaped path outside that root falls back to full verification
//!    instead of silently trusting.
//! 3. **Full verification stays available and is used where it decides
//!    something.** [`open_verified_lease`] re-hashes when an existing object's
//!    digest claim must be re-established;
//!    `openasr model-pack verify` re-hashes the whole store on demand, and
//!    re-seals each intact object afterwards so a store whose seals were lost
//!    (a backup restore without permissions, say) earns the fast path back.
//!
//! What deliberately does *not* happen is re-hashing on every load. Reading a
//! multi-gigabyte pack again on each model switch costs real startup latency,
//! and the only thing it would catch is someone who defeated the seal by hand
//! inside their own home directory -- who could equally rewrite the ref beside
//! it. [`open_declared_lease`] therefore checks structure and length only, and
//! the runtime content identity (`GgmlRuntimeSource::content_id`, which keys
//! every family's runtime cache) takes the digest straight from the object's
//! own path via [`trusted_object_digest`]. Both hot paths rest on the same
//! three points; neither is a separate trust decision.
//!
//! # What may write into the object namespace
//!
//! Trusting a digest without reading the bytes is only sound if every object
//! that can exist under **a given models root's** `objects/sha256/` was
//! verified at least once before it became reachable there. That qualifier is
//! load-bearing: the layout `objects/sha256/<digest>/content` is just a shape,
//! and a shape can be recreated anywhere on disk by anyone who can write a
//! file -- a user-supplied pack directory, a dev fixture, a zip extracted by a
//! converter tool. [`trusted_object_digest`] therefore never trusts the shape
//! alone; it additionally requires the path to fall under the caller's own
//! `models_root`, the one directory whose `objects/sha256/` this module's sole
//! writer actually populates. [`admit_file`] upholds the
//! verified-before-reachable half of the invariant: it hashes every byte while
//! writing its private staging copy, maps that same held descriptor, runs the
//! caller's preflight, and only then links/renames
//! the bytes into place and seals them. Legacy migration deliberately uses
//! this same seam instead of owning a path-based zero-copy exception.
//! In-flight writes live in `staging/` beside `objects/`, never inside the
//! content namespace, so an object path can never observe a torn or
//! unverified write -- at worst a crash leaves a verified-but-unsealed object,
//! which the seal-gated identity fast path simply answers with a full hash
//! until `verify` re-seals it.

use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use memmap2::Mmap;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::atomic_file;

#[derive(Debug, Error)]
pub enum ContentStoreError {
    #[error("could not read or write immutable model content '{path}': {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("immutable model content digest must be a lowercase SHA-256 hex string: {digest}")]
    InvalidDigest { digest: String },
    #[error("immutable model content changed while being admitted: {path}")]
    SourceChanged { path: PathBuf },
    #[error("immutable object collision at '{path}'")]
    ObjectCollision { path: PathBuf },
    #[error(transparent)]
    Preflight(#[from] Box<crate::PullError>),
}

/// An owned, immutable checkout of one content-addressed object.
///
/// The descriptor is intentionally retained with the mapping. Consumers that need
/// metadata, tensor indexes, or runtime contract validation must use this lease,
/// rather than reopening `path()`: a pathname can be renamed between independent
/// validation stages while an opened descriptor cannot change identity.
pub struct ContentLease {
    digest: String,
    path: PathBuf,
    file: File,
    mmap: Arc<Mmap>,
}

impl std::fmt::Debug for ContentLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ContentLease")
            .field("digest", &self.digest)
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl ContentLease {
    pub fn digest(&self) -> &str {
        &self.digest
    }
    /// Display and diagnostics only. It is never an authority for rereading pack bytes.
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn file(&self) -> &File {
        &self.file
    }
    pub fn bytes(&self) -> &[u8] {
        self.mmap.as_ref()
    }
    pub fn mmap(&self) -> Arc<Mmap> {
        Arc::clone(&self.mmap)
    }

    pub fn read_magic(&self) -> Result<[u8; 4], ContentStoreError> {
        let Some(head) = self.bytes().get(..4) else {
            return Err(ContentStoreError::Io {
                path: self.path.clone(),
                source: io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "pack is shorter than GGUF magic",
                ),
            });
        };
        Ok([head[0], head[1], head[2], head[3]])
    }

    /// Hash the bytes held by this lease, never by reopening its display path.
    pub fn sha256(&self) -> String {
        sha256_bytes(self.bytes())
    }

    fn from_file(digest: String, path: PathBuf, file: File) -> Result<Self, ContentStoreError> {
        let mmap = unsafe { Mmap::map(&file) }
            .map(Arc::new)
            .map_err(|source| ContentStoreError::Io {
                path: path.clone(),
                source,
            })?;
        Ok(Self {
            digest,
            path,
            file,
            mmap,
        })
    }

    fn with_path(mut self, path: PathBuf) -> Self {
        self.path = path;
        self
    }
}

#[derive(Debug)]
pub(crate) struct AdmittedContent<P = ()> {
    pub digest: String,
    pub size_bytes: u64,
    pub object_path: PathBuf,
    lease: ContentLease,
    proof: P,
}

impl<P> AdmittedContent<P> {
    /// Consume the validation lease for the object just admitted. It remains
    /// valid even after the private staging name has been unlinked.
    pub(crate) fn into_parts(self) -> (ContentLease, P) {
        (self.lease, self.proof)
    }

    pub(crate) fn proof(&self) -> &P {
        &self.proof
    }

    pub(crate) fn map_proof<Q>(self, map: impl FnOnce(P) -> Q) -> AdmittedContent<Q> {
        AdmittedContent {
            digest: self.digest,
            size_bytes: self.size_bytes,
            object_path: self.object_path,
            lease: self.lease,
            proof: map(self.proof),
        }
    }
}

impl AdmittedContent<()> {
    #[cfg(test)]
    pub(crate) fn into_lease(self) -> ContentLease {
        self.lease
    }
}

/// Every in-flight write lands here first, one level *beside* `objects/` rather
/// than inside it. Keeping transient bytes out of the content namespace is what
/// lets garbage collection treat `objects/sha256/*` as "digest names only", with
/// no special-casing of transaction prefixes.
pub(crate) const STAGING_DIR_NAME: &str = "staging";

/// Prefix for the private staging file one `admit_file` call owns end to end.
///
/// The name carries the creating pid because that admission is *not* resumable:
/// nothing but the process that started it can ever finish it, so once that pid
/// is gone the bytes are unconditionally garbage. Download partials share this
/// directory but are deliberately resumable across processes, hence the distinct
/// prefix -- collection must be able to tell the two apart by name alone.
pub(crate) const ADMIT_STAGING_PREFIX: &str = "admit-";
pub(crate) const ADMIT_STAGING_SUFFIX: &str = ".tmp";

/// Owning pid of an `admit_file` staging entry, or `None` for anything else in
/// the staging directory (download partials, locks, foreign files).
pub(crate) fn admit_staging_owner_pid(file_name: &str) -> Option<u32> {
    file_name
        .strip_prefix(ADMIT_STAGING_PREFIX)?
        .strip_suffix(ADMIT_STAGING_SUFFIX)?
        .split_once('-')
        .and_then(|(pid, _nonce)| pid.parse::<u32>().ok())
}

pub(crate) fn objects_root(models_root: &Path) -> PathBuf {
    models_root.join("objects").join("sha256")
}

/// Make a landed object read-only.
///
/// Content addressing rests entirely on "the path is the checksum". An
/// accidental in-place rewrite would break that premise *silently*: every later
/// dedup, ref, and size check would keep trusting a digest the bytes no longer
/// have. Sealing costs one `chmod` and removes the whole class.
///
/// Two callers: admission/migration seal once the verified bytes land, and
/// `verify_model_store` re-seals an object immediately after a full re-hash
/// has confirmed it intact -- that is how a store whose seals were lost (a
/// backup restore without permission bits) re-establishes the invariant the
/// seal-gated hot paths trust. Re-sealing is idempotent.
///
/// Removal is unaffected on Unix (unlink is governed by the parent directory),
/// and `unseal_object_for_removal` clears the bit where it is not.
pub(crate) fn seal_object(path: &Path) -> Result<(), ContentStoreError> {
    let mut permissions = fs::metadata(path)
        .map_err(|source| ContentStoreError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .permissions();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        permissions.set_mode(0o444);
    }
    #[cfg(not(unix))]
    {
        permissions.set_readonly(true);
    }
    fs::set_permissions(path, permissions).map_err(|source| ContentStoreError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Windows refuses to unlink a read-only file, so collection has to clear the
/// bit first. On Unix this is a no-op: the seal never blocked unlink there.
#[cfg_attr(not(unix), allow(clippy::permissions_set_readonly_false))]
fn unseal_object_for_removal(path: &Path) {
    #[cfg(not(unix))]
    if let Ok(metadata) = fs::metadata(path) {
        let mut permissions = metadata.permissions();
        permissions.set_readonly(false);
        let _ = fs::set_permissions(path, permissions);
    }
    #[cfg(unix)]
    let _ = path;
}

/// Layout of one immutable object: `<models>/objects/sha256/<digest>/content`.
/// The per-digest directory is part of the contract -- `InstalledModelStore`
/// (and the CLI/server stores already on disk) resolve refs against exactly this
/// path, so the bytes must never be written directly at `<digest>`.
///
/// Installed packs carry no `.oasr` suffix on disk: content addressing names
/// the file by its role under a digest directory. So the user-facing extension
/// contract (`has_openasr_runtime_pack_extension`) cannot be the only way a
/// caller-supplied path is recognised as a pack -- without this predicate,
/// pointing any CLI command at an installed pack is rejected as "must end with
/// .oasr". This is the layout half of that contract and lives here because this
/// module owns the layout.
///
/// Purely structural: it authorizes nothing on its own, and every consumer
/// still probes the container itself.
pub fn is_content_addressed_object_path(path: &Path) -> bool {
    object_digest_from_path(path).is_some()
}

/// The digest named by an object path, when `path` has exactly the
/// `<models>/objects/sha256/<digest>/content` layout.
///
/// This is the single parser for the object layout: [`is_content_addressed_object_path`]
/// and every "trust the digest without hashing" fast path are built on it, so
/// the layout contract has exactly one implementation. What the digest
/// *means* for a caller depends on the caller -- see [`trusted_object_digest`]
/// for the integrity-gated form identity consumers must use.
pub(crate) fn object_digest_from_path(path: &Path) -> Option<&str> {
    if path.file_name().and_then(|name| name.to_str()) != Some("content") {
        return None;
    }
    let digest_dir = path.parent()?;
    let digest = digest_dir.file_name().and_then(|name| name.to_str())?;
    validate_digest(digest).ok()?;
    let algorithm_dir = digest_dir.parent()?;
    if algorithm_dir.file_name().and_then(|name| name.to_str()) != Some("sha256") {
        return None;
    }
    if algorithm_dir
        .parent()
        .and_then(|objects| objects.file_name())
        .and_then(|name| name.to_str())
        != Some("objects")
    {
        return None;
    }
    Some(digest)
}

/// The digest an object path names, when the object may be trusted without
/// reading its bytes: the path has the exact object layout, falls under
/// `models_root`'s own `objects/sha256/`, *and* the seal is observably intact
/// (the file is read-only).
///
/// This is the hot-path form of the module's integrity chain. Content
/// addressing's premise is that the path *is* the checksum, and the three
/// points at the top of this module make that premise hold for any sealed
/// object *inside the store the caller actually owns*: the digest was
/// established once over the bytes actually written (and, above this module,
/// checked against the signed catalog), and the read-only seal is what keeps
/// those bytes from changing afterwards. Neither of those points says
/// anything about a file elsewhere on disk that merely happens to be named
/// `<anything>/objects/sha256/<64 lowercase hex>/content` -- the layout is a
/// public shape, not a proof of provenance, and admission never ran against
/// bytes outside `models_root`. The `models_root` check is what ties the
/// shape back to a store this module's sole writer ([`admit_file`]) actually
/// populated; see the module docs' "What
/// may write into the object namespace" section. While all three hold,
/// handing the digest out without a multi-gigabyte re-read is not skipping
/// verification -- the verification already happened at admission and its
/// result has been pinned in place ever since. Full re-verification remains
/// one `openasr model-pack verify` away for anyone who wants to test the
/// claim again (bit rot, a bad backup restore, suspicion of tampering).
///
/// The seal check is deliberately an *observable gate*, not a tamper-proof
/// guarantee: anyone with write access inside the store could chmod, rewrite,
/// and chmod back -- but that same actor can rewrite the ref beside the
/// object, and the documented threat model does not try to defend the user
/// from themselves inside their own home directory. What the gate does buy is
/// graceful fail-closed degradation for everything short of deliberate
/// tampering: a seal lost to a permissions-stripping restore, or defeated by a
/// buggy tool, flips every consumer of this function back onto a full hash --
/// which both verifies the bytes and (via `verify`) re-seals them.
///
/// On mounts where permission bits carry no real signal -- some SMB/exFAT
/// setups report every file as writable, others every file as read-only --
/// the gate degrades in whichever direction the mount lies, and neither
/// direction breaks safety. Nothing reads as sealed and every consumer falls
/// back to the hashing path it had before this gate existed; or everything
/// reads as sealed and the trust is still sound, because an object's
/// correctness never came from the permission bit -- admission hashed the
/// bytes before the object existed at all, and only the admission writer can
/// create one *within the given models root* (see the module docs). Exploiting a
/// mount that lies about *writability* still takes local write access, which
/// the threat model already excludes; bit rot under such a mount is exactly
/// what `verify` re-hashes for.
///
/// `sealed` must describe the file `path` actually refers to -- take it from
/// an already-open descriptor's metadata where one exists, so a path swapped
/// between stat and open cannot change which file the seal verdict applies to.
///
/// `models_root` must be the caller's own resolved model-store root (the same
/// value `admit_file` was called with), never a value
/// derived from `path` itself -- deriving it from `path` would make the
/// anchor check trivially satisfiable by construction and defeat the point.
pub(crate) fn trusted_object_digest<'a>(
    path: &'a Path,
    sealed: bool,
    models_root: &Path,
) -> Option<&'a str> {
    if !sealed || !path.starts_with(objects_root(models_root)) {
        return None;
    }
    object_digest_from_path(path)
}

/// The model-store root this process resolves installed packs under, when no
/// more specific root is already in scope: `OPENASR_HOME` (or the user-home
/// default) plus any `config.json`/`OPENASR_MODELS_DIR` override -- the exact
/// resolution every install path in this crate uses to decide where
/// [`admit_file`] is allowed to write (see
/// `pull.rs`'s own `models_root(home)` and `config::models_dir`'s doc
/// comment). This product commits to exactly one home per process (the
/// `OPENASR_HOME=...` convention documented in `AGENTS.md`), so this is not a
/// guess about where the store might be -- it is the only root a caller
/// without a more precise one (an already-resolved `home`/`PullPaths`) could
/// mean.
///
/// Prefer a locally-known root over this whenever one is already in scope
/// (e.g. `pull`'s `PullPaths`, which may have been built against an explicit,
/// non-default `home` in a test or a future multi-home caller): this function
/// exists for identity resolvers that only ever hold a bare path, such as
/// [`crate::GgmlRuntimeSource::content_id`].
///
/// Returns `None` when the process has no resolvable home at all. Callers
/// must treat that as "no anchor to check against" and fall back to hashing
/// -- never as license to trust an unanchored path anyway.
pub(crate) fn default_models_root() -> Option<PathBuf> {
    let home = crate::home::openasr_home().ok()?;
    let config = crate::config::load_config(&home).unwrap_or_default();
    Some(crate::config::models_dir(&home, &config))
}

pub(crate) fn object_path(models_root: &Path, digest: &str) -> Result<PathBuf, ContentStoreError> {
    validate_digest(digest)?;
    Ok(objects_root(models_root).join(digest).join("content"))
}

/// Open and map one immutable object, re-hashing its bytes.
///
/// The cold path of the store's hot/cold split: use this wherever the answer
/// authorizes destroying or skipping some *other* copy of the same content --
/// adopting an existing object instead of the bytes just staged, dropping a
/// legacy pack because an object already holds it, or an explicit `verify`.
/// In those places the digest is a claim being tested, and paying a full read
/// is the entire point.
///
/// For simply loading an installed pack use [`open_declared_lease`]: re-reading
/// a gigabyte on every model switch buys nothing the admission-time check and
/// the read-only seal have not already established.
pub(crate) fn open_verified_lease(
    models_root: &Path,
    digest: &str,
) -> Result<ContentLease, ContentStoreError> {
    let (path, lease) = open_object(models_root, digest)?;
    if lease.sha256() != digest {
        return Err(ContentStoreError::SourceChanged { path });
    }
    Ok(lease)
}

/// Open and map one immutable object for use, checking only that it is the
/// regular file of the expected length that its ref describes.
///
/// `size_bytes` comes from the ref, which is the store's own record of what it
/// admitted. The digest is *not* recomputed here on purpose -- see the module
/// documentation's integrity chain for what stands behind it.
pub(crate) fn open_declared_lease(
    models_root: &Path,
    digest: &str,
    size_bytes: u64,
) -> Result<ContentLease, ContentStoreError> {
    let (path, lease) = open_object(models_root, digest)?;
    if lease.bytes().len() as u64 != size_bytes {
        return Err(ContentStoreError::SourceChanged { path });
    }
    Ok(lease)
}

/// Shared open+map, with the structural checks both entry points need.
///
/// The regular-file check is made against the *descriptor*, not the path, so a
/// pathname swapped between the open and the check cannot change what was
/// actually mapped.
fn open_object(
    models_root: &Path,
    digest: &str,
) -> Result<(PathBuf, ContentLease), ContentStoreError> {
    let path = object_path(models_root, digest)?;
    let file = File::open(&path).map_err(|source| ContentStoreError::Io {
        path: path.clone(),
        source,
    })?;
    let metadata = file.metadata().map_err(|source| ContentStoreError::Io {
        path: path.clone(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(ContentStoreError::Io {
            path: path.clone(),
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "content-addressed object is not a regular file",
            ),
        });
    }
    let lease = ContentLease::from_file(digest.to_string(), path.clone(), file)?;
    Ok((path, lease))
}

/// Copy one opened source descriptor into a private staging file, fsync it,
/// validate its mapped bytes, then create an immutable object. `preflight` is
/// deliberately lease-based: it cannot accidentally reopen a mutable pathname.
pub(crate) fn admit_file<P>(
    source_path: &Path,
    models_root: &Path,
    preflight: impl Fn(&ContentLease) -> Result<P, crate::PullError>,
) -> Result<AdmittedContent<P>, ContentStoreError> {
    let source_metadata =
        fs::symlink_metadata(source_path).map_err(|source| ContentStoreError::Io {
            path: source_path.to_path_buf(),
            source,
        })?;
    if source_metadata.file_type().is_symlink() {
        return Err(ContentStoreError::SourceChanged {
            path: source_path.to_path_buf(),
        });
    }
    let mut source = File::open(source_path).map_err(|source| ContentStoreError::Io {
        path: source_path.to_path_buf(),
        source,
    })?;
    let before = source.metadata().map_err(|source| ContentStoreError::Io {
        path: source_path.to_path_buf(),
        source,
    })?;
    if !before.is_file() {
        return Err(ContentStoreError::Io {
            path: source_path.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "model pack must be a regular file",
            ),
        });
    }

    let staging_dir = models_root.join(STAGING_DIR_NAME);
    fs::create_dir_all(&staging_dir).map_err(|source| ContentStoreError::Io {
        path: staging_dir.clone(),
        source,
    })?;
    let staging = staging_dir.join(format!(
        "{ADMIT_STAGING_PREFIX}{}{ADMIT_STAGING_SUFFIX}",
        unique_suffix()
    ));
    let result = (|| {
        // Read+write keeps the sole staging descriptor open through copy, sync,
        // mapping, hashing, and preflight. There is no pathname reopen in that chain.
        let mut output = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&staging)
            .map_err(|source| ContentStoreError::Io {
                path: staging.clone(),
                source,
            })?;
        let mut hash = Sha256::new();
        let mut size = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let count = source
                .read(&mut buffer)
                .map_err(|source| ContentStoreError::Io {
                    path: source_path.to_path_buf(),
                    source,
                })?;
            if count == 0 {
                break;
            }
            output
                .write_all(&buffer[..count])
                .map_err(|source| ContentStoreError::Io {
                    path: staging.clone(),
                    source,
                })?;
            hash.update(&buffer[..count]);
            size = size
                .checked_add(count as u64)
                .ok_or_else(|| ContentStoreError::Io {
                    path: staging.clone(),
                    source: io::Error::other("model pack exceeds supported size"),
                })?;
        }
        output.sync_all().map_err(|source| ContentStoreError::Io {
            path: staging.clone(),
            source,
        })?;
        let after = source.metadata().map_err(|source| ContentStoreError::Io {
            path: source_path.to_path_buf(),
            source,
        })?;
        if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
            return Err(ContentStoreError::SourceChanged {
                path: source_path.to_path_buf(),
            });
        }
        output
            .seek(SeekFrom::Start(0))
            .map_err(|source| ContentStoreError::Io {
                path: staging.clone(),
                source,
            })?;
        let digest = format!("{:x}", hash.finalize());
        let lease = ContentLease::from_file(digest.clone(), staging.clone(), output)?;
        // `digest` was updated from the exact slices successfully written to
        // this private, still-held descriptor. Re-hashing the mmap here would
        // add a second O(n) SHA-256 pass over multi-gigabyte packs without a
        // distinct trust boundary; length plus the held descriptor is the
        // needed write/map consistency check.
        if lease.bytes().len() as u64 != size {
            return Err(ContentStoreError::SourceChanged {
                path: source_path.to_path_buf(),
            });
        }
        let proof =
            preflight(&lease).map_err(|error| ContentStoreError::Preflight(Box::new(error)))?;

        let object = object_path(models_root, &digest)?;
        let parent = object.parent().expect("object path has parent");
        fs::create_dir_all(parent).map_err(|source| ContentStoreError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
        let (lease, proof) = match fs::hard_link(&staging, &object) {
            Ok(()) => {
                atomic_file::sync_parent_dir_best_effort(&object);
                // Drop the staging name before sealing: it shares the inode with
                // the object, and a read-only staging entry is needlessly awkward
                // to unlink on the platforms that enforce the bit.
                fs::remove_file(&staging).map_err(|source| ContentStoreError::Io {
                    path: staging.clone(),
                    source,
                })?;
                seal_object(&object)?;
                (lease.with_path(object.clone()), proof)
            }
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
                // A digest name is not evidence. Revalidate the existing object's
                // descriptor and contract before reusing it; never replace it.
                let existing = open_verified_lease(models_root, &digest)?;
                if existing.bytes().len() as u64 != size {
                    return Err(ContentStoreError::ObjectCollision { path: object });
                }
                let proof = preflight(&existing)
                    .map_err(|error| ContentStoreError::Preflight(Box::new(error)))?;
                fs::remove_file(&staging).map_err(|source| ContentStoreError::Io {
                    path: staging.clone(),
                    source,
                })?;
                (existing, proof)
            }
            Err(_) => match fs::rename(&staging, &object) {
                Ok(()) => {
                    atomic_file::sync_parent_dir_best_effort(&object);
                    seal_object(&object)?;
                    (lease.with_path(object.clone()), proof)
                }
                Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
                    let existing = open_verified_lease(models_root, &digest)?;
                    if existing.bytes().len() as u64 != size {
                        return Err(ContentStoreError::ObjectCollision { path: object });
                    }
                    let proof = preflight(&existing)
                        .map_err(|error| ContentStoreError::Preflight(Box::new(error)))?;
                    fs::remove_file(&staging).map_err(|source| ContentStoreError::Io {
                        path: staging.clone(),
                        source,
                    })?;
                    (existing, proof)
                }
                Err(source) => {
                    return Err(ContentStoreError::Io {
                        path: object.clone(),
                        source,
                    });
                }
            },
        };
        Ok(AdmittedContent {
            digest,
            size_bytes: size,
            object_path: object,
            lease,
            proof,
        })
    })();
    if result.is_err() {
        let _ = fs::remove_file(&staging);
    }
    result
}

pub(crate) fn remove_object_if_unreferenced(
    models_root: &Path,
    digest: &str,
    referenced: bool,
) -> Result<(), ContentStoreError> {
    if !referenced {
        remove_object(models_root, digest)?;
    }
    Ok(())
}

/// Unlink one object and its per-digest directory. Returns the bytes reclaimed,
/// or `0` when the object was already gone.
pub(crate) fn remove_object(models_root: &Path, digest: &str) -> Result<u64, ContentStoreError> {
    let path = object_path(models_root, digest)?;
    let size = fs::symlink_metadata(&path).map(|metadata| metadata.len());
    unseal_object_for_removal(&path);
    match fs::remove_file(&path) {
        Ok(()) => {
            atomic_file::sync_parent_dir_best_effort(&path);
            // The per-digest directory only ever holds `content`; drop it
            // once empty so a collected object leaves nothing behind.
            if let Some(parent) = path.parent() {
                let _ = fs::remove_dir(parent);
            }
            Ok(size.unwrap_or(0))
        }
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(0),
        Err(source) => Err(ContentStoreError::Io { path, source }),
    }
}

/// One object present under `objects/sha256/`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoredObject {
    pub digest: String,
    pub size_bytes: u64,
    /// Last modification of the object's own directory entry, used as the
    /// "how long has this been here" clock for orphan grace.
    pub modified: Option<std::time::SystemTime>,
}

/// Enumerate every well-formed object in the store.
///
/// A directory whose name is not a valid digest, or that holds no readable
/// `content`, is skipped rather than reported: collection must never act on a
/// path it could not positively identify as an object.
pub(crate) fn stored_objects(models_root: &Path) -> Result<Vec<StoredObject>, ContentStoreError> {
    let root = objects_root(models_root);
    let entries = match fs::read_dir(&root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => return Err(ContentStoreError::Io { path: root, source }),
    };
    let mut objects = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| ContentStoreError::Io {
            path: root.clone(),
            source,
        })?;
        let Some(digest) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if validate_digest(&digest).is_err() {
            continue;
        }
        let content = entry.path().join("content");
        let Ok(metadata) = fs::symlink_metadata(&content) else {
            continue;
        };
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            continue;
        }
        objects.push(StoredObject {
            digest,
            size_bytes: metadata.len(),
            modified: metadata.modified().ok(),
        });
    }
    objects.sort_by(|left, right| left.digest.cmp(&right.digest));
    Ok(objects)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_digest(digest: &str) -> Result<(), ContentStoreError> {
    if digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        Ok(())
    } else {
        Err(ContentStoreError::InvalidDigest {
            digest: digest.to_string(),
        })
    }
}

fn unique_suffix() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    )
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn admission_keeps_preflight_on_the_copied_descriptor_when_source_is_replaced() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.oasr");
        let replacement = temp.path().join("replacement.oasr");
        fs::write(&source, b"GGUF-original-pack").unwrap();
        fs::write(&replacement, b"GGUF-replacement-pack").unwrap();
        let expected = sha256_bytes(b"GGUF-original-pack");

        let admitted = admit_file(&source, &temp.path().join("models"), |lease| {
            fs::rename(&replacement, &source).unwrap();
            assert_eq!(lease.bytes(), b"GGUF-original-pack");
            assert_eq!(lease.sha256(), expected);
            Ok(())
        })
        .unwrap();

        assert_eq!(admitted.digest, expected);
        assert_eq!(admitted.into_lease().bytes(), b"GGUF-original-pack");
        assert_eq!(fs::read(&source).unwrap(), b"GGUF-replacement-pack");
    }

    #[test]
    fn admission_returns_the_proof_for_the_exact_landed_generation() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.oasr");
        fs::write(&source, b"GGUF-proof-bearing-pack").unwrap();

        let admitted = admit_file(&source, &temp.path().join("models"), |lease| {
            Ok(lease.sha256())
        })
        .unwrap();
        assert_eq!(admitted.proof(), &admitted.digest);
        let (lease, proof) = admitted.into_parts();
        assert_eq!(proof, lease.sha256());
    }

    /// Overwrite a sealed object the way a bug or a stray tool would have to:
    /// by defeating the seal first. Tests that simulate corruption must not
    /// silently depend on objects being writable.
    #[cfg_attr(not(unix), allow(clippy::permissions_set_readonly_false))]
    fn force_overwrite_object(path: &Path, bytes: &[u8]) {
        let mut permissions = fs::metadata(path).unwrap().permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            permissions.set_mode(0o644);
        }
        #[cfg(not(unix))]
        permissions.set_readonly(false);
        fs::set_permissions(path, permissions).unwrap();
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn corrupt_existing_digest_object_is_never_replaced() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.oasr");
        fs::write(&source, b"GGUF-original-pack").unwrap();
        let root = temp.path().join("models");
        let first = admit_file(&source, &root, |_| Ok(())).unwrap();
        let object_path = first.object_path.clone();
        drop(first);
        force_overwrite_object(&object_path, b"corrupt");

        let error = admit_file(&source, &root, |_| Ok(())).unwrap_err();
        assert!(matches!(error, ContentStoreError::SourceChanged { .. }));
        assert_eq!(fs::read(object_path).unwrap(), b"corrupt");
    }

    #[test]
    fn admitted_object_is_sealed_read_only() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.oasr");
        fs::write(&source, b"GGUF-sealed-pack").unwrap();
        let root = temp.path().join("models");

        let admitted = admit_file(&source, &root, |_| Ok(())).unwrap();
        let permissions = fs::metadata(&admitted.object_path).unwrap().permissions();
        assert!(permissions.readonly(), "landed object must be read-only");
        assert!(
            fs::OpenOptions::new()
                .write(true)
                .open(&admitted.object_path)
                .is_err(),
            "a sealed object must reject being reopened for writing"
        );
        // Reading the pack -- the only thing production does with an object --
        // keeps working through the seal.
        assert_eq!(
            fs::read(&admitted.object_path).unwrap(),
            b"GGUF-sealed-pack"
        );
    }

    #[test]
    fn sealed_object_is_still_collectable() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.oasr");
        fs::write(&source, b"GGUF-collectable-pack").unwrap();
        let root = temp.path().join("models");
        let admitted = admit_file(&source, &root, |_| Ok(())).unwrap();
        let digest = admitted.digest.clone();
        let object_path = admitted.object_path.clone();
        drop(admitted);

        let freed = remove_object(&root, &digest).unwrap();
        assert_eq!(freed, b"GGUF-collectable-pack".len() as u64);
        assert!(!object_path.exists());
        // The per-digest directory goes with it.
        assert!(!object_path.parent().unwrap().exists());
    }

    #[test]
    fn admit_staging_names_carry_their_owning_pid() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.oasr");
        fs::write(&source, b"GGUF-staged-pack").unwrap();
        let root = temp.path().join("models");

        // Observed mid-admission: this is the only moment the staging entry and
        // the (still absent) object coexist.
        let observed = std::cell::RefCell::new(Vec::new());
        let objects_during = std::cell::Cell::new(usize::MAX);
        admit_file(&source, &root, |_| {
            for entry in fs::read_dir(root.join(STAGING_DIR_NAME)).unwrap() {
                observed.borrow_mut().push(entry.unwrap().path());
            }
            objects_during.set(
                fs::read_dir(objects_root(&root))
                    .map(Iterator::count)
                    .unwrap_or(0),
            );
            Ok(())
        })
        .unwrap();

        let staged = observed.into_inner();
        assert_eq!(staged.len(), 1, "one admission stages exactly one file");
        let name = staged[0].file_name().unwrap().to_str().unwrap();
        assert_eq!(
            admit_staging_owner_pid(name),
            Some(std::process::id()),
            "staging name must name the process that can finish it: {name}"
        );
        // In-flight bytes never sit inside the content namespace, so collection
        // can treat every name under `objects/sha256/` as a digest.
        assert!(!staged[0].starts_with(objects_root(&root)));
        assert_eq!(
            objects_during.get(),
            0,
            "no object may exist while its content is still staging"
        );
    }

    #[test]
    fn stored_objects_lists_only_valid_digest_directories() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.oasr");
        fs::write(&source, b"GGUF-listed-pack").unwrap();
        let root = temp.path().join("models");
        let admitted = admit_file(&source, &root, |_| Ok(())).unwrap();

        // Junk that a GC must positively refuse to identify as an object.
        fs::create_dir_all(objects_root(&root).join("not-a-digest")).unwrap();
        fs::create_dir_all(objects_root(&root).join("a".repeat(64))).unwrap();

        let objects = stored_objects(&root).unwrap();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].digest, admitted.digest);
        assert_eq!(objects[0].size_bytes, b"GGUF-listed-pack".len() as u64);
    }

    /// The load path must not re-derive what admission already established.
    ///
    /// Asserted by corrupting a sealed object's bytes *without* changing its
    /// length: a loader that re-hashes would reject this, and one that only
    /// checks structure and length accepts it. Pinning the accepting behaviour
    /// is what keeps a gigabyte-scale re-read from creeping back into every
    /// model switch.
    #[test]
    fn declared_lease_does_not_rehash_the_object() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.oasr");
        fs::write(&source, b"GGUF-load-path-pack").unwrap();
        let root = temp.path().join("models");
        let admitted = admit_file(&source, &root, |_| Ok(())).unwrap();
        let digest = admitted.digest.clone();
        let size = admitted.size_bytes;
        let object = admitted.object_path.clone();
        drop(admitted);

        force_overwrite_object(&object, b"GGUF-load-path-XXXX");
        assert_eq!(fs::metadata(&object).unwrap().len(), size);

        let lease = open_declared_lease(&root, &digest, size).unwrap();
        assert_eq!(lease.bytes(), b"GGUF-load-path-XXXX");

        // The verifying variant is the one that still catches it, and
        // `verify_model_store` is built on exactly that.
        assert!(matches!(
            open_verified_lease(&root, &digest).unwrap_err(),
            ContentStoreError::SourceChanged { .. }
        ));
    }

    #[test]
    fn declared_lease_rejects_an_object_of_the_wrong_length() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.oasr");
        fs::write(&source, b"GGUF-length-checked-pack").unwrap();
        let root = temp.path().join("models");
        let admitted = admit_file(&source, &root, |_| Ok(())).unwrap();
        let digest = admitted.digest.clone();
        let size = admitted.size_bytes;
        let object = admitted.object_path.clone();
        drop(admitted);

        // Truncation is the failure a cheap check must still catch: a partially
        // written or clipped object would otherwise be mapped and parsed.
        force_overwrite_object(&object, b"short");
        assert!(matches!(
            open_declared_lease(&root, &digest, size).unwrap_err(),
            ContentStoreError::SourceChanged { .. }
        ));
    }

    #[test]
    fn both_lease_variants_reject_a_missing_object() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("models");
        let digest = sha256_bytes(b"never-admitted");
        assert!(open_declared_lease(&root, &digest, 14).is_err());
        assert!(open_verified_lease(&root, &digest).is_err());
    }

    #[test]
    fn content_addressed_object_paths_are_recognised_without_an_extension() {
        let digest = "a".repeat(64);
        let object = Path::new("/home/u/.openasr/models")
            .join("objects")
            .join("sha256")
            .join(&digest)
            .join("content");
        assert!(is_content_addressed_object_path(&object));
    }

    #[test]
    fn only_the_exact_object_layout_is_recognised() {
        let digest = "a".repeat(64);
        let objects = Path::new("/m").join("objects").join("sha256");
        // Right layout, wrong file role.
        assert!(!is_content_addressed_object_path(
            &objects.join(&digest).join("weights.bin")
        ));
        // Right file role, digest directory is not a digest.
        assert!(!is_content_addressed_object_path(
            &objects.join("NOTADIGEST").join("content")
        ));
        // Right file role and digest, wrong algorithm directory.
        assert!(!is_content_addressed_object_path(
            &Path::new("/m")
                .join("objects")
                .join("md5")
                .join(&digest)
                .join("content")
        ));
        // Right tail, not under an objects root.
        assert!(!is_content_addressed_object_path(
            &Path::new("/m")
                .join("packs")
                .join("sha256")
                .join(&digest)
                .join("content")
        ));
        // A bare file named content.
        assert!(!is_content_addressed_object_path(Path::new("/m/content")));
    }

    #[test]
    fn object_digest_from_path_extracts_the_named_digest() {
        let digest = "ab".repeat(32);
        let object = Path::new("/any/prefix/models")
            .join("objects")
            .join("sha256")
            .join(&digest)
            .join("content");
        assert_eq!(object_digest_from_path(&object), Some(digest.as_str()));
        // The layout predicate is exactly "extraction succeeded".
        assert!(is_content_addressed_object_path(&object));
        assert_eq!(object_digest_from_path(Path::new("/m/content")), None);
    }

    #[test]
    fn trusted_object_digest_requires_layout_seal_and_root_anchor() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.oasr");
        fs::write(&source, b"GGUF-trusted-gate-pack").unwrap();
        let root = temp.path().join("models");
        let admitted = admit_file(&source, &root, |_| Ok(())).unwrap();

        // Sealed, correctly laid out, and under the caller's own root: the
        // path digest alone is the trust.
        assert!(
            fs::metadata(&admitted.object_path)
                .unwrap()
                .permissions()
                .readonly()
        );
        assert_eq!(
            trusted_object_digest(&admitted.object_path, true, &root),
            Some(admitted.digest.as_str())
        );

        // Defeating the seal must fail closed -- back to the hashing path
        // (None here) -- never keep handing out the trusted digest.
        let object_path = admitted.object_path.clone();
        drop(admitted);
        force_overwrite_object(&object_path, b"GGUF-trusted-gate-XXXX");
        assert!(!fs::metadata(&object_path).unwrap().permissions().readonly());
        assert_eq!(trusted_object_digest(&object_path, false, &root), None);

        // Layout and seal alone are not enough either: a caller anchored to a
        // *different* root must not trust this object, even sealed.
        let unrelated_root = temp.path().join("other-models");
        assert_eq!(
            trusted_object_digest(&object_path, true, &unrelated_root),
            None,
            "an object real and sealed under one root must not be trusted by a caller \
             anchored to a different root"
        );

        // The plain non-object source path is rejected regardless of root.
        assert_eq!(trusted_object_digest(&source, true, &root), None);
    }

    /// The adversarial case this gate exists for: a file placed *outside* any
    /// models root that merely has the object layout's shape --
    /// `<attacker-controlled dir>/objects/sha256/<64 hex>/content`, read-only
    /// -- must never be trusted, no matter how convincing the shape and seal
    /// look. Before the `models_root` anchor this returned the digest named
    /// by the path for arbitrary bytes; the fix is exactly the anchor check.
    #[test]
    fn trusted_object_digest_rejects_a_same_shaped_path_outside_the_models_root() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("models");

        // A real, honestly-admitted object under the caller's own root, for
        // contrast: this one must still be trusted.
        let source = temp.path().join("source.oasr");
        fs::write(&source, b"GGUF-real-object-under-root").unwrap();
        let admitted = admit_file(&source, &root, |_| Ok(())).unwrap();
        assert_eq!(
            trusted_object_digest(&admitted.object_path, true, &root),
            Some(admitted.digest.as_str())
        );

        // An attacker-controlled tree entirely outside `root`, shaped exactly
        // like a content-addressed object, sealed read-only, naming a digest
        // that has nothing to do with its actual bytes.
        let attacker_digest = "99".repeat(32);
        let attacker_object = temp
            .path()
            .join("totally-unrelated")
            .join("objects")
            .join("sha256")
            .join(&attacker_digest)
            .join("content");
        fs::create_dir_all(attacker_object.parent().unwrap()).unwrap();
        fs::write(&attacker_object, b"GGUFattacker-controlled-bytes").unwrap();
        let mut permissions = fs::metadata(&attacker_object).unwrap().permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            permissions.set_mode(0o444);
        }
        #[cfg(not(unix))]
        permissions.set_readonly(true);
        fs::set_permissions(&attacker_object, permissions).unwrap();
        assert!(
            fs::metadata(&attacker_object)
                .unwrap()
                .permissions()
                .readonly(),
            "the probe file must actually be sealed for this to test anything"
        );

        // Structurally, this path is indistinguishable from a real object.
        assert_eq!(
            object_digest_from_path(&attacker_object),
            Some(attacker_digest.as_str())
        );
        // But it is not under `root`, so it must never be trusted -- the
        // caller must fall back to a full hash instead of handing out
        // `attacker_digest` for bytes that do not match it.
        assert_eq!(
            trusted_object_digest(&attacker_object, true, &root),
            None,
            "a same-shaped sealed path outside the models root must never be trusted"
        );
    }
}
