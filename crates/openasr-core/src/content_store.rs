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
pub(crate) struct AdmittedContent {
    pub digest: String,
    pub size_bytes: u64,
    pub object_path: PathBuf,
    lease: ContentLease,
}

impl AdmittedContent {
    /// Consume the validation lease for the object just admitted. It remains
    /// valid even after the private staging name has been unlinked.
    pub(crate) fn into_lease(self) -> ContentLease {
        self.lease
    }
}

pub(crate) fn objects_root(models_root: &Path) -> PathBuf {
    models_root.join("objects").join("sha256")
}

/// Layout of one immutable object: `<models>/objects/sha256/<digest>/content`.
/// The per-digest directory is part of the contract -- `InstalledModelStore`
/// (and the CLI/server stores already on disk) resolve refs against exactly this
/// path, so the bytes must never be written directly at `<digest>`.
pub(crate) fn object_path(models_root: &Path, digest: &str) -> Result<PathBuf, ContentStoreError> {
    validate_digest(digest)?;
    Ok(objects_root(models_root).join(digest).join("content"))
}

/// Open, map, and hash exactly one immutable object descriptor.
pub(crate) fn open_lease(
    models_root: &Path,
    digest: &str,
) -> Result<ContentLease, ContentStoreError> {
    let path = object_path(models_root, digest)?;
    let file = File::open(&path).map_err(|source| ContentStoreError::Io {
        path: path.clone(),
        source,
    })?;
    let lease = ContentLease::from_file(digest.to_string(), path.clone(), file)?;
    if lease.sha256() != digest {
        return Err(ContentStoreError::SourceChanged { path });
    }
    Ok(lease)
}

/// Copy one opened source descriptor into a private staging file, fsync it,
/// validate its mapped bytes, then create an immutable object. `preflight` is
/// deliberately lease-based: it cannot accidentally reopen a mutable pathname.
pub(crate) fn admit_file(
    source_path: &Path,
    models_root: &Path,
    preflight: impl Fn(&ContentLease) -> Result<(), crate::PullError>,
) -> Result<AdmittedContent, ContentStoreError> {
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

    let staging_dir = models_root.join("staging");
    fs::create_dir_all(&staging_dir).map_err(|source| ContentStoreError::Io {
        path: staging_dir.clone(),
        source,
    })?;
    let staging = staging_dir.join(format!("{}.partial", unique_suffix()));
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
        if lease.bytes().len() as u64 != size || lease.sha256() != digest {
            return Err(ContentStoreError::SourceChanged {
                path: source_path.to_path_buf(),
            });
        }
        preflight(&lease).map_err(|error| ContentStoreError::Preflight(Box::new(error)))?;

        let object = object_path(models_root, &digest)?;
        let parent = object.parent().expect("object path has parent");
        fs::create_dir_all(parent).map_err(|source| ContentStoreError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
        let lease = match fs::hard_link(&staging, &object) {
            Ok(()) => {
                atomic_file::sync_parent_dir_best_effort(&object);
                fs::remove_file(&staging).map_err(|source| ContentStoreError::Io {
                    path: staging.clone(),
                    source,
                })?;
                lease.with_path(object.clone())
            }
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
                // A digest name is not evidence. Revalidate the existing object's
                // descriptor and contract before reusing it; never replace it.
                let existing = open_lease(models_root, &digest)?;
                if existing.bytes().len() as u64 != size || existing.sha256() != digest {
                    return Err(ContentStoreError::ObjectCollision { path: object });
                }
                preflight(&existing)
                    .map_err(|error| ContentStoreError::Preflight(Box::new(error)))?;
                fs::remove_file(&staging).map_err(|source| ContentStoreError::Io {
                    path: staging.clone(),
                    source,
                })?;
                existing
            }
            Err(_) => match fs::rename(&staging, &object) {
                Ok(()) => {
                    atomic_file::sync_parent_dir_best_effort(&object);
                    lease.with_path(object.clone())
                }
                Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
                    let existing = open_lease(models_root, &digest)?;
                    if existing.bytes().len() as u64 != size || existing.sha256() != digest {
                        return Err(ContentStoreError::ObjectCollision { path: object });
                    }
                    preflight(&existing)
                        .map_err(|error| ContentStoreError::Preflight(Box::new(error)))?;
                    fs::remove_file(&staging).map_err(|source| ContentStoreError::Io {
                        path: staging.clone(),
                        source,
                    })?;
                    existing
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
        let path = object_path(models_root, digest)?;
        match fs::remove_file(&path) {
            Ok(()) => {
                atomic_file::sync_parent_dir_best_effort(&path);
                // The per-digest directory only ever holds `content`; drop it
                // once empty so a collected object leaves nothing behind.
                if let Some(parent) = path.parent() {
                    let _ = fs::remove_dir(parent);
                }
            }
            Err(source) if source.kind() == io::ErrorKind::NotFound => {}
            Err(source) => return Err(ContentStoreError::Io { path, source }),
        }
    }
    Ok(())
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
    fn corrupt_existing_digest_object_is_never_replaced() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.oasr");
        fs::write(&source, b"GGUF-original-pack").unwrap();
        let root = temp.path().join("models");
        let first = admit_file(&source, &root, |_| Ok(())).unwrap();
        fs::write(&first.object_path, b"corrupt").unwrap();

        let error = admit_file(&source, &root, |_| Ok(())).unwrap_err();
        assert!(matches!(error, ContentStoreError::SourceChanged { .. }));
        assert_eq!(fs::read(first.object_path).unwrap(), b"corrupt");
    }
}
