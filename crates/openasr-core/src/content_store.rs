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
    #[error(transparent)]
    Preflight(#[from] Box<crate::PullError>),
}

/// A checkout of one immutable object. The owned descriptor and mapping refer to
/// the same published object; callers must retain this value for as long as they
/// consume bytes from the pack.
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
        let bytes = self.bytes();
        let Some(head) = bytes.get(..4) else {
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
}

pub(crate) struct AdmittedContent {
    pub digest: String,
    pub size_bytes: u64,
    pub object_path: PathBuf,
}

pub(crate) fn objects_root(models_root: &Path) -> PathBuf {
    models_root.join("objects").join("sha256")
}

pub(crate) fn object_path(models_root: &Path, digest: &str) -> Result<PathBuf, ContentStoreError> {
    validate_digest(digest)?;
    Ok(objects_root(models_root).join(digest))
}

pub(crate) fn open_lease(
    models_root: &Path,
    digest: &str,
) -> Result<ContentLease, ContentStoreError> {
    let path = object_path(models_root, digest)?;
    let file = File::open(&path).map_err(|source| ContentStoreError::Io {
        path: path.clone(),
        source,
    })?;
    let mmap = unsafe { Mmap::map(&file) }
        .map(Arc::new)
        .map_err(|source| ContentStoreError::Io {
            path: path.clone(),
            source,
        })?;
    let actual = sha256_bytes(&mmap);
    if actual != digest {
        return Err(ContentStoreError::SourceChanged { path });
    }
    Ok(ContentLease {
        digest: digest.to_string(),
        path,
        file,
        mmap,
    })
}

/// Copy from the opened admission descriptor into a private staging file, sync
/// it, then atomically publish by content digest. No caller ever treats the
/// mutable source pathname as runtime authority.
pub(crate) fn admit_file(
    source_path: &Path,
    models_root: &Path,
    preflight: impl FnOnce(&Path) -> Result<(), crate::PullError>,
) -> Result<AdmittedContent, ContentStoreError> {
    let source = File::open(source_path).map_err(|source| ContentStoreError::Io {
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
        let mut input = source;
        let mut output = OpenOptions::new()
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
            let count = input
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
            size += count as u64;
        }
        output.sync_all().map_err(|source| ContentStoreError::Io {
            path: staging.clone(),
            source,
        })?;
        let after = input.metadata().map_err(|source| ContentStoreError::Io {
            path: source_path.to_path_buf(),
            source,
        })?;
        if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
            return Err(ContentStoreError::SourceChanged {
                path: source_path.to_path_buf(),
            });
        }
        let digest = format!("{:x}", hash.finalize());
        let verified = hash_file(&staging)?;
        if verified.0 != size || verified.1 != digest {
            return Err(ContentStoreError::SourceChanged {
                path: source_path.to_path_buf(),
            });
        }
        preflight(&staging).map_err(|error| ContentStoreError::Preflight(Box::new(error)))?;
        let object = object_path(models_root, &digest)?;
        let parent = object.parent().expect("object path has parent");
        fs::create_dir_all(parent).map_err(|source| ContentStoreError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
        match fs::hard_link(&staging, &object) {
            Ok(()) => {
                fs::remove_file(&staging).map_err(|source| ContentStoreError::Io {
                    path: staging.clone(),
                    source,
                })?;
            }
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
                let existing = hash_file(&object)?;
                if existing.0 == size && existing.1 == digest {
                    fs::remove_file(&staging).map_err(|source| ContentStoreError::Io {
                        path: staging.clone(),
                        source,
                    })?;
                } else {
                    fs::remove_file(&object).map_err(|source| ContentStoreError::Io {
                        path: object.clone(),
                        source,
                    })?;
                    fs::rename(&staging, &object).map_err(|source| ContentStoreError::Io {
                        path: object.clone(),
                        source,
                    })?;
                    atomic_file::sync_parent_dir_best_effort(&object);
                }
            }
            Err(_) => match fs::rename(&staging, &object) {
                Ok(()) => atomic_file::sync_parent_dir_best_effort(&object),
                Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
                    fs::remove_file(&staging).map_err(|source| ContentStoreError::Io {
                        path: staging.clone(),
                        source,
                    })?;
                }
                Err(source) => {
                    return Err(ContentStoreError::Io {
                        path: object.clone(),
                        source,
                    });
                }
            },
        }
        Ok(AdmittedContent {
            digest,
            size_bytes: size,
            object_path: object,
        })
    })();
    if result.is_err() {
        let _ = fs::remove_file(&staging);
    }
    result
}

pub(crate) fn remove_object_if_unreferenced(models_root: &Path, digest: &str, referenced: bool) {
    if !referenced && let Ok(path) = object_path(models_root, digest) {
        let _ = fs::remove_file(path);
    }
}

fn hash_file(path: &Path) -> Result<(u64, String), ContentStoreError> {
    let mut file = File::open(path).map_err(|source| ContentStoreError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    file.seek(SeekFrom::Start(0))
        .map_err(|source| ContentStoreError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let mut hash = Sha256::new();
    let size = io::copy(&mut file, &mut HashWriter(&mut hash)).map_err(|source| {
        ContentStoreError::Io {
            path: path.to_path_buf(),
            source,
        }
    })?;
    Ok((size, format!("{:x}", hash.finalize())))
}

struct HashWriter<'a>(&'a mut Sha256);
impl Write for HashWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
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
