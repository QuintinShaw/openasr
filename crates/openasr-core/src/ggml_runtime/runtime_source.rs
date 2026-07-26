use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use memmap2::Mmap;
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::{GgmlPackageFormat, GgmlPackageProbe, GgmlPackageProbeError, probe_ggml_package_path};

/// A validated ggml runtime source: the file has been opened and mapped
/// exactly once. Its content id (full-file sha256) is derived from that same
/// mapping, lazily, the first time a caller actually asks for one.
///
/// This is the fix for a reopen TOCTOU that used to exist between building a
/// [`super::GgufTensorIndex`] (path-based) and mapping tensor *data*
/// (previously a fresh `File::open` of the same path in
/// `GgufTensorDataReader::from_tensor_index_and_alignment`): metadata, the
/// tensor index, and the mapped weight bytes could come from different file
/// generations if the pack was replaced between the two opens. Holding the
/// open mapping here and threading it through
/// [`super::GgufTensorDataReader::from_runtime_source`] means the bytes a
/// caller hashes for identity are the exact same bytes later read for
/// weights -- there is no second open to race against.
///
/// `content_id` is deliberately **not** computed at validation time: this
/// constructor sits on the per-request admission path (see
/// `validate_local_native_runtime_source`), and hashing a multi-GB pack on
/// every request is the exact per-request full-file-sha256 cost the runtime
/// cache coordinator's warm path is designed to avoid. Only a caller that
/// actually calls [`GgmlRuntimeSource::content_id`] pays the one-time hash;
/// callers that only need [`GgmlRuntimeSource::path`] / [`GgmlRuntimeSource::package_probe`]
/// (the common case) never do.
///
/// `path()` is downgraded to an admission / diagnostics / GC / fixture-lookup
/// helper: it must never be re-derived into a content identity by a caller
/// that already holds a `GgmlRuntimeSource` (use [`GgmlRuntimeSource::content_id`]
/// instead).
pub struct GgmlRuntimeSource {
    path: PathBuf,
    package_probe: GgmlPackageProbe,
    mmap: Arc<Mmap>,
    /// `sha256:<hex>` of the full mapped file. Computed once, lazily, from
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
            mmap: Arc::clone(&self.mmap),
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

    pub fn package_probe(&self) -> &GgmlPackageProbe {
        &self.package_probe
    }

    /// `sha256:<hex>` content id of the full mapped file. Computed on first
    /// call from the mapping this source already holds open (never a fresh
    /// `File::open` of `path`), then cached on this instance. This is the
    /// identity authority for this source -- prefer it over re-deriving an id
    /// from [`GgmlRuntimeSource::path`].
    pub fn content_id(&self) -> &str {
        self.content_id
            .get_or_init(|| format!("sha256:{:x}", Sha256::digest(&self.mmap[..])))
    }

    /// The open mapping backing this source. Sharing this `Arc` (rather than
    /// re-opening `path()`) is what lets metadata / tensor-index / weight
    /// readers agree on exactly the bytes this source's `content_id` was
    /// computed from.
    pub(crate) fn backing_mmap(&self) -> Arc<Mmap> {
        Arc::clone(&self.mmap)
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
    let mmap =
        unsafe { Mmap::map(&file) }.map_err(|source| GgmlRuntimeSourcePathError::MapFile {
            path: path.to_path_buf(),
            source,
        })?;

    Ok(GgmlRuntimeSource {
        path: path.to_path_buf(),
        package_probe,
        mmap: Arc::new(mmap),
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
    use std::{fs, path::Path};

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
}
