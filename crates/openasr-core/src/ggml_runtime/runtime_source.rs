use std::{
    fs,
    path::{Path, PathBuf},
};

use thiserror::Error;

use super::{GgmlPackageFormat, GgmlPackageProbe, GgmlPackageProbeError, probe_ggml_package_path};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgmlRuntimeSource {
    path: PathBuf,
    package_probe: GgmlPackageProbe,
}

impl GgmlRuntimeSource {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn package_probe(&self) -> &GgmlPackageProbe {
        &self.package_probe
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

    Ok(GgmlRuntimeSource {
        path: path.to_path_buf(),
        package_probe,
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
