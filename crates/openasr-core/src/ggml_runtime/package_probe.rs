use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use thiserror::Error;

use super::read_gguf_metadata;

const MAGIC_SIZE: usize = 4;
const GGUF_MAGIC: [u8; MAGIC_SIZE] = *b"GGUF";
const OASR_MAGIC: [u8; MAGIC_SIZE] = *b"OASR";

/// The sole user-facing filename extension for OpenASR runtime packs (`.oasr`).
///
/// The on-disk container is GGUF-structured internally (see [`GgmlPackageFormat`]),
/// but packs are *presented and supported* only as `.oasr`; the legacy `.gguf`
/// extension is no longer accepted at any public boundary. This constant + the
/// [`has_openasr_runtime_pack_extension`] predicate are the single source of truth
/// for that contract, shared by the CLI (run input + import output) and the core
/// converters' output validation.
pub const OPENASR_RUNTIME_PACK_EXTENSION: &str = "oasr";

/// True iff `path` carries the user-facing OpenASR runtime-pack extension
/// (`.oasr`, case-insensitive).
///
/// This is an *extension* check (the user-facing naming contract), deliberately
/// distinct from [`probe_ggml_package_path`], which validates the *container
/// magic* and stays permissive so internal GGUF fixtures remain readable. Public
/// producers/consumers gate on this; the low-level container reader does not.
pub fn has_openasr_runtime_pack_extension(path: impl AsRef<Path>) -> bool {
    path.as_ref()
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case(OPENASR_RUNTIME_PACK_EXTENSION))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlPackageFormat {
    GgufCompatible,
    UnsupportedOpenAsrContainerReserved,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlPackageExtensionHint {
    Oasr,
    Gguf,
    OtherOrMissing,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgmlPackageProbe {
    pub format: GgmlPackageFormat,
    pub extension_hint: GgmlPackageExtensionHint,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgmlPackageModelIdentityProbe {
    pub model_id: Option<String>,
    pub source_key: Option<String>,
    pub metadata_read_error: Option<String>,
}

#[derive(Debug, Error)]
pub enum GgmlPackageProbeError {
    #[error("could not read ggml package file '{path}': {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "ggml package file '{path}' is too short to contain a magic header: expected at least {expected} bytes, got {actual}"
    )]
    FileTooShort {
        path: PathBuf,
        expected: usize,
        actual: usize,
    },
    #[error("ggml package file '{path}' has unknown magic bytes: {magic:?}")]
    UnknownMagic {
        path: PathBuf,
        magic: [u8; MAGIC_SIZE],
    },
}

pub fn probe_ggml_package_path(
    path: impl AsRef<Path>,
) -> Result<GgmlPackageProbe, GgmlPackageProbeError> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|source| GgmlPackageProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    let mut magic = [0_u8; MAGIC_SIZE];
    let mut read_bytes = 0usize;
    while read_bytes < MAGIC_SIZE {
        let count =
            file.read(&mut magic[read_bytes..])
                .map_err(|source| GgmlPackageProbeError::Io {
                    path: path.to_path_buf(),
                    source,
                })?;
        if count == 0 {
            break;
        }
        read_bytes += count;
    }

    if read_bytes < MAGIC_SIZE {
        return Err(GgmlPackageProbeError::FileTooShort {
            path: path.to_path_buf(),
            expected: MAGIC_SIZE,
            actual: read_bytes,
        });
    }

    let format = match magic {
        GGUF_MAGIC => GgmlPackageFormat::GgufCompatible,
        OASR_MAGIC => GgmlPackageFormat::UnsupportedOpenAsrContainerReserved,
        _ => {
            return Err(GgmlPackageProbeError::UnknownMagic {
                path: path.to_path_buf(),
                magic,
            });
        }
    };

    Ok(GgmlPackageProbe {
        format,
        extension_hint: GgmlPackageExtensionHint::from_path(path),
    })
}

pub fn probe_ggml_package_model_identity(path: impl AsRef<Path>) -> GgmlPackageModelIdentityProbe {
    let path = path.as_ref();
    match parse_model_identity_from_metadata(path) {
        Ok(Some((source_key, model_id))) => GgmlPackageModelIdentityProbe {
            model_id: Some(model_id),
            source_key: Some(source_key),
            metadata_read_error: None,
        },
        Ok(None) => GgmlPackageModelIdentityProbe {
            model_id: None,
            source_key: None,
            metadata_read_error: None,
        },
        Err(error) => GgmlPackageModelIdentityProbe {
            model_id: None,
            source_key: None,
            metadata_read_error: Some(error),
        },
    }
}

impl GgmlPackageExtensionHint {
    fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("oasr") => Self::Oasr,
            Some(ext) if ext.eq_ignore_ascii_case("gguf") => Self::Gguf,
            _ => Self::OtherOrMissing,
        }
    }
}

const GGUF_MODEL_ID_CANDIDATE_KEYS: [&str; 3] =
    ["openasr.model.id", "general.basename", "general.name"];
const RUNTIME_SOURCE_FILE_STEM_SOURCE_KEY: &str = "<runtime-source.file-stem>";

fn parse_model_identity_from_metadata(path: &Path) -> Result<Option<(String, String)>, String> {
    let metadata = read_gguf_metadata(path).map_err(|error| error.to_string())?;
    for key in GGUF_MODEL_ID_CANDIDATE_KEYS {
        if let Some(value) = metadata.get_string(key) {
            let normalized = value.trim();
            if !normalized.is_empty() {
                return Ok(Some((key.to_string(), normalized.to_string())));
            }
        }
    }
    if let Some(stem) = path.file_stem().and_then(|value| value.to_str()) {
        let normalized = stem.trim();
        if !normalized.is_empty() {
            return Ok(Some((
                RUNTIME_SOURCE_FILE_STEM_SOURCE_KEY.to_string(),
                normalized.to_string(),
            )));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use std::{ffi::CString, fs, path::Path};

    use tempfile::NamedTempFile;

    use super::{
        super::ffi, GgmlPackageExtensionHint, GgmlPackageFormat, GgmlPackageProbeError,
        probe_ggml_package_model_identity, probe_ggml_package_path,
    };

    fn write_magic_file(path: &Path, bytes: &[u8]) {
        fs::write(path, bytes).expect("write probe fixture");
    }

    #[test]
    fn openasr_pack_extension_predicate_accepts_only_oasr() {
        use super::has_openasr_runtime_pack_extension;
        assert!(has_openasr_runtime_pack_extension(Path::new(
            "/tmp/model.oasr"
        )));
        assert!(has_openasr_runtime_pack_extension(Path::new(
            "/tmp/model.OASR"
        )));
        assert!(!has_openasr_runtime_pack_extension(Path::new(
            "/tmp/model.gguf"
        )));
        assert!(!has_openasr_runtime_pack_extension(Path::new("/tmp/model")));
        assert!(!has_openasr_runtime_pack_extension(Path::new(
            "/tmp/model.oasr.bak"
        )));
    }

    #[test]
    fn probes_gguf_magic_with_gguf_hint() {
        let file = NamedTempFile::new().expect("temp file");
        let probe_path = file.path().with_extension("gguf");
        write_magic_file(&probe_path, b"GGUFpayload");

        let probe = probe_ggml_package_path(&probe_path).expect("probe gguf");
        assert_eq!(probe.format, GgmlPackageFormat::GgufCompatible);
        assert_eq!(probe.extension_hint, GgmlPackageExtensionHint::Gguf);
    }

    #[test]
    fn probes_gguf_magic_with_oasr_hint() {
        let file = NamedTempFile::new().expect("temp file");
        let probe_path = file.path().with_extension("oasr");
        write_magic_file(&probe_path, b"GGUFpayload");

        let probe = probe_ggml_package_path(&probe_path).expect("probe gguf");
        assert_eq!(probe.format, GgmlPackageFormat::GgufCompatible);
        assert_eq!(probe.extension_hint, GgmlPackageExtensionHint::Oasr);
    }

    #[test]
    fn probes_reserved_oasr_magic() {
        let file = NamedTempFile::new().expect("temp file");
        write_magic_file(file.path(), b"OASRpayload");

        let probe = probe_ggml_package_path(file.path()).expect("probe reserved");
        assert_eq!(
            probe.format,
            GgmlPackageFormat::UnsupportedOpenAsrContainerReserved
        );
        assert_eq!(
            probe.extension_hint,
            GgmlPackageExtensionHint::OtherOrMissing
        );
    }

    #[test]
    fn rejects_unknown_magic() {
        let file = NamedTempFile::new().expect("temp file");
        write_magic_file(file.path(), b"ABCDpayload");

        let error = probe_ggml_package_path(file.path()).expect_err("unknown magic must fail");
        match error {
            GgmlPackageProbeError::UnknownMagic { magic, .. } => {
                assert_eq!(magic, *b"ABCD");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn rejects_short_file() {
        let file = NamedTempFile::new().expect("temp file");
        write_magic_file(file.path(), b"GG");

        let error = probe_ggml_package_path(file.path()).expect_err("short file must fail");
        match error {
            GgmlPackageProbeError::FileTooShort {
                expected, actual, ..
            } => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 2);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn returns_io_error_for_missing_file() {
        let file = NamedTempFile::new().expect("temp file");
        let missing_path = file.path().to_path_buf();
        drop(file);

        let error = probe_ggml_package_path(&missing_path).expect_err("missing file must fail");
        match error {
            GgmlPackageProbeError::Io { path, .. } => {
                assert_eq!(path, missing_path);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    fn write_gguf_string_metadata(path: &Path, entries: &[(&str, &str)]) {
        let path_c = CString::new(path.to_string_lossy().to_string()).expect("fixture path");
        let ctx = unsafe { ffi::gguf_init_empty() };
        assert!(!ctx.is_null(), "gguf_init_empty must succeed");

        for (key, value) in entries {
            let key_c = CString::new(*key).expect("key cstring");
            let value_c = CString::new(*value).expect("value cstring");
            unsafe { ffi::gguf_set_val_str(ctx, key_c.as_ptr(), value_c.as_ptr()) };
        }

        let ok = unsafe { ffi::gguf_write_to_file(ctx, path_c.as_ptr(), true) };
        unsafe { ffi::gguf_free(ctx) };
        assert!(ok, "gguf_write_to_file must succeed");
    }

    #[test]
    fn metadata_probe_reads_openasr_model_id() {
        let file = NamedTempFile::new().expect("temp file");
        write_gguf_string_metadata(
            file.path(),
            &[
                ("openasr.package.version", "1"),
                ("openasr.model.id", "whisper-small:gguf-q4"),
            ],
        );

        let probe = probe_ggml_package_model_identity(file.path());
        assert_eq!(probe.model_id.as_deref(), Some("whisper-small:gguf-q4"));
        assert_eq!(probe.source_key.as_deref(), Some("openasr.model.id"));
        assert_eq!(probe.metadata_read_error, None);
    }

    #[test]
    fn metadata_probe_falls_back_to_general_basename() {
        let file = NamedTempFile::new().expect("temp file");
        write_gguf_string_metadata(file.path(), &[("general.basename", "whisper-tiny")]);

        let probe = probe_ggml_package_model_identity(file.path());
        assert_eq!(probe.model_id.as_deref(), Some("whisper-tiny"));
        assert_eq!(probe.source_key.as_deref(), Some("general.basename"));
        assert_eq!(probe.metadata_read_error, None);
    }

    #[test]
    fn metadata_probe_reports_parse_error_for_truncated_metadata() {
        let file = NamedTempFile::new().expect("temp file");
        fs::write(file.path(), b"GGUF\x03\0\0\0").expect("write malformed gguf");

        let probe = probe_ggml_package_model_identity(file.path());
        assert_eq!(probe.model_id, None);
        assert_eq!(probe.source_key, None);
        assert!(probe.metadata_read_error.is_some());
    }

    #[test]
    fn metadata_probe_falls_back_to_general_name_after_basename() {
        let file = NamedTempFile::new().expect("temp file");
        write_gguf_string_metadata(
            file.path(),
            &[
                ("general.basename", "   "),
                ("general.name", "whisper-base"),
            ],
        );

        let probe = probe_ggml_package_model_identity(file.path());
        assert_eq!(probe.model_id.as_deref(), Some("whisper-base"));
        assert_eq!(probe.source_key.as_deref(), Some("general.name"));
        assert_eq!(probe.metadata_read_error, None);
    }

    #[test]
    fn metadata_probe_falls_back_to_runtime_source_file_stem() {
        let file = NamedTempFile::new().expect("temp file");
        let gguf = file.path().with_file_name("qwen3-asr-0.6b-q4_k.gguf");
        write_gguf_string_metadata(&gguf, &[("general.architecture", "qwen3-asr")]);

        let probe = probe_ggml_package_model_identity(&gguf);
        assert_eq!(probe.model_id.as_deref(), Some("qwen3-asr-0.6b-q4_k"));
        assert_eq!(
            probe.source_key.as_deref(),
            Some(super::RUNTIME_SOURCE_FILE_STEM_SOURCE_KEY)
        );
        assert_eq!(probe.metadata_read_error, None);
    }
}
