#[cfg(test)]
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;

#[cfg(test)]
use super::validate_ggml_runtime_source_path;
use super::{
    GgmlRuntimeSource, GgmlRuntimeSourcePathError, GgufCParserSandboxError, GgufMetadata,
    GgufTensorDataReadError, GgufTensorDataReader, GgufTensorIndex,
    load_gguf_metadata_and_tensor_index_with_c_parser_sandbox,
};

/// One immutable GGUF generation parsed exactly once before admission.
///
/// The open mapping, metadata and tensor index are one provenance unit. Quote,
/// contract validation and materialization must consume this value instead of
/// reopening the path or reparsing the header inside an allocation
/// transaction.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufRuntimeSourcePreflight {
    pub(crate) runtime_source: GgmlRuntimeSource,
    pub(crate) metadata: Arc<GgufMetadata>,
    pub(crate) tensor_index: Arc<GgufTensorIndex>,
}

impl GgufRuntimeSourcePreflight {
    /// Bounded-parses the exact mapping already held by `runtime_source`.
    pub fn from_runtime_source(
        runtime_source: &GgmlRuntimeSource,
    ) -> Result<Self, RuntimeSourceMetadataAndTensorIndexPreflightError> {
        load_runtime_source_metadata_and_tensor_index_from_source(runtime_source)
    }

    pub fn runtime_source(&self) -> &GgmlRuntimeSource {
        &self.runtime_source
    }

    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    pub fn tensor_index(&self) -> &GgufTensorIndex {
        &self.tensor_index
    }

    /// Rebinds diagnostics to a new hard-link/installed name without
    /// reopening or reparsing the exact mapping represented by this proof.
    pub(crate) fn with_display_path(mut self, path: PathBuf) -> Self {
        self.runtime_source = self.runtime_source.with_display_path(path.clone());
        Arc::make_mut(&mut self.tensor_index).set_display_path(path);
        self
    }

    /// Copy this exact admitted generation into anonymous immutable storage
    /// while retaining the already-validated header views. Tensor offsets are
    /// safe to reuse because the snapshot must hash to the same content id and
    /// keeps the same logical path.
    pub(crate) fn immutable_snapshot_matching_content_id(
        &self,
        expected_content_id: &str,
    ) -> Result<Self, GgmlRuntimeSourcePathError> {
        let runtime_source = self
            .runtime_source
            .immutable_snapshot_matching_content_id(expected_content_id)?;
        debug_assert_eq!(self.tensor_index.path(), runtime_source.path());
        Ok(Self {
            runtime_source,
            metadata: Arc::clone(&self.metadata),
            tensor_index: Arc::clone(&self.tensor_index),
        })
    }
}

#[derive(Debug, Error)]
pub enum RuntimeSourceMetadataAndTensorIndexPreflightError {
    #[error("runtime source path is invalid: {source}")]
    RuntimeSourcePath {
        source: Box<GgmlRuntimeSourcePathError>,
    },
    #[error("sandboxed C-side GGUF parse failed for '{runtime_source_path}': {source}")]
    SandboxedCParser {
        runtime_source_path: PathBuf,
        source: Box<GgufCParserSandboxError>,
    },
}

#[derive(Debug, Error)]
pub(crate) enum RuntimeSourceTensorReaderError {
    #[error("could not create GGUF tensor reader from '{runtime_source_path}': {source}")]
    Build {
        runtime_source_path: PathBuf,
        #[source]
        source: Box<GgufTensorDataReadError>,
    },
}

#[cfg(test)]
pub(crate) fn load_runtime_source_metadata_and_tensor_index(
    runtime_source_path: &Path,
) -> Result<GgufRuntimeSourcePreflight, RuntimeSourceMetadataAndTensorIndexPreflightError> {
    let runtime_source =
        validate_ggml_runtime_source_path(runtime_source_path).map_err(|source| {
            RuntimeSourceMetadataAndTensorIndexPreflightError::RuntimeSourcePath {
                source: Box::new(source),
            }
        })?;
    load_runtime_source_metadata_and_tensor_index_from_source(&runtime_source)
}

pub(crate) fn load_runtime_source_metadata_and_tensor_index_from_source(
    runtime_source: &GgmlRuntimeSource,
) -> Result<GgufRuntimeSourcePreflight, RuntimeSourceMetadataAndTensorIndexPreflightError> {
    let (metadata, tensor_index) =
        load_gguf_metadata_and_tensor_index_with_c_parser_sandbox(runtime_source).map_err(
            |source| RuntimeSourceMetadataAndTensorIndexPreflightError::SandboxedCParser {
                runtime_source_path: runtime_source.path().to_path_buf(),
                source: Box::new(source),
            },
        )?;
    Ok(GgufRuntimeSourcePreflight {
        runtime_source: runtime_source.clone(),
        metadata: Arc::new(metadata),
        tensor_index: Arc::new(tensor_index),
    })
}

pub(crate) fn build_runtime_tensor_reader_from_preflight(
    preflight: &GgufRuntimeSourcePreflight,
) -> Result<GgufTensorDataReader, RuntimeSourceTensorReaderError> {
    GgufTensorDataReader::from_preflight_parts(
        &preflight.runtime_source,
        &preflight.metadata,
        Arc::clone(&preflight.tensor_index),
    )
    .map_err(|source| RuntimeSourceTensorReaderError::Build {
        runtime_source_path: preflight.runtime_source.path().to_path_buf(),
        source: Box::new(source),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;

    use super::*;
    use crate::ggml_runtime::write_gguf_file_v0;

    #[test]
    fn preflight_rejects_missing_runtime_source_path() {
        let temp = tempfile::NamedTempFile::new().expect("temp file");
        let missing_path = temp.path().to_path_buf();
        drop(temp);

        let error = load_runtime_source_metadata_and_tensor_index(&missing_path)
            .expect_err("missing path must fail preflight");
        assert!(matches!(
            error,
            RuntimeSourceMetadataAndTensorIndexPreflightError::RuntimeSourcePath { .. }
        ));
    }

    #[test]
    fn preflight_surfaces_metadata_read_errors_with_runtime_path() {
        let temp = tempfile::NamedTempFile::new().expect("temp file");
        let runtime_path = temp.path().with_extension("gguf");
        fs::write(&runtime_path, b"GGUFpayload").expect("write gguf magic fixture");

        let error = load_runtime_source_metadata_and_tensor_index(&runtime_path)
            .expect_err("invalid gguf payload should fail metadata read");
        match error {
            RuntimeSourceMetadataAndTensorIndexPreflightError::SandboxedCParser {
                runtime_source_path,
                ..
            } => assert_eq!(runtime_source_path, runtime_path),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn immutable_snapshot_reuses_the_admitted_header_views() {
        let temp = tempfile::tempdir().expect("temp dir");
        let runtime_path = temp.path().join("minimal.gguf");
        write_gguf_file_v0(&runtime_path, &BTreeMap::new(), &[]).expect("write minimal GGUF");

        let preflight = load_runtime_source_metadata_and_tensor_index(&runtime_path)
            .expect("preflight minimal GGUF");
        let content_id = preflight.runtime_source.content_id().to_string();
        let snapshot = preflight
            .immutable_snapshot_matching_content_id(&content_id)
            .expect("snapshot admitted generation");

        assert!(Arc::ptr_eq(&preflight.metadata, &snapshot.metadata));
        assert!(Arc::ptr_eq(&preflight.tensor_index, &snapshot.tensor_index));
        assert_eq!(snapshot.runtime_source.content_id(), content_id);
        build_runtime_tensor_reader_from_preflight(&snapshot)
            .expect("build tensor reader without reparsing");
    }

    #[test]
    fn preflight_extracts_metadata_and_tensor_index_from_one_bounded_parse() {
        let temp = tempfile::tempdir().expect("temp dir");
        let runtime_path = temp.path().join("single-pass.gguf");
        write_gguf_file_v0(&runtime_path, &BTreeMap::new(), &[]).expect("write minimal GGUF");
        let before =
            crate::ggml_runtime::gguf_metadata::bounded_parse_call_count_for_current_thread();

        let preflight = load_runtime_source_metadata_and_tensor_index(&runtime_path)
            .expect("preflight minimal GGUF");

        let after_preflight =
            crate::ggml_runtime::gguf_metadata::bounded_parse_call_count_for_current_thread();
        assert_eq!(
            after_preflight - before,
            1,
            "preflight must parse the GGUF header once"
        );

        for _ in 0..3 {
            build_runtime_tensor_reader_from_preflight(&preflight)
                .expect("preflight reader must reuse parsed header views");
        }
        let after_readers =
            crate::ggml_runtime::gguf_metadata::bounded_parse_call_count_for_current_thread();
        assert_eq!(
            after_readers, after_preflight,
            "building any number of readers from one preflight must not parse again"
        );
    }
}
