use std::path::{Path, PathBuf};
use std::sync::Arc;

use thiserror::Error;

use super::ggml_asr_executor::GgmlAsrRuntimeSourcePreflight;
use crate::ggml_runtime::load_gguf_metadata_and_tensor_index_with_c_parser_sandbox;
use crate::{
    GgmlRuntimeSource, GgmlRuntimeSourcePathError, GgufTensorDataReadError, GgufTensorDataReader,
    validate_ggml_runtime_source_path,
};

#[derive(Debug, Error)]
pub(crate) enum RuntimeSourceMetadataAndTensorIndexPreflightError {
    #[error("runtime source path is invalid: {source}")]
    RuntimeSourcePath {
        source: Box<GgmlRuntimeSourcePathError>,
    },
    #[error("sandboxed C-side GGUF parse failed for '{runtime_source_path}': {source}")]
    SandboxedCParser {
        runtime_source_path: PathBuf,
        source: Box<crate::GgufCParserSandboxError>,
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

pub(crate) fn load_runtime_source_metadata_and_tensor_index(
    runtime_source_path: &Path,
) -> Result<GgmlAsrRuntimeSourcePreflight, RuntimeSourceMetadataAndTensorIndexPreflightError> {
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
) -> Result<GgmlAsrRuntimeSourcePreflight, RuntimeSourceMetadataAndTensorIndexPreflightError> {
    let (metadata, tensor_index) =
        load_gguf_metadata_and_tensor_index_with_c_parser_sandbox(runtime_source).map_err(
            |source| RuntimeSourceMetadataAndTensorIndexPreflightError::SandboxedCParser {
                runtime_source_path: runtime_source.path().to_path_buf(),
                source: Box::new(source),
            },
        )?;
    Ok(GgmlAsrRuntimeSourcePreflight {
        runtime_source: runtime_source.clone(),
        metadata,
        tensor_index: Arc::new(tensor_index),
    })
}

pub(crate) fn build_runtime_tensor_reader_from_preflight(
    preflight: &GgmlAsrRuntimeSourcePreflight,
) -> Result<GgufTensorDataReader, RuntimeSourceTensorReaderError> {
    GgufTensorDataReader::from_tensor_index_shared(Arc::clone(&preflight.tensor_index)).map_err(
        |source| RuntimeSourceTensorReaderError::Build {
            runtime_source_path: preflight.runtime_source.path().to_path_buf(),
            source: Box::new(source),
        },
    )
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn preflight_rejects_missing_runtime_source_path() {
        let temp = tempfile::NamedTempFile::new().expect("temp file");
        let missing_path = temp.path().to_path_buf();
        drop(temp);

        let error = load_runtime_source_metadata_and_tensor_index(&missing_path)
            .expect_err("missing path must fail preflight");
        match error {
            RuntimeSourceMetadataAndTensorIndexPreflightError::RuntimeSourcePath { .. } => {}
            other => panic!("unexpected error: {other}"),
        }
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
}
