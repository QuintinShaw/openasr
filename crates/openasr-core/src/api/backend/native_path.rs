use std::path::{Path, PathBuf};

use crate::ggml_runtime::{GgmlRuntimeSource, validate_ggml_runtime_source_path};

use super::BackendError;

pub(super) fn validate_local_native_model_pack_path(path: &Path) -> Result<PathBuf, BackendError> {
    validate_local_native_runtime_source(path).map(|source| source.path().to_path_buf())
}

pub(super) fn validate_local_native_runtime_source(
    path: &Path,
) -> Result<GgmlRuntimeSource, BackendError> {
    validate_ggml_runtime_source_path(path).map_err(|error| {
        BackendError::NativeModelPackPathRejected {
            reason: format!(
                "{error}. Expected a local GGUF-backed runtime file (.gguf or .oasr). \
                 Directories and reserved non-GGUF OASR containers are not accepted on this path."
            ),
        }
    })
}
