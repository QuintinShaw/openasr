use std::path::{Path, PathBuf};

use crate::ggml_runtime::{
    GgmlRuntimeSource, has_openasr_runtime_pack_extension, validate_ggml_runtime_source_path,
};

use super::BackendError;

pub(super) fn validate_local_native_model_pack_path(path: &Path) -> Result<PathBuf, BackendError> {
    validate_local_native_runtime_source(path).map(|source| source.path().to_path_buf())
}

pub(super) fn validate_local_native_runtime_source(
    path: &Path,
) -> Result<GgmlRuntimeSource, BackendError> {
    if !has_openasr_runtime_pack_extension(path) {
        return Err(BackendError::NativeModelPackPathRejected {
            reason: format!(
                "'{}' is not an OpenASR runtime package path; expected the .oasr extension",
                path.display()
            ),
        });
    }
    validate_ggml_runtime_source_path(path).map_err(|error| {
        BackendError::NativeModelPackPathRejected {
            reason: format!(
                "{error}. Expected a local GGUF-backed OpenASR runtime package (.oasr). \
                 Directories and reserved non-GGUF OASR containers are not accepted on this path."
            ),
        }
    })
}
