use std::path::{Path, PathBuf};

use crate::ggml_runtime::{
    GgmlRuntimeSource, has_openasr_runtime_pack_extension, validate_ggml_runtime_source_path,
};

use super::BackendError;

/// Whether a local path has one of the two supported product pack shapes.
///
/// User-supplied packs keep the `.oasr` suffix. Installed packs are immutable
/// content-store objects named `objects/sha256/<digest>/content`, so they have
/// no extension. This predicate is naming/layout policy only; callers must
/// still validate the container and run `PackVerifier` before execution.
pub(super) fn has_supported_native_runtime_pack_path_shape(path: &Path) -> bool {
    has_openasr_runtime_pack_extension(path) || crate::is_content_addressed_object_path(path)
}

pub(super) fn validate_local_native_model_pack_path(path: &Path) -> Result<PathBuf, BackendError> {
    validate_local_native_runtime_source(path).map(|source| source.path().to_path_buf())
}

pub(super) fn validate_local_native_runtime_source(
    path: &Path,
) -> Result<GgmlRuntimeSource, BackendError> {
    if !has_supported_native_runtime_pack_path_shape(path) {
        return Err(BackendError::NativeModelPackPathRejected {
            reason: format!(
                "'{}' is not an OpenASR runtime package path; expected a .oasr file or an installed content-addressed pack object",
                path.display()
            ),
        });
    }
    validate_ggml_runtime_source_path(path).map_err(|error| {
        BackendError::NativeModelPackPathRejected {
            reason: format!(
                "{error}. Expected a local GGUF-backed OpenASR runtime package (.oasr). \
                 Installed content-addressed pack objects are also accepted; directories and \
                 reserved non-GGUF OASR containers are not."
            ),
        }
    })
}
