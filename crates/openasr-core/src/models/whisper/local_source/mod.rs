use std::{
    fmt,
    path::{Path, PathBuf},
};

use thiserror::Error;

mod safetensors;
pub(super) mod source_io;

pub use safetensors::{SafetensorsHeaderV0, SafetensorsTensorHeaderV0, load_safetensors_header_v0};

// Duplicate-JSON-key rejection is model-agnostic and shared with the
// `local_source_import` importer used by every other family; see
// `crate::models::safetensors_json` for the implementation and rationale.
pub(super) use crate::models::safetensors_json::reject_duplicate_json_keys;

#[derive(Debug, Error)]
pub enum WhisperLocalSourceError {
    #[error("could not read model source file '{path}': {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not write model source file '{path}': {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not parse model source artifact '{path}': {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("{0}")]
    Validate(String),
}

pub(super) fn validate_error(message: String) -> WhisperLocalSourceError {
    WhisperLocalSourceError::Validate(message)
}

pub(super) fn checked_u64_add_with_context(
    left: u64,
    right: u64,
    context: impl Into<String>,
) -> Result<u64, WhisperLocalSourceError> {
    left.checked_add(right)
        .ok_or_else(|| validate_error(context.into()))
}

pub(super) fn tensor_validation_error(
    name: &str,
    detail: impl fmt::Display,
) -> WhisperLocalSourceError {
    validate_error(format!("tensor '{name}' {detail}"))
}
