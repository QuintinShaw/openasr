use std::path::PathBuf;

use thiserror::Error;

use crate::models::local_source_import::LocalSourceImportError;

pub(super) mod source_io;

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

// The whisper package importer reads `.safetensors` sources through the shared
// hardened `crate::models::local_source_import::SafetensorsFile` parser; both
// error enums carry identical `#[error]` renderings for the shared variants,
// so the conversion is loss-free.
impl From<LocalSourceImportError> for WhisperLocalSourceError {
    fn from(error: LocalSourceImportError) -> Self {
        match error {
            LocalSourceImportError::Read { path, source } => Self::Read { path, source },
            LocalSourceImportError::Parse { path, source } => Self::Parse { path, source },
            LocalSourceImportError::Validate(message) => Self::Validate(message),
        }
    }
}
