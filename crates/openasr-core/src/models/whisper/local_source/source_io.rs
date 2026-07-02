use std::{fs, path::Path};

use serde::Deserialize;

use super::WhisperLocalSourceError;

pub(crate) fn read_source_json_file<T: for<'de> Deserialize<'de>>(
    root: &Path,
    relative_path: &str,
) -> Result<T, WhisperLocalSourceError> {
    let path = root.join(relative_path);
    let bytes = read_source_file_bytes(root, relative_path)?;
    serde_json::from_slice(&bytes).map_err(|source| WhisperLocalSourceError::Parse { path, source })
}

pub(crate) fn read_source_file_bytes(
    root: &Path,
    relative_path: &str,
) -> Result<Vec<u8>, WhisperLocalSourceError> {
    let path = root.join(relative_path);
    fs::read(&path).map_err(|source| WhisperLocalSourceError::Read { path, source })
}
