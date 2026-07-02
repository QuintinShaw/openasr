use std::{fs, path::Path};

use crate::audio::AudioInputError;

pub(crate) fn validate_regular_file(path: &Path) -> Result<(), AudioInputError> {
    let metadata = fs::metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            AudioInputError::NotFound {
                path: path.to_path_buf(),
            }
        } else {
            AudioInputError::Metadata {
                path: path.to_path_buf(),
            }
        }
    })?;

    if metadata.is_dir() {
        return Err(AudioInputError::Directory {
            path: path.to_path_buf(),
        });
    }
    if !metadata.is_file() {
        return Err(AudioInputError::NotRegularFile {
            path: path.to_path_buf(),
        });
    }

    Ok(())
}
