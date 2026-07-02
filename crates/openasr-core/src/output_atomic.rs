#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{fs, io::Write, path::Path};

use tempfile::NamedTempFile;

use super::OutputWriteError;

pub(super) fn validate_existing_output_writable(path: &Path) -> Result<(), OutputWriteError> {
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(OutputWriteError::ExistingOutputNotWritable {
                path: path.to_path_buf(),
                source,
            });
        }
    };

    if !metadata.is_file() {
        return Err(OutputWriteError::ExistingOutputNotWritable {
            path: path.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "existing output is not a regular file",
            ),
        });
    }
    #[cfg(unix)]
    if metadata.permissions().mode() & 0o222 == 0 {
        return Err(OutputWriteError::ExistingOutputNotWritable {
            path: path.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "existing output has no writable permission bits",
            ),
        });
    }

    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .map(|_| ())
        .map_err(|source| OutputWriteError::ExistingOutputNotWritable {
            path: path.to_path_buf(),
            source,
        })
}

pub(super) fn write_text_via_tempfile(
    path: &Path,
    parent: &Path,
    temp_prefix: String,
    content: &str,
) -> Result<(), OutputWriteError> {
    let existing_permissions = fs::metadata(path)
        .ok()
        .map(|metadata| metadata.permissions());
    let mut builder = tempfile::Builder::new();
    builder.prefix(&temp_prefix).suffix(".part");
    #[cfg(unix)]
    if existing_permissions.is_none() {
        builder.permissions(fs::Permissions::from_mode(0o666));
    }

    let mut temp = builder
        .tempfile_in(parent)
        .map_err(|source| OutputWriteError::CreateTemp {
            path: path.to_path_buf(),
            source,
        })?;

    if let Some(permissions) = existing_permissions
        && let Err(source) = fs::set_permissions(temp.path(), permissions)
    {
        let cleanup_warning = close_temp_file(temp);
        return Err(OutputWriteError::SetTempPermissions {
            path: path.to_path_buf(),
            source,
            cleanup_warning,
        });
    }

    if let Err(source) = temp.write_all(content.as_bytes()) {
        let cleanup_warning = close_temp_file(temp);
        return Err(OutputWriteError::Write {
            path: path.to_path_buf(),
            source,
            cleanup_warning,
        });
    }

    if let Err(source) = temp.flush() {
        let cleanup_warning = close_temp_file(temp);
        return Err(OutputWriteError::Flush {
            path: path.to_path_buf(),
            source,
            cleanup_warning,
        });
    }

    if let Err(source) = temp.as_file().sync_all() {
        let cleanup_warning = close_temp_file(temp);
        return Err(OutputWriteError::Sync {
            path: path.to_path_buf(),
            source,
            cleanup_warning,
        });
    }

    match temp.persist(path) {
        Ok(_) => Ok(()),
        Err(error) => {
            let source = error.error;
            let cleanup_warning = close_temp_file(error.file);
            Err(OutputWriteError::Persist {
                path: path.to_path_buf(),
                source,
                cleanup_warning,
            })
        }
    }
}

fn close_temp_file(temp: NamedTempFile) -> Option<String> {
    temp.close()
        .err()
        .map(|error| format!("Warning: could not remove temporary output file: {error}"))
}
