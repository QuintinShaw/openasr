use std::{
    fs,
    path::{Component, Path, PathBuf},
};

use super::OutputWriteError;

pub(super) fn resolve_output_path(path: &Path) -> Result<PathBuf, OutputWriteError> {
    let original = path.to_path_buf();
    let mut current = path.to_path_buf();
    for _ in 0..40 {
        let metadata = match fs::symlink_metadata(&current) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return canonicalized_final_target(&current, &original);
            }
            Err(source) => {
                return Err(OutputWriteError::ResolveSymlink {
                    path: original,
                    source,
                });
            }
        };

        if !metadata.file_type().is_symlink() {
            return canonicalized_final_target(&current, &original);
        }

        let target =
            fs::read_link(&current).map_err(|source| OutputWriteError::ResolveSymlink {
                path: original.clone(),
                source,
            })?;
        current = if target.is_absolute() {
            target
        } else {
            output_parent(&current).join(target)
        };
    }

    Err(OutputWriteError::ResolveSymlink {
        path: original,
        source: std::io::Error::new(std::io::ErrorKind::InvalidInput, "too many symlink levels"),
    })
}

fn lexically_normalized_absolute(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = normalized.pop();
            }
            Component::Normal(name) => normalized.push(name),
        }
    }
    normalized
}

fn canonicalized_final_target(path: &Path, original: &Path) -> Result<PathBuf, OutputWriteError> {
    let normalized = lexically_normalized_absolute(path);
    let parent = output_parent(&normalized);
    let canonical_parent = fs::canonicalize(parent).map_err(|source| {
        if source.kind() == std::io::ErrorKind::NotFound {
            OutputWriteError::ParentNotFound {
                parent: parent.to_path_buf(),
            }
        } else {
            OutputWriteError::ResolveSymlink {
                path: original.to_path_buf(),
                source,
            }
        }
    })?;
    let basename = normalized
        .file_name()
        .ok_or_else(|| OutputWriteError::ResolveSymlink {
            path: original.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "output path has no file name",
            ),
        })?;
    Ok(canonical_parent.join(basename))
}

pub(super) fn output_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

pub(super) fn validate_output_parent(parent: &Path) -> Result<(), OutputWriteError> {
    let metadata = metadata_for_parent(parent)?;

    if metadata.is_dir() {
        return Ok(());
    }
    Err(OutputWriteError::ParentNotDirectory {
        parent: parent.to_path_buf(),
    })
}

fn metadata_for_parent(parent: &Path) -> Result<fs::Metadata, OutputWriteError> {
    fs::metadata(parent).map_err(|source| {
        if source.kind() == std::io::ErrorKind::NotFound {
            OutputWriteError::ParentNotFound {
                parent: parent.to_path_buf(),
            }
        } else {
            OutputWriteError::ParentMetadata {
                parent: parent.to_path_buf(),
                source,
            }
        }
    })
}

pub(super) fn temp_prefix(path: &Path) -> String {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("output");
    format!(".{file_name}.")
}
