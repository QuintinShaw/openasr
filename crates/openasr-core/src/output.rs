#[cfg(test)]
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[path = "output_atomic.rs"]
mod output_atomic;
#[path = "output_path.rs"]
mod output_path;

#[derive(Debug, Error)]
pub enum OutputWriteError {
    #[error(
        "Output directory not found: {parent}\nPlease create the directory or choose an existing directory."
    )]
    ParentNotFound { parent: PathBuf },
    #[error(
        "Could not read output directory: {parent}\nPlease check the path and directory permissions. Details: {source}"
    )]
    ParentMetadata {
        parent: PathBuf,
        source: std::io::Error,
    },
    #[error(
        "Output parent path is not a directory: {parent}\nPlease choose a path inside an existing directory."
    )]
    ParentNotDirectory { parent: PathBuf },
    #[error(
        "Could not create temporary output file next to: {path}\nPlease choose a writable output path. Details: {source}"
    )]
    CreateTemp {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error(
        "Could not resolve symlinked output path: {path}\nPlease choose a writable output path. Details: {source}"
    )]
    ResolveSymlink {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error(
        "Could not open existing output for writing: {path}\nThe final output was not replaced. Details: {source}"
    )]
    ExistingOutputNotWritable {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error(
        "Could not prepare temporary output file next to: {path}\nThe final output was not replaced. Details: {source}"
    )]
    SetTempPermissions {
        path: PathBuf,
        source: std::io::Error,
        cleanup_warning: Option<String>,
    },
    #[error(
        "Could not write output to temporary file for: {path}\nThe final output was not replaced. Details: {source}"
    )]
    Write {
        path: PathBuf,
        source: std::io::Error,
        cleanup_warning: Option<String>,
    },
    #[error(
        "Could not flush temporary output file for: {path}\nThe final output was not replaced. Details: {source}"
    )]
    Flush {
        path: PathBuf,
        source: std::io::Error,
        cleanup_warning: Option<String>,
    },
    #[error(
        "Could not sync temporary output file for: {path}\nThe final output was not replaced. Details: {source}"
    )]
    Sync {
        path: PathBuf,
        source: std::io::Error,
        cleanup_warning: Option<String>,
    },
    #[error(
        "Could not replace output atomically: {path}\nThe existing final output was left unchanged when possible. Details: {source}"
    )]
    Persist {
        path: PathBuf,
        source: std::io::Error,
        cleanup_warning: Option<String>,
    },
}

impl OutputWriteError {
    pub fn cleanup_warning(&self) -> Option<&str> {
        match self {
            Self::Write {
                cleanup_warning, ..
            }
            | Self::Flush {
                cleanup_warning, ..
            }
            | Self::Sync {
                cleanup_warning, ..
            }
            | Self::Persist {
                cleanup_warning, ..
            }
            | Self::SetTempPermissions {
                cleanup_warning, ..
            } => cleanup_warning.as_deref(),
            Self::ParentNotFound { .. }
            | Self::ParentMetadata { .. }
            | Self::ParentNotDirectory { .. }
            | Self::CreateTemp { .. }
            | Self::ResolveSymlink { .. }
            | Self::ExistingOutputNotWritable { .. } => None,
        }
    }
}

pub fn atomic_write_text(path: impl AsRef<Path>, content: &str) -> Result<(), OutputWriteError> {
    let requested_path = path.as_ref();
    let resolved_path = output_path::resolve_output_path(requested_path)?;
    let path = resolved_path.as_path();
    let parent = output_path::output_parent(path);
    output_path::validate_output_parent(parent)?;
    output_atomic::validate_existing_output_writable(path)?;
    output_atomic::write_text_via_tempfile(path, parent, output_path::temp_prefix(path), content)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::FileTypeExt;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(unix)]
    use std::os::unix::fs::symlink;

    #[test]
    fn atomic_write_text_writes_and_replaces_existing_file() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("transcript.txt");
        fs::write(&output, "old transcript\n").unwrap();

        atomic_write_text(&output, "new transcript\n").unwrap();

        assert_eq!(fs::read_to_string(&output).unwrap(), "new transcript\n");
        assert!(part_files(temp.path()).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_text_preserves_existing_output_permissions() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("transcript.txt");
        fs::write(&output, "old transcript\n").unwrap();
        fs::set_permissions(&output, fs::Permissions::from_mode(0o664)).unwrap();

        atomic_write_text(&output, "new transcript\n").unwrap();

        assert_eq!(fs::read_to_string(&output).unwrap(), "new transcript\n");
        assert_eq!(
            fs::metadata(&output).unwrap().permissions().mode() & 0o777,
            0o664
        );
        assert!(part_files(temp.path()).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_text_rejects_read_only_existing_output_without_replacing() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("transcript.txt");
        fs::write(&output, "old transcript\n").unwrap();
        fs::set_permissions(&output, fs::Permissions::from_mode(0o444)).unwrap();

        let error = atomic_write_text(&output, "new transcript\n").unwrap_err();

        fs::set_permissions(&output, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(
            error
                .to_string()
                .contains("Could not open existing output for writing:")
        );
        assert_eq!(fs::read_to_string(&output).unwrap(), "old transcript\n");
        assert!(part_files(temp.path()).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_text_preserves_symlink_and_updates_target() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("target.txt");
        let link = temp.path().join("link.txt");
        fs::write(&target, "old transcript\n").unwrap();
        symlink("target.txt", &link).unwrap();

        atomic_write_text(&link, "new transcript\n").unwrap();

        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read_to_string(&target).unwrap(), "new transcript\n");
        assert_eq!(fs::read_to_string(&link).unwrap(), "new transcript\n");
        assert!(part_files(temp.path()).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_text_preserves_symlink_chain_and_updates_final_target() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("target.txt");
        let middle = temp.path().join("middle.txt");
        let link = temp.path().join("link.txt");
        fs::write(&target, "old transcript\n").unwrap();
        symlink("target.txt", &middle).unwrap();
        symlink("middle.txt", &link).unwrap();

        atomic_write_text(&link, "new transcript\n").unwrap();

        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert!(
            fs::symlink_metadata(&middle)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read_to_string(&target).unwrap(), "new transcript\n");
        assert_eq!(fs::read_to_string(&middle).unwrap(), "new transcript\n");
        assert_eq!(fs::read_to_string(&link).unwrap(), "new transcript\n");
        assert!(part_files(temp.path()).is_empty());
    }

    #[test]
    fn atomic_write_text_reports_missing_parent_without_part_file() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("missing").join("transcript.txt");

        let error = atomic_write_text(&output, "transcript\n").unwrap_err();

        assert!(error.to_string().contains("Output directory not found:"));
        assert!(!output.exists());
        assert!(part_files(temp.path()).is_empty());
    }

    #[test]
    fn atomic_write_text_rejects_existing_directory_without_part_file() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("existing-output");
        fs::create_dir(&output).unwrap();

        let error = atomic_write_text(&output, "transcript\n").unwrap_err();

        assert!(
            error
                .to_string()
                .contains("Could not open existing output for writing:")
        );
        assert!(output.is_dir());
        assert!(part_files(temp.path()).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_text_rejects_fifo_without_part_file() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("transcript.pipe");
        let status = std::process::Command::new("mkfifo")
            .arg(&output)
            .status()
            .unwrap();
        assert!(status.success());

        let error = atomic_write_text(&output, "transcript\n").unwrap_err();

        assert!(
            error
                .to_string()
                .contains("Could not open existing output for writing:")
        );
        assert!(fs::symlink_metadata(&output).unwrap().file_type().is_fifo());
        assert!(part_files(temp.path()).is_empty());
    }

    fn part_files(dir: &Path) -> Vec<PathBuf> {
        fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension().and_then(|extension| extension.to_str()) == Some("part")
            })
            .collect()
    }
}
