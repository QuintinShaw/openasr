use std::fs;
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::{
    ffi::CString,
    io::Write,
    os::unix::{
        ffi::OsStrExt,
        fs::OpenOptionsExt,
        io::{AsRawFd, FromRawFd},
    },
    sync::atomic::{AtomicU64, Ordering},
};
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

/// Resolve exactly the destination that [`atomic_write_text`] would replace.
///
/// Callers that publish a sibling artifact before the primary output can use
/// this to reject output aliases, including dangling symlink chains, without
/// duplicating the write path's security semantics.
pub fn resolve_output_target(path: impl AsRef<Path>) -> Result<PathBuf, OutputWriteError> {
    output_path::resolve_output_path(path.as_ref())
}

/// A fixed output target rooted in a directory handle. Unix mutations stay
/// relative to the held directory FD, so renaming that directory and replacing
/// its old path with a symlink cannot redirect a receipt or trace publication.
pub struct ResolvedOutputTarget {
    path: PathBuf,
    #[cfg(unix)]
    directory: fs::File,
    #[cfg(unix)]
    basename: CString,
}

impl ResolvedOutputTarget {
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[cfg(unix)]
    pub fn atomic_write_text(&self, content: &str) -> Result<(), OutputWriteError> {
        write_relative_to_directory(self, content, false)
    }

    #[cfg(unix)]
    pub fn create_new_text_and_sync_parent(&self, content: &str) -> Result<(), OutputWriteError> {
        write_relative_to_directory(self, content, true)
    }

    #[cfg(not(unix))]
    pub fn atomic_write_text(&self, _content: &str) -> Result<(), OutputWriteError> {
        Err(unsupported_pinned_target(&self.path))
    }

    #[cfg(not(unix))]
    pub fn create_new_text_and_sync_parent(&self, _content: &str) -> Result<(), OutputWriteError> {
        Err(unsupported_pinned_target(&self.path))
    }
}

/// Resolve a target and bind it to a controlled parent directory handle.
/// Non-Unix builds fail closed rather than claiming equivalent path pinning.
pub fn resolve_output_target_handle(
    path: impl AsRef<Path>,
) -> Result<ResolvedOutputTarget, OutputWriteError> {
    let path = resolve_output_target(path)?;
    #[cfg(unix)]
    {
        let parent = output_path::output_parent(&path);
        output_path::validate_output_parent(parent)?;
        let basename = CString::new(
            path.file_name()
                .ok_or_else(|| OutputWriteError::ResolveSymlink {
                    path: path.clone(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "output has no basename",
                    ),
                })?
                .as_bytes(),
        )
        .map_err(|source| OutputWriteError::ResolveSymlink {
            path: path.clone(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidInput, source),
        })?;
        let directory = fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(parent)
            .map_err(|source| OutputWriteError::ResolveSymlink {
                path: path.clone(),
                source,
            })?;
        Ok(ResolvedOutputTarget {
            path,
            directory,
            basename,
        })
    }
    #[cfg(not(unix))]
    {
        Err(unsupported_pinned_target(&path))
    }
}

#[cfg(not(unix))]
fn unsupported_pinned_target(path: &Path) -> OutputWriteError {
    OutputWriteError::ResolveSymlink {
        path: path.to_path_buf(),
        source: std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "pinned directory handles are unavailable on this platform",
        ),
    }
}

#[cfg(unix)]
static PINNED_TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

#[cfg(unix)]
fn pinned_existing_mode(
    target: &ResolvedOutputTarget,
) -> Result<Option<libc::mode_t>, OutputWriteError> {
    let mut stat = unsafe { std::mem::zeroed::<libc::stat>() };
    let result = unsafe {
        libc::fstatat(
            target.directory.as_raw_fd(),
            target.basename.as_ptr(),
            &mut stat,
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if result != 0 {
        let source = std::io::Error::last_os_error();
        if source.raw_os_error() == Some(libc::ENOENT) {
            return Ok(None);
        }
        return Err(OutputWriteError::ExistingOutputNotWritable {
            path: target.path.clone(),
            source,
        });
    }
    let file_type = stat.st_mode & libc::S_IFMT;
    // A final-component symlink introduced after the directory was pinned is
    // an untrusted race, not a target to follow. `renameat` below replaces the
    // directory entry atomically, so accept it without inheriting permissions
    // from the symlink target.
    if file_type == libc::S_IFLNK {
        return Ok(None);
    }
    if file_type != libc::S_IFREG {
        return Err(OutputWriteError::ExistingOutputNotWritable {
            path: target.path.clone(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "existing pinned output is not a regular file",
            ),
        });
    }
    let mode = stat.st_mode & 0o777;
    if mode & 0o222 == 0 {
        return Err(OutputWriteError::ExistingOutputNotWritable {
            path: target.path.clone(),
            source: std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "existing pinned output has no writable permission bits",
            ),
        });
    }
    Ok(Some(mode))
}

#[cfg(unix)]
fn write_relative_to_directory(
    target: &ResolvedOutputTarget,
    content: &str,
    create_new: bool,
) -> Result<(), OutputWriteError> {
    let existing_mode = if create_new {
        None
    } else {
        pinned_existing_mode(target)?
    };
    let name = CString::new(format!(
        ".openasr-{}-{}",
        std::process::id(),
        PINNED_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ))
    .expect("generated temp name has no NUL");
    let fd = unsafe {
        libc::openat(
            target.directory.as_raw_fd(),
            name.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC,
            0o666,
        )
    };
    if fd < 0 {
        return Err(OutputWriteError::CreateTemp {
            path: target.path.clone(),
            source: std::io::Error::last_os_error(),
        });
    }
    if let Some(mode) = existing_mode
        && unsafe { libc::fchmod(fd, mode) } != 0
    {
        let source = std::io::Error::last_os_error();
        unsafe {
            libc::close(fd);
            libc::unlinkat(target.directory.as_raw_fd(), name.as_ptr(), 0);
        }
        return Err(OutputWriteError::SetTempPermissions {
            path: target.path.clone(),
            source,
            cleanup_warning: None,
        });
    }
    let mut temp = unsafe { fs::File::from_raw_fd(fd) };
    if let Err(source) = temp
        .write_all(content.as_bytes())
        .and_then(|_| temp.sync_all())
    {
        unsafe { libc::unlinkat(target.directory.as_raw_fd(), name.as_ptr(), 0) };
        return Err(OutputWriteError::Write {
            path: target.path.clone(),
            source,
            cleanup_warning: None,
        });
    }
    drop(temp);
    let publish = if create_new {
        unsafe {
            libc::linkat(
                target.directory.as_raw_fd(),
                name.as_ptr(),
                target.directory.as_raw_fd(),
                target.basename.as_ptr(),
                0,
            )
        }
    } else {
        unsafe {
            libc::renameat(
                target.directory.as_raw_fd(),
                name.as_ptr(),
                target.directory.as_raw_fd(),
                target.basename.as_ptr(),
            )
        }
    };
    if publish != 0 {
        let source = std::io::Error::last_os_error();
        unsafe { libc::unlinkat(target.directory.as_raw_fd(), name.as_ptr(), 0) };
        return Err(OutputWriteError::Persist {
            path: target.path.clone(),
            source,
            cleanup_warning: None,
        });
    }
    if create_new {
        unsafe { libc::unlinkat(target.directory.as_raw_fd(), name.as_ptr(), 0) };
    }
    target
        .directory
        .sync_all()
        .map_err(|source| OutputWriteError::Sync {
            path: target.path.clone(),
            source,
            cleanup_warning: None,
        })
}

/// Write through a target previously returned by [`resolve_output_target`]
/// without resolving a caller-controlled path again. A final symlink is
/// rejected; an atomic rename that races a later symlink swap replaces that
/// link rather than following it.
pub fn atomic_write_text_to_resolved_target(
    target: impl AsRef<Path>,
    content: &str,
) -> Result<(), OutputWriteError> {
    let path = target.as_ref();
    if fs::symlink_metadata(path)
        .map(|metadata| metadata.file_type().is_symlink())
        .unwrap_or(false)
    {
        return Err(OutputWriteError::ResolveSymlink {
            path: path.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "fixed output target became a symlink",
            ),
        });
    }
    let parent = output_path::output_parent(path);
    output_path::validate_output_parent(parent)?;
    output_atomic::validate_existing_output_writable(path)?;
    output_atomic::write_text_via_tempfile(path, parent, output_path::temp_prefix(path), content)
}

pub fn atomic_write_text(path: impl AsRef<Path>, content: &str) -> Result<(), OutputWriteError> {
    let resolved_path = resolve_output_target(path)?;
    atomic_write_text_to_resolved_target(resolved_path, content)
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

    #[cfg(unix)]
    #[test]
    fn pinned_target_rejects_read_only_and_non_regular_existing_entries() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        let readonly = temp.path().join("readonly.txt");
        fs::write(&readonly, "old").unwrap();
        fs::set_permissions(&readonly, fs::Permissions::from_mode(0o444)).unwrap();
        let target = resolve_output_target_handle(&readonly).unwrap();
        assert!(matches!(
            target.atomic_write_text("new"),
            Err(OutputWriteError::ExistingOutputNotWritable { .. })
        ));

        let directory = temp.path().join("directory");
        fs::create_dir(&directory).unwrap();
        let target = resolve_output_target_handle(&directory).unwrap();
        assert!(matches!(
            target.atomic_write_text("new"),
            Err(OutputWriteError::ExistingOutputNotWritable { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn pinned_target_preserves_existing_permissions_and_creates_new_output() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        let existing = temp.path().join("existing.txt");
        fs::write(&existing, "old").unwrap();
        fs::set_permissions(&existing, fs::Permissions::from_mode(0o640)).unwrap();
        resolve_output_target_handle(&existing)
            .unwrap()
            .atomic_write_text("new")
            .unwrap();
        assert_eq!(
            fs::metadata(&existing).unwrap().permissions().mode() & 0o777,
            0o640
        );

        let created = temp.path().join("created.txt");
        resolve_output_target_handle(&created)
            .unwrap()
            .atomic_write_text("new")
            .unwrap();
        assert_ne!(
            fs::metadata(&created).unwrap().permissions().mode() & 0o200,
            0,
        );
    }

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
