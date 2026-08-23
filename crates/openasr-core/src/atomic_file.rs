use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(test)]
use std::cell::Cell;

static ATOMIC_FILE_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AtomicFileFailpoint {
    BeforeSync,
    BeforeReplace,
    AfterReplace,
}

#[cfg(test)]
thread_local! {
    static ATOMIC_FILE_FAILPOINT: Cell<u8> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn set_atomic_file_failpoint(failpoint: Option<AtomicFileFailpoint>) {
    let value = match failpoint {
        None => 0,
        Some(AtomicFileFailpoint::BeforeSync) => 1,
        Some(AtomicFileFailpoint::BeforeReplace) => 2,
        Some(AtomicFileFailpoint::AfterReplace) => 3,
    };
    ATOMIC_FILE_FAILPOINT.with(|current| current.set(value));
}

fn failpoint_before_sync(explicit: Option<AtomicFileFailpoint>) -> io::Result<()> {
    if explicit == Some(AtomicFileFailpoint::BeforeSync) {
        return Err(io::Error::other("injected failure before staging sync"));
    }
    #[cfg(test)]
    if ATOMIC_FILE_FAILPOINT.with(Cell::get) == 1 {
        return Err(io::Error::other("injected failure before staging sync"));
    }
    Ok(())
}

fn failpoint_before_replace(explicit: Option<AtomicFileFailpoint>) -> io::Result<()> {
    if explicit == Some(AtomicFileFailpoint::BeforeReplace) {
        return Err(io::Error::other(
            "injected failure before atomic replacement",
        ));
    }
    #[cfg(test)]
    if ATOMIC_FILE_FAILPOINT.with(Cell::get) == 2 {
        return Err(io::Error::other(
            "injected failure before atomic replacement",
        ));
    }
    Ok(())
}

fn failpoint_after_replace(explicit: Option<AtomicFileFailpoint>) -> io::Result<()> {
    if explicit == Some(AtomicFileFailpoint::AfterReplace) {
        return Err(io::Error::other(
            "injected failure after atomic replacement",
        ));
    }
    #[cfg(test)]
    if ATOMIC_FILE_FAILPOINT.with(Cell::get) == 3 {
        return Err(io::Error::other(
            "injected failure after atomic replacement",
        ));
    }
    Ok(())
}

pub(crate) fn write_file_atomically(path: &Path, contents: &[u8]) -> io::Result<()> {
    write_file_atomically_detailed(path, contents, AtomicFileMode::Default)
        .map(|_| ())
        .map_err(|AtomicWriteError::NotCommitted(source)| source)
}

pub(crate) fn write_file_atomically_detailed(
    path: &Path,
    contents: &[u8],
    mode: AtomicFileMode,
) -> Result<AtomicWriteOutcome, AtomicWriteError> {
    write_file_atomically_detailed_with(&RealAtomicFileSystem, path, contents, mode)
}

pub(crate) fn write_file_atomically_detailed_with_failpoint(
    path: &Path,
    contents: &[u8],
    mode: AtomicFileMode,
    failpoint: AtomicFileFailpoint,
) -> Result<AtomicWriteOutcome, AtomicWriteError> {
    write_file_atomically_detailed_with_injected(
        &RealAtomicFileSystem,
        path,
        contents,
        mode,
        Some(failpoint),
    )
}

/// Result of replacing a staged file. A committed replacement whose parent
/// directory could not be synced is still a successful commit; callers may
/// surface the durability warning without treating the target as absent.
#[derive(Debug)]
pub(crate) enum AtomicReplaceOutcome {
    Replaced,
    CommittedWithSyncWarning { source: io::Error },
}

#[derive(Debug)]
pub(crate) enum AtomicReplaceError {
    NotReplaced(io::Error),
}

impl std::fmt::Display for AtomicReplaceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotReplaced(error) => write!(formatter, "atomic replacement failed: {error}"),
        }
    }
}

impl std::error::Error for AtomicReplaceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NotReplaced(error) => Some(error),
        }
    }
}

/// Replace a staged sibling and report whether the rename committed before a
/// parent-directory sync error. This is the recovery boundary used by V2.
pub(crate) fn replace_file_atomically_detailed(
    from: &Path,
    path: &Path,
) -> Result<AtomicReplaceOutcome, AtomicReplaceError> {
    replace_file_atomically_detailed_with(&RealAtomicFileSystem, from, path)
}

fn replace_file_atomically_detailed_with(
    filesystem: &impl AtomicFileSystem,
    from: &Path,
    path: &Path,
) -> Result<AtomicReplaceOutcome, AtomicReplaceError> {
    filesystem
        .rename(from, path)
        .map_err(AtomicReplaceError::NotReplaced)?;
    match filesystem.sync_parent_dir(path) {
        Ok(()) => Ok(AtomicReplaceOutcome::Replaced),
        Err(source) => Ok(AtomicReplaceOutcome::CommittedWithSyncWarning { source }),
    }
}

pub(crate) fn replace_file_atomically(from: &Path, path: &Path) -> io::Result<()> {
    match replace_file_atomically_detailed(from, path) {
        Ok(AtomicReplaceOutcome::Replaced) => Ok(()),
        Ok(AtomicReplaceOutcome::CommittedWithSyncWarning { source }) => {
            eprintln!(
                "openasr-core: atomic replacement committed but parent sync failed: {source}"
            );
            Ok(())
        }
        Err(AtomicReplaceError::NotReplaced(source)) => Err(source),
    }
}

/// A detailed result for the temporary-file writer. The target is authoritative
/// after `rename` succeeds, even when the post-rename durability step reports
/// an error (including Windows `MoveFileExW` callers).
#[derive(Debug)]
pub(crate) enum AtomicWriteOutcome {
    Written,
    CommittedWithSyncWarning { source: io::Error },
}

#[derive(Debug)]
pub(crate) enum AtomicWriteError {
    NotCommitted(io::Error),
}

impl std::fmt::Display for AtomicWriteError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotCommitted(error) => write!(formatter, "atomic write failed: {error}"),
        }
    }
}

impl std::error::Error for AtomicWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NotCommitted(error) => Some(error),
        }
    }
}

/// Atomically writes `contents` to `path`, creating the temporary file with
/// owner-only (0600) permissions from the moment it is created (not
/// after-the-fact via a post-rename `chmod`), then re-asserting 0600 on the
/// renamed target as a defense-in-depth belt-and-suspenders step. Callers
/// with secret-bearing files (API key hashes, voiceprint enrollments, TLS
/// private keys) use this instead of [`write_file_atomically`] so there is
/// never a window where the file is readable by group/other, regardless of
/// umask. Exposed crate-externally (via the crate-root re-export) for
/// `openasr-server`'s TLS identity store, which holds a raw private key.
pub fn write_owner_only_file_atomically(path: &Path, contents: &[u8]) -> io::Result<()> {
    write_owner_only_file_atomically_with(&RealAtomicFileSystem, path, contents)
}

fn write_owner_only_file_atomically_with(
    fs: &impl AtomicFileSystem,
    path: &Path,
    contents: &[u8],
) -> io::Result<()> {
    write_file_atomically_detailed_with(fs, path, contents, AtomicFileMode::OwnerOnly)
        .map(|_| ())
        .map_err(|AtomicWriteError::NotCommitted(source)| source)
}

#[cfg(test)]
fn write_file_atomically_with(
    fs: &impl AtomicFileSystem,
    path: &Path,
    contents: &[u8],
    mode: AtomicFileMode,
) -> io::Result<()> {
    write_file_atomically_detailed_with(fs, path, contents, mode)
        .map(|_| ())
        .map_err(|AtomicWriteError::NotCommitted(source)| source)
}

fn write_file_atomically_detailed_with(
    fs: &impl AtomicFileSystem,
    path: &Path,
    contents: &[u8],
    mode: AtomicFileMode,
) -> Result<AtomicWriteOutcome, AtomicWriteError> {
    write_file_atomically_detailed_with_injected(fs, path, contents, mode, None)
}

fn write_file_atomically_detailed_with_injected(
    fs: &impl AtomicFileSystem,
    path: &Path,
    contents: &[u8],
    mode: AtomicFileMode,
    failpoint: Option<AtomicFileFailpoint>,
) -> Result<AtomicWriteOutcome, AtomicWriteError> {
    let temp_path = atomic_temp_path(path);
    let result = (|| {
        let mut file = fs
            .create_new(&temp_path, mode)
            .map_err(AtomicWriteError::NotCommitted)?;
        if mode == AtomicFileMode::OwnerOnly {
            fs.set_owner_only_permissions(&temp_path)
                .map_err(AtomicWriteError::NotCommitted)?;
        }
        file.write_all(contents)
            .map_err(AtomicWriteError::NotCommitted)?;
        file.flush().map_err(AtomicWriteError::NotCommitted)?;
        failpoint_before_sync(failpoint).map_err(AtomicWriteError::NotCommitted)?;
        file.sync_all().map_err(AtomicWriteError::NotCommitted)?;
        drop(file);
        failpoint_before_replace(failpoint).map_err(AtomicWriteError::NotCommitted)?;
        fs.rename(&temp_path, path)
            .map_err(AtomicWriteError::NotCommitted)?;

        let mut warning = failpoint_after_replace(failpoint).err();
        if warning.is_none() && mode == AtomicFileMode::OwnerOnly {
            warning = fs.set_owner_only_permissions(path).err();
        }
        if warning.is_none() {
            warning = fs.sync_parent_dir(path).err();
        }
        Ok(match warning {
            Some(source) => AtomicWriteOutcome::CommittedWithSyncWarning { source },
            None => AtomicWriteOutcome::Written,
        })
    })();

    if result.is_err() {
        let _ = fs.remove_file(&temp_path);
    }
    result
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AtomicFileMode {
    Default,
    OwnerOnly,
}

trait AtomicFile: Write {
    fn sync_all(&mut self) -> io::Result<()>;
}

trait AtomicFileSystem {
    type File: AtomicFile;

    fn create_new(&self, path: &Path, mode: AtomicFileMode) -> io::Result<Self::File>;
    fn set_owner_only_permissions(&self, path: &Path) -> io::Result<()>;
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;
    fn remove_file(&self, path: &Path) -> io::Result<()>;
    fn sync_parent_dir(&self, path: &Path) -> io::Result<()>;
}

struct RealAtomicFileSystem;

impl AtomicFile for fs::File {
    fn sync_all(&mut self) -> io::Result<()> {
        fs::File::sync_all(self)
    }
}

impl AtomicFileSystem for RealAtomicFileSystem {
    type File = fs::File;

    fn create_new(&self, path: &Path, mode: AtomicFileMode) -> io::Result<Self::File> {
        #[cfg(not(unix))]
        let _ = mode;
        #[cfg(unix)]
        let _ = mode;

        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            // Every staging file is private from creation, including ordinary
            // config/catalog files. This avoids a permission window before rename.
            options.mode(0o600);
        }
        options.open(path)
    }

    // TODO(windows): this is a no-op on Windows. An owner-only equivalent
    // would mean building and applying a DACL (via
    // Win32_Security_Authorization's SetNamedSecurityInfoW or similar) that
    // grants access only to the current user's SID -- meaningfully more than
    // a `windows-sys` feature-flag addition, so it is not done here. Secret
    // stores written through `write_owner_only_file_atomically` (API keys,
    // voiceprint enrollments, TLS private keys) rely on the Windows user
    // profile directory's default ACL for protection on that platform.
    fn set_owner_only_permissions(&self, path: &Path) -> io::Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        }
        #[cfg(not(unix))]
        let _ = path;
        Ok(())
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            use windows_sys::Win32::Storage::FileSystem::{
                MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
            };

            let from_wide: Vec<u16> = from.as_os_str().encode_wide().chain(Some(0)).collect();
            let to_wide: Vec<u16> = to.as_os_str().encode_wide().chain(Some(0)).collect();
            // SAFETY: both buffers are live, NUL-terminated UTF-16 paths. The
            // flags request the Windows equivalent of Unix rename-over-target
            // and flush the rename before returning.
            if unsafe {
                MoveFileExW(
                    from_wide.as_ptr(),
                    to_wide.as_ptr(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
                )
            } == 0
            {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }
        #[cfg(not(windows))]
        {
            fs::rename(from, to)
        }
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        fs::remove_file(path)
    }

    fn sync_parent_dir(&self, path: &Path) -> io::Result<()> {
        let Some(parent) = path.parent() else {
            return Ok(());
        };
        fs::File::open(parent).and_then(|file| file.sync_all())
    }
}

fn atomic_temp_path(path: &Path) -> PathBuf {
    let sequence = ATOMIC_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    path.with_file_name(format!(
        ".openasr-{:x}-{now:x}-{sequence:x}.tmp",
        std::process::id(),
    ))
}

fn is_private_staging_name(name: &str) -> bool {
    let Some(body) = name
        .strip_prefix(".openasr-")
        .and_then(|value| value.strip_suffix(".tmp"))
    else {
        return false;
    };
    let parts = body.split('-').collect::<Vec<_>>();
    parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

/// must hold the process/file writer boundary before invoking this explicit
/// recovery operation; ordinary writes ignore orphans rather than racing them.
#[allow(dead_code)]
pub(crate) fn cleanup_orphan_staging_files(parent: &Path) -> io::Result<usize> {
    let mut removed = 0;
    let entries = match fs::read_dir(parent) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !is_private_staging_name(name) {
            continue;
        }
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_file() && !metadata.file_type().is_symlink() {
            fs::remove_file(&path)?;
            removed += 1;
        }
    }
    Ok(removed)
}

pub(crate) fn sync_parent_dir_best_effort(path: &Path) {
    let Some(parent) = path.parent() else {
        return;
    };
    let _ = fs::File::open(parent).and_then(|file| file.sync_all());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        cell::{Cell, RefCell},
        collections::BTreeMap,
        rc::Rc,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum FailurePoint {
        Write,
        Sync,
        Rename,
        ParentSync,
    }

    #[derive(Default)]
    struct FakeAtomicFileSystemState {
        files: RefCell<BTreeMap<PathBuf, Vec<u8>>>,
        temp_path: RefCell<Option<PathBuf>>,
        created_modes: RefCell<Vec<AtomicFileMode>>,
        owner_only_permission_paths: RefCell<Vec<PathBuf>>,
        removed_temp: Cell<bool>,
        synced_parent: Cell<bool>,
        failure_point: Cell<Option<FailurePoint>>,
    }

    #[derive(Clone, Default)]
    struct FakeAtomicFileSystem {
        state: Rc<FakeAtomicFileSystemState>,
    }

    struct FakeAtomicFile {
        path: PathBuf,
        state: Rc<FakeAtomicFileSystemState>,
    }

    impl FakeAtomicFileSystem {
        fn with_target(path: &Path, contents: &[u8]) -> Self {
            let fs = Self::default();
            fs.state
                .files
                .borrow_mut()
                .insert(path.to_path_buf(), contents.to_vec());
            fs
        }

        fn fail_at(&self, failure_point: FailurePoint) {
            self.state.failure_point.set(Some(failure_point));
        }

        fn target_contents(&self, path: &Path) -> Option<Vec<u8>> {
            self.state.files.borrow().get(path).cloned()
        }

        fn temp_exists(&self) -> bool {
            self.state
                .temp_path
                .borrow()
                .as_ref()
                .is_some_and(|path| self.state.files.borrow().contains_key(path))
        }

        fn temp_path(&self) -> PathBuf {
            self.state
                .temp_path
                .borrow()
                .clone()
                .expect("temp path should be recorded")
        }
    }

    impl Write for FakeAtomicFile {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.state.failure_point.get() == Some(FailurePoint::Write) {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "injected write failure",
                ));
            }
            self.state
                .files
                .borrow_mut()
                .entry(self.path.clone())
                .or_default()
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl AtomicFile for FakeAtomicFile {
        fn sync_all(&mut self) -> io::Result<()> {
            if self.state.failure_point.get() == Some(FailurePoint::Sync) {
                return Err(io::Error::other("injected sync failure"));
            }
            Ok(())
        }
    }

    impl AtomicFileSystem for FakeAtomicFileSystem {
        type File = FakeAtomicFile;

        fn create_new(&self, path: &Path, mode: AtomicFileMode) -> io::Result<Self::File> {
            let mut files = self.state.files.borrow_mut();
            if files.contains_key(path) {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "temp already exists",
                ));
            }
            files.insert(path.to_path_buf(), Vec::new());
            *self.state.temp_path.borrow_mut() = Some(path.to_path_buf());
            self.state.created_modes.borrow_mut().push(mode);
            Ok(FakeAtomicFile {
                path: path.to_path_buf(),
                state: Rc::clone(&self.state),
            })
        }

        fn set_owner_only_permissions(&self, path: &Path) -> io::Result<()> {
            self.state
                .owner_only_permission_paths
                .borrow_mut()
                .push(path.to_path_buf());
            Ok(())
        }

        fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
            if self.state.failure_point.get() == Some(FailurePoint::Rename) {
                return Err(io::Error::other("injected rename failure"));
            }
            let mut files = self.state.files.borrow_mut();
            let contents = files.remove(from).ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, "temp file missing before rename")
            })?;
            files.insert(to.to_path_buf(), contents);
            Ok(())
        }

        fn remove_file(&self, path: &Path) -> io::Result<()> {
            self.state.removed_temp.set(true);
            self.state.files.borrow_mut().remove(path);
            Ok(())
        }

        fn sync_parent_dir(&self, _path: &Path) -> io::Result<()> {
            if self.state.failure_point.get() == Some(FailurePoint::ParentSync) {
                return Err(io::Error::other("injected parent sync failure"));
            }
            self.state.synced_parent.set(true);
            Ok(())
        }
    }

    fn assert_failure_cleans_temp_and_preserves_target(failure_point: FailurePoint) {
        let target = Path::new("/tmp/openasr/config.json");
        let fs = FakeAtomicFileSystem::with_target(target, b"old");
        fs.fail_at(failure_point);

        let error =
            write_file_atomically_with(&fs, target, b"new", AtomicFileMode::Default).unwrap_err();

        assert!(!error.to_string().is_empty());
        assert_eq!(fs.target_contents(target), Some(b"old".to_vec()));
        assert!(fs.state.removed_temp.get());
        assert!(!fs.temp_exists());
        assert!(!fs.state.synced_parent.get());
        assert_eq!(
            fs.state.created_modes.borrow().as_slice(),
            &[AtomicFileMode::Default]
        );
        assert!(fs.state.owner_only_permission_paths.borrow().is_empty());
    }

    #[test]
    fn write_failure_cleans_temp_and_preserves_target() {
        assert_failure_cleans_temp_and_preserves_target(FailurePoint::Write);
    }

    #[test]
    fn sync_failure_cleans_temp_and_preserves_target() {
        assert_failure_cleans_temp_and_preserves_target(FailurePoint::Sync);
    }

    #[test]
    fn rename_failure_cleans_temp_and_preserves_target() {
        assert_failure_cleans_temp_and_preserves_target(FailurePoint::Rename);
    }

    #[test]
    fn successful_write_renames_and_syncs_parent() {
        let target = Path::new("/tmp/openasr/config.json");
        let fs = FakeAtomicFileSystem::with_target(target, b"old");

        write_file_atomically_with(&fs, target, b"new", AtomicFileMode::Default).unwrap();

        assert_eq!(fs.target_contents(target), Some(b"new".to_vec()));
        assert!(!fs.state.removed_temp.get());
        assert!(!fs.temp_exists());
        assert!(fs.state.synced_parent.get());
    }

    #[test]
    fn owner_only_success_uses_sibling_temp_permissions_and_syncs_parent() {
        let target = Path::new("/tmp/openasr/voiceprints.json");
        let fs = FakeAtomicFileSystem::with_target(target, b"old");

        write_owner_only_file_atomically_with(&fs, target, b"new").unwrap();

        let temp_path = fs.temp_path();
        assert_eq!(temp_path.parent(), target.parent());
        let temp_file_name = temp_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap();
        assert!(temp_file_name.starts_with(".openasr-"));
        assert!(temp_file_name.ends_with(".tmp"));
        assert_eq!(fs.target_contents(target), Some(b"new".to_vec()));
        assert!(!fs.temp_exists());
        assert!(fs.state.synced_parent.get());
        assert_eq!(
            fs.state.created_modes.borrow().as_slice(),
            &[AtomicFileMode::OwnerOnly]
        );
        assert_eq!(
            fs.state.owner_only_permission_paths.borrow().as_slice(),
            &[temp_path, target.to_path_buf()]
        );
    }

    #[test]
    fn atomic_temp_name_stays_short_for_long_backend_metadata_names() {
        let target = Path::new("/tmp/openasr/deep")
            .join(".openasr-0.1.34-windows-x86_64-cuda-sm_86-plugin.dll.partial.json");
        let temp = atomic_temp_path(&target);

        assert_eq!(temp.parent(), target.parent());
        let temp_name = temp.file_name().unwrap().to_string_lossy();
        assert!(temp_name.starts_with(".openasr-"));
        assert!(temp_name.ends_with(".tmp"));
        assert!(temp_name.len() < target.file_name().unwrap().len());
    }

    #[test]
    fn owner_only_sync_failure_cleans_temp_and_preserves_target() {
        let target = Path::new("/tmp/openasr/voiceprints.json");
        let fs = FakeAtomicFileSystem::with_target(target, b"old");
        fs.fail_at(FailurePoint::Sync);

        let error = write_owner_only_file_atomically_with(&fs, target, b"new").unwrap_err();

        assert!(!error.to_string().is_empty());
        assert_eq!(fs.target_contents(target), Some(b"old".to_vec()));
        assert!(fs.state.removed_temp.get());
        assert!(!fs.temp_exists());
        assert!(!fs.state.synced_parent.get());
        assert_eq!(
            fs.state.created_modes.borrow().as_slice(),
            &[AtomicFileMode::OwnerOnly]
        );
        assert_eq!(
            fs.state.owner_only_permission_paths.borrow().as_slice(),
            &[fs.temp_path()]
        );
    }

    #[test]
    fn failpoint_before_replace_preserves_target_and_cleans_staging() {
        let target = Path::new("/tmp/openasr/config.json");
        let fs = FakeAtomicFileSystem::with_target(target, b"old");
        set_atomic_file_failpoint(Some(AtomicFileFailpoint::BeforeReplace));
        let error = write_file_atomically_with(&fs, target, b"new", AtomicFileMode::Default);
        set_atomic_file_failpoint(None);

        assert!(error.is_err());
        assert_eq!(fs.target_contents(target), Some(b"old".to_vec()));
        assert!(!fs.temp_exists());
    }

    #[test]
    fn failpoint_after_replace_reports_committed_new_record() {
        let target = Path::new("/tmp/openasr/config.json");
        let fs = FakeAtomicFileSystem::with_target(target, b"old");
        set_atomic_file_failpoint(Some(AtomicFileFailpoint::AfterReplace));
        let outcome =
            write_file_atomically_detailed_with(&fs, target, b"new", AtomicFileMode::Default)
                .unwrap();
        set_atomic_file_failpoint(None);

        assert!(matches!(
            outcome,
            AtomicWriteOutcome::CommittedWithSyncWarning { .. }
        ));
        assert_eq!(fs.target_contents(target), Some(b"new".to_vec()));
    }

    #[test]
    fn detailed_replace_distinguishes_not_replaced() {
        let target = Path::new("/tmp/openasr/config.json");
        let staged = Path::new("/tmp/openasr/config.stage");
        let fs = FakeAtomicFileSystem::with_target(target, b"old");
        fs.state
            .files
            .borrow_mut()
            .insert(staged.to_path_buf(), b"new".to_vec());
        fs.fail_at(FailurePoint::Rename);

        let error = replace_file_atomically_detailed_with(&fs, staged, target).unwrap_err();
        assert!(matches!(error, AtomicReplaceError::NotReplaced(_)));
        assert_eq!(fs.target_contents(target), Some(b"old".to_vec()));
    }

    #[test]
    fn detailed_replace_reports_committed_target_when_parent_sync_fails() {
        let target = Path::new("/tmp/openasr/config.json");
        let staged = Path::new("/tmp/openasr/config.stage");
        let fs = FakeAtomicFileSystem::with_target(target, b"old");
        fs.state
            .files
            .borrow_mut()
            .insert(staged.to_path_buf(), b"new".to_vec());
        fs.fail_at(FailurePoint::ParentSync);

        let outcome = replace_file_atomically_detailed_with(&fs, staged, target).unwrap();
        assert!(matches!(
            outcome,
            AtomicReplaceOutcome::CommittedWithSyncWarning { .. }
        ));
        assert_eq!(fs.target_contents(target), Some(b"new".to_vec()));
    }

    #[test]
    fn detailed_write_reports_committed_target_when_parent_sync_fails() {
        let target = Path::new("/tmp/openasr/config.json");
        let fs = FakeAtomicFileSystem::with_target(target, b"old");
        fs.fail_at(FailurePoint::ParentSync);

        let outcome =
            write_file_atomically_detailed_with(&fs, target, b"new", AtomicFileMode::Default)
                .unwrap();
        assert!(matches!(
            outcome,
            AtomicWriteOutcome::CommittedWithSyncWarning { .. }
        ));
        assert_eq!(fs.target_contents(target), Some(b"new".to_vec()));
    }

    #[test]
    fn orphan_cleanup_only_removes_matching_regular_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(".openasr-a-b-c.tmp"), b"stale").unwrap();
        fs::write(dir.path().join(".openasr-user-export.tmp"), b"keep").unwrap();
        fs::write(dir.path().join("keep.tmp"), b"keep").unwrap();
        fs::create_dir(dir.path().join(".openasr-dir.tmp")).unwrap();

        assert_eq!(cleanup_orphan_staging_files(dir.path()).unwrap(), 1);
        assert!(!dir.path().join(".openasr-a-b-c.tmp").exists());
        assert!(dir.path().join(".openasr-user-export.tmp").exists());
        assert!(dir.path().join("keep.tmp").exists());
        assert!(dir.path().join(".openasr-dir.tmp").exists());
    }

    #[cfg(windows)]
    #[test]
    fn windows_replace_uses_existing_target_semantics() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("record.json");
        let staged = dir.path().join("record.stage");
        fs::write(&target, b"old").unwrap();
        fs::write(&staged, b"new").unwrap();

        replace_file_atomically_detailed(&staged, &target).unwrap();

        assert_eq!(fs::read(&target).unwrap(), b"new");
        assert!(!staged.exists());
    }

    #[cfg(unix)]
    #[test]
    fn owner_only_temp_file_is_0600_from_the_moment_it_is_created() {
        // Regression test for a write-then-chmod window: exercises the real
        // (non-faked) `RealAtomicFileSystem::create_new` in `OwnerOnly` mode
        // and stats the temp file immediately -- before any `write_all`,
        // `rename`, or the later explicit `set_owner_only_permissions` call
        // that `write_file_atomically_with` performs on the renamed target --
        // to pin down that the file is 0600 from creation, not merely 0600 in
        // steady state. A temp file created with a permissive default mode
        // (0666 & ~umask, commonly 0644) and only tightened to 0600 after
        // rename would leave a group/other-readable window during which a
        // local user could read a raw private key off disk.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let temp_path = dir.path().join(".probe.tmp");
        let file = RealAtomicFileSystem
            .create_new(&temp_path, AtomicFileMode::OwnerOnly)
            .unwrap();
        let mode = file.metadata().unwrap().permissions().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "temp file must be 0600 the instant create_new returns, before any later chmod"
        );
    }
}
