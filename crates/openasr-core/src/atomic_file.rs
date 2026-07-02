use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

static ATOMIC_FILE_COUNTER: AtomicU64 = AtomicU64::new(1);

pub(crate) fn write_file_atomically(path: &Path, contents: &[u8]) -> io::Result<()> {
    write_file_atomically_with(
        &RealAtomicFileSystem,
        path,
        contents,
        AtomicFileMode::Default,
    )
}

pub(crate) fn write_owner_only_file_atomically(path: &Path, contents: &[u8]) -> io::Result<()> {
    write_owner_only_file_atomically_with(&RealAtomicFileSystem, path, contents)
}

fn write_owner_only_file_atomically_with(
    fs: &impl AtomicFileSystem,
    path: &Path,
    contents: &[u8],
) -> io::Result<()> {
    write_file_atomically_with(fs, path, contents, AtomicFileMode::OwnerOnly)
}

fn write_file_atomically_with(
    fs: &impl AtomicFileSystem,
    path: &Path,
    contents: &[u8],
    mode: AtomicFileMode,
) -> io::Result<()> {
    let temp_path = atomic_temp_path(path);
    let result = (|| {
        let mut file = fs.create_new(&temp_path, mode)?;
        if mode == AtomicFileMode::OwnerOnly {
            fs.set_owner_only_permissions(&temp_path)?;
        }
        file.write_all(contents)?;
        file.flush()?;
        file.sync_all()?;
        drop(file);
        fs.rename(&temp_path, path)?;
        if mode == AtomicFileMode::OwnerOnly {
            fs.set_owner_only_permissions(path)?;
        }
        fs.sync_parent_dir_best_effort(path);
        Ok(())
    })();

    if result.is_err() {
        let _ = fs.remove_file(&temp_path);
    }
    result
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AtomicFileMode {
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
    fn sync_parent_dir_best_effort(&self, path: &Path);
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

        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        if mode == AtomicFileMode::OwnerOnly {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        options.open(path)
    }

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
        fs::rename(from, to)
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        fs::remove_file(path)
    }

    fn sync_parent_dir_best_effort(&self, path: &Path) {
        sync_parent_dir_best_effort(path);
    }
}

fn atomic_temp_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("openasr.tmp");
    let sequence = ATOMIC_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    path.with_file_name(format!(
        ".{file_name}.{}.{}.{}.tmp",
        std::process::id(),
        now,
        sequence
    ))
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

        fn sync_parent_dir_best_effort(&self, _path: &Path) {
            self.state.synced_parent.set(true);
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
        assert!(temp_file_name.starts_with(".voiceprints.json."));
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
}
