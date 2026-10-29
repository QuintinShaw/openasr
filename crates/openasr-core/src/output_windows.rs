use std::{
    ffi::OsString,
    fs,
    io::{self, Write},
    os::windows::{
        ffi::OsStrExt, ffi::OsStringExt, fs::MetadataExt, fs::OpenOptionsExt, io::AsRawHandle,
        io::FromRawHandle,
    },
    path::{Component, Path, PathBuf},
};

use windows_sys::Win32::Storage::FileSystem::{
    BY_HANDLE_FILE_INFORMATION, DELETE, FILE_ATTRIBUTE_REPARSE_POINT, FILE_DISPOSITION_INFO,
    FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_GENERIC_READ,
    FILE_GENERIC_WRITE, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, FileDispositionInfo,
    GetFileInformationByHandle, GetFinalPathNameByHandleW, SetFileInformationByHandle,
    VOLUME_NAME_GUID,
};
use windows_sys::Win32::{Foundation::HANDLE, System::IO::IO_STATUS_BLOCK};

use windows_sys::{
    Wdk::{
        Foundation::OBJECT_ATTRIBUTES,
        Storage::FileSystem::{
            FILE_CREATE, FILE_NON_DIRECTORY_FILE, FILE_OPEN, FILE_OPEN_REPARSE_POINT,
            FILE_RENAME_INFORMATION, FILE_RENAME_INFORMATION_0, FILE_SYNCHRONOUS_IO_NONALERT,
            FILE_WRITE_THROUGH, FileRenameInformation, NtCreateFile, NtSetInformationFile,
        },
    },
    Win32::{
        Foundation::{
            OBJ_CASE_INSENSITIVE, OBJ_DONT_REPARSE, RtlNtStatusToDosError, UNICODE_STRING,
        },
        Globalization::CompareStringOrdinal,
    },
};

fn relative_file(parent: &fs::File, name: &std::ffi::OsStr, create: bool) -> io::Result<fs::File> {
    let mut name = name.encode_wide().collect::<Vec<_>>();
    if name.is_empty()
        || name.iter().any(|value| matches!(*value, 0 | 47 | 58 | 92))
        || name == [46]
        || name == [46, 46]
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "relative output must be one ordinary basename",
        ));
    }
    let length = u16::try_from(
        name.len()
            .checked_mul(2)
            .ok_or_else(|| io::Error::other("output name too long"))?,
    )
    .map_err(|_| io::Error::other("output name too long"))?;
    name.push(0);
    let name = UNICODE_STRING {
        Length: length,
        MaximumLength: length
            .checked_add(2)
            .ok_or_else(|| io::Error::other("output name too long"))?,
        Buffer: name.as_mut_ptr(),
    };
    let attributes = OBJECT_ATTRIBUTES {
        Length: std::mem::size_of::<OBJECT_ATTRIBUTES>() as u32,
        RootDirectory: parent.as_raw_handle(),
        ObjectName: &name,
        Attributes: OBJ_CASE_INSENSITIVE | OBJ_DONT_REPARSE,
        SecurityDescriptor: std::ptr::null(),
        SecurityQualityOfService: std::ptr::null(),
    };
    let mut handle: HANDLE = std::ptr::null_mut();
    let mut status = IO_STATUS_BLOCK::default();
    let access = if create {
        FILE_GENERIC_READ | FILE_GENERIC_WRITE | DELETE
    } else {
        windows_sys::Win32::Storage::FileSystem::FILE_READ_ATTRIBUTES
            | windows_sys::Win32::Storage::FileSystem::SYNCHRONIZE
    };
    let options = FILE_SYNCHRONOUS_IO_NONALERT
        | FILE_OPEN_REPARSE_POINT
        | if create {
            FILE_NON_DIRECTORY_FILE | FILE_WRITE_THROUGH
        } else {
            0
        };
    // SAFETY: all NT buffers and the held root remain live. OBJ_DONT_REPARSE
    // refuses reparse traversal; a single basename cannot escape the root.
    // Synchronous IO prevents a pending operation from outliving the buffers.
    let result = unsafe {
        NtCreateFile(
            &mut handle,
            access,
            &attributes,
            &mut status,
            std::ptr::null(),
            0,
            FILE_SHARE_READ | FILE_SHARE_DELETE | if create { 0 } else { FILE_SHARE_WRITE },
            if create { FILE_CREATE } else { FILE_OPEN },
            options,
            std::ptr::null(),
            0,
        )
    };
    if result < 0 {
        // SAFETY: pure scalar status translation.
        return Err(io::Error::from_raw_os_error(
            unsafe { RtlNtStatusToDosError(result) } as i32,
        ));
    }
    // SAFETY: successful synchronous NtCreateFile transfers this unique handle.
    Ok(unsafe { fs::File::from_raw_handle(handle) })
}

use super::{OutputWriteError, ResolvedOutputTarget, output_path};

pub(super) fn same_destination(
    first: &ResolvedOutputTarget,
    second: &ResolvedOutputTarget,
) -> Result<bool, OutputWriteError> {
    let compare = || -> io::Result<bool> {
        if identity(first._directories.last().expect("resolved parent"))?
            != identity(second._directories.last().expect("resolved parent"))?
        {
            return Ok(false);
        }
        let a = first
            .path()
            .file_name()
            .expect("resolved basename")
            .encode_wide()
            .collect::<Vec<_>>();
        let b = second
            .path()
            .file_name()
            .expect("resolved basename")
            .encode_wide()
            .collect::<Vec<_>>();
        // Windows ordinal case folding is conservative even in a directory
        // configured for case sensitivity: ambiguous sibling names are refused.
        // SAFETY: both length-delimited UTF-16 buffers remain live.
        let result = unsafe {
            CompareStringOrdinal(a.as_ptr(), a.len() as i32, b.as_ptr(), b.len() as i32, 1)
        };
        if result == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(result == 2)
        }
    };
    compare().map_err(|source| OutputWriteError::ResolveSymlink {
        path: first.path.clone(),
        source,
    })
}

fn open_directory(path: &Path) -> io::Result<fs::File> {
    let directory = fs::OpenOptions::new()
        .access_mode(FILE_GENERIC_READ)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)?;
    let metadata = directory.metadata()?;
    if !metadata.is_dir() || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "pinned ancestor is not an ordinary directory",
        ));
    }
    Ok(directory)
}

fn identity(file: &fs::File) -> io::Result<(u32, u32, u32)> {
    let mut information = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: file owns a live handle and information is writable for the call.
    if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut information) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok((
        information.dwVolumeSerialNumber,
        information.nFileIndexHigh,
        information.nFileIndexLow,
    ))
}

fn volume_path(file: &fs::File) -> io::Result<PathBuf> {
    // A canonical DOS drive path still depends on mutable drive mappings.
    let mut buffer = vec![0u16; 32768];
    // SAFETY: the handle and full output buffer remain live for the call.
    let length = unsafe {
        GetFinalPathNameByHandleW(
            file.as_raw_handle(),
            buffer.as_mut_ptr(),
            buffer.len() as u32,
            VOLUME_NAME_GUID,
        )
    };
    if length == 0 {
        return Err(io::Error::last_os_error());
    }
    if length as usize >= buffer.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "volume path exceeds the Windows path bound",
        ));
    }
    let path = PathBuf::from(OsString::from_wide(&buffer[..length as usize]));
    if !path
        .as_os_str()
        .to_string_lossy()
        .starts_with(r"\\?\Volume{")
    {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "pinned output requires a local volume GUID path",
        ));
    }
    Ok(path)
}

pub(super) fn resolve(path: PathBuf) -> Result<ResolvedOutputTarget, OutputWriteError> {
    let resolve = || -> io::Result<ResolvedOutputTarget> {
        let initial = open_directory(output_path::output_parent(&path))?;
        let parent = volume_path(&initial)?;
        let mut directories = Vec::new();
        let mut prefix = PathBuf::new();
        for component in parent.components() {
            prefix.push(component.as_os_str());
            if matches!(component, Component::Prefix(_)) {
                continue;
            }
            // Deny ancestor renames. Mutations use the held parent directly
            // with OBJ_DONT_REPARSE; directory locks do not lock child names.
            directories.push(open_directory(&prefix)?);
        }
        if directories.last().map(identity).transpose()? != Some(identity(&initial)?) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "output parent changed while pinning",
            ));
        }
        let basename = path
            .file_name()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "output has no basename"))?;
        directories.push(initial);
        Ok(ResolvedOutputTarget {
            path: parent.join(basename),
            _directories: directories,
        })
    };
    resolve().map_err(|source| OutputWriteError::ResolveSymlink { path, source })
}

fn rename_open_file(
    file: &fs::File,
    directory: &fs::File,
    basename: &std::ffi::OsStr,
    replace: bool,
) -> io::Result<()> {
    let name = basename.encode_wide().collect::<Vec<_>>();
    let offset = std::mem::offset_of!(FILE_RENAME_INFORMATION, FileName);
    let length = offset
        .checked_add(
            (name.len() + 1)
                .checked_mul(2)
                .ok_or_else(|| io::Error::other("rename name too long"))?,
        )
        .ok_or_else(|| io::Error::other("rename buffer too long"))?;
    let length = u32::try_from(length.max(std::mem::size_of::<FILE_RENAME_INFORMATION>()))
        .map_err(|_| io::Error::other("rename buffer too long"))?;
    let mut storage = vec![
        0usize;
        (length as usize)
            .max(std::mem::size_of::<FILE_RENAME_INFORMATION>())
            .div_ceil(std::mem::size_of::<usize>())
    ];
    let info = storage.as_mut_ptr().cast::<FILE_RENAME_INFORMATION>();
    // SAFETY: aligned storage covers the fixed header and UTF-16 name. The
    // destination is relative to the held parent. Publication names the held
    // file, never its mutable staging name. Native FileRenameInformation is
    // used because the Win32 wrapper rejects relative roots on older Windows.
    let mut status = IO_STATUS_BLOCK::default();
    let result = unsafe {
        info.write(FILE_RENAME_INFORMATION {
            Anonymous: FILE_RENAME_INFORMATION_0 {
                // Initialize the entire union, including bytes outside the
                // BOOLEAN spelling used by FileRenameInfo.
                Flags: u32::from(replace),
            },
            RootDirectory: directory.as_raw_handle(),
            FileNameLength: (name.len() * 2) as u32,
            FileName: [0],
        });
        std::ptr::copy_nonoverlapping(
            name.as_ptr(),
            std::ptr::addr_of_mut!((*info).FileName).cast::<u16>(),
            name.len(),
        );
        NtSetInformationFile(
            file.as_raw_handle(),
            &mut status,
            info.cast(),
            length,
            FileRenameInformation,
        )
    };
    if result < 0 {
        // SAFETY: translating NTSTATUS is a pure scalar operation.
        let error = io::Error::from_raw_os_error(unsafe { RtlNtStatusToDosError(result) } as i32);
        return Err(io::Error::new(
            error.kind(),
            format!("native rename failed (NTSTATUS {result:#x}): {error}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
fn open_staging_file(path: &Path) -> io::Result<fs::File> {
    fs::OpenOptions::new()
        .read(true)
        .write(true)
        .access_mode(FILE_GENERIC_READ | FILE_GENERIC_WRITE | DELETE)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_DELETE)
        .create_new(true)
        .custom_flags(
            FILE_FLAG_OPEN_REPARSE_POINT
                | windows_sys::Win32::Storage::FileSystem::FILE_FLAG_WRITE_THROUGH,
        )
        .open(path)
}

fn discard_staging(mut temp: tempfile::NamedTempFile) -> Option<String> {
    temp.disable_cleanup(true);
    mark_staging_for_deletion(temp.as_file())
        .err()
        .map(|error| format!("could not mark owned staging file for deletion: {error}"))
}

fn mark_staging_for_deletion(file: &fs::File) -> io::Result<()> {
    let disposition = FILE_DISPOSITION_INFO { DeleteFile: true };
    // SAFETY: the live file owns DELETE access. Mark that exact file for
    // deletion before closing it; never delete by a reusable staging pathname.
    if unsafe {
        SetFileInformationByHandle(
            file.as_raw_handle(),
            FileDispositionInfo,
            (&disposition as *const FILE_DISPOSITION_INFO).cast(),
            std::mem::size_of::<FILE_DISPOSITION_INFO>() as u32,
        )
    } == 0
    {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

pub(super) fn write(
    target: &ResolvedOutputTarget,
    content: &str,
    create_new: bool,
) -> Result<(), OutputWriteError> {
    let path = target.path();
    let parent = target._directories.last().expect("resolved Windows parent");
    let metadata = parent
        .metadata()
        .map_err(|source| OutputWriteError::ResolveSymlink {
            path: path.to_path_buf(),
            source,
        })?;
    if !metadata.is_dir() || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(OutputWriteError::ResolveSymlink {
            path: path.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "held output parent became a reparse point",
            ),
        });
    }
    let basename = path.file_name().expect("resolved basename");
    match relative_file(parent, basename, false).and_then(|file| file.metadata()) {
        Ok(metadata)
            if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
                || !metadata.is_file()
                || metadata.permissions().readonly() =>
        {
            return Err(OutputWriteError::ExistingOutputNotWritable {
                path: path.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "pinned output is read-only, reparse, or non-regular",
                ),
            });
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(OutputWriteError::ExistingOutputNotWritable {
                path: path.to_path_buf(),
                source,
            });
        }
    }
    let mut temp = tempfile::Builder::new()
        .prefix(".openasr-")
        .disable_cleanup(true)
        .make_in(output_path::output_parent(path), |path| {
            relative_file(
                parent,
                path.file_name().expect("generated staging basename"),
                true,
            )
        })
        .map_err(|source| OutputWriteError::CreateTemp {
            path: path.to_path_buf(),
            source,
        })?;
    if let Err(source) = temp
        .write_all(content.as_bytes())
        .and_then(|_| temp.as_file().sync_all())
    {
        let cleanup_warning = discard_staging(temp);
        return Err(OutputWriteError::Write {
            path: path.to_path_buf(),
            source,
            cleanup_warning,
        });
    }
    if let Err(source) = rename_open_file(
        temp.as_file(),
        parent,
        path.file_name().expect("resolved basename"),
        !create_new,
    ) {
        let cleanup_warning = discard_staging(temp);
        return Err(OutputWriteError::Persist {
            path: path.to_path_buf(),
            source,
            cleanup_warning,
        });
    }
    // Keep the published file if the final flush fails, and never delete a
    // new file another producer might create at the old staging name.
    temp.disable_cleanup(true);
    temp.as_file()
        .sync_all()
        .map_err(|source| OutputWriteError::PublishedSync {
            path: path.to_path_buf(),
            source,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::resolve_output_target_handle;

    #[test]
    fn pinned_windows_compares_case_aliases_before_any_publication() {
        let temp = tempfile::tempdir().unwrap();
        for (a, b) in [
            ("Receipt.json", "receipt.JSON"),
            ("École.json", "éCOLE.JSON"),
        ] {
            let a = resolve_output_target_handle(temp.path().join(a)).unwrap();
            let b = resolve_output_target_handle(temp.path().join(b)).unwrap();
            assert!(same_destination(&a, &b).unwrap());
        }
        let a = resolve_output_target_handle(temp.path().join("receipt.json")).unwrap();
        let b = resolve_output_target_handle(temp.path().join("trace.jsonl")).unwrap();
        assert!(!same_destination(&a, &b).unwrap());
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);
    }

    #[test]
    fn pinned_windows_rejects_a_parent_changed_to_a_junction() {
        use windows_sys::Win32::System::IO::DeviceIoControl;
        let temp = tempfile::tempdir().unwrap();
        let parent = temp.path().join("parent");
        let outside = temp.path().join("outside");
        fs::create_dir(&parent).unwrap();
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("sentinel"), "outside").unwrap();
        let target = resolve_output_target_handle(parent.join("receipt.json")).unwrap();
        let canonical = fs::canonicalize(&outside).unwrap();
        let canonical = canonical.as_os_str().encode_wide().collect::<Vec<_>>();
        assert_eq!(&canonical[..4], &[92, 92, 63, 92]);
        let mut substitute = vec![92u16, 63, 63, 92];
        substitute.extend_from_slice(&canonical[4..]);
        let print = outside.as_os_str().encode_wide().collect::<Vec<_>>();
        let data_length = 8 + (substitute.len() + print.len() + 2) * 2;
        let mut data = Vec::new();
        data.extend_from_slice(&0xa0000003u32.to_le_bytes()); // mount-point reparse tag
        data.extend_from_slice(&(data_length as u16).to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes());
        for value in [
            0,
            substitute.len() as u16 * 2,
            (substitute.len() as u16 + 1) * 2,
            print.len() as u16 * 2,
        ] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        for value in substitute
            .into_iter()
            .chain(Some(0))
            .chain(print)
            .chain(Some(0))
        {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let mutation = fs::OpenOptions::new()
            .write(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
            .open(&parent)
            .unwrap();
        let mut returned = 0u32;
        // SAFETY: this test sends a bounded mount-point buffer only to its own
        // empty directory. The target and sentinel belong to the same temp root.
        let changed = unsafe {
            DeviceIoControl(
                mutation.as_raw_handle(),
                0x000900a4,
                data.as_ptr().cast(),
                data.len() as u32,
                std::ptr::null_mut(),
                0,
                &mut returned,
                std::ptr::null_mut(),
            )
        };
        assert_ne!(changed, 0, "{}", io::Error::last_os_error());
        assert!(
            target
                .create_new_text_and_sync_parent("must not escape")
                .is_err()
        );
        // Exercise the low-level operation after reparse mutation as well as
        // the preflight rejection above. A check/operation race must not escape
        // the held directory object.
        let held = target._directories.last().unwrap();
        if let Ok(mut file) = relative_file(held, std::ffi::OsStr::new("racy-stage"), true) {
            file.write_all(b"held parent only").unwrap();
            let _ = rename_open_file(&file, held, std::ffi::OsStr::new("racy-receipt"), false);
            assert!(!outside.join("racy-stage").exists());
            assert!(!outside.join("racy-receipt").exists());
            mark_staging_for_deletion(&file).unwrap();
        }
        assert_eq!(
            fs::read_to_string(outside.join("sentinel")).unwrap(),
            "outside"
        );
        assert_eq!(fs::read_dir(&outside).unwrap().count(), 1);
    }

    #[test]
    fn pinned_windows_unicode_create_new_and_atomic_replace_preserve_other_files() {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("验收-输出");
        fs::create_dir(&directory).unwrap();
        let destination = directory.join("字幕-法语.txt");
        let outside = temp.path().join("outside.txt");
        fs::write(&outside, "outside").unwrap();
        let target = resolve_output_target_handle(&destination).unwrap();
        assert!(
            target
                .path()
                .as_os_str()
                .to_string_lossy()
                .starts_with(r"\\?\Volume{")
        );
        target.create_new_text_and_sync_parent("été-中文").unwrap();
        assert!(target.create_new_text_and_sync_parent("overwrite").is_err());
        assert_eq!(fs::read_to_string(&destination).unwrap(), "été-中文");
        target.atomic_write_text("changed").unwrap();
        assert_eq!(fs::read_to_string(&destination).unwrap(), "changed");
        assert_eq!(fs::read_to_string(&outside).unwrap(), "outside");
        assert_eq!(
            fs::read_dir(&directory).unwrap().count(),
            1,
            "staging files leaked after a failed publish"
        );
        for basename in ["a", "字"] {
            let short = resolve_output_target_handle(directory.join(basename)).unwrap();
            short
                .create_new_text_and_sync_parent("short Unicode basename")
                .unwrap();
            assert_eq!(
                fs::read_to_string(directory.join(basename)).unwrap(),
                "short Unicode basename"
            );
        }
    }

    #[test]
    fn pinned_windows_rejects_ancestor_rename_until_targets_drop() {
        let temp = tempfile::tempdir().unwrap();
        let ancestor = temp.path().join("ancestor");
        let parent = ancestor.join("parent");
        fs::create_dir_all(&parent).unwrap();
        let target = resolve_output_target_handle(parent.join("receipt.json")).unwrap();
        assert!(fs::rename(&parent, ancestor.join("moved-parent")).is_err());
        assert!(fs::rename(&ancestor, temp.path().join("moved-ancestor")).is_err());
        target.create_new_text_and_sync_parent("receipt").unwrap();
        drop(target);
        fs::rename(&ancestor, temp.path().join("moved-ancestor")).unwrap();
    }

    #[test]
    fn pinned_windows_replaces_final_hardlink_without_changing_its_source() {
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("receipt.json");
        let outside = temp.path().join("outside.txt");
        fs::write(&outside, "outside").unwrap();
        // Tiny text fixtures only; model files are never linked.
        fs::hard_link(&outside, &destination).unwrap();
        let target = resolve_output_target_handle(&destination).unwrap();
        assert!(
            target
                .create_new_text_and_sync_parent("must not overwrite")
                .is_err()
        );
        target.atomic_write_text("receipt").unwrap();
        assert_eq!(fs::read_to_string(&outside).unwrap(), "outside");
        assert_eq!(fs::read_to_string(&destination).unwrap(), "receipt");
    }

    #[test]
    fn pinned_windows_rejects_readonly_and_directory_leaves() {
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("readonly.txt");
        fs::write(&destination, "old").unwrap();
        let permissions = fs::metadata(&destination).unwrap().permissions();
        let mut readonly = permissions.clone();
        readonly.set_readonly(true);
        fs::set_permissions(&destination, readonly).unwrap();
        let target = resolve_output_target_handle(&destination).unwrap();
        assert!(target.atomic_write_text("new").is_err());
        assert_eq!(fs::read_to_string(&destination).unwrap(), "old");
        drop(target);
        fs::set_permissions(&destination, permissions).unwrap();
        let directory = temp.path().join("directory");
        fs::create_dir(&directory).unwrap();
        let target = resolve_output_target_handle(&directory).unwrap();
        assert!(target.atomic_write_text("new").is_err());
    }

    #[test]
    fn pinned_windows_rejects_a_final_symlink_introduced_after_resolution() {
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("receipt.json");
        let outside = temp.path().join("outside.txt");
        fs::write(&outside, "outside").unwrap();
        let target = resolve_output_target_handle(&destination).unwrap();
        std::os::windows::fs::symlink_file(&outside, &destination).unwrap();
        assert!(target.atomic_write_text("must not follow").is_err());
        assert!(
            target
                .create_new_text_and_sync_parent("must not follow")
                .is_err()
        );
        assert_eq!(fs::read_to_string(&outside).unwrap(), "outside");
    }

    #[test]
    fn pinned_windows_publishes_the_held_staging_file_not_a_replacement_path() {
        let temp = tempfile::tempdir().unwrap();
        let target = resolve_output_target_handle(temp.path().join("receipt.json")).unwrap();
        let source = temp.path().join("staged");
        let mut file = open_staging_file(&source).unwrap();
        file.write_all(b"original staged bytes").unwrap();
        file.sync_all().unwrap();
        // Namespace mutation is allowed, but cannot change which held file is
        // published. Another producer may reuse the old staging name safely.
        let moved = temp.path().join("moved-staging");
        fs::rename(&source, &moved).unwrap();
        fs::write(&source, "replacement bytes").unwrap();
        rename_open_file(
            &file,
            target._directories.last().unwrap(),
            target.path().file_name().unwrap(),
            false,
        )
        .unwrap();
        file.sync_all().unwrap();
        assert!(!moved.exists());
        assert_eq!(fs::read_to_string(&source).unwrap(), "replacement bytes");
        assert_eq!(
            fs::read_to_string(temp.path().join("receipt.json")).unwrap(),
            "original staged bytes"
        );
    }
}
