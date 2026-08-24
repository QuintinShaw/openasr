use std::fs::File;

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
#[cfg(windows)]
use std::os::windows::io::AsRawHandle;

use serde::{Deserialize, Serialize};

#[cfg(windows)]
use windows_sys::Win32::Storage::FileSystem::{
    BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
};

/// Strong cross-platform identity for an already-opened regular file.
///
/// Identity is always derived from the held handle (`fstat` on Unix and
/// `GetFileInformationByHandle` on Windows), never from a second path lookup.
/// Unsupported native targets return `None`; callers must fail closed rather
/// than substitute a path or a weak `(length, seconds-mtime)` key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StrongFileIdentity {
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
    #[cfg(windows)]
    volume_serial_number: u32,
    #[cfg(windows)]
    file_index: u64,
    len: u64,
    mtime_secs: u64,
    mtime_nanos: u32,
}

impl StrongFileIdentity {
    #[cfg(test)]
    pub(crate) const fn test_fixture(seed: u64) -> Self {
        Self {
            #[cfg(unix)]
            dev: 1,
            #[cfg(unix)]
            ino: seed,
            #[cfg(windows)]
            volume_serial_number: 1,
            #[cfg(windows)]
            file_index: seed,
            len: 4096,
            mtime_secs: 1,
            mtime_nanos: 1,
        }
    }

    /// Returns `None` when the platform cannot provide every strong identity
    /// field or the modification time cannot be represented after Unix epoch.
    pub fn of_file(file: &File, metadata: &std::fs::Metadata) -> Option<Self> {
        #[cfg(not(any(unix, windows)))]
        {
            let _ = (file, metadata);
            return None;
        }
        #[cfg(unix)]
        let _ = file;
        #[cfg(windows)]
        let handle_identity = {
            let mut information = BY_HANDLE_FILE_INFORMATION::default();
            // SAFETY: `file` owns a live Windows file handle for this call and
            // `information` is writable storage of the required structure.
            let succeeded =
                unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut information) };
            if succeeded == 0 {
                return None;
            }
            (
                information.dwVolumeSerialNumber,
                (u64::from(information.nFileIndexHigh) << 32)
                    | u64::from(information.nFileIndexLow),
            )
        };
        let modified = metadata.modified().ok()?;
        let since_epoch = modified.duration_since(std::time::UNIX_EPOCH).ok()?;
        Some(Self {
            #[cfg(unix)]
            dev: metadata.dev(),
            #[cfg(unix)]
            ino: metadata.ino(),
            #[cfg(windows)]
            volume_serial_number: handle_identity.0,
            #[cfg(windows)]
            file_index: handle_identity.1,
            len: metadata.len(),
            mtime_secs: since_epoch.as_secs(),
            mtime_nanos: since_epoch.subsec_nanos(),
        })
    }
}
