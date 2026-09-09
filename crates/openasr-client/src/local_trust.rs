use std::{
    fs::{File, OpenOptions},
    net::SocketAddr,
    path::{Path, PathBuf},
};

use thiserror::Error;

use openasr_core::StrongFileIdentity;

/// Fail-closed proof error for local audio routing.
#[derive(Debug, Error)]
pub enum LocalTrustError {
    #[error("managed daemon endpoint must be a plain HTTP IP-literal origin with an explicit port")]
    InvalidManagedDaemonEndpoint,
    #[error("managed daemon endpoint must use a loopback IP address")]
    NonLoopbackManagedDaemonEndpoint,
    #[error(
        "managed daemon request path must be an absolute path without an authority or fragment"
    )]
    InvalidManagedDaemonPath,
    #[error("selected media path could not be inspected: {0}")]
    InspectMedia(#[source] std::io::Error),
    #[error(
        "selected media must be a regular file, not a directory, symlink, reparse point, or special file"
    )]
    MediaIsNotRegularFile,
    #[error("selected media path could not be resolved: {0}")]
    ResolveMedia(#[source] std::io::Error),
    #[error("selected media could not be opened safely: {0}")]
    OpenMedia(#[source] std::io::Error),
    #[error("selected media handle could not be inspected: {0}")]
    InspectOpenedMedia(#[source] std::io::Error),
    #[error("selected media filesystem does not expose a strong opened-file identity")]
    UnsupportedMediaIdentity,
    #[error("selected media handle could not be cloned: {0}")]
    CloneMedia(#[source] std::io::Error),
    #[error("selected media changed after the grant was issued")]
    MediaIdentityChanged,
}

/// A parsed proof that a managed-daemon origin is exact plain HTTP loopback.
///
/// Parsing accepts only an IP-literal [`SocketAddr`] with an explicit non-zero
/// port. DNS names (including `localhost`), userinfo, query strings, fragments,
/// and non-root paths are rejected so name resolution and URL rewriting cannot
/// silently move local audio off the machine.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ManagedDaemonEndpoint {
    address: SocketAddr,
    origin: String,
}

impl ManagedDaemonEndpoint {
    pub fn parse(value: &str) -> Result<Self, LocalTrustError> {
        let trimmed = value.trim();
        let authority = trimmed
            .strip_prefix("http://")
            .ok_or(LocalTrustError::InvalidManagedDaemonEndpoint)?;
        let authority = authority.strip_suffix('/').unwrap_or(authority);
        if authority.is_empty()
            || authority.contains('/')
            || authority.contains('@')
            || authority.contains('?')
            || authority.contains('#')
        {
            return Err(LocalTrustError::InvalidManagedDaemonEndpoint);
        }
        let address: SocketAddr = authority
            .parse()
            .map_err(|_| LocalTrustError::InvalidManagedDaemonEndpoint)?;
        if address.port() == 0 {
            return Err(LocalTrustError::InvalidManagedDaemonEndpoint);
        }
        if address.ip() != std::net::Ipv4Addr::LOCALHOST
            && address.ip() != std::net::Ipv6Addr::LOCALHOST
        {
            return Err(LocalTrustError::NonLoopbackManagedDaemonEndpoint);
        }
        Ok(Self {
            address,
            origin: format!("http://{address}"),
        })
    }

    pub const fn address(&self) -> SocketAddr {
        self.address
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }

    /// Build a request URL from a code-owned absolute path.
    pub fn request_url(&self, path: &str) -> Result<String, LocalTrustError> {
        if !path.starts_with('/')
            || path.starts_with("//")
            || path.contains('#')
            || path.chars().any(char::is_whitespace)
        {
            return Err(LocalTrustError::InvalidManagedDaemonPath);
        }
        Ok(format!("{}{path}", self.origin))
    }
}

/// An opened-once proof of user-selected local media.
///
/// The proof owns the file handle that was validated. Callers clone that
/// handle for streaming; they never reopen the path. Replacing the directory
/// entry after selection therefore cannot change the bytes an authorized
/// request reads. The canonical path is retained only for native playback
/// scope and display-name projection, never as later read authority.
#[derive(Debug)]
pub struct SelectedMedia {
    file: File,
    identity: StrongFileIdentity,
    canonical_path: PathBuf,
    display_name: String,
    len: u64,
}

impl SelectedMedia {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LocalTrustError> {
        let path = path.as_ref();
        let entry_metadata =
            std::fs::symlink_metadata(path).map_err(LocalTrustError::InspectMedia)?;
        if !is_plain_regular_file(&entry_metadata) {
            return Err(LocalTrustError::MediaIsNotRegularFile);
        }

        let canonical_path = path.canonicalize().map_err(LocalTrustError::ResolveMedia)?;
        let file = open_without_following_final_link(&canonical_path)
            .map_err(LocalTrustError::OpenMedia)?;
        let opened_metadata = file
            .metadata()
            .map_err(LocalTrustError::InspectOpenedMedia)?;
        if !is_plain_regular_file(&opened_metadata) {
            return Err(LocalTrustError::MediaIsNotRegularFile);
        }
        let identity = StrongFileIdentity::of_file(&file, &opened_metadata)
            .ok_or(LocalTrustError::UnsupportedMediaIdentity)?;
        let display_name = canonical_path
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .unwrap_or("Selected file")
            .to_string();

        Ok(Self {
            file,
            identity,
            canonical_path,
            display_name,
            len: opened_metadata.len(),
        })
    }

    pub fn try_clone_file(&self) -> Result<File, LocalTrustError> {
        self.ensure_identity()?;
        self.file.try_clone().map_err(LocalTrustError::CloneMedia)
    }

    pub fn canonical_path(&self) -> &Path {
        &self.canonical_path
    }

    pub const fn identity(&self) -> StrongFileIdentity {
        self.identity
    }

    /// Stable path projection of the already-open descriptor for native
    /// path-only consumers. The registry must keep this proof alive for the
    /// duration of the consumer operation.
    #[cfg(unix)]
    pub fn descriptor_path(&self) -> Result<PathBuf, LocalTrustError> {
        use std::os::fd::AsRawFd;

        self.ensure_identity()?;
        Ok(PathBuf::from(format!("/dev/fd/{}", self.file.as_raw_fd())))
    }

    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    pub const fn len(&self) -> u64 {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn ensure_identity(&self) -> Result<(), LocalTrustError> {
        let metadata = self
            .file
            .metadata()
            .map_err(LocalTrustError::InspectOpenedMedia)?;
        let current = StrongFileIdentity::of_file(&self.file, &metadata)
            .ok_or(LocalTrustError::UnsupportedMediaIdentity)?;
        if current == self.identity {
            Ok(())
        } else {
            Err(LocalTrustError::MediaIdentityChanged)
        }
    }
}

/// Build the only HTTP client permitted to carry local media to a managed
/// daemon. It never consults proxy environment and never follows redirects;
/// the destination proven by [`ManagedDaemonEndpoint`] is the destination
/// that receives the request.
pub fn managed_daemon_http_client() -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
}

fn is_plain_regular_file(metadata: &std::fs::Metadata) -> bool {
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return false;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return false;
        }
    }
    true
}

#[cfg(unix)]
fn open_without_following_final_link(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
}

#[cfg(windows)]
fn open_without_following_final_link(path: &Path) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT;

    OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
}

#[cfg(not(any(unix, windows)))]
fn open_without_following_final_link(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().read(true).open(path)
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read as _, Write as _},
        net::TcpListener,
        thread,
    };

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn managed_daemon_endpoint_accepts_only_exact_loopback_origins() {
        let ipv4 = ManagedDaemonEndpoint::parse("http://127.0.0.1:49152/").unwrap();
        assert_eq!(ipv4.origin(), "http://127.0.0.1:49152");
        assert_eq!(
            ipv4.request_url("/v1/audio/transcriptions").unwrap(),
            "http://127.0.0.1:49152/v1/audio/transcriptions"
        );
        let ipv6 = ManagedDaemonEndpoint::parse("http://[::1]:49152").unwrap();
        assert_eq!(ipv6.address().ip().to_string(), "::1");

        for rejected in [
            "https://127.0.0.1:49152",
            "http://localhost:49152",
            "http://127.0.0.2:49152",
            "http://192.168.1.2:49152",
            "http://user@127.0.0.1:49152",
            "http://127.0.0.1:49152/path",
            "http://127.0.0.1:49152?next=http://evil.example",
            "http://127.0.0.1:0",
        ] {
            assert!(
                ManagedDaemonEndpoint::parse(rejected).is_err(),
                "unexpectedly accepted {rejected}"
            );
        }
    }

    #[test]
    fn selected_media_holds_the_opened_generation_after_path_replacement() {
        let directory = tempdir().unwrap();
        let selected_path = directory.path().join("selected.wav");
        let replacement_path = directory.path().join("replacement.wav");
        std::fs::write(&selected_path, b"selected-generation").unwrap();
        std::fs::write(&replacement_path, b"replacement-generation").unwrap();

        let selected = SelectedMedia::open(&selected_path).unwrap();
        std::fs::rename(&replacement_path, &selected_path).unwrap();

        let mut bytes = Vec::new();
        selected
            .try_clone_file()
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        assert_eq!(bytes, b"selected-generation");
        assert_eq!(selected.display_name(), "selected.wav");
        assert_eq!(selected.len(), b"selected-generation".len() as u64);
    }

    #[test]
    fn selected_media_rejects_directories_and_special_files() {
        let directory = tempdir().unwrap();
        assert!(matches!(
            SelectedMedia::open(directory.path()),
            Err(LocalTrustError::MediaIsNotRegularFile)
        ));
    }

    #[test]
    fn selected_media_rejects_in_place_mutation_before_cloning() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("selected.wav");
        std::fs::write(&path, b"selected-generation").unwrap();
        let selected = SelectedMedia::open(&path).unwrap();
        std::fs::write(&path, b"replacement-generation").unwrap();
        assert!(matches!(
            selected.try_clone_file(),
            Err(LocalTrustError::MediaIdentityChanged)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn selected_media_rejects_symlinks() {
        use std::os::unix::fs::symlink;

        let directory = tempdir().unwrap();
        let target = directory.path().join("target.wav");
        let link = directory.path().join("link.wav");
        std::fs::write(&target, b"audio").unwrap();
        symlink(&target, &link).unwrap();
        assert!(matches!(
            SelectedMedia::open(&link),
            Err(LocalTrustError::MediaIsNotRegularFile)
        ));
    }

    #[tokio::test]
    async fn managed_daemon_client_never_follows_redirects() {
        let production = include_str!("local_trust.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(production.contains(".no_proxy()"));
        assert!(production.contains(".redirect(reqwest::redirect::Policy::none())"));

        let target = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        target.set_nonblocking(true).unwrap();
        let target_address = target.local_addr().unwrap();
        let redirect = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let redirect_address = redirect.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = redirect.accept().unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).unwrap();
            write!(
                stream,
                "HTTP/1.1 302 Found\r\nLocation: http://{target_address}/capture\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            .unwrap();
        });

        let response = managed_daemon_http_client()
            .unwrap()
            .get(format!("http://{redirect_address}/source"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::FOUND);
        server.join().unwrap();
        assert!(matches!(
            target.accept(),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock
        ));
    }
}
