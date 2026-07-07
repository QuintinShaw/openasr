//! Local server API-key store.
//!
//! `openasr serve` trusts loopback (127.0.0.1) callers by default (no key
//! required) -- this store backs the *opt-in* escape hatch for callers (coding
//! agents, scripts) that want an explicit bearer credential even on loopback.
//! Only a SHA-256 hash of each key is ever persisted; the plaintext token is
//! generated once by `ApiKeyStore::create` and must be shown to the caller
//! immediately -- it cannot be recovered from the store afterward.
//!
//! The hash algorithm here (SHA-256 over the raw token bytes, lowercase hex)
//! must match `openasr_server::ServerAuth`'s bearer-token hashing, since the
//! server compares an incoming `Authorization: Bearer <token>` header against
//! the hashes persisted here. `openasr-server`'s test suite asserts the two
//! stay in lockstep.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::realtime::events::realtime_timestamp_now;

/// Version of the on-disk API-key store.
pub const API_KEY_STORE_VERSION: u32 = 1;
/// Plaintext token prefix, so a leaked/pasted key is recognizable at a glance.
pub const API_KEY_TOKEN_PREFIX: &str = "oasr_sk_";
/// Public key-record id prefix (distinct from the secret token itself).
pub const API_KEY_ID_PREFIX: &str = "key_";
/// Random key material length in bytes (before hex-encoding).
const API_KEY_TOKEN_RANDOM_BYTES: usize = 32;
/// Env var override for the store path, mirroring
/// `diarize::enrollment::VOICEPRINT_STORE_ENV`, so tests do not touch a real
/// `OPENASR_HOME`.
pub const API_KEY_STORE_ENV: &str = "OPENASR_API_KEYS_PATH";

static API_KEY_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Error)]
pub enum ApiKeyError {
    #[error("unsupported API key store version {found}; expected {expected}")]
    UnsupportedVersion { found: u32, expected: u32 },
    #[error("could not read API key store {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("could not parse API key store {path}: {source}")]
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("could not create API key store directory {path}: {source}")]
    CreateDir {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("could not write API key store {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("could not serialize API key store: {0}")]
    Serialize(serde_json::Error),
    #[error("could not generate random key material: {0}")]
    Random(getrandom::Error),
    #[error("API key not found: {0}")]
    NotFound(String),
}

/// One issued API key. Only the hash is load-bearing for auth; `token_preview`
/// is a display-only fragment (prefix + first 4 hex chars) so a user can tell
/// keys apart in `openasr apikey list` without re-displaying the secret.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApiKeyRecord {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    pub created_at: String,
    pub token_hash: String,
    pub token_preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApiKeyStore {
    pub version: u32,
    #[serde(default)]
    pub keys: Vec<ApiKeyRecord>,
}

impl Default for ApiKeyStore {
    fn default() -> Self {
        Self {
            version: API_KEY_STORE_VERSION,
            keys: Vec::new(),
        }
    }
}

/// Canonical store location: `OPENASR_API_KEYS_PATH` when set, otherwise
/// `openasr_home()/apikeys.json`.
pub fn api_key_store_path() -> Option<PathBuf> {
    std::env::var(API_KEY_STORE_ENV)
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            crate::openasr_home()
                .ok()
                .map(|home| home.join("apikeys.json"))
        })
}

/// SHA-256 hex digest of a raw token. Must match
/// `openasr_server`'s internal `bearer_token_hash`.
pub fn hash_api_key_token(token: &str) -> String {
    let digest = Sha256::digest(token.trim().as_bytes());
    hex_encode(&digest)
}

impl ApiKeyStore {
    pub fn load(path: &Path) -> Result<Self, ApiKeyError> {
        if !path.is_file() {
            return Ok(Self::default());
        }
        let bytes = fs::read(path).map_err(|source| ApiKeyError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let store: Self = serde_json::from_slice(&bytes).map_err(|source| ApiKeyError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
        store.validate_version()?;
        Ok(store)
    }

    pub fn save(&self, path: &Path) -> Result<(), ApiKeyError> {
        self.validate_version()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| ApiKeyError::CreateDir {
                path: parent.to_path_buf(),
                source,
            })?;
            set_owner_only_dir_permissions(parent);
        }
        let json = serde_json::to_vec_pretty(self).map_err(ApiKeyError::Serialize)?;
        crate::atomic_file::write_owner_only_file_atomically(path, &json).map_err(|source| {
            ApiKeyError::Write {
                path: path.to_path_buf(),
                source,
            }
        })
    }

    /// Generates a new key, appends it (hash-only) to the store, and returns
    /// the plaintext token alongside its record. Callers must persist the
    /// store (`save`) and display the plaintext exactly once -- it is not
    /// recoverable afterward.
    pub fn create(&mut self, name: Option<String>) -> Result<(String, ApiKeyRecord), ApiKeyError> {
        let token = format!(
            "{API_KEY_TOKEN_PREFIX}{}",
            random_hex(API_KEY_TOKEN_RANDOM_BYTES).map_err(ApiKeyError::Random)?
        );
        let preview_len = API_KEY_TOKEN_PREFIX.len() + 4;
        let token_preview = token.chars().take(preview_len).collect::<String>() + "...";
        let record = ApiKeyRecord {
            id: generate_key_id(),
            name: name
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty()),
            created_at: realtime_timestamp_now(),
            token_hash: hash_api_key_token(&token),
            token_preview,
        };
        self.keys.push(record.clone());
        Ok((token, record))
    }

    /// Removes a key by id. Returns `true` if a key was found and removed.
    pub fn revoke(&mut self, id: &str) -> bool {
        let before = self.keys.len();
        self.keys.retain(|key| key.id != id);
        self.keys.len() != before
    }

    /// Hashes of all currently-active keys, for wiring into
    /// `openasr_server::ServerAuth::from_token_hashes`.
    pub fn active_token_hashes(&self) -> Vec<String> {
        self.keys.iter().map(|key| key.token_hash.clone()).collect()
    }

    fn validate_version(&self) -> Result<(), ApiKeyError> {
        if self.version == API_KEY_STORE_VERSION {
            Ok(())
        } else {
            Err(ApiKeyError::UnsupportedVersion {
                found: self.version,
                expected: API_KEY_STORE_VERSION,
            })
        }
    }
}

fn generate_key_id() -> String {
    let counter = API_KEY_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut hasher = Sha256::new();
    hasher.update(realtime_timestamp_now().as_bytes());
    hasher.update(counter.to_le_bytes());
    hasher.update(std::process::id().to_le_bytes());
    let digest = format!("{:x}", hasher.finalize());
    format!("{API_KEY_ID_PREFIX}{}", &digest[..16])
}

fn random_hex(byte_count: usize) -> Result<String, getrandom::Error> {
    let mut bytes = vec![0u8; byte_count];
    getrandom::fill(&mut bytes)?;
    Ok(hex_encode(&bytes))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn set_owner_only_dir_permissions(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o700));
    }
    #[cfg(not(unix))]
    let _ = path;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_returns_prefixed_token_and_stores_only_hash() {
        let mut store = ApiKeyStore::default();
        let (token, record) = store.create(Some("agent-laptop".to_string())).unwrap();
        assert!(token.starts_with(API_KEY_TOKEN_PREFIX));
        assert_eq!(token.len(), API_KEY_TOKEN_PREFIX.len() + 64);
        assert_eq!(record.token_hash, hash_api_key_token(&token));
        assert_ne!(record.token_hash, token, "must never store the plaintext");
        assert_eq!(record.name.as_deref(), Some("agent-laptop"));
        assert_eq!(store.keys.len(), 1);
        assert_eq!(store.active_token_hashes(), vec![record.token_hash]);
    }

    #[test]
    fn create_trims_and_drops_empty_name() {
        let mut store = ApiKeyStore::default();
        let (_, record) = store.create(Some("  ".to_string())).unwrap();
        assert_eq!(record.name, None);
    }

    #[test]
    fn revoke_removes_matching_key_only() {
        let mut store = ApiKeyStore::default();
        let (_, first) = store.create(None).unwrap();
        let (_, second) = store.create(None).unwrap();

        assert!(store.revoke(&first.id));
        assert_eq!(store.keys.len(), 1);
        assert_eq!(store.keys[0].id, second.id);
        assert!(!store.revoke("key_doesnotexist"));
    }

    #[test]
    fn round_trips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("apikeys.json");
        let mut store = ApiKeyStore::default();
        store.create(Some("ci".to_string())).unwrap();
        store.save(&path).unwrap();

        let loaded = ApiKeyStore::load(&path).unwrap();
        assert_eq!(loaded, store);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
    }

    #[test]
    fn missing_store_loads_as_empty_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.json");
        let store = ApiKeyStore::load(&path).unwrap();
        assert_eq!(store, ApiKeyStore::default());
    }

    #[test]
    fn rejects_unsupported_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("apikeys.json");
        fs::write(&path, br#"{"version":99,"keys":[]}"#).unwrap();
        let error = ApiKeyStore::load(&path).unwrap_err();
        assert!(matches!(error, ApiKeyError::UnsupportedVersion { .. }));
    }

    #[test]
    fn two_created_keys_have_distinct_ids_and_hashes() {
        let mut store = ApiKeyStore::default();
        let (token_a, record_a) = store.create(None).unwrap();
        let (token_b, record_b) = store.create(None).unwrap();
        assert_ne!(token_a, token_b);
        assert_ne!(record_a.id, record_b.id);
        assert_ne!(record_a.token_hash, record_b.token_hash);
    }
}
