use std::{
    fs, io,
    path::{Path, PathBuf},
};

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::atomic_file;

pub const CATALOG_SIGNATURE_SCHEMA_VERSION: u32 = 1;
pub const CATALOG_SIGNATURE_FILE_NAME: &str = "catalog.signature.json";
pub const CATALOG_EPOCH_FILE_NAME: &str = "catalog.epoch";
pub const CATALOG_SIGNATURE_ALGORITHM: &str = "ed25519";
pub const CATALOG_SIGNATURE_KEY_ID: &str = "openasr-catalog-v1";

const CATALOG_SIGNATURE_DOMAIN: &str = "openasr.catalog_manifest.v1";
const OPENASR_CATALOG_V1_PUBLIC_KEY_HEX: &str =
    "92331f1048a70b70fb00818f263b4f532ff536f21b7e86df2eb11c175105c2ad";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogTrustRoot {
    pub key_id: &'static str,
    pub public_key_hex: &'static str,
}

pub const OPENASR_CATALOG_TRUST_ROOTS: &[CatalogTrustRoot] = &[CatalogTrustRoot {
    key_id: CATALOG_SIGNATURE_KEY_ID,
    public_key_hex: OPENASR_CATALOG_V1_PUBLIC_KEY_HEX,
}];

/// Key id + public key for the **local-catalog development signing key**.
///
/// A local/`file://`/filesystem catalog source is only ever reached through an
/// explicit `catalog_url` override (CLI `--catalog-url`, `OPENASR_CATALOG_URL`,
/// or the server's equivalent) -- never the production HTTPS endpoint, the
/// on-disk cache tier, or the embedded snapshot. Whoever supplies that catalog
/// file already fully controls its contents, so this key adds no
/// confidentiality; its only job is to force every local catalog through the
/// same signature/sha256/catalog_url/schema checks a production catalog goes
/// through, closing the "a local path skips verification entirely" bypass.
///
/// The seed is therefore intentionally NOT secret: it is the deterministic
/// `sha256` of a fixed public label (see [`LOCAL_CATALOG_DEV_SIGNING_KEY_SEED_HEX`]),
/// so any contributor can re-derive it without a shared secret. This key is
/// added ONLY to [`OPENASR_LOCAL_CATALOG_TRUST_ROOTS`], never to
/// [`OPENASR_CATALOG_TRUST_ROOTS`] -- a widely-known dev key must never be
/// able to validate an HTTPS/cached/embedded production catalog.
pub const CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID: &str = "openasr-catalog-local-dev-v1";
const OPENASR_CATALOG_LOCAL_DEV_PUBLIC_KEY_HEX: &str =
    "bc1306d4cc4a1cbc817a862ee0223713ff79208c39bc8ce732da851db3c6b6a1";

/// The deterministic, publicly documented seed behind
/// [`CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID`] --
/// `sha256("openasr.catalog_manifest.v1.local-dev-signing-key-seed")`. Not a
/// secret (see the trust-root doc comment above); exposed so tooling/tests can
/// sign a local/dev catalog without touching the production signing seed.
pub const LOCAL_CATALOG_DEV_SIGNING_KEY_SEED_HEX: &str =
    "7181d685f3c226e1c111574368512b603d67964c057165ad004683b84998960e";

/// Trust roots accepted for a LOCAL (`file://` / bare filesystem path) catalog
/// source: the production key (so a local copy of the real, committed catalog
/// and its production signature still verifies) plus the public local-dev key
/// above. Never used for an `https://` source; see
/// [`verify_local_catalog_signature_manifest`].
pub const OPENASR_LOCAL_CATALOG_TRUST_ROOTS: &[CatalogTrustRoot] = &[
    CatalogTrustRoot {
        key_id: CATALOG_SIGNATURE_KEY_ID,
        public_key_hex: OPENASR_CATALOG_V1_PUBLIC_KEY_HEX,
    },
    CatalogTrustRoot {
        key_id: CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID,
        public_key_hex: OPENASR_CATALOG_LOCAL_DEV_PUBLIC_KEY_HEX,
    },
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogSignatureManifest {
    pub schema_version: u32,
    pub catalog_url: String,
    pub catalog_sha256: String,
    pub catalog_epoch: u64,
    pub signature: CatalogSignature,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogSignature {
    pub algorithm: String,
    pub key_id: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedCatalogSignature {
    pub catalog_epoch: u64,
    pub catalog_sha256: String,
    pub key_id: String,
}

#[derive(Debug, Error)]
pub enum CatalogSecurityError {
    #[error("Could not parse catalog signature manifest '{source}': {source_error}")]
    ParseManifest {
        source: String,
        #[source]
        source_error: serde_json::Error,
    },
    #[error("Could not serialize catalog signature manifest: {source}")]
    SerializeManifest {
        #[source]
        source: serde_json::Error,
    },
    #[error("Unsupported catalog signature schema_version {found}")]
    UnsupportedSchema { found: u32 },
    #[error("Invalid catalog signature manifest field '{field}': {message}")]
    InvalidField {
        field: &'static str,
        message: String,
    },
    #[error("Catalog signature manifest URL mismatch: expected '{expected}', got '{actual}'")]
    CatalogUrlMismatch { expected: String, actual: String },
    #[error("Catalog sha256 mismatch: expected {expected}, got {actual}")]
    CatalogShaMismatch { expected: String, actual: String },
    #[error("Unknown catalog signature key id '{key_id}'")]
    UnknownKey { key_id: String },
    #[error("Invalid catalog signature public key for '{key_id}': {message}")]
    InvalidPublicKey { key_id: String, message: String },
    #[error("Invalid catalog signature bytes: {message}")]
    InvalidSignature { message: String },
    #[error("Catalog signature verification failed for key '{key_id}'")]
    SignatureRejected { key_id: String },
    #[error("Catalog epoch rollback rejected: received {received}, highest seen {stored}")]
    EpochRollback { received: u64, stored: u64 },
    #[error("Could not read catalog epoch '{path}': {source}")]
    ReadEpoch {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("Could not write catalog epoch '{path}': {source}")]
    WriteEpoch {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("Could not write catalog signature manifest '{path}': {source}")]
    WriteManifest {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

pub fn default_catalog_signature_cache_path(openasr_home: impl AsRef<Path>) -> PathBuf {
    openasr_home.as_ref().join(CATALOG_SIGNATURE_FILE_NAME)
}

pub fn default_catalog_epoch_path(openasr_home: impl AsRef<Path>) -> PathBuf {
    openasr_home.as_ref().join(CATALOG_EPOCH_FILE_NAME)
}

pub fn catalog_signature_source(catalog_source: &str) -> String {
    if let Some(path) = catalog_source.strip_prefix("file://") {
        return format!(
            "file://{}",
            adjacent_signature_path(Path::new(path)).display()
        );
    }

    if catalog_source.starts_with("https://")
        && let Some((prefix, _)) = catalog_source.rsplit_once('/')
    {
        return format!("{prefix}/{CATALOG_SIGNATURE_FILE_NAME}");
    }

    adjacent_signature_path(Path::new(catalog_source))
        .display()
        .to_string()
}

pub fn derive_catalog_public_key_hex(
    signing_key_seed_hex: &str,
) -> Result<String, CatalogSecurityError> {
    let seed = decode_hex_exact::<32>(signing_key_seed_hex, "signing_key_seed_hex")?;
    let signing_key = SigningKey::from_bytes(&seed);
    Ok(hex_lower(&signing_key.verifying_key().to_bytes()))
}

pub fn render_catalog_signature_manifest(
    catalog_contents: &str,
    catalog_url: &str,
    catalog_epoch: u64,
    key_id: &str,
    signing_key_seed_hex: &str,
) -> Result<String, CatalogSecurityError> {
    validate_manifest_text_field("catalog_url", catalog_url)?;
    validate_manifest_text_field("signature.key_id", key_id)?;
    validate_catalog_epoch(catalog_epoch)?;

    let seed = decode_hex_exact::<32>(signing_key_seed_hex, "signing_key_seed_hex")?;
    let signing_key = SigningKey::from_bytes(&seed);
    let catalog_sha256 = sha256_hex(catalog_contents.as_bytes());
    let signature = signing_key.sign(
        signature_payload(
            CATALOG_SIGNATURE_ALGORITHM,
            key_id,
            catalog_url,
            &catalog_sha256,
            catalog_epoch,
        )
        .as_bytes(),
    );
    let manifest = CatalogSignatureManifest {
        schema_version: CATALOG_SIGNATURE_SCHEMA_VERSION,
        catalog_url: catalog_url.to_string(),
        catalog_sha256,
        catalog_epoch,
        signature: CatalogSignature {
            algorithm: CATALOG_SIGNATURE_ALGORITHM.to_string(),
            key_id: key_id.to_string(),
            value: hex_lower(&signature.to_bytes()),
        },
    };

    serde_json::to_string_pretty(&manifest)
        .map(|mut value| {
            value.push('\n');
            value
        })
        .map_err(|source| CatalogSecurityError::SerializeManifest { source })
}

pub fn verify_catalog_signature_manifest(
    catalog_contents: &str,
    manifest_contents: &str,
    expected_catalog_url: &str,
) -> Result<VerifiedCatalogSignature, CatalogSecurityError> {
    verify_catalog_signature_manifest_with_roots(
        catalog_contents,
        manifest_contents,
        expected_catalog_url,
        OPENASR_CATALOG_TRUST_ROOTS,
    )
}

/// Like [`verify_catalog_signature_manifest`], but for a LOCAL (`file://` /
/// bare filesystem path) catalog source: trusts the production key or the
/// public local-dev key (see [`OPENASR_LOCAL_CATALOG_TRUST_ROOTS`]). Never use
/// this for an `https://`/embedded/cached-network source -- that must stay
/// scoped to [`verify_catalog_signature_manifest`]'s production-only roots.
pub fn verify_local_catalog_signature_manifest(
    catalog_contents: &str,
    manifest_contents: &str,
    expected_catalog_url: &str,
) -> Result<VerifiedCatalogSignature, CatalogSecurityError> {
    verify_catalog_signature_manifest_with_roots(
        catalog_contents,
        manifest_contents,
        expected_catalog_url,
        OPENASR_LOCAL_CATALOG_TRUST_ROOTS,
    )
}

pub(crate) fn verify_catalog_signature_manifest_with_roots(
    catalog_contents: &str,
    manifest_contents: &str,
    expected_catalog_url: &str,
    trust_roots: &[CatalogTrustRoot],
) -> Result<VerifiedCatalogSignature, CatalogSecurityError> {
    let manifest: CatalogSignatureManifest =
        serde_json::from_str(manifest_contents).map_err(|source_error| {
            CatalogSecurityError::ParseManifest {
                source: CATALOG_SIGNATURE_FILE_NAME.to_string(),
                source_error,
            }
        })?;
    validate_manifest(&manifest, expected_catalog_url)?;

    let actual_sha = sha256_hex(catalog_contents.as_bytes());
    if actual_sha != manifest.catalog_sha256 {
        return Err(CatalogSecurityError::CatalogShaMismatch {
            expected: manifest.catalog_sha256,
            actual: actual_sha,
        });
    }

    let trust_root = trust_roots
        .iter()
        .find(|root| root.key_id == manifest.signature.key_id)
        .ok_or_else(|| CatalogSecurityError::UnknownKey {
            key_id: manifest.signature.key_id.clone(),
        })?;
    let public_key =
        decode_hex_exact::<32>(trust_root.public_key_hex, "public_key_hex").map_err(|error| {
            CatalogSecurityError::InvalidPublicKey {
                key_id: trust_root.key_id.to_string(),
                message: error.to_string(),
            }
        })?;
    let verifying_key = VerifyingKey::from_bytes(&public_key).map_err(|error| {
        CatalogSecurityError::InvalidPublicKey {
            key_id: trust_root.key_id.to_string(),
            message: error.to_string(),
        }
    })?;
    let signature_bytes = decode_hex_exact::<64>(&manifest.signature.value, "signature.value")?;
    let signature = Signature::from_bytes(&signature_bytes);
    verifying_key
        .verify(
            signature_payload(
                &manifest.signature.algorithm,
                &manifest.signature.key_id,
                &manifest.catalog_url,
                &manifest.catalog_sha256,
                manifest.catalog_epoch,
            )
            .as_bytes(),
            &signature,
        )
        .map_err(|_| CatalogSecurityError::SignatureRejected {
            key_id: manifest.signature.key_id.clone(),
        })?;

    Ok(VerifiedCatalogSignature {
        catalog_epoch: manifest.catalog_epoch,
        catalog_sha256: manifest.catalog_sha256,
        key_id: manifest.signature.key_id,
    })
}

pub(crate) fn enforce_catalog_epoch(
    openasr_home: &Path,
    received_epoch: u64,
) -> Result<(), CatalogSecurityError> {
    validate_catalog_epoch(received_epoch)?;
    let epoch_path = default_catalog_epoch_path(openasr_home);
    let Some(stored_epoch) = read_catalog_epoch(&epoch_path)? else {
        return Ok(());
    };
    if received_epoch < stored_epoch {
        return Err(CatalogSecurityError::EpochRollback {
            received: received_epoch,
            stored: stored_epoch,
        });
    }
    Ok(())
}

/// Whether a signature verified under `key_id` participates in the shared,
/// cross-source anti-rollback epoch floor (`enforce_catalog_epoch_for_verified`
/// / `record_catalog_epoch_for_verified`).
///
/// Scoped to the production key only. The local-dev key is public and
/// self-signed by definition (see the doc comment on
/// [`CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID`]), so a dev-key-verified local catalog
/// must never touch the shared floor: not to advance it, and not to be
/// rejected by it. Without this gate, loading a single local catalog signed
/// with the (widely-known, publicly derivable) dev key at a very high
/// `catalog_epoch` would permanently raise the floor every production source
/// (HTTPS, on-disk signed cache, and the embedded offline snapshot) is checked
/// against, bricking all of them until an operator manually deleted
/// `catalog.epoch` -- a persistent, self-inflicted DoS with no signing-key
/// compromise required. The local dev workflow does not need the anti-rollback
/// floor's protection in the first place: it is a developer signing content
/// for their own preview, not a production distribution channel that needs
/// protecting against a stale/rolled-back re-serve.
pub(crate) fn participates_in_epoch_floor(key_id: &str) -> bool {
    key_id != CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID
}

/// Runs [`enforce_catalog_epoch`] for `verified`, but only if its key
/// [`participates_in_epoch_floor`]. Use this (not the raw `enforce_catalog_epoch`)
/// at every catalog-load call site so the local-dev key can never be rejected
/// by -- or, via [`record_catalog_epoch_for_verified`], advance -- the shared
/// production floor.
pub(crate) fn enforce_catalog_epoch_for_verified(
    openasr_home: &Path,
    verified: &VerifiedCatalogSignature,
) -> Result<(), CatalogSecurityError> {
    if !participates_in_epoch_floor(&verified.key_id) {
        return Ok(());
    }
    enforce_catalog_epoch(openasr_home, verified.catalog_epoch)
}

/// Runs [`record_catalog_epoch`] for `verified`, but only if its key
/// [`participates_in_epoch_floor`]. See [`enforce_catalog_epoch_for_verified`].
pub(crate) fn record_catalog_epoch_for_verified(
    openasr_home: &Path,
    verified: &VerifiedCatalogSignature,
) -> Result<(), CatalogSecurityError> {
    if !participates_in_epoch_floor(&verified.key_id) {
        return Ok(());
    }
    record_catalog_epoch(openasr_home, verified.catalog_epoch)
}

/// Classifies a catalog `catalog_url`/identity into the two trust domains a
/// catalog signature can be verified under. This single classification is
/// deliberately the ONE place that decides both (a) which trust roots a
/// signature may be verified against (production-only for [`Remote`],
/// additionally the public local-dev key for [`Local`] --
/// [`CatalogSourceKind::Remote`]/[`CatalogSourceKind::Local`]) and (b), in
/// `registry::read_catalog_source`, which transport reads the bytes. Routing
/// both decisions through the same function means a future catalog source
/// scheme cannot be added to one without also changing the other -- there is
/// no second `starts_with` check to forget and leave classified as `Local`
/// (and therefore local-dev-key-eligible) by omission.
///
/// [`Remote`]: CatalogSourceKind::Remote
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CatalogSourceKind {
    /// Fetched over the network (`https://`); only the production key may
    /// sign it.
    Remote,
    /// Read from the local filesystem (`file://` or a bare path), OR an
    /// identity that is not itself a network address (e.g. a caller-supplied
    /// non-production `expected_catalog_url`); additionally accepts the
    /// public local-dev key.
    Local,
}

pub(crate) fn classify_catalog_identity(identity: &str) -> CatalogSourceKind {
    if identity.starts_with("https://") {
        CatalogSourceKind::Remote
    } else {
        CatalogSourceKind::Local
    }
}

pub(crate) fn cache_catalog_manifest(
    openasr_home: &Path,
    manifest_contents: &str,
) -> Result<(), CatalogSecurityError> {
    let path = default_catalog_signature_cache_path(openasr_home);
    atomic_file::write_file_atomically(&path, manifest_contents.as_bytes())
        .map_err(|source| CatalogSecurityError::WriteManifest { path, source })
}

pub(crate) fn record_catalog_epoch(
    openasr_home: &Path,
    epoch: u64,
) -> Result<(), CatalogSecurityError> {
    validate_catalog_epoch(epoch)?;
    let path = default_catalog_epoch_path(openasr_home);
    let contents = format!("{epoch}\n");
    atomic_file::write_file_atomically(&path, contents.as_bytes())
        .map_err(|source| CatalogSecurityError::WriteEpoch { path, source })
}

pub(crate) fn read_catalog_epoch(path: &Path) -> Result<Option<u64>, CatalogSecurityError> {
    match fs::read_to_string(path) {
        Ok(contents) => {
            let value = contents.trim().parse::<u64>().map_err(|error| {
                CatalogSecurityError::InvalidField {
                    field: "catalog_epoch",
                    message: format!("could not parse stored epoch: {error}"),
                }
            })?;
            validate_catalog_epoch(value)?;
            Ok(Some(value))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(CatalogSecurityError::ReadEpoch {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn validate_manifest(
    manifest: &CatalogSignatureManifest,
    expected_catalog_url: &str,
) -> Result<(), CatalogSecurityError> {
    if manifest.schema_version != CATALOG_SIGNATURE_SCHEMA_VERSION {
        return Err(CatalogSecurityError::UnsupportedSchema {
            found: manifest.schema_version,
        });
    }
    validate_manifest_text_field("catalog_url", &manifest.catalog_url)?;
    validate_manifest_text_field("catalog_sha256", &manifest.catalog_sha256)?;
    validate_manifest_text_field("signature.algorithm", &manifest.signature.algorithm)?;
    validate_manifest_text_field("signature.key_id", &manifest.signature.key_id)?;
    validate_manifest_text_field("signature.value", &manifest.signature.value)?;
    validate_catalog_epoch(manifest.catalog_epoch)?;
    if manifest.catalog_url != expected_catalog_url {
        return Err(CatalogSecurityError::CatalogUrlMismatch {
            expected: expected_catalog_url.to_string(),
            actual: manifest.catalog_url.clone(),
        });
    }
    if manifest.signature.algorithm != CATALOG_SIGNATURE_ALGORITHM {
        return Err(CatalogSecurityError::InvalidField {
            field: "signature.algorithm",
            message: format!(
                "expected {CATALOG_SIGNATURE_ALGORITHM}, got {}",
                manifest.signature.algorithm
            ),
        });
    }
    if manifest.catalog_sha256.len() != 64
        || !manifest
            .catalog_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(CatalogSecurityError::InvalidField {
            field: "catalog_sha256",
            message: "expected 64 lowercase hex characters".to_string(),
        });
    }
    if manifest.signature.value.len() != 128
        || !manifest
            .signature
            .value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(CatalogSecurityError::InvalidField {
            field: "signature.value",
            message: "expected 128 hex characters".to_string(),
        });
    }
    Ok(())
}

fn validate_catalog_epoch(epoch: u64) -> Result<(), CatalogSecurityError> {
    if epoch == 0 {
        return Err(CatalogSecurityError::InvalidField {
            field: "catalog_epoch",
            message: "must be greater than zero".to_string(),
        });
    }
    Ok(())
}

fn validate_manifest_text_field(
    field: &'static str,
    value: &str,
) -> Result<(), CatalogSecurityError> {
    if value.trim().is_empty() {
        return Err(CatalogSecurityError::InvalidField {
            field,
            message: "must not be empty".to_string(),
        });
    }
    if value.contains('\n') || value.contains('\r') {
        return Err(CatalogSecurityError::InvalidField {
            field,
            message: "must not contain newlines".to_string(),
        });
    }
    Ok(())
}

fn signature_payload(
    algorithm: &str,
    key_id: &str,
    catalog_url: &str,
    catalog_sha256: &str,
    catalog_epoch: u64,
) -> String {
    format!(
        "{CATALOG_SIGNATURE_DOMAIN}\nalgorithm:{algorithm}\nkey_id:{key_id}\ncatalog_url:{catalog_url}\ncatalog_sha256:{catalog_sha256}\ncatalog_epoch:{catalog_epoch}\n"
    )
}

fn adjacent_signature_path(catalog_path: &Path) -> PathBuf {
    catalog_path.with_file_name(CATALOG_SIGNATURE_FILE_NAME)
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

fn decode_hex_exact<const N: usize>(
    value: &str,
    field: &'static str,
) -> Result<[u8; N], CatalogSecurityError> {
    let value = value.trim();
    let bytes = value.as_bytes();
    if bytes.len() != N * 2 {
        return Err(CatalogSecurityError::InvalidField {
            field,
            message: format!("expected {} hex characters", N * 2),
        });
    }
    let mut out = [0_u8; N];
    for index in 0..N {
        let hi = hex_nibble(bytes[index * 2], field)?;
        let lo = hex_nibble(bytes[index * 2 + 1], field)?;
        out[index] = (hi << 4) | lo;
    }
    Ok(out)
}

fn hex_nibble(byte: u8, field: &'static str) -> Result<u8, CatalogSecurityError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(CatalogSecurityError::InvalidField {
            field,
            message: "invalid hex".to_string(),
        }),
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SEED_HEX: &str = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60";
    const TEST_PUBLIC_KEY_HEX: &str =
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";
    const TEST_KEY_ID: &str = "test-catalog-key";
    const TEST_CATALOG_URL: &str = "https://catalog.openasr.org/v1/catalog.json";

    fn test_roots() -> [CatalogTrustRoot; 1] {
        [CatalogTrustRoot {
            key_id: TEST_KEY_ID,
            public_key_hex: TEST_PUBLIC_KEY_HEX,
        }]
    }

    #[test]
    fn derives_public_key_from_seed() {
        assert_eq!(
            derive_catalog_public_key_hex(TEST_SEED_HEX).unwrap(),
            TEST_PUBLIC_KEY_HEX
        );
    }

    #[test]
    fn signed_manifest_verifies_catalog_bytes_and_epoch() {
        let catalog = r#"{"schema_version":1,"models":[]}"#;
        let manifest = render_catalog_signature_manifest(
            catalog,
            TEST_CATALOG_URL,
            42,
            TEST_KEY_ID,
            TEST_SEED_HEX,
        )
        .unwrap();

        let verified = verify_catalog_signature_manifest_with_roots(
            catalog,
            &manifest,
            TEST_CATALOG_URL,
            &test_roots(),
        )
        .unwrap();

        assert_eq!(verified.catalog_epoch, 42);
        assert_eq!(verified.key_id, TEST_KEY_ID);
    }

    #[test]
    fn signed_manifest_rejects_tampered_catalog() {
        let manifest = render_catalog_signature_manifest(
            r#"{"schema_version":1,"models":[]}"#,
            TEST_CATALOG_URL,
            42,
            TEST_KEY_ID,
            TEST_SEED_HEX,
        )
        .unwrap();

        let error = verify_catalog_signature_manifest_with_roots(
            r#"{"schema_version":1,"models":[{"id":"tampered"}]}"#,
            &manifest,
            TEST_CATALOG_URL,
            &test_roots(),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("Catalog sha256 mismatch"));
    }

    #[test]
    fn local_dev_public_key_matches_its_documented_seed() {
        // The dev seed is intentionally public (see the doc comment on
        // `CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID`); pin the derivation so the
        // committed public key and the committed seed can never silently drift
        // apart from each other.
        assert_eq!(
            derive_catalog_public_key_hex(LOCAL_CATALOG_DEV_SIGNING_KEY_SEED_HEX).unwrap(),
            OPENASR_CATALOG_LOCAL_DEV_PUBLIC_KEY_HEX
        );
    }

    #[test]
    fn local_catalog_trust_roots_include_the_production_key() {
        // A local copy of the real, committed catalog + its production
        // signature must still verify through the local trust-root set.
        assert!(
            OPENASR_LOCAL_CATALOG_TRUST_ROOTS
                .iter()
                .any(|root| root.key_id == CATALOG_SIGNATURE_KEY_ID
                    && root.public_key_hex == OPENASR_CATALOG_V1_PUBLIC_KEY_HEX)
        );
    }

    #[test]
    fn local_dev_signed_manifest_verifies_through_local_roots() {
        let catalog = r#"{"schema_version":1,"models":[]}"#;
        let source = "file:///tmp/local-catalog.json";
        let manifest = render_catalog_signature_manifest(
            catalog,
            source,
            7,
            CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID,
            LOCAL_CATALOG_DEV_SIGNING_KEY_SEED_HEX,
        )
        .unwrap();

        let verified = verify_local_catalog_signature_manifest(catalog, &manifest, source).unwrap();
        assert_eq!(verified.catalog_epoch, 7);
        assert_eq!(verified.key_id, CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID);
    }

    #[test]
    fn local_dev_signed_manifest_never_verifies_as_a_production_catalog() {
        // The whole point of keeping the dev key out of
        // `OPENASR_CATALOG_TRUST_ROOTS`: a widely-known dev key must never
        // authorize an HTTPS/embedded/cached production catalog.
        let catalog = r#"{"schema_version":1,"models":[]}"#;
        let source = "https://catalog.openasr.org/v1/catalog.json";
        let manifest = render_catalog_signature_manifest(
            catalog,
            source,
            7,
            CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID,
            LOCAL_CATALOG_DEV_SIGNING_KEY_SEED_HEX,
        )
        .unwrap();

        let error = verify_catalog_signature_manifest(catalog, &manifest, source)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("Unknown catalog signature key id"),
            "{error}"
        );
    }

    #[test]
    fn epoch_guard_rejects_rollback() {
        let temp = tempfile::tempdir().unwrap();
        record_catalog_epoch(temp.path(), 10).unwrap();

        let error = enforce_catalog_epoch(temp.path(), 9)
            .unwrap_err()
            .to_string();

        assert!(error.contains("rollback"));
    }

    #[test]
    fn only_the_production_key_participates_in_the_epoch_floor() {
        // B1 unit guard: the gate the higher-level `registry.rs` call sites
        // rely on to keep a dev-key-verified local catalog out of the shared
        // anti-rollback floor.
        assert!(participates_in_epoch_floor(CATALOG_SIGNATURE_KEY_ID));
        assert!(!participates_in_epoch_floor(
            CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID
        ));
        assert!(participates_in_epoch_floor("some-unrelated-key"));
    }

    #[test]
    fn enforce_and_record_for_verified_skip_the_floor_for_the_dev_key() {
        let temp = tempfile::tempdir().unwrap();
        let dev_verified = VerifiedCatalogSignature {
            catalog_epoch: u64::MAX,
            catalog_sha256: "0".repeat(64),
            key_id: CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID.to_string(),
        };

        // Recording a dev-key verification must not create the shared epoch
        // file at all.
        record_catalog_epoch_for_verified(temp.path(), &dev_verified).unwrap();
        assert!(!default_catalog_epoch_path(temp.path()).exists());

        // A production catalog at a low epoch must not be rejected as a
        // rollback against the dev catalog's (never persisted) high epoch.
        let production_verified = VerifiedCatalogSignature {
            catalog_epoch: 1,
            catalog_sha256: "0".repeat(64),
            key_id: CATALOG_SIGNATURE_KEY_ID.to_string(),
        };
        enforce_catalog_epoch_for_verified(temp.path(), &production_verified)
            .expect("no floor was ever recorded, so epoch 1 must be accepted");
    }

    #[test]
    fn enforce_and_record_for_verified_still_apply_the_floor_for_the_production_key() {
        let temp = tempfile::tempdir().unwrap();
        let high = VerifiedCatalogSignature {
            catalog_epoch: 10,
            catalog_sha256: "0".repeat(64),
            key_id: CATALOG_SIGNATURE_KEY_ID.to_string(),
        };
        record_catalog_epoch_for_verified(temp.path(), &high).unwrap();
        assert_eq!(
            read_catalog_epoch(&default_catalog_epoch_path(temp.path())).unwrap(),
            Some(10)
        );

        let low = VerifiedCatalogSignature {
            catalog_epoch: 9,
            catalog_sha256: "0".repeat(64),
            key_id: CATALOG_SIGNATURE_KEY_ID.to_string(),
        };
        let error = enforce_catalog_epoch_for_verified(temp.path(), &low)
            .unwrap_err()
            .to_string();
        assert!(error.contains("rollback"), "{error}");
    }

    #[test]
    fn classify_catalog_identity_only_treats_https_as_remote() {
        // S2 unit guard: this single classification is what both
        // `registry::read_catalog_source` (transport) and
        // `registry::verify_catalog_manifest_for_source` (trust roots) must
        // consult, so they can never drift apart on a new scheme.
        assert_eq!(
            classify_catalog_identity("https://catalog.openasr.org/v1/catalog.json"),
            CatalogSourceKind::Remote
        );
        assert_eq!(
            classify_catalog_identity("file:///tmp/catalog.json"),
            CatalogSourceKind::Local
        );
        assert_eq!(
            classify_catalog_identity("/tmp/catalog.json"),
            CatalogSourceKind::Local
        );
        assert_eq!(
            classify_catalog_identity("http://catalog.openasr.org/v1/catalog.json"),
            CatalogSourceKind::Local
        );
    }
}
