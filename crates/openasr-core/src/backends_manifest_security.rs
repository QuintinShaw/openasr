//! Ed25519 signing/verification for the `backends-manifest.json` release
//! asset (the per-release index of downloadable Windows GPU-kernel sidecars
//! -- vulkan/cuda/hip -- that the desktop app switches between at runtime).
//!
//! This deliberately reuses the SAME signing key material and the SAME
//! production trust root as the model catalog
//! (`catalog_security::OPENASR_CATALOG_TRUST_ROOTS`, key id
//! [`CATALOG_SIGNATURE_KEY_ID`]) rather than minting a second keypair: one
//! signing seed, one trust root, one place a maintainer manages key custody
//! and rotation. Only the domain-separation label baked into the signed
//! payload differs from the catalog's
//! (`openasr.backends_manifest.v1` vs `openasr.catalog_manifest.v1`), so a
//! signature produced for one manifest kind can never be replayed as a valid
//! signature for the other, even though both verify under the same public
//! key.
//!
//! Signing stays a LOCAL, maintainer-run operation, exactly like
//! `tooling/publish-model/scripts/publish_catalog.sh`: the seed
//! (`OPENASR_CATALOG_SIGNING_KEY_SEED_HEX`) never enters CI. Unlike the
//! catalog, this manifest carries no anti-rollback `epoch` -- it is generated
//! fresh per immutable, version-namespaced release URL
//! (`https://dl.openasr.org/core/v<version>/backends-manifest.json`), so
//! there is no shared mutable endpoint a stale signature could roll back.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::catalog_security::{
    CATALOG_SIGNATURE_KEY_ID, CatalogTrustRoot, OPENASR_CATALOG_TRUST_ROOTS,
};

pub const BACKENDS_MANIFEST_SIGNATURE_SCHEMA_VERSION: u32 = 1;
pub const BACKENDS_MANIFEST_SIGNATURE_FILE_NAME: &str = "backends-manifest.signature.json";
pub const BACKENDS_MANIFEST_SIGNATURE_ALGORITHM: &str = "ed25519";

const BACKENDS_MANIFEST_SIGNATURE_DOMAIN: &str = "openasr.backends_manifest.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendsManifestSignature {
    pub schema_version: u32,
    pub manifest_url: String,
    pub manifest_sha256: String,
    pub signature: BackendsManifestSignatureValue,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendsManifestSignatureValue {
    pub algorithm: String,
    pub key_id: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedBackendsManifestSignature {
    pub manifest_sha256: String,
    pub key_id: String,
}

#[derive(Debug, Error)]
pub enum BackendsManifestSecurityError {
    #[error("Could not parse backends-manifest signature '{source}': {source_error}")]
    ParseSignature {
        source: String,
        #[source]
        source_error: serde_json::Error,
    },
    #[error("Could not serialize backends-manifest signature: {source}")]
    SerializeSignature {
        #[source]
        source: serde_json::Error,
    },
    #[error("Unsupported backends-manifest signature schema_version {found}")]
    UnsupportedSchema { found: u32 },
    #[error("Invalid backends-manifest signature field '{field}': {message}")]
    InvalidField {
        field: &'static str,
        message: String,
    },
    #[error("backends-manifest signature URL mismatch: expected '{expected}', got '{actual}'")]
    ManifestUrlMismatch { expected: String, actual: String },
    #[error("backends-manifest sha256 mismatch: expected {expected}, got {actual}")]
    ManifestShaMismatch { expected: String, actual: String },
    #[error("Unknown backends-manifest signature key id '{key_id}'")]
    UnknownKey { key_id: String },
    #[error("Invalid backends-manifest signature public key for '{key_id}': {message}")]
    InvalidPublicKey { key_id: String, message: String },
    #[error("Invalid backends-manifest signature bytes: {message}")]
    InvalidSignature { message: String },
    #[error("backends-manifest signature verification failed for key '{key_id}'")]
    SignatureRejected { key_id: String },
}

/// Renders a signed `backends-manifest.signature.json` sidecar for
/// `manifest_contents` (the exact bytes of the already-written
/// `backends-manifest.json`). Mirrors
/// `catalog_security::render_catalog_signature_manifest`'s shape: caller
/// supplies the signing seed (never read from the environment here -- the
/// CLI entry point owns that so it stays a single, auditable place that
/// touches `OPENASR_CATALOG_SIGNING_KEY_SEED_HEX`).
pub fn render_backends_manifest_signature(
    manifest_contents: &str,
    manifest_url: &str,
    key_id: &str,
    signing_key_seed_hex: &str,
) -> Result<String, BackendsManifestSecurityError> {
    validate_text_field("manifest_url", manifest_url)?;
    validate_text_field("signature.key_id", key_id)?;

    let seed = decode_hex_exact::<32>(signing_key_seed_hex, "signing_key_seed_hex")?;
    let signing_key = SigningKey::from_bytes(&seed);
    let manifest_sha256 = sha256_hex(manifest_contents.as_bytes());
    let signature = signing_key.sign(
        signature_payload(
            BACKENDS_MANIFEST_SIGNATURE_ALGORITHM,
            key_id,
            manifest_url,
            &manifest_sha256,
        )
        .as_bytes(),
    );
    let manifest = BackendsManifestSignature {
        schema_version: BACKENDS_MANIFEST_SIGNATURE_SCHEMA_VERSION,
        manifest_url: manifest_url.to_string(),
        manifest_sha256,
        signature: BackendsManifestSignatureValue {
            algorithm: BACKENDS_MANIFEST_SIGNATURE_ALGORITHM.to_string(),
            key_id: key_id.to_string(),
            value: hex_lower(&signature.to_bytes()),
        },
    };

    serde_json::to_string_pretty(&manifest)
        .map(|mut value| {
            value.push('\n');
            value
        })
        .map_err(|source| BackendsManifestSecurityError::SerializeSignature { source })
}

/// Verifies a `backends-manifest.signature.json` sidecar against the
/// production catalog trust root
/// (`catalog_security::OPENASR_CATALOG_TRUST_ROOTS`) -- the same key that
/// signs the model catalog. There is no "local dev" variant of this
/// signature (unlike the catalog's `CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID`):
/// the backends manifest is only ever fetched from the production
/// `dl.openasr.org` / GitHub Releases endpoints, never a local dev override.
pub fn verify_backends_manifest_signature(
    manifest_contents: &str,
    signature_contents: &str,
    expected_manifest_url: &str,
) -> Result<VerifiedBackendsManifestSignature, BackendsManifestSecurityError> {
    verify_backends_manifest_signature_with_roots(
        manifest_contents,
        signature_contents,
        expected_manifest_url,
        OPENASR_CATALOG_TRUST_ROOTS,
    )
}

/// Like [`verify_backends_manifest_signature`], but against an injectable
/// trust-root set -- exists so tests can verify the sign/verify round trip
/// with a throwaway test keypair instead of the real production seed (which
/// never lives in this repo or CI; see the module doc).
pub(crate) fn verify_backends_manifest_signature_with_roots(
    manifest_contents: &str,
    signature_contents: &str,
    expected_manifest_url: &str,
    trust_roots: &[CatalogTrustRoot],
) -> Result<VerifiedBackendsManifestSignature, BackendsManifestSecurityError> {
    let signature: BackendsManifestSignature =
        serde_json::from_str(signature_contents).map_err(|source_error| {
            BackendsManifestSecurityError::ParseSignature {
                source: BACKENDS_MANIFEST_SIGNATURE_FILE_NAME.to_string(),
                source_error,
            }
        })?;
    validate_signature(&signature, expected_manifest_url)?;

    let actual_sha = sha256_hex(manifest_contents.as_bytes());
    if actual_sha != signature.manifest_sha256 {
        return Err(BackendsManifestSecurityError::ManifestShaMismatch {
            expected: signature.manifest_sha256,
            actual: actual_sha,
        });
    }

    let trust_root = trust_roots
        .iter()
        .find(|root| root.key_id == signature.signature.key_id)
        .ok_or_else(|| BackendsManifestSecurityError::UnknownKey {
            key_id: signature.signature.key_id.clone(),
        })?;
    let public_key =
        decode_hex_exact::<32>(trust_root.public_key_hex, "public_key_hex").map_err(|error| {
            BackendsManifestSecurityError::InvalidPublicKey {
                key_id: trust_root.key_id.to_string(),
                message: error.to_string(),
            }
        })?;
    let verifying_key = VerifyingKey::from_bytes(&public_key).map_err(|error| {
        BackendsManifestSecurityError::InvalidPublicKey {
            key_id: trust_root.key_id.to_string(),
            message: error.to_string(),
        }
    })?;
    let signature_bytes = decode_hex_exact::<64>(&signature.signature.value, "signature.value")?;
    let ed25519_signature = Signature::from_bytes(&signature_bytes);
    verifying_key
        .verify(
            signature_payload(
                &signature.signature.algorithm,
                &signature.signature.key_id,
                &signature.manifest_url,
                &signature.manifest_sha256,
            )
            .as_bytes(),
            &ed25519_signature,
        )
        .map_err(|_| BackendsManifestSecurityError::SignatureRejected {
            key_id: signature.signature.key_id.clone(),
        })?;

    Ok(VerifiedBackendsManifestSignature {
        manifest_sha256: signature.manifest_sha256,
        key_id: signature.signature.key_id,
    })
}

fn validate_signature(
    signature: &BackendsManifestSignature,
    expected_manifest_url: &str,
) -> Result<(), BackendsManifestSecurityError> {
    if signature.schema_version != BACKENDS_MANIFEST_SIGNATURE_SCHEMA_VERSION {
        return Err(BackendsManifestSecurityError::UnsupportedSchema {
            found: signature.schema_version,
        });
    }
    validate_text_field("manifest_url", &signature.manifest_url)?;
    validate_text_field("manifest_sha256", &signature.manifest_sha256)?;
    validate_text_field("signature.algorithm", &signature.signature.algorithm)?;
    validate_text_field("signature.key_id", &signature.signature.key_id)?;
    validate_text_field("signature.value", &signature.signature.value)?;
    if signature.manifest_url != expected_manifest_url {
        return Err(BackendsManifestSecurityError::ManifestUrlMismatch {
            expected: expected_manifest_url.to_string(),
            actual: signature.manifest_url.clone(),
        });
    }
    if signature.signature.algorithm != BACKENDS_MANIFEST_SIGNATURE_ALGORITHM {
        return Err(BackendsManifestSecurityError::InvalidField {
            field: "signature.algorithm",
            message: format!(
                "expected {BACKENDS_MANIFEST_SIGNATURE_ALGORITHM}, got {}",
                signature.signature.algorithm
            ),
        });
    }
    if signature.manifest_sha256.len() != 64
        || !signature
            .manifest_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(BackendsManifestSecurityError::InvalidField {
            field: "manifest_sha256",
            message: "expected 64 lowercase hex characters".to_string(),
        });
    }
    if signature.signature.value.len() != 128
        || !signature
            .signature
            .value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(BackendsManifestSecurityError::InvalidField {
            field: "signature.value",
            message: "expected 128 hex characters".to_string(),
        });
    }
    Ok(())
}

fn signature_payload(
    algorithm: &str,
    key_id: &str,
    manifest_url: &str,
    manifest_sha256: &str,
) -> String {
    format!(
        "{BACKENDS_MANIFEST_SIGNATURE_DOMAIN}\nalgorithm:{algorithm}\nkey_id:{key_id}\nmanifest_url:{manifest_url}\nmanifest_sha256:{manifest_sha256}\n"
    )
}

fn validate_text_field(
    field: &'static str,
    value: &str,
) -> Result<(), BackendsManifestSecurityError> {
    if value.trim().is_empty() {
        return Err(BackendsManifestSecurityError::InvalidField {
            field,
            message: "must not be empty".to_string(),
        });
    }
    if value.contains('\n') || value.contains('\r') {
        return Err(BackendsManifestSecurityError::InvalidField {
            field,
            message: "must not contain newlines".to_string(),
        });
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

fn decode_hex_exact<const N: usize>(
    value: &str,
    field: &'static str,
) -> Result<[u8; N], BackendsManifestSecurityError> {
    let value = value.trim();
    let bytes = value.as_bytes();
    if bytes.len() != N * 2 {
        return Err(BackendsManifestSecurityError::InvalidField {
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

fn hex_nibble(byte: u8, field: &'static str) -> Result<u8, BackendsManifestSecurityError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(BackendsManifestSecurityError::InvalidField {
            field,
            message: "invalid hex digit".to_string(),
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

/// Convenience re-export so callers do not need to import
/// `catalog_security::CATALOG_SIGNATURE_KEY_ID` separately to sign/verify
/// with the production key id.
pub const BACKENDS_MANIFEST_PRODUCTION_KEY_ID: &str = CATALOG_SIGNATURE_KEY_ID;

#[cfg(test)]
mod tests {
    use super::*;

    // Well-known RFC 8032 Ed25519 test vector (same pair
    // catalog_security's own tests use) -- NOT the real production seed,
    // which never lives in this repo. `verify_backends_manifest_signature`
    // itself is hardcoded to the real production trust root, so tests go
    // through `verify_backends_manifest_signature_with_roots` with this
    // throwaway keypair instead.
    const TEST_SEED_HEX: &str = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60";
    const TEST_PUBLIC_KEY_HEX: &str =
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";
    const TEST_KEY_ID: &str = "test-backends-key";
    const TEST_MANIFEST_URL: &str = "https://dl.openasr.org/core/v0.1.10/backends-manifest.json";

    fn test_roots() -> [CatalogTrustRoot; 1] {
        [CatalogTrustRoot {
            key_id: TEST_KEY_ID,
            public_key_hex: TEST_PUBLIC_KEY_HEX,
        }]
    }

    #[test]
    fn signed_manifest_verifies_bytes_and_url() {
        let manifest = r#"{"schema_version":1,"core_version":"0.1.10"}"#;
        let signature = render_backends_manifest_signature(
            manifest,
            TEST_MANIFEST_URL,
            TEST_KEY_ID,
            TEST_SEED_HEX,
        )
        .unwrap();

        let verified = verify_backends_manifest_signature_with_roots(
            manifest,
            &signature,
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .unwrap();

        assert_eq!(verified.key_id, TEST_KEY_ID);
        assert_eq!(verified.manifest_sha256.len(), 64);
    }

    #[test]
    fn signed_manifest_rejects_tampered_bytes() {
        let signature = render_backends_manifest_signature(
            r#"{"schema_version":1,"core_version":"0.1.10"}"#,
            TEST_MANIFEST_URL,
            TEST_KEY_ID,
            TEST_SEED_HEX,
        )
        .unwrap();

        let error = verify_backends_manifest_signature_with_roots(
            r#"{"schema_version":1,"core_version":"0.1.11-tampered"}"#,
            &signature,
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("sha256 mismatch"));
    }

    #[test]
    fn signed_manifest_rejects_url_mismatch() {
        let manifest = r#"{"schema_version":1,"core_version":"0.1.10"}"#;
        let signature = render_backends_manifest_signature(
            manifest,
            TEST_MANIFEST_URL,
            TEST_KEY_ID,
            TEST_SEED_HEX,
        )
        .unwrap();

        let error = verify_backends_manifest_signature_with_roots(
            manifest,
            &signature,
            "https://dl.openasr.org/core/v9.9.9/backends-manifest.json",
            &test_roots(),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("URL mismatch"));
    }

    #[test]
    fn signed_manifest_rejects_unknown_key_id() {
        let manifest = r#"{"schema_version":1,"core_version":"0.1.10"}"#;
        // Signed with the real production key id but verified against a
        // trust-root set that only knows the test key id -- must reject as
        // unknown rather than silently accept.
        let signature = render_backends_manifest_signature(
            manifest,
            TEST_MANIFEST_URL,
            CATALOG_SIGNATURE_KEY_ID,
            TEST_SEED_HEX,
        )
        .unwrap();

        let error = verify_backends_manifest_signature_with_roots(
            manifest,
            &signature,
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("Unknown backends-manifest signature key id"));
    }

    #[test]
    fn backends_manifest_signature_cannot_be_replayed_as_a_catalog_signature() {
        // Cross-protocol replay guard: the domain-separated payload differs
        // from the catalog's, so a signature minted here must not verify as
        // a valid CATALOG signature over the "same" bytes/url pairing, even
        // though both use the same production key.
        let payload = r#"{"schema_version":1,"core_version":"0.1.10"}"#;
        let backends_signature = render_backends_manifest_signature(
            payload,
            TEST_MANIFEST_URL,
            TEST_KEY_ID,
            TEST_SEED_HEX,
        )
        .unwrap();

        // A catalog signature manifest is shaped differently
        // (catalog_url/catalog_sha256/catalog_epoch vs manifest_url/
        // manifest_sha256), so it cannot even parse as one -- this simply
        // pins that the two schemas stay structurally distinct.
        assert!(
            serde_json::from_str::<crate::catalog_security::CatalogSignatureManifest>(
                &backends_signature
            )
            .is_err()
        );
    }

    #[test]
    fn production_key_id_constant_matches_catalog() {
        assert_eq!(
            BACKENDS_MANIFEST_PRODUCTION_KEY_ID,
            CATALOG_SIGNATURE_KEY_ID
        );
    }
}
