//! Domain-separated Ed25519 signatures for inert qualification manifests.
//!
//! Qualification manifests reuse the production catalog trust root, but the
//! signed payload has its own domain. A catalog signature therefore cannot be
//! replayed as qualification authority. There is no local
//! development trust root: qualification is meaningful only for immutable
//! release artifacts signed by the production key.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::catalog_security::{
    CATALOG_SIGNATURE_KEY_ID, CatalogTrustRoot, OPENASR_CATALOG_TRUST_ROOTS,
};

pub const QUALIFICATION_MANIFEST_SIGNATURE_SCHEMA_VERSION: u32 = 1;
pub const QUALIFICATION_MANIFEST_SIGNATURE_FILE_NAME: &str =
    "qualification-manifest.signature.json";
pub const QUALIFICATION_MANIFEST_SIGNATURE_ALGORITHM: &str = "ed25519";
pub const QUALIFICATION_MANIFEST_PRODUCTION_KEY_ID: &str = CATALOG_SIGNATURE_KEY_ID;

const QUALIFICATION_MANIFEST_SIGNATURE_DOMAIN: &str = "openasr.qualification_manifest.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct QualificationManifestSignature {
    pub schema_version: u32,
    pub manifest_url: String,
    pub manifest_sha256: String,
    pub signature: QualificationManifestSignatureValue,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct QualificationManifestSignatureValue {
    pub algorithm: String,
    pub key_id: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerifiedQualificationManifestSignature {
    pub manifest_sha256: String,
    pub key_id: String,
}

#[derive(Debug, Error)]
pub enum QualificationManifestSecurityError {
    #[error("could not parse qualification-manifest signature '{source}': {source_error}")]
    ParseSignature {
        source: String,
        #[source]
        source_error: serde_json::Error,
    },
    #[error("could not serialize qualification-manifest signature: {source}")]
    SerializeSignature {
        #[source]
        source: serde_json::Error,
    },
    #[error("unsupported qualification-manifest signature schema_version {found}")]
    UnsupportedSchema { found: u32 },
    #[error("invalid qualification-manifest signature field '{field}': {message}")]
    InvalidField {
        field: &'static str,
        message: String,
    },
    #[error("qualification-manifest signature URL mismatch: expected '{expected}', got '{actual}'")]
    ManifestUrlMismatch { expected: String, actual: String },
    #[error("qualification-manifest sha256 mismatch: expected {expected}, got {actual}")]
    ManifestShaMismatch { expected: String, actual: String },
    #[error("unknown qualification-manifest signature key id '{key_id}'")]
    UnknownKey { key_id: String },
    #[error("invalid qualification-manifest signature public key for '{key_id}': {message}")]
    InvalidPublicKey { key_id: String, message: String },
    #[error("invalid qualification-manifest signature bytes: {message}")]
    InvalidSignature { message: String },
    #[error("qualification-manifest signature verification failed for key '{key_id}'")]
    SignatureRejected { key_id: String },
}

pub(crate) fn render_qualification_manifest_signature(
    manifest_contents: &str,
    manifest_url: &str,
    key_id: &str,
    signing_key_seed_hex: &str,
) -> Result<String, QualificationManifestSecurityError> {
    validate_manifest_url(manifest_url)?;
    validate_text_field("signature.key_id", key_id)?;

    let seed = decode_hex_exact::<32>(signing_key_seed_hex, "signing_key_seed_hex")?;
    let signing_key = SigningKey::from_bytes(&seed);
    let manifest_sha256 = sha256_hex(manifest_contents.as_bytes());
    let signature = signing_key.sign(
        signature_payload(
            QUALIFICATION_MANIFEST_SIGNATURE_ALGORITHM,
            key_id,
            manifest_url,
            &manifest_sha256,
        )
        .as_bytes(),
    );
    let envelope = QualificationManifestSignature {
        schema_version: QUALIFICATION_MANIFEST_SIGNATURE_SCHEMA_VERSION,
        manifest_url: manifest_url.to_string(),
        manifest_sha256,
        signature: QualificationManifestSignatureValue {
            algorithm: QUALIFICATION_MANIFEST_SIGNATURE_ALGORITHM.to_string(),
            key_id: key_id.to_string(),
            value: hex_lower(&signature.to_bytes()),
        },
    };

    serde_json::to_string_pretty(&envelope)
        .map(|mut value| {
            value.push('\n');
            value
        })
        .map_err(|source| QualificationManifestSecurityError::SerializeSignature { source })
}

pub(crate) fn verify_qualification_manifest_signature(
    manifest_contents: &str,
    signature_contents: &str,
    expected_manifest_url: &str,
) -> Result<VerifiedQualificationManifestSignature, QualificationManifestSecurityError> {
    verify_qualification_manifest_signature_with_roots(
        manifest_contents,
        signature_contents,
        expected_manifest_url,
        OPENASR_CATALOG_TRUST_ROOTS,
    )
}

pub(crate) fn verify_qualification_manifest_signature_with_roots(
    manifest_contents: &str,
    signature_contents: &str,
    expected_manifest_url: &str,
    trust_roots: &[CatalogTrustRoot],
) -> Result<VerifiedQualificationManifestSignature, QualificationManifestSecurityError> {
    let signature: QualificationManifestSignature = serde_json::from_str(signature_contents)
        .map_err(
            |source_error| QualificationManifestSecurityError::ParseSignature {
                source: QUALIFICATION_MANIFEST_SIGNATURE_FILE_NAME.to_string(),
                source_error,
            },
        )?;
    validate_signature(&signature, expected_manifest_url)?;

    let actual_sha = sha256_hex(manifest_contents.as_bytes());
    if actual_sha != signature.manifest_sha256 {
        return Err(QualificationManifestSecurityError::ManifestShaMismatch {
            expected: signature.manifest_sha256,
            actual: actual_sha,
        });
    }

    let trust_root = trust_roots
        .iter()
        .find(|root| root.key_id == signature.signature.key_id)
        .ok_or_else(|| QualificationManifestSecurityError::UnknownKey {
            key_id: signature.signature.key_id.clone(),
        })?;
    let public_key =
        decode_hex_exact::<32>(trust_root.public_key_hex, "public_key_hex").map_err(|error| {
            QualificationManifestSecurityError::InvalidPublicKey {
                key_id: trust_root.key_id.to_string(),
                message: error.to_string(),
            }
        })?;
    let verifying_key = VerifyingKey::from_bytes(&public_key).map_err(|error| {
        QualificationManifestSecurityError::InvalidPublicKey {
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
        .map_err(|_| QualificationManifestSecurityError::SignatureRejected {
            key_id: signature.signature.key_id.clone(),
        })?;

    Ok(VerifiedQualificationManifestSignature {
        manifest_sha256: signature.manifest_sha256,
        key_id: signature.signature.key_id,
    })
}

fn validate_signature(
    signature: &QualificationManifestSignature,
    expected_manifest_url: &str,
) -> Result<(), QualificationManifestSecurityError> {
    if signature.schema_version != QUALIFICATION_MANIFEST_SIGNATURE_SCHEMA_VERSION {
        return Err(QualificationManifestSecurityError::UnsupportedSchema {
            found: signature.schema_version,
        });
    }
    validate_manifest_url(&signature.manifest_url)?;
    validate_lower_hex("manifest_sha256", &signature.manifest_sha256, 64)?;
    validate_text_field("signature.algorithm", &signature.signature.algorithm)?;
    validate_text_field("signature.key_id", &signature.signature.key_id)?;
    validate_lower_hex("signature.value", &signature.signature.value, 128)?;
    if signature.manifest_url != expected_manifest_url {
        return Err(QualificationManifestSecurityError::ManifestUrlMismatch {
            expected: expected_manifest_url.to_string(),
            actual: signature.manifest_url.clone(),
        });
    }
    if signature.signature.algorithm != QUALIFICATION_MANIFEST_SIGNATURE_ALGORITHM {
        return Err(QualificationManifestSecurityError::InvalidField {
            field: "signature.algorithm",
            message: format!(
                "expected {QUALIFICATION_MANIFEST_SIGNATURE_ALGORITHM}, got {}",
                signature.signature.algorithm
            ),
        });
    }
    Ok(())
}

fn validate_manifest_url(value: &str) -> Result<(), QualificationManifestSecurityError> {
    validate_text_field("manifest_url", value)?;
    let parsed = reqwest::Url::parse(value).map_err(|error| {
        QualificationManifestSecurityError::InvalidField {
            field: "manifest_url",
            message: format!("invalid URL: {error}"),
        }
    })?;
    let file_name = parsed
        .path_segments()
        .and_then(|mut segments| segments.next_back())
        .unwrap_or_default();
    if parsed.scheme() != "https"
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || !file_name.starts_with("openasr-")
        || !file_name.contains("-qualification-")
        || !file_name.ends_with(".json")
        || crate::safety::validate_safe_relative_path("manifest_url", file_name).is_err()
    {
        return Err(QualificationManifestSecurityError::InvalidField {
            field: "manifest_url",
            message: "must be a credential-free immutable HTTPS URL ending in a safe exact-cell qualification JSON basename without query or fragment"
                .to_string(),
        });
    }
    Ok(())
}

fn validate_text_field(
    field: &'static str,
    value: &str,
) -> Result<(), QualificationManifestSecurityError> {
    if value.trim().is_empty() || value.trim() != value || value.contains(['\r', '\n']) {
        Err(QualificationManifestSecurityError::InvalidField {
            field,
            message: "must be non-empty, trimmed, single-line text".to_string(),
        })
    } else {
        Ok(())
    }
}

fn validate_lower_hex(
    field: &'static str,
    value: &str,
    length: usize,
) -> Result<(), QualificationManifestSecurityError> {
    if value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(QualificationManifestSecurityError::InvalidField {
            field,
            message: format!("must be {length} lowercase hexadecimal characters"),
        })
    }
}

fn decode_hex_exact<const N: usize>(
    value: &str,
    field: &'static str,
) -> Result<[u8; N], QualificationManifestSecurityError> {
    if value.len() != N * 2 {
        return Err(QualificationManifestSecurityError::InvalidSignature {
            message: format!("{field} must contain exactly {} hex characters", N * 2),
        });
    }
    let mut out = [0_u8; N];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(chunk).map_err(|error| {
            QualificationManifestSecurityError::InvalidSignature {
                message: format!("{field} is not UTF-8: {error}"),
            }
        })?;
        out[index] = u8::from_str_radix(text, 16).map_err(|error| {
            QualificationManifestSecurityError::InvalidSignature {
                message: format!("{field} is not hexadecimal: {error}"),
            }
        })?;
    }
    Ok(out)
}

fn signature_payload(
    algorithm: &str,
    key_id: &str,
    manifest_url: &str,
    manifest_sha256: &str,
) -> String {
    format!(
        "{QUALIFICATION_MANIFEST_SIGNATURE_DOMAIN}\nalgorithm={algorithm}\nkey_id={key_id}\nmanifest_url={manifest_url}\nmanifest_sha256={manifest_sha256}\n"
    )
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        catalog_security::render_catalog_signature_manifest, derive_catalog_public_key_hex,
    };

    const TEST_SEED: &str = "0101010101010101010101010101010101010101010101010101010101010101";
    const TEST_KEY_ID: &str = "test-qualification-key";
    const URL: &str =
        "https://dl.openasr.org/core/v0.1.37/openasr-0.1.37-qualification-cuda-sm_89.json";

    fn roots() -> [CatalogTrustRoot; 1] {
        [CatalogTrustRoot {
            key_id: TEST_KEY_ID,
            public_key_hex: Box::leak(
                derive_catalog_public_key_hex(TEST_SEED)
                    .expect("derive test key")
                    .into_boxed_str(),
            ),
        }]
    }

    #[test]
    fn qualification_signature_round_trips_and_binds_url() {
        let body = r#"{"schema_version":1}"#;
        let signature = render_qualification_manifest_signature(body, URL, TEST_KEY_ID, TEST_SEED)
            .expect("sign qualification manifest");
        let verified =
            verify_qualification_manifest_signature_with_roots(body, &signature, URL, &roots())
                .expect("verify qualification manifest");
        assert_eq!(verified.key_id, TEST_KEY_ID);
        assert_eq!(verified.manifest_sha256, sha256_hex(body.as_bytes()));
        assert!(matches!(
            verify_qualification_manifest_signature_with_roots(
                body,
                &signature,
                "https://dl.openasr.org/other.json",
                &roots(),
            ),
            Err(QualificationManifestSecurityError::ManifestUrlMismatch { .. })
        ));
        assert!(matches!(
            verify_qualification_manifest_signature_with_roots(
                r#"{"schema_version":2}"#,
                &signature,
                URL,
                &roots(),
            ),
            Err(QualificationManifestSecurityError::ManifestShaMismatch { .. })
        ));
    }

    #[test]
    fn qualification_domain_rejects_catalog_signatures() {
        let body = r#"{"schema_version":1}"#;
        let catalog = render_catalog_signature_manifest(body, URL, 1, TEST_KEY_ID, TEST_SEED)
            .expect("sign catalog domain");
        assert!(
            verify_qualification_manifest_signature_with_roots(body, &catalog, URL, &roots())
                .is_err()
        );
    }

    #[test]
    fn signer_defaults_are_bound_to_the_production_trust_identity() {
        assert_eq!(CATALOG_SIGNATURE_KEY_ID, "openasr-catalog-v1");
        assert_ne!(
            QUALIFICATION_MANIFEST_SIGNATURE_DOMAIN,
            "openasr.catalog_manifest.v1"
        );
    }
}
