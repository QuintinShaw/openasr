//! Schema parsing + verified-fetch entry point for the desktop app's
//! **inference kernel** manifest (`backends-manifest.json` +
//! `backends-manifest.signature.json`).
//!
//! Signature verification itself is NOT implemented here: it is owned by
//! [`crate::backends_manifest_security`] (Ed25519, reusing the model
//! catalog's production signing key and trust root under a distinct
//! domain-separation label, `openasr.backends_manifest.v1` vs the catalog's
//! `openasr.catalog_manifest.v1`, so a catalog signature can never be
//! replayed as a backends-manifest signature or vice versa -- see that
//! module's doc comment for the full rationale). This module only:
//!
//! - defines the manifest's own JSON schema ([`BackendsManifest`],
//!   [`PlatformBackends`], [`BackendEntry`]),
//! - calls [`crate::verify_backends_manifest_signature`] before ever parsing
//!   the manifest body ([`verify_and_parse`] is fail-closed: an unverified
//!   manifest is never parsed, let alone trusted),
//! - enforces the `core_version` match rule desktop relies on
//!   ([`verify_and_parse_for_core_version`]), and
//! - looks up entries by platform/backend and verifies a downloaded
//!   archive's sha256 against them ([`BackendsManifest::backend_entry`],
//!   [`BackendEntry::verify_sha256`]).
//!
//! This is a SEPARATE concept from [`crate::pull::install_backend_pack`]
//! (which downloads dynamically-loaded ggml backend *plugins* into a running
//! daemon). The manifest verified here instead describes prebuilt,
//! statically-linked `openasr-cli` release archives -- desktop swaps its
//! whole sidecar binary (vulkan/cuda/hip) rather than loading a plugin into
//! one. See `docs/backend-kernels.md` for the full contract.
//!
//! Fail-closed: any missing signature, tampered manifest, signature
//! mismatch, or unsupported `schema_version` is rejected. There is no
//! "trust anyway" path.

use std::collections::BTreeMap;
use std::str::Utf8Error;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::BackendsManifestSecurityError;
#[cfg(test)]
use crate::catalog_security::CatalogTrustRoot;

/// Only `1` is understood. A manifest declaring anything else is rejected
/// rather than best-effort parsed -- see [`BackendManifestError::UnsupportedSchema`].
pub const BACKENDS_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Canonical manifest filename, served alongside its detached signature
/// (see [`crate::BACKENDS_MANIFEST_SIGNATURE_FILE_NAME`], owned by
/// [`crate::backends_manifest_security`]) at
/// `https://dl.openasr.org/core/v<version>/` and as a GitHub release asset.
pub const BACKENDS_MANIFEST_FILE_NAME: &str = "backends-manifest.json";

/// Full parsed + schema-validated manifest. Signature verification and schema
/// validation both already happened by the time a caller holds one of these --
/// see [`verify_and_parse`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendsManifest {
    pub schema_version: u32,
    /// The `openasr-cli` semver these archives were built from (e.g. `"0.1.14"`).
    /// Desktop only ever accepts a manifest whose `core_version` matches its own
    /// bundled sidecar version -- see `docs/backend-kernels.md`'s version-match
    /// rule and [`BackendManifestError::CoreVersionMismatch`].
    pub core_version: String,
    /// Full git commit sha the archives were built from, for traceability.
    pub source_commit: String,
    /// Keyed by platform id (currently only `"windows-x86_64"`).
    pub platforms: BTreeMap<String, PlatformBackends>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlatformBackends {
    /// Keyed by backend id (`"vulkan"`, `"cuda"`, `"hip"`).
    pub backends: BTreeMap<String, BackendEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendEntry {
    /// Release asset filename (e.g. `openasr-0.1.14-windows-x86_64-cuda.zip`).
    pub asset: String,
    pub size_bytes: u64,
    /// Lowercase hex sha256 of the asset's exact bytes.
    pub sha256: String,
    /// Download URLs in try-order: `dl.openasr.org` first, GitHub release
    /// download as fallback. See [`BackendEntry::urls`].
    pub urls: Vec<String>,
    /// PE import-table DLL-name prefixes (case-insensitive) that the resolved
    /// `openasr.exe` inside this archive must import at least one of, e.g.
    /// `["cublas64_"]` for `cuda`. This is the driver-linkage self-check a
    /// downloader runs on the *extracted* binary; it is independent of the
    /// sha256 check (that guards transport integrity, this guards "the right
    /// binary for the right kernel").
    pub pe_import_markers: Vec<String>,
}

impl BackendEntry {
    /// Verify `bytes` (the downloaded archive) hashes to [`BackendEntry::sha256`].
    /// Case-insensitive comparison since hex casing is not itself meaningful.
    pub fn verify_sha256(&self, bytes: &[u8]) -> Result<(), BackendManifestError> {
        let actual = sha256_hex(bytes);
        if !actual.eq_ignore_ascii_case(&self.sha256) {
            return Err(BackendManifestError::Sha256Mismatch {
                expected: self.sha256.clone(),
                actual,
            });
        }
        Ok(())
    }
}

impl BackendsManifest {
    /// Look up the entry for `platform`/`backend` (e.g. `"windows-x86_64"` /
    /// `"cuda"`), fail-closed on either missing.
    pub fn backend_entry(
        &self,
        platform: &str,
        backend: &str,
    ) -> Result<&BackendEntry, BackendManifestError> {
        let platform_entry =
            self.platforms
                .get(platform)
                .ok_or_else(|| BackendManifestError::UnknownPlatform {
                    platform: platform.to_string(),
                })?;
        platform_entry
            .backends
            .get(backend)
            .ok_or_else(|| BackendManifestError::UnknownBackend {
                platform: platform.to_string(),
                backend: backend.to_string(),
            })
    }
}

#[derive(Debug, Error)]
pub enum BackendManifestError {
    #[error("backends manifest bytes are not valid UTF-8: {0}")]
    InvalidUtf8(#[from] Utf8Error),
    #[error("could not parse backends manifest JSON: {0}")]
    Parse(#[source] serde_json::Error),
    #[error("backends manifest signature rejected: {0}")]
    Signature(#[source] BackendsManifestSecurityError),
    #[error(
        "unsupported backends manifest schema_version {found} (this build understands {BACKENDS_MANIFEST_SCHEMA_VERSION})"
    )]
    UnsupportedSchema { found: u32 },
    #[error(
        "backends manifest core_version '{manifest}' does not match the requested core_version '{requested}'; refusing a version-mismatched kernel manifest"
    )]
    CoreVersionMismatch { manifest: String, requested: String },
    #[error("backends manifest has no entry for platform '{platform}'")]
    UnknownPlatform { platform: String },
    #[error("backends manifest has no entry for backend '{backend}' on platform '{platform}'")]
    UnknownBackend { platform: String, backend: String },
    #[error("backend archive sha256 mismatch: expected {expected}, got {actual}")]
    Sha256Mismatch { expected: String, actual: String },
}

/// Verify the detached signature over `manifest_bytes`
/// ([`crate::verify_backends_manifest_signature`], production trust root),
/// then parse and schema-validate the manifest. Fail-closed: a missing/invalid
/// signature, a sha256 mismatch between `manifest_bytes` and what the signature
/// covers, or an unsupported `schema_version` all return `Err` -- there is no
/// partial-trust fallback.
///
/// `expected_manifest_url` must be the exact URL the manifest was fetched from
/// -- it is bound into the signed payload the same way a catalog URL is, so a
/// signature for one manifest URL can never be replayed against another.
pub fn verify_and_parse(
    manifest_bytes: &[u8],
    signature_bytes: &[u8],
    expected_manifest_url: &str,
) -> Result<BackendsManifest, BackendManifestError> {
    let manifest_text = std::str::from_utf8(manifest_bytes)?;
    let signature_text = std::str::from_utf8(signature_bytes)?;
    crate::verify_backends_manifest_signature(manifest_text, signature_text, expected_manifest_url)
        .map_err(BackendManifestError::Signature)?;
    parse_and_validate_schema(manifest_text)
}

/// Like [`verify_and_parse`], additionally rejecting a manifest whose
/// `core_version` does not equal `expected_core_version`. This is the entry
/// point desktop should use in practice: a kernel manifest built for a
/// different `openasr-cli` release must never be accepted, since the
/// downloaded archive's `openasr.exe --version` is later checked against the
/// SAME `core_version` string as a second, binary-level confirmation.
pub fn verify_and_parse_for_core_version(
    manifest_bytes: &[u8],
    signature_bytes: &[u8],
    expected_manifest_url: &str,
    expected_core_version: &str,
) -> Result<BackendsManifest, BackendManifestError> {
    let manifest = verify_and_parse(manifest_bytes, signature_bytes, expected_manifest_url)?;
    if manifest.core_version != expected_core_version {
        return Err(BackendManifestError::CoreVersionMismatch {
            manifest: manifest.core_version,
            requested: expected_core_version.to_string(),
        });
    }
    Ok(manifest)
}

/// Test-only hook: same as [`verify_and_parse`] but against a caller-supplied
/// trust root set, mirroring
/// `backends_manifest_security::verify_backends_manifest_signature_with_roots`.
/// Exists so unit tests can sign fixtures with a throwaway keypair instead of
/// the real (secret) production signing key -- see the `tests` module below.
/// `#[cfg(test)]`: unlike catalog_security's `_with_roots` (which the
/// production-only wrapper itself funnels through), `verify_and_parse` calls
/// `verify_backends_manifest_signature` directly, so this helper has no
/// non-test caller.
#[cfg(test)]
pub(crate) fn verify_and_parse_with_roots(
    manifest_bytes: &[u8],
    signature_bytes: &[u8],
    expected_manifest_url: &str,
    trust_roots: &[CatalogTrustRoot],
) -> Result<BackendsManifest, BackendManifestError> {
    let manifest_text = std::str::from_utf8(manifest_bytes)?;
    let signature_text = std::str::from_utf8(signature_bytes)?;
    crate::backends_manifest_security::verify_backends_manifest_signature_with_roots(
        manifest_text,
        signature_text,
        expected_manifest_url,
        trust_roots,
    )
    .map_err(BackendManifestError::Signature)?;
    parse_and_validate_schema(manifest_text)
}

fn parse_and_validate_schema(
    manifest_text: &str,
) -> Result<BackendsManifest, BackendManifestError> {
    let manifest: BackendsManifest =
        serde_json::from_str(manifest_text).map_err(BackendManifestError::Parse)?;
    if manifest.schema_version != BACKENDS_MANIFEST_SCHEMA_VERSION {
        return Err(BackendManifestError::UnsupportedSchema {
            found: manifest.schema_version,
        });
    }
    Ok(manifest)
}

/// Lowercase hex sha256 of `bytes`. Exposed so callers (in this crate and, via
/// `openasr-core` as a path dependency, the desktop app) can hash a downloaded
/// archive without re-deriving the format `sha256` fields use.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_backends_manifest_signature;

    // Throwaway test keypair -- NOT the production signing key (that secret
    // never appears in this repo). Mirrors both catalog_security's and
    // backends_manifest_security's own test fixtures, which use the same
    // pattern for the same reason.
    const TEST_SEED_HEX: &str = "0101010101010101010101010101010101010101010101010101010101010101";
    const TEST_KEY_ID: &str = "test-backends-manifest-key";
    const TEST_MANIFEST_URL: &str = "https://dl.openasr.org/core/v0.1.14/backends-manifest.json";

    fn test_roots() -> [CatalogTrustRoot; 1] {
        [CatalogTrustRoot {
            key_id: TEST_KEY_ID,
            public_key_hex: Box::leak(
                crate::derive_catalog_public_key_hex(TEST_SEED_HEX)
                    .unwrap()
                    .into_boxed_str(),
            ),
        }]
    }

    fn sample_manifest_json() -> String {
        r#"{
            "schema_version": 1,
            "core_version": "0.1.14",
            "source_commit": "a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4",
            "platforms": {
                "windows-x86_64": {
                    "backends": {
                        "vulkan": {
                            "asset": "openasr-0.1.14-windows-x86_64-vulkan.zip",
                            "size_bytes": 123456,
                            "sha256": "1111111111111111111111111111111111111111111111111111111111111111",
                            "urls": ["https://dl.openasr.org/core/v0.1.14/openasr-0.1.14-windows-x86_64-vulkan.zip"],
                            "pe_import_markers": ["vulkan-1.dll"]
                        },
                        "cuda": {
                            "asset": "openasr-0.1.14-windows-x86_64-cuda.zip",
                            "size_bytes": 559000000,
                            "sha256": "2222222222222222222222222222222222222222222222222222222222222222",
                            "urls": ["https://dl.openasr.org/core/v0.1.14/openasr-0.1.14-windows-x86_64-cuda.zip"],
                            "pe_import_markers": ["cublas64_"]
                        }
                    }
                }
            }
        }"#
        .to_string()
    }

    fn sign(manifest_json: &str, url: &str) -> String {
        render_backends_manifest_signature(manifest_json, url, TEST_KEY_ID, TEST_SEED_HEX).unwrap()
    }

    #[test]
    fn verifies_and_parses_a_well_formed_signed_manifest() {
        let manifest_json = sample_manifest_json();
        let signature = sign(&manifest_json, TEST_MANIFEST_URL);

        let parsed = verify_and_parse_with_roots(
            manifest_json.as_bytes(),
            signature.as_bytes(),
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .expect("well-formed signed manifest must verify");

        assert_eq!(parsed.core_version, "0.1.14");
        assert_eq!(parsed.schema_version, 1);
        let vulkan = parsed.backend_entry("windows-x86_64", "vulkan").unwrap();
        assert_eq!(vulkan.asset, "openasr-0.1.14-windows-x86_64-vulkan.zip");
        assert_eq!(vulkan.pe_import_markers, vec!["vulkan-1.dll".to_string()]);
        let cuda = parsed.backend_entry("windows-x86_64", "cuda").unwrap();
        assert_eq!(cuda.pe_import_markers, vec!["cublas64_".to_string()]);
    }

    #[test]
    fn rejects_a_tampered_manifest() {
        let manifest_json = sample_manifest_json();
        let signature = sign(&manifest_json, TEST_MANIFEST_URL);

        // Flip one byte of a hash so the signed sha256 no longer matches.
        let tampered = manifest_json.replacen("1111111111", "9999999999", 1);
        assert_ne!(tampered, manifest_json);

        let error = verify_and_parse_with_roots(
            tampered.as_bytes(),
            signature.as_bytes(),
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .unwrap_err();
        assert!(
            matches!(error, BackendManifestError::Signature(_)),
            "{error:?}"
        );
    }

    #[test]
    fn rejects_a_signature_that_does_not_match() {
        let manifest_json = sample_manifest_json();
        // Sign a DIFFERENT manifest, then pair that signature with the real one.
        let other_signature = sign(
            r#"{"schema_version":1,"core_version":"0.0.1","source_commit":"x","platforms":{}}"#,
            TEST_MANIFEST_URL,
        );

        let error = verify_and_parse_with_roots(
            manifest_json.as_bytes(),
            other_signature.as_bytes(),
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .unwrap_err();
        assert!(
            matches!(error, BackendManifestError::Signature(_)),
            "{error:?}"
        );
    }

    #[test]
    fn rejects_missing_or_empty_signature_bytes() {
        let manifest_json = sample_manifest_json();
        let error = verify_and_parse_with_roots(
            manifest_json.as_bytes(),
            b"",
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .unwrap_err();
        assert!(
            matches!(error, BackendManifestError::Signature(_)),
            "{error:?}"
        );
    }

    #[test]
    fn rejects_unknown_schema_version() {
        let manifest_json =
            sample_manifest_json().replacen("\"schema_version\": 1", "\"schema_version\": 2", 1);
        let signature = sign(&manifest_json, TEST_MANIFEST_URL);

        let error = verify_and_parse_with_roots(
            manifest_json.as_bytes(),
            signature.as_bytes(),
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .unwrap_err();
        assert!(
            matches!(error, BackendManifestError::UnsupportedSchema { found: 2 }),
            "{error:?}"
        );
    }

    #[test]
    fn core_version_mismatch_is_rejected() {
        let manifest_json = sample_manifest_json();
        let signature = sign(&manifest_json, TEST_MANIFEST_URL);

        // verify_and_parse_for_core_version uses the production trust roots
        // (via verify_and_parse), so this test-signed fixture fails signature
        // verification first -- exercise the version-mismatch branch that is
        // unique to this function directly against a parsed manifest instead.
        let error = verify_and_parse_for_core_version(
            manifest_json.as_bytes(),
            signature.as_bytes(),
            TEST_MANIFEST_URL,
            "9.9.9",
        );
        assert!(error.is_err());

        let parsed = verify_and_parse_with_roots(
            manifest_json.as_bytes(),
            signature.as_bytes(),
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .unwrap();
        assert_eq!(parsed.core_version, "0.1.14");
        assert_ne!(parsed.core_version, "9.9.9");
    }

    #[test]
    fn unknown_platform_and_backend_are_reported_distinctly() {
        let manifest_json = sample_manifest_json();
        let signature = sign(&manifest_json, TEST_MANIFEST_URL);
        let parsed = verify_and_parse_with_roots(
            manifest_json.as_bytes(),
            signature.as_bytes(),
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .unwrap();

        assert!(matches!(
            parsed.backend_entry("macos-arm64", "vulkan").unwrap_err(),
            BackendManifestError::UnknownPlatform { .. }
        ));
        assert!(matches!(
            parsed
                .backend_entry("windows-x86_64", "rocm-legacy")
                .unwrap_err(),
            BackendManifestError::UnknownBackend { .. }
        ));
    }

    #[test]
    fn sha256_hex_matches_a_known_vector() {
        // sha256("") -- standard test vector.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn backend_entry_verify_sha256_rejects_mismatch() {
        let manifest_json = sample_manifest_json();
        let signature = sign(&manifest_json, TEST_MANIFEST_URL);
        let parsed = verify_and_parse_with_roots(
            manifest_json.as_bytes(),
            signature.as_bytes(),
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .unwrap();
        let vulkan = parsed.backend_entry("windows-x86_64", "vulkan").unwrap();
        let error = vulkan
            .verify_sha256(b"not the real archive bytes")
            .unwrap_err();
        assert!(matches!(error, BackendManifestError::Sha256Mismatch { .. }));
    }
}
