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
//!
//! ## Schema v2 (vendor layers)
//!
//! `schema_version: 2` adds an optional top-level `vendor_layers` map and an
//! optional `BackendEntry.vendor_layer` field, so a GPU backend's large,
//! content-addressed, core-version-independent vendor runtime (NVIDIA
//! cudart/cuBLAS, AMD rocBLAS/hipBLAS) can be split out of the small,
//! per-release sidecar archive (`openasr.exe` + its own build artifacts) --
//! see [`VendorLayer`] and [`BackendEntry::vendor_layer`]. A backend entry
//! with no `vendor_layer` (v1's shape, and `vulkan` under v2) is
//! self-contained: nothing else to fetch or verify. This build's reader
//! accepts BOTH `schema_version` `1` and `2` ([`BACKENDS_MANIFEST_SCHEMA_VERSIONS`])
//! -- a v1 manifest simply has an empty `vendor_layers` map and no backend
//! entry ever sets `vendor_layer`, so the same lookup/verification code path
//! handles both without a version branch at the call site. See
//! `docs/backend-kernels.md` for the full v2 contract and the layout this
//! unlocks on disk.

use std::collections::BTreeMap;
use std::str::Utf8Error;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::BackendsManifestSecurityError;
#[cfg(test)]
use crate::catalog_security::CatalogTrustRoot;

/// `schema_version` values this build understands and parses -- currently
/// `1` (no vendor layers) and `2` (optional vendor layers). A manifest
/// declaring anything else is rejected rather than best-effort parsed -- see
/// [`BackendManifestError::UnsupportedSchema`]. Kept as a slice (not a single
/// constant) specifically so both trains can be accepted at once: the
/// manifest is generated per-release, so an old desktop build only ever sees
/// the schema version live at the time it shipped, while a new desktop build
/// (or this crate's own tests) must keep reading last release's v1 manifests
/// too.
pub const BACKENDS_MANIFEST_SCHEMA_VERSIONS: &[u32] = &[1, 2];

/// The schema version this build's own tooling *produces* (signing self-check,
/// docs examples). Readers must still go through
/// [`BACKENDS_MANIFEST_SCHEMA_VERSIONS`], not this constant.
pub const BACKENDS_MANIFEST_CURRENT_SCHEMA_VERSION: u32 = 2;

/// Canonical manifest filename, served alongside its detached signature
/// (see [`crate::BACKENDS_MANIFEST_SIGNATURE_FILE_NAME`], owned by
/// [`crate::backends_manifest_security`]) at
/// `https://dl.openasr.org/core/v<version>/` and as a GitHub release asset.
pub const BACKENDS_MANIFEST_FILE_NAME: &str = "backends-manifest.json";

/// The one true `backends-manifest.json` URL for a given `core_version`,
/// e.g. `https://dl.openasr.org/core/v0.1.20/backends-manifest.json`. Both
/// the LOCAL signing step (`__openasr-sign-backends-manifest --manifest-url`)
/// and every desktop fetch path must bind the signature to THIS string,
/// regardless of which mirror/base URL the bytes were actually downloaded
/// from (`dl.openasr.org` direct, the China-accel proxy, or the GitHub
/// Releases fallback) -- see [`crate::verify_backends_manifest_signature`]'s
/// `expected_manifest_url` parameter and this module's `verify_and_parse*`
/// entry points. Using the real fetch URL there instead (the pre-fix bug,
/// #145) makes every mirror except the primary CDN fail signature
/// verification, since the signed payload only ever names the canonical
/// host. Host choice is not a security property here -- integrity is
/// entirely carried by the Ed25519 signature + sha256, not by which URL the
/// bytes happened to arrive from -- so pinning verification to one
/// canonical string is safe and (per the URL-mismatch check in
/// `backends_manifest_security::validate_signature`) still fail-closed
/// against a signature minted for a different core_version's manifest.
pub fn canonical_manifest_url(core_version: &str) -> String {
    format!("https://dl.openasr.org/core/v{core_version}/{BACKENDS_MANIFEST_FILE_NAME}")
}

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
    /// Content-addressed vendor GPU runtimes (NVIDIA cudart/cuBLAS, AMD
    /// rocBLAS/hipBLAS), keyed by a stable layer id (e.g. `"cuda-runtime"`,
    /// `"rocm-runtime"`). Introduced in `schema_version: 2`; absent (empty)
    /// on a v1 manifest, in which case every [`BackendEntry::vendor_layer`]
    /// is also `None` -- see [`BackendsManifest::vendor_layer`].
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub vendor_layers: BTreeMap<String, VendorLayer>,
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
    /// Key into [`BackendsManifest::vendor_layers`] this backend needs
    /// installed alongside its own (small, sidecar-only) archive before it
    /// can launch, e.g. `"cuda-runtime"`. `None` means self-contained (v1's
    /// only shape; `vulkan` stays self-contained under v2 too, since the
    /// Vulkan loader redistributable is small enough to ship inline). See
    /// [`BackendsManifest::vendor_layer`] to resolve this key, and
    /// `docs/backend-kernels.md`'s "Disk layout" section for the install
    /// ordering this implies (vendor layer first, then the sidecar's own
    /// `--version` probe, which needs the vendor DLLs on `PATH` to resolve
    /// its PE imports).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor_layer: Option<String>,
}

/// A large, content-addressed, core-version-independent GPU vendor runtime
/// layer (e.g. NVIDIA's cudart/cuBLAS redistributables, or AMD's rocBLAS/
/// hipBLAS + the `rocblas/library`/`hipblaslt/library` Tensile subtrees).
/// Shared across every core release that pins a compatible toolchain, so it
/// is downloaded and verified once and then reused by later `switchKernel`
/// upgrades that keep the same GPU backend -- see the module doc and
/// `docs/backend-kernels.md`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VendorLayer {
    /// Lowercase hex sha256 of the archive's exact bytes. Also the content
    /// address this layer is installed under on disk
    /// (`sidecars/vendor/<sha256>/...`) and uploaded under in object storage
    /// (`core/vendor/<sha256>/<asset>`) -- see [`VendorLayer::verify_sha256`].
    pub sha256: String,
    /// Archive filename, e.g. `openasr-vendor-cuda-runtime-<sha12>.zip`.
    pub asset: String,
    pub size_bytes: u64,
    /// Download URLs in try-order, same convention as [`BackendEntry::urls`].
    pub urls: Vec<String>,
    /// Human-readable build toolchain identifier for traceability (e.g.
    /// `"cuda-13.0"`, `"rocm-7.2"`) -- not used for any verification decision.
    pub toolchain: String,
}

impl VendorLayer {
    /// Verify `bytes` (the downloaded vendor archive) hashes to
    /// [`VendorLayer::sha256`]. Case-insensitive, mirroring
    /// [`BackendEntry::verify_sha256`].
    pub fn verify_sha256(&self, bytes: &[u8]) -> Result<(), BackendManifestError> {
        verify_sha256_matches(&self.sha256, bytes)
    }
}

impl BackendEntry {
    /// Verify `bytes` (the downloaded archive) hashes to [`BackendEntry::sha256`].
    /// Case-insensitive comparison since hex casing is not itself meaningful.
    pub fn verify_sha256(&self, bytes: &[u8]) -> Result<(), BackendManifestError> {
        verify_sha256_matches(&self.sha256, bytes)
    }
}

fn verify_sha256_matches(expected: &str, bytes: &[u8]) -> Result<(), BackendManifestError> {
    let actual = sha256_hex(bytes);
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(BackendManifestError::Sha256Mismatch {
            expected: expected.to_string(),
            actual,
        });
    }
    Ok(())
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

    /// Look up a vendor layer by its [`BackendEntry::vendor_layer`] key,
    /// fail-closed on missing -- e.g. a backend entry naming a vendor layer
    /// the manifest never actually defines (a producer-side bug this reader
    /// must not silently ignore).
    pub fn vendor_layer(&self, key: &str) -> Result<&VendorLayer, BackendManifestError> {
        self.vendor_layers
            .get(key)
            .ok_or_else(|| BackendManifestError::UnknownVendorLayer {
                key: key.to_string(),
            })
    }

    /// Resolve `entry`'s [`BackendEntry::vendor_layer`] (if any) through this
    /// manifest's `vendor_layers` map in one call. Returns `Ok(None)` for a
    /// self-contained backend (`vendor_layer` is `None`), `Err` if the entry
    /// names a layer this manifest does not define.
    pub fn resolve_vendor_layer(
        &self,
        entry: &BackendEntry,
    ) -> Result<Option<&VendorLayer>, BackendManifestError> {
        entry
            .vendor_layer
            .as_deref()
            .map(|key| self.vendor_layer(key))
            .transpose()
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
        "unsupported backends manifest schema_version {found} (this build understands {BACKENDS_MANIFEST_SCHEMA_VERSIONS:?})"
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
    #[error("backends manifest has no vendor_layers entry for key '{key}'")]
    UnknownVendorLayer { key: String },
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
    if !BACKENDS_MANIFEST_SCHEMA_VERSIONS.contains(&manifest.schema_version) {
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

    /// schema_version 2 fixture: `cuda` points at a `vendor_layers` entry,
    /// `vulkan` stays self-contained (no `vendor_layer`) -- mirrors the
    /// design doc's example manifest.
    fn sample_manifest_v2_json() -> String {
        r#"{
            "schema_version": 2,
            "core_version": "0.1.20",
            "source_commit": "a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4",
            "vendor_layers": {
                "cuda-runtime": {
                    "sha256": "3333333333333333333333333333333333333333333333333333333333333333",
                    "asset": "openasr-vendor-cuda-runtime-abc123abc123.zip",
                    "size_bytes": 520000000,
                    "urls": [
                        "https://dl.openasr.org/core/vendor/3333333333333333333333333333333333333333333333333333333333333333/openasr-vendor-cuda-runtime-abc123abc123.zip",
                        "https://github.com/QuintinShaw/openasr/releases/download/v0.1.20/openasr-vendor-cuda-runtime-abc123abc123.zip"
                    ],
                    "toolchain": "cuda-13.0"
                }
            },
            "platforms": {
                "windows-x86_64": {
                    "backends": {
                        "vulkan": {
                            "asset": "openasr-0.1.20-windows-x86_64-vulkan.zip",
                            "size_bytes": 123456,
                            "sha256": "1111111111111111111111111111111111111111111111111111111111111111",
                            "urls": ["https://dl.openasr.org/core/v0.1.20/openasr-0.1.20-windows-x86_64-vulkan.zip"],
                            "pe_import_markers": ["vulkan-1.dll"]
                        },
                        "cuda": {
                            "asset": "openasr-0.1.20-windows-x86_64-cuda-sidecar.zip",
                            "size_bytes": 9000000,
                            "sha256": "2222222222222222222222222222222222222222222222222222222222222222",
                            "urls": ["https://dl.openasr.org/core/v0.1.20/openasr-0.1.20-windows-x86_64-cuda-sidecar.zip"],
                            "pe_import_markers": ["cublas64_"],
                            "vendor_layer": "cuda-runtime"
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
        // 3 is not (yet) a recognized schema_version -- 2 is now accepted
        // (see accepts_a_schema_v2_manifest_with_vendor_layers below), so this
        // test must reach past both understood versions to exercise the
        // reject-unknown-schema path.
        let manifest_json =
            sample_manifest_json().replacen("\"schema_version\": 1", "\"schema_version\": 3", 1);
        let signature = sign(&manifest_json, TEST_MANIFEST_URL);

        let error = verify_and_parse_with_roots(
            manifest_json.as_bytes(),
            signature.as_bytes(),
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .unwrap_err();
        assert!(
            matches!(error, BackendManifestError::UnsupportedSchema { found: 3 }),
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

    #[test]
    fn accepts_a_schema_v2_manifest_with_vendor_layers() {
        let manifest_json = sample_manifest_v2_json();
        let signature = sign(&manifest_json, TEST_MANIFEST_URL);

        let parsed = verify_and_parse_with_roots(
            manifest_json.as_bytes(),
            signature.as_bytes(),
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .expect("well-formed schema v2 manifest must verify");

        assert_eq!(parsed.schema_version, 2);
        assert_eq!(parsed.vendor_layers.len(), 1);

        let cuda = parsed.backend_entry("windows-x86_64", "cuda").unwrap();
        assert_eq!(cuda.vendor_layer.as_deref(), Some("cuda-runtime"));
        let vendor = parsed.vendor_layer("cuda-runtime").unwrap();
        assert_eq!(vendor.toolchain, "cuda-13.0");
        assert_eq!(
            vendor.sha256,
            "3333333333333333333333333333333333333333333333333333333333333333"
        );

        let vulkan = parsed.backend_entry("windows-x86_64", "vulkan").unwrap();
        assert_eq!(
            vulkan.vendor_layer, None,
            "vulkan stays self-contained under v2"
        );
    }

    #[test]
    fn resolve_vendor_layer_follows_the_backend_entrys_key() {
        let manifest_json = sample_manifest_v2_json();
        let signature = sign(&manifest_json, TEST_MANIFEST_URL);
        let parsed = verify_and_parse_with_roots(
            manifest_json.as_bytes(),
            signature.as_bytes(),
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .unwrap();

        let cuda = parsed.backend_entry("windows-x86_64", "cuda").unwrap();
        let resolved = parsed.resolve_vendor_layer(cuda).unwrap();
        assert_eq!(
            resolved.map(|layer| layer.asset.as_str()),
            Some("openasr-vendor-cuda-runtime-abc123abc123.zip")
        );

        let vulkan = parsed.backend_entry("windows-x86_64", "vulkan").unwrap();
        assert_eq!(
            parsed.resolve_vendor_layer(vulkan).unwrap(),
            None,
            "self-contained backend resolves to no vendor layer"
        );
    }

    #[test]
    fn unknown_vendor_layer_key_is_rejected() {
        let manifest_json = sample_manifest_v2_json();
        let signature = sign(&manifest_json, TEST_MANIFEST_URL);
        let parsed = verify_and_parse_with_roots(
            manifest_json.as_bytes(),
            signature.as_bytes(),
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .unwrap();

        assert!(matches!(
            parsed.vendor_layer("rocm-runtime").unwrap_err(),
            BackendManifestError::UnknownVendorLayer { .. }
        ));
    }

    #[test]
    fn a_v1_manifest_defaults_to_empty_vendor_layers_and_no_vendor_layer_refs() {
        // Plain v1 JSON (no `vendor_layers` key at the top level, no
        // `vendor_layer` key on any backend entry) must still parse under
        // this v2-aware reader -- the whole point of #[serde(default)] on
        // both new fields is that old, already-released manifests keep
        // working forever without a version branch at the call site.
        let manifest_json = sample_manifest_json();
        let signature = sign(&manifest_json, TEST_MANIFEST_URL);
        let parsed = verify_and_parse_with_roots(
            manifest_json.as_bytes(),
            signature.as_bytes(),
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .unwrap();

        assert_eq!(parsed.schema_version, 1);
        assert!(parsed.vendor_layers.is_empty());
        let cuda = parsed.backend_entry("windows-x86_64", "cuda").unwrap();
        assert_eq!(cuda.vendor_layer, None);
    }

    #[test]
    fn rejects_unknown_top_level_fields_even_under_schema_v2() {
        // deny_unknown_fields must survive the v2 addition -- a manifest
        // shape drift (a field neither this schema nor its v2 extension
        // knows about) has to fail loudly, not silently drop data.
        let manifest_json = sample_manifest_v2_json().replacen(
            "\"schema_version\": 2,",
            "\"schema_version\": 2,\n            \"unexpected_field\": true,",
            1,
        );
        let signature = sign(&manifest_json, TEST_MANIFEST_URL);
        let error = verify_and_parse_with_roots(
            manifest_json.as_bytes(),
            signature.as_bytes(),
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .unwrap_err();
        assert!(matches!(error, BackendManifestError::Parse(_)), "{error:?}");
    }

    #[test]
    fn vendor_layer_verify_sha256_rejects_mismatch() {
        let manifest_json = sample_manifest_v2_json();
        let signature = sign(&manifest_json, TEST_MANIFEST_URL);
        let parsed = verify_and_parse_with_roots(
            manifest_json.as_bytes(),
            signature.as_bytes(),
            TEST_MANIFEST_URL,
            &test_roots(),
        )
        .unwrap();
        let vendor = parsed.vendor_layer("cuda-runtime").unwrap();
        let error = vendor
            .verify_sha256(b"not the real vendor bytes")
            .unwrap_err();
        assert!(matches!(error, BackendManifestError::Sha256Mismatch { .. }));
    }

    #[test]
    fn canonical_manifest_url_matches_the_documented_shape() {
        assert_eq!(
            canonical_manifest_url("0.1.20"),
            "https://dl.openasr.org/core/v0.1.20/backends-manifest.json"
        );
    }

    #[test]
    fn canonical_manifest_url_is_the_only_url_signature_verification_accepts() {
        // Reproduces the fix for #145: signing (and verifying) against the
        // canonical URL must succeed regardless of which mirror the bytes
        // were actually fetched from -- desktop is expected to always pass
        // `canonical_manifest_url(core_version)` as `expected_manifest_url`,
        // never the per-mirror fetch URL.
        let manifest_json = sample_manifest_v2_json();
        let canonical = canonical_manifest_url("0.1.20");
        let signature = sign(&manifest_json, &canonical);

        let parsed = verify_and_parse_with_roots(
            manifest_json.as_bytes(),
            signature.as_bytes(),
            &canonical,
            &test_roots(),
        )
        .expect("canonical URL must verify regardless of fetch mirror");
        assert_eq!(parsed.core_version, "0.1.20");

        // A signature minted for the canonical URL must NOT verify against a
        // *different* URL string (e.g. an accel-mirror or GitHub-fallback
        // fetch URL) -- that is the whole point of binding to one canonical
        // string rather than "whatever URL we fetched from".
        let mirror_url = "https://accel.example.com/core/v0.1.20/backends-manifest.json";
        let error = verify_and_parse_with_roots(
            manifest_json.as_bytes(),
            signature.as_bytes(),
            mirror_url,
            &test_roots(),
        )
        .unwrap_err();
        assert!(
            matches!(error, BackendManifestError::Signature(_)),
            "{error:?}"
        );
    }
}
