//! Signed, inert release artifacts available only to the qualification runner.
//!
//! This schema is intentionally not part of [`crate::ModelCatalog`]. It has no
//! activation mode, model, pack, or execution-policy fields and therefore
//! cannot become a second candidate-generation authority. Holding a verified
//! manifest proves only which immutable bytes a qualification child process is
//! allowed to download and load.

use std::{collections::BTreeSet, str::Utf8Error};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    BACKEND_HOST_ABI_SCHEMA_VERSION,
    qualification_manifest_security::{
        QualificationManifestSecurityError, VerifiedQualificationManifestSignature,
        render_qualification_manifest_signature, verify_qualification_manifest_signature,
    },
};

pub const QUALIFICATION_MANIFEST_SCHEMA_VERSION: u32 = 2;
pub const QUALIFICATION_ATTESTATION_REPOSITORY: &str = "QuintinShaw/openasr";
pub const QUALIFICATION_ATTESTATION_SIGNER_WORKFLOW: &str =
    "QuintinShaw/openasr/.github/workflows/release-binaries.yml";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationManifest {
    pub schema_version: u32,
    pub release_subject: String,
    pub host_abi: QualificationHostAbi,
    pub provider_target: QualificationProviderTarget,
    pub artifacts: QualificationArtifacts,
    pub attestation: QualificationAttestation,
}

/// Closed copy of the neutral host ABI identity. The ordinary catalog keeps
/// its own forward-compatible wire type; qualification deliberately denies
/// unknown nested fields so policy data cannot be hidden inside `host_abi`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationHostAbi {
    pub schema_version: u32,
    pub fingerprint: String,
    pub target: String,
    pub crt: String,
    pub toolchain: String,
    pub compile_flags_sha256: String,
    pub ggml_backend_api_version: u32,
    pub ggml_revision: String,
    pub ggml_headers_sha256: String,
    pub openasr_ffi_sha256: String,
    pub openasr_extension_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationProvider {
    Cuda,
    Hip,
    Vulkan,
    #[serde(other)]
    Unknown,
}

impl QualificationProvider {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Hip => "hip",
            Self::Vulkan => "vulkan",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationProviderTarget {
    pub provider: QualificationProvider,
    /// Immutable artifact target: one exact compiled target for CUDA/HIP (for
    /// example `sm_89` or `gfx1200`) or `generic` for the generic Vulkan
    /// plugin. This is not the live physical-device target. A manifest never
    /// represents an equivalence set; any cross-target projection is a later
    /// capability-finalizer decision with its own proof digest.
    pub target: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationArtifacts {
    /// Final host executable used for qualification. The executable is a
    /// member of `bundle`, because Windows release subjects are immutable ZIP
    /// archives rather than separately published executable assets.
    pub binary: QualificationBinaryArtifact,
    /// Neutral-dynamic backend plugin. Every qualification provider requires
    /// this artifact; physical Vulkan is qualified through the same generic,
    /// attested plugin chain as CUDA/HIP rather than a bundled fast path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin: Option<QualificationArtifact>,
    /// Content-addressed vendor runtime archives, if the plugin needs them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vendor: Vec<QualificationArtifact>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationBinaryArtifact {
    /// Basename of the executable member inside the release archive. The
    /// runner requires exactly one case-insensitive member with this name.
    pub file_name: String,
    pub sha256: String,
    pub size_bytes: u64,
    /// Attested immutable release subject containing the executable and every
    /// companion DLL. Its unpacked tree identity prevents qualification from
    /// combining the right executable with unbound sibling libraries.
    pub bundle: QualificationArtifact,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationArtifact {
    /// Basename only. It is diagnostic identity, never a local load path.
    pub file_name: String,
    pub format: QualificationArtifactFormat,
    pub sha256: String,
    pub size_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unpacked_size_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unpacked_tree_sha256: Option<String>,
    /// Immutable signed download locations. The runner chooses only from this
    /// list and never accepts a caller-provided plugin path.
    pub urls: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationArtifactFormat {
    NativeLibrary,
    ZipArchive,
    AttestationBundle,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationAttestation {
    pub predicate_type: String,
    pub repository: String,
    pub signer_workflow: String,
    pub source_digest: String,
    pub deny_self_hosted_runners: bool,
    pub bundle: QualificationArtifact,
}

/// Typestate returned only after the detached production signature and every
/// inert schema invariant have been verified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedQualificationManifest {
    manifest: QualificationManifest,
    signature: VerifiedQualificationManifestSignature,
}

impl VerifiedQualificationManifest {
    pub fn manifest(&self) -> &QualificationManifest {
        &self.manifest
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.signature.manifest_sha256
    }

    pub fn signature_key_id(&self) -> &str {
        &self.signature.key_id
    }
}

#[derive(Debug, Error)]
pub enum QualificationManifestError {
    #[error("qualification manifest bytes are not valid UTF-8: {0}")]
    InvalidUtf8(#[from] Utf8Error),
    #[error("qualification manifest signature rejected: {0}")]
    Signature(#[source] QualificationManifestSecurityError),
    #[error("could not parse qualification manifest JSON: {0}")]
    Parse(#[source] serde_json::Error),
    #[error("unsupported qualification manifest schema_version {found}")]
    UnsupportedSchema { found: u32 },
    #[error("qualification manifest field '{field}' is invalid: {reason}")]
    InvalidField { field: &'static str, reason: String },
    #[error("qualification artifact packaging is invalid: {reason}")]
    InvalidPackaging { reason: String },
}

#[derive(Debug, Error)]
pub enum QualificationManifestSigningError {
    #[error("qualification manifest schema is not safe to sign: {0}")]
    Manifest(#[source] QualificationManifestError),
    #[error("could not sign qualification manifest: {0}")]
    Signature(#[source] QualificationManifestSecurityError),
}

pub fn verify_and_parse_qualification_manifest(
    manifest_bytes: &[u8],
    signature_bytes: &[u8],
    expected_manifest_url: &str,
) -> Result<VerifiedQualificationManifest, QualificationManifestError> {
    let manifest_text = std::str::from_utf8(manifest_bytes)?;
    let signature_text = std::str::from_utf8(signature_bytes)?;
    let signature = verify_qualification_manifest_signature(
        manifest_text,
        signature_text,
        expected_manifest_url,
    )
    .map_err(QualificationManifestError::Signature)?;
    let manifest = parse_and_validate_qualification_manifest(manifest_text)?;
    validate_canonical_manifest_url(&manifest, expected_manifest_url)?;
    Ok(VerifiedQualificationManifest {
        manifest,
        signature,
    })
}

/// Parse and validate an unsigned body before the maintainer signs it. The
/// returned plain value carries no verification typestate and must never be
/// accepted by a qualification runner or ordinary runtime.
pub(crate) fn validate_qualification_manifest_for_signing(
    manifest_contents: &str,
) -> Result<QualificationManifest, QualificationManifestError> {
    parse_and_validate_qualification_manifest(manifest_contents)
}

/// The only public qualification signing API. It validates the closed inert
/// schema before invoking the domain-separated signer, so callers cannot sign
/// arbitrary bytes while accidentally bypassing the policy-field guard.
pub fn render_validated_qualification_manifest_signature(
    manifest_contents: &str,
    manifest_url: &str,
    key_id: &str,
    signing_key_seed_hex: &str,
) -> Result<String, QualificationManifestSigningError> {
    let manifest = validate_qualification_manifest_for_signing(manifest_contents)
        .map_err(QualificationManifestSigningError::Manifest)?;
    validate_canonical_manifest_url(&manifest, manifest_url)
        .map_err(QualificationManifestSigningError::Manifest)?;
    let signature = render_qualification_manifest_signature(
        manifest_contents,
        manifest_url,
        key_id,
        signing_key_seed_hex,
    )
    .map_err(QualificationManifestSigningError::Signature)?;
    verify_qualification_manifest_signature(manifest_contents, &signature, manifest_url)
        .map_err(QualificationManifestSigningError::Signature)?;
    Ok(signature)
}

fn parse_and_validate_qualification_manifest(
    manifest_text: &str,
) -> Result<QualificationManifest, QualificationManifestError> {
    let manifest: QualificationManifest =
        serde_json::from_str(manifest_text).map_err(QualificationManifestError::Parse)?;
    manifest.validate()?;
    Ok(manifest)
}

impl QualificationManifest {
    fn validate(&self) -> Result<(), QualificationManifestError> {
        if self.schema_version != QUALIFICATION_MANIFEST_SCHEMA_VERSION {
            return Err(QualificationManifestError::UnsupportedSchema {
                found: self.schema_version,
            });
        }
        validate_release_subject(&self.release_subject)?;
        validate_host_abi(&self.host_abi)?;
        if self.host_abi.target != "x86_64-pc-windows-msvc" {
            return Err(invalid(
                "host_abi.target",
                "qualification v2 requires the x86_64 Windows MSVC release host",
            ));
        }
        if self.host_abi.crt != "msvc-md" {
            return Err(invalid(
                "host_abi.crt",
                "qualification v2 requires the dynamic MSVC CRT",
            ));
        }
        if self.provider_target.provider == QualificationProvider::Unknown {
            return Err(invalid("provider_target.provider", "unknown provider"));
        }
        require_token("provider_target.target", &self.provider_target.target)?;
        validate_provider_target(&self.provider_target)?;
        self.artifacts.binary.validate()?;
        if let Some(plugin) = &self.artifacts.plugin {
            plugin.validate(
                "artifacts.plugin",
                QualificationArtifactFormat::NativeLibrary,
            )?;
        }
        for vendor in &self.artifacts.vendor {
            vendor.validate("artifacts.vendor", QualificationArtifactFormat::ZipArchive)?;
        }
        if self.attestation.predicate_type != "https://slsa.dev/provenance/v1" {
            return Err(invalid(
                "attestation.predicate_type",
                "must be https://slsa.dev/provenance/v1",
            ));
        }
        if self.attestation.repository != QUALIFICATION_ATTESTATION_REPOSITORY {
            return Err(invalid(
                "attestation.repository",
                format!("must be {QUALIFICATION_ATTESTATION_REPOSITORY}"),
            ));
        }
        if self.attestation.signer_workflow != QUALIFICATION_ATTESTATION_SIGNER_WORKFLOW {
            return Err(invalid(
                "attestation.signer_workflow",
                format!("must be {QUALIFICATION_ATTESTATION_SIGNER_WORKFLOW}"),
            ));
        }
        require_lower_hex(
            "attestation.source_digest",
            &self.attestation.source_digest,
            40,
        )?;
        if !self.attestation.deny_self_hosted_runners {
            return Err(invalid(
                "attestation.deny_self_hosted_runners",
                "must be true",
            ));
        }
        self.attestation.bundle.validate(
            "attestation.bundle",
            QualificationArtifactFormat::AttestationBundle,
        )?;
        self.validate_packaging()?;
        self.validate_unique_artifacts()?;
        Ok(())
    }

    fn validate_packaging(&self) -> Result<(), QualificationManifestError> {
        let version = self
            .release_subject
            .strip_prefix('v')
            .expect("release subject was validated before packaging");
        if self.artifacts.binary.file_name != "openasr.exe" {
            return Err(invalid(
                "artifacts.binary.file_name",
                "qualification v2 requires the openasr.exe member",
            ));
        }
        let expected_binary_bundle = format!("openasr-{version}-windows-x86_64-neutral.zip");
        if self.artifacts.binary.bundle.file_name != expected_binary_bundle {
            return Err(invalid(
                "artifacts.binary.bundle.file_name",
                format!("must bind {expected_binary_bundle}"),
            ));
        }
        let expected_attestation_bundle = format!("openasr-{version}-build-provenance.bundle.json");
        if self.attestation.bundle.file_name != expected_attestation_bundle {
            return Err(invalid(
                "attestation.bundle.file_name",
                format!("must bind {expected_attestation_bundle}"),
            ));
        }
        let plugin = self.artifacts.plugin.as_ref().ok_or_else(|| {
            QualificationManifestError::InvalidPackaging {
                reason: format!(
                    "{} qualification requires a neutral-dynamic plugin artifact",
                    self.provider_target.provider.as_str()
                ),
            }
        })?;
        if self.artifacts.vendor.is_empty() {
            return Err(QualificationManifestError::InvalidPackaging {
                reason: format!(
                    "{} qualification requires at least one signed vendor artifact",
                    self.provider_target.provider.as_str()
                ),
            });
        }
        let expected_plugin = format!(
            "openasr-{version}-windows-x86_64-{}-{}-plugin.dll",
            plugin_release_prefix(self.provider_target.provider),
            self.provider_target.target
        );
        if plugin.file_name != expected_plugin {
            return Err(invalid(
                "artifacts.plugin.file_name",
                format!("must bind exact provider/artifact target as {expected_plugin}"),
            ));
        }
        for vendor in &self.artifacts.vendor {
            validate_vendor_artifact_name(vendor, vendor_layer_key(self.provider_target.provider))?;
        }
        let artifacts = std::iter::once(&self.artifacts.binary.bundle)
            .chain(self.artifacts.plugin.iter())
            .chain(self.artifacts.vendor.iter())
            .chain(std::iter::once(&self.attestation.bundle));
        for artifact in artifacts {
            validate_release_artifact_urls(&self.release_subject, artifact)?;
        }
        Ok(())
    }

    fn validate_unique_artifacts(&self) -> Result<(), QualificationManifestError> {
        let mut windows_names = BTreeSet::new();
        let mut digests = BTreeSet::new();
        windows_names.insert(self.artifacts.binary.file_name.to_ascii_lowercase());
        digests.insert(self.artifacts.binary.sha256.as_str());
        let artifacts = std::iter::once(&self.artifacts.binary.bundle)
            .chain(self.artifacts.plugin.iter())
            .chain(self.artifacts.vendor.iter())
            .chain(std::iter::once(&self.attestation.bundle));
        for artifact in artifacts {
            if !windows_names.insert(artifact.file_name.to_ascii_lowercase()) {
                return Err(invalid("artifacts", "artifact basenames must be unique"));
            }
            if !digests.insert(artifact.sha256.as_str()) {
                return Err(invalid(
                    "artifacts",
                    "artifact sha256 identities must be unique",
                ));
            }
        }
        Ok(())
    }
}

fn plugin_release_prefix(provider: QualificationProvider) -> &'static str {
    match provider {
        QualificationProvider::Cuda => "cuda",
        QualificationProvider::Hip => "rocm",
        QualificationProvider::Vulkan => "vulkan",
        QualificationProvider::Unknown => {
            unreachable!("rejected before packaging validation")
        }
    }
}

fn vendor_layer_key(provider: QualificationProvider) -> &'static str {
    match provider {
        QualificationProvider::Cuda => "cuda-runtime",
        QualificationProvider::Hip => "rocm-runtime",
        QualificationProvider::Vulkan => "vulkan-loader",
        QualificationProvider::Unknown => {
            unreachable!("rejected before packaging validation")
        }
    }
}

fn validate_vendor_artifact_name(
    artifact: &QualificationArtifact,
    vendor_layer: &str,
) -> Result<(), QualificationManifestError> {
    let prefix = format!("openasr-vendor-{vendor_layer}-");
    let short_digest = artifact
        .file_name
        .strip_prefix(&prefix)
        .and_then(|value| value.strip_suffix(".zip"))
        .ok_or_else(|| {
            invalid(
                "artifacts.vendor.file_name",
                format!("must use {prefix}<sha12>.zip"),
            )
        })?;
    if short_digest.len() != 12
        || !short_digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || !artifact.sha256.starts_with(short_digest)
    {
        return Err(invalid(
            "artifacts.vendor.file_name",
            "content-addressed suffix must equal the artifact sha256 prefix",
        ));
    }
    Ok(())
}

fn validate_release_artifact_urls(
    release_subject: &str,
    artifact: &QualificationArtifact,
) -> Result<(), QualificationManifestError> {
    let expected = [
        format!(
            "https://dl.openasr.org/core/{release_subject}/{}",
            artifact.file_name
        ),
        format!(
            "https://github.com/QuintinShaw/openasr/releases/download/{release_subject}/{}",
            artifact.file_name
        ),
    ];
    if artifact.urls != expected {
        return Err(invalid(
            "artifacts.urls",
            "must contain the canonical CDN URL followed by the immutable GitHub release mirror",
        ));
    }
    Ok(())
}

fn qualification_manifest_asset_file_name(manifest: &QualificationManifest) -> String {
    let version = manifest
        .release_subject
        .strip_prefix('v')
        .expect("validated qualification release subject has a v prefix");
    let cell = format!(
        "{}-{}",
        manifest.provider_target.provider.as_str(),
        manifest.provider_target.target
    );
    format!("openasr-{version}-qualification-{cell}.json")
}

fn validate_release_subject(value: &str) -> Result<(), QualificationManifestError> {
    let Some(version) = value.strip_prefix('v') else {
        return Err(invalid("release_subject", "must use canonical vX.Y.Z"));
    };
    let parts = version.split('.').collect::<Vec<_>>();
    if parts.len() != 3
        || parts.iter().any(|part| {
            part.is_empty()
                || !part.bytes().all(|byte| byte.is_ascii_digit())
                || (part.len() > 1 && part.starts_with('0'))
                || part.parse::<u64>().is_err()
        })
    {
        return Err(invalid("release_subject", "must use canonical vX.Y.Z"));
    }
    Ok(())
}

fn validate_canonical_manifest_url(
    manifest: &QualificationManifest,
    manifest_url: &str,
) -> Result<(), QualificationManifestError> {
    let expected = format!(
        "https://dl.openasr.org/core/{}/{}",
        manifest.release_subject,
        qualification_manifest_asset_file_name(manifest)
    );
    if manifest_url != expected {
        return Err(invalid(
            "manifest_url",
            format!("must bind the exact release/cell URL {expected}"),
        ));
    }
    Ok(())
}

impl QualificationBinaryArtifact {
    fn validate(&self) -> Result<(), QualificationManifestError> {
        require_file_name("artifacts.binary.file_name", &self.file_name)?;
        if !self.file_name.to_ascii_lowercase().ends_with(".exe") {
            return Err(invalid(
                "artifacts.binary.file_name",
                "Windows qualification binary must use an .exe basename",
            ));
        }
        require_lower_hex("artifacts.binary.sha256", &self.sha256, 64)?;
        if self.size_bytes == 0 {
            return Err(invalid(
                "artifacts.binary.size_bytes",
                "must be greater than zero",
            ));
        }
        self.bundle.validate(
            "artifacts.binary.bundle",
            QualificationArtifactFormat::ZipArchive,
        )
    }
}

impl QualificationArtifact {
    fn validate(
        &self,
        field: &'static str,
        expected_format: QualificationArtifactFormat,
    ) -> Result<(), QualificationManifestError> {
        require_file_name(field, &self.file_name)?;
        if self.format != expected_format || self.format == QualificationArtifactFormat::Unknown {
            return Err(invalid(
                field,
                format!(
                    "artifact format {:?} does not match required role {:?}",
                    self.format, expected_format
                ),
            ));
        }
        let lower_name = self.file_name.to_ascii_lowercase();
        let extension_matches = match self.format {
            QualificationArtifactFormat::NativeLibrary => lower_name.ends_with(".dll"),
            QualificationArtifactFormat::ZipArchive => lower_name.ends_with(".zip"),
            QualificationArtifactFormat::AttestationBundle => {
                lower_name.ends_with(".json") || lower_name.ends_with(".jsonl")
            }
            QualificationArtifactFormat::Unknown => false,
        };
        if !extension_matches {
            return Err(invalid(
                field,
                "artifact basename does not match its format",
            ));
        }
        require_lower_hex(field, &self.sha256, 64)?;
        if self.size_bytes == 0 {
            return Err(invalid(field, "size_bytes must be greater than zero"));
        }
        if self.urls.is_empty() {
            return Err(invalid(
                field,
                "at least one immutable HTTPS URL is required",
            ));
        }
        let mut seen = BTreeSet::new();
        for url in &self.urls {
            if !seen.insert(url.as_str()) {
                return Err(invalid(field, "download URLs must be unique"));
            }
            validate_download_url(field, url, &self.file_name)?;
        }
        match self.format {
            QualificationArtifactFormat::ZipArchive => {
                if self.unpacked_size_bytes.is_none_or(|size| size == 0) {
                    return Err(invalid(
                        field,
                        "zip archive requires non-zero unpacked_size_bytes",
                    ));
                }
                require_lower_hex(
                    field,
                    self.unpacked_tree_sha256.as_deref().unwrap_or_default(),
                    64,
                )?;
            }
            QualificationArtifactFormat::NativeLibrary
            | QualificationArtifactFormat::AttestationBundle => {
                if self.unpacked_size_bytes.is_some() || self.unpacked_tree_sha256.is_some() {
                    return Err(invalid(
                        field,
                        "non-archive artifact cannot declare unpacked identity",
                    ));
                }
            }
            QualificationArtifactFormat::Unknown => {
                return Err(invalid(field, "unknown artifact format"));
            }
        }
        Ok(())
    }
}

fn validate_provider_target(
    target: &QualificationProviderTarget,
) -> Result<(), QualificationManifestError> {
    let valid = match target.provider {
        QualificationProvider::Cuda => target.target.strip_prefix("sm_").is_some_and(|suffix| {
            matches!(suffix.len(), 2 | 3) && suffix.bytes().all(|byte| byte.is_ascii_digit())
        }),
        QualificationProvider::Hip => target.target.strip_prefix("gfx").is_some_and(|suffix| {
            (3..=8).contains(&suffix.len())
                && suffix
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        }),
        QualificationProvider::Vulkan => target.target == "generic",
        QualificationProvider::Unknown => false,
    };
    valid.then_some(()).ok_or_else(|| {
        invalid(
            "provider_target.target",
            format!(
                "target is not canonical for provider {}",
                target.provider.as_str()
            ),
        )
    })
}

fn validate_host_abi(host_abi: &QualificationHostAbi) -> Result<(), QualificationManifestError> {
    if host_abi.schema_version != BACKEND_HOST_ABI_SCHEMA_VERSION {
        return Err(invalid(
            "host_abi.schema_version",
            format!(
                "expected {BACKEND_HOST_ABI_SCHEMA_VERSION}, got {}",
                host_abi.schema_version
            ),
        ));
    }
    for (field, value) in [
        ("host_abi.fingerprint", host_abi.fingerprint.as_str()),
        (
            "host_abi.compile_flags_sha256",
            host_abi.compile_flags_sha256.as_str(),
        ),
        (
            "host_abi.ggml_headers_sha256",
            host_abi.ggml_headers_sha256.as_str(),
        ),
        (
            "host_abi.openasr_ffi_sha256",
            host_abi.openasr_ffi_sha256.as_str(),
        ),
        (
            "host_abi.openasr_extension_sha256",
            host_abi.openasr_extension_sha256.as_str(),
        ),
    ] {
        require_lower_hex(field, value, 64)?;
    }
    require_token("host_abi.target", &host_abi.target)?;
    require_token("host_abi.crt", &host_abi.crt)?;
    require_token("host_abi.toolchain", &host_abi.toolchain)?;
    require_lower_hex("host_abi.ggml_revision", &host_abi.ggml_revision, 40)?;
    if host_abi.ggml_backend_api_version == 0 {
        return Err(invalid(
            "host_abi.ggml_backend_api_version",
            "must be greater than zero",
        ));
    }
    Ok(())
}

fn validate_download_url(
    field: &'static str,
    value: &str,
    expected_file_name: &str,
) -> Result<(), QualificationManifestError> {
    let parsed = reqwest::Url::parse(value)
        .map_err(|error| invalid(field, format!("invalid URL: {error}")))?;
    if parsed.scheme() != "https"
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || parsed
            .path_segments()
            .and_then(|mut segments| segments.next_back())
            != Some(expected_file_name)
    {
        return Err(invalid(
            field,
            "URL must be credential-free HTTPS, have no fragment, and end with the signed basename",
        ));
    }
    Ok(())
}

fn require_text(field: &'static str, value: &str) -> Result<(), QualificationManifestError> {
    if value.trim().is_empty() || value.trim() != value || value.contains(['\r', '\n']) {
        Err(invalid(
            field,
            "must be non-empty, trimmed, single-line text",
        ))
    } else {
        Ok(())
    }
}

fn require_token(field: &'static str, value: &str) -> Result<(), QualificationManifestError> {
    require_text(field, value)?;
    if value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':'))
    {
        Ok(())
    } else {
        Err(invalid(field, "contains unsupported characters"))
    }
}

fn require_file_name(field: &'static str, value: &str) -> Result<(), QualificationManifestError> {
    require_text(field, value)?;
    let allowed = value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
    let windows_stem = value.split('.').next().unwrap_or_default();
    let upper_stem = windows_stem.to_ascii_uppercase();
    let windows_device = matches!(upper_stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || upper_stem.strip_prefix("COM").is_some_and(|suffix| {
            matches!(suffix, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
        })
        || upper_stem.strip_prefix("LPT").is_some_and(|suffix| {
            matches!(suffix, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
        });
    if !allowed
        || value.starts_with('.')
        || value.ends_with('.')
        || windows_device
        || value == "."
        || value == ".."
    {
        Err(invalid(field, "must be a safe basename"))
    } else {
        Ok(())
    }
}

fn require_lower_hex(
    field: &'static str,
    value: &str,
    length: usize,
) -> Result<(), QualificationManifestError> {
    if value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(invalid(
            field,
            format!("must be {length} lowercase hexadecimal characters"),
        ))
    }
}

fn invalid(field: &'static str, reason: impl Into<String>) -> QualificationManifestError {
    QualificationManifestError::InvalidField {
        field,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        catalog_security::CatalogTrustRoot,
        derive_catalog_public_key_hex,
        qualification_manifest_security::{
            render_qualification_manifest_signature,
            verify_qualification_manifest_signature_with_roots,
        },
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

    fn artifact(file_name: &str, fill: char) -> serde_json::Value {
        let format = if file_name.ends_with(".dll") {
            "native_library"
        } else if file_name.ends_with(".zip") {
            "zip_archive"
        } else {
            "attestation_bundle"
        };
        let (unpacked_size_bytes, unpacked_tree_sha256) = if format == "zip_archive" {
            (serde_json::json!(1), serde_json::json!("c".repeat(64)))
        } else {
            (serde_json::Value::Null, serde_json::Value::Null)
        };
        serde_json::json!({
            "file_name": file_name,
            "format": format,
            "sha256": fill.to_string().repeat(64),
            "size_bytes": 1,
            "unpacked_size_bytes": unpacked_size_bytes,
            "unpacked_tree_sha256": unpacked_tree_sha256,
            "urls": [
                format!("https://dl.openasr.org/core/v0.1.37/{file_name}"),
                format!("https://github.com/QuintinShaw/openasr/releases/download/v0.1.37/{file_name}"),
            ],
        })
    }

    fn binary_artifact() -> serde_json::Value {
        serde_json::json!({
            "file_name": "openasr.exe",
            "sha256": "7".repeat(64),
            "size_bytes": 1,
            "bundle": artifact("openasr-0.1.37-windows-x86_64-neutral.zip", 'd'),
        })
    }

    fn manifest_value() -> serde_json::Value {
        serde_json::json!({
            "schema_version": QUALIFICATION_MANIFEST_SCHEMA_VERSION,
            "release_subject": "v0.1.37",
            "host_abi": {
                "schema_version": BACKEND_HOST_ABI_SCHEMA_VERSION,
                "fingerprint": "1".repeat(64),
                "target": "x86_64-pc-windows-msvc",
                "crt": "msvc-md",
                "toolchain": "msvc-v143",
                "compile_flags_sha256": "2".repeat(64),
                "ggml_backend_api_version": 3,
                "ggml_revision": "3".repeat(40),
                "ggml_headers_sha256": "4".repeat(64),
                "openasr_ffi_sha256": "5".repeat(64),
                "openasr_extension_sha256": "6".repeat(64),
            },
            "provider_target": {"provider": "cuda", "target": "sm_89"},
            "artifacts": {
                "binary": binary_artifact(),
                "plugin": artifact("openasr-0.1.37-windows-x86_64-cuda-sm_89-plugin.dll", '8'),
                "vendor": [artifact("openasr-vendor-cuda-runtime-999999999999.zip", '9')],
            },
            "attestation": {
                "predicate_type": "https://slsa.dev/provenance/v1",
                "repository": QUALIFICATION_ATTESTATION_REPOSITORY,
                "signer_workflow": QUALIFICATION_ATTESTATION_SIGNER_WORKFLOW,
                "source_digest": "b".repeat(40),
                "deny_self_hosted_runners": true,
                "bundle": artifact("openasr-0.1.37-build-provenance.bundle.json", 'a'),
            },
        })
    }

    fn verify_with_test_root(
        value: &serde_json::Value,
    ) -> Result<QualificationManifest, QualificationManifestError> {
        let body = serde_json::to_string(value).expect("serialize manifest");
        let signature = render_qualification_manifest_signature(&body, URL, TEST_KEY_ID, TEST_SEED)
            .expect("sign manifest");
        let verified =
            verify_qualification_manifest_signature_with_roots(&body, &signature, URL, &roots())
                .expect("verify test signature");
        assert_eq!(verified.key_id, TEST_KEY_ID);
        parse_and_validate_qualification_manifest(&body)
    }

    #[test]
    fn signed_manifest_contains_only_inert_artifact_facts() {
        let manifest = verify_with_test_root(&manifest_value()).expect("valid manifest");
        assert_eq!(
            manifest.provider_target.provider,
            QualificationProvider::Cuda
        );
        assert_eq!(manifest.provider_target.target, "sm_89");
        assert!(manifest.artifacts.plugin.is_some());
        validate_canonical_manifest_url(&manifest, URL).expect("canonical cell URL");
        assert!(
            validate_canonical_manifest_url(
                &manifest,
                "https://dl.openasr.org/core/v0.1.37/openasr-0.1.37-qualification-cuda-sm_90.json",
            )
            .is_err()
        );
    }

    #[test]
    fn release_subject_is_a_canonical_stable_semver() {
        for release_subject in ["0.1.37", "v0.1.37-alpha.1", "v01.1.37"] {
            let mut manifest = manifest_value();
            manifest["release_subject"] = serde_json::json!(release_subject);
            assert!(
                verify_with_test_root(&manifest).is_err(),
                "{release_subject}"
            );
        }
    }

    #[test]
    fn activation_and_local_plugin_path_fields_are_rejected() {
        for (field, value) in [
            ("activation_modes", serde_json::json!(["explicit"])),
            ("plugin_path", serde_json::json!("C:\\temp\\plugin.dll")),
        ] {
            let mut manifest = manifest_value();
            manifest
                .as_object_mut()
                .expect("object")
                .insert(field.to_string(), value);
            assert!(matches!(
                verify_with_test_root(&manifest),
                Err(QualificationManifestError::Parse(_))
            ));
        }
    }

    #[test]
    fn public_signer_cannot_sign_policy_fields() {
        let mut manifest = manifest_value();
        manifest["activation_modes"] = serde_json::json!(["explicit"]);
        let body = serde_json::to_string(&manifest).expect("serialize unsafe manifest");
        assert!(matches!(
            render_validated_qualification_manifest_signature(&body, URL, TEST_KEY_ID, TEST_SEED,),
            Err(QualificationManifestSigningError::Manifest(_))
        ));
    }

    #[test]
    fn nested_host_abi_cannot_hide_policy_fields() {
        let mut manifest = manifest_value();
        manifest["host_abi"]["activation_modes"] = serde_json::json!(["explicit"]);
        assert!(matches!(
            verify_with_test_root(&manifest),
            Err(QualificationManifestError::Parse(_))
        ));
    }

    #[test]
    fn attestation_identity_is_pinned_and_self_hosted_is_denied() {
        for (field, value) in [
            (
                "predicate_type",
                serde_json::json!("https://example.com/provenance/v1"),
            ),
            ("repository", serde_json::json!("attacker/repo")),
            (
                "signer_workflow",
                serde_json::json!("attacker/repo/.github/workflows/build.yml"),
            ),
            ("source_digest", serde_json::json!("c".repeat(39))),
            ("deny_self_hosted_runners", serde_json::json!(false)),
        ] {
            let mut manifest = manifest_value();
            manifest["attestation"][field] = value;
            assert!(matches!(
                verify_with_test_root(&manifest),
                Err(QualificationManifestError::InvalidField { .. })
            ));
        }
    }

    #[test]
    fn vulkan_generic_uses_the_same_plugin_and_vendor_chain() {
        let mut vulkan = manifest_value();
        vulkan["provider_target"]["provider"] = serde_json::json!("vulkan");
        vulkan["provider_target"]["target"] = serde_json::json!("generic");
        vulkan["artifacts"]["plugin"] = artifact(
            "openasr-0.1.37-windows-x86_64-vulkan-generic-plugin.dll",
            '8',
        );
        vulkan["artifacts"]["vendor"] = serde_json::json!([artifact(
            "openasr-vendor-vulkan-loader-999999999999.zip",
            '9'
        )]);
        assert!(verify_with_test_root(&vulkan).is_ok());

        let mut live_target = vulkan.clone();
        live_target["provider_target"]["target"] =
            serde_json::json!("vk_caps_00001002_0000744c_00112233445566778899aabbccddeeff");
        assert!(matches!(
            verify_with_test_root(&live_target),
            Err(QualificationManifestError::InvalidField { .. })
        ));

        vulkan["artifacts"]
            .as_object_mut()
            .expect("artifacts")
            .remove("plugin");
        assert!(matches!(
            verify_with_test_root(&vulkan),
            Err(QualificationManifestError::InvalidPackaging { .. })
        ));

        let mut hip = manifest_value();
        hip["provider_target"]["provider"] = serde_json::json!("hip");
        hip["provider_target"]["target"] = serde_json::json!("gfx1200");
        hip["artifacts"]
            .as_object_mut()
            .expect("artifacts")
            .remove("plugin");
        assert!(matches!(
            verify_with_test_root(&hip),
            Err(QualificationManifestError::InvalidPackaging { .. })
        ));
    }

    #[test]
    fn provider_targets_and_windows_artifact_extensions_are_canonical() {
        for (provider, target) in [
            ("cuda", "gfx1200"),
            ("hip", "sm_89"),
            ("vulkan", "vulkan-any"),
        ] {
            let mut manifest = manifest_value();
            manifest["provider_target"]["provider"] = serde_json::json!(provider);
            manifest["provider_target"]["target"] = serde_json::json!(target);
            assert!(matches!(
                verify_with_test_root(&manifest),
                Err(QualificationManifestError::InvalidField { .. })
            ));
        }

        let mut wrong_plugin_extension = manifest_value();
        wrong_plugin_extension["artifacts"]["plugin"]["file_name"] =
            serde_json::json!("openasr-cuda.bin");
        wrong_plugin_extension["artifacts"]["plugin"]["urls"] =
            serde_json::json!(["https://dl.openasr.org/core/v0.1.37/openasr-cuda.bin"]);
        assert!(matches!(
            verify_with_test_root(&wrong_plugin_extension),
            Err(QualificationManifestError::InvalidField { .. })
        ));

        for (field, value) in [
            ("target", "aarch64-pc-windows-msvc"),
            ("crt", "msvc-static"),
        ] {
            let mut manifest = manifest_value();
            manifest["host_abi"][field] = serde_json::json!(value);
            assert!(matches!(
                verify_with_test_root(&manifest),
                Err(QualificationManifestError::InvalidField { .. })
            ));
        }
    }

    #[test]
    fn release_artifact_names_bind_version_provider_target_and_content_digest() {
        let mut mutations = Vec::new();

        let mut wrong_binary = manifest_value();
        wrong_binary["artifacts"]["binary"]["file_name"] = serde_json::json!("other-openasr.exe");
        mutations.push(wrong_binary);

        let mut wrong_bundle = manifest_value();
        wrong_bundle["artifacts"]["binary"]["bundle"] =
            artifact("openasr-0.1.38-windows-x86_64-neutral.zip", 'd');
        mutations.push(wrong_bundle);

        let mut wrong_plugin_target = manifest_value();
        wrong_plugin_target["artifacts"]["plugin"] =
            artifact("openasr-0.1.37-windows-x86_64-cuda-sm_90-plugin.dll", '8');
        mutations.push(wrong_plugin_target);

        let mut wrong_vendor_provider = manifest_value();
        wrong_vendor_provider["artifacts"]["vendor"][0] =
            artifact("openasr-vendor-rocm-runtime-999999999999.zip", '9');
        mutations.push(wrong_vendor_provider);

        let mut wrong_vendor_digest = manifest_value();
        wrong_vendor_digest["artifacts"]["vendor"][0] =
            artifact("openasr-vendor-cuda-runtime-888888888888.zip", '9');
        mutations.push(wrong_vendor_digest);

        let mut wrong_attestation = manifest_value();
        wrong_attestation["attestation"]["bundle"] =
            artifact("openasr-0.1.38-build-provenance.bundle.json", 'a');
        mutations.push(wrong_attestation);

        for manifest in mutations {
            assert!(matches!(
                verify_with_test_root(&manifest),
                Err(QualificationManifestError::InvalidField { .. })
            ));
        }
    }

    #[test]
    fn artifact_roles_and_archive_tree_identity_are_closed() {
        let mut wrong_role = manifest_value();
        wrong_role["artifacts"]["plugin"]["format"] = serde_json::json!("zip_archive");
        assert!(matches!(
            verify_with_test_root(&wrong_role),
            Err(QualificationManifestError::InvalidField { .. })
        ));

        for missing in ["unpacked_size_bytes", "unpacked_tree_sha256"] {
            let mut manifest = manifest_value();
            manifest["artifacts"]["vendor"][0]
                .as_object_mut()
                .expect("vendor artifact")
                .remove(missing);
            assert!(matches!(
                verify_with_test_root(&manifest),
                Err(QualificationManifestError::InvalidField { .. })
            ));
        }

        let mut binary_bundle_without_tree = manifest_value();
        binary_bundle_without_tree["artifacts"]["binary"]["bundle"]
            .as_object_mut()
            .expect("binary bundle")
            .remove("unpacked_tree_sha256");
        assert!(matches!(
            verify_with_test_root(&binary_bundle_without_tree),
            Err(QualificationManifestError::InvalidField { .. })
        ));
    }

    #[test]
    fn manifest_rejects_mutable_or_path_like_download_authority() {
        for url in [
            "file:///C:/temp/openasr-0.1.37-windows-x86_64-cuda-sm_89-plugin.dll",
            "https://dl.openasr.org/core/v0.1.37/openasr-0.1.37-windows-x86_64-cuda-sm_89-plugin.dll?download=temporary",
            "https://dl.openasr.org/core/v0.1.37/different.dll",
            "https://example.com/openasr-0.1.37-windows-x86_64-cuda-sm_89-plugin.dll",
        ] {
            let mut manifest = manifest_value();
            manifest["artifacts"]["plugin"]["urls"] = serde_json::json!([url]);
            assert!(matches!(
                verify_with_test_root(&manifest),
                Err(QualificationManifestError::InvalidField { .. })
            ));
        }
    }

    #[test]
    fn manifest_rejects_windows_unsafe_or_colliding_basenames() {
        for file_name in [
            "C:plugin.dll",
            "CON",
            "nul.dll",
            "COM1.dll",
            "LPT9.txt",
            "plugin.dll.",
            "plugin name.dll",
        ] {
            let mut manifest = manifest_value();
            manifest["artifacts"]["plugin"]["file_name"] = serde_json::json!(file_name);
            manifest["artifacts"]["plugin"]["urls"] =
                serde_json::json!([format!("https://dl.openasr.org/core/v0.1.37/{file_name}")]);
            assert!(matches!(
                verify_with_test_root(&manifest),
                Err(QualificationManifestError::InvalidField { .. })
            ));
        }

        let mut colliding = manifest_value();
        colliding["artifacts"]["plugin"]["file_name"] = serde_json::json!("OPENASR.EXE");
        colliding["artifacts"]["plugin"]["urls"] =
            serde_json::json!(["https://dl.openasr.org/core/v0.1.37/OPENASR.EXE"]);
        assert!(matches!(
            verify_with_test_root(&colliding),
            Err(QualificationManifestError::InvalidField { .. })
        ));
    }
}
