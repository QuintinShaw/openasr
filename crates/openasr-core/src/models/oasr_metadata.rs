use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::NativeAsrError;
use crate::ggml_runtime::{GgufMetadata, GgufWriteTensor, GgufWriteValue};
use thiserror::Error;

// Build provenance lives at the single GGUF write choke point (ggml_runtime::
// gguf_write), not in this per-family metadata module; re-exported here so
// every `openasr.*` pack-metadata key is discoverable in one place.
pub use crate::ggml_runtime::{BUILD_COMMIT_ENV, OASR_METADATA_KEY_BUILD_COMMIT};

pub const OASR_METADATA_KEY_PACKAGE_VERSION: &str = "openasr.package.version";
pub const OASR_METADATA_KEY_MODEL_FAMILY: &str = "openasr.model.family";
pub const OASR_METADATA_KEY_MODEL_ARCHITECTURE: &str = "openasr.model.architecture";
pub const OASR_METADATA_KEY_AUDIO_FRONTEND: &str = "openasr.audio.frontend";
pub const OASR_METADATA_KEY_DECODE_POLICY: &str = "openasr.decode.policy";
/// Self-description written by an auxiliary diarization pack (the pyannote
/// segmenter) so `openasr model inspect` can name what a pack is. NOT a
/// capability judge: which ASR family carries speaker structure in its own
/// decode is declared once on the arch descriptor
/// (`arch::SpeakerSegmentationSource`), never re-derived from pack metadata --
/// a published pack that predates any such key is still the same architecture.
pub const OASR_METADATA_KEY_FEATURE_DIARIZATION: &str = "openasr.features.diarization";

pub const OASR_PACKAGE_VERSION_V1: &str = "1";

/// Route-specific metadata envelope. ASR fields are derived from the single
/// architecture inventory; auxiliary packs declare only their real route and
/// never fabricate ASR frontend/decode/tokenizer fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PackEnvelope {
    Asr { model_architecture: &'static str },
    Aux { model_architecture: &'static str },
}

impl PackEnvelope {
    pub(crate) const fn asr(model_architecture: &'static str) -> Self {
        Self::Asr { model_architecture }
    }

    pub(crate) const fn aux(model_architecture: &'static str) -> Self {
        Self::Aux { model_architecture }
    }

    fn validate_verified_route(
        self,
        route: &super::pack_verifier::PackRoute,
    ) -> Result<(), OasrPackWriteError> {
        let matches = match (self, route) {
            (
                Self::Asr {
                    model_architecture: expected,
                },
                super::pack_verifier::PackRoute::Asr {
                    model_architecture: actual,
                    ..
                },
            ) => expected == *actual,
            (
                Self::Aux {
                    model_architecture: expected,
                },
                super::pack_verifier::PackRoute::Aux {
                    model_architecture: actual,
                    ..
                },
            ) => expected == actual,
            (Self::Asr { .. }, super::pack_verifier::PackRoute::Aux { .. })
            | (Self::Aux { .. }, super::pack_verifier::PackRoute::Asr { .. }) => false,
        };
        if matches {
            Ok(())
        } else {
            Err(OasrPackWriteError::RouteMismatch {
                expected: format!("{self:?}"),
                actual: format!("{route:?}"),
            })
        }
    }

    fn protected_metadata(self) -> Result<BTreeMap<String, GgufWriteValue>, PackEnvelopeError> {
        let mut metadata = BTreeMap::new();
        insert_metadata(
            &mut metadata,
            OASR_METADATA_KEY_PACKAGE_VERSION,
            OASR_PACKAGE_VERSION_V1,
        );
        match self {
            Self::Asr { model_architecture } => {
                let descriptor = crate::arch::OpenAsrArchitectureRegistry::with_builtins()
                    .find_by_model_architecture(model_architecture)
                    .ok_or(PackEnvelopeError::UnknownAsrArchitecture { model_architecture })?;
                insert_metadata(
                    &mut metadata,
                    crate::arch::GENERAL_ARCHITECTURE_KEY,
                    descriptor
                        .identity
                        .runtime_architecture_aliases
                        .first()
                        .copied()
                        .unwrap_or(descriptor.identity.model_architecture),
                );
                insert_metadata(
                    &mut metadata,
                    OASR_METADATA_KEY_MODEL_FAMILY,
                    descriptor.identity.model_family,
                );
                insert_metadata(
                    &mut metadata,
                    OASR_METADATA_KEY_MODEL_ARCHITECTURE,
                    descriptor.identity.model_architecture,
                );
                insert_metadata(
                    &mut metadata,
                    OASR_METADATA_KEY_AUDIO_FRONTEND,
                    descriptor.pack_contract.audio_frontend_id,
                );
                insert_metadata(
                    &mut metadata,
                    OASR_METADATA_KEY_DECODE_POLICY,
                    descriptor
                        .topology_contract
                        .decode_driver
                        .decode_policy_id(),
                );
                insert_metadata(
                    &mut metadata,
                    super::ggml_family_adapter::GGML_TOKENIZER_ID_KEY,
                    descriptor.pack_contract.tokenizer_id,
                );
            }
            Self::Aux { model_architecture } => {
                insert_metadata(
                    &mut metadata,
                    crate::arch::GENERAL_ARCHITECTURE_KEY,
                    model_architecture,
                );
            }
        }
        Ok(metadata)
    }

    fn seal(
        self,
        family_metadata: BTreeMap<String, GgufWriteValue>,
    ) -> Result<BTreeMap<String, GgufWriteValue>, PackEnvelopeError> {
        self.seal_with_existing(family_metadata, &BTreeMap::new())
    }

    fn seal_with_existing(
        self,
        mut family_metadata: BTreeMap<String, GgufWriteValue>,
        existing_metadata: &BTreeMap<String, GgufWriteValue>,
    ) -> Result<BTreeMap<String, GgufWriteValue>, PackEnvelopeError> {
        if family_metadata.contains_key(OASR_METADATA_KEY_BUILD_COMMIT) {
            return Err(PackEnvelopeError::ProtectedMetadataOverride {
                key: OASR_METADATA_KEY_BUILD_COMMIT.to_string(),
            });
        }
        let protected = self.protected_metadata()?;
        if let Some(key) = protected
            .keys()
            .find(|key| family_metadata.contains_key(*key))
        {
            return Err(PackEnvelopeError::ProtectedMetadataOverride { key: key.clone() });
        }
        for (key, expected) in protected {
            match existing_metadata.get(&key) {
                Some(actual) if actual == &expected => {}
                Some(actual) => {
                    return Err(PackEnvelopeError::InheritedMetadataMismatch {
                        key,
                        expected,
                        actual: actual.clone(),
                    });
                }
                None => {
                    family_metadata.insert(key, expected);
                }
            }
        }
        Ok(family_metadata)
    }
}

#[derive(Debug, Error)]
pub(crate) enum PackEnvelopeError {
    #[error("unknown ASR architecture '{model_architecture}' cannot produce an OASR envelope")]
    UnknownAsrArchitecture { model_architecture: &'static str },
    #[error("family metadata must not override envelope-owned key '{key}'")]
    ProtectedMetadataOverride { key: String },
    #[error(
        "inherited GGUF metadata key '{key}' conflicts with the OASR envelope: expected {expected:?}, got {actual:?}"
    )]
    InheritedMetadataMismatch {
        key: String,
        expected: GgufWriteValue,
        actual: GgufWriteValue,
    },
}

#[derive(Debug, Error)]
pub(crate) enum OasrPackWriteError {
    #[error(transparent)]
    Envelope(#[from] PackEnvelopeError),
    #[error("OASR output path already exists: {path}")]
    OutputExists { path: PathBuf },
    #[error("could not construct OASR staging path for '{path}'")]
    InvalidOutputPath { path: PathBuf },
    #[error("GGUF writer failed: {source}")]
    GgufWrite {
        #[source]
        source: crate::ggml_runtime::GgufWriteError,
    },
    #[error("written OASR pack failed its production verifier: {source}")]
    Verification {
        #[source]
        source: super::pack_verifier::PackVerificationError,
    },
    #[error(
        "verified OASR route does not match its sealed envelope: expected {expected}, got {actual}"
    )]
    RouteMismatch { expected: String, actual: String },
    #[error("could not expose verified OASR pack '{path}': {source}")]
    Expose {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not durably persist OASR pack '{path}': {source}")]
    Durability {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Transactional production writer: protected envelope -> raw staging GGUF ->
/// exact production verifier -> no-clobber atomic exposure.
pub(crate) struct OasrPackWriter;

impl OasrPackWriter {
    fn seal_output_metadata(
        envelope: PackEnvelope,
        family_metadata: BTreeMap<String, GgufWriteValue>,
        existing_metadata: &BTreeMap<String, GgufWriteValue>,
    ) -> Result<BTreeMap<String, GgufWriteValue>, OasrPackWriteError> {
        let mut metadata = if existing_metadata.is_empty() {
            envelope.seal(family_metadata)?
        } else {
            envelope.seal_with_existing(family_metadata, existing_metadata)?
        };
        if let Some((key, value)) = crate::ggml_runtime::build_provenance_from_env()
            .map_err(|source| OasrPackWriteError::GgufWrite { source })?
        {
            metadata.insert(key, value);
        }
        Ok(metadata)
    }

    /// Starts the only supported custom `.oasr` write path. Importers that
    /// must preserve an upstream GGUF payload byte-for-byte write the sealed
    /// metadata to this transaction's private staging path and then call
    /// [`OasrPackTransaction::commit`]. They cannot obtain a commit-capable
    /// transaction without first sealing the common envelope.
    pub(crate) fn begin(
        output_path: &Path,
        envelope: PackEnvelope,
        family_metadata: BTreeMap<String, GgufWriteValue>,
    ) -> Result<OasrPackTransaction, OasrPackWriteError> {
        if output_path.exists() {
            return Err(OasrPackWriteError::OutputExists {
                path: output_path.to_path_buf(),
            });
        }
        Ok(OasrPackTransaction {
            output_path: output_path.to_path_buf(),
            staging_path: staging_path_for(output_path)?,
            envelope,
            sealed_metadata: Self::seal_output_metadata(
                envelope,
                family_metadata,
                &BTreeMap::new(),
            )?,
            committed: false,
        })
    }

    pub(crate) fn write(
        output_path: &Path,
        envelope: PackEnvelope,
        family_metadata: BTreeMap<String, GgufWriteValue>,
        tensors: &[GgufWriteTensor],
    ) -> Result<super::pack_verifier::VerifiedPack, OasrPackWriteError> {
        let transaction = Self::begin(output_path, envelope, family_metadata)?;
        crate::ggml_runtime::write_gguf_file_v0(
            transaction.staging_path(),
            transaction.sealed_metadata(),
            tensors,
        )
        .map_err(|source| OasrPackWriteError::GgufWrite { source })?;
        transaction.commit()
    }
}

/// Envelope-sealed staging capability for nonstandard writers.
///
/// Dropping an uncommitted transaction removes its private staging file. The
/// final output name is exposed only after the exact bytes pass PackVerifier.
pub(crate) struct OasrPackTransaction {
    output_path: PathBuf,
    staging_path: PathBuf,
    envelope: PackEnvelope,
    sealed_metadata: BTreeMap<String, GgufWriteValue>,
    committed: bool,
}

impl OasrPackTransaction {
    pub(crate) fn staging_path(&self) -> &Path {
        &self.staging_path
    }

    pub(crate) fn sealed_metadata(&self) -> &BTreeMap<String, GgufWriteValue> {
        &self.sealed_metadata
    }

    pub(crate) fn commit(
        mut self,
    ) -> Result<super::pack_verifier::VerifiedPack, OasrPackWriteError> {
        // `File::open` creates a read-only handle. Windows requires write
        // access for `FlushFileBuffers`, which backs `sync_all`, so reopening
        // the completed staging file read-only makes every pack commit fail
        // with `ERROR_ACCESS_DENIED`. Keep the handle private and request the
        // minimum additional access needed for the durability barrier.
        fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.staging_path)
            .and_then(|file| file.sync_all())
            .map_err(|source| OasrPackWriteError::Durability {
                path: self.staging_path.clone(),
                source,
            })?;
        let verified = super::pack_verifier::PackVerifier
            .verify_candidate(super::pack_verifier::PackCandidate::new(&self.staging_path))
            .map_err(|source| OasrPackWriteError::Verification { source })?;
        self.envelope.validate_verified_route(verified.route())?;
        fs::hard_link(&self.staging_path, &self.output_path).map_err(|source| {
            OasrPackWriteError::Expose {
                path: self.output_path.clone(),
                source,
            }
        })?;
        // From this point the verified generation is durably named. Cleanup
        // of the private sibling must not turn a successful publish into an
        // error that prompts the caller to retry against an existing output.
        self.committed = true;
        crate::atomic_file::sync_parent_dir_best_effort(&self.output_path);
        let _ = fs::remove_file(&self.staging_path);
        Ok(verified.with_display_path(self.output_path.clone()))
    }
}

impl Drop for OasrPackTransaction {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_file(&self.staging_path);
        }
    }
}

fn staging_path_for(output_path: &Path) -> Result<PathBuf, OasrPackWriteError> {
    static NEXT_STAGING_ID: AtomicU64 = AtomicU64::new(0);
    let parent = output_path
        .parent()
        .ok_or_else(|| OasrPackWriteError::InvalidOutputPath {
            path: output_path.to_path_buf(),
        })?;
    let file_name = output_path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| OasrPackWriteError::InvalidOutputPath {
            path: output_path.to_path_buf(),
        })?;
    Ok(parent.join(format!(
        ".{file_name}.openasr-write-{}-{}.tmp",
        std::process::id(),
        NEXT_STAGING_ID.fetch_add(1, Ordering::Relaxed)
    )))
}

/// Shared `tokenizer.ggml.*` GGUF key names. Every builtin tokenizer family
/// (cohere, moonshine, whisper, qwen) reads/writes the same three keys
/// under these exact names; only the accepted `tokenizer.ggml.model` *value*
/// differs per family (llama/SentencePiece vs gpt2/BPE) and stays declared
/// locally in each family, since merging the values would collapse a real
/// distinction into a false one.
pub(crate) const TOKENIZER_GGML_MODEL_KEY: &str = "tokenizer.ggml.model";
pub(crate) const TOKENIZER_GGML_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
pub(crate) const TOKENIZER_GGML_MERGES_KEY: &str = "tokenizer.ggml.merges";

/// Insert a string-valued GGUF metadata entry. Shared by every family's
/// `*_runtime_gguf_metadata` builder in place of a per-file copy of the same
/// four-line helper.
pub(crate) fn insert_metadata(
    metadata: &mut BTreeMap<String, GgufWriteValue>,
    key: &str,
    value: impl ToString,
) {
    metadata.insert(key.to_string(), GgufWriteValue::String(value.to_string()));
}

/// Insert a `u32`-valued GGUF metadata entry.
pub(crate) fn insert_metadata_u32(
    metadata: &mut BTreeMap<String, GgufWriteValue>,
    key: &str,
    value: u32,
) {
    metadata.insert(key.to_string(), GgufWriteValue::U32(value));
}

/// Insert a string-array-valued GGUF metadata entry (e.g. `tokenizer.ggml.tokens`).
pub(crate) fn insert_metadata_string_array(
    metadata: &mut BTreeMap<String, GgufWriteValue>,
    key: &str,
    values: &[String],
) {
    metadata.insert(
        key.to_string(),
        GgufWriteValue::StringArray(values.to_vec()),
    );
}

/// Insert a `u32`-array-valued GGUF metadata entry.
pub(crate) fn insert_metadata_u32_array(
    metadata: &mut BTreeMap<String, GgufWriteValue>,
    key: &str,
    values: &[u32],
) {
    metadata.insert(key.to_string(), GgufWriteValue::U32Array(values.to_vec()));
}

/// Fluent wrapper around the four `insert_metadata*` helpers above. It can
/// finish only as family-owned metadata; the protected envelope is injected
/// later by [`OasrPackWriter`].
#[derive(Debug, Default)]
pub(crate) struct OasrMetadataBuilder {
    metadata: BTreeMap<String, GgufWriteValue>,
}

impl OasrMetadataBuilder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn str(mut self, key: &str, value: impl ToString) -> Self {
        insert_metadata(&mut self.metadata, key, value);
        self
    }

    pub(crate) fn u32(mut self, key: &str, value: u32) -> Self {
        insert_metadata_u32(&mut self.metadata, key, value);
        self
    }

    pub(crate) fn string_array(mut self, key: &str, values: &[String]) -> Self {
        insert_metadata_string_array(&mut self.metadata, key, values);
        self
    }

    /// Finishes family-owned metadata only. Envelope keys are injected and
    /// protected later by [`OasrPackWriter`].
    pub(crate) fn build_family_metadata(self) -> BTreeMap<String, GgufWriteValue> {
        self.metadata
    }
}

// --- Read-side accessors -------------------------------------------------
//
// Every builtin tokenizer family's `from_gguf_metadata` loader (cohere,
// moonshine, whisper, qwen) parsed its GGUF metadata through a
// byte-for-byte copy of these helpers, differing only in the family name
// spliced into the error text. Centralizing them here (mirroring the
// write-side `insert_metadata*` helpers above) means a metadata-parsing fix
// lands once instead of five times.

/// Read a required string-valued GGUF metadata key, trimmed and rejected if
/// empty after trimming.
pub(crate) fn required_metadata_string<'a>(
    metadata: &'a GgufMetadata,
    key: &'static str,
    family: &str,
) -> Result<&'a str, NativeAsrError> {
    let value = metadata
        .get_string(key)
        .ok_or_else(|| NativeAsrError::UnsupportedModelPack {
            reason: format!("{family} GGUF tokenizer is missing required key '{key}'"),
        })?;
    let normalized = value.trim();
    if normalized.is_empty() {
        return Err(NativeAsrError::UnsupportedModelPack {
            reason: format!("{family} GGUF tokenizer key '{key}' cannot be empty"),
        });
    }
    Ok(normalized)
}

/// Read a required `array[string]`-valued GGUF metadata key.
pub(crate) fn required_metadata_string_array<'a>(
    metadata: &'a GgufMetadata,
    key: &'static str,
    family: &str,
) -> Result<&'a [String], NativeAsrError> {
    metadata
        .get_string_array(key)
        .ok_or_else(|| NativeAsrError::UnsupportedModelPack {
            reason: format!("{family} GGUF tokenizer requires key '{key}' as array[string]"),
        })
}

/// Read a required `array[uint32]`-valued GGUF metadata key.
pub(crate) fn required_metadata_u32_array<'a>(
    metadata: &'a GgufMetadata,
    key: &'static str,
    family: &str,
) -> Result<&'a [u32], NativeAsrError> {
    metadata
        .get_u32_array(key)
        .ok_or_else(|| NativeAsrError::UnsupportedModelPack {
            reason: format!("{family} GGUF tokenizer requires key '{key}' as array[uint32]"),
        })
}

/// Read an optional `u32`-valued GGUF metadata key, accepting a native u32, a
/// native u64 that fits u32, or a numeric string (some importers write ints
/// as strings). Returns `None` when the key is absent.
pub(crate) fn optional_metadata_u32(
    metadata: &GgufMetadata,
    key: &'static str,
    family: &str,
) -> Result<Option<u32>, NativeAsrError> {
    if let Some(value) = metadata.get_u32(key) {
        return Ok(Some(value));
    }
    if let Some(value) = metadata.get_u64(key) {
        return u32::try_from(value)
            .map(Some)
            .map_err(|_| NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "{family} GGUF tokenizer key '{key}' value {value} does not fit u32"
                ),
            });
    }
    if let Some(value) = metadata.get_string(key) {
        let parsed =
            value
                .trim()
                .parse::<u32>()
                .map_err(|error| NativeAsrError::UnsupportedModelPack {
                    reason: format!(
                        "{family} GGUF tokenizer key '{key}' cannot parse '{value}' as u32: {error}"
                    ),
                })?;
        return Ok(Some(parsed));
    }
    Ok(None)
}

/// Read a required `u32`-valued GGUF metadata key (see [`optional_metadata_u32`]
/// for the accepted encodings).
pub(crate) fn required_metadata_u32(
    metadata: &GgufMetadata,
    key: &'static str,
    family: &str,
) -> Result<u32, NativeAsrError> {
    optional_metadata_u32(metadata, key, family)?.ok_or_else(|| {
        NativeAsrError::UnsupportedModelPack {
            reason: format!("{family} GGUF tokenizer is missing required key '{key}'"),
        }
    })
}

#[cfg(test)]
mod envelope_tests {
    use super::*;

    fn string_value<'a>(
        metadata: &'a BTreeMap<String, GgufWriteValue>,
        key: &str,
    ) -> Option<&'a str> {
        match metadata.get(key) {
            Some(GgufWriteValue::String(value)) => Some(value),
            _ => None,
        }
    }

    #[test]
    fn asr_envelope_derives_every_routing_key_from_inventory() {
        let descriptor = crate::arch::OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(crate::WHISPER_GGML_ARCHITECTURE_ID)
            .expect("Whisper inventory descriptor");
        let metadata = PackEnvelope::asr(crate::WHISPER_GGML_ARCHITECTURE_ID)
            .seal(BTreeMap::new())
            .expect("seal ASR envelope");

        assert_eq!(
            string_value(&metadata, OASR_METADATA_KEY_PACKAGE_VERSION),
            Some(OASR_PACKAGE_VERSION_V1)
        );
        assert_eq!(
            string_value(&metadata, crate::arch::GENERAL_ARCHITECTURE_KEY),
            descriptor
                .identity
                .runtime_architecture_aliases
                .first()
                .copied()
        );
        assert_eq!(
            string_value(&metadata, OASR_METADATA_KEY_MODEL_FAMILY),
            Some(crate::WHISPER_MODEL_FAMILY)
        );
        assert_eq!(
            string_value(
                &metadata,
                super::super::ggml_family_adapter::GGML_TOKENIZER_ID_KEY
            ),
            Some(crate::WHISPER_TOKENIZER_ID)
        );
    }

    #[test]
    fn asr_envelope_keeps_internal_architecture_distinct_from_gguf_alias() {
        let metadata = PackEnvelope::asr(crate::QWEN3_ASR_GGML_ARCHITECTURE_ID)
            .seal(BTreeMap::new())
            .expect("seal Qwen ASR envelope");

        assert_eq!(
            string_value(&metadata, crate::arch::GENERAL_ARCHITECTURE_KEY),
            Some("qwen3-asr")
        );
        assert_eq!(
            string_value(&metadata, OASR_METADATA_KEY_MODEL_ARCHITECTURE),
            Some(crate::QWEN3_ASR_GGML_ARCHITECTURE_ID)
        );
    }

    #[test]
    fn family_metadata_cannot_override_envelope_keys() {
        let mut metadata = BTreeMap::new();
        insert_metadata(
            &mut metadata,
            OASR_METADATA_KEY_PACKAGE_VERSION,
            "attacker-controlled",
        );
        let error = PackEnvelope::asr(crate::WHISPER_GGML_ARCHITECTURE_ID)
            .seal(metadata)
            .expect_err("protected key override must fail");
        assert!(matches!(
            error,
            PackEnvelopeError::ProtectedMetadataOverride { key }
                if key == OASR_METADATA_KEY_PACKAGE_VERSION
        ));
    }

    #[test]
    fn auxiliary_envelope_does_not_fabricate_asr_contract_fields() {
        let metadata =
            PackEnvelope::aux(crate::models::aux_pack_registry::REDIMNET2_GGML_ARCHITECTURE_ID)
                .seal(BTreeMap::new())
                .expect("seal aux envelope");

        assert_eq!(
            string_value(&metadata, OASR_METADATA_KEY_PACKAGE_VERSION),
            Some(OASR_PACKAGE_VERSION_V1)
        );
        assert_eq!(
            string_value(&metadata, crate::arch::GENERAL_ARCHITECTURE_KEY),
            Some(crate::models::aux_pack_registry::REDIMNET2_GGML_ARCHITECTURE_ID)
        );
        for key in [
            OASR_METADATA_KEY_MODEL_FAMILY,
            OASR_METADATA_KEY_MODEL_ARCHITECTURE,
            OASR_METADATA_KEY_AUDIO_FRONTEND,
            OASR_METADATA_KEY_DECODE_POLICY,
            super::super::ggml_family_adapter::GGML_TOKENIZER_ID_KEY,
        ] {
            assert!(
                !metadata.contains_key(key),
                "aux envelope must not write {key}"
            );
        }
    }

    #[test]
    fn inherited_protected_metadata_must_match_the_envelope() {
        let envelope =
            PackEnvelope::aux(crate::models::aux_pack_registry::REDIMNET2_GGML_ARCHITECTURE_ID);
        let mut inherited = BTreeMap::new();
        insert_metadata(
            &mut inherited,
            crate::arch::GENERAL_ARCHITECTURE_KEY,
            "wrong-architecture",
        );
        let error = envelope
            .seal_with_existing(BTreeMap::new(), &inherited)
            .expect_err("conflicting inherited routing metadata must fail closed");
        assert!(matches!(
            error,
            PackEnvelopeError::InheritedMetadataMismatch { key, .. }
                if key == crate::arch::GENERAL_ARCHITECTURE_KEY
        ));
    }

    #[test]
    fn sealed_envelope_rejects_a_verified_route_of_another_kind() {
        let error = PackEnvelope::asr(crate::WHISPER_GGML_ARCHITECTURE_ID)
            .validate_verified_route(&super::super::pack_verifier::PackRoute::Aux {
                kind: crate::models::aux_pack_registry::AuxPackKind::Diarization,
                model_architecture:
                    crate::models::aux_pack_registry::REDIMNET2_GGML_ARCHITECTURE_ID.to_string(),
            })
            .expect_err("ASR envelope must reject an auxiliary verified route");
        assert!(matches!(error, OasrPackWriteError::RouteMismatch { .. }));
    }

    #[test]
    fn failed_transaction_verification_never_exposes_the_output() {
        let directory = tempfile::tempdir().expect("tempdir");
        let output = directory.path().join("invalid.oasr");
        let transaction = OasrPackWriter::begin(
            &output,
            PackEnvelope::aux(crate::models::aux_pack_registry::REDIMNET2_GGML_ARCHITECTURE_ID),
            BTreeMap::new(),
        )
        .expect("begin transaction");
        let staging = transaction.staging_path().to_path_buf();
        std::fs::write(&staging, b"not a GGUF").expect("write invalid staging bytes");

        let error = transaction
            .commit()
            .expect_err("invalid staging bytes must fail production verification");
        assert!(matches!(error, OasrPackWriteError::Verification { .. }));
        assert!(
            !output.exists(),
            "an unverified output must never be exposed"
        );
        assert!(
            !staging.exists(),
            "failed transaction staging must be cleaned"
        );
    }
}
