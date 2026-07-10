use std::collections::BTreeMap;

use crate::NativeAsrError;
use crate::ggml_runtime::{GgufMetadata, GgufWriteValue};

pub const OASR_METADATA_KEY_PACKAGE_VERSION: &str = "openasr.package.version";
pub const OASR_METADATA_KEY_MODEL_FAMILY: &str = "openasr.model.family";
pub const OASR_METADATA_KEY_MODEL_ARCHITECTURE: &str = "openasr.model.architecture";
pub const OASR_METADATA_KEY_AUDIO_FRONTEND: &str = "openasr.audio.frontend";
pub const OASR_METADATA_KEY_DECODE_POLICY: &str = "openasr.decode.policy";
pub const OASR_METADATA_KEY_FEATURE_DIARIZATION: &str = "openasr.features.diarization";

pub const OASR_PACKAGE_VERSION_V1: &str = "1";
pub const OASR_FEATURE_DIARIZATION_COHERE_TOKEN_STREAM_V1: &str = "cohere-token-stream-v1";

/// Shared `tokenizer.ggml.*` GGUF key names. Every builtin tokenizer family
/// (cohere, hymt2, moonshine, whisper, qwen) reads/writes the same three keys
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

/// Fluent wrapper around the four `insert_metadata*` helpers above, for
/// importers that build up their GGUF metadata map in one pass rather than via
/// a closure over a local `BTreeMap`. Equivalent to calling the free functions
/// directly -- `build()` just returns the accumulated map -- so adopting it is
/// a pure naming/ergonomics choice, not a behavior change.
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

    // No current importer needs a chained u32-array insert (whisper is the only
    // u32-array user today and still calls the free function directly); kept
    // for parity with the other three primitives for the next adopter.
    #[allow(dead_code)]
    pub(crate) fn u32_array(mut self, key: &str, values: &[u32]) -> Self {
        insert_metadata_u32_array(&mut self.metadata, key, values);
        self
    }

    pub(crate) fn build(self) -> BTreeMap<String, GgufWriteValue> {
        self.metadata
    }
}

// --- Read-side accessors -------------------------------------------------
//
// Every builtin tokenizer family's `from_gguf_metadata` loader (cohere,
// hymt2, moonshine, whisper, qwen) parsed its GGUF metadata through a
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
