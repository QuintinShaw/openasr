use std::collections::BTreeMap;

use crate::ggml_runtime::GgufWriteValue;

pub const OASR_METADATA_KEY_PACKAGE_VERSION: &str = "openasr.package.version";
pub const OASR_METADATA_KEY_MODEL_FAMILY: &str = "openasr.model.family";
pub const OASR_METADATA_KEY_MODEL_ARCHITECTURE: &str = "openasr.model.architecture";
pub const OASR_METADATA_KEY_AUDIO_FRONTEND: &str = "openasr.audio.frontend";
pub const OASR_METADATA_KEY_DECODE_POLICY: &str = "openasr.decode.policy";
pub const OASR_METADATA_KEY_FEATURE_DIARIZATION: &str = "openasr.features.diarization";

pub const OASR_PACKAGE_VERSION_V1: &str = "1";
pub const OASR_FEATURE_DIARIZATION_COHERE_TOKEN_STREAM_V1: &str = "cohere-token-stream-v1";

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
