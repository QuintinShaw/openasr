use std::collections::BTreeMap;

use crate::models::oasr_metadata::{
    OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_DECODE_POLICY,
    OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
    OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
};

pub const GGML_TOKENIZER_ID_KEY: &str = "openasr.tokenizer.id";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlAdapterMetadataSource {
    GgufKvV1,
    OasrV1Metadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlExecutionCapability {
    DedicatedRuntimeExecutorV1,
    NativeGraphLoweringV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlFamilyAdapterSelectionSpec<'a> {
    pub source: GgmlAdapterMetadataSource,
    pub metadata: &'a BTreeMap<String, String>,
    pub tokenizer_id: Option<&'a str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlFamilyAdapterSelectionFields<'a> {
    pub source: GgmlAdapterMetadataSource,
    pub package_version: &'a str,
    pub model_family: &'a str,
    pub model_architecture: &'a str,
    pub audio_frontend_id: &'a str,
    pub decode_policy_id: &'a str,
    pub tokenizer_id: Option<&'a str>,
}

/// Compile-time, per-family hint for how this architecture handles a source
/// language. The concrete per-pack `LanguageMode` is resolved from this plus the
/// pack vocab at runtime (see `crate::api::backend::language`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageFamilyHint {
    /// Whisper: detects at decode time and accepts explicit selection. The pack
    /// vocab decides multilingual (DetectAndSpecify) vs English-only (Fixed).
    WhisperVocabGated,
    /// Qwen3-ASR: self-detects internally; an explicit hint is rejected with this reason.
    SelfDetectsRejectsHint { reject_reason: &'static str },
    /// Cohere transcribe: accepts explicit selection via a prompt language token;
    /// no decode-time auto-detect.
    SelectsViaPrompt { default_language: &'static str },
    /// SenseVoice: accepts explicit selection via a prompt token AND detects at
    /// decode time when unset (the model emits a readable `<|lang|>` tag).
    DetectAndSelectsViaPrompt,
    /// Intrinsically a single language (CTC / Moonshine).
    FixedMonolingual { language: &'static str },
    /// Intrinsically a fixed multilingual set with no per-request steering (XASR zh-en).
    FixedMultilingual { languages: &'static [&'static str] },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlFamilyAdapterDescriptor {
    pub adapter_id: &'static str,
    pub model_family: &'static str,
    pub model_architecture: &'static str,
    pub audio_frontend_id: &'static str,
    pub tokenizer_id: &'static str,
    pub decode_policy_id: &'static str,
    pub execution_capability: GgmlExecutionCapability,
    pub language_family_hint: LanguageFamilyHint,
    /// Whether this family's own decode loop produces the diarization tokens
    /// (e.g. the cohere token-stream's `<|diarize|>`/`<|spltoken0|>`), the
    /// single declaration of this architecture-level fact. Mirrored from
    /// `arch::OpenAsrArchitectureDescriptor::self_diarizes`; a pack still has
    /// to carry the actual runtime metadata for this to activate (see
    /// `native_runtime_metadata_supports_diarization`), so this flag alone is
    /// "capable of", not "this exact pack does".
    pub self_diarizes: bool,
}

impl GgmlFamilyAdapterDescriptor {
    pub fn matches_selection_fields(&self, fields: &GgmlFamilyAdapterSelectionFields<'_>) -> bool {
        if fields.package_version != OASR_PACKAGE_VERSION_V1 {
            return false;
        }
        if fields.model_family != self.model_family {
            return false;
        }
        if fields.model_architecture != self.model_architecture {
            return false;
        }
        if fields.audio_frontend_id != self.audio_frontend_id {
            return false;
        }
        if fields.decode_policy_id != self.decode_policy_id {
            return false;
        }
        match fields.tokenizer_id {
            Some(id) => id == self.tokenizer_id,
            None => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OasrV1MetadataError {
    MissingKey(&'static str),
    EmptyValue(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OasrV1AdapterSelectionMetadata<'a> {
    pub package_version: &'a str,
    pub model_family: &'a str,
    pub model_architecture: &'a str,
    pub audio_frontend_id: &'a str,
    pub decode_policy_id: &'a str,
}

impl<'a> GgmlFamilyAdapterSelectionSpec<'a> {
    pub fn from_gguf_metadata_v1(metadata: &'a BTreeMap<String, String>) -> Self {
        Self::from_metadata_map(GgmlAdapterMetadataSource::GgufKvV1, metadata)
    }

    pub fn from_oasr_metadata_v1(metadata: &'a BTreeMap<String, String>) -> Self {
        Self::from_metadata_map(GgmlAdapterMetadataSource::OasrV1Metadata, metadata)
    }

    pub fn try_from_gguf_metadata_v1(
        metadata: &'a BTreeMap<String, String>,
    ) -> Result<Self, OasrV1MetadataError> {
        let spec = Self::from_gguf_metadata_v1(metadata);
        let _ = spec.parse_selection_fields()?;
        Ok(spec)
    }

    pub fn try_from_oasr_metadata_v1(
        metadata: &'a BTreeMap<String, String>,
    ) -> Result<Self, OasrV1MetadataError> {
        let spec = Self::from_oasr_metadata_v1(metadata);
        let _ = spec.parse_selection_fields()?;
        Ok(spec)
    }

    pub fn parse_oasr_v1_metadata(
        &self,
    ) -> Result<OasrV1AdapterSelectionMetadata<'_>, OasrV1MetadataError> {
        let package_version =
            required_metadata_value(self.metadata, OASR_METADATA_KEY_PACKAGE_VERSION)?;
        let model_family = required_metadata_value(self.metadata, OASR_METADATA_KEY_MODEL_FAMILY)?;
        let model_architecture =
            required_metadata_value(self.metadata, OASR_METADATA_KEY_MODEL_ARCHITECTURE)?;
        let audio_frontend_id =
            required_metadata_value(self.metadata, OASR_METADATA_KEY_AUDIO_FRONTEND)?;
        let decode_policy_id =
            required_metadata_value(self.metadata, OASR_METADATA_KEY_DECODE_POLICY)?;

        Ok(OasrV1AdapterSelectionMetadata {
            package_version,
            model_family,
            model_architecture,
            audio_frontend_id,
            decode_policy_id,
        })
    }

    pub fn parse_selection_fields(
        &self,
    ) -> Result<GgmlFamilyAdapterSelectionFields<'_>, OasrV1MetadataError> {
        let parsed = self.parse_oasr_v1_metadata()?;
        if self
            .tokenizer_id
            .is_some_and(|tokenizer| tokenizer.trim().is_empty())
        {
            return Err(OasrV1MetadataError::EmptyValue(GGML_TOKENIZER_ID_KEY));
        }
        Ok(GgmlFamilyAdapterSelectionFields {
            source: self.source,
            package_version: parsed.package_version,
            model_family: parsed.model_family,
            model_architecture: parsed.model_architecture,
            audio_frontend_id: parsed.audio_frontend_id,
            decode_policy_id: parsed.decode_policy_id,
            tokenizer_id: self.tokenizer_id,
        })
    }

    fn from_metadata_map(
        source: GgmlAdapterMetadataSource,
        metadata: &'a BTreeMap<String, String>,
    ) -> Self {
        let tokenizer_id = metadata.get(GGML_TOKENIZER_ID_KEY).map(String::as_str);
        Self {
            source,
            metadata,
            tokenizer_id,
        }
    }
}

fn required_metadata_value<'a>(
    metadata: &'a BTreeMap<String, String>,
    key: &'static str,
) -> Result<&'a str, OasrV1MetadataError> {
    let Some(value) = metadata.get(key).map(String::as_str) else {
        return Err(OasrV1MetadataError::MissingKey(key));
    };
    if value.trim().is_empty() {
        return Err(OasrV1MetadataError::EmptyValue(key));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::oasr_metadata::{
        OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_DECODE_POLICY,
        OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
        OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
    };

    fn base_metadata() -> BTreeMap<String, String> {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            OASR_METADATA_KEY_PACKAGE_VERSION.to_string(),
            OASR_PACKAGE_VERSION_V1.to_string(),
        );
        metadata.insert(
            OASR_METADATA_KEY_MODEL_FAMILY.to_string(),
            "whisper".to_string(),
        );
        metadata.insert(
            OASR_METADATA_KEY_MODEL_ARCHITECTURE.to_string(),
            "whisper-encoder-decoder".to_string(),
        );
        metadata.insert(
            OASR_METADATA_KEY_AUDIO_FRONTEND.to_string(),
            "whisper.logmel.16khz.mono.v0".to_string(),
        );
        metadata.insert(
            OASR_METADATA_KEY_DECODE_POLICY.to_string(),
            "whisper.greedy.seq2seq.v1".to_string(),
        );
        metadata
    }

    #[test]
    fn parses_oasr_v1_metadata_from_selection_spec() {
        let metadata = base_metadata();
        let selection = GgmlFamilyAdapterSelectionSpec::from_gguf_metadata_v1(&metadata);
        let parsed = selection.parse_oasr_v1_metadata().expect("must parse");

        assert_eq!(parsed.package_version, OASR_PACKAGE_VERSION_V1);
        assert_eq!(parsed.model_family, "whisper");
        assert_eq!(parsed.model_architecture, "whisper-encoder-decoder");
        assert_eq!(parsed.audio_frontend_id, "whisper.logmel.16khz.mono.v0");
        assert_eq!(parsed.decode_policy_id, "whisper.greedy.seq2seq.v1");
    }

    #[test]
    fn parse_fails_closed_when_required_key_missing() {
        let mut metadata = base_metadata();
        metadata.remove(OASR_METADATA_KEY_DECODE_POLICY);
        let selection = GgmlFamilyAdapterSelectionSpec::from_gguf_metadata_v1(&metadata);
        let error = selection
            .parse_oasr_v1_metadata()
            .expect_err("missing key must fail");

        assert_eq!(
            error,
            OasrV1MetadataError::MissingKey(OASR_METADATA_KEY_DECODE_POLICY)
        );
    }

    #[test]
    fn parse_selection_fields_fails_when_optional_tokenizer_key_is_empty() {
        let mut metadata = base_metadata();
        metadata.insert(GGML_TOKENIZER_ID_KEY.to_string(), " ".to_string());
        let selection = GgmlFamilyAdapterSelectionSpec::from_gguf_metadata_v1(&metadata);
        let error = selection
            .parse_selection_fields()
            .expect_err("empty tokenizer id must fail");

        assert_eq!(
            error,
            OasrV1MetadataError::EmptyValue(GGML_TOKENIZER_ID_KEY)
        );
    }

    #[test]
    fn try_from_oasr_metadata_v1_validates_required_fields() {
        let mut metadata = base_metadata();
        metadata.remove(OASR_METADATA_KEY_MODEL_FAMILY);
        let error = GgmlFamilyAdapterSelectionSpec::try_from_oasr_metadata_v1(&metadata)
            .expect_err("required field validation must fail");

        assert_eq!(
            error,
            OasrV1MetadataError::MissingKey(OASR_METADATA_KEY_MODEL_FAMILY)
        );
    }
}
