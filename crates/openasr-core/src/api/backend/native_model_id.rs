use std::{collections::BTreeMap, path::Path};

use thiserror::Error;

use crate::{
    GgmlRuntimeSource, parse_model_ref, probe_ggml_package_model_identity,
    validate_ggml_runtime_source_path,
};

const RUNTIME_SOURCE_FILE_STEM_SOURCE_KEY: &str = "<runtime-source.file-stem>";
const METADATA_MODEL_ID_CANDIDATE_KEYS: [&str; 3] =
    ["openasr.model.id", "general.basename", "general.name"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeRuntimeModelIdSource {
    MetadataGgufKey { key: String },
    ExplicitModelIdFallback,
    RuntimeSourcePathStemFallback,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeRuntimeModelIdentity {
    pub model_id: String,
    pub source: NativeRuntimeModelIdSource,
}

#[derive(Debug, Clone, Error)]
pub enum NativeRuntimeModelIdentityError {
    #[error("could not validate GGUF-backed runtime source '{path}': {reason}")]
    RuntimeSourceValidation { path: String, reason: String },
    #[error(
        "could not resolve native model id from GGUF-backed runtime source '{path}'; expected valid GGUF metadata key ('openasr.model.id', 'general.basename', or 'general.name'), explicit model id fallback, or runtime source file stem.{metadata_error}"
    )]
    MissingModelId {
        path: String,
        metadata_error: String,
    },
}

pub fn resolve_local_native_runtime_model_identity(
    runtime_path: &Path,
    explicit_model_id_fallback: Option<&str>,
) -> Result<NativeRuntimeModelIdentity, NativeRuntimeModelIdentityError> {
    let runtime_source = validate_ggml_runtime_source_path(runtime_path).map_err(|error| {
        NativeRuntimeModelIdentityError::RuntimeSourceValidation {
            path: runtime_path.display().to_string(),
            reason: error.to_string(),
        }
    })?;
    resolve_native_runtime_model_identity_from_source(&runtime_source, explicit_model_id_fallback)
}

pub(crate) fn resolve_native_runtime_model_identity_from_source(
    runtime_source: &GgmlRuntimeSource,
    explicit_model_id_fallback: Option<&str>,
) -> Result<NativeRuntimeModelIdentity, NativeRuntimeModelIdentityError> {
    let metadata_identity = probe_ggml_package_model_identity(runtime_source.path());
    let mut rejected_candidates: Vec<String> = Vec::new();

    if let (Some(model_id), Some(source_key)) = (
        metadata_identity.model_id.as_deref(),
        metadata_identity.source_key.as_deref(),
    ) && source_key != RUNTIME_SOURCE_FILE_STEM_SOURCE_KEY
    {
        if let Ok(model_id) =
            normalize_and_validate_model_id(model_id, &format!("GGUF metadata key '{source_key}'"))
        {
            return Ok(NativeRuntimeModelIdentity {
                model_id,
                source: NativeRuntimeModelIdSource::MetadataGgufKey {
                    key: source_key.to_string(),
                },
            });
        }
        rejected_candidates.push(format!(
            "metadata candidate from key '{source_key}' was rejected"
        ));
    }

    if let Some(explicit_model_id) = explicit_model_id_fallback {
        if let Ok(model_id) =
            normalize_and_validate_model_id(explicit_model_id, "explicit fallback")
        {
            return Ok(NativeRuntimeModelIdentity {
                model_id,
                source: NativeRuntimeModelIdSource::ExplicitModelIdFallback,
            });
        }
        rejected_candidates.push("explicit fallback candidate was rejected".to_string());
    }

    if let Some(stem) = runtime_source
        .path()
        .file_stem()
        .and_then(|value| value.to_str())
    {
        if let Ok(model_id) = normalize_and_validate_model_id(stem, "runtime source file stem") {
            return Ok(NativeRuntimeModelIdentity {
                model_id,
                source: NativeRuntimeModelIdSource::RuntimeSourcePathStemFallback,
            });
        }
        rejected_candidates.push("runtime source file stem candidate was rejected".to_string());
    }

    let mut metadata_error = metadata_identity
        .metadata_read_error
        .as_deref()
        .map(|error| format!(" metadata_read_error={error};"))
        .unwrap_or_default();
    if !rejected_candidates.is_empty() {
        metadata_error.push_str(&format!(
            " candidate_rejections={};",
            rejected_candidates.join(", ")
        ));
    }
    Err(NativeRuntimeModelIdentityError::MissingModelId {
        path: runtime_source.path().display().to_string(),
        metadata_error,
    })
}

pub(crate) fn resolve_native_runtime_model_identity_from_string_metadata(
    metadata: &BTreeMap<String, String>,
    runtime_source_path: &Path,
    explicit_model_id_fallback: Option<&str>,
) -> Result<NativeRuntimeModelIdentity, NativeRuntimeModelIdentityError> {
    let mut rejected_candidates: Vec<String> = Vec::new();
    for key in METADATA_MODEL_ID_CANDIDATE_KEYS {
        if let Some(model_id) = metadata.get(key)
            && let Ok(model_id) =
                normalize_and_validate_model_id(model_id, &format!("GGUF metadata key '{key}'"))
        {
            return Ok(NativeRuntimeModelIdentity {
                model_id,
                source: NativeRuntimeModelIdSource::MetadataGgufKey {
                    key: key.to_string(),
                },
            });
        }
        if metadata.contains_key(key) {
            rejected_candidates.push(format!("metadata candidate from key '{key}' was rejected"));
        }
    }

    if let Some(explicit_model_id) = explicit_model_id_fallback {
        if let Ok(model_id) =
            normalize_and_validate_model_id(explicit_model_id, "explicit fallback")
        {
            return Ok(NativeRuntimeModelIdentity {
                model_id,
                source: NativeRuntimeModelIdSource::ExplicitModelIdFallback,
            });
        }
        rejected_candidates.push("explicit fallback candidate was rejected".to_string());
    }

    if let Some(stem) = runtime_source_path
        .file_stem()
        .and_then(|value| value.to_str())
    {
        if let Ok(model_id) = normalize_and_validate_model_id(stem, "runtime source file stem") {
            return Ok(NativeRuntimeModelIdentity {
                model_id,
                source: NativeRuntimeModelIdSource::RuntimeSourcePathStemFallback,
            });
        }
        rejected_candidates.push("runtime source file stem candidate was rejected".to_string());
    }

    let mut metadata_error = String::new();
    if !rejected_candidates.is_empty() {
        metadata_error.push_str(&format!(
            " candidate_rejections={};",
            rejected_candidates.join(", ")
        ));
    }
    Err(NativeRuntimeModelIdentityError::MissingModelId {
        path: runtime_source_path.display().to_string(),
        metadata_error,
    })
}

fn normalize_and_validate_model_id(model_id: &str, _origin: &str) -> Result<String, String> {
    let normalized = model_id.trim();
    if normalized.is_empty() {
        return Err("model id is empty after whitespace normalization".to_string());
    }
    parse_model_ref(normalized)
        .map(|_| normalized.to_string())
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};

    #[test]
    fn resolves_model_id_from_metadata_key() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("fixture.oasr");
        write_tiny_gguf_runtime_source(
            &runtime_path,
            &TinyGgufFixtureSpec::whisper_oasr_v1_non_streaming_cpu("whisper-large-v3-turbo"),
        )
        .unwrap();

        let identity = resolve_local_native_runtime_model_identity(&runtime_path, None).unwrap();
        assert_eq!(identity.model_id, "whisper-large-v3-turbo");
        assert_eq!(
            identity.source,
            NativeRuntimeModelIdSource::MetadataGgufKey {
                key: "openasr.model.id".to_string()
            }
        );
    }

    #[test]
    fn prefers_explicit_fallback_before_runtime_stem() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("stem-only-runtime.gguf");
        write_tiny_gguf_runtime_source(
            &runtime_path,
            &TinyGgufFixtureSpec::new(Default::default()),
        )
        .unwrap();

        let identity = resolve_local_native_runtime_model_identity(
            &runtime_path,
            Some("whisper-large-v3-turbo"),
        )
        .unwrap();
        assert_eq!(identity.model_id, "whisper-large-v3-turbo");
        assert_eq!(
            identity.source,
            NativeRuntimeModelIdSource::ExplicitModelIdFallback
        );
    }

    #[test]
    fn resolves_model_id_from_runtime_stem_when_metadata_missing() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-small.gguf");
        write_tiny_gguf_runtime_source(
            &runtime_path,
            &TinyGgufFixtureSpec::new(Default::default()),
        )
        .unwrap();

        let identity = resolve_local_native_runtime_model_identity(&runtime_path, None).unwrap();
        assert_eq!(identity.model_id, "whisper-small");
        assert_eq!(
            identity.source,
            NativeRuntimeModelIdSource::RuntimeSourcePathStemFallback
        );
    }

    #[test]
    fn falls_back_to_runtime_stem_when_explicit_model_id_is_invalid() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-small.gguf");
        write_tiny_gguf_runtime_source(
            &runtime_path,
            &TinyGgufFixtureSpec::new(Default::default()),
        )
        .unwrap();

        let identity =
            resolve_local_native_runtime_model_identity(&runtime_path, Some(":::")).unwrap();
        assert_eq!(identity.model_id, "whisper-small");
        assert_eq!(
            identity.source,
            NativeRuntimeModelIdSource::RuntimeSourcePathStemFallback
        );
    }

    #[test]
    fn falls_back_to_runtime_stem_when_metadata_model_id_is_invalid() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("native-pack.oasr");
        write_tiny_gguf_runtime_source(
            &runtime_path,
            &TinyGgufFixtureSpec::whisper_oasr_v1_non_streaming_cpu("bad::id"),
        )
        .unwrap();

        let identity = resolve_local_native_runtime_model_identity(&runtime_path, None).unwrap();
        assert_eq!(identity.model_id, "native-pack");
        assert_eq!(
            identity.source,
            NativeRuntimeModelIdSource::RuntimeSourcePathStemFallback
        );
    }
}
