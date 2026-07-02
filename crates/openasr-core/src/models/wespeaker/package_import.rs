//! Convert pyannote/WeSpeaker ResNet34 safetensors into a diarization `.oasr`
//! (GGUF-v0) pack.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::ggml_runtime::GgufWriteValue;
use crate::models::diarize_pack_import::convert_diarize_safetensors_to_oasr;
use crate::models::local_source_import::{LocalSourceImportError, SafetensorsFile, validate_error};
use crate::models::oasr_metadata::{
    OASR_METADATA_KEY_FEATURE_DIARIZATION, OASR_METADATA_KEY_MODEL_ARCHITECTURE,
    OASR_METADATA_KEY_MODEL_FAMILY, OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
};

use super::{WESPEAKER_GGML_ARCHITECTURE_ID, WESPEAKER_MODEL_FAMILY};

pub(crate) const OASR_FEATURE_DIARIZATION_WESPEAKER_EMBEDDER_V1: &str =
    "wespeaker-resnet34-speaker-embedder-v1";
pub(crate) const WESPEAKER_EXPECTED_SOURCE_NAME: &str = "pyannote/wespeaker-voxceleb-resnet34-LM";
pub(crate) const WESPEAKER_EXPECTED_SOURCE_REVISION: &str =
    "837717ddb9ff5507820346191109dc79c958d614";
pub(crate) const WESPEAKER_EXPECTED_LICENSE_NAME: &str = "CC-BY-4.0";

const SAFETENSORS_METADATA_SOURCE_NAME: &str = "source_name";
const SAFETENSORS_METADATA_SOURCE_REVISION: &str = "source_revision";
const SAFETENSORS_METADATA_LICENSE: &str = "license";

#[derive(Debug, Clone)]
pub struct WeSpeakerImportRequest {
    /// Path to the source WeSpeaker ResNet34 safetensors weight file.
    pub source_safetensors: PathBuf,
    /// Output `.oasr` pack path (must end in `.oasr`).
    pub output_root: PathBuf,
    /// Catalog/local model id recorded in the pack metadata.
    pub model_id: String,
    /// Upstream source identifier, e.g. `pyannote/wespeaker-voxceleb-resnet34-LM`.
    pub source_name: String,
    /// Upstream source revision or local provenance label.
    pub source_revision: String,
    /// License name recorded in the pack metadata. For the pyannote community-1
    /// WeSpeaker weights this must be `CC-BY-4.0`.
    pub license_name: String,
    /// License/source URL recorded in the pack metadata.
    pub license_source: String,
    /// Runtime tensor storage mode. WeSpeaker is shipped f32-only.
    pub quantization: WeSpeakerRuntimeQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeSpeakerImportResult {
    pub output_path: PathBuf,
    pub tensor_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeSpeakerRuntimeQuantizationMode {
    F32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WeSpeakerSourceMetadata {
    source_name: String,
    source_revision: String,
    license_name: String,
}

pub fn convert_local_wespeaker_source_to_runtime_pack(
    request: &WeSpeakerImportRequest,
) -> Result<WeSpeakerImportResult, LocalSourceImportError> {
    validate_request(request)?;
    let source_metadata = validate_source_artifact_metadata(request)?;
    let tensor_count = convert_diarize_safetensors_to_oasr(
        &request.source_safetensors,
        &request.output_root,
        &runtime_metadata(request, &source_metadata),
    )?;
    Ok(WeSpeakerImportResult {
        output_path: request.output_root.clone(),
        tensor_count,
    })
}

fn validate_request(request: &WeSpeakerImportRequest) -> Result<(), LocalSourceImportError> {
    for (value, field) in [
        (&request.model_id, "model_id"),
        (&request.source_name, "source_name"),
        (&request.source_revision, "source_revision"),
        (&request.license_name, "license_name"),
        (&request.license_source, "license_source"),
    ] {
        if value.trim().is_empty() {
            return Err(validate_error(format!(
                "WeSpeaker local-source converter requires non-empty {field}"
            )));
        }
    }
    Ok(())
}

fn validate_source_artifact_metadata(
    request: &WeSpeakerImportRequest,
) -> Result<WeSpeakerSourceMetadata, LocalSourceImportError> {
    let safetensors = SafetensorsFile::open(&request.source_safetensors)?;
    let metadata = &safetensors.header().metadata;
    let source_metadata = WeSpeakerSourceMetadata {
        source_name: required_artifact_metadata(metadata, SAFETENSORS_METADATA_SOURCE_NAME)?,
        source_revision: required_artifact_metadata(
            metadata,
            SAFETENSORS_METADATA_SOURCE_REVISION,
        )?,
        license_name: required_artifact_metadata(metadata, SAFETENSORS_METADATA_LICENSE)?,
    };
    require_expected_artifact_metadata(
        SAFETENSORS_METADATA_SOURCE_NAME,
        &source_metadata.source_name,
        WESPEAKER_EXPECTED_SOURCE_NAME,
    )?;
    require_expected_artifact_metadata(
        SAFETENSORS_METADATA_SOURCE_REVISION,
        &source_metadata.source_revision,
        WESPEAKER_EXPECTED_SOURCE_REVISION,
    )?;
    require_expected_artifact_metadata(
        SAFETENSORS_METADATA_LICENSE,
        &source_metadata.license_name,
        WESPEAKER_EXPECTED_LICENSE_NAME,
    )?;
    require_request_matches_artifact(
        "source_name",
        &request.source_name,
        &source_metadata.source_name,
    )?;
    require_request_matches_artifact(
        "source_revision",
        &request.source_revision,
        &source_metadata.source_revision,
    )?;
    require_request_matches_artifact(
        "license_name",
        &request.license_name,
        &source_metadata.license_name,
    )?;
    Ok(source_metadata)
}

fn required_artifact_metadata(
    metadata: &BTreeMap<String, String>,
    key: &str,
) -> Result<String, LocalSourceImportError> {
    let value = metadata.get(key).ok_or_else(|| {
        validate_error(format!(
            "WeSpeaker source safetensors must include __metadata__.{key}"
        ))
    })?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(validate_error(format!(
            "WeSpeaker source safetensors __metadata__.{key} must not be empty"
        )));
    }
    Ok(trimmed.to_string())
}

fn require_expected_artifact_metadata(
    key: &str,
    actual: &str,
    expected: &str,
) -> Result<(), LocalSourceImportError> {
    if actual == expected {
        return Ok(());
    }
    Err(validate_error(format!(
        "WeSpeaker source safetensors __metadata__.{key} mismatch: expected '{expected}', got '{actual}'"
    )))
}

fn require_request_matches_artifact(
    field: &str,
    actual: &str,
    expected: &str,
) -> Result<(), LocalSourceImportError> {
    if actual.trim() == expected {
        return Ok(());
    }
    Err(validate_error(format!(
        "WeSpeaker import argument {field} must match source safetensors metadata: expected '{expected}', got '{}'",
        actual.trim()
    )))
}

fn runtime_metadata(
    request: &WeSpeakerImportRequest,
    source_metadata: &WeSpeakerSourceMetadata,
) -> BTreeMap<String, GgufWriteValue> {
    let mut metadata = BTreeMap::new();
    let mut put = |key: &str, value: &str| {
        metadata.insert(key.to_string(), GgufWriteValue::String(value.to_string()));
    };
    put("general.architecture", WESPEAKER_GGML_ARCHITECTURE_ID);
    put(OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1);
    put(OASR_METADATA_KEY_MODEL_FAMILY, WESPEAKER_MODEL_FAMILY);
    put(
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        WESPEAKER_GGML_ARCHITECTURE_ID,
    );
    put(
        OASR_METADATA_KEY_FEATURE_DIARIZATION,
        OASR_FEATURE_DIARIZATION_WESPEAKER_EMBEDDER_V1,
    );
    put(
        "openasr.quantization",
        match request.quantization {
            WeSpeakerRuntimeQuantizationMode::F32 => "f32",
        },
    );
    put("openasr.model.id", &request.model_id);
    put("openasr.source.name", &source_metadata.source_name);
    put("openasr.source.revision", &source_metadata.source_revision);
    put("openasr.license.name", &source_metadata.license_name);
    put("openasr.license.source", &request.license_source);
    metadata
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::Write;

    use super::*;

    fn request(path: PathBuf) -> WeSpeakerImportRequest {
        WeSpeakerImportRequest {
            source_safetensors: path,
            output_root: PathBuf::from("out.oasr"),
            model_id: "wespeaker-test".to_string(),
            source_name: WESPEAKER_EXPECTED_SOURCE_NAME.to_string(),
            source_revision: WESPEAKER_EXPECTED_SOURCE_REVISION.to_string(),
            license_name: WESPEAKER_EXPECTED_LICENSE_NAME.to_string(),
            license_source: "https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM"
                .to_string(),
            quantization: WeSpeakerRuntimeQuantizationMode::F32,
        }
    }

    fn write_tiny_safetensors(path: &std::path::Path, metadata: BTreeMap<&str, &str>) {
        let mut header = serde_json::Map::new();
        header.insert(
            "__metadata__".to_string(),
            serde_json::to_value(metadata).unwrap(),
        );
        header.insert(
            "weight".to_string(),
            serde_json::json!({
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            }),
        );
        let header_bytes = serde_json::Value::Object(header).to_string().into_bytes();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&0.0f32.to_le_bytes()).unwrap();
    }

    fn expected_metadata() -> BTreeMap<&'static str, &'static str> {
        BTreeMap::from([
            (
                SAFETENSORS_METADATA_SOURCE_NAME,
                WESPEAKER_EXPECTED_SOURCE_NAME,
            ),
            (
                SAFETENSORS_METADATA_SOURCE_REVISION,
                WESPEAKER_EXPECTED_SOURCE_REVISION,
            ),
            (
                SAFETENSORS_METADATA_LICENSE,
                WESPEAKER_EXPECTED_LICENSE_NAME,
            ),
        ])
    }

    #[test]
    fn source_artifact_metadata_must_match_pinned_wespeaker_source() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("wespeaker.safetensors");
        let mut metadata = expected_metadata();
        metadata.insert(SAFETENSORS_METADATA_LICENSE, "MIT");
        write_tiny_safetensors(&path, metadata);

        let error = validate_source_artifact_metadata(&request(path))
            .expect_err("license mismatch must fail");
        assert!(
            error
                .to_string()
                .contains("__metadata__.license mismatch: expected 'CC-BY-4.0', got 'MIT'"),
            "{error}"
        );
    }

    #[test]
    fn import_arguments_must_match_wespeaker_artifact_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("wespeaker.safetensors");
        write_tiny_safetensors(&path, expected_metadata());
        let mut request = request(path);
        request.license_name = "MIT".to_string();

        let error = validate_source_artifact_metadata(&request)
            .expect_err("caller-supplied license mismatch must fail");
        assert!(
            error
                .to_string()
                .contains("argument license_name must match source safetensors metadata"),
            "{error}"
        );
    }

    #[test]
    fn source_artifact_metadata_is_required() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("wespeaker.safetensors");
        write_tiny_safetensors(&path, BTreeMap::new());

        let error = validate_source_artifact_metadata(&request(path))
            .expect_err("missing artifact metadata must fail");
        assert!(
            error
                .to_string()
                .contains("must include __metadata__.source_name"),
            "{error}"
        );
    }
}
