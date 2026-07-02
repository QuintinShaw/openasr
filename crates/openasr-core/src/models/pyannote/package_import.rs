//! Convert the extracted pyannote segmentation-3.0 safetensors weights into a
//! diarization `.oasr` (GGUF-v0) pack.
//!
//! The raw-F32 passthrough conversion (dims = logical shape, no ggml reversal)
//! lives in [`crate::models::diarize_pack_import`], shared with the WeSpeaker
//! embedder; this module only supplies the pyannote pack metadata. The source
//! safetensors is produced from the un-gated `onnx-community` ONNX mirror by
//! `tooling/publish-model/scripts/pyannote_extract.py`. The runtime loader is
//! [`crate::diarize::segment::PyannoteSegmenter::from_oasr`]; the
//! `diarize::segment::tests::oasr_roundtrip_matches_safetensors` test asserts the
//! converted pack yields a byte-identical forward pass to the safetensors fast
//! path.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::ggml_runtime::GgufWriteValue;
use crate::models::diarize_pack_import::convert_diarize_safetensors_to_oasr;
use crate::models::local_source_import::LocalSourceImportError;
use crate::models::oasr_metadata::{
    OASR_METADATA_KEY_FEATURE_DIARIZATION, OASR_METADATA_KEY_MODEL_ARCHITECTURE,
    OASR_METADATA_KEY_MODEL_FAMILY, OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
};

use super::{PYANNOTE_GGML_ARCHITECTURE_ID, PYANNOTE_MODEL_FAMILY};

/// `openasr.features.diarization` value tagging a pyannote segmenter pack.
pub(crate) const OASR_FEATURE_DIARIZATION_PYANNOTE_SEGMENTER_V1: &str = "pyannote-segmenter-v1";

#[derive(Debug, Clone)]
pub struct PyannoteImportRequest {
    /// Path to the source `pyannote_seg.safetensors` weight file.
    pub source_safetensors: PathBuf,
    /// Output `.oasr` pack path (must end in `.oasr`).
    pub output_root: PathBuf,
    /// Catalog model id recorded in the pack metadata.
    pub model_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyannoteImportResult {
    pub output_path: PathBuf,
    pub tensor_count: usize,
}

/// Convert a local pyannote-seg safetensors source into a diarization `.oasr` pack.
pub fn convert_local_pyannote_source_to_runtime_pack(
    request: &PyannoteImportRequest,
) -> Result<PyannoteImportResult, LocalSourceImportError> {
    let tensor_count = convert_diarize_safetensors_to_oasr(
        &request.source_safetensors,
        &request.output_root,
        &runtime_metadata(request),
    )?;
    Ok(PyannoteImportResult {
        output_path: request.output_root.clone(),
        tensor_count,
    })
}

fn runtime_metadata(request: &PyannoteImportRequest) -> BTreeMap<String, GgufWriteValue> {
    let mut metadata = BTreeMap::new();
    let mut put = |key: &str, value: &str| {
        metadata.insert(key.to_string(), GgufWriteValue::String(value.to_string()));
    };
    put("general.architecture", PYANNOTE_GGML_ARCHITECTURE_ID);
    put(OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1);
    put(OASR_METADATA_KEY_MODEL_FAMILY, PYANNOTE_MODEL_FAMILY);
    put(
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        PYANNOTE_GGML_ARCHITECTURE_ID,
    );
    put(
        OASR_METADATA_KEY_FEATURE_DIARIZATION,
        OASR_FEATURE_DIARIZATION_PYANNOTE_SEGMENTER_V1,
    );
    put("openasr.model.id", &request.model_id);
    metadata
}
