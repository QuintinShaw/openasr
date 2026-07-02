//! Shared safetensors → diarization `.oasr` (GGUF-v0) conversion for the
//! speaker-diarization models (WeSpeaker embedder, pyannote segmenter).
//!
//! By default tensors are stored as raw `F32` with the GGUF dims equal to the
//! logical (safetensors) shape — **no** ggml dim reversal, because these weights
//! are consumed by pure-Rust forward passes.

use std::collections::BTreeMap;
use std::path::Path;

use crate::ggml_runtime::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, write_gguf_file_v0,
};
use crate::models::local_source_import::{
    LocalSourceImportError, SafetensorsFile, decode_safetensors_payload_as_f32, validate_error,
    validate_output_pack_extension,
};

/// Convert `source_safetensors` into a diarization `.oasr` pack at `output_root`
/// with the given pack `metadata`, returning the tensor count. Every tensor is
/// passed through as raw `F32`.
pub(crate) fn convert_diarize_safetensors_to_oasr(
    source_safetensors: &Path,
    output_root: &Path,
    metadata: &BTreeMap<String, GgufWriteValue>,
) -> Result<usize, LocalSourceImportError> {
    validate_output_pack_extension(output_root)?;
    let safetensors = SafetensorsFile::open(source_safetensors)?;
    let tensors = build_diarize_tensors(&safetensors)?;
    write_gguf_file_v0(output_root, metadata, &tensors).map_err(|error| {
        validate_error(format!(
            "diarization GGUF writer failed for '{}': {error}",
            output_root.display()
        ))
    })?;
    Ok(tensors.len())
}

fn build_diarize_tensors(
    safetensors: &SafetensorsFile,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut out = Vec::with_capacity(safetensors.header().tensors.len());
    for tensor in &safetensors.header().tensors {
        let data = safetensors.tensor_data(tensor)?;
        let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
        // GGUF requires rank 1..=4; a safetensors scalar (rank 0) becomes `[1]`.
        let dims: Vec<u64> = if tensor.shape.is_empty() {
            vec![1]
        } else {
            tensor.shape.clone()
        };
        let expected: u64 = dims.iter().product();
        if expected as usize != values.len() {
            return Err(validate_error(format!(
                "diarization tensor '{}' has {} values but shape {dims:?} needs {expected}",
                tensor.name,
                values.len(),
            )));
        }
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in &values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        out.push(GgufWriteTensor {
            name: tensor.name.clone(),
            dims,
            tensor_type: GgufWriteTensorType::F32,
            data: bytes,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tiny_safetensors(path: &Path, name: &str, shape: &[u64], values: &[f32]) {
        let mut header = serde_json::Map::new();
        header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [0, values.len() * 4],
            }),
        );
        let header_bytes = serde_json::Value::Object(header).to_string().into_bytes();
        let mut bytes = Vec::with_capacity(8 + header_bytes.len() + values.len() * 4);
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn diarize_pack_stores_f32_tensor() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.safetensors");
        let output = temp.path().join("wespeaker-f32.oasr");
        let values: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.17).sin()).collect();
        write_tiny_safetensors(&source, "resnet.conv1.weight", &[32], &values);

        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufWriteValue::String("wespeaker-resnet34".to_string()),
        );
        let count = convert_diarize_safetensors_to_oasr(&source, &output, &metadata).unwrap();
        assert_eq!(count, 1);

        let index = crate::ggml_runtime::read_gguf_tensor_index(&output).unwrap();
        let tensor = index.get("resnet.conv1.weight").unwrap();
        assert_eq!(tensor.type_name, "f32");
        let reader = crate::ggml_runtime::GgufTensorDataReader::from_path(&output).unwrap();
        let restored = reader
            .host_tensor_f32_copy_dequantized_by_name("resnet.conv1.weight", &[32])
            .unwrap();
        assert_eq!(restored, values);
    }
}
