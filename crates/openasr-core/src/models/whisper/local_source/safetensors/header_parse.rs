use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

use super::dtype::{
    safetensors_dtype_size_bytes, tensor_element_count, validate_safetensors_tensor_offset_ranges,
};
use super::*;

pub(super) fn parse_safetensors_header_json_object(
    header_length_bytes: u64,
    data_length_bytes: u64,
    header_bytes: &[u8],
) -> Result<SafetensorsHeaderV0, WhisperLocalSourceError> {
    let header_text = std::str::from_utf8(header_bytes).map_err(|error| {
        validate_error(format!(
            "safetensors header is not valid UTF-8 JSON: {error}"
        ))
    })?;
    reject_duplicate_json_keys(header_text).map_err(|error| {
        validate_error(format!(
            "safetensors header has duplicate JSON keys: {error}"
        ))
    })?;
    let header_value: Value = serde_json::from_str(header_text).map_err(|error| {
        validate_error(format!("safetensors header JSON parse failed: {error}"))
    })?;
    let Value::Object(entries) = header_value else {
        return Err(validate_error(
            "safetensors header must be a JSON object".to_string(),
        ));
    };

    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::new();
    for (name, value) in entries {
        if name == "__metadata__" {
            metadata = serde_json::from_value(value).map_err(|error| {
                validate_error(format!(
                    "safetensors __metadata__ must be an object of string values: {error}"
                ))
            })?;
            continue;
        }
        tensors.push(parse_safetensors_tensor_header_entry(
            name,
            value,
            data_length_bytes,
        )?);
    }
    if tensors.is_empty() {
        return Err(validate_error(
            "safetensors header must include at least one tensor entry".to_string(),
        ));
    }
    validate_safetensors_tensor_offset_ranges(&tensors, data_length_bytes)?;
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(SafetensorsHeaderV0 {
        header_length_bytes,
        data_length_bytes,
        metadata,
        tensors,
    })
}

fn parse_safetensors_tensor_header_entry(
    name: String,
    value: Value,
    data_length_bytes: u64,
) -> Result<SafetensorsTensorHeaderV0, WhisperLocalSourceError> {
    let raw = parse_raw_safetensors_tensor_header(&name, value)?;
    let dtype_size_bytes = parse_safetensors_tensor_dtype_size(&name, &raw.dtype)?;
    let expected_elements = tensor_element_count(&raw.shape, &name)?;
    let [start, end] = raw.data_offsets;
    validate_safetensors_data_offsets(&name, start, end, data_length_bytes)?;
    let expected_bytes =
        checked_expected_safetensors_tensor_bytes(&name, expected_elements, dtype_size_bytes)?;
    let actual_bytes = end - start;
    if actual_bytes != expected_bytes {
        return Err(validate_error(format!(
            "safetensors tensor '{name}' byte range ({actual_bytes}) does not match expected bytes ({expected_bytes}) from dtype '{}' and shape {:?}",
            raw.dtype, raw.shape
        )));
    }
    Ok(SafetensorsTensorHeaderV0 {
        name,
        dtype: raw.dtype,
        shape: raw.shape,
        data_offsets: [start, end],
    })
}

fn parse_raw_safetensors_tensor_header(
    name: &str,
    value: Value,
) -> Result<RawSafetensorsTensorHeader, WhisperLocalSourceError> {
    let raw: RawSafetensorsTensorHeader = serde_json::from_value(value).map_err(|error| {
        validate_error(format!(
            "safetensors tensor '{name}' metadata is malformed: {error}"
        ))
    })?;
    if raw.dtype.trim().is_empty() {
        return Err(validate_error(format!(
            "safetensors tensor '{name}' dtype must not be empty"
        )));
    }
    Ok(raw)
}

fn parse_safetensors_tensor_dtype_size(
    name: &str,
    dtype: &str,
) -> Result<u64, WhisperLocalSourceError> {
    safetensors_dtype_size_bytes(dtype).ok_or_else(|| {
        validate_error(format!(
            "safetensors tensor '{name}' dtype '{dtype}' is not supported by M64C parser",
        ))
    })
}

fn validate_safetensors_data_offsets(
    name: &str,
    start: u64,
    end: u64,
    data_length_bytes: u64,
) -> Result<(), WhisperLocalSourceError> {
    if end < start {
        return Err(tensor_validation_error(
            name,
            format!("data_offsets end ({end}) must be greater than or equal to start ({start})"),
        ));
    }
    if end > data_length_bytes {
        return Err(tensor_validation_error(
            name,
            format!("data_offsets end ({end}) exceeds data section length ({data_length_bytes})"),
        ));
    }
    Ok(())
}

fn checked_expected_safetensors_tensor_bytes(
    name: &str,
    expected_elements: u64,
    dtype_size_bytes: u64,
) -> Result<u64, WhisperLocalSourceError> {
    expected_elements
        .checked_mul(dtype_size_bytes)
        .ok_or_else(|| {
            validate_error(format!(
                "safetensors tensor '{name}' expected byte size overflow from shape/dtype"
            ))
        })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSafetensorsTensorHeader {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}
