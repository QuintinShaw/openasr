use super::*;

pub(super) fn safetensors_dtype_size_bytes(dtype: &str) -> Option<u64> {
    match dtype.trim().to_ascii_uppercase().as_str() {
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" => Some(1),
        "I16" | "U16" | "F16" | "BF16" => Some(2),
        "I32" | "U32" | "F32" => Some(4),
        "I64" | "U64" | "F64" => Some(8),
        _ => None,
    }
}

pub(super) fn tensor_element_count(
    shape: &[u64],
    tensor_name: &str,
) -> Result<u64, WhisperLocalSourceError> {
    shape.iter().try_fold(1_u64, |count, dimension| {
        count.checked_mul(*dimension).ok_or_else(|| {
            validate_error(format!(
                "safetensors tensor '{tensor_name}' shape element-count overflow"
            ))
        })
    })
}

pub(super) fn validate_safetensors_tensor_offset_ranges(
    tensors: &[SafetensorsTensorHeaderV0],
    data_length_bytes: u64,
) -> Result<(), WhisperLocalSourceError> {
    let mut ranges = tensors
        .iter()
        .map(|tensor| {
            (
                tensor.data_offsets[0],
                tensor.data_offsets[1],
                tensor.name.as_str(),
            )
        })
        .collect::<Vec<_>>();
    ranges.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

    let mut expected_offset = 0_u64;
    for (start, end, name) in ranges {
        if start > expected_offset {
            return Err(validate_error(format!(
                "safetensors tensor '{name}' data_offsets start ({start}) leaves a gap before expected offset {expected_offset}"
            )));
        }
        if start < expected_offset {
            return Err(validate_error(format!(
                "safetensors tensor '{name}' data_offsets start ({start}) overlaps previous tensor range ending at {expected_offset}"
            )));
        }
        if end > expected_offset {
            expected_offset = end;
        }
    }
    if expected_offset != data_length_bytes {
        return Err(validate_error(format!(
            "safetensors tensor ranges must fully cover data section length {data_length_bytes}; covered length is {expected_offset}"
        )));
    }
    Ok(())
}
