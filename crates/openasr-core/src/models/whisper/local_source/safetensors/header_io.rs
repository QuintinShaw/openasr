use std::{fs::File, io::Read, path::Path};

use super::*;

pub(super) fn read_header_and_data_lengths_from_file(
    path: &Path,
) -> Result<(u64, u64, Vec<u8>), WhisperLocalSourceError> {
    let mut file = File::open(path).map_err(|source| WhisperLocalSourceError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let header_length_bytes = read_header_length_prefix(&mut file, path)?;
    let header_bytes = read_header_bytes(&mut file, path, header_length_bytes)?;
    let data_length_bytes = compute_data_length_bytes(&file, path, header_length_bytes)?;
    Ok((header_length_bytes, data_length_bytes, header_bytes))
}

fn read_header_length_prefix(file: &mut File, path: &Path) -> Result<u64, WhisperLocalSourceError> {
    let mut header_length_prefix = [0_u8; SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES];
    file.read_exact(&mut header_length_prefix)
        .map_err(|source| WhisperLocalSourceError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    let header_length_bytes = u64::from_le_bytes(header_length_prefix);
    validate_header_length_max(header_length_bytes)?;
    Ok(header_length_bytes)
}

fn read_header_bytes(
    file: &mut File,
    path: &Path,
    header_length_bytes: u64,
) -> Result<Vec<u8>, WhisperLocalSourceError> {
    let header_length_usize = header_length_to_usize(header_length_bytes)?;
    let mut header_bytes = vec![0_u8; header_length_usize];
    file.read_exact(&mut header_bytes)
        .map_err(|source| WhisperLocalSourceError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(header_bytes)
}

fn compute_data_length_bytes(
    file: &File,
    path: &Path,
    header_length_bytes: u64,
) -> Result<u64, WhisperLocalSourceError> {
    let file_size_bytes = file
        .metadata()
        .map_err(|source| WhisperLocalSourceError::Read {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    let header_section_len = checked_u64_add_with_context(
        SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES as u64,
        header_length_bytes,
        format!("safetensors header length {header_length_bytes} overflows file indexing bounds"),
    )?;
    if file_size_bytes < header_section_len {
        return Err(header_length_exceeds_available_error(
            header_length_bytes,
            file_size_bytes.saturating_sub(SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES as u64),
        ));
    }
    Ok(file_size_bytes - header_section_len)
}

fn validate_header_length_max(header_length_bytes: u64) -> Result<(), WhisperLocalSourceError> {
    if header_length_bytes > SAFETENSORS_HEADER_MAX_BYTES_V0 {
        return Err(validate_error(format!(
            "safetensors header length {header_length_bytes} exceeds max allowed {SAFETENSORS_HEADER_MAX_BYTES_V0} bytes"
        )));
    }
    Ok(())
}

fn header_length_to_usize(header_length_bytes: u64) -> Result<usize, WhisperLocalSourceError> {
    usize::try_from(header_length_bytes).map_err(|_| {
        validate_error(format!(
            "safetensors header length {header_length_bytes} is not representable on this platform"
        ))
    })
}

fn header_length_exceeds_available_error(
    header_length_bytes: u64,
    available_file_bytes: u64,
) -> WhisperLocalSourceError {
    validate_error(format!(
        "safetensors header length {header_length_bytes} exceeds available file bytes {available_file_bytes}",
    ))
}
