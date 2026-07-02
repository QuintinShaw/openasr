use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LocalSourceImportError {
    #[error("could not read model source file '{path}': {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not parse model source artifact '{path}': {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("{0}")]
    Validate(String),
}

pub(crate) fn validate_error(message: impl Into<String>) -> LocalSourceImportError {
    LocalSourceImportError::Validate(message.into())
}

/// Enforce the user-facing runtime-pack extension contract on a converter's
/// output path. Every `convert_local_*_to_runtime_pack` entry point calls this so
/// a direct library caller is held to the same `.oasr`-only rule as the CLI — the
/// on-disk container stays GGUF-structured, but it is only ever *produced* as
/// `.oasr`. Shares [`crate::has_openasr_runtime_pack_extension`] as the single
/// source of truth (see also the CLI's run/import path validation).
pub(crate) fn validate_output_pack_extension(
    output_root: &Path,
) -> Result<(), LocalSourceImportError> {
    if crate::has_openasr_runtime_pack_extension(output_root) {
        return Ok(());
    }
    Err(validate_error(format!(
        "local-source converter output '{}' must end with .oasr (OpenASR native runtime pack)",
        output_root.display()
    )))
}

pub(crate) fn read_source_json_file<T: for<'de> Deserialize<'de>>(
    root: &Path,
    relative_path: &str,
) -> Result<T, LocalSourceImportError> {
    let path = root.join(relative_path);
    let bytes = read_source_file_bytes(root, relative_path)?;
    serde_json::from_slice(&bytes).map_err(|source| LocalSourceImportError::Parse { path, source })
}

pub(crate) fn read_source_file_bytes(
    root: &Path,
    relative_path: &str,
) -> Result<Vec<u8>, LocalSourceImportError> {
    let path = root.join(relative_path);
    std::fs::read(&path).map_err(|source| LocalSourceImportError::Read { path, source })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SafetensorsTensorHeader {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SafetensorsHeader {
    pub header_length_bytes: u64,
    pub data_length_bytes: u64,
    pub metadata: BTreeMap<String, String>,
    pub tensors: Vec<SafetensorsTensorHeader>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSafetensorsTensorHeader {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

pub(crate) struct SafetensorsFile {
    path: PathBuf,
    _file: File,
    mmap: Mmap,
    data_offset_bytes: usize,
    header: SafetensorsHeader,
    by_name: BTreeMap<String, SafetensorsTensorHeader>,
}

impl SafetensorsFile {
    pub(crate) fn open(path: impl AsRef<Path>) -> Result<Self, LocalSourceImportError> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path).map_err(|source| LocalSourceImportError::Read {
            path: path.clone(),
            source,
        })?;
        let mut header_len_prefix = [0_u8; 8];
        file.read_exact(&mut header_len_prefix)
            .map_err(|source| LocalSourceImportError::Read {
                path: path.clone(),
                source,
            })?;
        let header_length_bytes = u64::from_le_bytes(header_len_prefix);
        let header_length = usize::try_from(header_length_bytes).map_err(|_| {
            validate_error(format!(
                "safetensors header length {header_length_bytes} is not representable on this platform"
            ))
        })?;
        let mut header_bytes = vec![0_u8; header_length];
        file.read_exact(&mut header_bytes)
            .map_err(|source| LocalSourceImportError::Read {
                path: path.clone(),
                source,
            })?;
        let raw_value: serde_json::Value =
            serde_json::from_slice(&header_bytes).map_err(|source| {
                LocalSourceImportError::Parse {
                    path: path.clone(),
                    source,
                }
            })?;
        let raw_object = raw_value
            .as_object()
            .ok_or_else(|| validate_error("safetensors header must be a JSON object"))?;
        let mut metadata = BTreeMap::new();
        let mut tensors = Vec::new();
        for (name, value) in raw_object {
            if name == "__metadata__" {
                metadata = serde_json::from_value(value.clone()).map_err(|source| {
                    LocalSourceImportError::Parse {
                        path: path.clone(),
                        source,
                    }
                })?;
                continue;
            }
            let raw: RawSafetensorsTensorHeader =
                serde_json::from_value(value.clone()).map_err(|source| {
                    LocalSourceImportError::Parse {
                        path: path.clone(),
                        source,
                    }
                })?;
            if raw.dtype.trim().is_empty() {
                return Err(validate_error(format!(
                    "safetensors tensor '{name}' dtype must not be empty"
                )));
            }
            if raw.data_offsets[1] < raw.data_offsets[0] {
                return Err(validate_error(format!(
                    "safetensors tensor '{name}' has inverted offsets {:?}",
                    raw.data_offsets
                )));
            }
            tensors.push(SafetensorsTensorHeader {
                name: name.clone(),
                dtype: raw.dtype,
                shape: raw.shape,
                data_offsets: raw.data_offsets,
            });
        }
        if tensors.is_empty() {
            return Err(validate_error(
                "safetensors header must include at least one tensor entry",
            ));
        }
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        let file_metadata = file
            .metadata()
            .map_err(|source| LocalSourceImportError::Read {
                path: path.clone(),
                source,
            })?;
        let total_len = file_metadata.len();
        let data_offset_bytes = usize::try_from(8_u64.saturating_add(header_length_bytes))
            .map_err(|_| validate_error("safetensors data offset overflowed platform usize"))?;
        if total_len < (8_u64.saturating_add(header_length_bytes)) {
            return Err(validate_error(format!(
                "safetensors file '{}' is smaller than its declared header",
                path.display()
            )));
        }
        let data_length_bytes = total_len - 8_u64 - header_length_bytes;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|source| LocalSourceImportError::Read {
            path: path.clone(),
            source,
        })?;
        let mut by_name = BTreeMap::new();
        for tensor in &tensors {
            by_name.insert(tensor.name.clone(), tensor.clone());
        }
        Ok(Self {
            path,
            _file: file,
            mmap,
            data_offset_bytes,
            header: SafetensorsHeader {
                header_length_bytes,
                data_length_bytes,
                metadata,
                tensors,
            },
            by_name,
        })
    }

    pub(crate) fn header(&self) -> &SafetensorsHeader {
        &self.header
    }

    pub(crate) fn tensor(&self, name: &str) -> Option<&SafetensorsTensorHeader> {
        self.by_name.get(name)
    }

    pub(crate) fn tensor_data(
        &self,
        tensor: &SafetensorsTensorHeader,
    ) -> Result<&[u8], LocalSourceImportError> {
        let start = usize::try_from(tensor.data_offsets[0]).map_err(|_| {
            validate_error(format!(
                "safetensors tensor '{}' start offset does not fit usize",
                tensor.name
            ))
        })?;
        let end = usize::try_from(tensor.data_offsets[1]).map_err(|_| {
            validate_error(format!(
                "safetensors tensor '{}' end offset does not fit usize",
                tensor.name
            ))
        })?;
        let absolute_start = self
            .data_offset_bytes
            .checked_add(start)
            .ok_or_else(|| validate_error("safetensors absolute start offset overflow"))?;
        let absolute_end = self
            .data_offset_bytes
            .checked_add(end)
            .ok_or_else(|| validate_error("safetensors absolute end offset overflow"))?;
        self.mmap.get(absolute_start..absolute_end).ok_or_else(|| {
            validate_error(format!(
                "safetensors tensor '{}' data range is out of bounds in '{}'",
                tensor.name,
                self.path.display()
            ))
        })
    }
}

pub(crate) fn decode_safetensors_payload_as_f32(
    tensor_name: &str,
    dtype: &str,
    data: &[u8],
) -> Result<Vec<f32>, LocalSourceImportError> {
    match dtype {
        "F32" => decode_f32_payload(tensor_name, data),
        "F16" => decode_f16_payload_as_f32(tensor_name, data),
        "BF16" => decode_bf16_payload_as_f32(tensor_name, data),
        other => Err(validate_error(format!(
            "safetensors tensor '{tensor_name}' dtype '{other}' is not supported for runtime import"
        ))),
    }
}

pub(crate) fn decode_safetensors_payload_as_f16_bits(
    tensor_name: &str,
    dtype: &str,
    data: &[u8],
) -> Result<Vec<u16>, LocalSourceImportError> {
    match dtype {
        "F16" => decode_f16_payload_bits(tensor_name, data),
        "F32" | "BF16" => {
            let values = decode_safetensors_payload_as_f32(tensor_name, dtype, data)?;
            Ok(values.into_iter().map(f32_to_f16_bits).collect())
        }
        other => Err(validate_error(format!(
            "safetensors tensor '{tensor_name}' dtype '{other}' cannot be converted to f16"
        ))),
    }
}

pub(crate) fn encode_f16_bits_le(values: Vec<u16>) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(values.len() * 2);
    for value in values {
        encoded.extend_from_slice(&value.to_le_bytes());
    }
    encoded
}

pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x03ff) as u32;
    let out = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut mant = mantissa;
            let mut exp = 113_u32;
            while mant & 0x0400 == 0 {
                mant <<= 1;
                exp = exp.saturating_sub(1);
            }
            mant &= 0x03ff;
            sign | (exp << 23) | (mant << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 112) << 23) | (mantissa << 13)
    };
    f32::from_bits(out)
}

pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ff_ff;

    if exponent == 255 {
        return sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 };
    }
    if exponent <= 112 {
        if exponent < 103 {
            return sign;
        }
        let shift = (126 - exponent) as u32;
        let mut subnormal = (mantissa | 0x80_00_00) >> shift;
        if (subnormal & 0x0000_1000) != 0 {
            subnormal = subnormal.saturating_add(0x0000_2000);
        }
        return sign | ((subnormal >> 13) as u16);
    }
    if exponent >= 143 {
        return sign | 0x7c00;
    }

    let exp = (exponent - 112) as u16;
    let mut frac = mantissa;
    if (frac & 0x0000_1000) != 0 {
        frac = frac.saturating_add(0x0000_2000);
        if (frac & 0x0080_0000) != 0 {
            return sign | ((exp + 1) << 10);
        }
    }
    sign | (exp << 10) | ((frac >> 13) as u16)
}

pub(crate) fn tensor_element_count(
    tensor_name: &str,
    dims: &[u64],
) -> Result<usize, LocalSourceImportError> {
    let mut count = 1_u64;
    for dim in dims {
        count = count.checked_mul(*dim).ok_or_else(|| {
            validate_error(format!(
                "tensor '{tensor_name}' element-count overflow for dims {dims:?}"
            ))
        })?;
    }
    usize::try_from(count).map_err(|_| {
        validate_error(format!(
            "tensor '{tensor_name}' element-count does not fit usize for dims {dims:?}"
        ))
    })
}

fn decode_f32_payload(tensor_name: &str, data: &[u8]) -> Result<Vec<f32>, LocalSourceImportError> {
    if !data.len().is_multiple_of(4) {
        return Err(validate_error(format!(
            "safetensors tensor '{tensor_name}' f32 payload length {} is not divisible by 4",
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn decode_f16_payload_bits(
    tensor_name: &str,
    data: &[u8],
) -> Result<Vec<u16>, LocalSourceImportError> {
    if !data.len().is_multiple_of(2) {
        return Err(validate_error(format!(
            "safetensors tensor '{tensor_name}' f16 payload length {} is not divisible by 2",
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        out.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(out)
}

fn decode_f16_payload_as_f32(
    tensor_name: &str,
    data: &[u8],
) -> Result<Vec<f32>, LocalSourceImportError> {
    let bits = decode_f16_payload_bits(tensor_name, data)?;
    Ok(bits.into_iter().map(f16_bits_to_f32).collect())
}

fn decode_bf16_payload_as_f32(
    tensor_name: &str,
    data: &[u8],
) -> Result<Vec<f32>, LocalSourceImportError> {
    if !data.len().is_multiple_of(2) {
        return Err(validate_error(format!(
            "safetensors tensor '{tensor_name}' bf16 payload length {} is not divisible by 2",
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let upper = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
        out.push(f32::from_bits(upper << 16));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{LocalSourceImportError, validate_output_pack_extension};
    use std::path::Path;

    #[test]
    fn output_pack_extension_accepts_oasr() {
        validate_output_pack_extension(Path::new("/tmp/whisper-small.oasr"))
            .expect(".oasr output must be accepted");
        validate_output_pack_extension(Path::new("/tmp/whisper-small.OASR"))
            .expect(".oasr output is case-insensitive");
    }

    #[test]
    fn output_pack_extension_rejects_legacy_gguf() {
        // The library boundary holds direct converter callers to the same
        // `.oasr`-only contract the CLI enforces — legacy `.gguf` is rejected.
        let error = validate_output_pack_extension(Path::new("/tmp/whisper-small.gguf"))
            .expect_err(".gguf output must be rejected");
        match error {
            LocalSourceImportError::Validate(message) => {
                assert!(
                    message.contains(".oasr"),
                    "message should cite .oasr: {message}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn output_pack_extension_rejects_missing_extension() {
        validate_output_pack_extension(Path::new("/tmp/whisper-small"))
            .expect_err("extensionless output must be rejected");
    }
}
