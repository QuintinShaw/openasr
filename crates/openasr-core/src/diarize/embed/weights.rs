//! Loader for speaker-embedder weight packs.
//!
//! Unlike the tiny vendored Stream-VAD model, speaker embedders are delivered as
//! pulled `.oasr` packs and production runtime construction receives a verified
//! preflight rather than reopening a path. Raw safetensors remain a test-only
//! converter/parity fixture. Packs are materialized into logical f32 buffers
//! for the pure-Rust forward passes.

use std::collections::BTreeMap;
#[cfg(test)]
use std::path::Path;

use thiserror::Error;

#[cfg(test)]
use crate::models::{
    aux_pack_registry::AuxPackKind,
    pack_verifier::{PackCandidate, PackRoute, PackVerifier},
};

#[derive(Debug, Error)]
pub enum WeightsError {
    #[error("weights file is truncated (len {len}, need {need})")]
    Truncated { len: usize, need: usize },
    #[error("weights header is not valid JSON: {0}")]
    Header(String),
    #[error("weights are missing tensor '{0}'")]
    Missing(String),
    #[error("tensor '{name}' has dtype '{dtype}', only F32 is supported in raw safetensors")]
    Dtype { name: String, dtype: String },
    #[error("tensor '{name}' data range is out of bounds")]
    Bounds { name: String },
    #[error("tensor '{name}' has {got} floats but shape {shape:?} needs {want}")]
    SizeMismatch {
        name: String,
        got: usize,
        want: usize,
        shape: Vec<usize>,
    },
    #[error("tensor '{name}' has shape {got:?}, expected {want:?}")]
    ShapeMismatch {
        name: String,
        got: Vec<usize>,
        want: Vec<usize>,
    },
    #[error("weights contain unexpected tensor '{0}'")]
    Unexpected(String),
    #[error("{0}")]
    InvalidInput(String),
    #[error("gguf `.oasr` pack read failed: {0}")]
    Gguf(String),
}

struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// A name-keyed bag of `f32` tensors loaded from a safetensors file.
pub(crate) struct Weights {
    tensors: BTreeMap<String, Tensor>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct SafetensorsWeightsQuote {
    pub(crate) retained_bytes: u64,
    pub(crate) parser_peak_bytes: u64,
}

impl Weights {
    pub(crate) fn quoted_persistent_host_commitment_bytes(
        tensor_index: &crate::GgufTensorIndex,
    ) -> Result<u64, WeightsError> {
        let mut bytes = allocation_commitment(std::mem::size_of::<Self>())?;
        for tensor in tensor_index.tensors() {
            let elements = tensor.num_elements().ok_or_else(|| {
                WeightsError::InvalidInput(format!(
                    "redimnet tensor '{}' element count overflow",
                    tensor.name
                ))
            })?;
            let data_bytes = elements
                .checked_mul(std::mem::size_of::<f32>() as u64)
                .ok_or_else(|| {
                    WeightsError::InvalidInput(format!(
                        "redimnet tensor '{}' f32 byte count overflow",
                        tensor.name
                    ))
                })?;
            let shape_bytes = (tensor.dims.len() as u64)
                .checked_mul(std::mem::size_of::<usize>() as u64)
                .ok_or_else(|| {
                    WeightsError::InvalidInput(format!(
                        "redimnet tensor '{}' shape byte count overflow",
                        tensor.name
                    ))
                })?;
            for commitment in [
                allocation_commitment_u64(tensor.name.len() as u64)?,
                allocation_commitment_u64(shape_bytes)?,
                allocation_commitment_u64(data_bytes)?,
                HOST_ALLOCATION_PAGE_BYTES,
            ] {
                bytes = bytes.checked_add(commitment).ok_or_else(|| {
                    WeightsError::InvalidInput(
                        "redimnet quoted weight byte sum overflow".to_string(),
                    )
                })?;
            }
        }
        Ok(bytes)
    }

    #[cfg(test)]
    pub(crate) fn quoted_safetensors_materialization(
        bytes: &[u8],
    ) -> Result<SafetensorsWeightsQuote, WeightsError> {
        let path = Path::new("<raw-safetensors-runtime-source>");
        let (_, header) = crate::models::local_source_import::parse_safetensors_header(path, bytes)
            .map_err(|error| WeightsError::Header(error.to_string()))?;
        let retained_bytes = quote_safetensors_header(&header)?;
        // Header parsing has two non-overlapping phases: duplicate-key
        // validation and the typed descriptor tree. A page per descriptor
        // sub-allocation (top-level node, name/dtype, shape and offsets), plus
        // two copies of the serialized header, is an allocator-independent
        // upper commitment without tying admission to serde_json internals.
        let descriptor_bytes = u64::try_from(header.tensors.len())
            .unwrap_or(u64::MAX)
            .checked_mul(HOST_ALLOCATION_PAGE_BYTES.saturating_mul(4))
            .ok_or_else(|| {
                WeightsError::InvalidInput(
                    "safetensors descriptor parser byte quote overflow".to_string(),
                )
            })?;
        let parser_peak_bytes = header
            .header_length_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(descriptor_bytes))
            .ok_or_else(|| {
                WeightsError::InvalidInput(
                    "safetensors parser peak byte quote overflow".to_string(),
                )
            })?;
        Ok(SafetensorsWeightsQuote {
            retained_bytes,
            parser_peak_bytes,
        })
    }

    /// Parse a safetensors byte buffer.
    #[cfg(test)]
    pub(crate) fn from_safetensors(bytes: &[u8]) -> Result<Self, WeightsError> {
        let path = Path::new("<raw-safetensors-runtime-source>");
        let (data_offset, header) =
            crate::models::local_source_import::parse_safetensors_header(path, bytes)
                .map_err(|error| WeightsError::Header(error.to_string()))?;
        let mut tensors = BTreeMap::new();
        for info in header.tensors {
            let name = info.name;
            if info.dtype != "F32" {
                return Err(WeightsError::Dtype {
                    name,
                    dtype: info.dtype,
                });
            }
            let start = usize::try_from(info.data_offsets[0])
                .ok()
                .and_then(|offset| data_offset.checked_add(offset))
                .ok_or_else(|| WeightsError::Bounds { name: name.clone() })?;
            let end = usize::try_from(info.data_offsets[1])
                .ok()
                .and_then(|offset| data_offset.checked_add(offset))
                .ok_or_else(|| WeightsError::Bounds { name: name.clone() })?;
            let payload = bytes
                .get(start..end)
                .ok_or_else(|| WeightsError::Bounds { name: name.clone() })?;
            let shape = info
                .shape
                .into_iter()
                .map(usize::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| {
                    WeightsError::InvalidInput(format!(
                        "tensor '{name}' shape does not fit platform usize"
                    ))
                })?;
            let floats = payload
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            tensors.insert(
                name,
                Tensor {
                    shape,
                    data: floats,
                },
            );
        }
        Ok(Self { tensors })
    }

    /// Parse a diarization `.oasr` (GGUF-v0) pack. Diarization packs keep GGUF
    /// dims equal to the logical safetensors shape — these weights are consumed
    /// by pure-Rust forward passes, so no ggml dim reversal is applied on write
    /// or read. Quantized tensors are dequantized here into that same logical
    /// f32 order.
    #[cfg(test)]
    pub(crate) fn from_oasr(path: &Path) -> Result<Self, WeightsError> {
        let verified_pack = PackVerifier
            .verify_candidate(PackCandidate::new(path))
            .map_err(|error| WeightsError::Gguf(error.to_string()))?;
        ensure_diarization_pack_route(&verified_pack)?;
        Self::from_preflight(verified_pack.preflight())
    }

    pub(crate) fn from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    ) -> Result<Self, WeightsError> {
        let reader = crate::ggml_runtime::build_runtime_tensor_reader_from_preflight(preflight)
            .map_err(|e| WeightsError::Gguf(e.to_string()))?;
        let mut tensors = BTreeMap::new();
        for metadata in reader.tensor_index().tensors() {
            let shape: Vec<usize> = metadata
                .dims
                .iter()
                .map(|&dim| dim as usize)
                .collect::<Vec<_>>();
            let data = reader
                .host_tensor_f32_copy_dequantized_by_name(&metadata.name, &metadata.dims)
                .map_err(|e| WeightsError::Gguf(e.to_string()))?;
            tensors.insert(metadata.name.clone(), Tensor { shape, data });
        }
        Ok(Self { tensors })
    }

    /// Capacity-derived commitment upper bound for every retained heap owner.
    /// Each independently allocated payload is page-rounded with allocator
    /// header room; a full page per logical tensor conservatively covers the
    /// private BTree node layout without depending on std internals.
    pub(crate) fn persistent_host_commitment_bytes(&self) -> Result<u64, WeightsError> {
        let mut bytes = allocation_commitment(std::mem::size_of::<Self>())?;
        for (name, tensor) in &self.tensors {
            let shape_bytes = tensor
                .shape
                .capacity()
                .checked_mul(std::mem::size_of::<usize>())
                .ok_or_else(|| {
                    WeightsError::InvalidInput("redimnet shape capacity byte overflow".to_string())
                })?;
            let data_bytes = tensor
                .data
                .capacity()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| {
                    WeightsError::InvalidInput("redimnet tensor capacity byte overflow".to_string())
                })?;
            for commitment in [
                allocation_commitment(name.capacity())?,
                allocation_commitment(shape_bytes)?,
                allocation_commitment(data_bytes)?,
                HOST_ALLOCATION_PAGE_BYTES,
            ] {
                bytes = bytes.checked_add(commitment).ok_or_else(|| {
                    WeightsError::InvalidInput(
                        "redimnet retained weight byte sum overflow".to_string(),
                    )
                })?;
            }
        }
        Ok(bytes)
    }

    pub(crate) fn get(&self, name: &str) -> Result<&[f32], WeightsError> {
        self.tensors
            .get(name)
            .map(|t| t.data.as_slice())
            .ok_or_else(|| WeightsError::Missing(name.to_string()))
    }

    pub(crate) fn shape(&self, name: &str) -> Result<&[usize], WeightsError> {
        self.tensors
            .get(name)
            .map(|t| t.shape.as_slice())
            .ok_or_else(|| WeightsError::Missing(name.to_string()))
    }
}

#[cfg(test)]
fn ensure_diarization_pack_route(
    verified_pack: &crate::models::pack_verifier::VerifiedPack,
) -> Result<(), WeightsError> {
    if matches!(
        verified_pack.route(),
        PackRoute::Aux {
            kind: AuxPackKind::Diarization,
            ..
        }
    ) {
        return Ok(());
    }
    Err(WeightsError::Gguf(format!(
        "ReDimNet pack route is not auxiliary diarization: {:?}",
        verified_pack.route()
    )))
}

#[cfg(test)]
fn quote_safetensors_header(
    header: &crate::models::local_source_import::SafetensorsHeader,
) -> Result<u64, WeightsError> {
    let mut bytes = allocation_commitment(std::mem::size_of::<Weights>())?;
    for tensor in &header.tensors {
        if tensor.dtype != "F32" {
            return Err(WeightsError::Dtype {
                name: tensor.name.clone(),
                dtype: tensor.dtype.clone(),
            });
        }
        let data_bytes = tensor.data_offsets[1]
            .checked_sub(tensor.data_offsets[0])
            .ok_or_else(|| WeightsError::Bounds {
                name: tensor.name.clone(),
            })?;
        let shape_bytes = u64::try_from(tensor.shape.len())
            .unwrap_or(u64::MAX)
            .checked_mul(std::mem::size_of::<usize>() as u64)
            .ok_or_else(|| {
                WeightsError::InvalidInput(format!(
                    "tensor '{}' shape byte quote overflow",
                    tensor.name
                ))
            })?;
        for commitment in [
            allocation_commitment_u64(tensor.name.len() as u64)?,
            allocation_commitment_u64(shape_bytes)?,
            allocation_commitment_u64(data_bytes)?,
            HOST_ALLOCATION_PAGE_BYTES,
        ] {
            bytes = bytes.checked_add(commitment).ok_or_else(|| {
                WeightsError::InvalidInput(
                    "safetensors retained weight byte quote overflow".to_string(),
                )
            })?;
        }
    }
    Ok(bytes)
}

pub(crate) const HOST_ALLOCATION_PAGE_BYTES: u64 = 4096;

pub(crate) fn allocation_commitment(requested_bytes: usize) -> Result<u64, WeightsError> {
    let requested = u64::try_from(requested_bytes).map_err(|_| {
        WeightsError::InvalidInput("redimnet allocation size does not fit u64".to_string())
    })?;
    allocation_commitment_u64(requested)
}

pub(crate) fn allocation_commitment_u64(requested: u64) -> Result<u64, WeightsError> {
    let with_header = requested
        .checked_add((std::mem::size_of::<usize>() * 2) as u64)
        .ok_or_else(|| {
            WeightsError::InvalidInput("redimnet allocation header byte overflow".to_string())
        })?;
    let remainder = with_header % HOST_ALLOCATION_PAGE_BYTES;
    if remainder == 0 {
        Ok(with_header)
    } else {
        with_header
            .checked_add(HOST_ALLOCATION_PAGE_BYTES - remainder)
            .ok_or_else(|| {
                WeightsError::InvalidInput("redimnet allocation rounding overflow".to_string())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_f32_safetensors() -> Vec<u8> {
        let header = br#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"long_tensor_name":{"dtype":"F32","shape":[1,2],"data_offsets":[8,16]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        for value in [1.0_f32, 2.0, 3.0, 4.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn raw_safetensors_quote_bounds_measured_retained_capacity() {
        let bytes = raw_f32_safetensors();
        let quote = Weights::quoted_safetensors_materialization(&bytes).expect("quote");
        let weights = Weights::from_safetensors(&bytes).expect("materialize");
        let measured = weights
            .persistent_host_commitment_bytes()
            .expect("measure retained");
        assert!(quote.retained_bytes >= measured);
        assert!(quote.parser_peak_bytes > 0);
    }

    #[test]
    fn raw_safetensors_loader_reuses_shared_duplicate_key_hardening() {
        let header = br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        let error = match Weights::from_safetensors(&bytes) {
            Ok(_) => panic!("duplicate key must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("duplicate JSON object key"));
    }
}
