//! Loader for speaker-embedder weight packs.
//!
//! Unlike the tiny vendored Stream-VAD model, speaker embedders are delivered as
//! pulled `.oasr` packs, so weights are read from a file path at runtime — never
//! `include_bytes!`. Raw safetensors remain supported as a dev fast path. `.oasr`
//! packs are materialized into logical f32 buffers for the pure-Rust forward
//! passes.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;
use thiserror::Error;

use crate::ggml_runtime::{GgmlRuntimeSource, GgufTensorDataReader};

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

#[derive(Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// A name-keyed bag of `f32` tensors loaded from a safetensors file.
pub(crate) struct Weights {
    tensors: BTreeMap<String, Tensor>,
}

impl Weights {
    /// Logical f32 resident size from the already-open GGUF tensor index.
    /// This performs only header/index/range validation and never copies or
    /// dequantizes tensor payloads, so admission can run before materializing
    /// the ReDimNet model while retaining the exact same mapped source.
    pub(crate) fn logical_f32_bytes_from_runtime_source(
        runtime_source: &GgmlRuntimeSource,
    ) -> Result<u64, WeightsError> {
        let reader = GgufTensorDataReader::from_runtime_source(runtime_source)
            .map_err(|error| WeightsError::Gguf(error.to_string()))?;
        if reader.tensor_index().tensors().is_empty() {
            return Err(WeightsError::InvalidInput(
                "speaker-embedder pack contains no tensors".to_string(),
            ));
        }
        reader.tensor_index().tensors().iter().enumerate().try_fold(
            0_u64,
            |total, (tensor_id, tensor)| {
                // Validate that this tensor index entry addresses bytes inside
                // the held mmap without copying those bytes.
                reader
                    .host_tensor_bytes_by_id(tensor_id)
                    .map_err(|error| WeightsError::Gguf(error.to_string()))?;
                let elements = tensor.num_elements().ok_or_else(|| {
                    WeightsError::InvalidInput(format!(
                        "tensor '{}' logical element count overflows",
                        tensor.name
                    ))
                })?;
                total
                    .checked_add(elements.checked_mul(4).ok_or_else(|| {
                        WeightsError::InvalidInput(format!(
                            "tensor '{}' logical f32 byte count overflows",
                            tensor.name
                        ))
                    })?)
                    .ok_or_else(|| {
                        WeightsError::InvalidInput(
                            "speaker-embedder logical f32 byte total overflows".to_string(),
                        )
                    })
            },
        )
    }

    pub(crate) fn logical_f32_bytes(&self) -> u64 {
        self.tensors.values().fold(0u64, |total, tensor| {
            total.saturating_add(
                u64::try_from(tensor.data.len())
                    .unwrap_or(u64::MAX)
                    .saturating_mul(std::mem::size_of::<f32>() as u64),
            )
        })
    }

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

    /// Parse a safetensors byte buffer.
    pub(crate) fn from_safetensors(bytes: &[u8]) -> Result<Self, WeightsError> {
        if bytes.len() < 8 {
            return Err(WeightsError::Truncated {
                len: bytes.len(),
                need: 8,
            });
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")) as usize;
        let header_end = 8usize
            .checked_add(header_len)
            .filter(|end| *end <= bytes.len())
            .ok_or(WeightsError::Truncated {
                len: bytes.len(),
                need: 8 + header_len,
            })?;
        let header: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..header_end])
                .map_err(|e| WeightsError::Header(e.to_string()))?;
        let data = &bytes[header_end..];

        let mut tensors = BTreeMap::new();
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let info: TensorInfo =
                TensorInfo::deserialize(value).map_err(|e| WeightsError::Header(e.to_string()))?;
            if info.dtype != "F32" {
                return Err(WeightsError::Dtype {
                    name,
                    dtype: info.dtype,
                });
            }
            let [start, end] = info.data_offsets;
            if end < start || end > data.len() || (end - start) % 4 != 0 {
                return Err(WeightsError::Bounds { name });
            }
            let want: usize = info.shape.iter().product();
            let got = (end - start) / 4;
            if got != want {
                return Err(WeightsError::SizeMismatch {
                    name,
                    got,
                    want,
                    shape: info.shape,
                });
            }
            let floats = data[start..end]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            tensors.insert(
                name,
                Tensor {
                    shape: info.shape,
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
    pub(crate) fn from_oasr(path: &Path) -> Result<Self, WeightsError> {
        // Explicit validate + `from_runtime_source` (equivalent to
        // `from_path`, which does exactly this internally) so the open
        // mapping identity/bytes contract is visible at this call site too;
        // this loader is the sole opener of this pack (no earlier admission
        // step to reuse), so there is no reopen race here either way.
        let runtime_source = crate::ggml_runtime::validate_ggml_runtime_source_path(path)
            .map_err(|e| WeightsError::Gguf(e.to_string()))?;
        Self::from_runtime_source(&runtime_source)
    }

    /// Parse weights from the same already-open mapping whose content id keys
    /// the resident runtime. This prevents a path replacement between cache-key
    /// resolution and weight loading from binding different bytes to that key.
    pub(crate) fn from_runtime_source(
        runtime_source: &GgmlRuntimeSource,
    ) -> Result<Self, WeightsError> {
        let reader = GgufTensorDataReader::from_runtime_source(runtime_source)
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
