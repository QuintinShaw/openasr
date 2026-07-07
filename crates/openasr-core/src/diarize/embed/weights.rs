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

use crate::ggml_runtime::GgufTensorDataReader;

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
        let reader =
            GgufTensorDataReader::from_path(path).map_err(|e| WeightsError::Gguf(e.to_string()))?;
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

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}
