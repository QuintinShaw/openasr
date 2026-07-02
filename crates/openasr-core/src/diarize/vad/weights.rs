//! Vendored Silero VAD v6.2 (16 kHz) weights and a minimal safetensors loader.
//!
//! The weights are the upstream `silero_vad_16k_op15.onnx` deployment
//! (`reparam_conv`) tensors from snakers4/silero-vad v6.2.1 (MIT), re-serialized
//! into a small safetensors file with stable names. Parsing uses the crate's
//! existing `serde_json` for the header; tensor data is little-endian `f32`.

use std::collections::BTreeMap;

use serde::Deserialize;
use thiserror::Error;

/// Vendored weights blob (safetensors). ~1.2 MB; MIT-licensed upstream model.
const WEIGHTS_BYTES: &[u8] = include_bytes!("assets/silero_vad_v6_16k.safetensors");

/// Learned-STFT filters: `[258, 1, 256]` flattened to `[258, 256]` (the singular
/// middle dim is dropped); first 129 rows are the real basis, next 129 imaginary.
pub(crate) const STFT_FILTERS: usize = 258;
pub(crate) const STFT_KERNEL: usize = 256;
/// Frequency bins fed to the encoder (`258 / 2`).
pub(crate) const FREQ_BINS: usize = 129;
/// LSTM hidden width and the encoder's final channel count.
pub(crate) const HIDDEN: usize = 128;

#[derive(Debug, Error)]
pub enum SileroWeightsError {
    #[error("silero weights blob is truncated (len {len}, need at least {need})")]
    Truncated { len: usize, need: usize },
    #[error("silero weights header is not valid JSON: {0}")]
    Header(String),
    #[error("silero weights are missing tensor '{0}'")]
    MissingTensor(&'static str),
    #[error("silero tensor '{name}' has unexpected dtype '{dtype}' (only F32 is supported)")]
    Dtype { name: &'static str, dtype: String },
    #[error("silero tensor '{name}' has shape {got:?}, expected {want:?}")]
    Shape {
        name: &'static str,
        got: Vec<usize>,
        want: Vec<usize>,
    },
    #[error("silero tensor '{name}' data range {0:?} is out of bounds", .range)]
    Bounds {
        name: &'static str,
        range: [usize; 2],
    },
}

#[derive(Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// All 15 Silero tensors held as flat row-major `f32`, ready for the forward
/// pass. Convolution weights are `[out, in, k]`, LSTM weights `[4*hidden, in]`.
pub(crate) struct SileroWeights {
    pub stft_basis: Vec<f32>, // [258, 256]
    pub conv1_w: Vec<f32>,    // [128, 129, 3]
    pub conv1_b: Vec<f32>,    // [128]
    pub conv2_w: Vec<f32>,    // [64, 128, 3]
    pub conv2_b: Vec<f32>,    // [64]
    pub conv3_w: Vec<f32>,    // [64, 64, 3]
    pub conv3_b: Vec<f32>,    // [64]
    pub conv4_w: Vec<f32>,    // [128, 64, 3]
    pub conv4_b: Vec<f32>,    // [128]
    pub lstm_w_ih: Vec<f32>,  // [512, 128]
    pub lstm_w_hh: Vec<f32>,  // [512, 128]
    pub lstm_b_ih: Vec<f32>,  // [512]
    pub lstm_b_hh: Vec<f32>,  // [512]
    pub final_w: Vec<f32>,    // [128]
    pub final_b: f32,
}

impl SileroWeights {
    /// Load the vendored, validated weights. Infallible in practice (the blob is
    /// committed), but returns a typed error rather than panicking so callers
    /// can fall back to the energy gate.
    pub(crate) fn embedded() -> Result<Self, SileroWeightsError> {
        Self::parse(WEIGHTS_BYTES)
    }

    fn parse(bytes: &[u8]) -> Result<Self, SileroWeightsError> {
        if bytes.len() < 8 {
            return Err(SileroWeightsError::Truncated {
                len: bytes.len(),
                need: 8,
            });
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")) as usize;
        let header_end = 8usize
            .checked_add(header_len)
            .filter(|end| *end <= bytes.len())
            .ok_or(SileroWeightsError::Truncated {
                len: bytes.len(),
                need: 8 + header_len,
            })?;
        let header: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..header_end])
                .map_err(|error| SileroWeightsError::Header(error.to_string()))?;
        let data = &bytes[header_end..];

        let load = |name: &'static str, want: &[usize]| -> Result<Vec<f32>, SileroWeightsError> {
            let value = header
                .get(name)
                .ok_or(SileroWeightsError::MissingTensor(name))?;
            let info: TensorInfo = TensorInfo::deserialize(value)
                .map_err(|error| SileroWeightsError::Header(error.to_string()))?;
            if info.dtype != "F32" {
                return Err(SileroWeightsError::Dtype {
                    name,
                    dtype: info.dtype,
                });
            }
            if info.shape != want {
                return Err(SileroWeightsError::Shape {
                    name,
                    got: info.shape,
                    want: want.to_vec(),
                });
            }
            let [start, end] = info.data_offsets;
            if end < start || end > data.len() {
                return Err(SileroWeightsError::Bounds {
                    name,
                    range: [start, end],
                });
            }
            Ok(read_f32_le(&data[start..end]))
        };

        let final_w = load("final_conv.weight", &[1, HIDDEN, 1])?;
        let final_b = load("final_conv.bias", &[1])?[0];
        Ok(Self {
            stft_basis: load("stft_conv.weight", &[STFT_FILTERS, 1, STFT_KERNEL])?,
            conv1_w: load("conv1.weight", &[HIDDEN, FREQ_BINS, 3])?,
            conv1_b: load("conv1.bias", &[HIDDEN])?,
            conv2_w: load("conv2.weight", &[64, HIDDEN, 3])?,
            conv2_b: load("conv2.bias", &[64])?,
            conv3_w: load("conv3.weight", &[64, 64, 3])?,
            conv3_b: load("conv3.bias", &[64])?,
            conv4_w: load("conv4.weight", &[HIDDEN, 64, 3])?,
            conv4_b: load("conv4.bias", &[HIDDEN])?,
            lstm_w_ih: load("lstm_cell.weight_ih", &[4 * HIDDEN, HIDDEN])?,
            lstm_w_hh: load("lstm_cell.weight_hh", &[4 * HIDDEN, HIDDEN])?,
            lstm_b_ih: load("lstm_cell.bias_ih", &[4 * HIDDEN])?,
            lstm_b_hh: load("lstm_cell.bias_hh", &[4 * HIDDEN])?,
            final_w,
            final_b,
        })
    }
}

fn read_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod weights_tests {
    use super::*;

    #[test]
    fn embedded_weights_parse_with_expected_shapes() {
        let w = SileroWeights::embedded().expect("vendored weights parse");
        assert_eq!(w.stft_basis.len(), STFT_FILTERS * STFT_KERNEL);
        assert_eq!(w.conv1_w.len(), HIDDEN * FREQ_BINS * 3);
        assert_eq!(w.lstm_w_ih.len(), 4 * HIDDEN * HIDDEN);
        assert_eq!(w.final_w.len(), HIDDEN);
        assert!(w.final_b.is_finite());
    }
}
