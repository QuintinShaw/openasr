//! Vendored FireRedVAD (`FireRedTeam/FireRedVAD`, Apache-2.0) DFSMN weights
//! and CMVN stats, plus a minimal safetensors loader.
//!
//! The tensors are the upstream `VAD/model.pth.tar` checkpoint's
//! `model_state_dict` (a `DetectModel` -- `dfsmn.*` + `out.*`) and
//! `VAD/cmvn.ark` global CMVN stats, re-serialized into a small safetensors
//! file with flattened, stable names (mirrors
//! `crate::diarize::vad::weights`'s Silero vendoring). Parsing reuses the
//! crate's existing `serde_json` for the header; tensor data is
//! little-endian `f32`. A trailing `hparams` `I32` tensor records the
//! upstream `DetectModel` hyperparameters so a mismatched export fails
//! loudly instead of silently producing wrong numbers.

use std::collections::BTreeMap;

use serde::Deserialize;
use thiserror::Error;

use super::frontend::NUM_MEL_BINS;
use super::model::{HIDDEN, LOOKAHEAD_ORDER, LOOKBACK_ORDER, NUM_BLOCKS, PROJ};

/// Vendored weights blob (safetensors). ~2.3 MB; Apache-2.0 upstream model.
const WEIGHTS_BYTES: &[u8] = include_bytes!("../assets/firered_vad_16k.safetensors");

#[derive(Debug, Error)]
pub enum FireRedVadWeightsError {
    #[error("firered VAD weights blob is truncated (len {len}, need at least {need})")]
    Truncated { len: usize, need: usize },
    #[error("firered VAD weights header is not valid JSON: {0}")]
    Header(String),
    #[error("firered VAD weights are missing tensor '{0}'")]
    MissingTensor(String),
    #[error("firered VAD tensor '{name}' has unexpected dtype '{dtype}' (only F32/I32 supported)")]
    Dtype { name: String, dtype: String },
    #[error("firered VAD tensor '{name}' has {got} elements, expected {want}")]
    Len {
        name: String,
        got: usize,
        want: usize,
    },
    #[error("firered VAD tensor '{name}' data range {range:?} is out of bounds")]
    Bounds { name: String, range: [usize; 2] },
    #[error(
        "firered VAD checkpoint hyperparameters {got:?} do not match the hand-written forward \
         pass's compiled-in constants {want:?}"
    )]
    HparamMismatch { got: Vec<i32>, want: Vec<i32> },
}

#[derive(Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// One DFSMN block's parameters (the `R - 1` repeated blocks after `fsmn1`).
pub(crate) struct BlockWeights {
    pub fc1_w: Vec<f32>,     // [HIDDEN, PROJ]
    pub fc1_b: Vec<f32>,     // [HIDDEN]
    pub fc2_w: Vec<f32>,     // [PROJ, HIDDEN], no bias
    pub lookback: Vec<f32>,  // [PROJ, LOOKBACK_ORDER]
    pub lookahead: Vec<f32>, // [PROJ, LOOKAHEAD_ORDER]
}

/// All FireRedVAD tensors held as flat row-major `f32`, ready for the
/// forward pass. Linear weights are PyTorch-layout `[out, in]`; FSMN filters
/// are depthwise `[channels, kernel]` (the singular `in_channels=1` conv1d
/// dim is dropped).
pub(crate) struct FireRedVadWeights {
    pub fc1_w: Vec<f32>,           // [HIDDEN, NUM_MEL_BINS]
    pub fc1_b: Vec<f32>,           // [HIDDEN]
    pub fc2_w: Vec<f32>,           // [PROJ, HIDDEN]
    pub fc2_b: Vec<f32>,           // [PROJ]
    pub fsmn1_lookback: Vec<f32>,  // [PROJ, LOOKBACK_ORDER]
    pub fsmn1_lookahead: Vec<f32>, // [PROJ, LOOKAHEAD_ORDER]
    pub blocks: Vec<BlockWeights>, // len NUM_BLOCKS
    pub dnn_w: Vec<f32>,           // [HIDDEN, PROJ]
    pub dnn_b: Vec<f32>,           // [HIDDEN]
    pub out_w: Vec<f32>,           // [1, HIDDEN]
    pub out_b: f32,
    pub cmvn_mean: [f32; NUM_MEL_BINS],
    pub cmvn_inv_stddev: [f32; NUM_MEL_BINS],
}

impl FireRedVadWeights {
    /// Load the vendored, validated weights. Infallible in practice (the
    /// blob is committed), but returns a typed error rather than panicking
    /// so callers can decline to register the engine.
    pub(crate) fn embedded() -> Result<Self, FireRedVadWeightsError> {
        Self::parse(WEIGHTS_BYTES)
    }

    fn parse(bytes: &[u8]) -> Result<Self, FireRedVadWeightsError> {
        if bytes.len() < 8 {
            return Err(FireRedVadWeightsError::Truncated {
                len: bytes.len(),
                need: 8,
            });
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")) as usize;
        let header_end = 8usize
            .checked_add(header_len)
            .filter(|end| *end <= bytes.len())
            .ok_or(FireRedVadWeightsError::Truncated {
                len: bytes.len(),
                need: 8 + header_len,
            })?;
        let header: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..header_end])
                .map_err(|error| FireRedVadWeightsError::Header(error.to_string()))?;
        let data = &bytes[header_end..];

        let load_named =
            |name: String, want_len: usize| -> Result<Vec<f32>, FireRedVadWeightsError> {
                let value = header
                    .get(&name)
                    .ok_or_else(|| FireRedVadWeightsError::MissingTensor(name.clone()))?;
                let info: TensorInfo = TensorInfo::deserialize(value)
                    .map_err(|error| FireRedVadWeightsError::Header(error.to_string()))?;
                if info.dtype != "F32" {
                    return Err(FireRedVadWeightsError::Dtype {
                        name,
                        dtype: info.dtype,
                    });
                }
                let got_len: usize = info.shape.iter().product();
                if got_len != want_len {
                    return Err(FireRedVadWeightsError::Len {
                        name,
                        got: got_len,
                        want: want_len,
                    });
                }
                let [start, end] = info.data_offsets;
                if end < start || end > data.len() {
                    return Err(FireRedVadWeightsError::Bounds {
                        name,
                        range: [start, end],
                    });
                }
                Ok(read_f32_le(&data[start..end]))
            };
        let load = |name: &str, want_len: usize| load_named(name.to_string(), want_len);

        // Hyperparameter guard: the hand-written forward pass hard-codes
        // R/M/H/P/N1/S1/N2/S2/idim/odim as Rust constants, so assert the
        // vendored checkpoint still matches before trusting the tensor shapes.
        {
            let value = header
                .get("hparams")
                .ok_or_else(|| FireRedVadWeightsError::MissingTensor("hparams".to_string()))?;
            let info: TensorInfo = TensorInfo::deserialize(value)
                .map_err(|error| FireRedVadWeightsError::Header(error.to_string()))?;
            if info.dtype != "I32" {
                return Err(FireRedVadWeightsError::Dtype {
                    name: "hparams".to_string(),
                    dtype: info.dtype,
                });
            }
            let [start, end] = info.data_offsets;
            if end < start || end > data.len() {
                return Err(FireRedVadWeightsError::Bounds {
                    name: "hparams".to_string(),
                    range: [start, end],
                });
            }
            let got: Vec<i32> = data[start..end]
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let want: Vec<i32> = vec![
                (NUM_BLOCKS + 1) as i32,
                1,
                HIDDEN as i32,
                PROJ as i32,
                LOOKBACK_ORDER as i32,
                1,
                LOOKAHEAD_ORDER as i32,
                1,
                NUM_MEL_BINS as i32,
                1,
            ];
            if got != want {
                return Err(FireRedVadWeightsError::HparamMismatch { got, want });
            }
        }

        let mut blocks = Vec::with_capacity(NUM_BLOCKS);
        for i in 0..NUM_BLOCKS {
            blocks.push(BlockWeights {
                fc1_w: load(&format!("dfsmn.block{i}.fc1.weight"), HIDDEN * PROJ)?,
                fc1_b: load(&format!("dfsmn.block{i}.fc1.bias"), HIDDEN)?,
                fc2_w: load(&format!("dfsmn.block{i}.fc2.weight"), PROJ * HIDDEN)?,
                lookback: load(&format!("dfsmn.block{i}.lookback"), PROJ * LOOKBACK_ORDER)?,
                lookahead: load(&format!("dfsmn.block{i}.lookahead"), PROJ * LOOKAHEAD_ORDER)?,
            });
        }

        let out_w = load("out.weight", HIDDEN)?;
        let out_b = load("out.bias", 1)?[0];
        let cmvn_mean_vec = load("frontend.cmvn.mean", NUM_MEL_BINS)?;
        let cmvn_istd_vec = load("frontend.cmvn.inv_stddev", NUM_MEL_BINS)?;
        let mut cmvn_mean = [0.0f32; NUM_MEL_BINS];
        let mut cmvn_inv_stddev = [0.0f32; NUM_MEL_BINS];
        cmvn_mean.copy_from_slice(&cmvn_mean_vec);
        cmvn_inv_stddev.copy_from_slice(&cmvn_istd_vec);

        Ok(Self {
            fc1_w: load("dfsmn.fc1.weight", HIDDEN * NUM_MEL_BINS)?,
            fc1_b: load("dfsmn.fc1.bias", HIDDEN)?,
            fc2_w: load("dfsmn.fc2.weight", PROJ * HIDDEN)?,
            fc2_b: load("dfsmn.fc2.bias", PROJ)?,
            fsmn1_lookback: load("dfsmn.fsmn1.lookback", PROJ * LOOKBACK_ORDER)?,
            fsmn1_lookahead: load("dfsmn.fsmn1.lookahead", PROJ * LOOKAHEAD_ORDER)?,
            blocks,
            dnn_w: load("dfsmn.dnn.weight", HIDDEN * PROJ)?,
            dnn_b: load("dfsmn.dnn.bias", HIDDEN)?,
            out_w,
            out_b,
            cmvn_mean,
            cmvn_inv_stddev,
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
        let w = FireRedVadWeights::embedded().expect("vendored firered VAD weights parse");
        assert_eq!(w.fc1_w.len(), HIDDEN * NUM_MEL_BINS);
        assert_eq!(w.blocks.len(), NUM_BLOCKS);
        assert_eq!(w.out_w.len(), HIDDEN);
        assert!(w.out_b.is_finite());
        assert!(w.cmvn_mean.iter().all(|v| v.is_finite()));
        assert!(w.cmvn_inv_stddev.iter().all(|v| v.is_finite()));
    }
}
