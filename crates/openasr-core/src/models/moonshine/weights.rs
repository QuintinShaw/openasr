use thiserror::Error;

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};

use super::runtime_contract::MoonshineExecutionMetadata;

/// A weight tensor dequantized to row-major f32 with its GGUF dims (ne0-first).
#[derive(Debug, Clone)]
pub(crate) struct MoonshineWeight {
    pub name: String,
    pub dims: Vec<usize>,
    pub values: Vec<f32>,
}

impl MoonshineWeight {
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MoonshineEncoderLayerWeights {
    pub attn_norm: MoonshineWeight,
    pub attn_q: MoonshineWeight,
    pub attn_k: MoonshineWeight,
    pub attn_v: MoonshineWeight,
    pub attn_o: MoonshineWeight,
    pub ffn_norm: MoonshineWeight,
    pub ffn_up: MoonshineWeight,
    pub ffn_up_bias: MoonshineWeight,
    pub ffn_down: MoonshineWeight,
    pub ffn_down_bias: MoonshineWeight,
}

#[derive(Debug, Clone)]
pub(crate) struct MoonshineDecoderLayerWeights {
    pub attn_norm: MoonshineWeight,
    pub attn_q: MoonshineWeight,
    pub attn_k: MoonshineWeight,
    pub attn_v: MoonshineWeight,
    pub attn_o: MoonshineWeight,
    pub cross_norm: MoonshineWeight,
    pub cross_q: MoonshineWeight,
    pub cross_k: MoonshineWeight,
    pub cross_v: MoonshineWeight,
    pub cross_o: MoonshineWeight,
    pub ffn_norm: MoonshineWeight,
    pub ffn_up: MoonshineWeight,
    pub ffn_up_bias: MoonshineWeight,
    pub ffn_down: MoonshineWeight,
    pub ffn_down_bias: MoonshineWeight,
}

#[derive(Debug, Clone)]
pub(crate) struct MoonshineEncoderWeights {
    pub conv1_weight: MoonshineWeight,
    pub conv2_weight: MoonshineWeight,
    pub conv2_bias: MoonshineWeight,
    pub conv3_weight: MoonshineWeight,
    pub conv3_bias: MoonshineWeight,
    pub groupnorm_weight: MoonshineWeight,
    pub groupnorm_bias: MoonshineWeight,
    pub out_norm: MoonshineWeight,
    pub layers: Vec<MoonshineEncoderLayerWeights>,
}

#[derive(Debug, Clone)]
pub(crate) struct MoonshineDecoderWeights {
    pub embedding: MoonshineWeight,
    pub out_norm: MoonshineWeight,
    pub layers: Vec<MoonshineDecoderLayerWeights>,
}

#[derive(Debug, Error)]
pub(crate) enum MoonshineWeightLoadError {
    #[error("moonshine tensor read failed: {reason}")]
    TensorReadFailed { reason: String },
    #[error("moonshine tensor '{name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        name: String,
        shape: String,
        reason: String,
    },
}

pub(crate) fn load_moonshine_encoder_weights(
    reader: &GgufTensorDataReader,
    metadata: MoonshineExecutionMetadata,
) -> Result<MoonshineEncoderWeights, MoonshineWeightLoadError> {
    let conv1_weight = load_any(reader, "enc.conv1.weight")?;
    let conv2_weight = load_any(reader, "enc.conv2.weight")?;
    let conv2_bias = load_any(reader, "enc.conv2.bias")?;
    let conv3_weight = load_any(reader, "enc.conv3.weight")?;
    let conv3_bias = load_vector(reader, "enc.conv3.bias", metadata.d_model)?;
    let groupnorm_weight = load_vector(reader, "enc.groupnorm.weight", metadata.d_model)?;
    let groupnorm_bias = load_vector(reader, "enc.groupnorm.bias", metadata.d_model)?;
    let out_norm = load_vector(reader, "enc.out_norm.weight", metadata.d_model)?;

    let mut layers = Vec::with_capacity(metadata.encoder_layers);
    for layer_idx in 0..metadata.encoder_layers {
        let prefix = format!("enc.blk.{layer_idx}.");
        layers.push(MoonshineEncoderLayerWeights {
            attn_norm: load_vector(
                reader,
                &format!("{prefix}attn_norm.weight"),
                metadata.d_model,
            )?,
            attn_q: load_matrix_meta_only(
                reader,
                &format!("{prefix}attn_q.weight"),
                metadata.d_model,
                metadata.d_model,
            )?,
            attn_k: load_matrix_meta_only(
                reader,
                &format!("{prefix}attn_k.weight"),
                metadata.d_model,
                metadata.d_model,
            )?,
            attn_v: load_matrix_meta_only(
                reader,
                &format!("{prefix}attn_v.weight"),
                metadata.d_model,
                metadata.d_model,
            )?,
            attn_o: load_matrix_meta_only(
                reader,
                &format!("{prefix}attn_o.weight"),
                metadata.d_model,
                metadata.d_model,
            )?,
            ffn_norm: load_vector(
                reader,
                &format!("{prefix}ffn_norm.weight"),
                metadata.d_model,
            )?,
            ffn_up: load_matrix_meta_only(
                reader,
                &format!("{prefix}ffn_up.weight"),
                metadata.d_model,
                metadata.encoder_ffn_dim,
            )?,
            ffn_up_bias: load_vector(
                reader,
                &format!("{prefix}ffn_up.bias"),
                metadata.encoder_ffn_dim,
            )?,
            ffn_down: load_matrix_meta_only(
                reader,
                &format!("{prefix}ffn_down.weight"),
                metadata.encoder_ffn_dim,
                metadata.d_model,
            )?,
            ffn_down_bias: load_vector(
                reader,
                &format!("{prefix}ffn_down.bias"),
                metadata.d_model,
            )?,
        });
    }

    Ok(MoonshineEncoderWeights {
        conv1_weight,
        conv2_weight,
        conv2_bias,
        conv3_weight,
        conv3_bias,
        groupnorm_weight,
        groupnorm_bias,
        out_norm,
        layers,
    })
}

pub(crate) fn load_moonshine_decoder_weights(
    reader: &GgufTensorDataReader,
    metadata: MoonshineExecutionMetadata,
) -> Result<MoonshineDecoderWeights, MoonshineWeightLoadError> {
    let embedding = load_matrix(
        reader,
        "dec.emb.weight",
        metadata.d_model,
        metadata.vocab_size,
    )?;
    let out_norm = load_vector(reader, "dec.out_norm.weight", metadata.d_model)?;
    let fc1_width = metadata.decoder_ffn_dim.saturating_mul(2);

    let mut layers = Vec::with_capacity(metadata.decoder_layers);
    for layer_idx in 0..metadata.decoder_layers {
        let prefix = format!("dec.blk.{layer_idx}.");
        layers.push(MoonshineDecoderLayerWeights {
            attn_norm: load_vector(
                reader,
                &format!("{prefix}attn_norm.weight"),
                metadata.d_model,
            )?,
            attn_q: load_matrix_meta_only(
                reader,
                &format!("{prefix}attn_q.weight"),
                metadata.d_model,
                metadata.d_model,
            )?,
            attn_k: load_matrix_meta_only(
                reader,
                &format!("{prefix}attn_k.weight"),
                metadata.d_model,
                metadata.d_model,
            )?,
            attn_v: load_matrix_meta_only(
                reader,
                &format!("{prefix}attn_v.weight"),
                metadata.d_model,
                metadata.d_model,
            )?,
            attn_o: load_matrix_meta_only(
                reader,
                &format!("{prefix}attn_o.weight"),
                metadata.d_model,
                metadata.d_model,
            )?,
            cross_norm: load_vector(
                reader,
                &format!("{prefix}cross_norm.weight"),
                metadata.d_model,
            )?,
            cross_q: load_matrix_meta_only(
                reader,
                &format!("{prefix}cross_q.weight"),
                metadata.d_model,
                metadata.d_model,
            )?,
            cross_k: load_matrix_meta_only(
                reader,
                &format!("{prefix}cross_k.weight"),
                metadata.d_model,
                metadata.d_model,
            )?,
            cross_v: load_matrix_meta_only(
                reader,
                &format!("{prefix}cross_v.weight"),
                metadata.d_model,
                metadata.d_model,
            )?,
            cross_o: load_matrix_meta_only(
                reader,
                &format!("{prefix}cross_o.weight"),
                metadata.d_model,
                metadata.d_model,
            )?,
            ffn_norm: load_vector(
                reader,
                &format!("{prefix}ffn_norm.weight"),
                metadata.d_model,
            )?,
            ffn_up: load_matrix_meta_only(
                reader,
                &format!("{prefix}ffn_up.weight"),
                metadata.d_model,
                fc1_width,
            )?,
            ffn_up_bias: load_vector(reader, &format!("{prefix}ffn_up.bias"), fc1_width)?,
            ffn_down: load_matrix_meta_only(
                reader,
                &format!("{prefix}ffn_down.weight"),
                metadata.decoder_ffn_dim,
                metadata.d_model,
            )?,
            ffn_down_bias: load_vector(
                reader,
                &format!("{prefix}ffn_down.bias"),
                metadata.d_model,
            )?,
        });
    }

    Ok(MoonshineDecoderWeights {
        embedding,
        out_norm,
        layers,
    })
}

fn load_any(
    reader: &GgufTensorDataReader,
    name: &str,
) -> Result<MoonshineWeight, MoonshineWeightLoadError> {
    let tensor = require_tensor(reader, name)?;
    let dims: Vec<usize> = tensor.dims.iter().map(|value| *value as usize).collect();
    let dims_u64 = tensor.dims.clone();
    let values = reader
        .host_tensor_f32_copy_dequantized_by_name(name, &dims_u64)
        .map_err(map_read_error)?;
    Ok(MoonshineWeight {
        name: name.to_string(),
        dims,
        values,
    })
}

/// Load a rank-2 matrix stored as ggml `[ne0, ne1]` (column-major over HF row-major bytes).
fn load_matrix(
    reader: &GgufTensorDataReader,
    name: &str,
    ne0: usize,
    ne1: usize,
) -> Result<MoonshineWeight, MoonshineWeightLoadError> {
    let tensor = require_tensor(reader, name)?;
    if tensor.dims.as_slice() != [ne0 as u64, ne1 as u64] {
        return Err(MoonshineWeightLoadError::InvalidTensorShape {
            name: name.to_string(),
            shape: render_shape(&tensor.dims),
            reason: format!("expected [{ne0}, {ne1}]"),
        });
    }
    let values = reader
        .host_tensor_f32_copy_dequantized_by_name(name, &[ne0 as u64, ne1 as u64])
        .map_err(map_read_error)?;
    Ok(MoonshineWeight {
        name: name.to_string(),
        dims: vec![ne0, ne1],
        values,
    })
}

/// Load a rank-2 matrix's NAME + DIMS only, WITHOUT dequantizing its bytes to a
/// resident f32 host Vec (`values` stays empty). Used for the per-layer 2-D
/// linears that the graph binds zero-copy from the mmap'd pack — never
/// materializing the f32 is what actually lowers peak RSS (a high-water mark).
fn load_matrix_meta_only(
    reader: &GgufTensorDataReader,
    name: &str,
    ne0: usize,
    ne1: usize,
) -> Result<MoonshineWeight, MoonshineWeightLoadError> {
    let tensor = require_tensor(reader, name)?;
    if tensor.dims.as_slice() != [ne0 as u64, ne1 as u64] {
        return Err(MoonshineWeightLoadError::InvalidTensorShape {
            name: name.to_string(),
            shape: render_shape(&tensor.dims),
            reason: format!("expected [{ne0}, {ne1}]"),
        });
    }
    Ok(MoonshineWeight {
        name: name.to_string(),
        dims: vec![ne0, ne1],
        values: Vec::new(),
    })
}

fn load_vector(
    reader: &GgufTensorDataReader,
    name: &str,
    expected_len: usize,
) -> Result<MoonshineWeight, MoonshineWeightLoadError> {
    let tensor = require_tensor(reader, name)?;
    if tensor.dims.as_slice() != [expected_len as u64] {
        return Err(MoonshineWeightLoadError::InvalidTensorShape {
            name: name.to_string(),
            shape: render_shape(&tensor.dims),
            reason: format!("expected [{expected_len}]"),
        });
    }
    let values = reader
        .host_tensor_f32_copy_dequantized_by_name(name, &[expected_len as u64])
        .map_err(map_read_error)?;
    Ok(MoonshineWeight {
        name: name.to_string(),
        dims: vec![expected_len],
        values,
    })
}

fn require_tensor<'a>(
    reader: &'a GgufTensorDataReader,
    name: &str,
) -> Result<&'a crate::GgufTensorMetadata, MoonshineWeightLoadError> {
    reader
        .tensor_index()
        .get(name)
        .ok_or_else(|| MoonshineWeightLoadError::InvalidTensorShape {
            name: name.to_string(),
            shape: "[]".to_string(),
            reason: "tensor is missing from GGUF tensor index".to_string(),
        })
}

fn map_read_error(error: GgufTensorDataReadError) -> MoonshineWeightLoadError {
    MoonshineWeightLoadError::TensorReadFailed {
        reason: error.to_string(),
    }
}

fn render_shape(shape: &[u64]) -> String {
    let parts = shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{parts}]")
}
