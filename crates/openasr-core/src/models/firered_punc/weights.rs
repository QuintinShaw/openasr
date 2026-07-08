//! Load a FireRedPunc `.oasr` pack into host f32 weights.
//!
//! FireRedPunc is BERT-base (~110M params) and runs as an occasional
//! finalize-only pass, so -- unlike the ASR encoders that keep the dominant
//! linears quantized and zero-copy -- every tensor is simply dequantized to f32
//! and uploaded to the graph arena. That keeps the graph and its synthetic-
//! weight numeric test straightforward; the conversion recipe may still store
//! the pack quantized (the loader dequantizes on read).

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};

use super::config::FireRedPuncExecutionMetadata;
use super::tensor_names::{
    EMBD_NORM_BIAS, EMBD_NORM_WEIGHT, POSITION_EMBD_WEIGHT, PUNC_HEAD_BIAS, PUNC_HEAD_WEIGHT,
    TOKEN_EMBD_WEIGHT, TOKEN_TYPE_EMBD_WEIGHT, firered_punc_layer_tensor_names,
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum FireRedPuncWeightsError {
    #[error("firered-punc weight read failed: {0}")]
    Read(#[from] GgufTensorDataReadError),
    #[error("firered-punc tensor '{name}' has {got} elements, expected {expected}")]
    ElementCount {
        name: String,
        got: usize,
        expected: usize,
    },
}

/// A host weight: stored dims (from the GGUF index) + dequantized f32 values.
#[derive(Debug, Clone)]
pub(crate) struct NamedTensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub values: Vec<f32>,
}

/// One BERT block's weights (`blk.{i}.*`). Linear weights are stored `[in, out]`
/// (ne0 = in_features) so the graph's `mul_mat(weight, x)` yields `[out, seq]`.
#[derive(Debug, Clone)]
pub(crate) struct FireRedPuncLayerWeights {
    pub attn_q_weight: NamedTensor,
    pub attn_q_bias: NamedTensor,
    pub attn_k_weight: NamedTensor,
    pub attn_k_bias: NamedTensor,
    pub attn_v_weight: NamedTensor,
    pub attn_v_bias: NamedTensor,
    pub attn_output_weight: NamedTensor,
    pub attn_output_bias: NamedTensor,
    pub attn_norm_weight: NamedTensor,
    pub attn_norm_bias: NamedTensor,
    pub ffn_up_weight: NamedTensor,
    pub ffn_up_bias: NamedTensor,
    pub ffn_down_weight: NamedTensor,
    pub ffn_down_bias: NamedTensor,
    pub ffn_norm_weight: NamedTensor,
    pub ffn_norm_bias: NamedTensor,
}

#[derive(Debug, Clone)]
pub(crate) struct FireRedPuncWeights {
    pub token_embd: NamedTensor,
    pub token_type_embd: NamedTensor,
    pub position_embd: NamedTensor,
    pub embd_norm_weight: NamedTensor,
    pub embd_norm_bias: NamedTensor,
    pub layers: Vec<FireRedPuncLayerWeights>,
    pub punc_head_weight: NamedTensor,
    pub punc_head_bias: NamedTensor,
}

fn load_named(
    reader: &GgufTensorDataReader,
    name: &str,
) -> Result<NamedTensor, FireRedPuncWeightsError> {
    let tensor = reader.tensor_index().get(name).ok_or_else(|| {
        FireRedPuncWeightsError::Read(GgufTensorDataReadError::TensorNotFound {
            path: reader.tensor_index().path().to_path_buf(),
            tensor_name: name.to_string(),
        })
    })?;
    let dims: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
    let shape_u64: Vec<u64> = tensor.dims.clone();
    let values = reader.host_tensor_f32_copy_dequantized_by_name(name, &shape_u64)?;
    Ok(NamedTensor {
        name: name.to_string(),
        dims,
        values,
    })
}

fn load_expected(
    reader: &GgufTensorDataReader,
    name: &str,
    expected: usize,
) -> Result<NamedTensor, FireRedPuncWeightsError> {
    let tensor = load_named(reader, name)?;
    if tensor.values.len() != expected {
        return Err(FireRedPuncWeightsError::ElementCount {
            name: name.to_string(),
            got: tensor.values.len(),
            expected,
        });
    }
    Ok(tensor)
}

fn load_layer(
    reader: &GgufTensorDataReader,
    layer: usize,
    metadata: &FireRedPuncExecutionMetadata,
) -> Result<FireRedPuncLayerWeights, FireRedPuncWeightsError> {
    let names = firered_punc_layer_tensor_names(layer);
    let d = metadata.d_model;
    let ffn = metadata.ffn_dim;
    Ok(FireRedPuncLayerWeights {
        attn_q_weight: load_expected(reader, &names.attn_q_weight, d * d)?,
        attn_q_bias: load_expected(reader, &names.attn_q_bias, d)?,
        attn_k_weight: load_expected(reader, &names.attn_k_weight, d * d)?,
        attn_k_bias: load_expected(reader, &names.attn_k_bias, d)?,
        attn_v_weight: load_expected(reader, &names.attn_v_weight, d * d)?,
        attn_v_bias: load_expected(reader, &names.attn_v_bias, d)?,
        attn_output_weight: load_expected(reader, &names.attn_output_weight, d * d)?,
        attn_output_bias: load_expected(reader, &names.attn_output_bias, d)?,
        attn_norm_weight: load_expected(reader, &names.attn_norm_weight, d)?,
        attn_norm_bias: load_expected(reader, &names.attn_norm_bias, d)?,
        ffn_up_weight: load_expected(reader, &names.ffn_up_weight, d * ffn)?,
        ffn_up_bias: load_expected(reader, &names.ffn_up_bias, ffn)?,
        ffn_down_weight: load_expected(reader, &names.ffn_down_weight, ffn * d)?,
        ffn_down_bias: load_expected(reader, &names.ffn_down_bias, d)?,
        ffn_norm_weight: load_expected(reader, &names.ffn_norm_weight, d)?,
        ffn_norm_bias: load_expected(reader, &names.ffn_norm_bias, d)?,
    })
}

pub(crate) fn load_firered_punc_weights(
    reader: &GgufTensorDataReader,
    metadata: &FireRedPuncExecutionMetadata,
) -> Result<FireRedPuncWeights, FireRedPuncWeightsError> {
    let d = metadata.d_model;
    let mut layers = Vec::with_capacity(metadata.layers);
    for layer in 0..metadata.layers {
        layers.push(load_layer(reader, layer, metadata)?);
    }
    Ok(FireRedPuncWeights {
        token_embd: load_expected(reader, TOKEN_EMBD_WEIGHT, d * metadata.vocab_size)?,
        token_type_embd: load_named(reader, TOKEN_TYPE_EMBD_WEIGHT)?,
        position_embd: load_expected(reader, POSITION_EMBD_WEIGHT, d * metadata.max_positions)?,
        embd_norm_weight: load_expected(reader, EMBD_NORM_WEIGHT, d)?,
        embd_norm_bias: load_expected(reader, EMBD_NORM_BIAS, d)?,
        layers,
        punc_head_weight: load_expected(reader, PUNC_HEAD_WEIGHT, d * metadata.label_count)?,
        punc_head_bias: load_expected(reader, PUNC_HEAD_BIAS, metadata.label_count)?,
    })
}
