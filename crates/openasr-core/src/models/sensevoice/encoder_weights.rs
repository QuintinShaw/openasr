//! Load a sensevoice `.oasr` pack into host weights.
//!
//! Mirrors `parakeet_ctc::encoder_weights`: every tensor is read generically
//! (dims from the GGUF index, values dequantized to f32); the 2-D linear
//! projections (`attn.qkv/out`, `ffn.up/down`, `ctc.head.weight`) are bound
//! zero-copy from the mmap'd pack by the encoder graph, so their f32 host
//! payloads are dropped after shape validation (keep-quantized: the graph's
//! `mul_mat` consumes the native q8_0/q4_k blocks straight from the pack).
//! Norms, biases, and the FSMN depthwise kernels stay host-resident (arena
//! uploads); the CMVN vectors and the 16x560 prompt-embedding table are
//! consumed host-side by the frontend/prompt splice, never by the graph.

#![allow(dead_code)]

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};

use super::runtime_contract::SenseVoiceExecutionMetadata;

#[derive(Debug, thiserror::Error)]
pub(crate) enum SenseVoiceEncoderWeightsError {
    #[error("sensevoice encoder weight read failed: {0}")]
    Read(#[from] GgufTensorDataReadError),
    #[error("sensevoice encoder tensor '{name}' has {got} elements, expected {expected}")]
    ElementCount {
        name: String,
        got: usize,
        expected: usize,
    },
}

/// A host weight: its stored dims (from the GGUF index) + dequantized f32 values.
#[derive(Debug, Clone)]
pub(crate) struct NamedTensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub values: Vec<f32>,
}

impl NamedTensor {
    fn element_count(&self) -> usize {
        self.values.len()
    }

    /// Drop the resident f32 host `values` (keeping name + dims) for a weight
    /// the encoder graph binds zero-copy from the mmap'd pack.
    fn drop_bound_payload(&mut self) {
        self.values = Vec::new();
    }
}

/// One SAN-M block's weights (`enc.blk.{i}.*` or `tp.blk.{i}.*`).
#[derive(Debug, Clone)]
pub(crate) struct SenseVoiceLayerWeights {
    pub attn_norm_weight: NamedTensor,
    pub attn_norm_bias: NamedTensor,
    /// Fused `[in, 3*d_model]` QKV projection (bound zero-copy).
    pub attn_qkv_weight: NamedTensor,
    pub attn_qkv_bias: NamedTensor,
    pub attn_out_weight: NamedTensor,
    pub attn_out_bias: NamedTensor,
    /// FSMN depthwise conv kernel `[kernel, 1, d_model]` (f16 arena upload).
    pub attn_fsmn_weight: NamedTensor,
    pub ffn_norm_weight: NamedTensor,
    pub ffn_norm_bias: NamedTensor,
    pub ffn_up_weight: NamedTensor,
    pub ffn_up_bias: NamedTensor,
    pub ffn_down_weight: NamedTensor,
    pub ffn_down_bias: NamedTensor,
}

#[derive(Debug, Clone)]
pub(crate) struct SenseVoiceEncoderWeights {
    /// `enc.blk.0..n_layers-1` (block 0 consumes the 560-dim LFR+prompt input).
    pub enc_layers: Vec<SenseVoiceLayerWeights>,
    /// `tp.blk.0..tp_layers-1`, run after `enc_after_norm`.
    pub tp_layers: Vec<SenseVoiceLayerWeights>,
    pub enc_after_norm_weight: NamedTensor,
    pub enc_after_norm_bias: NamedTensor,
    pub tp_norm_weight: NamedTensor,
    pub tp_norm_bias: NamedTensor,
    pub ctc_head_weight: NamedTensor,
    pub ctc_head_bias: NamedTensor,
    /// 16x560 prompt-embedding table (host f32, spliced by the frontend).
    pub prompt_embed: NamedTensor,
    /// `am.mvn` CMVN vectors (host f32, applied by the frontend).
    pub cmvn_neg_mean: NamedTensor,
    pub cmvn_inv_stddev: NamedTensor,
}

fn load_named(
    reader: &GgufTensorDataReader,
    name: &str,
) -> Result<NamedTensor, SenseVoiceEncoderWeightsError> {
    let tensor = reader.tensor_index().get(name).ok_or_else(|| {
        SenseVoiceEncoderWeightsError::Read(GgufTensorDataReadError::TensorNotFound {
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

fn load_layer(
    reader: &GgufTensorDataReader,
    scope: &str,
    layer: usize,
) -> Result<SenseVoiceLayerWeights, SenseVoiceEncoderWeightsError> {
    let n = |suffix: &str| format!("{scope}.{layer}.{suffix}");
    let mut weights = SenseVoiceLayerWeights {
        attn_norm_weight: load_named(reader, &n("attn.norm.weight"))?,
        attn_norm_bias: load_named(reader, &n("attn.norm.bias"))?,
        attn_qkv_weight: load_named(reader, &n("attn.qkv.weight"))?,
        attn_qkv_bias: load_named(reader, &n("attn.qkv.bias"))?,
        attn_out_weight: load_named(reader, &n("attn.out.weight"))?,
        attn_out_bias: load_named(reader, &n("attn.out.bias"))?,
        attn_fsmn_weight: load_named(reader, &n("attn.fsmn.weight"))?,
        ffn_norm_weight: load_named(reader, &n("ffn.norm.weight"))?,
        ffn_norm_bias: load_named(reader, &n("ffn.norm.bias"))?,
        ffn_up_weight: load_named(reader, &n("ffn.up.weight"))?,
        ffn_up_bias: load_named(reader, &n("ffn.up.bias"))?,
        ffn_down_weight: load_named(reader, &n("ffn.down.weight"))?,
        ffn_down_bias: load_named(reader, &n("ffn.down.bias"))?,
    };
    // Bound zero-copy by the graph: drop the dominant f32 host payloads.
    for w in [
        &mut weights.attn_qkv_weight,
        &mut weights.attn_out_weight,
        &mut weights.ffn_up_weight,
        &mut weights.ffn_down_weight,
    ] {
        w.drop_bound_payload();
    }
    Ok(weights)
}

pub(crate) fn load_sensevoice_encoder_weights(
    reader: &GgufTensorDataReader,
    metadata: &SenseVoiceExecutionMetadata,
) -> Result<SenseVoiceEncoderWeights, SenseVoiceEncoderWeightsError> {
    let mut enc_layers = Vec::with_capacity(metadata.n_layers);
    for layer in 0..metadata.n_layers {
        enc_layers.push(load_layer(reader, "enc.blk", layer)?);
    }
    let mut tp_layers = Vec::with_capacity(metadata.tp_layers);
    for layer in 0..metadata.tp_layers {
        tp_layers.push(load_layer(reader, "tp.blk", layer)?);
    }

    let mut ctc_head_weight = load_named(reader, "ctc.head.weight")?;
    let ctc_head_bias = load_named(reader, "ctc.head.bias")?;
    let expected_head = metadata.vocab_size * metadata.d_model;
    if ctc_head_weight.element_count() != expected_head {
        return Err(SenseVoiceEncoderWeightsError::ElementCount {
            name: ctc_head_weight.name.clone(),
            got: ctc_head_weight.element_count(),
            expected: expected_head,
        });
    }
    ctc_head_weight.drop_bound_payload();

    let prompt_embed = load_named(reader, "embed.prompt.weight")?;
    if !prompt_embed
        .element_count()
        .is_multiple_of(metadata.feature_dim)
    {
        return Err(SenseVoiceEncoderWeightsError::ElementCount {
            name: prompt_embed.name.clone(),
            got: prompt_embed.element_count(),
            expected: metadata.feature_dim,
        });
    }
    let cmvn_neg_mean = load_named(reader, "frontend.cmvn.neg_mean")?;
    let cmvn_inv_stddev = load_named(reader, "frontend.cmvn.inv_stddev")?;
    for cmvn in [&cmvn_neg_mean, &cmvn_inv_stddev] {
        if cmvn.element_count() != metadata.feature_dim {
            return Err(SenseVoiceEncoderWeightsError::ElementCount {
                name: cmvn.name.clone(),
                got: cmvn.element_count(),
                expected: metadata.feature_dim,
            });
        }
    }

    Ok(SenseVoiceEncoderWeights {
        enc_layers,
        tp_layers,
        enc_after_norm_weight: load_named(reader, "enc.after_norm.weight")?,
        enc_after_norm_bias: load_named(reader, "enc.after_norm.bias")?,
        tp_norm_weight: load_named(reader, "tp.norm.weight")?,
        tp_norm_bias: load_named(reader, "tp.norm.bias")?,
        ctc_head_weight,
        ctc_head_bias,
        prompt_embed,
        cmvn_neg_mean,
        cmvn_inv_stddev,
    })
}
