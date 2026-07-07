//! Loaded-GGUF-tensor handles for the firered-aed Conformer encoder.
//!
//! Unlike cohere's dual-path (CPU-side value vec + optional raw-ggml fast
//! path), firered-aed is greenfield: every encoder tensor is referenced
//! directly from the `.oasr` GGUF's mmap'd, already-resident buffer via
//! [`crate::ggml_runtime::GgmlLoadedWeightContext::tensor`]. There is no
//! separate static-tensor-arena upload step for weights (matches
//! keep-quantized: whatever `mul_mat`-compatible type the pack stores a
//! projection in is used as-is, unchanged by this loader).

#![allow(dead_code)]

use crate::ggml_runtime::{GgmlLoadedTensor, GgmlLoadedWeightContext};

#[derive(Debug, thiserror::Error)]
pub(crate) enum FireRedEncoderWeightsError {
    #[error("firered-aed encoder runtime is missing tensor '{name}'")]
    MissingTensor { name: String },
}

#[derive(Clone, Copy)]
pub(crate) struct FireRedEncoderLayerWeights {
    pub ffn1_norm_weight: GgmlLoadedTensor,
    pub ffn1_norm_bias: GgmlLoadedTensor,
    pub ffn1_up_weight: GgmlLoadedTensor,
    pub ffn1_up_bias: GgmlLoadedTensor,
    pub ffn1_down_weight: GgmlLoadedTensor,
    pub ffn1_down_bias: GgmlLoadedTensor,
    pub attn_norm_q_weight: GgmlLoadedTensor,
    pub attn_norm_q_bias: GgmlLoadedTensor,
    pub attn_norm_k_weight: GgmlLoadedTensor,
    pub attn_norm_k_bias: GgmlLoadedTensor,
    pub attn_norm_v_weight: GgmlLoadedTensor,
    pub attn_norm_v_bias: GgmlLoadedTensor,
    pub attn_q_weight: GgmlLoadedTensor,
    pub attn_k_weight: GgmlLoadedTensor,
    pub attn_v_weight: GgmlLoadedTensor,
    pub attn_out_weight: GgmlLoadedTensor,
    pub attn_pos_weight: GgmlLoadedTensor,
    pub attn_pos_bias_u: GgmlLoadedTensor,
    pub attn_pos_bias_v: GgmlLoadedTensor,
    pub conv_norm_weight: GgmlLoadedTensor,
    pub conv_norm_bias: GgmlLoadedTensor,
    pub conv_pw1_weight: GgmlLoadedTensor,
    pub conv_dw_weight: GgmlLoadedTensor,
    pub conv_ln_weight: GgmlLoadedTensor,
    pub conv_ln_bias: GgmlLoadedTensor,
    pub conv_pw2_weight: GgmlLoadedTensor,
    pub ffn2_norm_weight: GgmlLoadedTensor,
    pub ffn2_norm_bias: GgmlLoadedTensor,
    pub ffn2_up_weight: GgmlLoadedTensor,
    pub ffn2_up_bias: GgmlLoadedTensor,
    pub ffn2_down_weight: GgmlLoadedTensor,
    pub ffn2_down_bias: GgmlLoadedTensor,
    pub out_norm_weight: GgmlLoadedTensor,
    pub out_norm_bias: GgmlLoadedTensor,
}

pub(crate) struct FireRedEncoderWeights {
    pub subsample_conv1_weight: GgmlLoadedTensor,
    pub subsample_conv1_bias: GgmlLoadedTensor,
    pub subsample_conv2_weight: GgmlLoadedTensor,
    pub subsample_conv2_bias: GgmlLoadedTensor,
    pub subsample_out_weight: GgmlLoadedTensor,
    pub subsample_out_bias: GgmlLoadedTensor,
    pub layers: Vec<FireRedEncoderLayerWeights>,
}

fn tensor(
    loaded: &GgmlLoadedWeightContext,
    name: &str,
) -> Result<GgmlLoadedTensor, FireRedEncoderWeightsError> {
    loaded
        .tensor(name)
        .ok_or_else(|| FireRedEncoderWeightsError::MissingTensor {
            name: name.to_string(),
        })
}

impl FireRedEncoderWeights {
    pub(crate) fn load(
        loaded: &GgmlLoadedWeightContext,
        n_layers: usize,
    ) -> Result<Self, FireRedEncoderWeightsError> {
        let mut layers = Vec::with_capacity(n_layers);
        for layer_idx in 0..n_layers {
            let p = format!("enc.blk.{layer_idx}");
            layers.push(FireRedEncoderLayerWeights {
                ffn1_norm_weight: tensor(loaded, &format!("{p}.ffn1.norm.weight"))?,
                ffn1_norm_bias: tensor(loaded, &format!("{p}.ffn1.norm.bias"))?,
                ffn1_up_weight: tensor(loaded, &format!("{p}.ffn1.up.weight"))?,
                ffn1_up_bias: tensor(loaded, &format!("{p}.ffn1.up.bias"))?,
                ffn1_down_weight: tensor(loaded, &format!("{p}.ffn1.down.weight"))?,
                ffn1_down_bias: tensor(loaded, &format!("{p}.ffn1.down.bias"))?,
                attn_norm_q_weight: tensor(loaded, &format!("{p}.attn.norm_q.weight"))?,
                attn_norm_q_bias: tensor(loaded, &format!("{p}.attn.norm_q.bias"))?,
                attn_norm_k_weight: tensor(loaded, &format!("{p}.attn.norm_k.weight"))?,
                attn_norm_k_bias: tensor(loaded, &format!("{p}.attn.norm_k.bias"))?,
                attn_norm_v_weight: tensor(loaded, &format!("{p}.attn.norm_v.weight"))?,
                attn_norm_v_bias: tensor(loaded, &format!("{p}.attn.norm_v.bias"))?,
                attn_q_weight: tensor(loaded, &format!("{p}.attn.q.weight"))?,
                attn_k_weight: tensor(loaded, &format!("{p}.attn.k.weight"))?,
                attn_v_weight: tensor(loaded, &format!("{p}.attn.v.weight"))?,
                attn_out_weight: tensor(loaded, &format!("{p}.attn.out.weight"))?,
                attn_pos_weight: tensor(loaded, &format!("{p}.attn.pos.weight"))?,
                attn_pos_bias_u: tensor(loaded, &format!("{p}.attn.pos_bias_u"))?,
                attn_pos_bias_v: tensor(loaded, &format!("{p}.attn.pos_bias_v"))?,
                conv_norm_weight: tensor(loaded, &format!("{p}.conv.norm.weight"))?,
                conv_norm_bias: tensor(loaded, &format!("{p}.conv.norm.bias"))?,
                conv_pw1_weight: tensor(loaded, &format!("{p}.conv.pw1.weight"))?,
                conv_dw_weight: tensor(loaded, &format!("{p}.conv.dw.weight"))?,
                conv_ln_weight: tensor(loaded, &format!("{p}.conv.ln.weight"))?,
                conv_ln_bias: tensor(loaded, &format!("{p}.conv.ln.bias"))?,
                conv_pw2_weight: tensor(loaded, &format!("{p}.conv.pw2.weight"))?,
                ffn2_norm_weight: tensor(loaded, &format!("{p}.ffn2.norm.weight"))?,
                ffn2_norm_bias: tensor(loaded, &format!("{p}.ffn2.norm.bias"))?,
                ffn2_up_weight: tensor(loaded, &format!("{p}.ffn2.up.weight"))?,
                ffn2_up_bias: tensor(loaded, &format!("{p}.ffn2.up.bias"))?,
                ffn2_down_weight: tensor(loaded, &format!("{p}.ffn2.down.weight"))?,
                ffn2_down_bias: tensor(loaded, &format!("{p}.ffn2.down.bias"))?,
                out_norm_weight: tensor(loaded, &format!("{p}.out_norm.weight"))?,
                out_norm_bias: tensor(loaded, &format!("{p}.out_norm.bias"))?,
            });
        }
        Ok(Self {
            subsample_conv1_weight: tensor(loaded, "enc.subsample.conv1.weight")?,
            subsample_conv1_bias: tensor(loaded, "enc.subsample.conv1.bias")?,
            subsample_conv2_weight: tensor(loaded, "enc.subsample.conv2.weight")?,
            subsample_conv2_bias: tensor(loaded, "enc.subsample.conv2.bias")?,
            subsample_out_weight: tensor(loaded, "enc.subsample.out.weight")?,
            subsample_out_bias: tensor(loaded, "enc.subsample.out.bias")?,
            layers,
        })
    }
}
