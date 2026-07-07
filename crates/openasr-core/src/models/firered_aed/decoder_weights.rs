//! Loaded-GGUF-tensor handles for the firered-aed Transformer decoder.
//!
//! Mirrors [`super::encoder_weights`]: every decoder tensor is referenced
//! directly from the `.oasr` GGUF's mmap'd, already-resident buffer via
//! [`crate::ggml_runtime::GgmlLoadedWeightContext::tensor`] (keep-quantized,
//! no separate arena upload for weights).
//!
//! `self_attn.w_ks` and `cross_attn.w_ks` are upstream `nn.Linear(bias=False)`
//! (see `package_import`'s decoder tensor-name mapping), so this loader has no
//! bias field for either -- the decoder graph supplies one shared zero-filled
//! bias tensor for both K projections instead of baking a redundant all-zero
//! tensor into the pack.

#![allow(dead_code)]

use crate::ggml_runtime::{GgmlLoadedTensor, GgmlLoadedWeightContext};

#[derive(Debug, thiserror::Error)]
pub(crate) enum FireRedDecoderWeightsError {
    #[error("firered-aed decoder runtime is missing tensor '{name}'")]
    MissingTensor { name: String },
}

#[derive(Clone, Copy)]
pub(crate) struct FireRedDecoderLayerWeights {
    pub self_attn_norm_weight: GgmlLoadedTensor,
    pub self_attn_norm_bias: GgmlLoadedTensor,
    pub self_attn_q_weight: GgmlLoadedTensor,
    pub self_attn_q_bias: GgmlLoadedTensor,
    pub self_attn_k_weight: GgmlLoadedTensor,
    pub self_attn_v_weight: GgmlLoadedTensor,
    pub self_attn_v_bias: GgmlLoadedTensor,
    pub self_attn_out_weight: GgmlLoadedTensor,
    pub self_attn_out_bias: GgmlLoadedTensor,
    pub cross_attn_norm_weight: GgmlLoadedTensor,
    pub cross_attn_norm_bias: GgmlLoadedTensor,
    pub cross_attn_q_weight: GgmlLoadedTensor,
    pub cross_attn_q_bias: GgmlLoadedTensor,
    pub cross_attn_k_weight: GgmlLoadedTensor,
    pub cross_attn_v_weight: GgmlLoadedTensor,
    pub cross_attn_v_bias: GgmlLoadedTensor,
    pub cross_attn_out_weight: GgmlLoadedTensor,
    pub cross_attn_out_bias: GgmlLoadedTensor,
    pub ffn_norm_weight: GgmlLoadedTensor,
    pub ffn_norm_bias: GgmlLoadedTensor,
    pub ffn_up_weight: GgmlLoadedTensor,
    pub ffn_up_bias: GgmlLoadedTensor,
    pub ffn_down_weight: GgmlLoadedTensor,
    pub ffn_down_bias: GgmlLoadedTensor,
}

pub(crate) struct FireRedDecoderWeights {
    /// `decoder.tgt_word_emb.weight` -- an `ggml_get_rows` embedding table,
    /// stored separately from `out_proj_weight` even though the checkpoint
    /// ties them at training time (upstream state dict keeps two copies).
    pub token_embedding: GgmlLoadedTensor,
    /// `decoder.positional_encoding.pe`, absolute sinusoidal, `[max_positions, d_model]`.
    pub positional_encoding: GgmlLoadedTensor,
    pub layers: Vec<FireRedDecoderLayerWeights>,
    pub out_norm_weight: GgmlLoadedTensor,
    pub out_norm_bias: GgmlLoadedTensor,
    /// `decoder.tgt_word_prj.weight`, untied `mul_mat` operand, bias-free
    /// (upstream `nn.Linear(d_model, vocab, bias=False)`).
    pub out_proj_weight: GgmlLoadedTensor,
}

fn tensor(
    loaded: &GgmlLoadedWeightContext,
    name: &str,
) -> Result<GgmlLoadedTensor, FireRedDecoderWeightsError> {
    loaded
        .tensor(name)
        .ok_or_else(|| FireRedDecoderWeightsError::MissingTensor {
            name: name.to_string(),
        })
}

impl FireRedDecoderWeights {
    pub(crate) fn load(
        loaded: &GgmlLoadedWeightContext,
        n_layers: usize,
    ) -> Result<Self, FireRedDecoderWeightsError> {
        let mut layers = Vec::with_capacity(n_layers);
        for layer_idx in 0..n_layers {
            let p = format!("dec.blk.{layer_idx}");
            layers.push(FireRedDecoderLayerWeights {
                self_attn_norm_weight: tensor(loaded, &format!("{p}.self_attn.norm.weight"))?,
                self_attn_norm_bias: tensor(loaded, &format!("{p}.self_attn.norm.bias"))?,
                self_attn_q_weight: tensor(loaded, &format!("{p}.self_attn.q.weight"))?,
                self_attn_q_bias: tensor(loaded, &format!("{p}.self_attn.q.bias"))?,
                self_attn_k_weight: tensor(loaded, &format!("{p}.self_attn.k.weight"))?,
                self_attn_v_weight: tensor(loaded, &format!("{p}.self_attn.v.weight"))?,
                self_attn_v_bias: tensor(loaded, &format!("{p}.self_attn.v.bias"))?,
                self_attn_out_weight: tensor(loaded, &format!("{p}.self_attn.out.weight"))?,
                self_attn_out_bias: tensor(loaded, &format!("{p}.self_attn.out.bias"))?,
                cross_attn_norm_weight: tensor(loaded, &format!("{p}.cross_attn.norm.weight"))?,
                cross_attn_norm_bias: tensor(loaded, &format!("{p}.cross_attn.norm.bias"))?,
                cross_attn_q_weight: tensor(loaded, &format!("{p}.cross_attn.q.weight"))?,
                cross_attn_q_bias: tensor(loaded, &format!("{p}.cross_attn.q.bias"))?,
                cross_attn_k_weight: tensor(loaded, &format!("{p}.cross_attn.k.weight"))?,
                cross_attn_v_weight: tensor(loaded, &format!("{p}.cross_attn.v.weight"))?,
                cross_attn_v_bias: tensor(loaded, &format!("{p}.cross_attn.v.bias"))?,
                cross_attn_out_weight: tensor(loaded, &format!("{p}.cross_attn.out.weight"))?,
                cross_attn_out_bias: tensor(loaded, &format!("{p}.cross_attn.out.bias"))?,
                ffn_norm_weight: tensor(loaded, &format!("{p}.ffn.norm.weight"))?,
                ffn_norm_bias: tensor(loaded, &format!("{p}.ffn.norm.bias"))?,
                ffn_up_weight: tensor(loaded, &format!("{p}.ffn.up.weight"))?,
                ffn_up_bias: tensor(loaded, &format!("{p}.ffn.up.bias"))?,
                ffn_down_weight: tensor(loaded, &format!("{p}.ffn.down.weight"))?,
                ffn_down_bias: tensor(loaded, &format!("{p}.ffn.down.bias"))?,
            });
        }
        Ok(Self {
            token_embedding: tensor(loaded, "dec.tok_emb.weight")?,
            positional_encoding: tensor(loaded, "dec.pos_enc.pe")?,
            layers,
            out_norm_weight: tensor(loaded, "dec.out_norm.weight")?,
            out_norm_bias: tensor(loaded, "dec.out_norm.bias")?,
            out_proj_weight: tensor(loaded, "dec.out_proj.weight")?,
        })
    }
}
