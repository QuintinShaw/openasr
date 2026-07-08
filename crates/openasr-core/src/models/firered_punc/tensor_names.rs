//! GGUF tensor names for the FireRedPunc BERT encoder + classification head.
//!
//! BERT uses biased projections and full (weight + bias) LayerNorms throughout,
//! unlike the RMS-norm, bias-free LLM families. The conversion recipe in
//! `tooling/publish-model` maps the upstream `chinese-lert-base` / classifier
//! state dict onto these names; the runtime binds them back by the same names.

use crate::models::tensor_schema::layer_tensor_names;

/// Word embedding table `[d_model, vocab_size]`.
pub(crate) const TOKEN_EMBD_WEIGHT: &str = "token_embd.weight";
/// Segment (token-type) embedding table `[d_model, 2]`. Punctuation inference
/// is single-segment, so only row 0 is ever gathered, but the tensor is kept
/// for a faithful checkpoint round-trip.
pub(crate) const TOKEN_TYPE_EMBD_WEIGHT: &str = "token_type_embd.weight";
/// Learned absolute position table `[d_model, max_positions]`.
pub(crate) const POSITION_EMBD_WEIGHT: &str = "position_embd.weight";
/// Post-embedding LayerNorm (BERT `embeddings.LayerNorm`).
pub(crate) const EMBD_NORM_WEIGHT: &str = "embd_norm.weight";
pub(crate) const EMBD_NORM_BIAS: &str = "embd_norm.bias";
/// Token-classification head `[d_model, label_count]` + bias `[label_count]`.
pub(crate) const PUNC_HEAD_WEIGHT: &str = "punc_head.weight";
pub(crate) const PUNC_HEAD_BIAS: &str = "punc_head.bias";

layer_tensor_names! {
    pub(crate) struct FireRedPuncLayerTensorNames;
    pub(crate) fn firered_punc_layer_tensor_names @ "blk";
    {
        attn_q_weight => "attn_q.weight",
        attn_q_bias => "attn_q.bias",
        attn_k_weight => "attn_k.weight",
        attn_k_bias => "attn_k.bias",
        attn_v_weight => "attn_v.weight",
        attn_v_bias => "attn_v.bias",
        attn_output_weight => "attn_output.weight",
        attn_output_bias => "attn_output.bias",
        attn_norm_weight => "attn_norm.weight",
        attn_norm_bias => "attn_norm.bias",
        ffn_up_weight => "ffn_up.weight",
        ffn_up_bias => "ffn_up.bias",
        ffn_down_weight => "ffn_down.weight",
        ffn_down_bias => "ffn_down.bias",
        ffn_norm_weight => "ffn_norm.weight",
        ffn_norm_bias => "ffn_norm.bias",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn firered_punc_layer_tensor_names_are_block_scoped() {
        let names = firered_punc_layer_tensor_names(7);
        assert_eq!(names.attn_q_weight, "blk.7.attn_q.weight");
        assert_eq!(names.attn_norm_bias, "blk.7.attn_norm.bias");
        assert_eq!(names.ffn_down_weight, "blk.7.ffn_down.weight");
        assert_eq!(names.ffn_norm_weight, "blk.7.ffn_norm.weight");
    }
}
