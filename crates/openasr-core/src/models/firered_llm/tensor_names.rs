//! `.oasr` tensor-name constants for the firered-llm family (Encoder-Adapter-
//! Qwen2 decoder). The encoder branch reuses the exact `enc.blk.{i}.*` /
//! `enc.subsample.*` / `enc.pos_enc.pe` naming the `firered_aed` importer
//! already establishes (see `package_import.rs`'s `map_firered_encoder_tensor_name`,
//! which is a direct trim of `firered_aed::package_import::map_firered_tensor_name`
//! down to its encoder branch -- no decoder exists in this family). The
//! adapter and LLM branches are new to this family.

use crate::models::tensor_schema::layer_tensor_names;

pub(crate) const ADAPTER_LINEAR1_WEIGHT: &str = "adapter.linear1.weight";
pub(crate) const ADAPTER_LINEAR1_BIAS: &str = "adapter.linear1.bias";
pub(crate) const ADAPTER_LINEAR2_WEIGHT: &str = "adapter.linear2.weight";
pub(crate) const ADAPTER_LINEAR2_BIAS: &str = "adapter.linear2.bias";

pub(crate) const LLM_TOKEN_EMBD_WEIGHT: &str = "llm.tok_emb.weight";
pub(crate) const LLM_OUTPUT_NORM_WEIGHT: &str = "llm.out_norm.weight";
pub(crate) const LLM_OUTPUT_WEIGHT: &str = "llm.lm_head.weight";

// Qwen2 (not Qwen3): has q/k/v projection biases, has NO q_norm/k_norm
// (QK-norm) -- the inverse of the qwen3-asr LLM branch's tensor set. See
// scratchpad/fr2/T1-findings.md S1's "重大修正" for why the base weights come
// from the official Qwen2-7B-Instruct checkpoint (LoRA-merged upstream of
// this importer) rather than from `model.pth.tar` directly.
layer_tensor_names! {
    pub(crate) struct FireRedLlmQwen2LayerTensorNames;
    pub(crate) fn qwen2_llm_layer_tensor_names @ "llm.blk";
    {
        attn_norm_weight => "attn_norm.weight",
        attn_q_weight => "attn_q.weight",
        attn_q_bias => "attn_q.bias",
        attn_k_weight => "attn_k.weight",
        attn_k_bias => "attn_k.bias",
        attn_v_weight => "attn_v.weight",
        attn_v_bias => "attn_v.bias",
        attn_out_weight => "attn_out.weight",
        ffn_norm_weight => "ffn_norm.weight",
        ffn_gate_weight => "ffn_gate.weight",
        ffn_up_weight => "ffn_up.weight",
        ffn_down_weight => "ffn_down.weight",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen2_llm_layer_tensor_names_match_runtime_convention() {
        let names = qwen2_llm_layer_tensor_names(5);
        assert_eq!(names.attn_q_weight, "llm.blk.5.attn_q.weight");
        assert_eq!(names.attn_q_bias, "llm.blk.5.attn_q.bias");
        assert_eq!(names.ffn_gate_weight, "llm.blk.5.ffn_gate.weight");
    }
}
