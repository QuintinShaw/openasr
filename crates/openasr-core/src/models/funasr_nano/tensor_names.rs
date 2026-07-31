//! `.oasr` tensor-name constants for the funasr-nano family (FunASR SAN-M
//! encoder -> 2-layer transformer adaptor -> Qwen3-0.6B decoder).
//!
//! The SAN-M encoder branch reuses the exact `enc.blk.{i}.*` / `tp.blk.{i}.*` /
//! `enc.after_norm.*` / `tp.norm.*` naming the `sensevoice` family already
//! establishes (both are FunASR SAN-M/DFSMN encoders, byte-for-byte the same
//! per-layer op sequence), so this family's encoder tensors load through the
//! same `attn.{norm,qkv,out,fsmn}` / `ffn.{norm,up,down}` slots. The adaptor
//! branch (2x standard transformer blocks) and the LLM branch are new. The
//! decoder is a stock Qwen3-0.6B (QK-norm, no attention bias, GQA, tied
//! embeddings), so it reuses `qwen3-asr`'s exact per-layer tensor slot names
//! (`attn_q_norm`/`attn_k_norm`, no `*_bias`) under a bare `blk.N.*` scope, so a
//! runtime loader written against `qwen::QwenFamilyLlmLayerTensorNames`'s
//! generic loaders consumes the decoder branch without modification.

use crate::models::tensor_schema::layer_tensor_names;

// --- SAN-M encoder (50 enc blocks + 20 tp blocks) --------------------------

pub(crate) const ENC_AFTER_NORM_WEIGHT: &str = "enc.after_norm.weight";
pub(crate) const ENC_AFTER_NORM_BIAS: &str = "enc.after_norm.bias";
pub(crate) const TP_NORM_WEIGHT: &str = "tp.norm.weight";
pub(crate) const TP_NORM_BIAS: &str = "tp.norm.bias";

// --- 2-layer transformer adaptor -------------------------------------------

pub(crate) const ADAPTOR_LINEAR1_WEIGHT: &str = "adaptor.linear1.weight";
pub(crate) const ADAPTOR_LINEAR1_BIAS: &str = "adaptor.linear1.bias";
pub(crate) const ADAPTOR_LINEAR2_WEIGHT: &str = "adaptor.linear2.weight";
pub(crate) const ADAPTOR_LINEAR2_BIAS: &str = "adaptor.linear2.bias";

// --- Qwen3-0.6B decoder (tied embeddings + a materialized `output.weight`) --

pub(crate) const LLM_TOKEN_EMBD_WEIGHT: &str = "token_embd.weight";
pub(crate) const LLM_OUTPUT_NORM_WEIGHT: &str = "output_norm.weight";
pub(crate) const LLM_OUTPUT_WEIGHT: &str = "output.weight";

layer_tensor_names! {
    pub(crate) struct FunasrNanoLlmLayerTensorNames;
    pub(crate) fn funasr_nano_llm_layer_tensor_names @ "blk";
    {
        attn_norm_weight => "attn_norm.weight",
        attn_q_weight => "attn_q.weight",
        attn_k_weight => "attn_k.weight",
        attn_v_weight => "attn_v.weight",
        attn_output_weight => "attn_output.weight",
        attn_q_norm_weight => "attn_q_norm.weight",
        attn_k_norm_weight => "attn_k_norm.weight",
        ffn_norm_weight => "ffn_norm.weight",
        ffn_gate_weight => "ffn_gate.weight",
        ffn_up_weight => "ffn_up.weight",
        ffn_down_weight => "ffn_down.weight",
    }
}

/// Encoder-half tensor name prefixes (SAN-M encoder + adaptor), used by
/// `pack_quant_audit` to classify the audio-encoder half of this two-part pack
/// (the LLM half's `blk.` / `token_embd` / `output*` tensors are the Qwen3
/// decoder, not an audio encoder).
pub(crate) const AUDIO_ENCODER_TENSOR_NAME_PREFIXES: &[&str] = &["enc.", "tp.", "adaptor."];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_layer_tensor_names_match_runtime_convention() {
        let names = funasr_nano_llm_layer_tensor_names(5);
        assert_eq!(names.attn_q_norm_weight, "blk.5.attn_q_norm.weight");
        assert_eq!(names.ffn_gate_weight, "blk.5.ffn_gate.weight");
        assert_eq!(names.attn_output_weight, "blk.5.attn_output.weight");
    }
}
