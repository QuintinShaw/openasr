//! `.oasr` tensor-name constants for the moss-transcribe-diarize family
//! (Whisper-Medium-style encoder -> VQAdaptor bridge -> Qwen3-0.6B decoder).
//!
//! The encoder is architecturally a standard HF `WhisperEncoder` (see
//! `package_import`'s module doc for the upstream tensor names this maps
//! from), but this family does not reuse `whisper::package_import`'s tensor
//! names directly: that importer writes verbatim HF names
//! (`model.encoder.*`) into a GGUF that also always carries a whisper
//! *decoder* branch, which this family has none of. Scoping this family's
//! own `moss.enc.*` namespace (mirroring the `firered_llm` vs `firered_aed`
//! precedent: architecturally identical encoder, independent tensor
//! namespace because the owning pack has a different tensor-name contract)
//! keeps this pack self-describing without depending on whisper's
//! decoder-inclusive binding contract.
//!
//! The decoder branch is genuinely Qwen3-parameterized (QK-norm, no
//! attention bias, GQA) -- the same LLM parameter family `qwen3-asr`
//! already uses -- so this reuses `qwen3-asr`'s exact per-layer tensor slot
//! names (`attn_q_norm`/`attn_k_norm`, no `*_bias` slots) under this
//! family's own `moss.llm.blk.N.*` scope, so a runtime loader written
//! against `qwen::QwenFamilyLlmLayerTensorNames`'s generic tensor-name-driven
//! loaders can consume this pack's decoder branch without modification.

use crate::models::tensor_schema::layer_tensor_names;

// --- Whisper-Medium-style encoder (24 layers, d_model=1024, 16 heads) ------

pub(crate) const ENC_CONV1_WEIGHT: &str = "moss.enc.conv1.weight";
pub(crate) const ENC_CONV1_BIAS: &str = "moss.enc.conv1.bias";
pub(crate) const ENC_CONV2_WEIGHT: &str = "moss.enc.conv2.weight";
pub(crate) const ENC_CONV2_BIAS: &str = "moss.enc.conv2.bias";
pub(crate) const ENC_POS_EMBD_WEIGHT: &str = "moss.enc.pos_embd.weight";
pub(crate) const ENC_OUT_NORM_WEIGHT: &str = "moss.enc.out_norm.weight";
pub(crate) const ENC_OUT_NORM_BIAS: &str = "moss.enc.out_norm.bias";

layer_tensor_names! {
    pub(crate) struct MossEncoderLayerTensorNames;
    pub(crate) fn moss_encoder_layer_tensor_names @ "moss.enc.blk";
    {
        attn_norm_weight => "attn_norm.weight",
        attn_norm_bias => "attn_norm.bias",
        attn_q_weight => "attn_q.weight",
        attn_q_bias => "attn_q.bias",
        attn_k_weight => "attn_k.weight",
        attn_v_weight => "attn_v.weight",
        attn_v_bias => "attn_v.bias",
        attn_out_weight => "attn_out.weight",
        attn_out_bias => "attn_out.bias",
        ffn_norm_weight => "ffn_norm.weight",
        ffn_norm_bias => "ffn_norm.bias",
        ffn_up_weight => "ffn_up.weight",
        ffn_up_bias => "ffn_up.bias",
        ffn_down_weight => "ffn_down.weight",
        ffn_down_bias => "ffn_down.bias",
    }
}

// --- VQAdaptor bridge (4x time-merge is a pure reshape; only these 3 -----
// --- weighted layers exist: Linear -> SiLU -> Linear -> LayerNorm) --------

pub(crate) const ADAPTOR_LINEAR1_WEIGHT: &str = "moss.adaptor.linear1.weight";
pub(crate) const ADAPTOR_LINEAR1_BIAS: &str = "moss.adaptor.linear1.bias";
pub(crate) const ADAPTOR_LINEAR2_WEIGHT: &str = "moss.adaptor.linear2.weight";
pub(crate) const ADAPTOR_LINEAR2_BIAS: &str = "moss.adaptor.linear2.bias";
pub(crate) const ADAPTOR_NORM_WEIGHT: &str = "moss.adaptor.norm.weight";
pub(crate) const ADAPTOR_NORM_BIAS: &str = "moss.adaptor.norm.bias";

// --- Qwen3-0.6B decoder (tied embeddings: no separate lm_head tensor) -----

pub(crate) const LLM_TOKEN_EMBD_WEIGHT: &str = "moss.llm.tok_embd.weight";
pub(crate) const LLM_OUTPUT_NORM_WEIGHT: &str = "moss.llm.out_norm.weight";

layer_tensor_names! {
    pub(crate) struct MossLlmLayerTensorNames;
    pub(crate) fn moss_llm_layer_tensor_names @ "moss.llm.blk";
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_layer_tensor_names_match_runtime_convention() {
        let names = moss_encoder_layer_tensor_names(3);
        assert_eq!(names.attn_q_weight, "moss.enc.blk.3.attn_q.weight");
        assert_eq!(names.ffn_down_bias, "moss.enc.blk.3.ffn_down.bias");
    }

    #[test]
    fn llm_layer_tensor_names_match_runtime_convention() {
        let names = moss_llm_layer_tensor_names(5);
        assert_eq!(
            names.attn_q_norm_weight,
            "moss.llm.blk.5.attn_q_norm.weight"
        );
        assert_eq!(names.ffn_gate_weight, "moss.llm.blk.5.ffn_gate.weight");
    }
}
