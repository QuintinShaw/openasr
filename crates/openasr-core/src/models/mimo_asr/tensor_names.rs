//! `.oasr` tensor-name constants for the mimo-asr family (audio tokenizer
//! encoder -> input-local transformer -> 36L Qwen2 backbone), matching
//! `tooling/mimo-asr/convert_mimo_asr.py`'s remap tables and
//! `GGUF_MANIFEST.md` byte-for-byte.

use crate::models::tensor_schema::layer_tensor_names;

pub(crate) const TOKEN_EMBD_WEIGHT: &str = "token_embd.weight";
pub(crate) const OUTPUT_WEIGHT: &str = "output.weight";
pub(crate) const OUTPUT_NORM_WEIGHT: &str = "output_norm.weight";
pub(crate) const SPEECH_GROUP_PROJ_WEIGHT: &str = "speech_group_proj.weight";
pub(crate) const INLOCAL_NORM_WEIGHT: &str = "inlocal.norm.weight";

pub(crate) const AUDIOTOK_CONV1_WEIGHT: &str = "audiotok.conv1.weight";
pub(crate) const AUDIOTOK_CONV1_BIAS: &str = "audiotok.conv1.bias";
pub(crate) const AUDIOTOK_CONV2_WEIGHT: &str = "audiotok.conv2.weight";
pub(crate) const AUDIOTOK_CONV2_BIAS: &str = "audiotok.conv2.bias";
pub(crate) const AUDIOTOK_DOWN_SAMPLE_WEIGHT: &str = "audiotok.down_sample.weight";
pub(crate) const AUDIOTOK_DOWN_SAMPLE_NORM_WEIGHT: &str = "audiotok.down_sample_norm.weight";
pub(crate) const AUDIOTOK_DOWN_SAMPLE_NORM_BIAS: &str = "audiotok.down_sample_norm.bias";
pub(crate) const AUDIOTOK_NORM_WEIGHT: &str = "audiotok.norm.weight";
pub(crate) const AUDIOTOK_NORM_BIAS: &str = "audiotok.norm.bias";
pub(crate) const AUDIOTOK_MEL_FILTERS: &str = "audiotok.mel_filters";
pub(crate) const AUDIOTOK_MEL_WINDOW: &str = "audiotok.mel_window";

/// `speech_embd.{i}.weight` (8 RVQ-codebook embedding tables) and
/// `audiotok.quant.{i}.codebook` (the first 8 packed RVQ codebooks, encode
/// side only) share the same `{prefix}.{i}.{leaf}` shape as the per-layer
/// scheme, so [`crate::models::tensor_schema::indexed_tensor_name`] covers
/// them too.
pub(crate) fn speech_embd_weight_name(channel: usize) -> String {
    crate::models::tensor_schema::indexed_tensor_name("speech_embd", channel, "weight")
}

pub(crate) fn audiotok_codebook_name(level: usize) -> String {
    crate::models::tensor_schema::indexed_tensor_name("audiotok.quant", level, "codebook")
}

// 36L Qwen2 backbone: attention has qkv bias, no QK-norm (the inverse of
// qwen3-asr's own `blk.N.*` shape, reusing the SAME tensor namespace since
// the converter mirrors qwen3-asr's naming convention byte-for-byte).
layer_tensor_names! {
    pub(crate) struct MimoLlmLayerTensorNames;
    pub(crate) fn mimo_llm_layer_tensor_names @ "blk";
    {
        attn_norm_weight => "attn_norm.weight",
        attn_q_weight => "attn_q.weight",
        attn_q_bias => "attn_q.bias",
        attn_k_weight => "attn_k.weight",
        attn_k_bias => "attn_k.bias",
        attn_v_weight => "attn_v.weight",
        attn_v_bias => "attn_v.bias",
        attn_output_weight => "attn_output.weight",
        ffn_norm_weight => "ffn_norm.weight",
        ffn_gate_weight => "ffn_gate.weight",
        ffn_up_weight => "ffn_up.weight",
        ffn_down_weight => "ffn_down.weight",
    }
}

// 6L input-local transformer: same Qwen2 shape (qkv bias, no QK-norm,
// RMSNorm, SwiGLU) at a smaller width (1024 hidden / 64 heads x 16 head_dim),
// applied per 4-frame group (bidirectional, no causal mask).
layer_tensor_names! {
    pub(crate) struct MimoInlocalLayerTensorNames;
    pub(crate) fn mimo_inlocal_layer_tensor_names @ "inlocal.blk";
    {
        attn_norm_weight => "attn_norm.weight",
        attn_q_weight => "attn_q.weight",
        attn_q_bias => "attn_q.bias",
        attn_k_weight => "attn_k.weight",
        attn_k_bias => "attn_k.bias",
        attn_v_weight => "attn_v.weight",
        attn_v_bias => "attn_v.bias",
        attn_output_weight => "attn_output.weight",
        ffn_norm_weight => "ffn_norm.weight",
        ffn_gate_weight => "ffn_gate.weight",
        ffn_up_weight => "ffn_up.weight",
        ffn_down_weight => "ffn_down.weight",
    }
}

// 32L audio-tokenizer encoder: pre-LN, LayerNorm (weight+bias, not RMSNorm),
// plain GELU FFN (fc1/fc2, no gating), ASYMMETRIC attention bias (q/v have
// bias, k does not -- see `audio_tokenizer_graph`'s zero-bias handling).
layer_tensor_names! {
    pub(crate) struct MimoAudiotokLayerTensorNames;
    pub(crate) fn mimo_audiotok_layer_tensor_names @ "audiotok.blk";
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_layer_tensor_names_match_pack_convention() {
        let names = mimo_llm_layer_tensor_names(5);
        assert_eq!(names.attn_q_weight, "blk.5.attn_q.weight");
        assert_eq!(names.attn_q_bias, "blk.5.attn_q.bias");
        assert_eq!(names.ffn_gate_weight, "blk.5.ffn_gate.weight");
    }

    #[test]
    fn inlocal_layer_tensor_names_match_pack_convention() {
        let names = mimo_inlocal_layer_tensor_names(2);
        assert_eq!(names.attn_output_weight, "inlocal.blk.2.attn_output.weight");
    }

    #[test]
    fn audiotok_layer_tensor_names_match_pack_convention() {
        let names = mimo_audiotok_layer_tensor_names(9);
        assert_eq!(names.attn_q_bias, "audiotok.blk.9.attn_q.bias");
        assert_eq!(names.attn_out_weight, "audiotok.blk.9.attn_out.weight");
        assert_eq!(names.ffn_up_weight, "audiotok.blk.9.ffn_up.weight");
    }

    #[test]
    fn speech_embd_and_codebook_names_match_pack_convention() {
        assert_eq!(speech_embd_weight_name(0), "speech_embd.0.weight");
        assert_eq!(speech_embd_weight_name(7), "speech_embd.7.weight");
        assert_eq!(audiotok_codebook_name(3), "audiotok.quant.3.codebook");
    }
}
