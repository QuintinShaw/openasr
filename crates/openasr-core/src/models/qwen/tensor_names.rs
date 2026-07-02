use crate::models::tensor_schema::layer_tensor_names;

pub(crate) const AUDIO_MEL_FILTERS: &str = "audio.mel_filters";
pub(crate) const AUDIO_MEL_WINDOW: &str = "audio.mel_window";
pub(crate) const AUDIO_CONV1_WEIGHT: &str = "audio.conv.1.weight";
pub(crate) const AUDIO_CONV1_BIAS: &str = "audio.conv.1.bias";
pub(crate) const AUDIO_CONV2_WEIGHT: &str = "audio.conv.2.weight";
pub(crate) const AUDIO_CONV2_BIAS: &str = "audio.conv.2.bias";
pub(crate) const AUDIO_CONV3_WEIGHT: &str = "audio.conv.3.weight";
pub(crate) const AUDIO_CONV3_BIAS: &str = "audio.conv.3.bias";
pub(crate) const AUDIO_CONV_OUT_WEIGHT: &str = "audio.conv_out.weight";
pub(crate) const AUDIO_CONV_OUT_BIAS: &str = "audio.conv_out.bias";
pub(crate) const AUDIO_LN_POST_WEIGHT: &str = "audio.ln_post.weight";
pub(crate) const AUDIO_LN_POST_BIAS: &str = "audio.ln_post.bias";
pub(crate) const AUDIO_PROJ1_WEIGHT: &str = "audio.proj1.weight";
pub(crate) const AUDIO_PROJ1_BIAS: &str = "audio.proj1.bias";
pub(crate) const AUDIO_PROJ2_WEIGHT: &str = "audio.proj2.weight";
pub(crate) const AUDIO_PROJ2_BIAS: &str = "audio.proj2.bias";
pub(crate) const TOKEN_EMBD_WEIGHT: &str = "token_embd.weight";
pub(crate) const OUTPUT_WEIGHT: &str = "output.weight";
pub(crate) const OUTPUT_NORM_WEIGHT: &str = "output_norm.weight";

layer_tensor_names! {
    pub(crate) struct Qwen3AsrAudioLayerTensorNames;
    pub(crate) fn audio_layer_tensor_names @ "audio.blk";
    {
        attn_norm_weight => "attn_norm.weight",
        attn_norm_bias => "attn_norm.bias",
        attn_q_weight => "attn_q.weight",
        attn_q_bias => "attn_q.bias",
        attn_k_weight => "attn_k.weight",
        attn_k_bias => "attn_k.bias",
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

layer_tensor_names! {
    pub(crate) struct Qwen3AsrLlmLayerTensorNames;
    pub(crate) fn llm_layer_tensor_names @ "blk";
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
    fn audio_layer_tensor_names_match_runtime_convention() {
        let names = audio_layer_tensor_names(7);
        assert_eq!(names.attn_q_weight, "audio.blk.7.attn_q.weight");
        assert_eq!(names.ffn_down_bias, "audio.blk.7.ffn_down.bias");
    }

    #[test]
    fn llm_layer_tensor_names_match_runtime_convention() {
        let names = llm_layer_tensor_names(3);
        assert_eq!(names.attn_output_weight, "blk.3.attn_output.weight");
        assert_eq!(names.ffn_gate_weight, "blk.3.ffn_gate.weight");
    }
}
