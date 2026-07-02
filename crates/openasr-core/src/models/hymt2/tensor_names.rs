use crate::models::tensor_schema::layer_tensor_names;

pub(crate) const TOKEN_EMBD_WEIGHT: &str = "token_embd.weight";
pub(crate) const OUTPUT_NORM_WEIGHT: &str = "output_norm.weight";

layer_tensor_names! {
    pub(crate) struct Hymt2LlmLayerTensorNames;
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
    fn llm_layer_tensor_names_match_hunyuan_dense_gguf() {
        let names = llm_layer_tensor_names(3);
        assert_eq!(names.attn_output_weight, "blk.3.attn_output.weight");
        assert_eq!(names.ffn_gate_weight, "blk.3.ffn_gate.weight");
    }
}
