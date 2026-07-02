use crate::models::tensor_schema::layer_tensor_names;

pub(crate) const FE_MEL_FB: &str = "fe.mel_fb";
pub(crate) const FE_WINDOW: &str = "fe.window";
pub(crate) const ENC_PRE_OUT_WEIGHT: &str = "enc.pre.out.weight";
pub(crate) const ENC_PRE_OUT_BIAS: &str = "enc.pre.out.bias";
pub(crate) const ENC_PROJ_WEIGHT: &str = "enc.proj.weight";
pub(crate) const ENC_PROJ_BIAS: &str = "enc.proj.bias";
pub(crate) const DEC_EMB_WEIGHT: &str = "dec.emb.weight";
pub(crate) const DEC_POS_WEIGHT: &str = "dec.pos.weight";
pub(crate) const DEC_EMB_LN_WEIGHT: &str = "dec.emb_ln.weight";
pub(crate) const DEC_EMB_LN_BIAS: &str = "dec.emb_ln.bias";
pub(crate) const DEC_OUT_LN_WEIGHT: &str = "dec.out_ln.weight";
pub(crate) const DEC_OUT_LN_BIAS: &str = "dec.out_ln.bias";
pub(crate) const DEC_HEAD_WEIGHT: &str = "dec.head.weight";
pub(crate) const DEC_HEAD_BIAS: &str = "dec.head.bias";

pub(crate) fn enc_pre_conv_weight(stage_idx: usize) -> String {
    format!("enc.pre.conv.{stage_idx}.weight")
}

pub(crate) fn enc_pre_conv_bias(stage_idx: usize) -> String {
    format!("enc.pre.conv.{stage_idx}.bias")
}

layer_tensor_names! {
    pub(crate) struct CohereEncoderLayerTensorNames;
    pub(crate) fn encoder_layer_tensor_names @ "enc.blk";
    {
        ff1_norm_weight => "ff1.norm.weight",
        ff1_norm_bias => "ff1.norm.bias",
        ff1_up_weight => "ff1.up.weight",
        ff1_up_bias => "ff1.up.bias",
        ff1_down_weight => "ff1.down.weight",
        ff1_down_bias => "ff1.down.bias",
        attn_norm_weight => "attn.norm.weight",
        attn_norm_bias => "attn.norm.bias",
        attn_q_weight => "attn.q.weight",
        attn_q_bias => "attn.q.bias",
        attn_k_weight => "attn.k.weight",
        attn_k_bias => "attn.k.bias",
        attn_v_weight => "attn.v.weight",
        attn_v_bias => "attn.v.bias",
        attn_out_weight => "attn.out.weight",
        attn_out_bias => "attn.out.bias",
        attn_pos_weight => "attn.pos.weight",
        attn_pos_bias_u => "attn.pos_bias_u",
        attn_pos_bias_v => "attn.pos_bias_v",
        conv_norm_weight => "conv.norm.weight",
        conv_norm_bias => "conv.norm.bias",
        conv_pw1_weight => "conv.pw1.weight",
        conv_pw1_bias => "conv.pw1.bias",
        conv_dw_weight => "conv.dw.weight",
        conv_dw_bias => "conv.dw.bias",
        conv_bn_weight => "conv.bn.weight",
        conv_bn_bias => "conv.bn.bias",
        conv_bn_mean => "conv.bn.mean",
        conv_bn_var => "conv.bn.var",
        conv_pw2_weight => "conv.pw2.weight",
        conv_pw2_bias => "conv.pw2.bias",
        ff2_norm_weight => "ff2.norm.weight",
        ff2_norm_bias => "ff2.norm.bias",
        ff2_up_weight => "ff2.up.weight",
        ff2_up_bias => "ff2.up.bias",
        ff2_down_weight => "ff2.down.weight",
        ff2_down_bias => "ff2.down.bias",
        out_norm_weight => "out_norm.weight",
        out_norm_bias => "out_norm.bias",
    }
}

layer_tensor_names! {
    pub(crate) struct CohereDecoderLayerTensorNames;
    pub(crate) fn decoder_layer_tensor_names @ "dec.blk";
    {
        attn_ln_weight => "attn_ln.weight",
        attn_ln_bias => "attn_ln.bias",
        attn_q_weight => "attn_q.weight",
        attn_q_bias => "attn_q.bias",
        attn_k_weight => "attn_k.weight",
        attn_k_bias => "attn_k.bias",
        attn_v_weight => "attn_v.weight",
        attn_v_bias => "attn_v.bias",
        attn_o_weight => "attn_o.weight",
        attn_o_bias => "attn_o.bias",
        cross_ln_weight => "cross_ln.weight",
        cross_ln_bias => "cross_ln.bias",
        cross_q_weight => "cross_q.weight",
        cross_q_bias => "cross_q.bias",
        cross_k_weight => "cross_k.weight",
        cross_k_bias => "cross_k.bias",
        cross_v_weight => "cross_v.weight",
        cross_v_bias => "cross_v.bias",
        cross_o_weight => "cross_o.weight",
        cross_o_bias => "cross_o.bias",
        ffn_ln_weight => "ffn_ln.weight",
        ffn_ln_bias => "ffn_ln.bias",
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
    fn encoder_layer_tensor_names_match_runtime_convention() {
        let names = encoder_layer_tensor_names(4);
        assert_eq!(names.attn_q_weight, "enc.blk.4.attn.q.weight");
        assert_eq!(names.conv_bn_var, "enc.blk.4.conv.bn.var");
    }

    #[test]
    fn decoder_layer_tensor_names_match_runtime_convention() {
        let names = decoder_layer_tensor_names(2);
        assert_eq!(names.cross_o_bias, "dec.blk.2.cross_o.bias");
        assert_eq!(names.ffn_down_weight, "dec.blk.2.ffn_down.weight");
    }
}
