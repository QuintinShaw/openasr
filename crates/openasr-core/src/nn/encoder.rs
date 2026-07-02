//! Encoder layer blocks for the shared `nn/` IR boundary.
//!
//! `conformer_block` is the reusable, config-driven Conformer encoder layer:
//! macaron feed-forward halves bracketing relative-position self-attention and a
//! GLU + depthwise convolution module, with a final affine layer norm. It is a
//! faithful extraction of cohere-transcribe's hand-written per-layer graph —
//! the same ggml op sequence, step labels, f32 view strides, selective
//! contiguity, macaron 0.5 scaling, and `1/sqrt(head_dim)` attention scale — so
//! a migrated family stays bit-identical while the composition becomes shared.
//!
//! Like every `nn/` builder, the block is generic over the caller's error `E`
//! via a `map_err` closure and never owns a model-specific error type. The
//! relative-shift byte strides are passed in (the caller keeps its
//! overflow-checked arithmetic), keeping the block free of fallible pre-math.

use crate::ggml_runtime::{GgmlCpuGraphBuilder, GgmlCpuGraphError, GgmlCpuTensor};
use crate::nn::attn::{
    AttentionHeadLayout, AttentionReshapeSteps, AttentionValueMergeSteps,
    STANDARD_HEAD_PERMUTE_AXES, attention_context_from_probs,
    reshape_projection_to_attention_heads,
};
use crate::nn::ffn::{
    FeedForwardActivation, FeedForwardResidualSteps, apply_feed_forward_residual,
};
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

/// Scalar/shape knobs for one Conformer encoder block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ConformerBlockConfig {
    pub d_model: usize,
    pub attention_heads: usize,
    pub head_dim: usize,
    /// Subsampled encoder sequence length (frames).
    pub frame_count: usize,
    pub conv_kernel: usize,
    pub layer_norm_epsilon: f32,
    /// Macaron feed-forward output scale (Conformer uses 0.5).
    pub macaron_scale: f32,
    /// Relative-shift view strides/offset in bytes, precomputed by the caller
    /// (which keeps overflow-checked arithmetic and its own error type):
    /// `nb1 = (2*frame_count-2)*4`, `nb2 = (2*frame_count-1)*frame_count*4`,
    /// `offset = (frame_count-1)*4`.
    pub rel_shift_nb1: usize,
    pub rel_shift_nb2: usize,
    pub rel_shift_offset: usize,
}

/// Per-block graph tensors, in submodule order: FF1 (macaron) → rel-pos
/// attention → conv (GLU + depthwise) → FF2 (macaron) → final norm. `_weight`
/// is a projection matrix (caller uploads it transposed for `mul_mat`); biases
/// are f32 vectors; `attn_pos_bias_u/v` are 1-D `[d_model]` and reshaped in
/// graph; `conv_dw_weight` is f16 post BN-fold (the caller's responsibility).
#[derive(Clone, Copy)]
pub(crate) struct ConformerBlockWeights<'a> {
    pub ff1_norm_weight: GgmlCpuTensor<'a>,
    pub ff1_norm_bias: GgmlCpuTensor<'a>,
    pub ff1_up_weight: GgmlCpuTensor<'a>,
    pub ff1_up_bias: GgmlCpuTensor<'a>,
    pub ff1_down_weight: GgmlCpuTensor<'a>,
    pub ff1_down_bias: GgmlCpuTensor<'a>,
    pub attn_norm_weight: GgmlCpuTensor<'a>,
    pub attn_norm_bias: GgmlCpuTensor<'a>,
    pub attn_q_weight: GgmlCpuTensor<'a>,
    pub attn_q_bias: GgmlCpuTensor<'a>,
    pub attn_k_weight: GgmlCpuTensor<'a>,
    pub attn_k_bias: GgmlCpuTensor<'a>,
    pub attn_v_weight: GgmlCpuTensor<'a>,
    pub attn_v_bias: GgmlCpuTensor<'a>,
    pub attn_out_weight: GgmlCpuTensor<'a>,
    pub attn_out_bias: GgmlCpuTensor<'a>,
    pub attn_pos_weight: GgmlCpuTensor<'a>,
    pub attn_pos_bias_u: GgmlCpuTensor<'a>,
    pub attn_pos_bias_v: GgmlCpuTensor<'a>,
    pub conv_norm_weight: GgmlCpuTensor<'a>,
    pub conv_norm_bias: GgmlCpuTensor<'a>,
    pub conv_pw1_weight: GgmlCpuTensor<'a>,
    pub conv_pw1_bias: GgmlCpuTensor<'a>,
    pub conv_dw_weight: GgmlCpuTensor<'a>,
    pub conv_dw_bias: GgmlCpuTensor<'a>,
    pub conv_pw2_weight: GgmlCpuTensor<'a>,
    pub conv_pw2_bias: GgmlCpuTensor<'a>,
    pub ff2_norm_weight: GgmlCpuTensor<'a>,
    pub ff2_norm_bias: GgmlCpuTensor<'a>,
    pub ff2_up_weight: GgmlCpuTensor<'a>,
    pub ff2_up_bias: GgmlCpuTensor<'a>,
    pub ff2_down_weight: GgmlCpuTensor<'a>,
    pub ff2_down_bias: GgmlCpuTensor<'a>,
    pub out_norm_weight: GgmlCpuTensor<'a>,
    pub out_norm_bias: GgmlCpuTensor<'a>,
}

/// Residual-boundary tensors for optional debug capture, matching the six taps
/// cohere's hand-written layer exposed. The caller decides whether to read them.
#[derive(Clone, Copy)]
pub(crate) struct ConformerBlockTaps<'a> {
    pub ff1: GgmlCpuTensor<'a>,
    pub attn: GgmlCpuTensor<'a>,
    pub conv_glu: GgmlCpuTensor<'a>,
    pub conv_dw_act: GgmlCpuTensor<'a>,
    pub conv: GgmlCpuTensor<'a>,
    pub ff2: GgmlCpuTensor<'a>,
}

pub(crate) struct ConformerBlockOutput<'a> {
    pub output: GgmlCpuTensor<'a>,
    pub taps: ConformerBlockTaps<'a>,
}

/// Assemble one Conformer encoder block, reproducing cohere's hand-written op
/// sequence bit-identically. `pos_enc` is the per-encoder relative-position
/// encoding (uploaded once, shared across layers).
#[allow(clippy::too_many_lines)]
pub(crate) fn conformer_block<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    pos_enc: GgmlCpuTensor<'a>,
    config: ConformerBlockConfig,
    weights: ConformerBlockWeights<'a>,
    map_err: F,
) -> Result<ConformerBlockOutput<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let d_model = config.d_model;
    let heads = config.attention_heads;
    let head_dim = config.head_dim;
    let frame_count = config.frame_count;
    let conv_kernel = config.conv_kernel;

    // ----- FF1: macaron feed-forward half (pre-norm + residual, scale 0.5) -----
    let ff1_norm = apply_affine_layer_norm(
        graph,
        input,
        config.layer_norm_epsilon,
        weights.ff1_norm_weight,
        weights.ff1_norm_bias,
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "ff1_norm",
            bias: "ff1_norm",
        },
        map_err,
    )?;
    let mut state = apply_feed_forward_residual(
        graph,
        ff1_norm,
        input,
        FeedForwardActivation::Silu,
        Some(config.macaron_scale),
        FeedForwardResidualSteps {
            activation: "ggml_silu(ff1)",
            scale: Some("ggml_scale(ff1_half)"),
            residual: "ggml_add(ff1_residual)",
        },
        |graph, value| {
            let ff1 = graph
                .mul_mat(weights.ff1_up_weight, value)
                .map_err(|source| map_err("ggml_mul_mat(ff1_up)", source))?;
            graph
                .add(ff1, weights.ff1_up_bias)
                .map_err(|source| map_err("ggml_add(ff1_up_bias)", source))
        },
        |graph, value| {
            let ff1 = graph
                .mul_mat(weights.ff1_down_weight, value)
                .map_err(|source| map_err("ggml_mul_mat(ff1_down)", source))?;
            graph
                .add(ff1, weights.ff1_down_bias)
                .map_err(|source| map_err("ggml_add(ff1_down_bias)", source))
        },
        map_err,
    )?;
    let ff1_tap = state;
    let attn_input = state;

    // ----- Self-attention with relative positional encoding -----
    let attn_norm = apply_affine_layer_norm(
        graph,
        state,
        config.layer_norm_epsilon,
        weights.attn_norm_weight,
        weights.attn_norm_bias,
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "attn_norm",
            bias: "attn_norm",
        },
        map_err,
    )?;
    let mut q = graph
        .mul_mat(weights.attn_q_weight, attn_norm)
        .map_err(|source| map_err("ggml_mul_mat(attn_q)", source))?;
    q = graph
        .add(q, weights.attn_q_bias)
        .map_err(|source| map_err("ggml_add(attn_q_bias)", source))?;
    let mut k = graph
        .mul_mat(weights.attn_k_weight, attn_norm)
        .map_err(|source| map_err("ggml_mul_mat(attn_k)", source))?;
    k = graph
        .add(k, weights.attn_k_bias)
        .map_err(|source| map_err("ggml_add(attn_k_bias)", source))?;
    let v = graph
        .mul_mat(weights.attn_v_weight, attn_norm)
        .map_err(|source| map_err("ggml_mul_mat(attn_v)", source))?;
    let v = graph
        .add(v, weights.attn_v_bias)
        .map_err(|source| map_err("ggml_add(attn_v_bias)", source))?;
    let r = graph
        .mul_mat(weights.attn_pos_weight, pos_enc)
        .map_err(|source| map_err("ggml_mul_mat(attn_pos)", source))?;
    let q_u = graph
        .add(
            q,
            graph
                .reshape_1d(weights.attn_pos_bias_u, d_model)
                .map_err(|source| map_err("ggml_reshape_1d(attn_pos_bias_u)", source))?,
        )
        .map_err(|source| map_err("ggml_add(attn_pos_bias_u)", source))?;
    let q_v = graph
        .add(
            q,
            graph
                .reshape_1d(weights.attn_pos_bias_v, d_model)
                .map_err(|source| map_err("ggml_reshape_1d(attn_pos_bias_v)", source))?,
        )
        .map_err(|source| map_err("ggml_add(attn_pos_bias_v)", source))?;
    let attention_layout = AttentionHeadLayout {
        head_dim,
        attention_heads: heads,
        sequence_len: frame_count,
    };
    let q_u = reshape_projection_to_attention_heads(
        graph,
        q_u,
        attention_layout,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(attn_q_u)",
            permute: "ggml_permute(attn_q_u)",
            cont: "ggml_cont(attn_q_u)",
        },
        map_err,
    )?;
    let q_v = reshape_projection_to_attention_heads(
        graph,
        q_v,
        attention_layout,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(attn_q_v)",
            permute: "ggml_permute(attn_q_v)",
            cont: "ggml_cont(attn_q_v)",
        },
        map_err,
    )?;
    let k = reshape_projection_to_attention_heads(
        graph,
        k,
        attention_layout,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(attn_k)",
            permute: "ggml_permute(attn_k)",
            cont: "ggml_cont(attn_k)",
        },
        map_err,
    )?;
    let r = reshape_projection_to_attention_heads(
        graph,
        r,
        AttentionHeadLayout {
            sequence_len: 2 * frame_count - 1,
            ..attention_layout
        },
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(attn_r)",
            permute: "ggml_permute(attn_r)",
            cont: "ggml_cont(attn_r)",
        },
        map_err,
    )?;
    let ac = graph
        .mul_mat(
            graph
                .cont(k)
                .map_err(|source| map_err("ggml_cont(attn_k)", source))?,
            q_u,
        )
        .map_err(|source| map_err("ggml_mul_mat(attn_ac)", source))?;
    let bd_raw = graph
        .mul_mat(
            graph
                .cont(r)
                .map_err(|source| map_err("ggml_cont(attn_r)", source))?,
            q_v,
        )
        .map_err(|source| map_err("ggml_mul_mat(attn_bd_raw)", source))?;
    let bd = graph
        .view_3d(
            bd_raw,
            frame_count,
            frame_count,
            heads,
            config.rel_shift_nb1,
            config.rel_shift_nb2,
            config.rel_shift_offset,
        )
        .map_err(|source| map_err("ggml_view_3d(relative_shift)", source))?;
    let mut scores = graph
        .add(ac, bd)
        .map_err(|source| map_err("ggml_add(attn_scores)", source))?;
    scores = graph
        .scale(scores, 1.0 / (head_dim as f32).sqrt())
        .map_err(|source| map_err("ggml_scale(attn_scores)", source))?;
    let scores = graph
        .soft_max(scores)
        .map_err(|source| map_err("ggml_soft_max(attn_scores)", source))?;
    let v_heads = reshape_projection_to_attention_heads(
        graph,
        v,
        attention_layout,
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(attn_v)",
            permute: "ggml_permute(attn_v)",
            cont: "ggml_cont(attn_v)",
        },
        map_err,
    )?;
    let mut attn_out = attention_context_from_probs(
        graph,
        v_heads,
        scores,
        attention_layout,
        AttentionValueMergeSteps {
            value_permute: "ggml_permute(attn_v_t)",
            value_cont: "ggml_cont(attn_v_t)",
            context_mul: "ggml_mul_mat(attn_ctx)",
            context_merge_permute: "ggml_permute(attn_ctx_merge)",
            context_merge_cont: "ggml_cont(attn_ctx_merge)",
            context_merge_reshape: "ggml_reshape_2d(attn_ctx_merge)",
        },
        map_err,
    )?;
    attn_out = graph
        .mul_mat(weights.attn_out_weight, attn_out)
        .map_err(|source| map_err("ggml_mul_mat(attn_out)", source))?;
    attn_out = graph
        .add(attn_out, weights.attn_out_bias)
        .map_err(|source| map_err("ggml_add(attn_out_bias)", source))?;
    state = graph
        .add(attn_input, attn_out)
        .map_err(|source| map_err("ggml_add(attn_residual)", source))?;
    let attn_tap = state;
    let conv_input = state;

    // ----- Conv module: pre-norm → pw1 GLU → depthwise → pw2 → residual -----
    let conv_norm = apply_affine_layer_norm(
        graph,
        state,
        config.layer_norm_epsilon,
        weights.conv_norm_weight,
        weights.conv_norm_bias,
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "conv_norm",
            bias: "conv_norm",
        },
        map_err,
    )?;
    let pw1 = graph
        .reshape_2d(weights.conv_pw1_weight, d_model, d_model * 2)
        .map_err(|source| map_err("ggml_reshape_2d(conv_pw1_weight)", source))?;
    let mut conv = graph
        .mul_mat(pw1, conv_norm)
        .map_err(|source| map_err("ggml_mul_mat(conv_pw1)", source))?;
    conv = graph
        .add(conv, weights.conv_pw1_bias)
        .map_err(|source| map_err("ggml_add(conv_pw1_bias)", source))?;
    let conv_view_row_stride = d_model * std::mem::size_of::<f32>();
    let conv_plane_stride = (d_model * 2) * std::mem::size_of::<f32>();
    let conv_main = graph
        .view_3d(
            conv,
            d_model,
            1,
            frame_count,
            conv_view_row_stride,
            conv_plane_stride,
            0,
        )
        .map_err(|source| map_err("ggml_view_3d(conv_main)", source))?;
    let conv_gate = graph
        .view_3d(
            conv,
            d_model,
            1,
            frame_count,
            conv_view_row_stride,
            conv_plane_stride,
            d_model * std::mem::size_of::<f32>(),
        )
        .map_err(|source| map_err("ggml_view_3d(conv_gate)", source))?;
    let conv_main = graph
        .cont(conv_main)
        .map_err(|source| map_err("ggml_cont(conv_main)", source))?;
    let conv_gate = graph
        .cont(conv_gate)
        .map_err(|source| map_err("ggml_cont(conv_gate)", source))?;
    let conv_main = graph
        .reshape_2d(conv_main, d_model, frame_count)
        .map_err(|source| map_err("ggml_reshape_2d(conv_main)", source))?;
    let conv_gate = graph
        .reshape_2d(conv_gate, d_model, frame_count)
        .map_err(|source| map_err("ggml_reshape_2d(conv_gate)", source))?;
    conv = graph
        .mul(
            conv_main,
            graph
                .sigmoid(conv_gate)
                .map_err(|source| map_err("ggml_sigmoid(conv_gate)", source))?,
        )
        .map_err(|source| map_err("ggml_mul(conv_glu)", source))?;
    let conv_glu_tap = conv;
    let dw_weight = graph
        .reshape_4d(weights.conv_dw_weight, conv_kernel, 1, 1, d_model)
        .map_err(|source| map_err("ggml_reshape_4d(conv_dw_weight)", source))?;
    conv = graph
        .transpose(conv)
        .map_err(|source| map_err("ggml_transpose(conv_glu)", source))?;
    conv = graph
        .cont(conv)
        .map_err(|source| map_err("ggml_cont(conv_glu_t)", source))?;
    let conv_4d = graph
        .reshape_4d(conv, frame_count, 1, d_model, 1)
        .map_err(|source| map_err("ggml_reshape_4d(conv_in_4d)", source))?;
    let mut conv = graph
        .conv_2d_dw(dw_weight, conv_4d, 1, 1, (conv_kernel - 1) / 2, 0, 1, 1)
        .map_err(|source| map_err("ggml_conv_2d_dw(conv_dw)", source))?;
    conv = graph
        .permute(conv, 1, 2, 0, 3)
        .map_err(|source| map_err("ggml_permute(conv_dw_out)", source))?;
    conv = graph
        .cont(conv)
        .map_err(|source| map_err("ggml_cont(conv_dw_out)", source))?;
    let conv_dw_bias = graph
        .reshape_4d(weights.conv_dw_bias, d_model, 1, 1, 1)
        .map_err(|source| map_err("ggml_reshape_4d(conv_dw_bias)", source))?;
    let pw2 = graph
        .reshape_2d(weights.conv_pw2_weight, d_model, d_model)
        .map_err(|source| map_err("ggml_reshape_2d(conv_pw2_weight)", source))?;
    conv = graph
        .add(conv, conv_dw_bias)
        .map_err(|source| map_err("ggml_add(conv_dw_bias)", source))?;
    conv = graph
        .silu(conv)
        .map_err(|source| map_err("ggml_silu(conv_dw)", source))?;
    let conv_dw_act_tap = conv;
    conv = graph
        .mul_mat(pw2, conv)
        .map_err(|source| map_err("ggml_mul_mat(conv_pw2)", source))?;
    conv = graph
        .add(conv, weights.conv_pw2_bias)
        .map_err(|source| map_err("ggml_add(conv_pw2_bias)", source))?;
    state = graph
        .add(conv_input, conv)
        .map_err(|source| map_err("ggml_add(conv_residual)", source))?;
    let conv_tap = state;
    let ff2_input = state;

    // ----- FF2: macaron feed-forward half -----
    let ff2_norm = apply_affine_layer_norm(
        graph,
        state,
        config.layer_norm_epsilon,
        weights.ff2_norm_weight,
        weights.ff2_norm_bias,
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "ff2_norm",
            bias: "ff2_norm",
        },
        map_err,
    )?;
    state = apply_feed_forward_residual(
        graph,
        ff2_norm,
        ff2_input,
        FeedForwardActivation::Silu,
        Some(config.macaron_scale),
        FeedForwardResidualSteps {
            activation: "ggml_silu(ff2)",
            scale: Some("ggml_scale(ff2_half)"),
            residual: "ggml_add(ff2_residual)",
        },
        |graph, value| {
            let ff2 = graph
                .mul_mat(weights.ff2_up_weight, value)
                .map_err(|source| map_err("ggml_mul_mat(ff2_up)", source))?;
            graph
                .add(ff2, weights.ff2_up_bias)
                .map_err(|source| map_err("ggml_add(ff2_up_bias)", source))
        },
        |graph, value| {
            let ff2 = graph
                .mul_mat(weights.ff2_down_weight, value)
                .map_err(|source| map_err("ggml_mul_mat(ff2_down)", source))?;
            graph
                .add(ff2, weights.ff2_down_bias)
                .map_err(|source| map_err("ggml_add(ff2_down_bias)", source))
        },
        map_err,
    )?;
    let ff2_tap = state;

    // ----- Final affine layer norm (no residual) -----
    let output = apply_affine_layer_norm(
        graph,
        state,
        config.layer_norm_epsilon,
        weights.out_norm_weight,
        weights.out_norm_bias,
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "out_norm",
            bias: "out_norm",
        },
        map_err,
    )?;

    Ok(ConformerBlockOutput {
        output,
        taps: ConformerBlockTaps {
            ff1: ff1_tap,
            attn: attn_tap,
            conv_glu: conv_glu_tap,
            conv_dw_act: conv_dw_act_tap,
            conv: conv_tap,
            ff2: ff2_tap,
        },
    })
}

/// Scalar/shape knobs for one standard pre-norm Transformer encoder block
/// (masked scaled-dot self-attention + a single biased FFN — the Whisper /
/// Qwen-audio encoder shape).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TransformerEncoderConfig {
    pub head_dim: usize,
    pub attention_heads: usize,
    pub token_count: usize,
    pub layer_norm_epsilon: f32,
    pub ffn_activation: FeedForwardActivation,
}

/// Per-block graph tensors: attn (norm, q/k/v/out + biases) → ffn (norm,
/// up/down + biases). Standard transformer encoder layout, all projections
/// biased.
#[derive(Clone, Copy)]
pub(crate) struct TransformerEncoderLayerWeights<'a> {
    pub attn_norm_weight: GgmlCpuTensor<'a>,
    pub attn_norm_bias: GgmlCpuTensor<'a>,
    pub attn_q_weight: GgmlCpuTensor<'a>,
    pub attn_q_bias: GgmlCpuTensor<'a>,
    pub attn_k_weight: GgmlCpuTensor<'a>,
    pub attn_k_bias: GgmlCpuTensor<'a>,
    pub attn_v_weight: GgmlCpuTensor<'a>,
    pub attn_v_bias: GgmlCpuTensor<'a>,
    pub attn_out_weight: GgmlCpuTensor<'a>,
    pub attn_out_bias: GgmlCpuTensor<'a>,
    pub ffn_norm_weight: GgmlCpuTensor<'a>,
    pub ffn_norm_bias: GgmlCpuTensor<'a>,
    pub ffn_up_weight: GgmlCpuTensor<'a>,
    pub ffn_up_bias: GgmlCpuTensor<'a>,
    pub ffn_down_weight: GgmlCpuTensor<'a>,
    pub ffn_down_bias: GgmlCpuTensor<'a>,
}

fn linear_with_bias<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    input: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    step: &'static str,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let projected = graph
        .mul_mat(weight, input)
        .map_err(|source| map_err(step, source))?;
    graph
        .add(projected, bias)
        .map_err(|source| map_err(step, source))
}

/// Assemble one standard pre-norm Transformer encoder block, reproducing the
/// hand-written op sequence bit-identically. `mask` is the additive attention
/// mask added to the raw scores before the scaled softmax.
#[allow(clippy::too_many_lines)]
pub(crate) fn transformer_layer<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    mask: GgmlCpuTensor<'a>,
    config: TransformerEncoderConfig,
    weights: TransformerEncoderLayerWeights<'a>,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let layout = AttentionHeadLayout {
        head_dim: config.head_dim,
        attention_heads: config.attention_heads,
        sequence_len: config.token_count,
    };

    // ----- Self-attention (pre-norm + residual) -----
    let residual = input;
    let x = apply_affine_layer_norm(
        graph,
        input,
        config.layer_norm_epsilon,
        weights.attn_norm_weight,
        weights.attn_norm_bias,
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "attn_norm",
            bias: "attn_norm",
        },
        map_err,
    )?;
    let q = linear_with_bias(
        graph,
        weights.attn_q_weight,
        x,
        weights.attn_q_bias,
        "attn_q",
        map_err,
    )?;
    let k = linear_with_bias(
        graph,
        weights.attn_k_weight,
        x,
        weights.attn_k_bias,
        "attn_k",
        map_err,
    )?;
    let v = linear_with_bias(
        graph,
        weights.attn_v_weight,
        x,
        weights.attn_v_bias,
        "attn_v",
        map_err,
    )?;
    let q = reshape_projection_to_attention_heads(
        graph,
        q,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(attn_q)",
            permute: "ggml_permute(attn_q)",
            cont: "ggml_cont(attn_q)",
        },
        map_err,
    )?;
    let k = reshape_projection_to_attention_heads(
        graph,
        k,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(attn_k)",
            permute: "ggml_permute(attn_k)",
            cont: "ggml_cont(attn_k)",
        },
        map_err,
    )?;
    let v = reshape_projection_to_attention_heads(
        graph,
        v,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(attn_v)",
            permute: "ggml_permute(attn_v)",
            cont: "ggml_cont(attn_v)",
        },
        map_err,
    )?;
    let scores = graph
        .mul_mat(k, q)
        .map_err(|source| map_err("ggml_mul_mat(attn_scores)", source))?;
    let scores = graph
        .add(scores, mask)
        .map_err(|source| map_err("ggml_add(attn_mask)", source))?;
    let scores = graph
        .cont(scores)
        .map_err(|source| map_err("ggml_cont(attn_scores)", source))?;
    let scores = graph
        .soft_max_ext(scores, None, 1.0 / (config.head_dim as f32).sqrt(), 0.0)
        .map_err(|source| map_err("ggml_soft_max_ext(attn_probs)", source))?;
    let context = attention_context_from_probs(
        graph,
        v,
        scores,
        layout,
        AttentionValueMergeSteps {
            value_permute: "ggml_permute(attn_v_t)",
            value_cont: "ggml_cont(attn_v_t)",
            context_mul: "ggml_mul_mat(attn_context)",
            context_merge_permute: "ggml_permute(attn_merge)",
            context_merge_cont: "ggml_cont(attn_merge)",
            context_merge_reshape: "ggml_reshape_2d(attn_merge)",
        },
        map_err,
    )?;
    let attn_out = linear_with_bias(
        graph,
        weights.attn_out_weight,
        context,
        weights.attn_out_bias,
        "attn_out",
        map_err,
    )?;
    let state = graph
        .add(attn_out, residual)
        .map_err(|source| map_err("ggml_add(attn_residual)", source))?;

    // ----- FFN (pre-norm + residual) -----
    let residual = state;
    let x = apply_affine_layer_norm(
        graph,
        state,
        config.layer_norm_epsilon,
        weights.ffn_norm_weight,
        weights.ffn_norm_bias,
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "ffn_norm",
            bias: "ffn_norm",
        },
        map_err,
    )?;
    apply_feed_forward_residual(
        graph,
        x,
        residual,
        config.ffn_activation,
        None,
        FeedForwardResidualSteps {
            activation: "ggml_act(ffn_up)",
            scale: None,
            residual: "ggml_add(ffn_residual)",
        },
        |graph, value| {
            linear_with_bias(
                graph,
                weights.ffn_up_weight,
                value,
                weights.ffn_up_bias,
                "ffn_up",
                map_err,
            )
        },
        |graph, value| {
            linear_with_bias(
                graph,
                weights.ffn_down_weight,
                value,
                weights.ffn_down_bias,
                "ffn_down",
                map_err,
            )
        },
        map_err,
    )
}
