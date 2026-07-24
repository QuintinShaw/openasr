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
        .depthwise_conv_2d(dw_weight, conv_4d, 1, 1, (conv_kernel - 1) / 2, 0, 1, 1)
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

/// Conformer Transformer-XL relative-position sinusoidal table: the per-frame
/// `[2*frame_count-1, d_model]` sin/cos table that `conformer_block`'s
/// `rel_shift` view consumes. Pure math, no model-specific state -- generic
/// over the caller's error `E` (via `overflow_err`) like every other `nn/`
/// builder, so no family's error type leaks into another's.
pub(crate) fn build_relative_positional_encoding<E>(
    d_model: usize,
    frame_count: usize,
    overflow_err: impl Fn() -> E,
) -> Result<Vec<f32>, E> {
    let n_positions = frame_count
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(&overflow_err)?;
    let total = n_positions.checked_mul(d_model).ok_or_else(overflow_err)?;
    let mut values = vec![0.0_f32; total];
    for position_idx in 0..n_positions {
        let pos = (frame_count - 1) as f32 - position_idx as f32;
        for j in 0..(d_model / 2) {
            let div = 10000.0_f32.powf((2.0 * j as f32) / d_model as f32);
            let base = position_idx * d_model + 2 * j;
            values[base] = (pos / div).sin();
            if base + 1 < values.len() {
                values[base + 1] = (pos / div).cos();
            }
        }
    }
    Ok(values)
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
    /// Route self-attention through `ggml_flash_attn_ext` instead of the
    /// naive `mul_mat -> add(mask) -> soft_max_ext` sequence. Mirrors the
    /// Whisper encoder's `use_flash_attention` convention (see
    /// `models/whisper/ggml_executor.rs`): fused softmax/scale, no materialized
    /// `[seq, seq]` scores tensor. Callers building the additive `mask` passed
    /// to `transformer_layer` keep it f32; the flash branch casts its own
    /// F16 copy since `flash_attn_ext` requires an F16 contiguous mask.
    pub use_flash_attention: bool,
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
    // Flash-attn on a GPU-class backend consumes strided (permuted, not
    // `cont`'d) q/k/v views directly -- mirrors the Whisper encoder's
    // `reshape_encoder_projection_to_heads_for_flash` GPU/CPU split in
    // `models/whisper/ggml_executor.rs`. The naive path (and flash on CPU)
    // keeps the original `cont`'d heads unchanged.
    let use_strided_views = config.use_flash_attention && graph.backend_kind().is_gpu_class();
    let heads_contiguous = !use_strided_views;
    let q = reshape_projection_to_attention_heads(
        graph,
        q,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        heads_contiguous,
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
        heads_contiguous,
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
        heads_contiguous,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(attn_v)",
            permute: "ggml_permute(attn_v)",
            cont: "ggml_cont(attn_v)",
        },
        map_err,
    )?;
    let scale = 1.0 / (config.head_dim as f32).sqrt();
    let context = if config.use_flash_attention {
        // `flash_attn_ext` requires an F16 contiguous mask (unlike
        // `soft_max_ext`, which also accepts f32); cast the caller's additive
        // f32 padding mask once per layer. See `nn/decoder.rs`'s shared
        // seq2seq decoder for the same F16-mask convention.
        let flash_mask = graph
            .cast_to_f16(mask)
            .map_err(|source| map_err("ggml_cast(attn_flash_mask)", source))?;
        let flash = graph
            .flash_attn_ext(q, k, v, Some(flash_mask), scale, 0.0, 0.0)
            .map_err(|source| map_err("ggml_flash_attn_ext(attn)", source))?;
        graph
            .reshape_2d(
                flash,
                config.head_dim * config.attention_heads,
                config.token_count,
            )
            .map_err(|source| map_err("ggml_reshape_2d(attn_flash_merge)", source))?
    } else {
        let scores = graph
            .mul_mat(k, q)
            .map_err(|source| map_err("ggml_mul_mat(attn_scores)", source))?;
        let scores = graph
            .add(scores, mask)
            .map_err(|source| map_err("ggml_add(attn_mask)", source))?;
        let scores = graph
            .soft_max_ext(scores, None, scale, 0.0)
            .map_err(|source| map_err("ggml_soft_max_ext(attn_probs)", source))?;
        attention_context_from_probs(
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
        )?
    };
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

/// Scalar/shape knobs for one SenseVoice SAN-M encoder block (self-attention
/// with a DFSMN depthwise-conv memory branch, then a plain ReLU FFN).
///
/// `input_dim` may differ from `d_model` on the stack's first layer (SenseVoice
/// feeds the 560-dim LFR+prompt feature straight into `enc.blk.0`, whose QKV
/// projects 560 -> 3*512); when it differs the attention residual is skipped,
/// matching FunASR's `EncoderLayerSANM` (`in_size != size` => no residual).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SanMFsmnBlockConfig {
    pub d_model: usize,
    pub input_dim: usize,
    pub attention_heads: usize,
    pub head_dim: usize,
    pub frame_count: usize,
    pub fsmn_kernel: usize,
    pub layer_norm_epsilon: f32,
}

/// Weight handles for one SAN-M block. `attn_qkv_weight` is the fused
/// `[input_dim, 3*d_model]` projection; `attn_fsmn_weight` is the depthwise
/// conv kernel `[kernel, 1, d_model]` and MUST be f16 (ggml `conv_2d_dw`
/// requires an f16 kernel, mirroring the conformer depthwise conv).
#[derive(Debug, Clone, Copy)]
pub(crate) struct SanMFsmnBlockWeights<'a> {
    pub attn_norm_weight: GgmlCpuTensor<'a>,
    pub attn_norm_bias: GgmlCpuTensor<'a>,
    pub attn_qkv_weight: GgmlCpuTensor<'a>,
    pub attn_qkv_bias: GgmlCpuTensor<'a>,
    pub attn_out_weight: GgmlCpuTensor<'a>,
    pub attn_out_bias: GgmlCpuTensor<'a>,
    pub attn_fsmn_weight: GgmlCpuTensor<'a>,
    pub ffn_norm_weight: GgmlCpuTensor<'a>,
    pub ffn_norm_bias: GgmlCpuTensor<'a>,
    pub ffn_up_weight: GgmlCpuTensor<'a>,
    pub ffn_up_bias: GgmlCpuTensor<'a>,
    pub ffn_down_weight: GgmlCpuTensor<'a>,
    pub ffn_down_bias: GgmlCpuTensor<'a>,
}

/// One SenseVoice/Paraformer SAN-M encoder block:
///
/// ```text
///   xn  = LN(x)
///   qkv = W_qkv xn + b        (fused; split into q/k/v views)
///   mem = v + depthwise_conv1d(v, kernel, symmetric pad)   [DFSMN branch]
///   att = W_out softmax(q k^T / sqrt(d_k)) v + b_out + mem
///   x   = x + att             (skipped when input_dim != d_model)
///   x   = x + W2 relu(W1 LN(x) + b1) + b2
/// ```
///
/// Bidirectional (no attention mask): SenseVoice is a non-autoregressive CTC
/// encoder over one utterance.
pub(crate) fn sanm_fsmn_encoder_layer<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    config: SanMFsmnBlockConfig,
    weights: SanMFsmnBlockWeights<'a>,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let d_model = config.d_model;
    let frame_count = config.frame_count;
    let element = std::mem::size_of::<f32>();

    // ----- pre-norm + fused QKV -----
    let attn_norm = apply_affine_layer_norm(
        graph,
        input,
        config.layer_norm_epsilon,
        weights.attn_norm_weight,
        weights.attn_norm_bias,
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "sanm_attn_norm",
            bias: "sanm_attn_norm",
        },
        map_err,
    )?;
    let mut qkv = graph
        .mul_mat(weights.attn_qkv_weight, attn_norm)
        .map_err(|source| map_err("ggml_mul_mat(sanm_qkv)", source))?;
    qkv = graph
        .add(qkv, weights.attn_qkv_bias)
        .map_err(|source| map_err("ggml_add(sanm_qkv_bias)", source))?;
    qkv = graph
        .cont(qkv)
        .map_err(|source| map_err("ggml_cont(sanm_qkv)", source))?;
    let qkv_row = 3 * d_model * element;
    let split = |offset_units: usize, step: &'static str| {
        graph
            .view_2d(qkv, d_model, frame_count, qkv_row, offset_units * element)
            .and_then(|view| graph.cont(view))
            .map_err(|source| map_err(step, source))
    };
    let q = split(0, "ggml_view_2d(sanm_q)")?;
    let k = split(d_model, "ggml_view_2d(sanm_k)")?;
    let v = split(2 * d_model, "ggml_view_2d(sanm_v)")?;

    // ----- DFSMN memory branch: v + depthwise conv1d over time -----
    // Same im2col-backed depthwise pattern as the conformer conv module:
    // kernel [kernel,1,d_model] -> 4-D, v [d_model,T] -> [T,1,d_model,1],
    // conv_2d_dw with symmetric (kernel-1)/2 padding (sanm_shift = 0).
    let dw_weight = graph
        .reshape_4d(weights.attn_fsmn_weight, config.fsmn_kernel, 1, 1, d_model)
        .map_err(|source| map_err("ggml_reshape_4d(sanm_fsmn_weight)", source))?;
    let v_t = graph
        .transpose(v)
        .map_err(|source| map_err("ggml_transpose(sanm_fsmn_in)", source))?;
    let v_t = graph
        .cont(v_t)
        .map_err(|source| map_err("ggml_cont(sanm_fsmn_in)", source))?;
    let v_4d = graph
        .reshape_4d(v_t, frame_count, 1, d_model, 1)
        .map_err(|source| map_err("ggml_reshape_4d(sanm_fsmn_in)", source))?;
    let mut fsmn = graph
        .depthwise_conv_2d(dw_weight, v_4d, 1, 1, (config.fsmn_kernel - 1) / 2, 0, 1, 1)
        .map_err(|source| map_err("ggml_conv_2d_dw(sanm_fsmn)", source))?;
    fsmn = graph
        .permute(fsmn, 1, 2, 0, 3)
        .map_err(|source| map_err("ggml_permute(sanm_fsmn_out)", source))?;
    fsmn = graph
        .cont(fsmn)
        .map_err(|source| map_err("ggml_cont(sanm_fsmn_out)", source))?;
    fsmn = graph
        .reshape_2d(fsmn, d_model, frame_count)
        .map_err(|source| map_err("ggml_reshape_2d(sanm_fsmn_out)", source))?;
    let fsmn_mem = graph
        .add(fsmn, v)
        .map_err(|source| map_err("ggml_add(sanm_fsmn_residual)", source))?;

    // ----- bidirectional scaled-dot self-attention -----
    let layout = AttentionHeadLayout {
        head_dim: config.head_dim,
        attention_heads: config.attention_heads,
        sequence_len: frame_count,
    };
    let q_heads = reshape_projection_to_attention_heads(
        graph,
        q,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(sanm_q)",
            permute: "ggml_permute(sanm_q)",
            cont: "ggml_cont(sanm_q)",
        },
        map_err,
    )?;
    let k_heads = reshape_projection_to_attention_heads(
        graph,
        k,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(sanm_k)",
            permute: "ggml_permute(sanm_k)",
            cont: "ggml_cont(sanm_k)",
        },
        map_err,
    )?;
    let k_heads = graph
        .cont(k_heads)
        .map_err(|source| map_err("ggml_cont(sanm_k)", source))?;
    let mut scores = graph
        .mul_mat(k_heads, q_heads)
        .map_err(|source| map_err("ggml_mul_mat(sanm_scores)", source))?;
    scores = graph
        .scale(scores, 1.0 / (config.head_dim as f32).sqrt())
        .map_err(|source| map_err("ggml_scale(sanm_scores)", source))?;
    let probs = graph
        .soft_max(scores)
        .map_err(|source| map_err("ggml_soft_max(sanm_scores)", source))?;
    let v_heads = reshape_projection_to_attention_heads(
        graph,
        v,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(sanm_v)",
            permute: "ggml_permute(sanm_v)",
            cont: "ggml_cont(sanm_v)",
        },
        map_err,
    )?;
    let context = attention_context_from_probs(
        graph,
        v_heads,
        probs,
        layout,
        AttentionValueMergeSteps {
            value_permute: "ggml_permute(sanm_v_t)",
            value_cont: "ggml_cont(sanm_v_t)",
            context_mul: "ggml_mul_mat(sanm_ctx)",
            context_merge_permute: "ggml_permute(sanm_ctx_merge)",
            context_merge_cont: "ggml_cont(sanm_ctx_merge)",
            context_merge_reshape: "ggml_reshape_2d(sanm_ctx_merge)",
        },
        map_err,
    )?;
    let mut attn_out = graph
        .mul_mat(weights.attn_out_weight, context)
        .map_err(|source| map_err("ggml_mul_mat(sanm_attn_out)", source))?;
    attn_out = graph
        .add(attn_out, weights.attn_out_bias)
        .map_err(|source| map_err("ggml_add(sanm_attn_out_bias)", source))?;
    attn_out = graph
        .add(attn_out, fsmn_mem)
        .map_err(|source| map_err("ggml_add(sanm_fsmn_mem)", source))?;
    // FunASR EncoderLayerSANM: the attention residual only applies when the
    // block's input width equals d_model (the 560-dim first layer skips it).
    let state = if config.input_dim == d_model {
        graph
            .add(input, attn_out)
            .map_err(|source| map_err("ggml_add(sanm_attn_residual)", source))?
    } else {
        attn_out
    };

    // ----- position-wise FFN (ReLU) with residual -----
    let ffn_norm = apply_affine_layer_norm(
        graph,
        state,
        config.layer_norm_epsilon,
        weights.ffn_norm_weight,
        weights.ffn_norm_bias,
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "sanm_ffn_norm",
            bias: "sanm_ffn_norm",
        },
        map_err,
    )?;
    apply_feed_forward_residual(
        graph,
        ffn_norm,
        state,
        FeedForwardActivation::Relu,
        None,
        FeedForwardResidualSteps {
            activation: "ggml_relu(sanm_ffn)",
            scale: None,
            residual: "ggml_add(sanm_ffn_residual)",
        },
        |graph, value| {
            let up = graph
                .mul_mat(weights.ffn_up_weight, value)
                .map_err(|source| map_err("ggml_mul_mat(sanm_ffn_up)", source))?;
            graph
                .add(up, weights.ffn_up_bias)
                .map_err(|source| map_err("ggml_add(sanm_ffn_up_bias)", source))
        },
        |graph, value| {
            let down = graph
                .mul_mat(weights.ffn_down_weight, value)
                .map_err(|source| map_err("ggml_mul_mat(sanm_ffn_down)", source))?;
            graph
                .add(down, weights.ffn_down_bias)
                .map_err(|source| map_err("ggml_add(sanm_ffn_down_bias)", source))
        },
        map_err,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphRunner};

    /// Regression pin from the 2026-07-23 firered-aed v2 bisection
    /// (`crates/openasr-core/src/models/firered_aed/encoder_graph.rs`
    /// `parity_tests` doc, evidence chain step 2): this table was one of the
    /// suspects for the v2 encoder residual (a relative-position-encoding
    /// detail specific to the v2 checkpoint's `pe_maxlen`/window/sign
    /// convention), ruled out by comparing the full 549x1280 table this
    /// function produces (T=275) against the official
    /// `RelPositionalEncoding(1280, max_len=5000).forward` for the same T --
    /// max diff 3.0e-5 over 702,720 values, pure fp32 noise. Pinned here as
    /// first-row (relative position T-1=274) / middle-row (relative position
    /// 0) / last-row (relative position -(T-1)=-274) first-8 values so a
    /// future regression in this shared helper doesn't silently reopen that
    /// question for firered-aed (parakeet-ctc/cohere also depend on this
    /// function, but only firered-aed's residual investigation needed this
    /// exact pin).
    #[test]
    fn relative_positional_encoding_matches_python_reference_for_firered_aed_v2_shape() {
        let values = build_relative_positional_encoding(1280, 275, || "overflow").unwrap();
        assert_eq!(values.len(), 549 * 1280);

        let row = |idx: usize| &values[idx * 1280..idx * 1280 + 8];
        let first_row_expected = [
            -0.629_911_4_f32,
            -0.776_667,
            -0.091_786_35,
            0.995_778_74,
            0.723_826_35,
            -0.689_982_24,
            -0.995_081_3,
            0.099_061_38,
        ];
        let middle_row_expected = [0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let last_row_expected = [
            0.629_911_4_f32,
            -0.776_667,
            0.091_786_35,
            0.995_778_74,
            -0.723_826_35,
            -0.689_982_24,
            0.995_081_3,
            0.099_061_38,
        ];

        for (row_idx, expected) in [
            (0_usize, first_row_expected),
            (274, middle_row_expected),
            (548, last_row_expected),
        ] {
            for (col, &expected_value) in expected.iter().enumerate() {
                let actual = row(row_idx)[col];
                let diff = (actual - expected_value).abs();
                assert!(
                    diff < 1e-4,
                    "row {row_idx} col {col}: actual={actual} expected={expected_value} diff={diff}"
                );
            }
        }
    }

    // Mirrors the tiny fixed-size harness `nn/decoder.rs` uses for its shared
    // seq2seq block tests (HIDDEN=4, HEAD_DIM=2, HEADS=2): small enough to hand
    // -verify, big enough to actually exercise 2 attention heads.
    const HIDDEN: usize = 4;
    const HEAD_DIM: usize = 2;
    const HEADS: usize = 2;
    const TOKENS: usize = 3;
    const IDENTITY_4X4: [f32; HIDDEN * HIDDEN] = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];
    const ZERO_4X4: [f32; HIDDEN * HIDDEN] = [0.0; HIDDEN * HIDDEN];
    const NORM_WEIGHT: [f32; HIDDEN] = [1.0; HIDDEN];
    const ZERO_1D: [f32; HIDDEN] = [0.0; HIDDEN];

    fn identity_map_err(step: &'static str, source: GgmlCpuGraphError) -> GgmlCpuGraphError {
        let _ = step;
        source
    }

    /// Build a tiny `transformer_layer` graph (identity QKV/out projections,
    /// zero-bias FFN so it degenerates to a residual passthrough after
    /// attention) and return the computed output. `mask_values` is the raw
    /// f32 additive mask fed to both the naive and flash branches -- flash
    /// casts its own F16 copy internally, so this is the one input the two
    /// branches genuinely share unmodified.
    fn run_transformer_layer(
        backend: GgmlCpuGraphBackend,
        use_flash_attention: bool,
        state_values: &[f32; HIDDEN * TOKENS],
        mask_values: &[f32; TOKENS * TOKENS],
    ) -> Vec<f32> {
        let config = GgmlCpuGraphConfig {
            backend,
            ..GgmlCpuGraphConfig::conservative_default()
        };
        let mut runner = GgmlCpuGraphRunner::new(config).expect("cpu graph runner should init");
        let mut graph = runner.start_graph();

        let input = graph
            .new_tensor_2d_f32(HIDDEN, TOKENS, "input")
            .expect("input tensor");
        let mask = graph
            .new_tensor_2d_f32(TOKENS, TOKENS, "mask")
            .expect("mask tensor");
        let attn_norm_weight = graph
            .new_tensor_1d_f32(HIDDEN, "attn_norm_weight")
            .expect("attn_norm_weight");
        let attn_norm_bias = graph
            .new_tensor_1d_f32(HIDDEN, "attn_norm_bias")
            .expect("attn_norm_bias");
        let attn_q_weight = graph
            .new_tensor_2d_f32(HIDDEN, HIDDEN, "attn_q_weight")
            .expect("attn_q_weight");
        let attn_k_weight = graph
            .new_tensor_2d_f32(HIDDEN, HIDDEN, "attn_k_weight")
            .expect("attn_k_weight");
        let attn_v_weight = graph
            .new_tensor_2d_f32(HIDDEN, HIDDEN, "attn_v_weight")
            .expect("attn_v_weight");
        let attn_out_weight = graph
            .new_tensor_2d_f32(HIDDEN, HIDDEN, "attn_out_weight")
            .expect("attn_out_weight");
        let attn_bias = graph
            .new_tensor_1d_f32(HIDDEN, "attn_bias")
            .expect("attn_bias");
        let ffn_norm_weight = graph
            .new_tensor_1d_f32(HIDDEN, "ffn_norm_weight")
            .expect("ffn_norm_weight");
        let ffn_norm_bias = graph
            .new_tensor_1d_f32(HIDDEN, "ffn_norm_bias")
            .expect("ffn_norm_bias");
        let ffn_up_weight = graph
            .new_tensor_2d_f32(HIDDEN, HIDDEN, "ffn_up_weight")
            .expect("ffn_up_weight");
        let ffn_down_weight = graph
            .new_tensor_2d_f32(HIDDEN, HIDDEN, "ffn_down_weight")
            .expect("ffn_down_weight");
        let ffn_bias = graph
            .new_tensor_1d_f32(HIDDEN, "ffn_bias")
            .expect("ffn_bias");

        for tensor in [
            input,
            mask,
            attn_norm_weight,
            attn_norm_bias,
            attn_q_weight,
            attn_k_weight,
            attn_v_weight,
            attn_out_weight,
            attn_bias,
            ffn_norm_weight,
            ffn_norm_bias,
            ffn_up_weight,
            ffn_down_weight,
            ffn_bias,
        ] {
            graph.set_input(tensor).expect("tensor should be an input");
        }

        let output = transformer_layer(
            &mut graph,
            input,
            mask,
            TransformerEncoderConfig {
                head_dim: HEAD_DIM,
                attention_heads: HEADS,
                token_count: TOKENS,
                layer_norm_epsilon: 1.0e-5,
                ffn_activation: FeedForwardActivation::GeluErf,
                use_flash_attention,
            },
            TransformerEncoderLayerWeights {
                attn_norm_weight,
                attn_norm_bias,
                attn_q_weight,
                attn_q_bias: attn_bias,
                attn_k_weight,
                attn_k_bias: attn_bias,
                attn_v_weight,
                attn_v_bias: attn_bias,
                attn_out_weight,
                attn_out_bias: attn_bias,
                ffn_norm_weight,
                ffn_norm_bias,
                ffn_up_weight,
                ffn_up_bias: ffn_bias,
                ffn_down_weight,
                ffn_down_bias: ffn_bias,
            },
            identity_map_err,
        )
        .expect("transformer_layer should build");
        graph.set_output(output).expect("output should be settable");

        graph
            .set_f32_slice(input, state_values, "input")
            .expect("input upload");
        graph
            .set_f32_slice(mask, mask_values, "mask")
            .expect("mask upload");
        graph
            .set_f32_slice(attn_norm_weight, &NORM_WEIGHT, "attn_norm_weight")
            .expect("attn_norm_weight upload");
        graph
            .set_f32_slice(attn_norm_bias, &ZERO_1D, "attn_norm_bias")
            .expect("attn_norm_bias upload");
        graph
            .set_f32_slice(ffn_norm_weight, &NORM_WEIGHT, "ffn_norm_weight")
            .expect("ffn_norm_weight upload");
        graph
            .set_f32_slice(ffn_norm_bias, &ZERO_1D, "ffn_norm_bias")
            .expect("ffn_norm_bias upload");
        graph
            .set_f32_slice(attn_bias, &ZERO_1D, "attn_bias")
            .expect("attn_bias upload");
        graph
            .set_f32_slice(ffn_bias, &ZERO_1D, "ffn_bias")
            .expect("ffn_bias upload");
        for (tensor, values, name) in [
            (attn_q_weight, &IDENTITY_4X4, "attn_q_weight"),
            (attn_k_weight, &IDENTITY_4X4, "attn_k_weight"),
            (attn_v_weight, &IDENTITY_4X4, "attn_v_weight"),
            (attn_out_weight, &IDENTITY_4X4, "attn_out_weight"),
            // Zero the FFN so the block degenerates to a residual passthrough
            // after attention; this test's job is comparing the ATTENTION
            // branches, not exercising the (config-shared, unmodified) FFN.
            (ffn_up_weight, &ZERO_4X4, "ffn_up_weight"),
            (ffn_down_weight, &ZERO_4X4, "ffn_down_weight"),
        ] {
            graph.set_f32_slice(tensor, values, name).expect(name);
        }

        graph
            .compute_output_f32(output, HIDDEN * TOKENS)
            .expect("transformer_layer graph should compute")
    }

    #[test]
    fn naive_fallback_builds_and_stays_finite_when_flash_disabled() {
        let state = [
            0.4, -0.3, 0.2, 0.1, //
            -0.1, 0.5, -0.2, 0.3, //
            0.2, 0.1, 0.4, -0.4,
        ];
        let zero_mask = [0.0_f32; TOKENS * TOKENS];
        let output = run_transformer_layer(GgmlCpuGraphBackend::Cpu, false, &state, &zero_mask);
        assert_eq!(output.len(), HIDDEN * TOKENS);
        assert!(
            output.iter().all(|value| value.is_finite()),
            "naive attention fallback must produce a finite output: {output:?}"
        );
    }

    /// Flash and naive attention reduce in different float orders and must
    /// never be asserted bit-identical (see AGENTS-level golden-test
    /// guidance); this only checks they land in the same numerical ballpark
    /// for a tiny, hand-checkable case, including one masked (-inf) position
    /// to exercise the f32->F16 flash-mask cast on a non-trivial mask.
    #[test]
    fn flash_attention_matches_naive_within_tolerance_on_cpu() {
        let state = [
            0.4, -0.3, 0.2, 0.1, //
            -0.1, 0.5, -0.2, 0.3, //
            0.2, 0.1, 0.4, -0.4,
        ];
        let mut mask = [0.0_f32; TOKENS * TOKENS];
        // Mask token 2 out of every other token's attention (row-major
        // [query, key]-ish additive mask, matching how the naive branch
        // adds it to `mul_mat(k, q)` before softmax).
        mask[2] = f32::NEG_INFINITY;
        mask[TOKENS + 2] = f32::NEG_INFINITY;

        let naive = run_transformer_layer(GgmlCpuGraphBackend::Cpu, false, &state, &mask);
        let flash = run_transformer_layer(GgmlCpuGraphBackend::Cpu, true, &state, &mask);

        assert_eq!(naive.len(), flash.len());
        for (index, (a, b)) in naive.iter().zip(flash.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1.0e-3,
                "flash vs naive diverge at element {index}: naive={a} flash={b}"
            );
        }
    }

    // ----- SAN-M (SenseVoice/Paraformer) `input_dim == d_model` vs `!=` -----
    //
    // The only real caller (`models::sensevoice::encoder_graph`) exercises the
    // `input_dim != d_model` branch (the 560-dim LFR+prompt feature feeding
    // `enc.blk.0`) only through a bring-up test gated on
    // `#[ignore = "requires SENSEVOICE_BRINGUP_DIR + SENSEVOICE_PACK ..."]`,
    // which needs a local oracle pack and never runs in CI. These tiny
    // synthetic-tensor tests exercise both sides of the residual-skip
    // conditional (`config.input_dim == d_model` above) directly, independent
    // of any real pack.

    const SANM_D_MODEL: usize = HIDDEN;
    const SANM_MISMATCHED_INPUT_DIM: usize = 6;
    const SANM_FSMN_KERNEL: usize = 1;

    /// Build a tiny `sanm_fsmn_encoder_layer` graph with every weight/bias
    /// zeroed except the two LayerNorm scale vectors (kept at 1 so the norm
    /// itself stays well-defined). Every affine transform downstream of the
    /// norms (QKV, attention-out, the FSMN depthwise conv, the FFN) then
    /// multiplies by zero, so the whole attention+FFN contribution collapses
    /// to exactly zero and the block's output reduces to plain `state`: the
    /// original `input` when the residual add fires (`input_dim == d_model`),
    /// or exactly the zero vector when it is skipped (`input_dim !=
    /// d_model`, so `state = attn_out = 0`). That makes both branches of the
    /// conditional hand-verifiable without needing a full identity-attention
    /// reference implementation.
    fn run_sanm_fsmn_encoder_layer_zeroed(input_dim: usize, input_values: &[f32]) -> Vec<f32> {
        assert_eq!(input_values.len(), input_dim * TOKENS);
        let config = GgmlCpuGraphConfig {
            backend: GgmlCpuGraphBackend::Cpu,
            ..GgmlCpuGraphConfig::conservative_default()
        };
        let mut runner = GgmlCpuGraphRunner::new(config).expect("cpu graph runner should init");
        let mut graph = runner.start_graph();

        let input = graph
            .new_tensor_2d_f32(input_dim, TOKENS, "sanm_input")
            .expect("input tensor");
        let attn_norm_weight = graph
            .new_tensor_1d_f32(input_dim, "sanm_attn_norm_weight")
            .expect("attn_norm_weight");
        let attn_norm_bias = graph
            .new_tensor_1d_f32(input_dim, "sanm_attn_norm_bias")
            .expect("attn_norm_bias");
        let attn_qkv_weight = graph
            .new_tensor_2d_f32(input_dim, 3 * SANM_D_MODEL, "sanm_qkv_weight")
            .expect("attn_qkv_weight");
        let attn_qkv_bias = graph
            .new_tensor_1d_f32(3 * SANM_D_MODEL, "sanm_qkv_bias")
            .expect("attn_qkv_bias");
        let attn_out_weight = graph
            .new_tensor_2d_f32(SANM_D_MODEL, SANM_D_MODEL, "sanm_out_weight")
            .expect("attn_out_weight");
        let attn_out_bias = graph
            .new_tensor_1d_f32(SANM_D_MODEL, "sanm_out_bias")
            .expect("attn_out_bias");
        let attn_fsmn_weight = graph
            .new_tensor_3d_f16(SANM_FSMN_KERNEL, 1, SANM_D_MODEL, "sanm_fsmn_weight")
            .expect("attn_fsmn_weight");
        let ffn_norm_weight = graph
            .new_tensor_1d_f32(SANM_D_MODEL, "sanm_ffn_norm_weight")
            .expect("ffn_norm_weight");
        let ffn_norm_bias = graph
            .new_tensor_1d_f32(SANM_D_MODEL, "sanm_ffn_norm_bias")
            .expect("ffn_norm_bias");
        let ffn_up_weight = graph
            .new_tensor_2d_f32(SANM_D_MODEL, SANM_D_MODEL, "sanm_ffn_up_weight")
            .expect("ffn_up_weight");
        let ffn_up_bias = graph
            .new_tensor_1d_f32(SANM_D_MODEL, "sanm_ffn_up_bias")
            .expect("ffn_up_bias");
        let ffn_down_weight = graph
            .new_tensor_2d_f32(SANM_D_MODEL, SANM_D_MODEL, "sanm_ffn_down_weight")
            .expect("ffn_down_weight");
        let ffn_down_bias = graph
            .new_tensor_1d_f32(SANM_D_MODEL, "sanm_ffn_down_bias")
            .expect("ffn_down_bias");

        for tensor in [
            input,
            attn_norm_weight,
            attn_norm_bias,
            attn_qkv_weight,
            attn_qkv_bias,
            attn_out_weight,
            attn_out_bias,
            attn_fsmn_weight,
            ffn_norm_weight,
            ffn_norm_bias,
            ffn_up_weight,
            ffn_up_bias,
            ffn_down_weight,
            ffn_down_bias,
        ] {
            graph.set_input(tensor).expect("tensor should be an input");
        }

        let output = sanm_fsmn_encoder_layer(
            &mut graph,
            input,
            SanMFsmnBlockConfig {
                d_model: SANM_D_MODEL,
                input_dim,
                attention_heads: HEADS,
                head_dim: HEAD_DIM,
                frame_count: TOKENS,
                fsmn_kernel: SANM_FSMN_KERNEL,
                layer_norm_epsilon: 1.0e-5,
            },
            SanMFsmnBlockWeights {
                attn_norm_weight,
                attn_norm_bias,
                attn_qkv_weight,
                attn_qkv_bias,
                attn_out_weight,
                attn_out_bias,
                attn_fsmn_weight,
                ffn_norm_weight,
                ffn_norm_bias,
                ffn_up_weight,
                ffn_up_bias,
                ffn_down_weight,
                ffn_down_bias,
            },
            identity_map_err,
        )
        .expect("sanm_fsmn_encoder_layer should build");
        graph.set_output(output).expect("output should be settable");

        graph
            .set_f32_slice(input, input_values, "sanm_input")
            .expect("input upload");
        graph
            .set_f32_slice(
                attn_norm_weight,
                &vec![1.0f32; input_dim],
                "sanm_attn_norm_weight",
            )
            .expect("attn_norm_weight upload");
        graph
            .set_f32_slice(
                attn_norm_bias,
                &vec![0.0f32; input_dim],
                "sanm_attn_norm_bias",
            )
            .expect("attn_norm_bias upload");
        graph
            .set_f32_slice(
                attn_qkv_weight,
                &vec![0.0f32; input_dim * 3 * SANM_D_MODEL],
                "sanm_qkv_weight",
            )
            .expect("attn_qkv_weight upload");
        graph
            .set_f32_slice(attn_qkv_bias, &[0.0f32; 3 * SANM_D_MODEL], "sanm_qkv_bias")
            .expect("attn_qkv_bias upload");
        graph
            .set_f32_slice(attn_out_weight, &ZERO_4X4, "sanm_out_weight")
            .expect("attn_out_weight upload");
        graph
            .set_f32_slice(attn_out_bias, &ZERO_1D, "sanm_out_bias")
            .expect("attn_out_bias upload");
        graph
            .set_f16_bits_slice(
                attn_fsmn_weight,
                &[0u16; SANM_FSMN_KERNEL * SANM_D_MODEL],
                "sanm_fsmn_weight",
            )
            .expect("attn_fsmn_weight upload");
        graph
            .set_f32_slice(ffn_norm_weight, &NORM_WEIGHT, "sanm_ffn_norm_weight")
            .expect("ffn_norm_weight upload");
        graph
            .set_f32_slice(ffn_norm_bias, &ZERO_1D, "sanm_ffn_norm_bias")
            .expect("ffn_norm_bias upload");
        graph
            .set_f32_slice(ffn_up_weight, &ZERO_4X4, "sanm_ffn_up_weight")
            .expect("ffn_up_weight upload");
        graph
            .set_f32_slice(ffn_up_bias, &ZERO_1D, "sanm_ffn_up_bias")
            .expect("ffn_up_bias upload");
        graph
            .set_f32_slice(ffn_down_weight, &ZERO_4X4, "sanm_ffn_down_weight")
            .expect("ffn_down_weight upload");
        graph
            .set_f32_slice(ffn_down_bias, &ZERO_1D, "sanm_ffn_down_bias")
            .expect("ffn_down_bias upload");

        graph
            .compute_output_f32(output, SANM_D_MODEL * TOKENS)
            .expect("sanm_fsmn_encoder_layer graph should compute")
    }

    #[test]
    fn sanm_residual_adds_input_when_input_dim_matches_d_model() {
        let input = [
            0.4, -0.3, 0.2, 0.1, //
            -0.1, 0.5, -0.2, 0.3, //
            0.2, 0.1, 0.4, -0.4,
        ];
        let output = run_sanm_fsmn_encoder_layer_zeroed(SANM_D_MODEL, &input);
        assert_eq!(output.len(), input.len());
        for (index, (actual, expected)) in output.iter().zip(input.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "element {index}: with attention/FFN weights zeroed, the residual add \
                 must pass `input` through unchanged when input_dim == d_model, got \
                 {actual} expected {expected}"
            );
        }
    }

    #[test]
    fn sanm_residual_is_skipped_when_input_dim_differs_from_d_model() {
        // A wider first-layer input (SenseVoice's real shape feeds a 560-dim
        // LFR+prompt feature into a 512-dim d_model); the specific values
        // don't matter here since every weight downstream is zeroed -- only
        // that input_dim != d_model.
        let input: Vec<f32> = (0..(SANM_MISMATCHED_INPUT_DIM * TOKENS))
            .map(|i| 0.1 * (i as f32 + 1.0))
            .collect();
        let output = run_sanm_fsmn_encoder_layer_zeroed(SANM_MISMATCHED_INPUT_DIM, &input);
        assert_eq!(output.len(), SANM_D_MODEL * TOKENS);
        for (index, actual) in output.iter().enumerate() {
            assert!(
                actual.abs() <= 1.0e-5,
                "element {index}: with attention/FFN weights zeroed, the residual add \
                 must be SKIPPED when input_dim != d_model, so the block output should \
                 reduce to exactly zero, got {actual}"
            );
        }
    }
}
