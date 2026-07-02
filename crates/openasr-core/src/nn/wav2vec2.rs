//! wav2vec2 building blocks (the genuinely-new math for the CTC family).
//!
//! Three pieces that are NOT reusable from the existing pre-norm transformer /
//! conformer blocks:
//!
//! - `fold_pos_conv_weight_norm`: fold the positional-conv weight-norm
//!   parametrization (`weight_g[1,1,K] * weight_v / ||weight_v||_dim2`) into one
//!   effective kernel at IMPORT time (there is no weight-norm graph op). The norm
//!   is over the (out_channels, in_per_group) axes per kernel position — getting
//!   this dim wrong silently corrupts every position embedding.
//! - `grouped_conv_1d`: arbitrary-groups Conv1d (`groups != 1` and `!= channels`)
//!   built as a per-group loop of `conv_1d` over channel-sliced views + `concat`
//!   (ggml has no single op for it). Used for the `num_conv_pos_embedding_groups=16`
//!   positional conv.
//! - `wav2vec2_post_norm_encoder_layer`: the `do_stable_layer_norm=False`
//!   post-norm transformer layer (`x = ln(x + attn(x)); x = final_ln(x + ffn(x))`),
//!   re-sequenced from `nn::attn` / `nn::norm` sub-builders (the shared
//!   `transformer_layer` is strictly pre-norm and not reusable as-is).

#![allow(dead_code)]

use crate::ggml_runtime::{GgmlCpuGraphBuilder, GgmlCpuGraphError, GgmlCpuTensor};
use crate::nn::attn::{
    AttentionHeadLayout, AttentionReshapeSteps, AttentionValueMergeSteps,
    STANDARD_HEAD_PERMUTE_AXES, attention_context_from_probs,
    reshape_projection_to_attention_heads,
};
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

/// Fold the wav2vec2 positional-conv weight-norm into one effective kernel.
///
/// `weight_v` is the PyTorch `[out_channels, in_per_group, kernel]` C-order
/// tensor (out outermost, kernel innermost). `weight_g` is `[kernel]` (the
/// `[1,1,K]` tensor flattened). Output is the same `[out, in_per_group, kernel]`
/// C-order layout with `effective[o,i,k] = g[k] * v[o,i,k] / norm_k`, where
/// `norm_k = sqrt(sum_{o,i} v[o,i,k]^2)` (PyTorch `weight_norm(dim=2)`).
pub(crate) fn fold_pos_conv_weight_norm(
    weight_v: &[f32],
    weight_g: &[f32],
    out_channels: usize,
    in_per_group: usize,
    kernel: usize,
) -> Result<Vec<f32>, String> {
    if weight_v.len() != out_channels * in_per_group * kernel {
        return Err(format!(
            "pos-conv weight_v has {} elements, expected {}x{}x{}={}",
            weight_v.len(),
            out_channels,
            in_per_group,
            kernel,
            out_channels * in_per_group * kernel
        ));
    }
    if weight_g.len() != kernel {
        return Err(format!(
            "pos-conv weight_g has {} elements, expected kernel {kernel}",
            weight_g.len()
        ));
    }
    // norm over (out, in_per_group) per kernel position k.
    let mut norm_sq = vec![0.0f64; kernel];
    for o in 0..out_channels {
        for i in 0..in_per_group {
            let base = (o * in_per_group + i) * kernel;
            for k in 0..kernel {
                let v = weight_v[base + k] as f64;
                norm_sq[k] += v * v;
            }
        }
    }
    let norm: Vec<f64> = norm_sq.iter().map(|s| s.sqrt()).collect();
    for (k, n) in norm.iter().enumerate() {
        if *n == 0.0 {
            return Err(format!("pos-conv weight_v norm at kernel pos {k} is zero"));
        }
    }
    let mut effective = vec![0.0f32; weight_v.len()];
    for o in 0..out_channels {
        for i in 0..in_per_group {
            let base = (o * in_per_group + i) * kernel;
            for k in 0..kernel {
                effective[base + k] =
                    (weight_g[k] as f64 * weight_v[base + k] as f64 / norm[k]) as f32;
            }
        }
    }
    Ok(effective)
}

/// Group descriptor for `grouped_conv_1d`: each entry is one group's kernel
/// tensor in ggml `conv_1d` layout `[kernel, in_per_group, out_per_group]`.
pub(crate) struct GroupedConv1dParams {
    pub groups: usize,
    /// Time dimension (ne0) of the contiguous `[T, in_channels]` input.
    pub time: usize,
    pub in_per_group: usize,
    pub out_per_group: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

/// Arbitrary-groups Conv1d via a per-group loop of `conv_1d` over channel-sliced
/// views + `concat`. `data` is ggml `[T, in_channels]` (time fastest within each
/// channel row); `group_kernels[g]` is `[kernel, in_per_group, out_per_group]`.
/// Returns `[T_out, out_channels]`. The channel slicing assumes `data` is
/// contiguous (the caller `cont`s it first if needed).
pub(crate) fn grouped_conv_1d<'a, E, F>(
    graph: &GgmlCpuGraphBuilder<'a>,
    data: GgmlCpuTensor<'a>,
    group_kernels: &[GgmlCpuTensor<'a>],
    params: &GroupedConv1dParams,
    step: &'static str,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    debug_assert_eq!(group_kernels.len(), params.groups);
    let element = std::mem::size_of::<f32>();
    let t = params.time;
    let mut group_outputs = Vec::with_capacity(params.groups);
    // `g` indexes `group_kernels` AND computes per-group byte offsets into `data`,
    // so a range loop is clearest here.
    #[allow(clippy::needless_range_loop)]
    for g in 0..params.groups {
        // View this group's `in_per_group` channels: ne1-slice of [T, in_channels].
        let in_view = graph
            .view_2d(
                data,
                t,
                params.in_per_group,
                t * element,
                g * params.in_per_group * t * element,
            )
            .map_err(|source| map_err(step, source))?;
        let in_view = graph
            .cont(in_view)
            .map_err(|source| map_err(step, source))?;
        let conv = graph
            .conv_1d(
                group_kernels[g],
                in_view,
                params.stride,
                params.padding,
                params.dilation,
            )
            .map_err(|source| map_err(step, source))?;
        let conv = graph.cont(conv).map_err(|source| map_err(step, source))?;
        group_outputs.push(conv);
    }
    let mut grouped = group_outputs[0];
    for &next in &group_outputs[1..] {
        grouped = graph
            .concat(grouped, next, 1)
            .map_err(|source| map_err(step, source))?;
    }
    Ok(grouped)
}

/// Per-layer weights for `wav2vec2_post_norm_encoder_layer`. The 2-D linears are
/// the projection tensors (bound zero-copy / arena), the norms/biases are 1-D
/// arena tensors.
pub(crate) struct Wav2Vec2EncoderLayerWeights<'a> {
    pub q_weight: GgmlCpuTensor<'a>,
    pub q_bias: GgmlCpuTensor<'a>,
    pub k_weight: GgmlCpuTensor<'a>,
    pub k_bias: GgmlCpuTensor<'a>,
    pub v_weight: GgmlCpuTensor<'a>,
    pub v_bias: GgmlCpuTensor<'a>,
    pub out_weight: GgmlCpuTensor<'a>,
    pub out_bias: GgmlCpuTensor<'a>,
    pub layer_norm_weight: GgmlCpuTensor<'a>,
    pub layer_norm_bias: GgmlCpuTensor<'a>,
    pub ff_intermediate_weight: GgmlCpuTensor<'a>,
    pub ff_intermediate_bias: GgmlCpuTensor<'a>,
    pub ff_output_weight: GgmlCpuTensor<'a>,
    pub ff_output_bias: GgmlCpuTensor<'a>,
    pub final_layer_norm_weight: GgmlCpuTensor<'a>,
    pub final_layer_norm_bias: GgmlCpuTensor<'a>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Wav2Vec2EncoderLayerConfig {
    pub d_model: usize,
    pub attention_heads: usize,
    pub head_dim: usize,
    pub sequence_len: usize,
    pub layer_norm_epsilon: f32,
}

/// One wav2vec2 post-norm transformer encoder layer (`do_stable_layer_norm=False`):
/// `x = layer_norm(x + self_attn(x)); x = final_layer_norm(x + ffn(x))`. Full
/// bidirectional attention (no mask, no rel-pos), scaling `1/sqrt(head_dim)`,
/// GELU-erf FFN. `state` is `[d_model, T]` ggml layout.
pub(crate) fn wav2vec2_post_norm_encoder_layer<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    state: GgmlCpuTensor<'a>,
    config: Wav2Vec2EncoderLayerConfig,
    weights: &Wav2Vec2EncoderLayerWeights<'a>,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let attn_residual = state;
    let attn_out = wav2vec2_self_attention(graph, state, config, weights, map_err)?;
    // residual + post-attention layer_norm.
    let x = graph
        .add(attn_out, attn_residual)
        .map_err(|source| map_err("w2v2_attn_residual", source))?;
    let x = apply_affine_layer_norm(
        graph,
        x,
        config.layer_norm_epsilon,
        weights.layer_norm_weight,
        weights.layer_norm_bias,
        AffineLayerNormSteps {
            norm: "w2v2_attn_norm",
            scale: "w2v2_attn_norm_scale",
            bias: "w2v2_attn_norm_bias",
        },
        map_err,
    )?;

    // FFN: up -> gelu_erf -> down, residual, then final_layer_norm.
    let ffn_residual = x;
    let up = graph
        .mul_mat(weights.ff_intermediate_weight, x)
        .map_err(|source| map_err("w2v2_ffn_up", source))?;
    let up = graph
        .add(up, weights.ff_intermediate_bias)
        .map_err(|source| map_err("w2v2_ffn_up_bias", source))?;
    let activated = graph
        .gelu_erf(up)
        .map_err(|source| map_err("w2v2_ffn_gelu", source))?;
    let down = graph
        .mul_mat(weights.ff_output_weight, activated)
        .map_err(|source| map_err("w2v2_ffn_down", source))?;
    let down = graph
        .add(down, weights.ff_output_bias)
        .map_err(|source| map_err("w2v2_ffn_down_bias", source))?;
    let x = graph
        .add(down, ffn_residual)
        .map_err(|source| map_err("w2v2_ffn_residual", source))?;
    apply_affine_layer_norm(
        graph,
        x,
        config.layer_norm_epsilon,
        weights.final_layer_norm_weight,
        weights.final_layer_norm_bias,
        AffineLayerNormSteps {
            norm: "w2v2_final_norm",
            scale: "w2v2_final_norm_scale",
            bias: "w2v2_final_norm_bias",
        },
        map_err,
    )
}

/// One wav2vec2 STABLE-layer-norm (pre-norm) transformer encoder layer
/// (`do_stable_layer_norm=True`): `x = x + attn(ln(x)); x = x + ffn(final_ln(x))`.
/// The two LayerNorms come BEFORE the sublayers, residuals add the raw input.
/// A separate FINAL encoder LayerNorm (after the whole stack) is applied by the
/// caller. Same attention/FFN math as the post-norm layer, just re-sequenced.
/// `state` is `[d_model, T]`.
pub(crate) fn wav2vec2_stable_layer_norm_encoder_layer<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    state: GgmlCpuTensor<'a>,
    config: Wav2Vec2EncoderLayerConfig,
    weights: &Wav2Vec2EncoderLayerWeights<'a>,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    // attn: residual = x; h = ln(x); h = attn(h); x = residual + h.
    let attn_residual = state;
    let normed = apply_affine_layer_norm(
        graph,
        state,
        config.layer_norm_epsilon,
        weights.layer_norm_weight,
        weights.layer_norm_bias,
        AffineLayerNormSteps {
            norm: "w2v2_pre_attn_norm",
            scale: "w2v2_pre_attn_norm_scale",
            bias: "w2v2_pre_attn_norm_bias",
        },
        map_err,
    )?;
    let attn_out = wav2vec2_self_attention(graph, normed, config, weights, map_err)?;
    let x = graph
        .add(attn_out, attn_residual)
        .map_err(|source| map_err("w2v2_pre_attn_residual", source))?;

    // ffn: residual = x; h = final_ln(x); h = ffn(h); x = residual + h.
    let ffn_residual = x;
    let normed = apply_affine_layer_norm(
        graph,
        x,
        config.layer_norm_epsilon,
        weights.final_layer_norm_weight,
        weights.final_layer_norm_bias,
        AffineLayerNormSteps {
            norm: "w2v2_pre_ffn_norm",
            scale: "w2v2_pre_ffn_norm_scale",
            bias: "w2v2_pre_ffn_norm_bias",
        },
        map_err,
    )?;
    let up = graph
        .mul_mat(weights.ff_intermediate_weight, normed)
        .map_err(|source| map_err("w2v2_pre_ffn_up", source))?;
    let up = graph
        .add(up, weights.ff_intermediate_bias)
        .map_err(|source| map_err("w2v2_pre_ffn_up_bias", source))?;
    let activated = graph
        .gelu_erf(up)
        .map_err(|source| map_err("w2v2_pre_ffn_gelu", source))?;
    let down = graph
        .mul_mat(weights.ff_output_weight, activated)
        .map_err(|source| map_err("w2v2_pre_ffn_down", source))?;
    let down = graph
        .add(down, weights.ff_output_bias)
        .map_err(|source| map_err("w2v2_pre_ffn_down_bias", source))?;
    graph
        .add(down, ffn_residual)
        .map_err(|source| map_err("w2v2_pre_ffn_residual", source))
}

/// Full bidirectional multi-head self-attention (no mask, no rel-pos). `state`
/// is `[d_model, T]`. Mirrors the qwen audio-encoder attention math but with no
/// attention mask. Returns `[d_model, T]`.
fn wav2vec2_self_attention<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    state: GgmlCpuTensor<'a>,
    config: Wav2Vec2EncoderLayerConfig,
    weights: &Wav2Vec2EncoderLayerWeights<'a>,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let layout = AttentionHeadLayout {
        head_dim: config.head_dim,
        attention_heads: config.attention_heads,
        sequence_len: config.sequence_len,
    };
    let project = |graph: &GgmlCpuGraphBuilder<'a>,
                   weight: GgmlCpuTensor<'a>,
                   bias: GgmlCpuTensor<'a>,
                   mm: &'static str,
                   ba: &'static str|
     -> Result<GgmlCpuTensor<'a>, E> {
        let proj = graph
            .mul_mat(weight, state)
            .map_err(|source| map_err(mm, source))?;
        graph.add(proj, bias).map_err(|source| map_err(ba, source))
    };

    let q = project(
        graph,
        weights.q_weight,
        weights.q_bias,
        "w2v2_q",
        "w2v2_q_bias",
    )?;
    let k = project(
        graph,
        weights.k_weight,
        weights.k_bias,
        "w2v2_k",
        "w2v2_k_bias",
    )?;
    let v = project(
        graph,
        weights.v_weight,
        weights.v_bias,
        "w2v2_v",
        "w2v2_v_bias",
    )?;

    let reshape_steps = AttentionReshapeSteps {
        reshape: "w2v2_head_reshape",
        permute: "w2v2_head_permute",
        cont: "w2v2_head_cont",
    };
    let q_heads = reshape_projection_to_attention_heads(
        graph,
        q,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        reshape_steps,
        map_err,
    )?;
    let k_heads = reshape_projection_to_attention_heads(
        graph,
        k,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        reshape_steps,
        map_err,
    )?;
    let v_heads = reshape_projection_to_attention_heads(
        graph,
        v,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        reshape_steps,
        map_err,
    )?;

    // scores = k_heads^T(q) via mul_mat(k_heads, q_heads) -> [T_k, T_q, heads].
    let scores = graph
        .mul_mat(k_heads, q_heads)
        .map_err(|source| map_err("w2v2_attn_scores", source))?;
    let scale = (config.head_dim as f32).sqrt().recip();
    let probs = graph
        .soft_max_ext(scores, None, scale, 0.0)
        .map_err(|source| map_err("w2v2_attn_softmax", source))?;
    let context = attention_context_from_probs(
        graph,
        v_heads,
        probs,
        layout,
        AttentionValueMergeSteps {
            value_permute: "w2v2_value_permute",
            value_cont: "w2v2_value_cont",
            context_mul: "w2v2_context_mul",
            context_merge_permute: "w2v2_context_merge_permute",
            context_merge_cont: "w2v2_context_merge_cont",
            context_merge_reshape: "w2v2_context_merge_reshape",
        },
        map_err,
    )?;
    let out = graph
        .mul_mat(weights.out_weight, context)
        .map_err(|source| map_err("w2v2_attn_out", source))?;
    graph
        .add(out, weights.out_bias)
        .map_err(|source| map_err("w2v2_attn_out_bias", source))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Weight-norm fold matches a numpy `weight_g[k] * weight_v / ||weight_v||_k`
    /// reference (norm over the out_channels x in_per_group axes per kernel pos).
    /// Catches the #1 silent-corruption risk (wrong norm dim) in isolation.
    #[test]
    fn weight_norm_fold_matches_numpy_reference() {
        const OUT: usize = 4;
        const IN_G: usize = 2;
        const K: usize = 3;
        // weight_v in PyTorch [out, in/g, K] C-order (out outer, K inner).
        let weight_v: [f32; OUT * IN_G * K] = [
            0.536589, 0.130953, 0.028949, -0.559048, -0.083216, -0.106428, -0.024822, -0.1881,
            -0.013145, -0.143165, -0.394159, 0.265387, 0.264395, 0.512872, 0.01501, -0.121403,
            -0.163608, -0.463943, 0.29471, -0.33032, -0.355514, -0.061695, 0.445845, 0.071015,
        ];
        let weight_g: [f32; K] = [1.5, 2.0, 0.5];
        let expected: [f32; OUT * IN_G * K] = [
            0.901648, 0.290557, 0.022082, -0.939387, -0.184638, -0.081182, -0.041709, -0.417354,
            -0.010027, -0.240565, -0.874554, 0.202433, 0.444272, 1.137953, 0.011449, -0.203997,
            -0.363011, -0.353889, 0.495211, -0.732909, -0.271181, -0.103668, 0.989235, 0.054169,
        ];
        let folded = fold_pos_conv_weight_norm(&weight_v, &weight_g, OUT, IN_G, K).expect("fold");
        for (i, (actual, expected)) in folded.iter().copied().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "fold elem {i}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn weight_norm_fold_rejects_shape_mismatch() {
        // weight_v element count disagrees with out*in_per_group*kernel.
        assert!(fold_pos_conv_weight_norm(&[1.0, 2.0, 3.0], &[1.0], 2, 1, 1).is_err());
        // weight_g element count disagrees with kernel.
        assert!(fold_pos_conv_weight_norm(&[1.0, 2.0], &[1.0], 1, 1, 2).is_err());
    }
}
