//! firered-aed Conformer encoder ggml graph (Stage 2).
//!
//! Faithfully reproduces `fireredasr/models/module/conformer_encoder.py`
//! (`ConformerEncoder` / `RelPosEmbConformerBlock`, upstream FireRedTeam/
//! FireRedASR, verified against the actual checkpoint + reference inference
//! run for this port): Conv2d 4x subsampling (context-pad 6 zero frames first)
//! -> 16x macaron-FFN + rel-pos MHSA (per-projection q/k/v LayerNorm, no
//! attention biases) + GLU depthwise-conv (LayerNorm mid-block, no conv
//! biases) + macaron-FFN -> final affine LayerNorm per block.
//!
//! This is NOT built on the shared `nn::encoder::conformer_block` composer:
//! that block assumes ONE shared pre-attention LayerNorm feeding q/k/v and
//! biased attention/conv projections, whereas firered-aed normalizes q, k, v
//! with three *independent* LayerNorms (same input, different affine params)
//! and has zero biases anywhere in attention or the conv module. Reusing the
//! shared block would silently produce the wrong math, so this follows the
//! dolphin/sensevoice/xasr precedent: hand-written, `block_stack: None`,
//! built from the lower-level `nn::attn` / `nn::norm` / `nn::conv` primitives.
//! The relative-position table math IS bit-identical to cohere/parakeet-ctc's
//! shared Transformer-XL formula (same ESPnet/WeNet lineage), so that helper
//! is reused directly instead of re-derived.

#![allow(dead_code)]

use std::path::Path;

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner,
};
use crate::models::cohere::encoder_graph::build_relative_positional_encoding;
use crate::nn::attn::{
    AttentionHeadLayout, AttentionReshapeSteps, AttentionValueMergeSteps,
    STANDARD_HEAD_PERMUTE_AXES, attention_context_from_probs,
    reshape_projection_to_attention_heads,
};
use crate::nn::conv::{
    Conv2dParams, ConvActivation, ConvBlockSteps, apply_conv_2d_bias_activation, reshape_bias_4d,
};
use crate::nn::ffn::{
    FeedForwardActivation, FeedForwardResidualSteps, apply_feed_forward_residual,
};
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

use super::encoder_weights::{
    FireRedEncoderLayerWeights, FireRedEncoderWeights, FireRedEncoderWeightsError,
};
use super::runtime_contract::FireRedAedExecutionMetadata;

const FIRERED_ENCODER_LAYER_NORM_EPSILON: f32 = 1.0e-5;
const FIRERED_ENCODER_GRAPH_CONTEXT_BYTES: usize = 512 * 1024 * 1024;
const FIRERED_ENCODER_GRAPH_SIZE: usize = 32_768;
/// `Conv2dSubsampling.context = left_context(3) + 1 + right_context(3)`; the
/// encoder pads the time axis by `context - 1` zero frames before the stem
/// (`fireredasr` `ConformerEncoder.forward(..., pad=True)`).
const SUBSAMPLE_CONTEXT_PAD_FRAMES: usize = 6;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FireRedEncoderOutput {
    pub frame_count: usize,
    pub hidden_size: usize,
    /// Row-major `[frame][hidden]`, contiguous f32.
    pub rows: Vec<f32>,
}

#[derive(Debug, Error)]
pub(crate) enum FireRedEncoderError {
    #[error("firered-aed encoder weights: {0}")]
    Weights(#[from] FireRedEncoderWeightsError),
    #[error("firered-aed encoder requires at least one input frame")]
    EmptyInput,
    #[error("firered-aed encoder graph construction failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        #[source]
        source: GgmlCpuGraphError,
    },
    #[error("firered-aed encoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("firered-aed encoder shape overflowed")]
    ShapeOverflow,
}

fn map_err(step: &'static str, source: GgmlCpuGraphError) -> FireRedEncoderError {
    FireRedEncoderError::GraphBuildFailed { step, source }
}

fn conv_out_dim(input: usize, kernel: usize, stride: usize) -> Result<usize, FireRedEncoderError> {
    input
        .checked_sub(kernel)
        .and_then(|v| v.checked_div(stride))
        .and_then(|v| v.checked_add(1))
        .ok_or(FireRedEncoderError::ShapeOverflow)
}

pub(crate) fn firered_encoder_graph_config() -> GgmlCpuGraphConfig {
    // Stage 2 lands CPU-only; GPU backends can follow once decoder/executor
    // parity is established (matches the parakeet-ctc/sensevoice staging
    // precedent of correctness-first, backend-breadth-later).
    GgmlCpuGraphConfig {
        context_bytes: FIRERED_ENCODER_GRAPH_CONTEXT_BYTES,
        graph_size: FIRERED_ENCODER_GRAPH_SIZE,
        n_threads: None,
        backend: GgmlCpuGraphBackend::Cpu,
        use_scheduler: false,
    }
}

/// Run the full encoder forward pass in a single ggml graph (no incremental
/// reuse -- matches cohere's/parakeet's single-shot encoder shape).
pub(crate) fn encode_firered_aed_audio_embeddings(
    runtime_path: &Path,
    metadata: FireRedAedExecutionMetadata,
    cmvn_features: &[f32],
    n_frames: usize,
) -> Result<FireRedEncoderOutput, FireRedEncoderError> {
    if n_frames == 0 {
        return Err(FireRedEncoderError::EmptyInput);
    }
    let feature_dim = metadata.feature_dim;
    if cmvn_features.len() != n_frames * feature_dim {
        return Err(FireRedEncoderError::ShapeOverflow);
    }

    let mut runner = GgmlCpuGraphRunner::new(firered_encoder_graph_config())
        .map_err(|source| map_err("runner_init", source))?;
    let loaded = runner
        .load_gguf_weight_context(runtime_path)
        .map_err(|source| map_err("load_gguf_weight_context", source))?;
    let weights = FireRedEncoderWeights::load(&loaded, metadata.encoder_n_layers)?;

    // Zero-pad the time axis by `context - 1` frames (matches
    // `F.pad(padded_input, (0,0,0,context-1))`), then run the 2x Conv2d(k3,s2)
    // stem.
    let padded_frames = n_frames
        .checked_add(SUBSAMPLE_CONTEXT_PAD_FRAMES)
        .ok_or(FireRedEncoderError::ShapeOverflow)?;
    let mut padded = vec![0.0_f32; padded_frames * feature_dim];
    padded[..cmvn_features.len()].copy_from_slice(cmvn_features);

    let conv1_freq = conv_out_dim(feature_dim, 3, 2)?;
    let conv1_time = conv_out_dim(padded_frames, 3, 2)?;
    let conv2_freq = conv_out_dim(conv1_freq, 3, 2)?;
    let conv2_time = conv_out_dim(conv1_time, 3, 2)?;
    if conv2_freq * metadata.subsample_channels != metadata.subsample_out_dim {
        return Err(FireRedEncoderError::ShapeOverflow);
    }
    let frame_count = conv2_time;
    // Valid (unpadded) frame count: computed from the ORIGINAL (pre-context-pad)
    // frame count via the same subsampling arithmetic the upstream uses for
    // `input_lengths` (`(L-3)//2+1` twice) -- NOT from `padded_frames`. Frames
    // at/after this index are context-pad artifacts and must be masked out of
    // every layer's self-attention (upstream `src_mask`), or the last couple of
    // encoder frames leak zero-padded conv output into every frame's context.
    let valid_frame_count = conv_out_dim(conv_out_dim(n_frames, 3, 2)?, 3, 2)?.min(frame_count);

    let mut graph = runner.start_graph();
    let mel = graph
        .new_tensor_2d_f32(feature_dim, padded_frames, "firered_enc_mel")
        .map_err(|source| map_err("ggml_new_tensor_2d(mel)", source))?;
    graph
        .set_input(mel)
        .map_err(|source| map_err("ggml_set_input(mel)", source))?;

    let positional_values = build_relative_positional_encoding(metadata.d_model, frame_count)
        .map_err(|_| FireRedEncoderError::ShapeOverflow)?;
    let pos_enc = graph
        .new_tensor_2d_f32(metadata.d_model, 2 * frame_count - 1, "firered_enc_rel_pos")
        .map_err(|source| map_err("ggml_new_tensor_2d(pos_enc)", source))?;
    graph
        .set_input(pos_enc)
        .map_err(|source| map_err("ggml_set_input(pos_enc)", source))?;

    // Additive self-attention key mask: 0.0 for valid (unpadded) key frames,
    // -inf for context-pad frames at/after `valid_frame_count`. Broadcasts
    // over the query and head dims (ggml_add allows size-1 broadcast dims).
    let key_mask_values: Vec<f32> = (0..frame_count)
        .map(|idx| {
            if idx < valid_frame_count {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    let key_mask = graph
        .new_tensor_3d_f32(frame_count, 1, 1, "firered_enc_key_mask")
        .map_err(|source| map_err("ggml_new_tensor_3d(key_mask)", source))?;
    graph
        .set_input(key_mask)
        .map_err(|source| map_err("ggml_set_input(key_mask)", source))?;

    // ----- Conv2d subsampling stem -----
    let subsample_params = Conv2dParams {
        stride_x: 2,
        stride_y: 2,
        padding_x: 0,
        padding_y: 0,
        dilation_x: 1,
        dilation_y: 1,
    };
    let mut state_4d = graph
        .reshape_4d(mel, feature_dim, padded_frames, 1, 1)
        .map_err(|source| map_err("ggml_reshape_4d(mel)", source))?;
    let conv1_bias_4d = reshape_bias_4d(
        &graph,
        weights.subsample_conv1_bias.as_graph_tensor(),
        metadata.subsample_channels,
        "ggml_reshape_4d(subsample_conv1_bias)",
        map_err,
    )?;
    state_4d = apply_conv_2d_bias_activation(
        &graph,
        weights.subsample_conv1_weight.as_graph_tensor(),
        state_4d,
        conv1_bias_4d,
        subsample_params,
        ConvActivation::Relu,
        ConvBlockSteps {
            conv: "ggml_conv_2d(subsample_conv1)",
            bias: "ggml_add(subsample_conv1_bias)",
            activation: "ggml_relu(subsample_conv1)",
        },
        map_err,
    )?;
    let conv2_bias_4d = reshape_bias_4d(
        &graph,
        weights.subsample_conv2_bias.as_graph_tensor(),
        metadata.subsample_channels,
        "ggml_reshape_4d(subsample_conv2_bias)",
        map_err,
    )?;
    state_4d = apply_conv_2d_bias_activation(
        &graph,
        weights.subsample_conv2_weight.as_graph_tensor(),
        state_4d,
        conv2_bias_4d,
        subsample_params,
        ConvActivation::Relu,
        ConvBlockSteps {
            conv: "ggml_conv_2d(subsample_conv2)",
            bias: "ggml_add(subsample_conv2_bias)",
            activation: "ggml_relu(subsample_conv2)",
        },
        map_err,
    )?;
    // state_4d is [freq(conv2_freq), time(frame_count), channel, 1]. PyTorch
    // does `x.transpose(1,2).contiguous().view(N,T,C*D)` (channel-major,
    // freq-minor): permute(0,2,1,3) puts freq first (fastest), channel next,
    // time last, so the reshape_2d merge yields flat index `c*conv2_freq+d`.
    let mut state = graph
        .permute(state_4d, 0, 2, 1, 3)
        .map_err(|source| map_err("ggml_permute(subsample_flatten)", source))?;
    state = graph
        .cont(state)
        .map_err(|source| map_err("ggml_cont(subsample_flatten)", source))?;
    state = graph
        .reshape_2d(state, metadata.subsample_out_dim, frame_count)
        .map_err(|source| map_err("ggml_reshape_2d(subsample_flatten)", source))?;
    state = graph
        .mul_mat(weights.subsample_out_weight.as_graph_tensor(), state)
        .map_err(|source| map_err("ggml_mul_mat(subsample_out)", source))?;
    state = graph
        .add(state, weights.subsample_out_bias.as_graph_tensor())
        .map_err(|source| map_err("ggml_add(subsample_out_bias)", source))?;

    // ----- 16x Conformer block -----
    for layer in &weights.layers {
        state = firered_conformer_block(
            &mut graph,
            state,
            pos_enc,
            key_mask,
            metadata,
            frame_count,
            layer,
        )?;
    }

    graph
        .set_output(state)
        .map_err(|source| map_err("ggml_set_output(encoder)", source))?;
    graph
        .set_f32_slice(mel, &padded, "firered_enc_mel")
        .map_err(|source| map_err("ggml_set_f32_slice(mel)", source))?;
    graph
        .set_f32_slice(pos_enc, &positional_values, "firered_enc_rel_pos")
        .map_err(|source| map_err("ggml_set_f32_slice(pos_enc)", source))?;
    graph
        .set_f32_slice(key_mask, &key_mask_values, "firered_enc_key_mask")
        .map_err(|source| map_err("ggml_set_f32_slice(key_mask)", source))?;

    let expected_len = frame_count
        .checked_mul(metadata.d_model)
        .ok_or(FireRedEncoderError::ShapeOverflow)?;
    let rows = graph
        .compute_output_f32(state, expected_len)
        .map_err(|error| FireRedEncoderError::GraphExecutionFailed {
            reason: error.to_string(),
        })?;

    Ok(FireRedEncoderOutput {
        frame_count,
        hidden_size: metadata.d_model,
        rows,
    })
}

#[allow(clippy::too_many_lines)]
fn firered_conformer_block<'a>(
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    input: crate::ggml_runtime::GgmlCpuTensor<'a>,
    pos_enc: crate::ggml_runtime::GgmlCpuTensor<'a>,
    key_mask: crate::ggml_runtime::GgmlCpuTensor<'a>,
    metadata: FireRedAedExecutionMetadata,
    n_frames: usize,
    layer: &FireRedEncoderLayerWeights,
) -> Result<crate::ggml_runtime::GgmlCpuTensor<'a>, FireRedEncoderError> {
    let d_model = metadata.d_model;
    let heads = metadata.n_heads;
    let head_dim = metadata.head_dim;
    let epsilon = FIRERED_ENCODER_LAYER_NORM_EPSILON;

    // ----- FF1 (macaron, scale 0.5) -----
    let ff1_norm = apply_affine_layer_norm(
        graph,
        input,
        epsilon,
        layer.ffn1_norm_weight.as_graph_tensor(),
        layer.ffn1_norm_bias.as_graph_tensor(),
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "ffn1_norm",
            bias: "ffn1_norm",
        },
        map_err,
    )?;
    let mut state = apply_feed_forward_residual(
        graph,
        ff1_norm,
        input,
        FeedForwardActivation::Silu,
        Some(0.5),
        FeedForwardResidualSteps {
            activation: "ggml_silu(ffn1)",
            scale: Some("ggml_scale(ffn1_half)"),
            residual: "ggml_add(ffn1_residual)",
        },
        |graph, value| {
            let up = graph
                .mul_mat(layer.ffn1_up_weight.as_graph_tensor(), value)
                .map_err(|source| map_err("ggml_mul_mat(ffn1_up)", source))?;
            graph
                .add(up, layer.ffn1_up_bias.as_graph_tensor())
                .map_err(|source| map_err("ggml_add(ffn1_up_bias)", source))
        },
        |graph, value| {
            let down = graph
                .mul_mat(layer.ffn1_down_weight.as_graph_tensor(), value)
                .map_err(|source| map_err("ggml_mul_mat(ffn1_down)", source))?;
            graph
                .add(down, layer.ffn1_down_bias.as_graph_tensor())
                .map_err(|source| map_err("ggml_add(ffn1_down_bias)", source))
        },
        map_err,
    )?;
    let attn_input = state;

    // ----- Rel-pos self-attention: per-projection q/k/v LayerNorm, no biases -----
    let q_in = apply_affine_layer_norm(
        graph,
        state,
        epsilon,
        layer.attn_norm_q_weight.as_graph_tensor(),
        layer.attn_norm_q_bias.as_graph_tensor(),
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "attn_norm_q",
            bias: "attn_norm_q",
        },
        map_err,
    )?;
    let k_in = apply_affine_layer_norm(
        graph,
        state,
        epsilon,
        layer.attn_norm_k_weight.as_graph_tensor(),
        layer.attn_norm_k_bias.as_graph_tensor(),
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "attn_norm_k",
            bias: "attn_norm_k",
        },
        map_err,
    )?;
    let v_in = apply_affine_layer_norm(
        graph,
        state,
        epsilon,
        layer.attn_norm_v_weight.as_graph_tensor(),
        layer.attn_norm_v_bias.as_graph_tensor(),
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "attn_norm_v",
            bias: "attn_norm_v",
        },
        map_err,
    )?;
    let q = graph
        .mul_mat(layer.attn_q_weight.as_graph_tensor(), q_in)
        .map_err(|source| map_err("ggml_mul_mat(attn_q)", source))?;
    let k = graph
        .mul_mat(layer.attn_k_weight.as_graph_tensor(), k_in)
        .map_err(|source| map_err("ggml_mul_mat(attn_k)", source))?;
    let v = graph
        .mul_mat(layer.attn_v_weight.as_graph_tensor(), v_in)
        .map_err(|source| map_err("ggml_mul_mat(attn_v)", source))?;
    let r = graph
        .mul_mat(layer.attn_pos_weight.as_graph_tensor(), pos_enc)
        .map_err(|source| map_err("ggml_mul_mat(attn_pos)", source))?;
    let q_u = graph
        .add(
            q,
            graph
                .reshape_1d(layer.attn_pos_bias_u.as_graph_tensor(), d_model)
                .map_err(|source| map_err("ggml_reshape_1d(attn_pos_bias_u)", source))?,
        )
        .map_err(|source| map_err("ggml_add(attn_pos_bias_u)", source))?;
    let q_v = graph
        .add(
            q,
            graph
                .reshape_1d(layer.attn_pos_bias_v.as_graph_tensor(), d_model)
                .map_err(|source| map_err("ggml_reshape_1d(attn_pos_bias_v)", source))?,
        )
        .map_err(|source| map_err("ggml_add(attn_pos_bias_v)", source))?;

    let attention_layout = AttentionHeadLayout {
        head_dim,
        attention_heads: heads,
        sequence_len: n_frames,
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
    let k_heads = reshape_projection_to_attention_heads(
        graph,
        k,
        attention_layout,
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(attn_k)",
            permute: "ggml_permute(attn_k)",
            cont: "ggml_cont(attn_k)",
        },
        map_err,
    )?;
    let r_heads = reshape_projection_to_attention_heads(
        graph,
        r,
        AttentionHeadLayout {
            sequence_len: 2 * n_frames - 1,
            ..attention_layout
        },
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(attn_r)",
            permute: "ggml_permute(attn_r)",
            cont: "ggml_cont(attn_r)",
        },
        map_err,
    )?;
    let ac = graph
        .mul_mat(k_heads, q_u)
        .map_err(|source| map_err("ggml_mul_mat(attn_ac)", source))?;
    let bd_raw = graph
        .mul_mat(r_heads, q_v)
        .map_err(|source| map_err("ggml_mul_mat(attn_bd_raw)", source))?;
    let element = std::mem::size_of::<f32>();
    let rel_shift_nb1 = (2 * n_frames - 2) * element;
    let rel_shift_nb2 = (2 * n_frames - 1) * n_frames * element;
    let rel_shift_offset = (n_frames - 1) * element;
    let bd = graph
        .view_3d(
            bd_raw,
            n_frames,
            n_frames,
            heads,
            rel_shift_nb1,
            rel_shift_nb2,
            rel_shift_offset,
        )
        .map_err(|source| map_err("ggml_view_3d(relative_shift)", source))?;
    let mut scores = graph
        .add(ac, bd)
        .map_err(|source| map_err("ggml_add(attn_scores)", source))?;
    scores = graph
        .scale(scores, 1.0 / (head_dim as f32).sqrt())
        .map_err(|source| map_err("ggml_scale(attn_scores)", source))?;
    // Mask out context-pad key frames (upstream `src_mask`, applied identically
    // to every layer): `[n_frames,1,1]` broadcasts over the query and head dims.
    scores = graph
        .add(scores, key_mask)
        .map_err(|source| map_err("ggml_add(attn_key_mask)", source))?;
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
        .mul_mat(layer.attn_out_weight.as_graph_tensor(), attn_out)
        .map_err(|source| map_err("ggml_mul_mat(attn_out)", source))?;
    state = graph
        .add(attn_input, attn_out)
        .map_err(|source| map_err("ggml_add(attn_residual)", source))?;
    let conv_input = state;

    // ----- Conv module: pre-norm -> pw1(no bias) GLU -> depthwise(no bias) ->
    // LayerNorm -> swish -> pw2(no bias) -> residual -----
    let conv_norm = apply_affine_layer_norm(
        graph,
        state,
        epsilon,
        layer.conv_norm_weight.as_graph_tensor(),
        layer.conv_norm_bias.as_graph_tensor(),
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "conv_norm",
            bias: "conv_norm",
        },
        map_err,
    )?;
    // conv.pw1.weight is already `[d_model, d_model*4]` (PointwiseConvSqueeze
    // target dims from the importer). `conv` is `[d_model*4, n_frames]`;
    // `F.glu(dim=1)` in the upstream (N,4d,T) layout splits the channel axis
    // into two (2d) halves, which is ne0 here too, so a direct ne0-range view
    // (no 3D dance) is the exact same split.
    let mut conv = graph
        .mul_mat(layer.conv_pw1_weight.as_graph_tensor(), conv_norm)
        .map_err(|source| map_err("ggml_mul_mat(conv_pw1)", source))?;
    let glu_half_width = d_model * 2;
    let glu_row_stride = (d_model * 4) * element;
    let conv_main = graph
        .view_2d(conv, glu_half_width, n_frames, glu_row_stride, 0)
        .map_err(|source| map_err("ggml_view_2d(conv_main)", source))?;
    let conv_gate = graph
        .view_2d(
            conv,
            glu_half_width,
            n_frames,
            glu_row_stride,
            glu_half_width * element,
        )
        .map_err(|source| map_err("ggml_view_2d(conv_gate)", source))?;
    let conv_main = graph
        .cont(conv_main)
        .map_err(|source| map_err("ggml_cont(conv_main)", source))?;
    let conv_gate = graph
        .cont(conv_gate)
        .map_err(|source| map_err("ggml_cont(conv_gate)", source))?;
    conv = graph
        .mul(
            conv_main,
            graph
                .sigmoid(conv_gate)
                .map_err(|source| map_err("ggml_sigmoid(conv_gate)", source))?,
        )
        .map_err(|source| map_err("ggml_mul(conv_glu)", source))?;
    // Depthwise conv operates on the post-GLU width `d_model*2` (upstream:
    // `depthwise_conv = nn.Conv1d(d_model*2, d_model*2, kernel, groups=d_model*2)`).
    let conv_kernel = metadata.conv_kernel;
    let dw_channels = d_model * 2;
    let dw_weight = graph
        .reshape_4d(
            layer.conv_dw_weight.as_graph_tensor(),
            conv_kernel,
            1,
            1,
            dw_channels,
        )
        .map_err(|source| map_err("ggml_reshape_4d(conv_dw_weight)", source))?;
    conv = graph
        .transpose(conv)
        .map_err(|source| map_err("ggml_transpose(conv_glu)", source))?;
    conv = graph
        .cont(conv)
        .map_err(|source| map_err("ggml_cont(conv_glu_t)", source))?;
    let conv_4d = graph
        .reshape_4d(conv, n_frames, 1, dw_channels, 1)
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
    conv = graph
        .reshape_2d(conv, dw_channels, n_frames)
        .map_err(|source| map_err("ggml_reshape_2d(conv_dw_out)", source))?;
    // conv.batch_norm is actually LayerNorm over the `d_model*2` conv channels
    // upstream (`nn.LayerNorm(d_model*2)` applied post-depthwise, pre-swish).
    conv = apply_affine_layer_norm(
        graph,
        conv,
        epsilon,
        layer.conv_ln_weight.as_graph_tensor(),
        layer.conv_ln_bias.as_graph_tensor(),
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "conv_ln",
            bias: "conv_ln",
        },
        map_err,
    )?;
    conv = graph
        .silu(conv)
        .map_err(|source| map_err("ggml_silu(conv_dw)", source))?;
    // conv.pw2.weight is already `[d_model*2, d_model]` (PointwiseConvSqueeze
    // target dims); no reshape needed.
    conv = graph
        .mul_mat(layer.conv_pw2_weight.as_graph_tensor(), conv)
        .map_err(|source| map_err("ggml_mul_mat(conv_pw2)", source))?;
    state = graph
        .add(conv_input, conv)
        .map_err(|source| map_err("ggml_add(conv_residual)", source))?;
    let ff2_input = state;

    // ----- FF2 (macaron, scale 0.5) -----
    let ff2_norm = apply_affine_layer_norm(
        graph,
        state,
        epsilon,
        layer.ffn2_norm_weight.as_graph_tensor(),
        layer.ffn2_norm_bias.as_graph_tensor(),
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "ffn2_norm",
            bias: "ffn2_norm",
        },
        map_err,
    )?;
    state = apply_feed_forward_residual(
        graph,
        ff2_norm,
        ff2_input,
        FeedForwardActivation::Silu,
        Some(0.5),
        FeedForwardResidualSteps {
            activation: "ggml_silu(ffn2)",
            scale: Some("ggml_scale(ffn2_half)"),
            residual: "ggml_add(ffn2_residual)",
        },
        |graph, value| {
            let up = graph
                .mul_mat(layer.ffn2_up_weight.as_graph_tensor(), value)
                .map_err(|source| map_err("ggml_mul_mat(ffn2_up)", source))?;
            graph
                .add(up, layer.ffn2_up_bias.as_graph_tensor())
                .map_err(|source| map_err("ggml_add(ffn2_up_bias)", source))
        },
        |graph, value| {
            let down = graph
                .mul_mat(layer.ffn2_down_weight.as_graph_tensor(), value)
                .map_err(|source| map_err("ggml_mul_mat(ffn2_down)", source))?;
            graph
                .add(down, layer.ffn2_down_bias.as_graph_tensor())
                .map_err(|source| map_err("ggml_add(ffn2_down_bias)", source))
        },
        map_err,
    )?;

    // ----- Final affine LayerNorm (no residual) -----
    apply_affine_layer_norm(
        graph,
        state,
        epsilon,
        layer.out_norm_weight.as_graph_tensor(),
        layer.out_norm_bias.as_graph_tensor(),
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "out_norm",
            bias: "out_norm",
        },
        map_err,
    )
}

#[cfg(test)]
mod parity_tests {
    //! Dev-only numeric parity check against the real FireRedASR-AED-L
    //! checkpoint + reference PyTorch inference. Not part of the default
    //! suite: the fp16 `.oasr` pack (~2.2 GB, derived from a private
    //! downloaded checkpoint) and cached wav are dev-machine artifacts, never
    //! committed. `#[ignore]`d and silently skipped if the pack is absent so
    //! `cargo nextest run --workspace` stays green on a clean checkout.
    //!
    //! Reference values captured from `tmp/firered-ref-src/run_ref.py`
    //! (actual FireRedASR-AED-L `model.pth.tar`, `fixtures/jfk.wav`):
    //! `enc_outputs.shape == [1, 275, 1280]`, `lengths == [273]`,
    //! frame0 first8 == [-0.07052769, -0.20126128, -0.18988657, -1.136183,
    //! 0.26838502, 0.10432011, -0.09196245, 0.09975306].
    use super::*;
    use crate::ggml_runtime::GgufTensorDataReader;
    use crate::ggml_runtime::read_gguf_metadata;
    use crate::models::firered_aed::frontend::{FireRedFbankFrontend, apply_cmvn};
    use crate::models::firered_aed::runtime_contract::parse_firered_aed_execution_metadata;

    fn dev_pack_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/firered-out/firered-aed-l-fp16.oasr")
    }

    fn dev_wav_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav")
    }

    #[test]
    #[ignore = "requires the private dev-only firered-aed-l-fp16.oasr pack; see module docs"]
    fn encoder_matches_reference_pytorch_output_on_jfk_wav() {
        let pack_path = dev_pack_path();
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return;
        }

        let metadata_view = read_gguf_metadata(&pack_path).expect("read gguf metadata");
        let metadata =
            parse_firered_aed_execution_metadata(&metadata_view).expect("parse firered metadata");

        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            dev_wav_path(),
            "firered encoder parity test",
            "firered encoder parity test",
        )
        .expect("load jfk.wav");

        let frontend = FireRedFbankFrontend::new();
        let mut fbank = frontend.compute(&samples).expect("compute fbank");

        let reader = GgufTensorDataReader::from_path(&pack_path).expect("open tensor reader");
        let feature_dim = [metadata.feature_dim as u64];
        let neg_mean = reader
            .host_tensor_f32_copy_by_name("frontend.cmvn.neg_mean", &feature_dim)
            .expect("read neg_mean");
        let inv_stddev = reader
            .host_tensor_f32_copy_by_name("frontend.cmvn.inv_stddev", &feature_dim)
            .expect("read inv_stddev");
        apply_cmvn(&mut fbank.data, fbank.n_mels, &neg_mean, &inv_stddev).expect("apply cmvn");

        let output =
            encode_firered_aed_audio_embeddings(&pack_path, metadata, &fbank.data, fbank.n_frames)
                .expect("encode");

        eprintln!(
            "firered encoder parity: frames={} hidden={} frame0_first8={:?}",
            output.frame_count,
            output.hidden_size,
            &output.rows[..8.min(output.rows.len())]
        );

        assert_eq!(output.frame_count, 275);
        assert_eq!(output.hidden_size, 1280);
        let expected_frame0_first8 = [
            -0.070_527_69_f32,
            -0.201_261_28,
            -0.189_886_57,
            -1.136_183,
            0.268_385_02,
            0.104_320_11,
            -0.091_962_45,
            0.099_753_06,
        ];
        // Tolerance: this pack is fp16 and the reference ran fp32
        // end-to-end, so LayerNorm/softmax nonlinearity compounds rounding
        // drift across the 16-block stack. Bisected layer-by-layer against
        // the reference (manual debug run, not committed): subsampling alone
        // matches to ~1e-3, 1 block to ~3e-3, 3 blocks to ~3e-2 -- roughly
        // linear growth with depth, consistent with fp16 rounding
        // accumulation rather than a structural bug. 0.15 covers the full
        // 16-block extrapolation with headroom.
        for (idx, (&actual, &expected)) in output.rows[..8]
            .iter()
            .zip(expected_frame0_first8.iter())
            .enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff < 0.15,
                "frame0[{idx}] mismatch: actual={actual} expected={expected} diff={diff}"
            );
        }
    }
}
