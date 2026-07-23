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
//! shared Transformer-XL formula (same ESPnet/WeNet lineage), so the shared
//! `nn::encoder::build_relative_positional_encoding` helper is reused directly
//! instead of re-derived.

#![allow(dead_code)]

use std::path::Path;

use thiserror::Error;

use crate::ggml_runtime::{GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlLoadedWeightContext};
use crate::nn::attn::{
    AttentionHeadLayout, AttentionReshapeSteps, AttentionValueMergeSteps,
    STANDARD_HEAD_PERMUTE_AXES, attention_context_from_probs,
    reshape_projection_to_attention_heads,
};
use crate::nn::conv::{
    Conv2dParams, ConvActivation, ConvBlockSteps, apply_conv_2d_bias_activation, reshape_bias_4d,
};
use crate::nn::encoder::build_relative_positional_encoding;
use crate::nn::ffn::{
    FeedForwardActivation, FeedForwardResidualSteps, apply_feed_forward_residual,
};
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

use super::encoder_weights::{
    FireRedEncoderLayerWeights, FireRedEncoderWeights, FireRedEncoderWeightsError,
};
use super::graph_config::firered_encoder_graph_config;
use super::runtime_contract::FireRedAedExecutionMetadata;

const FIRERED_ENCODER_LAYER_NORM_EPSILON: f32 = 1.0e-5;
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

/// Owns the encoder's mmap'd weight context plus the ggml graph runner across
/// calls, so a caller (see the thread-local cache in `executor.rs`) can reuse
/// the same runtime for every transcription on this pack instead of
/// re-loading the GGUF weight context from scratch each time -- the encoder
/// forward itself stays a fresh single-shot graph per call (no incremental
/// reuse across time steps, matching cohere's/parakeet's shape).
pub(crate) struct FireRedEncoderGraphRuntime {
    runner: GgmlCpuGraphRunner,
    _loaded: GgmlLoadedWeightContext,
    weights: FireRedEncoderWeights,
    metadata: FireRedAedExecutionMetadata,
}

impl FireRedEncoderGraphRuntime {
    pub(crate) fn new(
        runtime_path: &Path,
        metadata: FireRedAedExecutionMetadata,
    ) -> Result<Self, FireRedEncoderError> {
        let runner = GgmlCpuGraphRunner::new(firered_encoder_graph_config())
            .map_err(|source| map_err("runner_init", source))?;
        let loaded = runner
            .load_gguf_weight_context(runtime_path)
            .map_err(|source| map_err("load_gguf_weight_context", source))?;
        let weights = FireRedEncoderWeights::load(&loaded, metadata.encoder_n_layers)?;
        Ok(Self {
            runner,
            _loaded: loaded,
            weights,
            metadata,
        })
    }

    pub(crate) fn encode(
        &mut self,
        cmvn_features: &[f32],
        n_frames: usize,
    ) -> Result<FireRedEncoderOutput, FireRedEncoderError> {
        encode_firered_aed_audio_embeddings(
            &mut self.runner,
            &self.weights,
            self.metadata,
            cmvn_features,
            n_frames,
        )
    }
}

/// Post-subsampling encoder time-frame count the 2x Conv2d(k3,s2) stem
/// produces for `n_frames` raw (pre-context-pad) fbank frames -- the same
/// time-axis arithmetic `encode_firered_aed_audio_embeddings` runs, factored
/// out so a caller can predict the frame count a window will encode to
/// *before* building the graph (see `FireRedAedGgmlExecutor`'s PE-capacity
/// preflight check, which must reject an oversized window with a typed error
/// rather than let it reach `ggml_prepare_outputs` and fail on an opaque
/// allocation error).
pub(crate) fn predicted_encoder_time_frames(n_frames: usize) -> Result<usize, FireRedEncoderError> {
    if n_frames == 0 {
        return Err(FireRedEncoderError::EmptyInput);
    }
    let padded_frames = n_frames
        .checked_add(SUBSAMPLE_CONTEXT_PAD_FRAMES)
        .ok_or(FireRedEncoderError::ShapeOverflow)?;
    let conv1_time = conv_out_dim(padded_frames, 3, 2)?;
    conv_out_dim(conv1_time, 3, 2)
}

/// Run the full encoder forward pass in a single ggml graph (no incremental
/// reuse -- matches cohere's/parakeet's single-shot encoder shape) against an
/// already-loaded runner/weights pair.
fn encode_firered_aed_audio_embeddings(
    runner: &mut GgmlCpuGraphRunner,
    weights: &FireRedEncoderWeights,
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

    // Zero-pad the time axis by `context - 1` frames (matches
    // `F.pad(padded_input, (0,0,0,context-1))`), then run the 2x Conv2d(k3,s2)
    // stem.
    let padded_frames = n_frames
        .checked_add(SUBSAMPLE_CONTEXT_PAD_FRAMES)
        .ok_or(FireRedEncoderError::ShapeOverflow)?;
    let mut padded = vec![0.0_f32; padded_frames * feature_dim];
    padded[..cmvn_features.len()].copy_from_slice(cmvn_features);

    let conv1_freq = conv_out_dim(feature_dim, 3, 2)?;
    let conv2_freq = conv_out_dim(conv1_freq, 3, 2)?;
    if conv2_freq * metadata.subsample_channels != metadata.subsample_out_dim {
        return Err(FireRedEncoderError::ShapeOverflow);
    }
    let frame_count = predicted_encoder_time_frames(n_frames)?;
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

    let positional_values =
        build_relative_positional_encoding(metadata.d_model, frame_count, || {
            FireRedEncoderError::ShapeOverflow
        })?;
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
    // Peak-RSS lever (mirrors the cohere/moonshine encoder path): allocate the
    // forward graph through the scheduler's gallocr (liveness-based buffer
    // REUSE) BEFORE uploading inputs, collapsing the per-conformer-layer
    // intermediate accumulation to the working-set peak instead of giving every
    // non-view tensor its own allocation. The three inputs below are marked
    // `set_input`, so gallocr keeps them live across the whole graph.
    graph
        .prepare_outputs_for_upload(&[state])
        .map_err(|source| map_err("ggml_prepare_outputs(encoder)", source))?;
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
    // No `ggml_cont` here: `attention_context_from_probs` immediately re-permutes
    // this tensor (its own `value_permute` step) before the value_cont that
    // materializes it, so the two permutes compose into one and only the final
    // `ggml_cont` needs to run.
    let v_heads = reshape_projection_to_attention_heads(
        graph,
        v,
        attention_layout,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
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
    // No `ggml_cont` needed for either GLU half: `ggml_sigmoid` is elementwise and
    // only requires `is_contiguous_rows` (satisfied here since nb0 on both views is
    // still the element size), and `ggml_mul` accepts a strided src0 against a
    // freshly-allocated (contiguous) dst.
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
        .depthwise_conv_2d(dw_weight, conv_4d, 1, 1, (conv_kernel - 1) / 2, 0, 1, 1)
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

// ---------------------------------------------------------------------------
// Test-only layer-tap instrumentation (bisection harness for the 2026-07-23
// v2 checkpoint parity investigation; zero release-path impact, entirely
// `#[cfg(test)]`-gated -- see `parity_tests` below for how it's driven).
// ---------------------------------------------------------------------------

/// The five named tap points inside one Conformer block, in forward order.
/// Mirrors the `attn_input` / `conv_input` / `ff2_input` local bindings
/// `firered_conformer_block` already has, plus one more (`ffn2_out`) that the
/// production function doesn't need to bind because it feeds straight into
/// the final norm.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct FireRedEncoderLayerTaps<'a> {
    /// After macaron FF1 residual (`x + 0.5*ffn1(x)`), before attention.
    pub ffn1_out: crate::ggml_runtime::GgmlCpuTensor<'a>,
    /// After the rel-pos MHSA residual, before the conv module.
    pub attn_out: crate::ggml_runtime::GgmlCpuTensor<'a>,
    /// After the conv module residual, before macaron FF2.
    pub conv_out: crate::ggml_runtime::GgmlCpuTensor<'a>,
    /// After macaron FF2 residual, before the block's final LayerNorm.
    pub ffn2_out: crate::ggml_runtime::GgmlCpuTensor<'a>,
    /// The block's actual output (after the final LayerNorm) -- identical to
    /// what `firered_conformer_block` returns for the same inputs/weights.
    pub block_out: crate::ggml_runtime::GgmlCpuTensor<'a>,
}

/// Test-only twin of `firered_conformer_block` that additionally exposes the
/// four intra-block tap points, so a bisection test can pin down *which*
/// sub-step first diverges from the PyTorch reference instead of only
/// knowing "somewhere in block N". Delegates to the exact same shared
/// primitives (`apply_affine_layer_norm`, `apply_feed_forward_residual`, the
/// attention helpers) as the production function -- only the block-level
/// orchestration is duplicated, not the math.
#[cfg(test)]
#[allow(clippy::too_many_lines)]
fn firered_conformer_block_with_taps<'a>(
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    input: crate::ggml_runtime::GgmlCpuTensor<'a>,
    pos_enc: crate::ggml_runtime::GgmlCpuTensor<'a>,
    key_mask: crate::ggml_runtime::GgmlCpuTensor<'a>,
    metadata: FireRedAedExecutionMetadata,
    n_frames: usize,
    layer: &FireRedEncoderLayerWeights,
) -> Result<FireRedEncoderLayerTaps<'a>, FireRedEncoderError> {
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
    let ffn1_out = state;
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
    let mut attn_out_proj = attention_context_from_probs(
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
    attn_out_proj = graph
        .mul_mat(layer.attn_out_weight.as_graph_tensor(), attn_out_proj)
        .map_err(|source| map_err("ggml_mul_mat(attn_out)", source))?;
    state = graph
        .add(attn_input, attn_out_proj)
        .map_err(|source| map_err("ggml_add(attn_residual)", source))?;
    let attn_out = state;
    let conv_input = state;

    // ----- Conv module -----
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
        .depthwise_conv_2d(dw_weight, conv_4d, 1, 1, (conv_kernel - 1) / 2, 0, 1, 1)
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
    conv = graph
        .mul_mat(layer.conv_pw2_weight.as_graph_tensor(), conv)
        .map_err(|source| map_err("ggml_mul_mat(conv_pw2)", source))?;
    state = graph
        .add(conv_input, conv)
        .map_err(|source| map_err("ggml_add(conv_residual)", source))?;
    let conv_out = state;
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
    let ffn2_out = state;

    // ----- Final affine LayerNorm (no residual) -----
    let block_out = apply_affine_layer_norm(
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
    )?;

    Ok(FireRedEncoderLayerTaps {
        ffn1_out,
        attn_out,
        conv_out,
        ffn2_out,
        block_out,
    })
}

/// Test-only twin of `encode_firered_aed_audio_embeddings` for the bisection
/// harness: builds the identical graph (subsampling + 16 Conformer blocks),
/// but (a) taps every block's final output (for the "which layer first
/// diverges" scan) and (b) if `tap_layer_idx` is `Some`, additionally taps
/// the four intra-block points of that one layer (for the "which sub-step"
/// follow-up). All requested tensors are read back in a single graph
/// execution via `compute_outputs_f32`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_with_layer_taps(
    runner: &mut GgmlCpuGraphRunner,
    weights: &FireRedEncoderWeights,
    metadata: FireRedAedExecutionMetadata,
    cmvn_features: &[f32],
    n_frames: usize,
    tap_layer_idx: Option<usize>,
) -> Result<FireRedEncoderTapDump, FireRedEncoderError> {
    if n_frames == 0 {
        return Err(FireRedEncoderError::EmptyInput);
    }
    let feature_dim = metadata.feature_dim;
    if cmvn_features.len() != n_frames * feature_dim {
        return Err(FireRedEncoderError::ShapeOverflow);
    }

    let padded_frames = n_frames
        .checked_add(SUBSAMPLE_CONTEXT_PAD_FRAMES)
        .ok_or(FireRedEncoderError::ShapeOverflow)?;
    let mut padded = vec![0.0_f32; padded_frames * feature_dim];
    padded[..cmvn_features.len()].copy_from_slice(cmvn_features);

    let conv1_freq = conv_out_dim(feature_dim, 3, 2)?;
    let conv2_freq = conv_out_dim(conv1_freq, 3, 2)?;
    if conv2_freq * metadata.subsample_channels != metadata.subsample_out_dim {
        return Err(FireRedEncoderError::ShapeOverflow);
    }
    let frame_count = predicted_encoder_time_frames(n_frames)?;
    let valid_frame_count = conv_out_dim(conv_out_dim(n_frames, 3, 2)?, 3, 2)?.min(frame_count);

    let mut graph = runner.start_graph();
    let mel = graph
        .new_tensor_2d_f32(feature_dim, padded_frames, "firered_enc_mel")
        .map_err(|source| map_err("ggml_new_tensor_2d(mel)", source))?;
    graph
        .set_input(mel)
        .map_err(|source| map_err("ggml_set_input(mel)", source))?;

    let positional_values =
        build_relative_positional_encoding(metadata.d_model, frame_count, || {
            FireRedEncoderError::ShapeOverflow
        })?;
    let pos_enc = graph
        .new_tensor_2d_f32(metadata.d_model, 2 * frame_count - 1, "firered_enc_rel_pos")
        .map_err(|source| map_err("ggml_new_tensor_2d(pos_enc)", source))?;
    graph
        .set_input(pos_enc)
        .map_err(|source| map_err("ggml_set_input(pos_enc)", source))?;

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

    let subsample_out = state;
    let mut block_outputs: Vec<crate::ggml_runtime::GgmlCpuTensor<'_>> =
        Vec::with_capacity(weights.layers.len());
    let mut intra_block_taps: Option<FireRedEncoderLayerTaps<'_>> = None;
    for (idx, layer) in weights.layers.iter().enumerate() {
        if Some(idx) == tap_layer_idx {
            let taps = firered_conformer_block_with_taps(
                &mut graph,
                state,
                pos_enc,
                key_mask,
                metadata,
                frame_count,
                layer,
            )?;
            state = taps.block_out;
            intra_block_taps = Some(taps);
        } else {
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
        block_outputs.push(state);
    }

    let mut all_outputs = vec![subsample_out];
    all_outputs.extend(block_outputs.iter().copied());
    if let Some(taps) = &intra_block_taps {
        all_outputs.extend([
            taps.ffn1_out,
            taps.attn_out,
            taps.conv_out,
            taps.ffn2_out,
            taps.block_out,
        ]);
    }
    // Every tap must be marked `ggml_set_output` (not just passed to
    // `build_forward_graph`/`prepare_outputs_for_upload`, which only adds it
    // as a graph root): without the OUTPUT flag, gallocr's liveness-based
    // buffer reuse is free to recycle a tap's buffer for a later tensor once
    // its last in-graph consumer has read it, and a subsequent readback sees
    // whatever later computation overwrote it instead of the real tap value.
    for &tensor in &all_outputs {
        graph
            .set_output(tensor)
            .map_err(|source| map_err("ggml_set_output(encoder_tap)", source))?;
    }

    graph
        .prepare_outputs_for_upload(&all_outputs)
        .map_err(|source| map_err("ggml_prepare_outputs(encoder_taps)", source))?;
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
    // `subsample_out` is `state` AFTER the subsample stem's `self.out` Linear
    // projection (matches Python's `embed_output`, already d_model-wide, not
    // the raw pre-projection `subsample_out_dim`-wide conv output).
    let subsample_len = expected_len;

    let mut requests: Vec<(crate::ggml_runtime::GgmlCpuTensor<'_>, usize)> =
        vec![(subsample_out, subsample_len)];
    requests.extend(block_outputs.iter().map(|&t| (t, expected_len)));
    if let Some(taps) = &intra_block_taps {
        requests.extend([
            (taps.ffn1_out, expected_len),
            (taps.attn_out, expected_len),
            (taps.conv_out, expected_len),
            (taps.ffn2_out, expected_len),
            (taps.block_out, expected_len),
        ]);
    }

    let mut computed = graph.compute_outputs_f32(&requests).map_err(|error| {
        FireRedEncoderError::GraphExecutionFailed {
            reason: error.to_string(),
        }
    })?;

    let mut iter = computed.drain(..);
    let subsample_rows = iter.next().expect("subsample_out present");
    let block_rows: Vec<Vec<f32>> = (0..block_outputs.len())
        .map(|_| iter.next().expect("block output present"))
        .collect();
    let intra_taps = if intra_block_taps.is_some() {
        Some(FireRedEncoderIntraBlockDump {
            ffn1_out: iter.next().expect("ffn1_out present"),
            attn_out: iter.next().expect("attn_out present"),
            conv_out: iter.next().expect("conv_out present"),
            ffn2_out: iter.next().expect("ffn2_out present"),
            block_out: iter.next().expect("block_out present"),
        })
    } else {
        None
    };

    Ok(FireRedEncoderTapDump {
        frame_count,
        hidden_size: metadata.d_model,
        subsample_out_dim: metadata.subsample_out_dim,
        subsample_rows,
        block_rows,
        intra_taps,
    })
}

/// Owned bisection dump: every block's final output plus, if requested, the
/// intra-block taps of one layer -- all as row-major `[frame][hidden]` f32.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct FireRedEncoderTapDump {
    pub frame_count: usize,
    pub hidden_size: usize,
    pub subsample_out_dim: usize,
    pub subsample_rows: Vec<f32>,
    /// `block_rows[i]` is Conformer block `i`'s final output (0-indexed).
    pub block_rows: Vec<Vec<f32>>,
    pub intra_taps: Option<FireRedEncoderIntraBlockDump>,
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct FireRedEncoderIntraBlockDump {
    pub ffn1_out: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub conv_out: Vec<f32>,
    pub ffn2_out: Vec<f32>,
    pub block_out: Vec<f32>,
}

#[cfg(test)]
mod frame_count_tests {
    use super::predicted_encoder_time_frames;

    #[test]
    fn zero_frames_is_empty_input_error() {
        assert!(matches!(
            predicted_encoder_time_frames(0),
            Err(super::FireRedEncoderError::EmptyInput)
        ));
    }

    #[test]
    fn matches_the_inline_arithmetic_encode_used_before_extraction() {
        // Same conv_out_dim(conv_out_dim(n+6,3,2),3,2) the encoder forward
        // itself runs; regression-pins the extraction in
        // `predicted_encoder_time_frames` against silent drift.
        for n_frames in [1usize, 10, 273, 1000, 20_000] {
            let padded = n_frames + 6;
            let conv1 = (padded - 3) / 2 + 1;
            let expected = (conv1 - 3) / 2 + 1;
            assert_eq!(
                predicted_encoder_time_frames(n_frames).unwrap(),
                expected,
                "mismatch for n_frames={n_frames}"
            );
        }
    }

    /// A 210s window (16 kHz mono, 10ms fbank hop) predicts an encoder frame
    /// count comfortably past the `pe_len=9999` -> 5000-frame capacity
    /// (~200s) declared in `runtime_contract`'s parsed-metadata test --
    /// this is the exact shape `FireRedAedGgmlExecutor::execute_inner`'s
    /// PE-capacity preflight check (issue #158) must catch before ever
    /// building the encoder graph.
    #[test]
    fn oversized_window_predicts_past_pe_capacity() {
        let raw_fbank_frames = 210 * 100; // 10ms hop => 100 frames/sec
        let predicted = predicted_encoder_time_frames(raw_fbank_frames).unwrap();
        assert!(
            predicted > 5000,
            "expected a 210s window to predict past the 5000-frame PE capacity, got {predicted}"
        );
    }
}

#[cfg(test)]
mod parity_tests {
    //! Dev-only numeric parity check against the real FireRedASR2-AED
    //! checkpoint + reference PyTorch inference. Not part of the default
    //! suite: the fp16 `.oasr` pack (~2.2 GB, derived from a private
    //! downloaded checkpoint) and cached wav are dev-machine artifacts, never
    //! committed. `#[ignore]`d and silently skipped if the pack is absent so
    //! `cargo nextest run --workspace` stays green on a clean checkout.
    //!
    //! ## Pin history (read this before touching the constants below)
    //!
    //! This pin was originally captured 2026-07 (#32) against **v1**
    //! (`FireRedTeam/FireRedASR-AED-L`) in the same commit that swapped the
    //! shipped importer/runtime over to **v2** (`FireRedTeam/FireRedASR2-AED`,
    //! 8667-token vocab) -- the reference was never regenerated for the
    //! checkpoint the test actually exercises, so it silently pinned the
    //! wrong model's numbers from day one. Re-pinned 2026-07-23 against the
    //! real v2 checkpoint (see below); if you ever port to a v3 checkpoint,
    //! **regenerate this pin against v3**, do not carry it forward.
    //!
    //! ## v2 reference generation (2026-07-23)
    //!
    //! - Checkpoint: `FireRedTeam/FireRedASR2-AED` `model.pth.tar`
    //!   (sha256 `4677cbd30988d63ed3e777f6a42a1e5260a3865317f6e15e488bef40954f7054`,
    //!   matches the HF-hosted LFS blob).
    //! - Official code: `FireRedTeam/FireRedASR2S` pinned commit
    //!   `4e7d9aaf4482a47cec1724807026b9b151926eb5`.
    //! - Generator: `tooling/firered2-reference-dumper/dump_aed_encoder.py`
    //!   (see that tool's README for the full command).
    //! - `enc_outputs.shape == [1, 275, 1280]`, `lengths == [275]`.
    //! - Full frame-0 (all 1280 dims, not just the first 8) is committed as a
    //!   5 KB fixture, `testdata/firered_aed_v2_encoder_frame0_reference.f32`
    //!   (row-major f32, little-endian) -- see "assertion design" below for
    //!   why the pin now covers the whole vector instead of 8 values.
    //!
    //! ## Evidence chain for the tolerance (do not re-tighten without re-reading this)
    //!
    //! 1. **fp16-weight-storage discriminator**: round-tripping every loaded
    //!    encoder weight through `w.half().float()` before running the pure
    //!    PyTorch reference (`dump_aed_encoder.py --fp16-weights`) perturbs
    //!    frame0 by at most **8.5e-4** -- three orders of magnitude below the
    //!    residual against our `.oasr` pack. Weight storage precision is NOT
    //!    the source of the gap.
    //! 2. **Relative positional encoding**: `build_relative_positional_encoding`
    //!    vs the official `RelPositionalEncoding.forward` for T=275/d_model=1280
    //!    match to max diff **3.0e-5** (702,720 values compared) -- pure fp32
    //!    noise, ruled out as a suspect.
    //! 3. **Per-block bisection** (subsample stem + all 16 Conformer blocks,
    //!    valid frames only, mean `|diff|` over the whole `[273, 1280]` tensor):
    //!    subsample 0.0006 -> block00 0.0057 -> block01 0.0122 -> block02
    //!    0.0300 -> ... -> block13 0.1081 -> **block14 0.1374** -> block15
    //!    (final output) **0.0208**. Growth is roughly a steady ~1.1-1.3x per
    //!    block (consistent with compounding fp32 rounding through repeated
    //!    LayerNorm/softmax nonlinearities), until the *final block's own
    //!    output LayerNorm* compresses the accumulated error back down --
    //!    every block ends in its own affine LayerNorm, and LayerNorm
    //!    (dividing by per-frame variance) is exactly the operation that
    //!    claws back a scale-like perturbation.
    //! 4. **Intra-block14 tap breakdown** (ffn1_out -> attn_out -> conv_out ->
    //!    ffn2_out -> block_out, the four sub-steps plus the block's own
    //!    final norm) found the block14 `max` spike (31.8, vs ~4 in
    //!    neighboring blocks) traces to `ffn2_out` (post FF2 residual, PRE the
    //!    block's own final LayerNorm): max diff **297.7** at one specific
    //!    (frame, channel) position, immediately compressed to 31.8 by that
    //!    same block's `out_norm`. Inspecting the raw activations there found
    //!    the textbook "massive activations" phenomenon (a small number of
    //!    channels a trained transformer/conformer uses as an implicit
    //!    bias/attention-sink, reaching values 100-1000x the typical
    //!    magnitude): **17 of 1280 channels** in `ffn2_out` reach absolute
    //!    values of 500-3439 (median |activation| across the tensor is only
    //!    2.2) -- e.g. channel 237 is rust=2756.16 vs the PyTorch reference's
    //!    3053.89, channel 1141 is rust=-59.83 vs -66.38. Both implementations
    //!    land on the *same order of magnitude and sign* at these channels;
    //!    they are not missing or wrong, they are numerically delicate: any
    //!    ~1% relative discrepancy from ggml's vs PyTorch's non-bit-identical
    //!    fp32 reduction order (`mul_mat`/`softmax`/`LayerNorm` do not sum in
    //!    the same order across two independently-implemented kernels) turns
    //!    into a double- or triple-digit *absolute* diff exactly at these
    //!    channels, while the other ~1260 channels stay at 1e-2-1e-1.
    //! 5. **Cross-thread-count self-noise baseline**: re-ran the pure PyTorch
    //!    reference twice on the same machine with `OMP_NUM_THREADS`/
    //!    `MKL_NUM_THREADS` set to 1 vs 8. Every block's full output,
    //!    including the hot channels, was **bit-identical** (max diff
    //!    0.0 across all 16 blocks) between the two runs. This CPU/PyTorch
    //!    build has zero self-noise from thread-count at this shape -- so the
    //!    residual we see is not generic "any two fp32 runs jitter a bit", it
    //!    specifically requires crossing implementations (ggml's kernels vs
    //!    PyTorch's), consistent with (4)'s reduction-order explanation and
    //!    ruling out thread-count nondeterminism as a contributor.
    //!
    //! **Conclusion**: the residual is real, reproducible, and fully
    //! explained by non-bit-identical fp32 reductions landing on a handful of
    //! massive-activation channels this checkpoint happens to have -- not a
    //! missing operation, wrong shape, sign error, or precision bug on either
    //! side. There is nothing to "fix": forcing bit-identical reduction order
    //! against an arbitrary PyTorch build is not a goal this runtime has ever
    //! had elsewhere (whisper/qwen/cohere/etc. all carry similar fp32-vs-fp32
    //! tolerances). **Do not "fix" a future failure here by chasing exact
    //! reduction-order parity with PyTorch** -- if this test ever starts
    //! failing, re-run the bisection harness below first (does the failure
    //! look like this evidence chain, i.e. concentrated at a few channels and
    //! compressed by the next LayerNorm? or does it look like a real
    //! structural regression, i.e. spread evenly and NOT compressed by
    //! LayerNorm?) before assuming a checkpoint or tolerance problem.
    //!
    //! ## Assertion design
    //!
    //! Two independent bounds over the **full 1280-dim frame0 vector** (not
    //! just 8 values -- the per-block bisection above found meaningfully
    //! larger diffs outside the first 8 dims):
    //! - `max_abs_diff < 0.4`: covers the measured 0.332 with ~20% headroom.
    //!   This bound alone is a weak regression detector, because it's exactly
    //!   the metric the massive-activation channels dominate (see (4) above)
    //!   -- a single hot channel jittering a bit more on some future
    //!   checkpoint/build can move this number without anything being wrong.
    //! - `mean_abs_diff < 0.15`: the sensitive bound. Frame0's measured mean
    //!   is 0.076 (roughly 2x the whole-tensor average of 0.021 -- frame0
    //!   happens to be a somewhat-worse-than-typical frame, not the best
    //!   case). A hot-channel-only noise source cannot move this metric much
    //!   (it's one channel out of 1280, diluted 1280x by the mean), so a real
    //!   regression -- a wrong operation, wrong sign, wrong shape, a missing
    //!   mask -- would spread error across most/all channels and move `mean`
    //!   by roughly an order of magnitude, not the ~2x headroom this bound
    //!   allows for.
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

        let mut runtime =
            FireRedEncoderGraphRuntime::new(&pack_path, metadata).expect("build encoder runtime");
        let output = runtime.encode(&fbank.data, fbank.n_frames).expect("encode");

        eprintln!(
            "firered encoder parity: frames={} hidden={} frame0_first8={:?}",
            output.frame_count,
            output.hidden_size,
            &output.rows[..8.min(output.rows.len())]
        );

        assert_eq!(output.frame_count, 275);
        assert_eq!(output.hidden_size, 1280);

        // Full frame-0 (all 1280 dims) fp32 PyTorch reference against the real
        // v2 checkpoint -- see the module doc above ("Pin history" /
        // "v2 reference generation" / "Evidence chain") for provenance and
        // why the bound is two-part (max + mean) instead of a single max.
        const REFERENCE_BYTES: &[u8] =
            include_bytes!("testdata/firered_aed_v2_encoder_frame0_reference.f32");
        assert_eq!(REFERENCE_BYTES.len(), metadata.d_model * 4);
        let expected_frame0: Vec<f32> = REFERENCE_BYTES
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("4-byte chunk")))
            .collect();

        let actual_frame0 = &output.rows[..metadata.d_model];
        let diffs: Vec<f32> = actual_frame0
            .iter()
            .zip(expected_frame0.iter())
            .map(|(&actual, &expected)| (actual - expected).abs())
            .collect();
        let max_abs_diff = diffs.iter().copied().fold(0.0_f32, f32::max);
        let mean_abs_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;
        eprintln!(
            "firered encoder parity: max_abs_diff={max_abs_diff} mean_abs_diff={mean_abs_diff}"
        );

        assert!(
            max_abs_diff < 0.4,
            "frame0 max_abs_diff too large: {max_abs_diff} (see module doc evidence chain)"
        );
        assert!(
            mean_abs_diff < 0.15,
            "frame0 mean_abs_diff too large: {mean_abs_diff} -- unlike max_abs_diff, this bound \
             is NOT dominated by massive-activation channels (one outlier channel out of 1280 \
             barely moves it), so crossing it is a real regression signal, not checkpoint noise"
        );
    }

    /// Bisection harness for the 2026-07-23 v2 residual investigation: dumps
    /// the subsample-stem output and all 16 Conformer blocks' final outputs
    /// (and, if `FIRERED_AED_TAP_LAYER` is set, one block's four intra-block
    /// taps too) to `tmp/rust_layers/` as row-major f32 files, for a python
    /// script to diff against `dump_aed_encoder.py --dump-layers-dir`. Not a
    /// correctness assertion by itself -- `--nocapture` only, paired with an
    /// out-of-band python diff. Silently skipped like the test above if the
    /// dev pack is absent.
    #[test]
    #[ignore = "bisection harness, requires the private dev-only firered-aed-l-fp16.oasr pack"]
    fn dump_encoder_layer_taps_for_v2_bisection() {
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
            "firered encoder bisection test",
            "firered encoder bisection test",
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

        let tap_layer_idx = std::env::var("FIRERED_AED_TAP_LAYER")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());

        let mut runner =
            GgmlCpuGraphRunner::new(firered_encoder_graph_config()).expect("build runner");
        let loaded = runner
            .load_gguf_weight_context(&pack_path)
            .expect("load gguf weight context");
        let weights =
            FireRedEncoderWeights::load(&loaded, metadata.encoder_n_layers).expect("load weights");

        let dump = encode_with_layer_taps(
            &mut runner,
            &weights,
            metadata,
            &fbank.data,
            fbank.n_frames,
            tap_layer_idx,
        )
        .expect("encode_with_layer_taps");

        let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/rust_layers");
        std::fs::create_dir_all(&out_dir).expect("create out_dir");

        let write_f32 = |name: &str, values: &[f32]| {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(out_dir.join(name), bytes).expect("write dump file");
        };

        write_f32("subsample_out.f32", &dump.subsample_rows);
        for (idx, rows) in dump.block_rows.iter().enumerate() {
            write_f32(&format!("block_{idx:02}.f32"), rows);
        }
        if let Some(taps) = &dump.intra_taps {
            write_f32("tap_ffn1_out.f32", &taps.ffn1_out);
            write_f32("tap_attn_out.f32", &taps.attn_out);
            write_f32("tap_conv_out.f32", &taps.conv_out);
            write_f32("tap_ffn2_out.f32", &taps.ffn2_out);
            write_f32("tap_block_out.f32", &taps.block_out);
        }

        eprintln!(
            "wrote subsample_out + {} block outputs{} to {}",
            dump.block_rows.len(),
            if dump.intra_taps.is_some() {
                " + intra-block taps"
            } else {
                ""
            },
            out_dir.display()
        );
    }
}
