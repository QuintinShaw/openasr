//! Pure Rust Zipformer2 encoder fragments used as parity scaffolding.
//!
//! The production target is GGML execution. This reference keeps operator
//! semantics explicit while the GGML graph is brought up stage by stage.

use super::encoder_ops::{apply_swoosh_l, apply_swoosh_r, bias_norm_last_dim};
use super::encoder_weights::{
    XasrConv1dWeights, XasrConv2dWeights, XasrConvolutionModuleWeights, XasrEncoderEmbedWeights,
    XasrEncoderLayerWeights, XasrLinearPairWeights, XasrLinearWithBias, XasrNonlinAttentionWeights,
    XasrSelfAttentionWeightsWeights,
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrEncoderEmbedReferenceOutput {
    pub frames: usize,
    pub dim: usize,
    /// Frame-major `[frames, dim]` rows, matching ONNX `/out_norm/Mul_1_output_0`.
    pub rows: Vec<f32>,
    /// NCHW `[1, 128, 3, 19]`, matching ONNX `new_embed_states`.
    pub new_embed_states: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrConvolutionReferenceOutput {
    /// Frame-major `[frames, dim]`, matching ONNX conv module output.
    pub rows: Vec<f32>,
    /// Channel-major `[dim, frames]`, matching the depthwise conv module's NCT
    /// output before transpose and out projection.
    pub depthwise_rows: Vec<f32>,
    /// Channel-major `[dim, kernel / 2]` updated streaming cache.
    pub new_cache: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrAttentionWeightsReferenceOutput {
    /// `[num_heads, frames, left_context + frames]`, matching ONNX Softmax
    /// without the batch axis.
    pub weights: Vec<f32>,
    /// Frame-major `[left_context, num_heads * query_head_dim]`.
    pub new_cached_key: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrValueProjectionReferenceOutput {
    /// Frame-major `[frames, dim]`.
    pub rows: Vec<f32>,
    /// Frame-major cache without batch axis.
    pub new_cache: Vec<f32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct XasrZipformerLayerReferenceCaches<'a> {
    pub cached_key: Option<&'a [f32]>,
    pub cached_nonlin_attention: Option<&'a [f32]>,
    pub cached_val1: Option<&'a [f32]>,
    pub cached_val2: Option<&'a [f32]>,
    pub cached_conv1: Option<&'a [f32]>,
    pub cached_conv2: Option<&'a [f32]>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrZipformerLayerReferenceOutput {
    /// Frame-major `[frames, dim]`, after the layer's final bypass.
    pub rows: Vec<f32>,
    pub new_cached_key: Vec<f32>,
    pub new_cached_nonlin_attention: Vec<f32>,
    pub new_cached_val1: Vec<f32>,
    pub new_cached_val2: Vec<f32>,
    pub new_cached_conv1: Vec<f32>,
    pub new_cached_conv2: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrDownsampleReferenceOutput {
    /// Frame-major padded/truncated input before factor grouping.
    pub padded_rows: Vec<f32>,
    /// Same values as `padded_rows`, matching ONNX reshape order
    /// `[out_frames, factor, 1, target_dim]`.
    pub reshaped_rows: Vec<f32>,
    /// Frame-major `[out_frames, target_dim]` weighted downsample output.
    pub rows: Vec<f32>,
    pub frames: usize,
    pub dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct Tensor4 {
    data: Vec<f32>,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
}

pub(crate) fn encode_embed_reference(
    weights: &XasrEncoderEmbedWeights,
    x: &[f32],
    frames: usize,
    feature_dim: usize,
    embed_states: Option<&[f32]>,
) -> Result<XasrEncoderEmbedReferenceOutput, String> {
    if feature_dim != 80 {
        return Err(format!(
            "xasr encoder embed expected feature_dim=80, got {feature_dim}"
        ));
    }
    if x.len() != frames * feature_dim {
        return Err(format!(
            "xasr encoder embed input has {} values, expected {}",
            x.len(),
            frames * feature_dim
        ));
    }
    let mut state = Tensor4 {
        data: x.to_vec(),
        n: 1,
        c: 1,
        h: frames,
        w: feature_dim,
    };

    state = conv2d_nchw(&state, &weights.conv0, 1, 1, (0, 1, 0, 1), 1)?;
    apply_swoosh_r(&mut state.data);
    state = conv2d_nchw(&state, &weights.conv4, 2, 2, (0, 0, 0, 0), 1)?;
    apply_swoosh_r(&mut state.data);
    state = conv2d_nchw(&state, &weights.conv7, 1, 2, (0, 0, 0, 0), 1)?;
    apply_swoosh_r(&mut state.data);

    let residual = slice_h(&state, 0, state.h.checked_sub(3).ok_or("xasr embed h < 3")?)?;
    let cached = match embed_states {
        Some(values) => Tensor4 {
            data: values.to_vec(),
            n: 1,
            c: 128,
            h: 3,
            w: 19,
        },
        None => Tensor4 {
            data: vec![0.0; 128 * 3 * 19],
            n: 1,
            c: 128,
            h: 3,
            w: 19,
        },
    };
    cached.validate_len("embed_states")?;
    let concat = concat_h(&cached, &state)?;
    let new_embed_states = slice_h(&concat, state.h - 3, state.h)?.data;

    let mut convnext = conv2d_nchw(
        &concat,
        &weights.convnext_depthwise,
        1,
        1,
        (0, 3, 0, 3),
        128,
    )?;
    convnext = conv2d_nchw(
        &convnext,
        &weights.convnext_pointwise1,
        1,
        1,
        (0, 0, 0, 0),
        1,
    )?;
    apply_swoosh_l(&mut convnext.data);
    convnext = conv2d_nchw(
        &convnext,
        &weights.convnext_pointwise2,
        1,
        1,
        (0, 0, 0, 0),
        1,
    )?;
    let state = add_nchw(&residual, &convnext)?;
    let projected = project_embed_out(&state, weights)?;
    Ok(XasrEncoderEmbedReferenceOutput {
        frames: projected.len() / weights.out.weight.output_dim,
        dim: weights.out.weight.output_dim,
        rows: projected,
        new_embed_states,
    })
}

pub(crate) fn feed_forward_reference(
    weights: &XasrLinearPairWeights,
    rows: &[f32],
    frames: usize,
    dim: usize,
) -> Result<Vec<f32>, String> {
    validate_frame_rows(rows, frames, dim, "feed-forward input")?;
    let mut output = Vec::with_capacity(rows.len());
    for frame in rows.chunks_exact(dim) {
        let mut hidden = weights
            .in_proj
            .weight
            .apply(frame, Some(&weights.in_proj.bias))?;
        apply_swoosh_l(&mut hidden);
        let projected = weights
            .out_proj
            .weight
            .apply(&hidden, Some(&weights.out_proj.bias))?;
        output.extend_from_slice(&projected);
    }
    Ok(output)
}

pub(crate) fn bypass_reference(
    original_rows: &[f32],
    current_rows: &[f32],
    scale: &[f32],
    frames: usize,
    dim: usize,
) -> Result<Vec<f32>, String> {
    validate_frame_rows(original_rows, frames, dim, "bypass original input")?;
    validate_frame_rows(current_rows, frames, dim, "bypass current input")?;
    if scale.len() != dim {
        return Err(format!(
            "xasr bypass scale has {} values, expected {dim}",
            scale.len()
        ));
    }
    let mut output = vec![0.0_f32; original_rows.len()];
    for frame in 0..frames {
        for (c, &scale) in scale.iter().enumerate().take(dim) {
            let idx = frame * dim + c;
            output[idx] = original_rows[idx] + (current_rows[idx] - original_rows[idx]) * scale;
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn zipformer_layer_streaming_reference(
    layer: &XasrEncoderLayerWeights,
    rows: &[f32],
    frames: usize,
    dim: usize,
    num_heads: usize,
    query_head_dim: usize,
    left_context_len: usize,
    valid_left_context_len: usize,
    caches: XasrZipformerLayerReferenceCaches<'_>,
) -> Result<XasrZipformerLayerReferenceOutput, String> {
    validate_frame_rows(rows, frames, dim, "zipformer layer input")?;
    let key_padding_mask =
        streaming_key_padding_mask(left_context_len, frames, valid_left_context_len)?;
    let attention = self_attention_weights_streaming_reference(
        &layer.self_attn_weights,
        rows,
        frames,
        dim,
        num_heads,
        query_head_dim,
        left_context_len,
        caches.cached_key,
        Some(&key_padding_mask),
    )?;
    let k_len = left_context_len + frames;
    let nonlin_attention_weights = &attention.weights[..frames * k_len];

    let ff1 = feed_forward_reference(&layer.feed_forward1, rows, frames, dim)?;
    let add6 = add_frame_rows(rows, &ff1)?;
    let nonlin = nonlin_attention_streaming_reference(
        &layer.nonlin_attention,
        &add6,
        nonlin_attention_weights,
        frames,
        dim,
        1,
        left_context_len,
        caches.cached_nonlin_attention,
    )?;
    let add8 = add_frame_rows(&add6, &nonlin.rows)?;
    let self1 = self_attention_streaming_reference(
        &layer.self_attn1,
        &add8,
        &attention.weights,
        frames,
        dim,
        num_heads,
        left_context_len,
        caches.cached_val1,
    )?;
    let add10 = add_frame_rows(&add8, &self1.rows)?;
    let conv1 = convolution_module_streaming_reference(
        &layer.conv_module1,
        &add10,
        frames,
        dim,
        caches.cached_conv1,
        None,
    )?;
    let add15 = add_frame_rows(&add10, &conv1.rows)?;
    let ff2 = feed_forward_reference(&layer.feed_forward2, &add15, frames, dim)?;
    let add16 = add_frame_rows(&add15, &ff2)?;
    let bypass_mid = bypass_reference(rows, &add16, &layer.bypass_mid_scale, frames, dim)?;

    let self2 = self_attention_streaming_reference(
        &layer.self_attn2,
        &bypass_mid,
        &attention.weights,
        frames,
        dim,
        num_heads,
        left_context_len,
        caches.cached_val2,
    )?;
    let add18 = add_frame_rows(&bypass_mid, &self2.rows)?;
    let conv2 = convolution_module_streaming_reference(
        &layer.conv_module2,
        &add18,
        frames,
        dim,
        caches.cached_conv2,
        None,
    )?;
    let add23 = add_frame_rows(&add18, &conv2.rows)?;
    let ff3 = feed_forward_reference(&layer.feed_forward3, &add23, frames, dim)?;
    let add24 = add_frame_rows(&add23, &ff3)?;
    let mut norm = add24;
    bias_norm_last_dim(&mut norm, dim, &layer.norm_bias, layer.norm_log_scale[0])?;
    let output = bypass_reference(rows, &norm, &layer.bypass_scale, frames, dim)?;

    Ok(XasrZipformerLayerReferenceOutput {
        rows: output,
        new_cached_key: attention.new_cached_key,
        new_cached_nonlin_attention: nonlin.new_cache,
        new_cached_val1: self1.new_cache,
        new_cached_val2: self2.new_cache,
        new_cached_conv1: conv1.new_cache,
        new_cached_conv2: conv2.new_cache,
    })
}

pub(crate) fn downsample_streaming_reference(
    rows: &[f32],
    frames: usize,
    input_dim: usize,
    target_dim: usize,
    bias_logits: &[f32],
) -> Result<XasrDownsampleReferenceOutput, String> {
    validate_frame_rows(rows, frames, input_dim, "downsample input")?;
    if target_dim == 0 {
        return Err("xasr downsample target dim must be > 0".to_string());
    }
    let factor = bias_logits.len();
    if factor == 0 {
        return Err("xasr downsample factor must be > 0".to_string());
    }
    let out_frames = frames.div_ceil(factor);
    let padded_frames = out_frames * factor;
    let mut padded_rows = resize_frame_rows_reference(rows, frames, input_dim, target_dim)?;
    padded_rows.resize(padded_frames * target_dim, 0.0);

    let reshaped_rows = padded_rows.clone();
    let mut weights = bias_logits.to_vec();
    softmax_last_dim(&mut weights, factor)?;
    let mut output = vec![0.0_f32; out_frames * target_dim];
    for out_frame in 0..out_frames {
        for (factor_index, &weight) in weights.iter().enumerate().take(factor) {
            let src_frame = out_frame * factor + factor_index;
            for c in 0..target_dim {
                output[out_frame * target_dim + c] +=
                    padded_rows[src_frame * target_dim + c] * weight;
            }
        }
    }

    Ok(XasrDownsampleReferenceOutput {
        padded_rows,
        reshaped_rows,
        rows: output,
        frames: out_frames,
        dim: target_dim,
    })
}

pub(crate) fn upsample_streaming_reference(
    rows: &[f32],
    frames: usize,
    dim: usize,
    factor: usize,
    target_frames: usize,
) -> Result<Vec<f32>, String> {
    validate_frame_rows(rows, frames, dim, "upsample input")?;
    if factor == 0 {
        return Err("xasr upsample factor must be > 0".to_string());
    }
    let mut output = vec![0.0_f32; target_frames * dim];
    for out_frame in 0..target_frames {
        let src_frame = out_frame / factor;
        if src_frame >= frames {
            continue;
        }
        let src = src_frame * dim;
        let dst = out_frame * dim;
        output[dst..dst + dim].copy_from_slice(&rows[src..src + dim]);
    }
    Ok(output)
}

pub(crate) fn resize_frame_rows_reference(
    rows: &[f32],
    frames: usize,
    input_dim: usize,
    target_dim: usize,
) -> Result<Vec<f32>, String> {
    validate_frame_rows(rows, frames, input_dim, "resize input")?;
    if target_dim == 0 {
        return Err("xasr resize target dim must be > 0".to_string());
    }
    let mut output = vec![0.0_f32; frames * target_dim];
    let copy_dim = input_dim.min(target_dim);
    for frame in 0..frames {
        let src = frame * input_dim;
        let dst = frame * target_dim;
        output[dst..dst + copy_dim].copy_from_slice(&rows[src..src + copy_dim]);
    }
    Ok(output)
}

pub(crate) fn self_attention_weights_streaming_reference(
    weights: &XasrSelfAttentionWeightsWeights,
    rows: &[f32],
    frames: usize,
    dim: usize,
    num_heads: usize,
    query_head_dim: usize,
    left_context_len: usize,
    cached_key: Option<&[f32]>,
    key_padding_mask: Option<&[bool]>,
) -> Result<XasrAttentionWeightsReferenceOutput, String> {
    validate_frame_rows(rows, frames, dim, "attention weights input")?;
    let query_dim = num_heads
        .checked_mul(query_head_dim)
        .ok_or("xasr attention query dim overflow")?;
    let pos_output_dim = weights.self_attn_pos_output_dim()?;
    if !pos_output_dim.is_multiple_of(num_heads) {
        return Err(format!(
            "xasr attention pos output dim {pos_output_dim} not divisible by heads {num_heads}"
        ));
    }
    let pos_head_dim = pos_output_dim / num_heads;
    let expected_in_proj_output = 2 * query_dim + pos_output_dim;
    if weights.in_proj.weight.output_dim != expected_in_proj_output {
        return Err(format!(
            "xasr attention in_proj output dim {}, expected {expected_in_proj_output}",
            weights.in_proj.weight.output_dim
        ));
    }

    let k_len = left_context_len + frames;
    if let Some(mask) = key_padding_mask
        && mask.len() != k_len
    {
        return Err(format!(
            "xasr attention key padding mask has {} entries, expected {k_len}",
            mask.len()
        ));
    }

    let mut q = vec![0.0_f32; frames * query_dim];
    let mut current_k = vec![0.0_f32; frames * query_dim];
    let mut p = vec![0.0_f32; frames * pos_output_dim];
    for (t, frame) in rows.chunks_exact(dim).enumerate() {
        let projected = weights
            .in_proj
            .weight
            .apply(frame, Some(&weights.in_proj.bias))?;
        q[t * query_dim..(t + 1) * query_dim].copy_from_slice(&projected[..query_dim]);
        current_k[t * query_dim..(t + 1) * query_dim]
            .copy_from_slice(&projected[query_dim..2 * query_dim]);
        p[t * pos_output_dim..(t + 1) * pos_output_dim]
            .copy_from_slice(&projected[2 * query_dim..]);
    }

    let cached_key = match cached_key {
        Some(cache) => {
            if cache.len() != left_context_len * query_dim {
                return Err(format!(
                    "xasr attention cached_key has {} values, expected {}",
                    cache.len(),
                    left_context_len * query_dim
                ));
            }
            cache.to_vec()
        }
        None => vec![0.0; left_context_len * query_dim],
    };
    let mut all_keys = vec![0.0_f32; k_len * query_dim];
    all_keys[..cached_key.len()].copy_from_slice(&cached_key);
    all_keys[cached_key.len()..].copy_from_slice(&current_k);
    let new_cached_key =
        all_keys[frames * query_dim..(frames + left_context_len) * query_dim].to_vec();

    let pos_dim = weights.linear_pos.input_dim;
    let pos_head_dim_from_weight = weights.linear_pos.output_dim / num_heads;
    if pos_head_dim_from_weight != pos_head_dim {
        return Err(format!(
            "xasr attention linear_pos output/head dim {pos_head_dim_from_weight}, expected {pos_head_dim}"
        ));
    }
    let pos_embedding = compact_relative_positional_encoding(frames, left_context_len, pos_dim);
    let mut projected_pos =
        Vec::with_capacity((left_context_len + 2 * frames - 1) * pos_output_dim);
    for row in pos_embedding.chunks_exact(pos_dim) {
        let projected = weights.linear_pos.apply(row, None)?;
        projected_pos.extend_from_slice(&projected);
    }

    let mut attn = vec![0.0_f32; num_heads * frames * k_len];
    for head in 0..num_heads {
        for target in 0..frames {
            for source in 0..k_len {
                let mut score = 0.0_f32;
                for i in 0..query_head_dim {
                    let q_idx = target * query_dim + head * query_head_dim + i;
                    let k_idx = source * query_dim + head * query_head_dim + i;
                    score += q[q_idx] * all_keys[k_idx];
                }
                let relative = frames - 1 - target + source;
                for i in 0..pos_head_dim {
                    let p_idx = target * pos_output_dim + head * pos_head_dim + i;
                    let pos_idx = relative * pos_output_dim + head * pos_head_dim + i;
                    score += p[p_idx] * projected_pos[pos_idx];
                }
                if key_padding_mask.is_some_and(|mask| mask[source]) {
                    score = -1000.0;
                }
                attn[(head * frames + target) * k_len + source] = score;
            }
        }
    }
    softmax_last_dim(&mut attn, k_len)?;
    Ok(XasrAttentionWeightsReferenceOutput {
        weights: attn,
        new_cached_key,
    })
}

pub(crate) fn streaming_key_padding_mask(
    left_context_len: usize,
    frames: usize,
    valid_left_context_len: usize,
) -> Result<Vec<bool>, String> {
    if valid_left_context_len > left_context_len {
        return Err(format!(
            "xasr attention valid left context {valid_left_context_len} exceeds {left_context_len}"
        ));
    }
    let mut mask = vec![false; left_context_len + frames];
    let masked_left = left_context_len - valid_left_context_len;
    for item in mask.iter_mut().take(masked_left) {
        *item = true;
    }
    Ok(mask)
}

pub(crate) fn nonlin_attention_streaming_reference(
    weights: &XasrNonlinAttentionWeights,
    rows: &[f32],
    attn_weights: &[f32],
    frames: usize,
    dim: usize,
    num_heads: usize,
    left_context_len: usize,
    cached_x: Option<&[f32]>,
) -> Result<XasrValueProjectionReferenceOutput, String> {
    validate_frame_rows(rows, frames, dim, "nonlin attention input")?;
    let hidden_dim = weights.out_proj.weight.input_dim;
    let in_proj_output = 3 * hidden_dim;
    if weights.in_proj.weight.output_dim != in_proj_output {
        return Err(format!(
            "xasr nonlin attention in_proj output dim {}, expected {in_proj_output}",
            weights.in_proj.weight.output_dim
        ));
    }
    if !hidden_dim.is_multiple_of(num_heads) {
        return Err(format!(
            "xasr nonlin attention hidden dim {hidden_dim} not divisible by heads {num_heads}"
        ));
    }
    let head_dim = hidden_dim / num_heads;
    let k_len = left_context_len + frames;
    validate_attention_weights(attn_weights, num_heads, frames, k_len)?;

    let mut x = vec![0.0_f32; frames * hidden_dim];
    let mut y = vec![0.0_f32; frames * hidden_dim];
    for (t, frame) in rows.chunks_exact(dim).enumerate() {
        let projected = weights
            .in_proj
            .weight
            .apply(frame, Some(&weights.in_proj.bias))?;
        for i in 0..hidden_dim {
            let s = projected[i].tanh();
            x[t * hidden_dim + i] = projected[hidden_dim + i] * s;
            y[t * hidden_dim + i] = projected[2 * hidden_dim + i];
        }
    }

    let cached_x = match cached_x {
        Some(cache) => {
            if cache.len() != num_heads * left_context_len * head_dim {
                return Err(format!(
                    "xasr nonlin attention cache has {} values, expected {}",
                    cache.len(),
                    num_heads * left_context_len * head_dim
                ));
            }
            cache.to_vec()
        }
        None => vec![0.0; num_heads * left_context_len * head_dim],
    };
    let mut x_pad = vec![0.0_f32; num_heads * k_len * head_dim];
    x_pad[..cached_x.len()].copy_from_slice(&cached_x);
    for t in 0..frames {
        for head in 0..num_heads {
            for d in 0..head_dim {
                x_pad[(head * k_len + left_context_len + t) * head_dim + d] =
                    x[t * hidden_dim + head * head_dim + d];
            }
        }
    }
    let new_cache = x_pad
        .chunks_exact(k_len * head_dim)
        .flat_map(|head_rows| {
            head_rows[frames * head_dim..(frames + left_context_len) * head_dim]
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();

    let attended =
        attention_weighted_values(attn_weights, &x_pad, num_heads, frames, k_len, head_dim)?;
    let mut attended_rows = vec![0.0_f32; frames * hidden_dim];
    for t in 0..frames {
        for head in 0..num_heads {
            for d in 0..head_dim {
                let dst = t * hidden_dim + head * head_dim + d;
                let src = (head * frames + t) * head_dim + d;
                attended_rows[dst] = attended[src] * y[dst];
            }
        }
    }
    let rows = apply_linear_to_rows(&weights.out_proj, &attended_rows, frames, hidden_dim)?;
    Ok(XasrValueProjectionReferenceOutput { rows, new_cache })
}

pub(crate) fn self_attention_streaming_reference(
    weights: &XasrLinearPairWeights,
    rows: &[f32],
    attn_weights: &[f32],
    frames: usize,
    dim: usize,
    num_heads: usize,
    left_context_len: usize,
    cached_val: Option<&[f32]>,
) -> Result<XasrValueProjectionReferenceOutput, String> {
    validate_frame_rows(rows, frames, dim, "self attention input")?;
    let value_dim = weights.in_proj.weight.output_dim;
    if !value_dim.is_multiple_of(num_heads) {
        return Err(format!(
            "xasr self attention value dim {value_dim} not divisible by heads {num_heads}"
        ));
    }
    let value_head_dim = value_dim / num_heads;
    let k_len = left_context_len + frames;
    validate_attention_weights(attn_weights, num_heads, frames, k_len)?;

    let mut current = vec![0.0_f32; frames * value_dim];
    for (t, frame) in rows.chunks_exact(dim).enumerate() {
        let projected = weights
            .in_proj
            .weight
            .apply(frame, Some(&weights.in_proj.bias))?;
        current[t * value_dim..(t + 1) * value_dim].copy_from_slice(&projected);
    }
    let cached_val = match cached_val {
        Some(cache) => {
            if cache.len() != left_context_len * value_dim {
                return Err(format!(
                    "xasr self attention cache has {} values, expected {}",
                    cache.len(),
                    left_context_len * value_dim
                ));
            }
            cache.to_vec()
        }
        None => vec![0.0; left_context_len * value_dim],
    };
    let mut values = vec![0.0_f32; num_heads * k_len * value_head_dim];
    for s in 0..left_context_len {
        for head in 0..num_heads {
            for d in 0..value_head_dim {
                values[(head * k_len + s) * value_head_dim + d] =
                    cached_val[s * value_dim + head * value_head_dim + d];
            }
        }
    }
    for t in 0..frames {
        for head in 0..num_heads {
            for d in 0..value_head_dim {
                values[(head * k_len + left_context_len + t) * value_head_dim + d] =
                    current[t * value_dim + head * value_head_dim + d];
            }
        }
    }
    let mut flat_values = vec![0.0_f32; k_len * value_dim];
    for s in 0..k_len {
        for head in 0..num_heads {
            for d in 0..value_head_dim {
                flat_values[s * value_dim + head * value_head_dim + d] =
                    values[(head * k_len + s) * value_head_dim + d];
            }
        }
    }
    let new_cache =
        flat_values[frames * value_dim..(frames + left_context_len) * value_dim].to_vec();

    let attended = attention_weighted_values(
        attn_weights,
        &values,
        num_heads,
        frames,
        k_len,
        value_head_dim,
    )?;
    let mut attended_rows = vec![0.0_f32; frames * value_dim];
    for t in 0..frames {
        for head in 0..num_heads {
            for d in 0..value_head_dim {
                attended_rows[t * value_dim + head * value_head_dim + d] =
                    attended[(head * frames + t) * value_head_dim + d];
            }
        }
    }
    let rows = apply_linear_to_rows(&weights.out_proj, &attended_rows, frames, value_dim)?;
    Ok(XasrValueProjectionReferenceOutput { rows, new_cache })
}

pub(crate) fn convolution_module_streaming_reference(
    weights: &XasrConvolutionModuleWeights,
    rows: &[f32],
    frames: usize,
    dim: usize,
    cache: Option<&[f32]>,
    padding_mask: Option<&[bool]>,
) -> Result<XasrConvolutionReferenceOutput, String> {
    validate_frame_rows(rows, frames, dim, "convolution module input")?;
    if let Some(mask) = padding_mask
        && mask.len() != frames
    {
        return Err(format!(
            "xasr convolution padding mask has {} entries, expected {frames}",
            mask.len()
        ));
    }

    let mut gated = vec![0.0_f32; frames * dim];
    for (t, frame) in rows.chunks_exact(dim).enumerate() {
        let projected = weights
            .in_proj
            .weight
            .apply(frame, Some(&weights.in_proj.bias))?;
        if projected.len() != 2 * dim {
            return Err(format!(
                "xasr convolution in_proj produced {}, expected {}",
                projected.len(),
                2 * dim
            ));
        }
        for c in 0..dim {
            gated[t * dim + c] = projected[c] * sigmoid(projected[dim + c]);
        }
    }

    let mut channel_major = vec![0.0_f32; dim * frames];
    for t in 0..frames {
        let masked = padding_mask.is_some_and(|mask| mask[t]);
        for c in 0..dim {
            channel_major[c * frames + t] = if masked { 0.0 } else { gated[t * dim + c] };
        }
    }

    let kernel = weights.depthwise_chunkwise_conv.weight.dims[0];
    let left_pad = kernel / 2;
    let expected_cache_len = dim * left_pad;
    let cache = match cache {
        Some(cache) => {
            if cache.len() != expected_cache_len {
                return Err(format!(
                    "xasr convolution cache has {} values, expected {expected_cache_len}",
                    cache.len()
                ));
            }
            cache.to_vec()
        }
        None => vec![0.0; expected_cache_len],
    };

    let mut cached_input = vec![0.0_f32; dim * (left_pad + frames)];
    for c in 0..dim {
        let dst = c * (left_pad + frames);
        let cache_src = c * left_pad;
        cached_input[dst..dst + left_pad].copy_from_slice(&cache[cache_src..cache_src + left_pad]);
        let frame_dst = dst + left_pad;
        let frame_src = c * frames;
        cached_input[frame_dst..frame_dst + frames]
            .copy_from_slice(&channel_major[frame_src..frame_src + frames]);
    }
    let mut new_cache = vec![0.0_f32; expected_cache_len];
    for c in 0..dim {
        let src = c * (left_pad + frames) + frames;
        let dst = c * left_pad;
        new_cache[dst..dst + left_pad].copy_from_slice(&cached_input[src..src + left_pad]);
    }

    let causal = depthwise_conv1d_valid_channel_major(
        &cached_input,
        dim,
        left_pad + frames,
        &weights.depthwise_causal_conv,
    )?;
    let chunk = depthwise_conv1d_same_channel_major(
        &channel_major,
        dim,
        frames,
        &weights.depthwise_chunkwise_conv,
    )?;
    let mut depthwise_rows = vec![0.0_f32; dim * frames];
    for c in 0..dim {
        for t in 0..frames {
            let idx = c * frames + t;
            depthwise_rows[idx] =
                causal[idx] + chunk[idx] * chunkwise_conv_scale(weights, c, t, frames)?;
        }
    }

    let mut transposed = vec![0.0_f32; frames * dim];
    for t in 0..frames {
        for c in 0..dim {
            transposed[t * dim + c] = depthwise_rows[c * frames + t];
        }
    }
    apply_swoosh_r(&mut transposed);
    let rows = apply_linear_to_rows(&weights.out_proj, &transposed, frames, dim)?;
    Ok(XasrConvolutionReferenceOutput {
        rows,
        depthwise_rows,
        new_cache,
    })
}

fn project_embed_out(
    state: &Tensor4,
    weights: &XasrEncoderEmbedWeights,
) -> Result<Vec<f32>, String> {
    if state.n != 1 {
        return Err(format!(
            "xasr encoder embed expected batch=1, got {}",
            state.n
        ));
    }
    let frame_input_dim = state.c * state.w;
    if frame_input_dim != weights.out.weight.input_dim {
        return Err(format!(
            "xasr encoder embed projection input dim {frame_input_dim}, expected {}",
            weights.out.weight.input_dim
        ));
    }
    let mut rows = Vec::with_capacity(state.h * weights.out.weight.output_dim);
    let mut frame = vec![0.0_f32; frame_input_dim];
    for h in 0..state.h {
        for c in 0..state.c {
            for w in 0..state.w {
                frame[c * state.w + w] = state.get(0, c, h, w);
            }
        }
        let out = weights.out.weight.apply(&frame, Some(&weights.out.bias))?;
        rows.extend_from_slice(&out);
    }
    bias_norm_last_dim(
        &mut rows,
        weights.out.weight.output_dim,
        &weights.out_norm_bias,
        weights.out_norm_log_scale[0],
    )?;
    Ok(rows)
}

trait XasrSelfAttentionWeightExt {
    fn self_attn_pos_output_dim(&self) -> Result<usize, String>;
}

impl XasrSelfAttentionWeightExt for XasrSelfAttentionWeightsWeights {
    fn self_attn_pos_output_dim(&self) -> Result<usize, String> {
        if self.linear_pos.output_dim == 0 {
            return Err("xasr attention linear_pos output dim must be > 0".to_string());
        }
        Ok(self.linear_pos.output_dim)
    }
}

fn apply_linear_to_rows(
    weights: &XasrLinearWithBias,
    rows: &[f32],
    frames: usize,
    input_dim: usize,
) -> Result<Vec<f32>, String> {
    validate_frame_rows(rows, frames, input_dim, "linear rows")?;
    let mut output = Vec::with_capacity(frames * weights.weight.output_dim);
    for frame in rows.chunks_exact(input_dim) {
        let projected = weights.weight.apply(frame, Some(&weights.bias))?;
        output.extend_from_slice(&projected);
    }
    Ok(output)
}

fn depthwise_conv1d_valid_channel_major(
    input: &[f32],
    channels: usize,
    input_len: usize,
    weights: &XasrConv1dWeights,
) -> Result<Vec<f32>, String> {
    let kernel = conv1d_kernel(weights, channels)?;
    if input.len() != channels * input_len {
        return Err(format!(
            "xasr conv1d valid input has {} values, expected {}",
            input.len(),
            channels * input_len
        ));
    }
    if input_len < kernel {
        return Err(format!(
            "xasr conv1d valid kernel {kernel} exceeds input len {input_len}"
        ));
    }
    let output_len = input_len - kernel + 1;
    let mut output = vec![0.0_f32; channels * output_len];
    for c in 0..channels {
        for t in 0..output_len {
            let mut sum = weights.bias[c];
            for k in 0..kernel {
                sum += input[c * input_len + t + k] * weights.weight.values[c * kernel + k];
            }
            output[c * output_len + t] = sum;
        }
    }
    Ok(output)
}

fn depthwise_conv1d_same_channel_major(
    input: &[f32],
    channels: usize,
    input_len: usize,
    weights: &XasrConv1dWeights,
) -> Result<Vec<f32>, String> {
    let kernel = conv1d_kernel(weights, channels)?;
    if input.len() != channels * input_len {
        return Err(format!(
            "xasr conv1d same input has {} values, expected {}",
            input.len(),
            channels * input_len
        ));
    }
    let pad = kernel / 2;
    let mut output = vec![0.0_f32; channels * input_len];
    for c in 0..channels {
        for t in 0..input_len {
            let mut sum = weights.bias[c];
            for k in 0..kernel {
                let Some(input_t) = (t + k).checked_sub(pad) else {
                    continue;
                };
                if input_t >= input_len {
                    continue;
                }
                sum += input[c * input_len + input_t] * weights.weight.values[c * kernel + k];
            }
            output[c * input_len + t] = sum;
        }
    }
    Ok(output)
}

fn conv1d_kernel(weights: &XasrConv1dWeights, channels: usize) -> Result<usize, String> {
    let [kernel, in_per_group, out_channels]: [usize; 3] =
        weights.weight.dims.as_slice().try_into().map_err(|_| {
            format!(
                "xasr conv1d '{}' expected rank 3, got {:?}",
                weights.weight.name, weights.weight.dims
            )
        })?;
    if in_per_group != 1 || out_channels != channels || weights.bias.len() != channels {
        return Err(format!(
            "xasr conv1d '{}' expected depthwise dims [{kernel},1,{channels}], got {:?} with bias {}",
            weights.weight.name,
            weights.weight.dims,
            weights.bias.len()
        ));
    }
    if weights.weight.values.len() != channels * kernel {
        return Err(format!(
            "xasr conv1d '{}' has {} weight values, expected {}",
            weights.weight.name,
            weights.weight.values.len(),
            channels * kernel
        ));
    }
    Ok(kernel)
}

fn chunkwise_conv_scale(
    weights: &XasrConvolutionModuleWeights,
    channel: usize,
    frame: usize,
    chunk_size: usize,
) -> Result<f32, String> {
    let dims = &weights.chunkwise_conv_scale.dims;
    if dims
        != &[
            2,
            weights.depthwise_chunkwise_conv.weight.dims[2],
            weights.depthwise_chunkwise_conv.weight.dims[0],
        ]
    {
        return Err(format!(
            "xasr chunkwise_conv_scale has dims {:?}, expected [2, {}, {}]",
            dims,
            weights.depthwise_chunkwise_conv.weight.dims[2],
            weights.depthwise_chunkwise_conv.weight.dims[0]
        ));
    }
    let channels = dims[1];
    let kernel = dims[2];
    let values = &weights.chunkwise_conv_scale.values;
    let left = if frame < kernel {
        values[channel * kernel + frame]
    } else {
        0.0
    };
    let right_base = channels * kernel;
    let right = if chunk_size < kernel {
        values[right_base + channel * kernel + kernel - chunk_size + frame]
    } else {
        let pad = chunk_size - kernel;
        if frame >= pad {
            values[right_base + channel * kernel + frame - pad]
        } else {
            0.0
        }
    };
    Ok(1.0 + left + right)
}

fn add_frame_rows(lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "xasr residual add shape mismatch lhs={} rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }
    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(&lhs, &rhs)| lhs + rhs)
        .collect())
}

fn validate_frame_rows(rows: &[f32], frames: usize, dim: usize, name: &str) -> Result<(), String> {
    if rows.len() != frames * dim {
        return Err(format!(
            "xasr {name} has {} values, expected {}",
            rows.len(),
            frames * dim
        ));
    }
    Ok(())
}

fn validate_attention_weights(
    attn_weights: &[f32],
    num_heads: usize,
    frames: usize,
    k_len: usize,
) -> Result<(), String> {
    let expected = num_heads * frames * k_len;
    if attn_weights.len() != expected {
        return Err(format!(
            "xasr attention weights have {} values, expected {expected}",
            attn_weights.len()
        ));
    }
    Ok(())
}

fn attention_weighted_values(
    attn_weights: &[f32],
    values: &[f32],
    num_heads: usize,
    frames: usize,
    k_len: usize,
    head_dim: usize,
) -> Result<Vec<f32>, String> {
    validate_attention_weights(attn_weights, num_heads, frames, k_len)?;
    if values.len() != num_heads * k_len * head_dim {
        return Err(format!(
            "xasr attention values have {} values, expected {}",
            values.len(),
            num_heads * k_len * head_dim
        ));
    }
    let mut output = vec![0.0_f32; num_heads * frames * head_dim];
    for head in 0..num_heads {
        for target in 0..frames {
            for source in 0..k_len {
                let weight = attn_weights[(head * frames + target) * k_len + source];
                for d in 0..head_dim {
                    output[(head * frames + target) * head_dim + d] +=
                        weight * values[(head * k_len + source) * head_dim + d];
                }
            }
        }
    }
    Ok(output)
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn softmax_last_dim(values: &mut [f32], width: usize) -> Result<(), String> {
    if width == 0 {
        return Err("xasr softmax width must be > 0".to_string());
    }
    if !values.len().is_multiple_of(width) {
        return Err(format!(
            "xasr softmax input has {} values, not divisible by width {width}",
            values.len()
        ));
    }
    for row in values.chunks_exact_mut(width) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for value in row.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        if sum == 0.0 || !sum.is_finite() {
            return Err("xasr softmax produced non-finite normalization".to_string());
        }
        for value in row.iter_mut() {
            *value /= sum;
        }
    }
    Ok(())
}

fn compact_relative_positional_encoding(
    frames: usize,
    left_context_len: usize,
    embed_dim: usize,
) -> Vec<f32> {
    debug_assert!(embed_dim.is_multiple_of(2));
    let total_context = frames + left_context_len;
    let seq_len = left_context_len + 2 * frames - 1;
    let compression_length = (embed_dim as f32).sqrt();
    let length_scale = embed_dim as f32 / (2.0 * std::f32::consts::PI);
    let mut output = vec![0.0_f32; seq_len * embed_dim];
    for row in 0..seq_len {
        let offset = row as isize - (total_context as isize - 1);
        let sign = (offset as f32).signum();
        let abs = (offset as f32).abs();
        let compressed =
            compression_length * sign * ((abs + compression_length).ln() - compression_length.ln());
        let atan = (compressed / length_scale).atan();
        for i in 0..embed_dim / 2 {
            let value = atan * (i + 1) as f32;
            output[row * embed_dim + 2 * i] = value.cos();
            output[row * embed_dim + 2 * i + 1] = value.sin();
        }
        output[row * embed_dim + embed_dim - 1] = 1.0;
    }
    output
}

fn conv2d_nchw(
    input: &Tensor4,
    weights: &XasrConv2dWeights,
    stride_h: usize,
    stride_w: usize,
    pads: (usize, usize, usize, usize),
    groups: usize,
) -> Result<Tensor4, String> {
    let [kernel_w, kernel_h, in_per_group, out_channels]: [usize; 4] =
        weights.weight.dims.as_slice().try_into().map_err(|_| {
            format!(
                "xasr conv2d '{}' expected rank 4, got {:?}",
                weights.weight.name, weights.weight.dims
            )
        })?;
    if groups == 0 || !input.c.is_multiple_of(groups) || !out_channels.is_multiple_of(groups) {
        return Err(format!(
            "xasr conv2d '{}' invalid groups {groups} for input {} output {out_channels}",
            weights.weight.name, input.c
        ));
    }
    if input.c / groups != in_per_group {
        return Err(format!(
            "xasr conv2d '{}' has in_per_group {in_per_group}, expected {}",
            weights.weight.name,
            input.c / groups
        ));
    }
    let (pad_top, pad_left, pad_bottom, pad_right) = pads;
    let padded_h = input.h + pad_top + pad_bottom;
    let padded_w = input.w + pad_left + pad_right;
    if padded_h < kernel_h || padded_w < kernel_w {
        return Err(format!(
            "xasr conv2d '{}' kernel [{kernel_h},{kernel_w}] exceeds padded input [{padded_h},{padded_w}]",
            weights.weight.name
        ));
    }
    let out_h = (padded_h - kernel_h) / stride_h + 1;
    let out_w = (padded_w - kernel_w) / stride_w + 1;
    let mut out = Tensor4 {
        data: vec![0.0; input.n * out_channels * out_h * out_w],
        n: input.n,
        c: out_channels,
        h: out_h,
        w: out_w,
    };
    let out_per_group = out_channels / groups;
    for n in 0..input.n {
        for oc in 0..out_channels {
            let group = oc / out_per_group;
            let input_base = group * in_per_group;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = weights.bias[oc];
                    for icg in 0..in_per_group {
                        let ic = input_base + icg;
                        for kh in 0..kernel_h {
                            let ih_padded = oh * stride_h + kh;
                            if ih_padded < pad_top {
                                continue;
                            }
                            let ih = ih_padded - pad_top;
                            if ih >= input.h {
                                continue;
                            }
                            for kw in 0..kernel_w {
                                let iw_padded = ow * stride_w + kw;
                                if iw_padded < pad_left {
                                    continue;
                                }
                                let iw = iw_padded - pad_left;
                                if iw >= input.w {
                                    continue;
                                }
                                let weight = weights.weight.values
                                    [(((oc * in_per_group + icg) * kernel_h + kh) * kernel_w) + kw];
                                sum += input.get(n, ic, ih, iw) * weight;
                            }
                        }
                    }
                    out.set(n, oc, oh, ow, sum);
                }
            }
        }
    }
    Ok(out)
}

fn slice_h(input: &Tensor4, start: usize, end: usize) -> Result<Tensor4, String> {
    if start > end || end > input.h {
        return Err(format!(
            "xasr slice_h invalid range {start}..{end} for h={}",
            input.h
        ));
    }
    let out_h = end - start;
    let mut out = Tensor4 {
        data: vec![0.0; input.n * input.c * out_h * input.w],
        n: input.n,
        c: input.c,
        h: out_h,
        w: input.w,
    };
    for n in 0..input.n {
        for c in 0..input.c {
            for h in 0..out_h {
                for w in 0..input.w {
                    out.set(n, c, h, w, input.get(n, c, start + h, w));
                }
            }
        }
    }
    Ok(out)
}

fn concat_h(lhs: &Tensor4, rhs: &Tensor4) -> Result<Tensor4, String> {
    if lhs.n != rhs.n || lhs.c != rhs.c || lhs.w != rhs.w {
        return Err(format!(
            "xasr concat_h shape mismatch lhs=[{},{},{},{}] rhs=[{},{},{},{}]",
            lhs.n, lhs.c, lhs.h, lhs.w, rhs.n, rhs.c, rhs.h, rhs.w
        ));
    }
    let mut out = Tensor4 {
        data: vec![0.0; lhs.n * lhs.c * (lhs.h + rhs.h) * lhs.w],
        n: lhs.n,
        c: lhs.c,
        h: lhs.h + rhs.h,
        w: lhs.w,
    };
    for n in 0..out.n {
        for c in 0..out.c {
            for h in 0..lhs.h {
                for w in 0..out.w {
                    out.set(n, c, h, w, lhs.get(n, c, h, w));
                }
            }
            for h in 0..rhs.h {
                for w in 0..out.w {
                    out.set(n, c, lhs.h + h, w, rhs.get(n, c, h, w));
                }
            }
        }
    }
    Ok(out)
}

fn add_nchw(lhs: &Tensor4, rhs: &Tensor4) -> Result<Tensor4, String> {
    if (lhs.n, lhs.c, lhs.h, lhs.w) != (rhs.n, rhs.c, rhs.h, rhs.w) {
        return Err(format!(
            "xasr add_nchw shape mismatch lhs=[{},{},{},{}] rhs=[{},{},{},{}]",
            lhs.n, lhs.c, lhs.h, lhs.w, rhs.n, rhs.c, rhs.h, rhs.w
        ));
    }
    Ok(Tensor4 {
        data: lhs
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(lhs, rhs)| lhs + rhs)
            .collect(),
        n: lhs.n,
        c: lhs.c,
        h: lhs.h,
        w: lhs.w,
    })
}

impl Tensor4 {
    fn validate_len(&self, name: &str) -> Result<(), String> {
        let expected = self.n * self.c * self.h * self.w;
        if self.data.len() == expected {
            return Ok(());
        }
        Err(format!(
            "xasr tensor '{name}' has {} values, expected {expected}",
            self.data.len()
        ))
    }

    fn get(&self, n: usize, c: usize, h: usize, w: usize) -> f32 {
        self.data[((n * self.c + c) * self.h + h) * self.w + w]
    }

    fn set(&mut self, n: usize, c: usize, h: usize, w: usize, value: f32) {
        self.data[((n * self.c + c) * self.h + h) * self.w + w] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{GgufTensorDataReader, read_gguf_metadata};
    use crate::models::xasr_zipformer::encoder_weights::load_xasr_encoder_weights;
    use crate::models::xasr_zipformer::runtime_contract::parse_xasr_zipformer_execution_metadata;
    use std::path::Path;

    #[test]
    fn conv2d_groups_match_manual_depthwise_reference() {
        let input = Tensor4 {
            data: vec![
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                10.0, 20.0, 30.0, //
                40.0, 50.0, 60.0,
            ],
            n: 1,
            c: 2,
            h: 2,
            w: 3,
        };
        let weights = XasrConv2dWeights {
            weight: crate::models::xasr_zipformer::weights::NamedTensor {
                name: "dw".to_string(),
                dims: vec![2, 1, 1, 2],
                values: vec![1.0, 2.0, 10.0, 20.0],
            },
            bias: vec![0.5, 1.0],
        };
        let out = conv2d_nchw(&input, &weights, 1, 1, (0, 0, 0, 0), 2).unwrap();
        assert_eq!(
            out.data,
            vec![5.5, 8.5, 14.5, 17.5, 501.0, 801.0, 1401.0, 1701.0]
        );
        assert_eq!((out.n, out.c, out.h, out.w), (1, 2, 2, 2));
    }

    #[test]
    fn conv1d_depthwise_valid_and_same_match_manual_reference() {
        let weights = XasrConv1dWeights {
            weight: crate::models::xasr_zipformer::weights::NamedTensor {
                name: "dw1d".to_string(),
                dims: vec![3, 1, 2],
                values: vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            },
            bias: vec![0.5, 1.0],
        };
        let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];

        let valid = depthwise_conv1d_valid_channel_major(&input, 2, 4, &weights).unwrap();
        assert_eq!(valid, vec![14.5, 20.5, 1401.0, 2001.0]);

        let same = depthwise_conv1d_same_channel_major(&input, 2, 4, &weights).unwrap();
        assert_eq!(
            same,
            vec![8.5, 14.5, 20.5, 11.5, 801.0, 1401.0, 2001.0, 1101.0]
        );
    }

    #[test]
    fn compact_relative_positional_encoding_keeps_bias_channel() {
        let encoding = compact_relative_positional_encoding(3, 2, 4);
        assert_eq!(encoding.len(), (2 + 2 * 3 - 1) * 4);
        for row in encoding.chunks_exact(4) {
            assert_eq!(row[3], 1.0);
        }
    }

    #[test]
    fn streaming_key_padding_mask_keeps_recent_left_context_and_current_frames() {
        let mask = streaming_key_padding_mask(5, 3, 2).unwrap();
        assert_eq!(
            mask,
            vec![true, true, true, false, false, false, false, false]
        );
    }

    #[test]
    #[ignore = "host-local: compares X-ASR stack1 downsample reference with exported ONNX debug tensors"]
    fn stack1_downsample_reference_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let input_path = root.join("oracle-stack0-debug-480ms.layer1.f32");
        let padded_path = root.join("oracle-downsample-stack1-debug-480ms.padded.f32");
        let reshaped_path = root.join("oracle-downsample-stack1-debug-480ms.reshaped.f32");
        let output_path = root.join("oracle-downsample-stack1-debug-480ms.output.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !input_path.exists()
            || !padded_path.exists()
            || !reshaped_path.exists()
            || !output_path.exists()
            || !pack.exists()
        {
            eprintln!("skipping: missing stack1 downsample oracle files");
            return;
        }

        let input = read_f32_file(&input_path);
        let expected_padded = read_f32_file(&padded_path);
        let expected_reshaped = read_f32_file(&reshaped_path);
        let expected_output = read_f32_file(&output_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let stack1 = &weights.stacks[1];
        let downsample = downsample_streaming_reference(
            &input,
            24,
            metadata.encoder_dims[0],
            metadata.encoder_dims[1],
            stack1.downsample_bias.as_ref().expect("stack1 downsample"),
        )
        .expect("downsample");

        assert_eq!(downsample.frames, 12);
        assert_eq!(downsample.dim, 256);
        assert_max_abs_diff(
            "stack1 downsample padded",
            &downsample.padded_rows,
            &expected_padded,
            2.0e-2,
        );
        assert_max_abs_diff(
            "stack1 downsample reshaped",
            &downsample.reshaped_rows,
            &expected_reshaped,
            2.0e-2,
        );
        assert_max_abs_diff(
            "stack1 downsample output",
            &downsample.rows,
            &expected_output,
            2.0e-2,
        );
    }

    #[test]
    #[ignore = "host-local: compares X-ASR stack1 upsample/out-combiner reference with exported ONNX debug tensors"]
    fn stack1_out_combiner_reference_matches_onnx_debug_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let original_path = root.join("oracle-downsample-stack1-debug-480ms.padded.f32");
        let stack1_path = root.join("oracle-stack1-debug-480ms.layer1.f32");
        let upsample_path = root.join("oracle-stack1-combine-debug-480ms.upsample.f32");
        let slice_path = root.join("oracle-stack1-combine-debug-480ms.slice.f32");
        let out_combiner_path = root.join("oracle-stack1-combine-debug-480ms.out_combiner.f32");
        let padded_path = root.join("oracle-stack1-combine-debug-480ms.padded.f32");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !original_path.exists()
            || !stack1_path.exists()
            || !upsample_path.exists()
            || !slice_path.exists()
            || !out_combiner_path.exists()
            || !padded_path.exists()
            || !pack.exists()
        {
            eprintln!("skipping: missing stack1 out-combiner oracle files");
            return;
        }

        let original = read_f32_file(&original_path);
        let stack1 = read_f32_file(&stack1_path);
        let expected_upsample = read_f32_file(&upsample_path);
        let expected_slice = read_f32_file(&slice_path);
        let expected_out_combiner = read_f32_file(&out_combiner_path);
        let expected_padded = read_f32_file(&padded_path);
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("metadata parse");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("weights");
        let stack = &weights.stacks[1];

        let upsample = upsample_streaming_reference(&stack1, 12, 256, 2, 24).expect("upsample");
        let out_combiner = bypass_reference(
            &original,
            &upsample,
            stack
                .out_combiner_bypass_scale
                .as_ref()
                .expect("stack1 out combiner"),
            24,
            256,
        )
        .expect("out combiner");
        let padded =
            resize_frame_rows_reference(&out_combiner, 24, 256, 512).expect("resize to stack2");

        assert_max_abs_diff("stack1 upsample", &upsample, &expected_upsample, 2.0e-2);
        assert_max_abs_diff("stack1 upsample slice", &upsample, &expected_slice, 2.0e-2);
        assert_max_abs_diff(
            "stack1 out combiner",
            &out_combiner,
            &expected_out_combiner,
            2.0e-2,
        );
        assert_max_abs_diff("stack1 combiner padded", &padded, &expected_padded, 2.0e-2);
    }

    fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert_eq!(lhs.len(), rhs.len());
        lhs.iter()
            .zip(rhs.iter())
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0.0_f32, f32::max)
    }

    fn assert_max_abs_diff(name: &str, lhs: &[f32], rhs: &[f32], tolerance: f32) {
        let diff = max_abs_diff(lhs, rhs);
        assert!(
            diff <= tolerance,
            "{name} max abs diff {diff} exceeds tolerance {tolerance}"
        );
    }

    fn add_same_shape(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        assert_eq!(lhs.len(), rhs.len());
        lhs.iter()
            .zip(rhs.iter())
            .map(|(&lhs, &rhs)| lhs + rhs)
            .collect()
    }

    fn read_f32_file(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("read f32 file");
        assert_eq!(bytes.len() % 4, 0, "unaligned f32 file {}", path.display());
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }
}
