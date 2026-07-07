//! Decoder layer blocks for the shared `nn/` IR boundary.
//!
//! `seq2seq_layer` is the reusable, config-driven cross-attending decoder layer:
//! pre-norm causal self-attention with an incremental f16 KV cache, pre-norm
//! cross-attention over precomputed encoder K/V, and a ReLU feed-forward — the
//! standard Whisper/cohere/paraformer seq2seq decoder shape. It is a faithful
//! extraction of cohere-transcribe's hand-written `apply_decoder_layer`: the
//! same ggml op sequence, step labels, f16/f32 view strides, `1/sqrt(head_dim)`
//! scale, and `flash_attn_ext` calls, so a migrated family stays bit-identical.
//!
//! Like every `nn/` builder it is generic over the caller's error `E` via a
//! `map_err` closure and owns no model-specific error type. Cross-attention K/V
//! are precomputed and passed in as a handle — the block only *views* them,
//! never recomputes or writes them. The self-attention causal mask is built but
//! its f16 bit upload is **deferred**: the block returns the mask tensor + bits
//! so the caller applies the upload after `set_output`, preserving the exact
//! side-effect ordering of the original graph.

use std::sync::Arc;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor,
    GgmlPersistentGraphSession, GgmlRopeExtParams, GgmlStaticTensor, GgmlStaticTensorArena,
};
use crate::nn::attn::{
    AttentionHeadLayout, AttentionReshapeSteps, STANDARD_HEAD_PERMUTE_AXES,
    reshape_projection_to_attention_heads,
};
use crate::nn::ffn::{
    FeedForwardActivation, GatedFeedForwardResidualSteps, apply_gated_feed_forward_residual,
};
use crate::nn::norm::{
    AffineLayerNormSteps, RmsNormSteps, apply_affine_layer_norm, apply_rms_norm,
};

/// Reused decode graphs with in-place resident KV are only correct on the
/// single-backend GPU path. CPU direct execution mis-recomputes reused graphs
/// with in-place KV writes, and the multi-backend scheduler drops refreshed
/// per-token inputs.
pub(crate) fn reusable_decode_graph_supported(
    backend: GgmlCpuGraphBackend,
    uses_scheduler: bool,
) -> bool {
    backend.is_gpu_class() && !uses_scheduler
}

pub(crate) fn reusable_decode_graph_supported_for_runner(runner: &GgmlCpuGraphRunner) -> bool {
    reusable_decode_graph_supported(runner.backend_kind(), runner.uses_scheduler())
}

/// Scalar/shape knobs for one seq2seq (cross-attending) decoder block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Seq2SeqLayerConfig {
    pub hidden: usize,
    pub attention_heads: usize,
    pub head_dim: usize,
    /// Tokens processed this step (prefill = N, incremental decode = 1).
    pub token_count: usize,
    /// Number of independent decoder streams represented by this graph step.
    /// `1` preserves the existing single-stream seq2seq graph.
    pub n_seq: usize,
    /// Accumulated self-KV length read back (prefix + this step).
    pub total_token_count: usize,
    /// Write slot into the self-KV cache for this step.
    pub position_offset: usize,
    pub layer_norm_epsilon: f32,
    /// FFN activation between `ffn_up`/`ffn_down` (Whisper/cohere/paraformer use
    /// ReLU; FireRedASR's decoder FFN uses GELU). Additive field, defaults kept
    /// explicit at each call site.
    pub ffn_activation: FeedForwardActivation,
    /// Self-KV persistent f16 storage geometry:
    /// `n_seq == 1`: `[head_dim, max_positions, heads]`;
    /// `n_seq > 1`: `[head_dim, max_positions, heads, n_seq]`.
    pub self_kv_max_positions: usize,
    /// Precomputed cross-KV persistent f32 geometry:
    /// `n_seq == 1`: `[hidden_size, frame_count]`;
    /// `n_seq > 1`: `[hidden_size, frame_count, n_seq]`.
    pub cross_frame_count: usize,
    pub cross_hidden_size: usize,
}

/// Handle to one layer's persistent self-attention KV cache (f16,
/// `[head_dim, max_positions, heads]` or batched
/// `[head_dim, max_positions, heads, n_seq]`). The block appends this step's
/// K/V at `position_offset` then reads `total_token_count` rows back.
#[derive(Clone, Copy)]
pub(crate) struct SelfKvHandle<'a> {
    pub key: GgmlCpuTensor<'a>,
    pub value: GgmlCpuTensor<'a>,
    /// Optional runtime row-index tensor for `ggml_set_rows`-based KV writes.
    /// When absent, the legacy view+copy writer uses `position_offset`.
    pub row_indices: Option<GgmlCpuTensor<'a>>,
    /// Optional externally-managed additive mask for fixed-span reusable graphs.
    /// When present, the block uses it directly instead of allocating/defer-
    /// uploading its legacy prompt causal mask. Batched seq2seq callers pass
    /// `[max_positions, token_count, 1, n_seq]` planes here.
    pub attention_mask: Option<GgmlCpuTensor<'a>>,
}

/// Handle to one layer's precomputed cross-attention KV (f32,
/// `[hidden_size, frame_count]` or batched `[hidden_size, frame_count, n_seq]`).
/// The block only *views* these.
#[derive(Clone, Copy)]
pub(crate) struct CrossKvHandle<'a> {
    pub key: GgmlCpuTensor<'a>,
    pub value: GgmlCpuTensor<'a>,
}

/// Per-block graph weights in submodule order: self-attn (norm,q,k,v,o) →
/// cross-attn (norm,q,o; k/v come from `CrossKvHandle`, no in-step projection)
/// → FFN (norm,up,down). `_weight` are `mul_mat`-ready; biases/norms are f32.
#[derive(Clone, Copy)]
pub(crate) struct Seq2SeqLayerWeights<'a> {
    pub self_attn_norm_weight: GgmlCpuTensor<'a>,
    pub self_attn_norm_bias: GgmlCpuTensor<'a>,
    pub self_attn_q_weight: GgmlCpuTensor<'a>,
    pub self_attn_q_bias: GgmlCpuTensor<'a>,
    pub self_attn_k_weight: GgmlCpuTensor<'a>,
    pub self_attn_k_bias: GgmlCpuTensor<'a>,
    pub self_attn_v_weight: GgmlCpuTensor<'a>,
    pub self_attn_v_bias: GgmlCpuTensor<'a>,
    pub self_attn_o_weight: GgmlCpuTensor<'a>,
    pub self_attn_o_bias: GgmlCpuTensor<'a>,
    pub cross_attn_norm_weight: GgmlCpuTensor<'a>,
    pub cross_attn_norm_bias: GgmlCpuTensor<'a>,
    pub cross_attn_q_weight: GgmlCpuTensor<'a>,
    pub cross_attn_q_bias: GgmlCpuTensor<'a>,
    pub cross_attn_o_weight: GgmlCpuTensor<'a>,
    pub cross_attn_o_bias: GgmlCpuTensor<'a>,
    pub ffn_norm_weight: GgmlCpuTensor<'a>,
    pub ffn_norm_bias: GgmlCpuTensor<'a>,
    pub ffn_up_weight: GgmlCpuTensor<'a>,
    pub ffn_up_bias: GgmlCpuTensor<'a>,
    pub ffn_down_weight: GgmlCpuTensor<'a>,
    pub ffn_down_bias: GgmlCpuTensor<'a>,
}

/// Residual-boundary / projection taps matching cohere's first-layer prompt
/// debug capture.
#[derive(Clone, Copy)]
pub(crate) struct Seq2SeqLayerTaps<'a> {
    pub self_attn_norm: GgmlCpuTensor<'a>,
    pub q_proj: GgmlCpuTensor<'a>,
    pub k_proj: GgmlCpuTensor<'a>,
    pub v_proj: GgmlCpuTensor<'a>,
    pub after_self_attn: GgmlCpuTensor<'a>,
    pub after_cross_attn: GgmlCpuTensor<'a>,
    pub after_ffn: GgmlCpuTensor<'a>,
}

pub(crate) struct Seq2SeqLayerOutput<'a> {
    pub output: GgmlCpuTensor<'a>,
    pub taps: Seq2SeqLayerTaps<'a>,
    /// `Some((mask, bits))` when `token_count > 1`. The caller must defer this
    /// f16 upload to after `set_output`, preserving the original ordering.
    pub deferred_self_mask: Option<(GgmlCpuTensor<'a>, Arc<[u16]>)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Seq2SeqLayerStackLength {
    pub layers: usize,
    pub cross_layers: usize,
    pub self_kv_layers: usize,
}

pub(crate) fn seq2seq_layer_stack<'a, L, C, S, E, F, M>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    mut state: GgmlCpuTensor<'a>,
    layers: &[L],
    cross_layers: &[C],
    self_kv_layers: &[S],
    length_error: M,
    mut apply_layer: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: FnMut(
        &mut GgmlCpuGraphBuilder<'a>,
        GgmlCpuTensor<'a>,
        usize,
        &L,
        &C,
        &S,
    ) -> Result<GgmlCpuTensor<'a>, E>,
    M: FnOnce(Seq2SeqLayerStackLength) -> E,
{
    if layers.len() != cross_layers.len() || layers.len() != self_kv_layers.len() {
        return Err(length_error(Seq2SeqLayerStackLength {
            layers: layers.len(),
            cross_layers: cross_layers.len(),
            self_kv_layers: self_kv_layers.len(),
        }));
    }
    for layer_idx in 0..layers.len() {
        state = apply_layer(
            graph,
            state,
            layer_idx,
            &layers[layer_idx],
            &cross_layers[layer_idx],
            &self_kv_layers[layer_idx],
        )?;
    }
    Ok(state)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Seq2SeqLayerStackPosition {
    pub layer_index: usize,
    pub is_last: bool,
}

pub(crate) fn seq2seq_indexed_layer_stack<'a, L, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    mut state: GgmlCpuTensor<'a>,
    layers: &[L],
    mut apply_layer: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: FnMut(
        &mut GgmlCpuGraphBuilder<'a>,
        GgmlCpuTensor<'a>,
        Seq2SeqLayerStackPosition,
        &L,
    ) -> Result<GgmlCpuTensor<'a>, E>,
{
    let last_layer_index = layers.len().saturating_sub(1);
    for (layer_index, layer) in layers.iter().enumerate() {
        state = apply_layer(
            graph,
            state,
            Seq2SeqLayerStackPosition {
                layer_index,
                is_last: layer_index == last_layer_index,
            },
            layer,
        )?;
    }
    Ok(state)
}

fn overflow(step: &'static str) -> (&'static str, GgmlCpuGraphError) {
    (
        step,
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "nn::decoder shape overflow",
        },
    )
}

fn apply_linear_with_bias<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    weight: GgmlCpuTensor<'a>,
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

fn reshape_to_heads<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    sequence_len: usize,
    attention_heads: usize,
    n_seq: usize,
    step: &'static str,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    if n_seq == 0 {
        return Err(map_err_tuple(
            map_err,
            (
                step,
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "seq2seq n_seq must be positive",
                },
            ),
        ));
    }
    if n_seq > 1 {
        let reshaped = graph
            .reshape_4d(projection, head_dim, attention_heads, sequence_len, n_seq)
            .map_err(|source| map_err("ggml_reshape_4d(attn_heads)", source))?;
        let permuted = graph
            .permute(
                reshaped,
                STANDARD_HEAD_PERMUTE_AXES[0],
                STANDARD_HEAD_PERMUTE_AXES[1],
                STANDARD_HEAD_PERMUTE_AXES[2],
                STANDARD_HEAD_PERMUTE_AXES[3],
            )
            .map_err(|source| map_err("ggml_permute(attn_heads)", source))?;
        return graph.cont(permuted).map_err(|source| map_err(step, source));
    }
    reshape_projection_to_attention_heads(
        graph,
        projection,
        AttentionHeadLayout {
            head_dim,
            attention_heads,
            sequence_len,
        },
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(attn_heads)",
            permute: "ggml_permute(attn_heads)",
            cont: step,
        },
        map_err,
    )
}

/// Append a step's projected K or V into the persistent f16 self-KV cache, one
/// head at a time, forcing write-before-read ordering via side-effect roots.
fn write_self_kv<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    source: GgmlCpuTensor<'a>,
    destination: GgmlCpuTensor<'a>,
    hidden: usize,
    token_count: usize,
    attention_heads: usize,
    max_positions: usize,
    position_offset: usize,
    step: &'static str,
    map_err: F,
) -> Result<(), E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let head_dim = hidden
        .checked_div(attention_heads)
        .ok_or_else(|| map_err_tuple(map_err, overflow(step)))?;
    let src_element_size = std::mem::size_of::<f32>();
    let dst_element_size = std::mem::size_of::<u16>();
    let head_elements = head_dim
        .checked_mul(token_count)
        .ok_or_else(|| map_err_tuple(map_err, overflow(step)))?;
    let src_head_stride_bytes = head_elements
        .checked_mul(src_element_size)
        .ok_or_else(|| map_err_tuple(map_err, overflow(step)))?;
    for head_idx in 0..attention_heads {
        let src = graph
            .view_1d(
                source,
                head_elements,
                head_idx
                    .checked_mul(src_head_stride_bytes)
                    .ok_or_else(|| map_err_tuple(map_err, overflow(step)))?,
            )
            .map_err(|source| map_err(step, source))?;
        let dst_offset_elements = head_idx
            .checked_mul(max_positions)
            .and_then(|value| value.checked_mul(head_dim))
            .and_then(|value| value.checked_add(position_offset.saturating_mul(head_dim)))
            .ok_or_else(|| map_err_tuple(map_err, overflow(step)))?;
        let dst = graph
            .view_1d(
                destination,
                head_elements,
                dst_offset_elements
                    .checked_mul(dst_element_size)
                    .ok_or_else(|| map_err_tuple(map_err, overflow(step)))?,
            )
            .map_err(|source| map_err(step, source))?;
        let write = graph
            .cpy(src, dst)
            .map_err(|source| map_err(step, source))?;
        graph
            .add_side_effect_root(write)
            .map_err(|source| map_err(step, source))?;
    }
    Ok(())
}

/// View the f16 self-KV cache as `[head_dim, sequence_len, heads]`.
fn view_self_kv_heads<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    tensor: GgmlCpuTensor<'a>,
    hidden: usize,
    head_dim: usize,
    sequence_len: usize,
    attention_heads: usize,
    max_positions: usize,
    n_seq: usize,
    step: &'static str,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    if n_seq == 0 {
        return Err(map_err_tuple(
            map_err,
            (
                step,
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "seq2seq n_seq must be positive",
                },
            ),
        ));
    }
    let element_size = std::mem::size_of::<u16>();
    let nb1 = head_dim
        .checked_mul(element_size)
        .ok_or_else(|| map_err_tuple(map_err, overflow(step)))?;
    let nb2 = hidden
        .checked_div(attention_heads)
        .and_then(|value| value.checked_mul(max_positions))
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| map_err_tuple(map_err, overflow(step)))?;
    if n_seq > 1 {
        let nb3 = head_dim
            .checked_mul(max_positions)
            .and_then(|value| value.checked_mul(attention_heads))
            .and_then(|value| value.checked_mul(element_size))
            .ok_or_else(|| map_err_tuple(map_err, overflow(step)))?;
        return graph
            .view_4d(
                tensor,
                head_dim,
                sequence_len,
                attention_heads,
                n_seq,
                nb1,
                nb2,
                nb3,
                0,
            )
            .map_err(|source| map_err(step, source));
    }
    graph
        .view_3d(tensor, head_dim, sequence_len, attention_heads, nb1, nb2, 0)
        .map_err(|source| map_err(step, source))
}

/// View the precomputed f32 cross-KV cache as `[head_dim, sequence_len, heads]`.
fn view_cross_kv_heads<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    tensor: GgmlCpuTensor<'a>,
    hidden: usize,
    head_dim: usize,
    sequence_len: usize,
    attention_heads: usize,
    n_seq: usize,
    step: &'static str,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    if n_seq == 0 {
        return Err(map_err_tuple(
            map_err,
            (
                step,
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "seq2seq n_seq must be positive",
                },
            ),
        ));
    }
    let element_size = std::mem::size_of::<f32>();
    let nb1 = hidden
        .checked_mul(element_size)
        .ok_or_else(|| map_err_tuple(map_err, overflow(step)))?;
    let nb2 = head_dim
        .checked_mul(element_size)
        .ok_or_else(|| map_err_tuple(map_err, overflow(step)))?;
    if n_seq > 1 {
        let nb3 = hidden
            .checked_mul(sequence_len)
            .and_then(|value| value.checked_mul(element_size))
            .ok_or_else(|| map_err_tuple(map_err, overflow(step)))?;
        return graph
            .view_4d(
                tensor,
                head_dim,
                sequence_len,
                attention_heads,
                n_seq,
                nb1,
                nb2,
                nb3,
                0,
            )
            .map_err(|source| map_err(step, source));
    }
    graph
        .view_3d(tensor, head_dim, sequence_len, attention_heads, nb1, nb2, 0)
        .map_err(|source| map_err(step, source))
}

fn merge_seq2seq_attention_context<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    context: GgmlCpuTensor<'a>,
    hidden: usize,
    token_count: usize,
    n_seq: usize,
    step: &'static str,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    if n_seq == 0 {
        return Err(map_err_tuple(
            map_err,
            (
                step,
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "seq2seq n_seq must be positive",
                },
            ),
        ));
    }
    let output_tokens = token_count
        .checked_mul(n_seq)
        .ok_or_else(|| map_err_tuple(map_err, overflow(step)))?;
    if n_seq == 1 || token_count == 1 {
        return graph
            .reshape_2d(context, hidden, output_tokens)
            .map_err(|source| map_err(step, source));
    }
    let merged = graph
        .permute(context, 0, 2, 1, 3)
        .map_err(|source| map_err("ggml_permute(seq2seq_context_merge)", source))?;
    let merged = graph
        .cont(merged)
        .map_err(|source| map_err("ggml_cont(seq2seq_context_merge)", source))?;
    graph
        .reshape_2d(merged, hidden, output_tokens)
        .map_err(|source| map_err(step, source))
}

fn map_err_tuple<E, F>(map_err: F, parts: (&'static str, GgmlCpuGraphError)) -> E
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E,
{
    map_err(parts.0, parts.1)
}

/// Build dense f16 causal-mask bits `[token_count, token_count]` (-inf above
/// the diagonal).
pub(crate) fn build_causal_mask_f16_bits<E, F>(
    token_count: usize,
    step: &'static str,
    map_err: F,
) -> Result<Arc<[u16]>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let total = token_count
        .checked_mul(token_count)
        .ok_or_else(|| map_err_tuple(map_err, overflow(step)))?;
    let mut values = vec![f32_to_f16_bits(0.0); total];
    let neg_inf_bits = f32_to_f16_bits(-f32::INFINITY);
    for token_idx in 0..token_count {
        let row_offset = token_idx
            .checked_mul(token_count)
            .ok_or_else(|| map_err_tuple(map_err, overflow(step)))?;
        for kv_idx in 0..token_count {
            if kv_idx > token_idx {
                values[row_offset + kv_idx] = neg_inf_bits;
            }
        }
    }
    Ok(Arc::<[u16]>::from(values.into_boxed_slice()))
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa_with_hidden = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let mut half_mantissa = (mantissa_with_hidden >> shift) as u16;
        let round_bit = 1_u32 << shift.saturating_sub(1);
        if shift > 0
            && (mantissa_with_hidden & round_bit) != 0
            && ((mantissa_with_hidden & (round_bit - 1)) != 0 || (half_mantissa & 1) != 0)
        {
            half_mantissa = half_mantissa.wrapping_add(1);
        }
        return sign | half_mantissa;
    }
    let mut half = sign | ((half_exponent as u16) << 10) | ((mantissa >> 13) as u16);
    if (mantissa & 0x1000) != 0 {
        half = half.wrapping_add(1);
    }
    half
}

/// Assemble one seq2seq decoder block, reproducing cohere's hand-written op
/// sequence bit-identically. The caller has already validated the self-KV step
/// invariant and must push `deferred_self_mask` onto its upload queue.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub(crate) fn seq2seq_layer<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    config: Seq2SeqLayerConfig,
    weights: Seq2SeqLayerWeights<'a>,
    self_kv: SelfKvHandle<'a>,
    cross_kv: CrossKvHandle<'a>,
    map_err: F,
) -> Result<Seq2SeqLayerOutput<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let hidden = config.hidden;
    let heads = config.attention_heads;
    let head_dim = config.head_dim;
    let token_count = config.token_count;
    let n_seq = config.n_seq;
    if n_seq == 0 {
        return Err(map_err_tuple(
            map_err,
            (
                "seq2seq_layer",
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "seq2seq n_seq must be positive",
                },
            ),
        ));
    }
    if n_seq > 1 && self_kv.row_indices.is_none() {
        return Err(map_err_tuple(
            map_err,
            (
                "seq2seq_layer",
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "batched seq2seq self-KV requires row indices",
                },
            ),
        ));
    }
    if n_seq > 1 && self_kv.attention_mask.is_none() {
        return Err(map_err_tuple(
            map_err,
            (
                "seq2seq_layer",
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "batched seq2seq self-attention requires a fixed mask",
                },
            ),
        ));
    }
    let scale = 1.0 / (head_dim as f32).sqrt();

    // ----- Self-attention (causal, incremental f16 KV cache) -----
    let self_attn_input = input;
    let attn_norm = apply_affine_layer_norm(
        graph,
        input,
        config.layer_norm_epsilon,
        weights.self_attn_norm_weight,
        weights.self_attn_norm_bias,
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "decoder_self_attn_norm",
            bias: "decoder_self_attn_norm",
        },
        map_err,
    )?;
    let q_proj = apply_linear_with_bias(
        graph,
        attn_norm,
        weights.self_attn_q_weight,
        weights.self_attn_q_bias,
        "decoder_self_attn_q",
        map_err,
    )?;
    let k_proj = apply_linear_with_bias(
        graph,
        attn_norm,
        weights.self_attn_k_weight,
        weights.self_attn_k_bias,
        "decoder_self_attn_k",
        map_err,
    )?;
    let v_proj = apply_linear_with_bias(
        graph,
        attn_norm,
        weights.self_attn_v_weight,
        weights.self_attn_v_bias,
        "decoder_self_attn_v",
        map_err,
    )?;
    let q = reshape_to_heads(
        graph,
        q_proj,
        head_dim,
        token_count,
        heads,
        n_seq,
        "decoder_self_q",
        map_err,
    )?;
    let k = reshape_to_heads(
        graph,
        k_proj,
        head_dim,
        token_count,
        heads,
        n_seq,
        "decoder_self_k",
        map_err,
    )?;
    let v = reshape_to_heads(
        graph,
        v_proj,
        head_dim,
        token_count,
        heads,
        n_seq,
        "decoder_self_v",
        map_err,
    )?;
    let (self_k_source, self_v_source) = if let Some(row_indices) = self_kv.row_indices {
        let k = graph
            .set_rows(self_kv.key, k, row_indices)
            .map_err(|source| map_err("decoder_self_k_set_rows", source))?;
        let v = graph
            .set_rows(self_kv.value, v, row_indices)
            .map_err(|source| map_err("decoder_self_v_set_rows", source))?;
        (k, v)
    } else {
        write_self_kv(
            graph,
            k,
            self_kv.key,
            hidden,
            token_count,
            heads,
            config.self_kv_max_positions,
            config.position_offset,
            "decoder_self_k_cache",
            map_err,
        )?;
        write_self_kv(
            graph,
            v,
            self_kv.value,
            hidden,
            token_count,
            heads,
            config.self_kv_max_positions,
            config.position_offset,
            "decoder_self_v_cache",
            map_err,
        )?;
        (self_kv.key, self_kv.value)
    };
    let k = view_self_kv_heads(
        graph,
        self_k_source,
        hidden,
        head_dim,
        config.total_token_count,
        heads,
        config.self_kv_max_positions,
        n_seq,
        "decoder_self_k_persistent",
        map_err,
    )?;
    let v = view_self_kv_heads(
        graph,
        self_v_source,
        hidden,
        head_dim,
        config.total_token_count,
        heads,
        config.self_kv_max_positions,
        n_seq,
        "decoder_self_v_persistent",
        map_err,
    )?;
    let (self_mask, deferred_self_mask) = if let Some(mask) = self_kv.attention_mask {
        (Some(mask), None)
    } else if token_count == 1 {
        (None, None)
    } else {
        let mask = graph
            .new_tensor_3d_f16(
                token_count,
                token_count,
                1,
                "cohere_decoder_layer_self_mask",
            )
            .map_err(|source| map_err("ggml_new_tensor_3d(layer_self_mask)", source))?;
        graph
            .set_input(mask)
            .map_err(|source| map_err("ggml_set_input(layer_self_mask)", source))?;
        let bits = build_causal_mask_f16_bits(token_count, "decoder_self_mask", map_err)?;
        (Some(mask), Some((mask, bits)))
    };
    let context = graph
        .flash_attn_ext(q, k, v, self_mask, scale, 0.0, 0.0)
        .map_err(|source| map_err("ggml_flash_attn_ext(self)", source))?;
    let context = merge_seq2seq_attention_context(
        graph,
        context,
        hidden,
        token_count,
        n_seq,
        "ggml_reshape_2d(self_merge)",
        map_err,
    )?;
    let self_attn = apply_linear_with_bias(
        graph,
        context,
        weights.self_attn_o_weight,
        weights.self_attn_o_bias,
        "decoder_self_attn_o",
        map_err,
    )?;
    let state = graph
        .add(self_attn_input, self_attn)
        .map_err(|source| map_err("ggml_add(self_residual)", source))?;
    let after_self_attn = state;

    // ----- Cross-attention (queries from state, K/V from precomputed cache) -----
    let cross_attn_input = state;
    let cross_norm = apply_affine_layer_norm(
        graph,
        state,
        config.layer_norm_epsilon,
        weights.cross_attn_norm_weight,
        weights.cross_attn_norm_bias,
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "decoder_cross_attn_norm",
            bias: "decoder_cross_attn_norm",
        },
        map_err,
    )?;
    let q = apply_linear_with_bias(
        graph,
        cross_norm,
        weights.cross_attn_q_weight,
        weights.cross_attn_q_bias,
        "decoder_cross_attn_q",
        map_err,
    )?;
    let q = reshape_to_heads(
        graph,
        q,
        head_dim,
        token_count,
        heads,
        n_seq,
        "decoder_cross_q",
        map_err,
    )?;
    let cross_k = view_cross_kv_heads(
        graph,
        cross_kv.key,
        config.cross_hidden_size,
        head_dim,
        config.cross_frame_count,
        heads,
        n_seq,
        "decoder_cross_k",
        map_err,
    )?;
    let cross_v = view_cross_kv_heads(
        graph,
        cross_kv.value,
        config.cross_hidden_size,
        head_dim,
        config.cross_frame_count,
        heads,
        n_seq,
        "decoder_cross_v",
        map_err,
    )?;
    let context = graph
        .flash_attn_ext(q, cross_k, cross_v, None, scale, 0.0, 0.0)
        .map_err(|source| map_err("ggml_flash_attn_ext(cross)", source))?;
    let context = merge_seq2seq_attention_context(
        graph,
        context,
        hidden,
        token_count,
        n_seq,
        "ggml_reshape_2d(cross_merge)",
        map_err,
    )?;
    let cross_attn = apply_linear_with_bias(
        graph,
        context,
        weights.cross_attn_o_weight,
        weights.cross_attn_o_bias,
        "decoder_cross_attn_o",
        map_err,
    )?;
    let state = graph
        .add(cross_attn_input, cross_attn)
        .map_err(|source| map_err("ggml_add(cross_residual)", source))?;
    let after_cross_attn = state;

    // ----- Feed-forward (ReLU) -----
    let ffn_input = state;
    let ffn_norm = apply_affine_layer_norm(
        graph,
        state,
        config.layer_norm_epsilon,
        weights.ffn_norm_weight,
        weights.ffn_norm_bias,
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "decoder_ffn_norm",
            bias: "decoder_ffn_norm",
        },
        map_err,
    )?;
    let ff = apply_linear_with_bias(
        graph,
        ffn_norm,
        weights.ffn_up_weight,
        weights.ffn_up_bias,
        "decoder_ffn_up",
        map_err,
    )?;
    let ff = match config.ffn_activation {
        FeedForwardActivation::Relu => graph
            .relu(ff)
            .map_err(|source| map_err("ggml_relu(ffn_up)", source))?,
        FeedForwardActivation::Gelu => graph
            .gelu(ff)
            .map_err(|source| map_err("ggml_gelu(ffn_up)", source))?,
        FeedForwardActivation::GeluErf => graph
            .gelu_erf(ff)
            .map_err(|source| map_err("ggml_gelu_erf(ffn_up)", source))?,
        FeedForwardActivation::Silu => graph
            .silu(ff)
            .map_err(|source| map_err("ggml_silu(ffn_up)", source))?,
    };
    let ff = apply_linear_with_bias(
        graph,
        ff,
        weights.ffn_down_weight,
        weights.ffn_down_bias,
        "decoder_ffn_down",
        map_err,
    )?;
    let state = graph
        .add(ffn_input, ff)
        .map_err(|source| map_err("ggml_add(ffn_residual)", source))?;
    let after_ffn = state;

    Ok(Seq2SeqLayerOutput {
        output: state,
        taps: Seq2SeqLayerTaps {
            self_attn_norm: attn_norm,
            q_proj,
            k_proj,
            v_proj,
            after_self_attn,
            after_cross_attn,
            after_ffn,
        },
        deferred_self_mask,
    })
}

/// Scalar / shape / mode knobs for one decoder-only LLM transformer block
/// (Qwen3-ASR shape: fused-or-split GQA self-attention with QK-norm + RoPE,
/// then SwiGLU FFN, no biases).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LlmLayerConfig {
    pub d_model: usize,
    pub head_dim: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub q_width: usize,
    pub k_width: usize,
    pub v_width: usize,
    /// Tokens processed for each sequence in this graph step. Incremental
    /// decode is `1`; prefill can be `>1`.
    pub token_count: usize,
    /// Number of independent decode streams represented by this graph step.
    /// `1` is today's single-stream shape.
    pub n_seq: usize,
    /// KV-cache read-back geometry for this step.
    pub total_tokens: usize,
    pub rms_norm_epsilon: f32,
    /// Pre-resolved RoPE params (`GgmlRopeExtParams::qwen_neox(head_dim, pos+1, theta)`).
    pub rope: GgmlRopeExtParams,
    /// `false` → manually reshape/repeat/cont K&V across the GQA group;
    /// `true` → leave K&V as `[head_dim, total_tokens, kv_heads]` for native ggml GQA.
    pub use_native_gqa: bool,
    /// `true` → fused `flash_attn_ext` self-attention (the default everywhere it
    /// is numerically trusted); `false` → the unfused mul_mat + soft_max_ext +
    /// mul_mat path for multi-token prefill on backends whose fused flash kernel
    /// mis-handles wide causal queries (HIP/ROCm MMA/TILE; see
    /// `qwen_llm_safe_gpu_prefill_query_tokens_for_backend_name`). Requires an
    /// explicit `attention_mask` when disabled.
    pub use_flash_attention: bool,
}

/// A LoRA side-path for one 2-D linear in the LLM decoder stack.
/// `y = W@x + b_scaled@(a@x)` where `b_scaled` is B pre-multiplied by
/// `alpha/rank` at load time (same convention as Moonshine's `LoraSlot`).
/// Both tensors are arena-resident `GgmlCpuTensor`s, already uploaded by the
/// caller before graph construction begins.
#[derive(Clone, Copy)]
pub(crate) struct LlmLoraSlot<'a> {
    /// `[input_dim, rank]` f32, ne0-major.
    pub a: GgmlCpuTensor<'a>,
    /// `[rank, output_dim]` f32, ne0-major, pre-scaled by `alpha/rank`.
    pub b_scaled: GgmlCpuTensor<'a>,
}

/// Per-block graph weights, submodule order: attn-norm → fused-or-split QKV →
/// q/k-norm → out-proj → ffn-norm → gate/up/down. `qkv_weight` present ⇒ fused
/// path (single `mul_mat` + `view_2d` split); absent ⇒ three separate `mul_mat`.
/// The fused-vs-split decision is already resolved (fail-closed) by the caller.
///
/// LoRA slots are optional and default to `None`. When ANY of `q_lora`,
/// `k_lora`, `v_lora` is `Some`, the QKV projection uses the SPLIT path
/// regardless of `qkv_weight`, so each projection can be individually adapted.
/// Consumers that leave all slots `None` produce a byte-identical graph to the
/// pre-LoRA code path.
#[derive(Clone, Copy)]
pub(crate) struct LlmLayerWeights<'a> {
    pub attn_norm_weight: GgmlCpuTensor<'a>,
    pub qkv_weight: Option<GgmlCpuTensor<'a>>,
    pub q_weight: GgmlCpuTensor<'a>,
    pub k_weight: GgmlCpuTensor<'a>,
    pub v_weight: GgmlCpuTensor<'a>,
    pub q_norm_weight: GgmlCpuTensor<'a>,
    pub k_norm_weight: GgmlCpuTensor<'a>,
    pub output_weight: GgmlCpuTensor<'a>,
    pub ffn_norm_weight: GgmlCpuTensor<'a>,
    pub ffn_gate_weight: GgmlCpuTensor<'a>,
    pub ffn_up_weight: GgmlCpuTensor<'a>,
    pub ffn_down_weight: GgmlCpuTensor<'a>,
    // LoRA side-paths — all None by default (additive, no-op).
    pub q_lora: Option<LlmLoraSlot<'a>>,
    pub k_lora: Option<LlmLoraSlot<'a>>,
    pub v_lora: Option<LlmLoraSlot<'a>>,
    pub output_lora: Option<LlmLoraSlot<'a>>,
    pub ffn_gate_lora: Option<LlmLoraSlot<'a>>,
    pub ffn_up_lora: Option<LlmLoraSlot<'a>>,
    pub ffn_down_lora: Option<LlmLoraSlot<'a>>,
}

/// KV-cache graph tensors (allocated + `set_input`'d by the caller) plus the i32
/// row-index and position input tensors. The block only `set_rows`-writes the new
/// K/V into the history tensors and reads them back; it never allocates them and
/// never uploads their host data (the caller does that around the block,
/// preserving the exact side-effect ordering).
#[derive(Clone, Copy)]
pub(crate) struct LlmKvCacheHandle<'a> {
    pub key_history: GgmlCpuTensor<'a>,
    pub value_history: GgmlCpuTensor<'a>,
    pub row_indices: GgmlCpuTensor<'a>,
    pub positions: GgmlCpuTensor<'a>,
    /// Optional additive attention mask `[total_tokens]` for the self-attention
    /// softmax. `None` ⇒ attend over the exact `total_tokens` history (every row
    /// valid — the growing-KV path). `Some` ⇒ the history is a fixed max-size
    /// tensor and the mask is `0` for valid positions and `-inf` for not-yet-
    /// written rows, so the graph shape is constant across decode steps (the
    /// prerequisite for graph reuse).
    pub attention_mask: Option<GgmlCpuTensor<'a>>,
}

pub(crate) struct LlmLayerOutput<'a> {
    /// Final block output `[d_model, token_count * n_seq]` (post-FFN residual).
    pub output: GgmlCpuTensor<'a>,
    /// Per-token projected K `[head_dim, kv_heads, token_count * n_seq]`
    /// after q/k-norm + RoPE, pre-`set_rows`.
    /// The caller reads this back to write the host KV cache.
    pub projected_k: GgmlCpuTensor<'a>,
    /// Per-token projected V `[head_dim, kv_heads, token_count * n_seq]`
    /// after reshape, pre-`set_rows`.
    pub projected_v: GgmlCpuTensor<'a>,
}

/// Stack-level shape/mode knobs for assembling multiple decoder-only LLM blocks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LlmDecoderStackConfig {
    pub d_model: usize,
    pub head_dim: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub q_width: usize,
    pub k_width: usize,
    pub v_width: usize,
    pub token_count: usize,
    pub n_seq: usize,
    pub rms_norm_epsilon: f32,
    pub rope: GgmlRopeExtParams,
    pub use_native_gqa: bool,
    /// See `LlmLayerConfig::use_flash_attention`.
    pub use_flash_attention: bool,
}

/// Per-step inputs for one composition of an LLM decoder layer stack.
/// The fields that differ between growing-KV and fixed-max reuse paths are
/// exactly `kv_span`, `attention_mask`, and the KV tensor-name scope.
pub(crate) struct LlmDecoderStackInputs<'a> {
    pub state: GgmlCpuTensor<'a>,
    pub row_indices: GgmlCpuTensor<'a>,
    pub positions: GgmlCpuTensor<'a>,
    pub attention_mask: Option<GgmlCpuTensor<'a>>,
    pub kv_span: usize,
    pub key_history_name: &'static str,
    pub value_history_name: &'static str,
}

pub(crate) struct LlmDecoderStackOutputs<'a> {
    pub state: GgmlCpuTensor<'a>,
    pub kv_inputs: Vec<(GgmlCpuTensor<'a>, GgmlCpuTensor<'a>)>,
    pub kv_outputs: Vec<(GgmlCpuTensor<'a>, GgmlCpuTensor<'a>)>,
}

#[derive(Clone, Copy)]
pub(crate) struct LlmResidentKvLayer {
    pub key: GgmlStaticTensor,
    pub value: GgmlStaticTensor,
}

pub(crate) struct LlmResidentKvArena {
    pub arena: GgmlStaticTensorArena,
    pub layers: Vec<LlmResidentKvLayer>,
}

impl LlmResidentKvArena {
    pub(crate) fn graph_tensors<'a>(&self) -> Vec<(GgmlCpuTensor<'a>, GgmlCpuTensor<'a>)> {
        self.layers
            .iter()
            .map(|layer| {
                (
                    self.arena.graph_tensor(layer.key),
                    self.arena.graph_tensor(layer.value),
                )
            })
            .collect()
    }
}

/// Build-once/re-run seq2seq decode graph state for a single-token incremental
/// decoder step. Field order is load-bearing: `session` must drop before
/// `kv_arena`, because graph tensors can point into the resident KV arena's
/// backend buffer.
pub(crate) struct Seq2SeqReusableDecodeGraph {
    session: GgmlPersistentGraphSession,
    #[allow(dead_code)]
    kv_arena: Option<GgmlStaticTensorArena>,
    pub max_positions: usize,
    pub n_seq: usize,
    pub token_id: GgmlCpuTensor<'static>,
    pub row_index: GgmlCpuTensor<'static>,
    pub position: GgmlCpuTensor<'static>,
    pub attention_mask: GgmlCpuTensor<'static>,
    pub logits: GgmlCpuTensor<'static>,
}

impl Seq2SeqReusableDecodeGraph {
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn new(
        session: GgmlPersistentGraphSession,
        kv_arena: GgmlStaticTensorArena,
        max_positions: usize,
        n_seq: usize,
        token_id: GgmlCpuTensor<'static>,
        row_index: GgmlCpuTensor<'static>,
        position: GgmlCpuTensor<'static>,
        attention_mask: GgmlCpuTensor<'static>,
        logits: GgmlCpuTensor<'static>,
    ) -> Self {
        Self {
            session,
            kv_arena: Some(kv_arena),
            max_positions,
            n_seq,
            token_id,
            row_index,
            position,
            attention_mask,
            logits,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_borrowed_kv_arena(
        session: GgmlPersistentGraphSession,
        max_positions: usize,
        n_seq: usize,
        token_id: GgmlCpuTensor<'static>,
        row_index: GgmlCpuTensor<'static>,
        position: GgmlCpuTensor<'static>,
        attention_mask: GgmlCpuTensor<'static>,
        logits: GgmlCpuTensor<'static>,
    ) -> Self {
        Self {
            session,
            kv_arena: None,
            max_positions,
            n_seq,
            token_id,
            row_index,
            position,
            attention_mask,
            logits,
        }
    }

    pub(crate) fn builder(&mut self) -> &mut GgmlCpuGraphBuilder<'static> {
        self.session.builder()
    }
}

/// Build-once/re-run LLM decode graph state. Field order is load-bearing:
/// `session` must drop before `kv_arena`, because graph tensors can point into
/// the resident KV arena's backend buffer.
pub(crate) struct LlmReusableDecodeGraph {
    session: GgmlPersistentGraphSession,
    #[allow(dead_code)]
    kv_arena: LlmResidentKvArena,
    pub max_positions: usize,
    pub n_seq: usize,
    pub hidden_tensor: GgmlCpuTensor<'static>,
    pub row_indices: GgmlCpuTensor<'static>,
    pub positions: GgmlCpuTensor<'static>,
    pub attention_mask: GgmlCpuTensor<'static>,
    pub state: GgmlCpuTensor<'static>,
    pub top1: Option<GgmlCpuTensor<'static>>,
}

impl LlmReusableDecodeGraph {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        session: GgmlPersistentGraphSession,
        kv_arena: LlmResidentKvArena,
        max_positions: usize,
        n_seq: usize,
        hidden_tensor: GgmlCpuTensor<'static>,
        row_indices: GgmlCpuTensor<'static>,
        positions: GgmlCpuTensor<'static>,
        attention_mask: GgmlCpuTensor<'static>,
        state: GgmlCpuTensor<'static>,
        top1: Option<GgmlCpuTensor<'static>>,
    ) -> Self {
        Self {
            session,
            kv_arena,
            max_positions,
            n_seq,
            hidden_tensor,
            row_indices,
            positions,
            attention_mask,
            state,
            top1,
        }
    }

    pub(crate) fn builder(&mut self) -> &mut GgmlCpuGraphBuilder<'static> {
        self.session.builder()
    }

    pub(crate) fn resident_kv_arena_mut(&mut self) -> &mut LlmResidentKvArena {
        &mut self.kv_arena
    }
}

const RMS_NORM_STEPS: RmsNormSteps = RmsNormSteps {
    norm: "rms_norm",
    scale: "mul",
};

/// LoRA-aware matmul: `W@x + b_scaled@(a@x)` when `lora` is `Some`, else plain
/// `W@x`. Used for the output and FFN projections in `llm_layer`. For the QKV
/// projections the split-path below handles LoRA inline.
fn llm_lora_matmul<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    lora: Option<LlmLoraSlot<'a>>,
    input: GgmlCpuTensor<'a>,
    step: &'static str,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let base = graph
        .mul_mat(weight, input)
        .map_err(|source| map_err(step, source))?;
    let Some(lora) = lora else {
        return Ok(base);
    };
    let lora_step = "llm_lora_inner";
    let ax = graph
        .mul_mat(lora.a, input)
        .map_err(|source| map_err(lora_step, source))?;
    let delta = graph
        .mul_mat(lora.b_scaled, ax)
        .map_err(|source| map_err(lora_step, source))?;
    graph
        .add(base, delta)
        .map_err(|source| map_err(lora_step, source))
}

/// Project Q/K/V from the normalized hidden state. Fused path: one `mul_mat` on a
/// concatenated `[q|k|v]` weight, then three `view_2d` slices at f32 byte offsets.
/// Split path: three separate `mul_mat`. Byte strides assume f32 (matching the
/// qwen layout the fused weight is built against).
///
/// If ANY of `q_lora`, `k_lora`, `v_lora` is `Some`, the SPLIT path is used
/// regardless of `qkv_weight` so each projection can be independently adapted.
/// When all three are `None` the existing fused/split logic is unchanged.
#[allow(clippy::too_many_arguments)]
fn build_projected_qkv<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    normed: GgmlCpuTensor<'a>,
    qkv_weight: Option<GgmlCpuTensor<'a>>,
    q_weight: GgmlCpuTensor<'a>,
    k_weight: GgmlCpuTensor<'a>,
    v_weight: GgmlCpuTensor<'a>,
    q_lora: Option<LlmLoraSlot<'a>>,
    k_lora: Option<LlmLoraSlot<'a>>,
    v_lora: Option<LlmLoraSlot<'a>>,
    q_width: usize,
    k_width: usize,
    v_width: usize,
    output_tokens: usize,
    map_err: F,
) -> Result<(GgmlCpuTensor<'a>, GgmlCpuTensor<'a>, GgmlCpuTensor<'a>), E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let element = std::mem::size_of::<f32>();
    // When any QKV projection has a LoRA slot, force the SPLIT path so each
    // projection can be independently adapted. The fused path cannot apply
    // per-projection LoRA because the output is a single concatenated tensor.
    let any_qkv_lora = q_lora.is_some() || k_lora.is_some() || v_lora.is_some();
    if let Some(qkv_weight) = qkv_weight.filter(|_| !any_qkv_lora) {
        let qkv_width = q_width
            .checked_add(k_width)
            .and_then(|width| width.checked_add(v_width))
            .ok_or_else(|| {
                map_err(
                    "llm_qkv_fused_width",
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "llm fused qkv width overflow",
                    },
                )
            })?;
        let column_stride = qkv_width.checked_mul(element).ok_or_else(|| {
            map_err(
                "llm_qkv_fused_stride",
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "llm fused qkv stride overflow",
                },
            )
        })?;
        let qkv = graph
            .mul_mat(qkv_weight, normed)
            .map_err(|source| map_err("llm_qkv_fused", source))?;
        let mut q = graph
            .view_2d(qkv, q_width, output_tokens, column_stride, 0)
            .map_err(|source| map_err("llm_qkv_q_view", source))?;
        let mut k = graph
            .view_2d(
                qkv,
                k_width,
                output_tokens,
                column_stride,
                q_width * element,
            )
            .map_err(|source| map_err("llm_qkv_k_view", source))?;
        let mut v = graph
            .view_2d(
                qkv,
                v_width,
                output_tokens,
                column_stride,
                (q_width + k_width) * element,
            )
            .map_err(|source| map_err("llm_qkv_v_view", source))?;
        if output_tokens > 1 {
            q = graph
                .cont(q)
                .map_err(|source| map_err("llm_qkv_q_cont", source))?;
            k = graph
                .cont(k)
                .map_err(|source| map_err("llm_qkv_k_cont", source))?;
            v = graph
                .cont(v)
                .map_err(|source| map_err("llm_qkv_v_cont", source))?;
        }
        return Ok((q, k, v));
    }
    let q = llm_lora_matmul(graph, q_weight, q_lora, normed, "llm_q_proj", map_err)?;
    let k = llm_lora_matmul(graph, k_weight, k_lora, normed, "llm_k_proj", map_err)?;
    let v = llm_lora_matmul(graph, v_weight, v_lora, normed, "llm_v_proj", map_err)?;
    Ok((q, k, v))
}

/// Expand grouped K/V across the query-head group for GQA. No-op when there is one
/// query per KV head or when the backend handles GQA natively (`use_native_gqa`).
fn expand_attention_kv<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    mut k_full: GgmlCpuTensor<'a>,
    mut v_full: GgmlCpuTensor<'a>,
    head_dim: usize,
    q_heads: usize,
    kv_heads: usize,
    use_native_gqa: bool,
    total_tokens: usize,
    map_err: F,
) -> Result<(GgmlCpuTensor<'a>, GgmlCpuTensor<'a>), E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let q_per_kv_group = q_heads / kv_heads;
    if q_per_kv_group <= 1 || use_native_gqa {
        return Ok((k_full, v_full));
    }
    k_full = graph
        .reshape_4d(k_full, head_dim, total_tokens, 1, kv_heads)
        .map_err(|source| map_err("llm_gqa_k_reshape4d", source))?;
    v_full = graph
        .reshape_4d(v_full, head_dim, total_tokens, 1, kv_heads)
        .map_err(|source| map_err("llm_gqa_v_reshape4d", source))?;
    k_full = graph
        .repeat_4d(k_full, head_dim, total_tokens, q_per_kv_group, kv_heads)
        .map_err(|source| map_err("llm_gqa_k_repeat4d", source))?;
    v_full = graph
        .repeat_4d(v_full, head_dim, total_tokens, q_per_kv_group, kv_heads)
        .map_err(|source| map_err("llm_gqa_v_repeat4d", source))?;
    k_full = graph
        .reshape_3d(k_full, head_dim, total_tokens, q_heads)
        .map_err(|source| map_err("llm_gqa_k_reshape3d", source))?;
    v_full = graph
        .reshape_3d(v_full, head_dim, total_tokens, q_heads)
        .map_err(|source| map_err("llm_gqa_v_reshape3d", source))?;
    k_full = graph
        .cont(k_full)
        .map_err(|source| map_err("llm_gqa_k_cont", source))?;
    v_full = graph
        .cont(v_full)
        .map_err(|source| map_err("llm_gqa_v_cont", source))?;
    Ok((k_full, v_full))
}

/// Assemble one decoder-only LLM transformer block, reproducing qwen's
/// hand-written op sequence bit-identically. `hidden` is `[d_model, 1]` and the
/// KV-cache history tensors are allocated + uploaded by the caller; the block
/// `set_rows`-writes this step's K/V and reads the full history back for
/// attention. Returns the pre-`set_rows` projected K/V so the caller can update
/// its host cache.
#[allow(clippy::too_many_lines)]
pub(crate) fn llm_layer<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    hidden: GgmlCpuTensor<'a>,
    config: LlmLayerConfig,
    weights: LlmLayerWeights<'a>,
    kv: LlmKvCacheHandle<'a>,
    map_err: F,
) -> Result<LlmLayerOutput<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let head_dim = config.head_dim;
    let token_count = config.token_count;
    let n_seq = config.n_seq;
    if token_count == 0 {
        return Err(map_err(
            "llm_token_count",
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "llm decoder token_count must be positive",
            },
        ));
    }
    if n_seq == 0 {
        return Err(map_err(
            "llm_n_seq",
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "llm decoder n_seq must be positive",
            },
        ));
    }
    let output_tokens = token_count.checked_mul(n_seq).ok_or_else(|| {
        map_err(
            "llm_output_tokens",
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "llm decoder token/sequence count overflow",
            },
        )
    })?;

    let normed = apply_rms_norm(
        graph,
        hidden,
        config.rms_norm_epsilon,
        weights.attn_norm_weight,
        RMS_NORM_STEPS,
        map_err,
    )?;
    let (q, k, v) = build_projected_qkv(
        graph,
        normed,
        weights.qkv_weight,
        weights.q_weight,
        weights.k_weight,
        weights.v_weight,
        weights.q_lora,
        weights.k_lora,
        weights.v_lora,
        config.q_width,
        config.k_width,
        config.v_width,
        output_tokens,
        map_err,
    )?;
    let q = graph
        .reshape_3d(q, head_dim, config.q_heads, output_tokens)
        .map_err(|source| map_err("llm_q_reshape3d", source))?;
    let k = graph
        .reshape_3d(k, head_dim, config.kv_heads, output_tokens)
        .map_err(|source| map_err("llm_k_reshape3d", source))?;
    let v = graph
        .reshape_3d(v, head_dim, config.kv_heads, output_tokens)
        .map_err(|source| map_err("llm_v_reshape3d", source))?;
    let q = apply_rms_norm(
        graph,
        q,
        config.rms_norm_epsilon,
        weights.q_norm_weight,
        RMS_NORM_STEPS,
        map_err,
    )?;
    let k = apply_rms_norm(
        graph,
        k,
        config.rms_norm_epsilon,
        weights.k_norm_weight,
        RMS_NORM_STEPS,
        map_err,
    )?;
    let q = graph
        .rope_ext(q, kv.positions, config.rope)
        .map_err(|source| map_err("llm_rope_q", source))?;
    let k = graph
        .rope_ext(k, kv.positions, config.rope)
        .map_err(|source| map_err("llm_rope_k", source))?;
    // Taps: the host KV-cache write consumes the post-q/k-norm + RoPE `k` and the
    // post-reshape `v`, BEFORE the permute/set_rows below. Bind them here.
    let projected_k = k;
    let projected_v = v;
    let (q_flash, k_new, v_new) = if token_count == 1 {
        let seq_batch_permute = if n_seq == 1 {
            (0, 2, 1, 3)
        } else {
            (0, 2, 3, 1)
        };
        let q_flash = graph
            .permute(
                q,
                seq_batch_permute.0,
                seq_batch_permute.1,
                seq_batch_permute.2,
                seq_batch_permute.3,
            )
            .map_err(|source| map_err("llm_q_permute", source))?;
        let k_new = graph
            .permute(
                k,
                seq_batch_permute.0,
                seq_batch_permute.1,
                seq_batch_permute.2,
                seq_batch_permute.3,
            )
            .map_err(|source| map_err("llm_k_permute", source))?;
        let v_new = graph
            .permute(
                v,
                seq_batch_permute.0,
                seq_batch_permute.1,
                seq_batch_permute.2,
                seq_batch_permute.3,
            )
            .map_err(|source| map_err("llm_v_permute", source))?;
        (q_flash, k_new, v_new)
    } else {
        let q = graph
            .cont(q)
            .map_err(|source| map_err("llm_q_query_seq_cont", source))?;
        let k = graph
            .cont(k)
            .map_err(|source| map_err("llm_k_query_seq_cont", source))?;
        let v = graph
            .cont(v)
            .map_err(|source| map_err("llm_v_query_seq_cont", source))?;
        let q = graph
            .reshape_4d(q, head_dim, config.q_heads, token_count, n_seq)
            .map_err(|source| map_err("llm_q_query_seq_reshape4d", source))?;
        let k = graph
            .reshape_4d(k, head_dim, config.kv_heads, token_count, n_seq)
            .map_err(|source| map_err("llm_k_query_seq_reshape4d", source))?;
        let v = graph
            .reshape_4d(v, head_dim, config.kv_heads, token_count, n_seq)
            .map_err(|source| map_err("llm_v_query_seq_reshape4d", source))?;
        let q_flash = graph
            .permute(q, 0, 2, 1, 3)
            .map_err(|source| map_err("llm_q_permute", source))?;
        let k_new = graph
            .permute(k, 0, 2, 1, 3)
            .map_err(|source| map_err("llm_k_permute", source))?;
        let v_new = graph
            .permute(v, 0, 2, 1, 3)
            .map_err(|source| map_err("llm_v_permute", source))?;
        (q_flash, k_new, v_new)
    };
    let q_flash = graph
        .cont(q_flash)
        .map_err(|source| map_err("llm_q_cont", source))?;
    let (k_new, v_new) = if token_count == 1 {
        (k_new, v_new)
    } else {
        let k_new = graph
            .cont(k_new)
            .map_err(|source| map_err("llm_k_set_rows_cont", source))?;
        let v_new = graph
            .cont(v_new)
            .map_err(|source| map_err("llm_v_set_rows_cont", source))?;
        (k_new, v_new)
    };
    let k_full = graph
        .set_rows(kv.key_history, k_new, kv.row_indices)
        .map_err(|source| map_err("llm_k_set_rows", source))?;
    let v_full = graph
        .set_rows(kv.value_history, v_new, kv.row_indices)
        .map_err(|source| map_err("llm_v_set_rows", source))?;
    let (k_full, v_full) = expand_attention_kv(
        graph,
        k_full,
        v_full,
        head_dim,
        config.q_heads,
        config.kv_heads,
        config.use_native_gqa || n_seq > 1,
        config.total_tokens,
        map_err,
    )?;
    let scale = (head_dim as f32).sqrt().recip();
    let attended = if config.use_flash_attention {
        graph
            .flash_attn_ext(q_flash, k_full, v_full, kv.attention_mask, scale, 0.0, 0.0)
            .map_err(|source| map_err("llm_flash_attn", source))?
    } else {
        llm_naive_masked_attention(
            graph,
            q_flash,
            k_full,
            v_full,
            kv.attention_mask,
            scale,
            map_err,
        )?
    };
    let attended = graph
        .reshape_2d(attended, config.q_width, output_tokens)
        .map_err(|source| map_err("llm_attn_reshape2d", source))?;
    let attn_hidden = llm_lora_matmul(
        graph,
        weights.output_weight,
        weights.output_lora,
        attended,
        "llm_out_proj",
        map_err,
    )?;
    let post_attn = graph
        .add(hidden, attn_hidden)
        .map_err(|source| map_err("llm_attn_residual", source))?;
    let ffn_normed = apply_rms_norm(
        graph,
        post_attn,
        config.rms_norm_epsilon,
        weights.ffn_norm_weight,
        RMS_NORM_STEPS,
        map_err,
    )?;
    let output = apply_gated_feed_forward_residual(
        graph,
        ffn_normed,
        post_attn,
        FeedForwardActivation::Silu,
        GatedFeedForwardResidualSteps {
            gate_activation: "silu",
            gate_mul: "mul",
            residual: "add",
        },
        |graph, x| {
            llm_lora_matmul(
                graph,
                weights.ffn_gate_weight,
                weights.ffn_gate_lora,
                x,
                "llm_ffn_gate",
                map_err,
            )
        },
        |graph, x| {
            llm_lora_matmul(
                graph,
                weights.ffn_up_weight,
                weights.ffn_up_lora,
                x,
                "llm_ffn_up",
                map_err,
            )
        },
        |graph, x| {
            llm_lora_matmul(
                graph,
                weights.ffn_down_weight,
                weights.ffn_down_lora,
                x,
                "llm_ffn_down",
                map_err,
            )
        },
        map_err,
    )?;

    Ok(LlmLayerOutput {
        output,
        projected_k,
        projected_v,
    })
}

/// Unfused self-attention (`Kᵀ·Q → soft_max_ext(mask, scale) → Vᵀ·P`) with the
/// same input/output geometry as the `flash_attn_ext` call it replaces:
/// q `[head_dim, n_q, q_heads, n_seq]`, K/V history `[head_dim, n_kv, kv_heads,
/// n_seq]` (GQA via `mul_mat` batch broadcast, so pre-expanded K/V also work),
/// additive f16/f32 mask `[n_kv, n_q, 1, n_seq]`, output `[head_dim, q_heads,
/// n_q, n_seq]` contiguous. The mask is mandatory: every caller that disables
/// flash runs against a fixed-span KV whose unwritten rows must stay at -inf.
fn llm_naive_masked_attention<'a, E, F>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    q: GgmlCpuTensor<'a>,
    k_full: GgmlCpuTensor<'a>,
    v_full: GgmlCpuTensor<'a>,
    attention_mask: Option<GgmlCpuTensor<'a>>,
    scale: f32,
    map_err: F,
) -> Result<GgmlCpuTensor<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
{
    let Some(mask) = attention_mask else {
        return Err(map_err(
            "llm_naive_attn_mask",
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "non-flash LLM attention requires an explicit attention mask",
            },
        ));
    };
    let scores = graph
        .mul_mat(k_full, q)
        .map_err(|source| map_err("llm_naive_attn_scores", source))?;
    let probabilities = graph
        .soft_max_ext(scores, Some(mask), scale, 0.0)
        .map_err(|source| map_err("llm_naive_attn_softmax", source))?;
    let v_t = graph
        .transpose(v_full)
        .map_err(|source| map_err("llm_naive_attn_v_transpose", source))?;
    let v_t = graph
        .cont(v_t)
        .map_err(|source| map_err("llm_naive_attn_v_cont", source))?;
    let attended = graph
        .mul_mat(v_t, probabilities)
        .map_err(|source| map_err("llm_naive_attn_context", source))?;
    let attended = graph
        .permute(attended, 0, 2, 1, 3)
        .map_err(|source| map_err("llm_naive_attn_merge_permute", source))?;
    graph
        .cont(attended)
        .map_err(|source| map_err("llm_naive_attn_merge_cont", source))
}

/// Walk a decoder-only LLM layer stack, emitting one `llm_layer` block per
/// layer and chaining `state` through them. The caller owns model-specific
/// weight handles and maps each layer into `LlmLayerWeights` with `layer_weights`;
/// this keeps the shared stack composer free of family-specific storage details.
pub(crate) fn compose_llm_decoder_layer_stack<'a, E, F, G>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    layer_count: usize,
    config: LlmDecoderStackConfig,
    inputs: LlmDecoderStackInputs<'a>,
    resident_kv: Option<&[(GgmlCpuTensor<'a>, GgmlCpuTensor<'a>)]>,
    mut layer_weights: G,
    map_err: F,
) -> Result<LlmDecoderStackOutputs<'a>, E>
where
    F: Fn(&'static str, GgmlCpuGraphError) -> E + Copy,
    G: FnMut(usize) -> LlmLayerWeights<'a>,
{
    if let Some(resident) = resident_kv
        && resident.len() != layer_count
    {
        return Err(map_err(
            "llm_decoder_stack_resident_kv",
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "resident KV layer count mismatch",
            },
        ));
    }

    let mut state = inputs.state;
    let mut kv_inputs = Vec::with_capacity(layer_count);
    let mut kv_outputs = Vec::with_capacity(layer_count);
    for layer_index in 0..layer_count {
        let (key_history, value_history) = match resident_kv {
            Some(resident) => resident[layer_index],
            None => {
                let key_history = graph
                    .new_tensor_4d_f32(
                        config.head_dim,
                        inputs.kv_span,
                        config.kv_heads,
                        config.n_seq,
                        inputs.key_history_name,
                    )
                    .map_err(|source| map_err("llm_decoder_stack_key_history", source))?;
                let value_history = graph
                    .new_tensor_4d_f32(
                        config.head_dim,
                        inputs.kv_span,
                        config.kv_heads,
                        config.n_seq,
                        inputs.value_history_name,
                    )
                    .map_err(|source| map_err("llm_decoder_stack_value_history", source))?;
                graph
                    .set_input(key_history)
                    .map_err(|source| map_err("llm_decoder_stack_key_input", source))?;
                graph
                    .set_input(value_history)
                    .map_err(|source| map_err("llm_decoder_stack_value_input", source))?;
                (key_history, value_history)
            }
        };
        let layer_out = llm_layer(
            graph,
            state,
            LlmLayerConfig {
                d_model: config.d_model,
                head_dim: config.head_dim,
                q_heads: config.q_heads,
                kv_heads: config.kv_heads,
                q_width: config.q_width,
                k_width: config.k_width,
                v_width: config.v_width,
                token_count: config.token_count,
                n_seq: config.n_seq,
                total_tokens: inputs.kv_span,
                rms_norm_epsilon: config.rms_norm_epsilon,
                rope: config.rope,
                use_native_gqa: config.use_native_gqa,
                use_flash_attention: config.use_flash_attention,
            },
            layer_weights(layer_index),
            LlmKvCacheHandle {
                key_history,
                value_history,
                row_indices: inputs.row_indices,
                positions: inputs.positions,
                attention_mask: inputs.attention_mask,
            },
            map_err,
        )?;
        state = layer_out.output;
        graph
            .set_output(layer_out.projected_k)
            .map_err(|source| map_err("llm_decoder_stack_projected_k", source))?;
        graph
            .set_output(layer_out.projected_v)
            .map_err(|source| map_err("llm_decoder_stack_projected_v", source))?;
        kv_inputs.push((key_history, value_history));
        kv_outputs.push((layer_out.projected_k, layer_out.projected_v));
    }
    Ok(LlmDecoderStackOutputs {
        state,
        kv_inputs,
        kv_outputs,
    })
}

/// Convert an f32 row to f16 bit patterns (for seeding f16 KV tensors from an
/// f32 host cache).
pub(crate) fn f32_slice_to_f16_bits(values: &[f32]) -> Vec<u16> {
    values.iter().copied().map(f32_to_f16_bits).collect()
}

/// Allocate a zero-filled resident KV cache arena. The cache element type is
/// always f16: it halves the arena footprint and the attention K/V read
/// bandwidth; `set_rows` casts the f32 projection rows on write and both
/// `flash_attn_ext` and `mul_mat` consume f16 K/V natively, so the decode
/// graph shape is unchanged.
pub(crate) fn allocate_zeroed_llm_resident_kv_arena(
    runner: &GgmlCpuGraphRunner,
    context_bytes: usize,
    layer_count: usize,
    head_dim: usize,
    max_positions: usize,
    kv_heads: usize,
    n_seq: usize,
    tensor_name_prefix: &str,
) -> Result<LlmResidentKvArena, GgmlCpuGraphError> {
    if n_seq == 0 {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "resident KV n_seq must be positive",
        });
    }
    let mut arena = runner.start_static_tensor_arena(context_bytes)?;
    let mut layers = Vec::with_capacity(layer_count);
    for layer_idx in 0..layer_count {
        let key_name = Box::leak(format!("{tensor_name_prefix}_key_{layer_idx}").into_boxed_str())
            as &'static str;
        let value_name =
            Box::leak(format!("{tensor_name_prefix}_value_{layer_idx}").into_boxed_str())
                as &'static str;
        let key = arena.new_tensor_4d_f16(head_dim, max_positions, kv_heads, n_seq, key_name)?;
        let value =
            arena.new_tensor_4d_f16(head_dim, max_positions, kv_heads, n_seq, value_name)?;
        layers.push(LlmResidentKvLayer { key, value });
    }
    arena.allocate_backend_buffer()?;
    let kv_elems = head_dim
        .checked_mul(max_positions)
        .and_then(|n| n.checked_mul(kv_heads))
        .and_then(|n| n.checked_mul(n_seq))
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "resident KV element count overflow",
        })?;
    // Zero-fill so masked (unwritten) positions never feed NaN/inf into
    // flash-attn; the f16 bit pattern for zero is zero.
    let kv_zero = vec![0_u16; kv_elems];
    for layer in &layers {
        arena.set_f16_bits_slice(layer.key, &kv_zero, "resident_kv_key")?;
        arena.set_f16_bits_slice(layer.value, &kv_zero, "resident_kv_value")?;
    }
    Ok(LlmResidentKvArena { arena, layers })
}

pub(crate) fn build_fixed_kv_attention_mask_bits(
    max_positions: usize,
    total_tokens: usize,
) -> Result<Vec<u16>, GgmlCpuGraphError> {
    build_fixed_kv_attention_mask_bits_for_sequences(max_positions, &[total_tokens])
}

pub(crate) fn build_fixed_kv_attention_mask_bits_for_sequences(
    max_positions: usize,
    total_tokens_by_sequence: &[usize],
) -> Result<Vec<u16>, GgmlCpuGraphError> {
    if total_tokens_by_sequence.is_empty() {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "fixed KV attention mask sequence count must be positive",
        });
    }
    if total_tokens_by_sequence
        .iter()
        .any(|&total_tokens| total_tokens > max_positions)
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "fixed KV attention mask token count exceeds max positions",
        });
    }
    const F16_ZERO: u16 = 0x0000;
    const F16_NEG_INF: u16 = 0xFC00;
    let mask_len = max_positions
        .checked_mul(total_tokens_by_sequence.len())
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "fixed KV attention mask element count overflow",
        })?;
    let mut mask_bits = vec![F16_ZERO; mask_len];
    for (sequence_index, &total_tokens) in total_tokens_by_sequence.iter().enumerate() {
        let plane = sequence_index * max_positions;
        for slot in mask_bits[plane + total_tokens..plane + max_positions].iter_mut() {
            *slot = F16_NEG_INF;
        }
    }
    Ok(mask_bits)
}

pub(crate) fn build_fixed_kv_attention_mask_bits_for_query_rows(
    max_positions: usize,
    token_count: usize,
    n_seq: usize,
    row_indices: &[usize],
) -> Result<Vec<u16>, GgmlCpuGraphError> {
    if token_count == 0 {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "fixed KV attention mask query count must be positive",
        });
    }
    if n_seq == 0 {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "fixed KV attention mask sequence count must be positive",
        });
    }
    let expected_rows =
        token_count
            .checked_mul(n_seq)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "fixed KV attention mask element count overflow",
            })?;
    if row_indices.len() != expected_rows {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "fixed KV attention mask row-index count mismatch",
        });
    }
    if row_indices.iter().any(|&row| row >= max_positions) {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "fixed KV attention mask token count exceeds max positions",
        });
    }
    const F16_ZERO: u16 = 0x0000;
    const F16_NEG_INF: u16 = 0xFC00;
    let mask_len = max_positions
        .checked_mul(token_count)
        .and_then(|n| n.checked_mul(n_seq))
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "fixed KV attention mask element count overflow",
        })?;
    let mut mask_bits = vec![F16_ZERO; mask_len];
    for sequence_index in 0..n_seq {
        for token_index in 0..token_count {
            let row_index = sequence_index
                .checked_mul(token_count)
                .and_then(|base| base.checked_add(token_index))
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "fixed KV attention mask element count overflow",
                })?;
            let total_tokens = row_indices[row_index] + 1;
            let plane = sequence_index
                .checked_mul(token_count)
                .and_then(|base| base.checked_add(token_index))
                .and_then(|plane| plane.checked_mul(max_positions))
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "fixed KV attention mask element count overflow",
                })?;
            for slot in mask_bits[plane + total_tokens..plane + max_positions].iter_mut() {
                *slot = F16_NEG_INF;
            }
        }
    }
    Ok(mask_bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{GgmlCpuGraphConfig, GgmlCpuGraphRunner};

    #[test]
    fn reusable_decode_graph_support_requires_gpu_class_without_scheduler() {
        assert!(!reusable_decode_graph_supported(
            GgmlCpuGraphBackend::Cpu,
            false
        ));
        assert!(!reusable_decode_graph_supported(
            GgmlCpuGraphBackend::Cpu,
            true
        ));
        assert!(reusable_decode_graph_supported(
            GgmlCpuGraphBackend::Metal,
            false
        ));
        assert!(!reusable_decode_graph_supported(
            GgmlCpuGraphBackend::Metal,
            true
        ));
        assert!(reusable_decode_graph_supported(
            GgmlCpuGraphBackend::Gpu,
            false
        ));
        assert!(!reusable_decode_graph_supported(
            GgmlCpuGraphBackend::Gpu,
            true
        ));
    }

    #[test]
    fn seq2seq_layer_stack_runs_layers_in_order() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let state = graph
            .new_tensor_2d_f32(1, 1, "state")
            .expect("state tensor");
        let layers = [10, 20];
        let cross_layers = [1, 2];
        let self_kv_layers = [3, 4];
        let mut seen = Vec::new();

        let _ = seq2seq_layer_stack(
            &mut graph,
            state,
            &layers,
            &cross_layers,
            &self_kv_layers,
            |_| "length mismatch",
            |_, state, layer_idx, layer, cross, self_kv| {
                seen.push((layer_idx, *layer, *cross, *self_kv));
                Ok::<_, &'static str>(state)
            },
        )
        .expect("matching layer slices should compose");

        assert_eq!(seen, vec![(0, 10, 1, 3), (1, 20, 2, 4)]);
    }

    #[test]
    fn seq2seq_layer_stack_fails_closed_on_length_mismatch() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let state = graph
            .new_tensor_2d_f32(1, 1, "state")
            .expect("state tensor");
        let layers = [10, 20];
        let cross_layers = [1];
        let self_kv_layers = [3, 4];

        let error = seq2seq_layer_stack(
            &mut graph,
            state,
            &layers,
            &cross_layers,
            &self_kv_layers,
            |length| length,
            |_, state, _, _, _, _| Ok::<_, Seq2SeqLayerStackLength>(state),
        )
        .expect_err("mismatched layer slices must fail closed");

        assert_eq!(
            error,
            Seq2SeqLayerStackLength {
                layers: 2,
                cross_layers: 1,
                self_kv_layers: 2,
            }
        );
    }

    #[test]
    fn seq2seq_indexed_layer_stack_marks_last_layer() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let state = graph
            .new_tensor_2d_f32(1, 1, "state")
            .expect("state tensor");
        let layers = [7, 8, 9];
        let mut seen = Vec::new();

        let _ =
            seq2seq_indexed_layer_stack(&mut graph, state, &layers, |_, state, position, layer| {
                seen.push((position.layer_index, position.is_last, *layer));
                Ok::<_, &'static str>(state)
            })
            .expect("indexed layer stack should compose");

        assert_eq!(seen, vec![(0, false, 7), (1, false, 8), (2, true, 9)]);
    }

    fn compute_single_layer_seq2seq_stack(
        n_seq: usize,
        token_count: usize,
        state_values: &[f32],
        row_indices: &[i32],
        cross_k_values: &[f32],
        cross_v_values: &[f32],
    ) -> Vec<f32> {
        const HIDDEN: usize = 4;
        const HEAD_DIM: usize = 2;
        const HEADS: usize = 2;
        const MAX_POSITIONS: usize = 3;
        const CROSS_FRAMES: usize = 2;
        const IDENTITY_4D: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        const ZERO_4D: [f32; 16] = [0.0; 16];
        const NORM_WEIGHT: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
        const ZERO_1D: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

        assert_eq!(state_values.len(), HIDDEN * token_count * n_seq);
        assert_eq!(row_indices.len(), token_count * n_seq);
        assert_eq!(cross_k_values.len(), HIDDEN * CROSS_FRAMES * n_seq);
        assert_eq!(cross_v_values.len(), HIDDEN * CROSS_FRAMES * n_seq);

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let state = graph
            .new_tensor_2d_f32(HIDDEN, token_count * n_seq, "seq2seq_state")
            .expect("state tensor");
        let row_indices_tensor = graph
            .new_tensor_4d_i32(token_count, 1, n_seq, 1, "seq2seq_row_indices")
            .expect("row indices");
        let attention_mask = graph
            .new_tensor_4d_f16(MAX_POSITIONS, token_count, 1, n_seq, "seq2seq_self_mask")
            .expect("attention mask");
        let self_k = graph
            .new_tensor_4d_f16(HEAD_DIM, MAX_POSITIONS, HEADS, n_seq, "seq2seq_self_k")
            .expect("self k");
        let self_v = graph
            .new_tensor_4d_f16(HEAD_DIM, MAX_POSITIONS, HEADS, n_seq, "seq2seq_self_v")
            .expect("self v");
        let cross_k = if n_seq == 1 {
            graph
                .new_tensor_2d_f32(HIDDEN, CROSS_FRAMES, "seq2seq_cross_k")
                .expect("cross k")
        } else {
            graph
                .new_tensor_3d_f32(HIDDEN, CROSS_FRAMES, n_seq, "seq2seq_cross_k")
                .expect("cross k")
        };
        let cross_v = if n_seq == 1 {
            graph
                .new_tensor_2d_f32(HIDDEN, CROSS_FRAMES, "seq2seq_cross_v")
                .expect("cross v")
        } else {
            graph
                .new_tensor_3d_f32(HIDDEN, CROSS_FRAMES, n_seq, "seq2seq_cross_v")
                .expect("cross v")
        };

        let norm_weight = graph
            .new_tensor_1d_f32(HIDDEN, "norm_weight")
            .expect("norm");
        let norm_bias = graph
            .new_tensor_1d_f32(HIDDEN, "norm_bias")
            .expect("norm bias");
        let self_q = graph
            .new_tensor_2d_f32(HIDDEN, HIDDEN, "self_q")
            .expect("self q");
        let self_k_weight = graph
            .new_tensor_2d_f32(HIDDEN, HIDDEN, "self_k_weight")
            .expect("self k weight");
        let self_v_weight = graph
            .new_tensor_2d_f32(HIDDEN, HIDDEN, "self_v_weight")
            .expect("self v weight");
        let self_o = graph
            .new_tensor_2d_f32(HIDDEN, HIDDEN, "self_o")
            .expect("self o");
        let cross_q = graph
            .new_tensor_2d_f32(HIDDEN, HIDDEN, "cross_q")
            .expect("cross q");
        let cross_o = graph
            .new_tensor_2d_f32(HIDDEN, HIDDEN, "cross_o")
            .expect("cross o");
        let ffn_up = graph
            .new_tensor_2d_f32(HIDDEN, HIDDEN, "ffn_up")
            .expect("ffn up");
        let ffn_down = graph
            .new_tensor_2d_f32(HIDDEN, HIDDEN, "ffn_down")
            .expect("ffn down");

        for tensor in [
            state,
            row_indices_tensor,
            attention_mask,
            self_k,
            self_v,
            cross_k,
            cross_v,
            norm_weight,
            norm_bias,
            self_q,
            self_k_weight,
            self_v_weight,
            self_o,
            cross_q,
            cross_o,
            ffn_up,
            ffn_down,
        ] {
            graph
                .set_input(tensor)
                .expect("test input should be settable");
        }

        let block = seq2seq_layer(
            &mut graph,
            state,
            Seq2SeqLayerConfig {
                hidden: HIDDEN,
                attention_heads: HEADS,
                head_dim: HEAD_DIM,
                token_count,
                n_seq,
                total_token_count: MAX_POSITIONS,
                position_offset: 0,
                layer_norm_epsilon: 1.0e-6,
                ffn_activation: FeedForwardActivation::Relu,
                self_kv_max_positions: MAX_POSITIONS,
                cross_frame_count: CROSS_FRAMES,
                cross_hidden_size: HIDDEN,
            },
            Seq2SeqLayerWeights {
                self_attn_norm_weight: norm_weight,
                self_attn_norm_bias: norm_bias,
                self_attn_q_weight: self_q,
                self_attn_q_bias: norm_bias,
                self_attn_k_weight: self_k_weight,
                self_attn_k_bias: norm_bias,
                self_attn_v_weight: self_v_weight,
                self_attn_v_bias: norm_bias,
                self_attn_o_weight: self_o,
                self_attn_o_bias: norm_bias,
                cross_attn_norm_weight: norm_weight,
                cross_attn_norm_bias: norm_bias,
                cross_attn_q_weight: cross_q,
                cross_attn_q_bias: norm_bias,
                cross_attn_o_weight: cross_o,
                cross_attn_o_bias: norm_bias,
                ffn_norm_weight: norm_weight,
                ffn_norm_bias: norm_bias,
                ffn_up_weight: ffn_up,
                ffn_up_bias: norm_bias,
                ffn_down_weight: ffn_down,
                ffn_down_bias: norm_bias,
            },
            SelfKvHandle {
                key: self_k,
                value: self_v,
                row_indices: Some(row_indices_tensor),
                attention_mask: Some(attention_mask),
            },
            CrossKvHandle {
                key: cross_k,
                value: cross_v,
            },
            |_step, source| source,
        )
        .expect("seq2seq layer should build");
        graph.set_output(block.output).expect("seq2seq output");

        graph
            .set_f32_slice(state, state_values, "seq2seq_state")
            .expect("state upload");
        graph
            .set_i32_slice(row_indices_tensor, row_indices, "seq2seq_row_indices")
            .expect("row upload");
        let row_indices_usize = row_indices
            .iter()
            .map(|&row| usize::try_from(row).expect("row index must be non-negative"))
            .collect::<Vec<_>>();
        let mask_bits = build_fixed_kv_attention_mask_bits_for_query_rows(
            MAX_POSITIONS,
            token_count,
            n_seq,
            &row_indices_usize,
        )
        .expect("mask should build");
        graph
            .set_f16_bits_slice(attention_mask, &mask_bits, "seq2seq_self_mask")
            .expect("mask upload");
        let zero_kv = vec![f32_to_f16_bits(0.0); HEAD_DIM * MAX_POSITIONS * HEADS * n_seq];
        graph
            .set_f16_bits_slice(self_k, &zero_kv, "seq2seq_self_k")
            .expect("self k upload");
        graph
            .set_f16_bits_slice(self_v, &zero_kv, "seq2seq_self_v")
            .expect("self v upload");
        graph
            .set_f32_slice(cross_k, cross_k_values, "seq2seq_cross_k")
            .expect("cross k upload");
        graph
            .set_f32_slice(cross_v, cross_v_values, "seq2seq_cross_v")
            .expect("cross v upload");
        graph
            .set_f32_slice(norm_weight, &NORM_WEIGHT, "norm_weight")
            .expect("norm weight upload");
        graph
            .set_f32_slice(norm_bias, &ZERO_1D, "norm_bias")
            .expect("norm bias upload");
        for (tensor, values, name) in [
            (self_q, &IDENTITY_4D[..], "self_q"),
            (self_k_weight, &IDENTITY_4D[..], "self_k_weight"),
            (self_v_weight, &IDENTITY_4D[..], "self_v_weight"),
            (self_o, &IDENTITY_4D[..], "self_o"),
            (cross_q, &IDENTITY_4D[..], "cross_q"),
            (cross_o, &IDENTITY_4D[..], "cross_o"),
            (ffn_up, &ZERO_4D[..], "ffn_up"),
            (ffn_down, &ZERO_4D[..], "ffn_down"),
        ] {
            graph
                .set_f32_slice(tensor, values, name)
                .expect("weight upload");
        }

        graph
            .compute_output_f32(block.output, HIDDEN * token_count * n_seq)
            .expect("seq2seq layer should compute")
    }

    #[test]
    fn seq2seq_layer_batched_sequence_output_matches_serial_runs() {
        let seq0_state = [0.25, -0.75, 0.5, 1.0];
        let seq1_state = [1.25, 0.5, -1.0, 0.75];
        let seq0_cross_k = [0.5, 0.25, -0.25, 0.75, 0.3, -0.4, 0.6, -0.8];
        let seq1_cross_k = [-0.5, 1.0, 0.75, -0.25, 0.9, 0.2, -0.7, 0.4];
        let seq0_cross_v = [0.1, 0.8, -0.4, 0.2, 0.55, -0.15, 0.35, -0.65];
        let seq1_cross_v = [0.9, -0.3, 0.6, -0.7, -0.2, 0.45, -0.55, 0.85];

        let batched = compute_single_layer_seq2seq_stack(
            2,
            1,
            &[
                seq0_state[0],
                seq0_state[1],
                seq0_state[2],
                seq0_state[3],
                seq1_state[0],
                seq1_state[1],
                seq1_state[2],
                seq1_state[3],
            ],
            &[0, 2],
            &[
                seq0_cross_k[0],
                seq0_cross_k[1],
                seq0_cross_k[2],
                seq0_cross_k[3],
                seq0_cross_k[4],
                seq0_cross_k[5],
                seq0_cross_k[6],
                seq0_cross_k[7],
                seq1_cross_k[0],
                seq1_cross_k[1],
                seq1_cross_k[2],
                seq1_cross_k[3],
                seq1_cross_k[4],
                seq1_cross_k[5],
                seq1_cross_k[6],
                seq1_cross_k[7],
            ],
            &[
                seq0_cross_v[0],
                seq0_cross_v[1],
                seq0_cross_v[2],
                seq0_cross_v[3],
                seq0_cross_v[4],
                seq0_cross_v[5],
                seq0_cross_v[6],
                seq0_cross_v[7],
                seq1_cross_v[0],
                seq1_cross_v[1],
                seq1_cross_v[2],
                seq1_cross_v[3],
                seq1_cross_v[4],
                seq1_cross_v[5],
                seq1_cross_v[6],
                seq1_cross_v[7],
            ],
        );
        let serial0 = compute_single_layer_seq2seq_stack(
            1,
            1,
            &seq0_state,
            &[0],
            &seq0_cross_k,
            &seq0_cross_v,
        );
        let serial1 = compute_single_layer_seq2seq_stack(
            1,
            1,
            &seq1_state,
            &[2],
            &seq1_cross_k,
            &seq1_cross_v,
        );

        assert_f32_slices_close(&batched[0..4], &serial0, 1.0e-4);
        assert_f32_slices_close(&batched[4..8], &serial1, 1.0e-4);
    }

    fn synthetic_seq2seq_state(sequence_index: usize) -> [f32; 4] {
        [
            0.25 + sequence_index as f32 * 0.125,
            -0.75 + sequence_index as f32 * 0.0625,
            0.5 - sequence_index as f32 * 0.03125,
            1.0 + sequence_index as f32 * 0.09375,
        ]
    }

    fn synthetic_seq2seq_cross_plane(sequence_index: usize, phase: f32) -> [f32; 8] {
        let mut values = [0.0; 8];
        for (index, value) in values.iter_mut().enumerate() {
            let mixed = sequence_index * 17 + index * 11;
            *value = ((mixed as f32 * 0.03125) + phase).sin();
        }
        values
    }

    #[test]
    fn seq2seq_layer_batched_ragged_n4_n8_outputs_match_serial_runs() {
        for n_seq in [4, 8] {
            let mut state_values = Vec::new();
            let mut row_indices = Vec::new();
            let mut cross_k_values = Vec::new();
            let mut cross_v_values = Vec::new();
            let mut states = Vec::new();
            let mut cross_ks = Vec::new();
            let mut cross_vs = Vec::new();
            for sequence_index in 0..n_seq {
                let state = synthetic_seq2seq_state(sequence_index);
                let cross_k = synthetic_seq2seq_cross_plane(sequence_index, 0.15);
                let cross_v = synthetic_seq2seq_cross_plane(sequence_index, -0.35);
                state_values.extend_from_slice(&state);
                row_indices.push(((sequence_index * 2) % 3) as i32);
                cross_k_values.extend_from_slice(&cross_k);
                cross_v_values.extend_from_slice(&cross_v);
                states.push(state);
                cross_ks.push(cross_k);
                cross_vs.push(cross_v);
            }

            let batched = compute_single_layer_seq2seq_stack(
                n_seq,
                1,
                &state_values,
                &row_indices,
                &cross_k_values,
                &cross_v_values,
            );
            for sequence_index in 0..n_seq {
                let serial = compute_single_layer_seq2seq_stack(
                    1,
                    1,
                    &states[sequence_index],
                    &[row_indices[sequence_index]],
                    &cross_ks[sequence_index],
                    &cross_vs[sequence_index],
                );
                let start = sequence_index * 4;
                assert_f32_slices_close(&batched[start..start + 4], &serial, 1.0e-4);
            }
        }
    }

    #[test]
    fn llm_decoder_stack_fails_closed_on_resident_kv_layer_count_mismatch() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let state = graph
            .new_tensor_2d_f32(1, 1, "state")
            .expect("state tensor");
        let row_indices = graph
            .new_tensor_1d_i32(1, "row_indices")
            .expect("row indices");
        let positions = graph.new_tensor_1d_i32(1, "positions").expect("positions");
        let key = graph
            .new_tensor_3d_f32(1, 1, 1, "key")
            .expect("resident key");
        let value = graph
            .new_tensor_3d_f32(1, 1, 1, "value")
            .expect("resident value");
        let resident = [(key, value)];
        let result = compose_llm_decoder_layer_stack(
            &mut graph,
            2,
            LlmDecoderStackConfig {
                d_model: 1,
                head_dim: 1,
                q_heads: 1,
                kv_heads: 1,
                q_width: 1,
                k_width: 1,
                v_width: 1,
                token_count: 1,
                n_seq: 1,
                rms_norm_epsilon: 1.0e-6,
                rope: GgmlRopeExtParams::qwen_neox(1, 1, 10_000.0).expect("rope params"),
                use_native_gqa: true,
                use_flash_attention: true,
            },
            LlmDecoderStackInputs {
                state,
                row_indices,
                positions,
                attention_mask: None,
                kv_span: 1,
                key_history_name: "key_history",
                value_history_name: "value_history",
            },
            Some(&resident),
            |_layer_index| panic!("layer weights must not be requested on mismatch"),
            |_step, source| source,
        );
        let error = match result {
            Ok(_) => panic!("resident KV mismatch must fail closed"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "resident KV layer count mismatch"
            }
        ));
    }

    #[test]
    fn llm_decoder_stack_builds_fused_qkv_batched_sequence_graph() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();

        let state = graph
            .new_tensor_2d_f32(2, 2, "state")
            .expect("state tensor");
        let row_indices = graph
            .new_tensor_4d_i32(1, 1, 2, 1, "row_indices")
            .expect("row indices");
        let positions = graph.new_tensor_1d_i32(2, "positions").expect("positions");
        let attention_mask = graph
            .new_tensor_4d_f16(3, 1, 1, 2, "attention_mask")
            .expect("attention mask");

        let norm = graph.new_tensor_1d_f32(2, "norm").expect("norm");
        let qkv = graph.new_tensor_2d_f32(2, 8, "qkv").expect("qkv");
        let q = graph.new_tensor_2d_f32(2, 4, "q").expect("q");
        let k = graph.new_tensor_2d_f32(2, 2, "k").expect("k");
        let v = graph.new_tensor_2d_f32(2, 2, "v").expect("v");
        let output = graph.new_tensor_2d_f32(4, 2, "output").expect("output");
        let ffn = graph.new_tensor_2d_f32(2, 2, "ffn").expect("ffn");

        let stack = compose_llm_decoder_layer_stack(
            &mut graph,
            1,
            LlmDecoderStackConfig {
                d_model: 2,
                head_dim: 2,
                q_heads: 2,
                kv_heads: 1,
                q_width: 4,
                k_width: 2,
                v_width: 2,
                token_count: 1,
                n_seq: 2,
                rms_norm_epsilon: 1.0e-6,
                rope: GgmlRopeExtParams::qwen_neox(2, 3, 10_000.0).expect("rope params"),
                use_native_gqa: false,
                use_flash_attention: true,
            },
            LlmDecoderStackInputs {
                state,
                row_indices,
                positions,
                attention_mask: Some(attention_mask),
                kv_span: 3,
                key_history_name: "key_history",
                value_history_name: "value_history",
            },
            None,
            |_layer_index| LlmLayerWeights {
                attn_norm_weight: norm,
                qkv_weight: Some(qkv),
                q_weight: q,
                k_weight: k,
                v_weight: v,
                q_norm_weight: norm,
                k_norm_weight: norm,
                output_weight: output,
                ffn_norm_weight: norm,
                ffn_gate_weight: ffn,
                ffn_up_weight: ffn,
                ffn_down_weight: ffn,
                q_lora: None,
                k_lora: None,
                v_lora: None,
                output_lora: None,
                ffn_gate_lora: None,
                ffn_up_lora: None,
                ffn_down_lora: None,
            },
            |_step, source| source,
        )
        .expect("n_seq=2 fused-QKV LLM stack should build");

        graph.set_output(stack.state).expect("state output");
    }

    fn assert_f32_slices_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let delta = (actual - expected).abs();
            assert!(
                delta <= tolerance,
                "index {index}: actual={actual}, expected={expected}, delta={delta}, tolerance={tolerance}"
            );
        }
    }

    fn compute_single_layer_llm_stack(
        token_count: usize,
        n_seq: usize,
        state_values: &[f32],
        row_indices: &[i32],
        positions: &[i32],
        total_tokens_by_sequence: &[usize],
        initial_key_history: Option<&[f32]>,
        initial_value_history: Option<&[f32]>,
    ) -> Vec<f32> {
        compute_single_layer_llm_stack_outputs(
            token_count,
            n_seq,
            state_values,
            row_indices,
            positions,
            total_tokens_by_sequence,
            initial_key_history,
            initial_value_history,
        )
        .state
    }

    struct LlmTestStackOutput {
        state: Vec<f32>,
        projected_k: Vec<f32>,
        projected_v: Vec<f32>,
    }

    fn compute_single_layer_llm_stack_outputs(
        token_count: usize,
        n_seq: usize,
        state_values: &[f32],
        row_indices: &[i32],
        positions: &[i32],
        total_tokens_by_sequence: &[usize],
        initial_key_history: Option<&[f32]>,
        initial_value_history: Option<&[f32]>,
    ) -> LlmTestStackOutput {
        const D_MODEL: usize = 2;
        const HEAD_DIM: usize = 2;
        const HEADS: usize = 1;
        const MAX_POSITIONS: usize = 3;
        const IDENTITY_2D: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        const NORM: [f32; 2] = [1.0, 1.0];

        let output_tokens = token_count.checked_mul(n_seq).expect("token count");
        assert_eq!(state_values.len(), D_MODEL * output_tokens);
        assert_eq!(row_indices.len(), output_tokens);
        assert_eq!(positions.len(), output_tokens);
        assert_eq!(total_tokens_by_sequence.len(), n_seq);

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let state = graph
            .new_tensor_2d_f32(D_MODEL, output_tokens, "state")
            .expect("state tensor");
        let row_indices_tensor = graph
            .new_tensor_4d_i32(token_count, 1, n_seq, 1, "row_indices")
            .expect("row indices");
        let positions_tensor = graph
            .new_tensor_1d_i32(output_tokens, "positions")
            .expect("positions");
        let attention_mask = graph
            .new_tensor_4d_f16(MAX_POSITIONS, token_count, 1, n_seq, "attention_mask")
            .expect("attention mask");
        let norm = graph.new_tensor_1d_f32(D_MODEL, "norm").expect("norm");
        let q = graph
            .new_tensor_2d_f32(D_MODEL, D_MODEL, "q")
            .expect("q weight");
        let k = graph
            .new_tensor_2d_f32(D_MODEL, D_MODEL, "k")
            .expect("k weight");
        let v = graph
            .new_tensor_2d_f32(D_MODEL, D_MODEL, "v")
            .expect("v weight");
        let output = graph
            .new_tensor_2d_f32(D_MODEL, D_MODEL, "output")
            .expect("output weight");
        let ffn = graph
            .new_tensor_2d_f32(D_MODEL, D_MODEL, "ffn")
            .expect("ffn weight");
        for input in [
            state,
            row_indices_tensor,
            positions_tensor,
            attention_mask,
            norm,
            q,
            k,
            v,
            output,
            ffn,
        ] {
            graph
                .set_input(input)
                .expect("test input should be settable");
        }

        let stack = compose_llm_decoder_layer_stack(
            &mut graph,
            1,
            LlmDecoderStackConfig {
                d_model: D_MODEL,
                head_dim: HEAD_DIM,
                q_heads: HEADS,
                kv_heads: HEADS,
                q_width: D_MODEL,
                k_width: D_MODEL,
                v_width: D_MODEL,
                token_count,
                n_seq,
                rms_norm_epsilon: 1.0e-6,
                rope: GgmlRopeExtParams::qwen_neox(HEAD_DIM, MAX_POSITIONS, 10_000.0)
                    .expect("rope params"),
                use_native_gqa: true,
                use_flash_attention: true,
            },
            LlmDecoderStackInputs {
                state,
                row_indices: row_indices_tensor,
                positions: positions_tensor,
                attention_mask: Some(attention_mask),
                kv_span: MAX_POSITIONS,
                key_history_name: "key_history",
                value_history_name: "value_history",
            },
            None,
            |_layer_index| LlmLayerWeights {
                attn_norm_weight: norm,
                qkv_weight: None,
                q_weight: q,
                k_weight: k,
                v_weight: v,
                q_norm_weight: norm,
                k_norm_weight: norm,
                output_weight: output,
                ffn_norm_weight: norm,
                ffn_gate_weight: ffn,
                ffn_up_weight: ffn,
                ffn_down_weight: ffn,
                q_lora: None,
                k_lora: None,
                v_lora: None,
                output_lora: None,
                ffn_gate_lora: None,
                ffn_up_lora: None,
                ffn_down_lora: None,
            },
            |_step, source| source,
        )
        .expect("LLM stack should build");
        graph.set_output(stack.state).expect("state output");

        graph
            .set_f32_slice(state, state_values, "state")
            .expect("state upload");
        graph
            .set_i32_slice(row_indices_tensor, row_indices, "row_indices")
            .expect("row-index upload");
        graph
            .set_i32_slice(positions_tensor, positions, "positions")
            .expect("position upload");
        let row_indices_usize: Vec<usize> = row_indices
            .iter()
            .map(|&row| usize::try_from(row).expect("non-negative row"))
            .collect();
        let mask_bits = build_fixed_kv_attention_mask_bits_for_query_rows(
            MAX_POSITIONS,
            token_count,
            n_seq,
            &row_indices_usize,
        )
        .expect("mask should build");
        graph
            .set_f16_bits_slice(attention_mask, &mask_bits, "attention_mask")
            .expect("mask upload");
        graph
            .set_f32_slice(norm, &NORM, "norm")
            .expect("norm upload");
        for (tensor, name) in [
            (q, "q"),
            (k, "k"),
            (v, "v"),
            (output, "output"),
            (ffn, "ffn"),
        ] {
            graph
                .set_f32_slice(tensor, &IDENTITY_2D, name)
                .expect("weight upload");
        }
        for (key_history, value_history) in stack.kv_inputs {
            let zero_history = vec![0.0_f32; HEAD_DIM * MAX_POSITIONS * HEADS * n_seq];
            let key_history_values = initial_key_history.unwrap_or(zero_history.as_slice());
            let value_history_values = initial_value_history.unwrap_or(zero_history.as_slice());
            graph
                .set_f32_slice(key_history, key_history_values, "key_history")
                .expect("key history upload");
            graph
                .set_f32_slice(value_history, value_history_values, "value_history")
                .expect("value history upload");
        }

        let projected_k = stack.kv_outputs[0].0;
        let projected_v = stack.kv_outputs[0].1;
        let mut outputs = graph
            .compute_outputs_f32(&[
                (stack.state, D_MODEL * output_tokens),
                (projected_k, D_MODEL * output_tokens),
                (projected_v, D_MODEL * output_tokens),
            ])
            .expect("LLM stack should compute");
        let projected_v = outputs.pop().expect("projected V output");
        let projected_k = outputs.pop().expect("projected K output");
        let state = outputs.pop().expect("state output");
        LlmTestStackOutput {
            state,
            projected_k,
            projected_v,
        }
    }

    #[test]
    fn llm_decoder_stack_batched_sequence_output_matches_serial_runs() {
        let seq0 = [0.25, -0.75];
        let seq1 = [1.25, 0.5];
        let batched = compute_single_layer_llm_stack(
            1,
            2,
            &[seq0[0], seq0[1], seq1[0], seq1[1]],
            &[0, 2],
            &[0, 2],
            &[1, 3],
            None,
            None,
        );
        let serial0 = compute_single_layer_llm_stack(1, 1, &seq0, &[0], &[0], &[1], None, None);
        let serial1 = compute_single_layer_llm_stack(1, 1, &seq1, &[2], &[2], &[3], None, None);

        assert_f32_slices_close(&batched[0..2], &serial0, 1.0e-4);
        assert_f32_slices_close(&batched[2..4], &serial1, 1.0e-4);
    }

    fn synthetic_llm_state(sequence_index: usize) -> [f32; 2] {
        [
            0.25 + sequence_index as f32 * 0.1875,
            -0.75 + sequence_index as f32 * 0.109375,
        ]
    }

    #[test]
    fn llm_decoder_stack_batched_ragged_n4_n8_outputs_match_serial_runs() {
        for n_seq in [4, 8] {
            let mut state_values = Vec::new();
            let mut row_indices = Vec::new();
            let mut positions = Vec::new();
            let mut total_tokens = Vec::new();
            let mut states = Vec::new();
            for sequence_index in 0..n_seq {
                let state = synthetic_llm_state(sequence_index);
                let row = (sequence_index * 2) % 3;
                state_values.extend_from_slice(&state);
                row_indices.push(row as i32);
                positions.push(row as i32);
                total_tokens.push(row + 1);
                states.push(state);
            }

            let batched = compute_single_layer_llm_stack(
                1,
                n_seq,
                &state_values,
                &row_indices,
                &positions,
                &total_tokens,
                None,
                None,
            );
            for sequence_index in 0..n_seq {
                let serial = compute_single_layer_llm_stack(
                    1,
                    1,
                    &states[sequence_index],
                    &[row_indices[sequence_index]],
                    &[positions[sequence_index]],
                    &[total_tokens[sequence_index]],
                    None,
                    None,
                );
                let start = sequence_index * 2;
                assert_f32_slices_close(&batched[start..start + 2], &serial, 1.0e-4);
            }
        }
    }

    #[test]
    fn llm_decoder_stack_prefill_query_output_matches_serial_runs() {
        let token0 = [0.25, -0.75];
        let token1 = [1.25, 0.5];
        let batched = compute_single_layer_llm_stack(
            2,
            1,
            &[token0[0], token0[1], token1[0], token1[1]],
            &[0, 1],
            &[0, 1],
            &[2],
            None,
            None,
        );
        let serial0 =
            compute_single_layer_llm_stack_outputs(1, 1, &token0, &[0], &[0], &[1], None, None);
        let mut seeded_k = vec![0.0_f32; 2 * 3];
        let mut seeded_v = vec![0.0_f32; 2 * 3];
        seeded_k[0..2].copy_from_slice(&serial0.projected_k);
        seeded_v[0..2].copy_from_slice(&serial0.projected_v);
        let serial1 = compute_single_layer_llm_stack(
            1,
            1,
            &token1,
            &[1],
            &[1],
            &[2],
            Some(&seeded_k),
            Some(&seeded_v),
        );

        assert_f32_slices_close(&batched[0..2], &serial0.state, 1.0e-4);
        assert_f32_slices_close(&batched[2..4], &serial1, 1.0e-4);
    }

    #[test]
    fn llm_decoder_stack_batched_prefill_query_output_matches_serial_runs() {
        let seq0_token0 = [0.25, -0.75];
        let seq0_token1 = [1.25, 0.5];
        let seq1_token0 = [-0.5, 0.75];
        let seq1_token1 = [0.5, 1.5];
        let batched = compute_single_layer_llm_stack(
            2,
            2,
            &[
                seq0_token0[0],
                seq0_token0[1],
                seq0_token1[0],
                seq0_token1[1],
                seq1_token0[0],
                seq1_token0[1],
                seq1_token1[0],
                seq1_token1[1],
            ],
            &[0, 1, 0, 1],
            &[0, 1, 0, 1],
            &[2, 2],
            None,
            None,
        );
        let serial0 = compute_single_layer_llm_stack(
            2,
            1,
            &[
                seq0_token0[0],
                seq0_token0[1],
                seq0_token1[0],
                seq0_token1[1],
            ],
            &[0, 1],
            &[0, 1],
            &[2],
            None,
            None,
        );
        let serial1 = compute_single_layer_llm_stack(
            2,
            1,
            &[
                seq1_token0[0],
                seq1_token0[1],
                seq1_token1[0],
                seq1_token1[1],
            ],
            &[0, 1],
            &[0, 1],
            &[2],
            None,
            None,
        );

        assert_f32_slices_close(&batched[0..4], &serial0, 1.0e-4);
        assert_f32_slices_close(&batched[4..8], &serial1, 1.0e-4);
    }

    #[test]
    fn resident_kv_arena_allocates_requested_layers() {
        let runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let resident = allocate_zeroed_llm_resident_kv_arena(
            &runner,
            GgmlCpuGraphConfig::default().context_bytes,
            2,
            3,
            4,
            1,
            1,
            "test_resident_kv",
        )
        .expect("resident KV arena should allocate");

        assert_eq!(resident.layers.len(), 2);
        assert_eq!(resident.graph_tensors().len(), 2);
    }

    #[test]
    fn fixed_kv_attention_mask_marks_unwritten_slots_negative_infinity() {
        let mask = build_fixed_kv_attention_mask_bits(5, 3).expect("mask should build");
        assert_eq!(mask, vec![0x0000, 0x0000, 0x0000, 0xFC00, 0xFC00]);
    }

    #[test]
    fn fixed_kv_attention_mask_stacks_sequence_planes() {
        let mask = build_fixed_kv_attention_mask_bits_for_sequences(4, &[1, 3])
            .expect("batched mask should build");
        assert_eq!(
            mask,
            vec![
                0x0000, 0xFC00, 0xFC00, 0xFC00, 0x0000, 0x0000, 0x0000, 0xFC00,
            ]
        );
    }

    #[test]
    fn fixed_kv_attention_mask_fails_closed_when_tokens_exceed_span() {
        let error = build_fixed_kv_attention_mask_bits(2, 3).expect_err("oversized mask must fail");
        assert!(matches!(
            error,
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "fixed KV attention mask token count exceeds max positions"
            }
        ));
    }
}
