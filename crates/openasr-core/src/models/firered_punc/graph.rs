//! FireRedPunc BERT forward graph: token/position/segment embeddings ->
//! post-embedding LayerNorm -> N post-norm transformer blocks (bidirectional
//! self-attention, no causal mask; erf-GELU FFN) -> token-classification head
//! -> `[label_count, seq]` logits.
//!
//! Composed from the shared `nn::{attn, ffn, norm}` building blocks (the same
//! `apply_affine_layer_norm` / `apply_feed_forward_residual` /
//! `reshape_projection_to_attention_heads` used by the ASR encoders). All
//! weights are host f32 uploaded to the graph arena; the forward is driven with
//! `GgmlCpuGraphRunner` + `compute_output_f32`.

use crate::ggml_runtime::{
    GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor,
    GgmlStaticTensor, GgmlStaticTensorArena,
};
use crate::nn::attn::{
    AttentionHeadLayout, AttentionReshapeSteps, AttentionValueMergeSteps,
    STANDARD_HEAD_PERMUTE_AXES, attention_context_from_probs,
    reshape_projection_to_attention_heads,
};
use crate::nn::ffn::{
    FeedForwardActivation, FeedForwardResidualSteps, apply_feed_forward_residual,
};
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

use super::config::{FIRERED_PUNC_LAYER_NORM_EPSILON, FireRedPuncExecutionMetadata};
use super::weights::{FireRedPuncLayerWeights, FireRedPuncWeights, NamedTensor};

const FIRERED_PUNC_GRAPH_CONTEXT_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub(crate) enum FireRedPuncGraphError {
    #[error("firered-punc graph build failed at '{step}': {source}")]
    Build {
        step: &'static str,
        source: GgmlCpuGraphError,
    },
    #[error("firered-punc graph execution failed: {reason}")]
    Execution { reason: String },
    #[error("firered-punc input length {got} exceeds max positions {max}")]
    SequenceTooLong { got: usize, max: usize },
    #[error("firered-punc input is empty")]
    EmptyInput,
}

fn bf(step: &'static str) -> impl Fn(GgmlCpuGraphError) -> FireRedPuncGraphError {
    move |source| FireRedPuncGraphError::Build { step, source }
}

fn bf2(step: &'static str, source: GgmlCpuGraphError) -> FireRedPuncGraphError {
    FireRedPuncGraphError::Build { step, source }
}

struct LayerArena {
    attn_q_weight: GgmlStaticTensor,
    attn_q_bias: GgmlStaticTensor,
    attn_k_weight: GgmlStaticTensor,
    attn_k_bias: GgmlStaticTensor,
    attn_v_weight: GgmlStaticTensor,
    attn_v_bias: GgmlStaticTensor,
    attn_output_weight: GgmlStaticTensor,
    attn_output_bias: GgmlStaticTensor,
    attn_norm_weight: GgmlStaticTensor,
    attn_norm_bias: GgmlStaticTensor,
    ffn_up_weight: GgmlStaticTensor,
    ffn_up_bias: GgmlStaticTensor,
    ffn_down_weight: GgmlStaticTensor,
    ffn_down_bias: GgmlStaticTensor,
    ffn_norm_weight: GgmlStaticTensor,
    ffn_norm_bias: GgmlStaticTensor,
}

pub(crate) struct FireRedPuncGraph {
    metadata: FireRedPuncExecutionMetadata,
    runner: GgmlCpuGraphRunner,
    arena: GgmlStaticTensorArena,
    token_embd: GgmlStaticTensor,
    token_type_embd: GgmlStaticTensor,
    position_embd: GgmlStaticTensor,
    embd_norm_weight: GgmlStaticTensor,
    embd_norm_bias: GgmlStaticTensor,
    layers: Vec<LayerArena>,
    punc_head_weight: GgmlStaticTensor,
    punc_head_bias: GgmlStaticTensor,
}

fn alloc_2d(
    arena: &GgmlStaticTensorArena,
    weight: &NamedTensor,
    ne0: usize,
    ne1: usize,
    step: &'static str,
) -> Result<GgmlStaticTensor, FireRedPuncGraphError> {
    debug_assert_eq!(weight.values.len(), ne0 * ne1, "{step} shape");
    arena.new_tensor_2d_f32(ne0, ne1, step).map_err(bf(step))
}

fn alloc_1d(
    arena: &GgmlStaticTensorArena,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<GgmlStaticTensor, FireRedPuncGraphError> {
    arena
        .new_tensor_1d_f32(weight.values.len(), step)
        .map_err(bf(step))
}

fn upload(
    arena: &mut GgmlStaticTensorArena,
    handle: GgmlStaticTensor,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<(), FireRedPuncGraphError> {
    arena
        .set_f32_slice(handle, &weight.values, step)
        .map_err(bf(step))
}

impl FireRedPuncGraph {
    pub(crate) fn new(
        weights: &FireRedPuncWeights,
        metadata: FireRedPuncExecutionMetadata,
    ) -> Result<Self, FireRedPuncGraphError> {
        let mut config = GgmlCpuGraphConfig::default();
        config.context_bytes = FIRERED_PUNC_GRAPH_CONTEXT_BYTES;
        config.graph_size = config.graph_size.max(metadata.layers * 64 + 512);
        let runner = GgmlCpuGraphRunner::new(config).map_err(bf("runner_init"))?;
        let mut arena = runner
            .start_static_tensor_arena(FIRERED_PUNC_GRAPH_CONTEXT_BYTES)
            .map_err(bf("arena_init"))?;

        let d = metadata.d_model;
        let ffn = metadata.ffn_dim;

        // ----- declare arena tensors (first upload freezes allocation) -----
        let token_embd = alloc_2d(
            &arena,
            &weights.token_embd,
            d,
            metadata.vocab_size,
            "token_embd",
        )?;
        let token_type_embd = alloc_2d(
            &arena,
            &weights.token_type_embd,
            d,
            weights.token_type_embd.values.len() / d,
            "token_type_embd",
        )?;
        let position_embd = alloc_2d(
            &arena,
            &weights.position_embd,
            d,
            metadata.max_positions,
            "position_embd",
        )?;
        let embd_norm_weight = alloc_1d(&arena, &weights.embd_norm_weight, "embd_norm_w")?;
        let embd_norm_bias = alloc_1d(&arena, &weights.embd_norm_bias, "embd_norm_b")?;
        let mut layers = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            layers.push(alloc_layer(&arena, layer, d, ffn)?);
        }
        let punc_head_weight = alloc_2d(
            &arena,
            &weights.punc_head_weight,
            d,
            metadata.label_count,
            "punc_head_w",
        )?;
        let punc_head_bias = alloc_1d(&arena, &weights.punc_head_bias, "punc_head_b")?;

        // ----- upload arena values -----
        upload(&mut arena, token_embd, &weights.token_embd, "token_embd")?;
        upload(
            &mut arena,
            token_type_embd,
            &weights.token_type_embd,
            "token_type_embd",
        )?;
        upload(
            &mut arena,
            position_embd,
            &weights.position_embd,
            "position_embd",
        )?;
        upload(
            &mut arena,
            embd_norm_weight,
            &weights.embd_norm_weight,
            "embd_norm_w",
        )?;
        upload(
            &mut arena,
            embd_norm_bias,
            &weights.embd_norm_bias,
            "embd_norm_b",
        )?;
        for (layer, handles) in weights.layers.iter().zip(&layers) {
            upload_layer(&mut arena, layer, handles)?;
        }
        upload(
            &mut arena,
            punc_head_weight,
            &weights.punc_head_weight,
            "punc_head_w",
        )?;
        upload(
            &mut arena,
            punc_head_bias,
            &weights.punc_head_bias,
            "punc_head_b",
        )?;

        Ok(Self {
            metadata,
            runner,
            arena,
            token_embd,
            token_type_embd,
            position_embd,
            embd_norm_weight,
            embd_norm_bias,
            layers,
            punc_head_weight,
            punc_head_bias,
        })
    }

    /// Run the BERT forward over `token_ids` (the full sequence, including the
    /// caller's `[CLS]`/`[SEP]` wrapping) and return `[label_count, seq]` logits
    /// laid out label-fastest: `logits[pos * label_count + label]`.
    pub(crate) fn forward(&mut self, token_ids: &[u32]) -> Result<Vec<f32>, FireRedPuncGraphError> {
        let seq = token_ids.len();
        if seq == 0 {
            return Err(FireRedPuncGraphError::EmptyInput);
        }
        if seq > self.metadata.max_positions {
            return Err(FireRedPuncGraphError::SequenceTooLong {
                got: seq,
                max: self.metadata.max_positions,
            });
        }
        let heads = self.metadata.heads;
        let head_dim = self.metadata.head_dim;
        let eps = FIRERED_PUNC_LAYER_NORM_EPSILON;
        let label_count = self.metadata.label_count;

        let ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
        let pos_i32: Vec<i32> = (0..seq as i32).collect();
        let type_i32: Vec<i32> = vec![0; seq];

        let mut graph = self.runner.start_graph();
        let ids_t = graph.new_tensor_1d_i32(seq, "ids").map_err(bf("new_ids"))?;
        graph.set_input(ids_t).map_err(bf("set_ids"))?;
        let pos_t = graph
            .new_tensor_1d_i32(seq, "posids")
            .map_err(bf("new_pos"))?;
        graph.set_input(pos_t).map_err(bf("set_pos"))?;
        let type_t = graph
            .new_tensor_1d_i32(seq, "typeids")
            .map_err(bf("new_type"))?;
        graph.set_input(type_t).map_err(bf("set_type"))?;

        // ----- embeddings: token + position + segment, then LayerNorm -----
        let tok = graph
            .get_rows(self.arena.graph_tensor(self.token_embd), ids_t)
            .map_err(bf("embd_token"))?;
        let pos = graph
            .get_rows(self.arena.graph_tensor(self.position_embd), pos_t)
            .map_err(bf("embd_pos"))?;
        let seg = graph
            .get_rows(self.arena.graph_tensor(self.token_type_embd), type_t)
            .map_err(bf("embd_seg"))?;
        let mut state = graph.add(tok, pos).map_err(bf("embd_add_pos"))?;
        state = graph.add(state, seg).map_err(bf("embd_add_seg"))?;
        state = apply_affine_layer_norm(
            &graph,
            state,
            eps,
            self.arena.graph_tensor(self.embd_norm_weight),
            self.arena.graph_tensor(self.embd_norm_bias),
            AffineLayerNormSteps {
                norm: "embd_norm",
                scale: "embd_norm_scale",
                bias: "embd_norm_bias",
            },
            bf2,
        )?;

        let layout = AttentionHeadLayout {
            head_dim,
            attention_heads: heads,
            sequence_len: seq,
        };
        for handles in &self.layers {
            state = bert_block(&mut graph, &self.arena, state, handles, layout, eps)?;
        }

        // ----- classification head: [d_model, label] x [d_model, seq] -----
        let mut logits = graph
            .mul_mat(self.arena.graph_tensor(self.punc_head_weight), state)
            .map_err(bf("head_matmul"))?;
        logits = graph
            .add(logits, self.arena.graph_tensor(self.punc_head_bias))
            .map_err(bf("head_bias"))?;
        graph.set_output(logits).map_err(bf("set_output"))?;
        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(bf("prepare_outputs"))?;

        graph
            .set_i32_slice(ids_t, &ids_i32, "upload_ids")
            .map_err(bf("upload_ids"))?;
        graph
            .set_i32_slice(pos_t, &pos_i32, "upload_pos")
            .map_err(bf("upload_pos"))?;
        graph
            .set_i32_slice(type_t, &type_i32, "upload_type")
            .map_err(bf("upload_type"))?;

        let want = label_count * seq;
        graph
            .compute_output_f32(logits, want)
            .map_err(|error| FireRedPuncGraphError::Execution {
                reason: error.to_string(),
            })
    }
}

fn bert_block<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    arena: &GgmlStaticTensorArena,
    input: GgmlCpuTensor<'a>,
    h: &LayerArena,
    layout: AttentionHeadLayout,
    eps: f32,
) -> Result<GgmlCpuTensor<'a>, FireRedPuncGraphError> {
    // ----- bidirectional self-attention (no causal mask) -----
    let q = linear(
        graph,
        arena,
        h.attn_q_weight,
        h.attn_q_bias,
        input,
        "attn_q",
    )?;
    let k = linear(
        graph,
        arena,
        h.attn_k_weight,
        h.attn_k_bias,
        input,
        "attn_k",
    )?;
    let v = linear(
        graph,
        arena,
        h.attn_v_weight,
        h.attn_v_bias,
        input,
        "attn_v",
    )?;

    let q_heads = reshape_projection_to_attention_heads(
        graph,
        q,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        AttentionReshapeSteps {
            reshape: "attn_q_reshape",
            permute: "attn_q_permute",
            cont: "attn_q_cont",
        },
        bf2,
    )?;
    let k_heads = reshape_projection_to_attention_heads(
        graph,
        k,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        AttentionReshapeSteps {
            reshape: "attn_k_reshape",
            permute: "attn_k_permute",
            cont: "attn_k_cont",
        },
        bf2,
    )?;
    let v_heads = reshape_projection_to_attention_heads(
        graph,
        v,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        AttentionReshapeSteps {
            reshape: "attn_v_reshape",
            permute: "attn_v_permute",
            cont: "attn_v_cont",
        },
        bf2,
    )?;
    let k_cont = graph.cont(k_heads).map_err(bf("attn_k_cont2"))?;
    let mut scores = graph.mul_mat(k_cont, q_heads).map_err(bf("attn_scores"))?;
    scores = graph
        .scale(scores, 1.0 / (layout.head_dim as f32).sqrt())
        .map_err(bf("attn_scale"))?;
    scores = graph.soft_max(scores).map_err(bf("attn_softmax"))?;
    let context = attention_context_from_probs(
        graph,
        v_heads,
        scores,
        layout,
        AttentionValueMergeSteps {
            value_permute: "attn_v_t_permute",
            value_cont: "attn_v_t_cont",
            context_mul: "attn_ctx_mul",
            context_merge_permute: "attn_ctx_permute",
            context_merge_cont: "attn_ctx_cont",
            context_merge_reshape: "attn_ctx_reshape",
        },
        bf2,
    )?;
    let attn_out = linear(
        graph,
        arena,
        h.attn_output_weight,
        h.attn_output_bias,
        context,
        "attn_out",
    )?;
    let attn_residual = graph.add(input, attn_out).map_err(bf("attn_residual"))?;
    let state = apply_affine_layer_norm(
        graph,
        attn_residual,
        eps,
        arena.graph_tensor(h.attn_norm_weight),
        arena.graph_tensor(h.attn_norm_bias),
        AffineLayerNormSteps {
            norm: "attn_norm",
            scale: "attn_norm_scale",
            bias: "attn_norm_bias",
        },
        bf2,
    )?;

    // ----- feed-forward (erf-GELU) + post-norm -----
    let up_w = arena.graph_tensor(h.ffn_up_weight);
    let up_b = arena.graph_tensor(h.ffn_up_bias);
    let down_w = arena.graph_tensor(h.ffn_down_weight);
    let down_b = arena.graph_tensor(h.ffn_down_bias);
    let ffn_out = apply_feed_forward_residual(
        graph,
        state,
        state,
        FeedForwardActivation::GeluErf,
        None,
        FeedForwardResidualSteps {
            activation: "ffn_gelu",
            scale: None,
            residual: "ffn_residual",
        },
        |g, x| {
            let up = g.mul_mat(up_w, x).map_err(bf("ffn_up"))?;
            g.add(up, up_b).map_err(bf("ffn_up_bias"))
        },
        |g, x| {
            let down = g.mul_mat(down_w, x).map_err(bf("ffn_down"))?;
            g.add(down, down_b).map_err(bf("ffn_down_bias"))
        },
        bf2,
    )?;
    apply_affine_layer_norm(
        graph,
        ffn_out,
        eps,
        arena.graph_tensor(h.ffn_norm_weight),
        arena.graph_tensor(h.ffn_norm_bias),
        AffineLayerNormSteps {
            norm: "ffn_norm",
            scale: "ffn_norm_scale",
            bias: "ffn_norm_bias",
        },
        bf2,
    )
}

/// `mul_mat(weight, x) + bias` -- a biased linear over `[in, seq]`.
fn linear<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    arena: &GgmlStaticTensorArena,
    weight: GgmlStaticTensor,
    bias: GgmlStaticTensor,
    x: GgmlCpuTensor<'a>,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, FireRedPuncGraphError> {
    let projected = graph
        .mul_mat(arena.graph_tensor(weight), x)
        .map_err(bf(step))?;
    graph
        .add(projected, arena.graph_tensor(bias))
        .map_err(bf(step))
}

fn alloc_layer(
    arena: &GgmlStaticTensorArena,
    layer: &FireRedPuncLayerWeights,
    d: usize,
    ffn: usize,
) -> Result<LayerArena, FireRedPuncGraphError> {
    Ok(LayerArena {
        attn_q_weight: alloc_2d(arena, &layer.attn_q_weight, d, d, "attn_q_w")?,
        attn_q_bias: alloc_1d(arena, &layer.attn_q_bias, "attn_q_b")?,
        attn_k_weight: alloc_2d(arena, &layer.attn_k_weight, d, d, "attn_k_w")?,
        attn_k_bias: alloc_1d(arena, &layer.attn_k_bias, "attn_k_b")?,
        attn_v_weight: alloc_2d(arena, &layer.attn_v_weight, d, d, "attn_v_w")?,
        attn_v_bias: alloc_1d(arena, &layer.attn_v_bias, "attn_v_b")?,
        attn_output_weight: alloc_2d(arena, &layer.attn_output_weight, d, d, "attn_out_w")?,
        attn_output_bias: alloc_1d(arena, &layer.attn_output_bias, "attn_out_b")?,
        attn_norm_weight: alloc_1d(arena, &layer.attn_norm_weight, "attn_norm_w")?,
        attn_norm_bias: alloc_1d(arena, &layer.attn_norm_bias, "attn_norm_b")?,
        ffn_up_weight: alloc_2d(arena, &layer.ffn_up_weight, d, ffn, "ffn_up_w")?,
        ffn_up_bias: alloc_1d(arena, &layer.ffn_up_bias, "ffn_up_b")?,
        ffn_down_weight: alloc_2d(arena, &layer.ffn_down_weight, ffn, d, "ffn_down_w")?,
        ffn_down_bias: alloc_1d(arena, &layer.ffn_down_bias, "ffn_down_b")?,
        ffn_norm_weight: alloc_1d(arena, &layer.ffn_norm_weight, "ffn_norm_w")?,
        ffn_norm_bias: alloc_1d(arena, &layer.ffn_norm_bias, "ffn_norm_b")?,
    })
}

/// Argmax label id per position over label-fastest `[label_count, seq]` logits.
pub(crate) fn argmax_labels_per_position(
    logits: &[f32],
    label_count: usize,
    seq: usize,
) -> Vec<usize> {
    (0..seq)
        .map(|pos| {
            let row = &logits[pos * label_count..(pos + 1) * label_count];
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect()
}

fn upload_layer(
    arena: &mut GgmlStaticTensorArena,
    layer: &FireRedPuncLayerWeights,
    h: &LayerArena,
) -> Result<(), FireRedPuncGraphError> {
    upload(arena, h.attn_q_weight, &layer.attn_q_weight, "attn_q_w")?;
    upload(arena, h.attn_q_bias, &layer.attn_q_bias, "attn_q_b")?;
    upload(arena, h.attn_k_weight, &layer.attn_k_weight, "attn_k_w")?;
    upload(arena, h.attn_k_bias, &layer.attn_k_bias, "attn_k_b")?;
    upload(arena, h.attn_v_weight, &layer.attn_v_weight, "attn_v_w")?;
    upload(arena, h.attn_v_bias, &layer.attn_v_bias, "attn_v_b")?;
    upload(
        arena,
        h.attn_output_weight,
        &layer.attn_output_weight,
        "attn_out_w",
    )?;
    upload(
        arena,
        h.attn_output_bias,
        &layer.attn_output_bias,
        "attn_out_b",
    )?;
    upload(
        arena,
        h.attn_norm_weight,
        &layer.attn_norm_weight,
        "attn_norm_w",
    )?;
    upload(
        arena,
        h.attn_norm_bias,
        &layer.attn_norm_bias,
        "attn_norm_b",
    )?;
    upload(arena, h.ffn_up_weight, &layer.ffn_up_weight, "ffn_up_w")?;
    upload(arena, h.ffn_up_bias, &layer.ffn_up_bias, "ffn_up_b")?;
    upload(
        arena,
        h.ffn_down_weight,
        &layer.ffn_down_weight,
        "ffn_down_w",
    )?;
    upload(arena, h.ffn_down_bias, &layer.ffn_down_bias, "ffn_down_b")?;
    upload(
        arena,
        h.ffn_norm_weight,
        &layer.ffn_norm_weight,
        "ffn_norm_w",
    )?;
    upload(arena, h.ffn_norm_bias, &layer.ffn_norm_bias, "ffn_norm_b")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nt(name: &str, dims: Vec<usize>, values: Vec<f32>) -> NamedTensor {
        NamedTensor {
            name: name.to_string(),
            dims,
            values,
        }
    }

    /// A tiny 1-layer BERT (d_model=4, 1 head, ffn=4, vocab=6, 5 labels) whose
    /// attention and FFN projections are zeroed, so each block reduces to a
    /// LayerNorm and the forward computes `logits = head . LayerNorm(embed)`.
    /// The classifier head is one-hot on dim 0 for label 1 with a +5 baseline
    /// on label 2, making per-token argmax hand-verifiable: a token embedded at
    /// [4,0,0,0] normalises to a positive dim-0 and wins label 1; the zero
    /// embedding normalises to all-zero and falls back to the label-2 baseline.
    fn tiny_metadata() -> FireRedPuncExecutionMetadata {
        FireRedPuncExecutionMetadata {
            layers: 1,
            d_model: 4,
            ffn_dim: 4,
            heads: 1,
            head_dim: 4,
            vocab_size: 6,
            max_positions: 8,
            label_count: 5,
        }
    }

    fn tiny_weights() -> FireRedPuncWeights {
        let zeros = |n: usize| vec![0.0f32; n];
        let ones = |n: usize| vec![1.0f32; n];
        let layer = FireRedPuncLayerWeights {
            attn_q_weight: nt("blk.0.attn_q.weight", vec![4, 4], zeros(16)),
            attn_q_bias: nt("blk.0.attn_q.bias", vec![4], zeros(4)),
            attn_k_weight: nt("blk.0.attn_k.weight", vec![4, 4], zeros(16)),
            attn_k_bias: nt("blk.0.attn_k.bias", vec![4], zeros(4)),
            attn_v_weight: nt("blk.0.attn_v.weight", vec![4, 4], zeros(16)),
            attn_v_bias: nt("blk.0.attn_v.bias", vec![4], zeros(4)),
            attn_output_weight: nt("blk.0.attn_output.weight", vec![4, 4], zeros(16)),
            attn_output_bias: nt("blk.0.attn_output.bias", vec![4], zeros(4)),
            attn_norm_weight: nt("blk.0.attn_norm.weight", vec![4], ones(4)),
            attn_norm_bias: nt("blk.0.attn_norm.bias", vec![4], zeros(4)),
            ffn_up_weight: nt("blk.0.ffn_up.weight", vec![4, 4], zeros(16)),
            ffn_up_bias: nt("blk.0.ffn_up.bias", vec![4], zeros(4)),
            ffn_down_weight: nt("blk.0.ffn_down.weight", vec![4, 4], zeros(16)),
            ffn_down_bias: nt("blk.0.ffn_down.bias", vec![4], zeros(4)),
            ffn_norm_weight: nt("blk.0.ffn_norm.weight", vec![4], ones(4)),
            ffn_norm_bias: nt("blk.0.ffn_norm.bias", vec![4], zeros(4)),
        };
        // token_embd [d=4, vocab=6], column-major per token: token 1 = [4,0,0,0].
        let mut token_embd = zeros(4 * 6);
        token_embd[4] = 4.0;
        // head [d=4, label=5]: label 1 column = [10,0,0,0], others 0.
        let mut head = zeros(4 * 5);
        head[4] = 10.0;
        FireRedPuncWeights {
            token_embd: nt("token_embd.weight", vec![4, 6], token_embd),
            token_type_embd: nt("token_type_embd.weight", vec![4, 2], zeros(8)),
            position_embd: nt("position_embd.weight", vec![4, 8], zeros(32)),
            embd_norm_weight: nt("embd_norm.weight", vec![4], ones(4)),
            embd_norm_bias: nt("embd_norm.bias", vec![4], zeros(4)),
            layers: vec![layer],
            punc_head_weight: nt("punc_head.weight", vec![4, 5], head),
            punc_head_bias: nt("punc_head.bias", vec![5], vec![0.0, 0.0, 5.0, 0.0, 0.0]),
        }
    }

    #[test]
    fn synthetic_forward_matches_hand_computed_argmax() {
        let metadata = tiny_metadata();
        let weights = tiny_weights();
        let mut graph = FireRedPuncGraph::new(&weights, metadata).expect("build graph");
        // Token 1 embeds to a positive dim-0 (-> label 1); token 0 to zero
        // (-> label-2 baseline).
        let logits = graph.forward(&[1, 0]).expect("forward");
        assert_eq!(logits.len(), metadata.label_count * 2);
        assert!(logits.iter().all(|v| v.is_finite()), "no NaN/Inf");
        let labels = argmax_labels_per_position(&logits, metadata.label_count, 2);
        assert_eq!(
            labels,
            vec![1, 2],
            "per-token argmax matches hand computation"
        );
    }

    #[test]
    fn synthetic_forward_is_deterministic() {
        let metadata = tiny_metadata();
        let weights = tiny_weights();
        let mut graph = FireRedPuncGraph::new(&weights, metadata).expect("build graph");
        let a = graph.forward(&[1, 0, 2]).expect("forward a");
        let b = graph.forward(&[1, 0, 2]).expect("forward b");
        assert_eq!(a, b, "same input -> same logits");
    }

    #[test]
    fn empty_and_overlong_inputs_fail_closed() {
        let metadata = tiny_metadata();
        let weights = tiny_weights();
        let mut graph = FireRedPuncGraph::new(&weights, metadata).expect("build graph");
        assert!(matches!(
            graph.forward(&[]),
            Err(FireRedPuncGraphError::EmptyInput)
        ));
        let too_long: Vec<u32> = vec![0; metadata.max_positions + 1];
        assert!(matches!(
            graph.forward(&too_long),
            Err(FireRedPuncGraphError::SequenceTooLong { .. })
        ));
    }
}
