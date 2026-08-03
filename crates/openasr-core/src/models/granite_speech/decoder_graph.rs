//! Granite (dense) decoder-only LLM ggml graph -- the `granite-4.0-1b-base`
//! checkpoint `granite-speech-4.1-2b` modality-aligns (`GraniteForCausalLM` in
//! HF `transformers.models.granite.modeling_granite`, generated from
//! `modular_granite.py`).
//!
//! This is a **prefill-only, one-shot forward** (no incremental KV-cache): it
//! takes the whole token sequence and returns final hidden states + logits for
//! every position in a single call. That is deliberately narrower than the
//! shared `nn::decoder::llm_layer`/`compose_llm_decoder_layer_stack` (which
//! carry the full incremental `set_rows` KV-cache session machinery qwen's
//! production decode needs) -- this pass only needs to validate the decoder's
//! *numerics* against an HF reference (see `parity`'s
//! `granite_speech_decoder_prefill_parity`); wiring a real incremental decode
//! session (executor + `decode_policy_component_registry`) is a further
//! follow-up.
//!
//! Reuse-vs-fork: `apply_rms_norm` (`nn::norm`) is reused as-is (identical
//! math). `nn::decoder::llm_layer` itself is NOT reused, and this is a
//! deliberate fork, not an oversight -- it hardcodes the attention scale at
//! `1/sqrt(head_dim)` and the residual add as plain `hidden + block_out` with
//! no multiplier, whereas Granite's `GraniteAttention`/`GraniteDecoderLayer`
//! (see HF `modeling_granite.py`) replace *both*:
//!   - `scaling = config.attention_multiplier` (`0.0078125` = `1/head_dim`,
//!     not `1/sqrt(head_dim)`) is used in place of the usual softmax scale;
//!   - every residual add is `residual + sublayer_out * config.
//!     residual_multiplier` (`0.22`), on both the attention and MLP branches;
//!   - `embedding_multiplier` (`12.0`) scales the token embedding once before
//!     the layer stack (applied by the caller, not this module);
//!   - `logits_scaling` (`8.0`) *divides* the final `lm_head` logits (also the
//!     caller's job, see `prefill_logits`).
//!
//! Granite also has no QK-norm (`nn::decoder`'s `q_norm_weight`/
//! `k_norm_weight` would both be `None` here, same as Qwen2's shape) and no
//! attention/MLP biases. Extending the shared `LlmLayerConfig`/`LlmLayerWeights`
//! with two more optional knobs (a scale override + a residual multiplier)
//! would let a future pass fold this back into the shared stack; not done here
//! to avoid touching qwen's live production decode path in the same change
//! that introduces Granite's numeric core.
//!
//! GQA (16 query heads / 4 KV heads) uses ggml's native `mul_mat` batch
//! broadcast (`k.ne2=4` divides `q.ne2=16`), the same "native GQA, no
//! `repeat_kv`" convention `nn::decoder::expand_attention_kv`'s
//! `use_native_gqa=true` path already uses for qwen. RoPE is `NEOX` mode
//! (matches HF's `rotate_half`/half-split convention), `theta=10000`, no
//! scaling (`rope_type: "default"`).

#![allow(dead_code)]

use std::collections::HashMap;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlLoadedTensor, GgmlLoadedWeightContext,
    GgmlRopeExtParams, GgmlStaticTensor, GgmlStaticTensorArena,
};
use crate::nn::norm::{RmsNormSteps, apply_rms_norm};

#[derive(Debug, thiserror::Error)]
pub(crate) enum GraniteSpeechDecoderError {
    #[error("granite-speech decoder shape error: {reason}")]
    Shape { reason: String },
    #[error("granite-speech decoder missing weight tensor '{name}'")]
    MissingWeight { name: String },
    #[error("granite-speech decoder weight '{name}' has {actual} values, expected {expected}")]
    WeightLen {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("granite-speech decoder GGML backend failed at {stage}: {source}")]
    Ggml {
        stage: &'static str,
        source: GgmlCpuGraphError,
    },
}

fn ggml_err(stage: &'static str) -> impl Fn(GgmlCpuGraphError) -> GraniteSpeechDecoderError + Copy {
    move |source| GraniteSpeechDecoderError::Ggml { stage, source }
}

const RMS_NORM_STEPS: RmsNormSteps = RmsNormSteps {
    norm: "rms_norm",
    scale: "rms_norm_scale",
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GraniteSpeechDecoderConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub attention_multiplier: f32,
    pub embedding_multiplier: f32,
    pub residual_multiplier: f32,
    pub logits_scaling: f32,
}

impl GraniteSpeechDecoderConfig {
    pub(crate) fn granite_speech_4_1_2b() -> Self {
        Self {
            hidden_size: 2048,
            num_layers: 40,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 128,
            intermediate_size: 4096,
            vocab_size: 100353,
            rms_norm_eps: 1.0e-5,
            rope_theta: 10000.0,
            attention_multiplier: 0.0078125,
            embedding_multiplier: 12.0,
            residual_multiplier: 0.22,
            logits_scaling: 8.0,
        }
    }
}

pub(crate) trait GraniteSpeechDecoderWeightProvider {
    fn tensor(&self, name: &str) -> Option<&[f32]>;
}

impl GraniteSpeechDecoderWeightProvider for HashMap<String, Vec<f32>> {
    fn tensor(&self, name: &str) -> Option<&[f32]> {
        self.get(name).map(Vec::as_slice)
    }
}

pub(crate) struct DecoderLayerWeights<'a> {
    attn_norm_w: GgmlCpuTensor<'a>,
    q_w: GgmlCpuTensor<'a>,
    k_w: GgmlCpuTensor<'a>,
    v_w: GgmlCpuTensor<'a>,
    o_w: GgmlCpuTensor<'a>,
    ffn_norm_w: GgmlCpuTensor<'a>,
    gate_w: GgmlCpuTensor<'a>,
    up_w: GgmlCpuTensor<'a>,
    down_w: GgmlCpuTensor<'a>,
}

struct DecoderWeights<'a> {
    layers: Vec<DecoderLayerWeights<'a>>,
    final_norm_w: GgmlCpuTensor<'a>,
    lm_head_w: GgmlCpuTensor<'a>,
}

struct WeightBuilder<'p> {
    provider: &'p dyn GraniteSpeechDecoderWeightProvider,
    uploads: Vec<(GgmlStaticTensor, &'p [f32], &'static str)>,
}

impl<'p> WeightBuilder<'p> {
    fn new(provider: &'p dyn GraniteSpeechDecoderWeightProvider) -> Self {
        Self {
            provider,
            uploads: Vec::new(),
        }
    }

    fn fetch(&self, name: &str, expected: usize) -> Result<&'p [f32], GraniteSpeechDecoderError> {
        let data =
            self.provider
                .tensor(name)
                .ok_or_else(|| GraniteSpeechDecoderError::MissingWeight {
                    name: name.to_string(),
                })?;
        if data.len() != expected {
            return Err(GraniteSpeechDecoderError::WeightLen {
                name: name.to_string(),
                expected,
                actual: data.len(),
            });
        }
        Ok(data)
    }

    fn w1<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        len: usize,
    ) -> Result<GgmlCpuTensor<'a>, GraniteSpeechDecoderError> {
        let data = self.fetch(name, len)?;
        let handle = arena
            .new_tensor_1d_f32(len, "granite_speech_decoder_weight")
            .map_err(ggml_err("weight_alloc_1d"))?;
        self.uploads
            .push((handle, data, "granite_speech_decoder_weight"));
        Ok(arena.graph_tensor(handle))
    }

    fn w2<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        ne0: usize,
        ne1: usize,
    ) -> Result<GgmlCpuTensor<'a>, GraniteSpeechDecoderError> {
        let data = self.fetch(name, ne0 * ne1)?;
        let handle = arena
            .new_tensor_2d_f32(ne0, ne1, "granite_speech_decoder_weight")
            .map_err(ggml_err("weight_alloc_2d"))?;
        self.uploads
            .push((handle, data, "granite_speech_decoder_weight"));
        Ok(arena.graph_tensor(handle))
    }

    fn upload(&self, arena: &mut GgmlStaticTensorArena) -> Result<(), GraniteSpeechDecoderError> {
        for (handle, data, name) in &self.uploads {
            arena
                .set_f32_slice(*handle, data, name)
                .map_err(ggml_err("upload_weight"))?;
        }
        Ok(())
    }
}

fn build_layer_weights<'a, 'p>(
    arena: &GgmlStaticTensorArena,
    builder: &mut WeightBuilder<'p>,
    config: &GraniteSpeechDecoderConfig,
    index: usize,
) -> Result<DecoderLayerWeights<'a>, GraniteSpeechDecoderError> {
    let d = config.hidden_size;
    let q_width = config.num_heads * config.head_dim;
    let kv_width = config.num_kv_heads * config.head_dim;
    let inter = config.intermediate_size;
    let p = |suffix: &str| format!("language_model.model.layers.{index}.{suffix}");
    Ok(DecoderLayerWeights {
        attn_norm_w: builder.w1(arena, &p("input_layernorm.weight"), d)?,
        q_w: builder.w2(arena, &p("self_attn.q_proj.weight"), d, q_width)?,
        k_w: builder.w2(arena, &p("self_attn.k_proj.weight"), d, kv_width)?,
        v_w: builder.w2(arena, &p("self_attn.v_proj.weight"), d, kv_width)?,
        o_w: builder.w2(arena, &p("self_attn.o_proj.weight"), q_width, d)?,
        ffn_norm_w: builder.w1(arena, &p("post_attention_layernorm.weight"), d)?,
        gate_w: builder.w2(arena, &p("mlp.gate_proj.weight"), d, inter)?,
        up_w: builder.w2(arena, &p("mlp.up_proj.weight"), d, inter)?,
        down_w: builder.w2(arena, &p("mlp.down_proj.weight"), inter, d)?,
    })
}

pub(crate) fn linear<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    input: GgmlCpuTensor<'a>,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechDecoderError> {
    graph.mul_mat(weight, input).map_err(ggml_err(stage))
}

/// Reinterpret a 2-D weight to ggml `[in, out]` order (the `mul_mat` operand
/// layout, `weight.ne0 == input.ne0 == in_dim`) with a pure metadata reshape.
///
/// A weight already stored `[in, out]` -- the quantized q8_0/q4_k packs and the
/// f32 static-arena path (which allocates the tensor `[in, out]` explicitly) --
/// reshapes to its own shape: a byte-for-byte view, so it stays bit-exact
/// (proven by `granite_incremental_decode_matches_full_recompute_bit_exact`,
/// which runs the arena path). A weight stored transposed as `[out, in]` -- the
/// f16 converter's torch-order layout (`package_import` writes each rank>=2
/// tensor's HF `[out, in]` shape verbatim) -- is reinterpreted to `[in, out]`;
/// its row-major flat buffer is byte-identical to the operand the arena path
/// materialized by uploading that same flat f32 into an explicitly
/// `[in, out]`-shaped tensor, so keep-quantized binding is correct regardless of
/// which dim convention a given pack was written with (no per-pack branch).
pub(crate) fn weight_in_major<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    in_dim: usize,
    out_dim: usize,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechDecoderError> {
    graph
        .reshape_2d(weight, in_dim, out_dim)
        .map_err(ggml_err(stage))
}

pub(crate) fn rms_norm<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    eps: f32,
    weight: GgmlCpuTensor<'a>,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechDecoderError> {
    apply_rms_norm(graph, input, eps, weight, RMS_NORM_STEPS, |s, source| {
        GraniteSpeechDecoderError::Ggml { stage: s, source }
    })
}

/// `[n_tokens, n_tokens]` additive causal mask (`0` where `key <= query`,
/// `f32::MIN` otherwise), row-major `[query][key]`.
pub(crate) fn causal_mask(n_tokens: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; n_tokens * n_tokens];
    for q in 0..n_tokens {
        for k in 0..n_tokens {
            if k > q {
                mask[q * n_tokens + k] = f32::MIN;
            }
        }
    }
    mask
}

/// Per-head query/key/value projections for one Granite decoder layer, already
/// RoPE-rotated and permuted to `[head_dim, n_tokens, heads]` (query-major, the
/// layout the batched `mul_mat` attention consumes; GQA broadcast relies on
/// `kv_heads` dividing `q_heads`). `k_perm`/`v_perm` are exactly the per-token
/// K/V an incremental KV-cache must persist (see
/// `decode_session::GraniteSpeechDecodeSession`).
pub(crate) struct GranitePreAttention<'a> {
    pub q_perm: GgmlCpuTensor<'a>,
    pub k_perm: GgmlCpuTensor<'a>,
    pub v_perm: GgmlCpuTensor<'a>,
}

/// The pre-attention half of a Granite decoder layer: `rms_norm -> q/k/v proj
/// -> reshape-to-heads -> RoPE(q,k) -> row-contiguous query-major views`. Shared,
/// byte-for-byte, by the one-shot prefill (`decoder_layer`) and the incremental
/// decode step, so a cached K/V produced here is provably identical to the K/V
/// a full recompute would produce at the same position.
pub(crate) fn granite_pre_attention<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    hidden: GgmlCpuTensor<'a>,
    positions: GgmlCpuTensor<'a>,
    weights: &DecoderLayerWeights<'a>,
    config: &GraniteSpeechDecoderConfig,
    n_tokens: usize,
    rope: GgmlRopeExtParams,
) -> Result<GranitePreAttention<'a>, GraniteSpeechDecoderError> {
    let map = ggml_err("decoder_layer");
    let head_dim = config.head_dim;
    let q_heads = config.num_heads;
    let kv_heads = config.num_kv_heads;

    let normed = rms_norm(graph, hidden, config.rms_norm_eps, weights.attn_norm_w)?;
    let hidden_size = config.hidden_size;
    let q_width = q_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let q_w = weight_in_major(graph, weights.q_w, hidden_size, q_width, "q_proj_reshape")?;
    let k_w = weight_in_major(graph, weights.k_w, hidden_size, kv_width, "k_proj_reshape")?;
    let v_w = weight_in_major(graph, weights.v_w, hidden_size, kv_width, "v_proj_reshape")?;
    let q = linear(graph, q_w, normed, "q_proj")?;
    let k = linear(graph, k_w, normed, "k_proj")?;
    let v = linear(graph, v_w, normed, "v_proj")?;

    let q = graph
        .reshape_3d(q, head_dim, q_heads, n_tokens)
        .map_err(map)?;
    let k = graph
        .reshape_3d(k, head_dim, kv_heads, n_tokens)
        .map_err(map)?;
    let v = graph
        .reshape_3d(v, head_dim, kv_heads, n_tokens)
        .map_err(map)?;

    let q = graph.rope_ext(q, positions, rope).map_err(map)?;
    let k = graph.rope_ext(k, positions, rope).map_err(map)?;

    // -> [head_dim, n_tokens, heads] (query-major) for the batched mul_mat
    // attention below; GQA broadcast relies on kv_heads dividing q_heads
    // (native ggml mul_mat batch broadcast, no repeat_kv materialization).
    let q_perm = graph.permute(q, 0, 2, 1, 3).map_err(map)?;
    let k_perm = graph.permute(k, 0, 2, 1, 3).map_err(map)?;
    let v_perm = graph.permute(v, 0, 2, 1, 3).map_err(map)?;
    let (q_perm, k_perm, v_perm) = if graph.backend_kind() == GgmlCpuGraphBackend::Metal {
        (q_perm, k_perm, v_perm)
    } else {
        // Preserve the established CPU/generic-GPU reduction and cache layout.
        // The strided row-view optimization is measured and transcript-gated
        // only on Metal's Flash Attention path.
        (
            graph.cont(q_perm).map_err(map)?,
            graph.cont(k_perm).map_err(map)?,
            graph.cont(v_perm).map_err(map)?,
        )
    };

    Ok(GranitePreAttention {
        q_perm,
        k_perm,
        v_perm,
    })
}

/// The post-attention half of a Granite decoder layer: fold the attention
/// context back to `[q_width, n_tokens]`, o-project + residual-scale into the
/// residual stream, then the SwiGLU MLP + residual-scale. Naive attention
/// returns `[head_dim, n_q, q_heads]` and needs a materializing layout merge;
/// `flash_attn_ext` already returns a reshape-compatible `[head_dim, q_heads,
/// n_q]` tensor. Shared by prefill and incremental decode.
pub(crate) fn granite_post_attention<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    hidden: GgmlCpuTensor<'a>,
    attended: GgmlCpuTensor<'a>,
    weights: &DecoderLayerWeights<'a>,
    config: &GraniteSpeechDecoderConfig,
    n_tokens: usize,
    flash_attention_output: bool,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechDecoderError> {
    let map = ggml_err("decoder_layer");
    let q_width = config.num_heads * config.head_dim;

    let attended = if !flash_attention_output {
        graph
            .cont(graph.permute(attended, 0, 2, 1, 3).map_err(map)?)
            .map_err(map)?
    } else {
        attended
    };
    let attended = graph.reshape_2d(attended, q_width, n_tokens).map_err(map)?;

    let hidden_size = config.hidden_size;
    let inter = config.intermediate_size;
    let o_w = weight_in_major(graph, weights.o_w, q_width, hidden_size, "o_proj_reshape")?;
    let attn_out = linear(graph, o_w, attended, "o_proj")?;
    let attn_out = graph
        .scale(attn_out, config.residual_multiplier)
        .map_err(map)?;
    let hidden = graph.add(hidden, attn_out).map_err(map)?;

    let ffn_normed = rms_norm(graph, hidden, config.rms_norm_eps, weights.ffn_norm_w)?;
    let gate_w = weight_in_major(
        graph,
        weights.gate_w,
        hidden_size,
        inter,
        "gate_proj_reshape",
    )?;
    let gate = linear(graph, gate_w, ffn_normed, "gate_proj")?;
    let gate = graph.silu(gate).map_err(map)?;
    let up_w = weight_in_major(graph, weights.up_w, hidden_size, inter, "up_proj_reshape")?;
    let up = linear(graph, up_w, ffn_normed, "up_proj")?;
    let gated = graph.mul(gate, up).map_err(map)?;
    let down_w = weight_in_major(
        graph,
        weights.down_w,
        inter,
        hidden_size,
        "down_proj_reshape",
    )?;
    let ffn_out = linear(graph, down_w, gated, "down_proj")?;
    let ffn_out = graph
        .scale(ffn_out, config.residual_multiplier)
        .map_err(map)?;
    graph.add(hidden, ffn_out).map_err(map)
}

#[allow(clippy::too_many_arguments)]
fn decoder_layer<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    hidden: GgmlCpuTensor<'a>,
    positions: GgmlCpuTensor<'a>,
    mask: GgmlCpuTensor<'a>,
    weights: &DecoderLayerWeights<'a>,
    config: &GraniteSpeechDecoderConfig,
    n_tokens: usize,
    rope: GgmlRopeExtParams,
) -> Result<GgmlCpuTensor<'a>, GraniteSpeechDecoderError> {
    let map = ggml_err("decoder_layer");
    let pre = granite_pre_attention(graph, hidden, positions, weights, config, n_tokens, rope)?;

    let scores = graph.mul_mat(pre.k_perm, pre.q_perm).map_err(map)?;
    let probs = graph
        .soft_max_ext(scores, Some(mask), config.attention_multiplier, 0.0)
        .map_err(map)?;
    let v_t = graph
        .cont(graph.transpose(pre.v_perm).map_err(map)?)
        .map_err(map)?;
    let attended = graph.mul_mat(v_t, probs).map_err(map)?;

    granite_post_attention(graph, hidden, attended, weights, config, n_tokens, false)
}

pub(crate) struct GraniteSpeechDecoderPrefillOutput {
    pub n_tokens: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,
    /// Final (post-final-RMSNorm) hidden states, `[n_tokens, hidden_size]`
    /// row-major.
    pub hidden_out: Vec<f32>,
    /// `lm_head` logits already divided by `logits_scaling`, `[n_tokens,
    /// vocab_size]` row-major.
    pub logits: Vec<f32>,
}

/// Look up one row of `language_model.model.embed_tokens.weight` (the raw,
/// un-scaled embedding table -- `embedding_multiplier` is applied once, to
/// the whole assembled sequence, by `prefill_logits_from_embeddings`, not
/// here). Exposed so `prompt.rs` can build the audio-spliced embedding
/// sequence (`get_merged_audio_embeddings`'s text-token half) on the host,
/// the same table `prefill_logits` itself gathers from.
pub(crate) fn embed_token_row<'p>(
    config: &GraniteSpeechDecoderConfig,
    provider: &'p dyn GraniteSpeechDecoderWeightProvider,
    token_id: u32,
) -> Result<&'p [f32], GraniteSpeechDecoderError> {
    let table = provider
        .tensor("language_model.model.embed_tokens.weight")
        .ok_or_else(|| GraniteSpeechDecoderError::MissingWeight {
            name: "language_model.model.embed_tokens.weight".to_string(),
        })?;
    let expected = config.hidden_size * config.vocab_size;
    if table.len() != expected {
        return Err(GraniteSpeechDecoderError::WeightLen {
            name: "language_model.model.embed_tokens.weight".to_string(),
            expected,
            actual: table.len(),
        });
    }
    if token_id as usize >= config.vocab_size {
        return Err(GraniteSpeechDecoderError::Shape {
            reason: format!(
                "token id {token_id} exceeds vocab_size {}",
                config.vocab_size
            ),
        });
    }
    let start = token_id as usize * config.hidden_size;
    Ok(&table[start..start + config.hidden_size])
}

/// One-shot prefill forward: embeds `token_ids` (scaled by
/// `embedding_multiplier`), runs the full decoder stack with a causal mask,
/// and returns hidden states + logits for every position. No KV-cache -- see
/// module doc. Implemented as a host-side embedding-table gather (bit-identical
/// to an in-graph `get_rows`) followed by `prefill_logits_from_embeddings`, so
/// text-only and audio-spliced prompts (`prompt.rs`) share one graph builder.
pub(crate) fn prefill_logits(
    config: &GraniteSpeechDecoderConfig,
    provider: &dyn GraniteSpeechDecoderWeightProvider,
    token_ids: &[u32],
    backend: GgmlCpuGraphBackend,
) -> Result<GraniteSpeechDecoderPrefillOutput, GraniteSpeechDecoderError> {
    if token_ids.is_empty() {
        return Err(GraniteSpeechDecoderError::Shape {
            reason: "token_ids must be non-empty".to_string(),
        });
    }
    let mut embeddings = Vec::with_capacity(token_ids.len() * config.hidden_size);
    for &id in token_ids {
        embeddings.extend_from_slice(embed_token_row(config, provider, id)?);
    }
    prefill_logits_from_embeddings(config, provider, &embeddings, token_ids.len(), backend)
}

/// Same forward as [`prefill_logits`], but takes an already-assembled
/// `[n_tokens, hidden_size]` embedding sequence directly (pre-`embedding_multiplier`,
/// post any audio splice -- see `prompt.rs::build_audio_prompt_embeddings`,
/// which mirrors HF's `GraniteSpeechModel.get_merged_audio_embeddings`).
pub(crate) fn prefill_logits_from_embeddings(
    config: &GraniteSpeechDecoderConfig,
    provider: &dyn GraniteSpeechDecoderWeightProvider,
    embeddings: &[f32],
    n_tokens: usize,
    backend: GgmlCpuGraphBackend,
) -> Result<GraniteSpeechDecoderPrefillOutput, GraniteSpeechDecoderError> {
    if n_tokens == 0 {
        return Err(GraniteSpeechDecoderError::Shape {
            reason: "n_tokens must be non-zero".to_string(),
        });
    }
    if embeddings.len() != n_tokens * config.hidden_size {
        return Err(GraniteSpeechDecoderError::Shape {
            reason: format!(
                "embeddings has {} values, expected {n_tokens}x{}",
                embeddings.len(),
                config.hidden_size
            ),
        });
    }

    let graph_config = GgmlCpuGraphConfig {
        context_bytes: 256 * 1024 * 1024,
        graph_size: 32768,
        n_threads: GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            backend,
            crate::ggml_runtime::GgmlCpuGraphThreadingWorkload::EncoderPrelude,
        ),
        backend,
        use_scheduler: true,
    };
    let mut runner = GgmlCpuGraphRunner::new(graph_config).map_err(ggml_err("runner_init"))?;
    let tensor_count = 32 + 32 * config.num_layers;
    let arena_bytes = GgmlCpuGraphConfig::metadata_context_bytes(tensor_count);
    let arena = runner
        .start_static_tensor_arena(arena_bytes)
        .map_err(ggml_err("static_tensor_arena"))?;

    let mut builder = WeightBuilder::new(provider);
    let mut layers = Vec::with_capacity(config.num_layers);
    for index in 0..config.num_layers {
        layers.push(build_layer_weights(&arena, &mut builder, config, index)?);
    }
    let final_norm_w = builder.w1(
        &arena,
        "language_model.model.norm.weight",
        config.hidden_size,
    )?;
    let lm_head_w = builder.w2(
        &arena,
        "language_model.lm_head.weight",
        config.hidden_size,
        config.vocab_size,
    )?;

    let positions_handle = arena
        .new_tensor_1d_i32(n_tokens, "granite_speech_decoder_positions")
        .map_err(ggml_err("weight_alloc_positions"))?;
    let mask_handle = arena
        .new_tensor_2d_f32(n_tokens, n_tokens, "granite_speech_decoder_mask")
        .map_err(ggml_err("weight_alloc_mask"))?;

    let mut arena = arena;
    builder.upload(&mut arena)?;
    let positions: Vec<i32> = (0..n_tokens as i32).collect();
    arena
        .set_i32_slice(
            positions_handle,
            &positions,
            "granite_speech_decoder_positions",
        )
        .map_err(ggml_err("upload_positions"))?;
    let mask_values = causal_mask(n_tokens);
    arena
        .set_f32_slice(mask_handle, &mask_values, "granite_speech_decoder_mask")
        .map_err(ggml_err("upload_mask"))?;

    let weights = DecoderWeights {
        layers,
        final_norm_w,
        lm_head_w,
    };

    let mut graph = runner.start_graph();
    let embed_tensor = graph
        .new_tensor_2d_f32(
            config.hidden_size,
            n_tokens,
            "granite_speech_decoder_embeds",
        )
        .map_err(ggml_err("input_alloc"))?;
    let positions_graph = arena.graph_tensor(positions_handle);
    let mask_graph = arena.graph_tensor(mask_handle);

    let mut hidden = graph
        .scale(embed_tensor, config.embedding_multiplier)
        .map_err(ggml_err("embed_scale"))?;

    let rope = GgmlRopeExtParams::qwen_neox(config.head_dim, n_tokens, config.rope_theta)
        .map_err(ggml_err("rope_params"))?;

    for layer in &weights.layers {
        hidden = decoder_layer(
            &mut graph,
            hidden,
            positions_graph,
            mask_graph,
            layer,
            config,
            n_tokens,
            rope,
        )?;
    }
    let hidden_out = rms_norm(&graph, hidden, config.rms_norm_eps, weights.final_norm_w)?;
    let lm_head_w = weight_in_major(
        &graph,
        weights.lm_head_w,
        config.hidden_size,
        config.vocab_size,
        "lm_head_reshape",
    )?;
    let logits_raw = linear(&graph, lm_head_w, hidden_out, "lm_head")?;
    let logits = graph
        .scale(logits_raw, 1.0 / config.logits_scaling)
        .map_err(ggml_err("logits_scale"))?;

    graph
        .set_output(hidden_out)
        .map_err(ggml_err("set_output_hidden"))?;
    graph
        .set_output(logits)
        .map_err(ggml_err("set_output_logits"))?;
    graph
        .set_input(embed_tensor)
        .map_err(ggml_err("mark_input(embeddings)"))?;
    graph
        .prepare_outputs_for_upload(&[hidden_out, logits])
        .map_err(ggml_err("prepare_outputs"))?;
    graph
        .set_f32_slice(embed_tensor, embeddings, "granite_speech_decoder_embeds")
        .map_err(ggml_err("upload_embeddings"))?;

    let mut outputs = graph
        .compute_outputs_f32(&[
            (hidden_out, n_tokens * config.hidden_size),
            (logits, n_tokens * config.vocab_size),
        ])
        .map_err(ggml_err("compute"))?;
    let logits = outputs.pop().expect("logits tap");
    let hidden_out = outputs.pop().expect("hidden tap");

    Ok(GraniteSpeechDecoderPrefillOutput {
        n_tokens,
        hidden_dim: config.hidden_size,
        vocab_size: config.vocab_size,
        hidden_out,
        logits,
    })
}

/// Per-layer weight-tensor handles, held in a persistent static arena.
struct GraniteLayerWeightHandles {
    attn_norm: GgmlStaticTensor,
    q: GgmlStaticTensor,
    k: GgmlStaticTensor,
    v: GgmlStaticTensor,
    o: GgmlStaticTensor,
    ffn_norm: GgmlStaticTensor,
    gate: GgmlStaticTensor,
    up: GgmlStaticTensor,
    down: GgmlStaticTensor,
}

/// All Granite decoder weights uploaded ONCE into a static tensor arena that
/// survives across every `GgmlCpuGraphRunner::start_graph` call (`start_graph`
/// only `ggml_reset`s the runner's own graph context, never this arena's --
/// see that method's doc). This is what lets the incremental decode session
/// (`decode_session`) prefill + run every single-token step against the same
/// 2B-parameter weight upload instead of re-uploading the whole decoder every
/// token; only the tiny per-step inputs (one embedding, one position, the K/V
/// history views) live in the reset-per-step graph context.
pub(crate) struct GraniteDecoderWeightArena {
    arena: GgmlStaticTensorArena,
    layers: Vec<GraniteLayerWeightHandles>,
    final_norm: GgmlStaticTensor,
    lm_head: GgmlStaticTensor,
}

impl GraniteDecoderWeightArena {
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add_vec(&self.layers, "granite decoder arena layer handles")?;
        Ok(bytes.finish())
    }

    /// Allocate every weight tensor in a fresh static arena and upload the
    /// provider's f32 weights once. All allocation happens before the first
    /// upload (the arena finalizes its backend buffer on first write and
    /// refuses further allocation afterward).
    pub(crate) fn load<'p>(
        runner: &GgmlCpuGraphRunner,
        config: &GraniteSpeechDecoderConfig,
        provider: &'p dyn GraniteSpeechDecoderWeightProvider,
    ) -> Result<Self, GraniteSpeechDecoderError> {
        let tensor_count = 32 + 32 * config.num_layers;
        let arena_bytes = GgmlCpuGraphConfig::metadata_context_bytes(tensor_count);
        let arena = runner
            .start_static_tensor_arena(arena_bytes)
            .map_err(ggml_err("session_static_tensor_arena"))?;

        let d = config.hidden_size;
        let q_width = config.num_heads * config.head_dim;
        let kv_width = config.num_kv_heads * config.head_dim;
        let inter = config.intermediate_size;

        // (handle, weight data) pairs, filled during allocation and uploaded
        // in one pass afterward (the arena finalizes its backend buffer on the
        // first write and refuses further allocation, so all `new_tensor_*`
        // calls must precede all uploads).
        let mut pending: Vec<(GgmlStaticTensor, &'p [f32])> = Vec::new();
        let alloc_1d = |name: &str,
                        len: usize,
                        pending: &mut Vec<(GgmlStaticTensor, &'p [f32])>|
         -> Result<GgmlStaticTensor, GraniteSpeechDecoderError> {
            let data = fetch_weight(provider, name, len)?;
            let handle = arena
                .new_tensor_1d_f32(len, "granite_speech_session_weight")
                .map_err(ggml_err("session_weight_alloc_1d"))?;
            pending.push((handle, data));
            Ok(handle)
        };
        let alloc_2d = |name: &str,
                        ne0: usize,
                        ne1: usize,
                        pending: &mut Vec<(GgmlStaticTensor, &'p [f32])>|
         -> Result<GgmlStaticTensor, GraniteSpeechDecoderError> {
            let data = fetch_weight(provider, name, ne0 * ne1)?;
            let handle = arena
                .new_tensor_2d_f32(ne0, ne1, "granite_speech_session_weight")
                .map_err(ggml_err("session_weight_alloc_2d"))?;
            pending.push((handle, data));
            Ok(handle)
        };

        let mut layers = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            let p = |suffix: &str| format!("language_model.model.layers.{index}.{suffix}");
            layers.push(GraniteLayerWeightHandles {
                attn_norm: alloc_1d(&p("input_layernorm.weight"), d, &mut pending)?,
                q: alloc_2d(&p("self_attn.q_proj.weight"), d, q_width, &mut pending)?,
                k: alloc_2d(&p("self_attn.k_proj.weight"), d, kv_width, &mut pending)?,
                v: alloc_2d(&p("self_attn.v_proj.weight"), d, kv_width, &mut pending)?,
                o: alloc_2d(&p("self_attn.o_proj.weight"), q_width, d, &mut pending)?,
                ffn_norm: alloc_1d(&p("post_attention_layernorm.weight"), d, &mut pending)?,
                gate: alloc_2d(&p("mlp.gate_proj.weight"), d, inter, &mut pending)?,
                up: alloc_2d(&p("mlp.up_proj.weight"), d, inter, &mut pending)?,
                down: alloc_2d(&p("mlp.down_proj.weight"), inter, d, &mut pending)?,
            });
        }
        let final_norm = alloc_1d("language_model.model.norm.weight", d, &mut pending)?;
        let lm_head = alloc_2d(
            "language_model.lm_head.weight",
            config.hidden_size,
            config.vocab_size,
            &mut pending,
        )?;

        let mut arena = arena;
        for (handle, data) in &pending {
            arena
                .set_f32_slice(*handle, data, "granite_speech_session_weight")
                .map_err(ggml_err("session_upload_weight"))?;
        }

        Ok(Self {
            arena,
            layers,
            final_norm,
            lm_head,
        })
    }

    /// Fresh per-graph `GgmlCpuTensor` wrappers over layer `index`'s persistent
    /// weight tensors (re-derived every step; the underlying arena storage is
    /// uploaded once).
    pub(crate) fn layer_weights<'a>(&self, index: usize) -> DecoderLayerWeights<'a> {
        let h = &self.layers[index];
        DecoderLayerWeights {
            attn_norm_w: self.arena.graph_tensor(h.attn_norm),
            q_w: self.arena.graph_tensor(h.q),
            k_w: self.arena.graph_tensor(h.k),
            v_w: self.arena.graph_tensor(h.v),
            o_w: self.arena.graph_tensor(h.o),
            ffn_norm_w: self.arena.graph_tensor(h.ffn_norm),
            gate_w: self.arena.graph_tensor(h.gate),
            up_w: self.arena.graph_tensor(h.up),
            down_w: self.arena.graph_tensor(h.down),
        }
    }

    pub(crate) fn final_norm_weight<'a>(&self) -> GgmlCpuTensor<'a> {
        self.arena.graph_tensor(self.final_norm)
    }

    pub(crate) fn lm_head_weight<'a>(&self) -> GgmlCpuTensor<'a> {
        self.arena.graph_tensor(self.lm_head)
    }
}

/// Per-layer weight handles bound zero-copy from the mmap'd `.oasr` pack (the
/// keep-quantized twin of [`GraniteLayerWeightHandles`], which lives in an
/// f32-uploaded static arena).
struct GraniteLayerLoadedHandles {
    attn_norm: GgmlLoadedTensor,
    q: GgmlLoadedTensor,
    k: GgmlLoadedTensor,
    v: GgmlLoadedTensor,
    o: GgmlLoadedTensor,
    ffn_norm: GgmlLoadedTensor,
    gate: GgmlLoadedTensor,
    up: GgmlLoadedTensor,
    down: GgmlLoadedTensor,
}

/// The keep-quantized decoder weights: every 2-D projection, the two per-layer
/// RMSNorm scales, the final norm, and the lm_head are bound zero-copy from the
/// pack's own already-resident (mmap'd, native q8_0/q4_k/f16/f32) tensor -- no
/// host dequant-to-f32 and no static-arena upload, exactly the shape
/// `firered_aed`/`cohere`/... already use. A q8_0 2-B decoder therefore stays
/// ~2.5 GB resident (its packed size) instead of the ~8 GB an f32 dequant +
/// arena upload cost. `mul_mat` consumes whatever `mul_mat`-compatible type the
/// pack stores each projection in, unchanged.
pub(crate) struct GraniteDecoderLoadedWeights {
    layers: Vec<GraniteLayerLoadedHandles>,
    final_norm: GgmlLoadedTensor,
    lm_head: GgmlLoadedTensor,
}

fn loaded_tensor(
    loaded: &GgmlLoadedWeightContext,
    name: &str,
) -> Result<GgmlLoadedTensor, GraniteSpeechDecoderError> {
    loaded
        .tensor(name)
        .ok_or_else(|| GraniteSpeechDecoderError::MissingWeight {
            name: name.to_string(),
        })
}

impl GraniteDecoderLoadedWeights {
    pub(crate) fn quoted_retained_system_memory_bytes(num_layers: usize) -> Result<u64, String> {
        let bytes = num_layers
            .checked_mul(std::mem::size_of::<GraniteLayerLoadedHandles>())
            .ok_or_else(|| "granite loaded decoder handle quote overflowed".to_string())?;
        u64::try_from(bytes)
            .map_err(|_| "granite loaded decoder handle quote exceeds u64".to_string())
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add_vec(&self.layers, "granite loaded decoder layer handles")?;
        Ok(bytes.finish())
    }

    /// Bind every decoder weight by its on-disk name (the converter stores the
    /// `language_model.*` tensors under their verbatim HF names, so these match
    /// the f32-arena path's `WeightBuilder` names exactly).
    pub(crate) fn load(
        loaded: &GgmlLoadedWeightContext,
        config: &GraniteSpeechDecoderConfig,
    ) -> Result<Self, GraniteSpeechDecoderError> {
        let mut layers = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            let p = |suffix: &str| format!("language_model.model.layers.{index}.{suffix}");
            layers.push(GraniteLayerLoadedHandles {
                attn_norm: loaded_tensor(loaded, &p("input_layernorm.weight"))?,
                q: loaded_tensor(loaded, &p("self_attn.q_proj.weight"))?,
                k: loaded_tensor(loaded, &p("self_attn.k_proj.weight"))?,
                v: loaded_tensor(loaded, &p("self_attn.v_proj.weight"))?,
                o: loaded_tensor(loaded, &p("self_attn.o_proj.weight"))?,
                ffn_norm: loaded_tensor(loaded, &p("post_attention_layernorm.weight"))?,
                gate: loaded_tensor(loaded, &p("mlp.gate_proj.weight"))?,
                up: loaded_tensor(loaded, &p("mlp.up_proj.weight"))?,
                down: loaded_tensor(loaded, &p("mlp.down_proj.weight"))?,
            });
        }
        Ok(Self {
            layers,
            final_norm: loaded_tensor(loaded, "language_model.model.norm.weight")?,
            lm_head: loaded_tensor(loaded, "language_model.lm_head.weight")?,
        })
    }

    fn layer_weights<'a>(&self, index: usize) -> DecoderLayerWeights<'a> {
        let h = &self.layers[index];
        DecoderLayerWeights {
            attn_norm_w: h.attn_norm.as_graph_tensor(),
            q_w: h.q.as_graph_tensor(),
            k_w: h.k.as_graph_tensor(),
            v_w: h.v.as_graph_tensor(),
            o_w: h.o.as_graph_tensor(),
            ffn_norm_w: h.ffn_norm.as_graph_tensor(),
            gate_w: h.gate.as_graph_tensor(),
            up_w: h.up.as_graph_tensor(),
            down_w: h.down.as_graph_tensor(),
        }
    }

    fn final_norm_weight<'a>(&self) -> GgmlCpuTensor<'a> {
        self.final_norm.as_graph_tensor()
    }

    fn lm_head_weight<'a>(&self) -> GgmlCpuTensor<'a> {
        self.lm_head.as_graph_tensor()
    }
}

/// The decoder weight residency backing a [`GraniteSpeechDecodeSession`]: either
/// the legacy f32 static-arena upload (used by the synthetic bit-exact test and
/// any host-`HashMap` provider) or the keep-quantized zero-copy bind from the
/// mmap'd pack. The session's forward code is identical for both -- it only ever
/// asks for `layer_weights`/`final_norm_weight`/`lm_head_weight`.
pub(crate) enum GraniteDecoderWeights {
    Arena(GraniteDecoderWeightArena),
    Loaded(GraniteDecoderLoadedWeights),
}

impl GraniteDecoderWeights {
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        match self {
            Self::Arena(arena) => arena.retained_system_memory_bytes(),
            Self::Loaded(loaded) => loaded.retained_system_memory_bytes(),
        }
    }

    pub(crate) fn layer_weights<'a>(&self, index: usize) -> DecoderLayerWeights<'a> {
        match self {
            Self::Arena(arena) => arena.layer_weights(index),
            Self::Loaded(loaded) => loaded.layer_weights(index),
        }
    }

    pub(crate) fn final_norm_weight<'a>(&self) -> GgmlCpuTensor<'a> {
        match self {
            Self::Arena(arena) => arena.final_norm_weight(),
            Self::Loaded(loaded) => loaded.final_norm_weight(),
        }
    }

    pub(crate) fn lm_head_weight<'a>(&self) -> GgmlCpuTensor<'a> {
        match self {
            Self::Arena(arena) => arena.lm_head_weight(),
            Self::Loaded(loaded) => loaded.lm_head_weight(),
        }
    }
}

fn fetch_weight<'p>(
    provider: &'p dyn GraniteSpeechDecoderWeightProvider,
    name: &str,
    expected: usize,
) -> Result<&'p [f32], GraniteSpeechDecoderError> {
    let data = provider
        .tensor(name)
        .ok_or_else(|| GraniteSpeechDecoderError::MissingWeight {
            name: name.to_string(),
        })?;
    if data.len() != expected {
        return Err(GraniteSpeechDecoderError::WeightLen {
            name: name.to_string(),
            expected,
            actual: data.len(),
        });
    }
    Ok(data)
}
