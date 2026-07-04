//! Dolphin `small.cn` Transformer decoder graph (WeNet format).
//!
//! Self-contained ggml graph assembler for the standard WeNet `TransformerDecoder`
//! that rides on top of the parity-verified E-Branchformer encoder. Like the
//! encoder graph it reuses the shared `nn/` building blocks (affine LayerNorm,
//! attention head reshape + context merge, ReLU feed-forward residual) but keeps
//! all family-specific tensor wiring here so nothing in the shared layers grows a
//! Dolphin special case.
//!
//! Architecture (WeNet `TransformerDecoder`, `normalize_before=True`, char
//! tokenizer, verified against the `small.cn.pt` state dict):
//!   token embed `decoder.embed.0` -> `* sqrt(d_model)` + absolute sinusoidal
//!   `decoder.embed.1.pe` -> 12 x DecoderLayer -> final LayerNorm
//!   `decoder.after_norm` -> `decoder.output_layer` (untied, `[vocab, d_model]`).
//! Each DecoderLayer is pre-norm: `norm1 -> causal self-attn -> residual`,
//! `norm2 -> cross-attn on the encoder output -> residual`,
//! `norm3 -> single ReLU FFN -> residual`. LayerNorm eps 1e-5, attention scale
//! `1/sqrt(head_dim)`, self-attention masked causally, cross-attention full-context.
//!
//! Numerics: the attention is assembled in f32 (`mul_mat` scores -> `soft_max_ext`
//! with an additive causal mask -> `mul_mat` context), the same pattern the
//! encoder attention branch and the moonshine decoder use to stay bit-close to the
//! PyTorch reference. This is deliberately NOT the `nn::decoder::seq2seq_layer`
//! path: that layer keeps an f16 self-attention KV cache for incremental GPU
//! decode, whose half-precision rounding cannot meet the <1e-3 golden-logit parity
//! bound this graph is validated against. `seq2seq_layer` remains the right home
//! for the later incremental-decode runtime; the reference-exact teacher-forced
//! forward lives here.
//!
//! WIP: this is the numeric core validated by the `parity` dev harness; the
//! CTC-prefix-beam + attention-rescoring joint decode wiring lands separately, so
//! the public surface is dead in a plain lib build until then.
#![allow(dead_code)]

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor,
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

use super::encoder_graph::DolphinWeightProvider;

#[derive(Debug, thiserror::Error)]
pub(crate) enum DolphinDecoderError {
    #[error("dolphin decoder shape error: {reason}")]
    Shape { reason: String },
    #[error("dolphin decoder missing weight tensor '{name}'")]
    MissingWeight { name: String },
    #[error("dolphin decoder weight '{name}' has {actual} values, expected {expected}")]
    WeightLen {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("dolphin decoder GGML backend failed at {stage}: {source}")]
    Ggml {
        stage: &'static str,
        source: GgmlCpuGraphError,
    },
}

fn ggml_err(stage: &'static str) -> impl Fn(GgmlCpuGraphError) -> DolphinDecoderError + Copy {
    move |source| DolphinDecoderError::Ggml { stage, source }
}

/// Scalar/shape configuration for the Dolphin `small.cn` Transformer decoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DolphinDecoderConfig {
    pub d_model: usize,
    pub attention_heads: usize,
    pub head_dim: usize,
    pub ffn_units: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    /// Length of the sinusoidal position table baked into `decoder.embed.1.pe`.
    pub max_positions: usize,
    pub layer_norm_epsilon: f32,
}

impl DolphinDecoderConfig {
    pub(crate) fn small_cn() -> Self {
        Self {
            d_model: 768,
            attention_heads: 12,
            head_dim: 64,
            ffn_units: 3072,
            num_layers: 12,
            vocab_size: 18173,
            max_positions: 5000,
            layer_norm_epsilon: 1e-5,
        }
    }
}

/// Decoder logits over the teacher-forced prefix.
#[derive(Debug, Clone)]
pub(crate) struct DolphinDecoderOutput {
    pub token_count: usize,
    pub vocab_size: usize,
    /// Row-major `[token_count, vocab_size]` raw logits (pre-softmax), vocab
    /// innermost. Row `i` is the distribution predicting the token that follows
    /// prompt position `i`.
    pub logits: Vec<f32>,
}

impl DolphinDecoderOutput {
    /// Logits of the last prefix position: the distribution over the first token
    /// emitted after the whole prompt.
    pub(crate) fn last_token_logits(&self) -> &[f32] {
        let start = (self.token_count - 1) * self.vocab_size;
        &self.logits[start..start + self.vocab_size]
    }
}

// --- weight upload plumbing (mirrors the encoder graph's WeightBuilder) --------

type Upload<'a, 'p> = (GgmlCpuTensor<'a>, &'p [f32], &'static str);

struct WeightBuilder<'a, 'p> {
    provider: &'p dyn DolphinWeightProvider,
    uploads: Vec<Upload<'a, 'p>>,
}

impl<'a, 'p> WeightBuilder<'a, 'p> {
    fn new(provider: &'p dyn DolphinWeightProvider) -> Self {
        Self {
            provider,
            uploads: Vec::new(),
        }
    }

    fn fetch(&self, name: &str, expected: usize) -> Result<&'p [f32], DolphinDecoderError> {
        let data =
            self.provider
                .tensor(name)
                .ok_or_else(|| DolphinDecoderError::MissingWeight {
                    name: name.to_string(),
                })?;
        if data.len() != expected {
            return Err(DolphinDecoderError::WeightLen {
                name: name.to_string(),
                expected,
                actual: data.len(),
            });
        }
        Ok(data)
    }

    /// A 1-D weight (bias / LayerNorm gamma-beta).
    fn w1(
        &mut self,
        graph: &GgmlCpuGraphBuilder<'a>,
        name: &str,
        len: usize,
    ) -> Result<GgmlCpuTensor<'a>, DolphinDecoderError> {
        let data = self.fetch(name, len)?;
        let tensor = graph
            .new_tensor_1d_f32(len, "dolphin_dec_weight")
            .map_err(ggml_err("weight_alloc_1d"))?;
        self.uploads.push((tensor, data, "dolphin_dec_weight"));
        Ok(tensor)
    }

    /// A 2-D weight uploaded as ggml `[ne0=in, ne1=out]` for `mul_mat(w, x)`.
    /// The PyTorch `Linear` weight `[out, in]` (and the `[vocab, d_model]` embed /
    /// output tables) is row-major with `in` innermost, which is exactly this ggml
    /// layout when uploaded raw.
    fn w2(
        &mut self,
        graph: &GgmlCpuGraphBuilder<'a>,
        name: &str,
        ne0: usize,
        ne1: usize,
    ) -> Result<GgmlCpuTensor<'a>, DolphinDecoderError> {
        let data = self.fetch(name, ne0 * ne1)?;
        let tensor = graph
            .new_tensor_2d_f32(ne0, ne1, "dolphin_dec_weight")
            .map_err(ggml_err("weight_alloc_2d"))?;
        self.uploads.push((tensor, data, "dolphin_dec_weight"));
        Ok(tensor)
    }

    /// The first `positions` rows of the `[1, max_positions, d_model]` absolute
    /// sinusoidal position table.
    fn pos_slice(
        &mut self,
        graph: &GgmlCpuGraphBuilder<'a>,
        name: &str,
        d_model: usize,
        positions: usize,
        max_positions: usize,
    ) -> Result<GgmlCpuTensor<'a>, DolphinDecoderError> {
        let full = self.fetch(name, d_model * max_positions)?;
        let slice = &full[..d_model * positions];
        let tensor = graph
            .new_tensor_2d_f32(d_model, positions, "dolphin_dec_weight")
            .map_err(ggml_err("weight_alloc_pos"))?;
        self.uploads.push((tensor, slice, "dolphin_dec_weight"));
        Ok(tensor)
    }
}

struct LinearWeights<'a> {
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
}

struct NormWeights<'a> {
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
}

struct DecoderLayerWeights<'a> {
    norm1: NormWeights<'a>,
    self_q: LinearWeights<'a>,
    self_k: LinearWeights<'a>,
    self_v: LinearWeights<'a>,
    self_o: LinearWeights<'a>,
    norm2: NormWeights<'a>,
    src_q: LinearWeights<'a>,
    src_k: LinearWeights<'a>,
    src_v: LinearWeights<'a>,
    src_o: LinearWeights<'a>,
    norm3: NormWeights<'a>,
    ff_w1: LinearWeights<'a>,
    ff_w2: LinearWeights<'a>,
}

struct DecoderWeights<'a> {
    token_embed: GgmlCpuTensor<'a>,
    pos_emb: GgmlCpuTensor<'a>,
    layers: Vec<DecoderLayerWeights<'a>>,
    after_norm: NormWeights<'a>,
    output_weight: GgmlCpuTensor<'a>,
    output_bias: GgmlCpuTensor<'a>,
}

fn build_linear_weights<'a, 'p>(
    graph: &GgmlCpuGraphBuilder<'a>,
    builder: &mut WeightBuilder<'a, 'p>,
    prefix: &str,
    d_in: usize,
    d_out: usize,
) -> Result<LinearWeights<'a>, DolphinDecoderError> {
    Ok(LinearWeights {
        weight: builder.w2(graph, &format!("{prefix}.weight"), d_in, d_out)?,
        bias: builder.w1(graph, &format!("{prefix}.bias"), d_out)?,
    })
}

fn build_norm_weights<'a, 'p>(
    graph: &GgmlCpuGraphBuilder<'a>,
    builder: &mut WeightBuilder<'a, 'p>,
    prefix: &str,
    d: usize,
) -> Result<NormWeights<'a>, DolphinDecoderError> {
    Ok(NormWeights {
        weight: builder.w1(graph, &format!("{prefix}.weight"), d)?,
        bias: builder.w1(graph, &format!("{prefix}.bias"), d)?,
    })
}

fn build_layer_weights<'a, 'p>(
    graph: &GgmlCpuGraphBuilder<'a>,
    builder: &mut WeightBuilder<'a, 'p>,
    config: &DolphinDecoderConfig,
    index: usize,
) -> Result<DecoderLayerWeights<'a>, DolphinDecoderError> {
    let d = config.d_model;
    let ffn = config.ffn_units;
    let p = |suffix: &str| format!("decoder.decoders.{index}.{suffix}");
    Ok(DecoderLayerWeights {
        norm1: build_norm_weights(graph, builder, &p("norm1"), d)?,
        self_q: build_linear_weights(graph, builder, &p("self_attn.linear_q"), d, d)?,
        self_k: build_linear_weights(graph, builder, &p("self_attn.linear_k"), d, d)?,
        self_v: build_linear_weights(graph, builder, &p("self_attn.linear_v"), d, d)?,
        self_o: build_linear_weights(graph, builder, &p("self_attn.linear_out"), d, d)?,
        norm2: build_norm_weights(graph, builder, &p("norm2"), d)?,
        src_q: build_linear_weights(graph, builder, &p("src_attn.linear_q"), d, d)?,
        src_k: build_linear_weights(graph, builder, &p("src_attn.linear_k"), d, d)?,
        src_v: build_linear_weights(graph, builder, &p("src_attn.linear_v"), d, d)?,
        src_o: build_linear_weights(graph, builder, &p("src_attn.linear_out"), d, d)?,
        norm3: build_norm_weights(graph, builder, &p("norm3"), d)?,
        ff_w1: build_linear_weights(graph, builder, &p("feed_forward.w_1"), d, ffn)?,
        ff_w2: build_linear_weights(graph, builder, &p("feed_forward.w_2"), ffn, d)?,
    })
}

// --- graph ops -------------------------------------------------------------

fn linear<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weights: &LinearWeights<'a>,
    input: GgmlCpuTensor<'a>,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, DolphinDecoderError> {
    let projected = graph
        .mul_mat(weights.weight, input)
        .map_err(ggml_err(stage))?;
    graph.add(projected, weights.bias).map_err(ggml_err(stage))
}

fn affine_ln<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    eps: f32,
    weights: &NormWeights<'a>,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, DolphinDecoderError> {
    apply_affine_layer_norm(
        graph,
        input,
        eps,
        weights.weight,
        weights.bias,
        AffineLayerNormSteps {
            norm: stage,
            scale: stage,
            bias: stage,
        },
        |s, source| DolphinDecoderError::Ggml { stage: s, source },
    )
}

fn reshape_heads<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    heads: usize,
    seq: usize,
) -> Result<GgmlCpuTensor<'a>, DolphinDecoderError> {
    reshape_projection_to_attention_heads(
        graph,
        projection,
        AttentionHeadLayout {
            head_dim,
            attention_heads: heads,
            sequence_len: seq,
        },
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        AttentionReshapeSteps {
            reshape: "attn_reshape",
            permute: "attn_permute",
            cont: "attn_cont",
        },
        |s, source| DolphinDecoderError::Ggml { stage: s, source },
    )
}

/// Scaled dot-product attention over head-major q/k/v (`[head_dim, seq, heads]`),
/// f32 throughout. `mask` is an additive `[kv_len, q_len]` bias applied inside the
/// softmax (self-attention causal mask; `None` for full-context cross-attention).
/// Returns the merged context `[d_model, q_len]`.
fn attention<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    q_heads: GgmlCpuTensor<'a>,
    k_heads: GgmlCpuTensor<'a>,
    v_heads: GgmlCpuTensor<'a>,
    mask: Option<GgmlCpuTensor<'a>>,
    config: &DolphinDecoderConfig,
    query_len: usize,
) -> Result<GgmlCpuTensor<'a>, DolphinDecoderError> {
    let map = ggml_err("attention");
    let scores = graph.mul_mat(k_heads, q_heads).map_err(map)?;
    let scale = 1.0 / (config.head_dim as f32).sqrt();
    let probs = graph.soft_max_ext(scores, mask, scale, 0.0).map_err(map)?;
    attention_context_from_probs(
        graph,
        v_heads,
        probs,
        AttentionHeadLayout {
            head_dim: config.head_dim,
            attention_heads: config.attention_heads,
            sequence_len: query_len,
        },
        AttentionValueMergeSteps {
            value_permute: "attn_v_t",
            value_cont: "attn_v_t",
            context_mul: "attn_ctx",
            context_merge_permute: "attn_merge",
            context_merge_cont: "attn_merge",
            context_merge_reshape: "attn_merge",
        },
        |s, source| DolphinDecoderError::Ggml { stage: s, source },
    )
}

#[allow(clippy::too_many_arguments)]
fn decoder_layer<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    encoder_out: GgmlCpuTensor<'a>,
    causal_mask: GgmlCpuTensor<'a>,
    weights: &DecoderLayerWeights<'a>,
    config: &DolphinDecoderConfig,
    tokens: usize,
    frames: usize,
) -> Result<GgmlCpuTensor<'a>, DolphinDecoderError> {
    let eps = config.layer_norm_epsilon;
    let hd = config.head_dim;
    let heads = config.attention_heads;
    let map = ggml_err("decoder_layer");

    // Self-attention (causal) sub-block: residual + self_attn(norm1(x)).
    let self_norm = affine_ln(graph, input, eps, &weights.norm1, "self_attn_norm")?;
    let q = linear(graph, &weights.self_q, self_norm, "self_attn_q")?;
    let k = linear(graph, &weights.self_k, self_norm, "self_attn_k")?;
    let v = linear(graph, &weights.self_v, self_norm, "self_attn_v")?;
    let q = reshape_heads(graph, q, hd, heads, tokens)?;
    let k = reshape_heads(graph, k, hd, heads, tokens)?;
    let v = reshape_heads(graph, v, hd, heads, tokens)?;
    let context = attention(graph, q, k, v, Some(causal_mask), config, tokens)?;
    let self_out = linear(graph, &weights.self_o, context, "self_attn_out")?;
    let x = graph.add(input, self_out).map_err(map)?;

    // Cross-attention sub-block: residual + src_attn(norm2(x)) over encoder_out.
    let cross_norm = affine_ln(graph, x, eps, &weights.norm2, "cross_attn_norm")?;
    let q = linear(graph, &weights.src_q, cross_norm, "cross_attn_q")?;
    let k = linear(graph, &weights.src_k, encoder_out, "cross_attn_k")?;
    let v = linear(graph, &weights.src_v, encoder_out, "cross_attn_v")?;
    let q = reshape_heads(graph, q, hd, heads, tokens)?;
    let k = reshape_heads(graph, k, hd, heads, frames)?;
    let v = reshape_heads(graph, v, hd, heads, frames)?;
    let context = attention(graph, q, k, v, None, config, tokens)?;
    let cross_out = linear(graph, &weights.src_o, context, "cross_attn_out")?;
    let x = graph.add(x, cross_out).map_err(map)?;

    // Feed-forward sub-block: residual + w_2(relu(w_1(norm3(x)))).
    let ff_norm = affine_ln(graph, x, eps, &weights.norm3, "ffn_norm")?;
    apply_feed_forward_residual(
        graph,
        ff_norm,
        x,
        FeedForwardActivation::Relu,
        None,
        FeedForwardResidualSteps {
            activation: "ffn_relu",
            scale: None,
            residual: "ffn_residual",
        },
        |graph, value| linear(graph, &weights.ff_w1, value, "ffn_up"),
        |graph, value| linear(graph, &weights.ff_w2, value, "ffn_down"),
        |s, source| DolphinDecoderError::Ggml { stage: s, source },
    )
}

/// Build a causal additive-bias mask `[kv=tokens, q=tokens]` (row-major, kv
/// innermost): `0.0` where the key position is `<=` the query position, `-inf`
/// otherwise.
fn build_causal_mask(tokens: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; tokens * tokens];
    for q in 0..tokens {
        for (k, cell) in mask[q * tokens..q * tokens + tokens].iter_mut().enumerate() {
            if k > q {
                *cell = f32::NEG_INFINITY;
            }
        }
    }
    mask
}

/// Run the Dolphin Transformer decoder over a teacher-forced prompt prefix and
/// return the per-position logits. `encoder_out` is the frame-major
/// `[frames, d_model]` encoder output (d_model innermost), matching the golden
/// `encoder_out` fixture layout.
pub(crate) fn decode_prompt_logits(
    config: &DolphinDecoderConfig,
    provider: &dyn DolphinWeightProvider,
    encoder_out: &[f32],
    frames: usize,
    prompt_tokens: &[u32],
    backend: GgmlCpuGraphBackend,
) -> Result<DolphinDecoderOutput, DolphinDecoderError> {
    let d = config.d_model;
    let tokens = prompt_tokens.len();
    if tokens == 0 {
        return Err(DolphinDecoderError::Shape {
            reason: "prompt must contain at least one token".to_string(),
        });
    }
    if tokens > config.max_positions {
        return Err(DolphinDecoderError::Shape {
            reason: format!(
                "prompt length {tokens} exceeds position table {}",
                config.max_positions
            ),
        });
    }
    if frames == 0 || encoder_out.len() != frames * d {
        return Err(DolphinDecoderError::Shape {
            reason: format!(
                "encoder_out has {} values, expected {frames}x{d}",
                encoder_out.len()
            ),
        });
    }
    if let Some(bad) = prompt_tokens
        .iter()
        .find(|&&t| t as usize >= config.vocab_size)
    {
        return Err(DolphinDecoderError::Shape {
            reason: format!(
                "prompt token {bad} out of vocab range {}",
                config.vocab_size
            ),
        });
    }

    let graph_config = GgmlCpuGraphConfig {
        context_bytes: 128 * 1024 * 1024,
        graph_size: 16384,
        n_threads: GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            backend,
            crate::ggml_runtime::GgmlCpuGraphThreadingWorkload::Decoder,
        ),
        backend,
        use_scheduler: backend.is_gpu_class(),
    };
    let mut runner = GgmlCpuGraphRunner::new(graph_config).map_err(ggml_err("runner_init"))?;
    let mut graph = runner.start_graph();

    // Phase A: create every weight tensor (must precede the first buffer alloc).
    let mut builder = WeightBuilder::new(provider);
    let token_embed = builder.w2(&graph, "decoder.embed.0.weight", d, config.vocab_size)?;
    let pos_emb = builder.pos_slice(
        &graph,
        "decoder.embed.1.pe",
        d,
        tokens,
        config.max_positions,
    )?;
    let mut layers = Vec::with_capacity(config.num_layers);
    for index in 0..config.num_layers {
        layers.push(build_layer_weights(&graph, &mut builder, config, index)?);
    }
    let after_norm = build_norm_weights(&graph, &mut builder, "decoder.after_norm", d)?;
    let output_weight = builder.w2(&graph, "decoder.output_layer.weight", d, config.vocab_size)?;
    let output_bias = builder.w1(&graph, "decoder.output_layer.bias", config.vocab_size)?;
    let weights = DecoderWeights {
        token_embed,
        pos_emb,
        layers,
        after_norm,
        output_weight,
        output_bias,
    };

    // Input tensors: token ids (i32), encoder memory (f32), causal mask (f32).
    let token_ids = graph
        .new_tensor_1d_i32(tokens, "dolphin_dec_tokens")
        .map_err(ggml_err("input_alloc_tokens"))?;
    let encoder_mem = graph
        .new_tensor_2d_f32(d, frames, "dolphin_dec_encoder_out")
        .map_err(ggml_err("input_alloc_encoder"))?;
    let causal_mask = graph
        .new_tensor_2d_f32(tokens, tokens, "dolphin_dec_causal_mask")
        .map_err(ggml_err("input_alloc_mask"))?;

    // Phase B: build the forward graph.
    // Embedding: token_embed(ids) * sqrt(d_model) + absolute positional encoding.
    let token_state = graph
        .get_rows(weights.token_embed, token_ids)
        .map_err(ggml_err("embed_get_rows"))?;
    let scaled = graph
        .scale(token_state, (d as f32).sqrt())
        .map_err(ggml_err("embed_xscale"))?;
    let mut hidden = graph
        .add(scaled, weights.pos_emb)
        .map_err(ggml_err("embed_pos"))?;
    for layer in &weights.layers {
        hidden = decoder_layer(
            &mut graph,
            hidden,
            encoder_mem,
            causal_mask,
            layer,
            config,
            tokens,
            frames,
        )?;
    }
    let normed = affine_ln(
        &graph,
        hidden,
        config.layer_norm_epsilon,
        &weights.after_norm,
        "after_norm",
    )?;
    let logits = graph
        .mul_mat(weights.output_weight, normed)
        .map_err(ggml_err("output_layer"))?;
    let logits = graph
        .add(logits, weights.output_bias)
        .map_err(ggml_err("output_layer_bias"))?;
    graph.set_output(logits).map_err(ggml_err("set_output"))?;

    // Phase C: upload inputs + weights, then compute.
    let token_ids_i32: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
    graph
        .set_i32_slice(token_ids, &token_ids_i32, "dolphin_dec_tokens")
        .map_err(ggml_err("upload_tokens"))?;
    graph
        .set_f32_slice(encoder_mem, encoder_out, "dolphin_dec_encoder_out")
        .map_err(ggml_err("upload_encoder"))?;
    graph
        .set_f32_slice(
            causal_mask,
            &build_causal_mask(tokens),
            "dolphin_dec_causal_mask",
        )
        .map_err(ggml_err("upload_mask"))?;
    for (tensor, data, name) in &builder.uploads {
        graph
            .set_f32_slice(*tensor, data, name)
            .map_err(ggml_err("upload_weight"))?;
    }

    let expected = tokens * config.vocab_size;
    let logits = graph
        .compute_output_f32(logits, expected)
        .map_err(ggml_err("compute"))?;

    Ok(DolphinDecoderOutput {
        token_count: tokens,
        vocab_size: config.vocab_size,
        logits,
    })
}
