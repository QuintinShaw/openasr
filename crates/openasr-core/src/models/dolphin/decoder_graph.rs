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
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlMatmulPrecision, GgmlStaticTensor,
    GgmlStaticTensorArena,
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

const DOLPHIN_DECODER_GRAPH_NODE_CAPACITY: usize = 16_384;

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

    /// Build the config from the pack's own parsed runtime metadata --
    /// checkpoint-size-agnostic. The importer's `derive_dolphin_architecture`
    /// asserts the decoder's own d_model equals the encoder's for every
    /// observed Dolphin checkpoint before it ever writes the pack, so this
    /// reuses `encoder_d_model` rather than tracking a redundant metadata key.
    /// `layer_norm_epsilon` stays fixed at `1e-5`, like `small_cn()`'s.
    pub(crate) fn from_execution_metadata(
        metadata: &super::runtime_contract::DolphinExecutionMetadata,
    ) -> Self {
        Self {
            d_model: metadata.encoder_d_model,
            attention_heads: metadata.decoder_n_heads,
            head_dim: metadata.encoder_head_dim,
            ffn_units: metadata.decoder_ffn_dim,
            num_layers: metadata.decoder_n_layers,
            vocab_size: metadata.vocab_size,
            max_positions: metadata.decoder_max_ctx,
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

// --- persistent weight arena -------------------------------------------------
//
// Perf (P5, then extended -- see joint_decode::attention_rescore):
// attention-rescoring teacher-forces up to DOLPHIN_BEAM_SIZE CTC n-best
// hypotheses against the same decoder weights and the same encoder memory;
// only the token ids / causal masks / sequence lengths differ. This mirrors
// the firered_aed decoder's `GgmlStaticTensorArena`-backed persistent weights
// (`decoder_weights.rs` / `decoder_graph.rs`): the ~200 weight tensors are
// declared and uploaded ONCE into a static arena by
// [`DolphinDecoderRescoreRuntime::new`], and the runtime itself is cached per
// `(pack, backend)` by the executor, so later utterances skip the upload
// entirely (the pre-cache behavior rebuilt runner + arena and re-uploaded
// every weight per utterance). Each
// [`DolphinDecoderRescoreRuntime::decode_nbest_prompt_logits`] call then
// builds ONE fused per-utterance graph: the encoder memory is a transient
// input uploaded once, the per-layer cross-attention K/V projections over it
// are shared graph nodes computed once, and every hypothesis contributes only
// its small chain (token embed lookup, absolute-position view, causal mask,
// stacked layers, output projection) referencing the resident weights and the
// shared K/V nodes. The forward math (`decoder_layer`/`linear`/`affine_ln`/
// etc. below) is untouched, so this is a pure execution-strategy change: same
// ops, same numbers, golden-identical output.

enum PendingUpload<'p> {
    F32(GgmlStaticTensor, &'p [f32], &'static str),
    Native(GgmlStaticTensor, &'p [u8], &'static str),
}

struct StaticWeightBuilder<'p> {
    provider: &'p dyn DolphinWeightProvider,
    uploads: Vec<PendingUpload<'p>>,
}

impl<'p> StaticWeightBuilder<'p> {
    fn new(provider: &'p dyn DolphinWeightProvider, tensor_count: usize) -> Self {
        Self {
            provider,
            uploads: Vec::with_capacity(tensor_count),
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
        arena: &GgmlStaticTensorArena,
        name: &str,
        len: usize,
    ) -> Result<GgmlStaticTensor, DolphinDecoderError> {
        let data = self.fetch(name, len)?;
        let tensor = arena
            .new_tensor_1d_f32(len, "dolphin_dec_weight")
            .map_err(ggml_err("weight_alloc_1d"))?;
        self.uploads
            .push(PendingUpload::F32(tensor, data, "dolphin_dec_weight"));
        Ok(tensor)
    }

    /// A 2-D `.weight` matmul operand bound as ggml `[ne0=in, ne1=out]` for
    /// `mul_mat(w, x)`. When the provider keeps this weight quantized/f16
    /// (`native_weight`) it is bound at its stored ggml type with the raw block
    /// bytes uploaded verbatim (stays quantized in the backend buffer, fed to
    /// `mul_mat`'s quantized-lhs path); otherwise it falls back to the f32 bind.
    /// NOT for the token embedding -- that is a `get_rows` operand and must stay
    /// f32 (see [`w2_embedding`]).
    fn w2(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        ne0: usize,
        ne1: usize,
    ) -> Result<GgmlStaticTensor, DolphinDecoderError> {
        if let Some(native) = self.provider.native_weight(name) {
            let tensor = arena
                .new_matmul_weight_2d_typed(ne0, ne1, native.ggml_type, "dolphin_dec_weight")
                .map_err(ggml_err("weight_alloc_2d_native"))?;
            self.uploads.push(PendingUpload::Native(
                tensor,
                native.bytes,
                "dolphin_dec_weight",
            ));
            return Ok(tensor);
        }
        let data = self.fetch(name, ne0 * ne1)?;
        let tensor = arena
            .new_tensor_2d_f32(ne0, ne1, "dolphin_dec_weight")
            .map_err(ggml_err("weight_alloc_2d"))?;
        self.uploads
            .push(PendingUpload::F32(tensor, data, "dolphin_dec_weight"));
        Ok(tensor)
    }

    /// The token embedding table `decoder.embed.0.weight`, always bound f32
    /// regardless of how the pack stored it: it is consumed by `ggml_get_rows`
    /// (row lookup), which only accepts f32/f16 embeddings, never a block-quant
    /// tensor. Keeping it f32 also matches the invariant that only rank-2 `.weight`
    /// *matmul* operands go native.
    fn w2_embedding(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        ne0: usize,
        ne1: usize,
    ) -> Result<GgmlStaticTensor, DolphinDecoderError> {
        let data = self.fetch(name, ne0 * ne1)?;
        let tensor = arena
            .new_tensor_2d_f32(ne0, ne1, "dolphin_dec_weight")
            .map_err(ggml_err("weight_alloc_2d"))?;
        self.uploads
            .push(PendingUpload::F32(tensor, data, "dolphin_dec_weight"));
        Ok(tensor)
    }

    /// The full `[1, max_positions, d_model]` absolute sinusoidal position
    /// table, uploaded once at its baked length. Per-call decode takes a
    /// contiguous leading view of the first `tokens` rows (see
    /// [`DolphinDecoderRescoreRuntime::decode_prompt_logits`]) instead of this
    /// builder re-slicing and re-uploading a `positions`-sized copy every call.
    fn pos_full(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        d_model: usize,
        max_positions: usize,
    ) -> Result<GgmlStaticTensor, DolphinDecoderError> {
        let data = self.fetch(name, d_model * max_positions)?;
        let tensor = arena
            .new_tensor_2d_f32(d_model, max_positions, "dolphin_dec_weight")
            .map_err(ggml_err("weight_alloc_pos"))?;
        self.uploads
            .push(PendingUpload::F32(tensor, data, "dolphin_dec_weight"));
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

/// Static (arena-resident) counterpart of [`LinearWeights`] / [`NormWeights`] /
/// [`DecoderLayerWeights`]: same shapes, but the tensors live in the runtime's
/// persistent [`GgmlStaticTensorArena`] instead of a per-call transient graph.
/// `to_transient` reborrows each field into the current call's graph lifetime
/// via [`GgmlStaticTensorArena::graph_tensor`] (mirrors `cohere::decoder_graph`).
#[derive(Clone, Copy)]
struct LinearStaticWeights {
    weight: GgmlStaticTensor,
    bias: GgmlStaticTensor,
}

impl LinearStaticWeights {
    fn to_transient<'a>(self, arena: &GgmlStaticTensorArena) -> LinearWeights<'a> {
        LinearWeights {
            weight: arena.graph_tensor(self.weight),
            bias: arena.graph_tensor(self.bias),
        }
    }
}

#[derive(Clone, Copy)]
struct NormStaticWeights {
    weight: GgmlStaticTensor,
    bias: GgmlStaticTensor,
}

impl NormStaticWeights {
    fn to_transient<'a>(self, arena: &GgmlStaticTensorArena) -> NormWeights<'a> {
        NormWeights {
            weight: arena.graph_tensor(self.weight),
            bias: arena.graph_tensor(self.bias),
        }
    }
}

#[derive(Clone, Copy)]
struct DecoderLayerStaticWeights {
    norm1: NormStaticWeights,
    self_q: LinearStaticWeights,
    self_k: LinearStaticWeights,
    self_v: LinearStaticWeights,
    self_o: LinearStaticWeights,
    norm2: NormStaticWeights,
    src_q: LinearStaticWeights,
    src_k: LinearStaticWeights,
    src_v: LinearStaticWeights,
    src_o: LinearStaticWeights,
    norm3: NormStaticWeights,
    ff_w1: LinearStaticWeights,
    ff_w2: LinearStaticWeights,
}

impl DecoderLayerStaticWeights {
    fn to_transient<'a>(self, arena: &GgmlStaticTensorArena) -> DecoderLayerWeights<'a> {
        DecoderLayerWeights {
            norm1: self.norm1.to_transient(arena),
            self_q: self.self_q.to_transient(arena),
            self_k: self.self_k.to_transient(arena),
            self_v: self.self_v.to_transient(arena),
            self_o: self.self_o.to_transient(arena),
            norm2: self.norm2.to_transient(arena),
            src_q: self.src_q.to_transient(arena),
            src_k: self.src_k.to_transient(arena),
            src_v: self.src_v.to_transient(arena),
            src_o: self.src_o.to_transient(arena),
            norm3: self.norm3.to_transient(arena),
            ff_w1: self.ff_w1.to_transient(arena),
            ff_w2: self.ff_w2.to_transient(arena),
        }
    }
}

struct DecoderStaticWeights {
    token_embed: GgmlStaticTensor,
    /// Full `[d_model, max_positions]` absolute position table (see
    /// [`StaticWeightBuilder::pos_full`]); sliced per call via a graph view.
    pos_emb_full: GgmlStaticTensor,
    layers: Vec<DecoderLayerStaticWeights>,
    after_norm: NormStaticWeights,
    output_weight: GgmlStaticTensor,
    output_bias: GgmlStaticTensor,
}

fn build_linear_static_weights(
    arena: &GgmlStaticTensorArena,
    builder: &mut StaticWeightBuilder<'_>,
    prefix: &str,
    d_in: usize,
    d_out: usize,
) -> Result<LinearStaticWeights, DolphinDecoderError> {
    Ok(LinearStaticWeights {
        weight: builder.w2(arena, &format!("{prefix}.weight"), d_in, d_out)?,
        bias: builder.w1(arena, &format!("{prefix}.bias"), d_out)?,
    })
}

fn build_norm_static_weights(
    arena: &GgmlStaticTensorArena,
    builder: &mut StaticWeightBuilder<'_>,
    prefix: &str,
    d: usize,
) -> Result<NormStaticWeights, DolphinDecoderError> {
    Ok(NormStaticWeights {
        weight: builder.w1(arena, &format!("{prefix}.weight"), d)?,
        bias: builder.w1(arena, &format!("{prefix}.bias"), d)?,
    })
}

fn build_layer_static_weights(
    arena: &GgmlStaticTensorArena,
    builder: &mut StaticWeightBuilder<'_>,
    config: &DolphinDecoderConfig,
    index: usize,
) -> Result<DecoderLayerStaticWeights, DolphinDecoderError> {
    let d = config.d_model;
    let ffn = config.ffn_units;
    let p = |suffix: &str| format!("decoder.decoders.{index}.{suffix}");
    Ok(DecoderLayerStaticWeights {
        norm1: build_norm_static_weights(arena, builder, &p("norm1"), d)?,
        self_q: build_linear_static_weights(arena, builder, &p("self_attn.linear_q"), d, d)?,
        self_k: build_linear_static_weights(arena, builder, &p("self_attn.linear_k"), d, d)?,
        self_v: build_linear_static_weights(arena, builder, &p("self_attn.linear_v"), d, d)?,
        self_o: build_linear_static_weights(arena, builder, &p("self_attn.linear_out"), d, d)?,
        norm2: build_norm_static_weights(arena, builder, &p("norm2"), d)?,
        src_q: build_linear_static_weights(arena, builder, &p("src_attn.linear_q"), d, d)?,
        src_k: build_linear_static_weights(arena, builder, &p("src_attn.linear_k"), d, d)?,
        src_v: build_linear_static_weights(arena, builder, &p("src_attn.linear_v"), d, d)?,
        src_o: build_linear_static_weights(arena, builder, &p("src_attn.linear_out"), d, d)?,
        norm3: build_norm_static_weights(arena, builder, &p("norm3"), d)?,
        ff_w1: build_linear_static_weights(arena, builder, &p("feed_forward.w_1"), d, ffn)?,
        ff_w2: build_linear_static_weights(arena, builder, &p("feed_forward.w_2"), ffn, d)?,
    })
}

/// Static-arena tensor count for [`GgmlCpuGraphConfig::metadata_context_bytes`]:
/// per layer, 13 `(weight, bias)`-style pairs (norm1/self_q/self_k/self_v/
/// self_o/norm2/src_q/src_k/src_v/src_o/norm3/ff_w1/ff_w2) = 26 tensors, plus
/// the fixed token embedding, full position table, after_norm (2), output
/// weight and output bias. The encoder memory is no longer arena-resident --
/// it is a per-call transient graph input of [`DolphinDecoderRescoreRuntime::
/// decode_nbest_prompt_logits`], uploaded once per utterance.
const DOLPHIN_DECODER_ARENA_TENSORS_PER_LAYER: usize = 26;
const DOLPHIN_DECODER_ARENA_FIXED_TENSORS: usize = 6;

fn dolphin_decoder_arena_context_bytes(num_layers: usize) -> usize {
    let tensor_count = DOLPHIN_DECODER_ARENA_FIXED_TENSORS
        .saturating_add(DOLPHIN_DECODER_ARENA_TENSORS_PER_LAYER.saturating_mul(num_layers));
    GgmlCpuGraphConfig::metadata_context_bytes(tensor_count)
}

// --- graph ops -------------------------------------------------------------

fn linear<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weights: &LinearWeights<'a>,
    input: GgmlCpuTensor<'a>,
    precision: GgmlMatmulPrecision,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, DolphinDecoderError> {
    let projected = graph
        .mul_mat_with_precision(weights.weight, input, precision)
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

/// One layer's cross-attention K/V heads, projected from the encoder memory.
/// These depend only on the encoder output (identical for every hypothesis of
/// one utterance), so [`DolphinDecoderRescoreRuntime::decode_nbest_prompt_logits`]
/// computes them once per utterance as shared graph nodes and every
/// per-hypothesis chain consumes the same nodes -- the exact ops (same inputs,
/// same order) each per-hypothesis graph used to run redundantly, so the
/// values are unchanged; they are just no longer recomputed `beam_size` times.
#[derive(Clone, Copy)]
struct CrossAttentionHeads<'a> {
    k_heads: GgmlCpuTensor<'a>,
    v_heads: GgmlCpuTensor<'a>,
}

fn build_cross_attention_heads<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    encoder_out: GgmlCpuTensor<'a>,
    weights: &DecoderLayerWeights<'a>,
    config: &DolphinDecoderConfig,
    precision: GgmlMatmulPrecision,
    frames: usize,
) -> Result<CrossAttentionHeads<'a>, DolphinDecoderError> {
    let hd = config.head_dim;
    let heads = config.attention_heads;
    let k = linear(
        graph,
        &weights.src_k,
        encoder_out,
        precision,
        "cross_attn_k",
    )?;
    let v = linear(
        graph,
        &weights.src_v,
        encoder_out,
        precision,
        "cross_attn_v",
    )?;
    Ok(CrossAttentionHeads {
        k_heads: reshape_heads(graph, k, hd, heads, frames)?,
        v_heads: reshape_heads(graph, v, hd, heads, frames)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn decoder_layer<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    cross: CrossAttentionHeads<'a>,
    causal_mask: GgmlCpuTensor<'a>,
    weights: &DecoderLayerWeights<'a>,
    config: &DolphinDecoderConfig,
    precision: GgmlMatmulPrecision,
    tokens: usize,
) -> Result<GgmlCpuTensor<'a>, DolphinDecoderError> {
    let eps = config.layer_norm_epsilon;
    let hd = config.head_dim;
    let heads = config.attention_heads;
    let map = ggml_err("decoder_layer");

    // Self-attention (causal) sub-block: residual + self_attn(norm1(x)).
    let self_norm = affine_ln(graph, input, eps, &weights.norm1, "self_attn_norm")?;
    let q = linear(graph, &weights.self_q, self_norm, precision, "self_attn_q")?;
    let k = linear(graph, &weights.self_k, self_norm, precision, "self_attn_k")?;
    let v = linear(graph, &weights.self_v, self_norm, precision, "self_attn_v")?;
    let q = reshape_heads(graph, q, hd, heads, tokens)?;
    let k = reshape_heads(graph, k, hd, heads, tokens)?;
    let v = reshape_heads(graph, v, hd, heads, tokens)?;
    let context = attention(graph, q, k, v, Some(causal_mask), config, tokens)?;
    let self_out = linear(graph, &weights.self_o, context, precision, "self_attn_out")?;
    let x = graph.add(input, self_out).map_err(map)?;

    // Cross-attention sub-block: residual + src_attn(norm2(x)) over the
    // shared, per-utterance K/V heads (see [`CrossAttentionHeads`]).
    let cross_norm = affine_ln(graph, x, eps, &weights.norm2, "cross_attn_norm")?;
    let q = linear(graph, &weights.src_q, cross_norm, precision, "cross_attn_q")?;
    let q = reshape_heads(graph, q, hd, heads, tokens)?;
    let context = attention(graph, q, cross.k_heads, cross.v_heads, None, config, tokens)?;
    let cross_out = linear(graph, &weights.src_o, context, precision, "cross_attn_out")?;
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
        |graph, value| linear(graph, &weights.ff_w1, value, precision, "ffn_up"),
        |graph, value| linear(graph, &weights.ff_w2, value, precision, "ffn_down"),
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

fn dolphin_decoder_runner_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    crate::models::graph_runtime_config::apply_request_execution_placement(GgmlCpuGraphConfig {
        context_bytes: GgmlCpuGraphConfig::metadata_context_bytes(
            DOLPHIN_DECODER_GRAPH_NODE_CAPACITY,
        ),
        graph_size: DOLPHIN_DECODER_GRAPH_NODE_CAPACITY,
        n_threads: GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            backend,
            crate::ggml_runtime::GgmlCpuGraphThreadingWorkload::Decoder,
        ),
        backend,
        // CPU/unscoped callers retain the bounded gallocr scheduler. The
        // active policy placement wins last so FullDevice is direct and the
        // qualified Vulkan split remains scheduler-backed Hybrid.
        use_scheduler: true,
    })
}

/// Build-once/run-many Dolphin decoder runtime for attention rescoring.
/// [`Self::new`] loads every decoder weight tensor into a persistent
/// [`GgmlStaticTensorArena`] exactly once; the runtime is then reusable
/// across utterances (the executor caches it per `(pack, backend)`), and
/// [`Self::decode_nbest_prompt_logits`] teacher-forces a whole CTC n-best in
/// a single fused graph per utterance. Owns a dedicated
/// [`GgmlCpuGraphRunner`] (rather than sharing the caller's) so the arena's
/// backend buffer and the per-call transient graphs agree on the same backend
/// instance.
pub(crate) struct DolphinDecoderRescoreRuntime {
    runner: GgmlCpuGraphRunner,
    arena: GgmlStaticTensorArena,
    config: DolphinDecoderConfig,
    weights: DecoderStaticWeights,
    matmul_precision: GgmlMatmulPrecision,
}

impl DolphinDecoderRescoreRuntime {
    pub(crate) fn quoted_system_memory_bytes(
        config: &DolphinDecoderConfig,
    ) -> Result<(u64, u64), String> {
        let retained = config
            .num_layers
            .checked_mul(std::mem::size_of::<DecoderLayerStaticWeights>())
            .ok_or_else(|| "dolphin decoder layer handle quote overflowed".to_string())?;
        let upload_count = DOLPHIN_DECODER_ARENA_TENSORS_PER_LAYER
            .checked_mul(config.num_layers)
            .and_then(|layers| layers.checked_add(DOLPHIN_DECODER_ARENA_FIXED_TENSORS))
            .ok_or_else(|| "dolphin decoder upload count overflowed".to_string())?;
        let upload_descriptors = upload_count
            .checked_mul(std::mem::size_of::<PendingUpload<'static>>())
            .ok_or_else(|| "dolphin decoder upload descriptor quote overflowed".to_string())?;
        let peak = retained
            .checked_add(upload_descriptors)
            .ok_or_else(|| "dolphin decoder construction quote overflowed".to_string())?;
        Ok((
            u64::try_from(peak)
                .map_err(|_| "dolphin decoder peak quote exceeds u64".to_string())?,
            u64::try_from(retained)
                .map_err(|_| "dolphin decoder retained quote exceeds u64".to_string())?,
        ))
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add_vec(&self.weights.layers, "dolphin decoder layer handles")?;
        Ok(bytes.finish())
    }

    pub(crate) fn new(
        config: &DolphinDecoderConfig,
        provider: &dyn DolphinWeightProvider,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, DolphinDecoderError> {
        Self::new_with_matmul_precision(config, provider, backend, GgmlMatmulPrecision::Default)
    }

    pub(crate) fn new_with_matmul_precision(
        config: &DolphinDecoderConfig,
        provider: &dyn DolphinWeightProvider,
        backend: GgmlCpuGraphBackend,
        matmul_precision: GgmlMatmulPrecision,
    ) -> Result<Self, DolphinDecoderError> {
        let d = config.d_model;
        let runner = GgmlCpuGraphRunner::new(dolphin_decoder_runner_config(backend))
            .map_err(ggml_err("runner_init"))?;
        let arena = runner
            .start_static_tensor_arena(dolphin_decoder_arena_context_bytes(config.num_layers))
            .map_err(ggml_err("static_tensor_arena"))?;

        // Phase A: create every weight tensor (must precede the arena's first
        // buffer alloc, which freezes further creation).
        let upload_count = DOLPHIN_DECODER_ARENA_TENSORS_PER_LAYER
            .saturating_mul(config.num_layers)
            .saturating_add(DOLPHIN_DECODER_ARENA_FIXED_TENSORS);
        let mut builder = StaticWeightBuilder::new(provider, upload_count);
        let token_embed =
            builder.w2_embedding(&arena, "decoder.embed.0.weight", d, config.vocab_size)?;
        let pos_emb_full =
            builder.pos_full(&arena, "decoder.embed.1.pe", d, config.max_positions)?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            layers.push(build_layer_static_weights(
                &arena,
                &mut builder,
                config,
                index,
            )?);
        }
        let after_norm = build_norm_static_weights(&arena, &mut builder, "decoder.after_norm", d)?;
        let output_weight =
            builder.w2(&arena, "decoder.output_layer.weight", d, config.vocab_size)?;
        let output_bias = builder.w1(&arena, "decoder.output_layer.bias", config.vocab_size)?;

        // Phase B: upload every weight exactly once.
        let mut arena = arena;
        for upload in &builder.uploads {
            match upload {
                PendingUpload::F32(tensor, data, name) => arena
                    .set_f32_slice(*tensor, data, name)
                    .map_err(ggml_err("upload_weight"))?,
                PendingUpload::Native(tensor, bytes, name) => arena
                    .set_bytes_slice(*tensor, bytes, name)
                    .map_err(ggml_err("upload_weight_native"))?,
            }
        }

        Ok(Self {
            runner,
            arena,
            config: *config,
            weights: DecoderStaticWeights {
                token_embed,
                pos_emb_full,
                layers,
                after_norm,
                output_weight,
                output_bias,
            },
            matmul_precision,
        })
    }

    pub(crate) fn config(&self) -> &DolphinDecoderConfig {
        &self.config
    }

    /// Teacher-force every prompt of one utterance's n-best in a single fused
    /// graph and return each prompt's per-position logits (index-aligned with
    /// `prompts`).
    ///
    /// `encoder_out` is the frame-major `[frames, d_model]` encoder output
    /// (d_model innermost), uploaded once per call as a transient graph input
    /// and shared by every prompt chain. Per decoder layer the cross-attention
    /// K/V heads are projected from it once as shared graph nodes (they do not
    /// depend on the prompt -- see [`CrossAttentionHeads`]); each prompt then
    /// contributes its own chain (token embed, absolute-position view, causal
    /// mask, self-attention, FFN, output projection) referencing those shared
    /// nodes. Compared to the previous one-graph-per-hypothesis loop this
    /// removes `beam_size - 1` redundant cross-K/V projections (the dominant
    /// FLOPs: `frames >> tokens`) and `beam_size - 1` graph build + dispatch +
    /// readback round-trips, while running the exact same ops on the exact
    /// same values per prompt -- the logits are unchanged.
    pub(crate) fn decode_nbest_prompt_logits(
        &mut self,
        encoder_out: &[f32],
        frames: usize,
        prompts: &[Vec<u32>],
    ) -> Result<Vec<DolphinDecoderOutput>, DolphinDecoderError> {
        let config = &self.config;
        let matmul_precision = self.matmul_precision;
        let d = config.d_model;
        if frames == 0 || encoder_out.len() != frames * d {
            return Err(DolphinDecoderError::Shape {
                reason: format!(
                    "encoder_out has {} values, expected {frames}x{d}",
                    encoder_out.len()
                ),
            });
        }
        if prompts.is_empty() {
            return Ok(Vec::new());
        }
        for prompt_tokens in prompts {
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
        }

        let arena = &self.arena;
        let mut graph = self.runner.start_graph();

        // Transient inputs: the encoder memory (one per utterance) plus each
        // prompt's token ids (i32) and causal mask (f32). Weights are
        // arena-resident (real backend buffer already allocated), so unlike
        // the transient inputs they need no `set_input`.
        let encoder_mem = graph
            .new_tensor_2d_f32(d, frames, "dolphin_dec_encoder_out")
            .map_err(ggml_err("input_alloc_encoder"))?;

        // Shared per-layer cross-attention K/V heads, projected once from the
        // encoder memory and consumed by every prompt chain below.
        let layer_transients: Vec<DecoderLayerWeights<'_>> = self
            .weights
            .layers
            .iter()
            .map(|layer| layer.to_transient(arena))
            .collect();
        let mut cross_heads = Vec::with_capacity(layer_transients.len());
        for layer in &layer_transients {
            cross_heads.push(build_cross_attention_heads(
                &graph,
                encoder_mem,
                layer,
                config,
                matmul_precision,
                frames,
            )?);
        }

        let after_norm = self.weights.after_norm.to_transient(arena);
        let row_stride = d * std::mem::size_of::<f32>();
        let mut prompt_inputs = Vec::with_capacity(prompts.len());
        let mut logits_outputs = Vec::with_capacity(prompts.len());
        for prompt_tokens in prompts {
            let tokens = prompt_tokens.len();
            let token_ids = graph
                .new_tensor_1d_i32(tokens, "dolphin_dec_tokens")
                .map_err(ggml_err("input_alloc_tokens"))?;
            let causal_mask = graph
                .new_tensor_2d_f32(tokens, tokens, "dolphin_dec_causal_mask")
                .map_err(ggml_err("input_alloc_mask"))?;

            // Embedding: token_embed(ids) * sqrt(d_model) + absolute positional
            // encoding, the latter a contiguous leading view (rows `0..tokens`)
            // of the arena's full position table -- no re-upload, no re-slice.
            let token_state = graph
                .get_rows(arena.graph_tensor(self.weights.token_embed), token_ids)
                .map_err(ggml_err("embed_get_rows"))?;
            let scaled = graph
                .scale(token_state, (d as f32).sqrt())
                .map_err(ggml_err("embed_xscale"))?;
            let pos_view = graph
                .view_2d(
                    arena.graph_tensor(self.weights.pos_emb_full),
                    d,
                    tokens,
                    row_stride,
                    0,
                )
                .map_err(ggml_err("embed_pos_view"))?;
            let mut hidden = graph.add(scaled, pos_view).map_err(ggml_err("embed_pos"))?;
            for (layer, cross) in layer_transients.iter().zip(&cross_heads) {
                hidden = decoder_layer(
                    &mut graph,
                    hidden,
                    *cross,
                    causal_mask,
                    layer,
                    config,
                    matmul_precision,
                    tokens,
                )?;
            }
            let normed = affine_ln(
                &graph,
                hidden,
                config.layer_norm_epsilon,
                &after_norm,
                "after_norm",
            )?;
            let logits = graph
                .mul_mat_with_precision(
                    arena.graph_tensor(self.weights.output_weight),
                    normed,
                    matmul_precision,
                )
                .map_err(ggml_err("output_layer"))?;
            let logits = graph
                .add(logits, arena.graph_tensor(self.weights.output_bias))
                .map_err(ggml_err("output_layer_bias"))?;
            graph.set_output(logits).map_err(ggml_err("set_output"))?;
            prompt_inputs.push((token_ids, causal_mask));
            logits_outputs.push(logits);
        }

        graph
            .set_input(encoder_mem)
            .map_err(ggml_err("mark_input(encoder_out)"))?;
        for (token_ids, causal_mask) in &prompt_inputs {
            graph
                .set_input(*token_ids)
                .map_err(ggml_err("mark_input(tokens)"))?;
            graph
                .set_input(*causal_mask)
                .map_err(ggml_err("mark_input(causal_mask)"))?;
        }
        // Allocate the forward graph through the scheduler's gallocr for
        // liveness-based buffer reuse before uploading inputs (mirrors the
        // encoder and the sibling cohere/moonshine decoders).
        graph
            .prepare_outputs_for_upload(&logits_outputs)
            .map_err(ggml_err("prepare_outputs"))?;

        graph
            .set_f32_slice(encoder_mem, encoder_out, "dolphin_dec_encoder_out")
            .map_err(ggml_err("upload_encoder"))?;
        for (prompt_tokens, (token_ids, causal_mask)) in prompts.iter().zip(&prompt_inputs) {
            let token_ids_i32: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
            graph
                .set_i32_slice(*token_ids, &token_ids_i32, "dolphin_dec_tokens")
                .map_err(ggml_err("upload_tokens"))?;
            graph
                .set_f32_slice(
                    *causal_mask,
                    &build_causal_mask(prompt_tokens.len()),
                    "dolphin_dec_causal_mask",
                )
                .map_err(ggml_err("upload_mask"))?;
        }

        let output_specs: Vec<(GgmlCpuTensor, usize)> = logits_outputs
            .iter()
            .zip(prompts)
            .map(|(logits, prompt_tokens)| (*logits, prompt_tokens.len() * config.vocab_size))
            .collect();
        let outputs = graph
            .compute_outputs_f32(&output_specs)
            .map_err(ggml_err("compute"))?;

        Ok(outputs
            .into_iter()
            .zip(prompts)
            .map(|(logits, prompt_tokens)| DolphinDecoderOutput {
                token_count: prompt_tokens.len(),
                vocab_size: config.vocab_size,
                logits,
            })
            .collect())
    }
}

/// One-shot convenience wrapper over [`DolphinDecoderRescoreRuntime`] for
/// callers that only need a single prompt's logits (the `parity` dev
/// harness). Attention-rescoring reuses the executor-cached runtime and
/// scores its whole CTC n-best in one fused call (see
/// `joint_decode::attention_rescore`) instead of calling this per hypothesis.
pub(crate) fn decode_prompt_logits(
    config: &DolphinDecoderConfig,
    provider: &dyn DolphinWeightProvider,
    encoder_out: &[f32],
    frames: usize,
    prompt_tokens: &[u32],
    backend: GgmlCpuGraphBackend,
) -> Result<DolphinDecoderOutput, DolphinDecoderError> {
    let mut outputs = DolphinDecoderRescoreRuntime::new(config, provider, backend)?
        .decode_nbest_prompt_logits(encoder_out, frames, &[prompt_tokens.to_vec()])?;
    Ok(outputs.pop().expect("single prompt yields single output"))
}

#[cfg(test)]
mod tests {
    use super::super::encoder_graph::synthetic_test_tensor;
    use super::*;
    use std::collections::HashMap;

    /// A tiny but structurally complete decoder config (2 layers, d_model 8)
    /// for the CPU fused-rescore equivalence tests below.
    fn tiny_config() -> DolphinDecoderConfig {
        DolphinDecoderConfig {
            d_model: 8,
            attention_heads: 2,
            head_dim: 4,
            ffn_units: 16,
            num_layers: 2,
            vocab_size: 12,
            max_positions: 16,
            layer_norm_epsilon: 1e-5,
        }
    }

    /// Every decoder tensor name at its exact expected length, filled with
    /// deterministic synthetic values.
    fn synthetic_provider(config: &DolphinDecoderConfig) -> HashMap<String, Vec<f32>> {
        let d = config.d_model;
        let ffn = config.ffn_units;
        let vocab = config.vocab_size;
        let mut map = HashMap::new();
        let mut put = |name: String, len: usize| {
            let values = synthetic_test_tensor(&name, len);
            map.insert(name, values);
        };
        put("decoder.embed.0.weight".into(), d * vocab);
        put("decoder.embed.1.pe".into(), d * config.max_positions);
        for index in 0..config.num_layers {
            let p = |suffix: &str| format!("decoder.decoders.{index}.{suffix}");
            for norm in ["norm1", "norm2", "norm3"] {
                put(p(&format!("{norm}.weight")), d);
                put(p(&format!("{norm}.bias")), d);
            }
            for attn in ["self_attn", "src_attn"] {
                for proj in ["linear_q", "linear_k", "linear_v", "linear_out"] {
                    put(p(&format!("{attn}.{proj}.weight")), d * d);
                    put(p(&format!("{attn}.{proj}.bias")), d);
                }
            }
            put(p("feed_forward.w_1.weight"), d * ffn);
            put(p("feed_forward.w_1.bias"), ffn);
            put(p("feed_forward.w_2.weight"), ffn * d);
            put(p("feed_forward.w_2.bias"), d);
        }
        put("decoder.after_norm.weight".into(), d);
        put("decoder.after_norm.bias".into(), d);
        put("decoder.output_layer.weight".into(), d * vocab);
        put("decoder.output_layer.bias".into(), vocab);
        map
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// The fused-rescore invariant the perf change rests on: teacher-forcing a
    /// whole n-best in ONE fused graph (shared encoder memory upload, shared
    /// per-layer cross-K/V nodes) must produce bit-identical per-prompt logits
    /// to decoding each prompt independently in its own graph -- scores and
    /// hypothesis selection cannot move. Also pins runtime reuse: a second
    /// fused call on the same (weight-resident) runtime is bit-identical, and
    /// batch composition does not leak between prompts (a different n-best
    /// containing the same prompt yields the same logits for it).
    #[test]
    fn fused_nbest_logits_match_independent_per_prompt_decode_bit_for_bit() {
        let config = tiny_config();
        let provider = synthetic_provider(&config);
        let frames = 5;
        let encoder_out = synthetic_test_tensor("encoder_out", frames * config.d_model);
        // Distinct lengths on purpose: the fused graph must keep each prompt's
        // own causal mask / position view / logits shape.
        let prompts: Vec<Vec<u32>> = vec![
            vec![2, 5, 10, 4, 9],
            vec![2, 5, 10],
            vec![2, 5, 10, 4, 9, 1, 7, 3],
            vec![11],
        ];

        let mut runtime =
            DolphinDecoderRescoreRuntime::new(&config, &provider, GgmlCpuGraphBackend::Cpu)
                .expect("runtime");
        let fused = runtime
            .decode_nbest_prompt_logits(&encoder_out, frames, &prompts)
            .expect("fused decode");
        assert_eq!(fused.len(), prompts.len());

        for (prompt, fused_output) in prompts.iter().zip(&fused) {
            assert_eq!(fused_output.token_count, prompt.len());
            let single = decode_prompt_logits(
                &config,
                &provider,
                &encoder_out,
                frames,
                prompt,
                GgmlCpuGraphBackend::Cpu,
            )
            .expect("single decode");
            assert_eq!(
                bits(&fused_output.logits),
                bits(&single.logits),
                "fused n-best logits diverged from the independent decode for prompt {prompt:?}"
            );
        }

        // Runtime reuse: the second fused call re-uploads nothing but inputs.
        let again = runtime
            .decode_nbest_prompt_logits(&encoder_out, frames, &prompts)
            .expect("fused decode #2");
        for (first, second) in fused.iter().zip(&again) {
            assert_eq!(bits(&first.logits), bits(&second.logits));
        }

        // Batch-composition independence: the same prompt inside a different
        // n-best must score identically.
        let subset = vec![prompts[1].clone()];
        let alone = runtime
            .decode_nbest_prompt_logits(&encoder_out, frames, &subset)
            .expect("subset decode");
        assert_eq!(bits(&alone[0].logits), bits(&fused[1].logits));
    }

    /// Fail-closed validation of the fused entry point: empty prompt, over-long
    /// prompt, out-of-vocab token, and a mismatched encoder buffer must all be
    /// typed errors (never a partial compute), and an empty n-best is a no-op.
    #[test]
    fn fused_nbest_validates_prompts_and_encoder_shape() {
        let config = tiny_config();
        let provider = synthetic_provider(&config);
        let frames = 4;
        let encoder_out = synthetic_test_tensor("encoder_out", frames * config.d_model);
        let mut runtime =
            DolphinDecoderRescoreRuntime::new(&config, &provider, GgmlCpuGraphBackend::Cpu)
                .expect("runtime");

        assert!(
            runtime
                .decode_nbest_prompt_logits(&encoder_out, frames, &[])
                .expect("empty n-best")
                .is_empty()
        );
        for bad in [
            vec![],                            // empty prompt
            vec![1; config.max_positions + 1], // beyond position table
            vec![config.vocab_size as u32],    // out of vocab
        ] {
            let result = runtime.decode_nbest_prompt_logits(
                &encoder_out,
                frames,
                std::slice::from_ref(&bad),
            );
            assert!(
                matches!(result, Err(DolphinDecoderError::Shape { .. })),
                "prompt {bad:?} must fail closed, got {result:?}"
            );
        }
        let result = runtime.decode_nbest_prompt_logits(&encoder_out, frames - 1, &[vec![1]]);
        assert!(matches!(result, Err(DolphinDecoderError::Shape { .. })));
    }
}
