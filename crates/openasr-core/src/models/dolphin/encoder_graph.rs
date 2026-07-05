//! Dolphin `small.cn` E-Branchformer encoder graph (WeNet format).
//!
//! Self-contained ggml graph assembler for the Dolphin dialect encoder. It
//! reuses the shared `nn/` building blocks (affine layer norm, attention head
//! reshapes, feed-forward residual) but keeps every family-specific tensor
//! wiring here so nothing in the shared layers has to grow a Dolphin special
//! case.
//!
//! Architecture (verified against the 862-tensor `small.cn.pt` state dict, char
//! tokenizer, `use_sdpa=true`, `causal=false`):
//!   Conv2dSubsampling4 -> `* sqrt(d_model)` -> 12 x E-Branchformer block ->
//!   final LayerNorm.
//! Each block: macaron FFN (half-step) + rel-pos MHSA (no `rel_shift`, pos_emb
//! length == T because `use_sdpa` folds the bias directly) as the global branch,
//! a cgMLP/CSGU local branch, a depthwise merge conv, a final FFN, and per-branch
//! norms. Swish/SiLU FFN activation, GELU (erf) cgMLP projection, identity CSGU
//! gate, LayerNorm eps 1e-5.
//!
//! WIP: this is the numeric core validated by the `parity` dev harness; the
//! executor/frontend wiring lands separately, so the public surface is dead in a
//! plain lib build until then.
#![allow(dead_code)]

use std::collections::HashMap;

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

const F32_BYTES: usize = std::mem::size_of::<f32>();

#[derive(Debug, thiserror::Error)]
pub(crate) enum DolphinEncoderError {
    #[error("dolphin encoder shape error: {reason}")]
    Shape { reason: String },
    #[error("dolphin encoder missing weight tensor '{name}'")]
    MissingWeight { name: String },
    #[error("dolphin encoder weight '{name}' has {actual} values, expected {expected}")]
    WeightLen {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("dolphin encoder GGML backend failed at {stage}: {source}")]
    Ggml {
        stage: &'static str,
        source: GgmlCpuGraphError,
    },
}

fn ggml_err(stage: &'static str) -> impl Fn(GgmlCpuGraphError) -> DolphinEncoderError + Copy {
    move |source| DolphinEncoderError::Ggml { stage, source }
}

/// Scalar/shape configuration for the Dolphin `small.cn` encoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DolphinEncoderConfig {
    pub d_model: usize,
    pub attention_heads: usize,
    pub head_dim: usize,
    pub ffn_units: usize,
    pub cgmlp_units: usize,
    pub cgmlp_kernel: usize,
    pub merge_kernel: usize,
    pub num_blocks: usize,
    pub feature_dim: usize,
    pub layer_norm_epsilon: f32,
}

impl DolphinEncoderConfig {
    pub(crate) fn small_cn() -> Self {
        Self {
            d_model: 768,
            attention_heads: 12,
            head_dim: 64,
            ffn_units: 3072,
            cgmlp_units: 3072,
            cgmlp_kernel: 31,
            merge_kernel: 31,
            num_blocks: 12,
            feature_dim: 80,
            layer_norm_epsilon: 1e-5,
        }
    }
}

/// Full encoder result plus per-stage taps for parity gating.
#[derive(Debug, Clone)]
pub(crate) struct DolphinEncoderOutput {
    pub frames: usize,
    pub dim: usize,
    /// Frame-major `[frames, dim]` output of `Conv2dSubsampling4 * sqrt(d_model)`
    /// (the hidden entering block 0).
    pub after_subsample: Vec<f32>,
    /// Frame-major `[frames, dim]` output after each block's `norm_final`.
    pub blocks: Vec<Vec<f32>>,
    /// Frame-major `[frames, dim]` encoder output = `after_norm(block_last)`.
    pub encoder_out: Vec<f32>,
}

/// A rank-2 `.weight` matmul operand served in its native ggml block layout
/// (quantized q8_0/q4_k or f16) instead of dequantized f32. `bytes` are the raw
/// ggml row-major blocks straight from the pack mmap (no dequant); the graph
/// binds them at `ggml_type` so the weight stays quantized in the backend buffer
/// and is fed directly to `mul_mat` (which whitelists these lhs types).
#[derive(Clone, Copy)]
pub(crate) struct DolphinNativeWeight<'a> {
    pub ggml_type: i32,
    pub bytes: &'a [u8],
}

/// Weight source keyed by the WeNet `encoder.*`/`decoder.*`/`ctc.*` tensor name.
pub(crate) trait DolphinWeightProvider {
    /// Dequantized (or raw-f32) view: 1-D vectors, convs, position tables, the
    /// decoder token embedding (get_rows), CMVN, and any rank-2 weight the
    /// provider keeps in f32.
    fn tensor(&self, name: &str) -> Option<&[f32]>;

    /// Native (quantized / f16) block bytes of a rank-2 `.weight` matmul operand,
    /// when the provider keeps it quantized. Default `None` means every tensor is
    /// served as f32 (the raw-safetensors parity provider), so the graph binds
    /// f32xf32 and stays bit-exact.
    fn native_weight(&self, _name: &str) -> Option<DolphinNativeWeight<'_>> {
        None
    }
}

impl DolphinWeightProvider for HashMap<String, Vec<f32>> {
    fn tensor(&self, name: &str) -> Option<&[f32]> {
        self.get(name).map(Vec::as_slice)
    }
}

/// Subsampled frame count after two `k3 s2` conv layers (4x time downsample).
fn subsample_len(frames: usize) -> usize {
    let after_first = (frames.saturating_sub(3)) / 2 + 1;
    (after_first.saturating_sub(3)) / 2 + 1
}

/// Subsampled feature width after the same two conv layers on the mel axis.
fn subsample_width(features: usize) -> usize {
    let after_first = (features.saturating_sub(3)) / 2 + 1;
    (after_first.saturating_sub(3)) / 2 + 1
}

// --- weight tensor handles -------------------------------------------------

struct EmbedWeights<'a> {
    conv0_w: GgmlCpuTensor<'a>,
    conv0_b: GgmlCpuTensor<'a>,
    conv1_w: GgmlCpuTensor<'a>,
    conv1_b: GgmlCpuTensor<'a>,
    out_w: GgmlCpuTensor<'a>,
    out_b: GgmlCpuTensor<'a>,
    /// `[d_model, frames]` slice of the shared sinusoidal position table.
    pos_emb: GgmlCpuTensor<'a>,
}

struct BlockWeights<'a> {
    ff_macaron_norm_w: GgmlCpuTensor<'a>,
    ff_macaron_norm_b: GgmlCpuTensor<'a>,
    ff_macaron_w1_w: GgmlCpuTensor<'a>,
    ff_macaron_w1_b: GgmlCpuTensor<'a>,
    ff_macaron_w2_w: GgmlCpuTensor<'a>,
    ff_macaron_w2_b: GgmlCpuTensor<'a>,
    norm_mha_w: GgmlCpuTensor<'a>,
    norm_mha_b: GgmlCpuTensor<'a>,
    q_w: GgmlCpuTensor<'a>,
    q_b: GgmlCpuTensor<'a>,
    k_w: GgmlCpuTensor<'a>,
    k_b: GgmlCpuTensor<'a>,
    v_w: GgmlCpuTensor<'a>,
    v_b: GgmlCpuTensor<'a>,
    pos_w: GgmlCpuTensor<'a>,
    pos_bias_u: GgmlCpuTensor<'a>,
    pos_bias_v: GgmlCpuTensor<'a>,
    out_w: GgmlCpuTensor<'a>,
    out_b: GgmlCpuTensor<'a>,
    norm_mlp_w: GgmlCpuTensor<'a>,
    norm_mlp_b: GgmlCpuTensor<'a>,
    cproj1_w: GgmlCpuTensor<'a>,
    cproj1_b: GgmlCpuTensor<'a>,
    csgu_norm_w: GgmlCpuTensor<'a>,
    csgu_norm_b: GgmlCpuTensor<'a>,
    csgu_conv_w: GgmlCpuTensor<'a>,
    csgu_conv_b: GgmlCpuTensor<'a>,
    cproj2_w: GgmlCpuTensor<'a>,
    cproj2_b: GgmlCpuTensor<'a>,
    fusion_conv_w: GgmlCpuTensor<'a>,
    fusion_conv_b: GgmlCpuTensor<'a>,
    merge_w: GgmlCpuTensor<'a>,
    merge_b: GgmlCpuTensor<'a>,
    norm_ff_w: GgmlCpuTensor<'a>,
    norm_ff_b: GgmlCpuTensor<'a>,
    ff_w1_w: GgmlCpuTensor<'a>,
    ff_w1_b: GgmlCpuTensor<'a>,
    ff_w2_w: GgmlCpuTensor<'a>,
    ff_w2_b: GgmlCpuTensor<'a>,
    norm_final_w: GgmlCpuTensor<'a>,
    norm_final_b: GgmlCpuTensor<'a>,
}

struct EncoderWeights<'a> {
    embed: EmbedWeights<'a>,
    blocks: Vec<BlockWeights<'a>>,
    after_norm_w: GgmlCpuTensor<'a>,
    after_norm_b: GgmlCpuTensor<'a>,
}

/// Pending f32 weight upload: `(tensor, source-slice, static-label)`.
type Upload<'a, 'p> = (GgmlCpuTensor<'a>, &'p [f32], &'static str);
/// Pending native (quantized / f16) weight upload: `(tensor, raw-bytes,
/// ggml-type, static-label)`.
type NativeUpload<'a, 'p> = (GgmlCpuTensor<'a>, &'p [u8], i32, &'static str);

struct WeightBuilder<'a, 'p> {
    provider: &'p dyn DolphinWeightProvider,
    uploads: Vec<Upload<'a, 'p>>,
    native_uploads: Vec<NativeUpload<'a, 'p>>,
}

impl<'a, 'p> WeightBuilder<'a, 'p> {
    fn new(provider: &'p dyn DolphinWeightProvider) -> Self {
        Self {
            provider,
            uploads: Vec::new(),
            native_uploads: Vec::new(),
        }
    }

    fn fetch(&self, name: &str, expected: usize) -> Result<&'p [f32], DolphinEncoderError> {
        let data =
            self.provider
                .tensor(name)
                .ok_or_else(|| DolphinEncoderError::MissingWeight {
                    name: name.to_string(),
                })?;
        if data.len() != expected {
            return Err(DolphinEncoderError::WeightLen {
                name: name.to_string(),
                expected,
                actual: data.len(),
            });
        }
        Ok(data)
    }

    /// A 1-D weight (bias / norm gamma-beta / packed pos bias).
    fn w1(
        &mut self,
        graph: &GgmlCpuGraphBuilder<'a>,
        name: &str,
        len: usize,
    ) -> Result<GgmlCpuTensor<'a>, DolphinEncoderError> {
        let data = self.fetch(name, len)?;
        let tensor = graph
            .new_tensor_1d_f32(len, "dolphin_weight")
            .map_err(ggml_err("weight_alloc_1d"))?;
        self.uploads.push((tensor, data, "dolphin_weight"));
        Ok(tensor)
    }

    /// A 2-D `.weight` matmul operand bound as ggml `[ne0=in, ne1=out]` for
    /// `mul_mat(w, x)`. When the provider keeps this weight quantized/f16
    /// (`native_weight`), it is bound at its stored ggml type and the raw block
    /// bytes are uploaded verbatim -- the weight stays quantized in the backend
    /// buffer, feeding `mul_mat`'s quantized-lhs path directly (no dequant-to-f32
    /// blow-up). Otherwise (the raw-safetensors parity provider) it falls back to
    /// the f32 bind. Both stored layouts (fp16's `[out, in]`, quant's reversed
    /// `[in, out]`) share the same in-innermost byte order, so uploading raw into
    /// the `[ne0=in, ne1=out]` graph tensor is order-preserving in either case.
    fn w2(
        &mut self,
        graph: &GgmlCpuGraphBuilder<'a>,
        name: &str,
        ne0: usize,
        ne1: usize,
    ) -> Result<GgmlCpuTensor<'a>, DolphinEncoderError> {
        if let Some(native) = self.provider.native_weight(name) {
            let tensor = graph
                .new_matmul_weight_2d_typed(ne0, ne1, native.ggml_type, "dolphin_weight")
                .map_err(ggml_err("weight_alloc_2d_native"))?;
            self.native_uploads
                .push((tensor, native.bytes, native.ggml_type, "dolphin_weight"));
            return Ok(tensor);
        }
        let data = self.fetch(name, ne0 * ne1)?;
        let tensor = graph
            .new_tensor_2d_f32(ne0, ne1, "dolphin_weight")
            .map_err(ggml_err("weight_alloc_2d"))?;
        self.uploads.push((tensor, data, "dolphin_weight"));
        Ok(tensor)
    }

    fn w4(
        &mut self,
        graph: &GgmlCpuGraphBuilder<'a>,
        name: &str,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
    ) -> Result<GgmlCpuTensor<'a>, DolphinEncoderError> {
        let data = self.fetch(name, ne0 * ne1 * ne2 * ne3)?;
        let tensor = graph
            .new_tensor_4d_f32(ne0, ne1, ne2, ne3, "dolphin_weight")
            .map_err(ggml_err("weight_alloc_4d"))?;
        self.uploads.push((tensor, data, "dolphin_weight"));
        Ok(tensor)
    }

    /// The first `frames` rows of the `[1, max_len, d_model]` position table.
    fn pos_slice(
        &mut self,
        graph: &GgmlCpuGraphBuilder<'a>,
        name: &str,
        d_model: usize,
        frames: usize,
        max_len: usize,
    ) -> Result<GgmlCpuTensor<'a>, DolphinEncoderError> {
        let full = self.fetch(name, d_model * max_len)?;
        let slice = &full[..d_model * frames];
        let tensor = graph
            .new_tensor_2d_f32(d_model, frames, "dolphin_weight")
            .map_err(ggml_err("weight_alloc_pos"))?;
        self.uploads.push((tensor, slice, "dolphin_weight"));
        Ok(tensor)
    }
}

fn build_embed_weights<'a, 'p>(
    graph: &GgmlCpuGraphBuilder<'a>,
    builder: &mut WeightBuilder<'a, 'p>,
    config: &DolphinEncoderConfig,
    frames: usize,
) -> Result<EmbedWeights<'a>, DolphinEncoderError> {
    let d = config.d_model;
    let flat = d * subsample_width(config.feature_dim);
    Ok(EmbedWeights {
        conv0_w: builder.w4(graph, "encoder.embed.conv.0.weight", 3, 3, 1, d)?,
        conv0_b: builder.w4(graph, "encoder.embed.conv.0.bias", 1, 1, d, 1)?,
        conv1_w: builder.w4(graph, "encoder.embed.conv.2.weight", 3, 3, d, d)?,
        conv1_b: builder.w4(graph, "encoder.embed.conv.2.bias", 1, 1, d, 1)?,
        out_w: builder.w2(graph, "encoder.embed.out.0.weight", flat, d)?,
        out_b: builder.w1(graph, "encoder.embed.out.0.bias", d)?,
        pos_emb: builder.pos_slice(graph, "encoder.embed.pos_enc.pe", d, frames, 5000)?,
    })
}

fn build_block_weights<'a, 'p>(
    graph: &GgmlCpuGraphBuilder<'a>,
    builder: &mut WeightBuilder<'a, 'p>,
    config: &DolphinEncoderConfig,
    index: usize,
) -> Result<BlockWeights<'a>, DolphinEncoderError> {
    let d = config.d_model;
    let ffn = config.ffn_units;
    let cg = config.cgmlp_units;
    let cg_half = cg / 2;
    let ck = config.cgmlp_kernel;
    let mk = config.merge_kernel;
    let p = |suffix: &str| format!("encoder.encoders.{index}.{suffix}");
    Ok(BlockWeights {
        ff_macaron_norm_w: builder.w1(graph, &p("norm_ff_macaron.weight"), d)?,
        ff_macaron_norm_b: builder.w1(graph, &p("norm_ff_macaron.bias"), d)?,
        ff_macaron_w1_w: builder.w2(graph, &p("feed_forward_macaron.w_1.weight"), d, ffn)?,
        ff_macaron_w1_b: builder.w1(graph, &p("feed_forward_macaron.w_1.bias"), ffn)?,
        ff_macaron_w2_w: builder.w2(graph, &p("feed_forward_macaron.w_2.weight"), ffn, d)?,
        ff_macaron_w2_b: builder.w1(graph, &p("feed_forward_macaron.w_2.bias"), d)?,
        norm_mha_w: builder.w1(graph, &p("norm_mha.weight"), d)?,
        norm_mha_b: builder.w1(graph, &p("norm_mha.bias"), d)?,
        q_w: builder.w2(graph, &p("attn.linear_q.weight"), d, d)?,
        q_b: builder.w1(graph, &p("attn.linear_q.bias"), d)?,
        k_w: builder.w2(graph, &p("attn.linear_k.weight"), d, d)?,
        k_b: builder.w1(graph, &p("attn.linear_k.bias"), d)?,
        v_w: builder.w2(graph, &p("attn.linear_v.weight"), d, d)?,
        v_b: builder.w1(graph, &p("attn.linear_v.bias"), d)?,
        pos_w: builder.w2(graph, &p("attn.linear_pos.weight"), d, d)?,
        pos_bias_u: builder.w1(graph, &p("attn.pos_bias_u"), d)?,
        pos_bias_v: builder.w1(graph, &p("attn.pos_bias_v"), d)?,
        out_w: builder.w2(graph, &p("attn.linear_out.weight"), d, d)?,
        out_b: builder.w1(graph, &p("attn.linear_out.bias"), d)?,
        norm_mlp_w: builder.w1(graph, &p("norm_mlp.weight"), d)?,
        norm_mlp_b: builder.w1(graph, &p("norm_mlp.bias"), d)?,
        cproj1_w: builder.w2(graph, &p("cgmlp.channel_proj1.0.weight"), d, cg)?,
        cproj1_b: builder.w1(graph, &p("cgmlp.channel_proj1.0.bias"), cg)?,
        csgu_norm_w: builder.w1(graph, &p("cgmlp.csgu.norm.weight"), cg_half)?,
        csgu_norm_b: builder.w1(graph, &p("cgmlp.csgu.norm.bias"), cg_half)?,
        csgu_conv_w: builder.w4(graph, &p("cgmlp.csgu.conv.weight"), ck, 1, 1, cg_half)?,
        csgu_conv_b: builder.w1(graph, &p("cgmlp.csgu.conv.bias"), cg_half)?,
        cproj2_w: builder.w2(graph, &p("cgmlp.channel_proj2.weight"), cg_half, d)?,
        cproj2_b: builder.w1(graph, &p("cgmlp.channel_proj2.bias"), d)?,
        fusion_conv_w: builder.w4(graph, &p("depthwise_conv_fusion.weight"), mk, 1, 1, d + d)?,
        fusion_conv_b: builder.w1(graph, &p("depthwise_conv_fusion.bias"), d + d)?,
        merge_w: builder.w2(graph, &p("merge_proj.weight"), d + d, d)?,
        merge_b: builder.w1(graph, &p("merge_proj.bias"), d)?,
        norm_ff_w: builder.w1(graph, &p("norm_ff.weight"), d)?,
        norm_ff_b: builder.w1(graph, &p("norm_ff.bias"), d)?,
        ff_w1_w: builder.w2(graph, &p("feed_forward.w_1.weight"), d, ffn)?,
        ff_w1_b: builder.w1(graph, &p("feed_forward.w_1.bias"), ffn)?,
        ff_w2_w: builder.w2(graph, &p("feed_forward.w_2.weight"), ffn, d)?,
        ff_w2_b: builder.w1(graph, &p("feed_forward.w_2.bias"), d)?,
        norm_final_w: builder.w1(graph, &p("norm_final.weight"), d)?,
        norm_final_b: builder.w1(graph, &p("norm_final.bias"), d)?,
    })
}

// --- graph ops -------------------------------------------------------------

fn linear<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    input: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, DolphinEncoderError> {
    let projected = graph.mul_mat(weight, input).map_err(ggml_err(stage))?;
    graph.add(projected, bias).map_err(ggml_err(stage))
}

fn affine_ln<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    eps: f32,
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, DolphinEncoderError> {
    apply_affine_layer_norm(
        graph,
        input,
        eps,
        weight,
        bias,
        AffineLayerNormSteps {
            norm: stage,
            scale: stage,
            bias: stage,
        },
        |s, source| DolphinEncoderError::Ggml { stage: s, source },
    )
}

/// Depthwise Conv1d over time with symmetric padding, mirroring the shared
/// conformer conv path: `[channels, frames]` in and out. `kernel` is ggml
/// `[k, 1, 1, channels]`, `bias` is `[channels]`.
fn depthwise_conv1d<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    kernel: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    channels: usize,
    frames: usize,
    padding: usize,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, DolphinEncoderError> {
    let map = ggml_err(stage);
    let transposed = graph.transpose(input).map_err(map)?;
    let transposed = graph.cont(transposed).map_err(map)?;
    let as_4d = graph
        .reshape_4d(transposed, frames, 1, channels, 1)
        .map_err(map)?;
    let conv = graph
        .depthwise_conv_2d(kernel, as_4d, 1, 1, padding, 0, 1, 1)
        .map_err(map)?;
    let conv = graph.permute(conv, 1, 2, 0, 3).map_err(map)?;
    let conv = graph.cont(conv).map_err(map)?;
    graph.add(conv, bias).map_err(map)
}

fn feed_forward_half<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    normed: GgmlCpuTensor<'a>,
    residual: GgmlCpuTensor<'a>,
    up_w: GgmlCpuTensor<'a>,
    up_b: GgmlCpuTensor<'a>,
    down_w: GgmlCpuTensor<'a>,
    down_b: GgmlCpuTensor<'a>,
    scale: f32,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, DolphinEncoderError> {
    apply_feed_forward_residual(
        graph,
        normed,
        residual,
        FeedForwardActivation::Silu,
        Some(scale),
        FeedForwardResidualSteps {
            activation: stage,
            scale: Some(stage),
            residual: stage,
        },
        |graph, value| linear(graph, up_w, value, up_b, stage),
        |graph, value| linear(graph, down_w, value, down_b, stage),
        |s, source| DolphinEncoderError::Ggml { stage: s, source },
    )
}

/// The rel-pos global branch (WeNet `RelPositionMultiHeadedAttention` with
/// `use_sdpa=true`): scores = `(q_u . k + q_v . p) / sqrt(head_dim)`, softmax,
/// context. No `rel_shift` and `pos_emb` length == T because sdpa folds the bias
/// straight into the scores; full-context single utterance so no mask term.
fn attention_branch<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    normed: GgmlCpuTensor<'a>,
    pos_emb: GgmlCpuTensor<'a>,
    weights: &BlockWeights<'a>,
    config: &DolphinEncoderConfig,
    frames: usize,
) -> Result<GgmlCpuTensor<'a>, DolphinEncoderError> {
    let map = ggml_err("attention");
    let q = linear(graph, weights.q_w, normed, weights.q_b, "attn_q")?;
    let k = linear(graph, weights.k_w, normed, weights.k_b, "attn_k")?;
    let v = linear(graph, weights.v_w, normed, weights.v_b, "attn_v")?;
    let p = graph.mul_mat(weights.pos_w, pos_emb).map_err(map)?;

    let q_u = graph.add(q, weights.pos_bias_u).map_err(map)?;
    let q_v = graph.add(q, weights.pos_bias_v).map_err(map)?;

    let layout = AttentionHeadLayout {
        head_dim: config.head_dim,
        attention_heads: config.attention_heads,
        sequence_len: frames,
    };
    let reshape_steps = AttentionReshapeSteps {
        reshape: "attn_reshape",
        permute: "attn_permute",
        cont: "attn_cont",
    };
    let map_err = |s, source| DolphinEncoderError::Ggml { stage: s, source };
    let q_u = reshape_projection_to_attention_heads(
        graph,
        q_u,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        reshape_steps,
        map_err,
    )?;
    let q_v = reshape_projection_to_attention_heads(
        graph,
        q_v,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        reshape_steps,
        map_err,
    )?;
    let k = reshape_projection_to_attention_heads(
        graph,
        k,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        reshape_steps,
        map_err,
    )?;
    let p = reshape_projection_to_attention_heads(
        graph,
        p,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        reshape_steps,
        map_err,
    )?;

    let ac = graph
        .mul_mat(graph.cont(k).map_err(map)?, q_u)
        .map_err(map)?;
    let bd = graph
        .mul_mat(graph.cont(p).map_err(map)?, q_v)
        .map_err(map)?;
    let scores = graph.add(ac, bd).map_err(map)?;
    let scores = graph
        .scale(scores, 1.0 / (config.head_dim as f32).sqrt())
        .map_err(map)?;
    let scores = graph.soft_max(scores).map_err(map)?;

    let v_heads = reshape_projection_to_attention_heads(
        graph,
        v,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        reshape_steps,
        map_err,
    )?;
    let context = attention_context_from_probs(
        graph,
        v_heads,
        scores,
        layout,
        AttentionValueMergeSteps {
            value_permute: "attn_v_t",
            value_cont: "attn_v_t",
            context_mul: "attn_ctx",
            context_merge_permute: "attn_merge",
            context_merge_cont: "attn_merge",
            context_merge_reshape: "attn_merge",
        },
        map_err,
    )?;
    linear(graph, weights.out_w, context, weights.out_b, "attn_out")
}

/// The cgMLP local branch: `channel_proj1 (GELU) -> CSGU -> channel_proj2`.
/// CSGU: split channels in half, LayerNorm + depthwise conv the gate half,
/// identity gate, multiply into the value half.
fn cgmlp_branch<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    normed: GgmlCpuTensor<'a>,
    weights: &BlockWeights<'a>,
    config: &DolphinEncoderConfig,
    frames: usize,
) -> Result<GgmlCpuTensor<'a>, DolphinEncoderError> {
    let map = ggml_err("cgmlp");
    let cg = config.cgmlp_units;
    let cg_half = cg / 2;

    let proj1 = linear(
        graph,
        weights.cproj1_w,
        normed,
        weights.cproj1_b,
        "cgmlp_proj1",
    )?;
    let proj1 = graph.gelu_erf(proj1).map_err(map)?;

    // chunk(2, dim=-1): first half is the value, second half is the gate.
    let row_stride = cg * F32_BYTES;
    let x_value = graph
        .view_2d(proj1, cg_half, frames, row_stride, 0)
        .map_err(map)?;
    let x_value = graph.cont(x_value).map_err(map)?;
    let x_gate = graph
        .view_2d(proj1, cg_half, frames, row_stride, cg_half * F32_BYTES)
        .map_err(map)?;
    let x_gate = graph.cont(x_gate).map_err(map)?;

    let x_gate = affine_ln(
        graph,
        x_gate,
        config.layer_norm_epsilon,
        weights.csgu_norm_w,
        weights.csgu_norm_b,
        "cgmlp_csgu_norm",
    )?;
    let x_gate = depthwise_conv1d(
        graph,
        x_gate,
        weights.csgu_conv_w,
        weights.csgu_conv_b,
        cg_half,
        frames,
        (config.cgmlp_kernel - 1) / 2,
        "cgmlp_csgu_conv",
    )?;
    // gate_activation = identity.
    let gated = graph.mul(x_value, x_gate).map_err(map)?;
    linear(
        graph,
        weights.cproj2_w,
        gated,
        weights.cproj2_b,
        "cgmlp_proj2",
    )
}

fn encoder_block<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    pos_emb: GgmlCpuTensor<'a>,
    weights: &BlockWeights<'a>,
    config: &DolphinEncoderConfig,
    frames: usize,
) -> Result<GgmlCpuTensor<'a>, DolphinEncoderError> {
    let eps = config.layer_norm_epsilon;
    let map = ggml_err("block");

    // macaron FFN half-step.
    let macaron_norm = affine_ln(
        graph,
        input,
        eps,
        weights.ff_macaron_norm_w,
        weights.ff_macaron_norm_b,
        "macaron_norm",
    )?;
    let x = feed_forward_half(
        graph,
        macaron_norm,
        input,
        weights.ff_macaron_w1_w,
        weights.ff_macaron_w1_b,
        weights.ff_macaron_w2_w,
        weights.ff_macaron_w2_b,
        0.5,
        "macaron_ffn",
    )?;

    // Two branches over the same post-macaron hidden.
    let attn_norm = affine_ln(
        graph,
        x,
        eps,
        weights.norm_mha_w,
        weights.norm_mha_b,
        "attn_norm",
    )?;
    let branch_attn = attention_branch(graph, attn_norm, pos_emb, weights, config, frames)?;

    let mlp_norm = affine_ln(
        graph,
        x,
        eps,
        weights.norm_mlp_w,
        weights.norm_mlp_b,
        "mlp_norm",
    )?;
    let branch_cgmlp = cgmlp_branch(graph, mlp_norm, weights, config, frames)?;

    // Merge: concat -> depthwise fusion conv -> merge_proj, residual on the
    // post-macaron hidden.
    let concat = graph.concat(branch_attn, branch_cgmlp, 0).map_err(map)?;
    let fusion = depthwise_conv1d(
        graph,
        concat,
        weights.fusion_conv_w,
        weights.fusion_conv_b,
        config.d_model + config.d_model,
        frames,
        (config.merge_kernel - 1) / 2,
        "merge_conv",
    )?;
    let merged = graph.add(concat, fusion).map_err(map)?;
    let merge_proj = linear(
        graph,
        weights.merge_w,
        merged,
        weights.merge_b,
        "merge_proj",
    )?;
    let x = graph.add(x, merge_proj).map_err(map)?;

    // Final FFN half-step.
    let ff_norm = affine_ln(
        graph,
        x,
        eps,
        weights.norm_ff_w,
        weights.norm_ff_b,
        "ff_norm",
    )?;
    let x = feed_forward_half(
        graph,
        ff_norm,
        x,
        weights.ff_w1_w,
        weights.ff_w1_b,
        weights.ff_w2_w,
        weights.ff_w2_b,
        0.5,
        "ff_ffn",
    )?;

    affine_ln(
        graph,
        x,
        eps,
        weights.norm_final_w,
        weights.norm_final_b,
        "norm_final",
    )
}

/// Conv2dSubsampling4 -> `* sqrt(d_model)`. Input is `[feature_dim, T]` in ggml
/// order (mel bin innermost). Returns `[d_model, frames]` (the block-0 hidden).
fn subsample<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    embed: &EmbedWeights<'a>,
    config: &DolphinEncoderConfig,
    frames_in: usize,
) -> Result<(GgmlCpuTensor<'a>, usize), DolphinEncoderError> {
    let map = ggml_err("subsample");
    let d = config.d_model;
    let feat = config.feature_dim;
    let width = subsample_width(feat);
    let frames = subsample_len(frames_in);

    // ggml conv_2d: data [W=feat, H=T, C_in=1, N=1], kernel [KW, KH, C_in, C_out].
    let data = graph
        .reshape_4d(input, feat, frames_in, 1, 1)
        .map_err(map)?;
    let conv0 = graph
        .conv_2d(embed.conv0_w, data, 2, 2, 0, 0, 1, 1)
        .map_err(map)?;
    let conv0 = graph.add(conv0, embed.conv0_b).map_err(map)?;
    let conv0 = graph.relu(conv0).map_err(map)?;
    let conv1 = graph
        .conv_2d(embed.conv1_w, conv0, 2, 2, 0, 0, 1, 1)
        .map_err(map)?;
    let conv1 = graph.add(conv1, embed.conv1_b).map_err(map)?;
    let conv1 = graph.relu(conv1).map_err(map)?;

    // conv1 is [W=width, H=frames, C=d, N=1]. PyTorch flattens per frame as
    // (channel, freq) with freq innermost -> reorder to [freq, channel, frame].
    let reordered = graph.permute(conv1, 0, 2, 1, 3).map_err(map)?;
    let reordered = graph.cont(reordered).map_err(map)?;
    let flat = graph
        .reshape_2d(reordered, width * d, frames)
        .map_err(map)?;
    let projected = linear(graph, embed.out_w, flat, embed.out_b, "subsample_out")?;
    let scaled = graph.scale(projected, (d as f32).sqrt()).map_err(map)?;
    Ok((scaled, frames))
}

/// Build and run the full encoder graph on the CPU backend. Returns the encoder
/// output plus per-stage taps for parity.
pub(crate) fn encode(
    config: &DolphinEncoderConfig,
    provider: &dyn DolphinWeightProvider,
    features: &[f32],
    frames_in: usize,
    backend: GgmlCpuGraphBackend,
) -> Result<DolphinEncoderOutput, DolphinEncoderError> {
    let feat = config.feature_dim;
    if features.len() != frames_in * feat {
        return Err(DolphinEncoderError::Shape {
            reason: format!(
                "features has {} values, expected {frames_in}x{feat}",
                features.len()
            ),
        });
    }
    let frames = subsample_len(frames_in);

    let graph_config = GgmlCpuGraphConfig {
        context_bytes: 64 * 1024 * 1024,
        graph_size: 16384,
        n_threads: GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            backend,
            crate::ggml_runtime::GgmlCpuGraphThreadingWorkload::EncoderPrelude,
        ),
        backend,
        use_scheduler: backend.is_gpu_class(),
    };
    let mut runner = GgmlCpuGraphRunner::new(graph_config).map_err(ggml_err("runner_init"))?;
    let mut graph = runner.start_graph();

    // Phase A: create every weight tensor (must precede the first buffer alloc).
    let mut builder = WeightBuilder::new(provider);
    let embed = build_embed_weights(&graph, &mut builder, config, frames)?;
    let mut blocks = Vec::with_capacity(config.num_blocks);
    for index in 0..config.num_blocks {
        blocks.push(build_block_weights(&graph, &mut builder, config, index)?);
    }
    let after_norm_w = builder.w1(&graph, "encoder.after_norm.weight", config.d_model)?;
    let after_norm_b = builder.w1(&graph, "encoder.after_norm.bias", config.d_model)?;
    let weights = EncoderWeights {
        embed,
        blocks,
        after_norm_w,
        after_norm_b,
    };

    let input = graph
        .new_tensor_2d_f32(feat, frames_in, "dolphin_features")
        .map_err(ggml_err("input_alloc"))?;

    // Phase B: build the forward graph and collect the taps.
    let (after_subsample, frames_check) =
        subsample(&graph, input, &weights.embed, config, frames_in)?;
    if frames_check != frames {
        return Err(DolphinEncoderError::Shape {
            reason: format!("subsample produced {frames_check} frames, expected {frames}"),
        });
    }
    let mut taps: Vec<GgmlCpuTensor> = Vec::with_capacity(config.num_blocks + 2);
    taps.push(after_subsample);
    let mut hidden = after_subsample;
    for block in &weights.blocks {
        hidden = encoder_block(
            &mut graph,
            hidden,
            weights.embed.pos_emb,
            block,
            config,
            frames,
        )?;
        taps.push(hidden);
    }
    let encoder_out = affine_ln(
        &graph,
        hidden,
        config.layer_norm_epsilon,
        weights.after_norm_w,
        weights.after_norm_b,
        "after_norm",
    )?;
    taps.push(encoder_out);

    for tap in &taps {
        graph.set_output(*tap).map_err(ggml_err("set_output"))?;
    }

    // Phase C: upload inputs + weights, then compute. Native (quantized/f16)
    // rank-2 `.weight` operands upload their raw block bytes verbatim so they stay
    // quantized in the backend buffer; everything else uploads dequantized f32.
    graph
        .set_f32_slice(input, features, "dolphin_features")
        .map_err(ggml_err("upload_features"))?;
    for (tensor, data, name) in &builder.uploads {
        graph
            .set_f32_slice(*tensor, data, name)
            .map_err(ggml_err("upload_weight"))?;
    }
    for (tensor, bytes, ggml_type, name) in &builder.native_uploads {
        graph
            .set_matmul_weight_bytes(*tensor, bytes, *ggml_type, name)
            .map_err(ggml_err("upload_weight_native"))?;
    }

    let expected = frames * config.d_model;
    let output_specs: Vec<(GgmlCpuTensor, usize)> =
        taps.iter().map(|tap| (*tap, expected)).collect();
    let mut outputs = graph
        .compute_outputs_f32(&output_specs)
        .map_err(ggml_err("compute"))?;

    let encoder_out = outputs.pop().expect("encoder_out tap");
    let blocks = outputs.split_off(1);
    let after_subsample = outputs.pop().expect("after_subsample tap");

    Ok(DolphinEncoderOutput {
        frames,
        dim: config.d_model,
        after_subsample,
        blocks,
        encoder_out,
    })
}
