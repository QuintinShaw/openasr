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
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlStaticTensor, GgmlStaticTensorArena,
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

/// Scalar/shape configuration for the Dolphin encoder. `language_scheme`
/// selects the encoder's relative-position-attention flavor, which differs
/// between the two Dolphin training pipelines (confirmed by reading
/// `DataoceanAI/Dolphin`'s own inference source, not assumed):
///
/// * [`DolphinLanguageScheme::CnDialect`] (`small.cn`/`cn-dialect-base`,
///   WeNet-trained, `use_sdpa: true`): the simple non-centered
///   `RelPositionalEncoding`, sliced `[0, frames)` from a baked/synthesized
///   table, folded into the SDPA bias with **no `rel_shift`** (`pos_emb`
///   length == `frames`) -- unchanged from before this scheme split existed.
/// * [`DolphinLanguageScheme::Multilingual`] (`dolphin-small`/`dolphin-base`,
///   ESPnet-trained, `use_sdpa: false`, `pos_enc_layer_type: rel_pos_v1`): the
///   centered Transformer-XL `RelPositionalEncodingV1` (`2*frames-1` positions,
///   `[frame-1 .. -(frame-1)]`), computed fresh per `frames` at graph-build
///   time (see `dolphin_relative_positional_table`) and consumed through the
///   real `rel_shift` (see `attention_branch`).
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
    /// Length of the sinusoidal position table baked into
    /// `encoder.embed.pos_enc.pe` (`dolphin.encoder.max_ctx`). Only consulted
    /// under [`DolphinLanguageScheme::CnDialect`]; the multilingual scheme
    /// computes its rel-pos table fresh per request instead.
    pub max_positions: usize,
    pub layer_norm_epsilon: f32,
    pub language_scheme: super::package_import::DolphinLanguageScheme,
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
            max_positions: 5000,
            layer_norm_epsilon: 1e-5,
            language_scheme: super::package_import::DolphinLanguageScheme::CnDialect,
        }
    }

    /// Build the config from the pack's own parsed runtime metadata --
    /// checkpoint-size-agnostic (`small.cn`/base/multilingual all resolve
    /// through this same path). `layer_norm_epsilon` is not a per-checkpoint
    /// metadata key: every observed Dolphin/WeNet checkpoint uses the same
    /// `1e-5` LayerNorm epsilon, so it stays a fixed architecture constant
    /// like `small_cn()`'s. `language_scheme` comes from the pack's
    /// `dolphin.language.scheme` metadata (see `executor::run_dolphin_pipeline`),
    /// same signal the decode-prefix builder and frontend already dispatch on.
    pub(crate) fn from_execution_metadata(
        metadata: &super::runtime_contract::DolphinExecutionMetadata,
        language_scheme: super::package_import::DolphinLanguageScheme,
    ) -> Self {
        Self {
            d_model: metadata.encoder_d_model,
            attention_heads: metadata.encoder_n_heads,
            head_dim: metadata.encoder_head_dim,
            ffn_units: metadata.encoder_ffn_dim,
            cgmlp_units: metadata.encoder_cgmlp_units,
            cgmlp_kernel: metadata.encoder_cgmlp_kernel,
            merge_kernel: metadata.encoder_merge_kernel,
            num_blocks: metadata.encoder_n_layers,
            feature_dim: metadata.feature_dim,
            max_positions: metadata.encoder_max_ctx,
            layer_norm_epsilon: 1e-5,
            language_scheme,
        }
    }
}

/// Full encoder result. `encoder_out` is always populated; `after_subsample`
/// and `blocks` are per-stage taps that [`encode`]'s `capture_taps` flag only
/// materializes for `#[cfg(test)]` parity gating (empty otherwise -- see
/// [`encode`]'s doc comment, P6).
#[derive(Debug, Clone)]
pub(crate) struct DolphinEncoderOutput {
    pub frames: usize,
    pub dim: usize,
    /// Frame-major `[frames, dim]` output of `Conv2dSubsampling4 * sqrt(d_model)`
    /// (the hidden entering block 0). Empty unless `encode` was called with
    /// `capture_taps: true`.
    pub after_subsample: Vec<f32>,
    /// Frame-major `[frames, dim]` output after each block's `norm_final`.
    /// Empty unless `encode` was called with `capture_taps: true`.
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

/// One `k3 s2` (no padding) Conv2d layer's output length along an axis, or
/// `Err` if `input` is smaller than the kernel (the ggml `im2col`/`conv_2d`
/// precondition `OH > 0` -- feeding it an under-sized input aborts the whole
/// process instead of returning a Rust error, so this must be checked before
/// the graph is ever built; see `subsample_len`/`subsample`/`encode`).
fn conv2d_no_pad_stride2_out_len(
    input: usize,
    kernel: usize,
) -> Result<usize, DolphinEncoderError> {
    const STRIDE: usize = 2;
    input
        .checked_sub(kernel)
        .and_then(|value| value.checked_div(STRIDE))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| DolphinEncoderError::Shape {
            reason: format!(
                "conv2d subsampling requires at least {kernel} input frames, got {input}"
            ),
        })
}

/// Subsampled frame count after two `k3 s2` (no padding) conv layers (4x time
/// downsample). `Err` when `frames_in` is too short for the two-layer
/// receptive field (7 frames minimum: `(7-3)/2+1=3` after layer one, `(3-3)/2+1=1`
/// after layer two) instead of silently producing a degenerate frame count that
/// would abort the ggml conv at graph-build time.
fn subsample_len(frames: usize) -> Result<usize, DolphinEncoderError> {
    let after_first = conv2d_no_pad_stride2_out_len(frames, 3)?;
    conv2d_no_pad_stride2_out_len(after_first, 3)
}

/// The smallest `frames_in` for which [`subsample_len`] succeeds. Derived by
/// walking the same checked formula forward instead of a hardcoded constant,
/// so it cannot drift out of sync with it.
pub(crate) fn minimum_subsample_input_frames() -> usize {
    (1..)
        .find(|&frames| subsample_len(frames).is_ok())
        .expect("subsample_len's two-layer k3/s2 chain has a finite minimum valid input")
}

/// Subsampled feature width after the same two conv layers on the mel axis.
/// Unlike `subsample_len` this always runs over the model's fixed
/// `feature_dim` config (e.g. 80 mel bins), never a runtime-variable audio
/// length, so it stays infallible (saturating is safe: `feature_dim` is always
/// far above the two-layer receptive field for every published Dolphin pack).
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
    // `pos_emb` moved out of `EmbedWeights`: its scheme-dependent shape/source
    // (a baked-table slice for `CnDialect`, a graph-build-time-computed
    // rel-pos-v1 table for `Multilingual`) is built directly in `encode()`
    // instead, see `dolphin_relative_positional_table`.
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

/// Static-arena tensor count for [`GgmlCpuGraphConfig::metadata_context_bytes`].
/// Per E-Branchformer block, `build_block_weights` allocates 41 weight tensors
/// (macaron FFN norm+w1+w2 = 6, MHA norm = 2, rel-pos attention q/k/v/pos/out +
/// pos_bias_u/v = 11, cgMLP norm = 2, cgMLP proj1/csgu-norm/csgu-conv/proj2 = 8,
/// fusion conv + merge proj = 4, final FFN norm+w1+w2 = 6, block final norm = 2);
/// the fixed set is the embed stem (2 convs + out = 6 tensors), the after_norm
/// pair (2), and the position table (1).
const DOLPHIN_ENCODER_ARENA_TENSORS_PER_BLOCK: usize = 41;
const DOLPHIN_ENCODER_ARENA_FIXED_TENSORS: usize = 9;

fn dolphin_encoder_arena_context_bytes(num_blocks: usize) -> usize {
    let tensor_count = DOLPHIN_ENCODER_ARENA_FIXED_TENSORS
        .saturating_add(DOLPHIN_ENCODER_ARENA_TENSORS_PER_BLOCK.saturating_mul(num_blocks));
    GgmlCpuGraphConfig::metadata_context_bytes(tensor_count)
}

/// Pending f32 weight upload into the static arena: `(handle, source-slice,
/// static-label)`.
type Upload<'p> = (GgmlStaticTensor, &'p [f32], &'static str);
/// Pending native (quantized / f16) weight upload: `(handle, raw-bytes,
/// static-label)`.
type NativeUpload<'p> = (GgmlStaticTensor, &'p [u8], &'static str);

/// Allocates every encoder weight into the runtime's persistent
/// [`GgmlStaticTensorArena`] (a `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` backend
/// buffer) rather than as per-call transient graph leaves. This is what lets the
/// ggml multi-backend scheduler offload the encoder's matmuls to an accelerator:
/// the scheduler only considers an op for `op_offload` when its weight `src`
/// lives in a WEIGHTS-usage buffer, which the old per-call graph-input weights
/// were not, so the whole E-Branchformer (the FLOPs-heavy stage) was pinned to
/// the CPU even under an explicit Metal backend. Mirrors the sibling
/// `decoder_graph::StaticWeightBuilder` and `sensevoice::SenseVoiceEncoderGraph`
/// exactly: allocate every tensor first (the arena's first upload freezes
/// further creation), then upload once. Weight placement changes no value the
/// graph computes, so the encoder output stays golden-identical -- only the
/// backend each op runs on changes.
struct WeightBuilder<'p> {
    provider: &'p dyn DolphinWeightProvider,
    uploads: Vec<Upload<'p>>,
    native_uploads: Vec<NativeUpload<'p>>,
}

impl<'p> WeightBuilder<'p> {
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
    fn w1<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        len: usize,
    ) -> Result<GgmlCpuTensor<'a>, DolphinEncoderError> {
        let data = self.fetch(name, len)?;
        let handle = arena
            .new_tensor_1d_f32(len, "dolphin_weight")
            .map_err(ggml_err("weight_alloc_1d"))?;
        self.uploads.push((handle, data, "dolphin_weight"));
        Ok(arena.graph_tensor(handle))
    }

    /// A 2-D `.weight` matmul operand bound as ggml `[ne0=in, ne1=out]` for
    /// `mul_mat(w, x)`. When the provider keeps this weight quantized/f16
    /// (`native_weight`), it is bound at its stored ggml type and the raw block
    /// bytes are uploaded verbatim -- the weight stays quantized in the backend
    /// buffer, feeding `mul_mat`'s quantized-lhs path directly (no dequant-to-f32
    /// blow-up). Otherwise (the raw-safetensors parity provider) it falls back to
    /// the f32 bind. Both stored layouts (fp16's `[out, in]`, quant's reversed
    /// `[in, out]`) share the same in-innermost byte order, so uploading raw into
    /// the `[ne0=in, ne1=out]` arena tensor is order-preserving in either case.
    fn w2<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        ne0: usize,
        ne1: usize,
    ) -> Result<GgmlCpuTensor<'a>, DolphinEncoderError> {
        if let Some(native) = self.provider.native_weight(name) {
            let handle = arena
                .new_matmul_weight_2d_typed(ne0, ne1, native.ggml_type, "dolphin_weight")
                .map_err(ggml_err("weight_alloc_2d_native"))?;
            self.native_uploads
                .push((handle, native.bytes, "dolphin_weight"));
            return Ok(arena.graph_tensor(handle));
        }
        let data = self.fetch(name, ne0 * ne1)?;
        let handle = arena
            .new_tensor_2d_f32(ne0, ne1, "dolphin_weight")
            .map_err(ggml_err("weight_alloc_2d"))?;
        self.uploads.push((handle, data, "dolphin_weight"));
        Ok(arena.graph_tensor(handle))
    }

    fn w4<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
    ) -> Result<GgmlCpuTensor<'a>, DolphinEncoderError> {
        let data = self.fetch(name, ne0 * ne1 * ne2 * ne3)?;
        let handle = arena
            .new_tensor_4d_f32(ne0, ne1, ne2, ne3, "dolphin_weight")
            .map_err(ggml_err("weight_alloc_4d"))?;
        self.uploads.push((handle, data, "dolphin_weight"));
        Ok(arena.graph_tensor(handle))
    }

    /// The first `frames` rows of the `[1, max_len, d_model]` position table.
    fn pos_slice<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        d_model: usize,
        frames: usize,
        max_len: usize,
    ) -> Result<GgmlCpuTensor<'a>, DolphinEncoderError> {
        let full = self.fetch(name, d_model * max_len)?;
        let slice = &full[..d_model * frames];
        let handle = arena
            .new_tensor_2d_f32(d_model, frames, "dolphin_weight")
            .map_err(ggml_err("weight_alloc_pos"))?;
        self.uploads.push((handle, slice, "dolphin_weight"));
        Ok(arena.graph_tensor(handle))
    }

    /// Upload every collected weight into the arena's backend buffer exactly
    /// once (the first upload allocates the buffer and freezes further tensor
    /// creation). Native (quantized / f16) rank-2 `.weight` operands upload
    /// their raw block bytes verbatim so they stay quantized in the buffer;
    /// everything else uploads dequantized f32.
    fn upload(&self, arena: &mut GgmlStaticTensorArena) -> Result<(), DolphinEncoderError> {
        for (handle, data, name) in &self.uploads {
            arena
                .set_f32_slice(*handle, data, name)
                .map_err(ggml_err("upload_weight"))?;
        }
        for (handle, bytes, name) in &self.native_uploads {
            arena
                .set_bytes_slice(*handle, bytes, name)
                .map_err(ggml_err("upload_weight_native"))?;
        }
        Ok(())
    }
}

fn build_embed_weights<'a, 'p>(
    arena: &GgmlStaticTensorArena,
    builder: &mut WeightBuilder<'p>,
    config: &DolphinEncoderConfig,
) -> Result<EmbedWeights<'a>, DolphinEncoderError> {
    let d = config.d_model;
    let flat = d * subsample_width(config.feature_dim);
    Ok(EmbedWeights {
        conv0_w: builder.w4(arena, "encoder.embed.conv.0.weight", 3, 3, 1, d)?,
        conv0_b: builder.w4(arena, "encoder.embed.conv.0.bias", 1, 1, d, 1)?,
        conv1_w: builder.w4(arena, "encoder.embed.conv.2.weight", 3, 3, d, d)?,
        conv1_b: builder.w4(arena, "encoder.embed.conv.2.bias", 1, 1, d, 1)?,
        out_w: builder.w2(arena, "encoder.embed.out.0.weight", flat, d)?,
        out_b: builder.w1(arena, "encoder.embed.out.0.bias", d)?,
    })
}

/// The centered Transformer-XL relative-position sinusoidal table ESPnet's
/// `RelPositionalEncodingV1` computes fresh per forward call (never baked as a
/// state-dict buffer): `2*frames-1` rows, position `frames-1` (row 0) down to
/// `-(frames-1)` (last row) -- `pe_positive` (flipped) concatenated with
/// `pe_negative[1:]` in the reference. Row-major `[position][d_model]`
/// (d_model innermost), matching `pos_slice`'s baked-table layout so both
/// schemes' `pos_emb` tensors share the same `[d_model, positions]` ggml
/// binding convention downstream.
fn dolphin_relative_positional_table(d_model: usize, frames: usize) -> Option<Vec<f32>> {
    let n_positions = frames.checked_mul(2)?.checked_sub(1)?;
    let total = n_positions.checked_mul(d_model)?;
    let mut table = vec![0.0f32; total];
    for position_idx in 0..n_positions {
        let pos = (frames - 1) as f64 - position_idx as f64;
        let row = &mut table[position_idx * d_model..(position_idx + 1) * d_model];
        let mut i = 0;
        while i < d_model {
            let div_term = (-((i as f64) / (d_model as f64)) * 10000.0_f64.ln()).exp();
            let angle = pos * div_term;
            row[i] = angle.sin() as f32;
            if i + 1 < d_model {
                row[i + 1] = angle.cos() as f32;
            }
            i += 2;
        }
    }
    Some(table)
}

fn build_block_weights<'a, 'p>(
    arena: &GgmlStaticTensorArena,
    builder: &mut WeightBuilder<'p>,
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
        ff_macaron_norm_w: builder.w1(arena, &p("norm_ff_macaron.weight"), d)?,
        ff_macaron_norm_b: builder.w1(arena, &p("norm_ff_macaron.bias"), d)?,
        ff_macaron_w1_w: builder.w2(arena, &p("feed_forward_macaron.w_1.weight"), d, ffn)?,
        ff_macaron_w1_b: builder.w1(arena, &p("feed_forward_macaron.w_1.bias"), ffn)?,
        ff_macaron_w2_w: builder.w2(arena, &p("feed_forward_macaron.w_2.weight"), ffn, d)?,
        ff_macaron_w2_b: builder.w1(arena, &p("feed_forward_macaron.w_2.bias"), d)?,
        norm_mha_w: builder.w1(arena, &p("norm_mha.weight"), d)?,
        norm_mha_b: builder.w1(arena, &p("norm_mha.bias"), d)?,
        q_w: builder.w2(arena, &p("attn.linear_q.weight"), d, d)?,
        q_b: builder.w1(arena, &p("attn.linear_q.bias"), d)?,
        k_w: builder.w2(arena, &p("attn.linear_k.weight"), d, d)?,
        k_b: builder.w1(arena, &p("attn.linear_k.bias"), d)?,
        v_w: builder.w2(arena, &p("attn.linear_v.weight"), d, d)?,
        v_b: builder.w1(arena, &p("attn.linear_v.bias"), d)?,
        pos_w: builder.w2(arena, &p("attn.linear_pos.weight"), d, d)?,
        pos_bias_u: builder.w1(arena, &p("attn.pos_bias_u"), d)?,
        pos_bias_v: builder.w1(arena, &p("attn.pos_bias_v"), d)?,
        out_w: builder.w2(arena, &p("attn.linear_out.weight"), d, d)?,
        out_b: builder.w1(arena, &p("attn.linear_out.bias"), d)?,
        norm_mlp_w: builder.w1(arena, &p("norm_mlp.weight"), d)?,
        norm_mlp_b: builder.w1(arena, &p("norm_mlp.bias"), d)?,
        cproj1_w: builder.w2(arena, &p("cgmlp.channel_proj1.0.weight"), d, cg)?,
        cproj1_b: builder.w1(arena, &p("cgmlp.channel_proj1.0.bias"), cg)?,
        csgu_norm_w: builder.w1(arena, &p("cgmlp.csgu.norm.weight"), cg_half)?,
        csgu_norm_b: builder.w1(arena, &p("cgmlp.csgu.norm.bias"), cg_half)?,
        csgu_conv_w: builder.w4(arena, &p("cgmlp.csgu.conv.weight"), ck, 1, 1, cg_half)?,
        csgu_conv_b: builder.w1(arena, &p("cgmlp.csgu.conv.bias"), cg_half)?,
        cproj2_w: builder.w2(arena, &p("cgmlp.channel_proj2.weight"), cg_half, d)?,
        cproj2_b: builder.w1(arena, &p("cgmlp.channel_proj2.bias"), d)?,
        fusion_conv_w: builder.w4(arena, &p("depthwise_conv_fusion.weight"), mk, 1, 1, d + d)?,
        fusion_conv_b: builder.w1(arena, &p("depthwise_conv_fusion.bias"), d + d)?,
        merge_w: builder.w2(arena, &p("merge_proj.weight"), d + d, d)?,
        merge_b: builder.w1(arena, &p("merge_proj.bias"), d)?,
        norm_ff_w: builder.w1(arena, &p("norm_ff.weight"), d)?,
        norm_ff_b: builder.w1(arena, &p("norm_ff.bias"), d)?,
        ff_w1_w: builder.w2(arena, &p("feed_forward.w_1.weight"), d, ffn)?,
        ff_w1_b: builder.w1(arena, &p("feed_forward.w_1.bias"), ffn)?,
        ff_w2_w: builder.w2(arena, &p("feed_forward.w_2.weight"), ffn, d)?,
        ff_w2_b: builder.w1(arena, &p("feed_forward.w_2.bias"), d)?,
        norm_final_w: builder.w1(arena, &p("norm_final.weight"), d)?,
        norm_final_b: builder.w1(arena, &p("norm_final.bias"), d)?,
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

/// The rel-pos global branch (`RelPositionMultiHeadedAttention`):
/// scores = `(q_u . k + q_v . p) / sqrt(head_dim)`, softmax, context.
///
/// Two schemes, dispatched on `config.language_scheme` (see
/// [`DolphinEncoderConfig`]'s doc comment):
/// * `CnDialect` (`use_sdpa: true`): `pos_emb` length == `frames`, **no
///   `rel_shift`** -- sdpa folds `matrix_bd` directly into the bias. Unchanged
///   from the parity-verified small.cn path.
/// * `Multilingual` (`use_sdpa: false`, `rel_pos_v1`): `pos_emb` length ==
///   `2*frames-1` (centered), and `matrix_bd` goes through the real
///   `rel_shift` (a strided `view_3d` reinterpretation, the same trick
///   `nn::encoder::conformer_block` uses for cohere/parakeet's Transformer-XL
///   rel-pos attention) before being added to `matrix_ac`.
///
/// Full-context single utterance either way, so no additive mask term.
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
    let is_multilingual = matches!(
        config.language_scheme,
        super::package_import::DolphinLanguageScheme::Multilingual
    );
    let pos_layout = if is_multilingual {
        AttentionHeadLayout {
            sequence_len: 2 * frames - 1,
            ..layout
        }
    } else {
        layout
    };
    let p = reshape_projection_to_attention_heads(
        graph,
        p,
        pos_layout,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        reshape_steps,
        map_err,
    )?;

    let ac = graph
        .mul_mat(graph.cont(k).map_err(map)?, q_u)
        .map_err(map)?;
    let bd_raw = graph
        .mul_mat(graph.cont(p).map_err(map)?, q_v)
        .map_err(map)?;
    let bd = if is_multilingual {
        // rel_shift: reinterpret `bd_raw` (`[2*frames-1, frames, heads]`) as
        // `[frames, frames, heads]` via the classic pad+reshape+slice trick,
        // done directly as a strided view (no extra copy) -- byte-identical
        // stride formula to `nn::encoder::conformer_block`'s `bd` view.
        let element = std::mem::size_of::<f32>();
        let nb1 =
            (2 * frames - 2)
                .checked_mul(element)
                .ok_or_else(|| DolphinEncoderError::Shape {
                    reason: "rel_shift nb1 overflow".to_string(),
                })?;
        let nb2 = (2 * frames - 1)
            .checked_mul(frames)
            .and_then(|value| value.checked_mul(element))
            .ok_or_else(|| DolphinEncoderError::Shape {
                reason: "rel_shift nb2 overflow".to_string(),
            })?;
        let offset =
            (frames - 1)
                .checked_mul(element)
                .ok_or_else(|| DolphinEncoderError::Shape {
                    reason: "rel_shift offset overflow".to_string(),
                })?;
        graph
            .view_3d(
                bd_raw,
                frames,
                frames,
                config.attention_heads,
                nb1,
                nb2,
                offset,
            )
            .map_err(map)?
    } else {
        bd_raw
    };
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
    let frames = subsample_len(frames_in)?;

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

/// Build and run the full encoder graph on the CPU backend. Always returns
/// `encoder_out`; when `capture_taps` is true it additionally materializes
/// `after_subsample` and every per-block hidden state for the parity harness
/// (see [`DolphinEncoderOutput`]).
///
/// Perf (P6): every per-block tap unconditionally `set_output` + f32-materialized
/// used to pin ~200MB+ of intermediate hidden state resident for the whole graph
/// (plus a host copy per block) on every production call, even though the
/// executor (`executor::run_dolphin_pipeline`) only ever reads `encoder_out` --
/// `blocks`/`after_subsample` exist solely for `#[cfg(test)]` parity
/// (`parity::dolphin_encoder_parity`). With `capture_taps: false` (the
/// production default via `executor::encode_dolphin_encoder_from_pack`), only
/// `encoder_out` is declared an output, so gallocr's liveness-based allocator is
/// free to recycle each block's buffer as soon as the next block stops reading
/// it, exactly like every other tap-free encoder in this codebase.
pub(crate) fn encode(
    config: &DolphinEncoderConfig,
    provider: &dyn DolphinWeightProvider,
    features: &[f32],
    frames_in: usize,
    backend: GgmlCpuGraphBackend,
    capture_taps: bool,
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
    // Reject a frames_in too short for the two-layer k3/s2 subsampling stem
    // before any graph/runner allocation: ggml's `conv_2d`/`im2col` asserts
    // `OH > 0` and aborts the whole process on an under-sized input rather
    // than returning a Rust error, so this must fail closed here first (see
    // `subsample_len`). A streaming FINAL over a too-short trailing window is
    // the reachable real-world trigger (short press-to-talk after idle
    // unload); the streaming driver also skips the encode call entirely in
    // that case (see `incremental_streaming_driver`), so this is defense in
    // depth for any other caller.
    let frames = subsample_len(frames_in)?;

    let graph_config = GgmlCpuGraphConfig {
        context_bytes: 64 * 1024 * 1024,
        graph_size: 16384,
        n_threads: GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            backend,
            crate::ggml_runtime::GgmlCpuGraphThreadingWorkload::EncoderPrelude,
        ),
        backend,
        // Ggml's gallocr scheduler reuses buffer space across tensors whose
        // lifetimes don't overlap instead of giving every non-view tensor its
        // own allocation; on the CPU backend both allocators produce
        // identical results, so unconditionally enabling it (like the
        // sibling cohere/moonshine encoders) only bounds memory footprint on
        // long audio, never the encoder's output.
        use_scheduler: true,
    };
    let mut runner = GgmlCpuGraphRunner::new(graph_config).map_err(ggml_err("runner_init"))?;
    // Persistent weight arena (a WEIGHTS-usage backend buffer). Placing every
    // encoder weight here -- instead of the per-call transient graph leaves the
    // pre-arena encoder used -- is what lets the ggml multi-backend scheduler
    // offload the E-Branchformer's matmuls to an accelerator (see
    // `WeightBuilder`). Mirrors `decoder_graph::DolphinDecoderRescoreRuntime` and
    // the sibling `sensevoice`/`cohere`/`moonshine` encoders. The arena is an
    // owned value carrying a raw pointer into the runner's backend, so it and the
    // per-call graph (a `&mut runner` borrow) coexist; `runner` outlives it.
    let arena = runner
        .start_static_tensor_arena(dolphin_encoder_arena_context_bytes(config.num_blocks))
        .map_err(ggml_err("static_tensor_arena"))?;

    // Phase A: allocate every weight tensor in the arena (allocation must precede
    // the arena's first upload, which freezes further creation).
    let mut builder = WeightBuilder::new(provider);
    let embed = build_embed_weights(&arena, &mut builder, config)?;
    let mut blocks = Vec::with_capacity(config.num_blocks);
    for index in 0..config.num_blocks {
        blocks.push(build_block_weights(&arena, &mut builder, config, index)?);
    }
    let after_norm_w = builder.w1(&arena, "encoder.after_norm.weight", config.d_model)?;
    let after_norm_b = builder.w1(&arena, "encoder.after_norm.bias", config.d_model)?;
    let weights = EncoderWeights {
        embed,
        blocks,
        after_norm_w,
        after_norm_b,
    };

    // The encoder's relative-position table: a baked-table slice for `CnDialect`
    // (via the provider, like every other weight), or a table computed fresh for
    // this request's `frames` for `Multilingual` (never baked -- see
    // `dolphin_relative_positional_table`). Both live in the arena (constant for
    // the whole call); the computed one is uploaded separately below since it is
    // an owned buffer, not a provider-borrowed slice `WeightBuilder` can hold.
    let is_multilingual = matches!(
        config.language_scheme,
        super::package_import::DolphinLanguageScheme::Multilingual
    );
    let mut computed_pos: Option<(GgmlStaticTensor, Vec<f32>)> = None;
    let pos_emb = if is_multilingual {
        let table = dolphin_relative_positional_table(config.d_model, frames).ok_or_else(|| {
            DolphinEncoderError::Shape {
                reason: "relative position table size overflow".to_string(),
            }
        })?;
        let handle = arena
            .new_tensor_2d_f32(config.d_model, 2 * frames - 1, "dolphin_rel_pos")
            .map_err(ggml_err("weight_alloc_relpos"))?;
        let tensor = arena.graph_tensor(handle);
        computed_pos = Some((handle, table));
        tensor
    } else {
        builder.pos_slice(
            &arena,
            "encoder.embed.pos_enc.pe",
            config.d_model,
            frames,
            config.max_positions,
        )?
    };

    // Phase B: upload every weight (+ the computed rel-pos table) into the arena
    // backend buffer exactly once. This freezes the arena.
    let mut arena = arena;
    builder.upload(&mut arena)?;
    if let Some((handle, table)) = &computed_pos {
        arena
            .set_f32_slice(*handle, table, "dolphin_rel_pos")
            .map_err(ggml_err("upload_rel_pos"))?;
    }

    // Phase C: build the per-call forward graph. Only the audio features are a
    // genuine per-call graph input; every weight and the position table are
    // already resident in the arena's backend buffer.
    let mut graph = runner.start_graph();
    let input = graph
        .new_tensor_2d_f32(feat, frames_in, "dolphin_features")
        .map_err(ggml_err("input_alloc"))?;

    let (after_subsample, frames_check) =
        subsample(&graph, input, &weights.embed, config, frames_in)?;
    if frames_check != frames {
        return Err(DolphinEncoderError::Shape {
            reason: format!("subsample produced {frames_check} frames, expected {frames}"),
        });
    }
    let mut taps: Vec<GgmlCpuTensor> = Vec::with_capacity(if capture_taps {
        config.num_blocks + 2
    } else {
        1
    });
    if capture_taps {
        taps.push(after_subsample);
    }
    let mut hidden = after_subsample;
    for block in &weights.blocks {
        hidden = encoder_block(&mut graph, hidden, pos_emb, block, config, frames)?;
        if capture_taps {
            taps.push(hidden);
        }
    }
    let encoder_out = affine_ln(
        &graph,
        hidden,
        config.layer_norm_epsilon,
        weights.after_norm_w,
        weights.after_norm_b,
        "after_norm",
    )?;
    // encoder_out is always the last (and, when `!capture_taps`, only) tap.
    taps.push(encoder_out);

    for tap in &taps {
        graph.set_output(*tap).map_err(ggml_err("set_output"))?;
    }
    // Only the audio-feature leaf is a fresh per-call graph tensor with no buffer
    // of its own; the weights and position table are arena-resident (their
    // backend buffer is already allocated), so -- like the decoder's arena path
    // -- they need no `set_input`.
    graph
        .set_input(input)
        .map_err(ggml_err("mark_input(features)"))?;
    // Allocate the forward graph through the scheduler's gallocr for
    // liveness-based buffer reuse before uploading inputs -- every tap above
    // is already marked as an output, so gallocr keeps each one's buffer
    // resident instead of recycling it once a later block stops reading it.
    graph
        .prepare_outputs_for_upload(&taps)
        .map_err(ggml_err("prepare_outputs"))?;

    // Phase D: upload the audio features, then compute.
    graph
        .set_f32_slice(input, features, "dolphin_features")
        .map_err(ggml_err("upload_features"))?;

    let expected = frames * config.d_model;
    let output_specs: Vec<(GgmlCpuTensor, usize)> =
        taps.iter().map(|tap| (*tap, expected)).collect();
    let mut outputs = graph
        .compute_outputs_f32(&output_specs)
        .map_err(ggml_err("compute"))?;

    let encoder_out = outputs.pop().expect("encoder_out tap");
    let (after_subsample, blocks) = if capture_taps {
        let blocks = outputs.split_off(1);
        let after_subsample = outputs.pop().expect("after_subsample tap");
        (after_subsample, blocks)
    } else {
        (Vec::new(), Vec::new())
    };

    Ok(DolphinEncoderOutput {
        frames,
        dim: config.d_model,
        after_subsample,
        blocks,
        encoder_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two k3/s2 (no padding) layers: `(7-3)/2+1=3` after layer one, `(3-3)/2+1=1`
    /// after layer two -- the smallest input that survives both. Regression check
    /// on `subsample_len`'s checked arithmetic (was a `saturating_sub` that
    /// silently produced a degenerate frame count for anything shorter instead
    /// of failing closed).
    #[test]
    fn minimum_subsample_input_frames_is_seven() {
        assert_eq!(minimum_subsample_input_frames(), 7);
        assert!(subsample_len(6).is_err());
        assert!(subsample_len(7).is_ok());
    }

    /// The historical crash (mac 0.1.8 field report): a streaming FINAL over a
    /// too-short trailing window (fewer frames than the Conv2dSubsampling4
    /// receptive field) reached `ggml_conv_2d`/`ggml_im2col`, which asserts
    /// `OH > 0` and `ggml_abort()`s the whole process -- not a catchable Rust
    /// panic. `encode` must reject this before ever building the graph.
    #[test]
    fn encode_rejects_frames_below_subsampling_receptive_field_instead_of_aborting() {
        let config = DolphinEncoderConfig::small_cn();
        let feat = config.feature_dim;
        let frames_in = minimum_subsample_input_frames() - 1;
        let features = vec![0.0f32; frames_in * feat];
        // The rejection happens before any weight is looked up, so an empty
        // provider is enough -- if this regresses to reading weights first,
        // the test will fail with a `MissingWeight` error instead of `Shape`.
        let provider: HashMap<String, Vec<f32>> = HashMap::new();
        let result = encode(
            &config,
            &provider,
            &features,
            frames_in,
            GgmlCpuGraphBackend::Cpu,
            false,
        );
        assert!(
            matches!(result, Err(DolphinEncoderError::Shape { .. })),
            "expected a typed Shape error for {frames_in} frames, got {result:?}"
        );
    }

    /// Pins the centered Transformer-XL `RelPositionalEncodingV1` table's shape
    /// and index direction (the multilingual dolphin encoder's only numeric
    /// path with no weights-backed parity harness): `2*frames-1` rows running
    /// from position `+(frames-1)` (row 0) down to `-(frames-1)` (last row),
    /// the center row at `frames-1` being position 0, and rows symmetric about
    /// that center differing only by `sin`'s sign (`cos` even). Catches an
    /// off-by-one in the row/position mapping without loading a pack.
    #[test]
    fn relative_positional_table_is_centered_and_sign_symmetric() {
        let d_model = 4usize;
        let frames = 3usize;
        let table = dolphin_relative_positional_table(d_model, frames).expect("table");
        let n_positions = 2 * frames - 1; // 5
        assert_eq!(table.len(), n_positions * d_model);

        let row = |idx: usize| &table[idx * d_model..(idx + 1) * d_model];

        // div_term for d_model=4: exp(0)=1 (pair 0), 10000^-0.5=0.01 (pair 1).
        let expected_for_pos = |pos: f64| {
            [
                (pos * 1.0).sin() as f32,
                (pos * 1.0).cos() as f32,
                (pos * 0.01).sin() as f32,
                (pos * 0.01).cos() as f32,
            ]
        };
        // Row 0 is the most-positive position (frames-1 = 2), last row is -2.
        for (bin, &expected) in expected_for_pos(2.0).iter().enumerate() {
            assert!((row(0)[bin] - expected).abs() < 1.0e-6, "row0 bin{bin}");
        }
        // Center row (index frames-1 = 2) is position 0: sin=0, cos=1 per pair.
        assert_eq!(row(frames - 1), &[0.0, 1.0, 0.0, 1.0]);
        // Sign symmetry: row k and row (n_positions-1-k) hold opposite positions
        // -- sin negated, cos preserved (off-by-one in the direction breaks this).
        for k in 0..n_positions {
            let mirror = n_positions - 1 - k;
            let (a, b) = (row(k), row(mirror));
            assert!((a[0] + b[0]).abs() < 1.0e-6, "sin pair0 antisymmetry k{k}");
            assert!((a[1] - b[1]).abs() < 1.0e-6, "cos pair0 symmetry k{k}");
            assert!((a[2] + b[2]).abs() < 1.0e-6, "sin pair1 antisymmetry k{k}");
            assert!((a[3] - b[3]).abs() < 1.0e-6, "cos pair1 symmetry k{k}");
        }
    }

    /// Odd `d_model` must still fill every row without indexing past the end
    /// (the loop writes `row[i+1]` only when `i + 1 < d_model`).
    #[test]
    fn relative_positional_table_handles_odd_d_model() {
        let table = dolphin_relative_positional_table(3, 2).expect("table");
        assert_eq!(table.len(), (2 * 2 - 1) * 3);
        assert!(table.iter().all(|v| v.is_finite()));
    }
}
