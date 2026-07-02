//! parakeet-ctc encoder graph (goal-1 S3c): dw-striding subsampling (cloned from
//! cohere's proven FastConformer prelude) → `conformer_block` × N (the shared
//! nn/ block) → final LayerNorm → CTC head → `[vocab, frames]` logits.
//!
//! The subsampling op sequence is a verbatim clone of cohere `encode()` (same
//! NeMo dw_striding: regular conv0+ReLU, depthwise conv2, pointwise conv3+ReLU,
//! depthwise conv5, pointwise conv6+ReLU), so the #1 frame-count risk is borrowed
//! from a proven impl. The conformer layers reuse `nn::encoder::conformer_block`
//! and the rel-pos table reuses cohere's `build_relative_positional_encoding`.

#![allow(dead_code)]

use std::path::Path;

use crate::ggml_runtime::{
    GGML_TYPE_F32, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor, GgmlLoadedTensor,
    GgmlLoadedWeightContext, GgmlStaticTensor, GgmlStaticTensorArena,
};
use crate::models::cohere::encoder_graph::{build_relative_positional_encoding, f32_to_f16_bits};
use crate::models::parakeet_ctc::graph_config::parakeet_ctc_encoder_graph_config;

const GGML_TYPE_F16: i32 = 1;
use crate::nn::conv::{
    Conv2dParams, ConvActivation, ConvBlockSteps, apply_conv_2d_bias_activation,
    apply_conv_2d_depthwise_bias_activation, reshape_bias_4d,
};
use crate::nn::encoder::{ConformerBlockConfig, ConformerBlockWeights, conformer_block};

use super::encoder_weights::{NamedTensor, ParakeetEncoderLayerWeights, ParakeetEncoderWeights};
use super::runtime_contract::ParakeetCtcExecutionMetadata;

const PARAKEET_ENCODER_GRAPH_CONTEXT_BYTES: usize = 768 * 1024 * 1024;
const ENCODER_LAYER_NORM_EPSILON: f32 = 1.0e-5;
const CONFORMER_MACARON_SCALE: f32 = 0.5;
const SUBSAMPLING_KERNEL: usize = 3;
const SUBSAMPLING_STRIDE: usize = 2;
const SUBSAMPLING_PADDING: usize = 1;

#[derive(Debug, thiserror::Error)]
pub(crate) enum ParakeetEncoderError {
    #[error("parakeet-ctc encoder graph build failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        source: GgmlCpuGraphError,
    },
    #[error("parakeet-ctc encoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("parakeet-ctc encoder shape error: {reason}")]
    Shape { reason: String },
}

/// 80-bin log-mel features for one (single-chunk) utterance: `data` is row-major
/// `[n_mels, n_frames]` (n_mels fastest), matching the encoder's mel input tensor.
#[derive(Debug, Clone)]
pub(crate) struct ParakeetMelFeatures {
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub n_mels: usize,
}

/// Encoder output: per-frame CTC logits. `logits[frame * vocab_size + v]` is the
/// logit for token `v` at output frame `frame` (ggml frame-major buffer).
#[derive(Debug, Clone)]
pub(crate) struct ParakeetCtcEncoderOutput {
    pub frame_count: usize,
    pub vocab_size: usize,
    pub logits: Vec<f32>,
}

fn bf(step: &'static str) -> impl Fn(GgmlCpuGraphError) -> ParakeetEncoderError {
    move |source| ParakeetEncoderError::GraphBuildFailed { step, source }
}

/// A 2-D linear weight: either an arena tensor (f32-uploaded — legacy / no
/// runtime pack) or a zero-copy leaf bound to the mmap'd pack (native
/// q4_K/f16/f32, no host copy + no arena upload). Goals 7+8 lever: parakeet's
/// packer reverses every rank>=2 `.weight` to ggml `[in,out]` at PACK time, so
/// the on-disk linears (`ff*`, `attn.{q,k,v,out,pos}`, `conv.pw{1,2}`, ctc head)
/// are directly `mul_mat`-consumable and safe to bind — mirroring cohere's
/// `loaded_or_static_tensor` + qwen's `bind_or_arena_llm`. NOT bound: the
/// BN-folded depthwise conv (its host values are mutated at load), 1-D
/// norms/biases, the rank-3/4 subsampling convs, and `attn.pos_bias_u/v`.
#[derive(Clone, Copy)]
enum WeightSlot {
    Arena(GgmlStaticTensor),
    Loaded(GgmlLoadedTensor),
}

impl WeightSlot {
    fn graph<'a>(self, arena: &GgmlStaticTensorArena) -> GgmlCpuTensor<'a> {
        match self {
            Self::Arena(handle) => arena.graph_tensor(handle),
            Self::Loaded(tensor) => tensor.as_graph_tensor(),
        }
    }
}

/// Bind a 2-D linear zero-copy from the mmap'd pack (`loaded`) by its on-disk
/// name. FAILS CLOSED if the loaded context is absent or the tensor is missing:
/// the host f32 `values` for bound weights are dropped at load, so there is no
/// arena fallback — uploading an empty buffer would silently corrupt the graph.
fn bind_loaded(
    loaded: Option<&GgmlLoadedWeightContext>,
    name: &str,
) -> Result<WeightSlot, ParakeetEncoderError> {
    match loaded.and_then(|ctx| ctx.tensor(name)) {
        Some(tensor) => Ok(WeightSlot::Loaded(tensor)),
        None => Err(ParakeetEncoderError::Shape {
            reason: format!(
                "2-D linear '{name}' could not be bound zero-copy from the runtime pack \
                 (loaded weight context missing or tensor absent); host payload was dropped"
            ),
        }),
    }
}

fn conv_out_dim(input: usize) -> usize {
    (input + 2 * SUBSAMPLING_PADDING - SUBSAMPLING_KERNEL) / SUBSAMPLING_STRIDE + 1
}

/// Allocate a static arena tensor matching a host weight's stored dims (the same
/// layout the importer wrote + cohere's loader uses, so `conformer_block`
/// consumes it identically).
fn alloc_static(
    arena: &GgmlStaticTensorArena,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<GgmlStaticTensor, ParakeetEncoderError> {
    let map = |source| ParakeetEncoderError::GraphBuildFailed { step, source };
    match weight.dims.as_slice() {
        [] | [_] => arena
            .new_tensor_1d_f32(weight.values.len(), step)
            .map_err(map),
        [ne0, ne1] => arena.new_tensor_2d_f32(*ne0, *ne1, step).map_err(map),
        [ne0, ne1, ne2] => arena.new_tensor_3d_f32(*ne0, *ne1, *ne2, step).map_err(map),
        [ne0, ne1, ne2, ne3] => arena
            .new_tensor_4d_typed(*ne0, *ne1, *ne2, *ne3, GGML_TYPE_F32, step)
            .map_err(map),
        _ => Err(ParakeetEncoderError::Shape {
            reason: format!(
                "tensor '{}' has unsupported rank {:?}",
                weight.name, weight.dims
            ),
        }),
    }
}

fn upload_static(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<(), ParakeetEncoderError> {
    arena
        .set_f32_slice(tensor, &weight.values, step)
        .map_err(|source| ParakeetEncoderError::GraphBuildFailed { step, source })
}

/// Allocate an f16 arena tensor for a depthwise conv kernel (ggml `conv_2d_dw`
/// requires an f16 kernel; regular `conv_2d` accepts f32).
fn alloc_static_f16(
    arena: &GgmlStaticTensorArena,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<GgmlStaticTensor, ParakeetEncoderError> {
    let map = |source| ParakeetEncoderError::GraphBuildFailed { step, source };
    match weight.dims.as_slice() {
        [ne0, ne1, ne2] => arena
            .new_tensor_3d_typed(*ne0, *ne1, *ne2, GGML_TYPE_F16, step)
            .map_err(map),
        [ne0, ne1, ne2, ne3] => arena
            .new_tensor_4d_typed(*ne0, *ne1, *ne2, *ne3, GGML_TYPE_F16, step)
            .map_err(map),
        _ => Err(ParakeetEncoderError::Shape {
            reason: format!("f16 depthwise '{}' rank {:?}", weight.name, weight.dims),
        }),
    }
}

fn upload_static_f16(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<(), ParakeetEncoderError> {
    let bits: Vec<u16> = weight.values.iter().copied().map(f32_to_f16_bits).collect();
    arena
        .set_f16_bits_slice(tensor, &bits, step)
        .map_err(|source| ParakeetEncoderError::GraphBuildFailed { step, source })
}

/// Per-layer handles for the conformer block weights. The 2-D linears
/// (`ff*.{up,down}`, `attn.{q,k,v,out,pos}`, `conv.pw{1,2}`) are `WeightSlot`
/// (bound zero-copy from the pack when a runtime path is supplied; else arena);
/// norms, biases, and the BN-folded depthwise conv stay plain arena tensors.
struct LayerArena {
    ff1_norm_weight: GgmlStaticTensor,
    ff1_norm_bias: GgmlStaticTensor,
    ff1_up_weight: WeightSlot,
    ff1_up_bias: GgmlStaticTensor,
    ff1_down_weight: WeightSlot,
    ff1_down_bias: GgmlStaticTensor,
    attn_norm_weight: GgmlStaticTensor,
    attn_norm_bias: GgmlStaticTensor,
    attn_q_weight: WeightSlot,
    attn_q_bias: GgmlStaticTensor,
    attn_k_weight: WeightSlot,
    attn_k_bias: GgmlStaticTensor,
    attn_v_weight: WeightSlot,
    attn_v_bias: GgmlStaticTensor,
    attn_out_weight: WeightSlot,
    attn_out_bias: GgmlStaticTensor,
    attn_pos_weight: WeightSlot,
    attn_pos_bias_u: GgmlStaticTensor,
    attn_pos_bias_v: GgmlStaticTensor,
    conv_norm_weight: GgmlStaticTensor,
    conv_norm_bias: GgmlStaticTensor,
    conv_pw1_weight: WeightSlot,
    conv_pw1_bias: GgmlStaticTensor,
    conv_dw_weight: GgmlStaticTensor,
    conv_dw_bias: GgmlStaticTensor,
    conv_pw2_weight: WeightSlot,
    conv_pw2_bias: GgmlStaticTensor,
    ff2_norm_weight: GgmlStaticTensor,
    ff2_norm_bias: GgmlStaticTensor,
    ff2_up_weight: WeightSlot,
    ff2_up_bias: GgmlStaticTensor,
    ff2_down_weight: WeightSlot,
    ff2_down_bias: GgmlStaticTensor,
    out_norm_weight: GgmlStaticTensor,
    out_norm_bias: GgmlStaticTensor,
}

struct SubArena {
    conv0_w: GgmlStaticTensor,
    conv0_b: GgmlStaticTensor,
    conv2_w: GgmlStaticTensor,
    conv2_b: GgmlStaticTensor,
    conv3_w: GgmlStaticTensor,
    conv3_b: GgmlStaticTensor,
    conv5_w: GgmlStaticTensor,
    conv5_b: GgmlStaticTensor,
    conv6_w: GgmlStaticTensor,
    conv6_b: GgmlStaticTensor,
    linear_w: WeightSlot,
    linear_b: GgmlStaticTensor,
    conv6_channels: usize,
}

pub(crate) struct ParakeetCtcEncoderGraph {
    metadata: ParakeetCtcExecutionMetadata,
    runner: GgmlCpuGraphRunner,
    // `loaded_weights` owns the mmap-backed buffer that the `Loaded` weight slots
    // below alias. Rust drops struct fields in declaration order (first-declared
    // drops first), so declaring it here does NOT make it outlive `arena`;
    // soundness relies on neither `arena` nor `runner` dereferencing weight memory
    // on drop. Field order mirrors cohere/qwen for consistency.
    loaded_weights: Option<GgmlLoadedWeightContext>,
    arena: GgmlStaticTensorArena,
    sub: SubArena,
    layers: Vec<LayerArena>,
    ctc_head_weight: WeightSlot,
    ctc_head_bias: GgmlStaticTensor,
}

fn find_sub<'a>(
    weights: &'a ParakeetEncoderWeights,
    name: &str,
) -> Result<&'a NamedTensor, ParakeetEncoderError> {
    weights
        .subsampling
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| ParakeetEncoderError::Shape {
            reason: format!("missing subsampling tensor '{name}'"),
        })
}

impl ParakeetCtcEncoderGraph {
    pub(crate) fn new(
        weights: &ParakeetEncoderWeights,
        metadata: ParakeetCtcExecutionMetadata,
        runtime_path: Option<&Path>,
    ) -> Result<Self, ParakeetEncoderError> {
        let mut config = parakeet_ctc_encoder_graph_config();
        config.context_bytes = PARAKEET_ENCODER_GRAPH_CONTEXT_BYTES;
        // FastConformer-XL (parakeet-ctc-1.1b, ~42 conformer layers) builds more
        // graph nodes than the default 4096-node cap, tripping
        // `GGML_ASSERT(cgraph->n_nodes < cgraph->size)`. Size the cgraph to the
        // actual (data-driven) layer count with generous per-layer headroom. This
        // is capacity only — the built graph and its op order are identical, so
        // the 24-layer 0.6b model's output stays byte-for-byte unchanged.
        config.graph_size = config.graph_size.max(weights.layers.len() * 256 + 2048);
        let runner = GgmlCpuGraphRunner::new(config).map_err(|source| {
            ParakeetEncoderError::GraphBuildFailed {
                step: "runner_init",
                source,
            }
        })?;
        // Goals 7+8 lever: bind the 2-D linears zero-copy from the mmap'd pack
        // (no f32 dequantize-to-host, no arena upload). Fails closed below if the
        // load failed but a bindable weight's host payload was dropped at load.
        let loaded_weights =
            runtime_path.and_then(|path| runner.load_gguf_weight_context(path).ok());
        let loaded = loaded_weights.as_ref();
        let mut arena = runner
            .start_static_tensor_arena(PARAKEET_ENCODER_GRAPH_CONTEXT_BYTES)
            .map_err(|source| ParakeetEncoderError::GraphBuildFailed {
                step: "static_tensor_arena",
                source,
            })?;

        // ----- declare (allocate) all arena tensors first (first upload freezes) -----
        let s = |n: &str| find_sub(weights, n);
        let conv0_w_t = alloc_static(&arena, s("enc.sub.layers.0.weight")?, "sub0_w")?;
        let conv0_b_t = alloc_static(&arena, s("enc.sub.layers.0.bias")?, "sub0_b")?;
        let conv2_w_t = alloc_static_f16(&arena, s("enc.sub.layers.2.weight")?, "sub2_w")?;
        let conv2_b_t = alloc_static(&arena, s("enc.sub.layers.2.bias")?, "sub2_b")?;
        let conv3_w_t = alloc_static(&arena, s("enc.sub.layers.3.weight")?, "sub3_w")?;
        let conv3_b_t = alloc_static(&arena, s("enc.sub.layers.3.bias")?, "sub3_b")?;
        let conv5_w_t = alloc_static_f16(&arena, s("enc.sub.layers.5.weight")?, "sub5_w")?;
        let conv5_b_t = alloc_static(&arena, s("enc.sub.layers.5.bias")?, "sub5_b")?;
        let conv6_w_t = alloc_static(&arena, s("enc.sub.layers.6.weight")?, "sub6_w")?;
        let conv6_b_t = alloc_static(&arena, s("enc.sub.layers.6.bias")?, "sub6_b")?;
        let linear_w_slot = bind_loaded(loaded, "enc.sub.linear.weight")?;
        let linear_b_t = alloc_static(&arena, s("enc.sub.linear.bias")?, "sub_lin_b")?;
        let conv6_channels = s("enc.sub.layers.6.bias")?.values.len();

        let mut layer_arenas = Vec::with_capacity(weights.layers.len());
        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            layer_arenas.push(alloc_layer(&arena, loaded, layer_idx, layer)?);
        }
        // CTC head: `ctc.head.weight` is f16 `[1, d_model, vocab]` on disk (the
        // packer's reversed-dims convention) — bound zero-copy + reshaped to
        // `[d_model, vocab]` for the head matmul. Its bias stays arena (1-D f32).
        let ctc_head_weight_slot = bind_loaded(loaded, &weights.ctc_head_weight.name)?;
        let ctc_head_bias_t = alloc_static(&arena, &weights.ctc_head_bias, "ctc_head_b")?;

        // ----- upload all values (arena now freezes on first set) -----
        upload_static(
            &mut arena,
            conv0_w_t,
            s("enc.sub.layers.0.weight")?,
            "sub0_w",
        )?;
        upload_static(&mut arena, conv0_b_t, s("enc.sub.layers.0.bias")?, "sub0_b")?;
        upload_static_f16(
            &mut arena,
            conv2_w_t,
            s("enc.sub.layers.2.weight")?,
            "sub2_w",
        )?;
        upload_static(&mut arena, conv2_b_t, s("enc.sub.layers.2.bias")?, "sub2_b")?;
        upload_static(
            &mut arena,
            conv3_w_t,
            s("enc.sub.layers.3.weight")?,
            "sub3_w",
        )?;
        upload_static(&mut arena, conv3_b_t, s("enc.sub.layers.3.bias")?, "sub3_b")?;
        upload_static_f16(
            &mut arena,
            conv5_w_t,
            s("enc.sub.layers.5.weight")?,
            "sub5_w",
        )?;
        upload_static(&mut arena, conv5_b_t, s("enc.sub.layers.5.bias")?, "sub5_b")?;
        upload_static(
            &mut arena,
            conv6_w_t,
            s("enc.sub.layers.6.weight")?,
            "sub6_w",
        )?;
        upload_static(&mut arena, conv6_b_t, s("enc.sub.layers.6.bias")?, "sub6_b")?;
        // `enc.sub.linear.weight` is bound zero-copy; only its bias is arena-uploaded.
        upload_static(
            &mut arena,
            linear_b_t,
            s("enc.sub.linear.bias")?,
            "sub_lin_b",
        )?;
        for (layer, handles) in weights.layers.iter().zip(&layer_arenas) {
            upload_layer(&mut arena, layer, handles)?;
        }
        // ctc head weight is bound zero-copy (no upload); only its bias is arena.
        upload_static(
            &mut arena,
            ctc_head_bias_t,
            &weights.ctc_head_bias,
            "ctc_head_b",
        )?;

        Ok(Self {
            metadata,
            runner,
            loaded_weights,
            arena,
            sub: SubArena {
                conv0_w: conv0_w_t,
                conv0_b: conv0_b_t,
                conv2_w: conv2_w_t,
                conv2_b: conv2_b_t,
                conv3_w: conv3_w_t,
                conv3_b: conv3_b_t,
                conv5_w: conv5_w_t,
                conv5_b: conv5_b_t,
                conv6_w: conv6_w_t,
                conv6_b: conv6_b_t,
                linear_w: linear_w_slot,
                linear_b: linear_b_t,
                conv6_channels,
            },
            layers: layer_arenas,
            ctc_head_weight: ctc_head_weight_slot,
            ctc_head_bias: ctc_head_bias_t,
        })
    }

    pub(crate) fn encode(
        &mut self,
        mel: &ParakeetMelFeatures,
    ) -> Result<ParakeetCtcEncoderOutput, ParakeetEncoderError> {
        let metadata = self.metadata;
        let d_model = metadata.hidden_size;
        let subsampled_frames = conv_out_dim(conv_out_dim(conv_out_dim(mel.n_frames)));
        let subsampled_freq = conv_out_dim(conv_out_dim(conv_out_dim(mel.n_mels)));
        let positional =
            build_relative_positional_encoding(d_model, subsampled_frames).map_err(|e| {
                ParakeetEncoderError::Shape {
                    reason: e.to_string(),
                }
            })?;

        let mut graph = self.runner.start_graph();

        let mel_t = graph
            .new_tensor_2d_f32(mel.n_mels, mel.n_frames, "parakeet_mel")
            .map_err(bf("new_mel"))?;
        let pos_t = graph
            .new_tensor_2d_f32(d_model, positional.len() / d_model, "parakeet_rel_pos")
            .map_err(bf("new_pos"))?;
        graph.set_input(mel_t).map_err(bf("set_input_mel"))?;
        graph.set_input(pos_t).map_err(bf("set_input_pos"))?;

        let conv_map = |step, source| ParakeetEncoderError::GraphBuildFailed { step, source };
        let stride2 = Conv2dParams {
            stride_x: 2,
            stride_y: 2,
            padding_x: 1,
            padding_y: 1,
            dilation_x: 1,
            dilation_y: 1,
        };
        let pointwise = Conv2dParams {
            stride_x: 1,
            stride_y: 1,
            padding_x: 0,
            padding_y: 0,
            dilation_x: 1,
            dilation_y: 1,
        };
        let bias4d = |g: &_, t: GgmlStaticTensor, len: usize, step| {
            reshape_bias_4d(g, self.arena.graph_tensor(t), len, step, conv_map)
        };

        // ----- dw-striding subsampling (verbatim cohere FastConformer prelude) -----
        let mut state_4d = graph
            .reshape_4d(mel_t, mel.n_mels, mel.n_frames, 1, 1)
            .map_err(bf("reshape_mel_4d"))?;
        let conv0_b = bias4d(
            &graph,
            self.sub.conv0_b,
            metadata.subsampling_channels,
            "sub0_bias4d",
        )?;
        state_4d = apply_conv_2d_bias_activation(
            &graph,
            self.arena.graph_tensor(self.sub.conv0_w),
            state_4d,
            conv0_b,
            stride2,
            ConvActivation::Relu,
            ConvBlockSteps {
                conv: "conv0",
                bias: "conv0_bias",
                activation: "conv0_relu",
            },
            conv_map,
        )?;
        let conv2_b = bias4d(
            &graph,
            self.sub.conv2_b,
            metadata.subsampling_channels,
            "sub2_bias4d",
        )?;
        state_4d = apply_conv_2d_depthwise_bias_activation(
            &graph,
            self.arena.graph_tensor(self.sub.conv2_w),
            state_4d,
            conv2_b,
            stride2,
            None,
            ConvBlockSteps {
                conv: "conv2_dw",
                bias: "conv2_bias",
                activation: "conv2_noact",
            },
            conv_map,
        )?;
        let conv3_b = bias4d(
            &graph,
            self.sub.conv3_b,
            metadata.subsampling_channels,
            "sub3_bias4d",
        )?;
        state_4d = apply_conv_2d_bias_activation(
            &graph,
            self.arena.graph_tensor(self.sub.conv3_w),
            state_4d,
            conv3_b,
            pointwise,
            ConvActivation::Relu,
            ConvBlockSteps {
                conv: "conv3_pw",
                bias: "conv3_bias",
                activation: "conv3_relu",
            },
            conv_map,
        )?;
        let conv5_b = bias4d(
            &graph,
            self.sub.conv5_b,
            metadata.subsampling_channels,
            "sub5_bias4d",
        )?;
        state_4d = apply_conv_2d_depthwise_bias_activation(
            &graph,
            self.arena.graph_tensor(self.sub.conv5_w),
            state_4d,
            conv5_b,
            stride2,
            None,
            ConvBlockSteps {
                conv: "conv5_dw",
                bias: "conv5_bias",
                activation: "conv5_noact",
            },
            conv_map,
        )?;
        let conv6_b = bias4d(
            &graph,
            self.sub.conv6_b,
            metadata.subsampling_channels,
            "sub6_bias4d",
        )?;
        state_4d = apply_conv_2d_bias_activation(
            &graph,
            self.arena.graph_tensor(self.sub.conv6_w),
            state_4d,
            conv6_b,
            pointwise,
            ConvActivation::Relu,
            ConvBlockSteps {
                conv: "conv6_pw",
                bias: "conv6_bias",
                activation: "conv6_relu",
            },
            conv_map,
        )?;

        // flatten [channels, freq] per frame -> [channels*freq, frames] -> linear -> d_model.
        let flattened = self
            .sub
            .conv6_channels
            .checked_mul(subsampled_freq)
            .ok_or_else(|| ParakeetEncoderError::Shape {
                reason: "flatten overflow".into(),
            })?;
        let mut state = graph
            .permute(state_4d, 0, 2, 1, 3)
            .map_err(bf("permute_flatten"))?;
        state = graph.cont(state).map_err(bf("cont_flatten"))?;
        state = graph
            .reshape_2d(state, flattened, subsampled_frames)
            .map_err(bf("reshape_flatten"))?;
        state = graph
            .mul_mat(self.sub.linear_w.graph(&self.arena), state)
            .map_err(bf("sub_linear"))?;
        state = graph
            .add(state, self.arena.graph_tensor(self.sub.linear_b))
            .map_err(bf("sub_linear_bias"))?;
        // scale_input: x *= sqrt(d_model)  (NeMo FastConformer).
        state = graph
            .scale(state, (d_model as f32).sqrt())
            .map_err(bf("scale_input"))?;

        // ----- conformer layers (shared nn/ block) -----
        let element = std::mem::size_of::<f32>();
        let frame = subsampled_frames;
        let config = ConformerBlockConfig {
            d_model,
            attention_heads: metadata.n_heads,
            head_dim: metadata.head_dim,
            frame_count: frame,
            conv_kernel: metadata.conv_kernel,
            layer_norm_epsilon: ENCODER_LAYER_NORM_EPSILON,
            macaron_scale: CONFORMER_MACARON_SCALE,
            rel_shift_nb1: (2 * frame - 2) * element,
            rel_shift_nb2: (2 * frame - 1) * frame * element,
            rel_shift_offset: (frame - 1) * element,
        };
        let pos_enc = pos_t;
        for handles in &self.layers {
            let weights = conformer_weights(&self.arena, handles);
            let block = conformer_block(&mut graph, state, pos_enc, config, weights, conv_map)?;
            state = block.output;
        }

        // ----- CTC head -----
        // No extra final norm: conformer_block already applies its per-layer
        // out_norm as its last step, and parakeet has no separate encoder-level
        // final-norm tensor — the CTC head consumes the last block's output.
        let head = graph
            .reshape_2d(
                self.ctc_head_weight.graph(&self.arena),
                d_model,
                metadata.vocab_size,
            )
            .map_err(bf("ctc_head_reshape"))?;
        let mut logits = graph.mul_mat(head, state).map_err(bf("ctc_head_matmul"))?;
        logits = graph
            .add(logits, self.arena.graph_tensor(self.ctc_head_bias))
            .map_err(bf("ctc_head_bias"))?;
        graph.set_output(logits).map_err(bf("set_output_logits"))?;

        // Peak-RSS lever: allocate the compute graph via the scheduler's gallocr
        // (liveness-based buffer REUSE) before uploading inputs, instead of the
        // per-tensor alloc_ctx_tensors that the uploads would otherwise force
        // (RSS = sum over conformer layers). Collapses peak RSS to the working set.
        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(bf("prepare_outputs"))?;
        upload_graph_f32(&mut graph, mel_t, &mel.data, "upload_mel")?;
        upload_graph_f32(&mut graph, pos_t, &positional, "upload_pos")?;

        let want = metadata
            .vocab_size
            .checked_mul(subsampled_frames)
            .ok_or_else(|| ParakeetEncoderError::Shape {
                reason: "logits overflow".into(),
            })?;
        let logits = graph.compute_output_f32(logits, want).map_err(|error| {
            ParakeetEncoderError::GraphExecutionFailed {
                reason: error.to_string(),
            }
        })?;
        Ok(ParakeetCtcEncoderOutput {
            frame_count: subsampled_frames,
            vocab_size: metadata.vocab_size,
            logits,
        })
    }
}

fn upload_graph_f32<'a>(
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    tensor: GgmlCpuTensor<'a>,
    values: &[f32],
    step: &'static str,
) -> Result<(), ParakeetEncoderError> {
    graph
        .set_f32_slice(tensor, values, step)
        .map_err(|source| ParakeetEncoderError::GraphBuildFailed { step, source })
}

/// Allocate one conformer layer's arena tensors + bind its 2-D linears zero-copy
/// from the mmap'd pack. The bound set (`ff*.{up,down}`, `attn.{q,k,v,out,pos}`,
/// `conv.pw{1,2}`) skips its `alloc_static`+`upload_static`: each is `[in,out]`
/// native on disk (the packer reversed dims) so it is bound + drawn straight from
/// the pack at `mul_mat`. Fails closed (`bind_loaded`) when a bound weight is
/// absent — its host payload was dropped at load. `layer_idx` is unused for
/// binding (weights carry their own `enc.blk.{i}.*` names) but threaded for
/// parity with the qwen/cohere mirrors.
fn alloc_layer(
    arena: &GgmlStaticTensorArena,
    loaded: Option<&GgmlLoadedWeightContext>,
    _layer_idx: usize,
    layer: &ParakeetEncoderLayerWeights,
) -> Result<LayerArena, ParakeetEncoderError> {
    Ok(LayerArena {
        ff1_norm_weight: alloc_static(arena, &layer.ff1_norm_weight, "ff1_norm_w")?,
        ff1_norm_bias: alloc_static(arena, &layer.ff1_norm_bias, "ff1_norm_b")?,
        ff1_up_weight: bind_loaded(loaded, &layer.ff1_up_weight.name)?,
        ff1_up_bias: alloc_static(arena, &layer.ff1_up_bias, "ff1_up_b")?,
        ff1_down_weight: bind_loaded(loaded, &layer.ff1_down_weight.name)?,
        ff1_down_bias: alloc_static(arena, &layer.ff1_down_bias, "ff1_down_b")?,
        attn_norm_weight: alloc_static(arena, &layer.attn_norm_weight, "attn_norm_w")?,
        attn_norm_bias: alloc_static(arena, &layer.attn_norm_bias, "attn_norm_b")?,
        attn_q_weight: bind_loaded(loaded, &layer.attn_q_weight.name)?,
        attn_q_bias: alloc_static(arena, &layer.attn_q_bias, "attn_q_b")?,
        attn_k_weight: bind_loaded(loaded, &layer.attn_k_weight.name)?,
        attn_k_bias: alloc_static(arena, &layer.attn_k_bias, "attn_k_b")?,
        attn_v_weight: bind_loaded(loaded, &layer.attn_v_weight.name)?,
        attn_v_bias: alloc_static(arena, &layer.attn_v_bias, "attn_v_b")?,
        attn_out_weight: bind_loaded(loaded, &layer.attn_out_weight.name)?,
        attn_out_bias: alloc_static(arena, &layer.attn_out_bias, "attn_out_b")?,
        attn_pos_weight: bind_loaded(loaded, &layer.attn_pos_weight.name)?,
        attn_pos_bias_u: alloc_static(arena, &layer.attn_pos_bias_u, "attn_pos_u")?,
        attn_pos_bias_v: alloc_static(arena, &layer.attn_pos_bias_v, "attn_pos_v")?,
        conv_norm_weight: alloc_static(arena, &layer.conv_norm_weight, "conv_norm_w")?,
        conv_norm_bias: alloc_static(arena, &layer.conv_norm_bias, "conv_norm_b")?,
        conv_pw1_weight: bind_loaded(loaded, &layer.conv_pw1_weight.name)?,
        conv_pw1_bias: alloc_static(arena, &layer.conv_pw1_bias, "conv_pw1_b")?,
        conv_dw_weight: alloc_static_f16(arena, &layer.conv_dw_weight, "conv_dw_w")?,
        conv_dw_bias: alloc_static(arena, &layer.conv_dw_bias, "conv_dw_b")?,
        conv_pw2_weight: bind_loaded(loaded, &layer.conv_pw2_weight.name)?,
        conv_pw2_bias: alloc_static(arena, &layer.conv_pw2_bias, "conv_pw2_b")?,
        ff2_norm_weight: alloc_static(arena, &layer.ff2_norm_weight, "ff2_norm_w")?,
        ff2_norm_bias: alloc_static(arena, &layer.ff2_norm_bias, "ff2_norm_b")?,
        ff2_up_weight: bind_loaded(loaded, &layer.ff2_up_weight.name)?,
        ff2_up_bias: alloc_static(arena, &layer.ff2_up_bias, "ff2_up_b")?,
        ff2_down_weight: bind_loaded(loaded, &layer.ff2_down_weight.name)?,
        ff2_down_bias: alloc_static(arena, &layer.ff2_down_bias, "ff2_down_b")?,
        out_norm_weight: alloc_static(arena, &layer.out_norm_weight, "out_norm_w")?,
        out_norm_bias: alloc_static(arena, &layer.out_norm_bias, "out_norm_b")?,
    })
}

/// Upload one layer's ARENA tensors (norms, biases, BN-folded depthwise conv).
/// The 2-D linears are bound zero-copy in `alloc_layer`, so they are absent here
/// (their host f32 `values` were dropped at load — see `encoder_weights`).
fn upload_layer(
    arena: &mut GgmlStaticTensorArena,
    layer: &ParakeetEncoderLayerWeights,
    h: &LayerArena,
) -> Result<(), ParakeetEncoderError> {
    upload_static_f16(arena, h.conv_dw_weight, &layer.conv_dw_weight, "conv_dw_w")?;
    let pairs: [(GgmlStaticTensor, &NamedTensor); 23] = [
        (h.ff1_norm_weight, &layer.ff1_norm_weight),
        (h.ff1_norm_bias, &layer.ff1_norm_bias),
        (h.ff1_up_bias, &layer.ff1_up_bias),
        (h.ff1_down_bias, &layer.ff1_down_bias),
        (h.attn_norm_weight, &layer.attn_norm_weight),
        (h.attn_norm_bias, &layer.attn_norm_bias),
        (h.attn_q_bias, &layer.attn_q_bias),
        (h.attn_k_bias, &layer.attn_k_bias),
        (h.attn_v_bias, &layer.attn_v_bias),
        (h.attn_out_bias, &layer.attn_out_bias),
        (h.attn_pos_bias_u, &layer.attn_pos_bias_u),
        (h.attn_pos_bias_v, &layer.attn_pos_bias_v),
        (h.conv_norm_weight, &layer.conv_norm_weight),
        (h.conv_norm_bias, &layer.conv_norm_bias),
        (h.conv_pw1_bias, &layer.conv_pw1_bias),
        (h.conv_dw_bias, &layer.conv_dw_bias),
        (h.conv_pw2_bias, &layer.conv_pw2_bias),
        (h.ff2_norm_weight, &layer.ff2_norm_weight),
        (h.ff2_norm_bias, &layer.ff2_norm_bias),
        (h.ff2_up_bias, &layer.ff2_up_bias),
        (h.ff2_down_bias, &layer.ff2_down_bias),
        (h.out_norm_weight, &layer.out_norm_weight),
        (h.out_norm_bias, &layer.out_norm_bias),
    ];
    for (tensor, weight) in pairs {
        upload_static(arena, tensor, weight, "layer_weight")?;
    }
    Ok(())
}

fn conformer_weights<'a>(
    arena: &'a GgmlStaticTensorArena,
    h: &LayerArena,
) -> ConformerBlockWeights<'a> {
    let g = |t: GgmlStaticTensor| arena.graph_tensor(t);
    // Bound 2-D linears resolve to their mmap'd leaf (or arena fallback) via the
    // `WeightSlot` handle; everything else is a plain arena tensor.
    let b = |slot: WeightSlot| slot.graph(arena);
    ConformerBlockWeights {
        ff1_norm_weight: g(h.ff1_norm_weight),
        ff1_norm_bias: g(h.ff1_norm_bias),
        ff1_up_weight: b(h.ff1_up_weight),
        ff1_up_bias: g(h.ff1_up_bias),
        ff1_down_weight: b(h.ff1_down_weight),
        ff1_down_bias: g(h.ff1_down_bias),
        attn_norm_weight: g(h.attn_norm_weight),
        attn_norm_bias: g(h.attn_norm_bias),
        attn_q_weight: b(h.attn_q_weight),
        attn_q_bias: g(h.attn_q_bias),
        attn_k_weight: b(h.attn_k_weight),
        attn_k_bias: g(h.attn_k_bias),
        attn_v_weight: b(h.attn_v_weight),
        attn_v_bias: g(h.attn_v_bias),
        attn_out_weight: b(h.attn_out_weight),
        attn_out_bias: g(h.attn_out_bias),
        attn_pos_weight: b(h.attn_pos_weight),
        attn_pos_bias_u: g(h.attn_pos_bias_u),
        attn_pos_bias_v: g(h.attn_pos_bias_v),
        conv_norm_weight: g(h.conv_norm_weight),
        conv_norm_bias: g(h.conv_norm_bias),
        conv_pw1_weight: b(h.conv_pw1_weight),
        conv_pw1_bias: g(h.conv_pw1_bias),
        conv_dw_weight: g(h.conv_dw_weight),
        conv_dw_bias: g(h.conv_dw_bias),
        conv_pw2_weight: b(h.conv_pw2_weight),
        conv_pw2_bias: g(h.conv_pw2_bias),
        ff2_norm_weight: g(h.ff2_norm_weight),
        ff2_norm_bias: g(h.ff2_norm_bias),
        ff2_up_weight: b(h.ff2_up_weight),
        ff2_up_bias: g(h.ff2_up_bias),
        ff2_down_weight: b(h.ff2_down_weight),
        ff2_down_bias: g(h.ff2_down_bias),
        out_norm_weight: g(h.out_norm_weight),
        out_norm_bias: g(h.out_norm_bias),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::GgufTensorDataReader;
    use crate::models::parakeet_ctc::encoder_weights::load_parakeet_ctc_encoder_weights;
    use crate::models::parakeet_ctc::runtime_contract::parse_parakeet_ctc_execution_metadata;
    use std::path::Path;

    /// S3c gate: build the full encoder graph from the real pack + run it on a
    /// dummy mel; assert finite [vocab, frames] logits with no ggml shape error
    /// (validates subsampling frame-count + conformer + head wiring). WER
    /// correctness is the S4 gate. Skipped when the pack is absent.
    #[test]
    fn encoder_graph_smoke_produces_finite_logits_when_pack_present() {
        let candidates = [Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/models/parakeet-ctc-0.6b/openasr/parakeet-ctc-0.6b-fp16.oasr")];
        let Some(path) = candidates.into_iter().find(|p| p.exists()) else {
            eprintln!("skipping: parakeet-ctc-0.6b pack not present");
            return;
        };
        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        let gguf_metadata = crate::ggml_runtime::read_gguf_metadata(&path).expect("gguf metadata");
        let metadata = parse_parakeet_ctc_execution_metadata(&gguf_metadata).expect("metadata");
        let weights = load_parakeet_ctc_encoder_weights(&reader, &metadata).expect("weights");

        let mut graph =
            ParakeetCtcEncoderGraph::new(&weights, metadata, Some(path.as_path())).expect("graph");
        let n_frames = 128usize;
        let mel = ParakeetMelFeatures {
            data: (0..metadata.n_mels * n_frames)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
                .collect(),
            n_frames,
            n_mels: metadata.n_mels,
        };
        let out = graph.encode(&mel).expect("encode");
        let expected_frames = conv_out_dim(conv_out_dim(conv_out_dim(n_frames)));
        assert_eq!(out.frame_count, expected_frames);
        assert_eq!(out.vocab_size, metadata.vocab_size);
        assert_eq!(out.logits.len(), metadata.vocab_size * expected_frames);
        assert!(
            out.logits.iter().all(|v| v.is_finite()),
            "logits must be finite"
        );
    }
}
