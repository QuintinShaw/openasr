//! parakeet-tdt encoder graph: the parakeet-ctc FastConformer encoder
//! (dw-striding subsampling prelude + shared `nn::encoder::conformer_block`)
//! with the TDT differences: `scale_input` honored from pack metadata (false
//! for v3) and the joint ENCODER PROJECTION (`enc.proj`, d_model -> joint
//! hidden) applied in-graph instead of a CTC head. Output is the per-frame
//! projected encoder representation the host-side TDT greedy loop consumes.

use std::path::Path;

use crate::ggml_runtime::{
    ArenaAllocError, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor, GgmlLoadedTensor,
    GgmlLoadedWeightContext, GgmlStaticTensor, GgmlStaticTensorArena,
    alloc_static_f16 as arena_alloc_static_f16, alloc_static_f32 as arena_alloc_static_f32,
    bind_loaded as arena_bind_loaded, upload_static_f16 as arena_upload_static_f16,
    upload_static_f32 as arena_upload_static_f32,
};
use crate::models::parakeet_tdt::graph_config::parakeet_tdt_encoder_graph_config;

use crate::nn::conv::{
    Conv2dParams, ConvActivation, ConvBlockSteps, apply_conv_2d_bias_activation,
    apply_conv_2d_depthwise_bias_activation, reshape_bias_4d,
};
use crate::nn::encoder::{
    ConformerBlockConfig, ConformerBlockWeights, build_relative_positional_encoding,
    conformer_block,
};
use crate::nn::half::f32_to_f16_bits;

use super::encoder_weights::{
    NamedTensor, ParakeetTdtEncoderLayerWeights, ParakeetTdtEncoderWeights,
};
use super::runtime_contract::ParakeetTdtExecutionMetadata;

const PARAKEET_TDT_ENCODER_GRAPH_CONTEXT_BYTES: usize = 768 * 1024 * 1024;
const ENCODER_LAYER_NORM_EPSILON: f32 = 1.0e-5;
const CONFORMER_MACARON_SCALE: f32 = 0.5;
const SUBSAMPLING_KERNEL: usize = 3;
const SUBSAMPLING_STRIDE: usize = 2;
const SUBSAMPLING_PADDING: usize = 1;

#[derive(Debug, thiserror::Error)]
pub(crate) enum ParakeetTdtEncoderError {
    #[error("parakeet-tdt encoder graph build failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        source: GgmlCpuGraphError,
    },
    #[error("parakeet-tdt encoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("parakeet-tdt encoder shape error: {reason}")]
    Shape { reason: String },
}

/// 128-bin log-mel features for one utterance: `data` is row-major
/// `[n_mels, n_frames]` (n_mels fastest).
#[derive(Debug, Clone)]
pub(crate) struct ParakeetTdtMelFeatures {
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub n_mels: usize,
}

/// Encoder output: per-frame joint-projected encoder features.
/// `features[frame * joint_hidden + j]` (frame-major).
#[derive(Debug, Clone)]
pub(crate) struct ParakeetTdtEncoderOutput {
    pub frame_count: usize,
    pub joint_hidden: usize,
    pub features: Vec<f32>,
}

fn bf(step: &'static str) -> impl Fn(GgmlCpuGraphError) -> ParakeetTdtEncoderError {
    move |source| ParakeetTdtEncoderError::GraphBuildFailed { step, source }
}

/// A 2-D linear bound zero-copy to the mmap'd pack (native q4_K/f16/f32).
/// Unlike parakeet-ctc there is no arena fallback variant: every bindable
/// linear's host payload is dropped at load, so binding failure fails closed
/// in `bind_loaded`.
#[derive(Clone, Copy)]
struct WeightSlot(GgmlLoadedTensor);

impl WeightSlot {
    fn graph<'a>(self, _arena: &GgmlStaticTensorArena) -> GgmlCpuTensor<'a> {
        self.0.as_graph_tensor()
    }
}

/// Bind a 2-D linear zero-copy from the mmap'd pack. FAILS CLOSED when absent:
/// the host f32 payload for bound weights was dropped at load, so there is no
/// arena fallback.
fn bind_loaded(
    loaded: Option<&GgmlLoadedWeightContext>,
    name: &str,
) -> Result<WeightSlot, ParakeetTdtEncoderError> {
    arena_bind_loaded(loaded, name)
        .map(WeightSlot)
        .map_err(|reason| ParakeetTdtEncoderError::Shape { reason })
}

fn conv_out_dim(input: usize) -> usize {
    (input + 2 * SUBSAMPLING_PADDING - SUBSAMPLING_KERNEL) / SUBSAMPLING_STRIDE + 1
}

fn alloc_static(
    arena: &GgmlStaticTensorArena,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<GgmlStaticTensor, ParakeetTdtEncoderError> {
    arena_alloc_static_f32(arena, &weight.dims, weight.values.len(), step, true).map_err(
        |e| match e {
            ArenaAllocError::Graph(source) => {
                ParakeetTdtEncoderError::GraphBuildFailed { step, source }
            }
            ArenaAllocError::UnsupportedRank(dims) => ParakeetTdtEncoderError::Shape {
                reason: format!("tensor '{}' has unsupported rank {:?}", weight.name, dims),
            },
        },
    )
}

fn upload_static(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<(), ParakeetTdtEncoderError> {
    arena_upload_static_f32(arena, tensor, &weight.values, step)
        .map_err(|source| ParakeetTdtEncoderError::GraphBuildFailed { step, source })
}

fn alloc_static_f16(
    arena: &GgmlStaticTensorArena,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<GgmlStaticTensor, ParakeetTdtEncoderError> {
    arena_alloc_static_f16(arena, &weight.dims, step, true).map_err(|e| match e {
        ArenaAllocError::Graph(source) => {
            ParakeetTdtEncoderError::GraphBuildFailed { step, source }
        }
        ArenaAllocError::UnsupportedRank(dims) => ParakeetTdtEncoderError::Shape {
            reason: format!("f16 depthwise '{}' rank {:?}", weight.name, dims),
        },
    })
}

fn upload_static_f16(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<(), ParakeetTdtEncoderError> {
    arena_upload_static_f16(arena, tensor, &weight.values, step, f32_to_f16_bits)
        .map_err(|source| ParakeetTdtEncoderError::GraphBuildFailed { step, source })
}

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

pub(crate) struct ParakeetTdtEncoderGraph {
    metadata: ParakeetTdtExecutionMetadata,
    runner: GgmlCpuGraphRunner,
    // `loaded_weights` owns the mmap-backed buffer the bound slots alias;
    // field order mirrors parakeet_ctc/cohere (see the soundness note there).
    // Never read directly -- it exists to keep the mapping alive.
    #[allow(dead_code)]
    loaded_weights: Option<GgmlLoadedWeightContext>,
    arena: GgmlStaticTensorArena,
    sub: SubArena,
    layers: Vec<LayerArena>,
    enc_proj_weight: WeightSlot,
    enc_proj_bias: GgmlStaticTensor,
}

fn find_sub<'a>(
    weights: &'a ParakeetTdtEncoderWeights,
    name: &str,
) -> Result<&'a NamedTensor, ParakeetTdtEncoderError> {
    weights
        .subsampling
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| ParakeetTdtEncoderError::Shape {
            reason: format!("missing subsampling tensor '{name}'"),
        })
}

impl ParakeetTdtEncoderGraph {
    pub(crate) fn new(
        weights: &ParakeetTdtEncoderWeights,
        metadata: ParakeetTdtExecutionMetadata,
        runtime_path: Option<&Path>,
    ) -> Result<Self, ParakeetTdtEncoderError> {
        let mut config = parakeet_tdt_encoder_graph_config();
        config.context_bytes = PARAKEET_TDT_ENCODER_GRAPH_CONTEXT_BYTES;
        config.graph_size = config.graph_size.max(weights.layers.len() * 256 + 2048);
        let runner = GgmlCpuGraphRunner::new(config).map_err(|source| {
            ParakeetTdtEncoderError::GraphBuildFailed {
                step: "runner_init",
                source,
            }
        })?;
        let loaded_weights =
            runtime_path.and_then(|path| runner.load_gguf_weight_context(path).ok());
        let loaded = loaded_weights.as_ref();
        let mut arena = runner
            .start_static_tensor_arena(PARAKEET_TDT_ENCODER_GRAPH_CONTEXT_BYTES)
            .map_err(|source| ParakeetTdtEncoderError::GraphBuildFailed {
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
        for layer in weights.layers.iter() {
            layer_arenas.push(alloc_layer(&arena, loaded, layer)?);
        }
        let enc_proj_weight_slot = bind_loaded(loaded, &weights.enc_proj_weight.name)?;
        let enc_proj_bias_t = alloc_static(&arena, &weights.enc_proj_bias, "enc_proj_b")?;

        // ----- upload all values -----
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
        upload_static(
            &mut arena,
            linear_b_t,
            s("enc.sub.linear.bias")?,
            "sub_lin_b",
        )?;
        for (layer, handles) in weights.layers.iter().zip(&layer_arenas) {
            upload_layer(&mut arena, layer, handles)?;
        }
        upload_static(
            &mut arena,
            enc_proj_bias_t,
            &weights.enc_proj_bias,
            "enc_proj_b",
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
            enc_proj_weight: enc_proj_weight_slot,
            enc_proj_bias: enc_proj_bias_t,
        })
    }

    pub(crate) fn encode(
        &mut self,
        mel: &ParakeetTdtMelFeatures,
    ) -> Result<ParakeetTdtEncoderOutput, ParakeetTdtEncoderError> {
        let metadata = self.metadata;
        let d_model = metadata.hidden_size;
        let subsampled_frames = conv_out_dim(conv_out_dim(conv_out_dim(mel.n_frames)));
        let subsampled_freq = conv_out_dim(conv_out_dim(conv_out_dim(mel.n_mels)));
        let positional = build_relative_positional_encoding(d_model, subsampled_frames, || {
            ParakeetTdtEncoderError::Shape {
                reason: "relative positional encoding shape overflow".to_string(),
            }
        })?;

        let mut graph = self.runner.start_graph();

        let mel_t = graph
            .new_tensor_2d_f32(mel.n_mels, mel.n_frames, "parakeet_tdt_mel")
            .map_err(bf("new_mel"))?;
        let pos_t = graph
            .new_tensor_2d_f32(d_model, positional.len() / d_model, "parakeet_tdt_rel_pos")
            .map_err(bf("new_pos"))?;
        graph.set_input(mel_t).map_err(bf("set_input_mel"))?;
        graph.set_input(pos_t).map_err(bf("set_input_pos"))?;

        let conv_map = |step, source| ParakeetTdtEncoderError::GraphBuildFailed { step, source };
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

        // ----- dw-striding subsampling (verbatim parakeet-ctc/cohere prelude) -----
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

        // flatten [channels, freq] per frame -> [channels*freq, frames] -> linear.
        let flattened = self
            .sub
            .conv6_channels
            .checked_mul(subsampled_freq)
            .ok_or_else(|| ParakeetTdtEncoderError::Shape {
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
        // scale_input: x *= sqrt(d_model). Metadata-driven: parakeet-ctc's HF
        // checkpoint scales, parakeet-tdt-0.6b-v3's does NOT (scale_input
        // false; the HF conversion folded NeMo's xscaling away).
        if metadata.scale_input {
            state = graph
                .scale(state, (d_model as f32).sqrt())
                .map_err(bf("scale_input"))?;
        }

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

        // ----- joint encoder projection: d_model -> joint_hidden -----
        // (In place of parakeet-ctc's CTC head; the last conformer block
        // already applied its out_norm, and there is no separate encoder-level
        // final norm in the checkpoint.)
        let proj = graph
            .reshape_2d(
                self.enc_proj_weight.graph(&self.arena),
                d_model,
                metadata.joint_hidden,
            )
            .map_err(bf("enc_proj_reshape"))?;
        let mut features = graph.mul_mat(proj, state).map_err(bf("enc_proj_matmul"))?;
        features = graph
            .add(features, self.arena.graph_tensor(self.enc_proj_bias))
            .map_err(bf("enc_proj_bias"))?;
        graph
            .set_output(features)
            .map_err(bf("set_output_features"))?;

        graph
            .prepare_outputs_for_upload(&[features])
            .map_err(bf("prepare_outputs"))?;
        upload_graph_f32(&mut graph, mel_t, &mel.data, "upload_mel")?;
        upload_graph_f32(&mut graph, pos_t, &positional, "upload_pos")?;

        let want = metadata
            .joint_hidden
            .checked_mul(subsampled_frames)
            .ok_or_else(|| ParakeetTdtEncoderError::Shape {
                reason: "features overflow".into(),
            })?;
        let features = graph.compute_output_f32(features, want).map_err(|error| {
            ParakeetTdtEncoderError::GraphExecutionFailed {
                reason: error.to_string(),
            }
        })?;
        Ok(ParakeetTdtEncoderOutput {
            frame_count: subsampled_frames,
            joint_hidden: metadata.joint_hidden,
            features,
        })
    }
}

fn upload_graph_f32<'a>(
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    tensor: GgmlCpuTensor<'a>,
    values: &[f32],
    step: &'static str,
) -> Result<(), ParakeetTdtEncoderError> {
    graph
        .set_f32_slice(tensor, values, step)
        .map_err(|source| ParakeetTdtEncoderError::GraphBuildFailed { step, source })
}

fn alloc_layer(
    arena: &GgmlStaticTensorArena,
    loaded: Option<&GgmlLoadedWeightContext>,
    layer: &ParakeetTdtEncoderLayerWeights,
) -> Result<LayerArena, ParakeetTdtEncoderError> {
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

fn upload_layer(
    arena: &mut GgmlStaticTensorArena,
    layer: &ParakeetTdtEncoderLayerWeights,
    h: &LayerArena,
) -> Result<(), ParakeetTdtEncoderError> {
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
    use crate::models::parakeet_tdt::encoder_weights::load_parakeet_tdt_encoder_weights;
    use crate::models::parakeet_tdt::runtime_contract::parse_parakeet_tdt_execution_metadata;
    use std::path::Path;

    /// Encoder smoke gate: build the full graph from the real pack + run it on
    /// a dummy mel; assert finite [joint_hidden, frames] features with no ggml
    /// shape error. Transcript correctness is the executor-stage gate.
    #[test]
    fn encoder_graph_smoke_produces_finite_features_when_pack_present() {
        let candidates = [Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../tmp/models/parakeet-tdt-0.6b-v3-source/openasr/parakeet-tdt-0.6b-v3-fp16.oasr",
        )];
        let Some(path) = candidates.into_iter().find(|p| p.exists()) else {
            eprintln!("skipping: parakeet-tdt-0.6b-v3 pack not present");
            return;
        };
        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        let gguf_metadata = crate::ggml_runtime::read_gguf_metadata(&path).expect("gguf metadata");
        let metadata = parse_parakeet_tdt_execution_metadata(&gguf_metadata).expect("metadata");
        let weights = load_parakeet_tdt_encoder_weights(&reader, &metadata).expect("weights");

        let mut graph =
            ParakeetTdtEncoderGraph::new(&weights, metadata, Some(path.as_path())).expect("graph");
        let n_frames = 128usize;
        let mel = ParakeetTdtMelFeatures {
            data: (0..metadata.n_mels * n_frames)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
                .collect(),
            n_frames,
            n_mels: metadata.n_mels,
        };
        let out = graph.encode(&mel).expect("encode");
        let expected_frames = conv_out_dim(conv_out_dim(conv_out_dim(n_frames)));
        assert_eq!(out.frame_count, expected_frames);
        assert_eq!(out.joint_hidden, metadata.joint_hidden);
        assert_eq!(out.features.len(), metadata.joint_hidden * expected_frames);
        assert!(
            out.features.iter().all(|v| v.is_finite()),
            "features must be finite"
        );
    }
}
