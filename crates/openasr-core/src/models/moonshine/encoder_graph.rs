use thiserror::Error;

use std::path::Path;

use crate::ggml_runtime::{
    GGML_TYPE_F16, GgmlCpuGraphBuilder, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor,
    GgmlLoadedTensor, GgmlLoadedWeightContext, GgmlRopeExtParams, GgmlStaticTensor,
    GgmlStaticTensorArena,
};

use super::frontend::MoonshineWaveformFeatures;
use super::graph_config::moonshine_encoder_graph_config;
use super::lora::{LoraSlot, MoonshineLoraAdapter, new_lora_slot_tensors};
use super::runtime_contract::MoonshineExecutionMetadata;
use super::weights::{MoonshineEncoderLayerWeights, MoonshineEncoderWeights, MoonshineWeight};

const MOONSHINE_LAYER_NORM_EPSILON: f32 = 1.0e-5;
const MOONSHINE_GROUP_NORM_EPSILON: f32 = 1.0e-5;
const MOONSHINE_ENCODER_GRAPH_CONTEXT_BYTES: usize = 512 * 1024 * 1024;

// Conv stem strides/kernels are architecture constants.
const CONV1_KERNEL: usize = 127;
const CONV1_STRIDE: usize = 64;
const CONV2_KERNEL: usize = 7;
const CONV2_STRIDE: usize = 3;
const CONV3_KERNEL: usize = 3;
const CONV3_STRIDE: usize = 2;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MoonshineEncoderOutput {
    pub frame_count: usize,
    pub hidden_size: usize,
    /// Layout: [frame][hidden] contiguous f32.
    pub rows: Vec<f32>,
}

#[derive(Debug, Error)]
pub(crate) enum MoonshineEncoderError {
    #[error("moonshine encoder features are invalid: {reason}")]
    InvalidFeatures { reason: String },
    #[error("moonshine encoder graph construction failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        #[source]
        source: GgmlCpuGraphError,
    },
    #[error("moonshine encoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("moonshine encoder shape overflowed")]
    ShapeOverflow,
}

/// A 2-D linear weight: bound zero-copy from the mmap'd pack (native q8_0
/// `[in,out]`) or, as a fallback, uploaded into the arena as f32. moonshine loads
/// its bindable linears meta-only (no f32 materialized), so binding is mandatory
/// and `Arena` is never constructed here — it is retained for parity with the
/// wav2vec2/cohere `WeightSlot` pattern and a future non-mmap fallback.
#[derive(Clone, Copy)]
enum WeightSlot {
    #[allow(dead_code)]
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

fn bind_loaded(
    loaded: Option<&GgmlLoadedWeightContext>,
    name: &str,
) -> Result<WeightSlot, MoonshineEncoderError> {
    match loaded.and_then(|ctx| ctx.tensor(name)) {
        Some(tensor) => Ok(WeightSlot::Loaded(tensor)),
        None => Err(MoonshineEncoderError::GraphExecutionFailed {
            reason: format!(
                "2-D linear '{name}' could not be bound zero-copy from the runtime pack \
                 (loaded weight context missing or tensor absent); host payload was meta-only"
            ),
        }),
    }
}

#[derive(Default, Clone, Copy)]
struct MoonshineEncoderLayerLora {
    attn_q: Option<LoraSlot>,
    attn_k: Option<LoraSlot>,
    attn_v: Option<LoraSlot>,
    attn_o: Option<LoraSlot>,
    ffn_up: Option<LoraSlot>,
    ffn_down: Option<LoraSlot>,
}

struct MoonshineEncoderLayerRuntime {
    attn_norm: GgmlStaticTensor,
    attn_q: WeightSlot,
    attn_k: WeightSlot,
    attn_v: WeightSlot,
    attn_o: WeightSlot,
    ffn_norm: GgmlStaticTensor,
    ffn_up: WeightSlot,
    ffn_up_bias: GgmlStaticTensor,
    ffn_down: WeightSlot,
    ffn_down_bias: GgmlStaticTensor,
    lora: MoonshineEncoderLayerLora,
}

/// Allocate (but do not yet upload) arena tensors for one optional LoRA
/// target; the f32 payloads are queued for upload after all arena tensors
/// exist (the arena cannot extend once its backend buffer is allocated).
fn new_lora_slot<'adapter>(
    arena: &GgmlStaticTensorArena,
    adapter: Option<&'adapter MoonshineLoraAdapter>,
    base_tensor_name: &str,
    pending_uploads: &mut Vec<(GgmlStaticTensor, &'adapter [f32], &'static str)>,
) -> Result<Option<LoraSlot>, MoonshineEncoderError> {
    let Some(target) = adapter.and_then(|adapter| adapter.target(base_tensor_name)) else {
        return Ok(None);
    };
    let slot =
        new_lora_slot_tensors(arena, target, "enc_lora_a", "enc_lora_b").map_err(|source| {
            MoonshineEncoderError::GraphBuildFailed {
                step: "enc_lora_alloc",
                source,
            }
        })?;
    pending_uploads.push((slot.a, target.a_values.as_slice(), "enc_lora_a"));
    pending_uploads.push((
        slot.b_scaled,
        target.b_scaled_values.as_slice(),
        "enc_lora_b",
    ));
    Ok(Some(slot))
}

pub(crate) struct MoonshineEncoderGraphRuntime {
    metadata: MoonshineExecutionMetadata,
    runner: GgmlCpuGraphRunner,
    // Owns the mmap'd pack backing every zero-copy WeightSlot::Loaded handle;
    // must outlive `layers`. Kept even when None (f32-fallback path).
    #[allow(dead_code)]
    loaded_weights: Option<GgmlLoadedWeightContext>,
    arena: GgmlStaticTensorArena,
    conv1_weight: GgmlStaticTensor,
    conv2_weight: GgmlStaticTensor,
    conv2_bias: GgmlStaticTensor,
    conv2_bias_len: usize,
    conv3_weight: GgmlStaticTensor,
    conv3_bias: GgmlStaticTensor,
    groupnorm_weight: GgmlStaticTensor,
    groupnorm_bias: GgmlStaticTensor,
    out_norm: GgmlStaticTensor,
    layers: Vec<MoonshineEncoderLayerRuntime>,
}

impl MoonshineEncoderGraphRuntime {
    pub(crate) fn new(
        weights: &MoonshineEncoderWeights,
        metadata: MoonshineExecutionMetadata,
        runtime_path: Option<&Path>,
        adapter: Option<&MoonshineLoraAdapter>,
    ) -> Result<Self, MoonshineEncoderError> {
        let mut config = moonshine_encoder_graph_config();
        config.context_bytes = MOONSHINE_ENCODER_GRAPH_CONTEXT_BYTES;
        let runner = GgmlCpuGraphRunner::new(config).map_err(build_err("runner_init"))?;
        // Goals 7+8 memory lever: bind the per-layer 2-D linears zero-copy from the
        // mmap'd pack (native q8_0 [in,out]) instead of dequantizing them to
        // resident f32 host Vecs. The weights loader supplies these meta-only
        // (empty values), so binding is mandatory (fails closed below).
        let loaded_weights =
            runtime_path.and_then(|path| runner.load_gguf_weight_context(path).ok());
        let loaded = loaded_weights.as_ref();
        let mut arena = runner
            .start_static_tensor_arena(config.context_bytes)
            .map_err(build_err("static_tensor_arena"))?;

        let conv1_weight = new_conv_kernel_f16(&arena, &weights.conv1_weight, "enc_conv1_w")?;
        let conv2_weight = new_conv_kernel_f16(&arena, &weights.conv2_weight, "enc_conv2_w")?;
        let conv2_bias = new_vector(&arena, weights.conv2_bias.len(), "enc_conv2_b")?;
        let conv3_weight = new_conv_kernel_f16(&arena, &weights.conv3_weight, "enc_conv3_w")?;
        let conv3_bias = new_vector(&arena, weights.conv3_bias.len(), "enc_conv3_b")?;
        let groupnorm_weight = new_vector(&arena, weights.groupnorm_weight.len(), "enc_gn_w")?;
        let groupnorm_bias = new_vector(&arena, weights.groupnorm_bias.len(), "enc_gn_b")?;
        let out_norm = new_vector(&arena, weights.out_norm.len(), "enc_out_norm")?;

        let mut layers = Vec::with_capacity(weights.layers.len());
        let mut pending_lora_uploads = Vec::new();
        for layer in &weights.layers {
            layers.push(MoonshineEncoderLayerRuntime {
                attn_norm: new_vector(&arena, layer.attn_norm.len(), "enc_attn_norm")?,
                attn_q: bind_loaded(loaded, &layer.attn_q.name)?,
                attn_k: bind_loaded(loaded, &layer.attn_k.name)?,
                attn_v: bind_loaded(loaded, &layer.attn_v.name)?,
                attn_o: bind_loaded(loaded, &layer.attn_o.name)?,
                ffn_norm: new_vector(&arena, layer.ffn_norm.len(), "enc_ffn_norm")?,
                ffn_up: bind_loaded(loaded, &layer.ffn_up.name)?,
                ffn_up_bias: new_vector(&arena, layer.ffn_up_bias.len(), "enc_ffn_up_b")?,
                ffn_down: bind_loaded(loaded, &layer.ffn_down.name)?,
                ffn_down_bias: new_vector(&arena, layer.ffn_down_bias.len(), "enc_ffn_down_b")?,
                lora: MoonshineEncoderLayerLora {
                    attn_q: new_lora_slot(
                        &arena,
                        adapter,
                        &layer.attn_q.name,
                        &mut pending_lora_uploads,
                    )?,
                    attn_k: new_lora_slot(
                        &arena,
                        adapter,
                        &layer.attn_k.name,
                        &mut pending_lora_uploads,
                    )?,
                    attn_v: new_lora_slot(
                        &arena,
                        adapter,
                        &layer.attn_v.name,
                        &mut pending_lora_uploads,
                    )?,
                    attn_o: new_lora_slot(
                        &arena,
                        adapter,
                        &layer.attn_o.name,
                        &mut pending_lora_uploads,
                    )?,
                    ffn_up: new_lora_slot(
                        &arena,
                        adapter,
                        &layer.ffn_up.name,
                        &mut pending_lora_uploads,
                    )?,
                    ffn_down: new_lora_slot(
                        &arena,
                        adapter,
                        &layer.ffn_down.name,
                        &mut pending_lora_uploads,
                    )?,
                },
            });
        }

        upload_f16(
            &mut arena,
            conv1_weight,
            &weights.conv1_weight,
            "enc_conv1_w",
        )?;
        upload_f16(
            &mut arena,
            conv2_weight,
            &weights.conv2_weight,
            "enc_conv2_w",
        )?;
        upload(&mut arena, conv2_bias, &weights.conv2_bias, "enc_conv2_b")?;
        upload_f16(
            &mut arena,
            conv3_weight,
            &weights.conv3_weight,
            "enc_conv3_w",
        )?;
        upload(&mut arena, conv3_bias, &weights.conv3_bias, "enc_conv3_b")?;
        upload(
            &mut arena,
            groupnorm_weight,
            &weights.groupnorm_weight,
            "enc_gn_w",
        )?;
        upload(
            &mut arena,
            groupnorm_bias,
            &weights.groupnorm_bias,
            "enc_gn_b",
        )?;
        upload(&mut arena, out_norm, &weights.out_norm, "enc_out_norm")?;
        for (runtime, layer) in layers.iter().zip(&weights.layers) {
            upload_layer(&mut arena, runtime, layer)?;
        }
        for (tensor, values, name) in pending_lora_uploads {
            arena
                .set_f32_slice(tensor, values, name)
                .map_err(|source| MoonshineEncoderError::GraphBuildFailed { step: name, source })?;
        }

        Ok(Self {
            metadata,
            runner,
            loaded_weights,
            arena,
            conv1_weight,
            conv2_weight,
            conv2_bias,
            conv2_bias_len: weights.conv2_bias.len(),
            conv3_weight,
            conv3_bias,
            groupnorm_weight,
            groupnorm_bias,
            out_norm,
            layers,
        })
    }

    pub(crate) fn encode(
        &mut self,
        features: &MoonshineWaveformFeatures,
    ) -> Result<MoonshineEncoderOutput, MoonshineEncoderError> {
        let n_samples = features.samples.len();
        let l1 = conv_out_len(n_samples, CONV1_KERNEL, CONV1_STRIDE)?;
        let l2 = conv_out_len(l1, CONV2_KERNEL, CONV2_STRIDE)?;
        let frame_count = conv_out_len(l2, CONV3_KERNEL, CONV3_STRIDE)?;
        if frame_count == 0 {
            return Err(MoonshineEncoderError::InvalidFeatures {
                reason: format!("audio too short: {n_samples} samples produce 0 encoder frames"),
            });
        }
        let d_model = self.metadata.d_model;
        let heads = self.metadata.n_heads;
        let head_dim = self.metadata.head_dim;

        let mut graph = self.runner.start_graph();

        // Raw waveform input as ggml conv_1d data [L, IC=1].
        let wave = graph
            .new_tensor_2d_f32(n_samples, 1, "moonshine_wave")
            .map_err(build_err("ggml_new_tensor_2d(wave)"))?;
        graph
            .set_input(wave)
            .map_err(build_err("ggml_set_input(wave)"))?;

        // Encoder self-attention positions 0..frame_count. Declared as an input up-front
        // (all inputs must exist before any op pulls allocation forward).
        let positions = graph
            .new_tensor_1d_i32(frame_count, "enc_positions")
            .map_err(build_err("ggml_new_tensor_1d(enc_pos)"))?;
        graph
            .set_input(positions)
            .map_err(build_err("ggml_set_input(enc_pos)"))?;

        // HF conv stem order (modeling_moonshine.MoonshineEncoder.forward):
        //   x = tanh(conv1(x)); x = groupnorm(x); x = gelu(conv2(x)); x = gelu(conv3(x))
        // GroupNorm(num_groups=1) is applied right after conv1, over the 288 channels.

        // conv1 (no bias) -> tanh. Output [conv1_len, d_model] (ne0=time, ne1=channels).
        let conv1_len = conv_out_len(n_samples, CONV1_KERNEL, CONV1_STRIDE)?;
        let mut state = graph
            .conv_1d(
                self.arena.graph_tensor(self.conv1_weight),
                wave,
                CONV1_STRIDE,
                0,
                1,
            )
            .map_err(build_err("conv1"))?;
        state = graph.tanh(state).map_err(build_err("conv1_tanh"))?;

        // GroupNorm(1) over channels per time-step: transpose to channel-major [d_model, time],
        // ggml_norm over ne0(channels), affine, then transpose back to [time, d_model] for conv2.
        let mut chan = graph
            .permute(state, 1, 0, 2, 3)
            .map_err(build_err("ggml_permute(gn_channel_major)"))?;
        chan = graph
            .cont(chan)
            .map_err(build_err("ggml_cont(gn_channel_major)"))?;
        chan = graph
            .reshape_2d(chan, d_model, conv1_len)
            .map_err(build_err("ggml_reshape_2d(gn_channel_major)"))?;
        chan = graph
            .norm(chan, MOONSHINE_GROUP_NORM_EPSILON)
            .map_err(build_err("ggml_norm(groupnorm)"))?;
        chan = graph
            .mul(chan, self.arena.graph_tensor(self.groupnorm_weight))
            .map_err(build_err("ggml_mul(groupnorm_w)"))?;
        chan = graph
            .add(chan, self.arena.graph_tensor(self.groupnorm_bias))
            .map_err(build_err("ggml_add(groupnorm_b)"))?;
        // transpose back to [time, d_model] for conv2 data layout.
        state = graph
            .permute(chan, 1, 0, 2, 3)
            .map_err(build_err("ggml_permute(gn_time_major)"))?;
        state = graph
            .cont(state)
            .map_err(build_err("ggml_cont(gn_time_major)"))?;
        state = graph
            .reshape_2d(state, conv1_len, d_model)
            .map_err(build_err("ggml_reshape_2d(gn_time_major)"))?;

        // conv2 + bias -> gelu (exact erf).
        state = graph
            .conv_1d(
                self.arena.graph_tensor(self.conv2_weight),
                state,
                CONV2_STRIDE,
                0,
                1,
            )
            .map_err(build_err("conv2"))?;
        let conv2_channels = self.conv2_bias_len;
        state = add_channel_bias(
            &graph,
            state,
            self.arena.graph_tensor(self.conv2_bias),
            conv2_channels,
        )?;
        state = graph.gelu_erf(state).map_err(build_err("conv2_gelu"))?;
        // conv3 + bias -> gelu (exact erf).
        state = graph
            .conv_1d(
                self.arena.graph_tensor(self.conv3_weight),
                state,
                CONV3_STRIDE,
                0,
                1,
            )
            .map_err(build_err("conv3"))?;
        state = add_channel_bias(
            &graph,
            state,
            self.arena.graph_tensor(self.conv3_bias),
            d_model,
        )?;
        state = graph.gelu_erf(state).map_err(build_err("conv3_gelu"))?;

        // After conv3: state is [frame_count, d_model] (ne0=frames, ne1=channels).
        // Transpose to channel-major [d_model, frame_count] sequence for the transformer.
        let mut seq = graph
            .permute(state, 1, 0, 2, 3)
            .map_err(build_err("ggml_permute(seq_channel_major)"))?;
        seq = graph
            .cont(seq)
            .map_err(build_err("ggml_cont(seq_channel_major)"))?;
        let mut hidden = graph
            .reshape_2d(seq, d_model, frame_count)
            .map_err(build_err("ggml_reshape_2d(seq_channel_major)"))?;

        // Transformer layers (pre-norm, bidirectional self-attn + partial RoPE, gelu FFN).
        let rope_params = GgmlRopeExtParams::moonshine_gptj(
            self.metadata.rotary_dim,
            self.metadata.decoder_max_context,
            self.metadata.rope_theta,
        )
        .map_err(build_err("rope_params"))?;
        for layer in &self.layers {
            hidden = run_encoder_layer(
                &mut graph,
                &self.arena,
                hidden,
                layer,
                positions,
                rope_params,
                frame_count,
                d_model,
                heads,
                head_dim,
            )?;
        }

        // Final weight-only LayerNorm.
        hidden = apply_weighted_norm(
            &graph,
            hidden,
            self.arena.graph_tensor(self.out_norm),
            "enc_out_norm",
        )?;
        graph
            .set_output(hidden)
            .map_err(build_err("ggml_set_output(enc)"))?;

        // Upload inputs after the graph is fully built (deferred allocation).
        graph
            .set_f32_slice(wave, &features.samples, "moonshine_wave")
            .map_err(build_err("ggml_set_f32_slice(wave)"))?;
        let position_values: Vec<i32> = (0..frame_count as i32).collect();
        graph
            .set_i32_slice(positions, &position_values, "enc_positions")
            .map_err(build_err("ggml_set_i32_slice(enc_pos)"))?;

        let total = d_model
            .checked_mul(frame_count)
            .ok_or(MoonshineEncoderError::ShapeOverflow)?;
        let rows = graph.compute_output_f32(hidden, total).map_err(|error| {
            MoonshineEncoderError::GraphExecutionFailed {
                reason: error.to_string(),
            }
        })?;
        Ok(MoonshineEncoderOutput {
            frame_count,
            hidden_size: d_model,
            rows,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn run_encoder_layer<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    arena: &GgmlStaticTensorArena,
    state: GgmlCpuTensor<'a>,
    layer: &MoonshineEncoderLayerRuntime,
    positions: GgmlCpuTensor<'a>,
    rope_params: GgmlRopeExtParams,
    frame_count: usize,
    d_model: usize,
    heads: usize,
    head_dim: usize,
) -> Result<GgmlCpuTensor<'a>, MoonshineEncoderError> {
    let attn_input = state;
    let normed = apply_weighted_norm(
        graph,
        state,
        arena.graph_tensor(layer.attn_norm),
        "enc_attn_norm",
    )?;
    let q = matmul(
        graph,
        arena,
        layer.attn_q,
        layer.lora.attn_q,
        normed,
        "enc_q",
    )?;
    let k = matmul(
        graph,
        arena,
        layer.attn_k,
        layer.lora.attn_k,
        normed,
        "enc_k",
    )?;
    let v = matmul(
        graph,
        arena,
        layer.attn_v,
        layer.lora.attn_v,
        normed,
        "enc_v",
    )?;

    let q = rope_heads(
        graph,
        q,
        head_dim,
        heads,
        frame_count,
        positions,
        rope_params,
        "enc_q_rope",
    )?;
    let k = rope_heads(
        graph,
        k,
        head_dim,
        heads,
        frame_count,
        positions,
        rope_params,
        "enc_k_rope",
    )?;

    // q,k are already [head_dim, heads, tokens] after rope; permute to [head_dim, tokens, heads].
    let q = roped_to_attn(graph, q, "enc_q_attn")?;
    let k = roped_to_attn(graph, k, "enc_k_attn")?;
    let v = reshape_heads_for_attn(graph, v, head_dim, heads, frame_count, "enc_v_attn")?;

    // Manual scaled dot-product attention (bidirectional, no mask). flash_attn_ext is avoided
    // because head_dim=36 is not a Metal flash-attention supported size.
    let scale = 1.0 / (head_dim as f32).sqrt();
    let context = scaled_dot_product_attention(
        graph,
        q,
        k,
        v,
        None,
        scale,
        head_dim,
        frame_count,
        heads,
        d_model,
    )?;
    let attn = matmul(
        graph,
        arena,
        layer.attn_o,
        layer.lora.attn_o,
        context,
        "enc_o",
    )?;
    let state = graph
        .add(attn_input, attn)
        .map_err(build_err("enc_attn_residual"))?;

    // FFN (post_attention_layernorm pre-norm, gelu, fc1 has bias, fc2 has bias).
    let ffn_input = state;
    let normed = apply_weighted_norm(
        graph,
        state,
        arena.graph_tensor(layer.ffn_norm),
        "enc_ffn_norm",
    )?;
    let mut ff = matmul(
        graph,
        arena,
        layer.ffn_up,
        layer.lora.ffn_up,
        normed,
        "enc_ffn_up",
    )?;
    ff = graph
        .add(ff, arena.graph_tensor(layer.ffn_up_bias))
        .map_err(build_err("enc_ffn_up_bias"))?;
    ff = graph.gelu_erf(ff).map_err(build_err("enc_ffn_gelu"))?;
    ff = matmul(
        graph,
        arena,
        layer.ffn_down,
        layer.lora.ffn_down,
        ff,
        "enc_ffn_down",
    )?;
    ff = graph
        .add(ff, arena.graph_tensor(layer.ffn_down_bias))
        .map_err(build_err("enc_ffn_down_bias"))?;
    let state = graph
        .add(ffn_input, ff)
        .map_err(build_err("enc_ffn_residual"))?;
    Ok(state)
}

#[allow(clippy::too_many_arguments)]
fn rope_heads<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    heads: usize,
    tokens: usize,
    positions: GgmlCpuTensor<'a>,
    params: GgmlRopeExtParams,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineEncoderError> {
    // ggml_rope_ext wants [head_dim, n_head, n_tokens] with positions along ne2.
    let reshaped = graph
        .reshape_3d(projection, head_dim, heads, tokens)
        .map_err(build_err("ggml_reshape_3d(rope)"))?;
    graph
        .rope_ext(reshaped, positions, params)
        .map_err(|source| MoonshineEncoderError::GraphBuildFailed { step, source })
}

/// Manual scaled dot-product attention over q,k,v laid out as [head_dim, seq, heads].
/// Returns merged context as [d_model, q_len]. Optional additive mask is [kv_len, q_len, 1].
#[allow(clippy::too_many_arguments)]
fn scaled_dot_product_attention<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    q: GgmlCpuTensor<'a>,
    k: GgmlCpuTensor<'a>,
    v: GgmlCpuTensor<'a>,
    mask: Option<GgmlCpuTensor<'a>>,
    scale: f32,
    _head_dim: usize,
    q_len: usize,
    _heads: usize,
    d_model: usize,
) -> Result<GgmlCpuTensor<'a>, MoonshineEncoderError> {
    // scores = kᵀ q -> [kv_len, q_len, heads]
    let scores = graph
        .mul_mat(k, q)
        .map_err(build_err("ggml_mul_mat(attn_scores)"))?;
    let probs = graph
        .soft_max_ext(scores, mask, scale, 0.0)
        .map_err(build_err("ggml_soft_max_ext(attn_scores)"))?;
    // v as [kv_len, head_dim, heads] so mul_mat contracts over kv_len.
    let v_t = graph
        .permute(v, 1, 0, 2, 3)
        .map_err(build_err("ggml_permute(attn_v_t)"))?;
    let v_t = graph.cont(v_t).map_err(build_err("ggml_cont(attn_v_t)"))?;
    // context = vᵀ probs -> [head_dim, q_len, heads]
    let context = graph
        .mul_mat(v_t, probs)
        .map_err(build_err("ggml_mul_mat(attn_ctx)"))?;
    // merge heads: [head_dim, q_len, heads] -> [head_dim, heads, q_len] -> [d_model, q_len]
    let merged = graph
        .permute(context, 0, 2, 1, 3)
        .map_err(build_err("ggml_permute(attn_merge)"))?;
    let merged = graph
        .cont(merged)
        .map_err(build_err("ggml_cont(attn_merge)"))?;
    graph
        .reshape_2d(merged, d_model, q_len)
        .map_err(build_err("ggml_reshape_2d(attn_merge)"))
}

/// Permute a roped [head_dim, heads, tokens] tensor into flash-attn layout [head_dim, tokens, heads].
fn roped_to_attn<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    roped: GgmlCpuTensor<'a>,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineEncoderError> {
    let permuted = graph
        .permute(roped, 0, 2, 1, 3)
        .map_err(build_err("ggml_permute(rope_attn)"))?;
    graph
        .cont(permuted)
        .map_err(|source| MoonshineEncoderError::GraphBuildFailed { step, source })
}

/// Reshape a [d_model, tokens] projection into flash-attn layout [head_dim, tokens, heads].
fn reshape_heads_for_attn<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    heads: usize,
    tokens: usize,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineEncoderError> {
    let reshaped = graph
        .reshape_3d(projection, head_dim, heads, tokens)
        .map_err(build_err("ggml_reshape_3d(attn)"))?;
    let permuted = graph
        .permute(reshaped, 0, 2, 1, 3)
        .map_err(build_err("ggml_permute(attn)"))?;
    graph
        .cont(permuted)
        .map_err(|source| MoonshineEncoderError::GraphBuildFailed { step, source })
}

/// `y = W@x`, optionally with the dynamic LoRA side branch
/// `y = W@x + B_scaled@(A@x)` when this linear is an adapter target.
fn matmul<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    arena: &GgmlStaticTensorArena,
    weight: WeightSlot,
    lora: Option<LoraSlot>,
    input: GgmlCpuTensor<'a>,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineEncoderError> {
    let build_failed = |source| MoonshineEncoderError::GraphBuildFailed { step, source };
    let base = graph
        .mul_mat(weight.graph(arena), input)
        .map_err(build_failed)?;
    let Some(lora) = lora else {
        return Ok(base);
    };
    let ax = graph
        .mul_mat(arena.graph_tensor(lora.a), input)
        .map_err(build_failed)?;
    let delta = graph
        .mul_mat(arena.graph_tensor(lora.b_scaled), ax)
        .map_err(build_failed)?;
    graph.add(base, delta).map_err(build_failed)
}

/// Weight-only mean-centered LayerNorm: ggml_norm(x) * weight (no bias, not RMS).
fn apply_weighted_norm<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    weight: GgmlCpuTensor<'a>,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MoonshineEncoderError> {
    let normed = graph
        .norm(input, MOONSHINE_LAYER_NORM_EPSILON)
        .map_err(build_err("ggml_norm(weighted_ln)"))?;
    graph
        .mul(normed, weight)
        .map_err(|source| MoonshineEncoderError::GraphBuildFailed { step, source })
}

/// Add a per-channel conv bias. After conv_1d the data is [out_len, out_ch]; the bias
/// must broadcast over out_len, so reshape it to [1, out_ch].
fn add_channel_bias<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    state: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    channels: usize,
) -> Result<GgmlCpuTensor<'a>, MoonshineEncoderError> {
    let bias_2d = graph
        .reshape_2d(bias, 1, channels)
        .map_err(build_err("ggml_reshape_2d(conv_bias)"))?;
    graph
        .add(state, bias_2d)
        .map_err(build_err("ggml_add(conv_bias)"))
}

fn conv_out_len(
    input: usize,
    kernel: usize,
    stride: usize,
) -> Result<usize, MoonshineEncoderError> {
    if input < kernel {
        return Ok(0);
    }
    input
        .checked_sub(kernel)
        .and_then(|value| value.checked_div(stride))
        .and_then(|value| value.checked_add(1))
        .ok_or(MoonshineEncoderError::ShapeOverflow)
}

fn new_vector(
    arena: &GgmlStaticTensorArena,
    len: usize,
    name: &'static str,
) -> Result<GgmlStaticTensor, MoonshineEncoderError> {
    arena
        .new_tensor_1d_f32(len, name)
        .map_err(|source| MoonshineEncoderError::GraphBuildFailed { step: name, source })
}

/// Conv kernels must be F16 because ggml's im2col conv path asserts a half-precision kernel.
fn new_conv_kernel_f16(
    arena: &GgmlStaticTensorArena,
    weight: &MoonshineWeight,
    name: &'static str,
) -> Result<GgmlStaticTensor, MoonshineEncoderError> {
    match weight.dims.as_slice() {
        [ne0, ne1, ne2] => arena
            .new_tensor_3d_typed(*ne0, *ne1, *ne2, GGML_TYPE_F16, name)
            .map_err(|source| MoonshineEncoderError::GraphBuildFailed { step: name, source }),
        _ => Err(MoonshineEncoderError::InvalidFeatures {
            reason: format!(
                "unsupported conv kernel rank for '{}': {:?}",
                weight.name, weight.dims
            ),
        }),
    }
}

fn upload(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &MoonshineWeight,
    name: &'static str,
) -> Result<(), MoonshineEncoderError> {
    arena
        .set_f32_slice(tensor, &weight.values, name)
        .map_err(|source| MoonshineEncoderError::GraphBuildFailed { step: name, source })
}

fn upload_f16(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &MoonshineWeight,
    name: &'static str,
) -> Result<(), MoonshineEncoderError> {
    let bits: Vec<u16> = weight.values.iter().copied().map(f32_to_f16_bits).collect();
    arena
        .set_f16_bits_slice(tensor, &bits, name)
        .map_err(|source| MoonshineEncoderError::GraphBuildFailed { step: name, source })
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

fn upload_layer(
    arena: &mut GgmlStaticTensorArena,
    runtime: &MoonshineEncoderLayerRuntime,
    layer: &MoonshineEncoderLayerWeights,
) -> Result<(), MoonshineEncoderError> {
    // The 2-D linears (attn_q/k/v/o, ffn_up/down) are bound zero-copy from the
    // mmap'd pack (WeightSlot::Loaded) and carry no host payload — only the
    // arena-resident norms + biases are uploaded here.
    upload(arena, runtime.attn_norm, &layer.attn_norm, "enc_attn_norm")?;
    upload(arena, runtime.ffn_norm, &layer.ffn_norm, "enc_ffn_norm")?;
    upload(
        arena,
        runtime.ffn_up_bias,
        &layer.ffn_up_bias,
        "enc_ffn_up_b",
    )?;
    upload(
        arena,
        runtime.ffn_down_bias,
        &layer.ffn_down_bias,
        "enc_ffn_down_b",
    )
}

fn build_err(step: &'static str) -> impl Fn(GgmlCpuGraphError) -> MoonshineEncoderError {
    move |source| MoonshineEncoderError::GraphBuildFailed { step, source }
}
