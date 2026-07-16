//! MiMo-Audio-Tokenizer encoder ggml graph: `conv1(k3,s1)+gelu -> conv2(k3,s2)+gelu`
//! -> 32 rope transformer layers (pre-LN, asymmetric qkv bias, plain GELU FFN)
//! **with the layer-3 (idx 2) output added back onto the layer-32 (idx 31)
//! output before the final LayerNorm** -> `down_sample(k2,s2,no bias)+gelu` ->
//! `down_sample_norm`. Produces the 25Hz `[T, 1280]` hidden-state rows that
//! [`super::rvq`] quantizes into 8 RVQ codebook indices per frame.
//!
//! This is the P2.0 "blood lesson #1/#2" surface: the skip connection and the
//! conv1/conv2 stride asymmetry are graph-shape facts, not weights -- get
//! either wrong and every RVQ code downstream mis-codes (see
//! `GGUF_MANIFEST.md`'s `mimo.tok.encoder.skip_layer_id` /
//! `mimo.tok.conv{1,2}.stride` doc comments and P2.0 findings SS2).

use std::path::Path;

use thiserror::Error;

use crate::ggml_runtime::{
    GGML_TYPE_F16, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner,
    GgmlCpuTensor, GgmlLoadedTensor, GgmlLoadedWeightContext, GgmlRopeExtParams, GgmlStaticTensor,
    GgmlStaticTensorArena, GgufTensorDataReadError, GgufTensorDataReader,
};
use crate::nn::conv::{
    Conv1dParams, ConvActivation, ConvBlockSteps, apply_conv_1d_bias_activation,
};
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

use super::mel_frontend::MimoMelFeatures;
use super::runtime_contract::MimoAudiotokMetadata;
use super::tensor_names::{
    AUDIOTOK_CONV1_BIAS, AUDIOTOK_CONV1_WEIGHT, AUDIOTOK_CONV2_BIAS, AUDIOTOK_CONV2_WEIGHT,
    AUDIOTOK_DOWN_SAMPLE_NORM_BIAS, AUDIOTOK_DOWN_SAMPLE_NORM_WEIGHT, AUDIOTOK_DOWN_SAMPLE_WEIGHT,
    AUDIOTOK_NORM_BIAS, AUDIOTOK_NORM_WEIGHT, mimo_audiotok_layer_tensor_names,
};

const LAYER_NORM_EPSILON: f32 = 1.0e-5;
const GRAPH_CONTEXT_BYTES: usize = 512 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MimoAudiotokEncoderOutput {
    pub frame_count: usize,
    pub d_model: usize,
    /// Frame-major (`[frame][d_model]`) contiguous f32.
    pub rows: Vec<f32>,
}

#[derive(Debug, Error)]
pub(crate) enum MimoAudiotokEncoderError {
    #[error("mimo-asr audiotok encoder graph construction failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        #[source]
        source: GgmlCpuGraphError,
    },
    #[error("mimo-asr audiotok encoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error(
        "mimo-asr audiotok encoder 2-D weight '{name}' could not be bound zero-copy from the runtime pack"
    )]
    WeightNotBound { name: String },
    #[error("mimo-asr audiotok encoder could not read tensor '{name}': {source}")]
    TensorRead {
        name: String,
        #[source]
        source: GgufTensorDataReadError,
    },
    #[error("mimo-asr audiotok encoder input mel features are invalid: {reason}")]
    InvalidMelFeatures { reason: String },
    #[error("mimo-asr audiotok encoder shape overflowed")]
    ShapeOverflow,
}

fn build_err(step: &'static str, source: GgmlCpuGraphError) -> MimoAudiotokEncoderError {
    MimoAudiotokEncoderError::GraphBuildFailed { step, source }
}

fn bind(
    loaded: &GgmlLoadedWeightContext,
    name: &str,
) -> Result<GgmlLoadedTensor, MimoAudiotokEncoderError> {
    loaded
        .tensor(name)
        .ok_or_else(|| MimoAudiotokEncoderError::WeightNotBound {
            name: name.to_string(),
        })
}

struct LayerRuntime {
    attn_norm: GgmlStaticTensor,
    attn_norm_bias: GgmlStaticTensor,
    attn_q: GgmlLoadedTensor,
    attn_q_bias: GgmlStaticTensor,
    attn_k: GgmlLoadedTensor,
    // No k bias (asymmetric qkv bias: q/v have bias, k does not).
    attn_v: GgmlLoadedTensor,
    attn_v_bias: GgmlStaticTensor,
    attn_out: GgmlLoadedTensor,
    attn_out_bias: GgmlStaticTensor,
    ffn_norm: GgmlStaticTensor,
    ffn_norm_bias: GgmlStaticTensor,
    ffn_up: GgmlLoadedTensor,
    ffn_up_bias: GgmlStaticTensor,
    ffn_down: GgmlLoadedTensor,
    ffn_down_bias: GgmlStaticTensor,
}

pub(crate) struct MimoAudiotokEncoderRuntime {
    metadata: MimoAudiotokMetadata,
    runner: GgmlCpuGraphRunner,
    #[allow(dead_code)]
    loaded_weights: GgmlLoadedWeightContext,
    arena: GgmlStaticTensorArena,
    conv1_weight: GgmlStaticTensor,
    conv1_bias: GgmlStaticTensor,
    conv2_weight: GgmlStaticTensor,
    conv2_bias: GgmlStaticTensor,
    norm_weight: GgmlStaticTensor,
    norm_bias: GgmlStaticTensor,
    down_sample_weight: GgmlStaticTensor,
    down_sample_zero_bias: GgmlStaticTensor,
    down_sample_norm_weight: GgmlStaticTensor,
    down_sample_norm_bias: GgmlStaticTensor,
    layers: Vec<LayerRuntime>,
}

impl MimoAudiotokEncoderRuntime {
    pub(crate) fn new(
        runtime_path: &Path,
        metadata: MimoAudiotokMetadata,
    ) -> Result<Self, MimoAudiotokEncoderError> {
        let mut config = GgmlCpuGraphConfig::default();
        config.context_bytes = config
            .context_bytes
            .max(GgmlCpuGraphConfig::metadata_context_bytes(
                config.graph_size,
            ))
            .max(GRAPH_CONTEXT_BYTES);
        let runner =
            GgmlCpuGraphRunner::new(config).map_err(|source| build_err("runner_init", source))?;
        // Zero-copy binding for the big 2-D attn/ffn matmul weights (mmap'd,
        // native f16 -- no host f32 duplicate).
        let loaded_weights = runner
            .load_gguf_weight_context(runtime_path)
            .map_err(|error| MimoAudiotokEncoderError::GraphExecutionFailed {
                reason: format!("load_gguf_weight_context: {error}"),
            })?;
        // Plain host reader for the small arena-resident tensors (norms,
        // biases, conv kernels) -- `GgmlLoadedWeightContext` only exposes
        // zero-copy graph bindings, not host f32 copies.
        let reader = GgufTensorDataReader::from_path(runtime_path).map_err(|error| {
            MimoAudiotokEncoderError::GraphExecutionFailed {
                reason: format!("GgufTensorDataReader::from_path: {error}"),
            }
        })?;
        let mut arena = runner
            .start_static_tensor_arena(config.context_bytes)
            .map_err(|source| build_err("static_tensor_arena", source))?;

        let d_model = metadata.d_model;
        let n_mels = 128usize; // fixed by the mel front-end (conv1's in-channel count)
        let conv1_weight = arena
            .new_tensor_3d_typed(
                metadata.conv_kernel_size,
                n_mels,
                d_model,
                GGML_TYPE_F16,
                "audiotok_conv1_w",
            )
            .map_err(|source| build_err("audiotok_conv1_w", source))?;
        // Conv biases are `[1, out_channels]` (not `[out_channels]`): ggml
        // conv_1d output is `[out_time, out_channels]` (time on ne0), so the
        // per-channel bias must broadcast over ne0=time -- exactly the shape
        // whisper's own conv bias uses (`new_tensor_2d_f32(1, out_channels)`).
        let conv1_bias = new_channel_bias(&arena, d_model, "audiotok_conv1_b")?;
        let conv2_weight = arena
            .new_tensor_3d_typed(
                metadata.conv_kernel_size,
                d_model,
                d_model,
                GGML_TYPE_F16,
                "audiotok_conv2_w",
            )
            .map_err(|source| build_err("audiotok_conv2_w", source))?;
        let conv2_bias = new_channel_bias(&arena, d_model, "audiotok_conv2_b")?;
        let norm_weight = new_vector(&arena, d_model, "audiotok_norm_w")?;
        let norm_bias = new_vector(&arena, d_model, "audiotok_norm_b")?;
        let down_sample_weight = arena
            .new_tensor_3d_typed(2, d_model, d_model, GGML_TYPE_F16, "audiotok_ds_w")
            .map_err(|source| build_err("audiotok_ds_w", source))?;
        // `down_sample_layer` has no bias (bias=False upstream); allocate a
        // dedicated always-zero vector rather than reuse another tensor
        // scaled by zero, so the graph's bias-add is a plain, self-explanatory
        // no-op tensor instead of a scale-trick.
        let down_sample_zero_bias = new_channel_bias(&arena, d_model, "audiotok_ds_zero_b")?;
        let down_sample_norm_weight = new_vector(&arena, d_model, "audiotok_ds_norm_w")?;
        let down_sample_norm_bias = new_vector(&arena, d_model, "audiotok_ds_norm_b")?;

        let mut layers = Vec::with_capacity(metadata.n_layers);
        for layer_idx in 0..metadata.n_layers {
            let names = mimo_audiotok_layer_tensor_names(layer_idx);
            layers.push(LayerRuntime {
                attn_norm: new_vector(&arena, d_model, "audiotok_attn_norm_w")?,
                attn_norm_bias: new_vector(&arena, d_model, "audiotok_attn_norm_b")?,
                attn_q: bind(&loaded_weights, &names.attn_q_weight)?,
                attn_q_bias: new_vector(&arena, d_model, "audiotok_attn_q_b")?,
                attn_k: bind(&loaded_weights, &names.attn_k_weight)?,
                attn_v: bind(&loaded_weights, &names.attn_v_weight)?,
                attn_v_bias: new_vector(&arena, d_model, "audiotok_attn_v_b")?,
                attn_out: bind(&loaded_weights, &names.attn_out_weight)?,
                attn_out_bias: new_vector(&arena, d_model, "audiotok_attn_out_b")?,
                ffn_norm: new_vector(&arena, d_model, "audiotok_ffn_norm_w")?,
                ffn_norm_bias: new_vector(&arena, d_model, "audiotok_ffn_norm_b")?,
                ffn_up: bind(&loaded_weights, &names.ffn_up_weight)?,
                ffn_up_bias: new_vector(&arena, metadata.ffn_dim, "audiotok_ffn_up_b")?,
                ffn_down: bind(&loaded_weights, &names.ffn_down_weight)?,
                ffn_down_bias: new_vector(&arena, d_model, "audiotok_ffn_down_b")?,
            });
        }

        let k = metadata.conv_kernel_size as u64;
        let dm = d_model as u64;
        upload_f16(
            &mut arena,
            conv1_weight,
            &reader,
            AUDIOTOK_CONV1_WEIGHT,
            &[k, n_mels as u64, dm],
        )?;
        upload(&mut arena, conv1_bias, &reader, AUDIOTOK_CONV1_BIAS, &[dm])?;
        upload_f16(
            &mut arena,
            conv2_weight,
            &reader,
            AUDIOTOK_CONV2_WEIGHT,
            &[k, dm, dm],
        )?;
        upload(&mut arena, conv2_bias, &reader, AUDIOTOK_CONV2_BIAS, &[dm])?;
        upload(
            &mut arena,
            norm_weight,
            &reader,
            AUDIOTOK_NORM_WEIGHT,
            &[dm],
        )?;
        upload(&mut arena, norm_bias, &reader, AUDIOTOK_NORM_BIAS, &[dm])?;
        upload_f16(
            &mut arena,
            down_sample_weight,
            &reader,
            AUDIOTOK_DOWN_SAMPLE_WEIGHT,
            &[2, dm, dm],
        )?;
        arena
            .set_f32_slice(
                down_sample_zero_bias,
                &vec![0.0_f32; d_model],
                "audiotok_ds_zero_b",
            )
            .map_err(|source| build_err("audiotok_ds_zero_b", source))?;
        upload(
            &mut arena,
            down_sample_norm_weight,
            &reader,
            AUDIOTOK_DOWN_SAMPLE_NORM_WEIGHT,
            &[dm],
        )?;
        upload(
            &mut arena,
            down_sample_norm_bias,
            &reader,
            AUDIOTOK_DOWN_SAMPLE_NORM_BIAS,
            &[dm],
        )?;
        for (layer_idx, layer) in layers.iter().enumerate() {
            let names = mimo_audiotok_layer_tensor_names(layer_idx);
            upload(
                &mut arena,
                layer.attn_norm,
                &reader,
                &names.attn_norm_weight,
                &[dm],
            )?;
            upload(
                &mut arena,
                layer.attn_norm_bias,
                &reader,
                &names.attn_norm_bias,
                &[dm],
            )?;
            upload(
                &mut arena,
                layer.attn_q_bias,
                &reader,
                &names.attn_q_bias,
                &[dm],
            )?;
            upload(
                &mut arena,
                layer.attn_v_bias,
                &reader,
                &names.attn_v_bias,
                &[dm],
            )?;
            upload(
                &mut arena,
                layer.attn_out_bias,
                &reader,
                &names.attn_out_bias,
                &[dm],
            )?;
            upload(
                &mut arena,
                layer.ffn_norm,
                &reader,
                &names.ffn_norm_weight,
                &[dm],
            )?;
            upload(
                &mut arena,
                layer.ffn_norm_bias,
                &reader,
                &names.ffn_norm_bias,
                &[dm],
            )?;
            upload(
                &mut arena,
                layer.ffn_up_bias,
                &reader,
                &names.ffn_up_bias,
                &[metadata.ffn_dim as u64],
            )?;
            upload(
                &mut arena,
                layer.ffn_down_bias,
                &reader,
                &names.ffn_down_bias,
                &[dm],
            )?;
        }

        Ok(Self {
            metadata,
            runner,
            loaded_weights,
            arena,
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            norm_weight,
            norm_bias,
            down_sample_weight,
            down_sample_zero_bias,
            down_sample_norm_weight,
            down_sample_norm_bias,
            layers,
        })
    }

    /// `mel`: `[n_mels][n_frames]` mel-major features from
    /// [`super::mel_frontend::mimo_mel_features_from_samples`]. Returns the
    /// down-sampled (25Hz) `[frame][d_model]` hidden-state rows the RVQ
    /// encoder quantizes.
    pub(crate) fn encode(
        &mut self,
        mel: &MimoMelFeatures,
    ) -> Result<MimoAudiotokEncoderOutput, MimoAudiotokEncoderError> {
        if mel.n_frames == 0 {
            return Err(MimoAudiotokEncoderError::InvalidMelFeatures {
                reason: "zero mel frames".to_string(),
            });
        }
        let d_model = self.metadata.d_model;
        let n_mels = mel.n_mels;
        let conv_pad = self.metadata.conv_kernel_size / 2;

        let mut graph = self.runner.start_graph();

        // ggml conv_1d data layout: [time, in_channels].
        let mel_input = graph
            .new_tensor_2d_f32(mel.n_frames, n_mels, "audiotok_mel")
            .map_err(|source| build_err("audiotok_mel", source))?;
        graph
            .set_input(mel_input)
            .map_err(|source| build_err("audiotok_mel_input", source))?;

        // conv1 (stride 1, no time downsample -- P2.0 blood lesson #2) -> gelu.
        let mut state = apply_conv_1d_bias_activation(
            &graph,
            self.arena.graph_tensor(self.conv1_weight),
            mel_input,
            self.arena.graph_tensor(self.conv1_bias),
            Conv1dParams {
                stride: self.metadata.conv1_stride,
                padding: conv_pad,
                dilation: 1,
            },
            ConvActivation::GeluErf,
            ConvBlockSteps {
                conv: "audiotok_conv1",
                bias: "audiotok_conv1_bias",
                activation: "audiotok_conv1_gelu",
            },
            build_err,
        )?;
        let conv1_frames = conv_out_len(
            mel.n_frames,
            self.metadata.conv_kernel_size,
            self.metadata.conv1_stride,
            conv_pad,
        )?;

        // conv2 (stride 2 -- the ONLY 2x time downsample in the encoder body) -> gelu.
        state = apply_conv_1d_bias_activation(
            &graph,
            self.arena.graph_tensor(self.conv2_weight),
            state,
            self.arena.graph_tensor(self.conv2_bias),
            Conv1dParams {
                stride: self.metadata.conv2_stride,
                padding: conv_pad,
                dilation: 1,
            },
            ConvActivation::GeluErf,
            ConvBlockSteps {
                conv: "audiotok_conv2",
                bias: "audiotok_conv2_bias",
                activation: "audiotok_conv2_gelu",
            },
            build_err,
        )?;
        let frame_count = conv_out_len(
            conv1_frames,
            self.metadata.conv_kernel_size,
            self.metadata.conv2_stride,
            conv_pad,
        )?;
        if frame_count == 0 {
            return Err(MimoAudiotokEncoderError::InvalidMelFeatures {
                reason: "audio too short: produces 0 encoder frames after conv stem".to_string(),
            });
        }

        // conv output is [time, channels]; transpose to [channels, time] (=
        // [d_model, frame_count]) for the transformer's row-per-token layout.
        let mut hidden = graph
            .permute(state, 1, 0, 2, 3)
            .map_err(|source| build_err("audiotok_seq_permute", source))?;
        hidden = graph
            .cont(hidden)
            .map_err(|source| build_err("audiotok_seq_cont", source))?;
        hidden = graph
            .reshape_2d(hidden, d_model, frame_count)
            .map_err(|source| build_err("audiotok_seq_reshape", source))?;

        let positions = graph
            .new_tensor_1d_i32(frame_count, "audiotok_positions")
            .map_err(|source| build_err("audiotok_positions", source))?;
        graph
            .set_input(positions)
            .map_err(|source| build_err("audiotok_positions_input", source))?;

        let rope_params = GgmlRopeExtParams::qwen_neox(
            self.metadata.head_dim,
            frame_count.max(1),
            self.metadata.rope_theta,
        )
        .map_err(|source| build_err("audiotok_rope_params", source))?;

        let mut skip_state: Option<GgmlCpuTensor> = None;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = run_encoder_layer(
                &mut graph,
                &self.arena,
                hidden,
                layer,
                positions,
                rope_params,
                frame_count,
                d_model,
                self.metadata.n_heads,
                self.metadata.head_dim,
            )?;
            // Blood lesson #1: capture the layer-`skip_layer_id` (1-indexed)
            // output; it gets added back onto the LAST layer's output below,
            // before the final LayerNorm -- it does NOT bypass layers
            // (skip_layer_id)..n_layers, which still run normally in between.
            if layer_idx + 1 == self.metadata.skip_layer_id {
                skip_state = Some(hidden);
            }
        }
        let skip_state =
            skip_state.ok_or_else(|| MimoAudiotokEncoderError::GraphExecutionFailed {
                reason: "skip_layer_id was never reached (n_layers too small)".to_string(),
            })?;
        hidden = graph
            .add(hidden, skip_state)
            .map_err(|source| build_err("audiotok_skip_add", source))?;
        hidden = apply_affine_layer_norm(
            &graph,
            hidden,
            LAYER_NORM_EPSILON,
            self.arena.graph_tensor(self.norm_weight),
            self.arena.graph_tensor(self.norm_bias),
            AffineLayerNormSteps {
                norm: "audiotok_final_norm",
                scale: "audiotok_final_norm_scale",
                bias: "audiotok_final_norm_bias",
            },
            build_err,
        )?;

        // Down-sample: [d_model, frame_count] -> [frame_count, d_model] (conv
        // data layout) -> conv1d(k=2,s=2,no bias) -> gelu -> back to
        // [d_model, down_frame_count], then down_sample_norm (blood lesson
        // #2's second 2x stride: conv2(2x) + down_sample(2x) = 4x total,
        // 100Hz mel -> 25Hz codec).
        let mut ds_input = graph
            .permute(hidden, 1, 0, 2, 3)
            .map_err(|source| build_err("audiotok_ds_permute_in", source))?;
        ds_input = graph
            .cont(ds_input)
            .map_err(|source| build_err("audiotok_ds_cont_in", source))?;
        ds_input = graph
            .reshape_2d(ds_input, frame_count, d_model)
            .map_err(|source| build_err("audiotok_ds_reshape_in", source))?;
        let mut down = apply_conv_1d_bias_activation(
            &graph,
            self.arena.graph_tensor(self.down_sample_weight),
            ds_input,
            self.arena.graph_tensor(self.down_sample_zero_bias),
            Conv1dParams {
                stride: self.metadata.down_sample_stride,
                padding: 0,
                dilation: 1,
            },
            ConvActivation::GeluErf,
            ConvBlockSteps {
                conv: "audiotok_ds_conv",
                bias: "audiotok_ds_conv_bias",
                activation: "audiotok_ds_gelu",
            },
            build_err,
        )?;
        let down_frame_count = conv_out_len(frame_count, 2, self.metadata.down_sample_stride, 0)?;
        if down_frame_count == 0 {
            return Err(MimoAudiotokEncoderError::InvalidMelFeatures {
                reason: "audio too short: produces 0 frames after down-sample".to_string(),
            });
        }
        down = graph
            .permute(down, 1, 0, 2, 3)
            .map_err(|source| build_err("audiotok_ds_permute_out", source))?;
        down = graph
            .cont(down)
            .map_err(|source| build_err("audiotok_ds_cont_out", source))?;
        down = graph
            .reshape_2d(down, d_model, down_frame_count)
            .map_err(|source| build_err("audiotok_ds_reshape_out", source))?;
        down = apply_affine_layer_norm(
            &graph,
            down,
            LAYER_NORM_EPSILON,
            self.arena.graph_tensor(self.down_sample_norm_weight),
            self.arena.graph_tensor(self.down_sample_norm_bias),
            AffineLayerNormSteps {
                norm: "audiotok_ds_norm",
                scale: "audiotok_ds_norm_scale",
                bias: "audiotok_ds_norm_bias",
            },
            build_err,
        )?;

        graph
            .set_output(down)
            .map_err(|source| build_err("audiotok_output", source))?;
        graph
            .prepare_outputs_for_upload(&[down])
            .map_err(|source| build_err("audiotok_prepare_outputs", source))?;

        graph
            .set_f32_slice(mel_input, &mel.data, "audiotok_mel")
            .map_err(|source| build_err("audiotok_mel_upload", source))?;
        let position_values: Vec<i32> = (0..frame_count as i32).collect();
        graph
            .set_i32_slice(positions, &position_values, "audiotok_positions")
            .map_err(|source| build_err("audiotok_positions_upload", source))?;

        let total = d_model
            .checked_mul(down_frame_count)
            .ok_or(MimoAudiotokEncoderError::ShapeOverflow)?;
        let rows = graph.compute_output_f32(down, total).map_err(|error| {
            MimoAudiotokEncoderError::GraphExecutionFailed {
                reason: error.to_string(),
            }
        })?;
        Ok(MimoAudiotokEncoderOutput {
            frame_count: down_frame_count,
            d_model,
            rows,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn run_encoder_layer<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    arena: &GgmlStaticTensorArena,
    state: GgmlCpuTensor<'a>,
    layer: &LayerRuntime,
    positions: GgmlCpuTensor<'a>,
    rope_params: GgmlRopeExtParams,
    frame_count: usize,
    d_model: usize,
    heads: usize,
    head_dim: usize,
) -> Result<GgmlCpuTensor<'a>, MimoAudiotokEncoderError> {
    let residual = state;
    let normed = apply_affine_layer_norm(
        graph,
        state,
        LAYER_NORM_EPSILON,
        arena.graph_tensor(layer.attn_norm),
        arena.graph_tensor(layer.attn_norm_bias),
        AffineLayerNormSteps {
            norm: "audiotok_attn_norm",
            scale: "audiotok_attn_norm_scale",
            bias: "audiotok_attn_norm_bias",
        },
        build_err,
    )?;

    let mut q = graph
        .mul_mat(layer.attn_q.as_graph_tensor(), normed)
        .map_err(|source| build_err("audiotok_q", source))?;
    q = graph
        .add(q, arena.graph_tensor(layer.attn_q_bias))
        .map_err(|source| build_err("audiotok_q_bias", source))?;
    // Asymmetric qkv bias: k has NO bias (P2.0 finding, `k_proj = nn.Linear(..., bias=False)`).
    let k = graph
        .mul_mat(layer.attn_k.as_graph_tensor(), normed)
        .map_err(|source| build_err("audiotok_k", source))?;
    let mut v = graph
        .mul_mat(layer.attn_v.as_graph_tensor(), normed)
        .map_err(|source| build_err("audiotok_v", source))?;
    v = graph
        .add(v, arena.graph_tensor(layer.attn_v_bias))
        .map_err(|source| build_err("audiotok_v_bias", source))?;

    let q = rope_heads(
        graph,
        q,
        head_dim,
        heads,
        frame_count,
        positions,
        rope_params,
        "audiotok_q_rope",
    )?;
    let k = rope_heads(
        graph,
        k,
        head_dim,
        heads,
        frame_count,
        positions,
        rope_params,
        "audiotok_k_rope",
    )?;
    let q = roped_to_attn(graph, q, "audiotok_q_attn")?;
    let k = roped_to_attn(graph, k, "audiotok_k_attn")?;
    let v = reshape_heads_for_attn(graph, v, head_dim, heads, frame_count, "audiotok_v_attn")?;

    let scale = 1.0 / (head_dim as f32).sqrt();
    let context = scaled_dot_product_attention(graph, q, k, v, scale, frame_count, d_model)?;
    let mut attn = graph
        .mul_mat(layer.attn_out.as_graph_tensor(), context)
        .map_err(|source| build_err("audiotok_out", source))?;
    attn = graph
        .add(attn, arena.graph_tensor(layer.attn_out_bias))
        .map_err(|source| build_err("audiotok_out_bias", source))?;
    let state = graph
        .add(residual, attn)
        .map_err(|source| build_err("audiotok_attn_residual", source))?;

    let ffn_residual = state;
    let normed = apply_affine_layer_norm(
        graph,
        state,
        LAYER_NORM_EPSILON,
        arena.graph_tensor(layer.ffn_norm),
        arena.graph_tensor(layer.ffn_norm_bias),
        AffineLayerNormSteps {
            norm: "audiotok_ffn_norm",
            scale: "audiotok_ffn_norm_scale",
            bias: "audiotok_ffn_norm_bias",
        },
        build_err,
    )?;
    let mut ff = graph
        .mul_mat(layer.ffn_up.as_graph_tensor(), normed)
        .map_err(|source| build_err("audiotok_ffn_up", source))?;
    ff = graph
        .add(ff, arena.graph_tensor(layer.ffn_up_bias))
        .map_err(|source| build_err("audiotok_ffn_up_bias", source))?;
    ff = graph
        .gelu_erf(ff)
        .map_err(|source| build_err("audiotok_ffn_gelu", source))?;
    ff = graph
        .mul_mat(layer.ffn_down.as_graph_tensor(), ff)
        .map_err(|source| build_err("audiotok_ffn_down", source))?;
    ff = graph
        .add(ff, arena.graph_tensor(layer.ffn_down_bias))
        .map_err(|source| build_err("audiotok_ffn_down_bias", source))?;
    graph
        .add(ffn_residual, ff)
        .map_err(|source| build_err("audiotok_ffn_residual", source))
}

fn rope_heads<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    heads: usize,
    tokens: usize,
    positions: GgmlCpuTensor<'a>,
    params: GgmlRopeExtParams,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MimoAudiotokEncoderError> {
    let reshaped = graph
        .reshape_3d(projection, head_dim, heads, tokens)
        .map_err(|source| build_err("audiotok_rope_reshape", source))?;
    graph
        .rope_ext(reshaped, positions, params)
        .map_err(|source| MimoAudiotokEncoderError::GraphBuildFailed { step, source })
}

fn roped_to_attn<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    roped: GgmlCpuTensor<'a>,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MimoAudiotokEncoderError> {
    let permuted = graph
        .permute(roped, 0, 2, 1, 3)
        .map_err(|source| build_err("audiotok_rope_permute", source))?;
    graph
        .cont(permuted)
        .map_err(|source| MimoAudiotokEncoderError::GraphBuildFailed { step, source })
}

fn reshape_heads_for_attn<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    heads: usize,
    tokens: usize,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, MimoAudiotokEncoderError> {
    let reshaped = graph
        .reshape_3d(projection, head_dim, heads, tokens)
        .map_err(|source| build_err("audiotok_heads_reshape", source))?;
    let permuted = graph
        .permute(reshaped, 0, 2, 1, 3)
        .map_err(|source| build_err("audiotok_heads_permute", source))?;
    graph
        .cont(permuted)
        .map_err(|source| MimoAudiotokEncoderError::GraphBuildFailed { step, source })
}

/// Bidirectional (no mask) scaled dot-product attention; q,k,v laid out as
/// `[head_dim, seq, heads]`. Returns merged context `[d_model, q_len]`.
fn scaled_dot_product_attention<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    q: GgmlCpuTensor<'a>,
    k: GgmlCpuTensor<'a>,
    v: GgmlCpuTensor<'a>,
    scale: f32,
    q_len: usize,
    d_model: usize,
) -> Result<GgmlCpuTensor<'a>, MimoAudiotokEncoderError> {
    let scores = graph
        .mul_mat(k, q)
        .map_err(|source| build_err("audiotok_attn_scores", source))?;
    let probs = graph
        .soft_max_ext(scores, None, scale, 0.0)
        .map_err(|source| build_err("audiotok_attn_softmax", source))?;
    let v_t = graph
        .permute(v, 1, 0, 2, 3)
        .map_err(|source| build_err("audiotok_attn_v_t", source))?;
    let v_t = graph
        .cont(v_t)
        .map_err(|source| build_err("audiotok_attn_v_t_cont", source))?;
    let context = graph
        .mul_mat(v_t, probs)
        .map_err(|source| build_err("audiotok_attn_ctx", source))?;
    let merged = graph
        .permute(context, 0, 2, 1, 3)
        .map_err(|source| build_err("audiotok_attn_merge", source))?;
    let merged = graph
        .cont(merged)
        .map_err(|source| build_err("audiotok_attn_merge_cont", source))?;
    graph
        .reshape_2d(merged, d_model, q_len)
        .map_err(|source| build_err("audiotok_attn_merge_reshape", source))
}

fn conv_out_len(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<usize, MimoAudiotokEncoderError> {
    let padded = input
        .checked_add(2 * padding)
        .ok_or(MimoAudiotokEncoderError::ShapeOverflow)?;
    if padded < kernel {
        return Ok(0);
    }
    padded
        .checked_sub(kernel)
        .and_then(|value| value.checked_div(stride))
        .and_then(|value| value.checked_add(1))
        .ok_or(MimoAudiotokEncoderError::ShapeOverflow)
}

fn new_vector(
    arena: &GgmlStaticTensorArena,
    len: usize,
    name: &'static str,
) -> Result<GgmlStaticTensor, MimoAudiotokEncoderError> {
    arena
        .new_tensor_1d_f32(len, name)
        .map_err(|source| MimoAudiotokEncoderError::GraphBuildFailed { step: name, source })
}

/// A `[1, channels]` per-channel conv bias (ne0=1 so it broadcasts over the
/// conv output's ne0=time axis; see `new`'s conv-bias doc comment).
fn new_channel_bias(
    arena: &GgmlStaticTensorArena,
    channels: usize,
    name: &'static str,
) -> Result<GgmlStaticTensor, MimoAudiotokEncoderError> {
    arena
        .new_tensor_2d_f32(1, channels, name)
        .map_err(|source| MimoAudiotokEncoderError::GraphBuildFailed { step: name, source })
}

fn upload(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    reader: &GgufTensorDataReader,
    name: &str,
    shape: &[u64],
) -> Result<(), MimoAudiotokEncoderError> {
    let values = reader
        .host_tensor_f32_copy_dequantized_by_name(name, shape)
        .map_err(|source| MimoAudiotokEncoderError::TensorRead {
            name: name.to_string(),
            source,
        })?;
    arena
        .set_f32_slice(tensor, &values, "mimo_audiotok_upload")
        .map_err(|source| MimoAudiotokEncoderError::GraphBuildFailed {
            step: "mimo_audiotok_upload",
            source,
        })
}

/// Uploads a conv kernel's exact stored f16 bits (no f32-dequant-then-f32-to-f16
/// round trip) into an arena tensor.
fn upload_f16(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    reader: &GgufTensorDataReader,
    name: &str,
    shape: &[u64],
) -> Result<(), MimoAudiotokEncoderError> {
    let bits = reader
        .host_tensor_f16_bits_copy_by_name(name, shape)
        .map_err(|source| MimoAudiotokEncoderError::TensorRead {
            name: name.to_string(),
            source,
        })?;
    arena
        .set_f16_bits_slice(tensor, &bits, "mimo_audiotok_upload_f16")
        .map_err(|source| MimoAudiotokEncoderError::GraphBuildFailed {
            step: "mimo_audiotok_upload_f16",
            source,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_out_len_matches_pytorch_formula() {
        // conv1: k=3,s=1,pad=1 -> same length.
        assert_eq!(conv_out_len(100, 3, 1, 1).unwrap(), 100);
        // conv2: k=3,s=2,pad=1 -> ceil-ish halving (pytorch floor formula).
        assert_eq!(conv_out_len(100, 3, 2, 1).unwrap(), 50);
        // down_sample: k=2,s=2,pad=0 -> exact halving.
        assert_eq!(conv_out_len(50, 2, 2, 0).unwrap(), 25);
    }
}
