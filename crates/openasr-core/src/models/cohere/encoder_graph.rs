use std::path::Path;
use std::time::Instant;

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlLoadedTensor, GgmlLoadedWeightContext,
    GgmlStaticTensor, GgmlStaticTensorArena,
};
use crate::nn::conv::{
    Conv2dParams, ConvActivation, ConvBlockSteps, apply_conv_2d_bias_activation,
    apply_conv_2d_depthwise_bias_activation, reshape_bias_4d as nn_reshape_bias_4d,
};

use super::encoder_weights::{CohereEncoderLayerWeights, CohereTranscribeEncoderWeights};
use super::frontend::CohereTranscribeMelFeatures;
use super::graph_config::cohere_encoder_graph_config;
use super::runtime_contract::CohereTranscribeExecutionMetadata;

const COHERE_ENCODER_LAYER_NORM_EPSILON: f32 = 1.0e-5;
const COHERE_ENCODER_GRAPH_CONTEXT_BYTES: usize = 512 * 1024 * 1024;
const GGML_TYPE_F16: i32 = 1;
const COHERE_DEBUG_ENCODER_ENV: &str = "OPENASR_COHERE_DEBUG_ENCODER";
const COHERE_DEBUG_ENCODER_BUILD_ENV: &str = "OPENASR_COHERE_DEBUG_ENCODER_BUILD";

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CohereTranscribeEncoderOutput {
    pub frame_count: usize,
    pub hidden_size: usize,
    // Layout: [frame][hidden] contiguous f32.
    pub rows: Vec<f32>,
}

#[derive(Debug, Error)]
pub(crate) enum CohereTranscribeEncoderError {
    #[error("cohere-transcribe encoder features are invalid: {reason}")]
    InvalidFeatures { reason: String },
    #[error("cohere-transcribe encoder tensor '{tensor_name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        tensor_name: String,
        shape: String,
        reason: String,
    },
    #[error("cohere-transcribe encoder graph construction failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        #[source]
        source: GgmlCpuGraphError,
    },
    #[error("cohere-transcribe encoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("cohere-transcribe encoder shape overflowed")]
    ShapeOverflow,
}

struct CohereEncoderPreludeRuntime {
    pre_conv0_weight: GgmlStaticTensor,
    pre_conv0_bias: GgmlStaticTensor,
    pre_conv0_bias_len: usize,
    pre_conv2_weight: GgmlStaticTensor,
    pre_conv2_bias: GgmlStaticTensor,
    pre_conv2_bias_len: usize,
    pre_conv3_weight: GgmlStaticTensor,
    pre_conv3_bias: GgmlStaticTensor,
    pre_conv3_bias_len: usize,
    pre_conv5_weight: GgmlStaticTensor,
    pre_conv5_bias: GgmlStaticTensor,
    pre_conv5_bias_len: usize,
    pre_conv6_weight: GgmlStaticTensor,
    pre_conv6_bias: GgmlStaticTensor,
    pre_conv6_bias_len: usize,
    pre_out_weight: GgmlStaticTensor,
    pre_out_bias: GgmlStaticTensor,
    enc_proj_weight: GgmlStaticTensor,
    enc_proj_bias: GgmlStaticTensor,
}

struct CohereEncoderLayerRuntime {
    ff1_norm_weight: GgmlStaticTensor,
    ff1_norm_bias: GgmlStaticTensor,
    ff1_up_weight: GgmlStaticTensor,
    ff1_up_bias: GgmlStaticTensor,
    ff1_down_weight: GgmlStaticTensor,
    ff1_down_bias: GgmlStaticTensor,
    attn_norm_weight: GgmlStaticTensor,
    attn_norm_bias: GgmlStaticTensor,
    attn_q_weight: GgmlStaticTensor,
    attn_q_bias: GgmlStaticTensor,
    attn_k_weight: GgmlStaticTensor,
    attn_k_bias: GgmlStaticTensor,
    attn_v_weight: GgmlStaticTensor,
    attn_v_bias: GgmlStaticTensor,
    attn_out_weight: GgmlStaticTensor,
    attn_out_bias: GgmlStaticTensor,
    attn_pos_weight: GgmlStaticTensor,
    attn_pos_bias_u: GgmlStaticTensor,
    attn_pos_bias_v: GgmlStaticTensor,
    conv_norm_weight: GgmlStaticTensor,
    conv_norm_bias: GgmlStaticTensor,
    conv_pw1_weight: GgmlStaticTensor,
    conv_pw1_bias: GgmlStaticTensor,
    conv_dw_weight: GgmlStaticTensor,
    conv_dw_bias: GgmlStaticTensor,
    conv_pw2_weight: GgmlStaticTensor,
    conv_pw2_bias: GgmlStaticTensor,
    ff2_norm_weight: GgmlStaticTensor,
    ff2_norm_bias: GgmlStaticTensor,
    ff2_up_weight: GgmlStaticTensor,
    ff2_up_bias: GgmlStaticTensor,
    ff2_down_weight: GgmlStaticTensor,
    ff2_down_bias: GgmlStaticTensor,
    out_norm_weight: GgmlStaticTensor,
    out_norm_bias: GgmlStaticTensor,
}

pub(crate) struct CohereTranscribeEncoderGraphRuntime {
    metadata: CohereTranscribeExecutionMetadata,
    runner: GgmlCpuGraphRunner,
    loaded_weights: Option<GgmlLoadedWeightContext>,
    arena: GgmlStaticTensorArena,
    prelude: CohereEncoderPreludeRuntime,
    layers: Vec<CohereEncoderLayerRuntime>,
}

fn has_loaded_tensor(loaded_weights: Option<&GgmlLoadedWeightContext>, name: &str) -> bool {
    loaded_weights
        .and_then(|loaded| loaded.tensor(name))
        .is_some()
}

fn loaded_or_static_tensor<'a>(
    loaded: Option<GgmlLoadedTensor>,
    arena: &GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
) -> crate::ggml_runtime::GgmlCpuTensor<'a> {
    loaded
        .map(GgmlLoadedTensor::as_graph_tensor)
        .unwrap_or_else(|| arena.graph_tensor(tensor))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn encode_cohere_transcribe_audio_embeddings_from_weights(
    weights: &CohereTranscribeEncoderWeights,
    metadata: CohereTranscribeExecutionMetadata,
    mel_features: &CohereTranscribeMelFeatures,
) -> Result<CohereTranscribeEncoderOutput, CohereTranscribeEncoderError> {
    let mut runtime = CohereTranscribeEncoderGraphRuntime::new(weights, metadata, None)?;
    runtime.encode(mel_features)
}

impl CohereTranscribeEncoderGraphRuntime {
    pub(crate) fn new(
        weights: &CohereTranscribeEncoderWeights,
        metadata: CohereTranscribeExecutionMetadata,
        runtime_path: Option<&Path>,
    ) -> Result<Self, CohereTranscribeEncoderError> {
        let build_debug = std::env::var_os(COHERE_DEBUG_ENCODER_BUILD_ENV).is_some();
        let runner_start = Instant::now();
        let mut config = cohere_encoder_graph_config();
        config.context_bytes = COHERE_ENCODER_GRAPH_CONTEXT_BYTES;
        let runner = GgmlCpuGraphRunner::new(config).map_err(|source| {
            CohereTranscribeEncoderError::GraphBuildFailed {
                step: "runner_init",
                source,
            }
        })?;
        let runner_ms = runner_start.elapsed().as_secs_f64() * 1000.0;
        let loaded_weights =
            runtime_path.and_then(|path| runner.load_gguf_weight_context(path).ok());
        let arena_start = Instant::now();
        let mut arena = runner
            .start_static_tensor_arena(config.context_bytes)
            .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed {
                step: "static_tensor_arena",
                source,
            })?;
        let arena_ms = arena_start.elapsed().as_secs_f64() * 1000.0;

        let prelude_decl_start = Instant::now();
        let prelude = CohereEncoderPreludeRuntime {
            pre_conv0_weight: new_static_tensor_4d_from_dims(
                &arena,
                &weights.pre_conv0_weight.dims,
                "enc_pre_conv0_weight",
            )?,
            pre_conv0_bias: new_static_tensor_1d_from_len(
                &arena,
                weights.pre_conv0_bias.values.len(),
                "enc_pre_conv0_bias",
            )?,
            pre_conv0_bias_len: weights.pre_conv0_bias.values.len(),
            pre_conv2_weight: new_static_tensor_4d_f16_from_dims(
                &arena,
                &weights.pre_conv2_weight.dims,
                "enc_pre_conv2_weight",
            )?,
            pre_conv2_bias: new_static_tensor_1d_from_len(
                &arena,
                weights.pre_conv2_bias.values.len(),
                "enc_pre_conv2_bias",
            )?,
            pre_conv2_bias_len: weights.pre_conv2_bias.values.len(),
            pre_conv3_weight: new_static_tensor_4d_from_dims(
                &arena,
                &weights.pre_conv3_weight.dims,
                "enc_pre_conv3_weight",
            )?,
            pre_conv3_bias: new_static_tensor_1d_from_len(
                &arena,
                weights.pre_conv3_bias.values.len(),
                "enc_pre_conv3_bias",
            )?,
            pre_conv3_bias_len: weights.pre_conv3_bias.values.len(),
            pre_conv5_weight: new_static_tensor_4d_f16_from_dims(
                &arena,
                &weights.pre_conv5_weight.dims,
                "enc_pre_conv5_weight",
            )?,
            pre_conv5_bias: new_static_tensor_1d_from_len(
                &arena,
                weights.pre_conv5_bias.values.len(),
                "enc_pre_conv5_bias",
            )?,
            pre_conv5_bias_len: weights.pre_conv5_bias.values.len(),
            pre_conv6_weight: new_static_tensor_4d_from_dims(
                &arena,
                &weights.pre_conv6_weight.dims,
                "enc_pre_conv6_weight",
            )?,
            pre_conv6_bias: new_static_tensor_1d_from_len(
                &arena,
                weights.pre_conv6_bias.values.len(),
                "enc_pre_conv6_bias",
            )?,
            pre_conv6_bias_len: weights.pre_conv6_bias.values.len(),
            pre_out_weight: new_static_projection_tensor(
                &arena,
                &weights.pre_out_weight,
                "enc_pre_out_weight",
            )?,
            pre_out_bias: new_static_tensor_1d_from_len(
                &arena,
                weights.pre_out_bias.values.len(),
                "enc_pre_out_bias",
            )?,
            enc_proj_weight: new_static_projection_tensor(
                &arena,
                &weights.encoder_projection_weight,
                "enc_proj_weight",
            )?,
            enc_proj_bias: new_static_tensor_1d_from_len(
                &arena,
                weights.encoder_projection_bias.values.len(),
                "enc_proj_bias",
            )?,
        };
        let prelude_decl_ms = prelude_decl_start.elapsed().as_secs_f64() * 1000.0;

        let layer_decl_start = Instant::now();
        let mut layers = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            let runtime = CohereEncoderLayerRuntime {
                ff1_norm_weight: new_static_tensor_1d_from_len(
                    &arena,
                    layer.ff1_norm_weight.values.len(),
                    "enc_ff1_norm_weight",
                )?,
                ff1_norm_bias: new_static_tensor_1d_from_len(
                    &arena,
                    layer.ff1_norm_bias.values.len(),
                    "enc_ff1_norm_bias",
                )?,
                ff1_up_weight: new_static_projection_tensor(
                    &arena,
                    &layer.ff1_up_weight,
                    "enc_ff1_up_weight",
                )?,
                ff1_up_bias: new_static_tensor_1d_from_len(
                    &arena,
                    layer.ff1_up_bias.values.len(),
                    "enc_ff1_up_bias",
                )?,
                ff1_down_weight: new_static_projection_tensor(
                    &arena,
                    &layer.ff1_down_weight,
                    "enc_ff1_down_weight",
                )?,
                ff1_down_bias: new_static_tensor_1d_from_len(
                    &arena,
                    layer.ff1_down_bias.values.len(),
                    "enc_ff1_down_bias",
                )?,
                attn_norm_weight: new_static_tensor_1d_from_len(
                    &arena,
                    layer.attn_norm_weight.values.len(),
                    "enc_attn_norm_weight",
                )?,
                attn_norm_bias: new_static_tensor_1d_from_len(
                    &arena,
                    layer.attn_norm_bias.values.len(),
                    "enc_attn_norm_bias",
                )?,
                attn_q_weight: new_static_projection_tensor(
                    &arena,
                    &layer.attn_q_weight,
                    "enc_attn_q_weight",
                )?,
                attn_q_bias: new_static_tensor_1d_from_len(
                    &arena,
                    layer.attn_q_bias.values.len(),
                    "enc_attn_q_bias",
                )?,
                attn_k_weight: new_static_projection_tensor(
                    &arena,
                    &layer.attn_k_weight,
                    "enc_attn_k_weight",
                )?,
                attn_k_bias: new_static_tensor_1d_from_len(
                    &arena,
                    layer.attn_k_bias.values.len(),
                    "enc_attn_k_bias",
                )?,
                attn_v_weight: new_static_projection_tensor(
                    &arena,
                    &layer.attn_v_weight,
                    "enc_attn_v_weight",
                )?,
                attn_v_bias: new_static_tensor_1d_from_len(
                    &arena,
                    layer.attn_v_bias.values.len(),
                    "enc_attn_v_bias",
                )?,
                attn_out_weight: new_static_projection_tensor(
                    &arena,
                    &layer.attn_out_weight,
                    "enc_attn_out_weight",
                )?,
                attn_out_bias: new_static_tensor_1d_from_len(
                    &arena,
                    layer.attn_out_bias.values.len(),
                    "enc_attn_out_bias",
                )?,
                attn_pos_weight: new_static_projection_tensor(
                    &arena,
                    &layer.attn_pos_weight,
                    "enc_attn_pos_weight",
                )?,
                attn_pos_bias_u: new_static_row_major_matrix_tensor(
                    &arena,
                    &layer.attn_pos_bias_u,
                    "enc_attn_pos_bias_u",
                )?,
                attn_pos_bias_v: new_static_row_major_matrix_tensor(
                    &arena,
                    &layer.attn_pos_bias_v,
                    "enc_attn_pos_bias_v",
                )?,
                conv_norm_weight: new_static_tensor_1d_from_len(
                    &arena,
                    layer.conv_norm_weight.values.len(),
                    "enc_conv_norm_weight",
                )?,
                conv_norm_bias: new_static_tensor_1d_from_len(
                    &arena,
                    layer.conv_norm_bias.values.len(),
                    "enc_conv_norm_bias",
                )?,
                conv_pw1_weight: new_static_tensor_2d_or_3d_from_weight(
                    &arena,
                    &layer.conv_pw1_weight,
                    "enc_conv_pw1_weight",
                )?,
                conv_pw1_bias: new_static_tensor_1d_from_len(
                    &arena,
                    layer.conv_pw1_bias.values.len(),
                    "enc_conv_pw1_bias",
                )?,
                conv_dw_weight: new_static_tensor_2d_or_3d_f16_from_dims(
                    &arena,
                    &layer.conv_dw_weight.dims,
                    "enc_conv_dw_weight",
                )?,
                conv_dw_bias: new_static_tensor_1d_from_len(
                    &arena,
                    layer.conv_dw_bias.values.len(),
                    "enc_conv_dw_bias",
                )?,
                conv_pw2_weight: new_static_tensor_2d_or_3d_from_weight(
                    &arena,
                    &layer.conv_pw2_weight,
                    "enc_conv_pw2_weight",
                )?,
                conv_pw2_bias: new_static_tensor_1d_from_len(
                    &arena,
                    layer.conv_pw2_bias.values.len(),
                    "enc_conv_pw2_bias",
                )?,
                ff2_norm_weight: new_static_tensor_1d_from_len(
                    &arena,
                    layer.ff2_norm_weight.values.len(),
                    "enc_ff2_norm_weight",
                )?,
                ff2_norm_bias: new_static_tensor_1d_from_len(
                    &arena,
                    layer.ff2_norm_bias.values.len(),
                    "enc_ff2_norm_bias",
                )?,
                ff2_up_weight: new_static_projection_tensor(
                    &arena,
                    &layer.ff2_up_weight,
                    "enc_ff2_up_weight",
                )?,
                ff2_up_bias: new_static_tensor_1d_from_len(
                    &arena,
                    layer.ff2_up_bias.values.len(),
                    "enc_ff2_up_bias",
                )?,
                ff2_down_weight: new_static_projection_tensor(
                    &arena,
                    &layer.ff2_down_weight,
                    "enc_ff2_down_weight",
                )?,
                ff2_down_bias: new_static_tensor_1d_from_len(
                    &arena,
                    layer.ff2_down_bias.values.len(),
                    "enc_ff2_down_bias",
                )?,
                out_norm_weight: new_static_tensor_1d_from_len(
                    &arena,
                    layer.out_norm_weight.values.len(),
                    "enc_out_norm_weight",
                )?,
                out_norm_bias: new_static_tensor_1d_from_len(
                    &arena,
                    layer.out_norm_bias.values.len(),
                    "enc_out_norm_bias",
                )?,
            };
            layers.push(runtime);
        }
        let layer_decl_ms = layer_decl_start.elapsed().as_secs_f64() * 1000.0;

        let prelude_upload_start = Instant::now();
        upload_static_tensor_weight_with_expected_type(
            &mut arena,
            prelude.pre_conv0_weight,
            &weights.pre_conv0_weight,
            crate::ggml_runtime::GGML_TYPE_F32,
            "enc_pre_conv0_weight",
        )?;
        upload_static_f32(
            &mut arena,
            prelude.pre_conv0_bias,
            &weights.pre_conv0_bias.values,
            "enc_pre_conv0_bias",
        )?;
        upload_static_f16_from_weight(
            &mut arena,
            prelude.pre_conv2_weight,
            &weights.pre_conv2_weight,
            "enc_pre_conv2_weight",
        )?;
        upload_static_f32(
            &mut arena,
            prelude.pre_conv2_bias,
            &weights.pre_conv2_bias.values,
            "enc_pre_conv2_bias",
        )?;
        upload_static_tensor_weight_with_expected_type(
            &mut arena,
            prelude.pre_conv3_weight,
            &weights.pre_conv3_weight,
            crate::ggml_runtime::GGML_TYPE_F32,
            "enc_pre_conv3_weight",
        )?;
        upload_static_f32(
            &mut arena,
            prelude.pre_conv3_bias,
            &weights.pre_conv3_bias.values,
            "enc_pre_conv3_bias",
        )?;
        upload_static_f16_from_weight(
            &mut arena,
            prelude.pre_conv5_weight,
            &weights.pre_conv5_weight,
            "enc_pre_conv5_weight",
        )?;
        upload_static_f32(
            &mut arena,
            prelude.pre_conv5_bias,
            &weights.pre_conv5_bias.values,
            "enc_pre_conv5_bias",
        )?;
        upload_static_tensor_weight_with_expected_type(
            &mut arena,
            prelude.pre_conv6_weight,
            &weights.pre_conv6_weight,
            crate::ggml_runtime::GGML_TYPE_F32,
            "enc_pre_conv6_weight",
        )?;
        upload_static_f32(
            &mut arena,
            prelude.pre_conv6_bias,
            &weights.pre_conv6_bias.values,
            "enc_pre_conv6_bias",
        )?;
        if !has_loaded_tensor(loaded_weights.as_ref(), &weights.pre_out_weight.name) {
            upload_static_projection_f32(
                &mut arena,
                prelude.pre_out_weight,
                &weights.pre_out_weight,
                "enc_pre_out_weight",
            )?;
        }
        upload_static_f32(
            &mut arena,
            prelude.pre_out_bias,
            &weights.pre_out_bias.values,
            "enc_pre_out_bias",
        )?;
        if !has_loaded_tensor(
            loaded_weights.as_ref(),
            &weights.encoder_projection_weight.name,
        ) {
            upload_static_projection_f32(
                &mut arena,
                prelude.enc_proj_weight,
                &weights.encoder_projection_weight,
                "enc_proj_weight",
            )?;
        }
        upload_static_f32(
            &mut arena,
            prelude.enc_proj_bias,
            &weights.encoder_projection_bias.values,
            "enc_proj_bias",
        )?;
        let prelude_upload_ms = prelude_upload_start.elapsed().as_secs_f64() * 1000.0;

        let layer_upload_start = Instant::now();
        for (layer, runtime) in weights.layers.iter().zip(layers.iter()) {
            upload_static_encoder_layer(&mut arena, loaded_weights.as_ref(), layer, runtime)?;
        }
        let layer_upload_ms = layer_upload_start.elapsed().as_secs_f64() * 1000.0;
        if build_debug {
            eprintln!(
                "openasr cohere encoder-build: runner_ms={runner_ms:.2} arena_ms={arena_ms:.2} prelude_decl_ms={prelude_decl_ms:.2} layer_decl_ms={layer_decl_ms:.2} prelude_upload_ms={prelude_upload_ms:.2} layer_upload_ms={layer_upload_ms:.2} layers={}",
                weights.layers.len(),
            );
        }

        Ok(Self {
            metadata,
            runner,
            loaded_weights,
            arena,
            prelude,
            layers,
        })
    }

    pub(crate) fn encode(
        &mut self,
        mel_features: &CohereTranscribeMelFeatures,
    ) -> Result<CohereTranscribeEncoderOutput, CohereTranscribeEncoderError> {
        let metadata = self.metadata;
        validate_mel_features(metadata, mel_features)?;
        let subsampled_freq = conv_out_dim(metadata.n_mels, 3, 2, 1)?;
        let subsampled_freq = conv_out_dim(subsampled_freq, 3, 2, 1)?;
        let subsampled_freq = conv_out_dim(subsampled_freq, 3, 2, 1)?;
        let subsampled_frames = conv_out_dim(mel_features.n_frames, 3, 2, 1)?;
        let subsampled_frames = conv_out_dim(subsampled_frames, 3, 2, 1)?;
        let subsampled_frames = conv_out_dim(subsampled_frames, 3, 2, 1)?;
        let positional =
            build_relative_positional_encoding(metadata.encoder_d_model, subsampled_frames)?;

        let mut graph = self.runner.start_graph();
        let mel = graph
            .new_tensor_2d_f32(metadata.n_mels, mel_features.n_frames, "cohere_mel")
            .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed {
                step: "ggml_new_tensor_2d(mel)",
                source,
            })?;
        let pos_enc = graph
            .new_tensor_2d_f32(
                metadata.encoder_d_model,
                positional.len() / metadata.encoder_d_model,
                "cohere_rel_pos",
            )
            .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed {
                step: "ggml_new_tensor_2d(pos_enc)",
                source,
            })?;
        graph
            .set_input(mel)
            .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed {
                step: "ggml_set_input(mel)",
                source,
            })?;
        graph.set_input(pos_enc).map_err(|source| {
            CohereTranscribeEncoderError::GraphBuildFailed {
                step: "ggml_set_input(pos_enc)",
                source,
            }
        })?;

        let map_graph_error =
            |step, source| CohereTranscribeEncoderError::GraphBuildFailed { step, source };
        let prelude_stride2 = Conv2dParams {
            stride_x: 2,
            stride_y: 2,
            padding_x: 1,
            padding_y: 1,
            dilation_x: 1,
            dilation_y: 1,
        };
        let prelude_pointwise = Conv2dParams {
            stride_x: 1,
            stride_y: 1,
            padding_x: 0,
            padding_y: 0,
            dilation_x: 1,
            dilation_y: 1,
        };
        let pre_conv0_bias_4d = nn_reshape_bias_4d(
            &graph,
            self.arena.graph_tensor(self.prelude.pre_conv0_bias),
            self.prelude.pre_conv0_bias_len,
            "ggml_reshape_4d(bias)",
            map_graph_error,
        )?;
        let pre_conv2_bias_4d = nn_reshape_bias_4d(
            &graph,
            self.arena.graph_tensor(self.prelude.pre_conv2_bias),
            self.prelude.pre_conv2_bias_len,
            "ggml_reshape_4d(bias)",
            map_graph_error,
        )?;
        let pre_conv3_bias_4d = nn_reshape_bias_4d(
            &graph,
            self.arena.graph_tensor(self.prelude.pre_conv3_bias),
            self.prelude.pre_conv3_bias_len,
            "ggml_reshape_4d(bias)",
            map_graph_error,
        )?;
        let pre_conv5_bias_4d = nn_reshape_bias_4d(
            &graph,
            self.arena.graph_tensor(self.prelude.pre_conv5_bias),
            self.prelude.pre_conv5_bias_len,
            "ggml_reshape_4d(bias)",
            map_graph_error,
        )?;
        let pre_conv6_bias_4d = nn_reshape_bias_4d(
            &graph,
            self.arena.graph_tensor(self.prelude.pre_conv6_bias),
            self.prelude.pre_conv6_bias_len,
            "ggml_reshape_4d(bias)",
            map_graph_error,
        )?;

        let mut state_4d = graph
            .reshape_4d(mel, metadata.n_mels, mel_features.n_frames, 1, 1)
            .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed {
                step: "ggml_reshape_4d(mel)",
                source,
            })?;
        state_4d = apply_conv_2d_bias_activation(
            &graph,
            self.arena.graph_tensor(self.prelude.pre_conv0_weight),
            state_4d,
            pre_conv0_bias_4d,
            prelude_stride2,
            ConvActivation::Relu,
            ConvBlockSteps {
                conv: "ggml_conv_2d(pre_conv0)",
                bias: "ggml_add(pre_conv0_bias)",
                activation: "ggml_relu(pre_conv0)",
            },
            map_graph_error,
        )?;

        state_4d = apply_conv_2d_depthwise_bias_activation(
            &graph,
            self.arena.graph_tensor(self.prelude.pre_conv2_weight),
            state_4d,
            pre_conv2_bias_4d,
            prelude_stride2,
            None,
            ConvBlockSteps {
                conv: "ggml_conv_2d_dw(pre_conv2)",
                bias: "ggml_add(pre_conv2_bias)",
                activation: "ggml_relu(pre_conv2)",
            },
            map_graph_error,
        )?;
        state_4d = apply_conv_2d_bias_activation(
            &graph,
            self.arena.graph_tensor(self.prelude.pre_conv3_weight),
            state_4d,
            pre_conv3_bias_4d,
            prelude_pointwise,
            ConvActivation::Relu,
            ConvBlockSteps {
                conv: "ggml_conv_2d(pre_conv3)",
                bias: "ggml_add(pre_conv3_bias)",
                activation: "ggml_relu(pre_conv3)",
            },
            map_graph_error,
        )?;

        state_4d = apply_conv_2d_depthwise_bias_activation(
            &graph,
            self.arena.graph_tensor(self.prelude.pre_conv5_weight),
            state_4d,
            pre_conv5_bias_4d,
            prelude_stride2,
            None,
            ConvBlockSteps {
                conv: "ggml_conv_2d_dw(pre_conv5)",
                bias: "ggml_add(pre_conv5_bias)",
                activation: "ggml_relu(pre_conv5)",
            },
            map_graph_error,
        )?;
        state_4d = apply_conv_2d_bias_activation(
            &graph,
            self.arena.graph_tensor(self.prelude.pre_conv6_weight),
            state_4d,
            pre_conv6_bias_4d,
            prelude_pointwise,
            ConvActivation::Relu,
            ConvBlockSteps {
                conv: "ggml_conv_2d(pre_conv6)",
                bias: "ggml_add(pre_conv6_bias)",
                activation: "ggml_relu(pre_conv6)",
            },
            map_graph_error,
        )?;

        let conv_channels = self.prelude.pre_conv6_bias_len;
        let flattened_width = conv_channels
            .checked_mul(subsampled_freq)
            .ok_or(CohereTranscribeEncoderError::ShapeOverflow)?;

        let mut state = graph.permute(state_4d, 0, 2, 1, 3).map_err(|source| {
            CohereTranscribeEncoderError::GraphBuildFailed {
                step: "ggml_permute(pre_flatten)",
                source,
            }
        })?;
        state =
            graph
                .cont(state)
                .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed {
                    step: "ggml_cont(pre_flatten)",
                    source,
                })?;
        state = graph
            .reshape_2d(state, flattened_width, subsampled_frames)
            .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed {
                step: "ggml_reshape_2d(pre_flatten)",
                source,
            })?;
        state = graph
            .mul_mat(
                loaded_or_static_tensor(
                    self.loaded_weights
                        .as_ref()
                        .and_then(|loaded| loaded.tensor("enc.pre.out.weight")),
                    &self.arena,
                    self.prelude.pre_out_weight,
                ),
                state,
            )
            .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed {
                step: "ggml_mul_mat(pre_out)",
                source,
            })?;
        state = graph
            .add(state, self.arena.graph_tensor(self.prelude.pre_out_bias))
            .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed {
                step: "ggml_add(pre_out_bias)",
                source,
            })?;
        let pre_subsample_state = state;

        let capture_debug = std::env::var_os(COHERE_DEBUG_ENCODER_ENV).is_some();
        let (next_state, layer0_debug) = compose_conformer_encoder_layer_stack(
            &mut graph,
            state,
            &self.layers,
            &self.arena,
            self.loaded_weights.as_ref(),
            pos_enc,
            metadata,
            subsampled_frames,
            capture_debug,
        )?;
        state = next_state;

        let pre_projection_state = state;
        state = graph
            .mul_mat(
                loaded_or_static_tensor(
                    self.loaded_weights
                        .as_ref()
                        .and_then(|loaded| loaded.tensor("enc.proj.weight")),
                    &self.arena,
                    self.prelude.enc_proj_weight,
                ),
                state,
            )
            .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed {
                step: "ggml_mul_mat(enc_proj)",
                source,
            })?;
        state = graph
            .add(state, self.arena.graph_tensor(self.prelude.enc_proj_bias))
            .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed {
                step: "ggml_add(enc_proj_bias)",
                source,
            })?;
        graph.set_output(state).map_err(|source| {
            CohereTranscribeEncoderError::GraphBuildFailed {
                step: "ggml_set_output(enc_out)",
                source,
            }
        })?;

        // Peak-RSS lever: allocate the compute graph via the scheduler's gallocr
        // (liveness-based buffer REUSE) before uploading inputs, collapsing the
        // per-conformer-layer intermediate accumulation to the working-set peak.
        // Only on the production path: the capture_debug path reads back many
        // intermediate stages as outputs, which gallocr reuse would free.
        if !capture_debug {
            graph
                .prepare_outputs_for_upload(&[state])
                .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed {
                    step: "ggml_prepare_outputs(enc_out)",
                    source,
                })?;
        }

        upload_f32(&mut graph, mel, &mel_features.data, "cohere_mel")?;
        upload_f32(&mut graph, pos_enc, &positional, "cohere_rel_pos")?;

        let rows = if capture_debug {
            let pre_subsample_len = metadata
                .encoder_d_model
                .checked_mul(subsampled_frames)
                .ok_or(CohereTranscribeEncoderError::ShapeOverflow)?;
            let pre_projection_len = metadata
                .encoder_d_model
                .checked_mul(subsampled_frames)
                .ok_or(CohereTranscribeEncoderError::ShapeOverflow)?;
            let output_len = metadata
                .decoder_d_model
                .checked_mul(subsampled_frames)
                .ok_or(CohereTranscribeEncoderError::ShapeOverflow)?;
            let mut requested_outputs = vec![
                (pre_subsample_state, pre_subsample_len),
                (pre_projection_state, pre_projection_len),
            ];
            if let Some(debug) = layer0_debug {
                requested_outputs.push((debug.ff1, pre_projection_len));
                requested_outputs.push((debug.attn, pre_projection_len));
                requested_outputs.push((debug.conv_glu, pre_projection_len));
                requested_outputs.push((debug.conv_dw_act, pre_projection_len));
                requested_outputs.push((debug.conv, pre_projection_len));
                requested_outputs.push((debug.ff2, pre_projection_len));
            }
            requested_outputs.push((state, output_len));
            let mut outputs = graph
                .compute_outputs_f32(&requested_outputs)
                .map_err(|error| CohereTranscribeEncoderError::GraphExecutionFailed {
                    reason: error.to_string(),
                })?;
            let pre_subsample_rows = outputs.remove(0);
            emit_cohere_debug_encoder_stage_preview(
                "pre_subsample_out",
                subsampled_frames,
                metadata.encoder_d_model,
                &pre_subsample_rows,
            );
            let pre_projection_rows = outputs.remove(0);
            emit_cohere_debug_encoder_stage_preview(
                "pre_proj",
                subsampled_frames,
                metadata.encoder_d_model,
                &pre_projection_rows,
            );
            if layer0_debug.is_some() {
                let ff1_rows = outputs.remove(0);
                emit_cohere_debug_encoder_stage_preview(
                    "L0_ff1",
                    subsampled_frames,
                    metadata.encoder_d_model,
                    &ff1_rows,
                );
                let attn_rows = outputs.remove(0);
                emit_cohere_debug_encoder_stage_preview(
                    "L0_attn",
                    subsampled_frames,
                    metadata.encoder_d_model,
                    &attn_rows,
                );
                let conv_glu_rows = outputs.remove(0);
                emit_cohere_debug_encoder_stage_preview(
                    "L0_conv_glu",
                    subsampled_frames,
                    metadata.encoder_d_model,
                    &conv_glu_rows,
                );
                let conv_dw_act_rows = outputs.remove(0);
                emit_cohere_debug_encoder_stage_preview(
                    "L0_conv_dw_act",
                    subsampled_frames,
                    metadata.encoder_d_model,
                    &conv_dw_act_rows,
                );
                let conv_rows = outputs.remove(0);
                emit_cohere_debug_encoder_stage_preview(
                    "L0_conv",
                    subsampled_frames,
                    metadata.encoder_d_model,
                    &conv_rows,
                );
                let ff2_rows = outputs.remove(0);
                emit_cohere_debug_encoder_stage_preview(
                    "L0_ff2",
                    subsampled_frames,
                    metadata.encoder_d_model,
                    &ff2_rows,
                );
            }
            outputs.remove(0)
        } else {
            graph
                .compute_output_f32(
                    state,
                    metadata
                        .decoder_d_model
                        .checked_mul(subsampled_frames)
                        .ok_or(CohereTranscribeEncoderError::ShapeOverflow)?,
                )
                .map_err(|error| CohereTranscribeEncoderError::GraphExecutionFailed {
                    reason: error.to_string(),
                })?
        };
        Ok(CohereTranscribeEncoderOutput {
            frame_count: subsampled_frames,
            hidden_size: metadata.decoder_d_model,
            rows,
        })
    }
}

#[derive(Clone, Copy)]
struct EncoderLayerGraphTensors<'a> {
    ff1_norm_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ff1_norm_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ff1_up_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ff1_up_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ff1_down_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ff1_down_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_norm_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_norm_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_q_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_q_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_k_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_k_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_v_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_v_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_out_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_out_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_pos_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_pos_bias_u: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_pos_bias_v: crate::ggml_runtime::GgmlCpuTensor<'a>,
    conv_norm_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    conv_norm_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    conv_pw1_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    conv_pw1_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    conv_dw_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    conv_dw_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    conv_pw2_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    conv_pw2_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ff2_norm_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ff2_norm_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ff2_up_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ff2_up_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ff2_down_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ff2_down_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    out_norm_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    out_norm_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
}

#[derive(Clone, Copy)]
struct EncoderLayerDebugTensors<'a> {
    ff1: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn: crate::ggml_runtime::GgmlCpuTensor<'a>,
    conv_glu: crate::ggml_runtime::GgmlCpuTensor<'a>,
    conv_dw_act: crate::ggml_runtime::GgmlCpuTensor<'a>,
    conv: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ff2: crate::ggml_runtime::GgmlCpuTensor<'a>,
}

struct EncoderLayerRunResult<'a> {
    output: crate::ggml_runtime::GgmlCpuTensor<'a>,
    debug: Option<EncoderLayerDebugTensors<'a>>,
}

impl CohereEncoderLayerRuntime {
    fn as_graph_tensors<'a>(
        &self,
        arena: &'a GgmlStaticTensorArena,
        loaded_weights: Option<&GgmlLoadedWeightContext>,
        layer_idx: usize,
    ) -> EncoderLayerGraphTensors<'a> {
        let prefix = format!("enc.blk.{layer_idx}.");
        EncoderLayerGraphTensors {
            ff1_norm_weight: arena.graph_tensor(self.ff1_norm_weight),
            ff1_norm_bias: arena.graph_tensor(self.ff1_norm_bias),
            ff1_up_weight: loaded_or_static_tensor(
                loaded_weights.and_then(|loaded| loaded.tensor(&format!("{prefix}ff1.up.weight"))),
                arena,
                self.ff1_up_weight,
            ),
            ff1_up_bias: arena.graph_tensor(self.ff1_up_bias),
            ff1_down_weight: loaded_or_static_tensor(
                loaded_weights
                    .and_then(|loaded| loaded.tensor(&format!("{prefix}ff1.down.weight"))),
                arena,
                self.ff1_down_weight,
            ),
            ff1_down_bias: arena.graph_tensor(self.ff1_down_bias),
            attn_norm_weight: arena.graph_tensor(self.attn_norm_weight),
            attn_norm_bias: arena.graph_tensor(self.attn_norm_bias),
            attn_q_weight: loaded_or_static_tensor(
                loaded_weights.and_then(|loaded| loaded.tensor(&format!("{prefix}attn.q.weight"))),
                arena,
                self.attn_q_weight,
            ),
            attn_q_bias: arena.graph_tensor(self.attn_q_bias),
            attn_k_weight: loaded_or_static_tensor(
                loaded_weights.and_then(|loaded| loaded.tensor(&format!("{prefix}attn.k.weight"))),
                arena,
                self.attn_k_weight,
            ),
            attn_k_bias: arena.graph_tensor(self.attn_k_bias),
            attn_v_weight: loaded_or_static_tensor(
                loaded_weights.and_then(|loaded| loaded.tensor(&format!("{prefix}attn.v.weight"))),
                arena,
                self.attn_v_weight,
            ),
            attn_v_bias: arena.graph_tensor(self.attn_v_bias),
            attn_out_weight: loaded_or_static_tensor(
                loaded_weights
                    .and_then(|loaded| loaded.tensor(&format!("{prefix}attn.out.weight"))),
                arena,
                self.attn_out_weight,
            ),
            attn_out_bias: arena.graph_tensor(self.attn_out_bias),
            attn_pos_weight: loaded_or_static_tensor(
                loaded_weights
                    .and_then(|loaded| loaded.tensor(&format!("{prefix}attn.pos.weight"))),
                arena,
                self.attn_pos_weight,
            ),
            attn_pos_bias_u: arena.graph_tensor(self.attn_pos_bias_u),
            attn_pos_bias_v: arena.graph_tensor(self.attn_pos_bias_v),
            conv_norm_weight: arena.graph_tensor(self.conv_norm_weight),
            conv_norm_bias: arena.graph_tensor(self.conv_norm_bias),
            conv_pw1_weight: arena.graph_tensor(self.conv_pw1_weight),
            conv_pw1_bias: arena.graph_tensor(self.conv_pw1_bias),
            conv_dw_weight: arena.graph_tensor(self.conv_dw_weight),
            conv_dw_bias: arena.graph_tensor(self.conv_dw_bias),
            conv_pw2_weight: arena.graph_tensor(self.conv_pw2_weight),
            conv_pw2_bias: arena.graph_tensor(self.conv_pw2_bias),
            ff2_norm_weight: arena.graph_tensor(self.ff2_norm_weight),
            ff2_norm_bias: arena.graph_tensor(self.ff2_norm_bias),
            ff2_up_weight: loaded_or_static_tensor(
                loaded_weights.and_then(|loaded| loaded.tensor(&format!("{prefix}ff2.up.weight"))),
                arena,
                self.ff2_up_weight,
            ),
            ff2_up_bias: arena.graph_tensor(self.ff2_up_bias),
            ff2_down_weight: loaded_or_static_tensor(
                loaded_weights
                    .and_then(|loaded| loaded.tensor(&format!("{prefix}ff2.down.weight"))),
                arena,
                self.ff2_down_weight,
            ),
            ff2_down_bias: arena.graph_tensor(self.ff2_down_bias),
            out_norm_weight: arena.graph_tensor(self.out_norm_weight),
            out_norm_bias: arena.graph_tensor(self.out_norm_bias),
        }
    }
}

/// Walk the cohere conformer encoder layer stack, emitting one
/// `nn::encoder::conformer_block` block per resident layer and chaining `state`
/// through them. This is the Seq2SeqEncoderDecoder shape's ConformerBlock-stage
/// assembly (P4 data-driven): the layer COUNT comes from `layers` (built from
/// `cohere_transcribe.encoder.n_layers`) and the block KIND is `conformer_block`
/// (the descriptor's `ConformerBlock`). The encoder mirror of the qwen S2/S3
/// composers; the emitted op sequence is unchanged. Returns the layer-0 debug
/// tensors when `capture_debug` is set (encoder-debug env), matching the prior
/// inline loop.
#[allow(clippy::too_many_arguments)]
fn compose_conformer_encoder_layer_stack<'a>(
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    mut state: crate::ggml_runtime::GgmlCpuTensor<'a>,
    layers: &[CohereEncoderLayerRuntime],
    arena: &'a GgmlStaticTensorArena,
    loaded_weights: Option<&GgmlLoadedWeightContext>,
    pos_enc: crate::ggml_runtime::GgmlCpuTensor<'a>,
    metadata: CohereTranscribeExecutionMetadata,
    frame_count: usize,
    capture_debug: bool,
) -> Result<
    (
        crate::ggml_runtime::GgmlCpuTensor<'a>,
        Option<EncoderLayerDebugTensors<'a>>,
    ),
    CohereTranscribeEncoderError,
> {
    let mut layer0_debug = None;
    for (layer_idx, layer) in layers.iter().enumerate() {
        let graph_tensors = layer.as_graph_tensors(arena, loaded_weights, layer_idx);
        let result = run_encoder_layer(
            graph,
            state,
            pos_enc,
            metadata,
            frame_count,
            &graph_tensors,
            capture_debug && layer_idx == 0,
        )?;
        state = result.output;
        if layer_idx == 0 {
            layer0_debug = result.debug;
        }
    }
    Ok((state, layer0_debug))
}

fn run_encoder_layer<'a>(
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    state: crate::ggml_runtime::GgmlCpuTensor<'a>,
    pos_enc: crate::ggml_runtime::GgmlCpuTensor<'a>,
    metadata: CohereTranscribeExecutionMetadata,
    frame_count: usize,
    layer: &EncoderLayerGraphTensors<'a>,
    capture_debug: bool,
) -> Result<EncoderLayerRunResult<'a>, CohereTranscribeEncoderError> {
    use crate::nn::encoder::{ConformerBlockConfig, ConformerBlockWeights, conformer_block};

    let element = std::mem::size_of::<f32>();
    let rel_shift_nb1 = (2 * frame_count - 2)
        .checked_mul(element)
        .ok_or(CohereTranscribeEncoderError::ShapeOverflow)?;
    let rel_shift_nb2 = (2 * frame_count - 1)
        .checked_mul(frame_count)
        .and_then(|value| value.checked_mul(element))
        .ok_or(CohereTranscribeEncoderError::ShapeOverflow)?;
    let rel_shift_offset = (frame_count - 1)
        .checked_mul(element)
        .ok_or(CohereTranscribeEncoderError::ShapeOverflow)?;

    let config = ConformerBlockConfig {
        d_model: metadata.encoder_d_model,
        attention_heads: metadata.encoder_heads,
        head_dim: metadata.encoder_head_dim,
        frame_count,
        conv_kernel: metadata.encoder_conv_kernel,
        layer_norm_epsilon: COHERE_ENCODER_LAYER_NORM_EPSILON,
        macaron_scale: 0.5,
        rel_shift_nb1,
        rel_shift_nb2,
        rel_shift_offset,
    };
    let weights = ConformerBlockWeights {
        ff1_norm_weight: layer.ff1_norm_weight,
        ff1_norm_bias: layer.ff1_norm_bias,
        ff1_up_weight: layer.ff1_up_weight,
        ff1_up_bias: layer.ff1_up_bias,
        ff1_down_weight: layer.ff1_down_weight,
        ff1_down_bias: layer.ff1_down_bias,
        attn_norm_weight: layer.attn_norm_weight,
        attn_norm_bias: layer.attn_norm_bias,
        attn_q_weight: layer.attn_q_weight,
        attn_q_bias: layer.attn_q_bias,
        attn_k_weight: layer.attn_k_weight,
        attn_k_bias: layer.attn_k_bias,
        attn_v_weight: layer.attn_v_weight,
        attn_v_bias: layer.attn_v_bias,
        attn_out_weight: layer.attn_out_weight,
        attn_out_bias: layer.attn_out_bias,
        attn_pos_weight: layer.attn_pos_weight,
        attn_pos_bias_u: layer.attn_pos_bias_u,
        attn_pos_bias_v: layer.attn_pos_bias_v,
        conv_norm_weight: layer.conv_norm_weight,
        conv_norm_bias: layer.conv_norm_bias,
        conv_pw1_weight: layer.conv_pw1_weight,
        conv_pw1_bias: layer.conv_pw1_bias,
        conv_dw_weight: layer.conv_dw_weight,
        conv_dw_bias: layer.conv_dw_bias,
        conv_pw2_weight: layer.conv_pw2_weight,
        conv_pw2_bias: layer.conv_pw2_bias,
        ff2_norm_weight: layer.ff2_norm_weight,
        ff2_norm_bias: layer.ff2_norm_bias,
        ff2_up_weight: layer.ff2_up_weight,
        ff2_up_bias: layer.ff2_up_bias,
        ff2_down_weight: layer.ff2_down_weight,
        ff2_down_bias: layer.ff2_down_bias,
        out_norm_weight: layer.out_norm_weight,
        out_norm_bias: layer.out_norm_bias,
    };

    let block = conformer_block(graph, state, pos_enc, config, weights, |step, source| {
        CohereTranscribeEncoderError::GraphBuildFailed { step, source }
    })?;

    Ok(EncoderLayerRunResult {
        output: block.output,
        debug: capture_debug.then_some(EncoderLayerDebugTensors {
            ff1: block.taps.ff1,
            attn: block.taps.attn,
            conv_glu: block.taps.conv_glu,
            conv_dw_act: block.taps.conv_dw_act,
            conv: block.taps.conv,
            ff2: block.taps.ff2,
        }),
    })
}

fn conv_out_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<usize, CohereTranscribeEncoderError> {
    input
        .checked_add(padding.saturating_mul(2))
        .and_then(|value| value.checked_sub(kernel))
        .and_then(|value| value.checked_div(stride))
        .and_then(|value| value.checked_add(1))
        .ok_or(CohereTranscribeEncoderError::ShapeOverflow)
}

fn validate_mel_features(
    metadata: CohereTranscribeExecutionMetadata,
    mel_features: &CohereTranscribeMelFeatures,
) -> Result<(), CohereTranscribeEncoderError> {
    if mel_features.n_mels != metadata.n_mels {
        return Err(CohereTranscribeEncoderError::InvalidFeatures {
            reason: format!(
                "expected n_mels={}, got {}",
                metadata.n_mels, mel_features.n_mels
            ),
        });
    }
    if mel_features.n_frames == 0 {
        return Err(CohereTranscribeEncoderError::InvalidFeatures {
            reason: "expected at least one frame".to_string(),
        });
    }
    if mel_features.data.len() != mel_features.n_frames.saturating_mul(mel_features.n_mels) {
        return Err(CohereTranscribeEncoderError::InvalidFeatures {
            reason: "feature buffer length does not match n_frames * n_mels".to_string(),
        });
    }
    if mel_features.data.iter().any(|value| !value.is_finite()) {
        return Err(CohereTranscribeEncoderError::InvalidFeatures {
            reason: "feature buffer contains non-finite values".to_string(),
        });
    }
    Ok(())
}

/// Conformer Transformer-XL relative-position sinusoidal table. Shared with the
/// parakeet-ctc encoder (S3c) — same conformer rel-pos that `conformer_block`'s
/// rel_shift consumes; exposed `pub(crate)` for reuse (additive, no behavior
/// change to cohere).
pub(crate) fn build_relative_positional_encoding(
    d_model: usize,
    frame_count: usize,
) -> Result<Vec<f32>, CohereTranscribeEncoderError> {
    let n_positions = frame_count
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(CohereTranscribeEncoderError::ShapeOverflow)?;
    let total = n_positions
        .checked_mul(d_model)
        .ok_or(CohereTranscribeEncoderError::ShapeOverflow)?;
    let mut values = vec![0.0_f32; total];
    for position_idx in 0..n_positions {
        let pos = (frame_count - 1) as f32 - position_idx as f32;
        for j in 0..(d_model / 2) {
            let div = 10000.0_f32.powf((2.0 * j as f32) / d_model as f32);
            let base = position_idx * d_model + 2 * j;
            values[base] = (pos / div).sin();
            if base + 1 < values.len() {
                values[base + 1] = (pos / div).cos();
            }
        }
    }
    Ok(values)
}

fn emit_cohere_debug_encoder_stage_preview(
    stage: &str,
    frame_count: usize,
    hidden_size: usize,
    rows: &[f32],
) {
    if rows.is_empty() || hidden_size == 0 || frame_count == 0 {
        return;
    }

    let first_values = rows
        .iter()
        .take(8)
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    let first_frame = &rows[..hidden_size.min(rows.len())];
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;
    for value in first_frame {
        min_value = min_value.min(*value);
        max_value = max_value.max(*value);
        sum += f64::from(*value);
    }
    let mean_value = sum / first_frame.len() as f64;
    eprintln!(
        "openasr cohere encoder {stage}: frames={} hidden={} first8=[{}] frame0_mean={:.6} frame0_min={:.6} frame0_max={:.6}",
        frame_count, hidden_size, first_values, mean_value, min_value, max_value
    );
}

fn new_static_tensor_1d_from_len(
    arena: &GgmlStaticTensorArena,
    len: usize,
    step: &'static str,
) -> Result<GgmlStaticTensor, CohereTranscribeEncoderError> {
    arena
        .new_tensor_1d_f32(len, step)
        .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed { step, source })
}

fn new_static_tensor_2d_or_3d_from_dims(
    arena: &GgmlStaticTensorArena,
    dims: &[usize],
    step: &'static str,
) -> Result<GgmlStaticTensor, CohereTranscribeEncoderError> {
    match dims {
        [ne0, ne1] => arena
            .new_tensor_2d_f32(*ne0, *ne1, step)
            .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed { step, source }),
        [ne0, ne1, ne2] => arena
            .new_tensor_3d_f32(*ne0, *ne1, *ne2, step)
            .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed { step, source }),
        _ => Err(CohereTranscribeEncoderError::InvalidTensorShape {
            tensor_name: step.to_string(),
            shape: format!("{dims:?}"),
            reason: "expected rank-2 or rank-3 tensor".to_string(),
        }),
    }
}

fn new_static_tensor_2d_or_3d_from_weight(
    arena: &GgmlStaticTensorArena,
    weight: &super::weights::CohereTensorWeight,
    step: &'static str,
) -> Result<GgmlStaticTensor, CohereTranscribeEncoderError> {
    if let Some(raw) = &weight.raw_ggml
        && raw.dims.as_slice() == weight.dims.as_slice()
    {
        return match weight.dims.as_slice() {
            [ne0, ne1] => arena
                .new_tensor_2d_typed(*ne0, *ne1, raw.ggml_type, step)
                .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed { step, source }),
            [ne0, ne1, ne2] => arena
                .new_tensor_3d_typed(*ne0, *ne1, *ne2, raw.ggml_type, step)
                .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed { step, source }),
            _ => new_static_tensor_2d_or_3d_from_dims(arena, &weight.dims, step),
        };
    }
    new_static_tensor_2d_or_3d_from_dims(arena, &weight.dims, step)
}

fn new_static_tensor_4d_from_dims(
    arena: &GgmlStaticTensorArena,
    dims: &[usize],
    step: &'static str,
) -> Result<GgmlStaticTensor, CohereTranscribeEncoderError> {
    if dims.len() != 4 {
        return Err(CohereTranscribeEncoderError::InvalidTensorShape {
            tensor_name: step.to_string(),
            shape: format!("{dims:?}"),
            reason: "expected rank-4 tensor".to_string(),
        });
    }
    arena
        .new_tensor_4d_typed(
            dims[0],
            dims[1],
            dims[2],
            dims[3],
            crate::ggml_runtime::GGML_TYPE_F32,
            step,
        )
        .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed { step, source })
}

fn new_static_tensor_4d_f16_from_dims(
    arena: &GgmlStaticTensorArena,
    dims: &[usize],
    step: &'static str,
) -> Result<GgmlStaticTensor, CohereTranscribeEncoderError> {
    if dims.len() != 4 {
        return Err(CohereTranscribeEncoderError::InvalidTensorShape {
            tensor_name: step.to_string(),
            shape: format!("{dims:?}"),
            reason: "expected rank-4 tensor".to_string(),
        });
    }
    arena
        .new_tensor_4d_typed(dims[0], dims[1], dims[2], dims[3], GGML_TYPE_F16, step)
        .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed { step, source })
}

fn new_static_tensor_2d_or_3d_f16_from_dims(
    arena: &GgmlStaticTensorArena,
    dims: &[usize],
    step: &'static str,
) -> Result<GgmlStaticTensor, CohereTranscribeEncoderError> {
    match dims {
        [ne0, ne1] => arena
            .new_tensor_2d_f16(*ne0, *ne1, step)
            .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed { step, source }),
        [ne0, ne1, ne2] => arena
            .new_tensor_3d_f16(*ne0, *ne1, *ne2, step)
            .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed { step, source }),
        _ => Err(CohereTranscribeEncoderError::InvalidTensorShape {
            tensor_name: step.to_string(),
            shape: format!("{dims:?}"),
            reason: "expected rank-2 or rank-3 tensor".to_string(),
        }),
    }
}

fn new_static_projection_tensor(
    arena: &GgmlStaticTensorArena,
    weight: &super::weights::CohereMatrixWeight,
    step: &'static str,
) -> Result<GgmlStaticTensor, CohereTranscribeEncoderError> {
    if let Some(raw) = &weight.raw_ggml
        && raw.dims.as_slice() == [weight.cols, weight.rows]
    {
        return arena
            .new_matmul_weight_2d_typed(weight.cols, weight.rows, raw.ggml_type, step)
            .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed { step, source });
    }
    arena
        .new_tensor_2d_f32(weight.cols, weight.rows, step)
        .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed { step, source })
}

fn new_static_row_major_matrix_tensor(
    arena: &GgmlStaticTensorArena,
    weight: &super::weights::CohereMatrixWeight,
    step: &'static str,
) -> Result<GgmlStaticTensor, CohereTranscribeEncoderError> {
    arena
        .new_tensor_2d_f32(weight.rows, weight.cols, step)
        .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed { step, source })
}

fn upload_static_f32(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    values: &[f32],
    step: &'static str,
) -> Result<(), CohereTranscribeEncoderError> {
    arena
        .set_f32_slice(tensor, values, step)
        .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed { step, source })
}

fn upload_static_f16_from_f32(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    values: &[f32],
    step: &'static str,
) -> Result<(), CohereTranscribeEncoderError> {
    let bits = values
        .iter()
        .copied()
        .map(f32_to_f16_bits)
        .collect::<Vec<_>>();
    arena
        .set_f16_bits_slice(tensor, &bits, step)
        .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed { step, source })
}

fn upload_static_f16_from_weight(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &super::weights::CohereTensorWeight,
    step: &'static str,
) -> Result<(), CohereTranscribeEncoderError> {
    if let Some(raw) = &weight.raw_ggml
        && raw.dims == weight.dims
        && arena.set_bytes_slice(tensor, raw.bytes(), step).is_ok()
    {
        return Ok(());
    }
    upload_static_f16_from_f32(arena, tensor, &weight.values, step)
}

fn upload_static_tensor_weight_with_expected_type(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &super::weights::CohereTensorWeight,
    expected_type: i32,
    step: &'static str,
) -> Result<(), CohereTranscribeEncoderError> {
    if let Some(raw) = &weight.raw_ggml
        && raw.dims.as_slice() == weight.dims.as_slice()
        && raw.ggml_type == expected_type
        && arena.set_bytes_slice(tensor, raw.bytes(), step).is_ok()
    {
        return Ok(());
    }
    upload_static_f32(arena, tensor, &weight.values, step)
}

fn upload_static_tensor_weight(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &super::weights::CohereTensorWeight,
    step: &'static str,
) -> Result<(), CohereTranscribeEncoderError> {
    if let Some(raw) = &weight.raw_ggml
        && raw.dims.as_slice() == weight.dims.as_slice()
        && arena.set_bytes_slice(tensor, raw.bytes(), step).is_ok()
    {
        return Ok(());
    }
    upload_static_f32(arena, tensor, &weight.values, step)
}

fn upload_static_projection_f32(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &super::weights::CohereMatrixWeight,
    step: &'static str,
) -> Result<(), CohereTranscribeEncoderError> {
    if let Some(raw) = &weight.raw_ggml
        && raw.dims.as_slice() == [weight.cols, weight.rows]
        && arena.set_bytes_slice(tensor, raw.bytes(), step).is_ok()
    {
        return Ok(());
    }
    let values = match weight.layout {
        super::weights::CohereMatrixLayout::RowsByColumns => {
            transpose_matrix(&weight.values, weight.rows, weight.cols)?
        }
        super::weights::CohereMatrixLayout::ColumnsByRows => weight.values.clone(),
    };
    upload_static_f32(arena, tensor, &values, step)
}

fn upload_static_row_major_matrix_f32(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &super::weights::CohereMatrixWeight,
    step: &'static str,
) -> Result<(), CohereTranscribeEncoderError> {
    let values = match weight.layout {
        super::weights::CohereMatrixLayout::RowsByColumns => weight.values.clone(),
        super::weights::CohereMatrixLayout::ColumnsByRows => {
            transpose_matrix(&weight.values, weight.cols, weight.rows)?
        }
    };
    upload_static_f32(arena, tensor, &values, step)
}

fn upload_static_encoder_layer(
    arena: &mut GgmlStaticTensorArena,
    loaded_weights: Option<&GgmlLoadedWeightContext>,
    layer: &CohereEncoderLayerWeights,
    tensors: &CohereEncoderLayerRuntime,
) -> Result<(), CohereTranscribeEncoderError> {
    upload_static_f32(
        arena,
        tensors.ff1_norm_weight,
        &layer.ff1_norm_weight.values,
        "enc_ff1_norm_weight",
    )?;
    upload_static_f32(
        arena,
        tensors.ff1_norm_bias,
        &layer.ff1_norm_bias.values,
        "enc_ff1_norm_bias",
    )?;
    if !has_loaded_tensor(loaded_weights, &layer.ff1_up_weight.name) {
        upload_static_projection_f32(
            arena,
            tensors.ff1_up_weight,
            &layer.ff1_up_weight,
            "enc_ff1_up_weight",
        )?;
    }
    upload_static_f32(
        arena,
        tensors.ff1_up_bias,
        &layer.ff1_up_bias.values,
        "enc_ff1_up_bias",
    )?;
    if !has_loaded_tensor(loaded_weights, &layer.ff1_down_weight.name) {
        upload_static_projection_f32(
            arena,
            tensors.ff1_down_weight,
            &layer.ff1_down_weight,
            "enc_ff1_down_weight",
        )?;
    }
    upload_static_f32(
        arena,
        tensors.ff1_down_bias,
        &layer.ff1_down_bias.values,
        "enc_ff1_down_bias",
    )?;
    upload_static_f32(
        arena,
        tensors.attn_norm_weight,
        &layer.attn_norm_weight.values,
        "enc_attn_norm_weight",
    )?;
    upload_static_f32(
        arena,
        tensors.attn_norm_bias,
        &layer.attn_norm_bias.values,
        "enc_attn_norm_bias",
    )?;
    if !has_loaded_tensor(loaded_weights, &layer.attn_q_weight.name) {
        upload_static_projection_f32(
            arena,
            tensors.attn_q_weight,
            &layer.attn_q_weight,
            "enc_attn_q_weight",
        )?;
    }
    upload_static_f32(
        arena,
        tensors.attn_q_bias,
        &layer.attn_q_bias.values,
        "enc_attn_q_bias",
    )?;
    if !has_loaded_tensor(loaded_weights, &layer.attn_k_weight.name) {
        upload_static_projection_f32(
            arena,
            tensors.attn_k_weight,
            &layer.attn_k_weight,
            "enc_attn_k_weight",
        )?;
    }
    upload_static_f32(
        arena,
        tensors.attn_k_bias,
        &layer.attn_k_bias.values,
        "enc_attn_k_bias",
    )?;
    if !has_loaded_tensor(loaded_weights, &layer.attn_v_weight.name) {
        upload_static_projection_f32(
            arena,
            tensors.attn_v_weight,
            &layer.attn_v_weight,
            "enc_attn_v_weight",
        )?;
    }
    upload_static_f32(
        arena,
        tensors.attn_v_bias,
        &layer.attn_v_bias.values,
        "enc_attn_v_bias",
    )?;
    if !has_loaded_tensor(loaded_weights, &layer.attn_out_weight.name) {
        upload_static_projection_f32(
            arena,
            tensors.attn_out_weight,
            &layer.attn_out_weight,
            "enc_attn_out_weight",
        )?;
    }
    upload_static_f32(
        arena,
        tensors.attn_out_bias,
        &layer.attn_out_bias.values,
        "enc_attn_out_bias",
    )?;
    if !has_loaded_tensor(loaded_weights, &layer.attn_pos_weight.name) {
        upload_static_projection_f32(
            arena,
            tensors.attn_pos_weight,
            &layer.attn_pos_weight,
            "enc_attn_pos_weight",
        )?;
    }
    upload_static_row_major_matrix_f32(
        arena,
        tensors.attn_pos_bias_u,
        &layer.attn_pos_bias_u,
        "enc_attn_pos_bias_u",
    )?;
    upload_static_row_major_matrix_f32(
        arena,
        tensors.attn_pos_bias_v,
        &layer.attn_pos_bias_v,
        "enc_attn_pos_bias_v",
    )?;
    upload_static_f32(
        arena,
        tensors.conv_norm_weight,
        &layer.conv_norm_weight.values,
        "enc_conv_norm_weight",
    )?;
    upload_static_f32(
        arena,
        tensors.conv_norm_bias,
        &layer.conv_norm_bias.values,
        "enc_conv_norm_bias",
    )?;
    upload_static_tensor_weight(
        arena,
        tensors.conv_pw1_weight,
        &layer.conv_pw1_weight,
        "enc_conv_pw1_weight",
    )?;
    upload_static_f32(
        arena,
        tensors.conv_pw1_bias,
        &layer.conv_pw1_bias.values,
        "enc_conv_pw1_bias",
    )?;
    upload_static_f16_from_weight(
        arena,
        tensors.conv_dw_weight,
        &layer.conv_dw_weight,
        "enc_conv_dw_weight",
    )?;
    upload_static_f32(
        arena,
        tensors.conv_dw_bias,
        &layer.conv_dw_bias.values,
        "enc_conv_dw_bias",
    )?;
    upload_static_tensor_weight(
        arena,
        tensors.conv_pw2_weight,
        &layer.conv_pw2_weight,
        "enc_conv_pw2_weight",
    )?;
    upload_static_f32(
        arena,
        tensors.conv_pw2_bias,
        &layer.conv_pw2_bias.values,
        "enc_conv_pw2_bias",
    )?;
    upload_static_f32(
        arena,
        tensors.ff2_norm_weight,
        &layer.ff2_norm_weight.values,
        "enc_ff2_norm_weight",
    )?;
    upload_static_f32(
        arena,
        tensors.ff2_norm_bias,
        &layer.ff2_norm_bias.values,
        "enc_ff2_norm_bias",
    )?;
    if !has_loaded_tensor(loaded_weights, &layer.ff2_up_weight.name) {
        upload_static_projection_f32(
            arena,
            tensors.ff2_up_weight,
            &layer.ff2_up_weight,
            "enc_ff2_up_weight",
        )?;
    }
    upload_static_f32(
        arena,
        tensors.ff2_up_bias,
        &layer.ff2_up_bias.values,
        "enc_ff2_up_bias",
    )?;
    if !has_loaded_tensor(loaded_weights, &layer.ff2_down_weight.name) {
        upload_static_projection_f32(
            arena,
            tensors.ff2_down_weight,
            &layer.ff2_down_weight,
            "enc_ff2_down_weight",
        )?;
    }
    upload_static_f32(
        arena,
        tensors.ff2_down_bias,
        &layer.ff2_down_bias.values,
        "enc_ff2_down_bias",
    )?;
    upload_static_f32(
        arena,
        tensors.out_norm_weight,
        &layer.out_norm_weight.values,
        "enc_out_norm_weight",
    )?;
    upload_static_f32(
        arena,
        tensors.out_norm_bias,
        &layer.out_norm_bias.values,
        "enc_out_norm_bias",
    )
}

fn upload_f32<'a>(
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    tensor: crate::ggml_runtime::GgmlCpuTensor<'a>,
    values: &[f32],
    step: &'static str,
) -> Result<(), CohereTranscribeEncoderError> {
    graph
        .set_f32_slice(tensor, values, step)
        .map_err(|source| CohereTranscribeEncoderError::GraphBuildFailed { step, source })
}

fn transpose_matrix(
    values: &[f32],
    src_rows: usize,
    src_cols: usize,
) -> Result<Vec<f32>, CohereTranscribeEncoderError> {
    let expected = src_rows
        .checked_mul(src_cols)
        .ok_or(CohereTranscribeEncoderError::ShapeOverflow)?;
    if values.len() != expected {
        return Err(CohereTranscribeEncoderError::InvalidFeatures {
            reason: format!(
                "matrix transpose expected {} values, got {}",
                expected,
                values.len()
            ),
        });
    }
    let mut out = vec![0.0_f32; expected];
    for row in 0..src_rows {
        for col in 0..src_cols {
            out[col * src_rows + row] = values[row * src_cols + col];
        }
    }
    Ok(out)
}

pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
    use crate::validate_ggml_runtime_source_path;
    use crate::{
        GgmlAsrRuntimeSourcePreflight, read_gguf_metadata_from_runtime_source,
        read_gguf_tensor_index_from_runtime_source,
    };
    use tempfile::{NamedTempFile, TempPath};

    fn write_runtime_ready_preflight() -> (TempPath, GgmlAsrRuntimeSourcePreflight) {
        let file = NamedTempFile::new().expect("temp file");
        let persisted = file.into_temp_path();
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_non_streaming_cpu("cohere-runtime-fixture")
            .with_cohere_graph_metadata(2, 2, 16, 2, 8, 32, 5, 32, 32)
            .with_cohere_runtime_tensors_with_layers(2, 2);
        write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");

        let runtime_source =
            validate_ggml_runtime_source_path(&persisted).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        (
            persisted,
            GgmlAsrRuntimeSourcePreflight {
                runtime_source,
                metadata,
                tensor_index: Arc::new(tensor_index),
            },
        )
    }

    fn sample_features(metadata: CohereTranscribeExecutionMetadata) -> CohereTranscribeMelFeatures {
        let n_frames = 12;
        let mut data = Vec::with_capacity(n_frames * metadata.n_mels);
        for frame_idx in 0..n_frames {
            for mel_idx in 0..metadata.n_mels {
                let value = ((frame_idx * metadata.n_mels + mel_idx) as f32 * 0.03125).sin();
                data.push(value);
            }
        }
        CohereTranscribeMelFeatures {
            n_frames,
            n_mels: metadata.n_mels,
            data,
        }
    }

    #[test]
    fn encoder_emits_finite_projected_frames() {
        let (_runtime_path, preflight) = write_runtime_ready_preflight();
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");
        let features = sample_features(metadata);
        let weights =
            super::super::encoder_weights::load_cohere_transcribe_encoder_weights_from_reader(
                &reader, metadata,
            )
            .expect("weights");

        let output =
            encode_cohere_transcribe_audio_embeddings_from_weights(&weights, metadata, &features)
                .expect("encoder");

        assert!(output.frame_count > 0);
        assert_eq!(output.hidden_size, metadata.decoder_d_model);
        assert_eq!(output.rows.len(), output.frame_count * output.hidden_size);
        assert!(output.rows.iter().all(|value| value.is_finite()));
    }
}
