use thiserror::Error;

#[cfg(test)]
use crate::ggml_runtime::{
    GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor,
};
#[cfg(test)]
use crate::nn::attn::{
    AttentionHeadLayout, AttentionReshapeSteps, AttentionValueMergeSteps,
    STANDARD_HEAD_PERMUTE_AXES, attention_context_from_probs,
    reshape_projection_to_attention_heads,
};
#[cfg(test)]
use crate::nn::norm::{
    AffineLayerNormSteps, apply_affine_layer_norm as nn_apply_affine_layer_norm,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WhisperEncoderGraphInputShape {
    pub frames: usize,
    pub hidden_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WhisperEncoderGraphMetadata {
    pub encoder_layers: usize,
    pub encoder_hidden_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperEncoderGraphTensorRef {
    pub tensor_name: String,
    pub tensor_num_elements: usize,
    pub dims: Vec<u64>,
    pub runtime_linear_weight_layout: Option<WhisperEncoderLinearWeightLayout>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperEncoderLayerTensorBinding {
    pub self_attn_norm_weight: Option<WhisperEncoderGraphTensorRef>,
    pub self_attn_norm_bias: Option<WhisperEncoderGraphTensorRef>,
    pub self_attn_q_weight: Option<WhisperEncoderGraphTensorRef>,
    pub self_attn_k_weight: Option<WhisperEncoderGraphTensorRef>,
    pub self_attn_v_weight: Option<WhisperEncoderGraphTensorRef>,
    pub self_attn_out_weight: Option<WhisperEncoderGraphTensorRef>,
    pub mlp_norm_weight: Option<WhisperEncoderGraphTensorRef>,
    pub mlp_norm_bias: Option<WhisperEncoderGraphTensorRef>,
    pub mlp_fc1_weight: Option<WhisperEncoderGraphTensorRef>,
    pub mlp_fc2_weight: Option<WhisperEncoderGraphTensorRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperEncoderTensorBindingSeam {
    pub layers: Vec<WhisperEncoderLayerTensorBinding>,
    pub final_norm_weight: Option<WhisperEncoderGraphTensorRef>,
    pub final_norm_bias: Option<WhisperEncoderGraphTensorRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperEncoderTensorMaterializationSeam {
    pub source_label: &'static str,
    pub materialized_tensor_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperEncoderGraphPlan {
    pub input_shape: WhisperEncoderGraphInputShape,
    pub output_frames: usize,
    pub output_hidden_size: usize,
    pub layers: Vec<WhisperEncoderLayerPlan>,
    pub final_norm: WhisperEncoderNormPlan,
    pub required_primitives: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperEncoderLayerPlan {
    pub layer_idx: usize,
    pub self_attn_norm: WhisperEncoderNormPlan,
    pub self_attn_q: WhisperEncoderLinearProjectionPlan,
    pub self_attn_k: WhisperEncoderLinearProjectionPlan,
    pub self_attn_v: WhisperEncoderLinearProjectionPlan,
    pub self_attn_out: WhisperEncoderLinearProjectionPlan,
    pub mlp_norm: WhisperEncoderNormPlan,
    pub mlp_fc1: WhisperEncoderLinearProjectionPlan,
    pub mlp_fc2: WhisperEncoderLinearProjectionPlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperEncoderNormPlan {
    pub weight: WhisperEncoderGraphTensorRef,
    pub bias: WhisperEncoderGraphTensorRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperEncoderLinearProjectionPlan {
    pub weight: WhisperEncoderGraphTensorRef,
    pub weight_layout: WhisperEncoderLinearWeightLayout,
    pub input_dim: usize,
    pub output_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum WhisperEncoderLinearWeightLayout {
    InputOutput,
    OutputInput,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum WhisperEncoderGraphPlanError {
    #[error("whisper encoder graph input shape is invalid: {reason}")]
    InvalidInputShape { reason: String },
    #[error(
        "whisper encoder graph binding layer count mismatch: metadata={metadata_layers}, binding={binding_layers}"
    )]
    LayerCountMismatch {
        metadata_layers: usize,
        binding_layers: usize,
    },
    #[error("whisper encoder graph binding is missing layer {layer_idx}")]
    MissingLayerBinding { layer_idx: usize },
    #[error("whisper encoder graph is missing required tensor '{slot}' at {scope}")]
    MissingTensorBinding { scope: String, slot: &'static str },
    #[error(
        "whisper encoder graph tensor '{tensor_name}' for '{slot}' at {scope} has invalid shape {found_shape:?}: {reason}"
    )]
    TensorShapeMismatch {
        scope: String,
        slot: &'static str,
        tensor_name: String,
        found_shape: Vec<u64>,
        reason: String,
    },
    #[error("whisper encoder graph unsupported primitive '{primitive}': {reason}")]
    UnsupportedEncoderPrimitive {
        primitive: &'static str,
        reason: String,
    },
}

pub(crate) struct WhisperEncoderGraphBuilder<'a> {
    metadata: WhisperEncoderGraphMetadata,
    binding: &'a WhisperEncoderTensorBindingSeam,
    materialization: &'a WhisperEncoderTensorMaterializationSeam,
    input_shape: WhisperEncoderGraphInputShape,
}

impl<'a> WhisperEncoderGraphBuilder<'a> {
    pub(crate) fn new(
        metadata: WhisperEncoderGraphMetadata,
        binding: &'a WhisperEncoderTensorBindingSeam,
        materialization: &'a WhisperEncoderTensorMaterializationSeam,
        input_shape: WhisperEncoderGraphInputShape,
    ) -> Self {
        Self {
            metadata,
            binding,
            materialization,
            input_shape,
        }
    }

    pub(crate) fn build(&self) -> Result<WhisperEncoderGraphPlan, WhisperEncoderGraphPlanError> {
        if self.input_shape.frames == 0 {
            return Err(WhisperEncoderGraphPlanError::InvalidInputShape {
                reason: "frames must be > 0".to_string(),
            });
        }
        if self.input_shape.hidden_size == 0 {
            return Err(WhisperEncoderGraphPlanError::InvalidInputShape {
                reason: "hidden_size must be > 0".to_string(),
            });
        }
        if self.input_shape.hidden_size != self.metadata.encoder_hidden_size {
            return Err(WhisperEncoderGraphPlanError::InvalidInputShape {
                reason: format!(
                    "input hidden_size={} does not match whisper.encoder.embedding_length={}",
                    self.input_shape.hidden_size, self.metadata.encoder_hidden_size
                ),
            });
        }
        if self.materialization.materialized_tensor_count == 0 {
            return Err(WhisperEncoderGraphPlanError::UnsupportedEncoderPrimitive {
                primitive: "encoder.tensor_materialization",
                reason: format!(
                    "materialization seam '{}' resolved no tensors",
                    self.materialization.source_label
                ),
            });
        }
        if self.metadata.encoder_layers == 0 {
            return Err(WhisperEncoderGraphPlanError::InvalidInputShape {
                reason: "encoder_layers must be > 0".to_string(),
            });
        }
        if self.binding.layers.len() != self.metadata.encoder_layers {
            return Err(WhisperEncoderGraphPlanError::LayerCountMismatch {
                metadata_layers: self.metadata.encoder_layers,
                binding_layers: self.binding.layers.len(),
            });
        }

        let mut layers = Vec::with_capacity(self.metadata.encoder_layers);
        for layer_idx in 0..self.metadata.encoder_layers {
            let layer_binding = self
                .binding
                .layers
                .get(layer_idx)
                .ok_or(WhisperEncoderGraphPlanError::MissingLayerBinding { layer_idx })?;
            layers.push(self.parse_layer(layer_idx, layer_binding)?);
        }

        let final_norm = self.parse_norm(
            "encoder",
            "layer_norm",
            self.binding.final_norm_weight.as_ref(),
            self.binding.final_norm_bias.as_ref(),
            self.input_shape.hidden_size,
        )?;

        Ok(WhisperEncoderGraphPlan {
            input_shape: self.input_shape,
            output_frames: self.input_shape.frames,
            output_hidden_size: self.input_shape.hidden_size,
            layers,
            final_norm,
            required_primitives: required_encoder_primitives(),
        })
    }

    fn parse_layer(
        &self,
        layer_idx: usize,
        layer_binding: &WhisperEncoderLayerTensorBinding,
    ) -> Result<WhisperEncoderLayerPlan, WhisperEncoderGraphPlanError> {
        let hidden = self.input_shape.hidden_size;
        let scope = format!("encoder.layer[{layer_idx}]");
        let self_attn_norm = self.parse_norm(
            &scope,
            "self_attn_layer_norm",
            layer_binding.self_attn_norm_weight.as_ref(),
            layer_binding.self_attn_norm_bias.as_ref(),
            hidden,
        )?;
        let q = self.parse_linear(
            &scope,
            "self_attn.q_proj.weight",
            layer_binding.self_attn_q_weight.as_ref(),
            hidden,
            None,
        )?;
        let k = self.parse_linear(
            &scope,
            "self_attn.k_proj.weight",
            layer_binding.self_attn_k_weight.as_ref(),
            hidden,
            None,
        )?;
        let v = self.parse_linear(
            &scope,
            "self_attn.v_proj.weight",
            layer_binding.self_attn_v_weight.as_ref(),
            hidden,
            None,
        )?;
        let out = self.parse_linear(
            &scope,
            "self_attn.out_proj.weight",
            layer_binding.self_attn_out_weight.as_ref(),
            hidden,
            Some(hidden),
        )?;

        let mlp_norm = self.parse_norm(
            &scope,
            "final_layer_norm",
            layer_binding.mlp_norm_weight.as_ref(),
            layer_binding.mlp_norm_bias.as_ref(),
            hidden,
        )?;
        let fc1 = self.parse_linear(
            &scope,
            "fc1.weight",
            layer_binding.mlp_fc1_weight.as_ref(),
            hidden,
            None,
        )?;
        let fc2 = self.parse_linear(
            &scope,
            "fc2.weight",
            layer_binding.mlp_fc2_weight.as_ref(),
            fc1.output_dim,
            Some(hidden),
        )?;

        Ok(WhisperEncoderLayerPlan {
            layer_idx,
            self_attn_norm,
            self_attn_q: q,
            self_attn_k: k,
            self_attn_v: v,
            self_attn_out: out,
            mlp_norm,
            mlp_fc1: fc1,
            mlp_fc2: fc2,
        })
    }

    fn parse_norm(
        &self,
        scope: &str,
        slot_prefix: &'static str,
        weight: Option<&WhisperEncoderGraphTensorRef>,
        bias: Option<&WhisperEncoderGraphTensorRef>,
        hidden: usize,
    ) -> Result<WhisperEncoderNormPlan, WhisperEncoderGraphPlanError> {
        let weight = weight.ok_or_else(|| WhisperEncoderGraphPlanError::MissingTensorBinding {
            scope: scope.to_string(),
            slot: slot_prefix,
        })?;
        let bias = bias.ok_or_else(|| WhisperEncoderGraphPlanError::MissingTensorBinding {
            scope: scope.to_string(),
            slot: slot_prefix,
        })?;

        validate_norm_shape(scope, slot_prefix, weight, hidden)?;
        validate_norm_shape(scope, slot_prefix, bias, hidden)?;
        Ok(WhisperEncoderNormPlan {
            weight: weight.clone(),
            bias: bias.clone(),
        })
    }

    fn parse_linear(
        &self,
        scope: &str,
        slot: &'static str,
        tensor: Option<&WhisperEncoderGraphTensorRef>,
        expected_input_dim: usize,
        expected_output_dim: Option<usize>,
    ) -> Result<WhisperEncoderLinearProjectionPlan, WhisperEncoderGraphPlanError> {
        let tensor = tensor.ok_or_else(|| WhisperEncoderGraphPlanError::MissingTensorBinding {
            scope: scope.to_string(),
            slot,
        })?;
        if tensor.dims.len() != 2 {
            return Err(WhisperEncoderGraphPlanError::TensorShapeMismatch {
                scope: scope.to_string(),
                slot,
                tensor_name: tensor.tensor_name.clone(),
                found_shape: tensor.dims.clone(),
                reason: "expected rank-2 linear projection tensor".to_string(),
            });
        }

        let lhs = usize::try_from(tensor.dims[0]).map_err(|_| {
            WhisperEncoderGraphPlanError::TensorShapeMismatch {
                scope: scope.to_string(),
                slot,
                tensor_name: tensor.tensor_name.clone(),
                found_shape: tensor.dims.clone(),
                reason: "dimension does not fit target usize".to_string(),
            }
        })?;
        let rhs = usize::try_from(tensor.dims[1]).map_err(|_| {
            WhisperEncoderGraphPlanError::TensorShapeMismatch {
                scope: scope.to_string(),
                slot,
                tensor_name: tensor.tensor_name.clone(),
                found_shape: tensor.dims.clone(),
                reason: "dimension does not fit target usize".to_string(),
            }
        })?;

        let (input_dim, output_dim, weight_layout) = if let Some(runtime_layout) =
            tensor.runtime_linear_weight_layout
        {
            match runtime_layout {
                WhisperEncoderLinearWeightLayout::InputOutput => {
                    if lhs != expected_input_dim {
                        return Err(WhisperEncoderGraphPlanError::TensorShapeMismatch {
                            scope: scope.to_string(),
                            slot,
                            tensor_name: tensor.tensor_name.clone(),
                            found_shape: tensor.dims.clone(),
                            reason: format!(
                                "prepared input-output projection input_dim={lhs} does not match expected {expected_input_dim}"
                            ),
                        });
                    }
                    (lhs, rhs, runtime_layout)
                }
                WhisperEncoderLinearWeightLayout::OutputInput => {
                    if rhs != expected_input_dim {
                        return Err(WhisperEncoderGraphPlanError::TensorShapeMismatch {
                            scope: scope.to_string(),
                            slot,
                            tensor_name: tensor.tensor_name.clone(),
                            found_shape: tensor.dims.clone(),
                            reason: format!(
                                "prepared output-input projection input_dim={rhs} does not match expected {expected_input_dim}"
                            ),
                        });
                    }
                    (rhs, lhs, runtime_layout)
                }
            }
        } else if lhs == expected_input_dim && rhs == expected_input_dim {
            return Err(WhisperEncoderGraphPlanError::TensorShapeMismatch {
                scope: scope.to_string(),
                slot,
                tensor_name: tensor.tensor_name.clone(),
                found_shape: tensor.dims.clone(),
                reason: format!(
                    "ambiguous square projection requires runtime_linear_weight_layout metadata for input_dim={expected_input_dim}"
                ),
            });
        } else if lhs == expected_input_dim {
            (lhs, rhs, WhisperEncoderLinearWeightLayout::InputOutput)
        } else if rhs == expected_input_dim {
            (rhs, lhs, WhisperEncoderLinearWeightLayout::OutputInput)
        } else {
            return Err(WhisperEncoderGraphPlanError::TensorShapeMismatch {
                scope: scope.to_string(),
                slot,
                tensor_name: tensor.tensor_name.clone(),
                found_shape: tensor.dims.clone(),
                reason: format!("expected one dimension to match input_dim={expected_input_dim}"),
            });
        };

        if let Some(expected_output_dim) = expected_output_dim
            && output_dim != expected_output_dim
        {
            return Err(WhisperEncoderGraphPlanError::TensorShapeMismatch {
                scope: scope.to_string(),
                slot,
                tensor_name: tensor.tensor_name.clone(),
                found_shape: tensor.dims.clone(),
                reason: format!(
                    "projection output_dim={output_dim} does not match expected {expected_output_dim}"
                ),
            });
        }

        Ok(WhisperEncoderLinearProjectionPlan {
            weight: tensor.clone(),
            weight_layout,
            input_dim,
            output_dim,
        })
    }
}

pub(crate) fn build_whisper_encoder_graph_plan(
    metadata: WhisperEncoderGraphMetadata,
    binding: &WhisperEncoderTensorBindingSeam,
    materialization: &WhisperEncoderTensorMaterializationSeam,
    input_shape: WhisperEncoderGraphInputShape,
) -> Result<WhisperEncoderGraphPlan, WhisperEncoderGraphPlanError> {
    WhisperEncoderGraphBuilder::new(metadata, binding, materialization, input_shape).build()
}

fn validate_norm_shape(
    scope: &str,
    slot: &'static str,
    tensor: &WhisperEncoderGraphTensorRef,
    hidden: usize,
) -> Result<(), WhisperEncoderGraphPlanError> {
    let hidden_u64 = hidden as u64;
    let ok = match tensor.dims.as_slice() {
        [dim] => *dim == hidden_u64,
        [_, last] => *last == hidden_u64,
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(WhisperEncoderGraphPlanError::TensorShapeMismatch {
            scope: scope.to_string(),
            slot,
            tensor_name: tensor.tensor_name.clone(),
            found_shape: tensor.dims.clone(),
            reason: format!("expected rank-1 [hidden] or rank-2 [*, hidden={hidden}]"),
        })
    }
}

fn required_encoder_primitives() -> Vec<&'static str> {
    vec![
        "encoder.layer_norm",
        "encoder.self_attn.qkv_projection",
        "encoder.self_attn.qkv_reshape_permute",
        "encoder.self_attn.qk_attention",
        "encoder.self_attn.softmax(scale=1/sqrt(head_dim))",
        "encoder.self_attn.av_projection",
        "encoder.self_attn.out_projection",
        "encoder.residual_add",
        "encoder.mlp.fc1",
        "encoder.mlp.gelu",
        "encoder.mlp.fc2",
        "encoder.final_layer_norm",
    ]
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WhisperEncoderHiddenStateLayout {
    SequenceHidden,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WhisperEncoderGraphExecutionInput {
    pub hidden_state: Vec<f32>,
    pub layout: WhisperEncoderHiddenStateLayout,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WhisperEncoderGraphExecutionOutput {
    pub hidden_state: Vec<f32>,
    pub layout: WhisperEncoderHiddenStateLayout,
    pub frames: usize,
    pub hidden_size: usize,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WhisperEncoderGraphExecutionConfig {
    pub attention_heads: usize,
    pub use_flash_attention: bool,
    pub layer_norm_epsilon: f32,
}

#[cfg(test)]
impl Default for WhisperEncoderGraphExecutionConfig {
    fn default() -> Self {
        Self {
            attention_heads: 1,
            use_flash_attention: true,
            layer_norm_epsilon: 1.0e-5,
        }
    }
}

#[cfg(test)]
#[derive(Debug, Error)]
pub(crate) enum WhisperEncoderGraphExecutionError {
    #[error("whisper encoder execution input is invalid: {reason}")]
    InvalidInput { reason: String },
    #[error("whisper encoder execution is missing tensor '{tensor_name}': {reason}")]
    MissingMaterializedTensor { tensor_name: String, reason: String },
    #[error("whisper encoder graph tensor '{tensor_name}' materialization failed: {reason}")]
    TensorMaterializationFailed { tensor_name: String, reason: String },
    #[error("whisper encoder graph unsupported primitive '{primitive}': {reason}")]
    UnsupportedEncoderPrimitive {
        primitive: &'static str,
        reason: String,
    },
    #[error("whisper encoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
}

#[cfg(test)]
pub(crate) trait WhisperEncoderTensorSource {
    fn materialize_tensor_f32(
        &self,
        tensor: &WhisperEncoderGraphTensorRef,
    ) -> Result<Vec<f32>, WhisperEncoderGraphExecutionError>;
}

#[cfg(test)]
pub(crate) fn execute_whisper_encoder_graph_ggml_v0(
    plan: &WhisperEncoderGraphPlan,
    input: &WhisperEncoderGraphExecutionInput,
    source: &dyn WhisperEncoderTensorSource,
    config: WhisperEncoderGraphExecutionConfig,
) -> Result<WhisperEncoderGraphExecutionOutput, WhisperEncoderGraphExecutionError> {
    if config.attention_heads == 0 {
        return Err(WhisperEncoderGraphExecutionError::InvalidInput {
            reason: "attention_heads must be > 0".to_string(),
        });
    }
    if !plan
        .output_hidden_size
        .is_multiple_of(config.attention_heads)
    {
        return Err(WhisperEncoderGraphExecutionError::InvalidInput {
            reason: format!(
                "hidden_size {} is not divisible by attention_heads {}",
                plan.output_hidden_size, config.attention_heads
            ),
        });
    }
    if !(config.layer_norm_epsilon.is_finite() && config.layer_norm_epsilon > 0.0) {
        return Err(WhisperEncoderGraphExecutionError::InvalidInput {
            reason: "layer_norm_epsilon must be finite and > 0".to_string(),
        });
    }

    let expected_len = plan
        .output_frames
        .checked_mul(plan.output_hidden_size)
        .ok_or_else(|| WhisperEncoderGraphExecutionError::InvalidInput {
            reason: "input shape overflows usize".to_string(),
        })?;
    if input.hidden_state.len() != expected_len {
        return Err(WhisperEncoderGraphExecutionError::InvalidInput {
            reason: format!(
                "hidden input has {} elements but expected {} for [{}, {}]",
                input.hidden_state.len(),
                expected_len,
                plan.output_frames,
                plan.output_hidden_size
            ),
        });
    }

    let hidden_by_seq = normalize_hidden_layout(input, plan.output_frames, plan.output_hidden_size);
    let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default()).map_err(|error| {
        WhisperEncoderGraphExecutionError::GraphExecutionFailed {
            reason: format!("could not initialize ggml cpu graph runner: {error}"),
        }
    })?;
    let mut graph = runner.start_graph();
    let mut uploads = Vec::new();

    let state_input = graph
        .new_tensor_2d_f32(
            plan.output_hidden_size,
            plan.output_frames,
            "encoder_hidden_input",
        )
        .map_err(|error| map_encoder_execute_graph_error("ggml_new_tensor_2d(hidden)", error))?;
    graph
        .set_input(state_input)
        .map_err(|error| map_encoder_execute_graph_error("ggml_set_input(hidden)", error))?;
    uploads.push((state_input, hidden_by_seq, "encoder_hidden_input"));

    let mut state = state_input;
    for (layer_idx, layer) in plan.layers.iter().enumerate() {
        let attn_norm = apply_affine_layer_norm(
            &mut graph,
            &mut uploads,
            source,
            state,
            config.layer_norm_epsilon,
            &layer.self_attn_norm,
            &format!("encoder_layer_{layer_idx}_self_attn_norm"),
        )?;

        let q = apply_linear(
            &mut graph,
            &mut uploads,
            source,
            attn_norm,
            &layer.self_attn_q,
            &format!("encoder_layer_{layer_idx}_self_attn_q"),
        )?;
        let k = apply_linear(
            &mut graph,
            &mut uploads,
            source,
            attn_norm,
            &layer.self_attn_k,
            &format!("encoder_layer_{layer_idx}_self_attn_k"),
        )?;
        let v = apply_linear(
            &mut graph,
            &mut uploads,
            source,
            attn_norm,
            &layer.self_attn_v,
            &format!("encoder_layer_{layer_idx}_self_attn_v"),
        )?;

        let head_dim = plan.output_hidden_size / config.attention_heads;
        let q = reshape_projected_hidden_sequence_to_heads(
            &mut graph,
            q,
            head_dim,
            plan.output_frames,
            config.attention_heads,
            "attn_q_heads",
        )?;
        let k = reshape_projected_hidden_sequence_to_heads(
            &mut graph,
            k,
            head_dim,
            plan.output_frames,
            config.attention_heads,
            "attn_k_heads",
        )?;
        let v = reshape_projected_hidden_sequence_to_heads(
            &mut graph,
            v,
            head_dim,
            plan.output_frames,
            config.attention_heads,
            "attn_v_heads",
        )?;

        let attention_scale = 1.0f32 / (head_dim as f32).sqrt();
        let attn_context = if config.use_flash_attention {
            match graph.flash_attn_ext(k, q, v, None, attention_scale, 0.0, 0.0) {
                Ok(flash) => flash,
                Err(error) => {
                    return Err(map_encoder_execute_graph_error(
                        "ggml_flash_attn_ext(self_attn)",
                        error,
                    ));
                }
            }
        } else {
            let attn_scores = graph
                .mul_mat(k, q)
                .map_err(|error| map_encoder_execute_graph_error("ggml_mul_mat(attn_qk)", error))?;
            let attn_scores = graph
                .scale(attn_scores, attention_scale)
                .map_err(|error| map_encoder_execute_graph_error("ggml_scale(attn_qk)", error))?;
            let attn_scores = graph
                .cont(attn_scores)
                .map_err(|error| map_encoder_execute_graph_error("ggml_cont(attn_qk)", error))?;
            let attn_probs = graph.soft_max(attn_scores).map_err(|error| {
                map_encoder_execute_graph_error("ggml_soft_max(attn_qk_probs)", error)
            })?;
            attention_context_from_probs(
                &graph,
                v,
                attn_probs,
                AttentionHeadLayout {
                    head_dim,
                    attention_heads: config.attention_heads,
                    sequence_len: plan.output_frames,
                },
                AttentionValueMergeSteps {
                    value_permute: "ggml_permute(attn_v_t)",
                    value_cont: "ggml_cont(attn_v_t)",
                    context_mul: "ggml_mul_mat(attn_av)",
                    context_merge_permute: "ggml_permute(attn_merge)",
                    context_merge_cont: "ggml_cont(attn_merge)",
                    context_merge_reshape: "ggml_reshape_2d(attn_merge)",
                },
                map_encoder_execute_graph_error,
            )?
        };

        let attn_out = apply_linear(
            &mut graph,
            &mut uploads,
            source,
            attn_context,
            &layer.self_attn_out,
            &format!("encoder_layer_{layer_idx}_self_attn_out"),
        )?;
        state = graph
            .add(attn_out, state)
            .map_err(|error| map_encoder_execute_graph_error("ggml_add(attn_residual)", error))?;

        let mlp_norm = apply_affine_layer_norm(
            &mut graph,
            &mut uploads,
            source,
            state,
            config.layer_norm_epsilon,
            &layer.mlp_norm,
            &format!("encoder_layer_{layer_idx}_mlp_norm"),
        )?;
        let mlp_fc1 = apply_linear(
            &mut graph,
            &mut uploads,
            source,
            mlp_norm,
            &layer.mlp_fc1,
            &format!("encoder_layer_{layer_idx}_mlp_fc1"),
        )?;
        let mlp_fc1 = graph
            .gelu(mlp_fc1)
            .map_err(|error| map_encoder_execute_graph_error("ggml_gelu(mlp_fc1)", error))?;
        let mlp_fc2 = apply_linear(
            &mut graph,
            &mut uploads,
            source,
            mlp_fc1,
            &layer.mlp_fc2,
            &format!("encoder_layer_{layer_idx}_mlp_fc2"),
        )?;
        state = graph
            .add(mlp_fc2, state)
            .map_err(|error| map_encoder_execute_graph_error("ggml_add(mlp_residual)", error))?;
    }

    state = apply_affine_layer_norm(
        &mut graph,
        &mut uploads,
        source,
        state,
        config.layer_norm_epsilon,
        &plan.final_norm,
        "encoder_final_norm",
    )?;

    graph
        .set_output(state)
        .map_err(|error| map_encoder_execute_graph_error("ggml_set_output(state)", error))?;
    for (tensor, values, label) in uploads {
        graph
            .set_f32_slice(tensor, &values, label)
            .map_err(
                |error| WhisperEncoderGraphExecutionError::GraphExecutionFailed {
                    reason: format!("could not upload tensor '{label}': {error}"),
                },
            )?;
    }

    let hidden_by_seq = graph
        .compute_output_f32(state, expected_len)
        .map_err(
            |error| WhisperEncoderGraphExecutionError::GraphExecutionFailed {
                reason: format!("encoder graph compute failed: {error}"),
            },
        )?;
    let hidden_seq_major = transpose_hidden_sequence_to_sequence_hidden(
        &hidden_by_seq,
        plan.output_frames,
        plan.output_hidden_size,
    );
    Ok(WhisperEncoderGraphExecutionOutput {
        hidden_state: hidden_seq_major,
        layout: WhisperEncoderHiddenStateLayout::SequenceHidden,
        frames: plan.output_frames,
        hidden_size: plan.output_hidden_size,
    })
}

#[cfg(test)]
fn reshape_projected_hidden_sequence_to_heads<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    sequence_len: usize,
    attention_heads: usize,
    label: &'static str,
) -> Result<GgmlCpuTensor<'a>, WhisperEncoderGraphExecutionError> {
    reshape_projection_to_attention_heads(
        graph,
        projection,
        AttentionHeadLayout {
            head_dim,
            attention_heads,
            sequence_len,
        },
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(attn_heads)",
            permute: "ggml_permute(attn_heads)",
            cont: label,
        },
        map_encoder_execute_graph_error,
    )
}

#[cfg(test)]
type EncoderUpload<'a> = (GgmlCpuTensor<'a>, Vec<f32>, &'static str);

#[cfg(test)]
fn apply_linear<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    uploads: &mut Vec<EncoderUpload<'a>>,
    source: &dyn WhisperEncoderTensorSource,
    input_tensor: GgmlCpuTensor<'a>,
    projection: &WhisperEncoderLinearProjectionPlan,
    label_prefix: &str,
) -> Result<GgmlCpuTensor<'a>, WhisperEncoderGraphExecutionError> {
    let mut weights = source.materialize_tensor_f32(&projection.weight)?;
    let expected_len = projection
        .input_dim
        .checked_mul(projection.output_dim)
        .ok_or_else(|| WhisperEncoderGraphExecutionError::InvalidInput {
            reason: format!(
                "{} projection dimensions overflow: {}x{}",
                projection.weight.tensor_name, projection.input_dim, projection.output_dim
            ),
        })?;
    if weights.len() != expected_len {
        return Err(
            WhisperEncoderGraphExecutionError::TensorMaterializationFailed {
                tensor_name: projection.weight.tensor_name.clone(),
                reason: format!(
                    "tensor has {} elements but projection expects {}",
                    weights.len(),
                    expected_len
                ),
            },
        );
    }
    if projection.weight_layout == WhisperEncoderLinearWeightLayout::OutputInput {
        weights = transpose_weight_output_input_to_input_output(
            &weights,
            projection.input_dim,
            projection.output_dim,
        )?;
    }

    let weight_name = Box::leak(format!("{label_prefix}_weight").into_boxed_str());
    let weight = graph
        .new_tensor_2d_f32(projection.input_dim, projection.output_dim, weight_name)
        .map_err(|error| {
            map_encoder_execute_graph_error("ggml_new_tensor_2d(linear_weight)", error)
        })?;
    graph
        .set_input(weight)
        .map_err(|error| map_encoder_execute_graph_error("ggml_set_input(linear_weight)", error))?;
    uploads.push((weight, weights, weight_name));
    graph
        .mul_mat(weight, input_tensor)
        .map_err(|error| map_encoder_execute_graph_error("ggml_mul_mat(linear)", error))
}

#[cfg(test)]
fn apply_affine_layer_norm<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    uploads: &mut Vec<EncoderUpload<'a>>,
    source: &dyn WhisperEncoderTensorSource,
    input_tensor: GgmlCpuTensor<'a>,
    layer_norm_epsilon: f32,
    norm: &WhisperEncoderNormPlan,
    label_prefix: &str,
) -> Result<GgmlCpuTensor<'a>, WhisperEncoderGraphExecutionError> {
    let hidden = usize::try_from(*norm.weight.dims.last().ok_or_else(|| {
        WhisperEncoderGraphExecutionError::InvalidInput {
            reason: format!("{} missing weight dims", norm.weight.tensor_name),
        }
    })?)
    .map_err(|_| WhisperEncoderGraphExecutionError::InvalidInput {
        reason: format!(
            "{} hidden dimension does not fit usize",
            norm.weight.tensor_name
        ),
    })?;
    let weight = materialize_hidden_vector(source, &norm.weight, hidden)?;
    let bias = materialize_hidden_vector(source, &norm.bias, hidden)?;

    let bias_name = Box::leak(format!("{label_prefix}_bias").into_boxed_str());
    let bias_tensor = graph
        .new_tensor_1d_f32(hidden, bias_name)
        .map_err(|error| {
            map_encoder_execute_graph_error("ggml_new_tensor_1d(layer_norm_bias)", error)
        })?;
    graph.set_input(bias_tensor).map_err(|error| {
        map_encoder_execute_graph_error("ggml_set_input(layer_norm_bias)", error)
    })?;
    uploads.push((bias_tensor, bias, bias_name));
    let weight_name = Box::leak(format!("{label_prefix}_weight").into_boxed_str());
    let weight_tensor = graph
        .new_tensor_1d_f32(hidden, weight_name)
        .map_err(|error| {
            map_encoder_execute_graph_error("ggml_new_tensor_1d(layer_norm_weight)", error)
        })?;
    graph.set_input(weight_tensor).map_err(|error| {
        map_encoder_execute_graph_error("ggml_set_input(layer_norm_weight)", error)
    })?;
    uploads.push((weight_tensor, weight, weight_name));
    nn_apply_affine_layer_norm(
        graph,
        input_tensor,
        layer_norm_epsilon,
        weight_tensor,
        bias_tensor,
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "ggml_mul(layer_norm_weight)",
            bias: "ggml_add(layer_norm_bias)",
        },
        map_encoder_execute_graph_error,
    )
}

#[cfg(test)]
fn materialize_hidden_vector(
    source: &dyn WhisperEncoderTensorSource,
    tensor: &WhisperEncoderGraphTensorRef,
    hidden: usize,
) -> Result<Vec<f32>, WhisperEncoderGraphExecutionError> {
    let values = source.materialize_tensor_f32(tensor)?;
    let start = values.len().saturating_sub(hidden);
    let vector = values.get(start..).ok_or_else(|| {
        WhisperEncoderGraphExecutionError::TensorMaterializationFailed {
            tensor_name: tensor.tensor_name.clone(),
            reason: format!(
                "tensor has {} elements but hidden vector requires at least {}",
                values.len(),
                hidden
            ),
        }
    })?;
    if vector.len() != hidden {
        return Err(
            WhisperEncoderGraphExecutionError::TensorMaterializationFailed {
                tensor_name: tensor.tensor_name.clone(),
                reason: format!(
                    "tensor has {} elements but hidden vector requires exactly {hidden}",
                    values.len()
                ),
            },
        );
    }
    Ok(vector.to_vec())
}

#[cfg(test)]
fn transpose_weight_output_input_to_input_output(
    source: &[f32],
    input_dim: usize,
    output_dim: usize,
) -> Result<Vec<f32>, WhisperEncoderGraphExecutionError> {
    if source.len() != input_dim * output_dim {
        return Err(WhisperEncoderGraphExecutionError::InvalidInput {
            reason: format!(
                "cannot transpose weight with {} values for {}x{}",
                source.len(),
                output_dim,
                input_dim
            ),
        });
    }
    let mut transposed = vec![0.0f32; source.len()];
    for out_idx in 0..output_dim {
        for in_idx in 0..input_dim {
            let src = in_idx
                .checked_add(out_idx.saturating_mul(input_dim))
                .ok_or_else(|| WhisperEncoderGraphExecutionError::InvalidInput {
                    reason: "weight transpose source index overflow".to_string(),
                })?;
            let dst = in_idx
                .checked_add(out_idx.saturating_mul(input_dim))
                .ok_or_else(|| WhisperEncoderGraphExecutionError::InvalidInput {
                    reason: "weight transpose destination index overflow".to_string(),
                })?;
            transposed[dst] = source[src];
        }
    }
    Ok(transposed)
}

#[cfg(test)]
fn normalize_hidden_layout(
    input: &WhisperEncoderGraphExecutionInput,
    frames: usize,
    hidden: usize,
) -> Vec<f32> {
    transpose_sequence_hidden_to_hidden_sequence(&input.hidden_state, frames, hidden)
}

#[cfg(test)]
fn transpose_sequence_hidden_to_hidden_sequence(
    input: &[f32],
    frames: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for frame_idx in 0..frames {
        for hidden_idx in 0..hidden {
            let src = frame_idx * hidden + hidden_idx;
            let dst = hidden_idx * frames + frame_idx;
            output[dst] = input[src];
        }
    }
    output
}

#[cfg(test)]
fn transpose_hidden_sequence_to_sequence_hidden(
    input: &[f32],
    frames: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for hidden_idx in 0..hidden {
        for frame_idx in 0..frames {
            let src = hidden_idx * frames + frame_idx;
            let dst = frame_idx * hidden + hidden_idx;
            output[dst] = input[src];
        }
    }
    output
}

#[cfg(test)]
fn map_encoder_execute_graph_error(
    primitive: &'static str,
    error: GgmlCpuGraphError,
) -> WhisperEncoderGraphExecutionError {
    match error {
        GgmlCpuGraphError::UnsupportedOperation { .. }
        | GgmlCpuGraphError::UnsupportedInputs { .. }
        | GgmlCpuGraphError::GraphBuildFailed { .. } => {
            WhisperEncoderGraphExecutionError::UnsupportedEncoderPrimitive {
                primitive,
                reason: error.to_string(),
            }
        }
        _ => WhisperEncoderGraphExecutionError::GraphExecutionFailed {
            reason: format!("{primitive} failed: {error}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn one_layer_tiny_encoder_graph_plan_builds() {
        let metadata = WhisperEncoderGraphMetadata {
            encoder_layers: 1,
            encoder_hidden_size: 4,
        };
        let binding = WhisperEncoderTensorBindingSeam {
            layers: vec![one_layer_binding()],
            final_norm_weight: Some(tensor("model.encoder.layer_norm.weight", &[4])),
            final_norm_bias: Some(tensor("model.encoder.layer_norm.bias", &[4])),
        };
        let materialization = WhisperEncoderTensorMaterializationSeam {
            source_label: "test-fixture",
            materialized_tensor_count: 12,
        };
        let input_shape = WhisperEncoderGraphInputShape {
            frames: 8,
            hidden_size: 4,
        };

        let plan =
            build_whisper_encoder_graph_plan(metadata, &binding, &materialization, input_shape)
                .expect("one-layer encoder graph plan should succeed");
        assert_eq!(plan.layers.len(), 1);
        assert_eq!(plan.output_frames, 8);
        assert_eq!(plan.output_hidden_size, 4);
        assert_eq!(
            plan.layers[0].self_attn_q.weight_layout,
            WhisperEncoderLinearWeightLayout::OutputInput
        );
        assert_eq!(
            plan.layers[0].mlp_fc1.weight_layout,
            WhisperEncoderLinearWeightLayout::OutputInput
        );
        assert_eq!(plan.layers[0].mlp_fc1.output_dim, 8);
        assert_eq!(plan.layers[0].mlp_fc2.input_dim, 8);
    }

    #[test]
    fn missing_attn_norm_fails_closed() {
        let metadata = WhisperEncoderGraphMetadata {
            encoder_layers: 1,
            encoder_hidden_size: 4,
        };
        let mut layer = one_layer_binding();
        layer.self_attn_norm_weight = None;
        let binding = WhisperEncoderTensorBindingSeam {
            layers: vec![layer],
            final_norm_weight: Some(tensor("model.encoder.layer_norm.weight", &[4])),
            final_norm_bias: Some(tensor("model.encoder.layer_norm.bias", &[4])),
        };
        let materialization = WhisperEncoderTensorMaterializationSeam {
            source_label: "test-fixture",
            materialized_tensor_count: 12,
        };
        let input_shape = WhisperEncoderGraphInputShape {
            frames: 8,
            hidden_size: 4,
        };

        let error =
            build_whisper_encoder_graph_plan(metadata, &binding, &materialization, input_shape)
                .expect_err("missing attn norm must fail closed");
        assert!(matches!(
            error,
            WhisperEncoderGraphPlanError::MissingTensorBinding { .. }
        ));
    }

    #[test]
    fn fc2_shape_mismatch_fails_closed() {
        let metadata = WhisperEncoderGraphMetadata {
            encoder_layers: 1,
            encoder_hidden_size: 4,
        };
        let mut layer = one_layer_binding();
        layer.mlp_fc2_weight = Some(tensor("model.encoder.layers.0.fc2.weight", &[4, 7]));
        let binding = WhisperEncoderTensorBindingSeam {
            layers: vec![layer],
            final_norm_weight: Some(tensor("model.encoder.layer_norm.weight", &[4])),
            final_norm_bias: Some(tensor("model.encoder.layer_norm.bias", &[4])),
        };
        let materialization = WhisperEncoderTensorMaterializationSeam {
            source_label: "test-fixture",
            materialized_tensor_count: 12,
        };
        let input_shape = WhisperEncoderGraphInputShape {
            frames: 8,
            hidden_size: 4,
        };

        let error =
            build_whisper_encoder_graph_plan(metadata, &binding, &materialization, input_shape)
                .expect_err("fc2 mismatch must fail closed");
        assert!(matches!(
            error,
            WhisperEncoderGraphPlanError::TensorShapeMismatch { .. }
        ));
    }

    #[test]
    fn square_projection_without_runtime_layout_fails_closed() {
        let metadata = WhisperEncoderGraphMetadata {
            encoder_layers: 1,
            encoder_hidden_size: 4,
        };
        let mut layer = one_layer_binding();
        layer.self_attn_q_weight = Some(tensor_without_layout(
            "model.encoder.layers.0.self_attn.q_proj.weight",
            &[4, 4],
        ));
        let binding = WhisperEncoderTensorBindingSeam {
            layers: vec![layer],
            final_norm_weight: Some(tensor("model.encoder.layer_norm.weight", &[4])),
            final_norm_bias: Some(tensor("model.encoder.layer_norm.bias", &[4])),
        };
        let materialization = WhisperEncoderTensorMaterializationSeam {
            source_label: "test-fixture",
            materialized_tensor_count: 12,
        };
        let input_shape = WhisperEncoderGraphInputShape {
            frames: 8,
            hidden_size: 4,
        };

        let error =
            build_whisper_encoder_graph_plan(metadata, &binding, &materialization, input_shape)
                .expect_err("square projection without runtime layout must fail closed");
        let WhisperEncoderGraphPlanError::TensorShapeMismatch { reason, .. } = error else {
            panic!("expected shape mismatch error");
        };
        assert!(
            reason.contains("ambiguous square projection requires runtime_linear_weight_layout"),
            "unexpected reason: {reason}"
        );
    }

    #[test]
    fn tiny_one_layer_encoder_execution_returns_finite_output() {
        let metadata = WhisperEncoderGraphMetadata {
            encoder_layers: 1,
            encoder_hidden_size: 4,
        };
        let binding = WhisperEncoderTensorBindingSeam {
            layers: vec![one_layer_binding()],
            final_norm_weight: Some(tensor("model.encoder.layer_norm.weight", &[4])),
            final_norm_bias: Some(tensor("model.encoder.layer_norm.bias", &[4])),
        };
        let materialization = WhisperEncoderTensorMaterializationSeam {
            source_label: "test-fixture",
            materialized_tensor_count: 12,
        };
        let input_shape = WhisperEncoderGraphInputShape {
            frames: 3,
            hidden_size: 4,
        };
        let plan =
            build_whisper_encoder_graph_plan(metadata, &binding, &materialization, input_shape)
                .expect("one-layer encoder graph plan should succeed");
        let source = MockTensorSource::from_plan(&plan);

        let output = execute_whisper_encoder_graph_ggml_v0(
            &plan,
            &WhisperEncoderGraphExecutionInput {
                hidden_state: vec![
                    0.1, 0.2, 0.3, 0.4, //
                    0.5, 0.6, 0.7, 0.8, //
                    0.9, 1.0, 1.1, 1.2,
                ],
                layout: WhisperEncoderHiddenStateLayout::SequenceHidden,
            },
            &source,
            WhisperEncoderGraphExecutionConfig {
                attention_heads: 2,
                use_flash_attention: false,
                layer_norm_epsilon: 1.0e-5,
            },
        )
        .expect("tiny one-layer encoder must execute");

        assert_eq!(output.frames, 3);
        assert_eq!(output.hidden_size, 4);
        assert_eq!(output.hidden_state.len(), 12);
        assert!(
            output.hidden_state.iter().all(|value| value.is_finite()),
            "encoder output should be finite: {:?}",
            output.hidden_state
        );
    }

    fn one_layer_binding() -> WhisperEncoderLayerTensorBinding {
        WhisperEncoderLayerTensorBinding {
            self_attn_norm_weight: Some(tensor(
                "model.encoder.layers.0.self_attn_layer_norm.weight",
                &[4],
            )),
            self_attn_norm_bias: Some(tensor(
                "model.encoder.layers.0.self_attn_layer_norm.bias",
                &[4],
            )),
            self_attn_q_weight: Some(tensor(
                "model.encoder.layers.0.self_attn.q_proj.weight",
                &[4, 4],
            )),
            self_attn_k_weight: Some(tensor(
                "model.encoder.layers.0.self_attn.k_proj.weight",
                &[4, 4],
            )),
            self_attn_v_weight: Some(tensor(
                "model.encoder.layers.0.self_attn.v_proj.weight",
                &[4, 4],
            )),
            self_attn_out_weight: Some(tensor(
                "model.encoder.layers.0.self_attn.out_proj.weight",
                &[4, 4],
            )),
            mlp_norm_weight: Some(tensor(
                "model.encoder.layers.0.final_layer_norm.weight",
                &[4],
            )),
            mlp_norm_bias: Some(tensor("model.encoder.layers.0.final_layer_norm.bias", &[4])),
            mlp_fc1_weight: Some(tensor("model.encoder.layers.0.fc1.weight", &[8, 4])),
            mlp_fc2_weight: Some(tensor("model.encoder.layers.0.fc2.weight", &[4, 8])),
        }
    }

    fn tensor(name: &str, dims: &[u64]) -> WhisperEncoderGraphTensorRef {
        WhisperEncoderGraphTensorRef {
            tensor_name: name.to_string(),
            tensor_num_elements: dims.iter().copied().product::<u64>() as usize,
            dims: dims.to_vec(),
            runtime_linear_weight_layout: if dims.len() == 2 {
                Some(WhisperEncoderLinearWeightLayout::OutputInput)
            } else {
                None
            },
        }
    }

    fn tensor_without_layout(name: &str, dims: &[u64]) -> WhisperEncoderGraphTensorRef {
        WhisperEncoderGraphTensorRef {
            tensor_name: name.to_string(),
            tensor_num_elements: dims.iter().copied().product::<u64>() as usize,
            dims: dims.to_vec(),
            runtime_linear_weight_layout: None,
        }
    }

    struct MockTensorSource {
        values: BTreeMap<String, Vec<f32>>,
    }

    impl MockTensorSource {
        fn from_plan(plan: &WhisperEncoderGraphPlan) -> Self {
            let mut values = BTreeMap::new();
            for layer in &plan.layers {
                insert_default_tensor(&mut values, &layer.self_attn_norm.weight);
                insert_default_tensor(&mut values, &layer.self_attn_norm.bias);
                insert_default_tensor(&mut values, &layer.self_attn_q.weight);
                insert_default_tensor(&mut values, &layer.self_attn_k.weight);
                insert_default_tensor(&mut values, &layer.self_attn_v.weight);
                insert_default_tensor(&mut values, &layer.self_attn_out.weight);
                insert_default_tensor(&mut values, &layer.mlp_norm.weight);
                insert_default_tensor(&mut values, &layer.mlp_norm.bias);
                insert_default_tensor(&mut values, &layer.mlp_fc1.weight);
                insert_default_tensor(&mut values, &layer.mlp_fc2.weight);
            }
            insert_default_tensor(&mut values, &plan.final_norm.weight);
            insert_default_tensor(&mut values, &plan.final_norm.bias);
            Self { values }
        }
    }

    impl WhisperEncoderTensorSource for MockTensorSource {
        fn materialize_tensor_f32(
            &self,
            tensor: &WhisperEncoderGraphTensorRef,
        ) -> Result<Vec<f32>, WhisperEncoderGraphExecutionError> {
            let Some(values) = self.values.get(&tensor.tensor_name) else {
                return Err(
                    WhisperEncoderGraphExecutionError::MissingMaterializedTensor {
                        tensor_name: tensor.tensor_name.clone(),
                        reason: "test tensor is absent".to_string(),
                    },
                );
            };
            Ok(values.clone())
        }
    }

    fn insert_default_tensor(
        values: &mut BTreeMap<String, Vec<f32>>,
        tensor: &WhisperEncoderGraphTensorRef,
    ) {
        let data = if tensor.tensor_name.contains("norm.weight") {
            vec![1.0f32; tensor.tensor_num_elements]
        } else if tensor.tensor_name.contains("norm.bias") {
            vec![0.0f32; tensor.tensor_num_elements]
        } else if tensor.dims.len() == 2 && tensor.dims[0] == tensor.dims[1] {
            let dim = tensor.dims[0] as usize;
            let mut identity = vec![0.0f32; dim * dim];
            for idx in 0..dim {
                identity[idx * dim + idx] = 1.0;
            }
            identity
        } else {
            (0..tensor.tensor_num_elements)
                .map(|idx| ((idx % 11) as f32) * 0.01)
                .collect()
        };
        values.insert(tensor.tensor_name.clone(), data);
    }
}
