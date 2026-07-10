//! Hands-off: single-responsibility ggml graph transcription, guarded by
//! golden/parity tests. Do not split this module for "tidiness" -- the tensor
//! wiring is validated as a whole and refactoring here risks silent numeric
//! drift.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    path::Path,
    sync::Arc,
    time::Instant,
};

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlLoadedTensor, GgmlLoadedWeightContext, GgmlStaticTensor,
    GgmlStaticTensorArena,
};
use crate::nn::attn::{
    AttentionHeadLayout, AttentionReshapeSteps, STANDARD_HEAD_PERMUTE_AXES,
    reshape_projection_to_attention_heads,
};
use crate::nn::decoder::{
    Seq2SeqReusableDecodeGraph, build_fixed_kv_attention_mask_bits,
    build_fixed_kv_attention_mask_bits_for_query_rows,
    build_fixed_kv_attention_mask_bits_for_sequences, seq2seq_indexed_layer_stack,
};
use crate::nn::half::f32_to_f16_bits;

use super::execution_policy::{
    whisper_decoder_persistent_cross_cache_default_f32_rhs_on_cpu_enabled as decoder_persistent_cross_cache_default_f32_rhs_on_cpu_enabled,
    whisper_decoder_persistent_cross_cache_f16_upload_enabled as decoder_persistent_cross_cache_f16_upload_enabled,
};
#[cfg(test)]
use super::execution_policy::{
    whisper_decoder_persistent_cross_cache_default_f32_rhs_on_cpu_enabled_with_env as decoder_persistent_cross_cache_default_f32_rhs_on_cpu_enabled_with_env,
    whisper_decoder_persistent_cross_cache_f16_upload_enabled_with_env as decoder_persistent_cross_cache_f16_upload_enabled_with_env,
};
use super::graph_config::whisper_decoder_graph_config;

const GGML_TYPE_F16: i32 = 1;
const WHISPER_DECODER_REUSE_GRAPH_CONTEXT_BYTES: usize = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WhisperDecoderGraphInputShape {
    pub token_count: usize,
    pub encoder_frames: usize,
    pub hidden_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WhisperDecoderGraphMetadata {
    pub decoder_layers: usize,
    pub decoder_hidden_size: usize,
    pub decoder_attention_heads: usize,
    pub vocab_size: usize,
    pub max_target_positions: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperDecoderGraphTensorRef {
    pub tensor_name: String,
    pub tensor_num_elements: usize,
    pub dims: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperDecoderLayerTensorBinding {
    pub self_attn_norm_weight: Option<WhisperDecoderGraphTensorRef>,
    pub self_attn_norm_bias: Option<WhisperDecoderGraphTensorRef>,
    pub self_attn_q_weight: Option<WhisperDecoderGraphTensorRef>,
    pub self_attn_q_bias: Option<WhisperDecoderGraphTensorRef>,
    pub self_attn_k_weight: Option<WhisperDecoderGraphTensorRef>,
    pub self_attn_v_weight: Option<WhisperDecoderGraphTensorRef>,
    pub self_attn_v_bias: Option<WhisperDecoderGraphTensorRef>,
    pub self_attn_out_weight: Option<WhisperDecoderGraphTensorRef>,
    pub self_attn_out_bias: Option<WhisperDecoderGraphTensorRef>,
    pub cross_attn_norm_weight: Option<WhisperDecoderGraphTensorRef>,
    pub cross_attn_norm_bias: Option<WhisperDecoderGraphTensorRef>,
    pub cross_attn_q_weight: Option<WhisperDecoderGraphTensorRef>,
    pub cross_attn_q_bias: Option<WhisperDecoderGraphTensorRef>,
    pub cross_attn_k_weight: Option<WhisperDecoderGraphTensorRef>,
    pub cross_attn_v_weight: Option<WhisperDecoderGraphTensorRef>,
    pub cross_attn_v_bias: Option<WhisperDecoderGraphTensorRef>,
    pub cross_attn_out_weight: Option<WhisperDecoderGraphTensorRef>,
    pub cross_attn_out_bias: Option<WhisperDecoderGraphTensorRef>,
    pub mlp_norm_weight: Option<WhisperDecoderGraphTensorRef>,
    pub mlp_norm_bias: Option<WhisperDecoderGraphTensorRef>,
    pub mlp_fc1_weight: Option<WhisperDecoderGraphTensorRef>,
    pub mlp_fc1_bias: Option<WhisperDecoderGraphTensorRef>,
    pub mlp_fc2_weight: Option<WhisperDecoderGraphTensorRef>,
    pub mlp_fc2_bias: Option<WhisperDecoderGraphTensorRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperDecoderTensorBindingSeam {
    pub token_embedding_weight: Option<WhisperDecoderGraphTensorRef>,
    pub position_embedding_weight: Option<WhisperDecoderGraphTensorRef>,
    pub final_norm_weight: Option<WhisperDecoderGraphTensorRef>,
    pub final_norm_bias: Option<WhisperDecoderGraphTensorRef>,
    pub output_projection_weight: Option<WhisperDecoderGraphTensorRef>,
    pub output_projection_bias: Option<WhisperDecoderGraphTensorRef>,
    pub layers: Vec<WhisperDecoderLayerTensorBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperDecoderTensorMaterializationSeam {
    pub source_label: &'static str,
    pub materialized_tensor_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WhisperDecoderEmbeddingLayout {
    VocabHidden,
    HiddenVocab,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperDecoderEmbeddingPlan {
    pub weight: WhisperDecoderGraphTensorRef,
    pub layout: WhisperDecoderEmbeddingLayout,
    pub vocab_size: usize,
    pub hidden_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperDecoderNormPlan {
    pub weight: WhisperDecoderGraphTensorRef,
    pub bias: WhisperDecoderGraphTensorRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum WhisperDecoderLinearWeightLayout {
    InputOutput,
    OutputInput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperDecoderLinearProjectionPlan {
    pub weight: WhisperDecoderGraphTensorRef,
    pub weight_layout: WhisperDecoderLinearWeightLayout,
    pub input_dim: usize,
    pub output_dim: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperDecoderLinearWithBiasPlan {
    pub projection: WhisperDecoderLinearProjectionPlan,
    pub bias: WhisperDecoderGraphTensorRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperDecoderLayerPlan {
    pub layer_idx: usize,
    pub self_attn_norm: WhisperDecoderNormPlan,
    pub self_attn_q: WhisperDecoderLinearWithBiasPlan,
    pub self_attn_k: WhisperDecoderLinearProjectionPlan,
    pub self_attn_v: WhisperDecoderLinearWithBiasPlan,
    pub self_attn_out: WhisperDecoderLinearWithBiasPlan,
    pub cross_attn_norm: WhisperDecoderNormPlan,
    pub cross_attn_q: WhisperDecoderLinearWithBiasPlan,
    pub cross_attn_k: WhisperDecoderLinearProjectionPlan,
    pub cross_attn_v: WhisperDecoderLinearWithBiasPlan,
    pub cross_attn_out: WhisperDecoderLinearWithBiasPlan,
    pub mlp_norm: WhisperDecoderNormPlan,
    pub mlp_fc1: WhisperDecoderLinearWithBiasPlan,
    pub mlp_fc2: WhisperDecoderLinearWithBiasPlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperDecoderOutputProjectionPlan {
    pub projection: WhisperDecoderLinearProjectionPlan,
    pub bias: Option<WhisperDecoderGraphTensorRef>,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperDecoderGraphPlan {
    pub input_shape: WhisperDecoderGraphInputShape,
    pub decoder_attention_heads: usize,
    pub token_embedding: WhisperDecoderEmbeddingPlan,
    pub position_embedding: WhisperDecoderEmbeddingPlan,
    pub layers: Vec<WhisperDecoderLayerPlan>,
    pub final_norm: WhisperDecoderNormPlan,
    pub output_projection: WhisperDecoderOutputProjectionPlan,
    pub required_primitives: Vec<&'static str>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum WhisperDecoderGraphPlanError {
    #[error("whisper decoder graph input shape is invalid: {reason}")]
    InvalidInputShape { reason: String },
    #[error(
        "whisper decoder graph binding layer count mismatch: metadata={metadata_layers}, binding={binding_layers}"
    )]
    LayerCountMismatch {
        metadata_layers: usize,
        binding_layers: usize,
    },
    #[error("whisper decoder graph binding is missing layer {layer_idx}")]
    MissingLayerBinding { layer_idx: usize },
    #[error("whisper decoder graph is missing required tensor '{slot}' at {scope}")]
    MissingTensorBinding { scope: String, slot: &'static str },
    #[error(
        "whisper decoder graph tensor '{tensor_name}' for '{slot}' at {scope} has invalid shape {found_shape:?}: {reason}"
    )]
    TensorShapeMismatch {
        scope: String,
        slot: &'static str,
        tensor_name: String,
        found_shape: Vec<u64>,
        reason: String,
    },
    #[error("whisper decoder graph unsupported primitive '{primitive}': {reason}")]
    UnsupportedDecoderPrimitive {
        primitive: &'static str,
        reason: String,
    },
}

pub(crate) struct WhisperDecoderGraphBuilder<'a> {
    metadata: WhisperDecoderGraphMetadata,
    binding: &'a WhisperDecoderTensorBindingSeam,
    materialization: &'a WhisperDecoderTensorMaterializationSeam,
    input_shape: WhisperDecoderGraphInputShape,
}

impl<'a> WhisperDecoderGraphBuilder<'a> {
    pub(crate) fn new(
        metadata: WhisperDecoderGraphMetadata,
        binding: &'a WhisperDecoderTensorBindingSeam,
        materialization: &'a WhisperDecoderTensorMaterializationSeam,
        input_shape: WhisperDecoderGraphInputShape,
    ) -> Self {
        Self {
            metadata,
            binding,
            materialization,
            input_shape,
        }
    }

    pub(crate) fn build(&self) -> Result<WhisperDecoderGraphPlan, WhisperDecoderGraphPlanError> {
        if self.input_shape.token_count == 0 {
            return Err(WhisperDecoderGraphPlanError::InvalidInputShape {
                reason: "token_count must be > 0".to_string(),
            });
        }
        if self.input_shape.encoder_frames == 0 {
            return Err(WhisperDecoderGraphPlanError::InvalidInputShape {
                reason: "encoder_frames must be > 0".to_string(),
            });
        }
        if self.input_shape.hidden_size == 0 {
            return Err(WhisperDecoderGraphPlanError::InvalidInputShape {
                reason: "hidden_size must be > 0".to_string(),
            });
        }
        if self.metadata.decoder_hidden_size == 0 {
            return Err(WhisperDecoderGraphPlanError::InvalidInputShape {
                reason: "decoder_hidden_size must be > 0".to_string(),
            });
        }
        if self.input_shape.hidden_size != self.metadata.decoder_hidden_size {
            return Err(WhisperDecoderGraphPlanError::InvalidInputShape {
                reason: format!(
                    "input hidden_size={} does not match whisper.decoder.embedding_length={}",
                    self.input_shape.hidden_size, self.metadata.decoder_hidden_size
                ),
            });
        }
        if self.metadata.decoder_layers == 0 {
            return Err(WhisperDecoderGraphPlanError::InvalidInputShape {
                reason: "decoder_layers must be > 0".to_string(),
            });
        }
        if self.binding.layers.len() != self.metadata.decoder_layers {
            return Err(WhisperDecoderGraphPlanError::LayerCountMismatch {
                metadata_layers: self.metadata.decoder_layers,
                binding_layers: self.binding.layers.len(),
            });
        }
        if self.metadata.decoder_attention_heads == 0 {
            return Err(WhisperDecoderGraphPlanError::InvalidInputShape {
                reason: "decoder_attention_heads must be > 0".to_string(),
            });
        }
        if !self
            .metadata
            .decoder_hidden_size
            .is_multiple_of(self.metadata.decoder_attention_heads)
        {
            return Err(WhisperDecoderGraphPlanError::InvalidInputShape {
                reason: format!(
                    "decoder_hidden_size {} must be divisible by decoder_attention_heads {}",
                    self.metadata.decoder_hidden_size, self.metadata.decoder_attention_heads
                ),
            });
        }
        if self.input_shape.token_count > self.metadata.max_target_positions {
            return Err(WhisperDecoderGraphPlanError::InvalidInputShape {
                reason: format!(
                    "token_count {} exceeds max_target_positions {}",
                    self.input_shape.token_count, self.metadata.max_target_positions
                ),
            });
        }
        if self.metadata.vocab_size == 0 {
            return Err(WhisperDecoderGraphPlanError::InvalidInputShape {
                reason: "vocab_size must be > 0".to_string(),
            });
        }
        if self.materialization.materialized_tensor_count == 0 {
            return Err(WhisperDecoderGraphPlanError::UnsupportedDecoderPrimitive {
                primitive: "decoder.tensor_materialization",
                reason: format!(
                    "materialization seam '{}' resolved no tensors",
                    self.materialization.source_label
                ),
            });
        }

        let token_embedding = self.parse_embedding(
            "decoder",
            "embed_tokens.weight",
            self.binding.token_embedding_weight.as_ref(),
            self.metadata.vocab_size,
            self.input_shape.hidden_size,
        )?;
        let position_embedding = self.parse_embedding(
            "decoder",
            "embed_positions.weight",
            self.binding.position_embedding_weight.as_ref(),
            self.metadata.max_target_positions,
            self.input_shape.hidden_size,
        )?;
        let final_norm = self.parse_norm(
            "decoder",
            "layer_norm",
            self.binding.final_norm_weight.as_ref(),
            self.binding.final_norm_bias.as_ref(),
            self.input_shape.hidden_size,
        )?;
        let output_projection = self.parse_output_projection(
            "decoder",
            self.binding.output_projection_weight.as_ref(),
            self.binding.output_projection_bias.as_ref(),
            self.input_shape.hidden_size,
            self.metadata.vocab_size,
        )?;

        let mut layers = Vec::with_capacity(self.metadata.decoder_layers);
        for layer_idx in 0..self.metadata.decoder_layers {
            let layer = self
                .binding
                .layers
                .get(layer_idx)
                .ok_or(WhisperDecoderGraphPlanError::MissingLayerBinding { layer_idx })?;
            layers.push(self.parse_layer(layer_idx, layer)?);
        }

        Ok(WhisperDecoderGraphPlan {
            input_shape: self.input_shape,
            decoder_attention_heads: self.metadata.decoder_attention_heads,
            token_embedding,
            position_embedding,
            layers,
            final_norm,
            output_projection,
            required_primitives: required_decoder_primitives(),
        })
    }

    fn parse_layer(
        &self,
        layer_idx: usize,
        layer_binding: &WhisperDecoderLayerTensorBinding,
    ) -> Result<WhisperDecoderLayerPlan, WhisperDecoderGraphPlanError> {
        let hidden = self.input_shape.hidden_size;
        let scope = format!("decoder.layer[{layer_idx}]");
        let self_attn_norm = self.parse_norm(
            &scope,
            "self_attn_layer_norm",
            layer_binding.self_attn_norm_weight.as_ref(),
            layer_binding.self_attn_norm_bias.as_ref(),
            hidden,
        )?;
        let self_attn_q = self.parse_linear_with_bias(
            &scope,
            "self_attn.q_proj.weight",
            layer_binding.self_attn_q_weight.as_ref(),
            layer_binding.self_attn_q_bias.as_ref(),
            hidden,
            Some(hidden),
        )?;
        let self_attn_k = self.parse_linear(
            &scope,
            "self_attn.k_proj.weight",
            layer_binding.self_attn_k_weight.as_ref(),
            hidden,
            Some(hidden),
        )?;
        let self_attn_v = self.parse_linear_with_bias(
            &scope,
            "self_attn.v_proj.weight",
            layer_binding.self_attn_v_weight.as_ref(),
            layer_binding.self_attn_v_bias.as_ref(),
            hidden,
            Some(hidden),
        )?;
        let self_attn_out = self.parse_linear_with_bias(
            &scope,
            "self_attn.out_proj.weight",
            layer_binding.self_attn_out_weight.as_ref(),
            layer_binding.self_attn_out_bias.as_ref(),
            hidden,
            Some(hidden),
        )?;

        let cross_attn_norm = self.parse_norm(
            &scope,
            "encoder_attn_layer_norm",
            layer_binding.cross_attn_norm_weight.as_ref(),
            layer_binding.cross_attn_norm_bias.as_ref(),
            hidden,
        )?;
        let cross_attn_q = self.parse_linear_with_bias(
            &scope,
            "encoder_attn.q_proj.weight",
            layer_binding.cross_attn_q_weight.as_ref(),
            layer_binding.cross_attn_q_bias.as_ref(),
            hidden,
            Some(hidden),
        )?;
        let cross_attn_k = self.parse_linear(
            &scope,
            "encoder_attn.k_proj.weight",
            layer_binding.cross_attn_k_weight.as_ref(),
            hidden,
            Some(hidden),
        )?;
        let cross_attn_v = self.parse_linear_with_bias(
            &scope,
            "encoder_attn.v_proj.weight",
            layer_binding.cross_attn_v_weight.as_ref(),
            layer_binding.cross_attn_v_bias.as_ref(),
            hidden,
            Some(hidden),
        )?;
        let cross_attn_out = self.parse_linear_with_bias(
            &scope,
            "encoder_attn.out_proj.weight",
            layer_binding.cross_attn_out_weight.as_ref(),
            layer_binding.cross_attn_out_bias.as_ref(),
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
        let mlp_fc1 = self.parse_linear_with_bias(
            &scope,
            "fc1.weight",
            layer_binding.mlp_fc1_weight.as_ref(),
            layer_binding.mlp_fc1_bias.as_ref(),
            hidden,
            None,
        )?;
        let mlp_fc2 = self.parse_linear_with_bias(
            &scope,
            "fc2.weight",
            layer_binding.mlp_fc2_weight.as_ref(),
            layer_binding.mlp_fc2_bias.as_ref(),
            mlp_fc1.projection.output_dim,
            Some(hidden),
        )?;

        Ok(WhisperDecoderLayerPlan {
            layer_idx,
            self_attn_norm,
            self_attn_q,
            self_attn_k,
            self_attn_v,
            self_attn_out,
            cross_attn_norm,
            cross_attn_q,
            cross_attn_k,
            cross_attn_v,
            cross_attn_out,
            mlp_norm,
            mlp_fc1,
            mlp_fc2,
        })
    }

    fn parse_embedding(
        &self,
        scope: &str,
        slot: &'static str,
        tensor: Option<&WhisperDecoderGraphTensorRef>,
        expected_axis0: usize,
        expected_hidden: usize,
    ) -> Result<WhisperDecoderEmbeddingPlan, WhisperDecoderGraphPlanError> {
        let tensor = tensor.ok_or_else(|| WhisperDecoderGraphPlanError::MissingTensorBinding {
            scope: scope.to_string(),
            slot,
        })?;
        if tensor.dims.len() != 2 {
            return Err(WhisperDecoderGraphPlanError::TensorShapeMismatch {
                scope: scope.to_string(),
                slot,
                tensor_name: tensor.tensor_name.clone(),
                found_shape: tensor.dims.clone(),
                reason: "expected rank-2 embedding tensor".to_string(),
            });
        }

        let lhs = usize::try_from(tensor.dims[0]).map_err(|_| {
            WhisperDecoderGraphPlanError::TensorShapeMismatch {
                scope: scope.to_string(),
                slot,
                tensor_name: tensor.tensor_name.clone(),
                found_shape: tensor.dims.clone(),
                reason: "dimension does not fit usize".to_string(),
            }
        })?;
        let rhs = usize::try_from(tensor.dims[1]).map_err(|_| {
            WhisperDecoderGraphPlanError::TensorShapeMismatch {
                scope: scope.to_string(),
                slot,
                tensor_name: tensor.tensor_name.clone(),
                found_shape: tensor.dims.clone(),
                reason: "dimension does not fit usize".to_string(),
            }
        })?;

        let (layout, axis0, hidden_size) = if lhs == expected_axis0 && rhs == expected_hidden {
            (WhisperDecoderEmbeddingLayout::VocabHidden, lhs, rhs)
        } else if lhs == expected_hidden && rhs == expected_axis0 {
            (WhisperDecoderEmbeddingLayout::HiddenVocab, rhs, lhs)
        } else {
            return Err(WhisperDecoderGraphPlanError::TensorShapeMismatch {
                scope: scope.to_string(),
                slot,
                tensor_name: tensor.tensor_name.clone(),
                found_shape: tensor.dims.clone(),
                reason: format!(
                    "expected [{}x{}] or [{}x{}]",
                    expected_axis0, expected_hidden, expected_hidden, expected_axis0
                ),
            });
        };

        Ok(WhisperDecoderEmbeddingPlan {
            weight: tensor.clone(),
            layout,
            vocab_size: axis0,
            hidden_size,
        })
    }

    fn parse_output_projection(
        &self,
        scope: &str,
        weight: Option<&WhisperDecoderGraphTensorRef>,
        bias: Option<&WhisperDecoderGraphTensorRef>,
        expected_input_dim: usize,
        vocab_size: usize,
    ) -> Result<WhisperDecoderOutputProjectionPlan, WhisperDecoderGraphPlanError> {
        let projection = self.parse_linear(
            scope,
            "output_projection.weight",
            weight,
            expected_input_dim,
            Some(vocab_size),
        )?;
        let bias = if let Some(bias) = bias {
            validate_bias_shape(scope, "output_projection.bias", bias, vocab_size)?;
            Some(bias.clone())
        } else {
            None
        };
        Ok(WhisperDecoderOutputProjectionPlan {
            projection,
            bias,
            vocab_size,
        })
    }

    fn parse_linear_with_bias(
        &self,
        scope: &str,
        slot: &'static str,
        weight: Option<&WhisperDecoderGraphTensorRef>,
        bias: Option<&WhisperDecoderGraphTensorRef>,
        expected_input_dim: usize,
        expected_output_dim: Option<usize>,
    ) -> Result<WhisperDecoderLinearWithBiasPlan, WhisperDecoderGraphPlanError> {
        let projection =
            self.parse_linear(scope, slot, weight, expected_input_dim, expected_output_dim)?;
        let bias = bias.ok_or_else(|| WhisperDecoderGraphPlanError::MissingTensorBinding {
            scope: scope.to_string(),
            slot,
        })?;
        validate_bias_shape(scope, slot, bias, projection.output_dim)?;
        Ok(WhisperDecoderLinearWithBiasPlan {
            projection,
            bias: bias.clone(),
        })
    }

    fn parse_linear(
        &self,
        scope: &str,
        slot: &'static str,
        tensor: Option<&WhisperDecoderGraphTensorRef>,
        expected_input_dim: usize,
        expected_output_dim: Option<usize>,
    ) -> Result<WhisperDecoderLinearProjectionPlan, WhisperDecoderGraphPlanError> {
        let tensor = tensor.ok_or_else(|| WhisperDecoderGraphPlanError::MissingTensorBinding {
            scope: scope.to_string(),
            slot,
        })?;
        if tensor.dims.len() != 2 {
            return Err(WhisperDecoderGraphPlanError::TensorShapeMismatch {
                scope: scope.to_string(),
                slot,
                tensor_name: tensor.tensor_name.clone(),
                found_shape: tensor.dims.clone(),
                reason: "expected rank-2 linear projection tensor".to_string(),
            });
        }

        let lhs = usize::try_from(tensor.dims[0]).map_err(|_| {
            WhisperDecoderGraphPlanError::TensorShapeMismatch {
                scope: scope.to_string(),
                slot,
                tensor_name: tensor.tensor_name.clone(),
                found_shape: tensor.dims.clone(),
                reason: "dimension does not fit target usize".to_string(),
            }
        })?;
        let rhs = usize::try_from(tensor.dims[1]).map_err(|_| {
            WhisperDecoderGraphPlanError::TensorShapeMismatch {
                scope: scope.to_string(),
                slot,
                tensor_name: tensor.tensor_name.clone(),
                found_shape: tensor.dims.clone(),
                reason: "dimension does not fit target usize".to_string(),
            }
        })?;

        let (input_dim, output_dim, weight_layout) = if rhs == expected_input_dim {
            (rhs, lhs, WhisperDecoderLinearWeightLayout::OutputInput)
        } else if lhs == expected_input_dim {
            (lhs, rhs, WhisperDecoderLinearWeightLayout::InputOutput)
        } else {
            return Err(WhisperDecoderGraphPlanError::TensorShapeMismatch {
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
            return Err(WhisperDecoderGraphPlanError::TensorShapeMismatch {
                scope: scope.to_string(),
                slot,
                tensor_name: tensor.tensor_name.clone(),
                found_shape: tensor.dims.clone(),
                reason: format!(
                    "projection output_dim={output_dim} does not match expected {expected_output_dim}"
                ),
            });
        }

        Ok(WhisperDecoderLinearProjectionPlan {
            weight: tensor.clone(),
            weight_layout,
            input_dim,
            output_dim,
        })
    }

    fn parse_norm(
        &self,
        scope: &str,
        slot_prefix: &'static str,
        weight: Option<&WhisperDecoderGraphTensorRef>,
        bias: Option<&WhisperDecoderGraphTensorRef>,
        hidden: usize,
    ) -> Result<WhisperDecoderNormPlan, WhisperDecoderGraphPlanError> {
        let weight = weight.ok_or_else(|| WhisperDecoderGraphPlanError::MissingTensorBinding {
            scope: scope.to_string(),
            slot: slot_prefix,
        })?;
        let bias = bias.ok_or_else(|| WhisperDecoderGraphPlanError::MissingTensorBinding {
            scope: scope.to_string(),
            slot: slot_prefix,
        })?;
        validate_norm_shape(scope, slot_prefix, weight, hidden)?;
        validate_norm_shape(scope, slot_prefix, bias, hidden)?;
        Ok(WhisperDecoderNormPlan {
            weight: weight.clone(),
            bias: bias.clone(),
        })
    }
}

pub(crate) fn build_whisper_decoder_graph_plan(
    metadata: WhisperDecoderGraphMetadata,
    binding: &WhisperDecoderTensorBindingSeam,
    materialization: &WhisperDecoderTensorMaterializationSeam,
    input_shape: WhisperDecoderGraphInputShape,
) -> Result<WhisperDecoderGraphPlan, WhisperDecoderGraphPlanError> {
    WhisperDecoderGraphBuilder::new(metadata, binding, materialization, input_shape).build()
}

fn validate_norm_shape(
    scope: &str,
    slot: &'static str,
    tensor: &WhisperDecoderGraphTensorRef,
    hidden: usize,
) -> Result<(), WhisperDecoderGraphPlanError> {
    let hidden_u64 = hidden as u64;
    let ok = match tensor.dims.as_slice() {
        [dim] => *dim == hidden_u64,
        [_, last] => *last == hidden_u64,
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(WhisperDecoderGraphPlanError::TensorShapeMismatch {
            scope: scope.to_string(),
            slot,
            tensor_name: tensor.tensor_name.clone(),
            found_shape: tensor.dims.clone(),
            reason: format!("expected rank-1 [hidden] or rank-2 [*, hidden={hidden}]"),
        })
    }
}

fn validate_bias_shape(
    scope: &str,
    slot: &'static str,
    tensor: &WhisperDecoderGraphTensorRef,
    expected_dim: usize,
) -> Result<(), WhisperDecoderGraphPlanError> {
    let expected_u64 = expected_dim as u64;
    let ok = match tensor.dims.as_slice() {
        [dim] => *dim == expected_u64,
        [_, last] => *last == expected_u64,
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(WhisperDecoderGraphPlanError::TensorShapeMismatch {
            scope: scope.to_string(),
            slot,
            tensor_name: tensor.tensor_name.clone(),
            found_shape: tensor.dims.clone(),
            reason: format!("expected rank-1 [{expected_dim}] or rank-2 [*, {expected_dim}]"),
        })
    }
}

fn required_decoder_primitives() -> Vec<&'static str> {
    vec![
        "decoder.token_embedding",
        "decoder.positional_embedding",
        "decoder.self_attn.layer_norm",
        "decoder.self_attn.qkv_projection",
        "decoder.self_attn.causal_attention_softmax",
        "decoder.self_attn.out_projection",
        "decoder.cross_attn.layer_norm",
        "decoder.cross_attn.qkv_projection",
        "decoder.cross_attn.attention_softmax",
        "decoder.cross_attn.out_projection",
        "decoder.residual_add",
        "decoder.mlp.fc1",
        "decoder.mlp.gelu",
        "decoder.mlp.fc2",
        "decoder.final_layer_norm",
        "decoder.output_projection",
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WhisperDecoderHiddenStateLayout {
    SequenceHidden,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WhisperDecoderGraphExecutionInput {
    pub decoder_prefix_tokens: Vec<u32>,
    pub encoder_hidden_state: Vec<f32>,
    pub encoder_layout: WhisperDecoderHiddenStateLayout,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WhisperDecoderGraphExecutionOutput {
    pub logits: Vec<f32>,
    pub greedy_token: u32,
    pub prefix_len: usize,
    pub vocab_size: usize,
    pub last_token_cross_attention_frame_probs: Option<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WhisperDecoderBatchedStepOutput {
    pub logits: Vec<f32>,
    pub vocab_size: usize,
    pub n_seq: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WhisperDecoderGraphExecutionConfig {
    pub attention_heads: usize,
    pub use_self_flash_attention: bool,
    pub use_cross_flash_attention: bool,
    pub collect_cross_attention: bool,
    pub layer_norm_epsilon: f32,
}

impl Default for WhisperDecoderGraphExecutionConfig {
    fn default() -> Self {
        Self {
            attention_heads: 1,
            use_self_flash_attention: false,
            use_cross_flash_attention: false,
            collect_cross_attention: false,
            layer_norm_epsilon: 1.0e-5,
        }
    }
}

#[derive(Debug, Error)]
pub(crate) enum WhisperDecoderGraphExecutionError {
    #[error("whisper decoder execution input is invalid: {reason}")]
    InvalidInput { reason: String },
    #[error("whisper decoder execution is missing tensor '{tensor_name}': {reason}")]
    MissingMaterializedTensor { tensor_name: String, reason: String },
    #[error("whisper decoder graph tensor '{tensor_name}' materialization failed: {reason}")]
    TensorMaterializationFailed { tensor_name: String, reason: String },
    #[error("whisper decoder graph unsupported primitive '{primitive}': {reason}")]
    UnsupportedDecoderPrimitive {
        primitive: &'static str,
        reason: String,
    },
    #[error("whisper decoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
}

pub(crate) trait WhisperDecoderTensorSource {
    fn materialize_tensor_f32(
        &self,
        tensor: &WhisperDecoderGraphTensorRef,
    ) -> Result<Vec<f32>, WhisperDecoderGraphExecutionError>;

    fn materialize_tensor_f32_arc(
        &self,
        tensor: &WhisperDecoderGraphTensorRef,
    ) -> Result<Arc<[f32]>, WhisperDecoderGraphExecutionError> {
        self.materialize_tensor_f32(tensor)
            .map(|values| Arc::<[f32]>::from(values.into_boxed_slice()))
    }

    fn materialize_tensor_f16_bits(
        &self,
        _tensor: &WhisperDecoderGraphTensorRef,
    ) -> Result<Option<Vec<u16>>, WhisperDecoderGraphExecutionError> {
        Ok(None)
    }

    fn materialize_tensor_f16_bits_arc(
        &self,
        tensor: &WhisperDecoderGraphTensorRef,
    ) -> Result<Option<Arc<[u16]>>, WhisperDecoderGraphExecutionError> {
        self.materialize_tensor_f16_bits(tensor)
            .map(|values| values.map(|values| Arc::<[u16]>::from(values.into_boxed_slice())))
    }

    fn materialize_tensor_quantized(
        &self,
        _tensor: &WhisperDecoderGraphTensorRef,
    ) -> Result<Option<(i32, Vec<u8>)>, WhisperDecoderGraphExecutionError> {
        Ok(None)
    }

    fn materialize_tensor_quantized_arc(
        &self,
        tensor: &WhisperDecoderGraphTensorRef,
    ) -> Result<Option<(i32, Arc<[u8]>)>, WhisperDecoderGraphExecutionError> {
        self.materialize_tensor_quantized(tensor).map(|value| {
            value.map(|(ggml_type, bytes)| (ggml_type, Arc::<[u8]>::from(bytes.into_boxed_slice())))
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LocalLinearWeightCacheKey {
    tensor_name: String,
    input_dim: usize,
    output_dim: usize,
    source_layout: WhisperDecoderLinearWeightLayout,
}

fn linear_weight_cache_key(
    projection: &WhisperDecoderLinearProjectionPlan,
) -> LocalLinearWeightCacheKey {
    LocalLinearWeightCacheKey {
        tensor_name: projection.weight.tensor_name.clone(),
        input_dim: projection.input_dim,
        output_dim: projection.output_dim,
        source_layout: projection.weight_layout,
    }
}

#[derive(Debug, Clone)]
enum LocalLinearWeightPayload {
    F16Bits(Arc<[u16]>),
    Quantized { ggml_type: i32, bytes: Arc<[u8]> },
}

type LocalVectorPayload = Arc<[f32]>;

const PERSISTENT_CROSS_ATTENTION_LAYER_STRIDE_ALIGN: usize = 256;

fn persistent_cross_attention_layer_stride_frames(encoder_frames: usize) -> usize {
    if encoder_frames == 0 {
        0
    } else {
        encoder_frames.next_multiple_of(PERSISTENT_CROSS_ATTENTION_LAYER_STRIDE_ALIGN)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LocalVectorCacheKey {
    tensor_name: String,
    len: usize,
}

pub(crate) struct WhisperDecoderPersistentWeightCache {
    arena: GgmlStaticTensorArena,
    linear_weights: HashMap<LocalLinearWeightCacheKey, GgmlStaticTensor>,
    // Zero-copy linear weights bound directly to the mmap'd runtime pack (no host
    // copy, no arena upload). Only quantized input-output linear weights consumed
    // verbatim via `linear_weight()` are bound here; the cross-attention K/V
    // projections (referenced by the precompute task as arena tensors) and every
    // f32/f16 weight stay on the arena path. `_loaded_weights` owns the mmap + ggml
    // context the bound tensors point into and must outlive the graph.
    _loaded_weights: Option<GgmlLoadedWeightContext>,
    loaded_linear_weights: HashMap<LocalLinearWeightCacheKey, GgmlLoadedTensor>,
    vectors: HashMap<LocalVectorCacheKey, GgmlStaticTensor>,
    embeddings: WhisperDecoderPersistentEmbeddingCache,
    _cross_attention_storage: WhisperDecoderPersistentCrossAttentionStorage,
    cross_attention: Vec<WhisperDecoderPersistentCrossAttentionCache>,
    cross_attention_projection_tasks: Vec<PersistentCrossAttentionProjectionTask>,
    cross_attention_projection_tasks_by_slot: Vec<Vec<PersistentCrossAttentionProjectionTask>>,
    cross_attention_input_dim: usize,
    cross_attention_output_dim: usize,
    cross_attention_encoder_frames: usize,
    self_attention: WhisperDecoderPersistentSelfAttentionCache,
}

#[derive(Debug, Default)]
pub(crate) struct WhisperDecoderExecutionTensorCache {
    raw_tensors: HashMap<String, Arc<[f32]>>,
    linear_weights: HashMap<LocalLinearWeightCacheKey, LocalLinearWeightPayload>,
    cross_attention: HashMap<usize, WhisperDecoderCrossAttentionCache>,
}

enum PersistentTensorUpload {
    F16(GgmlStaticTensor, Arc<[u16]>, &'static str),
    F32(GgmlStaticTensor, Arc<[f32]>, &'static str),
    Bytes(GgmlStaticTensor, Arc<[u8]>, &'static str),
}

#[derive(Debug, Clone)]
struct PersistentCrossAttentionProjectionTask {
    layer_idx: usize,
    input_dim: usize,
    output_dim: usize,
    key_weight: GgmlStaticTensor,
    key_weight_accepts_f16_rhs: bool,
    value_weight: GgmlStaticTensor,
    value_weight_accepts_f16_rhs: bool,
    value_bias: GgmlStaticTensor,
    key_target: GgmlStaticTensor,
    value_target: GgmlStaticTensor,
}

#[derive(Debug, Clone)]
struct WhisperDecoderCrossAttentionCache {
    key: Arc<[u16]>,
    value: Arc<[u16]>,
}

#[derive(Debug, Clone, Copy)]
struct WhisperDecoderPersistentCrossAttentionStorage {
    _key: GgmlStaticTensor,
    _value: GgmlStaticTensor,
    _layer_stride_frames: usize,
    _n_seq: usize,
}

#[derive(Debug, Clone, Copy)]
struct WhisperDecoderPersistentCrossAttentionCache {
    key: GgmlStaticTensor,
    value: GgmlStaticTensor,
    layer_stride_frames: usize,
    n_seq: usize,
}

#[derive(Debug, Clone, Copy)]
struct WhisperDecoderPersistentSelfAttentionCache {
    key: GgmlStaticTensor,
    value: GgmlStaticTensor,
    max_positions: usize,
    layer_count: usize,
    hidden: usize,
    n_seq: usize,
}

#[derive(Debug, Clone, Copy)]
struct WhisperDecoderPersistentEmbeddingCache {
    token: GgmlStaticTensor,
    position: GgmlStaticTensor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperDecoderSelfKvCacheState {
    next_position: usize,
}

impl WhisperDecoderSelfKvCacheState {
    pub(crate) fn new() -> Self {
        Self { next_position: 0 }
    }

    pub(crate) fn next_position(&self) -> usize {
        self.next_position
    }

    pub(crate) fn advance(&mut self, token_count: usize) {
        self.next_position = self.next_position.saturating_add(token_count);
    }
}

impl WhisperDecoderExecutionTensorCache {
    fn materialize_tensor_f32(
        &mut self,
        source: &dyn WhisperDecoderTensorSource,
        tensor: &WhisperDecoderGraphTensorRef,
    ) -> Result<Arc<[f32]>, WhisperDecoderGraphExecutionError> {
        if let Some(values) = self.raw_tensors.get(&tensor.tensor_name) {
            return Ok(Arc::clone(values));
        }
        let values = source.materialize_tensor_f32_arc(tensor)?;
        self.raw_tensors
            .insert(tensor.tensor_name.clone(), Arc::clone(&values));
        Ok(values)
    }

    fn materialize_linear_weight_input_output(
        &mut self,
        source: &dyn WhisperDecoderTensorSource,
        projection: &WhisperDecoderLinearProjectionPlan,
    ) -> Result<LocalLinearWeightPayload, WhisperDecoderGraphExecutionError> {
        let local_key = LocalLinearWeightCacheKey {
            tensor_name: projection.weight.tensor_name.clone(),
            input_dim: projection.input_dim,
            output_dim: projection.output_dim,
            source_layout: projection.weight_layout,
        };
        if let Some(values) = self.linear_weights.get(&local_key) {
            return Ok(values.clone());
        }

        let expected_len = projection
            .input_dim
            .checked_mul(projection.output_dim)
            .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
                reason: format!(
                    "{} projection dimensions overflow: {}x{}",
                    projection.weight.tensor_name, projection.input_dim, projection.output_dim
                ),
            })?;

        let materialized = if let Some((ggml_type, bytes)) =
            source.materialize_tensor_quantized_arc(&projection.weight)?
        {
            if projection.weight_layout != WhisperDecoderLinearWeightLayout::InputOutput
                && projection.input_dim != projection.output_dim
            {
                return Err(
                    WhisperDecoderGraphExecutionError::TensorMaterializationFailed {
                        tensor_name: projection.weight.tensor_name.clone(),
                        reason: "quantized decoder weight must be input-output layout".to_string(),
                    },
                );
            }
            LocalLinearWeightPayload::Quantized { ggml_type, bytes }
        } else if let Some(weights) = source.materialize_tensor_f16_bits_arc(&projection.weight)? {
            if weights.len() != expected_len {
                return Err(
                    WhisperDecoderGraphExecutionError::TensorMaterializationFailed {
                        tensor_name: projection.weight.tensor_name.clone(),
                        reason: format!(
                            "tensor has {} f16 elements but projection expects {}",
                            weights.len(),
                            expected_len
                        ),
                    },
                );
            }
            let values = match projection.weight_layout {
                WhisperDecoderLinearWeightLayout::InputOutput => weights,
                WhisperDecoderLinearWeightLayout::OutputInput => Arc::<[u16]>::from(
                    transpose_weight_output_input_to_input_output(
                        &weights,
                        projection.input_dim,
                        projection.output_dim,
                    )?
                    .into_boxed_slice(),
                ),
            };
            LocalLinearWeightPayload::F16Bits(values)
        } else {
            let weights = self.materialize_tensor_f32(source, &projection.weight)?;
            if weights.len() != expected_len {
                return Err(
                    WhisperDecoderGraphExecutionError::TensorMaterializationFailed {
                        tensor_name: projection.weight.tensor_name.clone(),
                        reason: format!(
                            "tensor has {} elements but projection expects {}",
                            weights.len(),
                            expected_len
                        ),
                    },
                );
            }

            let values = match projection.weight_layout {
                WhisperDecoderLinearWeightLayout::InputOutput => Arc::clone(&weights),
                WhisperDecoderLinearWeightLayout::OutputInput => Arc::<[f32]>::from(
                    transpose_weight_output_input_to_input_output(
                        &weights,
                        projection.input_dim,
                        projection.output_dim,
                    )?
                    .into_boxed_slice(),
                ),
            };
            if values.iter().any(|value| !value.is_finite()) {
                return Err(
                    WhisperDecoderGraphExecutionError::TensorMaterializationFailed {
                        tensor_name: projection.weight.tensor_name.clone(),
                        reason: "tensor contains non-finite values".to_string(),
                    },
                );
            }
            LocalLinearWeightPayload::F16Bits(Arc::<[u16]>::from(
                values
                    .iter()
                    .copied()
                    .map(f32_to_f16_bits)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ))
        };

        self.linear_weights.insert(local_key, materialized.clone());
        Ok(materialized)
    }

    fn materialize_vector(
        &mut self,
        source: &dyn WhisperDecoderTensorSource,
        tensor: &WhisperDecoderGraphTensorRef,
        len: usize,
    ) -> Result<LocalVectorPayload, WhisperDecoderGraphExecutionError> {
        let values = materialize_hidden_vector(self, source, tensor, len)?;
        Ok(values)
    }

    fn materialize_embedding_hidden_vocab(
        &mut self,
        source: &dyn WhisperDecoderTensorSource,
        embedding: &WhisperDecoderEmbeddingPlan,
    ) -> Result<Arc<[f32]>, WhisperDecoderGraphExecutionError> {
        let values = self.materialize_tensor_f32(source, &embedding.weight)?;
        let expected = embedding
            .hidden_size
            .checked_mul(embedding.vocab_size)
            .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
                reason: format!(
                    "{} embedding shape overflows usize",
                    embedding.weight.tensor_name
                ),
            })?;
        if values.len() != expected {
            return Err(
                WhisperDecoderGraphExecutionError::TensorMaterializationFailed {
                    tensor_name: embedding.weight.tensor_name.clone(),
                    reason: format!(
                        "embedding has {} values but expected {}",
                        values.len(),
                        expected
                    ),
                },
            );
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(
                WhisperDecoderGraphExecutionError::TensorMaterializationFailed {
                    tensor_name: embedding.weight.tensor_name.clone(),
                    reason: "embedding contains non-finite values".to_string(),
                },
            );
        }
        let hidden_vocab = match embedding.layout {
            WhisperDecoderEmbeddingLayout::HiddenVocab => Arc::clone(&values),
            WhisperDecoderEmbeddingLayout::VocabHidden => {
                let mut transposed = vec![0.0_f32; expected];
                for vocab_idx in 0..embedding.vocab_size {
                    for hidden_idx in 0..embedding.hidden_size {
                        transposed[hidden_idx + vocab_idx * embedding.hidden_size] =
                            values[vocab_idx * embedding.hidden_size + hidden_idx];
                    }
                }
                Arc::<[f32]>::from(transposed.into_boxed_slice())
            }
        };
        Ok(hidden_vocab)
    }

    fn materialize_embedding_hidden_vocab_f16_bits(
        &mut self,
        source: &dyn WhisperDecoderTensorSource,
        embedding: &WhisperDecoderEmbeddingPlan,
    ) -> Result<Arc<[u16]>, WhisperDecoderGraphExecutionError> {
        let expected = embedding
            .hidden_size
            .checked_mul(embedding.vocab_size)
            .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
                reason: format!(
                    "{} embedding shape overflows usize",
                    embedding.weight.tensor_name
                ),
            })?;
        if let Some(values) = source.materialize_tensor_f16_bits_arc(&embedding.weight)? {
            if values.len() != expected {
                return Err(
                    WhisperDecoderGraphExecutionError::TensorMaterializationFailed {
                        tensor_name: embedding.weight.tensor_name.clone(),
                        reason: format!(
                            "embedding has {} f16 values but expected {}",
                            values.len(),
                            expected
                        ),
                    },
                );
            }
            let hidden_vocab = match embedding.layout {
                WhisperDecoderEmbeddingLayout::HiddenVocab => values,
                WhisperDecoderEmbeddingLayout::VocabHidden => {
                    let mut transposed = vec![0_u16; expected];
                    for vocab_idx in 0..embedding.vocab_size {
                        for hidden_idx in 0..embedding.hidden_size {
                            transposed[hidden_idx + vocab_idx * embedding.hidden_size] =
                                values[vocab_idx * embedding.hidden_size + hidden_idx];
                        }
                    }
                    Arc::<[u16]>::from(transposed.into_boxed_slice())
                }
            };
            return Ok(hidden_vocab);
        }

        let values = self.materialize_embedding_hidden_vocab(source, embedding)?;
        Ok(Arc::<[u16]>::from(
            values
                .iter()
                .copied()
                .map(f32_to_f16_bits)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ))
    }

    fn materialize_cross_attention_cache(
        &mut self,
        source: &dyn WhisperDecoderTensorSource,
        layer: &WhisperDecoderLayerPlan,
        encoder_hidden: Arc<[f32]>,
        hidden: usize,
        encoder_frames: usize,
        cache_misses: &mut usize,
    ) -> Result<WhisperDecoderCrossAttentionCache, WhisperDecoderGraphExecutionError> {
        if let Some(cache) = self.cross_attention.get(&layer.layer_idx) {
            return Ok(cache.clone());
        }
        *cache_misses = cache_misses.saturating_add(1);

        let key = materialize_linear_projection_output_ggml(
            self,
            source,
            Arc::clone(&encoder_hidden),
            encoder_frames,
            &layer.cross_attn_k,
            None,
            "decoder_cross_attn_k_cache",
        )?;
        let value = materialize_linear_projection_output_ggml(
            self,
            source,
            encoder_hidden,
            encoder_frames,
            &layer.cross_attn_v.projection,
            Some(&layer.cross_attn_v.bias),
            "decoder_cross_attn_v_cache",
        )?;

        let expected = hidden.checked_mul(encoder_frames).ok_or_else(|| {
            WhisperDecoderGraphExecutionError::InvalidInput {
                reason: "cross-attention cache shape overflows usize".to_string(),
            }
        })?;
        if key.len() != expected || value.len() != expected {
            return Err(WhisperDecoderGraphExecutionError::GraphExecutionFailed {
                reason: format!(
                    "cross-attention cache shape mismatch for layer {}: key={} value={} expected={}",
                    layer.layer_idx,
                    key.len(),
                    value.len(),
                    expected
                ),
            });
        }

        let cache = WhisperDecoderCrossAttentionCache {
            key: Arc::<[u16]>::from(
                key.iter()
                    .copied()
                    .map(f32_to_f16_bits)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ),
            value: Arc::<[u16]>::from(
                value
                    .iter()
                    .copied()
                    .map(f32_to_f16_bits)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ),
        };
        self.cross_attention.insert(layer.layer_idx, cache.clone());
        Ok(cache)
    }
}

impl WhisperDecoderPersistentWeightCache {
    fn validate_cross_attention_stage_plan(
        &self,
        plan: &WhisperDecoderGraphPlan,
    ) -> Result<(), WhisperDecoderGraphExecutionError> {
        let hidden = self.cross_attention_input_dim;
        let encoder_frames = self.cross_attention_encoder_frames;
        if hidden != plan.input_shape.hidden_size
            || encoder_frames != plan.input_shape.encoder_frames
            || self.cross_attention_projection_tasks.len() != plan.layers.len()
            || self.cross_attention.len() != plan.layers.len()
        {
            return Err(WhisperDecoderGraphExecutionError::InvalidInput {
                reason: "decoder persistent cross-attention stage received incompatible plan shape"
                    .to_string(),
            });
        }
        for (expected_layer_idx, layer) in plan.layers.iter().enumerate() {
            if layer.layer_idx != expected_layer_idx {
                return Err(WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: format!(
                        "decoder persistent cross-attention requires contiguous layer indices at populate stage; expected layer_idx={expected_layer_idx}, got {}",
                        layer.layer_idx
                    ),
                });
            }
        }
        Ok(())
    }

    pub(crate) fn prepare_cross_attention_stage<'a>(
        &self,
        runner: &'a mut GgmlCpuGraphRunner,
        plan: &WhisperDecoderGraphPlan,
    ) -> Result<PreparedCrossCachePopulateStage<'a>, WhisperDecoderGraphExecutionError> {
        self.validate_cross_attention_stage_plan(plan)?;
        if self.cross_attention_projection_tasks.is_empty() {
            return Err(WhisperDecoderGraphExecutionError::InvalidInput {
                reason: "decoder persistent cross-attention prepare requires non-empty layers"
                    .to_string(),
            });
        }
        let tasks = self.cross_attention_projection_tasks_for_slot(0)?;
        prepare_cross_attention_projection_pairs_with_persistent_weights_runner_ggml(
            runner,
            &self.arena,
            self.cross_attention_encoder_frames,
            self.cross_attention_input_dim,
            self.cross_attention_output_dim,
            &tasks,
            "decoder_persistent_cross_attn_cache",
        )
    }

    fn cross_attention_projection_tasks_for_slot(
        &self,
        slot_index: usize,
    ) -> Result<Vec<PersistentCrossAttentionProjectionTask>, WhisperDecoderGraphExecutionError>
    {
        self.cross_attention_projection_tasks_by_slot
            .get(slot_index)
            .cloned()
            .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
                reason: format!(
                    "decoder persistent cross-attention slot {slot_index} exceeds n_seq {}",
                    self.cross_attention_projection_tasks_by_slot.len()
                ),
            })
    }

    pub(crate) fn build_static_stage(
        runner: &mut GgmlCpuGraphRunner,
        plan: &WhisperDecoderGraphPlan,
        source: &dyn WhisperDecoderTensorSource,
        tensor_cache: &mut WhisperDecoderExecutionTensorCache,
        self_kv_max_positions: usize,
        runtime_path: Option<&Path>,
    ) -> Result<Self, WhisperDecoderGraphExecutionError> {
        Self::build_static_stage_with_n_seq(
            runner,
            plan,
            source,
            tensor_cache,
            self_kv_max_positions,
            runtime_path,
            1,
        )
    }

    pub(crate) fn build_static_stage_with_n_seq(
        runner: &mut GgmlCpuGraphRunner,
        plan: &WhisperDecoderGraphPlan,
        source: &dyn WhisperDecoderTensorSource,
        tensor_cache: &mut WhisperDecoderExecutionTensorCache,
        self_kv_max_positions: usize,
        runtime_path: Option<&Path>,
        n_seq: usize,
    ) -> Result<Self, WhisperDecoderGraphExecutionError> {
        if n_seq == 0 {
            return Err(WhisperDecoderGraphExecutionError::InvalidInput {
                reason: "whisper decoder persistent cache n_seq must be positive".to_string(),
            });
        }
        let persistent_build_start = Instant::now();
        let mut arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::default().context_bytes)
            .map_err(
                |error| WhisperDecoderGraphExecutionError::GraphExecutionFailed {
                    reason: format!(
                        "could not initialize decoder persistent weight arena: {error}"
                    ),
                },
            )?;
        let arena_init_ms = persistent_build_start.elapsed().as_millis();
        // Bind eligible quantized linear weights zero-copy to the mmap'd pack. The
        // cross-attention K/V projections are EXCLUDED: their static arena tensors
        // are referenced directly by the cross-attention precompute task, so they
        // must remain arena-resident.
        let loaded_weights =
            runtime_path.and_then(|path| runner.load_gguf_weight_context(path).ok());
        let cross_kv_excluded_keys: HashSet<LocalLinearWeightCacheKey> = plan
            .layers
            .iter()
            .flat_map(|layer| {
                [
                    linear_weight_cache_key(&layer.cross_attn_k),
                    linear_weight_cache_key(&layer.cross_attn_v.projection),
                ]
            })
            .collect();
        let mut linear_weights = HashMap::new();
        let mut loaded_linear_weights: HashMap<LocalLinearWeightCacheKey, GgmlLoadedTensor> =
            HashMap::new();
        let mut linear_weight_types = HashMap::new();
        let mut vectors = HashMap::new();
        let mut uploads = Vec::new();
        let embeddings_start = Instant::now();
        let token_embedding_name = "decoder_persistent_token_embedding";
        let token_embedding_values = tensor_cache
            .materialize_embedding_hidden_vocab_f16_bits(source, &plan.token_embedding)?;
        let token_embedding = arena
            .new_tensor_2d_f16(
                plan.token_embedding.hidden_size,
                plan.token_embedding.vocab_size,
                token_embedding_name,
            )
            .map_err(|error| {
                map_decoder_execute_graph_error(
                    "ggml_new_tensor_2d_f16(persistent_token_embedding)",
                    error,
                )
            })?;
        uploads.push(PersistentTensorUpload::F16(
            token_embedding,
            token_embedding_values,
            token_embedding_name,
        ));
        let position_embedding_name = "decoder_persistent_position_embedding";
        let position_embedding_values =
            tensor_cache.materialize_embedding_hidden_vocab(source, &plan.position_embedding)?;
        let position_embedding = arena
            .new_tensor_2d_f32(
                plan.position_embedding.hidden_size,
                plan.position_embedding.vocab_size,
                position_embedding_name,
            )
            .map_err(|error| {
                map_decoder_execute_graph_error(
                    "ggml_new_tensor_2d_f32(persistent_position_embedding)",
                    error,
                )
            })?;
        uploads.push(PersistentTensorUpload::F32(
            position_embedding,
            position_embedding_values,
            position_embedding_name,
        ));
        let embeddings_ms = embeddings_start.elapsed().as_millis();
        let self_kv_alloc_start = Instant::now();
        let self_kv_len = plan
            .input_shape
            .hidden_size
            .checked_mul(self_kv_max_positions)
            .and_then(|value| value.checked_mul(n_seq))
            .and_then(|value| value.checked_mul(plan.layers.len()))
            .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
                reason: "persistent self-attention KV cache shape overflows usize".to_string(),
            })?;
        let self_key_name = "decoder_persistent_self_attn_k_cache";
        let self_value_name = "decoder_persistent_self_attn_v_cache";
        let self_key = arena
            .new_tensor_4d_f16(
                plan.input_shape.hidden_size,
                self_kv_max_positions,
                n_seq,
                plan.layers.len(),
                self_key_name,
            )
            .map_err(|error| {
                map_decoder_execute_graph_error(
                    "ggml_new_tensor_4d_f16(persistent_self_k_cache)",
                    error,
                )
            })?;
        let self_value = arena
            .new_tensor_4d_f16(
                plan.input_shape.hidden_size,
                self_kv_max_positions,
                n_seq,
                plan.layers.len(),
                self_value_name,
            )
            .map_err(|error| {
                map_decoder_execute_graph_error(
                    "ggml_new_tensor_4d_f16(persistent_self_v_cache)",
                    error,
                )
            })?;
        let self_zero = Arc::<[u16]>::from(vec![0_u16; self_kv_len].into_boxed_slice());
        uploads.push(PersistentTensorUpload::F16(
            self_key,
            Arc::clone(&self_zero),
            self_key_name,
        ));
        uploads.push(PersistentTensorUpload::F16(
            self_value,
            self_zero,
            self_value_name,
        ));
        let self_kv_alloc_ms = self_kv_alloc_start.elapsed().as_millis();
        let linear_weights_start = Instant::now();
        for projection in decoder_linear_projection_plans(plan) {
            let key = LocalLinearWeightCacheKey {
                tensor_name: projection.weight.tensor_name.clone(),
                input_dim: projection.input_dim,
                output_dim: projection.output_dim,
                source_layout: projection.weight_layout,
            };
            if linear_weights.contains_key(&key) || loaded_linear_weights.contains_key(&key) {
                continue;
            }
            // Zero-copy bind: a quantized input-output linear weight that is NOT a
            // cross-attention K/V projection is consumed verbatim via mul_mat, so
            // the mmap'd pack bytes are bit-identical to the arena copy. Bind it
            // directly and skip the arena allocation + upload.
            if projection.weight_layout == WhisperDecoderLinearWeightLayout::InputOutput
                && !cross_kv_excluded_keys.contains(&key)
                && let Some((ggml_type, _bytes)) =
                    source.materialize_tensor_quantized_arc(&projection.weight)?
                && let Some(loaded) = loaded_weights
                    .as_ref()
                    .and_then(|ctx| ctx.tensor(&projection.weight.tensor_name))
            {
                linear_weight_types.insert(key.clone(), ggml_type);
                loaded_linear_weights.insert(key, loaded);
                continue;
            }
            let weights =
                tensor_cache.materialize_linear_weight_input_output(source, projection)?;
            let tensor_name: &'static str = "decoder_persistent_linear_weight";
            let tensor = arena
                .new_matmul_weight_2d_typed(
                    projection.input_dim,
                    projection.output_dim,
                    match &weights {
                        LocalLinearWeightPayload::F16Bits(_) => GGML_TYPE_F16,
                        LocalLinearWeightPayload::Quantized { ggml_type, .. } => *ggml_type,
                    },
                    tensor_name,
                )
                .map_err(|error| {
                    map_decoder_execute_graph_error(
                        "ggml_new_tensor_2d_typed(persistent_linear_weight)",
                        error,
                    )
                })?;
            match weights {
                LocalLinearWeightPayload::F16Bits(values) => {
                    linear_weight_types.insert(key.clone(), GGML_TYPE_F16);
                    linear_weights.insert(key, tensor);
                    uploads.push(PersistentTensorUpload::F16(tensor, values, tensor_name))
                }
                LocalLinearWeightPayload::Quantized { ggml_type, bytes } => {
                    linear_weight_types.insert(key.clone(), ggml_type);
                    linear_weights.insert(key, tensor);
                    uploads.push(PersistentTensorUpload::Bytes(tensor, bytes, tensor_name))
                }
            }
        }
        let linear_weights_ms = linear_weights_start.elapsed().as_millis();
        let vectors_start = Instant::now();
        for (tensor, len) in decoder_vector_tensor_plans(plan) {
            let key = LocalVectorCacheKey {
                tensor_name: tensor.tensor_name.clone(),
                len,
            };
            if vectors.contains_key(&key) {
                continue;
            }
            let values = tensor_cache.materialize_vector(source, tensor, len)?;
            let tensor_name: &'static str = "decoder_persistent_vector";
            let static_tensor = arena.new_tensor_1d_f32(len, tensor_name).map_err(|error| {
                map_decoder_execute_graph_error("ggml_new_tensor_1d_f32(persistent_vector)", error)
            })?;
            vectors.insert(key, static_tensor);
            uploads.push(PersistentTensorUpload::F32(
                static_tensor,
                values,
                tensor_name,
            ));
        }
        let vectors_ms = vectors_start.elapsed().as_millis();
        let hidden = plan.input_shape.hidden_size;
        let encoder_frames = plan.input_shape.encoder_frames;
        let cross_attention_layer_stride_frames =
            persistent_cross_attention_layer_stride_frames(encoder_frames);
        let mut cross_attention = Vec::with_capacity(plan.layers.len());
        let mut cross_attention_projection_tasks = Vec::with_capacity(plan.layers.len());
        let mut cross_attention_projection_tasks_by_slot =
            vec![Vec::with_capacity(plan.layers.len()); n_seq];
        let key_storage_name: &'static str = "decoder_persistent_cross_attn_k_storage";
        let value_storage_name: &'static str = "decoder_persistent_cross_attn_v_storage";
        let key_storage = arena
            .new_tensor_4d_f16(
                hidden,
                cross_attention_layer_stride_frames,
                n_seq,
                plan.layers.len(),
                key_storage_name,
            )
            .map_err(|error| {
                map_decoder_execute_graph_error(
                    "ggml_new_tensor_4d_f16(persistent_cross_k_storage)",
                    error,
                )
            })?;
        let value_storage = arena
            .new_tensor_4d_f16(
                hidden,
                cross_attention_layer_stride_frames,
                n_seq,
                plan.layers.len(),
                value_storage_name,
            )
            .map_err(|error| {
                map_decoder_execute_graph_error(
                    "ggml_new_tensor_4d_f16(persistent_cross_v_storage)",
                    error,
                )
            })?;
        let cross_attention_layer_stride_bytes = hidden
            .checked_mul(cross_attention_layer_stride_frames)
            .and_then(|value| value.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
                reason: "persistent cross-attention layer stride overflows usize".to_string(),
            })?;
        let cross_attention_row_stride_bytes = hidden
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
                reason: "persistent cross-attention row stride overflows usize".to_string(),
            })?;
        for (expected_layer_idx, layer) in plan.layers.iter().enumerate() {
            if layer.layer_idx != expected_layer_idx {
                return Err(WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: format!(
                        "decoder persistent cross-attention requires contiguous layer indices; expected layer_idx={expected_layer_idx}, got {}",
                        layer.layer_idx
                    ),
                });
            }
            let key_name: &'static str = "decoder_persistent_cross_attn_k_cache";
            let value_name: &'static str = "decoder_persistent_cross_attn_v_cache";
            let layer_offset = expected_layer_idx
                .checked_mul(cross_attention_layer_stride_bytes)
                .and_then(|value| value.checked_mul(n_seq))
                .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: "persistent cross-attention layer offset overflows usize".to_string(),
                })?;
            let layer_columns = if n_seq == 1 {
                encoder_frames
            } else {
                cross_attention_layer_stride_frames
                    .checked_mul(n_seq)
                    .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
                        reason: "persistent cross-attention layer view overflows usize".to_string(),
                    })?
            };
            let key_tensor = arena
                .view_2d(
                    key_storage,
                    hidden,
                    layer_columns,
                    cross_attention_row_stride_bytes,
                    layer_offset,
                    key_name,
                )
                .map_err(|error| {
                    map_decoder_execute_graph_error("ggml_view_2d(persistent_cross_k_cache)", error)
                })?;
            let value_tensor = arena
                .view_2d(
                    value_storage,
                    hidden,
                    layer_columns,
                    cross_attention_row_stride_bytes,
                    layer_offset,
                    value_name,
                )
                .map_err(|error| {
                    map_decoder_execute_graph_error("ggml_view_2d(persistent_cross_v_cache)", error)
                })?;
            let key_weight = persistent_linear_weight_handle(
                &linear_weights,
                &layer.cross_attn_k,
                "decoder_persistent_cross_attn_cache",
            )?;
            let key_weight_ggml_type = persistent_linear_weight_type_handle(
                &linear_weight_types,
                &layer.cross_attn_k,
                "decoder_persistent_cross_attn_cache",
            )?;
            let value_weight = persistent_linear_weight_handle(
                &linear_weights,
                &layer.cross_attn_v.projection,
                "decoder_persistent_cross_attn_cache",
            )?;
            let value_weight_ggml_type = persistent_linear_weight_type_handle(
                &linear_weight_types,
                &layer.cross_attn_v.projection,
                "decoder_persistent_cross_attn_cache",
            )?;
            let value_bias = persistent_vector_handle(
                &vectors,
                &layer.cross_attn_v.bias,
                layer.cross_attn_v.projection.output_dim,
                "decoder_persistent_cross_attn_cache",
            )?;
            cross_attention.push(WhisperDecoderPersistentCrossAttentionCache {
                key: key_tensor,
                value: value_tensor,
                layer_stride_frames: cross_attention_layer_stride_frames,
                n_seq,
            });
            for (slot_index, slot_tasks) in cross_attention_projection_tasks_by_slot
                .iter_mut()
                .enumerate()
            {
                let slot_offset = if n_seq == 1 {
                    0
                } else {
                    cross_attention_layer_stride_frames
                        .checked_mul(hidden)
                        .and_then(|value| value.checked_mul(std::mem::size_of::<u16>()))
                        .and_then(|value| value.checked_mul(slot_index))
                        .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
                            reason: "persistent cross-attention slot offset overflows usize"
                                .to_string(),
                        })?
                };
                let key_target = if n_seq == 1 {
                    key_tensor
                } else {
                    arena
                        .view_2d(
                            key_tensor,
                            hidden,
                            encoder_frames,
                            cross_attention_row_stride_bytes,
                            slot_offset,
                            key_name,
                        )
                        .map_err(|error| {
                            map_decoder_execute_graph_error(
                                "ggml_view_2d(persistent_cross_k_slot)",
                                error,
                            )
                        })?
                };
                let value_target = if n_seq == 1 {
                    value_tensor
                } else {
                    arena
                        .view_2d(
                            value_tensor,
                            hidden,
                            encoder_frames,
                            cross_attention_row_stride_bytes,
                            slot_offset,
                            value_name,
                        )
                        .map_err(|error| {
                            map_decoder_execute_graph_error(
                                "ggml_view_2d(persistent_cross_v_slot)",
                                error,
                            )
                        })?
                };
                let task = PersistentCrossAttentionProjectionTask {
                    layer_idx: layer.layer_idx,
                    input_dim: layer.cross_attn_k.input_dim,
                    output_dim: layer.cross_attn_k.output_dim,
                    key_weight,
                    key_weight_accepts_f16_rhs: key_weight_ggml_type == GGML_TYPE_F16,
                    value_weight,
                    value_weight_accepts_f16_rhs: value_weight_ggml_type == GGML_TYPE_F16,
                    value_bias,
                    key_target,
                    value_target,
                };
                if slot_index == 0 {
                    cross_attention_projection_tasks.push(task.clone());
                }
                slot_tasks.push(task);
            }
        }
        arena.allocate_backend_buffer().map_err(|error| {
            map_decoder_execute_graph_error("decoder_persistent_arena_alloc", error)
        })?;
        let upload_count = uploads.len();
        let upload_bytes: usize = uploads
            .iter()
            .map(|upload| match upload {
                PersistentTensorUpload::F16(_, values, _) => values.len() * 2,
                PersistentTensorUpload::F32(_, values, _) => values.len() * 4,
                PersistentTensorUpload::Bytes(_, values, _) => values.len(),
            })
            .sum();
        let upload_start = Instant::now();
        for upload in uploads {
            match upload {
                PersistentTensorUpload::F16(tensor, values, tensor_name) => arena
                    .set_f16_bits_slice(tensor, values.as_ref(), tensor_name)
                    .map_err(|error| {
                        map_decoder_execute_graph_error(
                            "ggml_backend_tensor_set(persistent_f16)",
                            error,
                        )
                    })?,
                PersistentTensorUpload::F32(tensor, values, tensor_name) => arena
                    .set_f32_slice(tensor, values.as_ref(), tensor_name)
                    .map_err(|error| {
                        map_decoder_execute_graph_error(
                            "ggml_backend_tensor_set(persistent_f32)",
                            error,
                        )
                    })?,
                PersistentTensorUpload::Bytes(tensor, values, tensor_name) => arena
                    .set_bytes_slice(tensor, values.as_ref(), tensor_name)
                    .map_err(|error| {
                        map_decoder_execute_graph_error(
                            "ggml_backend_tensor_set(persistent_bytes)",
                            error,
                        )
                    })?,
            }
        }
        let upload_ms = upload_start.elapsed().as_millis();
        let cross_attention_ms = 0;
        emit_decoder_persistent_cache_detail_trace(
            arena_init_ms,
            embeddings_ms,
            self_kv_alloc_ms,
            linear_weights_ms,
            vectors_ms,
            cross_attention_ms,
            upload_count,
            upload_bytes,
            upload_ms,
            persistent_build_start.elapsed().as_millis(),
        );
        Ok(Self {
            arena,
            linear_weights,
            _loaded_weights: loaded_weights,
            loaded_linear_weights,
            vectors,
            embeddings: WhisperDecoderPersistentEmbeddingCache {
                token: token_embedding,
                position: position_embedding,
            },
            _cross_attention_storage: WhisperDecoderPersistentCrossAttentionStorage {
                _key: key_storage,
                _value: value_storage,
                _layer_stride_frames: cross_attention_layer_stride_frames,
                _n_seq: n_seq,
            },
            cross_attention,
            cross_attention_projection_tasks,
            cross_attention_projection_tasks_by_slot,
            cross_attention_input_dim: hidden,
            cross_attention_output_dim: hidden,
            cross_attention_encoder_frames: encoder_frames,
            self_attention: WhisperDecoderPersistentSelfAttentionCache {
                key: self_key,
                value: self_value,
                max_positions: self_kv_max_positions,
                layer_count: plan.layers.len(),
                hidden: plan.input_shape.hidden_size,
                n_seq,
            },
        })
    }

    pub(crate) fn populate_cross_attention_stage(
        &self,
        runner: &mut GgmlCpuGraphRunner,
        plan: &WhisperDecoderGraphPlan,
        encoder_hidden_state: &[f32],
        encoder_layout: WhisperDecoderHiddenStateLayout,
    ) -> Result<(), WhisperDecoderGraphExecutionError> {
        self.populate_cross_attention_stage_slot(
            runner,
            plan,
            encoder_hidden_state,
            encoder_layout,
            0,
        )
    }

    pub(crate) fn populate_cross_attention_stage_slot(
        &self,
        runner: &mut GgmlCpuGraphRunner,
        plan: &WhisperDecoderGraphPlan,
        encoder_hidden_state: &[f32],
        encoder_layout: WhisperDecoderHiddenStateLayout,
        slot_index: usize,
    ) -> Result<(), WhisperDecoderGraphExecutionError> {
        if self.cross_attention_projection_tasks.is_empty() {
            return Ok(());
        }
        let populate_start = Instant::now();
        self.validate_cross_attention_stage_plan(plan)?;
        let hidden = self.cross_attention_input_dim;
        let encoder_frames = self.cross_attention_encoder_frames;
        let encoder_hidden =
            normalize_hidden_layout(encoder_hidden_state, encoder_layout, encoder_frames, hidden);
        let expected_cross_len = hidden.checked_mul(encoder_frames).ok_or_else(|| {
            WhisperDecoderGraphExecutionError::InvalidInput {
                reason: "persistent cross-attention cache shape overflows usize".to_string(),
            }
        })?;
        let input_bytes = encoder_hidden
            .as_ref()
            .len()
            .saturating_mul(std::mem::size_of::<f32>());
        let tasks = self.cross_attention_projection_tasks_for_slot(slot_index)?;
        let task_count = tasks.len();
        let populate_perf =
            populate_cross_attention_projection_pairs_with_persistent_weights_runner_ggml(
                runner,
                &self.arena,
                encoder_hidden.as_ref(),
                encoder_frames,
                self.cross_attention_input_dim,
                self.cross_attention_output_dim,
                &tasks,
                "decoder_persistent_cross_attn_cache",
            )?;
        emit_decoder_persistent_cross_cache_populate_trace(
            task_count,
            input_bytes,
            expected_cross_len,
            populate_perf.graph_build_ms,
            populate_perf.upload_ms,
            populate_perf.compute_ms,
            populate_start.elapsed().as_millis(),
        );
        Ok(())
    }

    pub(crate) fn populate_cross_attention_stage_with_prepared(
        &self,
        prepared: PreparedCrossCachePopulateStage<'_>,
        plan: &WhisperDecoderGraphPlan,
        encoder_hidden_state: &[f32],
        encoder_layout: WhisperDecoderHiddenStateLayout,
    ) -> Result<(), WhisperDecoderGraphExecutionError> {
        let populate_start = Instant::now();
        self.validate_cross_attention_stage_plan(plan)?;
        let hidden = self.cross_attention_input_dim;
        let encoder_frames = self.cross_attention_encoder_frames;
        let encoder_hidden =
            normalize_hidden_layout(encoder_hidden_state, encoder_layout, encoder_frames, hidden);
        let expected_cross_len = hidden.checked_mul(encoder_frames).ok_or_else(|| {
            WhisperDecoderGraphExecutionError::InvalidInput {
                reason: "persistent cross-attention cache shape overflows usize".to_string(),
            }
        })?;
        let input_bytes = encoder_hidden
            .as_ref()
            .len()
            .saturating_mul(std::mem::size_of::<f32>());
        let task_count = self.cross_attention_projection_tasks.len();
        let populate_perf = prepared.execute(encoder_hidden.as_ref())?;
        emit_decoder_persistent_cross_cache_populate_trace(
            task_count,
            input_bytes,
            expected_cross_len,
            populate_perf.graph_build_ms,
            populate_perf.upload_ms,
            populate_perf.compute_ms,
            populate_start.elapsed().as_millis(),
        );
        Ok(())
    }

    pub(crate) fn supports_cross_attention_for_plan(&self, plan: &WhisperDecoderGraphPlan) -> bool {
        plan.layers
            .iter()
            .all(|layer| self.has_cross_attention(layer.layer_idx))
    }

    fn linear_weight<'a>(
        &self,
        projection: &WhisperDecoderLinearProjectionPlan,
    ) -> Option<GgmlCpuTensor<'a>> {
        let key = LocalLinearWeightCacheKey {
            tensor_name: projection.weight.tensor_name.clone(),
            input_dim: projection.input_dim,
            output_dim: projection.output_dim,
            source_layout: projection.weight_layout,
        };
        if let Some(loaded) = self.loaded_linear_weights.get(&key) {
            return Some(loaded.as_graph_tensor());
        }
        self.linear_weights
            .get(&key)
            .map(|tensor| self.arena.graph_tensor(*tensor))
    }

    fn vector<'a>(
        &self,
        tensor: &WhisperDecoderGraphTensorRef,
        len: usize,
    ) -> Option<GgmlCpuTensor<'a>> {
        let key = LocalVectorCacheKey {
            tensor_name: tensor.tensor_name.clone(),
            len,
        };
        self.vectors
            .get(&key)
            .map(|tensor| self.arena.graph_tensor(*tensor))
    }

    fn token_embedding<'a>(&self) -> GgmlCpuTensor<'a> {
        self.arena.graph_tensor(self.embeddings.token)
    }

    fn position_embedding<'a>(&self) -> GgmlCpuTensor<'a> {
        self.arena.graph_tensor(self.embeddings.position)
    }

    fn has_cross_attention(&self, layer_idx: usize) -> bool {
        layer_idx < self.cross_attention.len()
    }

    fn self_attention_cache(&self) -> WhisperDecoderPersistentSelfAttentionCache {
        self.self_attention
    }

    fn self_attention_key<'a>(&self) -> GgmlCpuTensor<'a> {
        self.arena.graph_tensor(self.self_attention.key)
    }

    fn self_attention_value<'a>(&self) -> GgmlCpuTensor<'a> {
        self.arena.graph_tensor(self.self_attention.value)
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_decoder_persistent_cache_detail_trace(
    arena_init_ms: u128,
    embeddings_ms: u128,
    self_kv_alloc_ms: u128,
    linear_weights_ms: u128,
    vectors_ms: u128,
    cross_attention_ms: u128,
    upload_count: usize,
    upload_bytes: usize,
    upload_ms: u128,
    total_ms: u128,
) {
    if std::env::var_os("OPENASR_WHISPER_GGML_TRACE").is_none() {
        return;
    }
    eprintln!(
        "openasr_whisper_ggml_trace stage=decoder_persistent_cache event=detail status=ok arena_init_ms={arena_init_ms} embeddings_ms={embeddings_ms} self_kv_alloc_ms={self_kv_alloc_ms} linear_weights_ms={linear_weights_ms} vectors_ms={vectors_ms} cross_attention_ms={cross_attention_ms} upload_count={upload_count} upload_bytes={upload_bytes} upload_ms={upload_ms} total_ms={total_ms}"
    );
}

fn emit_decoder_persistent_cross_cache_populate_trace(
    task_count: usize,
    input_bytes: usize,
    cross_cache_len_per_layer: usize,
    graph_build_ms: u128,
    upload_ms: u128,
    compute_ms: u128,
    total_ms: u128,
) {
    if std::env::var_os("OPENASR_WHISPER_GGML_TRACE").is_none() {
        return;
    }
    eprintln!(
        "openasr_whisper_ggml_trace stage=decoder_persistent_cache event=populate_detail status=ok task_count={task_count} input_bytes={input_bytes} cross_cache_len_per_layer={cross_cache_len_per_layer} graph_build_ms={graph_build_ms} upload_ms={upload_ms} compute_ms={compute_ms} total_ms={total_ms}"
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CrossCachePopulatePerfStats {
    graph_build_ms: u128,
    upload_ms: u128,
    compute_ms: u128,
}

#[derive(Debug, Clone, Copy)]
enum PreparedCrossCacheInputUploadTensor<'a> {
    F16(GgmlCpuTensor<'a>),
    F32(GgmlCpuTensor<'a>),
}

pub(crate) struct PreparedCrossCachePopulateStage<'a> {
    graph: GgmlCpuGraphBuilder<'a>,
    input_tensor: PreparedCrossCacheInputUploadTensor<'a>,
    expected_input: usize,
    input_upload_is_f16: bool,
    label_prefix: &'static str,
    graph_build_ms: u128,
}

impl<'a> PreparedCrossCachePopulateStage<'a> {
    pub(crate) fn execute(
        mut self,
        input_values: &[f32],
    ) -> Result<CrossCachePopulatePerfStats, WhisperDecoderGraphExecutionError> {
        if input_values.len() != self.expected_input {
            return Err(WhisperDecoderGraphExecutionError::InvalidInput {
                reason: format!(
                    "{} input has {} elements but expected {}",
                    self.label_prefix,
                    input_values.len(),
                    self.expected_input
                ),
            });
        }
        let upload_start = Instant::now();
        match self.input_tensor {
            PreparedCrossCacheInputUploadTensor::F16(tensor) => {
                let input_f16_bits = Arc::<[u16]>::from(
                    input_values
                        .iter()
                        .copied()
                        .map(f32_to_f16_bits)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                );
                upload_decoder_tensor(
                    &mut self.graph,
                    tensor,
                    DecoderUploadData::F16Bits(input_f16_bits),
                    "cross_cache_input_f16",
                    Some(self.label_prefix),
                )?;
            }
            PreparedCrossCacheInputUploadTensor::F32(tensor) => {
                upload_decoder_tensor(
                    &mut self.graph,
                    tensor,
                    DecoderUploadData::F32Borrowed(input_values),
                    "cross_cache_input",
                    Some(self.label_prefix),
                )?;
            }
        }
        let upload_ms = upload_start.elapsed().as_millis();
        let compute_start = Instant::now();
        self.graph.compute_side_effects().map_err(|error| {
            WhisperDecoderGraphExecutionError::GraphExecutionFailed {
                reason: format!(
                    "{} cache projection write failed: {error}",
                    self.label_prefix
                ),
            }
        })?;
        let compute_ms = compute_start.elapsed().as_millis();
        if std::env::var_os("OPENASR_WHISPER_GGML_TRACE").is_some() {
            eprintln!(
                "openasr_whisper_ggml_trace stage=decoder_persistent_cache event=populate_upload status=ok input_upload_f16={} input_bytes={}",
                usize::from(self.input_upload_is_f16),
                if self.input_upload_is_f16 {
                    self.expected_input
                        .saturating_mul(std::mem::size_of::<u16>())
                } else {
                    self.expected_input
                        .saturating_mul(std::mem::size_of::<f32>())
                }
            );
        }
        Ok(CrossCachePopulatePerfStats {
            graph_build_ms: self.graph_build_ms,
            upload_ms,
            compute_ms,
        })
    }
}

fn decoder_linear_projection_plans(
    plan: &WhisperDecoderGraphPlan,
) -> Vec<&WhisperDecoderLinearProjectionPlan> {
    let mut projections = Vec::with_capacity(plan.layers.len().saturating_mul(10) + 1);
    for layer in &plan.layers {
        projections.push(&layer.self_attn_q.projection);
        projections.push(&layer.self_attn_k);
        projections.push(&layer.self_attn_v.projection);
        projections.push(&layer.self_attn_out.projection);
        projections.push(&layer.cross_attn_q.projection);
        projections.push(&layer.cross_attn_k);
        projections.push(&layer.cross_attn_v.projection);
        projections.push(&layer.cross_attn_out.projection);
        projections.push(&layer.mlp_fc1.projection);
        projections.push(&layer.mlp_fc2.projection);
    }
    projections.push(&plan.output_projection.projection);
    projections
}

fn decoder_vector_tensor_plans(
    plan: &WhisperDecoderGraphPlan,
) -> Vec<(&WhisperDecoderGraphTensorRef, usize)> {
    let mut tensors = Vec::with_capacity(plan.layers.len().saturating_mul(18) + 3);
    let hidden = plan.input_shape.hidden_size;
    for layer in &plan.layers {
        tensors.push((&layer.self_attn_norm.weight, hidden));
        tensors.push((&layer.self_attn_norm.bias, hidden));
        tensors.push((
            &layer.self_attn_q.bias,
            layer.self_attn_q.projection.output_dim,
        ));
        tensors.push((
            &layer.self_attn_v.bias,
            layer.self_attn_v.projection.output_dim,
        ));
        tensors.push((
            &layer.self_attn_out.bias,
            layer.self_attn_out.projection.output_dim,
        ));
        tensors.push((&layer.cross_attn_norm.weight, hidden));
        tensors.push((&layer.cross_attn_norm.bias, hidden));
        tensors.push((
            &layer.cross_attn_q.bias,
            layer.cross_attn_q.projection.output_dim,
        ));
        tensors.push((
            &layer.cross_attn_v.bias,
            layer.cross_attn_v.projection.output_dim,
        ));
        tensors.push((
            &layer.cross_attn_out.bias,
            layer.cross_attn_out.projection.output_dim,
        ));
        tensors.push((&layer.mlp_norm.weight, hidden));
        tensors.push((&layer.mlp_norm.bias, hidden));
        tensors.push((&layer.mlp_fc1.bias, layer.mlp_fc1.projection.output_dim));
        tensors.push((&layer.mlp_fc2.bias, layer.mlp_fc2.projection.output_dim));
    }
    tensors.push((&plan.final_norm.weight, hidden));
    tensors.push((&plan.final_norm.bias, hidden));
    if let Some(bias) = &plan.output_projection.bias {
        tensors.push((bias, plan.output_projection.vocab_size));
    }
    tensors
}

#[cfg(test)]
pub(crate) fn run_whisper_decoder_greedy_step_ggml_v0(
    plan: &WhisperDecoderGraphPlan,
    input: &WhisperDecoderGraphExecutionInput,
    source: &dyn WhisperDecoderTensorSource,
    config: WhisperDecoderGraphExecutionConfig,
) -> Result<WhisperDecoderGraphExecutionOutput, WhisperDecoderGraphExecutionError> {
    let mut tensor_cache = WhisperDecoderExecutionTensorCache::default();
    run_whisper_decoder_greedy_step_with_cache_ggml_v0(
        plan,
        input,
        source,
        config,
        &mut tensor_cache,
    )
}

#[cfg(test)]
pub(crate) fn run_whisper_decoder_greedy_step_with_cache_ggml_v0(
    plan: &WhisperDecoderGraphPlan,
    input: &WhisperDecoderGraphExecutionInput,
    source: &dyn WhisperDecoderTensorSource,
    config: WhisperDecoderGraphExecutionConfig,
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
) -> Result<WhisperDecoderGraphExecutionOutput, WhisperDecoderGraphExecutionError> {
    let mut runner = GgmlCpuGraphRunner::new(whisper_decoder_graph_config()).map_err(|error| {
        WhisperDecoderGraphExecutionError::GraphExecutionFailed {
            reason: format!("could not initialize ggml cpu graph runner: {error}"),
        }
    })?;
    run_whisper_decoder_greedy_step_with_cache_and_runner_ggml_v0(
        &mut runner,
        None,
        None,
        0,
        plan,
        input,
        source,
        config,
        tensor_cache,
    )
}

pub(crate) fn run_whisper_decoder_greedy_step_with_cache_and_runner_ggml_v0(
    runner: &mut GgmlCpuGraphRunner,
    persistent_weights: Option<&WhisperDecoderPersistentWeightCache>,
    self_kv_state: Option<&WhisperDecoderSelfKvCacheState>,
    position_offset: usize,
    plan: &WhisperDecoderGraphPlan,
    input: &WhisperDecoderGraphExecutionInput,
    source: &dyn WhisperDecoderTensorSource,
    config: WhisperDecoderGraphExecutionConfig,
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
) -> Result<WhisperDecoderGraphExecutionOutput, WhisperDecoderGraphExecutionError> {
    execute_whisper_decoder_with_position_offset_ggml_v0(
        runner,
        persistent_weights,
        self_kv_state,
        plan,
        "decoder_prefix_tokens",
        &input.decoder_prefix_tokens,
        position_offset,
        &input.encoder_hidden_state,
        input.encoder_layout,
        source,
        config,
        tensor_cache,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_whisper_decoder_with_position_offset_ggml_v0(
    runner: &mut GgmlCpuGraphRunner,
    persistent_weights: Option<&WhisperDecoderPersistentWeightCache>,
    self_kv_state: Option<&WhisperDecoderSelfKvCacheState>,
    plan: &WhisperDecoderGraphPlan,
    token_label: &'static str,
    decoder_tokens: &[u32],
    position_offset: usize,
    encoder_hidden_state: &[f32],
    encoder_layout: WhisperDecoderHiddenStateLayout,
    source: &dyn WhisperDecoderTensorSource,
    config: WhisperDecoderGraphExecutionConfig,
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
) -> Result<WhisperDecoderGraphExecutionOutput, WhisperDecoderGraphExecutionError> {
    let graph_total_start = Instant::now();
    validate_decoder_execution_config(plan, config)?;
    validate_decoder_tokens(plan, token_label, decoder_tokens, position_offset)?;
    let prefix_len = decoder_tokens.len();
    let hidden = plan.input_shape.hidden_size;
    let encoder_frames = plan.input_shape.encoder_frames;
    let needs_encoder_hidden = decoder_step_needs_encoder_hidden(plan, persistent_weights);
    let encoder_hidden = if needs_encoder_hidden {
        validate_encoder_hidden_input(plan, encoder_hidden_state)?;
        Some(Arc::<[f32]>::from(
            normalize_hidden_layout(encoder_hidden_state, encoder_layout, encoder_frames, hidden)
                .into_owned()
                .into_boxed_slice(),
        ))
    } else {
        None
    };

    let graph_build_start = Instant::now();
    let mut graph = runner.start_graph();
    let mut uploads = Vec::new();
    let mut single_token_id_i32: Option<[i32; 1]> = None;
    let mut single_position_id_i32: Option<[i32; 1]> = None;
    let mut self_kv_row_indices = None;

    let state_input = if let Some(persistent_weights) = persistent_weights {
        let token_ids_owned = if prefix_len == 1 {
            let token = *decoder_tokens.first().ok_or_else(|| {
                WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: "decoder token list unexpectedly empty".to_string(),
                }
            })?;
            single_token_id_i32 = Some([i32::try_from(token).map_err(|_| {
                WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: format!("token id {token} exceeds ggml i32 input range"),
                }
            })?]);
            None
        } else {
            Some(
                decoder_tokens
                    .iter()
                    .copied()
                    .map(|token| {
                        i32::try_from(token).map_err(|_| {
                            WhisperDecoderGraphExecutionError::InvalidInput {
                                reason: format!("token id {token} exceeds ggml i32 input range"),
                            }
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        };
        let position_ids_owned = if prefix_len == 1 {
            single_position_id_i32 = Some([i32::try_from(position_offset).map_err(|_| {
                WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: "decoder position id exceeds ggml i32 input range".to_string(),
                }
            })?]);
            None
        } else {
            Some(
                (0..prefix_len)
                    .map(|relative_position| {
                        position_offset
                            .checked_add(relative_position)
                            .and_then(|position| i32::try_from(position).ok())
                            .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
                                reason: "decoder position id exceeds ggml i32 input range"
                                    .to_string(),
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        };
        let token_ids_tensor = graph
            .new_tensor_1d_i32(prefix_len, "decoder_token_ids")
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_new_tensor_1d_i32(token_ids)", error)
            })?;
        graph
            .set_input(token_ids_tensor)
            .map_err(|error| map_decoder_execute_graph_error("ggml_set_input(token_ids)", error))?;
        if let Some(token_ids) = token_ids_owned {
            uploads.push((
                token_ids_tensor,
                DecoderUploadData::I32(Arc::<[i32]>::from(token_ids.into_boxed_slice())),
                "decoder_token_ids",
            ));
        } else {
            let token_ids = single_token_id_i32
                .as_ref()
                .expect("single-token decoder id must be prepared");
            uploads.push((
                token_ids_tensor,
                DecoderUploadData::I32Borrowed(token_ids.as_slice()),
                "decoder_token_ids",
            ));
        }
        let position_ids_tensor = graph
            .new_tensor_1d_i32(prefix_len, "decoder_position_ids")
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_new_tensor_1d_i32(position_ids)", error)
            })?;
        graph.set_input(position_ids_tensor).map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_input(position_ids)", error)
        })?;
        if prefix_len == 1 {
            self_kv_row_indices = Some(position_ids_tensor);
        }
        if let Some(position_ids) = position_ids_owned {
            uploads.push((
                position_ids_tensor,
                DecoderUploadData::I32(Arc::<[i32]>::from(position_ids.into_boxed_slice())),
                "decoder_position_ids",
            ));
        } else {
            let position_ids = single_position_id_i32
                .as_ref()
                .expect("single-position decoder id must be prepared");
            uploads.push((
                position_ids_tensor,
                DecoderUploadData::I32Borrowed(position_ids.as_slice()),
                "decoder_position_ids",
            ));
        }
        let token_state = graph
            .get_rows(persistent_weights.token_embedding(), token_ids_tensor)
            .map_err(|error| map_decoder_execute_graph_error("ggml_get_rows(token)", error))?;
        let position_state = graph
            .get_rows(persistent_weights.position_embedding(), position_ids_tensor)
            .map_err(|error| map_decoder_execute_graph_error("ggml_get_rows(position)", error))?;
        graph.add(token_state, position_state).map_err(|error| {
            map_decoder_execute_graph_error("ggml_add(decoder_embedding)", error)
        })?
    } else {
        let token_hidden = if position_offset == 0 {
            materialize_decoder_embeddings(
                tensor_cache,
                source,
                &plan.token_embedding,
                &plan.position_embedding,
                decoder_tokens,
            )?
        } else {
            materialize_decoder_embeddings_with_position_offset(
                tensor_cache,
                source,
                &plan.token_embedding,
                &plan.position_embedding,
                decoder_tokens,
                position_offset,
            )?
        };
        let state_input = graph
            .new_tensor_2d_f32(hidden, prefix_len, "decoder_state_input")
            .map_err(|error| map_decoder_execute_graph_error("ggml_new_tensor_2d(state)", error))?;
        graph
            .set_input(state_input)
            .map_err(|error| map_decoder_execute_graph_error("ggml_set_input(state)", error))?;
        uploads.push((
            state_input,
            DecoderUploadData::F32(Arc::<[f32]>::from(token_hidden.into_boxed_slice())),
            "decoder_state_input",
        ));
        state_input
    };

    let shared_self_attention_mask = if prefix_len == 1 {
        None
    } else {
        let n_kv = position_offset.checked_add(prefix_len).ok_or_else(|| {
            WhisperDecoderGraphExecutionError::InvalidInput {
                reason: "decoder self-attention shared mask KV length overflows usize".to_string(),
            }
        })?;
        let mask_tensor = graph
            .new_tensor_3d_f16(n_kv, prefix_len, 1, "decoder_self_kq_mask")
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_new_tensor_3d(self_kq_mask)", error)
            })?;
        graph.set_input(mask_tensor).map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_input(self_kq_mask)", error)
        })?;
        uploads.push((
            mask_tensor,
            DecoderUploadData::F16Bits(build_decoder_self_attention_causal_mask_f16_bits(
                n_kv,
                prefix_len,
                position_offset,
            )?),
            "decoder_self_kq_mask",
        ));
        Some(mask_tensor)
    };

    let mut state = state_input;
    let mut last_token_cross_attention_tensor = None;
    let mut cross_cache_misses = 0usize;
    state = seq2seq_indexed_layer_stack(
        &mut graph,
        state,
        &plan.layers,
        |graph, state, position, layer| {
            let state = apply_decoder_self_attention(
                graph,
                &mut uploads,
                tensor_cache,
                persistent_weights,
                self_kv_state,
                source,
                state,
                layer,
                hidden,
                prefix_len,
                position_offset,
                None,
                shared_self_attention_mask,
                self_kv_row_indices,
                config.attention_heads,
                config.layer_norm_epsilon,
                config.use_self_flash_attention,
                1,
            )?;
            let cross_cache = if persistent_weights
                .is_some_and(|weights| weights.has_cross_attention(layer.layer_idx))
            {
                None
            } else {
                let encoder_hidden = encoder_hidden.as_ref().ok_or_else(|| {
                    WhisperDecoderGraphExecutionError::GraphExecutionFailed {
                        reason:
                            "decoder cross-attention cache requested without encoder hidden input"
                                .to_string(),
                    }
                })?;
                Some(tensor_cache.materialize_cross_attention_cache(
                    source,
                    layer,
                    Arc::clone(encoder_hidden),
                    hidden,
                    encoder_frames,
                    &mut cross_cache_misses,
                )?)
            };
            let cross_attention = apply_decoder_cross_attention(
                graph,
                &mut uploads,
                tensor_cache,
                persistent_weights,
                source,
                state,
                cross_cache.as_ref(),
                layer,
                hidden,
                prefix_len,
                encoder_frames,
                config.attention_heads,
                config.layer_norm_epsilon,
                config.use_cross_flash_attention,
                config.collect_cross_attention && position.is_last,
                1,
            )?;
            if position.is_last {
                last_token_cross_attention_tensor = cross_attention.last_token_frame_probs;
            }
            apply_decoder_mlp(
                graph,
                &mut uploads,
                tensor_cache,
                persistent_weights,
                source,
                cross_attention.state,
                layer,
                config.layer_norm_epsilon,
            )
        },
    )?;

    let last_token_state = view_last_token_state(&mut graph, state, hidden, prefix_len)?;
    let last_token_state = apply_affine_layer_norm(
        &mut graph,
        &mut uploads,
        tensor_cache,
        persistent_weights,
        source,
        last_token_state,
        config.layer_norm_epsilon,
        &plan.final_norm,
        "decoder_final_norm",
    )?;
    let last_token_logits_tensor = apply_linear_with_optional_bias(
        &mut graph,
        &mut uploads,
        tensor_cache,
        persistent_weights,
        source,
        last_token_state,
        &plan.output_projection.projection,
        plan.output_projection.bias.as_ref(),
        "decoder_output_projection",
    )?;

    graph
        .set_output(last_token_logits_tensor)
        .map_err(|error| map_decoder_execute_graph_error("ggml_set_output(logits)", error))?;
    let graph_output_tensor = last_token_logits_tensor;
    if let Some(attention_tensor) = last_token_cross_attention_tensor {
        graph.set_output(attention_tensor).map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_output(cross_attention)", error)
        })?;
    }
    let graph_build_ms = graph_build_start.elapsed().as_millis();

    let upload_start = Instant::now();
    let upload_count = uploads.len();
    let upload_bytes: usize = uploads.iter().map(|(_, values, _)| values.byte_len()).sum();
    let mut output_tensors = vec![graph_output_tensor];
    if let Some(attention_tensor) = last_token_cross_attention_tensor {
        output_tensors.push(attention_tensor);
    }
    graph
        .prepare_outputs_for_upload(&output_tensors)
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_backend_sched_alloc_graph(decoder_logits)", error)
        })?;
    for (tensor, values, label) in uploads {
        upload_decoder_tensor(&mut graph, tensor, values, label, None)?;
    }
    let upload_ms = upload_start.elapsed().as_millis();

    let compute_start = Instant::now();
    let (logits, last_token_cross_attention_frame_probs) =
        if let Some(attention_tensor) = last_token_cross_attention_tensor {
            let attention_len = encoder_frames
                .checked_mul(prefix_len)
                .and_then(|value| value.checked_mul(config.attention_heads))
                .ok_or_else(|| WhisperDecoderGraphExecutionError::GraphExecutionFailed {
                    reason: "decoder cross-attention output shape overflowed".to_string(),
                })?;
            let mut outputs = graph
                .compute_outputs_f32(&[
                    (last_token_logits_tensor, plan.output_projection.vocab_size),
                    (attention_tensor, attention_len),
                ])
                .map_err(
                    |error| WhisperDecoderGraphExecutionError::GraphExecutionFailed {
                        reason: format!("decoder graph compute failed: {error}"),
                    },
                )?;
            let attention = outputs.pop().ok_or_else(|| {
                WhisperDecoderGraphExecutionError::GraphExecutionFailed {
                    reason: "decoder graph did not return cross-attention output".to_string(),
                }
            })?;
            let logits = outputs.pop().ok_or_else(|| {
                WhisperDecoderGraphExecutionError::GraphExecutionFailed {
                    reason: "decoder graph did not return logits output".to_string(),
                }
            })?;
            let frame_probs = average_last_token_cross_attention_frame_probs(
                &attention,
                encoder_frames,
                prefix_len,
                config.attention_heads,
            )?;
            (logits, Some(frame_probs))
        } else {
            let logits = graph
                .compute_output_f32(last_token_logits_tensor, plan.output_projection.vocab_size)
                .map_err(
                    |error| WhisperDecoderGraphExecutionError::GraphExecutionFailed {
                        reason: format!("decoder graph compute failed: {error}"),
                    },
                )?;
            (logits, None)
        };
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(WhisperDecoderGraphExecutionError::GraphExecutionFailed {
            reason: "decoder logits contain non-finite values".to_string(),
        });
    }
    let greedy_token = argmax_finite(&logits)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| WhisperDecoderGraphExecutionError::GraphExecutionFailed {
            reason: "could not select greedy token from decoder logits".to_string(),
        })?;
    let output = WhisperDecoderGraphExecutionOutput {
        logits,
        greedy_token,
        prefix_len,
        vocab_size: plan.output_projection.vocab_size,
        last_token_cross_attention_frame_probs,
    };
    let compute_ms = compute_start.elapsed().as_millis();

    emit_decoder_graph_detail_trace(
        token_label,
        prefix_len,
        upload_count,
        upload_bytes,
        cross_cache_misses,
        graph_build_ms,
        upload_ms,
        compute_ms,
        graph_total_start.elapsed().as_millis(),
    );

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_whisper_decoder_reused_incremental_step_ggml_v0(
    reuse: &mut Option<Seq2SeqReusableDecodeGraph>,
    runner: &mut GgmlCpuGraphRunner,
    persistent_weights: &WhisperDecoderPersistentWeightCache,
    self_kv_state: &WhisperDecoderSelfKvCacheState,
    position_offset: usize,
    plan: &WhisperDecoderGraphPlan,
    token_id: u32,
    source: &dyn WhisperDecoderTensorSource,
    config: WhisperDecoderGraphExecutionConfig,
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
) -> Result<WhisperDecoderGraphExecutionOutput, WhisperDecoderGraphExecutionError> {
    let graph_total_start = Instant::now();
    validate_decoder_execution_config(plan, config)?;
    validate_decoder_tokens(plan, "decoder_reuse_token", &[token_id], position_offset)?;
    if runner.uses_scheduler() {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper reusable decode graph requires scheduler-off execution".to_string(),
        });
    }
    if config.collect_cross_attention {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper reusable decode graph does not collect cross-attention outputs"
                .to_string(),
        });
    }
    if !persistent_weights.supports_cross_attention_for_plan(plan) {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper reusable decode graph requires resident cross-attention cache"
                .to_string(),
        });
    }
    if self_kv_state.next_position() != position_offset {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "decoder self KV state mismatch: next_position={} position_offset={position_offset}",
                self_kv_state.next_position()
            ),
        });
    }

    let max_positions = persistent_weights.self_attention_cache().max_positions;
    if position_offset >= max_positions {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "decoder position {position_offset} exceeds reusable KV cache size {max_positions}"
            ),
        });
    }

    let build_start = Instant::now();
    let needs_build = reuse
        .as_ref()
        .map(|reuse| reuse.max_positions != max_positions || reuse.n_seq != 1)
        .unwrap_or(true);
    if needs_build {
        *reuse = Some(build_whisper_decoder_reusable_incremental_graph(
            runner,
            persistent_weights,
            plan,
            source,
            config,
            tensor_cache,
            max_positions,
        )?);
    }
    let graph_build_ms = if needs_build {
        build_start.elapsed().as_millis()
    } else {
        0
    };

    let token_i32 =
        i32::try_from(token_id).map_err(|_| WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!("token id {token_id} exceeds ggml i32 input range"),
        })?;
    let position_i32 = i32::try_from(position_offset).map_err(|_| {
        WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "decoder position id exceeds ggml i32 input range".to_string(),
        }
    })?;
    let total_tokens = position_offset.checked_add(1).ok_or_else(|| {
        WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "decoder reusable total token count overflows usize".to_string(),
        }
    })?;

    let reuse = reuse
        .as_mut()
        .expect("whisper reusable decode graph built above");
    let token_tensor = reuse.token_id;
    let row_index = reuse.row_index;
    let position_tensor = reuse.position;
    let attention_mask = reuse.attention_mask;
    let logits_tensor = reuse.logits;
    let graph_max_positions = reuse.max_positions;
    let upload_start = Instant::now();
    let graph = reuse.builder();
    graph
        .set_i32_slice(token_tensor, &[token_i32], "whisper_reuse_token")
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_i32_slice(reuse_token)", error)
        })?;
    graph
        .set_i32_slice(row_index, &[position_i32], "whisper_reuse_row")
        .map_err(|error| map_decoder_execute_graph_error("ggml_set_i32_slice(reuse_row)", error))?;
    graph
        .set_i32_slice(position_tensor, &[position_i32], "whisper_reuse_position")
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_i32_slice(reuse_position)", error)
        })?;
    let mask_bits = build_fixed_kv_attention_mask_bits(graph_max_positions, total_tokens)
        .map_err(|error| map_decoder_execute_graph_error("whisper_reuse_self_mask", error))?;
    graph
        .set_f16_bits_slice(attention_mask, &mask_bits, "whisper_reuse_self_mask")
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_f16_bits_slice(reuse_mask)", error)
        })?;
    let upload_ms = upload_start.elapsed().as_millis();

    let compute_start = Instant::now();
    let logits = graph
        .compute_output_f32(logits_tensor, plan.output_projection.vocab_size)
        .map_err(
            |error| WhisperDecoderGraphExecutionError::GraphExecutionFailed {
                reason: format!("decoder reusable graph compute failed: {error}"),
            },
        )?;
    let compute_ms = compute_start.elapsed().as_millis();
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(WhisperDecoderGraphExecutionError::GraphExecutionFailed {
            reason: "decoder reusable logits contain non-finite values".to_string(),
        });
    }
    let greedy_token = argmax_finite(&logits)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| WhisperDecoderGraphExecutionError::GraphExecutionFailed {
            reason: "could not select greedy token from reusable decoder logits".to_string(),
        })?;

    emit_decoder_graph_detail_trace(
        "decoder_reuse_token",
        1,
        4,
        (3 * std::mem::size_of::<i32>()) + (mask_bits.len() * std::mem::size_of::<u16>()),
        0,
        graph_build_ms,
        upload_ms,
        compute_ms,
        graph_total_start.elapsed().as_millis(),
    );

    Ok(WhisperDecoderGraphExecutionOutput {
        logits,
        greedy_token,
        prefix_len: 1,
        vocab_size: plan.output_projection.vocab_size,
        last_token_cross_attention_frame_probs: None,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_whisper_decoder_reused_batched_incremental_step_ggml_v0(
    reuse: &mut Option<Seq2SeqReusableDecodeGraph>,
    runner: &mut GgmlCpuGraphRunner,
    persistent_weights: &WhisperDecoderPersistentWeightCache,
    plan: &WhisperDecoderGraphPlan,
    token_ids: &[u32],
    positions: &[usize],
    total_tokens_by_sequence: &[usize],
    source: &dyn WhisperDecoderTensorSource,
    config: WhisperDecoderGraphExecutionConfig,
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
) -> Result<WhisperDecoderBatchedStepOutput, WhisperDecoderGraphExecutionError> {
    let graph_total_start = Instant::now();
    validate_decoder_execution_config(plan, config)?;
    if runner.uses_scheduler() {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper batched reusable decode graph requires scheduler-off execution"
                .to_string(),
        });
    }
    if config.collect_cross_attention {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason:
                "whisper batched reusable decode graph does not collect cross-attention outputs"
                    .to_string(),
        });
    }
    let n_seq = token_ids.len();
    if n_seq <= 1 {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper batched reusable decode graph requires at least two sequences"
                .to_string(),
        });
    }
    if positions.len() != n_seq || total_tokens_by_sequence.len() != n_seq {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "whisper batched decode shape mismatch: tokens={} positions={} totals={}",
                token_ids.len(),
                positions.len(),
                total_tokens_by_sequence.len()
            ),
        });
    }
    if !persistent_weights.supports_cross_attention_for_plan(plan) {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper batched reusable decode graph requires resident cross-attention cache"
                .to_string(),
        });
    }

    let self_cache = persistent_weights.self_attention_cache();
    let max_positions = self_cache.max_positions;
    if self_cache.n_seq != n_seq {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "whisper batched self-KV cache n_seq mismatch: cache={} batch={n_seq}",
                self_cache.n_seq
            ),
        });
    }
    if persistent_weights
        .cross_attention
        .iter()
        .any(|cache| cache.n_seq != n_seq)
    {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper batched cross-attention cache n_seq mismatch".to_string(),
        });
    }
    if positions.iter().any(|&position| position >= max_positions) {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!("whisper batched positions must be < {max_positions}"),
        });
    }
    if total_tokens_by_sequence
        .iter()
        .any(|&total_tokens| total_tokens == 0 || total_tokens > max_positions)
    {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!("whisper batched total token count must be in 1..={max_positions}"),
        });
    }
    if positions
        .iter()
        .zip(total_tokens_by_sequence.iter())
        .any(|(&position, &total_tokens)| position >= total_tokens)
    {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper batched positions must be visible in the fixed KV mask".to_string(),
        });
    }
    if positions
        .iter()
        .any(|&position| position >= plan.position_embedding.vocab_size)
    {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper batched position id exceeds position embedding size".to_string(),
        });
    }
    if token_ids.iter().any(|token| {
        usize::try_from(*token)
            .ok()
            .map(|token| token >= plan.output_projection.vocab_size)
            .unwrap_or(true)
    }) {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper batched token ids contain out-of-vocabulary token".to_string(),
        });
    }

    let build_start = Instant::now();
    let needs_build = reuse
        .as_ref()
        .map(|reuse| reuse.max_positions != max_positions || reuse.n_seq != n_seq)
        .unwrap_or(true);
    if needs_build {
        *reuse = Some(build_whisper_decoder_reusable_incremental_graph_with_n_seq(
            runner,
            persistent_weights,
            plan,
            source,
            config,
            tensor_cache,
            max_positions,
            n_seq,
        )?);
    }
    let graph_build_ms = if needs_build {
        build_start.elapsed().as_millis()
    } else {
        0
    };

    let token_ids_i32 = token_ids
        .iter()
        .map(|&token_id| {
            i32::try_from(token_id).map_err(|_| WhisperDecoderGraphExecutionError::InvalidInput {
                reason: format!("token id {token_id} exceeds ggml i32 input range"),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let positions_i32 = positions
        .iter()
        .map(|&position| {
            i32::try_from(position).map_err(|_| WhisperDecoderGraphExecutionError::InvalidInput {
                reason: "decoder position id exceeds ggml i32 input range".to_string(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let reuse = reuse
        .as_mut()
        .expect("whisper batched reusable decode graph built above");
    let token_tensor = reuse.token_id;
    let row_index = reuse.row_index;
    let position_tensor = reuse.position;
    let attention_mask = reuse.attention_mask;
    let logits_tensor = reuse.logits;
    let graph_max_positions = reuse.max_positions;
    let upload_start = Instant::now();
    let graph = reuse.builder();
    graph
        .set_i32_slice(token_tensor, &token_ids_i32, "whisper_reuse_batch_token")
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_i32_slice(reuse_batch_token)", error)
        })?;
    graph
        .set_i32_slice(row_index, &positions_i32, "whisper_reuse_batch_row")
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_i32_slice(reuse_batch_row)", error)
        })?;
    graph
        .set_i32_slice(
            position_tensor,
            &positions_i32,
            "whisper_reuse_batch_position",
        )
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_i32_slice(reuse_batch_position)", error)
        })?;
    let mask_bits = build_fixed_kv_attention_mask_bits_for_sequences(
        graph_max_positions,
        total_tokens_by_sequence,
    )
    .map_err(|error| map_decoder_execute_graph_error("whisper_reuse_batch_self_mask", error))?;
    graph
        .set_f16_bits_slice(attention_mask, &mask_bits, "whisper_reuse_batch_self_mask")
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_f16_bits_slice(reuse_batch_mask)", error)
        })?;
    let upload_ms = upload_start.elapsed().as_millis();

    let output_len = plan
        .output_projection
        .vocab_size
        .checked_mul(n_seq)
        .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper batched logits shape overflows usize".to_string(),
        })?;
    let compute_start = Instant::now();
    let logits = graph
        .compute_output_f32(logits_tensor, output_len)
        .map_err(
            |error| WhisperDecoderGraphExecutionError::GraphExecutionFailed {
                reason: format!("decoder batched reusable graph compute failed: {error}"),
            },
        )?;
    let compute_ms = compute_start.elapsed().as_millis();
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(WhisperDecoderGraphExecutionError::GraphExecutionFailed {
            reason: "decoder batched reusable logits contain non-finite values".to_string(),
        });
    }

    emit_decoder_graph_detail_trace(
        "decoder_reuse_batch",
        n_seq,
        4,
        (3 * n_seq * std::mem::size_of::<i32>()) + (mask_bits.len() * std::mem::size_of::<u16>()),
        0,
        graph_build_ms,
        upload_ms,
        compute_ms,
        graph_total_start.elapsed().as_millis(),
    );

    Ok(WhisperDecoderBatchedStepOutput {
        logits,
        vocab_size: plan.output_projection.vocab_size,
        n_seq,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_whisper_decoder_batched_prefill_step_ggml_v0(
    runner: &mut GgmlCpuGraphRunner,
    persistent_weights: &WhisperDecoderPersistentWeightCache,
    plan: &WhisperDecoderGraphPlan,
    prompt_tokens: &[u32],
    source: &dyn WhisperDecoderTensorSource,
    config: WhisperDecoderGraphExecutionConfig,
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
) -> Result<WhisperDecoderBatchedStepOutput, WhisperDecoderGraphExecutionError> {
    let graph_total_start = Instant::now();
    validate_decoder_execution_config(plan, config)?;
    if runner.uses_scheduler() {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper batched prefill requires scheduler-off execution".to_string(),
        });
    }
    if config.collect_cross_attention {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper batched prefill does not collect cross-attention outputs".to_string(),
        });
    }
    validate_decoder_tokens(plan, "decoder_batched_prefill_token", prompt_tokens, 0)?;
    if !persistent_weights.supports_cross_attention_for_plan(plan) {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper batched prefill requires resident cross-attention cache".to_string(),
        });
    }

    let self_cache = persistent_weights.self_attention_cache();
    let max_positions = self_cache.max_positions;
    let n_seq = self_cache.n_seq;
    let token_count = prompt_tokens.len();
    if n_seq <= 1 {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper batched prefill requires at least two sequences".to_string(),
        });
    }
    if token_count > max_positions {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "whisper batched prefill token_count {token_count} exceeds KV cache size {max_positions}"
            ),
        });
    }
    if persistent_weights
        .cross_attention
        .iter()
        .any(|cache| cache.n_seq != n_seq)
    {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper batched prefill cross-attention cache n_seq mismatch".to_string(),
        });
    }

    let output_tokens = token_count.checked_mul(n_seq).ok_or_else(|| {
        WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper batched prefill token shape overflows usize".to_string(),
        }
    })?;
    let mut token_ids = Vec::with_capacity(output_tokens);
    let mut positions = Vec::with_capacity(output_tokens);
    let mut row_indices_i32 = Vec::with_capacity(output_tokens);
    let mut row_indices_usize = Vec::with_capacity(output_tokens);
    for _ in 0..n_seq {
        for (position, &token_id) in prompt_tokens.iter().enumerate() {
            let token_i32 = i32::try_from(token_id).map_err(|_| {
                WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: format!("token id {token_id} exceeds ggml i32 input range"),
                }
            })?;
            let position_i32 = i32::try_from(position).map_err(|_| {
                WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: "decoder position id exceeds ggml i32 input range".to_string(),
                }
            })?;
            token_ids.push(token_i32);
            positions.push(position_i32);
            row_indices_i32.push(position_i32);
            row_indices_usize.push(position);
        }
    }

    let hidden = plan.input_shape.hidden_size;
    let encoder_frames = plan.input_shape.encoder_frames;
    let graph_build_start = Instant::now();
    let mut graph = runner.start_graph();
    let mut uploads = Vec::new();
    let token_ids_tensor = graph
        .new_tensor_1d_i32(output_tokens, "whisper_prefill_token")
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_new_tensor_1d(prefill_token)", error)
        })?;
    let positions_tensor = graph
        .new_tensor_1d_i32(output_tokens, "whisper_prefill_position")
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_new_tensor_1d(prefill_position)", error)
        })?;
    let row_index_tensor = graph
        .new_tensor_4d_i32(token_count, 1, n_seq, 1, "whisper_prefill_row")
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_new_tensor_4d(prefill_row)", error)
        })?;
    let attention_mask = graph
        .new_tensor_4d_f16(
            max_positions,
            token_count,
            1,
            n_seq,
            "whisper_prefill_self_mask",
        )
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_new_tensor_4d(prefill_mask)", error)
        })?;
    graph
        .set_input(token_ids_tensor)
        .map_err(|error| map_decoder_execute_graph_error("ggml_set_input(prefill_token)", error))?;
    graph.set_input(positions_tensor).map_err(|error| {
        map_decoder_execute_graph_error("ggml_set_input(prefill_position)", error)
    })?;
    graph
        .set_input(row_index_tensor)
        .map_err(|error| map_decoder_execute_graph_error("ggml_set_input(prefill_row)", error))?;
    graph
        .set_input(attention_mask)
        .map_err(|error| map_decoder_execute_graph_error("ggml_set_input(prefill_mask)", error))?;

    let token_state = graph
        .get_rows(persistent_weights.token_embedding(), token_ids_tensor)
        .map_err(|error| map_decoder_execute_graph_error("ggml_get_rows(prefill_token)", error))?;
    let position_state = graph
        .get_rows(persistent_weights.position_embedding(), positions_tensor)
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_get_rows(prefill_position)", error)
        })?;
    let mut state = graph
        .add(token_state, position_state)
        .map_err(|error| map_decoder_execute_graph_error("ggml_add(prefill_embedding)", error))?;
    let build_self_kv_state = WhisperDecoderSelfKvCacheState::new();
    for layer in &plan.layers {
        state = apply_decoder_self_attention(
            &mut graph,
            &mut uploads,
            tensor_cache,
            Some(persistent_weights),
            Some(&build_self_kv_state),
            source,
            state,
            layer,
            hidden,
            token_count,
            0,
            Some(max_positions),
            Some(attention_mask),
            Some(row_index_tensor),
            config.attention_heads,
            config.layer_norm_epsilon,
            config.use_self_flash_attention,
            n_seq,
        )?;
        let cross_attention = apply_decoder_cross_attention(
            &mut graph,
            &mut uploads,
            tensor_cache,
            Some(persistent_weights),
            source,
            state,
            None,
            layer,
            hidden,
            token_count,
            encoder_frames,
            config.attention_heads,
            config.layer_norm_epsilon,
            config.use_cross_flash_attention,
            false,
            n_seq,
        )?;
        state = apply_decoder_mlp(
            &mut graph,
            &mut uploads,
            tensor_cache,
            Some(persistent_weights),
            source,
            cross_attention.state,
            layer,
            config.layer_norm_epsilon,
        )?;
    }

    let last_token_state =
        view_batched_last_token_state(&mut graph, state, hidden, token_count, n_seq)?;
    let last_token_state = apply_affine_layer_norm(
        &mut graph,
        &mut uploads,
        tensor_cache,
        Some(persistent_weights),
        source,
        last_token_state,
        config.layer_norm_epsilon,
        &plan.final_norm,
        "decoder_prefill_final_norm",
    )?;
    let logits = apply_linear_with_optional_bias(
        &mut graph,
        &mut uploads,
        tensor_cache,
        Some(persistent_weights),
        source,
        last_token_state,
        &plan.output_projection.projection,
        plan.output_projection.bias.as_ref(),
        "decoder_prefill_output_projection",
    )?;
    graph.set_output(logits).map_err(|error| {
        map_decoder_execute_graph_error("ggml_set_output(prefill_logits)", error)
    })?;
    let graph_build_ms = graph_build_start.elapsed().as_millis();

    graph
        .set_i32_slice(token_ids_tensor, &token_ids, "whisper_prefill_token")
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_i32_slice(prefill_token)", error)
        })?;
    graph
        .set_i32_slice(positions_tensor, &positions, "whisper_prefill_position")
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_i32_slice(prefill_position)", error)
        })?;
    graph
        .set_i32_slice(row_index_tensor, &row_indices_i32, "whisper_prefill_row")
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_i32_slice(prefill_row)", error)
        })?;
    let mask_bits = build_fixed_kv_attention_mask_bits_for_query_rows(
        max_positions,
        token_count,
        n_seq,
        &row_indices_usize,
    )
    .map_err(|error| map_decoder_execute_graph_error("whisper_prefill_self_mask", error))?;
    graph
        .set_f16_bits_slice(attention_mask, &mask_bits, "whisper_prefill_self_mask")
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_f16_bits_slice(prefill_mask)", error)
        })?;

    let upload_start = Instant::now();
    let upload_count = uploads.len();
    let upload_bytes: usize = uploads.iter().map(|(_, values, _)| values.byte_len()).sum();
    graph
        .prepare_outputs_for_upload(&[logits])
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_backend_sched_alloc_graph(prefill_logits)", error)
        })?;
    for (tensor, values, label) in uploads {
        upload_decoder_tensor(
            &mut graph,
            tensor,
            values,
            label,
            Some("decoder_prefill_static"),
        )?;
    }
    let upload_ms = upload_start.elapsed().as_millis();

    let output_len = plan
        .output_projection
        .vocab_size
        .checked_mul(n_seq)
        .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper batched prefill logits shape overflows usize".to_string(),
        })?;
    let compute_start = Instant::now();
    let logits = graph
        .compute_output_f32(logits, output_len)
        .map_err(
            |error| WhisperDecoderGraphExecutionError::GraphExecutionFailed {
                reason: format!("decoder batched prefill graph compute failed: {error}"),
            },
        )?;
    let compute_ms = compute_start.elapsed().as_millis();
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(WhisperDecoderGraphExecutionError::GraphExecutionFailed {
            reason: "decoder batched prefill logits contain non-finite values".to_string(),
        });
    }

    emit_decoder_graph_detail_trace(
        "decoder_batched_prefill",
        token_count,
        upload_count + 4,
        upload_bytes
            + (token_ids.len() * std::mem::size_of::<i32>())
            + (positions.len() * std::mem::size_of::<i32>())
            + (row_indices_i32.len() * std::mem::size_of::<i32>())
            + (mask_bits.len() * std::mem::size_of::<u16>()),
        0,
        graph_build_ms,
        upload_ms,
        compute_ms,
        graph_total_start.elapsed().as_millis(),
    );

    Ok(WhisperDecoderBatchedStepOutput {
        logits,
        vocab_size: plan.output_projection.vocab_size,
        n_seq,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_whisper_decoder_reusable_incremental_graph(
    runner: &mut GgmlCpuGraphRunner,
    persistent_weights: &WhisperDecoderPersistentWeightCache,
    plan: &WhisperDecoderGraphPlan,
    source: &dyn WhisperDecoderTensorSource,
    config: WhisperDecoderGraphExecutionConfig,
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
    max_positions: usize,
) -> Result<Seq2SeqReusableDecodeGraph, WhisperDecoderGraphExecutionError> {
    build_whisper_decoder_reusable_incremental_graph_with_n_seq(
        runner,
        persistent_weights,
        plan,
        source,
        config,
        tensor_cache,
        max_positions,
        1,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_whisper_decoder_reusable_incremental_graph_with_n_seq(
    runner: &mut GgmlCpuGraphRunner,
    persistent_weights: &WhisperDecoderPersistentWeightCache,
    plan: &WhisperDecoderGraphPlan,
    source: &dyn WhisperDecoderTensorSource,
    config: WhisperDecoderGraphExecutionConfig,
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
    max_positions: usize,
    n_seq: usize,
) -> Result<Seq2SeqReusableDecodeGraph, WhisperDecoderGraphExecutionError> {
    if n_seq == 0 {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "whisper reusable decode graph n_seq must be positive".to_string(),
        });
    }
    let hidden = plan.input_shape.hidden_size;
    let encoder_frames = plan.input_shape.encoder_frames;
    let mut session = runner
        .start_persistent_graph_session(WHISPER_DECODER_REUSE_GRAPH_CONTEXT_BYTES)
        .map_err(|error| map_decoder_execute_graph_error("whisper_reuse_session", error))?;
    let graph = session.builder();
    let token_id = graph
        .new_tensor_1d_i32(n_seq, "whisper_reuse_token")
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_new_tensor_1d(reuse_token)", error)
        })?;
    let row_index = if n_seq == 1 {
        graph
            .new_tensor_1d_i32(1, "whisper_reuse_row")
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_new_tensor_1d(reuse_row)", error)
            })?
    } else {
        graph
            .new_tensor_4d_i32(1, 1, n_seq, 1, "whisper_reuse_row")
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_new_tensor_4d(reuse_row)", error)
            })?
    };
    let position = graph
        .new_tensor_1d_i32(n_seq, "whisper_reuse_position")
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_new_tensor_1d(reuse_position)", error)
        })?;
    let attention_mask = if n_seq == 1 {
        graph
            .new_tensor_3d_f16(max_positions, 1, 1, "whisper_reuse_self_mask")
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_new_tensor_3d(reuse_mask)", error)
            })?
    } else {
        graph
            .new_tensor_4d_f16(max_positions, 1, 1, n_seq, "whisper_reuse_self_mask")
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_new_tensor_4d(reuse_mask)", error)
            })?
    };
    graph
        .set_input(token_id)
        .map_err(|error| map_decoder_execute_graph_error("ggml_set_input(reuse_token)", error))?;
    graph
        .set_input(row_index)
        .map_err(|error| map_decoder_execute_graph_error("ggml_set_input(reuse_row)", error))?;
    graph.set_input(position).map_err(|error| {
        map_decoder_execute_graph_error("ggml_set_input(reuse_position)", error)
    })?;
    graph
        .set_input(attention_mask)
        .map_err(|error| map_decoder_execute_graph_error("ggml_set_input(reuse_mask)", error))?;

    let token_state = graph
        .get_rows(persistent_weights.token_embedding(), token_id)
        .map_err(|error| map_decoder_execute_graph_error("ggml_get_rows(reuse_token)", error))?;
    let position_state = graph
        .get_rows(persistent_weights.position_embedding(), position)
        .map_err(|error| map_decoder_execute_graph_error("ggml_get_rows(reuse_position)", error))?;
    let mut state = graph
        .add(token_state, position_state)
        .map_err(|error| map_decoder_execute_graph_error("ggml_add(reuse_embedding)", error))?;
    let mut uploads = Vec::new();
    let build_self_kv_state = WhisperDecoderSelfKvCacheState::new();
    for layer in &plan.layers {
        state = apply_decoder_self_attention(
            graph,
            &mut uploads,
            tensor_cache,
            Some(persistent_weights),
            Some(&build_self_kv_state),
            source,
            state,
            layer,
            hidden,
            1,
            0,
            Some(max_positions),
            Some(attention_mask),
            Some(row_index),
            config.attention_heads,
            config.layer_norm_epsilon,
            config.use_self_flash_attention,
            n_seq,
        )?;
        let cross_attention = apply_decoder_cross_attention(
            graph,
            &mut uploads,
            tensor_cache,
            Some(persistent_weights),
            source,
            state,
            None,
            layer,
            hidden,
            1,
            encoder_frames,
            config.attention_heads,
            config.layer_norm_epsilon,
            config.use_cross_flash_attention,
            false,
            n_seq,
        )?;
        state = cross_attention.state;
        state = apply_decoder_mlp(
            graph,
            &mut uploads,
            tensor_cache,
            Some(persistent_weights),
            source,
            state,
            layer,
            config.layer_norm_epsilon,
        )?;
    }

    let last_token_state = if n_seq == 1 {
        view_last_token_state(graph, state, hidden, 1)?
    } else {
        state
    };
    let last_token_state = apply_affine_layer_norm(
        graph,
        &mut uploads,
        tensor_cache,
        Some(persistent_weights),
        source,
        last_token_state,
        config.layer_norm_epsilon,
        &plan.final_norm,
        "decoder_reuse_final_norm",
    )?;
    let logits = apply_linear_with_optional_bias(
        graph,
        &mut uploads,
        tensor_cache,
        Some(persistent_weights),
        source,
        last_token_state,
        &plan.output_projection.projection,
        plan.output_projection.bias.as_ref(),
        "decoder_reuse_output_projection",
    )?;

    graph
        .set_output(logits)
        .map_err(|error| map_decoder_execute_graph_error("ggml_set_output(reuse_logits)", error))?;
    graph
        .prepare_outputs_for_upload(&[logits])
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_backend_sched_alloc_graph(reuse_logits)", error)
        })?;
    for (tensor, values, label) in uploads {
        upload_decoder_tensor(graph, tensor, values, label, Some("decoder_reuse_static"))?;
    }

    Ok(Seq2SeqReusableDecodeGraph::new_with_borrowed_kv_arena(
        session,
        max_positions,
        n_seq,
        token_id,
        row_index,
        position,
        attention_mask,
        logits,
    ))
}

fn decoder_step_needs_encoder_hidden(
    plan: &WhisperDecoderGraphPlan,
    persistent_weights: Option<&WhisperDecoderPersistentWeightCache>,
) -> bool {
    let Some(persistent_weights) = persistent_weights else {
        return true;
    };
    plan.layers
        .iter()
        .any(|layer| !persistent_weights.has_cross_attention(layer.layer_idx))
}

#[allow(clippy::too_many_arguments)]
fn emit_decoder_graph_detail_trace(
    token_label: &'static str,
    token_count: usize,
    upload_count: usize,
    upload_bytes: usize,
    cross_cache_misses: usize,
    graph_build_ms: u128,
    upload_ms: u128,
    compute_ms: u128,
    total_ms: u128,
) {
    if std::env::var_os("OPENASR_WHISPER_GGML_TRACE").is_none() {
        return;
    }
    eprintln!(
        "openasr_whisper_ggml_trace stage=decoder_graph event=detail status=ok token_label={token_label} token_count={token_count} upload_count={upload_count} upload_bytes={upload_bytes} cross_cache_misses={cross_cache_misses} graph_build_ms={graph_build_ms} upload_ms={upload_ms} compute_ms={compute_ms} total_ms={total_ms}"
    );
}

fn validate_decoder_execution_config(
    plan: &WhisperDecoderGraphPlan,
    config: WhisperDecoderGraphExecutionConfig,
) -> Result<(), WhisperDecoderGraphExecutionError> {
    if config.attention_heads == 0 {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "attention_heads must be > 0".to_string(),
        });
    }
    if config.attention_heads != plan.decoder_attention_heads {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "attention_heads mismatch: config={} plan={}",
                config.attention_heads, plan.decoder_attention_heads
            ),
        });
    }
    if !(config.layer_norm_epsilon.is_finite() && config.layer_norm_epsilon > 0.0) {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "layer_norm_epsilon must be finite and > 0".to_string(),
        });
    }
    Ok(())
}

fn validate_decoder_tokens(
    plan: &WhisperDecoderGraphPlan,
    token_label: &'static str,
    decoder_tokens: &[u32],
    position_offset: usize,
) -> Result<(), WhisperDecoderGraphExecutionError> {
    if decoder_tokens.is_empty() {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!("{token_label} must be non-empty"),
        });
    }
    if decoder_tokens.len() > plan.input_shape.token_count {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "{token_label} length {} exceeds planned token_count {}",
                decoder_tokens.len(),
                plan.input_shape.token_count
            ),
        });
    }
    let token_end = position_offset
        .checked_add(decoder_tokens.len())
        .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "decoder token position range overflows usize".to_string(),
        })?;
    if token_end > plan.position_embedding.vocab_size {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "decoder token positions [{position_offset}, {}) exceed position embedding size {}",
                token_end, plan.position_embedding.vocab_size
            ),
        });
    }
    if decoder_tokens.iter().any(|token| {
        usize::try_from(*token)
            .ok()
            .map(|token| token >= plan.output_projection.vocab_size)
            .unwrap_or(true)
    }) {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!("{token_label} contains out-of-vocabulary token id"),
        });
    }
    Ok(())
}

fn validate_encoder_hidden_input(
    plan: &WhisperDecoderGraphPlan,
    encoder_hidden_state: &[f32],
) -> Result<(), WhisperDecoderGraphExecutionError> {
    let expected_encoder_values = plan
        .input_shape
        .encoder_frames
        .checked_mul(plan.input_shape.hidden_size)
        .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "encoder hidden input shape overflows usize".to_string(),
        })?;
    if encoder_hidden_state.len() != expected_encoder_values {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "encoder hidden input has {} elements but expected {} for [{}, {}]",
                encoder_hidden_state.len(),
                expected_encoder_values,
                plan.input_shape.encoder_frames,
                plan.input_shape.hidden_size
            ),
        });
    }
    if encoder_hidden_state.iter().any(|value| !value.is_finite()) {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "encoder_hidden_state contains non-finite values".to_string(),
        });
    }
    Ok(())
}

enum DecoderUploadData<'a> {
    F32(Arc<[f32]>),
    F32Borrowed(&'a [f32]),
    F16Bits(Arc<[u16]>),
    Bytes(Arc<[u8]>),
    I32(Arc<[i32]>),
    I32Borrowed(&'a [i32]),
}

impl DecoderUploadData<'_> {
    fn byte_len(&self) -> usize {
        match self {
            Self::F32(values) => values.len().saturating_mul(std::mem::size_of::<f32>()),
            Self::F32Borrowed(values) => values.len().saturating_mul(std::mem::size_of::<f32>()),
            Self::F16Bits(values) => values.len().saturating_mul(std::mem::size_of::<u16>()),
            Self::Bytes(values) => values.len(),
            Self::I32(values) => values.len().saturating_mul(std::mem::size_of::<i32>()),
            Self::I32Borrowed(values) => values.len().saturating_mul(std::mem::size_of::<i32>()),
        }
    }
}

type DecoderUpload<'a> = (GgmlCpuTensor<'a>, DecoderUploadData<'a>, &'static str);

fn upload_decoder_tensor<'g, 'v>(
    graph: &mut GgmlCpuGraphBuilder<'g>,
    tensor: GgmlCpuTensor<'g>,
    values: DecoderUploadData<'v>,
    label: &'static str,
    context: Option<&'static str>,
) -> Result<(), WhisperDecoderGraphExecutionError> {
    match values {
        DecoderUploadData::F32(values) => graph.set_f32_slice(tensor, values.as_ref(), label),
        DecoderUploadData::F32Borrowed(values) => graph.set_f32_slice(tensor, values, label),
        DecoderUploadData::F16Bits(values) => {
            graph.set_f16_bits_slice(tensor, values.as_ref(), label)
        }
        DecoderUploadData::Bytes(values) => graph.set_bytes_slice(tensor, values.as_ref(), label),
        DecoderUploadData::I32(values) => graph.set_i32_slice(tensor, values.as_ref(), label),
        DecoderUploadData::I32Borrowed(values) => graph.set_i32_slice(tensor, values, label),
    }
    .map_err(
        |error| WhisperDecoderGraphExecutionError::GraphExecutionFailed {
            reason: match context {
                Some(context) => {
                    format!("could not upload tensor '{label}' for {context}: {error}")
                }
                None => format!("could not upload tensor '{label}': {error}"),
            },
        },
    )
}

fn build_decoder_self_attention_causal_mask_f16_bits(
    n_kv: usize,
    token_count: usize,
    position_offset: usize,
) -> Result<Arc<[u16]>, WhisperDecoderGraphExecutionError> {
    let total = n_kv.checked_mul(token_count).ok_or_else(|| {
        WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "decoder self-attention mask shape overflows usize".to_string(),
        }
    })?;
    let mut values = vec![f32_to_f16_bits(0.0); total];
    let neg_inf_bits = f32_to_f16_bits(-f32::INFINITY);
    for token_idx in 0..token_count {
        let max_visible_kv = position_offset.checked_add(token_idx).ok_or_else(|| {
            WhisperDecoderGraphExecutionError::InvalidInput {
                reason: "decoder self-attention mask position overflows usize".to_string(),
            }
        })?;
        let row_offset = token_idx.checked_mul(n_kv).ok_or_else(|| {
            WhisperDecoderGraphExecutionError::InvalidInput {
                reason: "decoder self-attention mask row offset overflows usize".to_string(),
            }
        })?;
        for kv_idx in 0..n_kv {
            if kv_idx > max_visible_kv {
                values[row_offset + kv_idx] = neg_inf_bits;
            }
        }
    }
    Ok(Arc::<[u16]>::from(values.into_boxed_slice()))
}

fn view_last_token_state<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    state: GgmlCpuTensor<'a>,
    hidden: usize,
    prefix_len: usize,
) -> Result<GgmlCpuTensor<'a>, WhisperDecoderGraphExecutionError> {
    let contiguous_state = graph
        .cont(state)
        .map_err(|error| map_decoder_execute_graph_error("ggml_cont(last_token_state)", error))?;
    let row_stride = hidden
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "last token state row stride overflows usize".to_string(),
        })?;
    let offset = prefix_len
        .checked_sub(1)
        .and_then(|index| index.checked_mul(row_stride))
        .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "last token state offset overflows usize".to_string(),
        })?;
    graph
        .view_2d(contiguous_state, hidden, 1, row_stride, offset)
        .map_err(|error| map_decoder_execute_graph_error("ggml_view_2d(last_token_state)", error))
}

fn view_batched_last_token_state<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    state: GgmlCpuTensor<'a>,
    hidden: usize,
    token_count: usize,
    n_seq: usize,
) -> Result<GgmlCpuTensor<'a>, WhisperDecoderGraphExecutionError> {
    if token_count == 0 || n_seq == 0 {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "batched last-token state requires positive token_count and n_seq".to_string(),
        });
    }
    if n_seq == 1 {
        return view_last_token_state(graph, state, hidden, token_count);
    }
    let contiguous_state = graph
        .cont(state)
        .map_err(|error| map_decoder_execute_graph_error("ggml_cont(batched_last_token)", error))?;
    let element_size = std::mem::size_of::<f32>();
    let column_stride = hidden
        .checked_mul(token_count)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "batched last-token state column stride overflows usize".to_string(),
        })?;
    let offset = token_count
        .checked_sub(1)
        .and_then(|index| index.checked_mul(hidden))
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "batched last-token state offset overflows usize".to_string(),
        })?;
    graph
        .view_2d(contiguous_state, hidden, n_seq, column_stride, offset)
        .map_err(|error| map_decoder_execute_graph_error("ggml_view_2d(batched_last_token)", error))
}

fn merge_attention_context_with_n_seq<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    context: GgmlCpuTensor<'a>,
    hidden: usize,
    token_count: usize,
    n_seq: usize,
    label: &'static str,
) -> Result<GgmlCpuTensor<'a>, WhisperDecoderGraphExecutionError> {
    let output_columns = token_count.checked_mul(n_seq).ok_or_else(|| {
        WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "attention context output shape overflows usize".to_string(),
        }
    })?;
    if n_seq == 1 || token_count == 1 {
        return graph
            .reshape_2d(context, hidden, output_columns)
            .map_err(|error| map_decoder_execute_graph_error(label, error));
    }
    let context = graph
        .permute(context, 0, 2, 1, 3)
        .map_err(|error| map_decoder_execute_graph_error("ggml_permute(attn_merge)", error))?;
    let context = graph
        .cont(context)
        .map_err(|error| map_decoder_execute_graph_error("ggml_cont(attn_merge)", error))?;
    graph
        .reshape_2d(context, hidden, output_columns)
        .map_err(|error| map_decoder_execute_graph_error(label, error))
}

fn reshape_projected_hidden_sequence_to_heads<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    sequence_len: usize,
    attention_heads: usize,
    label: &'static str,
) -> Result<GgmlCpuTensor<'a>, WhisperDecoderGraphExecutionError> {
    // Whisper stores projection output as [hidden, sequence]. Split hidden into
    // [head_dim, heads] first, then move sequence before heads to match
    // whisper.cpp's [head_dim, sequence, heads] attention layout.
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
        map_decoder_execute_graph_error,
    )
}

fn reshape_projected_hidden_sequence_to_heads_with_n_seq<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    sequence_len: usize,
    attention_heads: usize,
    n_seq: usize,
    label: &'static str,
) -> Result<GgmlCpuTensor<'a>, WhisperDecoderGraphExecutionError> {
    if n_seq == 1 {
        return reshape_projected_hidden_sequence_to_heads(
            graph,
            projection,
            head_dim,
            sequence_len,
            attention_heads,
            label,
        );
    }
    let reshaped = graph
        .reshape_4d(projection, head_dim, attention_heads, sequence_len, n_seq)
        .map_err(|error| map_decoder_execute_graph_error("ggml_reshape_4d(attn_heads)", error))?;
    let permuted = graph
        .permute(reshaped, 0, 2, 1, 3)
        .map_err(|error| map_decoder_execute_graph_error("ggml_permute(attn_heads)", error))?;
    graph
        .cont(permuted)
        .map_err(|error| map_decoder_execute_graph_error(label, error))
}

fn reshape_projected_hidden_sequence_to_heads_view<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    sequence_len: usize,
    attention_heads: usize,
) -> Result<GgmlCpuTensor<'a>, WhisperDecoderGraphExecutionError> {
    reshape_projection_to_attention_heads(
        graph,
        projection,
        AttentionHeadLayout {
            head_dim,
            attention_heads,
            sequence_len,
        },
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(attn_heads)",
            permute: "ggml_permute(attn_heads)",
            cont: "ggml_cont(attn_heads_view)",
        },
        map_decoder_execute_graph_error,
    )
}

fn reshape_projected_hidden_sequence_to_heads_view_with_n_seq<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    sequence_len: usize,
    attention_heads: usize,
    n_seq: usize,
) -> Result<GgmlCpuTensor<'a>, WhisperDecoderGraphExecutionError> {
    if n_seq == 1 {
        return reshape_projected_hidden_sequence_to_heads_view(
            graph,
            projection,
            head_dim,
            sequence_len,
            attention_heads,
        );
    }
    reshape_projected_hidden_sequence_to_heads_with_n_seq(
        graph,
        projection,
        head_dim,
        sequence_len,
        attention_heads,
        n_seq,
        "ggml_cont(attn_heads_view)",
    )
}

fn view_f16_hidden_sequence_to_heads<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    tensor: GgmlCpuTensor<'a>,
    hidden: usize,
    head_dim: usize,
    sequence_len: usize,
    attention_heads: usize,
    label: &'static str,
) -> Result<GgmlCpuTensor<'a>, WhisperDecoderGraphExecutionError> {
    let element_size = std::mem::size_of::<u16>();
    let nb1 = hidden.checked_mul(element_size).ok_or_else(|| {
        WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "cross-attention hidden stride overflows usize".to_string(),
        }
    })?;
    let nb2 = head_dim.checked_mul(element_size).ok_or_else(|| {
        WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "cross-attention head stride overflows usize".to_string(),
        }
    })?;
    graph
        .view_3d(tensor, head_dim, sequence_len, attention_heads, nb1, nb2, 0)
        .map_err(|error| map_decoder_execute_graph_error(label, error))
}

fn view_f16_hidden_sequence_to_heads_with_n_seq<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    tensor: GgmlCpuTensor<'a>,
    hidden: usize,
    head_dim: usize,
    sequence_len: usize,
    attention_heads: usize,
    n_seq: usize,
    sequence_stride: usize,
    label: &'static str,
) -> Result<GgmlCpuTensor<'a>, WhisperDecoderGraphExecutionError> {
    if n_seq == 1 {
        return view_f16_hidden_sequence_to_heads(
            graph,
            tensor,
            hidden,
            head_dim,
            sequence_len,
            attention_heads,
            label,
        );
    }
    let element_size = std::mem::size_of::<u16>();
    let nb1 = hidden.checked_mul(element_size).ok_or_else(|| {
        WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "attention hidden stride overflows usize".to_string(),
        }
    })?;
    let nb2 = head_dim.checked_mul(element_size).ok_or_else(|| {
        WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "attention head stride overflows usize".to_string(),
        }
    })?;
    let nb3 = hidden
        .checked_mul(sequence_stride)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "attention sequence slot stride overflows usize".to_string(),
        })?;
    graph
        .view_4d(
            tensor,
            head_dim,
            sequence_len,
            attention_heads,
            n_seq,
            nb1,
            nb2,
            nb3,
            0,
        )
        .map_err(|error| map_decoder_execute_graph_error(label, error))
}

#[allow(clippy::too_many_arguments)]
fn apply_decoder_self_attention<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    uploads: &mut Vec<DecoderUpload<'a>>,
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
    persistent_weights: Option<&WhisperDecoderPersistentWeightCache>,
    self_kv_state: Option<&WhisperDecoderSelfKvCacheState>,
    source: &dyn WhisperDecoderTensorSource,
    input: GgmlCpuTensor<'a>,
    layer: &WhisperDecoderLayerPlan,
    hidden: usize,
    token_count: usize,
    position_offset: usize,
    self_attention_kv_len: Option<usize>,
    shared_self_attention_mask: Option<GgmlCpuTensor<'a>>,
    self_kv_row_indices: Option<GgmlCpuTensor<'a>>,
    attention_heads: usize,
    layer_norm_epsilon: f32,
    use_flash_attention: bool,
    n_seq: usize,
) -> Result<GgmlCpuTensor<'a>, WhisperDecoderGraphExecutionError> {
    if n_seq == 0 {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "decoder self-attention n_seq must be positive".to_string(),
        });
    }
    let head_dim = hidden / attention_heads;
    let norm = apply_affine_layer_norm(
        graph,
        uploads,
        tensor_cache,
        persistent_weights,
        source,
        input,
        layer_norm_epsilon,
        &layer.self_attn_norm,
        "decoder_self_attn_norm",
    )?;
    let q = apply_linear_with_bias(
        graph,
        uploads,
        tensor_cache,
        persistent_weights,
        source,
        norm,
        &layer.self_attn_q.projection,
        &layer.self_attn_q.bias,
        "decoder_self_attn_q",
    )?;
    let k = apply_linear(
        graph,
        uploads,
        tensor_cache,
        persistent_weights,
        source,
        norm,
        &layer.self_attn_k,
        "decoder_self_attn_k",
    )?;
    let v = apply_linear_with_bias(
        graph,
        uploads,
        tensor_cache,
        persistent_weights,
        source,
        norm,
        &layer.self_attn_v.projection,
        &layer.self_attn_v.bias,
        "decoder_self_attn_v",
    )?;
    let n_kv = if let Some(kv_len) = self_attention_kv_len {
        kv_len
    } else {
        position_offset.checked_add(token_count).ok_or_else(|| {
            WhisperDecoderGraphExecutionError::InvalidInput {
                reason: "decoder self-attention KV length overflows usize".to_string(),
            }
        })?
    };
    let written_end = position_offset.checked_add(token_count).ok_or_else(|| {
        WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "decoder self-attention KV write range overflows usize".to_string(),
        }
    })?;
    if n_kv < written_end {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "decoder self-attention KV length is smaller than the written prefix"
                .to_string(),
        });
    }
    let use_self_kv = persistent_weights.is_some() && self_kv_state.is_some();
    if n_seq > 1 && !use_self_kv {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "batched whisper self-attention requires resident self-KV cache".to_string(),
        });
    }
    let (k, v) = if use_self_kv {
        let weights = persistent_weights.expect("checked above");
        let self_cache = weights.self_attention_cache();
        let state = self_kv_state.expect("checked above");
        validate_self_kv_step(
            &self_cache,
            state,
            layer.layer_idx,
            hidden,
            token_count,
            position_offset,
            n_kv,
            attention_heads,
        )?;
        if n_seq != self_cache.n_seq {
            return Err(WhisperDecoderGraphExecutionError::InvalidInput {
                reason: format!(
                    "decoder self-attention n_seq mismatch: graph={n_seq} cache={}",
                    self_cache.n_seq
                ),
            });
        }
        let element_size = std::mem::size_of::<u16>();
        let layer_offset = layer
            .layer_idx
            .checked_mul(self_cache.max_positions)
            .and_then(|value| value.checked_mul(self_cache.n_seq))
            .and_then(|value| value.checked_mul(hidden))
            .and_then(|value| value.checked_mul(element_size))
            .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
                reason: "decoder self-attention KV layer offset overflows usize".to_string(),
            })?;
        if let Some(row_indices) = self_kv_row_indices {
            let row_stride = hidden.checked_mul(element_size).ok_or_else(|| {
                WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: "decoder self-attention KV row stride overflows usize".to_string(),
                }
            })?;
            if n_seq == 1 {
                let k_layer = graph
                    .view_2d(
                        weights.self_attention_key(),
                        hidden,
                        self_cache.max_positions,
                        row_stride,
                        layer_offset,
                    )
                    .map_err(|error| {
                        map_decoder_execute_graph_error("ggml_view_2d(self_k_layer)", error)
                    })?;
                let v_layer = graph
                    .view_2d(
                        weights.self_attention_value(),
                        hidden,
                        self_cache.max_positions,
                        row_stride,
                        layer_offset,
                    )
                    .map_err(|error| {
                        map_decoder_execute_graph_error("ggml_view_2d(self_v_layer)", error)
                    })?;
                let k_rows = graph.reshape_2d(k, hidden, token_count).map_err(|error| {
                    map_decoder_execute_graph_error("ggml_reshape_2d(self_k_rows)", error)
                })?;
                let v_rows = graph.reshape_2d(v, hidden, token_count).map_err(|error| {
                    map_decoder_execute_graph_error("ggml_reshape_2d(self_v_rows)", error)
                })?;
                let k_written = graph
                    .set_rows(k_layer, k_rows, row_indices)
                    .map_err(|error| {
                        map_decoder_execute_graph_error("ggml_set_rows(self_k_write)", error)
                    })?;
                let v_written = graph
                    .set_rows(v_layer, v_rows, row_indices)
                    .map_err(|error| {
                        map_decoder_execute_graph_error("ggml_set_rows(self_v_write)", error)
                    })?;
                let nb1 = row_stride;
                let nb2 = head_dim.checked_mul(element_size).ok_or_else(|| {
                    WhisperDecoderGraphExecutionError::InvalidInput {
                        reason: "decoder self-attention KV head stride overflows usize".to_string(),
                    }
                })?;
                let k_cached = graph
                    .view_3d(k_written, head_dim, n_kv, attention_heads, nb1, nb2, 0)
                    .map_err(|error| {
                        map_decoder_execute_graph_error("ggml_view_3d(self_k_cache)", error)
                    })?;
                let v_cached = graph
                    .view_3d(v_written, head_dim, n_kv, attention_heads, nb1, nb2, 0)
                    .map_err(|error| {
                        map_decoder_execute_graph_error("ggml_view_3d(self_v_cache)", error)
                    })?;
                (k_cached, v_cached)
            } else {
                let nb1 = row_stride;
                let nb2 = head_dim.checked_mul(element_size).ok_or_else(|| {
                    WhisperDecoderGraphExecutionError::InvalidInput {
                        reason: "decoder self-attention KV head stride overflows usize".to_string(),
                    }
                })?;
                let nb3 = hidden
                    .checked_mul(self_cache.max_positions)
                    .and_then(|value| value.checked_mul(element_size))
                    .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
                        reason: "decoder self-attention KV sequence stride overflows usize"
                            .to_string(),
                    })?;
                let k_layer = graph
                    .view_4d(
                        weights.self_attention_key(),
                        head_dim,
                        self_cache.max_positions,
                        attention_heads,
                        n_seq,
                        nb1,
                        nb2,
                        nb3,
                        layer_offset,
                    )
                    .map_err(|error| {
                        map_decoder_execute_graph_error("ggml_view_4d(self_k_layer)", error)
                    })?;
                let v_layer = graph
                    .view_4d(
                        weights.self_attention_value(),
                        head_dim,
                        self_cache.max_positions,
                        attention_heads,
                        n_seq,
                        nb1,
                        nb2,
                        nb3,
                        layer_offset,
                    )
                    .map_err(|error| {
                        map_decoder_execute_graph_error("ggml_view_4d(self_v_layer)", error)
                    })?;
                let k_rows = reshape_projected_hidden_sequence_to_heads_with_n_seq(
                    graph,
                    k,
                    head_dim,
                    token_count,
                    attention_heads,
                    n_seq,
                    "ggml_cont(self_k_rows)",
                )?;
                let v_rows = reshape_projected_hidden_sequence_to_heads_with_n_seq(
                    graph,
                    v,
                    head_dim,
                    token_count,
                    attention_heads,
                    n_seq,
                    "ggml_cont(self_v_rows)",
                )?;
                let k_written = graph
                    .set_rows(k_layer, k_rows, row_indices)
                    .map_err(|error| {
                        map_decoder_execute_graph_error("ggml_set_rows(self_k_write)", error)
                    })?;
                let v_written = graph
                    .set_rows(v_layer, v_rows, row_indices)
                    .map_err(|error| {
                        map_decoder_execute_graph_error("ggml_set_rows(self_v_write)", error)
                    })?;
                (k_written, v_written)
            }
        } else {
            if n_seq != 1 {
                return Err(WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: "batched whisper self-attention requires row-indexed KV writes"
                        .to_string(),
                });
            }
            let write_offset = layer_offset
                .checked_add(
                    position_offset
                        .checked_mul(hidden)
                        .and_then(|value| value.checked_mul(element_size))
                        .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
                            reason: "decoder self-attention KV write offset overflows usize"
                                .to_string(),
                        })?,
                )
                .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: "decoder self-attention KV write offset overflows usize".to_string(),
                })?;
            let flat_len = hidden.checked_mul(token_count).ok_or_else(|| {
                WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: "decoder self-attention KV write length overflows usize".to_string(),
                }
            })?;
            let k_flat = graph.reshape_1d(k, flat_len).map_err(|error| {
                map_decoder_execute_graph_error("ggml_reshape_1d(self_k)", error)
            })?;
            let k_dst = graph
                .view_1d(weights.self_attention_key(), flat_len, write_offset)
                .map_err(|error| {
                    map_decoder_execute_graph_error("ggml_view_1d(self_k_write)", error)
                })?;
            let k_write = graph.cpy(k_flat, k_dst).map_err(|error| {
                map_decoder_execute_graph_error("ggml_cpy(self_k_write)", error)
            })?;
            graph.add_side_effect_root(k_write).map_err(|error| {
                map_decoder_execute_graph_error("ggml_side_effect(self_k)", error)
            })?;

            let v_flat = graph.reshape_1d(v, flat_len).map_err(|error| {
                map_decoder_execute_graph_error("ggml_reshape_1d(self_v)", error)
            })?;
            let v_dst = graph
                .view_1d(weights.self_attention_value(), flat_len, write_offset)
                .map_err(|error| {
                    map_decoder_execute_graph_error("ggml_view_1d(self_v_write)", error)
                })?;
            let v_write = graph.cpy(v_flat, v_dst).map_err(|error| {
                map_decoder_execute_graph_error("ggml_cpy(self_v_write)", error)
            })?;
            graph.add_side_effect_root(v_write).map_err(|error| {
                map_decoder_execute_graph_error("ggml_side_effect(self_v)", error)
            })?;
            let nb1 = hidden.checked_mul(element_size).ok_or_else(|| {
                WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: "decoder self-attention KV row stride overflows usize".to_string(),
                }
            })?;
            let nb2 = head_dim.checked_mul(element_size).ok_or_else(|| {
                WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: "decoder self-attention KV head stride overflows usize".to_string(),
                }
            })?;
            let k_cached = graph
                .view_3d(
                    weights.self_attention_key(),
                    head_dim,
                    n_kv,
                    attention_heads,
                    nb1,
                    nb2,
                    layer_offset,
                )
                .map_err(|error| {
                    map_decoder_execute_graph_error("ggml_view_3d(self_k_cache)", error)
                })?;
            let v_cached = graph
                .view_3d(
                    weights.self_attention_value(),
                    head_dim,
                    n_kv,
                    attention_heads,
                    nb1,
                    nb2,
                    layer_offset,
                )
                .map_err(|error| {
                    map_decoder_execute_graph_error("ggml_view_3d(self_v_cache)", error)
                })?;
            (k_cached, v_cached)
        }
    } else {
        let k = reshape_projected_hidden_sequence_to_heads(
            graph,
            k,
            head_dim,
            token_count,
            attention_heads,
            "self_k",
        )?;
        let v = reshape_projected_hidden_sequence_to_heads(
            graph,
            v,
            head_dim,
            token_count,
            attention_heads,
            "self_v",
        )?;
        (k, v)
    };

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let use_self_flash_attention = use_flash_attention && use_self_kv;
    let output_columns = token_count.checked_mul(n_seq).ok_or_else(|| {
        WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "decoder self-attention output shape overflows usize".to_string(),
        }
    })?;
    let context = if use_self_flash_attention {
        let q = reshape_projected_hidden_sequence_to_heads_view_with_n_seq(
            graph,
            q,
            head_dim,
            token_count,
            attention_heads,
            n_seq,
        )?;
        let mask = if let Some(mask) = shared_self_attention_mask {
            Some(mask)
        } else if token_count == 1 && n_seq == 1 {
            None
        } else {
            Some(shared_self_attention_mask.ok_or_else(|| {
                WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: "decoder self-attention mask was not prepared for multi-token step"
                        .to_string(),
                }
            })?)
        };
        let context = graph
            .flash_attn_ext(q, k, v, mask, scale, 0.0, 0.0)
            .map_err(|error| map_decoder_execute_graph_error("ggml_flash_attn_ext(self)", error))?;
        merge_attention_context_with_n_seq(
            graph,
            context,
            hidden,
            token_count,
            n_seq,
            "ggml_reshape_2d(self_flash)",
        )?
    } else {
        let q = reshape_projected_hidden_sequence_to_heads_with_n_seq(
            graph,
            q,
            head_dim,
            token_count,
            attention_heads,
            n_seq,
            "self_q",
        )?;
        let scores = graph
            .mul_mat(k, q)
            .map_err(|error| map_decoder_execute_graph_error("ggml_mul_mat(self_qk)", error))?;
        let mask = if let Some(mask) = shared_self_attention_mask {
            Some(mask)
        } else if token_count == 1 && n_seq == 1 {
            None
        } else {
            Some(shared_self_attention_mask.ok_or_else(|| {
                WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: "decoder self-attention mask was not prepared for multi-token step"
                        .to_string(),
                }
            })?)
        };
        let scores = graph
            .cont(scores)
            .map_err(|error| map_decoder_execute_graph_error("ggml_cont(self_qk)", error))?;
        let probs = graph
            .soft_max_ext(scores, mask, scale, 0.0)
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_soft_max_ext(self_qk)", error)
            })?;
        let v_t = graph
            .permute(v, 1, 0, 2, 3)
            .map_err(|error| map_decoder_execute_graph_error("ggml_permute(self_v_t)", error))?;
        let v_t = graph
            .cont(v_t)
            .map_err(|error| map_decoder_execute_graph_error("ggml_cont(self_v_t)", error))?;
        let context = graph
            .mul_mat(v_t, probs)
            .map_err(|error| map_decoder_execute_graph_error("ggml_mul_mat(self_av)", error))?;
        let context = graph
            .permute(context, 0, 2, 1, 3)
            .map_err(|error| map_decoder_execute_graph_error("ggml_permute(self_merge)", error))?;
        let context = graph
            .cont(context)
            .map_err(|error| map_decoder_execute_graph_error("ggml_cont(self_merge)", error))?;
        graph
            .reshape_2d(context, hidden, output_columns)
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_reshape_2d(self_merge)", error)
            })?
    };

    let projected = apply_linear_with_bias(
        graph,
        uploads,
        tensor_cache,
        persistent_weights,
        source,
        context,
        &layer.self_attn_out.projection,
        &layer.self_attn_out.bias,
        "decoder_self_attn_out",
    )?;
    graph
        .add(projected, input)
        .map_err(|error| map_decoder_execute_graph_error("ggml_add(self_residual)", error))
}

#[allow(clippy::too_many_arguments)]
fn validate_self_kv_step(
    cache: &WhisperDecoderPersistentSelfAttentionCache,
    state: &WhisperDecoderSelfKvCacheState,
    layer_idx: usize,
    hidden: usize,
    token_count: usize,
    position_offset: usize,
    n_kv: usize,
    attention_heads: usize,
) -> Result<(), WhisperDecoderGraphExecutionError> {
    if cache.hidden != hidden {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "self KV hidden mismatch: cache={} graph={hidden}",
                cache.hidden
            ),
        });
    }
    if layer_idx >= cache.layer_count {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "self KV layer {layer_idx} exceeds layer count {}",
                cache.layer_count
            ),
        });
    }
    if n_kv > cache.max_positions {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "self KV positions {} exceed cache size {}",
                n_kv, cache.max_positions
            ),
        });
    }
    if position_offset != state.next_position() {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "self KV position mismatch: requested offset={position_offset} next_position={}",
                state.next_position()
            ),
        });
    }
    if position_offset > 0 && token_count != 1 {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "self KV incremental path only supports one token after prefill".to_string(),
        });
    }
    if attention_heads == 0 || !hidden.is_multiple_of(attention_heads) {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "self KV requires hidden size {hidden} divisible by attention heads {attention_heads}"
            ),
        });
    }
    Ok(())
}

struct DecoderCrossAttentionApplyOutput<'a> {
    state: GgmlCpuTensor<'a>,
    last_token_frame_probs: Option<GgmlCpuTensor<'a>>,
}

fn apply_decoder_cross_attention<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    uploads: &mut Vec<DecoderUpload<'a>>,
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
    persistent_weights: Option<&WhisperDecoderPersistentWeightCache>,
    source: &dyn WhisperDecoderTensorSource,
    input: GgmlCpuTensor<'a>,
    cache: Option<&WhisperDecoderCrossAttentionCache>,
    layer: &WhisperDecoderLayerPlan,
    hidden: usize,
    token_count: usize,
    encoder_frames: usize,
    attention_heads: usize,
    layer_norm_epsilon: f32,
    use_flash_attention: bool,
    collect_attention_probs: bool,
    n_seq: usize,
) -> Result<DecoderCrossAttentionApplyOutput<'a>, WhisperDecoderGraphExecutionError> {
    if n_seq == 0 {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "decoder cross-attention n_seq must be positive".to_string(),
        });
    }
    if n_seq > 1 && collect_attention_probs {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "batched whisper reusable cross-attention does not collect attention outputs"
                .to_string(),
        });
    }
    let head_dim = hidden / attention_heads;
    let norm = apply_affine_layer_norm(
        graph,
        uploads,
        tensor_cache,
        persistent_weights,
        source,
        input,
        layer_norm_epsilon,
        &layer.cross_attn_norm,
        "decoder_cross_attn_norm",
    )?;
    let q = apply_linear_with_bias(
        graph,
        uploads,
        tensor_cache,
        persistent_weights,
        source,
        norm,
        &layer.cross_attn_q.projection,
        &layer.cross_attn_q.bias,
        "decoder_cross_attn_q",
    )?;
    let persistent_cross = persistent_weights.and_then(|weights| {
        weights.cross_attention.get(layer.layer_idx).map(|cache| {
            (
                weights.arena.graph_tensor(cache.key),
                weights.arena.graph_tensor(cache.value),
                cache.layer_stride_frames,
                cache.n_seq,
            )
        })
    });
    let (k, v, cross_sequence_stride, cross_cache_n_seq) = if let Some(cross) = persistent_cross {
        cross
    } else {
        if n_seq != 1 {
            return Err(WhisperDecoderGraphExecutionError::InvalidInput {
                reason: "batched whisper cross-attention requires resident cross-attention cache"
                    .to_string(),
            });
        }
        let cache =
            cache.ok_or_else(|| WhisperDecoderGraphExecutionError::GraphExecutionFailed {
                reason: format!(
                    "missing cross-attention cache for decoder layer {}",
                    layer.layer_idx
                ),
            })?;
        let k = graph
            .new_tensor_2d_f16(hidden, encoder_frames, "decoder_cross_attn_k_cache")
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_new_tensor_2d_f16(cross_k_cache)", error)
            })?;
        graph.set_input(k).map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_input(cross_k_cache)", error)
        })?;
        uploads.push((
            k,
            DecoderUploadData::F16Bits(Arc::clone(&cache.key)),
            "decoder_cross_attn_k_cache",
        ));

        let v = graph
            .new_tensor_2d_f16(hidden, encoder_frames, "decoder_cross_attn_v_cache")
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_new_tensor_2d_f16(cross_v_cache)", error)
            })?;
        graph.set_input(v).map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_input(cross_v_cache)", error)
        })?;
        uploads.push((
            v,
            DecoderUploadData::F16Bits(Arc::clone(&cache.value)),
            "decoder_cross_attn_v_cache",
        ));
        (k, v, encoder_frames, 1)
    };
    if n_seq != cross_cache_n_seq {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "decoder cross-attention n_seq mismatch: graph={n_seq} cache={cross_cache_n_seq}"
            ),
        });
    }

    let k = view_f16_hidden_sequence_to_heads_with_n_seq(
        graph,
        k,
        hidden,
        head_dim,
        encoder_frames,
        attention_heads,
        n_seq,
        cross_sequence_stride,
        "ggml_view(cross_k)",
    )?;
    let v = view_f16_hidden_sequence_to_heads_with_n_seq(
        graph,
        v,
        hidden,
        head_dim,
        encoder_frames,
        attention_heads,
        n_seq,
        cross_sequence_stride,
        "ggml_view(cross_v)",
    )?;

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut last_token_frame_probs = None;
    let output_columns = token_count.checked_mul(n_seq).ok_or_else(|| {
        WhisperDecoderGraphExecutionError::InvalidInput {
            reason: "decoder cross-attention output shape overflows usize".to_string(),
        }
    })?;
    let context = if use_flash_attention && !collect_attention_probs {
        let q = reshape_projected_hidden_sequence_to_heads_view_with_n_seq(
            graph,
            q,
            head_dim,
            token_count,
            attention_heads,
            n_seq,
        )?;
        let context = graph
            .flash_attn_ext(q, k, v, None, scale, 0.0, 0.0)
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_flash_attn_ext(cross)", error)
            })?;
        merge_attention_context_with_n_seq(
            graph,
            context,
            hidden,
            token_count,
            n_seq,
            "ggml_reshape_2d(cross_flash)",
        )?
    } else {
        let q = reshape_projected_hidden_sequence_to_heads_with_n_seq(
            graph,
            q,
            head_dim,
            token_count,
            attention_heads,
            n_seq,
            "cross_q",
        )?;
        let scores = graph
            .mul_mat(k, q)
            .map_err(|error| map_decoder_execute_graph_error("ggml_mul_mat(cross_qk)", error))?;
        let scores = graph
            .cont(scores)
            .map_err(|error| map_decoder_execute_graph_error("ggml_cont(cross_qk)", error))?;
        let probs = graph
            .soft_max_ext(scores, None, scale, 0.0)
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_soft_max_ext(cross_qk)", error)
            })?;
        if collect_attention_probs {
            last_token_frame_probs = Some(probs);
        }
        let v_t = graph
            .permute(v, 1, 0, 2, 3)
            .map_err(|error| map_decoder_execute_graph_error("ggml_permute(cross_v_t)", error))?;
        let v_t = graph
            .cont(v_t)
            .map_err(|error| map_decoder_execute_graph_error("ggml_cont(cross_v_t)", error))?;
        let context = graph
            .mul_mat(v_t, probs)
            .map_err(|error| map_decoder_execute_graph_error("ggml_mul_mat(cross_av)", error))?;
        let context = graph
            .permute(context, 0, 2, 1, 3)
            .map_err(|error| map_decoder_execute_graph_error("ggml_permute(cross_merge)", error))?;
        let context = graph
            .cont(context)
            .map_err(|error| map_decoder_execute_graph_error("ggml_cont(cross_merge)", error))?;
        graph
            .reshape_2d(context, hidden, output_columns)
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_reshape_2d(cross_merge)", error)
            })?
    };

    let projected = apply_linear_with_bias(
        graph,
        uploads,
        tensor_cache,
        persistent_weights,
        source,
        context,
        &layer.cross_attn_out.projection,
        &layer.cross_attn_out.bias,
        "decoder_cross_attn_out",
    )?;
    graph
        .add(projected, input)
        .map(|state| DecoderCrossAttentionApplyOutput {
            state,
            last_token_frame_probs,
        })
        .map_err(|error| map_decoder_execute_graph_error("ggml_add(cross_residual)", error))
}

fn apply_decoder_mlp<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    uploads: &mut Vec<DecoderUpload<'a>>,
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
    persistent_weights: Option<&WhisperDecoderPersistentWeightCache>,
    source: &dyn WhisperDecoderTensorSource,
    input: GgmlCpuTensor<'a>,
    layer: &WhisperDecoderLayerPlan,
    layer_norm_epsilon: f32,
) -> Result<GgmlCpuTensor<'a>, WhisperDecoderGraphExecutionError> {
    let norm = apply_affine_layer_norm(
        graph,
        uploads,
        tensor_cache,
        persistent_weights,
        source,
        input,
        layer_norm_epsilon,
        &layer.mlp_norm,
        "decoder_mlp_norm",
    )?;
    let fc1 = apply_linear_with_bias(
        graph,
        uploads,
        tensor_cache,
        persistent_weights,
        source,
        norm,
        &layer.mlp_fc1.projection,
        &layer.mlp_fc1.bias,
        "decoder_mlp_fc1",
    )?;
    let fc1 = graph
        .gelu(fc1)
        .map_err(|error| map_decoder_execute_graph_error("ggml_gelu(decoder_mlp_fc1)", error))?;
    let fc2 = apply_linear_with_bias(
        graph,
        uploads,
        tensor_cache,
        persistent_weights,
        source,
        fc1,
        &layer.mlp_fc2.projection,
        &layer.mlp_fc2.bias,
        "decoder_mlp_fc2",
    )?;
    graph
        .add(fc2, input)
        .map_err(|error| map_decoder_execute_graph_error("ggml_add(mlp_residual)", error))
}

fn average_last_token_cross_attention_frame_probs(
    attention: &[f32],
    encoder_frames: usize,
    token_count: usize,
    attention_heads: usize,
) -> Result<Vec<f32>, WhisperDecoderGraphExecutionError> {
    if encoder_frames == 0 || token_count == 0 || attention_heads == 0 {
        return Err(WhisperDecoderGraphExecutionError::GraphExecutionFailed {
            reason: "decoder cross-attention output has an empty dimension".to_string(),
        });
    }
    let expected_len = encoder_frames
        .checked_mul(token_count)
        .and_then(|value| value.checked_mul(attention_heads))
        .ok_or_else(|| WhisperDecoderGraphExecutionError::GraphExecutionFailed {
            reason: "decoder cross-attention output shape overflowed".to_string(),
        })?;
    if attention.len() != expected_len {
        return Err(WhisperDecoderGraphExecutionError::GraphExecutionFailed {
            reason: format!(
                "decoder cross-attention output length mismatch: got {}, expected {}",
                attention.len(),
                expected_len
            ),
        });
    }
    let token_index = token_count - 1;
    let mut frame_probs = vec![0.0_f32; encoder_frames];
    for head_index in 0..attention_heads {
        let base = encoder_frames * (token_index + token_count * head_index);
        for frame_index in 0..encoder_frames {
            frame_probs[frame_index] += attention[base + frame_index];
        }
    }
    let inv_heads = 1.0_f32 / attention_heads as f32;
    let mut sum = 0.0_f32;
    for prob in &mut frame_probs {
        if !prob.is_finite() {
            return Err(WhisperDecoderGraphExecutionError::GraphExecutionFailed {
                reason: "decoder cross-attention output contains non-finite values".to_string(),
            });
        }
        *prob *= inv_heads;
        sum += *prob;
    }
    if sum > 0.0 && sum.is_finite() {
        for prob in &mut frame_probs {
            *prob /= sum;
        }
    }
    Ok(frame_probs)
}

fn apply_linear_with_optional_bias<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    uploads: &mut Vec<DecoderUpload<'a>>,
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
    persistent_weights: Option<&WhisperDecoderPersistentWeightCache>,
    source: &dyn WhisperDecoderTensorSource,
    input_tensor: GgmlCpuTensor<'a>,
    projection: &WhisperDecoderLinearProjectionPlan,
    bias: Option<&WhisperDecoderGraphTensorRef>,
    _label_prefix: &str,
) -> Result<GgmlCpuTensor<'a>, WhisperDecoderGraphExecutionError> {
    let projected = apply_linear(
        graph,
        uploads,
        tensor_cache,
        persistent_weights,
        source,
        input_tensor,
        projection,
        _label_prefix,
    )?;
    if let Some(bias) = bias {
        let bias_tensor = if let Some(tensor) =
            persistent_weights.and_then(|weights| weights.vector(bias, projection.output_dim))
        {
            tensor
        } else {
            let bias_values =
                materialize_hidden_vector(tensor_cache, source, bias, projection.output_dim)?;
            let bias_name = "decoder_linear_bias";
            let bias_tensor = graph
                .new_tensor_1d_f32(projection.output_dim, bias_name)
                .map_err(|error| {
                    map_decoder_execute_graph_error("ggml_new_tensor_1d(linear_bias)", error)
                })?;
            graph.set_input(bias_tensor).map_err(|error| {
                map_decoder_execute_graph_error("ggml_set_input(linear_bias)", error)
            })?;
            uploads.push((bias_tensor, DecoderUploadData::F32(bias_values), bias_name));
            bias_tensor
        };
        graph
            .add(projected, bias_tensor)
            .map_err(|error| map_decoder_execute_graph_error("ggml_add(linear_bias)", error))
    } else {
        Ok(projected)
    }
}

fn apply_linear_with_bias<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    uploads: &mut Vec<DecoderUpload<'a>>,
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
    persistent_weights: Option<&WhisperDecoderPersistentWeightCache>,
    source: &dyn WhisperDecoderTensorSource,
    input_tensor: GgmlCpuTensor<'a>,
    projection: &WhisperDecoderLinearProjectionPlan,
    bias: &WhisperDecoderGraphTensorRef,
    _label_prefix: &str,
) -> Result<GgmlCpuTensor<'a>, WhisperDecoderGraphExecutionError> {
    apply_linear_with_optional_bias(
        graph,
        uploads,
        tensor_cache,
        persistent_weights,
        source,
        input_tensor,
        projection,
        Some(bias),
        _label_prefix,
    )
}

fn materialize_linear_projection_output_ggml(
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
    source: &dyn WhisperDecoderTensorSource,
    input_values: Arc<[f32]>,
    input_columns: usize,
    projection: &WhisperDecoderLinearProjectionPlan,
    bias: Option<&WhisperDecoderGraphTensorRef>,
    label_prefix: &'static str,
) -> Result<Arc<[f32]>, WhisperDecoderGraphExecutionError> {
    let mut runner = GgmlCpuGraphRunner::new(whisper_decoder_graph_config()).map_err(|error| {
        WhisperDecoderGraphExecutionError::GraphExecutionFailed {
            reason: format!(
                "could not initialize ggml cpu graph runner for {label_prefix}: {error}"
            ),
        }
    })?;
    materialize_linear_projection_output_with_runner_ggml(
        &mut runner,
        tensor_cache,
        source,
        input_values,
        input_columns,
        projection,
        bias,
        label_prefix,
    )
}

fn materialize_linear_projection_output_with_runner_ggml(
    runner: &mut GgmlCpuGraphRunner,
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
    source: &dyn WhisperDecoderTensorSource,
    input_values: Arc<[f32]>,
    input_columns: usize,
    projection: &WhisperDecoderLinearProjectionPlan,
    bias: Option<&WhisperDecoderGraphTensorRef>,
    label_prefix: &'static str,
) -> Result<Arc<[f32]>, WhisperDecoderGraphExecutionError> {
    let expected_input = projection
        .input_dim
        .checked_mul(input_columns)
        .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!("{label_prefix} input shape overflows usize"),
        })?;
    if input_values.len() != expected_input {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "{label_prefix} input has {} elements but expected {} for [{}, {}]",
                input_values.len(),
                expected_input,
                projection.input_dim,
                input_columns
            ),
        });
    }

    let mut graph = runner.start_graph();
    let mut uploads = Vec::new();

    let input = graph
        .new_tensor_2d_f32(projection.input_dim, input_columns, "linear_cache_input")
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_new_tensor_2d(linear_cache_input)", error)
        })?;
    graph.set_input(input).map_err(|error| {
        map_decoder_execute_graph_error("ggml_set_input(linear_cache_input)", error)
    })?;
    uploads.push((
        input,
        DecoderUploadData::F32(input_values),
        "linear_cache_input",
    ));

    let output = apply_linear_with_optional_bias(
        &mut graph,
        &mut uploads,
        tensor_cache,
        None,
        source,
        input,
        projection,
        bias,
        label_prefix,
    )?;
    graph
        .set_output(output)
        .map_err(|error| map_decoder_execute_graph_error("ggml_set_output(linear_cache)", error))?;
    graph
        .prepare_outputs_for_upload(&[output])
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_backend_sched_alloc_graph(linear_cache)", error)
        })?;

    for (tensor, values, label) in uploads {
        upload_decoder_tensor(&mut graph, tensor, values, label, Some(label_prefix))?;
    }

    let expected_output = projection
        .output_dim
        .checked_mul(input_columns)
        .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!("{label_prefix} output shape overflows usize"),
        })?;
    graph
        .compute_output_f32(output, expected_output)
        .map(|values| Arc::<[f32]>::from(values.into_boxed_slice()))
        .map_err(
            |error| WhisperDecoderGraphExecutionError::GraphExecutionFailed {
                reason: format!("{label_prefix} cache projection failed: {error}"),
            },
        )
}

#[allow(clippy::too_many_arguments)]
fn populate_cross_attention_projection_pairs_with_persistent_weights_runner_ggml(
    runner: &mut GgmlCpuGraphRunner,
    arena: &GgmlStaticTensorArena,
    input_values: &[f32],
    input_columns: usize,
    input_dim: usize,
    output_dim: usize,
    tasks: &[PersistentCrossAttentionProjectionTask],
    label_prefix: &'static str,
) -> Result<CrossCachePopulatePerfStats, WhisperDecoderGraphExecutionError> {
    if tasks.is_empty() {
        return Ok(CrossCachePopulatePerfStats {
            graph_build_ms: 0,
            upload_ms: 0,
            compute_ms: 0,
        });
    }
    let prepared = prepare_cross_attention_projection_pairs_with_persistent_weights_runner_ggml(
        runner,
        arena,
        input_columns,
        input_dim,
        output_dim,
        tasks,
        label_prefix,
    )?;
    prepared.execute(input_values)
}

#[allow(clippy::too_many_arguments)]
fn prepare_cross_attention_projection_pairs_with_persistent_weights_runner_ggml<'a>(
    runner: &'a mut GgmlCpuGraphRunner,
    arena: &GgmlStaticTensorArena,
    input_columns: usize,
    input_dim: usize,
    output_dim: usize,
    tasks: &[PersistentCrossAttentionProjectionTask],
    label_prefix: &'static str,
) -> Result<PreparedCrossCachePopulateStage<'a>, WhisperDecoderGraphExecutionError> {
    let graph_stage_start = Instant::now();
    if tasks.is_empty() {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!("{label_prefix} prepare requires non-empty task list"),
        });
    }

    let expected_input = input_dim.checked_mul(input_columns).ok_or_else(|| {
        WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!("{label_prefix} input shape overflows usize"),
        }
    })?;

    let upload_backend = runner.backend_kind();
    let mut graph = runner.start_graph();
    let any_f16_rhs_task = tasks
        .iter()
        .any(|task| task.key_weight_accepts_f16_rhs || task.value_weight_accepts_f16_rhs);
    let requires_f32_rhs = tasks
        .iter()
        .any(|task| !task.key_weight_accepts_f16_rhs || !task.value_weight_accepts_f16_rhs);
    let use_f32_rhs_on_cpu = matches!(upload_backend, GgmlCpuGraphBackend::Cpu)
        && decoder_persistent_cross_cache_default_f32_rhs_on_cpu_enabled();
    let use_f16_upload = !use_f32_rhs_on_cpu
        && decoder_persistent_cross_cache_f16_upload_enabled(upload_backend, requires_f32_rhs)
        && any_f16_rhs_task;
    let (key_input, value_input, input_upload_is_f16, input_upload_tensor) = if use_f16_upload {
        let input_f16 = graph
            .new_tensor_2d_f16(input_dim, input_columns, "cross_cache_input_f16")
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_new_tensor_2d(cross_cache_input_f16)", error)
            })?;
        graph.set_input(input_f16).map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_input(cross_cache_input_f16)", error)
        })?;
        if requires_f32_rhs {
            let input_f32 = graph
                .new_tensor_2d_f32(input_dim, input_columns, "cross_cache_input_f32")
                .map_err(|error| {
                    map_decoder_execute_graph_error(
                        "ggml_new_tensor_2d(cross_cache_input_f32)",
                        error,
                    )
                })?;
            let input = graph.cpy(input_f16, input_f32).map_err(|error| {
                map_decoder_execute_graph_error("ggml_cpy(cross_cache_input_f16_to_f32)", error)
            })?;
            (
                input_f16,
                input,
                true,
                PreparedCrossCacheInputUploadTensor::F16(input_f16),
            )
        } else {
            (
                input_f16,
                input_f16,
                true,
                PreparedCrossCacheInputUploadTensor::F16(input_f16),
            )
        }
    } else {
        let input = graph
            .new_tensor_2d_f32(input_dim, input_columns, "cross_cache_input")
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_new_tensor_2d(cross_cache_input)", error)
            })?;
        graph.set_input(input).map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_input(cross_cache_input)", error)
        })?;
        (
            input,
            input,
            false,
            PreparedCrossCacheInputUploadTensor::F32(input),
        )
    };

    for task in tasks {
        if task.input_dim != input_dim {
            return Err(WhisperDecoderGraphExecutionError::InvalidInput {
                reason: format!(
                    "{label_prefix} layer {} input dim {} does not match batched input dim {}",
                    task.layer_idx, task.input_dim, input_dim
                ),
            });
        }
        if task.output_dim != output_dim {
            return Err(WhisperDecoderGraphExecutionError::InvalidInput {
                reason: format!(
                    "{label_prefix} layer {} output dim {} does not match persistent output dim {}",
                    task.layer_idx, task.output_dim, output_dim
                ),
            });
        }
        let key_mat_input = if input_upload_is_f16 && task.key_weight_accepts_f16_rhs {
            key_input
        } else {
            value_input
        };
        let value_mat_input = if input_upload_is_f16 && task.value_weight_accepts_f16_rhs {
            key_input
        } else {
            value_input
        };
        let key_output = graph
            .mul_mat(arena.graph_tensor(task.key_weight), key_mat_input)
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_mul_mat(cross_k_linear)", error)
            })?;
        let value_linear = graph
            .mul_mat(arena.graph_tensor(task.value_weight), value_mat_input)
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_mul_mat(cross_v_linear)", error)
            })?;
        let value_output = graph
            .add(value_linear, arena.graph_tensor(task.value_bias))
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_add(cross_v_linear_bias)", error)
            })?;
        let key_write = graph
            .cpy(key_output, arena.graph_tensor(task.key_target))
            .map_err(|error| map_decoder_execute_graph_error("ggml_cpy(cross_k_write)", error))?;
        graph.add_side_effect_root(key_write).map_err(|error| {
            map_decoder_execute_graph_error("ggml_build_forward_expand(cross_k_write)", error)
        })?;
        let value_write = graph
            .cpy(value_output, arena.graph_tensor(task.value_target))
            .map_err(|error| map_decoder_execute_graph_error("ggml_cpy(cross_v_write)", error))?;
        graph.add_side_effect_root(value_write).map_err(|error| {
            map_decoder_execute_graph_error("ggml_build_forward_expand(cross_v_write)", error)
        })?;
    }
    graph.prepare_side_effects_for_upload().map_err(|error| {
        map_decoder_execute_graph_error("ggml_backend_sched_alloc_graph(cross_kv_write)", error)
    })?;
    Ok(PreparedCrossCachePopulateStage {
        graph,
        input_tensor: input_upload_tensor,
        expected_input,
        input_upload_is_f16,
        label_prefix,
        graph_build_ms: graph_stage_start.elapsed().as_millis(),
    })
}

fn persistent_linear_weight_handle(
    linear_weights: &HashMap<LocalLinearWeightCacheKey, GgmlStaticTensor>,
    projection: &WhisperDecoderLinearProjectionPlan,
    label_prefix: &'static str,
) -> Result<GgmlStaticTensor, WhisperDecoderGraphExecutionError> {
    let key = LocalLinearWeightCacheKey {
        tensor_name: projection.weight.tensor_name.clone(),
        input_dim: projection.input_dim,
        output_dim: projection.output_dim,
        source_layout: projection.weight_layout,
    };
    let tensor = linear_weights
        .get(&key)
        .copied()
        .or_else(|| {
            linear_weights.iter().find_map(|(candidate_key, tensor)| {
                (candidate_key.tensor_name == projection.weight.tensor_name
                    && candidate_key.input_dim == projection.input_dim
                    && candidate_key.output_dim == projection.output_dim)
                    .then_some(*tensor)
            })
        })
        .ok_or_else(|| WhisperDecoderGraphExecutionError::GraphExecutionFailed {
            reason: format!(
                "{label_prefix} missing persistent linear weight '{}' for [{}x{}]",
                projection.weight.tensor_name, projection.input_dim, projection.output_dim
            ),
        })?;
    Ok(tensor)
}

fn persistent_linear_weight_type_handle(
    linear_weight_types: &HashMap<LocalLinearWeightCacheKey, i32>,
    projection: &WhisperDecoderLinearProjectionPlan,
    label_prefix: &'static str,
) -> Result<i32, WhisperDecoderGraphExecutionError> {
    let key = LocalLinearWeightCacheKey {
        tensor_name: projection.weight.tensor_name.clone(),
        input_dim: projection.input_dim,
        output_dim: projection.output_dim,
        source_layout: projection.weight_layout,
    };
    let ggml_type = linear_weight_types
        .get(&key)
        .copied()
        .or_else(|| {
            linear_weight_types
                .iter()
                .find_map(|(candidate_key, ggml_type)| {
                    (candidate_key.tensor_name == projection.weight.tensor_name
                        && candidate_key.input_dim == projection.input_dim
                        && candidate_key.output_dim == projection.output_dim)
                        .then_some(*ggml_type)
                })
        })
        .ok_or_else(|| WhisperDecoderGraphExecutionError::GraphExecutionFailed {
            reason: format!(
                "{label_prefix} missing persistent linear weight type '{}' for [{}x{}]",
                projection.weight.tensor_name, projection.input_dim, projection.output_dim
            ),
        })?;
    Ok(ggml_type)
}

fn persistent_vector_handle(
    vectors: &HashMap<LocalVectorCacheKey, GgmlStaticTensor>,
    tensor: &WhisperDecoderGraphTensorRef,
    len: usize,
    label_prefix: &'static str,
) -> Result<GgmlStaticTensor, WhisperDecoderGraphExecutionError> {
    let key = LocalVectorCacheKey {
        tensor_name: tensor.tensor_name.clone(),
        len,
    };
    let static_tensor = vectors.get(&key).copied().ok_or_else(|| {
        WhisperDecoderGraphExecutionError::GraphExecutionFailed {
            reason: format!(
                "{label_prefix} missing persistent vector '{}' len={len}",
                tensor.tensor_name
            ),
        }
    })?;
    Ok(static_tensor)
}

fn apply_linear<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    uploads: &mut Vec<DecoderUpload<'a>>,
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
    persistent_weights: Option<&WhisperDecoderPersistentWeightCache>,
    source: &dyn WhisperDecoderTensorSource,
    input_tensor: GgmlCpuTensor<'a>,
    projection: &WhisperDecoderLinearProjectionPlan,
    _label_prefix: &str,
) -> Result<GgmlCpuTensor<'a>, WhisperDecoderGraphExecutionError> {
    if let Some(weight) = persistent_weights.and_then(|cache| cache.linear_weight(projection)) {
        return graph
            .mul_mat(weight, input_tensor)
            .map_err(|error| map_decoder_execute_graph_error("ggml_mul_mat(linear)", error));
    }

    let weights = tensor_cache.materialize_linear_weight_input_output(source, projection)?;

    let weight_name = "decoder_linear_weight";
    let (ggml_type, upload) = match weights {
        LocalLinearWeightPayload::F16Bits(values) => {
            (GGML_TYPE_F16, DecoderUploadData::F16Bits(values))
        }
        LocalLinearWeightPayload::Quantized { ggml_type, bytes } => {
            (ggml_type, DecoderUploadData::Bytes(bytes))
        }
    };
    let weight = graph
        .new_matmul_weight_2d_typed(
            projection.input_dim,
            projection.output_dim,
            ggml_type,
            weight_name,
        )
        .map_err(|error| {
            map_decoder_execute_graph_error("ggml_new_tensor_2d_typed(linear_weight)", error)
        })?;
    graph
        .set_input(weight)
        .map_err(|error| map_decoder_execute_graph_error("ggml_set_input(linear_weight)", error))?;
    uploads.push((weight, upload, weight_name));
    graph
        .mul_mat(weight, input_tensor)
        .map_err(|error| map_decoder_execute_graph_error("ggml_mul_mat(linear)", error))
}

fn apply_affine_layer_norm<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    uploads: &mut Vec<DecoderUpload<'a>>,
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
    persistent_weights: Option<&WhisperDecoderPersistentWeightCache>,
    source: &dyn WhisperDecoderTensorSource,
    input_tensor: GgmlCpuTensor<'a>,
    layer_norm_epsilon: f32,
    norm: &WhisperDecoderNormPlan,
    _label_prefix: &str,
) -> Result<GgmlCpuTensor<'a>, WhisperDecoderGraphExecutionError> {
    let hidden = usize::try_from(*norm.weight.dims.last().ok_or_else(|| {
        WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!("{} missing weight dims", norm.weight.tensor_name),
        }
    })?)
    .map_err(|_| WhisperDecoderGraphExecutionError::InvalidInput {
        reason: format!(
            "{} hidden dimension does not fit usize",
            norm.weight.tensor_name
        ),
    })?;
    let normalized = graph
        .norm(input_tensor, layer_norm_epsilon)
        .map_err(|error| map_decoder_execute_graph_error("ggml_norm(layer_norm)", error))?;
    let weight_tensor = if let Some(tensor) =
        persistent_weights.and_then(|weights| weights.vector(&norm.weight, hidden))
    {
        tensor
    } else {
        let weight = materialize_hidden_vector(tensor_cache, source, &norm.weight, hidden)?;
        let weight_name = "decoder_layer_norm_weight";
        let weight_tensor = graph
            .new_tensor_1d_f32(hidden, weight_name)
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_new_tensor_1d(layer_norm_weight)", error)
            })?;
        graph.set_input(weight_tensor).map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_input(layer_norm_weight)", error)
        })?;
        uploads.push((weight_tensor, DecoderUploadData::F32(weight), weight_name));
        weight_tensor
    };

    let scaled = graph
        .mul(normalized, weight_tensor)
        .map_err(|error| map_decoder_execute_graph_error("ggml_mul(layer_norm_affine)", error))?;
    let bias_tensor = if let Some(tensor) =
        persistent_weights.and_then(|weights| weights.vector(&norm.bias, hidden))
    {
        tensor
    } else {
        let bias = materialize_hidden_vector(tensor_cache, source, &norm.bias, hidden)?;
        let bias_name = "decoder_layer_norm_bias";
        let bias_tensor = graph
            .new_tensor_1d_f32(hidden, bias_name)
            .map_err(|error| {
                map_decoder_execute_graph_error("ggml_new_tensor_1d(layer_norm_bias)", error)
            })?;
        graph.set_input(bias_tensor).map_err(|error| {
            map_decoder_execute_graph_error("ggml_set_input(layer_norm_bias)", error)
        })?;
        uploads.push((bias_tensor, DecoderUploadData::F32(bias), bias_name));
        bias_tensor
    };
    graph
        .add(scaled, bias_tensor)
        .map_err(|error| map_decoder_execute_graph_error("ggml_add(layer_norm_bias)", error))
}

fn materialize_hidden_vector(
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
    source: &dyn WhisperDecoderTensorSource,
    tensor: &WhisperDecoderGraphTensorRef,
    hidden: usize,
) -> Result<Arc<[f32]>, WhisperDecoderGraphExecutionError> {
    let values = tensor_cache.materialize_tensor_f32(source, tensor)?;
    let start = values.len().saturating_sub(hidden);
    let vector = values.get(start..).ok_or_else(|| {
        WhisperDecoderGraphExecutionError::TensorMaterializationFailed {
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
            WhisperDecoderGraphExecutionError::TensorMaterializationFailed {
                tensor_name: tensor.tensor_name.clone(),
                reason: format!(
                    "tensor has {} elements but hidden vector requires exactly {hidden}",
                    values.len()
                ),
            },
        );
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(
            WhisperDecoderGraphExecutionError::TensorMaterializationFailed {
                tensor_name: tensor.tensor_name.clone(),
                reason: "hidden vector contains non-finite values".to_string(),
            },
        );
    }
    Ok(Arc::<[f32]>::from(vector.to_vec().into_boxed_slice()))
}

fn materialize_decoder_embeddings(
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
    source: &dyn WhisperDecoderTensorSource,
    token_embedding: &WhisperDecoderEmbeddingPlan,
    position_embedding: &WhisperDecoderEmbeddingPlan,
    tokens: &[u32],
) -> Result<Vec<f32>, WhisperDecoderGraphExecutionError> {
    materialize_decoder_embeddings_with_position_offset(
        tensor_cache,
        source,
        token_embedding,
        position_embedding,
        tokens,
        0,
    )
}

fn materialize_decoder_embeddings_with_position_offset(
    tensor_cache: &mut WhisperDecoderExecutionTensorCache,
    source: &dyn WhisperDecoderTensorSource,
    token_embedding: &WhisperDecoderEmbeddingPlan,
    position_embedding: &WhisperDecoderEmbeddingPlan,
    tokens: &[u32],
    position_offset: usize,
) -> Result<Vec<f32>, WhisperDecoderGraphExecutionError> {
    let token_weights = tensor_cache.materialize_tensor_f32(source, &token_embedding.weight)?;
    let position_weights =
        tensor_cache.materialize_tensor_f32(source, &position_embedding.weight)?;
    if token_weights.iter().any(|value| !value.is_finite()) {
        return Err(
            WhisperDecoderGraphExecutionError::TensorMaterializationFailed {
                tensor_name: token_embedding.weight.tensor_name.clone(),
                reason: "token embedding contains non-finite values".to_string(),
            },
        );
    }
    if position_weights.iter().any(|value| !value.is_finite()) {
        return Err(
            WhisperDecoderGraphExecutionError::TensorMaterializationFailed {
                tensor_name: position_embedding.weight.tensor_name.clone(),
                reason: "position embedding contains non-finite values".to_string(),
            },
        );
    }

    let hidden = token_embedding.hidden_size;
    let mut hidden_seq = vec![0.0f32; hidden * tokens.len()];
    for (relative_position, token) in tokens.iter().copied().enumerate() {
        let absolute_position =
            position_offset
                .checked_add(relative_position)
                .ok_or_else(|| WhisperDecoderGraphExecutionError::InvalidInput {
                    reason: "decoder absolute position overflows usize".to_string(),
                })?;
        if absolute_position >= position_embedding.vocab_size {
            return Err(WhisperDecoderGraphExecutionError::InvalidInput {
                reason: format!(
                    "decoder absolute position {absolute_position} exceeds position embedding size {}",
                    position_embedding.vocab_size
                ),
            });
        }
        let token_idx = usize::try_from(token).map_err(|_| {
            WhisperDecoderGraphExecutionError::InvalidInput {
                reason: format!("token id {token} does not fit usize"),
            }
        })?;
        if token_idx >= token_embedding.vocab_size {
            return Err(WhisperDecoderGraphExecutionError::InvalidInput {
                reason: format!(
                    "token id {token_idx} exceeds decoder vocab size {}",
                    token_embedding.vocab_size
                ),
            });
        }
        for hidden_idx in 0..hidden {
            let token_value = embedding_value(
                &token_weights,
                token_embedding,
                token_idx,
                hidden_idx,
                &token_embedding.weight.tensor_name,
            )?;
            let pos_value = embedding_value(
                &position_weights,
                position_embedding,
                absolute_position,
                hidden_idx,
                &position_embedding.weight.tensor_name,
            )?;
            hidden_seq[relative_position * hidden + hidden_idx] = token_value + pos_value;
        }
    }
    Ok(hidden_seq)
}

fn embedding_value(
    values: &[f32],
    plan: &WhisperDecoderEmbeddingPlan,
    row_idx: usize,
    hidden_idx: usize,
    tensor_name: &str,
) -> Result<f32, WhisperDecoderGraphExecutionError> {
    let index = match plan.layout {
        WhisperDecoderEmbeddingLayout::VocabHidden => hidden_idx
            .checked_add(row_idx.saturating_mul(plan.hidden_size))
            .ok_or_else(
                || WhisperDecoderGraphExecutionError::TensorMaterializationFailed {
                    tensor_name: tensor_name.to_string(),
                    reason: "embedding index overflow".to_string(),
                },
            )?,
        WhisperDecoderEmbeddingLayout::HiddenVocab => row_idx
            .checked_add(hidden_idx.saturating_mul(plan.vocab_size))
            .ok_or_else(
                || WhisperDecoderGraphExecutionError::TensorMaterializationFailed {
                    tensor_name: tensor_name.to_string(),
                    reason: "embedding index overflow".to_string(),
                },
            )?,
    };
    values.get(index).copied().ok_or_else(|| {
        WhisperDecoderGraphExecutionError::TensorMaterializationFailed {
            tensor_name: tensor_name.to_string(),
            reason: format!(
                "embedding index {} out of bounds (len={})",
                index,
                values.len()
            ),
        }
    })
}

fn transpose_weight_output_input_to_input_output<T: Copy>(
    source: &[T],
    input_dim: usize,
    output_dim: usize,
) -> Result<Vec<T>, WhisperDecoderGraphExecutionError> {
    let expected = input_dim.checked_mul(output_dim).ok_or_else(|| {
        WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "cannot transpose weight with overflowing shape {output_dim}x{input_dim}"
            ),
        }
    })?;
    if source.len() != expected {
        return Err(WhisperDecoderGraphExecutionError::InvalidInput {
            reason: format!(
                "cannot transpose weight with {} values for {}x{}",
                source.len(),
                output_dim,
                input_dim
            ),
        });
    }
    Ok(source.to_vec())
}

fn normalize_hidden_layout<'a>(
    input: &'a [f32],
    layout: WhisperDecoderHiddenStateLayout,
    _frames: usize,
    _hidden: usize,
) -> Cow<'a, [f32]> {
    match layout {
        WhisperDecoderHiddenStateLayout::SequenceHidden => Cow::Borrowed(input),
    }
}

fn argmax_finite(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then_with(|| rhs.0.cmp(&lhs.0)))
        .map(|(idx, _)| idx)
}

fn map_decoder_execute_graph_error(
    primitive: &'static str,
    error: GgmlCpuGraphError,
) -> WhisperDecoderGraphExecutionError {
    match error {
        GgmlCpuGraphError::UnsupportedOperation { .. }
        | GgmlCpuGraphError::UnsupportedInputs { .. }
        | GgmlCpuGraphError::GraphBuildFailed { .. } => {
            WhisperDecoderGraphExecutionError::UnsupportedDecoderPrimitive {
                primitive,
                reason: error.to_string(),
            }
        }
        _ => WhisperDecoderGraphExecutionError::GraphExecutionFailed {
            reason: format!("{primitive} failed: {error}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::BTreeMap};

    use crate::ggml_runtime::{GgmlCpuGraphConfig, GgmlCpuGraphRunner};

    use super::*;

    #[test]
    fn default_cpu_f32_rhs_policy_enables_for_auto_and_blas() {
        for value in [None, Some(""), Some("auto"), Some("default"), Some("blas")] {
            assert!(decoder_persistent_cross_cache_default_f32_rhs_on_cpu_enabled_with_env(value));
        }
    }

    #[test]
    fn default_cpu_f32_rhs_policy_disables_for_off_modes() {
        for value in [Some("off"), Some("none"), Some("0"), Some(" OFF ")] {
            assert!(!decoder_persistent_cross_cache_default_f32_rhs_on_cpu_enabled_with_env(value));
        }
    }

    #[test]
    fn persistent_cross_cache_f16_upload_default_enables_cpu_and_pure_f16_metal() {
        assert!(decoder_persistent_cross_cache_f16_upload_enabled_with_env(
            None, None, true
        ));
        assert!(decoder_persistent_cross_cache_f16_upload_enabled_with_env(
            None, None, true
        ));
        assert!(!decoder_persistent_cross_cache_f16_upload_enabled_with_env(
            None, None, false
        ));
    }

    #[test]
    fn persistent_cross_cache_f16_upload_env_overrides_default() {
        for value in [
            Some("1"),
            Some("true"),
            Some("TRUE"),
            Some("yes"),
            Some("on"),
        ] {
            assert!(decoder_persistent_cross_cache_f16_upload_enabled_with_env(
                None, value, false
            ));
            assert!(!decoder_persistent_cross_cache_f16_upload_enabled_with_env(
                value,
                Some("1"),
                true
            ));
        }
    }

    #[test]
    fn one_layer_tiny_decoder_graph_plan_builds() {
        let metadata = WhisperDecoderGraphMetadata {
            decoder_layers: 1,
            decoder_hidden_size: 4,
            decoder_attention_heads: 2,
            vocab_size: 16,
            max_target_positions: 8,
        };
        let binding = WhisperDecoderTensorBindingSeam {
            token_embedding_weight: Some(tensor("model.decoder.embed_tokens.weight", &[16, 4])),
            position_embedding_weight: Some(tensor(
                "model.decoder.embed_positions.weight",
                &[8, 4],
            )),
            final_norm_weight: Some(tensor("model.decoder.layer_norm.weight", &[4])),
            final_norm_bias: Some(tensor("model.decoder.layer_norm.bias", &[4])),
            output_projection_weight: Some(tensor("model.decoder.embed_tokens.weight", &[16, 4])),
            output_projection_bias: None,
            layers: vec![one_layer_binding()],
        };
        let materialization = WhisperDecoderTensorMaterializationSeam {
            source_label: "test-fixture",
            materialized_tensor_count: 32,
        };
        let input_shape = WhisperDecoderGraphInputShape {
            token_count: 3,
            encoder_frames: 2,
            hidden_size: 4,
        };

        let plan =
            build_whisper_decoder_graph_plan(metadata, &binding, &materialization, input_shape)
                .expect("decoder graph plan should succeed");
        assert_eq!(plan.layers.len(), 1);
        assert_eq!(plan.output_projection.vocab_size, 16);
        assert_eq!(plan.token_embedding.hidden_size, 4);
        assert_eq!(plan.position_embedding.vocab_size, 8);
    }

    #[test]
    fn missing_cross_attention_weight_fails_closed() {
        let metadata = WhisperDecoderGraphMetadata {
            decoder_layers: 1,
            decoder_hidden_size: 4,
            decoder_attention_heads: 2,
            vocab_size: 16,
            max_target_positions: 8,
        };
        let mut layer = one_layer_binding();
        layer.cross_attn_q_weight = None;
        let binding = WhisperDecoderTensorBindingSeam {
            token_embedding_weight: Some(tensor("model.decoder.embed_tokens.weight", &[16, 4])),
            position_embedding_weight: Some(tensor(
                "model.decoder.embed_positions.weight",
                &[8, 4],
            )),
            final_norm_weight: Some(tensor("model.decoder.layer_norm.weight", &[4])),
            final_norm_bias: Some(tensor("model.decoder.layer_norm.bias", &[4])),
            output_projection_weight: Some(tensor("model.decoder.embed_tokens.weight", &[16, 4])),
            output_projection_bias: None,
            layers: vec![layer],
        };
        let materialization = WhisperDecoderTensorMaterializationSeam {
            source_label: "test-fixture",
            materialized_tensor_count: 32,
        };
        let input_shape = WhisperDecoderGraphInputShape {
            token_count: 3,
            encoder_frames: 2,
            hidden_size: 4,
        };

        let error =
            build_whisper_decoder_graph_plan(metadata, &binding, &materialization, input_shape)
                .expect_err("missing cross attn q weight must fail closed");
        assert!(matches!(
            error,
            WhisperDecoderGraphPlanError::MissingTensorBinding { .. }
        ));
    }

    #[test]
    fn tiny_synthetic_one_step_logits_are_finite() {
        let metadata = WhisperDecoderGraphMetadata {
            decoder_layers: 1,
            decoder_hidden_size: 4,
            decoder_attention_heads: 2,
            vocab_size: 16,
            max_target_positions: 8,
        };
        let binding = WhisperDecoderTensorBindingSeam {
            token_embedding_weight: Some(tensor("model.decoder.embed_tokens.weight", &[16, 4])),
            position_embedding_weight: Some(tensor(
                "model.decoder.embed_positions.weight",
                &[8, 4],
            )),
            final_norm_weight: Some(tensor("model.decoder.layer_norm.weight", &[4])),
            final_norm_bias: Some(tensor("model.decoder.layer_norm.bias", &[4])),
            output_projection_weight: Some(tensor("model.decoder.embed_tokens.weight", &[16, 4])),
            output_projection_bias: None,
            layers: vec![one_layer_binding()],
        };
        let materialization = WhisperDecoderTensorMaterializationSeam {
            source_label: "test-fixture",
            materialized_tensor_count: 32,
        };
        let input_shape = WhisperDecoderGraphInputShape {
            token_count: 3,
            encoder_frames: 2,
            hidden_size: 4,
        };
        let plan =
            build_whisper_decoder_graph_plan(metadata, &binding, &materialization, input_shape)
                .expect("decoder graph plan should succeed");
        let source = MockTensorSource::from_plan(&plan);

        for use_cross_flash_attention in [false, true] {
            let output = run_whisper_decoder_greedy_step_ggml_v0(
                &plan,
                &WhisperDecoderGraphExecutionInput {
                    decoder_prefix_tokens: vec![1],
                    encoder_hidden_state: vec![
                        0.1, 0.2, 0.3, 0.4, //
                        0.5, 0.6, 0.7, 0.8,
                    ],
                    encoder_layout: WhisperDecoderHiddenStateLayout::SequenceHidden,
                },
                &source,
                WhisperDecoderGraphExecutionConfig {
                    attention_heads: 2,
                    use_self_flash_attention: false,
                    use_cross_flash_attention,
                    collect_cross_attention: false,
                    layer_norm_epsilon: 1.0e-5,
                },
            )
            .expect("one-step decoder graph must execute");

            assert_eq!(output.prefix_len, 1);
            assert_eq!(output.vocab_size, 16);
            assert_eq!(output.logits.len(), 16);
            assert!(
                output.logits.iter().all(|value| value.is_finite()),
                "decoder logits should be finite with cross_flash={use_cross_flash_attention}: {:?}",
                output.logits
            );
        }
        let output = run_whisper_decoder_greedy_step_ggml_v0(
            &plan,
            &WhisperDecoderGraphExecutionInput {
                decoder_prefix_tokens: vec![1],
                encoder_hidden_state: vec![
                    0.1, 0.2, 0.3, 0.4, //
                    0.5, 0.6, 0.7, 0.8,
                ],
                encoder_layout: WhisperDecoderHiddenStateLayout::SequenceHidden,
            },
            &source,
            WhisperDecoderGraphExecutionConfig {
                attention_heads: 2,
                use_self_flash_attention: false,
                use_cross_flash_attention: true,
                collect_cross_attention: true,
                layer_norm_epsilon: 1.0e-5,
            },
        )
        .expect("one-step decoder graph must expose cross attention");
        let frame_probs = output
            .last_token_cross_attention_frame_probs
            .expect("cross-attention collection should return frame probabilities");
        assert_eq!(frame_probs.len(), 2);
        assert!(frame_probs.iter().all(|value| value.is_finite()));
        let sum = frame_probs.iter().sum::<f32>();
        assert!((sum - 1.0).abs() < 1.0e-4, "sum={sum}");
    }

    #[test]
    fn batched_reused_incremental_logits_match_serial_slots() {
        let plan = tiny_decoder_plan();
        let source = MockTensorSource::from_plan(&plan);
        let config = WhisperDecoderGraphExecutionConfig {
            attention_heads: 2,
            use_self_flash_attention: false,
            use_cross_flash_attention: false,
            collect_cross_attention: false,
            layer_norm_epsilon: 1.0e-5,
        };
        let encoders = [
            vec![
                0.1, 0.2, 0.3, 0.4, //
                0.5, 0.6, 0.7, 0.8,
            ],
            vec![
                0.8, 0.7, 0.6, 0.5, //
                0.4, 0.3, 0.2, 0.1,
            ],
        ];
        let token_ids = [1_u32, 2_u32];
        let positions = [0_usize, 0_usize];
        let total_tokens = [1_usize, 1_usize];

        let serial_logits = token_ids
            .iter()
            .enumerate()
            .map(|(slot, &token_id)| {
                let mut runner =
                    GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
                        .expect("serial runner should initialize");
                let mut tensor_cache = WhisperDecoderExecutionTensorCache::default();
                let persistent = WhisperDecoderPersistentWeightCache::build_static_stage(
                    &mut runner,
                    &plan,
                    &source,
                    &mut tensor_cache,
                    plan.position_embedding.vocab_size,
                    None,
                )
                .expect("serial persistent cache should build");
                persistent
                    .populate_cross_attention_stage(
                        &mut runner,
                        &plan,
                        &encoders[slot],
                        WhisperDecoderHiddenStateLayout::SequenceHidden,
                    )
                    .expect("serial cross-cache should populate");
                let state = WhisperDecoderSelfKvCacheState::new();
                let mut reuse = None;
                run_whisper_decoder_reused_incremental_step_ggml_v0(
                    &mut reuse,
                    &mut runner,
                    &persistent,
                    &state,
                    positions[slot],
                    &plan,
                    token_id,
                    &source,
                    config,
                    &mut tensor_cache,
                )
                .expect("serial reusable step should run")
                .logits
            })
            .collect::<Vec<_>>();

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("batched runner should initialize");
        let mut tensor_cache = WhisperDecoderExecutionTensorCache::default();
        let persistent = WhisperDecoderPersistentWeightCache::build_static_stage_with_n_seq(
            &mut runner,
            &plan,
            &source,
            &mut tensor_cache,
            plan.position_embedding.vocab_size,
            None,
            token_ids.len(),
        )
        .expect("batched persistent cache should build");
        for (slot, encoder) in encoders.iter().enumerate() {
            persistent
                .populate_cross_attention_stage_slot(
                    &mut runner,
                    &plan,
                    encoder,
                    WhisperDecoderHiddenStateLayout::SequenceHidden,
                    slot,
                )
                .expect("batched cross-cache slot should populate");
        }
        let mut reuse = None;
        let batched = run_whisper_decoder_reused_batched_incremental_step_ggml_v0(
            &mut reuse,
            &mut runner,
            &persistent,
            &plan,
            &token_ids,
            &positions,
            &total_tokens,
            &source,
            config,
            &mut tensor_cache,
        )
        .expect("batched reusable step should run");

        assert_eq!(batched.vocab_size, plan.output_projection.vocab_size);
        assert_eq!(batched.n_seq, token_ids.len());
        assert_eq!(batched.logits.len(), batched.vocab_size * batched.n_seq);
        for (slot, expected) in serial_logits.iter().enumerate() {
            let start = slot * batched.vocab_size;
            let actual = &batched.logits[start..start + batched.vocab_size];
            assert_f32_slice_close(actual, expected, 1.0e-4);
        }
    }

    #[test]
    fn batched_prefill_logits_match_serial_prefixes_and_seeds_reused_step() {
        let plan = tiny_decoder_plan();
        let source = MockTensorSource::from_plan(&plan);
        let config = WhisperDecoderGraphExecutionConfig {
            attention_heads: 2,
            use_self_flash_attention: false,
            use_cross_flash_attention: false,
            collect_cross_attention: false,
            layer_norm_epsilon: 1.0e-5,
        };
        let encoders = [
            vec![
                0.1, 0.2, 0.3, 0.4, //
                0.5, 0.6, 0.7, 0.8,
            ],
            vec![
                0.8, 0.7, 0.6, 0.5, //
                0.4, 0.3, 0.2, 0.1,
            ],
        ];
        let prompt = vec![1_u32, 2_u32];
        let followup = 3_u32;

        let serial_prefill_logits = encoders
            .iter()
            .map(|encoder| {
                run_whisper_decoder_greedy_step_ggml_v0(
                    &plan,
                    &WhisperDecoderGraphExecutionInput {
                        decoder_prefix_tokens: prompt.clone(),
                        encoder_hidden_state: encoder.clone(),
                        encoder_layout: WhisperDecoderHiddenStateLayout::SequenceHidden,
                    },
                    &source,
                    config,
                )
                .expect("serial prefill graph should run")
                .logits
            })
            .collect::<Vec<_>>();
        let serial_followup_logits = encoders
            .iter()
            .map(|encoder| {
                let mut prefix = prompt.clone();
                prefix.push(followup);
                run_whisper_decoder_greedy_step_ggml_v0(
                    &plan,
                    &WhisperDecoderGraphExecutionInput {
                        decoder_prefix_tokens: prefix,
                        encoder_hidden_state: encoder.clone(),
                        encoder_layout: WhisperDecoderHiddenStateLayout::SequenceHidden,
                    },
                    &source,
                    config,
                )
                .expect("serial followup graph should run")
                .logits
            })
            .collect::<Vec<_>>();

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("batched runner should initialize");
        let mut tensor_cache = WhisperDecoderExecutionTensorCache::default();
        let persistent = WhisperDecoderPersistentWeightCache::build_static_stage_with_n_seq(
            &mut runner,
            &plan,
            &source,
            &mut tensor_cache,
            plan.position_embedding.vocab_size,
            None,
            encoders.len(),
        )
        .expect("batched persistent cache should build");
        for (slot, encoder) in encoders.iter().enumerate() {
            persistent
                .populate_cross_attention_stage_slot(
                    &mut runner,
                    &plan,
                    encoder,
                    WhisperDecoderHiddenStateLayout::SequenceHidden,
                    slot,
                )
                .expect("batched cross-cache slot should populate");
        }
        let batched_prefill = run_whisper_decoder_batched_prefill_step_ggml_v0(
            &mut runner,
            &persistent,
            &plan,
            &prompt,
            &source,
            config,
            &mut tensor_cache,
        )
        .expect("batched prefill should run");
        assert_eq!(
            batched_prefill.vocab_size,
            plan.output_projection.vocab_size
        );
        assert_eq!(batched_prefill.n_seq, encoders.len());
        assert_eq!(
            batched_prefill.logits.len(),
            batched_prefill.vocab_size * batched_prefill.n_seq
        );
        for (slot, expected) in serial_prefill_logits.iter().enumerate() {
            let start = slot * batched_prefill.vocab_size;
            let actual = &batched_prefill.logits[start..start + batched_prefill.vocab_size];
            assert_f32_slice_close(actual, expected, 1.0e-3);
        }

        let mut reuse = None;
        let batched_followup = run_whisper_decoder_reused_batched_incremental_step_ggml_v0(
            &mut reuse,
            &mut runner,
            &persistent,
            &plan,
            &[followup, followup],
            &[2, 2],
            &[3, 3],
            &source,
            config,
            &mut tensor_cache,
        )
        .expect("batched followup should run");
        for (slot, expected) in serial_followup_logits.iter().enumerate() {
            let start = slot * batched_followup.vocab_size;
            let actual = &batched_followup.logits[start..start + batched_followup.vocab_size];
            assert_f32_slice_close(actual, expected, 1.0e-3);
        }
    }

    #[test]
    fn linear_weights_are_cached_across_decoder_steps_for_same_source() {
        let metadata = WhisperDecoderGraphMetadata {
            decoder_layers: 1,
            decoder_hidden_size: 4,
            decoder_attention_heads: 2,
            vocab_size: 16,
            max_target_positions: 8,
        };
        let binding = WhisperDecoderTensorBindingSeam {
            token_embedding_weight: Some(tensor("model.decoder.embed_tokens.weight", &[16, 4])),
            position_embedding_weight: Some(tensor(
                "model.decoder.embed_positions.weight",
                &[8, 4],
            )),
            final_norm_weight: Some(tensor("model.decoder.layer_norm.weight", &[4])),
            final_norm_bias: Some(tensor("model.decoder.layer_norm.bias", &[4])),
            output_projection_weight: Some(tensor("model.decoder.embed_tokens.weight", &[16, 4])),
            output_projection_bias: None,
            layers: vec![one_layer_binding()],
        };
        let materialization = WhisperDecoderTensorMaterializationSeam {
            source_label: "test-fixture",
            materialized_tensor_count: 32,
        };
        let input_shape = WhisperDecoderGraphInputShape {
            token_count: 3,
            encoder_frames: 2,
            hidden_size: 4,
        };
        let plan =
            build_whisper_decoder_graph_plan(metadata, &binding, &materialization, input_shape)
                .expect("decoder graph plan should succeed");

        let source = CountingTensorSource::from_plan(&plan);
        let input = WhisperDecoderGraphExecutionInput {
            decoder_prefix_tokens: vec![1],
            encoder_hidden_state: vec![
                0.1, 0.2, 0.3, 0.4, //
                0.5, 0.6, 0.7, 0.8,
            ],
            encoder_layout: WhisperDecoderHiddenStateLayout::SequenceHidden,
        };

        let mut cache = WhisperDecoderExecutionTensorCache::default();
        run_whisper_decoder_greedy_step_with_cache_ggml_v0(
            &plan,
            &input,
            &source,
            WhisperDecoderGraphExecutionConfig {
                attention_heads: 2,
                use_self_flash_attention: false,
                use_cross_flash_attention: false,
                collect_cross_attention: false,
                layer_norm_epsilon: 1.0e-5,
            },
            &mut cache,
        )
        .expect("first decoder step should succeed");
        run_whisper_decoder_greedy_step_with_cache_ggml_v0(
            &plan,
            &input,
            &source,
            WhisperDecoderGraphExecutionConfig {
                attention_heads: 2,
                use_self_flash_attention: false,
                use_cross_flash_attention: false,
                collect_cross_attention: false,
                layer_norm_epsilon: 1.0e-5,
            },
            &mut cache,
        )
        .expect("second decoder step should succeed");

        assert_eq!(
            source.count_for("model.decoder.layers.0.self_attn.q_proj.weight"),
            1,
            "linear weight should be materialized once and reused from cache"
        );
    }

    #[test]
    fn quantized_square_output_input_projection_is_accepted() {
        let projection = WhisperDecoderLinearProjectionPlan {
            weight: tensor("decoder.square.weight", &[4, 4]),
            weight_layout: WhisperDecoderLinearWeightLayout::OutputInput,
            input_dim: 4,
            output_dim: 4,
        };
        let source = QuantizedOnlyTensorSource {
            tensor_name: "decoder.square.weight".to_string(),
            ggml_type: 8,
            bytes: vec![1, 2, 3, 4],
        };
        let mut cache = WhisperDecoderExecutionTensorCache::default();
        let payload = cache
            .materialize_linear_weight_input_output(&source, &projection)
            .expect("square quantized projection should be accepted");
        match payload {
            LocalLinearWeightPayload::Quantized { ggml_type, bytes } => {
                assert_eq!(ggml_type, 8);
                assert_eq!(bytes.as_ref(), [1, 2, 3, 4]);
            }
            LocalLinearWeightPayload::F16Bits(_) => {
                panic!("expected quantized payload");
            }
        }
    }

    #[test]
    fn quantized_non_square_output_input_projection_fails_closed() {
        let projection = WhisperDecoderLinearProjectionPlan {
            weight: tensor("decoder.rect.weight", &[6, 4]),
            weight_layout: WhisperDecoderLinearWeightLayout::OutputInput,
            input_dim: 4,
            output_dim: 6,
        };
        let source = QuantizedOnlyTensorSource {
            tensor_name: "decoder.rect.weight".to_string(),
            ggml_type: 8,
            bytes: vec![1, 2, 3, 4],
        };
        let mut cache = WhisperDecoderExecutionTensorCache::default();
        let error = cache
            .materialize_linear_weight_input_output(&source, &projection)
            .expect_err("non-square output-input quantized projection must fail closed");
        assert!(matches!(
            error,
            WhisperDecoderGraphExecutionError::TensorMaterializationFailed { .. }
        ));
        assert!(
            error
                .to_string()
                .contains("quantized decoder weight must be input-output layout"),
            "unexpected error: {error}"
        );
    }

    fn tiny_decoder_plan() -> WhisperDecoderGraphPlan {
        let metadata = WhisperDecoderGraphMetadata {
            decoder_layers: 1,
            decoder_hidden_size: 4,
            decoder_attention_heads: 2,
            vocab_size: 16,
            max_target_positions: 8,
        };
        let binding = WhisperDecoderTensorBindingSeam {
            token_embedding_weight: Some(tensor("model.decoder.embed_tokens.weight", &[16, 4])),
            position_embedding_weight: Some(tensor(
                "model.decoder.embed_positions.weight",
                &[8, 4],
            )),
            final_norm_weight: Some(tensor("model.decoder.layer_norm.weight", &[4])),
            final_norm_bias: Some(tensor("model.decoder.layer_norm.bias", &[4])),
            output_projection_weight: Some(tensor("model.decoder.embed_tokens.weight", &[16, 4])),
            output_projection_bias: None,
            layers: vec![one_layer_binding()],
        };
        let materialization = WhisperDecoderTensorMaterializationSeam {
            source_label: "test-fixture",
            materialized_tensor_count: 32,
        };
        let input_shape = WhisperDecoderGraphInputShape {
            token_count: 3,
            encoder_frames: 2,
            hidden_size: 4,
        };
        build_whisper_decoder_graph_plan(metadata, &binding, &materialization, input_shape)
            .expect("decoder graph plan should succeed")
    }

    fn assert_f32_slice_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (actual - expected).abs();
            assert!(
                delta <= tolerance,
                "mismatch at {idx}: actual={actual} expected={expected} delta={delta}"
            );
        }
    }

    fn one_layer_binding() -> WhisperDecoderLayerTensorBinding {
        WhisperDecoderLayerTensorBinding {
            self_attn_norm_weight: Some(tensor(
                "model.decoder.layers.0.self_attn_layer_norm.weight",
                &[4],
            )),
            self_attn_norm_bias: Some(tensor(
                "model.decoder.layers.0.self_attn_layer_norm.bias",
                &[4],
            )),
            self_attn_q_weight: Some(tensor(
                "model.decoder.layers.0.self_attn.q_proj.weight",
                &[4, 4],
            )),
            self_attn_q_bias: Some(tensor("model.decoder.layers.0.self_attn.q_proj.bias", &[4])),
            self_attn_k_weight: Some(tensor(
                "model.decoder.layers.0.self_attn.k_proj.weight",
                &[4, 4],
            )),
            self_attn_v_weight: Some(tensor(
                "model.decoder.layers.0.self_attn.v_proj.weight",
                &[4, 4],
            )),
            self_attn_v_bias: Some(tensor("model.decoder.layers.0.self_attn.v_proj.bias", &[4])),
            self_attn_out_weight: Some(tensor(
                "model.decoder.layers.0.self_attn.out_proj.weight",
                &[4, 4],
            )),
            self_attn_out_bias: Some(tensor(
                "model.decoder.layers.0.self_attn.out_proj.bias",
                &[4],
            )),
            cross_attn_norm_weight: Some(tensor(
                "model.decoder.layers.0.encoder_attn_layer_norm.weight",
                &[4],
            )),
            cross_attn_norm_bias: Some(tensor(
                "model.decoder.layers.0.encoder_attn_layer_norm.bias",
                &[4],
            )),
            cross_attn_q_weight: Some(tensor(
                "model.decoder.layers.0.encoder_attn.q_proj.weight",
                &[4, 4],
            )),
            cross_attn_q_bias: Some(tensor(
                "model.decoder.layers.0.encoder_attn.q_proj.bias",
                &[4],
            )),
            cross_attn_k_weight: Some(tensor(
                "model.decoder.layers.0.encoder_attn.k_proj.weight",
                &[4, 4],
            )),
            cross_attn_v_weight: Some(tensor(
                "model.decoder.layers.0.encoder_attn.v_proj.weight",
                &[4, 4],
            )),
            cross_attn_v_bias: Some(tensor(
                "model.decoder.layers.0.encoder_attn.v_proj.bias",
                &[4],
            )),
            cross_attn_out_weight: Some(tensor(
                "model.decoder.layers.0.encoder_attn.out_proj.weight",
                &[4, 4],
            )),
            cross_attn_out_bias: Some(tensor(
                "model.decoder.layers.0.encoder_attn.out_proj.bias",
                &[4],
            )),
            mlp_norm_weight: Some(tensor(
                "model.decoder.layers.0.final_layer_norm.weight",
                &[4],
            )),
            mlp_norm_bias: Some(tensor("model.decoder.layers.0.final_layer_norm.bias", &[4])),
            mlp_fc1_weight: Some(tensor("model.decoder.layers.0.fc1.weight", &[8, 4])),
            mlp_fc1_bias: Some(tensor("model.decoder.layers.0.fc1.bias", &[8])),
            mlp_fc2_weight: Some(tensor("model.decoder.layers.0.fc2.weight", &[4, 8])),
            mlp_fc2_bias: Some(tensor("model.decoder.layers.0.fc2.bias", &[4])),
        }
    }

    fn tensor(name: &str, dims: &[u64]) -> WhisperDecoderGraphTensorRef {
        WhisperDecoderGraphTensorRef {
            tensor_name: name.to_string(),
            tensor_num_elements: dims.iter().copied().product::<u64>() as usize,
            dims: dims.to_vec(),
        }
    }

    struct MockTensorSource {
        values: BTreeMap<String, Vec<f32>>,
    }

    impl MockTensorSource {
        fn from_plan(plan: &WhisperDecoderGraphPlan) -> Self {
            let mut values = BTreeMap::new();
            insert_default_tensor(&mut values, &plan.token_embedding.weight);
            insert_default_tensor(&mut values, &plan.position_embedding.weight);
            insert_default_tensor(&mut values, &plan.final_norm.weight);
            insert_default_tensor(&mut values, &plan.final_norm.bias);
            insert_default_tensor(&mut values, &plan.output_projection.projection.weight);
            if let Some(bias) = &plan.output_projection.bias {
                insert_default_tensor(&mut values, bias);
            }
            for layer in &plan.layers {
                insert_default_tensor(&mut values, &layer.self_attn_norm.weight);
                insert_default_tensor(&mut values, &layer.self_attn_norm.bias);
                insert_default_tensor(&mut values, &layer.self_attn_q.projection.weight);
                insert_default_tensor(&mut values, &layer.self_attn_q.bias);
                insert_default_tensor(&mut values, &layer.self_attn_k.weight);
                insert_default_tensor(&mut values, &layer.self_attn_v.projection.weight);
                insert_default_tensor(&mut values, &layer.self_attn_v.bias);
                insert_default_tensor(&mut values, &layer.self_attn_out.projection.weight);
                insert_default_tensor(&mut values, &layer.self_attn_out.bias);
                insert_default_tensor(&mut values, &layer.cross_attn_norm.weight);
                insert_default_tensor(&mut values, &layer.cross_attn_norm.bias);
                insert_default_tensor(&mut values, &layer.cross_attn_q.projection.weight);
                insert_default_tensor(&mut values, &layer.cross_attn_q.bias);
                insert_default_tensor(&mut values, &layer.cross_attn_k.weight);
                insert_default_tensor(&mut values, &layer.cross_attn_v.projection.weight);
                insert_default_tensor(&mut values, &layer.cross_attn_v.bias);
                insert_default_tensor(&mut values, &layer.cross_attn_out.projection.weight);
                insert_default_tensor(&mut values, &layer.cross_attn_out.bias);
                insert_default_tensor(&mut values, &layer.mlp_norm.weight);
                insert_default_tensor(&mut values, &layer.mlp_norm.bias);
                insert_default_tensor(&mut values, &layer.mlp_fc1.projection.weight);
                insert_default_tensor(&mut values, &layer.mlp_fc1.bias);
                insert_default_tensor(&mut values, &layer.mlp_fc2.projection.weight);
                insert_default_tensor(&mut values, &layer.mlp_fc2.bias);
            }
            Self { values }
        }
    }

    impl WhisperDecoderTensorSource for MockTensorSource {
        fn materialize_tensor_f32(
            &self,
            tensor: &WhisperDecoderGraphTensorRef,
        ) -> Result<Vec<f32>, WhisperDecoderGraphExecutionError> {
            let Some(values) = self.values.get(&tensor.tensor_name) else {
                return Err(
                    WhisperDecoderGraphExecutionError::MissingMaterializedTensor {
                        tensor_name: tensor.tensor_name.clone(),
                        reason: "test tensor is absent".to_string(),
                    },
                );
            };
            Ok(values.clone())
        }
    }

    struct CountingTensorSource {
        values: BTreeMap<String, Vec<f32>>,
        counts: RefCell<BTreeMap<String, usize>>,
    }

    impl CountingTensorSource {
        fn from_plan(plan: &WhisperDecoderGraphPlan) -> Self {
            Self {
                values: MockTensorSource::from_plan(plan).values,
                counts: RefCell::new(BTreeMap::new()),
            }
        }

        fn count_for(&self, tensor_name: &str) -> usize {
            self.counts.borrow().get(tensor_name).copied().unwrap_or(0)
        }
    }

    impl WhisperDecoderTensorSource for CountingTensorSource {
        fn materialize_tensor_f32(
            &self,
            tensor: &WhisperDecoderGraphTensorRef,
        ) -> Result<Vec<f32>, WhisperDecoderGraphExecutionError> {
            *self
                .counts
                .borrow_mut()
                .entry(tensor.tensor_name.clone())
                .or_default() += 1;
            let Some(values) = self.values.get(&tensor.tensor_name) else {
                return Err(
                    WhisperDecoderGraphExecutionError::MissingMaterializedTensor {
                        tensor_name: tensor.tensor_name.clone(),
                        reason: "test tensor is absent".to_string(),
                    },
                );
            };
            Ok(values.clone())
        }
    }

    struct QuantizedOnlyTensorSource {
        tensor_name: String,
        ggml_type: i32,
        bytes: Vec<u8>,
    }

    impl WhisperDecoderTensorSource for QuantizedOnlyTensorSource {
        fn materialize_tensor_f32(
            &self,
            tensor: &WhisperDecoderGraphTensorRef,
        ) -> Result<Vec<f32>, WhisperDecoderGraphExecutionError> {
            Err(
                WhisperDecoderGraphExecutionError::MissingMaterializedTensor {
                    tensor_name: tensor.tensor_name.clone(),
                    reason: "quantized-only test source has no f32 tensors".to_string(),
                },
            )
        }

        fn materialize_tensor_quantized(
            &self,
            tensor: &WhisperDecoderGraphTensorRef,
        ) -> Result<Option<(i32, Vec<u8>)>, WhisperDecoderGraphExecutionError> {
            if tensor.tensor_name == self.tensor_name {
                return Ok(Some((self.ggml_type, self.bytes.clone())));
            }
            Ok(None)
        }
    }

    fn insert_default_tensor(
        values: &mut BTreeMap<String, Vec<f32>>,
        tensor: &WhisperDecoderGraphTensorRef,
    ) {
        let data = if tensor.tensor_name.contains("norm.weight") {
            vec![1.0f32; tensor.tensor_num_elements]
        } else if tensor.tensor_name.contains("norm.bias") || tensor.tensor_name.ends_with(".bias")
        {
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
                .map(|idx| ((idx % 13) as f32) * 0.01)
                .collect()
        };
        values.insert(tensor.tensor_name.clone(), data);
    }
}
