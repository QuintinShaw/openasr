use std::collections::BTreeMap;

use thiserror::Error;

use crate::{GgufTensorIndex, GgufTensorMetadata};

const GGML_TYPE_F32: i32 = 0;
const GGML_TYPE_F16: i32 = 1;
const GGML_TYPE_BF16: i32 = 30;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperGgufTensorBindingContext {
    pub n_audio_layer: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_text_layer: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_ctx: usize,
    pub n_vocab: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum WhisperGgufTensorSlot {
    EncoderConv1Weight,
    EncoderConv1Bias,
    EncoderConv2Weight,
    EncoderConv2Bias,
    EncoderPositionalEmbedding,
    EncoderLayerSelfAttnNormWeight { layer_idx: usize },
    EncoderLayerSelfAttnNormBias { layer_idx: usize },
    EncoderLayerSelfAttnQWeight { layer_idx: usize },
    EncoderLayerSelfAttnQBias { layer_idx: usize },
    EncoderLayerSelfAttnKWeight { layer_idx: usize },
    EncoderLayerSelfAttnVWeight { layer_idx: usize },
    EncoderLayerSelfAttnVBias { layer_idx: usize },
    EncoderLayerSelfAttnOutWeight { layer_idx: usize },
    EncoderLayerSelfAttnOutBias { layer_idx: usize },
    EncoderLayerMlpNormWeight { layer_idx: usize },
    EncoderLayerMlpNormBias { layer_idx: usize },
    EncoderLayerMlpFc1Weight { layer_idx: usize },
    EncoderLayerMlpFc1Bias { layer_idx: usize },
    EncoderLayerMlpFc2Weight { layer_idx: usize },
    EncoderLayerMlpFc2Bias { layer_idx: usize },
    EncoderFinalLayerNormWeight,
    EncoderFinalLayerNormBias,
    DecoderTokenEmbedding,
    DecoderPositionalEmbedding,
    DecoderLayerSelfAttnNormWeight { layer_idx: usize },
    DecoderLayerSelfAttnNormBias { layer_idx: usize },
    DecoderLayerSelfAttnQWeight { layer_idx: usize },
    DecoderLayerSelfAttnQBias { layer_idx: usize },
    DecoderLayerSelfAttnKWeight { layer_idx: usize },
    DecoderLayerSelfAttnVWeight { layer_idx: usize },
    DecoderLayerSelfAttnVBias { layer_idx: usize },
    DecoderLayerSelfAttnOutWeight { layer_idx: usize },
    DecoderLayerSelfAttnOutBias { layer_idx: usize },
    DecoderLayerCrossAttnNormWeight { layer_idx: usize },
    DecoderLayerCrossAttnNormBias { layer_idx: usize },
    DecoderLayerCrossAttnQWeight { layer_idx: usize },
    DecoderLayerCrossAttnQBias { layer_idx: usize },
    DecoderLayerCrossAttnKWeight { layer_idx: usize },
    DecoderLayerCrossAttnVWeight { layer_idx: usize },
    DecoderLayerCrossAttnVBias { layer_idx: usize },
    DecoderLayerCrossAttnOutWeight { layer_idx: usize },
    DecoderLayerCrossAttnOutBias { layer_idx: usize },
    DecoderLayerMlpNormWeight { layer_idx: usize },
    DecoderLayerMlpNormBias { layer_idx: usize },
    DecoderLayerMlpFc1Weight { layer_idx: usize },
    DecoderLayerMlpFc1Bias { layer_idx: usize },
    DecoderLayerMlpFc2Weight { layer_idx: usize },
    DecoderLayerMlpFc2Bias { layer_idx: usize },
    DecoderFinalLayerNormWeight,
    DecoderFinalLayerNormBias,
    DecoderOutputProjectionWeight,
}

impl WhisperGgufTensorSlot {
    pub(crate) fn label(&self) -> String {
        match self {
            Self::EncoderConv1Weight => "encoder.conv1.weight".to_string(),
            Self::EncoderConv1Bias => "encoder.conv1.bias".to_string(),
            Self::EncoderConv2Weight => "encoder.conv2.weight".to_string(),
            Self::EncoderConv2Bias => "encoder.conv2.bias".to_string(),
            Self::EncoderPositionalEmbedding => "encoder.positional_embedding".to_string(),
            Self::EncoderLayerSelfAttnNormWeight { layer_idx } => {
                format!("encoder.layers.{layer_idx}.self_attn_layer_norm.weight")
            }
            Self::EncoderLayerSelfAttnNormBias { layer_idx } => {
                format!("encoder.layers.{layer_idx}.self_attn_layer_norm.bias")
            }
            Self::EncoderLayerSelfAttnQWeight { layer_idx } => {
                format!("encoder.layers.{layer_idx}.self_attn.q_proj.weight")
            }
            Self::EncoderLayerSelfAttnQBias { layer_idx } => {
                format!("encoder.layers.{layer_idx}.self_attn.q_proj.bias")
            }
            Self::EncoderLayerSelfAttnKWeight { layer_idx } => {
                format!("encoder.layers.{layer_idx}.self_attn.k_proj.weight")
            }
            Self::EncoderLayerSelfAttnVWeight { layer_idx } => {
                format!("encoder.layers.{layer_idx}.self_attn.v_proj.weight")
            }
            Self::EncoderLayerSelfAttnVBias { layer_idx } => {
                format!("encoder.layers.{layer_idx}.self_attn.v_proj.bias")
            }
            Self::EncoderLayerSelfAttnOutWeight { layer_idx } => {
                format!("encoder.layers.{layer_idx}.self_attn.out_proj.weight")
            }
            Self::EncoderLayerSelfAttnOutBias { layer_idx } => {
                format!("encoder.layers.{layer_idx}.self_attn.out_proj.bias")
            }
            Self::EncoderLayerMlpNormWeight { layer_idx } => {
                format!("encoder.layers.{layer_idx}.final_layer_norm.weight")
            }
            Self::EncoderLayerMlpNormBias { layer_idx } => {
                format!("encoder.layers.{layer_idx}.final_layer_norm.bias")
            }
            Self::EncoderLayerMlpFc1Weight { layer_idx } => {
                format!("encoder.layers.{layer_idx}.fc1.weight")
            }
            Self::EncoderLayerMlpFc1Bias { layer_idx } => {
                format!("encoder.layers.{layer_idx}.fc1.bias")
            }
            Self::EncoderLayerMlpFc2Weight { layer_idx } => {
                format!("encoder.layers.{layer_idx}.fc2.weight")
            }
            Self::EncoderLayerMlpFc2Bias { layer_idx } => {
                format!("encoder.layers.{layer_idx}.fc2.bias")
            }
            Self::EncoderFinalLayerNormWeight => "encoder.layer_norm.weight".to_string(),
            Self::EncoderFinalLayerNormBias => "encoder.layer_norm.bias".to_string(),
            Self::DecoderTokenEmbedding => "decoder.token_embedding.weight".to_string(),
            Self::DecoderPositionalEmbedding => "decoder.positional_embedding".to_string(),
            Self::DecoderLayerSelfAttnNormWeight { layer_idx } => {
                format!("decoder.layers.{layer_idx}.self_attn_layer_norm.weight")
            }
            Self::DecoderLayerSelfAttnNormBias { layer_idx } => {
                format!("decoder.layers.{layer_idx}.self_attn_layer_norm.bias")
            }
            Self::DecoderLayerSelfAttnQWeight { layer_idx } => {
                format!("decoder.layers.{layer_idx}.self_attn.q_proj.weight")
            }
            Self::DecoderLayerSelfAttnQBias { layer_idx } => {
                format!("decoder.layers.{layer_idx}.self_attn.q_proj.bias")
            }
            Self::DecoderLayerSelfAttnKWeight { layer_idx } => {
                format!("decoder.layers.{layer_idx}.self_attn.k_proj.weight")
            }
            Self::DecoderLayerSelfAttnVWeight { layer_idx } => {
                format!("decoder.layers.{layer_idx}.self_attn.v_proj.weight")
            }
            Self::DecoderLayerSelfAttnVBias { layer_idx } => {
                format!("decoder.layers.{layer_idx}.self_attn.v_proj.bias")
            }
            Self::DecoderLayerSelfAttnOutWeight { layer_idx } => {
                format!("decoder.layers.{layer_idx}.self_attn.out_proj.weight")
            }
            Self::DecoderLayerSelfAttnOutBias { layer_idx } => {
                format!("decoder.layers.{layer_idx}.self_attn.out_proj.bias")
            }
            Self::DecoderLayerCrossAttnNormWeight { layer_idx } => {
                format!("decoder.layers.{layer_idx}.encoder_attn_layer_norm.weight")
            }
            Self::DecoderLayerCrossAttnNormBias { layer_idx } => {
                format!("decoder.layers.{layer_idx}.encoder_attn_layer_norm.bias")
            }
            Self::DecoderLayerCrossAttnQWeight { layer_idx } => {
                format!("decoder.layers.{layer_idx}.cross_attn.q_proj.weight")
            }
            Self::DecoderLayerCrossAttnQBias { layer_idx } => {
                format!("decoder.layers.{layer_idx}.cross_attn.q_proj.bias")
            }
            Self::DecoderLayerCrossAttnKWeight { layer_idx } => {
                format!("decoder.layers.{layer_idx}.cross_attn.k_proj.weight")
            }
            Self::DecoderLayerCrossAttnVWeight { layer_idx } => {
                format!("decoder.layers.{layer_idx}.cross_attn.v_proj.weight")
            }
            Self::DecoderLayerCrossAttnVBias { layer_idx } => {
                format!("decoder.layers.{layer_idx}.cross_attn.v_proj.bias")
            }
            Self::DecoderLayerCrossAttnOutWeight { layer_idx } => {
                format!("decoder.layers.{layer_idx}.cross_attn.out_proj.weight")
            }
            Self::DecoderLayerCrossAttnOutBias { layer_idx } => {
                format!("decoder.layers.{layer_idx}.cross_attn.out_proj.bias")
            }
            Self::DecoderLayerMlpNormWeight { layer_idx } => {
                format!("decoder.layers.{layer_idx}.final_layer_norm.weight")
            }
            Self::DecoderLayerMlpNormBias { layer_idx } => {
                format!("decoder.layers.{layer_idx}.final_layer_norm.bias")
            }
            Self::DecoderLayerMlpFc1Weight { layer_idx } => {
                format!("decoder.layers.{layer_idx}.fc1.weight")
            }
            Self::DecoderLayerMlpFc1Bias { layer_idx } => {
                format!("decoder.layers.{layer_idx}.fc1.bias")
            }
            Self::DecoderLayerMlpFc2Weight { layer_idx } => {
                format!("decoder.layers.{layer_idx}.fc2.weight")
            }
            Self::DecoderLayerMlpFc2Bias { layer_idx } => {
                format!("decoder.layers.{layer_idx}.fc2.bias")
            }
            Self::DecoderFinalLayerNormWeight => "decoder.layer_norm.weight".to_string(),
            Self::DecoderFinalLayerNormBias => "decoder.layer_norm.bias".to_string(),
            Self::DecoderOutputProjectionWeight => "decoder.output_projection.weight".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperGgufTensorBinding {
    pub slot: WhisperGgufTensorSlot,
    pub resolved_name: String,
    pub metadata: GgufTensorMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperGgufEncoderPreludeTensorBindings {
    pub conv1_weight: WhisperGgufTensorBinding,
    pub conv1_bias: WhisperGgufTensorBinding,
    pub conv2_weight: WhisperGgufTensorBinding,
    pub conv2_bias: WhisperGgufTensorBinding,
    pub positional_embedding: WhisperGgufTensorBinding,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperGgufEncoderLayerTensorBindings {
    pub layer_idx: usize,
    pub self_attn_layer_norm_weight: WhisperGgufTensorBinding,
    pub self_attn_layer_norm_bias: WhisperGgufTensorBinding,
    pub self_attn_q_weight: WhisperGgufTensorBinding,
    pub self_attn_q_bias: WhisperGgufTensorBinding,
    pub self_attn_k_weight: WhisperGgufTensorBinding,
    pub self_attn_v_weight: WhisperGgufTensorBinding,
    pub self_attn_v_bias: WhisperGgufTensorBinding,
    pub self_attn_out_weight: WhisperGgufTensorBinding,
    pub self_attn_out_bias: WhisperGgufTensorBinding,
    pub mlp_norm_weight: WhisperGgufTensorBinding,
    pub mlp_norm_bias: WhisperGgufTensorBinding,
    pub fc1_weight: WhisperGgufTensorBinding,
    pub fc1_bias: WhisperGgufTensorBinding,
    pub fc2_weight: WhisperGgufTensorBinding,
    pub fc2_bias: WhisperGgufTensorBinding,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperGgufEncoderTensorBindings {
    pub n_audio_layer: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_ctx: usize,
    pub n_mels: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub prelude: WhisperGgufEncoderPreludeTensorBindings,
    pub layers: Vec<WhisperGgufEncoderLayerTensorBindings>,
    pub final_layer_norm_weight: WhisperGgufTensorBinding,
    pub final_layer_norm_bias: WhisperGgufTensorBinding,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperGgufDecoderLayerTensorBindings {
    pub layer_idx: usize,
    pub self_attn_layer_norm_weight: WhisperGgufTensorBinding,
    pub self_attn_layer_norm_bias: WhisperGgufTensorBinding,
    pub self_attn_q_weight: WhisperGgufTensorBinding,
    pub self_attn_q_bias: WhisperGgufTensorBinding,
    pub self_attn_k_weight: WhisperGgufTensorBinding,
    pub self_attn_v_weight: WhisperGgufTensorBinding,
    pub self_attn_v_bias: WhisperGgufTensorBinding,
    pub self_attn_out_weight: WhisperGgufTensorBinding,
    pub self_attn_out_bias: WhisperGgufTensorBinding,
    pub cross_attn_layer_norm_weight: WhisperGgufTensorBinding,
    pub cross_attn_layer_norm_bias: WhisperGgufTensorBinding,
    pub cross_attn_q_weight: WhisperGgufTensorBinding,
    pub cross_attn_q_bias: WhisperGgufTensorBinding,
    pub cross_attn_k_weight: WhisperGgufTensorBinding,
    pub cross_attn_v_weight: WhisperGgufTensorBinding,
    pub cross_attn_v_bias: WhisperGgufTensorBinding,
    pub cross_attn_out_weight: WhisperGgufTensorBinding,
    pub cross_attn_out_bias: WhisperGgufTensorBinding,
    pub mlp_norm_weight: WhisperGgufTensorBinding,
    pub mlp_norm_bias: WhisperGgufTensorBinding,
    pub fc1_weight: WhisperGgufTensorBinding,
    pub fc1_bias: WhisperGgufTensorBinding,
    pub fc2_weight: WhisperGgufTensorBinding,
    pub fc2_bias: WhisperGgufTensorBinding,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperGgufDecoderTensorBindings {
    pub n_text_layer: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_ctx: usize,
    pub n_vocab: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub token_embedding: WhisperGgufTensorBinding,
    pub positional_embedding: WhisperGgufTensorBinding,
    pub layers: Vec<WhisperGgufDecoderLayerTensorBindings>,
    pub final_layer_norm_weight: WhisperGgufTensorBinding,
    pub final_layer_norm_bias: WhisperGgufTensorBinding,
    pub output_projection_weight: WhisperGgufTensorBinding,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperGgufTensorBindings {
    encoder: WhisperGgufEncoderTensorBindings,
    decoder: WhisperGgufDecoderTensorBindings,
}

impl WhisperGgufTensorBindings {
    pub(crate) fn encoder(&self) -> &WhisperGgufEncoderTensorBindings {
        &self.encoder
    }

    pub(crate) fn decoder(&self) -> &WhisperGgufDecoderTensorBindings {
        &self.decoder
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum WhisperGgufTensorBindingError {
    #[error("invalid whisper gguf tensor binding context: {field} {reason}")]
    InvalidContext { field: &'static str, reason: String },
    #[error("missing required whisper gguf tensor slot '{slot}' (aliases: {aliases:?})")]
    MissingRequiredTensor { slot: String, aliases: Vec<String> },
    #[error(
        "whisper gguf tensor slot '{slot}' resolved to '{tensor_name}' with unsupported type '{found_type}' (expected {expected})"
    )]
    TensorTypeMismatch {
        slot: String,
        tensor_name: String,
        found_type: String,
        expected: String,
    },
    #[error(
        "whisper gguf tensor slot '{slot}' resolved to '{tensor_name}' has invalid shape {found_shape:?} (expected {expected})"
    )]
    TensorShapeMismatch {
        slot: String,
        tensor_name: String,
        found_shape: Vec<u64>,
        expected: String,
    },
    #[error("whisper gguf encoder layer {layer_idx} shape invariant failed: {reason}")]
    EncoderLayerInvariant { layer_idx: usize, reason: String },
    #[error("whisper gguf decoder layer {layer_idx} shape invariant failed: {reason}")]
    DecoderLayerInvariant { layer_idx: usize, reason: String },
    #[error("whisper gguf decoder binding invariant failed: {reason}")]
    DecoderInvariant { reason: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorTypeConstraint {
    Any,
    FloatLike,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TensorShapeConstraint {
    ExactAnyOf(Vec<Vec<u64>>),
    Rank1OrRank2LastDim(u64),
    Rank1OrRank2,
    Rank2Exact(u64, u64),
    Rank2AnyOrder(u64, u64),
    Rank2EitherDim(u64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WhisperTensorSlotDescriptor {
    slot: WhisperGgufTensorSlot,
    aliases: Vec<String>,
    type_constraint: TensorTypeConstraint,
    shape_constraint: TensorShapeConstraint,
}

pub(crate) fn bind_whisper_gguf_tensors(
    context: &WhisperGgufTensorBindingContext,
    index: &GgufTensorIndex,
) -> Result<WhisperGgufTensorBindings, WhisperGgufTensorBindingError> {
    validate_binding_context(context)?;

    let mut by_slot = BTreeMap::new();
    for descriptor in whisper_tensor_slot_descriptors(context) {
        let Some((resolved_name, metadata)) = resolve_alias(index, &descriptor.aliases) else {
            return Err(WhisperGgufTensorBindingError::MissingRequiredTensor {
                slot: descriptor.slot.label(),
                aliases: descriptor.aliases,
            });
        };
        validate_tensor_type(&descriptor, metadata)?;
        validate_tensor_shape(&descriptor, metadata)?;
        by_slot.insert(
            descriptor.slot.clone(),
            WhisperGgufTensorBinding {
                slot: descriptor.slot,
                resolved_name,
                metadata: metadata.clone(),
            },
        );
    }

    let encoder = build_encoder_tensor_bindings(context, &by_slot)?;
    let decoder = build_decoder_tensor_bindings(context, &by_slot)?;

    Ok(WhisperGgufTensorBindings { encoder, decoder })
}

fn validate_binding_context(
    context: &WhisperGgufTensorBindingContext,
) -> Result<(), WhisperGgufTensorBindingError> {
    if context.n_audio_layer == 0 {
        return Err(WhisperGgufTensorBindingError::InvalidContext {
            field: "n_audio_layer",
            reason: "must be > 0".to_string(),
        });
    }
    if context.n_audio_state == 0 {
        return Err(WhisperGgufTensorBindingError::InvalidContext {
            field: "n_audio_state",
            reason: "must be > 0".to_string(),
        });
    }
    if context.n_audio_head == 0 {
        return Err(WhisperGgufTensorBindingError::InvalidContext {
            field: "n_audio_head",
            reason: "must be > 0".to_string(),
        });
    }
    if !context.n_audio_state.is_multiple_of(context.n_audio_head) {
        return Err(WhisperGgufTensorBindingError::InvalidContext {
            field: "n_audio_head",
            reason: format!(
                "n_audio_state {} must be divisible by n_audio_head {}",
                context.n_audio_state, context.n_audio_head
            ),
        });
    }
    if context.n_mels == 0 {
        return Err(WhisperGgufTensorBindingError::InvalidContext {
            field: "n_mels",
            reason: "must be > 0".to_string(),
        });
    }
    if context.n_audio_ctx == 0 {
        return Err(WhisperGgufTensorBindingError::InvalidContext {
            field: "n_audio_ctx",
            reason: "must be > 0".to_string(),
        });
    }
    if context.n_text_layer == 0 {
        return Err(WhisperGgufTensorBindingError::InvalidContext {
            field: "n_text_layer",
            reason: "must be > 0".to_string(),
        });
    }
    if context.n_text_state == 0 {
        return Err(WhisperGgufTensorBindingError::InvalidContext {
            field: "n_text_state",
            reason: "must be > 0".to_string(),
        });
    }
    if context.n_text_head == 0 {
        return Err(WhisperGgufTensorBindingError::InvalidContext {
            field: "n_text_head",
            reason: "must be > 0".to_string(),
        });
    }
    if !context.n_text_state.is_multiple_of(context.n_text_head) {
        return Err(WhisperGgufTensorBindingError::InvalidContext {
            field: "n_text_head",
            reason: format!(
                "n_text_state {} must be divisible by n_text_head {}",
                context.n_text_state, context.n_text_head
            ),
        });
    }
    if context.n_text_ctx == 0 {
        return Err(WhisperGgufTensorBindingError::InvalidContext {
            field: "n_text_ctx",
            reason: "must be > 0".to_string(),
        });
    }
    if context.n_vocab == 0 {
        return Err(WhisperGgufTensorBindingError::InvalidContext {
            field: "n_vocab",
            reason: "must be > 0".to_string(),
        });
    }
    Ok(())
}

fn build_encoder_tensor_bindings(
    context: &WhisperGgufTensorBindingContext,
    by_slot: &BTreeMap<WhisperGgufTensorSlot, WhisperGgufTensorBinding>,
) -> Result<WhisperGgufEncoderTensorBindings, WhisperGgufTensorBindingError> {
    let prelude = WhisperGgufEncoderPreludeTensorBindings {
        conv1_weight: required_slot(by_slot, &WhisperGgufTensorSlot::EncoderConv1Weight)?,
        conv1_bias: required_slot(by_slot, &WhisperGgufTensorSlot::EncoderConv1Bias)?,
        conv2_weight: required_slot(by_slot, &WhisperGgufTensorSlot::EncoderConv2Weight)?,
        conv2_bias: required_slot(by_slot, &WhisperGgufTensorSlot::EncoderConv2Bias)?,
        positional_embedding: required_slot(
            by_slot,
            &WhisperGgufTensorSlot::EncoderPositionalEmbedding,
        )?,
    };

    let mut ffn_dim = None;
    let mut layers = Vec::with_capacity(context.n_audio_layer);
    for layer_idx in 0..context.n_audio_layer {
        let layer = WhisperGgufEncoderLayerTensorBindings {
            layer_idx,
            self_attn_layer_norm_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::EncoderLayerSelfAttnNormWeight { layer_idx },
            )?,
            self_attn_layer_norm_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::EncoderLayerSelfAttnNormBias { layer_idx },
            )?,
            self_attn_q_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::EncoderLayerSelfAttnQWeight { layer_idx },
            )?,
            self_attn_q_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::EncoderLayerSelfAttnQBias { layer_idx },
            )?,
            self_attn_k_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::EncoderLayerSelfAttnKWeight { layer_idx },
            )?,
            self_attn_v_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::EncoderLayerSelfAttnVWeight { layer_idx },
            )?,
            self_attn_v_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::EncoderLayerSelfAttnVBias { layer_idx },
            )?,
            self_attn_out_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::EncoderLayerSelfAttnOutWeight { layer_idx },
            )?,
            self_attn_out_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::EncoderLayerSelfAttnOutBias { layer_idx },
            )?,
            mlp_norm_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::EncoderLayerMlpNormWeight { layer_idx },
            )?,
            mlp_norm_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::EncoderLayerMlpNormBias { layer_idx },
            )?,
            fc1_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::EncoderLayerMlpFc1Weight { layer_idx },
            )?,
            fc1_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::EncoderLayerMlpFc1Bias { layer_idx },
            )?,
            fc2_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::EncoderLayerMlpFc2Weight { layer_idx },
            )?,
            fc2_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::EncoderLayerMlpFc2Bias { layer_idx },
            )?,
        };

        let layer_ffn_dim = infer_and_validate_encoder_layer_ffn_dim(context, &layer)?;
        match ffn_dim {
            Some(expected) if expected != layer_ffn_dim => {
                return Err(WhisperGgufTensorBindingError::EncoderLayerInvariant {
                    layer_idx,
                    reason: format!(
                        "fc1/fc2 imply ffn_dim={layer_ffn_dim}, expected {expected} from earlier layers"
                    ),
                });
            }
            Some(_) => {}
            None => ffn_dim = Some(layer_ffn_dim),
        }

        layers.push(layer);
    }

    let final_layer_norm_weight =
        required_slot(by_slot, &WhisperGgufTensorSlot::EncoderFinalLayerNormWeight)?;
    let final_layer_norm_bias =
        required_slot(by_slot, &WhisperGgufTensorSlot::EncoderFinalLayerNormBias)?;

    Ok(WhisperGgufEncoderTensorBindings {
        n_audio_layer: context.n_audio_layer,
        n_audio_state: context.n_audio_state,
        n_audio_head: context.n_audio_head,
        n_audio_ctx: context.n_audio_ctx,
        n_mels: context.n_mels,
        head_dim: context.n_audio_state / context.n_audio_head,
        ffn_dim: ffn_dim.unwrap_or(context.n_audio_state),
        prelude,
        layers,
        final_layer_norm_weight,
        final_layer_norm_bias,
    })
}

fn infer_and_validate_encoder_layer_ffn_dim(
    context: &WhisperGgufTensorBindingContext,
    layer: &WhisperGgufEncoderLayerTensorBindings,
) -> Result<usize, WhisperGgufTensorBindingError> {
    infer_layer_ffn_dim(
        context.n_audio_state,
        layer.layer_idx,
        &layer.fc1_weight.metadata.dims,
        &layer.fc1_bias.metadata.dims,
        &layer.fc2_weight.metadata.dims,
        &layer.fc2_bias.metadata.dims,
        |layer_idx, reason| WhisperGgufTensorBindingError::EncoderLayerInvariant {
            layer_idx,
            reason,
        },
    )
}

fn rank1_or_rank2_vector_matches(dims: &[u64], expected: u64) -> bool {
    match dims {
        [dim] => *dim == expected,
        [d0, d1] => (*d0 == expected && *d1 == 1) || (*d0 == 1 && *d1 == expected),
        _ => false,
    }
}

fn infer_layer_ffn_dim<E, F>(
    hidden_size: usize,
    layer_idx: usize,
    fc1_dims: &[u64],
    fc1_bias_dims: &[u64],
    fc2_dims: &[u64],
    fc2_bias_dims: &[u64],
    make_error: F,
) -> Result<usize, E>
where
    F: Fn(usize, String) -> E,
{
    let [fc1_dim0_u64, fc1_dim1_u64] = fc1_dims else {
        return Err(make_error(
            layer_idx,
            format!("fc1.weight must be rank-2, got {fc1_dims:?}"),
        ));
    };
    let hidden_u64 = hidden_size as u64;
    let fc1_out_u64 = if *fc1_dim0_u64 == hidden_u64 {
        *fc1_dim1_u64
    } else if *fc1_dim1_u64 == hidden_u64 {
        *fc1_dim0_u64
    } else {
        return Err(make_error(
            layer_idx,
            format!("fc1.weight shape {fc1_dims:?} must contain hidden_size={hidden_size}",),
        ));
    };

    let ffn_dim = usize::try_from(fc1_out_u64).map_err(|_| {
        make_error(
            layer_idx,
            format!("fc1.weight projected dim {fc1_out_u64} does not fit usize"),
        )
    })?;
    if ffn_dim == 0 {
        return Err(make_error(
            layer_idx,
            "fc1.weight projected dim must be > 0".to_string(),
        ));
    }

    if !rank1_or_rank2_vector_matches(fc1_bias_dims, fc1_out_u64) {
        return Err(make_error(
            layer_idx,
            format!("fc1.bias shape {fc1_bias_dims:?} must match inferred ffn_dim={ffn_dim}",),
        ));
    }

    let [fc2_dim0_u64, fc2_dim1_u64] = fc2_dims else {
        return Err(make_error(
            layer_idx,
            format!("fc2.weight must be rank-2, got {fc2_dims:?}"),
        ));
    };
    let fc2_matches = (*fc2_dim0_u64 == hidden_u64 && *fc2_dim1_u64 == fc1_out_u64)
        || (*fc2_dim0_u64 == fc1_out_u64 && *fc2_dim1_u64 == hidden_u64);
    if !fc2_matches {
        return Err(make_error(
            layer_idx,
            format!(
                "fc2.weight shape {fc2_dims:?} must contain hidden_size={hidden_size} and ffn_dim={ffn_dim}",
            ),
        ));
    }
    if !rank1_or_rank2_vector_matches(fc2_bias_dims, hidden_u64) {
        return Err(make_error(
            layer_idx,
            format!("fc2.bias shape {fc2_bias_dims:?} must match hidden_size={hidden_size}"),
        ));
    }

    Ok(ffn_dim)
}

fn has_rank2_exact(dims: &[u64], one: [u64; 2], other: [u64; 2]) -> bool {
    matches!(dims, [d0, d1] if [*d0, *d1] == one || [*d0, *d1] == other)
}

fn build_decoder_tensor_bindings(
    context: &WhisperGgufTensorBindingContext,
    by_slot: &BTreeMap<WhisperGgufTensorSlot, WhisperGgufTensorBinding>,
) -> Result<WhisperGgufDecoderTensorBindings, WhisperGgufTensorBindingError> {
    let token_embedding = required_slot(by_slot, &WhisperGgufTensorSlot::DecoderTokenEmbedding)?;
    let positional_embedding =
        required_slot(by_slot, &WhisperGgufTensorSlot::DecoderPositionalEmbedding)?;
    let output_projection_weight = required_slot(
        by_slot,
        &WhisperGgufTensorSlot::DecoderOutputProjectionWeight,
    )?;
    let final_layer_norm_weight =
        required_slot(by_slot, &WhisperGgufTensorSlot::DecoderFinalLayerNormWeight)?;
    let final_layer_norm_bias =
        required_slot(by_slot, &WhisperGgufTensorSlot::DecoderFinalLayerNormBias)?;

    let mut ffn_dim = None;
    let mut layers = Vec::with_capacity(context.n_text_layer);
    for layer_idx in 0..context.n_text_layer {
        let layer = WhisperGgufDecoderLayerTensorBindings {
            layer_idx,
            self_attn_layer_norm_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerSelfAttnNormWeight { layer_idx },
            )?,
            self_attn_layer_norm_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerSelfAttnNormBias { layer_idx },
            )?,
            self_attn_q_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerSelfAttnQWeight { layer_idx },
            )?,
            self_attn_q_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerSelfAttnQBias { layer_idx },
            )?,
            self_attn_k_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerSelfAttnKWeight { layer_idx },
            )?,
            self_attn_v_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerSelfAttnVWeight { layer_idx },
            )?,
            self_attn_v_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerSelfAttnVBias { layer_idx },
            )?,
            self_attn_out_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerSelfAttnOutWeight { layer_idx },
            )?,
            self_attn_out_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerSelfAttnOutBias { layer_idx },
            )?,
            cross_attn_layer_norm_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerCrossAttnNormWeight { layer_idx },
            )?,
            cross_attn_layer_norm_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerCrossAttnNormBias { layer_idx },
            )?,
            cross_attn_q_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerCrossAttnQWeight { layer_idx },
            )?,
            cross_attn_q_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerCrossAttnQBias { layer_idx },
            )?,
            cross_attn_k_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerCrossAttnKWeight { layer_idx },
            )?,
            cross_attn_v_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerCrossAttnVWeight { layer_idx },
            )?,
            cross_attn_v_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerCrossAttnVBias { layer_idx },
            )?,
            cross_attn_out_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerCrossAttnOutWeight { layer_idx },
            )?,
            cross_attn_out_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerCrossAttnOutBias { layer_idx },
            )?,
            mlp_norm_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerMlpNormWeight { layer_idx },
            )?,
            mlp_norm_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerMlpNormBias { layer_idx },
            )?,
            fc1_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerMlpFc1Weight { layer_idx },
            )?,
            fc1_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerMlpFc1Bias { layer_idx },
            )?,
            fc2_weight: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerMlpFc2Weight { layer_idx },
            )?,
            fc2_bias: required_slot(
                by_slot,
                &WhisperGgufTensorSlot::DecoderLayerMlpFc2Bias { layer_idx },
            )?,
        };

        let layer_ffn_dim = infer_and_validate_decoder_layer_ffn_dim(context, &layer)?;
        match ffn_dim {
            Some(expected) if expected != layer_ffn_dim => {
                return Err(WhisperGgufTensorBindingError::DecoderLayerInvariant {
                    layer_idx,
                    reason: format!(
                        "fc1/fc2 imply ffn_dim={layer_ffn_dim}, expected {expected} from earlier layers"
                    ),
                });
            }
            Some(_) => {}
            None => ffn_dim = Some(layer_ffn_dim),
        }

        layers.push(layer);
    }

    let hidden_u64 = context.n_text_state as u64;
    let vocab_u64 = context.n_vocab as u64;
    if !has_rank2_exact(
        &token_embedding.metadata.dims,
        [vocab_u64, hidden_u64],
        [hidden_u64, vocab_u64],
    ) {
        return Err(WhisperGgufTensorBindingError::DecoderInvariant {
            reason: format!(
                "token embedding shape {:?} must be [{}, {}] or [{}, {}]",
                token_embedding.metadata.dims,
                context.n_vocab,
                context.n_text_state,
                context.n_text_state,
                context.n_vocab
            ),
        });
    }
    if token_embedding.metadata.dims != output_projection_weight.metadata.dims {
        return Err(WhisperGgufTensorBindingError::DecoderInvariant {
            reason: format!(
                "output projection shape {:?} must match token embedding shape {:?}",
                output_projection_weight.metadata.dims, token_embedding.metadata.dims
            ),
        });
    }

    Ok(WhisperGgufDecoderTensorBindings {
        n_text_layer: context.n_text_layer,
        n_text_state: context.n_text_state,
        n_text_head: context.n_text_head,
        n_text_ctx: context.n_text_ctx,
        n_vocab: context.n_vocab,
        head_dim: context.n_text_state / context.n_text_head,
        ffn_dim: ffn_dim.unwrap_or(context.n_text_state),
        token_embedding,
        positional_embedding,
        layers,
        final_layer_norm_weight,
        final_layer_norm_bias,
        output_projection_weight,
    })
}

fn infer_and_validate_decoder_layer_ffn_dim(
    context: &WhisperGgufTensorBindingContext,
    layer: &WhisperGgufDecoderLayerTensorBindings,
) -> Result<usize, WhisperGgufTensorBindingError> {
    infer_layer_ffn_dim(
        context.n_text_state,
        layer.layer_idx,
        &layer.fc1_weight.metadata.dims,
        &layer.fc1_bias.metadata.dims,
        &layer.fc2_weight.metadata.dims,
        &layer.fc2_bias.metadata.dims,
        |layer_idx, reason| WhisperGgufTensorBindingError::DecoderLayerInvariant {
            layer_idx,
            reason,
        },
    )
}

fn required_slot(
    by_slot: &BTreeMap<WhisperGgufTensorSlot, WhisperGgufTensorBinding>,
    slot: &WhisperGgufTensorSlot,
) -> Result<WhisperGgufTensorBinding, WhisperGgufTensorBindingError> {
    by_slot
        .get(slot)
        .cloned()
        .ok_or_else(|| WhisperGgufTensorBindingError::MissingRequiredTensor {
            slot: slot.label(),
            aliases: vec![],
        })
}

fn resolve_alias<'a>(
    index: &'a GgufTensorIndex,
    aliases: &[String],
) -> Option<(String, &'a GgufTensorMetadata)> {
    aliases
        .iter()
        .find_map(|name| index.get(name).map(|metadata| (name.clone(), metadata)))
}

fn validate_tensor_type(
    descriptor: &WhisperTensorSlotDescriptor,
    metadata: &GgufTensorMetadata,
) -> Result<(), WhisperGgufTensorBindingError> {
    match descriptor.type_constraint {
        TensorTypeConstraint::Any => Ok(()),
        TensorTypeConstraint::FloatLike => {
            if matches!(
                metadata.ggml_type,
                GGML_TYPE_F32 | GGML_TYPE_F16 | GGML_TYPE_BF16
            ) {
                return Ok(());
            }
            Err(WhisperGgufTensorBindingError::TensorTypeMismatch {
                slot: descriptor.slot.label(),
                tensor_name: metadata.name.clone(),
                found_type: metadata.type_name.clone(),
                expected: "f32/f16/bf16".to_string(),
            })
        }
    }
}

fn validate_tensor_shape(
    descriptor: &WhisperTensorSlotDescriptor,
    metadata: &GgufTensorMetadata,
) -> Result<(), WhisperGgufTensorBindingError> {
    if metadata.dims.contains(&0) {
        return Err(WhisperGgufTensorBindingError::TensorShapeMismatch {
            slot: descriptor.slot.label(),
            tensor_name: metadata.name.clone(),
            found_shape: metadata.dims.clone(),
            expected: "all dimensions must be > 0".to_string(),
        });
    }
    let matched = match &descriptor.shape_constraint {
        TensorShapeConstraint::ExactAnyOf(shapes) => {
            shapes.iter().any(|shape| metadata.has_shape(shape))
        }
        TensorShapeConstraint::Rank1OrRank2LastDim(dim) => match metadata.dims.as_slice() {
            [last] => last == dim,
            [_, last] => last == dim,
            _ => false,
        },
        TensorShapeConstraint::Rank1OrRank2 => matches!(metadata.dims.as_slice(), [_] | [_, _]),
        TensorShapeConstraint::Rank2Exact(a, b) => metadata.dims.as_slice() == [*a, *b],
        TensorShapeConstraint::Rank2AnyOrder(a, b) => {
            metadata.dims.as_slice() == [*a, *b] || metadata.dims.as_slice() == [*b, *a]
        }
        TensorShapeConstraint::Rank2EitherDim(dim) => match metadata.dims.as_slice() {
            [d0, d1] => d0 == dim || d1 == dim,
            _ => false,
        },
    };
    if matched {
        Ok(())
    } else {
        Err(WhisperGgufTensorBindingError::TensorShapeMismatch {
            slot: descriptor.slot.label(),
            tensor_name: metadata.name.clone(),
            found_shape: metadata.dims.clone(),
            expected: describe_shape_constraint(&descriptor.shape_constraint),
        })
    }
}

fn describe_shape_constraint(constraint: &TensorShapeConstraint) -> String {
    match constraint {
        TensorShapeConstraint::ExactAnyOf(shapes) => {
            let formatted = shapes
                .iter()
                .map(|shape| format!("{shape:?}"))
                .collect::<Vec<_>>()
                .join(" | ");
            format!("one of [{formatted}]")
        }
        TensorShapeConstraint::Rank1OrRank2LastDim(dim) => {
            format!("rank-1 [{dim}] or rank-2 [*, {dim}]")
        }
        TensorShapeConstraint::Rank1OrRank2 => "rank-1 [*] or rank-2 [*, *]".to_string(),
        TensorShapeConstraint::Rank2Exact(a, b) => format!("rank-2 [{a}, {b}]"),
        TensorShapeConstraint::Rank2AnyOrder(a, b) => {
            format!("rank-2 [{a}, {b}] or [{b}, {a}]")
        }
        TensorShapeConstraint::Rank2EitherDim(dim) => {
            format!("rank-2 with one dimension equal to {dim}")
        }
    }
}

fn exact_shapes(shapes: impl IntoIterator<Item = Vec<u64>>) -> TensorShapeConstraint {
    TensorShapeConstraint::ExactAnyOf(shapes.into_iter().collect())
}

fn descriptor(
    slot: WhisperGgufTensorSlot,
    aliases: Vec<String>,
    type_constraint: TensorTypeConstraint,
    shape_constraint: TensorShapeConstraint,
) -> WhisperTensorSlotDescriptor {
    WhisperTensorSlotDescriptor {
        slot,
        aliases,
        type_constraint,
        shape_constraint,
    }
}

fn descriptor_any(
    slot: WhisperGgufTensorSlot,
    aliases: Vec<String>,
    shape_constraint: TensorShapeConstraint,
) -> WhisperTensorSlotDescriptor {
    descriptor(slot, aliases, TensorTypeConstraint::Any, shape_constraint)
}

fn descriptor_float(
    slot: WhisperGgufTensorSlot,
    aliases: Vec<String>,
    shape_constraint: TensorShapeConstraint,
) -> WhisperTensorSlotDescriptor {
    descriptor(
        slot,
        aliases,
        TensorTypeConstraint::FloatLike,
        shape_constraint,
    )
}

fn hidden_u64(hidden: usize) -> u64 {
    hidden as u64
}

fn whisper_tensor_slot_descriptors(
    context: &WhisperGgufTensorBindingContext,
) -> Vec<WhisperTensorSlotDescriptor> {
    let encoder_hidden = hidden_u64(context.n_audio_state);
    let decoder_hidden = hidden_u64(context.n_text_state);
    let decoder_ctx = hidden_u64(context.n_text_ctx);
    let vocab = hidden_u64(context.n_vocab);
    let encoder_mels = hidden_u64(context.n_mels);
    let encoder_ctx = hidden_u64(context.n_audio_ctx);

    let mut descriptors = vec![
        descriptor_any(
            WhisperGgufTensorSlot::EncoderConv1Weight,
            vec![
                "model.encoder.conv1.weight".to_string(),
                "encoder.conv1.weight".to_string(),
            ],
            exact_shapes([
                vec![3, encoder_mels, encoder_hidden],
                vec![encoder_hidden, encoder_mels, 3],
            ]),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::EncoderConv1Bias,
            vec![
                "model.encoder.conv1.bias".to_string(),
                "encoder.conv1.bias".to_string(),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(encoder_hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::EncoderConv2Weight,
            vec![
                "model.encoder.conv2.weight".to_string(),
                "encoder.conv2.weight".to_string(),
            ],
            exact_shapes([
                vec![3, encoder_hidden, encoder_hidden],
                vec![encoder_hidden, encoder_hidden, 3],
            ]),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::EncoderConv2Bias,
            vec![
                "model.encoder.conv2.bias".to_string(),
                "encoder.conv2.bias".to_string(),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(encoder_hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::EncoderPositionalEmbedding,
            vec![
                "model.encoder.embed_positions.weight".to_string(),
                "encoder.positional_embedding".to_string(),
            ],
            TensorShapeConstraint::Rank2AnyOrder(encoder_ctx, encoder_hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::EncoderFinalLayerNormWeight,
            vec![
                "model.encoder.layer_norm.weight".to_string(),
                "encoder.ln_post.weight".to_string(),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(encoder_hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::EncoderFinalLayerNormBias,
            vec![
                "model.encoder.layer_norm.bias".to_string(),
                "encoder.ln_post.bias".to_string(),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(encoder_hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::DecoderTokenEmbedding,
            vec![
                "model.decoder.embed_tokens.weight".to_string(),
                "model.decoder.token_embedding.weight".to_string(),
                "decoder.token_embedding.weight".to_string(),
            ],
            exact_shapes([vec![vocab, decoder_hidden], vec![decoder_hidden, vocab]]),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::DecoderPositionalEmbedding,
            vec![
                "model.decoder.embed_positions.weight".to_string(),
                "decoder.positional_embedding".to_string(),
            ],
            TensorShapeConstraint::Rank2AnyOrder(decoder_ctx, decoder_hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::DecoderFinalLayerNormWeight,
            vec![
                "model.decoder.layer_norm.weight".to_string(),
                "decoder.ln.weight".to_string(),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(decoder_hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::DecoderFinalLayerNormBias,
            vec![
                "model.decoder.layer_norm.bias".to_string(),
                "decoder.ln.bias".to_string(),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(decoder_hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::DecoderOutputProjectionWeight,
            vec![
                "model.decoder.output_projection.weight".to_string(),
                "decoder.output_projection.weight".to_string(),
                "model.decoder.embed_tokens.weight".to_string(),
                "model.decoder.token_embedding.weight".to_string(),
                "decoder.token_embedding.weight".to_string(),
            ],
            exact_shapes([vec![vocab, decoder_hidden], vec![decoder_hidden, vocab]]),
        ),
    ];

    for layer_idx in 0..context.n_audio_layer {
        descriptors.extend(encoder_layer_descriptors(layer_idx, encoder_hidden));
    }

    for layer_idx in 0..context.n_text_layer {
        descriptors.extend(decoder_layer_descriptors(layer_idx, decoder_hidden));
    }

    descriptors
}

fn encoder_layer_descriptors(layer_idx: usize, hidden: u64) -> Vec<WhisperTensorSlotDescriptor> {
    vec![
        descriptor_float(
            WhisperGgufTensorSlot::EncoderLayerSelfAttnNormWeight { layer_idx },
            vec![
                format!("model.encoder.layers.{layer_idx}.self_attn_layer_norm.weight"),
                format!("encoder.blocks.{layer_idx}.attn_ln.weight"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::EncoderLayerSelfAttnNormBias { layer_idx },
            vec![
                format!("model.encoder.layers.{layer_idx}.self_attn_layer_norm.bias"),
                format!("encoder.blocks.{layer_idx}.attn_ln.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::EncoderLayerSelfAttnQWeight { layer_idx },
            vec![
                format!("model.encoder.layers.{layer_idx}.self_attn.q_proj.weight"),
                format!("encoder.blocks.{layer_idx}.attn.query.weight"),
            ],
            TensorShapeConstraint::Rank2Exact(hidden, hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::EncoderLayerSelfAttnQBias { layer_idx },
            vec![
                format!("model.encoder.layers.{layer_idx}.self_attn.q_proj.bias"),
                format!("encoder.blocks.{layer_idx}.attn.query.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::EncoderLayerSelfAttnKWeight { layer_idx },
            vec![
                format!("model.encoder.layers.{layer_idx}.self_attn.k_proj.weight"),
                format!("encoder.blocks.{layer_idx}.attn.key.weight"),
            ],
            TensorShapeConstraint::Rank2Exact(hidden, hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::EncoderLayerSelfAttnVWeight { layer_idx },
            vec![
                format!("model.encoder.layers.{layer_idx}.self_attn.v_proj.weight"),
                format!("encoder.blocks.{layer_idx}.attn.value.weight"),
            ],
            TensorShapeConstraint::Rank2Exact(hidden, hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::EncoderLayerSelfAttnVBias { layer_idx },
            vec![
                format!("model.encoder.layers.{layer_idx}.self_attn.v_proj.bias"),
                format!("encoder.blocks.{layer_idx}.attn.value.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::EncoderLayerSelfAttnOutWeight { layer_idx },
            vec![
                format!("model.encoder.layers.{layer_idx}.self_attn.out_proj.weight"),
                format!("encoder.blocks.{layer_idx}.attn.out.weight"),
            ],
            TensorShapeConstraint::Rank2Exact(hidden, hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::EncoderLayerSelfAttnOutBias { layer_idx },
            vec![
                format!("model.encoder.layers.{layer_idx}.self_attn.out_proj.bias"),
                format!("encoder.blocks.{layer_idx}.attn.out.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::EncoderLayerMlpNormWeight { layer_idx },
            vec![
                format!("model.encoder.layers.{layer_idx}.final_layer_norm.weight"),
                format!("encoder.blocks.{layer_idx}.mlp_ln.weight"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::EncoderLayerMlpNormBias { layer_idx },
            vec![
                format!("model.encoder.layers.{layer_idx}.final_layer_norm.bias"),
                format!("encoder.blocks.{layer_idx}.mlp_ln.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::EncoderLayerMlpFc1Weight { layer_idx },
            vec![
                format!("model.encoder.layers.{layer_idx}.fc1.weight"),
                format!("encoder.blocks.{layer_idx}.mlp.0.weight"),
            ],
            TensorShapeConstraint::Rank2EitherDim(hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::EncoderLayerMlpFc1Bias { layer_idx },
            vec![
                format!("model.encoder.layers.{layer_idx}.fc1.bias"),
                format!("encoder.blocks.{layer_idx}.mlp.0.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2,
        ),
        descriptor_any(
            WhisperGgufTensorSlot::EncoderLayerMlpFc2Weight { layer_idx },
            vec![
                format!("model.encoder.layers.{layer_idx}.fc2.weight"),
                format!("encoder.blocks.{layer_idx}.mlp.2.weight"),
            ],
            TensorShapeConstraint::Rank2EitherDim(hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::EncoderLayerMlpFc2Bias { layer_idx },
            vec![
                format!("model.encoder.layers.{layer_idx}.fc2.bias"),
                format!("encoder.blocks.{layer_idx}.mlp.2.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
    ]
}

fn decoder_layer_descriptors(layer_idx: usize, hidden: u64) -> Vec<WhisperTensorSlotDescriptor> {
    vec![
        descriptor_float(
            WhisperGgufTensorSlot::DecoderLayerSelfAttnNormWeight { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.self_attn_layer_norm.weight"),
                format!("decoder.blocks.{layer_idx}.attn_ln.weight"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::DecoderLayerSelfAttnNormBias { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.self_attn_layer_norm.bias"),
                format!("decoder.blocks.{layer_idx}.attn_ln.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::DecoderLayerSelfAttnQWeight { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.self_attn.q_proj.weight"),
                format!("decoder.blocks.{layer_idx}.attn.query.weight"),
            ],
            TensorShapeConstraint::Rank2Exact(hidden, hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::DecoderLayerSelfAttnQBias { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.self_attn.q_proj.bias"),
                format!("decoder.blocks.{layer_idx}.attn.query.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::DecoderLayerSelfAttnKWeight { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.self_attn.k_proj.weight"),
                format!("decoder.blocks.{layer_idx}.attn.key.weight"),
            ],
            TensorShapeConstraint::Rank2Exact(hidden, hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::DecoderLayerSelfAttnVWeight { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.self_attn.v_proj.weight"),
                format!("decoder.blocks.{layer_idx}.attn.value.weight"),
            ],
            TensorShapeConstraint::Rank2Exact(hidden, hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::DecoderLayerSelfAttnVBias { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.self_attn.v_proj.bias"),
                format!("decoder.blocks.{layer_idx}.attn.value.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::DecoderLayerSelfAttnOutWeight { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.self_attn.out_proj.weight"),
                format!("decoder.blocks.{layer_idx}.attn.out.weight"),
            ],
            TensorShapeConstraint::Rank2Exact(hidden, hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::DecoderLayerSelfAttnOutBias { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.self_attn.out_proj.bias"),
                format!("decoder.blocks.{layer_idx}.attn.out.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::DecoderLayerCrossAttnNormWeight { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.encoder_attn_layer_norm.weight"),
                format!("decoder.blocks.{layer_idx}.cross_attn_ln.weight"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::DecoderLayerCrossAttnNormBias { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.encoder_attn_layer_norm.bias"),
                format!("decoder.blocks.{layer_idx}.cross_attn_ln.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::DecoderLayerCrossAttnQWeight { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.encoder_attn.q_proj.weight"),
                format!("decoder.blocks.{layer_idx}.cross_attn.query.weight"),
            ],
            TensorShapeConstraint::Rank2Exact(hidden, hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::DecoderLayerCrossAttnQBias { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.encoder_attn.q_proj.bias"),
                format!("decoder.blocks.{layer_idx}.cross_attn.query.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::DecoderLayerCrossAttnKWeight { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.encoder_attn.k_proj.weight"),
                format!("decoder.blocks.{layer_idx}.cross_attn.key.weight"),
            ],
            TensorShapeConstraint::Rank2Exact(hidden, hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::DecoderLayerCrossAttnVWeight { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.encoder_attn.v_proj.weight"),
                format!("decoder.blocks.{layer_idx}.cross_attn.value.weight"),
            ],
            TensorShapeConstraint::Rank2Exact(hidden, hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::DecoderLayerCrossAttnVBias { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.encoder_attn.v_proj.bias"),
                format!("decoder.blocks.{layer_idx}.cross_attn.value.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::DecoderLayerCrossAttnOutWeight { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.encoder_attn.out_proj.weight"),
                format!("decoder.blocks.{layer_idx}.cross_attn.out.weight"),
            ],
            TensorShapeConstraint::Rank2Exact(hidden, hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::DecoderLayerCrossAttnOutBias { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.encoder_attn.out_proj.bias"),
                format!("decoder.blocks.{layer_idx}.cross_attn.out.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::DecoderLayerMlpNormWeight { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.final_layer_norm.weight"),
                format!("decoder.blocks.{layer_idx}.mlp_ln.weight"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::DecoderLayerMlpNormBias { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.final_layer_norm.bias"),
                format!("decoder.blocks.{layer_idx}.mlp_ln.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
        descriptor_any(
            WhisperGgufTensorSlot::DecoderLayerMlpFc1Weight { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.fc1.weight"),
                format!("decoder.blocks.{layer_idx}.mlp.0.weight"),
            ],
            TensorShapeConstraint::Rank2EitherDim(hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::DecoderLayerMlpFc1Bias { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.fc1.bias"),
                format!("decoder.blocks.{layer_idx}.mlp.0.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2,
        ),
        descriptor_any(
            WhisperGgufTensorSlot::DecoderLayerMlpFc2Weight { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.fc2.weight"),
                format!("decoder.blocks.{layer_idx}.mlp.2.weight"),
            ],
            TensorShapeConstraint::Rank2EitherDim(hidden),
        ),
        descriptor_float(
            WhisperGgufTensorSlot::DecoderLayerMlpFc2Bias { layer_idx },
            vec![
                format!("model.decoder.layers.{layer_idx}.fc2.bias"),
                format!("decoder.blocks.{layer_idx}.mlp.2.bias"),
            ],
            TensorShapeConstraint::Rank1OrRank2LastDim(hidden),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use tempfile::NamedTempFile;

    use super::*;
    use crate::read_gguf_tensor_index;

    const GGUF_VERSION: u32 = 3;
    const GGUF_ALIGNMENT: usize = 32;

    #[test]
    fn full_tiny_encoder_binding_succeeds() {
        let file = NamedTempFile::new().expect("temp file");
        write_gguf_fixture(file.path(), &base_openasr_tensors());
        let index = read_gguf_tensor_index(file.path()).expect("read index");

        let bindings = bind_whisper_gguf_tensors(&test_context(), &index).expect("must bind");
        assert_eq!(bindings.encoder().layers.len(), 2);
        assert_eq!(bindings.encoder().n_audio_state, 4);
        assert_eq!(bindings.encoder().head_dim, 2);
        assert_eq!(bindings.encoder().ffn_dim, 8);
        assert_eq!(bindings.decoder().layers.len(), 2);
        assert_eq!(bindings.decoder().n_vocab, 32);
        assert_eq!(bindings.decoder().head_dim, 2);
    }

    #[test]
    fn whisper_cpp_mlp_weight_layout_succeeds() {
        let file = NamedTempFile::new().expect("temp file");
        let mut tensors = base_openasr_tensors();
        for tensor in &mut tensors {
            if tensor.name.ends_with(".fc1.weight") {
                tensor.dims = vec![4, 8];
            } else if tensor.name.ends_with(".fc2.weight") {
                tensor.dims = vec![8, 4];
            }
        }
        write_gguf_fixture(file.path(), &tensors);
        let index = read_gguf_tensor_index(file.path()).expect("read index");

        let bindings = bind_whisper_gguf_tensors(&test_context(), &index).expect("must bind");
        assert_eq!(bindings.encoder().ffn_dim, 8);
    }

    #[test]
    fn missing_layer_fails_closed() {
        let file = NamedTempFile::new().expect("temp file");
        let tensors = base_openasr_tensors()
            .into_iter()
            .filter(|tensor| tensor.name != "model.encoder.layers.1.fc2.weight")
            .collect::<Vec<_>>();
        write_gguf_fixture(file.path(), &tensors);
        let index = read_gguf_tensor_index(file.path()).expect("read index");

        let error = bind_whisper_gguf_tensors(&test_context(), &index).expect_err("must fail");
        assert!(matches!(
            error,
            WhisperGgufTensorBindingError::MissingRequiredTensor { .. }
        ));
    }

    #[test]
    fn missing_encoder_final_norm_fails_closed() {
        let file = NamedTempFile::new().expect("temp file");
        let tensors = base_openasr_tensors()
            .into_iter()
            .filter(|tensor| tensor.name != "model.encoder.layer_norm.weight")
            .collect::<Vec<_>>();
        write_gguf_fixture(file.path(), &tensors);
        let index = read_gguf_tensor_index(file.path()).expect("read index");

        let error = bind_whisper_gguf_tensors(&test_context(), &index).expect_err("must fail");
        assert!(matches!(
            error,
            WhisperGgufTensorBindingError::MissingRequiredTensor { .. }
        ));
    }

    #[test]
    fn shape_mismatch_fails_closed() {
        let file = NamedTempFile::new().expect("temp file");
        let mut tensors = base_openasr_tensors();
        for tensor in &mut tensors {
            if tensor.name == "model.encoder.layers.0.self_attn.k_proj.weight" {
                tensor.dims = vec![4, 2];
            }
        }
        write_gguf_fixture(file.path(), &tensors);
        let index = read_gguf_tensor_index(file.path()).expect("read index");

        let error = bind_whisper_gguf_tensors(&test_context(), &index).expect_err("must fail");
        assert!(matches!(
            error,
            WhisperGgufTensorBindingError::TensorShapeMismatch { .. }
        ));
    }

    #[test]
    fn layer_count_mismatch_fails_closed() {
        let file = NamedTempFile::new().expect("temp file");
        let tensors = base_openasr_tensors()
            .into_iter()
            .filter(|tensor| !tensor.name.starts_with("model.encoder.layers.1."))
            .collect::<Vec<_>>();
        write_gguf_fixture(file.path(), &tensors);
        let index = read_gguf_tensor_index(file.path()).expect("read index");

        let error = bind_whisper_gguf_tensors(&test_context(), &index).expect_err("must fail");
        assert!(matches!(
            error,
            WhisperGgufTensorBindingError::MissingRequiredTensor { .. }
        ));
    }

    #[derive(Clone)]
    struct OwnedTensorFixture {
        name: String,
        ggml_type: i32,
        dims: Vec<i64>,
    }

    fn base_openasr_tensors() -> Vec<OwnedTensorFixture> {
        let mut tensors = vec![
            tensor("model.encoder.conv1.weight", GGML_TYPE_F32, &[3, 2, 4]),
            tensor("model.encoder.conv1.bias", GGML_TYPE_F32, &[4]),
            tensor("model.encoder.conv2.weight", GGML_TYPE_F32, &[3, 4, 4]),
            tensor("model.encoder.conv2.bias", GGML_TYPE_F32, &[4]),
            tensor(
                "model.encoder.embed_positions.weight",
                GGML_TYPE_F32,
                &[8, 4],
            ),
            tensor("model.encoder.layer_norm.weight", GGML_TYPE_F32, &[4]),
            tensor("model.encoder.layer_norm.bias", GGML_TYPE_F32, &[4]),
            tensor("model.decoder.embed_tokens.weight", GGML_TYPE_F32, &[32, 4]),
            tensor(
                "model.decoder.embed_positions.weight",
                GGML_TYPE_F32,
                &[16, 4],
            ),
            tensor("model.decoder.layer_norm.weight", GGML_TYPE_F32, &[4]),
            tensor("model.decoder.layer_norm.bias", GGML_TYPE_F32, &[4]),
        ];

        for layer_idx in 0..2 {
            let prefix = format!("model.encoder.layers.{layer_idx}");
            tensors.extend([
                tensor(
                    &format!("{prefix}.self_attn_layer_norm.weight"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.self_attn_layer_norm.bias"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    GGML_TYPE_F32,
                    &[4, 4],
                ),
                tensor(
                    &format!("{prefix}.self_attn.q_proj.bias"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    GGML_TYPE_F32,
                    &[4, 4],
                ),
                tensor(
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    GGML_TYPE_F32,
                    &[4, 4],
                ),
                tensor(
                    &format!("{prefix}.self_attn.v_proj.bias"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.self_attn.out_proj.weight"),
                    GGML_TYPE_F32,
                    &[4, 4],
                ),
                tensor(
                    &format!("{prefix}.self_attn.out_proj.bias"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.final_layer_norm.weight"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.final_layer_norm.bias"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(&format!("{prefix}.fc1.weight"), GGML_TYPE_F32, &[8, 4]),
                tensor(&format!("{prefix}.fc1.bias"), GGML_TYPE_F32, &[8]),
                tensor(&format!("{prefix}.fc2.weight"), GGML_TYPE_F32, &[4, 8]),
                tensor(&format!("{prefix}.fc2.bias"), GGML_TYPE_F32, &[4]),
            ]);

            let prefix = format!("model.decoder.layers.{layer_idx}");
            tensors.extend([
                tensor(
                    &format!("{prefix}.self_attn_layer_norm.weight"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.self_attn_layer_norm.bias"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    GGML_TYPE_F32,
                    &[4, 4],
                ),
                tensor(
                    &format!("{prefix}.self_attn.q_proj.bias"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    GGML_TYPE_F32,
                    &[4, 4],
                ),
                tensor(
                    &format!("{prefix}.self_attn.k_proj.bias"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    GGML_TYPE_F32,
                    &[4, 4],
                ),
                tensor(
                    &format!("{prefix}.self_attn.v_proj.bias"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.self_attn.out_proj.weight"),
                    GGML_TYPE_F32,
                    &[4, 4],
                ),
                tensor(
                    &format!("{prefix}.self_attn.out_proj.bias"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.encoder_attn_layer_norm.weight"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.encoder_attn_layer_norm.bias"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.encoder_attn.q_proj.weight"),
                    GGML_TYPE_F32,
                    &[4, 4],
                ),
                tensor(
                    &format!("{prefix}.encoder_attn.q_proj.bias"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.encoder_attn.k_proj.weight"),
                    GGML_TYPE_F32,
                    &[4, 4],
                ),
                tensor(
                    &format!("{prefix}.encoder_attn.k_proj.bias"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.encoder_attn.v_proj.weight"),
                    GGML_TYPE_F32,
                    &[4, 4],
                ),
                tensor(
                    &format!("{prefix}.encoder_attn.v_proj.bias"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.encoder_attn.out_proj.weight"),
                    GGML_TYPE_F32,
                    &[4, 4],
                ),
                tensor(
                    &format!("{prefix}.encoder_attn.out_proj.bias"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.final_layer_norm.weight"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(
                    &format!("{prefix}.final_layer_norm.bias"),
                    GGML_TYPE_F32,
                    &[4],
                ),
                tensor(&format!("{prefix}.fc1.weight"), GGML_TYPE_F32, &[8, 4]),
                tensor(&format!("{prefix}.fc1.bias"), GGML_TYPE_F32, &[8]),
                tensor(&format!("{prefix}.fc2.weight"), GGML_TYPE_F32, &[4, 8]),
                tensor(&format!("{prefix}.fc2.bias"), GGML_TYPE_F32, &[4]),
            ]);
        }

        tensors
    }

    fn tensor(name: &str, ggml_type: i32, dims: &[i64]) -> OwnedTensorFixture {
        OwnedTensorFixture {
            name: name.to_string(),
            ggml_type,
            dims: dims.to_vec(),
        }
    }

    fn test_context() -> WhisperGgufTensorBindingContext {
        WhisperGgufTensorBindingContext {
            n_audio_layer: 2,
            n_audio_state: 4,
            n_audio_head: 2,
            n_mels: 2,
            n_audio_ctx: 8,
            n_text_layer: 2,
            n_text_state: 4,
            n_text_head: 2,
            n_text_ctx: 16,
            n_vocab: 32,
        }
    }

    fn write_gguf_fixture(path: &Path, tensors: &[OwnedTensorFixture]) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as i64).to_le_bytes());
        bytes.extend_from_slice(&0_i64.to_le_bytes()); // n_kv

        let tensor_payload_sizes = tensors
            .iter()
            .map(|tensor| payload_size_for_fixture(tensor.ggml_type, &tensor.dims))
            .collect::<Vec<_>>();

        let mut running_offset: u64 = 0;
        for (index, tensor) in tensors.iter().enumerate() {
            push_gguf_string(&mut bytes, &tensor.name);
            bytes.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
            for dim in &tensor.dims {
                bytes.extend_from_slice(&dim.to_le_bytes());
            }
            bytes.extend_from_slice(&tensor.ggml_type.to_le_bytes());
            bytes.extend_from_slice(&running_offset.to_le_bytes());
            running_offset = align_up_u64(
                running_offset + tensor_payload_sizes[index],
                GGUF_ALIGNMENT as u64,
            );
        }

        let aligned = align_up(bytes.len(), GGUF_ALIGNMENT);
        bytes.resize(aligned, 0);
        for tensor_size in tensor_payload_sizes {
            bytes.resize(bytes.len() + tensor_size as usize, 0);
            let next_aligned = align_up(bytes.len(), GGUF_ALIGNMENT);
            bytes.resize(next_aligned, 0);
        }

        fs::write(path, bytes).expect("write gguf fixture");
    }

    fn payload_size_for_fixture(ggml_type: i32, dims: &[i64]) -> u64 {
        let mut elements: u64 = 1;
        for dim in dims {
            let dim = u64::try_from(*dim.max(&0)).expect("non-negative dim");
            elements = elements.saturating_mul(dim);
        }
        match ggml_type {
            GGML_TYPE_F32 => elements.saturating_mul(4),
            GGML_TYPE_F16 | GGML_TYPE_BF16 => elements.saturating_mul(2),
            _ => elements.saturating_mul(4),
        }
    }

    fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn align_up(value: usize, alignment: usize) -> usize {
        debug_assert!(alignment > 0);
        (value + alignment - 1) & !(alignment - 1)
    }

    fn align_up_u64(value: u64, alignment: u64) -> u64 {
        debug_assert!(alignment > 0);
        (value + alignment - 1) & !(alignment - 1)
    }
}
