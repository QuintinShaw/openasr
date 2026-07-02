use thiserror::Error;

use super::ggml_tensor_binding::{WhisperGgufTensorBinding, WhisperGgufTensorBindings};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WhisperEncoderPreludeInputShape {
    pub mel_bins: usize,
    pub mel_frames: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperEncoderPreludePlan {
    pub input_shape: WhisperEncoderPreludeInputShape,
    pub conv1: WhisperEncoderPreludeConv1dPlan,
    pub conv2: WhisperEncoderPreludeConv1dPlan,
    pub positional_embedding: WhisperEncoderPreludePositionalEmbeddingPlan,
    pub output_frames: usize,
    pub output_hidden_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperEncoderPreludeConv1dPlan {
    pub weight_name: String,
    pub weight_num_elements: usize,
    pub bias_name: String,
    pub bias_num_elements: usize,
    pub layout: WhisperEncoderPreludeConv1dWeightLayout,
    pub kernel_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WhisperEncoderPreludeConv1dWeightLayout {
    KernelInOut,
    OutInKernel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperEncoderPreludePositionalEmbeddingPlan {
    pub tensor_name: String,
    pub tensor_num_elements: usize,
    pub max_positions: usize,
    pub hidden_size: usize,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum WhisperEncoderPreludePlanError {
    #[error("whisper encoder prelude input shape is invalid: {reason}")]
    InvalidInputShape { reason: String },
    #[error(
        "whisper encoder prelude tensor '{tensor_name}' for slot '{slot}' has invalid shape {found_shape:?}: {reason}"
    )]
    TensorShapeMismatch {
        slot: &'static str,
        tensor_name: String,
        found_shape: Vec<u64>,
        reason: String,
    },
    #[error("whisper encoder prelude unsupported primitive '{primitive}': {reason}")]
    UnsupportedPrimitive {
        primitive: &'static str,
        reason: String,
    },
}

pub(crate) fn build_whisper_encoder_prelude_plan(
    bindings: &WhisperGgufTensorBindings,
    input_shape: WhisperEncoderPreludeInputShape,
    encoder_hidden_size: usize,
    encoder_mels_count: usize,
) -> Result<WhisperEncoderPreludePlan, WhisperEncoderPreludePlanError> {
    if input_shape.mel_bins != encoder_mels_count {
        return Err(WhisperEncoderPreludePlanError::InvalidInputShape {
            reason: format!(
                "mel_bins={} does not match whisper.encoder.mels_count={encoder_mels_count}",
                input_shape.mel_bins
            ),
        });
    }
    if input_shape.mel_frames == 0 {
        return Err(WhisperEncoderPreludePlanError::InvalidInputShape {
            reason: "mel_frames must be > 0".to_string(),
        });
    }

    let conv1 = parse_conv1_plan(bindings, encoder_hidden_size, encoder_mels_count)?;
    let conv2 = parse_conv2_plan(bindings, encoder_hidden_size)?;
    let positional_embedding = parse_positional_embedding_plan(bindings, encoder_hidden_size)?;

    let conv1_frames = conv_output_frames(
        input_shape.mel_frames,
        conv1.kernel_size,
        conv1.stride,
        conv1.padding,
        conv1.dilation,
    )?;
    let output_frames = conv_output_frames(
        conv1_frames,
        conv2.kernel_size,
        conv2.stride,
        conv2.padding,
        conv2.dilation,
    )?;

    if output_frames > positional_embedding.max_positions {
        return Err(WhisperEncoderPreludePlanError::UnsupportedPrimitive {
            primitive: "encoder.positional_embedding.slice",
            reason: format!(
                "projected frames {output_frames} exceed positional capacity {}",
                positional_embedding.max_positions
            ),
        });
    }

    Ok(WhisperEncoderPreludePlan {
        input_shape,
        conv1,
        conv2,
        positional_embedding,
        output_frames,
        output_hidden_size: encoder_hidden_size,
    })
}

fn parse_conv1_plan(
    bindings: &WhisperGgufTensorBindings,
    hidden_size: usize,
    mels_count: usize,
) -> Result<WhisperEncoderPreludeConv1dPlan, WhisperEncoderPreludePlanError> {
    let prelude = &bindings.encoder().prelude;
    parse_conv_plan(
        &prelude.conv1_weight,
        &prelude.conv1_bias,
        "encoder.conv1.weight",
        "encoder.conv1.bias",
        hidden_size,
        mels_count,
        1,
    )
}

fn parse_conv2_plan(
    bindings: &WhisperGgufTensorBindings,
    hidden_size: usize,
) -> Result<WhisperEncoderPreludeConv1dPlan, WhisperEncoderPreludePlanError> {
    let prelude = &bindings.encoder().prelude;
    parse_conv_plan(
        &prelude.conv2_weight,
        &prelude.conv2_bias,
        "encoder.conv2.weight",
        "encoder.conv2.bias",
        hidden_size,
        hidden_size,
        2,
    )
}

fn parse_conv_plan(
    weight: &WhisperGgufTensorBinding,
    bias: &WhisperGgufTensorBinding,
    slot_label: &'static str,
    bias_slot_label: &'static str,
    out_channels: usize,
    in_channels: usize,
    stride: usize,
) -> Result<WhisperEncoderPreludeConv1dPlan, WhisperEncoderPreludePlanError> {
    let dims = &weight.metadata.dims;
    let kernel_size = 3_usize;
    let kernel_size_u64 = kernel_size as u64;
    let in_channels_u64 = in_channels as u64;
    let out_channels_u64 = out_channels as u64;
    let layout = if dims.as_slice() == [kernel_size_u64, in_channels_u64, out_channels_u64] {
        WhisperEncoderPreludeConv1dWeightLayout::KernelInOut
    } else if dims.as_slice() == [out_channels_u64, in_channels_u64, kernel_size_u64] {
        WhisperEncoderPreludeConv1dWeightLayout::OutInKernel
    } else {
        return Err(WhisperEncoderPreludePlanError::TensorShapeMismatch {
            slot: slot_label,
            tensor_name: weight.metadata.name.clone(),
            found_shape: dims.clone(),
            reason: format!(
                "expected [{kernel_size},{in_channels},{out_channels}] or [{out_channels},{in_channels},{kernel_size}]"
            ),
        });
    };

    let weight_num_elements =
        u64_to_usize_checked(weight.metadata.num_elements().ok_or_else(|| {
            WhisperEncoderPreludePlanError::TensorShapeMismatch {
                slot: slot_label,
                tensor_name: weight.metadata.name.clone(),
                found_shape: dims.clone(),
                reason: "shape element count overflow".to_string(),
            }
        })?)?;

    let bias_num_elements =
        u64_to_usize_checked(bias.metadata.num_elements().ok_or_else(|| {
            WhisperEncoderPreludePlanError::TensorShapeMismatch {
                slot: bias_slot_label,
                tensor_name: bias.metadata.name.clone(),
                found_shape: bias.metadata.dims.clone(),
                reason: "shape element count overflow".to_string(),
            }
        })?)?;
    if bias_num_elements < out_channels {
        return Err(WhisperEncoderPreludePlanError::TensorShapeMismatch {
            slot: bias_slot_label,
            tensor_name: bias.metadata.name.clone(),
            found_shape: bias.metadata.dims.clone(),
            reason: format!(
                "bias has {bias_num_elements} elements but requires at least {out_channels}"
            ),
        });
    }

    Ok(WhisperEncoderPreludeConv1dPlan {
        weight_name: weight.resolved_name.clone(),
        weight_num_elements,
        bias_name: bias.resolved_name.clone(),
        bias_num_elements,
        layout,
        kernel_size,
        in_channels,
        out_channels,
        stride,
        padding: 1,
        dilation: 1,
    })
}

fn parse_positional_embedding_plan(
    bindings: &WhisperGgufTensorBindings,
    hidden_size: usize,
) -> Result<WhisperEncoderPreludePositionalEmbeddingPlan, WhisperEncoderPreludePlanError> {
    let slot_label = "encoder.positional_embedding";
    let positional = &bindings.encoder().prelude.positional_embedding;
    let dims = positional.metadata.dims.as_slice();
    let hidden_u64 = hidden_size as u64;
    let max_positions = match dims {
        [positions, hidden] if *hidden == hidden_u64 => u64_to_usize_checked(*positions)?,
        [hidden, positions] if *hidden == hidden_u64 => u64_to_usize_checked(*positions)?,
        _ => {
            return Err(WhisperEncoderPreludePlanError::TensorShapeMismatch {
                slot: slot_label,
                tensor_name: positional.metadata.name.clone(),
                found_shape: positional.metadata.dims.clone(),
                reason: format!("expected rank-2 with one dimension equal to hidden={hidden_size}"),
            });
        }
    };
    if max_positions == 0 {
        return Err(WhisperEncoderPreludePlanError::TensorShapeMismatch {
            slot: slot_label,
            tensor_name: positional.metadata.name.clone(),
            found_shape: positional.metadata.dims.clone(),
            reason: "positional embedding sequence dimension must be > 0".to_string(),
        });
    }

    let tensor_num_elements =
        u64_to_usize_checked(positional.metadata.num_elements().ok_or_else(|| {
            WhisperEncoderPreludePlanError::TensorShapeMismatch {
                slot: slot_label,
                tensor_name: positional.metadata.name.clone(),
                found_shape: positional.metadata.dims.clone(),
                reason: "shape element count overflow".to_string(),
            }
        })?)?;

    Ok(WhisperEncoderPreludePositionalEmbeddingPlan {
        tensor_name: positional.resolved_name.clone(),
        tensor_num_elements,
        max_positions,
        hidden_size,
    })
}

fn conv_output_frames(
    input_frames: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize, WhisperEncoderPreludePlanError> {
    let padded = input_frames
        .checked_add(padding.checked_mul(2).ok_or_else(|| {
            WhisperEncoderPreludePlanError::InvalidInputShape {
                reason: "padding overflow".to_string(),
            }
        })?)
        .ok_or_else(|| WhisperEncoderPreludePlanError::InvalidInputShape {
            reason: "padded input frame count overflow".to_string(),
        })?;
    let receptive = dilation
        .checked_mul(kernel_size.saturating_sub(1))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| WhisperEncoderPreludePlanError::InvalidInputShape {
            reason: "convolution receptive field overflow".to_string(),
        })?;
    if padded < receptive {
        return Err(WhisperEncoderPreludePlanError::InvalidInputShape {
            reason: format!(
                "convolution receptive field {receptive} exceeds padded input frame count {padded}"
            ),
        });
    }
    let numer = padded.checked_sub(receptive).ok_or_else(|| {
        WhisperEncoderPreludePlanError::InvalidInputShape {
            reason: "convolution frame underflow".to_string(),
        }
    })?;
    let output = numer
        .checked_div(stride)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| WhisperEncoderPreludePlanError::InvalidInputShape {
            reason: "convolution output frame overflow".to_string(),
        })?;
    if output == 0 {
        return Err(WhisperEncoderPreludePlanError::InvalidInputShape {
            reason: "convolution output frame count resolved to zero".to_string(),
        });
    }
    Ok(output)
}

fn u64_to_usize_checked(value: u64) -> Result<usize, WhisperEncoderPreludePlanError> {
    usize::try_from(value).map_err(|_| WhisperEncoderPreludePlanError::InvalidInputShape {
        reason: format!("value {value} does not fit target usize"),
    })
}
