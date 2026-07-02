use thiserror::Error;

use crate::arch::{
    COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID, GENERAL_ARCHITECTURE_KEY, OpenAsrArchitectureRegistry,
};
use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_string_scalar, required_u64_scalar,
    u64_to_u32, u64_to_usize, validate_positive_usize,
};
use crate::models::runtime_tensor_contract_registry::{
    RuntimeTensorContractMetadata, resolve_builtin_runtime_tensor_contract_descriptors,
};
use crate::models::tensor_binding::{
    TensorBindingDescriptor, TensorBindingRequirement, TensorBindingSpec, render_shape,
    require_tensor as require_tensor_binding, tensor_binding_descriptors,
    validate_tensor_binding_descriptors,
};
use crate::{GgufTensorIndex, GgufTensorMetadata};

use super::tensor_names::{
    DEC_EMB_LN_BIAS, DEC_EMB_LN_WEIGHT, DEC_EMB_WEIGHT, DEC_HEAD_BIAS, DEC_HEAD_WEIGHT,
    DEC_OUT_LN_BIAS, DEC_OUT_LN_WEIGHT, DEC_POS_WEIGHT, ENC_PRE_OUT_BIAS, ENC_PRE_OUT_WEIGHT,
    ENC_PROJ_BIAS, ENC_PROJ_WEIGHT, FE_MEL_FB, FE_WINDOW, decoder_layer_tensor_names,
    enc_pre_conv_weight, encoder_layer_tensor_names,
};

pub(crate) use crate::arch::hparams::{
    COHERE_TRANSCRIBE_ARCHITECTURE_VALUE, COHERE_TRANSCRIBE_AUDIO_HOP_LENGTH_KEY,
    COHERE_TRANSCRIBE_AUDIO_MELS_COUNT_KEY, COHERE_TRANSCRIBE_AUDIO_N_FFT_KEY,
    COHERE_TRANSCRIBE_AUDIO_SAMPLE_RATE_KEY, COHERE_TRANSCRIBE_AUDIO_WIN_LENGTH_KEY,
    COHERE_TRANSCRIBE_DECODER_D_MODEL_KEY, COHERE_TRANSCRIBE_DECODER_FFN_DIM_KEY,
    COHERE_TRANSCRIBE_DECODER_HEAD_DIM_KEY, COHERE_TRANSCRIBE_DECODER_HEADS_KEY,
    COHERE_TRANSCRIBE_DECODER_LAYERS_KEY, COHERE_TRANSCRIBE_DECODER_MAX_CONTEXT_KEY,
    COHERE_TRANSCRIBE_DECODER_START_TOKEN_ID_KEY, COHERE_TRANSCRIBE_ENCODER_CONV_KERNEL_KEY,
    COHERE_TRANSCRIBE_ENCODER_D_MODEL_KEY, COHERE_TRANSCRIBE_ENCODER_FFN_DIM_KEY,
    COHERE_TRANSCRIBE_ENCODER_HEAD_DIM_KEY, COHERE_TRANSCRIBE_ENCODER_HEADS_KEY,
    COHERE_TRANSCRIBE_ENCODER_LAYERS_KEY, COHERE_TRANSCRIBE_VOCAB_SIZE_KEY,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CohereTranscribeExecutionMetadata {
    pub vocab_size: usize,
    pub encoder_layers: usize,
    pub encoder_d_model: usize,
    pub encoder_heads: usize,
    pub encoder_head_dim: usize,
    pub encoder_ffn_dim: usize,
    pub encoder_conv_kernel: usize,
    pub decoder_layers: usize,
    pub decoder_d_model: usize,
    pub decoder_heads: usize,
    pub decoder_head_dim: usize,
    pub decoder_ffn_dim: usize,
    pub decoder_max_context: usize,
    pub decoder_start_token_id: u32,
    pub sample_rate_hz: u32,
    pub n_mels: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum CohereTranscribeRuntimeContractError {
    #[error("cohere-transcribe missing required GGUF metadata key '{key}'")]
    MissingRequiredMetadata { key: &'static str },
    #[error("cohere-transcribe GGUF metadata '{key}' is invalid: {reason}")]
    InvalidMetadataValue { key: &'static str, reason: String },
    #[error("cohere-transcribe expected general.architecture='{expected}', got '{found}'")]
    UnexpectedArchitecture {
        expected: &'static str,
        found: String,
    },
    #[error("cohere-transcribe missing required GGUF tensor '{name}'")]
    MissingRequiredTensor { name: String },
    #[error("cohere-transcribe GGUF tensor '{name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        name: String,
        shape: String,
        reason: String,
    },
}

pub(crate) fn parse_cohere_transcribe_execution_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<CohereTranscribeExecutionMetadata, CohereTranscribeRuntimeContractError> {
    let architecture = required_string_scalar(metadata, GENERAL_ARCHITECTURE_KEY)
        .map_err(map_metadata_contract_error)?;
    if architecture != COHERE_TRANSCRIBE_ARCHITECTURE_VALUE {
        return Err(
            CohereTranscribeRuntimeContractError::UnexpectedArchitecture {
                expected: COHERE_TRANSCRIBE_ARCHITECTURE_VALUE,
                found: architecture.to_string(),
            },
        );
    }

    let vocab_size = required_usize(metadata, COHERE_TRANSCRIBE_VOCAB_SIZE_KEY)?;
    let encoder_layers = required_usize(metadata, COHERE_TRANSCRIBE_ENCODER_LAYERS_KEY)?;
    let encoder_d_model = required_usize(metadata, COHERE_TRANSCRIBE_ENCODER_D_MODEL_KEY)?;
    let encoder_heads = required_usize(metadata, COHERE_TRANSCRIBE_ENCODER_HEADS_KEY)?;
    let encoder_head_dim = required_usize(metadata, COHERE_TRANSCRIBE_ENCODER_HEAD_DIM_KEY)?;
    let encoder_ffn_dim = required_usize(metadata, COHERE_TRANSCRIBE_ENCODER_FFN_DIM_KEY)?;
    let encoder_conv_kernel = required_usize(metadata, COHERE_TRANSCRIBE_ENCODER_CONV_KERNEL_KEY)?;
    let decoder_layers = required_usize(metadata, COHERE_TRANSCRIBE_DECODER_LAYERS_KEY)?;
    let decoder_d_model = required_usize(metadata, COHERE_TRANSCRIBE_DECODER_D_MODEL_KEY)?;
    let decoder_heads = required_usize(metadata, COHERE_TRANSCRIBE_DECODER_HEADS_KEY)?;
    let decoder_head_dim = required_usize(metadata, COHERE_TRANSCRIBE_DECODER_HEAD_DIM_KEY)?;
    let decoder_ffn_dim = required_usize(metadata, COHERE_TRANSCRIBE_DECODER_FFN_DIM_KEY)?;
    let decoder_max_context = required_usize(metadata, COHERE_TRANSCRIBE_DECODER_MAX_CONTEXT_KEY)?;
    let decoder_start_token_id =
        required_u32(metadata, COHERE_TRANSCRIBE_DECODER_START_TOKEN_ID_KEY)?;
    let sample_rate_hz = required_u32(metadata, COHERE_TRANSCRIBE_AUDIO_SAMPLE_RATE_KEY)?;
    let n_mels = required_usize(metadata, COHERE_TRANSCRIBE_AUDIO_MELS_COUNT_KEY)?;
    let n_fft = required_usize(metadata, COHERE_TRANSCRIBE_AUDIO_N_FFT_KEY)?;
    let hop_length = required_usize(metadata, COHERE_TRANSCRIBE_AUDIO_HOP_LENGTH_KEY)?;
    let win_length = required_usize(metadata, COHERE_TRANSCRIBE_AUDIO_WIN_LENGTH_KEY)?;

    for (value, key) in [
        (vocab_size, COHERE_TRANSCRIBE_VOCAB_SIZE_KEY),
        (encoder_layers, COHERE_TRANSCRIBE_ENCODER_LAYERS_KEY),
        (encoder_d_model, COHERE_TRANSCRIBE_ENCODER_D_MODEL_KEY),
        (encoder_heads, COHERE_TRANSCRIBE_ENCODER_HEADS_KEY),
        (encoder_head_dim, COHERE_TRANSCRIBE_ENCODER_HEAD_DIM_KEY),
        (encoder_ffn_dim, COHERE_TRANSCRIBE_ENCODER_FFN_DIM_KEY),
        (
            encoder_conv_kernel,
            COHERE_TRANSCRIBE_ENCODER_CONV_KERNEL_KEY,
        ),
        (decoder_layers, COHERE_TRANSCRIBE_DECODER_LAYERS_KEY),
        (decoder_d_model, COHERE_TRANSCRIBE_DECODER_D_MODEL_KEY),
        (decoder_heads, COHERE_TRANSCRIBE_DECODER_HEADS_KEY),
        (decoder_head_dim, COHERE_TRANSCRIBE_DECODER_HEAD_DIM_KEY),
        (decoder_ffn_dim, COHERE_TRANSCRIBE_DECODER_FFN_DIM_KEY),
        (
            decoder_max_context,
            COHERE_TRANSCRIBE_DECODER_MAX_CONTEXT_KEY,
        ),
        (n_mels, COHERE_TRANSCRIBE_AUDIO_MELS_COUNT_KEY),
        (n_fft, COHERE_TRANSCRIBE_AUDIO_N_FFT_KEY),
        (hop_length, COHERE_TRANSCRIBE_AUDIO_HOP_LENGTH_KEY),
        (win_length, COHERE_TRANSCRIBE_AUDIO_WIN_LENGTH_KEY),
    ] {
        validate_positive_usize(value, key).map_err(map_metadata_contract_error)?;
    }

    if encoder_heads.saturating_mul(encoder_head_dim) != encoder_d_model {
        return Err(CohereTranscribeRuntimeContractError::InvalidMetadataValue {
            key: COHERE_TRANSCRIBE_ENCODER_D_MODEL_KEY,
            reason: format!(
                "{}={} must equal {}={} * {}={}",
                COHERE_TRANSCRIBE_ENCODER_D_MODEL_KEY,
                encoder_d_model,
                COHERE_TRANSCRIBE_ENCODER_HEADS_KEY,
                encoder_heads,
                COHERE_TRANSCRIBE_ENCODER_HEAD_DIM_KEY,
                encoder_head_dim,
            ),
        });
    }
    if decoder_heads.saturating_mul(decoder_head_dim) != decoder_d_model {
        return Err(CohereTranscribeRuntimeContractError::InvalidMetadataValue {
            key: COHERE_TRANSCRIBE_DECODER_D_MODEL_KEY,
            reason: format!(
                "{}={} must equal {}={} * {}={}",
                COHERE_TRANSCRIBE_DECODER_D_MODEL_KEY,
                decoder_d_model,
                COHERE_TRANSCRIBE_DECODER_HEADS_KEY,
                decoder_heads,
                COHERE_TRANSCRIBE_DECODER_HEAD_DIM_KEY,
                decoder_head_dim,
            ),
        });
    }
    if hop_length > win_length || win_length > n_fft {
        return Err(CohereTranscribeRuntimeContractError::InvalidMetadataValue {
            key: COHERE_TRANSCRIBE_AUDIO_WIN_LENGTH_KEY,
            reason: format!(
                "{}={} and {}={} must satisfy hop <= win <= fft ({})",
                COHERE_TRANSCRIBE_AUDIO_HOP_LENGTH_KEY,
                hop_length,
                COHERE_TRANSCRIBE_AUDIO_WIN_LENGTH_KEY,
                win_length,
                n_fft,
            ),
        });
    }

    Ok(CohereTranscribeExecutionMetadata {
        vocab_size,
        encoder_layers,
        encoder_d_model,
        encoder_heads,
        encoder_head_dim,
        encoder_ffn_dim,
        encoder_conv_kernel,
        decoder_layers,
        decoder_d_model,
        decoder_heads,
        decoder_head_dim,
        decoder_ffn_dim,
        decoder_max_context,
        decoder_start_token_id,
        sample_rate_hz,
        n_mels,
        n_fft,
        hop_length,
        win_length,
    })
}

pub(crate) fn validate_cohere_transcribe_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    metadata: CohereTranscribeExecutionMetadata,
) -> Result<(), CohereTranscribeRuntimeContractError> {
    let fft_bins = metadata
        .n_fft
        .checked_div(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(
            || CohereTranscribeRuntimeContractError::InvalidMetadataValue {
                key: COHERE_TRANSCRIBE_AUDIO_N_FFT_KEY,
                reason: "n_fft overflow while computing FFT bin count".to_string(),
            },
        )?;

    let mel_fb = require_tensor(index, FE_MEL_FB)?;
    if mel_fb.dims != vec![fft_bins as u64, metadata.n_mels as u64] {
        return Err(invalid_tensor_shape(
            FE_MEL_FB,
            &mel_fb.dims,
            format!("expected [{}, {}]", fft_bins, metadata.n_mels),
        ));
    }
    let descriptors = resolve_builtin_runtime_tensor_contract_descriptors(
        cohere_runtime_tensor_contract_id(),
        RuntimeTensorContractMetadata::CohereTranscribe(metadata),
    )
    .expect("cohere builtin runtime tensor contract must resolve");
    validate_tensor_binding_descriptors(
        index,
        &descriptors,
        missing_required_tensor,
        invalid_tensor_shape,
    )?;

    for tensor_name in [
        enc_pre_conv_weight(0),
        enc_pre_conv_weight(2),
        enc_pre_conv_weight(5),
    ] {
        let tensor = require_tensor(index, &tensor_name)?;
        if tensor.dims.len() != 4 {
            return Err(invalid_tensor_shape(
                &tensor_name,
                &tensor.dims,
                "expected rank-4 conv weight tensor".to_string(),
            ));
        }
    }
    for tensor_name in [enc_pre_conv_weight(3), enc_pre_conv_weight(6)] {
        let tensor = require_tensor(index, &tensor_name)?;
        if tensor.dims.len() != 4 && tensor.dims.len() != 2 {
            return Err(invalid_tensor_shape(
                &tensor_name,
                &tensor.dims,
                "expected rank-4 conv tensor or rank-2 folded 1x1 conv tensor".to_string(),
            ));
        }
    }

    Ok(())
}

pub(crate) fn cohere_transcribe_runtime_tensor_descriptors(
    metadata: CohereTranscribeExecutionMetadata,
) -> Vec<TensorBindingDescriptor> {
    let top_level_bindings = [
        TensorBindingSpec {
            tensor_name: FE_WINDOW,
            requirement: TensorBindingRequirement::VectorLen(metadata.win_length),
            reason: "expected window vector",
        },
        TensorBindingSpec {
            tensor_name: ENC_PRE_OUT_WEIGHT,
            requirement: TensorBindingRequirement::Rank2WithDim(metadata.encoder_d_model),
            reason: "expected rank-2 pre-out matrix with one dimension = encoder hidden size",
        },
        TensorBindingSpec {
            tensor_name: ENC_PRE_OUT_BIAS,
            requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
            reason: "expected encoder hidden-size bias",
        },
        TensorBindingSpec {
            tensor_name: ENC_PROJ_WEIGHT,
            requirement: TensorBindingRequirement::Rank2EitherDims(
                metadata.encoder_d_model,
                metadata.decoder_d_model,
            ),
            reason: "expected encoder->decoder projection matrix",
        },
        TensorBindingSpec {
            tensor_name: ENC_PROJ_BIAS,
            requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
            reason: "expected decoder hidden-size bias",
        },
        TensorBindingSpec {
            tensor_name: DEC_EMB_WEIGHT,
            requirement: TensorBindingRequirement::Rank2EitherDims(
                metadata.vocab_size,
                metadata.decoder_d_model,
            ),
            reason: "expected vocab/decoder embedding matrix",
        },
        TensorBindingSpec {
            tensor_name: DEC_POS_WEIGHT,
            requirement: TensorBindingRequirement::Rank2EitherDims(
                metadata.decoder_max_context,
                metadata.decoder_d_model,
            ),
            reason: "expected decoder positional embedding matrix",
        },
        TensorBindingSpec {
            tensor_name: DEC_EMB_LN_WEIGHT,
            requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
            reason: "expected decoder hidden-size vector",
        },
        TensorBindingSpec {
            tensor_name: DEC_EMB_LN_BIAS,
            requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
            reason: "expected decoder hidden-size vector",
        },
        TensorBindingSpec {
            tensor_name: DEC_OUT_LN_WEIGHT,
            requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
            reason: "expected decoder hidden-size vector",
        },
        TensorBindingSpec {
            tensor_name: DEC_OUT_LN_BIAS,
            requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
            reason: "expected decoder hidden-size vector",
        },
        TensorBindingSpec {
            tensor_name: DEC_HEAD_WEIGHT,
            requirement: TensorBindingRequirement::Rank2EitherDims(
                metadata.vocab_size,
                metadata.decoder_d_model,
            ),
            reason: "expected decoder vocab projection matrix",
        },
        TensorBindingSpec {
            tensor_name: DEC_HEAD_BIAS,
            requirement: TensorBindingRequirement::VectorLen(metadata.vocab_size),
            reason: "expected vocab-sized head bias",
        },
    ];
    let mut descriptors = tensor_binding_descriptors(&top_level_bindings);
    for layer_idx in 0..metadata.encoder_layers {
        let names = encoder_layer_tensor_names(layer_idx);
        let bindings = [
            TensorBindingSpec {
                tensor_name: names.ff1_norm_weight.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.ff1_norm_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.attn_norm_weight.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.attn_norm_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.conv_norm_weight.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.conv_norm_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.conv_bn_weight.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.conv_bn_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.conv_bn_mean.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.conv_bn_var.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.ff2_norm_weight.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.ff2_norm_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.out_norm_weight.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.out_norm_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.attn_q_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2WithDim(metadata.encoder_d_model),
                reason: "expected rank-2 encoder attention matrix with one dimension = encoder hidden size",
            },
            TensorBindingSpec {
                tensor_name: names.attn_k_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2WithDim(metadata.encoder_d_model),
                reason: "expected rank-2 encoder attention matrix with one dimension = encoder hidden size",
            },
            TensorBindingSpec {
                tensor_name: names.attn_v_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2WithDim(metadata.encoder_d_model),
                reason: "expected rank-2 encoder attention matrix with one dimension = encoder hidden size",
            },
            TensorBindingSpec {
                tensor_name: names.attn_out_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2WithDim(metadata.encoder_d_model),
                reason: "expected rank-2 encoder attention matrix with one dimension = encoder hidden size",
            },
            TensorBindingSpec {
                tensor_name: names.attn_pos_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2WithDim(metadata.encoder_d_model),
                reason: "expected rank-2 encoder attention matrix with one dimension = encoder hidden size",
            },
            TensorBindingSpec {
                tensor_name: names.attn_q_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size bias",
            },
            TensorBindingSpec {
                tensor_name: names.attn_k_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size bias",
            },
            TensorBindingSpec {
                tensor_name: names.attn_v_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size bias",
            },
            TensorBindingSpec {
                tensor_name: names.attn_out_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size bias",
            },
            TensorBindingSpec {
                tensor_name: names.attn_pos_bias_u.as_str(),
                requirement: TensorBindingRequirement::Rank2EitherDims(
                    metadata.encoder_heads,
                    metadata.encoder_head_dim,
                ),
                reason: "expected [heads, head_dim] positional bias matrix",
            },
            TensorBindingSpec {
                tensor_name: names.attn_pos_bias_v.as_str(),
                requirement: TensorBindingRequirement::Rank2EitherDims(
                    metadata.encoder_heads,
                    metadata.encoder_head_dim,
                ),
                reason: "expected [heads, head_dim] positional bias matrix",
            },
            TensorBindingSpec {
                tensor_name: names.ff1_up_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2EitherDims(
                    metadata.encoder_ffn_dim,
                    metadata.encoder_d_model,
                ),
                reason: "expected encoder FFN up matrix",
            },
            TensorBindingSpec {
                tensor_name: names.ff2_up_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2EitherDims(
                    metadata.encoder_ffn_dim,
                    metadata.encoder_d_model,
                ),
                reason: "expected encoder FFN up matrix",
            },
            TensorBindingSpec {
                tensor_name: names.ff1_down_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2EitherDims(
                    metadata.encoder_d_model,
                    metadata.encoder_ffn_dim,
                ),
                reason: "expected encoder FFN down matrix",
            },
            TensorBindingSpec {
                tensor_name: names.ff2_down_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2EitherDims(
                    metadata.encoder_d_model,
                    metadata.encoder_ffn_dim,
                ),
                reason: "expected encoder FFN down matrix",
            },
            TensorBindingSpec {
                tensor_name: names.ff1_up_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_ffn_dim),
                reason: "expected encoder FFN bias",
            },
            TensorBindingSpec {
                tensor_name: names.ff2_up_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_ffn_dim),
                reason: "expected encoder FFN bias",
            },
            TensorBindingSpec {
                tensor_name: names.ff1_down_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size bias",
            },
            TensorBindingSpec {
                tensor_name: names.ff2_down_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected encoder hidden-size bias",
            },
            TensorBindingSpec {
                tensor_name: names.conv_pw1_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2OrRank3WithDims(
                    metadata.encoder_d_model * 2,
                    metadata.encoder_d_model,
                ),
                reason: "expected pointwise conv tensor with 2*d_model and d_model dimensions",
            },
            TensorBindingSpec {
                tensor_name: names.conv_pw1_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model * 2),
                reason: "expected pointwise conv bias",
            },
            TensorBindingSpec {
                tensor_name: names.conv_dw_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2OrRank3WithDims(
                    metadata.encoder_d_model,
                    metadata.encoder_conv_kernel,
                ),
                reason: "expected depthwise conv tensor with d_model and conv_kernel dimensions",
            },
            TensorBindingSpec {
                tensor_name: names.conv_dw_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected depthwise conv bias",
            },
            TensorBindingSpec {
                tensor_name: names.conv_pw2_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2OrRank3WithDims(
                    metadata.encoder_d_model,
                    metadata.encoder_d_model,
                ),
                reason: "expected pointwise conv tensor with encoder hidden size dimensions",
            },
            TensorBindingSpec {
                tensor_name: names.conv_pw2_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.encoder_d_model),
                reason: "expected pointwise conv bias",
            },
        ];
        descriptors.extend(tensor_binding_descriptors(&bindings));
    }
    for layer_idx in 0..metadata.decoder_layers {
        let names = decoder_layer_tensor_names(layer_idx);
        let bindings = [
            TensorBindingSpec {
                tensor_name: names.attn_ln_weight.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
                reason: "expected decoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.attn_ln_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
                reason: "expected decoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.cross_ln_weight.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
                reason: "expected decoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.cross_ln_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
                reason: "expected decoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.ffn_ln_weight.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
                reason: "expected decoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.ffn_ln_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
                reason: "expected decoder hidden-size vector",
            },
            TensorBindingSpec {
                tensor_name: names.attn_q_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2WithDim(metadata.decoder_d_model),
                reason: "expected rank-2 decoder matrix with one dimension = decoder hidden size",
            },
            TensorBindingSpec {
                tensor_name: names.attn_k_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2WithDim(metadata.decoder_d_model),
                reason: "expected rank-2 decoder matrix with one dimension = decoder hidden size",
            },
            TensorBindingSpec {
                tensor_name: names.attn_v_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2WithDim(metadata.decoder_d_model),
                reason: "expected rank-2 decoder matrix with one dimension = decoder hidden size",
            },
            TensorBindingSpec {
                tensor_name: names.attn_o_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2WithDim(metadata.decoder_d_model),
                reason: "expected rank-2 decoder matrix with one dimension = decoder hidden size",
            },
            TensorBindingSpec {
                tensor_name: names.cross_q_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2WithDim(metadata.decoder_d_model),
                reason: "expected rank-2 decoder matrix with one dimension = decoder hidden size",
            },
            TensorBindingSpec {
                tensor_name: names.cross_k_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2WithDim(metadata.decoder_d_model),
                reason: "expected rank-2 decoder matrix with one dimension = decoder hidden size",
            },
            TensorBindingSpec {
                tensor_name: names.cross_v_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2WithDim(metadata.decoder_d_model),
                reason: "expected rank-2 decoder matrix with one dimension = decoder hidden size",
            },
            TensorBindingSpec {
                tensor_name: names.cross_o_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2WithDim(metadata.decoder_d_model),
                reason: "expected rank-2 decoder matrix with one dimension = decoder hidden size",
            },
            TensorBindingSpec {
                tensor_name: names.attn_q_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
                reason: "expected decoder hidden-size bias",
            },
            TensorBindingSpec {
                tensor_name: names.attn_k_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
                reason: "expected decoder hidden-size bias",
            },
            TensorBindingSpec {
                tensor_name: names.attn_v_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
                reason: "expected decoder hidden-size bias",
            },
            TensorBindingSpec {
                tensor_name: names.attn_o_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
                reason: "expected decoder hidden-size bias",
            },
            TensorBindingSpec {
                tensor_name: names.cross_q_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
                reason: "expected decoder hidden-size bias",
            },
            TensorBindingSpec {
                tensor_name: names.cross_k_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
                reason: "expected decoder hidden-size bias",
            },
            TensorBindingSpec {
                tensor_name: names.cross_v_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
                reason: "expected decoder hidden-size bias",
            },
            TensorBindingSpec {
                tensor_name: names.cross_o_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
                reason: "expected decoder hidden-size bias",
            },
            TensorBindingSpec {
                tensor_name: names.ffn_up_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2EitherDims(
                    metadata.decoder_ffn_dim,
                    metadata.decoder_d_model,
                ),
                reason: "expected decoder FFN up matrix",
            },
            TensorBindingSpec {
                tensor_name: names.ffn_up_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_ffn_dim),
                reason: "expected decoder FFN bias",
            },
            TensorBindingSpec {
                tensor_name: names.ffn_down_weight.as_str(),
                requirement: TensorBindingRequirement::Rank2EitherDims(
                    metadata.decoder_d_model,
                    metadata.decoder_ffn_dim,
                ),
                reason: "expected decoder FFN down matrix",
            },
            TensorBindingSpec {
                tensor_name: names.ffn_down_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
                reason: "expected decoder hidden-size bias",
            },
        ];
        descriptors.extend(tensor_binding_descriptors(&bindings));
    }
    descriptors
}

fn cohere_runtime_tensor_contract_id() -> &'static str {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID)
        .expect("cohere architecture must be registered")
        .runtime_tensor_contract_id
}

fn required_usize<M: ScalarMetadataView>(
    metadata: &M,
    key: &'static str,
) -> Result<usize, CohereTranscribeRuntimeContractError> {
    let value = required_u64_scalar(metadata, key).map_err(map_metadata_contract_error)?;
    u64_to_usize(value, key).map_err(map_metadata_contract_error)
}

fn required_u32<M: ScalarMetadataView>(
    metadata: &M,
    key: &'static str,
) -> Result<u32, CohereTranscribeRuntimeContractError> {
    let value = required_u64_scalar(metadata, key).map_err(map_metadata_contract_error)?;
    u64_to_u32(value, key).map_err(map_metadata_contract_error)
}

fn map_metadata_contract_error(
    error: MetadataContractError,
) -> CohereTranscribeRuntimeContractError {
    match error {
        MetadataContractError::MissingRequiredKey { key } => {
            CohereTranscribeRuntimeContractError::MissingRequiredMetadata { key }
        }
        MetadataContractError::InvalidValue { key, reason } => {
            CohereTranscribeRuntimeContractError::InvalidMetadataValue { key, reason }
        }
    }
}

fn require_tensor<'a>(
    index: &'a GgufTensorIndex,
    name: &str,
) -> Result<&'a GgufTensorMetadata, CohereTranscribeRuntimeContractError> {
    require_tensor_binding(index, name, missing_required_tensor)
}

fn missing_required_tensor(name: &str) -> CohereTranscribeRuntimeContractError {
    CohereTranscribeRuntimeContractError::MissingRequiredTensor {
        name: name.to_string(),
    }
}

fn invalid_tensor_shape(
    name: &str,
    shape: &[u64],
    reason: String,
) -> CohereTranscribeRuntimeContractError {
    CohereTranscribeRuntimeContractError::InvalidTensorShape {
        name: name.to_string(),
        shape: render_shape(shape),
        reason,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{
        read_gguf_tensor_index,
        testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source},
    };
    use tempfile::NamedTempFile;

    fn base_metadata() -> BTreeMap<String, String> {
        [
            (
                GENERAL_ARCHITECTURE_KEY,
                COHERE_TRANSCRIBE_ARCHITECTURE_VALUE,
            ),
            (COHERE_TRANSCRIBE_VOCAB_SIZE_KEY, "16384"),
            (COHERE_TRANSCRIBE_ENCODER_LAYERS_KEY, "48"),
            (COHERE_TRANSCRIBE_ENCODER_D_MODEL_KEY, "1280"),
            (COHERE_TRANSCRIBE_ENCODER_HEADS_KEY, "8"),
            (COHERE_TRANSCRIBE_ENCODER_HEAD_DIM_KEY, "160"),
            (COHERE_TRANSCRIBE_ENCODER_FFN_DIM_KEY, "5120"),
            (COHERE_TRANSCRIBE_ENCODER_CONV_KERNEL_KEY, "9"),
            (COHERE_TRANSCRIBE_DECODER_LAYERS_KEY, "8"),
            (COHERE_TRANSCRIBE_DECODER_D_MODEL_KEY, "1024"),
            (COHERE_TRANSCRIBE_DECODER_HEADS_KEY, "8"),
            (COHERE_TRANSCRIBE_DECODER_HEAD_DIM_KEY, "128"),
            (COHERE_TRANSCRIBE_DECODER_FFN_DIM_KEY, "4096"),
            (COHERE_TRANSCRIBE_DECODER_MAX_CONTEXT_KEY, "1024"),
            (COHERE_TRANSCRIBE_DECODER_START_TOKEN_ID_KEY, "13764"),
            (COHERE_TRANSCRIBE_AUDIO_SAMPLE_RATE_KEY, "16000"),
            (COHERE_TRANSCRIBE_AUDIO_MELS_COUNT_KEY, "128"),
            (COHERE_TRANSCRIBE_AUDIO_N_FFT_KEY, "512"),
            (COHERE_TRANSCRIBE_AUDIO_HOP_LENGTH_KEY, "160"),
            (COHERE_TRANSCRIBE_AUDIO_WIN_LENGTH_KEY, "400"),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
    }

    #[test]
    fn parses_reference_cohere_metadata() {
        let metadata = base_metadata();
        let parsed = parse_cohere_transcribe_execution_metadata(&metadata).expect("must parse");
        assert_eq!(parsed.encoder_d_model, 1280);
        assert_eq!(parsed.decoder_head_dim, 128);
        assert_eq!(parsed.decoder_start_token_id, 13_764);
        assert_eq!(parsed.sample_rate_hz, 16_000);
    }

    #[test]
    fn rejects_unexpected_architecture() {
        let mut metadata = base_metadata();
        metadata.insert(
            GENERAL_ARCHITECTURE_KEY.to_string(),
            "cohere-transcribe-typo".to_string(),
        );
        let error = parse_cohere_transcribe_execution_metadata(&metadata).expect_err("must fail");
        assert!(matches!(
            error,
            CohereTranscribeRuntimeContractError::UnexpectedArchitecture { .. }
        ));
    }

    #[test]
    fn rejects_inconsistent_head_geometry() {
        let mut metadata = base_metadata();
        metadata.insert(
            COHERE_TRANSCRIBE_ENCODER_HEAD_DIM_KEY.to_string(),
            "128".to_string(),
        );
        let error = parse_cohere_transcribe_execution_metadata(&metadata).expect_err("must fail");
        assert!(matches!(
            error,
            CohereTranscribeRuntimeContractError::InvalidMetadataValue {
                key: COHERE_TRANSCRIBE_ENCODER_D_MODEL_KEY,
                ..
            }
        ));
    }

    #[test]
    fn validates_runtime_ready_fixture_tensors() {
        let file = NamedTempFile::new().expect("temp file");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(file.path(), &spec).expect("write fixture");

        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");
        let metadata = parse_cohere_transcribe_execution_metadata(&spec.metadata)
            .expect("runtime-ready metadata must parse");

        validate_cohere_transcribe_runtime_tensors_with_index(&index, metadata)
            .expect("runtime-ready tensor fixture must validate");
    }

    #[test]
    fn rejects_runtime_fixture_missing_required_tensor() {
        let file = NamedTempFile::new().expect("temp file");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture")
            .without_tensor("dec.out_ln.weight");
        write_tiny_gguf_runtime_source(file.path(), &spec).expect("write fixture");

        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");
        let metadata = parse_cohere_transcribe_execution_metadata(&spec.metadata)
            .expect("runtime-ready metadata must parse");
        let error = validate_cohere_transcribe_runtime_tensors_with_index(&index, metadata)
            .expect_err("missing tensor must fail closed");

        assert!(matches!(
            error,
            CohereTranscribeRuntimeContractError::MissingRequiredTensor { ref name }
                if name == "dec.out_ln.weight"
        ));
    }

    #[test]
    fn rejects_runtime_fixture_shape_mismatch() {
        let file = NamedTempFile::new().expect("temp file");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture")
            .with_tensor_shape("fe.mel_fb", [99_u64, 8_u64]);
        write_tiny_gguf_runtime_source(file.path(), &spec).expect("write fixture");

        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");
        let metadata = parse_cohere_transcribe_execution_metadata(&spec.metadata)
            .expect("runtime-ready metadata must parse");
        let error = validate_cohere_transcribe_runtime_tensors_with_index(&index, metadata)
            .expect_err("shape mismatch must fail closed");

        assert!(matches!(
            error,
            CohereTranscribeRuntimeContractError::InvalidTensorShape { ref name, .. }
                if name == "fe.mel_fb"
        ));
    }
}
