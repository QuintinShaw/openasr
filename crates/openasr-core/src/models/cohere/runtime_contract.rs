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
    // The decoder start token indexes the same vocab the decode loop argmax-es
    // over; a start id at or beyond vocab_size can never be fed to the
    // embedding rows the pack ships. Fail closed here, the same "token ids
    // must stay inside the declared vocab" rule the moss-transcribe-diarize
    // contract applies to its audio control tokens.
    if (decoder_start_token_id as usize) >= vocab_size {
        return Err(CohereTranscribeRuntimeContractError::InvalidMetadataValue {
            key: COHERE_TRANSCRIBE_DECODER_START_TOKEN_ID_KEY,
            reason: format!(
                "token id {decoder_start_token_id} out of range for vocab_size {vocab_size}"
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
    // Ordered ggml [in, out] for mul_mat projections (and the pack embedding
    // layout the decoder get_rows path ships). Lifetime-bound ExactDims slices
    // below are copied into owned descriptors before this frame returns.
    let enc_proj_dims = [metadata.encoder_d_model, metadata.decoder_d_model];
    let dec_emb_dims = [metadata.vocab_size, metadata.decoder_d_model];
    let dec_pos_dims = [metadata.decoder_max_context, metadata.decoder_d_model];
    let dec_head_dims = [metadata.decoder_d_model, metadata.vocab_size];
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
            // mul_mat weight is ggml [enc_d_model, dec_d_model] = [in, out].
            requirement: TensorBindingRequirement::ExactDims(&enc_proj_dims),
            reason: "encoder->decoder projection must be ggml [encoder_d_model, decoder_d_model]",
        },
        TensorBindingSpec {
            tensor_name: ENC_PROJ_BIAS,
            requirement: TensorBindingRequirement::VectorLen(metadata.decoder_d_model),
            reason: "expected decoder hidden-size bias",
        },
        TensorBindingSpec {
            tensor_name: DEC_EMB_WEIGHT,
            // Pack ships RowsByColumns [vocab, d_model] (importer does not reverse
            // embeddings); decoder get_rows materializes from that layout.
            requirement: TensorBindingRequirement::ExactDims(&dec_emb_dims),
            reason: "token embedding table must be [vocab_size, decoder_d_model]",
        },
        TensorBindingSpec {
            tensor_name: DEC_POS_WEIGHT,
            requirement: TensorBindingRequirement::ExactDims(&dec_pos_dims),
            reason: "positional embedding table must be [decoder_max_context, decoder_d_model]",
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
            // mul_mat weight is ggml [d_model, vocab] = [in, out].
            requirement: TensorBindingRequirement::ExactDims(&dec_head_dims),
            reason: "decoder vocab projection must be ggml [decoder_d_model, vocab_size]",
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
        let enc_ffn_up_dims = [metadata.encoder_d_model, metadata.encoder_ffn_dim];
        let enc_ffn_down_dims = [metadata.encoder_ffn_dim, metadata.encoder_d_model];
        let enc_pos_bias_dims = [metadata.encoder_head_dim, metadata.encoder_heads];
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
                requirement: TensorBindingRequirement::ExactDims(&enc_pos_bias_dims),
                reason: "Transformer-XL pos_bias_u must be GGUF [head_dim, n_heads]; bytes stay HF head-major and are flattened without transpose",
            },
            TensorBindingSpec {
                tensor_name: names.attn_pos_bias_v.as_str(),
                requirement: TensorBindingRequirement::ExactDims(&enc_pos_bias_dims),
                reason: "Transformer-XL pos_bias_v must be GGUF [head_dim, n_heads]; bytes stay HF head-major and are flattened without transpose",
            },
            TensorBindingSpec {
                tensor_name: names.ff1_up_weight.as_str(),
                // mul_mat [in, out] = [d_model, ffn]
                requirement: TensorBindingRequirement::ExactDims(&enc_ffn_up_dims),
                reason: "encoder FFN up must be ggml [encoder_d_model, encoder_ffn_dim]",
            },
            TensorBindingSpec {
                tensor_name: names.ff2_up_weight.as_str(),
                requirement: TensorBindingRequirement::ExactDims(&enc_ffn_up_dims),
                reason: "encoder FFN up must be ggml [encoder_d_model, encoder_ffn_dim]",
            },
            TensorBindingSpec {
                tensor_name: names.ff1_down_weight.as_str(),
                // mul_mat [in, out] = [ffn, d_model]
                requirement: TensorBindingRequirement::ExactDims(&enc_ffn_down_dims),
                reason: "encoder FFN down must be ggml [encoder_ffn_dim, encoder_d_model]",
            },
            TensorBindingSpec {
                tensor_name: names.ff2_down_weight.as_str(),
                requirement: TensorBindingRequirement::ExactDims(&enc_ffn_down_dims),
                reason: "encoder FFN down must be ggml [encoder_ffn_dim, encoder_d_model]",
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
        let dec_ffn_up_dims = [metadata.decoder_d_model, metadata.decoder_ffn_dim];
        let dec_ffn_down_dims = [metadata.decoder_ffn_dim, metadata.decoder_d_model];
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
                // mul_mat [in, out] = [d_model, ffn]
                requirement: TensorBindingRequirement::ExactDims(&dec_ffn_up_dims),
                reason: "decoder FFN up must be ggml [decoder_d_model, decoder_ffn_dim]",
            },
            TensorBindingSpec {
                tensor_name: names.ffn_up_bias.as_str(),
                requirement: TensorBindingRequirement::VectorLen(metadata.decoder_ffn_dim),
                reason: "expected decoder FFN bias",
            },
            TensorBindingSpec {
                tensor_name: names.ffn_down_weight.as_str(),
                // mul_mat [in, out] = [ffn, d_model]
                requirement: TensorBindingRequirement::ExactDims(&dec_ffn_down_dims),
                reason: "decoder FFN down must be ggml [decoder_ffn_dim, decoder_d_model]",
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
        .pack_contract
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

pub(crate) fn validate_runtime_pack_contract(
    preflight: &crate::GgufRuntimeSourcePreflight,
) -> Result<(), String> {
    crate::models::runtime_tensor_contract_registry::validate_builtin_runtime_tensor_contract_for_architecture(
        crate::arch::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
        preflight.metadata(),
        preflight.tensor_index(),
    )
    .map(|_| ())
    .map_err(crate::models::runtime_pack_contract::tensor_validation_error)
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

    /// Capacity regression anchor: the shared KV byte derivation on this
    /// family's real-checkpoint decoder geometry (8 layers, MHA so 8 KV
    /// heads, head_dim 128 -- the fixture values above), split by storage
    /// copy. Runs the derivation golden for every `Derived` family, not just
    /// the one that consumes an integral window today.
    #[test]
    fn kv_bytes_per_position_matches_the_reference_decoder_geometry() {
        use crate::capacity::{KvGeometry, kv_bytes_per_position};
        use crate::nn::decoder::LlmKvCacheSpec;

        let geometry = KvGeometry {
            n_layers: 8,
            kv_heads: 8,
            head_dim: 128,
        };
        // 8 layers * 2 (K+V) * 8 kv-heads = 128 rows per position.
        let default = kv_bytes_per_position(&geometry, LlmKvCacheSpec::DEFAULT).expect("default");
        assert_eq!(default.host, 128 * 512); // f32 rows
        assert_eq!(default.resident, 128 * 256); // f16 rows
        let q8_0 = kv_bytes_per_position(&geometry, LlmKvCacheSpec::Q8_0).expect("q8_0");
        assert_eq!(q8_0.host, 128 * 136); // 128 / 32 * 34 B q8_0 rows
        assert_eq!(q8_0.resident, 128 * 136);
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
    fn rejects_decoder_start_token_id_out_of_vocab() {
        let mut metadata = base_metadata();
        metadata.insert(
            COHERE_TRANSCRIBE_DECODER_START_TOKEN_ID_KEY.to_string(),
            "16384".to_string(),
        );
        let error = parse_cohere_transcribe_execution_metadata(&metadata).expect_err("must fail");
        assert!(matches!(
            error,
            CohereTranscribeRuntimeContractError::InvalidMetadataValue {
                key: COHERE_TRANSCRIBE_DECODER_START_TOKEN_ID_KEY,
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

    /// Ordered ExactDims must reject HF [out, in] that Rank2EitherDims admitted
    /// for mul_mat projections (enc.proj / FFN / dec.head).
    #[test]
    fn rejects_transposed_encoder_projection_weight() {
        let file = NamedTempFile::new().expect("temp file");
        // Default fixture is square enc/dec d_model=16; use a non-square
        // geometry so the HF [out, in] transpose is distinguishable.
        let base = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture")
            .with_metadata("cohere_transcribe.decoder.d_model", "24")
            .with_metadata("cohere_transcribe.decoder.head_dim", "12")
            .with_metadata("cohere_transcribe.decoder.ffn_dim", "48")
            .with_cohere_runtime_tensors_with_layers(2, 2);
        let metadata = parse_cohere_transcribe_execution_metadata(&base.metadata)
            .expect("runtime-ready metadata must parse");
        assert_ne!(
            metadata.encoder_d_model, metadata.decoder_d_model,
            "test requires rectangular enc.proj"
        );
        // Canonical is [enc_d, dec_d]; force the HF [dec_d, enc_d] orientation.
        let spec = base.with_tensor_shape(
            ENC_PROJ_WEIGHT,
            [
                metadata.decoder_d_model as u64,
                metadata.encoder_d_model as u64,
            ],
        );
        write_tiny_gguf_runtime_source(file.path(), &spec).expect("write fixture");
        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");
        let error = validate_cohere_transcribe_runtime_tensors_with_index(&index, metadata)
            .expect_err("transposed enc.proj.weight must fail closed");
        assert!(matches!(
            error,
            CohereTranscribeRuntimeContractError::InvalidTensorShape { ref name, .. }
                if name == ENC_PROJ_WEIGHT
        ));
    }

    #[test]
    fn rejects_transposed_transformer_xl_positional_bias() {
        let file = NamedTempFile::new().expect("temp file");
        let base = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        let metadata = parse_cohere_transcribe_execution_metadata(&base.metadata)
            .expect("runtime-ready metadata must parse");
        assert_ne!(
            metadata.encoder_heads, metadata.encoder_head_dim,
            "test requires distinguishable positional-bias dimensions"
        );
        let name = encoder_layer_tensor_names(0).attn_pos_bias_u;
        // Canonical GGUF dims are [head_dim, n_heads]. Re-introducing source
        // HF metadata order [n_heads, head_dim] must fail before the loader can
        // transpose the already head-major payload a second time.
        let spec = base.with_tensor_shape(
            name.clone(),
            [
                metadata.encoder_heads as u64,
                metadata.encoder_head_dim as u64,
            ],
        );
        write_tiny_gguf_runtime_source(file.path(), &spec).expect("write fixture");
        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");
        let error = validate_cohere_transcribe_runtime_tensors_with_index(&index, metadata)
            .expect_err("transposed positional bias must fail closed");
        assert!(matches!(
            error,
            CohereTranscribeRuntimeContractError::InvalidTensorShape { name: ref bad, .. }
                if bad == &name
        ));
    }

    #[test]
    fn rejects_transposed_decoder_head_weight() {
        let file = NamedTempFile::new().expect("temp file");
        let base = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        let metadata = parse_cohere_transcribe_execution_metadata(&base.metadata)
            .expect("runtime-ready metadata must parse");
        // Canonical is [d_model, vocab]; force [vocab, d_model].
        let spec = base.with_tensor_shape(
            DEC_HEAD_WEIGHT,
            [metadata.vocab_size as u64, metadata.decoder_d_model as u64],
        );
        write_tiny_gguf_runtime_source(file.path(), &spec).expect("write fixture");
        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");
        let error = validate_cohere_transcribe_runtime_tensors_with_index(&index, metadata)
            .expect_err("transposed dec.head.weight must fail closed");
        assert!(matches!(
            error,
            CohereTranscribeRuntimeContractError::InvalidTensorShape { ref name, .. }
                if name == DEC_HEAD_WEIGHT
        ));
    }

    #[test]
    fn rejects_transposed_encoder_ffn_up_weight() {
        let file = NamedTempFile::new().expect("temp file");
        let base = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        let metadata = parse_cohere_transcribe_execution_metadata(&base.metadata)
            .expect("runtime-ready metadata must parse");
        let names = encoder_layer_tensor_names(0);
        // Canonical is [d_model, ffn]; force [ffn, d_model].
        let spec = base.with_tensor_shape(
            &names.ff1_up_weight,
            [
                metadata.encoder_ffn_dim as u64,
                metadata.encoder_d_model as u64,
            ],
        );
        write_tiny_gguf_runtime_source(file.path(), &spec).expect("write fixture");
        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");
        let error = validate_cohere_transcribe_runtime_tensors_with_index(&index, metadata)
            .expect_err("transposed ff1.up.weight must fail closed");
        assert!(matches!(
            error,
            CohereTranscribeRuntimeContractError::InvalidTensorShape { ref name, .. }
                if name.as_str() == names.ff1_up_weight
        ));
    }

    #[test]
    fn rejects_transposed_token_embedding_weight() {
        let file = NamedTempFile::new().expect("temp file");
        let base = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        let metadata = parse_cohere_transcribe_execution_metadata(&base.metadata)
            .expect("runtime-ready metadata must parse");
        // Canonical pack layout is [vocab, d_model]; force [d_model, vocab].
        let spec = base.with_tensor_shape(
            DEC_EMB_WEIGHT,
            [metadata.decoder_d_model as u64, metadata.vocab_size as u64],
        );
        write_tiny_gguf_runtime_source(file.path(), &spec).expect("write fixture");
        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");
        let error = validate_cohere_transcribe_runtime_tensors_with_index(&index, metadata)
            .expect_err("transposed dec.emb.weight must fail closed");
        assert!(matches!(
            error,
            CohereTranscribeRuntimeContractError::InvalidTensorShape { ref name, .. }
                if name == DEC_EMB_WEIGHT
        ));
    }
}
