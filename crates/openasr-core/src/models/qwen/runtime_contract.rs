use thiserror::Error;

use crate::GgufTensorIndex;
use crate::arch::{
    GENERAL_ARCHITECTURE_KEY, OpenAsrArchitectureRegistry, QWEN3_ASR_GGML_ARCHITECTURE_ID,
};
use crate::models::decode_policy_component_registry::BuiltinSeq2SeqDecodePolicyTokenSource;
use crate::models::phrase_bias_decode::PhraseBiasTokenEncoder;
use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_string_scalar, required_u64_scalar,
    u64_to_u32, u64_to_usize, validate_positive_usize,
};
use crate::models::runtime_tensor_contract_registry::{
    RuntimeTensorContractMetadata, resolve_builtin_runtime_tensor_contract_descriptors,
};
use crate::models::tensor_binding::{
    TensorBindingDescriptor, TensorBindingDescriptorRequirement, render_shape,
    require_tensor as require_tensor_binding, validate_tensor_binding_descriptors,
};

use super::tensor_names::{
    AUDIO_CONV_OUT_WEIGHT, AUDIO_MEL_FILTERS, AUDIO_MEL_WINDOW, OUTPUT_NORM_WEIGHT, OUTPUT_WEIGHT,
    TOKEN_EMBD_WEIGHT, audio_layer_tensor_names, llm_layer_tensor_names,
};

pub(crate) use crate::arch::hparams::{
    QWEN3_ARCHITECTURE_VALUE, QWEN3_AUDIO_D_MODEL_KEY, QWEN3_AUDIO_END_TOKEN_ID_KEY,
    QWEN3_AUDIO_HEADS_KEY, QWEN3_AUDIO_LAYERS_KEY, QWEN3_AUDIO_PAD_TOKEN_ID_KEY,
    QWEN3_AUDIO_START_TOKEN_ID_KEY, QWEN3_EOS_TOKEN_ID_KEY, QWEN3_HOP_LENGTH_KEY,
    QWEN3_LLM_D_MODEL_KEY, QWEN3_LLM_HEAD_DIM_KEY, QWEN3_LLM_HEADS_KEY, QWEN3_LLM_KV_HEADS_KEY,
    QWEN3_LLM_LAYERS_KEY, QWEN3_LLM_MAX_POSITIONS_KEY, QWEN3_LLM_VOCAB_SIZE_KEY,
    QWEN3_MELS_COUNT_KEY, QWEN3_N_FFT_KEY, QWEN3_PAD_TOKEN_ID_KEY, QWEN3_SAMPLE_RATE_KEY,
    QWEN3_WIN_LENGTH_KEY,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen3AsrExecutionMetadata {
    pub sample_rate_hz: u32,
    pub n_mels: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub audio_layers: usize,
    pub audio_d_model: usize,
    pub audio_heads: usize,
    pub llm_layers: usize,
    pub llm_d_model: usize,
    pub llm_heads: usize,
    pub llm_kv_heads: usize,
    pub llm_head_dim: usize,
    pub vocab_size: usize,
    pub llm_max_positions: usize,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub audio_pad_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}

impl BuiltinSeq2SeqDecodePolicyTokenSource for Qwen3AsrExecutionMetadata {
    fn audio_end_token_id(&self) -> Option<u32> {
        Some(self.audio_end_token_id)
    }

    fn audio_pad_token_id(&self) -> Option<u32> {
        Some(self.pad_token_id)
    }
}

impl PhraseBiasTokenEncoder for Qwen3AsrExecutionMetadata {
    fn encode_phrase_bias_tokens(&self, _phrase: &str) -> Result<Option<Vec<u32>>, String> {
        // The metadata fallback carries only special-token ids and no tokenizer
        // vocab, so it cannot encode phrase-bias phrases. Fail closed (Unsupported)
        // rather than silently dropping the requested bias.
        Ok(None)
    }
}

#[derive(Debug, Error)]
pub(crate) enum Qwen3AsrRuntimeContractError {
    #[error("qwen3-asr missing required GGUF metadata key '{key}'")]
    MissingRequiredMetadata { key: &'static str },
    #[error("qwen3-asr GGUF metadata '{key}' is invalid: {reason}")]
    InvalidMetadataValue { key: &'static str, reason: String },
    #[error("qwen3-asr expected general.architecture='{expected}', got '{found}'")]
    UnexpectedArchitecture {
        expected: &'static str,
        found: String,
    },
    #[error("qwen3-asr missing required GGUF tensor '{name}'")]
    MissingRequiredTensor { name: String },
    #[error("qwen3-asr GGUF tensor '{name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        name: String,
        shape: String,
        reason: String,
    },
}

pub(crate) fn parse_qwen3_execution_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<Qwen3AsrExecutionMetadata, Qwen3AsrRuntimeContractError> {
    let architecture = required_string_scalar(metadata, GENERAL_ARCHITECTURE_KEY)
        .map_err(map_metadata_contract_error)?;
    if architecture != QWEN3_ARCHITECTURE_VALUE {
        return Err(Qwen3AsrRuntimeContractError::UnexpectedArchitecture {
            expected: QWEN3_ARCHITECTURE_VALUE,
            found: architecture.to_string(),
        });
    }

    let sample_rate_hz = u64_to_u32(
        required_u64_scalar(metadata, QWEN3_SAMPLE_RATE_KEY)
            .map_err(map_metadata_contract_error)?,
        QWEN3_SAMPLE_RATE_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let n_mels = u64_to_usize(
        required_u64_scalar(metadata, QWEN3_MELS_COUNT_KEY).map_err(map_metadata_contract_error)?,
        QWEN3_MELS_COUNT_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let n_fft = u64_to_usize(
        required_u64_scalar(metadata, QWEN3_N_FFT_KEY).map_err(map_metadata_contract_error)?,
        QWEN3_N_FFT_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let win_length = u64_to_usize(
        required_u64_scalar(metadata, QWEN3_WIN_LENGTH_KEY).map_err(map_metadata_contract_error)?,
        QWEN3_WIN_LENGTH_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let hop_length = u64_to_usize(
        required_u64_scalar(metadata, QWEN3_HOP_LENGTH_KEY).map_err(map_metadata_contract_error)?,
        QWEN3_HOP_LENGTH_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let audio_layers = u64_to_usize(
        required_u64_scalar(metadata, QWEN3_AUDIO_LAYERS_KEY)
            .map_err(map_metadata_contract_error)?,
        QWEN3_AUDIO_LAYERS_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let audio_d_model = u64_to_usize(
        required_u64_scalar(metadata, QWEN3_AUDIO_D_MODEL_KEY)
            .map_err(map_metadata_contract_error)?,
        QWEN3_AUDIO_D_MODEL_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let audio_heads = u64_to_usize(
        required_u64_scalar(metadata, QWEN3_AUDIO_HEADS_KEY)
            .map_err(map_metadata_contract_error)?,
        QWEN3_AUDIO_HEADS_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let llm_layers = u64_to_usize(
        required_u64_scalar(metadata, QWEN3_LLM_LAYERS_KEY).map_err(map_metadata_contract_error)?,
        QWEN3_LLM_LAYERS_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let llm_d_model = u64_to_usize(
        required_u64_scalar(metadata, QWEN3_LLM_D_MODEL_KEY)
            .map_err(map_metadata_contract_error)?,
        QWEN3_LLM_D_MODEL_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let llm_heads = u64_to_usize(
        required_u64_scalar(metadata, QWEN3_LLM_HEADS_KEY).map_err(map_metadata_contract_error)?,
        QWEN3_LLM_HEADS_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let llm_kv_heads = u64_to_usize(
        required_u64_scalar(metadata, QWEN3_LLM_KV_HEADS_KEY)
            .map_err(map_metadata_contract_error)?,
        QWEN3_LLM_KV_HEADS_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let llm_head_dim = u64_to_usize(
        required_u64_scalar(metadata, QWEN3_LLM_HEAD_DIM_KEY)
            .map_err(map_metadata_contract_error)?,
        QWEN3_LLM_HEAD_DIM_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let vocab_size = u64_to_usize(
        required_u64_scalar(metadata, QWEN3_LLM_VOCAB_SIZE_KEY)
            .map_err(map_metadata_contract_error)?,
        QWEN3_LLM_VOCAB_SIZE_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let llm_max_positions = u64_to_usize(
        required_u64_scalar(metadata, QWEN3_LLM_MAX_POSITIONS_KEY)
            .map_err(map_metadata_contract_error)?,
        QWEN3_LLM_MAX_POSITIONS_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let audio_start_token_id = u64_to_u32(
        required_u64_scalar(metadata, QWEN3_AUDIO_START_TOKEN_ID_KEY)
            .map_err(map_metadata_contract_error)?,
        QWEN3_AUDIO_START_TOKEN_ID_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let audio_end_token_id = u64_to_u32(
        required_u64_scalar(metadata, QWEN3_AUDIO_END_TOKEN_ID_KEY)
            .map_err(map_metadata_contract_error)?,
        QWEN3_AUDIO_END_TOKEN_ID_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let audio_pad_token_id = u64_to_u32(
        required_u64_scalar(metadata, QWEN3_AUDIO_PAD_TOKEN_ID_KEY)
            .map_err(map_metadata_contract_error)?,
        QWEN3_AUDIO_PAD_TOKEN_ID_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let eos_token_id = u64_to_u32(
        required_u64_scalar(metadata, QWEN3_EOS_TOKEN_ID_KEY)
            .map_err(map_metadata_contract_error)?,
        QWEN3_EOS_TOKEN_ID_KEY,
    )
    .map_err(map_metadata_contract_error)?;
    let pad_token_id = u64_to_u32(
        required_u64_scalar(metadata, QWEN3_PAD_TOKEN_ID_KEY)
            .map_err(map_metadata_contract_error)?,
        QWEN3_PAD_TOKEN_ID_KEY,
    )
    .map_err(map_metadata_contract_error)?;

    validate_positive_usize(n_mels, QWEN3_MELS_COUNT_KEY).map_err(map_metadata_contract_error)?;
    validate_positive_usize(n_fft, QWEN3_N_FFT_KEY).map_err(map_metadata_contract_error)?;
    validate_positive_usize(win_length, QWEN3_WIN_LENGTH_KEY)
        .map_err(map_metadata_contract_error)?;
    validate_positive_usize(hop_length, QWEN3_HOP_LENGTH_KEY)
        .map_err(map_metadata_contract_error)?;
    validate_positive_usize(audio_layers, QWEN3_AUDIO_LAYERS_KEY)
        .map_err(map_metadata_contract_error)?;
    validate_positive_usize(audio_d_model, QWEN3_AUDIO_D_MODEL_KEY)
        .map_err(map_metadata_contract_error)?;
    validate_positive_usize(audio_heads, QWEN3_AUDIO_HEADS_KEY)
        .map_err(map_metadata_contract_error)?;
    validate_positive_usize(llm_layers, QWEN3_LLM_LAYERS_KEY)
        .map_err(map_metadata_contract_error)?;
    validate_positive_usize(llm_d_model, QWEN3_LLM_D_MODEL_KEY)
        .map_err(map_metadata_contract_error)?;
    validate_positive_usize(llm_heads, QWEN3_LLM_HEADS_KEY).map_err(map_metadata_contract_error)?;
    validate_positive_usize(llm_kv_heads, QWEN3_LLM_KV_HEADS_KEY)
        .map_err(map_metadata_contract_error)?;
    validate_positive_usize(llm_head_dim, QWEN3_LLM_HEAD_DIM_KEY)
        .map_err(map_metadata_contract_error)?;
    validate_positive_usize(vocab_size, QWEN3_LLM_VOCAB_SIZE_KEY)
        .map_err(map_metadata_contract_error)?;
    validate_positive_usize(llm_max_positions, QWEN3_LLM_MAX_POSITIONS_KEY)
        .map_err(map_metadata_contract_error)?;

    if !audio_d_model.is_multiple_of(audio_heads) {
        return Err(Qwen3AsrRuntimeContractError::InvalidMetadataValue {
            key: QWEN3_AUDIO_D_MODEL_KEY,
            reason: format!(
                "{QWEN3_AUDIO_D_MODEL_KEY}={audio_d_model} must be divisible by {QWEN3_AUDIO_HEADS_KEY}={audio_heads}"
            ),
        });
    }
    if llm_kv_heads > llm_heads {
        return Err(Qwen3AsrRuntimeContractError::InvalidMetadataValue {
            key: QWEN3_LLM_KV_HEADS_KEY,
            reason: format!(
                "{QWEN3_LLM_KV_HEADS_KEY}={llm_kv_heads} must be <= {QWEN3_LLM_HEADS_KEY}={llm_heads}"
            ),
        });
    }
    if !llm_heads.is_multiple_of(llm_kv_heads) {
        return Err(Qwen3AsrRuntimeContractError::InvalidMetadataValue {
            key: QWEN3_LLM_HEADS_KEY,
            reason: format!(
                "{QWEN3_LLM_HEADS_KEY}={llm_heads} must be divisible by {QWEN3_LLM_KV_HEADS_KEY}={llm_kv_heads}"
            ),
        });
    }

    Ok(Qwen3AsrExecutionMetadata {
        sample_rate_hz,
        n_mels,
        n_fft,
        win_length,
        hop_length,
        audio_layers,
        audio_d_model,
        audio_heads,
        llm_layers,
        llm_d_model,
        llm_heads,
        llm_kv_heads,
        llm_head_dim,
        vocab_size,
        llm_max_positions,
        audio_start_token_id,
        audio_end_token_id,
        audio_pad_token_id,
        eos_token_id,
        pad_token_id,
    })
}

pub(crate) fn validate_qwen3_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    metadata: Qwen3AsrExecutionMetadata,
) -> Result<(), Qwen3AsrRuntimeContractError> {
    let mel_filters = require_tensor(index, AUDIO_MEL_FILTERS)?;
    let expected_fft_bins = metadata
        .n_fft
        .checked_div(2)
        .and_then(|half| half.checked_add(1))
        .ok_or_else(|| Qwen3AsrRuntimeContractError::InvalidMetadataValue {
            key: QWEN3_N_FFT_KEY,
            reason: "n_fft overflow while computing FFT bin count".to_string(),
        })?;
    if mel_filters.dims != vec![metadata.n_mels as u64, expected_fft_bins as u64] {
        return Err(invalid_tensor_shape(
            AUDIO_MEL_FILTERS,
            &mel_filters.dims,
            format!(
                "expected [{} x {}] from metadata",
                metadata.n_mels, expected_fft_bins
            ),
        ));
    }

    let descriptors = resolve_builtin_runtime_tensor_contract_descriptors(
        qwen3_runtime_tensor_contract_id(),
        RuntimeTensorContractMetadata::Qwen3Asr(metadata),
    )
    .expect("qwen builtin runtime tensor contract must resolve");
    validate_tensor_binding_descriptors(
        index,
        &descriptors,
        missing_required_tensor,
        invalid_tensor_shape,
    )?;

    Ok(())
}

pub(crate) fn qwen3_runtime_tensor_descriptors(
    metadata: Qwen3AsrExecutionMetadata,
) -> Vec<TensorBindingDescriptor> {
    let mut descriptors = vec![
        TensorBindingDescriptor {
            tensor_name: AUDIO_MEL_WINDOW.to_string(),
            requirement: TensorBindingDescriptorRequirement::VectorLen(metadata.win_length),
            reason: "expected mel window vector".to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: AUDIO_CONV_OUT_WEIGHT.to_string(),
            requirement: TensorBindingDescriptorRequirement::RankAtLeastWithDimAt {
                min_rank: 2,
                axis: 1,
                dim: metadata.audio_d_model,
            },
            reason: "expected rank>=2 audio projection tensor with dims[1]=audio d_model"
                .to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: TOKEN_EMBD_WEIGHT.to_string(),
            requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                metadata.llm_d_model,
                metadata.vocab_size,
            ),
            reason: "expected token embedding matrix with llm hidden size and vocab dimensions"
                .to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: OUTPUT_WEIGHT.to_string(),
            requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                metadata.llm_d_model,
                metadata.vocab_size,
            ),
            reason: "expected output projection matrix with llm hidden size and vocab dimensions"
                .to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: OUTPUT_NORM_WEIGHT.to_string(),
            requirement: TensorBindingDescriptorRequirement::VectorLen(metadata.llm_d_model),
            reason: "expected output norm vector with llm hidden size length".to_string(),
        },
    ];
    for layer_idx in 0..metadata.audio_layers {
        let names = audio_layer_tensor_names(layer_idx);
        descriptors.extend([
            TensorBindingDescriptor {
                tensor_name: names.attn_norm_weight,
                requirement: TensorBindingDescriptorRequirement::VectorLen(metadata.audio_d_model),
                reason: "expected audio hidden-size vector".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_norm_bias,
                requirement: TensorBindingDescriptorRequirement::VectorLen(metadata.audio_d_model),
                reason: "expected audio hidden-size vector".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_q_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2WithDim(
                    metadata.audio_d_model,
                ),
                reason:
                    "expected rank-2 audio attention matrix with one dimension = audio hidden size"
                        .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_q_bias,
                requirement: TensorBindingDescriptorRequirement::VectorLen(metadata.audio_d_model),
                reason: "expected audio hidden-size bias".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_k_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2WithDim(
                    metadata.audio_d_model,
                ),
                reason:
                    "expected rank-2 audio attention matrix with one dimension = audio hidden size"
                        .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_k_bias,
                requirement: TensorBindingDescriptorRequirement::VectorLen(metadata.audio_d_model),
                reason: "expected audio hidden-size bias".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_v_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2WithDim(
                    metadata.audio_d_model,
                ),
                reason:
                    "expected rank-2 audio attention matrix with one dimension = audio hidden size"
                        .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_v_bias,
                requirement: TensorBindingDescriptorRequirement::VectorLen(metadata.audio_d_model),
                reason: "expected audio hidden-size bias".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_out_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2WithDim(
                    metadata.audio_d_model,
                ),
                reason:
                    "expected rank-2 audio attention matrix with one dimension = audio hidden size"
                        .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_out_bias,
                requirement: TensorBindingDescriptorRequirement::VectorLen(metadata.audio_d_model),
                reason: "expected audio hidden-size bias".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_norm_weight,
                requirement: TensorBindingDescriptorRequirement::VectorLen(metadata.audio_d_model),
                reason: "expected audio hidden-size vector".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_norm_bias,
                requirement: TensorBindingDescriptorRequirement::VectorLen(metadata.audio_d_model),
                reason: "expected audio hidden-size vector".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_up_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2WithDim(
                    metadata.audio_d_model,
                ),
                reason: "expected rank-2 audio FFN matrix with one dimension = audio hidden size"
                    .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_up_bias,
                requirement: TensorBindingDescriptorRequirement::NonEmptyVector,
                reason: "expected non-empty audio FFN bias vector".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_down_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2WithDim(
                    metadata.audio_d_model,
                ),
                reason: "expected rank-2 audio FFN matrix with one dimension = audio hidden size"
                    .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_down_bias,
                requirement: TensorBindingDescriptorRequirement::NonEmptyVector,
                reason: "expected non-empty audio FFN bias vector".to_string(),
            },
        ]);
    }
    for layer_idx in 0..metadata.llm_layers {
        let names = llm_layer_tensor_names(layer_idx);
        descriptors.extend([
            TensorBindingDescriptor {
                tensor_name: names.attn_norm_weight,
                requirement: TensorBindingDescriptorRequirement::VectorLen(metadata.llm_d_model),
                reason: "expected llm hidden-size vector".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_q_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2WithDim(metadata.llm_d_model),
                reason: "expected rank-2 attn_q matrix with one dimension = llm hidden size"
                    .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_k_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2WithDim(metadata.llm_d_model),
                reason: "expected rank-2 attn_k matrix with one dimension = llm hidden size"
                    .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_v_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2WithDim(metadata.llm_d_model),
                reason: "expected rank-2 attn_v matrix with one dimension = llm hidden size"
                    .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_output_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2WithDim(metadata.llm_d_model),
                reason: "expected rank-2 attn_output matrix with one dimension = llm hidden size"
                    .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_q_norm_weight,
                requirement: TensorBindingDescriptorRequirement::NonEmptyVector,
                reason: "expected non-empty rank-1 q_norm vector".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.attn_k_norm_weight,
                requirement: TensorBindingDescriptorRequirement::NonEmptyVector,
                reason: "expected non-empty rank-1 k_norm vector".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_norm_weight,
                requirement: TensorBindingDescriptorRequirement::VectorLen(metadata.llm_d_model),
                reason: "expected llm hidden-size vector".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_gate_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2WithDim(metadata.llm_d_model),
                reason: "expected rank-2 FFN gate matrix with one dimension = llm hidden size"
                    .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_up_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2WithDim(metadata.llm_d_model),
                reason: "expected rank-2 FFN up matrix with one dimension = llm hidden size"
                    .to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: names.ffn_down_weight,
                requirement: TensorBindingDescriptorRequirement::Rank2WithDim(metadata.llm_d_model),
                reason: "expected rank-2 FFN down matrix with one dimension = llm hidden size"
                    .to_string(),
            },
        ]);
    }
    descriptors
}

fn qwen3_runtime_tensor_contract_id() -> &'static str {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(QWEN3_ASR_GGML_ARCHITECTURE_ID)
        .expect("qwen architecture must be registered")
        .runtime_tensor_contract_id
}

fn require_tensor<'a>(
    index: &'a crate::GgufTensorIndex,
    name: &str,
) -> Result<&'a crate::GgufTensorMetadata, Qwen3AsrRuntimeContractError> {
    require_tensor_binding(index, name, missing_required_tensor)
}

fn missing_required_tensor(name: &str) -> Qwen3AsrRuntimeContractError {
    Qwen3AsrRuntimeContractError::MissingRequiredTensor {
        name: name.to_string(),
    }
}

fn invalid_tensor_shape(name: &str, shape: &[u64], reason: String) -> Qwen3AsrRuntimeContractError {
    Qwen3AsrRuntimeContractError::InvalidTensorShape {
        name: name.to_string(),
        shape: render_shape(shape),
        reason,
    }
}

fn map_metadata_contract_error(error: MetadataContractError) -> Qwen3AsrRuntimeContractError {
    match error {
        MetadataContractError::MissingRequiredKey { key } => {
            Qwen3AsrRuntimeContractError::MissingRequiredMetadata { key }
        }
        MetadataContractError::InvalidValue { key, reason } => {
            Qwen3AsrRuntimeContractError::InvalidMetadataValue { key, reason }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn metadata() -> BTreeMap<String, String> {
        [
            (GENERAL_ARCHITECTURE_KEY, QWEN3_ARCHITECTURE_VALUE),
            (QWEN3_SAMPLE_RATE_KEY, "16000"),
            (QWEN3_MELS_COUNT_KEY, "128"),
            (QWEN3_N_FFT_KEY, "400"),
            (QWEN3_WIN_LENGTH_KEY, "400"),
            (QWEN3_HOP_LENGTH_KEY, "160"),
            (QWEN3_AUDIO_LAYERS_KEY, "2"),
            (QWEN3_AUDIO_D_MODEL_KEY, "1280"),
            (QWEN3_AUDIO_HEADS_KEY, "20"),
            (QWEN3_LLM_LAYERS_KEY, "28"),
            (QWEN3_LLM_D_MODEL_KEY, "2048"),
            (QWEN3_LLM_HEADS_KEY, "16"),
            (QWEN3_LLM_KV_HEADS_KEY, "8"),
            (QWEN3_LLM_HEAD_DIM_KEY, "128"),
            (QWEN3_LLM_VOCAB_SIZE_KEY, "152064"),
            (QWEN3_LLM_MAX_POSITIONS_KEY, "4096"),
            (QWEN3_AUDIO_START_TOKEN_ID_KEY, "151647"),
            (QWEN3_AUDIO_END_TOKEN_ID_KEY, "151648"),
            (QWEN3_AUDIO_PAD_TOKEN_ID_KEY, "151649"),
            (QWEN3_EOS_TOKEN_ID_KEY, "151645"),
            (QWEN3_PAD_TOKEN_ID_KEY, "151643"),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
    }

    #[test]
    fn qwen3_metadata_accepts_grouped_query_attention_heads() {
        let metadata = metadata();
        let parsed = parse_qwen3_execution_metadata(&metadata).expect("metadata should parse");

        assert_eq!(parsed.llm_heads, 16);
        assert_eq!(parsed.llm_kv_heads, 8);
        assert_eq!(parsed.llm_head_dim, 128);
        assert_eq!(parsed.llm_d_model, 2048);
    }

    #[test]
    fn qwen3_metadata_accepts_expanded_query_projection_heads() {
        let mut metadata = metadata();
        metadata.insert(QWEN3_LLM_D_MODEL_KEY.to_string(), "1024".to_string());
        metadata.insert(QWEN3_LLM_HEADS_KEY.to_string(), "16".to_string());
        metadata.insert(QWEN3_LLM_KV_HEADS_KEY.to_string(), "8".to_string());
        metadata.insert(QWEN3_LLM_HEAD_DIM_KEY.to_string(), "128".to_string());

        let parsed = parse_qwen3_execution_metadata(&metadata).expect("metadata should parse");

        assert_eq!(parsed.llm_heads, 16);
        assert_eq!(parsed.llm_kv_heads, 8);
        assert_eq!(parsed.llm_head_dim, 128);
        assert_eq!(parsed.llm_d_model, 1024);
    }

    #[test]
    fn qwen3_metadata_rejects_heads_that_are_not_divisible_by_kv_heads() {
        let mut metadata = metadata();
        metadata.insert(QWEN3_LLM_KV_HEADS_KEY.to_string(), "6".to_string());

        let error = parse_qwen3_execution_metadata(&metadata).expect_err("invalid kv heads");
        match error {
            Qwen3AsrRuntimeContractError::InvalidMetadataValue { key, reason } => {
                assert_eq!(key, QWEN3_LLM_HEADS_KEY);
                assert!(reason.contains(QWEN3_LLM_HEADS_KEY), "{reason}");
                assert!(reason.contains(QWEN3_LLM_KV_HEADS_KEY), "{reason}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
