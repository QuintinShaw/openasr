use thiserror::Error;

use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_string_scalar, required_u64_scalar,
    u64_to_u32, u64_to_usize, validate_positive_usize,
};
use crate::{GgufTensorIndex, GgufTensorMetadata};

pub(crate) const GENERAL_ARCHITECTURE_KEY: &str = "general.architecture";
pub(crate) const MOONSHINE_ARCHITECTURE_VALUE: &str = "moonshine-encoder-decoder";

pub(crate) const MOONSHINE_VOCAB_SIZE_KEY: &str = "moonshine.vocab_size";
pub(crate) const MOONSHINE_D_MODEL_KEY: &str = "moonshine.d_model";
pub(crate) const MOONSHINE_ENCODER_LAYERS_KEY: &str = "moonshine.encoder.n_layers";
pub(crate) const MOONSHINE_DECODER_LAYERS_KEY: &str = "moonshine.decoder.n_layers";
pub(crate) const MOONSHINE_HEADS_KEY: &str = "moonshine.n_heads";
pub(crate) const MOONSHINE_HEAD_DIM_KEY: &str = "moonshine.head_dim";
pub(crate) const MOONSHINE_ROTARY_DIM_KEY: &str = "moonshine.rotary_dim";
pub(crate) const MOONSHINE_ROPE_THETA_KEY: &str = "moonshine.rope_theta";
pub(crate) const MOONSHINE_ENCODER_FFN_DIM_KEY: &str = "moonshine.encoder.ffn_dim";
pub(crate) const MOONSHINE_DECODER_FFN_DIM_KEY: &str = "moonshine.decoder.ffn_dim";
pub(crate) const MOONSHINE_MAX_CONTEXT_KEY: &str = "moonshine.decoder.max_ctx";
pub(crate) const MOONSHINE_BOS_TOKEN_ID_KEY: &str = "moonshine.decoder.bos_token_id";
pub(crate) const MOONSHINE_EOS_TOKEN_ID_KEY: &str = "moonshine.decoder.eos_token_id";
pub(crate) const MOONSHINE_SAMPLE_RATE_KEY: &str = "moonshine.audio.sample_rate";

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MoonshineExecutionMetadata {
    pub vocab_size: usize,
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub decoder_max_context: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub sample_rate_hz: u32,
    pub rope_theta: f32,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub(crate) enum MoonshineRuntimeContractError {
    #[error("moonshine missing required GGUF metadata key '{key}'")]
    MissingRequiredMetadata { key: &'static str },
    #[error("moonshine GGUF metadata '{key}' is invalid: {reason}")]
    InvalidMetadataValue { key: &'static str, reason: String },
    #[error("moonshine expected general.architecture='{expected}', got '{found}'")]
    UnexpectedArchitecture {
        expected: &'static str,
        found: String,
    },
    #[error("moonshine missing required GGUF tensor '{name}'")]
    MissingRequiredTensor { name: String },
    #[error("moonshine GGUF tensor '{name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        name: String,
        shape: String,
        reason: String,
    },
}

pub(crate) fn parse_moonshine_execution_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<MoonshineExecutionMetadata, MoonshineRuntimeContractError> {
    let architecture = required_string_scalar(metadata, GENERAL_ARCHITECTURE_KEY)
        .map_err(map_metadata_contract_error)?;
    if architecture != MOONSHINE_ARCHITECTURE_VALUE {
        return Err(MoonshineRuntimeContractError::UnexpectedArchitecture {
            expected: MOONSHINE_ARCHITECTURE_VALUE,
            found: architecture.to_string(),
        });
    }

    let vocab_size = required_usize(metadata, MOONSHINE_VOCAB_SIZE_KEY)?;
    let d_model = required_usize(metadata, MOONSHINE_D_MODEL_KEY)?;
    let encoder_layers = required_usize(metadata, MOONSHINE_ENCODER_LAYERS_KEY)?;
    let decoder_layers = required_usize(metadata, MOONSHINE_DECODER_LAYERS_KEY)?;
    let n_heads = required_usize(metadata, MOONSHINE_HEADS_KEY)?;
    let head_dim = required_usize(metadata, MOONSHINE_HEAD_DIM_KEY)?;
    let rotary_dim = required_usize(metadata, MOONSHINE_ROTARY_DIM_KEY)?;
    let encoder_ffn_dim = required_usize(metadata, MOONSHINE_ENCODER_FFN_DIM_KEY)?;
    let decoder_ffn_dim = required_usize(metadata, MOONSHINE_DECODER_FFN_DIM_KEY)?;
    let decoder_max_context = required_usize(metadata, MOONSHINE_MAX_CONTEXT_KEY)?;
    let bos_token_id = required_u32(metadata, MOONSHINE_BOS_TOKEN_ID_KEY)?;
    let eos_token_id = required_u32(metadata, MOONSHINE_EOS_TOKEN_ID_KEY)?;
    let sample_rate_hz = required_u32(metadata, MOONSHINE_SAMPLE_RATE_KEY)?;
    let rope_theta = required_f32(metadata, MOONSHINE_ROPE_THETA_KEY)?;

    for (value, key) in [
        (vocab_size, MOONSHINE_VOCAB_SIZE_KEY),
        (d_model, MOONSHINE_D_MODEL_KEY),
        (encoder_layers, MOONSHINE_ENCODER_LAYERS_KEY),
        (decoder_layers, MOONSHINE_DECODER_LAYERS_KEY),
        (n_heads, MOONSHINE_HEADS_KEY),
        (head_dim, MOONSHINE_HEAD_DIM_KEY),
        (rotary_dim, MOONSHINE_ROTARY_DIM_KEY),
        (encoder_ffn_dim, MOONSHINE_ENCODER_FFN_DIM_KEY),
        (decoder_ffn_dim, MOONSHINE_DECODER_FFN_DIM_KEY),
        (decoder_max_context, MOONSHINE_MAX_CONTEXT_KEY),
    ] {
        validate_positive_usize(value, key).map_err(map_metadata_contract_error)?;
    }

    if n_heads.saturating_mul(head_dim) != d_model {
        return Err(MoonshineRuntimeContractError::InvalidMetadataValue {
            key: MOONSHINE_D_MODEL_KEY,
            reason: format!(
                "{MOONSHINE_D_MODEL_KEY}={d_model} must equal {MOONSHINE_HEADS_KEY}={n_heads} * {MOONSHINE_HEAD_DIM_KEY}={head_dim}"
            ),
        });
    }
    if rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(MoonshineRuntimeContractError::InvalidMetadataValue {
            key: MOONSHINE_ROTARY_DIM_KEY,
            reason: format!("rotary_dim={rotary_dim} must be even and <= head_dim={head_dim}"),
        });
    }
    if !(rope_theta.is_finite() && rope_theta > 0.0) {
        return Err(MoonshineRuntimeContractError::InvalidMetadataValue {
            key: MOONSHINE_ROPE_THETA_KEY,
            reason: format!("rope_theta={rope_theta} must be finite and positive"),
        });
    }

    Ok(MoonshineExecutionMetadata {
        vocab_size,
        d_model,
        encoder_layers,
        decoder_layers,
        n_heads,
        head_dim,
        rotary_dim,
        encoder_ffn_dim,
        decoder_ffn_dim,
        decoder_max_context,
        bos_token_id,
        eos_token_id,
        sample_rate_hz,
        rope_theta,
    })
}

pub(crate) fn validate_moonshine_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    metadata: MoonshineExecutionMetadata,
) -> Result<(), MoonshineRuntimeContractError> {
    // Conv stem.
    let conv1 = require_tensor(index, "enc.conv1.weight")?;
    if conv1.dims.len() != 3 {
        return Err(invalid_tensor_shape(
            "enc.conv1.weight",
            &conv1.dims,
            "expected rank-3 conv1 weight".to_string(),
        ));
    }
    for name in ["enc.conv2.weight", "enc.conv3.weight"] {
        let tensor = require_tensor(index, name)?;
        if tensor.dims.len() != 3 {
            return Err(invalid_tensor_shape(
                name,
                &tensor.dims,
                "expected rank-3 conv weight".to_string(),
            ));
        }
    }
    // conv2 produces 2*d_model channels (576 for tiny); its bias matches conv2 out channels.
    let conv2_bias = require_tensor(index, "enc.conv2.bias")?;
    if conv2_bias.dims.len() != 1 || conv2_bias.dims[0] == 0 {
        return Err(invalid_tensor_shape(
            "enc.conv2.bias",
            &conv2_bias.dims,
            "expected non-empty conv2 bias vector".to_string(),
        ));
    }
    for name in [
        "enc.conv3.bias",
        "enc.groupnorm.weight",
        "enc.groupnorm.bias",
        "enc.out_norm.weight",
        "dec.out_norm.weight",
    ] {
        let tensor = require_tensor(index, name)?;
        validate_vector_len(
            name,
            &tensor.dims,
            metadata.d_model,
            "expected d_model-sized vector",
        )?;
    }

    let emb = require_tensor(index, "dec.emb.weight")?;
    validate_rank2_permuted(
        "dec.emb.weight",
        &emb.dims,
        metadata.vocab_size,
        metadata.d_model,
        "expected vocab/d_model embedding matrix",
    )?;

    validate_encoder_layer_tensors(index, metadata)?;
    validate_decoder_layer_tensors(index, metadata)?;
    Ok(())
}

fn validate_encoder_layer_tensors(
    index: &GgufTensorIndex,
    metadata: MoonshineExecutionMetadata,
) -> Result<(), MoonshineRuntimeContractError> {
    for layer_idx in 0..metadata.encoder_layers {
        let prefix = format!("enc.blk.{layer_idx}.");
        for slot in ["attn_norm.weight", "ffn_norm.weight", "ffn_down.bias"] {
            let name = format!("{prefix}{slot}");
            let tensor = require_tensor(index, &name)?;
            validate_vector_len(
                &name,
                &tensor.dims,
                metadata.d_model,
                "expected d_model vector",
            )?;
        }
        for slot in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_o.weight",
        ] {
            let name = format!("{prefix}{slot}");
            let tensor = require_tensor(index, &name)?;
            validate_rank2_permuted(
                &name,
                &tensor.dims,
                metadata.d_model,
                metadata.d_model,
                "expected d_model x d_model attention matrix",
            )?;
        }
        let up = require_tensor(index, &format!("{prefix}ffn_up.weight"))?;
        validate_rank2_permuted(
            &format!("{prefix}ffn_up.weight"),
            &up.dims,
            metadata.encoder_ffn_dim,
            metadata.d_model,
            "expected encoder FFN up matrix",
        )?;
        let up_b = require_tensor(index, &format!("{prefix}ffn_up.bias"))?;
        validate_vector_len(
            &format!("{prefix}ffn_up.bias"),
            &up_b.dims,
            metadata.encoder_ffn_dim,
            "expected encoder FFN up bias",
        )?;
        let down = require_tensor(index, &format!("{prefix}ffn_down.weight"))?;
        validate_rank2_permuted(
            &format!("{prefix}ffn_down.weight"),
            &down.dims,
            metadata.d_model,
            metadata.encoder_ffn_dim,
            "expected encoder FFN down matrix",
        )?;
    }
    Ok(())
}

fn validate_decoder_layer_tensors(
    index: &GgufTensorIndex,
    metadata: MoonshineExecutionMetadata,
) -> Result<(), MoonshineRuntimeContractError> {
    for layer_idx in 0..metadata.decoder_layers {
        let prefix = format!("dec.blk.{layer_idx}.");
        for slot in [
            "attn_norm.weight",
            "cross_norm.weight",
            "ffn_norm.weight",
            "ffn_down.bias",
        ] {
            let name = format!("{prefix}{slot}");
            let tensor = require_tensor(index, &name)?;
            validate_vector_len(
                &name,
                &tensor.dims,
                metadata.d_model,
                "expected d_model vector",
            )?;
        }
        for slot in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_o.weight",
            "cross_q.weight",
            "cross_k.weight",
            "cross_v.weight",
            "cross_o.weight",
        ] {
            let name = format!("{prefix}{slot}");
            let tensor = require_tensor(index, &name)?;
            validate_rank2_permuted(
                &name,
                &tensor.dims,
                metadata.d_model,
                metadata.d_model,
                "expected d_model x d_model attention matrix",
            )?;
        }
        // Gated FFN fc1 produces 2 * ffn_dim (hidden, gate).
        let up = require_tensor(index, &format!("{prefix}ffn_up.weight"))?;
        validate_rank2_permuted(
            &format!("{prefix}ffn_up.weight"),
            &up.dims,
            metadata.decoder_ffn_dim.saturating_mul(2),
            metadata.d_model,
            "expected decoder gated FFN fc1 matrix (2*ffn_dim x d_model)",
        )?;
        let up_b = require_tensor(index, &format!("{prefix}ffn_up.bias"))?;
        validate_vector_len(
            &format!("{prefix}ffn_up.bias"),
            &up_b.dims,
            metadata.decoder_ffn_dim.saturating_mul(2),
            "expected decoder gated FFN fc1 bias",
        )?;
        let down = require_tensor(index, &format!("{prefix}ffn_down.weight"))?;
        validate_rank2_permuted(
            &format!("{prefix}ffn_down.weight"),
            &down.dims,
            metadata.d_model,
            metadata.decoder_ffn_dim,
            "expected decoder FFN down matrix",
        )?;
    }
    Ok(())
}

fn required_usize<M: ScalarMetadataView>(
    metadata: &M,
    key: &'static str,
) -> Result<usize, MoonshineRuntimeContractError> {
    let value = required_u64_scalar(metadata, key).map_err(map_metadata_contract_error)?;
    u64_to_usize(value, key).map_err(map_metadata_contract_error)
}

fn required_u32<M: ScalarMetadataView>(
    metadata: &M,
    key: &'static str,
) -> Result<u32, MoonshineRuntimeContractError> {
    let value = required_u64_scalar(metadata, key).map_err(map_metadata_contract_error)?;
    u64_to_u32(value, key).map_err(map_metadata_contract_error)
}

fn required_f32<M: ScalarMetadataView>(
    metadata: &M,
    key: &'static str,
) -> Result<f32, MoonshineRuntimeContractError> {
    // rope_theta is stored as a stringified float in GGUF string metadata.
    let value = required_string_scalar(metadata, key).map_err(map_metadata_contract_error)?;
    value.trim().parse::<f32>().map_err(|error| {
        MoonshineRuntimeContractError::InvalidMetadataValue {
            key,
            reason: format!("could not parse f32 from '{value}': {error}"),
        }
    })
}

fn map_metadata_contract_error(error: MetadataContractError) -> MoonshineRuntimeContractError {
    match error {
        MetadataContractError::MissingRequiredKey { key } => {
            MoonshineRuntimeContractError::MissingRequiredMetadata { key }
        }
        MetadataContractError::InvalidValue { key, reason } => {
            MoonshineRuntimeContractError::InvalidMetadataValue { key, reason }
        }
    }
}

fn validate_vector_len(
    tensor_name: &str,
    dims: &[u64],
    expected_len: usize,
    reason: &str,
) -> Result<(), MoonshineRuntimeContractError> {
    if dims == [expected_len as u64] {
        return Ok(());
    }
    Err(invalid_tensor_shape(
        tensor_name,
        dims,
        format!("{reason}; expected [{expected_len}]"),
    ))
}

fn validate_rank2_permuted(
    tensor_name: &str,
    dims: &[u64],
    lhs: usize,
    rhs: usize,
    reason: &str,
) -> Result<(), MoonshineRuntimeContractError> {
    if dims.len() == 2
        && ((dims[0] as usize == lhs && dims[1] as usize == rhs)
            || (dims[0] as usize == rhs && dims[1] as usize == lhs))
    {
        return Ok(());
    }
    Err(invalid_tensor_shape(tensor_name, dims, reason.to_string()))
}

fn require_tensor<'a>(
    index: &'a GgufTensorIndex,
    name: &str,
) -> Result<&'a GgufTensorMetadata, MoonshineRuntimeContractError> {
    index
        .get(name)
        .ok_or_else(|| MoonshineRuntimeContractError::MissingRequiredTensor {
            name: name.to_string(),
        })
}

fn invalid_tensor_shape(
    name: &str,
    shape: &[u64],
    reason: String,
) -> MoonshineRuntimeContractError {
    MoonshineRuntimeContractError::InvalidTensorShape {
        name: name.to_string(),
        shape: render_shape(shape),
        reason,
    }
}

fn render_shape(shape: &[u64]) -> String {
    let parts = shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{parts}]")
}
