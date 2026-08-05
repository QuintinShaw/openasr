//! funasr-nano execution metadata + runtime tensor contract parsed from the
//! `.oasr` GGUF header. Key names match exactly what the pack importer writes
//! (`funasr.enc.*` / `funasr.adp.*` / `funasr.llm.*`). The validator is
//! depth-complete: a pack must satisfy all three metadata contracts AND the
//! full runtime tensor binding (SAN-M encoder, transformer adaptor, Qwen3
//! decoder) before it can be admitted.

use crate::GgufTensorIndex;
use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_u64_scalar, u64_to_u32, u64_to_usize,
    validate_bounded_usize, validate_positive_usize,
};
use crate::models::tensor_binding::{
    TensorBindingDescriptor, TensorBindingDescriptorRequirement, TensorReadGuard, render_shape,
    validate_tensor_binding_descriptors,
};

pub(crate) const ENC_N_LAYERS_KEY: &str = "funasr.enc.n_layers";
pub(crate) const ENC_TP_BLOCKS_KEY: &str = "funasr.enc.tp_blocks";
pub(crate) const ENC_D_MODEL_KEY: &str = "funasr.enc.d_model";
pub(crate) const ENC_N_HEADS_KEY: &str = "funasr.enc.n_heads";
pub(crate) const ENC_HEAD_DIM_KEY: &str = "funasr.enc.head_dim";
pub(crate) const ENC_FFN_DIM_KEY: &str = "funasr.enc.ffn_dim";
pub(crate) const ENC_FSMN_KERNEL_KEY: &str = "funasr.enc.fsmn_kernel";
pub(crate) const ENC_FEATURE_DIM_KEY: &str = "funasr.enc.feature_dim";

pub(crate) const ADP_N_LAYERS_KEY: &str = "funasr.adp.n_layers";
pub(crate) const ADP_N_HEADS_KEY: &str = "funasr.adp.n_heads";
pub(crate) const ADP_ENCODER_DIM_KEY: &str = "funasr.adp.encoder_dim";
pub(crate) const ADP_LLM_DIM_KEY: &str = "funasr.adp.llm_dim";

pub(crate) const LLM_N_LAYERS_KEY: &str = "funasr.llm.n_layers";
pub(crate) const LLM_D_MODEL_KEY: &str = "funasr.llm.d_model";
pub(crate) const LLM_N_HEADS_KEY: &str = "funasr.llm.n_heads";
pub(crate) const LLM_N_KV_HEADS_KEY: &str = "funasr.llm.n_kv_heads";
pub(crate) const LLM_HEAD_DIM_KEY: &str = "funasr.llm.head_dim";
pub(crate) const LLM_FFN_DIM_KEY: &str = "funasr.llm.ffn_dim";
pub(crate) const LLM_VOCAB_SIZE_KEY: &str = "funasr.llm.vocab_size";
pub(crate) const LLM_MAX_POSITIONS_KEY: &str = "funasr.llm.max_positions";
pub(crate) const LLM_CHATML_IM_START_TOKEN_ID_KEY: &str = "funasr.llm.chatml_im_start_token_id";
pub(crate) const LLM_CHATML_IM_END_TOKEN_ID_KEY: &str = "funasr.llm.chatml_im_end_token_id";
pub(crate) const LLM_ENDOFTEXT_TOKEN_ID_KEY: &str = "funasr.llm.endoftext_token_id";

/// Architecture ceilings for pack-supplied geometry, with generous headroom
/// over the production checkpoint (50 enc + 20 tp SAN-M blocks, d_model 512;
/// 2 adaptor blocks; 28 Qwen3 decoder layers, d_model 1024, vocab 151936).
/// They bound every contract-derived arithmetic expression and the
/// tensor-obligation count a malicious metadata set can construct, so
/// contract building stays allocation-bounded and overflow-free on untrusted
/// input; parse fails closed above them.
pub(crate) const FUNASR_NANO_MAX_LAYERS: usize = 512;
pub(crate) const FUNASR_NANO_MAX_D_MODEL: usize = 65_536;
pub(crate) const FUNASR_NANO_MAX_N_HEADS: usize = 1_024;
pub(crate) const FUNASR_NANO_MAX_HEAD_DIM: usize = 65_536;
pub(crate) const FUNASR_NANO_MAX_FFN_DIM: usize = 262_144;
pub(crate) const FUNASR_NANO_MAX_FSMN_KERNEL: usize = 4_096;
pub(crate) const FUNASR_NANO_MAX_FEATURE_DIM: usize = 4_096;
pub(crate) const FUNASR_NANO_MAX_VOCAB_SIZE: usize = 1_000_000;
pub(crate) const FUNASR_NANO_MAX_POSITIONS: usize = 1_048_576;
/// Global ceiling on the tensor obligations one pack's contract may
/// construct; far above the production 1233, far below anything that could
/// exhaust the verifier.
pub(crate) const FUNASR_NANO_MAX_TENSOR_OBLIGATIONS: usize = 1_000_000;

/// `rope_theta` (1e6) and RMSNorm epsilon (1e-6) are fixed properties of the
/// checkpoint's stock Qwen3-0.6B decoder (`Qwen3-0.6B/config.json`'s
/// `rope_theta` / `rms_norm_eps`), not per-pack metadata -- the same "family
/// constant, not a GGUF key" convention `moss_transcribe_diarize` /
/// `firered_llm` already use for their Qwen decoders.
pub(crate) const FUNASR_NANO_ROPE_THETA: f32 = 1_000_000.0;
pub(crate) const FUNASR_NANO_RMS_NORM_EPSILON: f32 = 1e-6;
/// The FunASR SAN-M encoder and the transformer adaptor use `nn.LayerNorm`'s
/// eps = 1e-5 (verified against the official funasr-nano llama.cpp runtime's
/// implementation notes and the model.pt-derived reference oracle), NOT the
/// 1e-12 the `sensevoice` (SenseVoiceSmall) encoder pins -- Fun-ASR-Nano's
/// encoder is retrained with the llama.cpp-standard 1e-5.
pub(crate) const FUNASR_NANO_ENCODER_LAYER_NORM_EPSILON: f32 = 1e-5;
pub(crate) const FUNASR_NANO_ADAPTOR_LAYER_NORM_EPSILON: f32 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FunasrNanoEncoderMetadata {
    pub n_layers: usize,
    pub tp_blocks: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub fsmn_kernel: usize,
    pub feature_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FunasrNanoAdapterMetadata {
    pub n_layers: usize,
    pub n_heads: usize,
    pub encoder_dim: usize,
    pub llm_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FunasrNanoDecoderMetadata {
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub max_positions: usize,
    pub chatml_im_start_token_id: u32,
    pub chatml_im_end_token_id: u32,
    pub endoftext_token_id: u32,
}

pub(crate) fn parse_funasr_nano_encoder_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<FunasrNanoEncoderMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let n_layers = usize_key(ENC_N_LAYERS_KEY)?;
    let tp_blocks = usize_key(ENC_TP_BLOCKS_KEY)?;
    let d_model = usize_key(ENC_D_MODEL_KEY)?;
    let n_heads = usize_key(ENC_N_HEADS_KEY)?;
    let head_dim = usize_key(ENC_HEAD_DIM_KEY)?;
    let ffn_dim = usize_key(ENC_FFN_DIM_KEY)?;
    let fsmn_kernel = usize_key(ENC_FSMN_KERNEL_KEY)?;
    let feature_dim = usize_key(ENC_FEATURE_DIM_KEY)?;
    for (key, value) in [
        (ENC_N_LAYERS_KEY, n_layers),
        (ENC_TP_BLOCKS_KEY, tp_blocks),
        (ENC_D_MODEL_KEY, d_model),
        (ENC_N_HEADS_KEY, n_heads),
        (ENC_HEAD_DIM_KEY, head_dim),
        (ENC_FFN_DIM_KEY, ffn_dim),
        (ENC_FSMN_KERNEL_KEY, fsmn_kernel),
        (ENC_FEATURE_DIM_KEY, feature_dim),
    ] {
        validate_positive_usize(value, key)?;
    }
    for (key, value, max) in [
        (ENC_N_LAYERS_KEY, n_layers, FUNASR_NANO_MAX_LAYERS),
        (ENC_TP_BLOCKS_KEY, tp_blocks, FUNASR_NANO_MAX_LAYERS),
        (ENC_D_MODEL_KEY, d_model, FUNASR_NANO_MAX_D_MODEL),
        (ENC_N_HEADS_KEY, n_heads, FUNASR_NANO_MAX_N_HEADS),
        (ENC_HEAD_DIM_KEY, head_dim, FUNASR_NANO_MAX_HEAD_DIM),
        (ENC_FFN_DIM_KEY, ffn_dim, FUNASR_NANO_MAX_FFN_DIM),
        (
            ENC_FSMN_KERNEL_KEY,
            fsmn_kernel,
            FUNASR_NANO_MAX_FSMN_KERNEL,
        ),
        (
            ENC_FEATURE_DIM_KEY,
            feature_dim,
            FUNASR_NANO_MAX_FEATURE_DIM,
        ),
    ] {
        validate_bounded_usize(value, key, max)?;
    }
    if n_heads.checked_mul(head_dim) != Some(d_model) {
        return Err(MetadataContractError::InvalidValue {
            key: ENC_HEAD_DIM_KEY,
            reason: format!("n_heads {n_heads} * head_dim {head_dim} != d_model {d_model}"),
        });
    }
    if fsmn_kernel.is_multiple_of(2) {
        return Err(MetadataContractError::InvalidValue {
            key: ENC_FSMN_KERNEL_KEY,
            reason: format!("fsmn kernel {fsmn_kernel} must be odd (symmetric padding)"),
        });
    }
    Ok(FunasrNanoEncoderMetadata {
        n_layers,
        tp_blocks,
        d_model,
        n_heads,
        head_dim,
        ffn_dim,
        fsmn_kernel,
        feature_dim,
    })
}

pub(crate) fn parse_funasr_nano_adapter_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<FunasrNanoAdapterMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let n_layers = usize_key(ADP_N_LAYERS_KEY)?;
    let n_heads = usize_key(ADP_N_HEADS_KEY)?;
    let encoder_dim = usize_key(ADP_ENCODER_DIM_KEY)?;
    let llm_dim = usize_key(ADP_LLM_DIM_KEY)?;
    for (key, value) in [
        (ADP_N_LAYERS_KEY, n_layers),
        (ADP_N_HEADS_KEY, n_heads),
        (ADP_ENCODER_DIM_KEY, encoder_dim),
        (ADP_LLM_DIM_KEY, llm_dim),
    ] {
        validate_positive_usize(value, key)?;
    }
    for (key, value, max) in [
        (ADP_N_LAYERS_KEY, n_layers, FUNASR_NANO_MAX_LAYERS),
        (ADP_N_HEADS_KEY, n_heads, FUNASR_NANO_MAX_N_HEADS),
        (ADP_ENCODER_DIM_KEY, encoder_dim, FUNASR_NANO_MAX_D_MODEL),
        (ADP_LLM_DIM_KEY, llm_dim, FUNASR_NANO_MAX_D_MODEL),
    ] {
        validate_bounded_usize(value, key, max)?;
    }
    if !llm_dim.is_multiple_of(n_heads) {
        return Err(MetadataContractError::InvalidValue {
            key: ADP_N_HEADS_KEY,
            reason: format!("llm_dim {llm_dim} is not a multiple of n_heads {n_heads}"),
        });
    }
    Ok(FunasrNanoAdapterMetadata {
        n_layers,
        n_heads,
        encoder_dim,
        llm_dim,
    })
}

pub(crate) fn parse_funasr_nano_decoder_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<FunasrNanoDecoderMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let u32_key = |key: &'static str| -> Result<u32, MetadataContractError> {
        u64_to_u32(required_u64_scalar(metadata, key)?, key)
    };
    let n_layers = usize_key(LLM_N_LAYERS_KEY)?;
    let d_model = usize_key(LLM_D_MODEL_KEY)?;
    let n_heads = usize_key(LLM_N_HEADS_KEY)?;
    let n_kv_heads = usize_key(LLM_N_KV_HEADS_KEY)?;
    let head_dim = usize_key(LLM_HEAD_DIM_KEY)?;
    let ffn_dim = usize_key(LLM_FFN_DIM_KEY)?;
    let vocab_size = usize_key(LLM_VOCAB_SIZE_KEY)?;
    let max_positions = usize_key(LLM_MAX_POSITIONS_KEY)?;
    let chatml_im_start_token_id = u32_key(LLM_CHATML_IM_START_TOKEN_ID_KEY)?;
    let chatml_im_end_token_id = u32_key(LLM_CHATML_IM_END_TOKEN_ID_KEY)?;
    let endoftext_token_id = u32_key(LLM_ENDOFTEXT_TOKEN_ID_KEY)?;

    for (key, value) in [
        (LLM_N_LAYERS_KEY, n_layers),
        (LLM_D_MODEL_KEY, d_model),
        (LLM_N_HEADS_KEY, n_heads),
        (LLM_N_KV_HEADS_KEY, n_kv_heads),
        (LLM_HEAD_DIM_KEY, head_dim),
        (LLM_FFN_DIM_KEY, ffn_dim),
        (LLM_VOCAB_SIZE_KEY, vocab_size),
        (LLM_MAX_POSITIONS_KEY, max_positions),
    ] {
        validate_positive_usize(value, key)?;
    }
    for (key, value, max) in [
        (LLM_N_LAYERS_KEY, n_layers, FUNASR_NANO_MAX_LAYERS),
        (LLM_D_MODEL_KEY, d_model, FUNASR_NANO_MAX_D_MODEL),
        (LLM_N_HEADS_KEY, n_heads, FUNASR_NANO_MAX_N_HEADS),
        (LLM_N_KV_HEADS_KEY, n_kv_heads, FUNASR_NANO_MAX_N_HEADS),
        (LLM_HEAD_DIM_KEY, head_dim, FUNASR_NANO_MAX_HEAD_DIM),
        (LLM_FFN_DIM_KEY, ffn_dim, FUNASR_NANO_MAX_FFN_DIM),
        (LLM_VOCAB_SIZE_KEY, vocab_size, FUNASR_NANO_MAX_VOCAB_SIZE),
        (
            LLM_MAX_POSITIONS_KEY,
            max_positions,
            FUNASR_NANO_MAX_POSITIONS,
        ),
    ] {
        validate_bounded_usize(value, key, max)?;
    }
    // Qwen3 decouples the per-head projection width from `d_model / n_heads`
    // (head_dim 128 * n_heads 16 = 2048 != d_model 1024), so there is no
    // `n_heads * head_dim == d_model` invariant to enforce here (matches
    // `qwen`/`moss_transcribe_diarize`, which never assert one either).
    if n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
        return Err(MetadataContractError::InvalidValue {
            key: LLM_N_KV_HEADS_KEY,
            reason: format!("n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}"),
        });
    }
    for (key, id) in [
        (LLM_CHATML_IM_START_TOKEN_ID_KEY, chatml_im_start_token_id),
        (LLM_CHATML_IM_END_TOKEN_ID_KEY, chatml_im_end_token_id),
        (LLM_ENDOFTEXT_TOKEN_ID_KEY, endoftext_token_id),
    ] {
        if (id as usize) >= vocab_size {
            return Err(MetadataContractError::InvalidValue {
                key,
                reason: format!("token id {id} out of range for vocab_size {vocab_size}"),
            });
        }
    }
    Ok(FunasrNanoDecoderMetadata {
        n_layers,
        d_model,
        n_heads,
        n_kv_heads,
        head_dim,
        ffn_dim,
        vocab_size,
        max_positions,
        chatml_im_start_token_id,
        chatml_im_end_token_id,
        endoftext_token_id,
    })
}

pub(crate) fn validate_runtime_pack_contract(
    preflight: &crate::GgufRuntimeSourcePreflight,
) -> Result<(), String> {
    let encoder = parse_funasr_nano_encoder_metadata(preflight.metadata()).map_err(|error| {
        crate::models::runtime_pack_contract::metadata_validation_error("funasr-nano", error)
    })?;
    let adapter = parse_funasr_nano_adapter_metadata(preflight.metadata()).map_err(|error| {
        crate::models::runtime_pack_contract::metadata_validation_error("funasr-nano", error)
    })?;
    let decoder = parse_funasr_nano_decoder_metadata(preflight.metadata()).map_err(|error| {
        crate::models::runtime_pack_contract::metadata_validation_error("funasr-nano", error)
    })?;
    validate_funasr_nano_runtime_tensors_with_index(
        preflight.tensor_index(),
        &encoder,
        &adapter,
        &decoder,
    )
    .map_err(crate::models::runtime_pack_contract::tensor_validation_error)
}

/// Fail-closed tensor-contract errors, surfaced by the pack verifier before a
/// funasr-nano pack can be admitted.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub(crate) enum FunasrNanoTensorContractError {
    #[error("funasr-nano runtime tensor contract is missing required tensor '{name}'")]
    MissingRequiredTensor { name: String },
    #[error("funasr-nano runtime tensor '{name}' has shape {shape}: {reason}")]
    InvalidTensorShape {
        name: String,
        shape: String,
        reason: String,
    },
    #[error(
        "funasr-nano geometry constructs {count} tensor obligations, exceeding the ceiling {max}"
    )]
    TooManyTensorObligations { count: usize, max: usize },
}

fn missing_required_tensor(name: &str) -> FunasrNanoTensorContractError {
    FunasrNanoTensorContractError::MissingRequiredTensor {
        name: name.to_string(),
    }
}

fn invalid_tensor_shape(
    name: &str,
    shape: &[u64],
    reason: String,
) -> FunasrNanoTensorContractError {
    FunasrNanoTensorContractError::InvalidTensorShape {
        name: name.to_string(),
        shape: render_shape(shape),
        reason,
    }
}

fn descriptor(
    tensor_name: String,
    requirement: TensorBindingDescriptorRequirement,
    reason: &str,
) -> TensorBindingDescriptor {
    TensorBindingDescriptor {
        tensor_name,
        requirement,
        reason: reason.to_string(),
    }
}

/// One SAN-M block's runtime tensor bindings: the 13 tensors
/// `encoder_graph::load_layer` reads and `nn::encoder::sanm_fsmn_encoder_layer`
/// consumes (the identical layout the sensevoice family contracts), shaped for
/// the block's `input_dim`. Suffixes come from the shared `tensor_names`
/// constants the loader resolves, so contract and loader enumerate one name
/// set. Derived extents use saturating arithmetic: parsing caps every input,
/// so saturation is unreachable defense in depth that stays fail-closed at
/// validation (no pack tensor can match a saturated requirement).
fn sanm_block_tensor_descriptors(
    encoder: &FunasrNanoEncoderMetadata,
    scope: &str,
    layer: usize,
    input_dim: usize,
) -> Vec<TensorBindingDescriptor> {
    use super::tensor_names::{
        SANM_ATTN_FSMN_WEIGHT, SANM_ATTN_NORM_BIAS, SANM_ATTN_NORM_WEIGHT, SANM_ATTN_OUT_BIAS,
        SANM_ATTN_OUT_WEIGHT, SANM_ATTN_QKV_BIAS, SANM_ATTN_QKV_WEIGHT, SANM_FFN_DOWN_BIAS,
        SANM_FFN_DOWN_WEIGHT, SANM_FFN_NORM_BIAS, SANM_FFN_NORM_WEIGHT, SANM_FFN_UP_BIAS,
        SANM_FFN_UP_WEIGHT,
    };
    let d_model = encoder.d_model;
    let qkv_dim = d_model.saturating_mul(3);
    let name = |suffix: &str| format!("{scope}.{layer}.{suffix}");
    let entries: [(&str, TensorBindingDescriptorRequirement, &str); 13] = [
        (
            SANM_ATTN_NORM_WEIGHT,
            TensorBindingDescriptorRequirement::VectorLen(input_dim),
            "pre-attention LayerNorm gamma must span the block input width",
        ),
        (
            SANM_ATTN_NORM_BIAS,
            TensorBindingDescriptorRequirement::VectorLen(input_dim),
            "pre-attention LayerNorm beta must span the block input width",
        ),
        (
            SANM_ATTN_QKV_WEIGHT,
            TensorBindingDescriptorRequirement::Rank2EitherDims(input_dim, qkv_dim),
            "fused QKV projection must map the block input width to 3*d_model",
        ),
        (
            SANM_ATTN_QKV_BIAS,
            TensorBindingDescriptorRequirement::VectorLen(qkv_dim),
            "fused QKV bias must span 3*d_model",
        ),
        (
            SANM_ATTN_OUT_WEIGHT,
            TensorBindingDescriptorRequirement::Rank2EitherDims(d_model, d_model),
            "attention output projection must be d_model x d_model",
        ),
        (
            SANM_ATTN_OUT_BIAS,
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "attention output bias must span d_model",
        ),
        (
            SANM_ATTN_FSMN_WEIGHT,
            TensorBindingDescriptorRequirement::ExactDims(vec![encoder.fsmn_kernel, 1, d_model]),
            "FSMN depthwise kernel must be [fsmn_kernel, 1, d_model] for the im2col conv path",
        ),
        (
            SANM_FFN_NORM_WEIGHT,
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "pre-FFN LayerNorm gamma must span d_model",
        ),
        (
            SANM_FFN_NORM_BIAS,
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "pre-FFN LayerNorm beta must span d_model",
        ),
        (
            SANM_FFN_UP_WEIGHT,
            TensorBindingDescriptorRequirement::Rank2EitherDims(d_model, encoder.ffn_dim),
            "FFN up projection must map d_model to ffn_dim",
        ),
        (
            SANM_FFN_UP_BIAS,
            TensorBindingDescriptorRequirement::VectorLen(encoder.ffn_dim),
            "FFN up bias must span ffn_dim",
        ),
        (
            SANM_FFN_DOWN_WEIGHT,
            TensorBindingDescriptorRequirement::Rank2EitherDims(encoder.ffn_dim, d_model),
            "FFN down projection must map ffn_dim to d_model",
        ),
        (
            SANM_FFN_DOWN_BIAS,
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "FFN down bias must span d_model",
        ),
    ];
    entries
        .into_iter()
        .map(|(suffix, requirement, reason)| descriptor(name(suffix), requirement, reason))
        .collect()
}

/// One adaptor transformer block's runtime tensor bindings: the 16 tensors
/// `adapter_graph::load_block` binds (`attn.{norm,q,k,v,out}` +
/// `ffn.{norm,up,down}`, each weight+bias), all llm_dim wide. Suffixes come
/// from the shared `tensor_names` constants the binder resolves, so contract
/// and loader enumerate one name set.
fn adaptor_block_tensor_descriptors(
    adapter: &FunasrNanoAdapterMetadata,
    layer: usize,
) -> Vec<TensorBindingDescriptor> {
    use super::tensor_names::{
        ADAPTOR_ATTN_K_BIAS, ADAPTOR_ATTN_K_WEIGHT, ADAPTOR_ATTN_NORM_BIAS,
        ADAPTOR_ATTN_NORM_WEIGHT, ADAPTOR_ATTN_OUT_BIAS, ADAPTOR_ATTN_OUT_WEIGHT,
        ADAPTOR_ATTN_Q_BIAS, ADAPTOR_ATTN_Q_WEIGHT, ADAPTOR_ATTN_V_BIAS, ADAPTOR_ATTN_V_WEIGHT,
        ADAPTOR_FFN_DOWN_BIAS, ADAPTOR_FFN_DOWN_WEIGHT, ADAPTOR_FFN_NORM_BIAS,
        ADAPTOR_FFN_NORM_WEIGHT, ADAPTOR_FFN_UP_BIAS, ADAPTOR_FFN_UP_WEIGHT,
    };
    let llm_dim = adapter.llm_dim;
    let name = |suffix: &str| format!("adaptor.blk.{layer}.{suffix}");
    let entries: [(&str, TensorBindingDescriptorRequirement, &str); 16] = [
        (
            ADAPTOR_ATTN_NORM_WEIGHT,
            TensorBindingDescriptorRequirement::VectorLen(llm_dim),
            "adaptor attention pre-norm gamma must span llm_dim",
        ),
        (
            ADAPTOR_ATTN_NORM_BIAS,
            TensorBindingDescriptorRequirement::VectorLen(llm_dim),
            "adaptor attention pre-norm beta must span llm_dim",
        ),
        (
            ADAPTOR_ATTN_Q_WEIGHT,
            TensorBindingDescriptorRequirement::Rank2EitherDims(llm_dim, llm_dim),
            "adaptor attention projection must be llm_dim x llm_dim",
        ),
        (
            ADAPTOR_ATTN_Q_BIAS,
            TensorBindingDescriptorRequirement::VectorLen(llm_dim),
            "adaptor attention projection bias must span llm_dim",
        ),
        (
            ADAPTOR_ATTN_K_WEIGHT,
            TensorBindingDescriptorRequirement::Rank2EitherDims(llm_dim, llm_dim),
            "adaptor attention projection must be llm_dim x llm_dim",
        ),
        (
            ADAPTOR_ATTN_K_BIAS,
            TensorBindingDescriptorRequirement::VectorLen(llm_dim),
            "adaptor attention projection bias must span llm_dim",
        ),
        (
            ADAPTOR_ATTN_V_WEIGHT,
            TensorBindingDescriptorRequirement::Rank2EitherDims(llm_dim, llm_dim),
            "adaptor attention projection must be llm_dim x llm_dim",
        ),
        (
            ADAPTOR_ATTN_V_BIAS,
            TensorBindingDescriptorRequirement::VectorLen(llm_dim),
            "adaptor attention projection bias must span llm_dim",
        ),
        (
            ADAPTOR_ATTN_OUT_WEIGHT,
            TensorBindingDescriptorRequirement::Rank2EitherDims(llm_dim, llm_dim),
            "adaptor attention output projection must be llm_dim x llm_dim",
        ),
        (
            ADAPTOR_ATTN_OUT_BIAS,
            TensorBindingDescriptorRequirement::VectorLen(llm_dim),
            "adaptor attention output bias must span llm_dim",
        ),
        (
            ADAPTOR_FFN_NORM_WEIGHT,
            TensorBindingDescriptorRequirement::VectorLen(llm_dim),
            "adaptor FFN pre-norm gamma must span llm_dim",
        ),
        (
            ADAPTOR_FFN_NORM_BIAS,
            TensorBindingDescriptorRequirement::VectorLen(llm_dim),
            "adaptor FFN pre-norm beta must span llm_dim",
        ),
        (
            ADAPTOR_FFN_UP_WEIGHT,
            TensorBindingDescriptorRequirement::Rank2WithDim(llm_dim),
            "adaptor FFN up projection must consume llm_dim-wide rows",
        ),
        (
            ADAPTOR_FFN_UP_BIAS,
            TensorBindingDescriptorRequirement::NonEmptyVector,
            "adaptor FFN up bias must be a non-empty vector",
        ),
        (
            ADAPTOR_FFN_DOWN_WEIGHT,
            TensorBindingDescriptorRequirement::Rank2WithDim(llm_dim),
            "adaptor FFN down projection must produce llm_dim-wide rows",
        ),
        (
            ADAPTOR_FFN_DOWN_BIAS,
            TensorBindingDescriptorRequirement::VectorLen(llm_dim),
            "adaptor FFN down bias must span llm_dim",
        ),
    ];
    entries
        .into_iter()
        .map(|(suffix, requirement, reason)| descriptor(name(suffix), requirement, reason))
        .collect()
}

/// One Qwen3 decoder layer's runtime tensor bindings: the 11 weight-only
/// tensors the loader resolves through `funasr_nano_llm_layer_tensor_names`
/// (RMSNorm carries no bias, Qwen3 attention is bias-free). The names come
/// from that exact loader name source, so contract and loader enumerate one
/// name set. Derived extents use saturating arithmetic: parsing caps every
/// input, so saturation is unreachable defense in depth that stays
/// fail-closed at validation.
fn llm_layer_tensor_descriptors(
    decoder: &FunasrNanoDecoderMetadata,
    layer: usize,
) -> Vec<TensorBindingDescriptor> {
    use super::tensor_names::funasr_nano_llm_layer_tensor_names;
    let names = funasr_nano_llm_layer_tensor_names(layer);
    let d_model = decoder.d_model;
    let q_dim = decoder.n_heads.saturating_mul(decoder.head_dim);
    let kv_dim = decoder.n_kv_heads.saturating_mul(decoder.head_dim);
    let entries: [(String, TensorBindingDescriptorRequirement, &str); 11] = [
        (
            names.attn_norm_weight,
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "attention RMSNorm must span d_model",
        ),
        (
            names.attn_q_weight,
            TensorBindingDescriptorRequirement::Rank2EitherDims(d_model, q_dim),
            "query projection must map d_model to n_heads*head_dim",
        ),
        (
            names.attn_k_weight,
            TensorBindingDescriptorRequirement::Rank2EitherDims(d_model, kv_dim),
            "key projection must map d_model to n_kv_heads*head_dim",
        ),
        (
            names.attn_v_weight,
            TensorBindingDescriptorRequirement::Rank2EitherDims(d_model, kv_dim),
            "value projection must map d_model to n_kv_heads*head_dim",
        ),
        (
            names.attn_output_weight,
            TensorBindingDescriptorRequirement::Rank2EitherDims(q_dim, d_model),
            "attention output projection must map n_heads*head_dim to d_model",
        ),
        (
            names.attn_q_norm_weight,
            TensorBindingDescriptorRequirement::VectorLen(decoder.head_dim),
            "QK-norm query RMSNorm must span head_dim",
        ),
        (
            names.attn_k_norm_weight,
            TensorBindingDescriptorRequirement::VectorLen(decoder.head_dim),
            "QK-norm key RMSNorm must span head_dim",
        ),
        (
            names.ffn_norm_weight,
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "FFN RMSNorm must span d_model",
        ),
        (
            names.ffn_gate_weight,
            TensorBindingDescriptorRequirement::Rank2EitherDims(d_model, decoder.ffn_dim),
            "FFN gate projection must map d_model to ffn_dim",
        ),
        (
            names.ffn_up_weight,
            TensorBindingDescriptorRequirement::Rank2EitherDims(d_model, decoder.ffn_dim),
            "FFN up projection must map d_model to ffn_dim",
        ),
        (
            names.ffn_down_weight,
            TensorBindingDescriptorRequirement::Rank2EitherDims(decoder.ffn_dim, d_model),
            "FFN down projection must map ffn_dim to d_model",
        ),
    ];
    entries
        .into_iter()
        .map(
            |(tensor_name, requirement, reason)| TensorBindingDescriptor {
                tensor_name,
                requirement,
                reason: reason.to_string(),
            },
        )
        .collect()
}

/// The SAN-M encoder half of the contract: every `enc.blk` / `tp.blk` block
/// plus the four tail LayerNorms. The encoder weight loader reads exactly
/// this set; the read guard below enforces it at load time.
pub(crate) fn funasr_nano_encoder_tensor_descriptors(
    encoder: &FunasrNanoEncoderMetadata,
) -> Vec<TensorBindingDescriptor> {
    use super::tensor_names::{
        ENC_AFTER_NORM_BIAS, ENC_AFTER_NORM_WEIGHT, TP_NORM_BIAS, TP_NORM_WEIGHT,
    };
    let mut descriptors = Vec::new();
    for layer in 0..encoder.n_layers {
        let input_dim = if layer == 0 {
            encoder.feature_dim
        } else {
            encoder.d_model
        };
        descriptors.extend(sanm_block_tensor_descriptors(
            encoder, "enc.blk", layer, input_dim,
        ));
    }
    for layer in 0..encoder.tp_blocks {
        descriptors.extend(sanm_block_tensor_descriptors(
            encoder,
            "tp.blk",
            layer,
            encoder.d_model,
        ));
    }
    let d_model = encoder.d_model;
    descriptors.extend([
        descriptor(
            ENC_AFTER_NORM_WEIGHT.to_string(),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "encoder tail LayerNorm gamma must span d_model",
        ),
        descriptor(
            ENC_AFTER_NORM_BIAS.to_string(),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "encoder tail LayerNorm beta must span d_model",
        ),
        descriptor(
            TP_NORM_WEIGHT.to_string(),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "two-pass tail LayerNorm gamma must span d_model",
        ),
        descriptor(
            TP_NORM_BIAS.to_string(),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "two-pass tail LayerNorm beta must span d_model",
        ),
    ]);
    descriptors
}

/// The adaptor half of the contract: the two flattening/projecting linears
/// plus every transformer block. The adapter binder resolves exactly this
/// set; the read guard below enforces it at bind time.
pub(crate) fn funasr_nano_adapter_tensor_descriptors(
    adapter: &FunasrNanoAdapterMetadata,
) -> Vec<TensorBindingDescriptor> {
    use super::tensor_names::{
        ADAPTOR_LINEAR1_BIAS, ADAPTOR_LINEAR1_WEIGHT, ADAPTOR_LINEAR2_BIAS, ADAPTOR_LINEAR2_WEIGHT,
    };
    let mut descriptors = vec![
        descriptor(
            ADAPTOR_LINEAR1_WEIGHT.to_string(),
            TensorBindingDescriptorRequirement::Rank2WithDim(adapter.encoder_dim),
            "adaptor linear1 must consume encoder_dim-wide rows",
        ),
        descriptor(
            ADAPTOR_LINEAR1_BIAS.to_string(),
            TensorBindingDescriptorRequirement::NonEmptyVector,
            "adaptor linear1 bias must be a non-empty vector",
        ),
        descriptor(
            ADAPTOR_LINEAR2_WEIGHT.to_string(),
            TensorBindingDescriptorRequirement::Rank2WithDim(adapter.llm_dim),
            "adaptor linear2 must produce llm_dim-wide rows",
        ),
        descriptor(
            ADAPTOR_LINEAR2_BIAS.to_string(),
            TensorBindingDescriptorRequirement::VectorLen(adapter.llm_dim),
            "adaptor linear2 bias must span llm_dim",
        ),
    ];
    for layer in 0..adapter.n_layers {
        descriptors.extend(adaptor_block_tensor_descriptors(adapter, layer));
    }
    descriptors
}

/// The Qwen3 decoder half of the contract: every decoder layer (named by the
/// loader's own name source) plus the final norm, logits head, and token
/// embedding (named by the shared `tensor_names` constants the loader reads).
pub(crate) fn funasr_nano_decoder_tensor_descriptors(
    decoder: &FunasrNanoDecoderMetadata,
) -> Vec<TensorBindingDescriptor> {
    use super::tensor_names::{LLM_OUTPUT_NORM_WEIGHT, LLM_OUTPUT_WEIGHT, LLM_TOKEN_EMBD_WEIGHT};
    let mut descriptors = Vec::new();
    for layer in 0..decoder.n_layers {
        descriptors.extend(llm_layer_tensor_descriptors(decoder, layer));
    }
    descriptors.extend([
        descriptor(
            LLM_OUTPUT_NORM_WEIGHT.to_string(),
            TensorBindingDescriptorRequirement::VectorLen(decoder.d_model),
            "final RMSNorm before the logits head must span d_model",
        ),
        descriptor(
            LLM_OUTPUT_WEIGHT.to_string(),
            TensorBindingDescriptorRequirement::Rank2EitherDims(
                decoder.d_model,
                decoder.vocab_size,
            ),
            "logits head must project d_model to the vocab",
        ),
        descriptor(
            LLM_TOKEN_EMBD_WEIGHT.to_string(),
            TensorBindingDescriptorRequirement::Rank2EitherDims(
                decoder.d_model,
                decoder.vocab_size,
            ),
            "token embedding table must be d_model x vocab",
        ),
    ]);
    descriptors
}

/// The runtime tensor contract for one funasr-nano pack: every tensor the
/// SAN-M encoder, transformer adaptor, and Qwen3 decoder materialize, with
/// the shapes the graphs consume. Derived from the parsed metadata, so a
/// checkpoint with different layer counts validates its own geometry.
pub(crate) fn funasr_nano_runtime_tensor_binding_descriptors(
    encoder: &FunasrNanoEncoderMetadata,
    adapter: &FunasrNanoAdapterMetadata,
    decoder: &FunasrNanoDecoderMetadata,
) -> Vec<TensorBindingDescriptor> {
    let mut descriptors = funasr_nano_encoder_tensor_descriptors(encoder);
    descriptors.extend(funasr_nano_adapter_tensor_descriptors(adapter));
    descriptors.extend(funasr_nano_decoder_tensor_descriptors(decoder));
    descriptors
}

/// Read guard for the encoder half: the encoder weight loader fails closed on
/// any tensor this guard (built from the contract enumeration) does not list.
pub(crate) fn funasr_nano_encoder_read_guard(
    encoder: &FunasrNanoEncoderMetadata,
) -> TensorReadGuard {
    TensorReadGuard::from_descriptors(&funasr_nano_encoder_tensor_descriptors(encoder))
}

/// Read guard for the adaptor half: the adapter binder fails closed on any
/// tensor this guard (built from the contract enumeration) does not list.
pub(crate) fn funasr_nano_adapter_read_guard(
    adapter: &FunasrNanoAdapterMetadata,
) -> TensorReadGuard {
    TensorReadGuard::from_descriptors(&funasr_nano_adapter_tensor_descriptors(adapter))
}

/// Read guard for the decoder half: built from the contract enumeration so
/// local name-set checks (and tests) can fail closed on any tensor the
/// decoder logic does not list. Production decoder load does not install this
/// onto the shared tensor index -- FunASR packs are encoder+adapter+decoder
/// combos, and the whole-decoder weight context must enumerate every pack
/// tensor. Admission-time
/// [`validate_funasr_nano_runtime_tensors_with_index`] plus known-name shape
/// checks in the shared Qwen planner remain the fail-closed path.
#[cfg(test)]
pub(crate) fn funasr_nano_decoder_read_guard(
    decoder: &FunasrNanoDecoderMetadata,
) -> TensorReadGuard {
    TensorReadGuard::from_descriptors(&funasr_nano_decoder_tensor_descriptors(decoder))
}

/// Validate the full runtime tensor set against the pack's tensor index,
/// fail-closed on a geometry whose obligation count exceeds the ceiling.
pub(crate) fn validate_funasr_nano_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    encoder: &FunasrNanoEncoderMetadata,
    adapter: &FunasrNanoAdapterMetadata,
    decoder: &FunasrNanoDecoderMetadata,
) -> Result<(), FunasrNanoTensorContractError> {
    let descriptors = funasr_nano_runtime_tensor_binding_descriptors(encoder, adapter, decoder);
    if descriptors.len() > FUNASR_NANO_MAX_TENSOR_OBLIGATIONS {
        return Err(FunasrNanoTensorContractError::TooManyTensorObligations {
            count: descriptors.len(),
            max: FUNASR_NANO_MAX_TENSOR_OBLIGATIONS,
        });
    }
    validate_tensor_binding_descriptors(
        index,
        &descriptors,
        missing_required_tensor,
        invalid_tensor_shape,
    )
}

/// Projects the single tensor contract into a runtime-ready fixture tensor set
/// (pack names plus valid dims); the runtime-ready test fixture stamps exactly
/// this set, so fixture and validator agree through one enumeration.
#[cfg(any(test, feature = "testing"))]
pub(crate) fn funasr_nano_runtime_tensors(
    encoder: &FunasrNanoEncoderMetadata,
    adapter: &FunasrNanoAdapterMetadata,
    decoder: &FunasrNanoDecoderMetadata,
) -> Vec<(String, Vec<u64>)> {
    crate::models::tensor_binding::project_fixture_tensors(
        &funasr_nano_runtime_tensor_binding_descriptors(encoder, adapter, decoder),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn full_metadata() -> BTreeMap<String, String> {
        [
            (ENC_N_LAYERS_KEY, "50"),
            (ENC_TP_BLOCKS_KEY, "20"),
            (ENC_D_MODEL_KEY, "512"),
            (ENC_N_HEADS_KEY, "4"),
            (ENC_HEAD_DIM_KEY, "128"),
            (ENC_FFN_DIM_KEY, "2048"),
            (ENC_FSMN_KERNEL_KEY, "11"),
            (ENC_FEATURE_DIM_KEY, "560"),
            (ADP_N_LAYERS_KEY, "2"),
            (ADP_N_HEADS_KEY, "8"),
            (ADP_ENCODER_DIM_KEY, "512"),
            (ADP_LLM_DIM_KEY, "1024"),
            (LLM_N_LAYERS_KEY, "28"),
            (LLM_D_MODEL_KEY, "1024"),
            (LLM_N_HEADS_KEY, "16"),
            (LLM_N_KV_HEADS_KEY, "8"),
            (LLM_HEAD_DIM_KEY, "128"),
            (LLM_FFN_DIM_KEY, "3072"),
            (LLM_VOCAB_SIZE_KEY, "151936"),
            (LLM_MAX_POSITIONS_KEY, "40960"),
            (LLM_CHATML_IM_START_TOKEN_ID_KEY, "151644"),
            (LLM_CHATML_IM_END_TOKEN_ID_KEY, "151645"),
            (LLM_ENDOFTEXT_TOKEN_ID_KEY, "151643"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    #[test]
    fn parses_encoder_metadata() {
        let parsed = parse_funasr_nano_encoder_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.n_layers, 50);
        assert_eq!(parsed.tp_blocks, 20);
        assert_eq!(parsed.d_model, 512);
        assert_eq!(parsed.feature_dim, 560);
    }

    #[test]
    fn parses_adapter_metadata() {
        let parsed = parse_funasr_nano_adapter_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.n_layers, 2);
        assert_eq!(parsed.n_heads, 8);
        assert_eq!(parsed.llm_dim, 1024);
    }

    #[test]
    fn parses_decoder_metadata() {
        let parsed = parse_funasr_nano_decoder_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.n_layers, 28);
        assert_eq!(parsed.n_kv_heads, 8);
        assert_eq!(parsed.chatml_im_end_token_id, 151_645);
    }

    #[test]
    fn rejects_kv_heads_not_dividing_heads() {
        let mut metadata = full_metadata();
        metadata.insert(LLM_N_KV_HEADS_KEY.to_string(), "3".to_string());
        assert!(parse_funasr_nano_decoder_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_even_fsmn_kernel() {
        let mut metadata = full_metadata();
        metadata.insert(ENC_FSMN_KERNEL_KEY.to_string(), "10".to_string());
        assert!(parse_funasr_nano_encoder_metadata(&metadata).is_err());
    }

    // --- Runtime tensor contract ---

    fn tiny_encoder() -> FunasrNanoEncoderMetadata {
        FunasrNanoEncoderMetadata {
            n_layers: 1,
            tp_blocks: 1,
            d_model: 16,
            n_heads: 2,
            head_dim: 8,
            ffn_dim: 32,
            fsmn_kernel: 5,
            feature_dim: 28,
        }
    }

    fn tiny_adapter() -> FunasrNanoAdapterMetadata {
        FunasrNanoAdapterMetadata {
            n_layers: 1,
            n_heads: 2,
            encoder_dim: 16,
            llm_dim: 24,
        }
    }

    fn tiny_decoder() -> FunasrNanoDecoderMetadata {
        FunasrNanoDecoderMetadata {
            n_layers: 1,
            d_model: 24,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 8,
            ffn_dim: 48,
            vocab_size: 32,
            max_positions: 64,
            chatml_im_start_token_id: 0,
            chatml_im_end_token_id: 1,
            endoftext_token_id: 2,
        }
    }

    fn tensor_index_from_shapes(shapes: &[(String, Vec<u64>)]) -> crate::GgufTensorIndex {
        let tensors = shapes
            .iter()
            .enumerate()
            .map(|(index, (name, dims))| crate::GgufTensorMetadata {
                name: name.clone(),
                dims: dims.clone(),
                ggml_type: 0,
                type_name: "f32".to_string(),
                size_bytes: 0,
                offset_bytes: index as u64,
            })
            .collect();
        crate::GgufTensorIndex::from_snapshot(crate::ggml_runtime::GgufTensorIndexSnapshot {
            path: std::path::PathBuf::from("funasr-nano-contract-test.oasr"),
            data_section_offset_bytes: 0,
            tensors,
        })
        .expect("unique tensor names")
    }

    /// Structural pins on the production geometry (50 enc + 20 tp SAN-M
    /// blocks of 13 tensors, 4 tail norms, the 2-layer adaptor (4 linears +
    /// 2x16 block tensors), and the 28-layer Qwen3 decoder (11 weights per
    /// layer + norm/logits/embedding)). Loader-equivalence evidence lives in
    /// the traced full-load tests: encoder half in `encoder_graph::trace_tests`,
    /// decoder half in `llm_transformer::trace_tests` (logical plan/logits/
    /// embedding access trace plus combo-pack `new_from_preflight`). Adaptor half still pins name-source
    /// equality below; this test holds the enumeration's production shape stable.
    #[test]
    fn descriptor_set_stays_pinned_on_production_geometry() {
        let encoder = parse_funasr_nano_encoder_metadata(&full_metadata()).expect("enc");
        let adapter = parse_funasr_nano_adapter_metadata(&full_metadata()).expect("adp");
        let decoder = parse_funasr_nano_decoder_metadata(&full_metadata()).expect("llm");
        let descriptors =
            funasr_nano_runtime_tensor_binding_descriptors(&encoder, &adapter, &decoder);
        assert_eq!(descriptors.len(), (50 + 20) * 13 + 4 + 36 + 28 * 11 + 3);
        let names: std::collections::BTreeSet<&str> = descriptors
            .iter()
            .map(|descriptor| descriptor.tensor_name.as_str())
            .collect();
        assert_eq!(names.len(), descriptors.len(), "names must be unique");
    }

    /// The adaptor + decoder descriptor names must equal the loader's own
    /// name sources (the shared suffix constants and the
    /// `funasr_nano_llm_layer_tensor_names` builder), so the two read sets
    /// are one name set by construction and any one-sided edit fails here.
    #[test]
    fn adapter_and_decoder_descriptor_names_match_the_loader_name_sources() {
        use super::super::tensor_names::{
            ADAPTOR_ATTN_K_BIAS, ADAPTOR_ATTN_K_WEIGHT, ADAPTOR_ATTN_NORM_BIAS,
            ADAPTOR_ATTN_NORM_WEIGHT, ADAPTOR_ATTN_OUT_BIAS, ADAPTOR_ATTN_OUT_WEIGHT,
            ADAPTOR_ATTN_Q_BIAS, ADAPTOR_ATTN_Q_WEIGHT, ADAPTOR_ATTN_V_BIAS, ADAPTOR_ATTN_V_WEIGHT,
            ADAPTOR_FFN_DOWN_BIAS, ADAPTOR_FFN_DOWN_WEIGHT, ADAPTOR_FFN_NORM_BIAS,
            ADAPTOR_FFN_NORM_WEIGHT, ADAPTOR_FFN_UP_BIAS, ADAPTOR_FFN_UP_WEIGHT,
            ADAPTOR_LINEAR1_BIAS, ADAPTOR_LINEAR1_WEIGHT, ADAPTOR_LINEAR2_BIAS,
            ADAPTOR_LINEAR2_WEIGHT, LLM_OUTPUT_NORM_WEIGHT, LLM_OUTPUT_WEIGHT,
            LLM_TOKEN_EMBD_WEIGHT, funasr_nano_llm_layer_tensor_names,
        };

        let encoder = parse_funasr_nano_encoder_metadata(&full_metadata()).expect("enc");
        let adapter = parse_funasr_nano_adapter_metadata(&full_metadata()).expect("adp");
        let decoder = parse_funasr_nano_decoder_metadata(&full_metadata()).expect("llm");

        let descriptor_names = |descriptors: Vec<TensorBindingDescriptor>| {
            descriptors
                .into_iter()
                .map(|descriptor| descriptor.tensor_name)
                .collect::<std::collections::BTreeSet<String>>()
        };

        // Adaptor half: the names the binder resolves from the shared
        // constants.
        let mut loader_adaptor_names = std::collections::BTreeSet::new();
        for suffix in [
            ADAPTOR_LINEAR1_WEIGHT,
            ADAPTOR_LINEAR1_BIAS,
            ADAPTOR_LINEAR2_WEIGHT,
            ADAPTOR_LINEAR2_BIAS,
        ] {
            loader_adaptor_names.insert(suffix.to_string());
        }
        for layer in 0..adapter.n_layers {
            for suffix in [
                ADAPTOR_ATTN_NORM_WEIGHT,
                ADAPTOR_ATTN_NORM_BIAS,
                ADAPTOR_ATTN_Q_WEIGHT,
                ADAPTOR_ATTN_Q_BIAS,
                ADAPTOR_ATTN_K_WEIGHT,
                ADAPTOR_ATTN_K_BIAS,
                ADAPTOR_ATTN_V_WEIGHT,
                ADAPTOR_ATTN_V_BIAS,
                ADAPTOR_ATTN_OUT_WEIGHT,
                ADAPTOR_ATTN_OUT_BIAS,
                ADAPTOR_FFN_NORM_WEIGHT,
                ADAPTOR_FFN_NORM_BIAS,
                ADAPTOR_FFN_UP_WEIGHT,
                ADAPTOR_FFN_UP_BIAS,
                ADAPTOR_FFN_DOWN_WEIGHT,
                ADAPTOR_FFN_DOWN_BIAS,
            ] {
                loader_adaptor_names.insert(format!("adaptor.blk.{layer}.{suffix}"));
            }
        }
        assert_eq!(
            descriptor_names(funasr_nano_adapter_tensor_descriptors(&adapter)),
            loader_adaptor_names,
            "adaptor contract names must equal the binder's name sources"
        );

        // Decoder half: the names the loader hands to the shared qwen
        // machinery, plus the tail constants it reads directly.
        let mut loader_decoder_names = std::collections::BTreeSet::new();
        for layer in 0..decoder.n_layers {
            let names = funasr_nano_llm_layer_tensor_names(layer);
            loader_decoder_names.extend([
                names.attn_norm_weight,
                names.attn_q_weight,
                names.attn_k_weight,
                names.attn_v_weight,
                names.attn_output_weight,
                names.attn_q_norm_weight,
                names.attn_k_norm_weight,
                names.ffn_norm_weight,
                names.ffn_gate_weight,
                names.ffn_up_weight,
                names.ffn_down_weight,
            ]);
        }
        for suffix in [
            LLM_OUTPUT_NORM_WEIGHT,
            LLM_OUTPUT_WEIGHT,
            LLM_TOKEN_EMBD_WEIGHT,
        ] {
            loader_decoder_names.insert(suffix.to_string());
        }
        assert_eq!(
            descriptor_names(funasr_nano_decoder_tensor_descriptors(&decoder)),
            loader_decoder_names,
            "decoder contract names must equal the loader's name sources"
        );

        // Encoder half suffixes are likewise shared constants; spot-check
        // both scopes' block-0 sets against the constant-derived names.
        let encoder_names = descriptor_names(funasr_nano_encoder_tensor_descriptors(&encoder));
        for scope in ["enc.blk.0", "tp.blk.0"] {
            for suffix in [
                "attn.norm.weight",
                "attn.qkv.weight",
                "attn.fsmn.weight",
                "ffn.down.bias",
            ] {
                assert!(
                    encoder_names.contains(&format!("{scope}.{suffix}")),
                    "encoder contract must cover {scope}.{suffix}"
                );
            }
        }
    }

    /// Architecture ceilings fail closed on untrusted metadata, keeping
    /// contract construction allocation-bounded and overflow-free.
    #[test]
    fn rejects_geometry_above_architecture_ceilings() {
        for (key, value) in [
            (ENC_N_LAYERS_KEY, FUNASR_NANO_MAX_LAYERS as u64 + 1),
            (ENC_TP_BLOCKS_KEY, FUNASR_NANO_MAX_LAYERS as u64 + 1),
            (ENC_D_MODEL_KEY, FUNASR_NANO_MAX_D_MODEL as u64 + 1),
            (ENC_FFN_DIM_KEY, FUNASR_NANO_MAX_FFN_DIM as u64 + 1),
            (ENC_FSMN_KERNEL_KEY, FUNASR_NANO_MAX_FSMN_KERNEL as u64 + 1),
            (ENC_FEATURE_DIM_KEY, FUNASR_NANO_MAX_FEATURE_DIM as u64 + 1),
        ] {
            let mut metadata = full_metadata();
            metadata.insert(key.to_string(), value.to_string());
            assert!(
                parse_funasr_nano_encoder_metadata(&metadata).is_err(),
                "must reject {key} = {value} above its ceiling"
            );
        }
        for (key, value) in [
            (ADP_N_LAYERS_KEY, FUNASR_NANO_MAX_LAYERS as u64 + 1),
            (ADP_ENCODER_DIM_KEY, FUNASR_NANO_MAX_D_MODEL as u64 + 1),
            (ADP_LLM_DIM_KEY, FUNASR_NANO_MAX_D_MODEL as u64 + 1),
        ] {
            let mut metadata = full_metadata();
            metadata.insert(key.to_string(), value.to_string());
            assert!(
                parse_funasr_nano_adapter_metadata(&metadata).is_err(),
                "must reject {key} = {value} above its ceiling"
            );
        }
        for (key, value) in [
            (LLM_N_LAYERS_KEY, FUNASR_NANO_MAX_LAYERS as u64 + 1),
            (LLM_D_MODEL_KEY, FUNASR_NANO_MAX_D_MODEL as u64 + 1),
            (LLM_FFN_DIM_KEY, FUNASR_NANO_MAX_FFN_DIM as u64 + 1),
            (LLM_VOCAB_SIZE_KEY, FUNASR_NANO_MAX_VOCAB_SIZE as u64 + 1),
            (LLM_MAX_POSITIONS_KEY, FUNASR_NANO_MAX_POSITIONS as u64 + 1),
        ] {
            let mut metadata = full_metadata();
            metadata.insert(key.to_string(), value.to_string());
            assert!(
                parse_funasr_nano_decoder_metadata(&metadata).is_err(),
                "must reject {key} = {value} above its ceiling"
            );
        }
    }

    /// Boundary: geometry exactly at the ceilings stays admissible (the
    /// ceilings bound, they do not shrink the production envelope). The
    /// encoder/decoder heads keep their divisibility invariants consistent.
    #[test]
    fn accepts_geometry_at_the_architecture_ceilings() {
        let mut metadata = full_metadata();
        metadata.insert(
            ENC_N_LAYERS_KEY.to_string(),
            FUNASR_NANO_MAX_LAYERS.to_string(),
        );
        metadata.insert(
            ENC_FFN_DIM_KEY.to_string(),
            FUNASR_NANO_MAX_FFN_DIM.to_string(),
        );
        assert!(parse_funasr_nano_encoder_metadata(&metadata).is_ok());

        let mut metadata = full_metadata();
        metadata.insert(
            LLM_N_LAYERS_KEY.to_string(),
            FUNASR_NANO_MAX_LAYERS.to_string(),
        );
        metadata.insert(
            LLM_FFN_DIM_KEY.to_string(),
            FUNASR_NANO_MAX_FFN_DIM.to_string(),
        );
        assert!(parse_funasr_nano_decoder_metadata(&metadata).is_ok());
    }

    /// Overflowing head geometry must fail closed through checked arithmetic
    /// instead of wrapping into an accidentally satisfying product.
    #[test]
    fn rejects_overflowing_head_geometry_without_wrapping() {
        let mut metadata = full_metadata();
        metadata.insert(ENC_N_HEADS_KEY.to_string(), u64::MAX.to_string());
        metadata.insert(ENC_HEAD_DIM_KEY.to_string(), u64::MAX.to_string());
        // The n_heads ceiling fires first; either way parse fails closed
        // without panicking or wrapping the product.
        assert!(parse_funasr_nano_encoder_metadata(&metadata).is_err());
    }

    #[test]
    fn validates_the_projected_tiny_tensor_set() {
        let (encoder, adapter, decoder) = (tiny_encoder(), tiny_adapter(), tiny_decoder());
        let shapes = funasr_nano_runtime_tensors(&encoder, &adapter, &decoder);
        let index = tensor_index_from_shapes(&shapes);
        validate_funasr_nano_runtime_tensors_with_index(&index, &encoder, &adapter, &decoder)
            .expect("projected tensor set must satisfy the contract");
    }

    #[test]
    fn rejects_a_missing_required_tensor() {
        let (encoder, adapter, decoder) = (tiny_encoder(), tiny_adapter(), tiny_decoder());
        let mut shapes = funasr_nano_runtime_tensors(&encoder, &adapter, &decoder);
        shapes.retain(|(name, _)| name != "adaptor.linear2.weight");
        let index = tensor_index_from_shapes(&shapes);
        let error =
            validate_funasr_nano_runtime_tensors_with_index(&index, &encoder, &adapter, &decoder)
                .expect_err("missing adaptor linear2 must fail closed");
        assert!(
            matches!(error, FunasrNanoTensorContractError::MissingRequiredTensor { ref name } if name == "adaptor.linear2.weight"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_a_wrong_shape() {
        let (encoder, adapter, decoder) = (tiny_encoder(), tiny_adapter(), tiny_decoder());
        let mut shapes = funasr_nano_runtime_tensors(&encoder, &adapter, &decoder);
        for (name, dims) in shapes.iter_mut() {
            if name == "enc.blk.0.attn.fsmn.weight" {
                *dims = vec![1, 1];
            }
        }
        let index = tensor_index_from_shapes(&shapes);
        let error =
            validate_funasr_nano_runtime_tensors_with_index(&index, &encoder, &adapter, &decoder)
                .expect_err("corrupted FSMN kernel must fail closed");
        assert!(
            matches!(error, FunasrNanoTensorContractError::InvalidTensorShape { ref name, .. } if name == "enc.blk.0.attn.fsmn.weight"),
            "unexpected error: {error}"
        );
    }
}
