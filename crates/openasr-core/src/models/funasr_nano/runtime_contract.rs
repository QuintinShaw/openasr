//! funasr-nano execution metadata parsed from the `.oasr` GGUF header. Key
//! names match exactly what the pack importer writes (`funasr.enc.*` /
//! `funasr.adp.*` / `funasr.llm.*`).

use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_u64_scalar, u64_to_u32, u64_to_usize,
    validate_positive_usize,
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
    if n_heads * head_dim != d_model {
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
}
