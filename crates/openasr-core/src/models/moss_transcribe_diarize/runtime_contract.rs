//! moss-transcribe-diarize execution metadata parsed from the `.oasr` GGUF
//! header. Key names match exactly what `package_import` writes.

use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_u64_scalar, u64_to_u32, u64_to_usize,
    validate_positive_usize,
};

pub(crate) const ENCODER_N_LAYERS_KEY: &str = "moss_td.encoder.n_layers";
pub(crate) const ENCODER_D_MODEL_KEY: &str = "moss_td.encoder.d_model";
pub(crate) const ENCODER_N_HEADS_KEY: &str = "moss_td.encoder.n_heads";
pub(crate) const ENCODER_FFN_DIM_KEY: &str = "moss_td.encoder.ffn_dim";
pub(crate) const ENCODER_N_MELS_KEY: &str = "moss_td.encoder.n_mels";
pub(crate) const ENCODER_MAX_SOURCE_POSITIONS_KEY: &str = "moss_td.encoder.max_source_positions";
pub(crate) const ADAPTOR_MERGE_SIZE_KEY: &str = "moss_td.adaptor.merge_size";
pub(crate) const ADAPTOR_INPUT_DIM_KEY: &str = "moss_td.adaptor.input_dim";
pub(crate) const LLM_N_LAYERS_KEY: &str = "moss_td.llm.n_layers";
pub(crate) const LLM_D_MODEL_KEY: &str = "moss_td.llm.d_model";
pub(crate) const LLM_FFN_DIM_KEY: &str = "moss_td.llm.ffn_dim";
pub(crate) const LLM_N_HEADS_KEY: &str = "moss_td.llm.n_heads";
pub(crate) const LLM_N_KV_HEADS_KEY: &str = "moss_td.llm.n_kv_heads";
pub(crate) const LLM_HEAD_DIM_KEY: &str = "moss_td.llm.head_dim";
pub(crate) const LLM_VOCAB_SIZE_KEY: &str = "moss_td.llm.vocab_size";
pub(crate) const LLM_MAX_POSITIONS_KEY: &str = "moss_td.llm.max_positions";
pub(crate) const LLM_AUDIO_START_TOKEN_ID_KEY: &str = "moss_td.llm.audio_start_token_id";
pub(crate) const LLM_AUDIO_END_TOKEN_ID_KEY: &str = "moss_td.llm.audio_end_token_id";
pub(crate) const LLM_AUDIO_PAD_TOKEN_ID_KEY: &str = "moss_td.llm.audio_pad_token_id";

/// `rope_theta` (1e6) and RMSNorm epsilon (1e-6) are fixed properties of the
/// checkpoint's Qwen3-0.6B decoder (`config.json`'s `text_config.rope_theta`
/// / `rms_norm_eps`, verified against the real checkpoint), not per-pack
/// metadata -- the same "family constant, not a GGUF key" convention
/// `firered_llm::runtime_contract`'s `FIRERED_LLM_ROPE_THETA` uses.
pub(crate) const MOSS_TD_ROPE_THETA: f32 = 1_000_000.0;
pub(crate) const MOSS_TD_RMS_NORM_EPSILON: f32 = 1e-6;
/// `nn.LayerNorm`'s `eps` in `VQAdaptor.__init__` (`config.py`:
/// `norm_eps=config.text_config.rms_norm_eps`) -- same value as the decoder's
/// RMSNorm epsilon, verified against the real checkpoint's `config.json`.
pub(crate) const MOSS_TD_ADAPTOR_NORM_EPSILON: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MossTdEncoderMetadata {
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub n_mels: usize,
    pub max_source_positions: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MossTdAdaptorMetadata {
    pub merge_size: usize,
    pub input_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MossTdDecoderMetadata {
    pub n_layers: usize,
    pub d_model: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_positions: usize,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub audio_pad_token_id: u32,
}

pub(crate) fn parse_encoder_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<MossTdEncoderMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let n_layers = usize_key(ENCODER_N_LAYERS_KEY)?;
    let d_model = usize_key(ENCODER_D_MODEL_KEY)?;
    let n_heads = usize_key(ENCODER_N_HEADS_KEY)?;
    let ffn_dim = usize_key(ENCODER_FFN_DIM_KEY)?;
    let n_mels = usize_key(ENCODER_N_MELS_KEY)?;
    let max_source_positions = usize_key(ENCODER_MAX_SOURCE_POSITIONS_KEY)?;
    for (key, value) in [
        (ENCODER_N_LAYERS_KEY, n_layers),
        (ENCODER_D_MODEL_KEY, d_model),
        (ENCODER_N_HEADS_KEY, n_heads),
        (ENCODER_FFN_DIM_KEY, ffn_dim),
        (ENCODER_N_MELS_KEY, n_mels),
        (ENCODER_MAX_SOURCE_POSITIONS_KEY, max_source_positions),
    ] {
        validate_positive_usize(value, key)?;
    }
    if n_heads == 0 || !d_model.is_multiple_of(n_heads) {
        return Err(MetadataContractError::InvalidValue {
            key: ENCODER_N_HEADS_KEY,
            reason: format!("d_model {d_model} is not a multiple of n_heads {n_heads}"),
        });
    }
    Ok(MossTdEncoderMetadata {
        n_layers,
        d_model,
        n_heads,
        ffn_dim,
        n_mels,
        max_source_positions,
    })
}

pub(crate) fn parse_adaptor_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<MossTdAdaptorMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let merge_size = usize_key(ADAPTOR_MERGE_SIZE_KEY)?;
    let input_dim = usize_key(ADAPTOR_INPUT_DIM_KEY)?;
    validate_positive_usize(merge_size, ADAPTOR_MERGE_SIZE_KEY)?;
    validate_positive_usize(input_dim, ADAPTOR_INPUT_DIM_KEY)?;
    Ok(MossTdAdaptorMetadata {
        merge_size,
        input_dim,
    })
}

pub(crate) fn parse_decoder_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<MossTdDecoderMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let u32_key = |key: &'static str| -> Result<u32, MetadataContractError> {
        u64_to_u32(required_u64_scalar(metadata, key)?, key)
    };
    let n_layers = usize_key(LLM_N_LAYERS_KEY)?;
    let d_model = usize_key(LLM_D_MODEL_KEY)?;
    let ffn_dim = usize_key(LLM_FFN_DIM_KEY)?;
    let n_heads = usize_key(LLM_N_HEADS_KEY)?;
    let n_kv_heads = usize_key(LLM_N_KV_HEADS_KEY)?;
    let head_dim = usize_key(LLM_HEAD_DIM_KEY)?;
    let vocab_size = usize_key(LLM_VOCAB_SIZE_KEY)?;
    let max_positions = usize_key(LLM_MAX_POSITIONS_KEY)?;
    let audio_start_token_id = u32_key(LLM_AUDIO_START_TOKEN_ID_KEY)?;
    let audio_end_token_id = u32_key(LLM_AUDIO_END_TOKEN_ID_KEY)?;
    let audio_pad_token_id = u32_key(LLM_AUDIO_PAD_TOKEN_ID_KEY)?;

    for (key, value) in [
        (LLM_N_LAYERS_KEY, n_layers),
        (LLM_D_MODEL_KEY, d_model),
        (LLM_FFN_DIM_KEY, ffn_dim),
        (LLM_N_HEADS_KEY, n_heads),
        (LLM_N_KV_HEADS_KEY, n_kv_heads),
        (LLM_HEAD_DIM_KEY, head_dim),
        (LLM_VOCAB_SIZE_KEY, vocab_size),
        (LLM_MAX_POSITIONS_KEY, max_positions),
    ] {
        validate_positive_usize(value, key)?;
    }
    // Unlike Qwen2/firered-llm, Qwen3 decouples the per-head projection width
    // from `d_model / n_heads`: the real checkpoint's `head_dim` (128) times
    // `n_heads` (16) is 2048, not `d_model`'s 1024 -- `q_proj`/`k_proj`/
    // `v_proj` project to `n_heads * head_dim` and `attn_output` projects
    // back down to `d_model` (verified against the real checkpoint's
    // `config.json`). So there is no `n_heads * head_dim == d_model`
    // invariant to enforce here (matches `qwen::runtime_contract`, which
    // never asserts one either).
    if n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
        return Err(MetadataContractError::InvalidValue {
            key: LLM_N_KV_HEADS_KEY,
            reason: format!("n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}"),
        });
    }
    for (key, id) in [
        (LLM_AUDIO_START_TOKEN_ID_KEY, audio_start_token_id),
        (LLM_AUDIO_END_TOKEN_ID_KEY, audio_end_token_id),
        (LLM_AUDIO_PAD_TOKEN_ID_KEY, audio_pad_token_id),
    ] {
        if (id as usize) >= vocab_size {
            return Err(MetadataContractError::InvalidValue {
                key,
                reason: format!("token id {id} out of range for vocab_size {vocab_size}"),
            });
        }
    }

    Ok(MossTdDecoderMetadata {
        n_layers,
        d_model,
        ffn_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        vocab_size,
        max_positions,
        audio_start_token_id,
        audio_end_token_id,
        audio_pad_token_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn full_metadata() -> BTreeMap<String, String> {
        [
            (ENCODER_N_LAYERS_KEY, "24"),
            (ENCODER_D_MODEL_KEY, "1024"),
            (ENCODER_N_HEADS_KEY, "16"),
            (ENCODER_FFN_DIM_KEY, "4096"),
            (ENCODER_N_MELS_KEY, "80"),
            (ENCODER_MAX_SOURCE_POSITIONS_KEY, "1500"),
            (ADAPTOR_MERGE_SIZE_KEY, "4"),
            (ADAPTOR_INPUT_DIM_KEY, "4096"),
            (LLM_N_LAYERS_KEY, "28"),
            (LLM_D_MODEL_KEY, "1024"),
            (LLM_FFN_DIM_KEY, "3072"),
            (LLM_N_HEADS_KEY, "16"),
            (LLM_N_KV_HEADS_KEY, "8"),
            (LLM_HEAD_DIM_KEY, "128"),
            (LLM_VOCAB_SIZE_KEY, "151936"),
            (LLM_MAX_POSITIONS_KEY, "131072"),
            (LLM_AUDIO_START_TOKEN_ID_KEY, "151669"),
            (LLM_AUDIO_END_TOKEN_ID_KEY, "151670"),
            (LLM_AUDIO_PAD_TOKEN_ID_KEY, "151671"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    #[test]
    fn parses_encoder_metadata_matching_real_checkpoint() {
        let parsed = parse_encoder_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.n_layers, 24);
        assert_eq!(parsed.d_model, 1024);
        assert_eq!(parsed.max_source_positions, 1500);
    }

    #[test]
    fn parses_adaptor_metadata_matching_real_checkpoint() {
        let parsed = parse_adaptor_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.merge_size, 4);
        assert_eq!(parsed.input_dim, 4096);
    }

    #[test]
    fn parses_decoder_metadata_matching_real_checkpoint() {
        let parsed = parse_decoder_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.n_kv_heads, 8);
        assert_eq!(parsed.audio_pad_token_id, 151_671);
    }

    #[test]
    fn rejects_kv_heads_not_dividing_heads() {
        let mut metadata = full_metadata();
        metadata.insert(LLM_N_KV_HEADS_KEY.to_string(), "3".to_string());
        assert!(parse_decoder_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_audio_token_id_out_of_vocab() {
        let mut metadata = full_metadata();
        metadata.insert(LLM_AUDIO_PAD_TOKEN_ID_KEY.to_string(), "999999".to_string());
        assert!(parse_decoder_metadata(&metadata).is_err());
    }
}
