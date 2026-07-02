//! wav2vec2-ctc execution metadata parsed from the `.oasr` GGUF header.

#![allow(dead_code)]

use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, optional_u64_scalar, required_u64_scalar,
    u64_to_u32, u64_to_usize, validate_positive_usize,
};

pub(crate) const WAV2VEC2_N_LAYERS_KEY: &str = "wav2vec2.n_layers";
pub(crate) const WAV2VEC2_HIDDEN_SIZE_KEY: &str = "wav2vec2.hidden_size";
pub(crate) const WAV2VEC2_N_HEADS_KEY: &str = "wav2vec2.n_heads";
pub(crate) const WAV2VEC2_HEAD_DIM_KEY: &str = "wav2vec2.head_dim";
pub(crate) const WAV2VEC2_FFN_DIM_KEY: &str = "wav2vec2.ffn_dim";
pub(crate) const WAV2VEC2_VOCAB_SIZE_KEY: &str = "wav2vec2.vocab_size";
pub(crate) const WAV2VEC2_NUM_CONV_POS_EMBEDDINGS_KEY: &str = "wav2vec2.num_conv_pos_embeddings";
pub(crate) const WAV2VEC2_NUM_CONV_POS_EMBEDDING_GROUPS_KEY: &str =
    "wav2vec2.num_conv_pos_embedding_groups";
/// Positional-conv stack depth. Optional; absent/1 = the single weight-norm conv
/// (wav2vec2/hubert, even kernel + SamePad crop). >1 = data2vec's stack of plain
/// grouped convs (odd kernel, no crop, sequential + residual add).
pub(crate) const WAV2VEC2_POS_CONV_DEPTH_KEY: &str = "wav2vec2.pos_conv_depth";
pub(crate) const WAV2VEC2_CTC_BLANK_TOKEN_ID_KEY: &str = "ctc.blank_token_id";
/// Feature-extractor norm mode: `"group"` (single GroupNorm on conv layer 0,
/// base-960h) vs `"layer"` (per-conv-layer LayerNorm over channels, large
/// variants). Optional; defaults to `"group"` for legacy base-960h packs.
pub(crate) const WAV2VEC2_FEAT_EXTRACT_NORM_KEY: &str = "wav2vec2.feat_extract_norm";
/// `1` for the pre-norm "stable layer norm" encoder + final encoder LayerNorm
/// (large variants), `0` for the post-norm encoder (base-960h). Optional;
/// defaults to `0` for legacy packs.
pub(crate) const WAV2VEC2_DO_STABLE_LAYER_NORM_KEY: &str = "wav2vec2.do_stable_layer_norm";
/// `1` if the feature-extractor conv layers carry a bias (hubert/lv60), `0`
/// otherwise (base-960h, data2vec). Optional; defaults to `0`.
pub(crate) const WAV2VEC2_CONV_BIAS_KEY: &str = "wav2vec2.conv_bias";

/// Feature-extractor channel-normalization mode (the `feat_extract_norm` config).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FeatExtractNorm {
    /// Single GroupNorm (n_groups == channels = per-channel) on conv layer 0 only.
    Group,
    /// LayerNorm over the channel dim after EVERY conv layer.
    Layer,
}

impl FeatExtractNorm {
    fn from_str(value: &str) -> Result<Self, MetadataContractError> {
        match value.trim() {
            "group" => Ok(Self::Group),
            "layer" => Ok(Self::Layer),
            other => Err(MetadataContractError::InvalidValue {
                key: WAV2VEC2_FEAT_EXTRACT_NORM_KEY,
                reason: format!("unknown feat_extract_norm '{other}' (want 'group' or 'layer')"),
            }),
        }
    }
}

/// The 7-layer feature-extractor conv stack (fixed for the base/large family).
pub(crate) const FEATURE_EXTRACTOR_CONV_DIM: [usize; 7] = [512, 512, 512, 512, 512, 512, 512];
pub(crate) const FEATURE_EXTRACTOR_CONV_KERNEL: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
pub(crate) const FEATURE_EXTRACTOR_CONV_STRIDE: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Wav2Vec2CtcExecutionMetadata {
    pub n_layers: usize,
    pub hidden_size: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub num_conv_pos_embeddings: usize,
    pub num_conv_pos_embedding_groups: usize,
    /// Positional-conv stack depth: 1 = single weight-norm conv (wav2vec2/hubert),
    /// >1 = data2vec's stacked plain grouped convs.
    pub pos_conv_depth: usize,
    pub blank_token_id: u32,
    /// Feature-extractor norm mode (`group` legacy default, `layer` for large).
    pub feat_extract_norm: FeatExtractNorm,
    /// Pre-norm "stable layer norm" encoder + final encoder LayerNorm (large
    /// variants) vs post-norm (base-960h, the legacy default).
    pub do_stable_layer_norm: bool,
    /// Whether the feature-extractor conv layers carry a bias (hubert/lv60).
    pub conv_bias: bool,
}

pub(crate) fn parse_wav2vec2_ctc_execution_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<Wav2Vec2CtcExecutionMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let n_layers = usize_key(WAV2VEC2_N_LAYERS_KEY)?;
    let hidden_size = usize_key(WAV2VEC2_HIDDEN_SIZE_KEY)?;
    let n_heads = usize_key(WAV2VEC2_N_HEADS_KEY)?;
    let head_dim = usize_key(WAV2VEC2_HEAD_DIM_KEY)?;
    let ffn_dim = usize_key(WAV2VEC2_FFN_DIM_KEY)?;
    let vocab_size = usize_key(WAV2VEC2_VOCAB_SIZE_KEY)?;
    let num_conv_pos_embeddings = usize_key(WAV2VEC2_NUM_CONV_POS_EMBEDDINGS_KEY)?;
    let num_conv_pos_embedding_groups = usize_key(WAV2VEC2_NUM_CONV_POS_EMBEDDING_GROUPS_KEY)?;
    let blank_token_id = u64_to_u32(
        required_u64_scalar(metadata, WAV2VEC2_CTC_BLANK_TOKEN_ID_KEY)?,
        WAV2VEC2_CTC_BLANK_TOKEN_ID_KEY,
    )?;
    // New config flags are OPTIONAL with base-960h defaults so legacy packs
    // (group norm, post-norm encoder, no conv bias) load unchanged.
    let feat_extract_norm = match metadata.get_string_scalar(WAV2VEC2_FEAT_EXTRACT_NORM_KEY) {
        Some(value) => FeatExtractNorm::from_str(value)?,
        None => FeatExtractNorm::Group,
    };
    let do_stable_layer_norm =
        optional_u64_scalar(metadata, WAV2VEC2_DO_STABLE_LAYER_NORM_KEY)?.unwrap_or(0) != 0;
    let conv_bias = optional_u64_scalar(metadata, WAV2VEC2_CONV_BIAS_KEY)?.unwrap_or(0) != 0;
    let pos_conv_depth = u64_to_usize(
        optional_u64_scalar(metadata, WAV2VEC2_POS_CONV_DEPTH_KEY)?.unwrap_or(1),
        WAV2VEC2_POS_CONV_DEPTH_KEY,
    )?
    .max(1);

    for (key, value) in [
        (WAV2VEC2_N_LAYERS_KEY, n_layers),
        (WAV2VEC2_HIDDEN_SIZE_KEY, hidden_size),
        (WAV2VEC2_N_HEADS_KEY, n_heads),
        (WAV2VEC2_HEAD_DIM_KEY, head_dim),
        (WAV2VEC2_FFN_DIM_KEY, ffn_dim),
        (WAV2VEC2_VOCAB_SIZE_KEY, vocab_size),
        (
            WAV2VEC2_NUM_CONV_POS_EMBEDDINGS_KEY,
            num_conv_pos_embeddings,
        ),
        (
            WAV2VEC2_NUM_CONV_POS_EMBEDDING_GROUPS_KEY,
            num_conv_pos_embedding_groups,
        ),
    ] {
        validate_positive_usize(value, key)?;
    }
    if (blank_token_id as usize) >= vocab_size {
        return Err(MetadataContractError::InvalidValue {
            key: WAV2VEC2_CTC_BLANK_TOKEN_ID_KEY,
            reason: format!("blank {blank_token_id} out of range for vocab_size {vocab_size}"),
        });
    }
    if head_dim * n_heads != hidden_size {
        return Err(MetadataContractError::InvalidValue {
            key: WAV2VEC2_HEAD_DIM_KEY,
            reason: format!("head_dim {head_dim} * n_heads {n_heads} != hidden_size {hidden_size}"),
        });
    }
    // For the SINGLE weight-norm conv (wav2vec2/hubert) an even kernel is required
    // for the SamePadLayer crop to be well-defined (drop the last output frame).
    // data2vec's STACKED convs (pos_conv_depth > 1) use an odd kernel (19) and no
    // crop, so the parity requirement only applies to the single-conv path.
    if pos_conv_depth == 1 && num_conv_pos_embeddings % 2 != 0 {
        return Err(MetadataContractError::InvalidValue {
            key: WAV2VEC2_NUM_CONV_POS_EMBEDDINGS_KEY,
            reason: format!(
                "num_conv_pos_embeddings {num_conv_pos_embeddings} must be even for the SamePad crop"
            ),
        });
    }
    if hidden_size % num_conv_pos_embedding_groups != 0 {
        return Err(MetadataContractError::InvalidValue {
            key: WAV2VEC2_NUM_CONV_POS_EMBEDDING_GROUPS_KEY,
            reason: format!(
                "hidden_size {hidden_size} not divisible by groups {num_conv_pos_embedding_groups}"
            ),
        });
    }

    Ok(Wav2Vec2CtcExecutionMetadata {
        n_layers,
        hidden_size,
        n_heads,
        head_dim,
        ffn_dim,
        vocab_size,
        num_conv_pos_embeddings,
        num_conv_pos_embedding_groups,
        pos_conv_depth,
        blank_token_id,
        feat_extract_norm,
        do_stable_layer_norm,
        conv_bias,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn wav2vec2_metadata() -> BTreeMap<String, String> {
        [
            (WAV2VEC2_N_LAYERS_KEY, "12"),
            (WAV2VEC2_HIDDEN_SIZE_KEY, "768"),
            (WAV2VEC2_N_HEADS_KEY, "12"),
            (WAV2VEC2_HEAD_DIM_KEY, "64"),
            (WAV2VEC2_FFN_DIM_KEY, "3072"),
            (WAV2VEC2_VOCAB_SIZE_KEY, "32"),
            (WAV2VEC2_NUM_CONV_POS_EMBEDDINGS_KEY, "128"),
            (WAV2VEC2_NUM_CONV_POS_EMBEDDING_GROUPS_KEY, "16"),
            (WAV2VEC2_CTC_BLANK_TOKEN_ID_KEY, "0"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    #[test]
    fn parses_wav2vec2_base_960h_metadata() {
        let parsed = parse_wav2vec2_ctc_execution_metadata(&wav2vec2_metadata()).expect("parse");
        assert_eq!(parsed.n_layers, 12);
        assert_eq!(parsed.hidden_size, 768);
        assert_eq!(parsed.head_dim, 64);
        assert_eq!(parsed.vocab_size, 32);
        assert_eq!(parsed.blank_token_id, 0);
        assert_eq!(parsed.num_conv_pos_embedding_groups, 16);
        // legacy packs (no new flags) default to the base-960h config.
        assert_eq!(parsed.feat_extract_norm, FeatExtractNorm::Group);
        assert!(!parsed.do_stable_layer_norm);
        assert!(!parsed.conv_bias);
    }

    #[test]
    fn parses_large_variant_flags() {
        let mut metadata = wav2vec2_metadata();
        metadata.insert(
            WAV2VEC2_FEAT_EXTRACT_NORM_KEY.to_string(),
            "layer".to_string(),
        );
        metadata.insert(
            WAV2VEC2_DO_STABLE_LAYER_NORM_KEY.to_string(),
            "1".to_string(),
        );
        metadata.insert(WAV2VEC2_CONV_BIAS_KEY.to_string(), "1".to_string());
        let parsed = parse_wav2vec2_ctc_execution_metadata(&metadata).expect("parse");
        assert_eq!(parsed.feat_extract_norm, FeatExtractNorm::Layer);
        assert!(parsed.do_stable_layer_norm);
        assert!(parsed.conv_bias);
    }

    #[test]
    fn rejects_unknown_feat_extract_norm() {
        let mut metadata = wav2vec2_metadata();
        metadata.insert(
            WAV2VEC2_FEAT_EXTRACT_NORM_KEY.to_string(),
            "instance".to_string(),
        );
        assert!(parse_wav2vec2_ctc_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_odd_pos_conv_kernel() {
        let mut metadata = wav2vec2_metadata();
        metadata.insert(
            WAV2VEC2_NUM_CONV_POS_EMBEDDINGS_KEY.to_string(),
            "127".to_string(),
        );
        assert!(parse_wav2vec2_ctc_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_inconsistent_head_dim() {
        let mut metadata = wav2vec2_metadata();
        metadata.insert(WAV2VEC2_HEAD_DIM_KEY.to_string(), "100".to_string());
        assert!(parse_wav2vec2_ctc_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_blank_out_of_vocab() {
        let mut metadata = wav2vec2_metadata();
        metadata.insert(
            WAV2VEC2_CTC_BLANK_TOKEN_ID_KEY.to_string(),
            "99".to_string(),
        );
        assert!(parse_wav2vec2_ctc_execution_metadata(&metadata).is_err());
    }
}
