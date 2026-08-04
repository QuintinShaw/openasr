//! wav2vec2-ctc execution metadata parsed from the `.oasr` GGUF header, plus
//! the admission-time runtime tensor contract that proves the pack carries
//! every tensor the runtime will load (metadata-derived shapes checked against
//! the tensor index) before the pack is admitted.

#![allow(dead_code)]

use thiserror::Error;

use crate::GgufTensorIndex;
use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, optional_u64_scalar, required_u64_scalar,
    u64_to_u32, u64_to_usize, validate_positive_usize,
};
use crate::models::tensor_binding::render_shape;

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

/// Typed tensor-contract failure for a single wav2vec2-ctc pack.
#[derive(Debug, Error, Clone, PartialEq)]
pub(crate) enum Wav2Vec2CtcTensorContractError {
    #[error("wav2vec2-ctc missing required runtime tensor '{name}'")]
    MissingRequiredTensor { name: String },
    #[error("wav2vec2-ctc runtime tensor '{name}' has shape {shape}: {reason}")]
    InvalidTensorShape {
        name: String,
        shape: String,
        reason: String,
    },
}

/// The single required-tensor enumeration for the wav2vec2-ctc family.
///
/// Every tensor the runtime loads has a shape fully determined by the parsed
/// execution metadata plus the architecture-constant feature-extractor
/// geometry, so the admission-time validator and the runtime-ready pack
/// fixture share this one list: a pack that is missing any entry, or carries a
/// different shape, fails closed at admission instead of mid-execution.
/// 2-D linear weights appear in their stored ggml `[in, out]` orientation (the
/// importer reverses the HF `[out, in]` layout); conv kernels appear in ggml
/// `[K, in_per_group, out_channels]` layout.
pub(crate) fn wav2vec2_ctc_runtime_tensors(
    metadata: &Wav2Vec2CtcExecutionMetadata,
) -> Vec<(String, Vec<u64>)> {
    let hidden = metadata.hidden_size as u64;
    let ffn = metadata.ffn_dim as u64;
    let vocab = metadata.vocab_size as u64;
    // Metadata parsing proves `hidden_size % num_conv_pos_embedding_groups == 0`.
    let pos_in_per_group = hidden / metadata.num_conv_pos_embedding_groups as u64;
    let pos_kernel = metadata.num_conv_pos_embeddings as u64;

    let mut out = Vec::new();

    // 7-layer strided conv feature extractor. Input channels chain
    // 1 -> 512 -> ...; the channel-norm + conv-bias tensors are flag-gated
    // exactly as the importer emits them (group norm: layer 0 only; layer
    // norm: every layer; conv bias: every layer when present).
    for (layer, kernel) in FEATURE_EXTRACTOR_CONV_KERNEL.iter().enumerate() {
        let in_channels = if layer == 0 {
            1
        } else {
            FEATURE_EXTRACTOR_CONV_DIM[layer - 1]
        } as u64;
        let out_channels = FEATURE_EXTRACTOR_CONV_DIM[layer] as u64;
        out.push((
            format!("enc.fe.{layer}.conv.weight"),
            vec![*kernel as u64, in_channels, out_channels],
        ));
        if metadata.conv_bias {
            out.push((format!("enc.fe.{layer}.conv.bias"), vec![out_channels]));
        }
        let has_channel_norm = match metadata.feat_extract_norm {
            FeatExtractNorm::Group => layer == 0,
            FeatExtractNorm::Layer => true,
        };
        if has_channel_norm {
            out.push((format!("enc.fe.{layer}.gn.weight"), vec![out_channels]));
            out.push((format!("enc.fe.{layer}.gn.bias"), vec![out_channels]));
        }
    }

    // Feature projection: LayerNorm over the extractor output, then the
    // `fe_out -> hidden` linear (stored `[fe_out, hidden]`).
    let fe_out = *FEATURE_EXTRACTOR_CONV_DIM.last().unwrap_or(&512) as u64;
    out.push(("enc.fp.norm.weight".to_string(), vec![fe_out]));
    out.push(("enc.fp.norm.bias".to_string(), vec![fe_out]));
    out.push(("enc.fp.proj.weight".to_string(), vec![fe_out, hidden]));
    out.push(("enc.fp.proj.bias".to_string(), vec![hidden]));

    // Positional conv stack: one folded weight-norm conv (wav2vec2/hubert) or
    // data2vec's stack of plain grouped convs.
    let pos_conv_depth = metadata.pos_conv_depth.max(1);
    if pos_conv_depth <= 1 {
        out.push((
            "enc.posconv.weight".to_string(),
            vec![pos_kernel, pos_in_per_group, hidden],
        ));
        out.push(("enc.posconv.bias".to_string(), vec![hidden]));
    } else {
        for layer in 0..pos_conv_depth {
            out.push((
                format!("enc.posconv.{layer}.weight"),
                vec![pos_kernel, pos_in_per_group, hidden],
            ));
            out.push((format!("enc.posconv.{layer}.bias"), vec![hidden]));
        }
    }

    // Encoder layer norm: applied before the stack on the post-norm encoder
    // (base) or after the stack on the stable encoder (large variants); both
    // paths consume it.
    out.push(("enc.norm.weight".to_string(), vec![hidden]));
    out.push(("enc.norm.bias".to_string(), vec![hidden]));

    // Post-norm / stable transformer layers.
    for layer in 0..metadata.n_layers {
        let prefix = format!("enc.blk.{layer}");
        for projection in ["attn.q", "attn.k", "attn.v", "attn.out"] {
            out.push((
                format!("{prefix}.{projection}.weight"),
                vec![hidden, hidden],
            ));
            out.push((format!("{prefix}.{projection}.bias"), vec![hidden]));
        }
        out.push((format!("{prefix}.attn.norm.weight"), vec![hidden]));
        out.push((format!("{prefix}.attn.norm.bias"), vec![hidden]));
        out.push((format!("{prefix}.ffn.up.weight"), vec![hidden, ffn]));
        out.push((format!("{prefix}.ffn.up.bias"), vec![ffn]));
        out.push((format!("{prefix}.ffn.down.weight"), vec![ffn, hidden]));
        out.push((format!("{prefix}.ffn.down.bias"), vec![hidden]));
        out.push((format!("{prefix}.final.norm.weight"), vec![hidden]));
        out.push((format!("{prefix}.final.norm.bias"), vec![hidden]));
    }

    // CTC head (the HF `lm_head` linear, stored `[hidden, vocab]`).
    out.push(("ctc.head.weight".to_string(), vec![hidden, vocab]));
    out.push(("ctc.head.bias".to_string(), vec![vocab]));
    out
}

/// Admission-time runtime tensor contract for wav2vec2-ctc. Validates the pack
/// tensor index against the single required-tensor enumeration derived from
/// `metadata`; a missing tensor or a shape the declared geometry cannot
/// construct fails closed with the offending tensor named.
pub(crate) fn validate_wav2vec2_ctc_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    metadata: &Wav2Vec2CtcExecutionMetadata,
) -> Result<(), Wav2Vec2CtcTensorContractError> {
    for (name, expected_dims) in wav2vec2_ctc_runtime_tensors(metadata) {
        let Some(tensor) = index.get(&name) else {
            return Err(Wav2Vec2CtcTensorContractError::MissingRequiredTensor { name });
        };
        if tensor.dims != expected_dims {
            return Err(Wav2Vec2CtcTensorContractError::InvalidTensorShape {
                name,
                shape: render_shape(&tensor.dims),
                reason: format!("expected shape {:?}", expected_dims),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_runtime_pack_contract(
    preflight: &crate::GgufRuntimeSourcePreflight,
) -> Result<(), String> {
    let metadata =
        parse_wav2vec2_ctc_execution_metadata(preflight.metadata()).map_err(|error| {
            crate::models::runtime_pack_contract::metadata_validation_error("wav2vec2-ctc", error)
        })?;
    validate_wav2vec2_ctc_runtime_tensors_with_index(preflight.tensor_index(), &metadata)
        .map_err(crate::models::runtime_pack_contract::tensor_validation_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{GgufTensorIndexSnapshot, GgufTensorMetadata};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

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

    // --- Runtime tensor contract ---

    /// Tiny internally-consistent geometry for tensor-level tests: one
    /// transformer layer, small widths. Every metadata invariant holds
    /// (head_dim * n_heads == hidden_size, hidden_size % groups == 0, even
    /// single-conv pos kernel, blank in vocab).
    fn tiny_metadata() -> Wav2Vec2CtcExecutionMetadata {
        Wav2Vec2CtcExecutionMetadata {
            n_layers: 1,
            hidden_size: 16,
            n_heads: 2,
            head_dim: 8,
            ffn_dim: 32,
            vocab_size: 4,
            num_conv_pos_embeddings: 4,
            num_conv_pos_embedding_groups: 2,
            pos_conv_depth: 1,
            blank_token_id: 0,
            feat_extract_norm: FeatExtractNorm::Group,
            do_stable_layer_norm: false,
            conv_bias: false,
        }
    }

    fn tensor_index_from_shapes(shapes: &[(String, Vec<u64>)]) -> crate::GgufTensorIndex {
        let tensors = shapes
            .iter()
            .enumerate()
            .map(|(index, (name, dims))| GgufTensorMetadata {
                name: name.clone(),
                dims: dims.clone(),
                ggml_type: 0,
                type_name: "f32".to_string(),
                size_bytes: 0,
                offset_bytes: index as u64,
            })
            .collect();
        crate::GgufTensorIndex::from_snapshot(GgufTensorIndexSnapshot {
            path: PathBuf::from("/tmp/wav2vec2-ctc-contract-test.oasr"),
            data_section_offset_bytes: 0,
            tensors,
        })
        .expect("unique tensor names")
    }

    #[test]
    fn validates_the_tiny_reference_tensor_set() {
        let metadata = tiny_metadata();
        let shapes = wav2vec2_ctc_runtime_tensors(&metadata);
        let index = tensor_index_from_shapes(&shapes);
        validate_wav2vec2_ctc_runtime_tensors_with_index(&index, &metadata)
            .expect("tiny tensor set must satisfy the contract");
    }

    #[test]
    fn tensor_enumeration_covers_every_runtime_loaded_tensor() {
        let metadata = tiny_metadata();
        let names: BTreeMap<String, Vec<u64>> = wav2vec2_ctc_runtime_tensors(&metadata)
            .into_iter()
            .collect();
        // feature extractor: 7 convs, group norm on layer 0 only, no conv bias.
        for layer in 0..7 {
            assert!(names.contains_key(&format!("enc.fe.{layer}.conv.weight")));
            assert!(!names.contains_key(&format!("enc.fe.{layer}.conv.bias")));
        }
        assert!(names.contains_key("enc.fe.0.gn.weight"));
        assert!(!names.contains_key("enc.fe.1.gn.weight"));
        // folded single pos-conv + projection + norms + one layer + head.
        for name in [
            "enc.fp.proj.weight",
            "enc.posconv.weight",
            "enc.posconv.bias",
            "enc.norm.weight",
            "enc.blk.0.attn.q.weight",
            "enc.blk.0.ffn.down.bias",
            "enc.blk.0.final.norm.bias",
            "ctc.head.weight",
            "ctc.head.bias",
        ] {
            assert!(names.contains_key(name), "missing {name}");
        }
        assert!(!names.contains_key("enc.posconv.0.weight"));
        assert!(!names.contains_key("enc.blk.1.attn.q.weight"));
        // Shapes follow the stored ggml orientation: fp proj `[fe_out, hidden]`,
        // ctc head `[hidden, vocab]`, pos-conv `[kernel, in_per_group, hidden]`.
        assert_eq!(names["enc.fp.proj.weight"], vec![512, 16]);
        assert_eq!(names["ctc.head.weight"], vec![16, 4]);
        assert_eq!(names["enc.posconv.weight"], vec![4, 8, 16]);
        assert_eq!(names["enc.fe.0.conv.weight"], vec![10, 1, 512]);
        assert_eq!(names["enc.fe.6.conv.weight"], vec![2, 512, 512]);
    }

    #[test]
    fn rejects_a_missing_required_tensor() {
        let metadata = tiny_metadata();
        let mut shapes = wav2vec2_ctc_runtime_tensors(&metadata);
        shapes.retain(|(name, _)| name != "enc.blk.0.ffn.down.weight");
        let index = tensor_index_from_shapes(&shapes);
        let error = validate_wav2vec2_ctc_runtime_tensors_with_index(&index, &metadata)
            .expect_err("missing layer tensor must fail closed");
        match error {
            Wav2Vec2CtcTensorContractError::MissingRequiredTensor { name } => {
                assert_eq!(name, "enc.blk.0.ffn.down.weight");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn rejects_a_ctc_head_with_the_wrong_vocab_width() {
        let metadata = tiny_metadata();
        let mut shapes = wav2vec2_ctc_runtime_tensors(&metadata);
        for (name, dims) in shapes.iter_mut() {
            if name == "ctc.head.weight" {
                *dims = vec![16, 99];
            }
        }
        let index = tensor_index_from_shapes(&shapes);
        let error = validate_wav2vec2_ctc_runtime_tensors_with_index(&index, &metadata)
            .expect_err("head width mismatch must fail closed");
        match error {
            Wav2Vec2CtcTensorContractError::InvalidTensorShape { name, .. } => {
                assert_eq!(name, "ctc.head.weight");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn conv_bias_flag_gates_the_feature_extractor_biases() {
        let mut metadata = tiny_metadata();
        metadata.conv_bias = true;
        let names: BTreeMap<String, Vec<u64>> = wav2vec2_ctc_runtime_tensors(&metadata)
            .into_iter()
            .collect();
        for layer in 0..7 {
            assert!(
                names.contains_key(&format!("enc.fe.{layer}.conv.bias")),
                "conv bias {layer}"
            );
        }
        assert_eq!(names["enc.fe.3.conv.bias"], vec![512]);
    }

    #[test]
    fn layer_feat_extract_norm_requires_every_channel_norm() {
        let mut metadata = tiny_metadata();
        metadata.feat_extract_norm = FeatExtractNorm::Layer;
        let names: BTreeMap<String, Vec<u64>> = wav2vec2_ctc_runtime_tensors(&metadata)
            .into_iter()
            .collect();
        for layer in 0..7 {
            assert!(names.contains_key(&format!("enc.fe.{layer}.gn.weight")));
            assert!(names.contains_key(&format!("enc.fe.{layer}.gn.bias")));
        }
    }

    #[test]
    fn data2vec_depth_enumerates_the_stacked_pos_convs() {
        let mut metadata = tiny_metadata();
        metadata.pos_conv_depth = 3;
        metadata.num_conv_pos_embeddings = 5;
        let names: BTreeMap<String, Vec<u64>> = wav2vec2_ctc_runtime_tensors(&metadata)
            .into_iter()
            .collect();
        assert!(!names.contains_key("enc.posconv.weight"));
        for layer in 0..3 {
            assert_eq!(
                names[&format!("enc.posconv.{layer}.weight")],
                vec![5, 8, 16]
            );
            assert_eq!(names[&format!("enc.posconv.{layer}.bias")], vec![16]);
        }
    }
}
