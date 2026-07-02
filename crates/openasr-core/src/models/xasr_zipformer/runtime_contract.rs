//! X-ASR Zipformer2 execution metadata parsed from the `.oasr` GGUF header.

use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_string_scalar, required_u64_scalar,
    u64_to_u32, u64_to_usize, validate_positive_usize,
};

pub(crate) const XASR_NUM_STACKS_KEY: &str = "xasr.num_stacks";
pub(crate) const XASR_NUM_ENCODER_LAYERS_KEY: &str = "xasr.num_encoder_layers";
pub(crate) const XASR_ENCODER_DIMS_KEY: &str = "xasr.encoder_dims";
pub(crate) const XASR_QUERY_HEAD_DIMS_KEY: &str = "xasr.query_head_dims";
pub(crate) const XASR_VALUE_HEAD_DIMS_KEY: &str = "xasr.value_head_dims";
pub(crate) const XASR_NUM_HEADS_KEY: &str = "xasr.num_heads";
pub(crate) const XASR_CNN_MODULE_KERNELS_KEY: &str = "xasr.cnn_module_kernels";
pub(crate) const XASR_LEFT_CONTEXT_LEN_KEY: &str = "xasr.left_context_len";
pub(crate) const XASR_DOWNSAMPLING_FACTORS_KEY: &str = "xasr.downsampling_factors";
pub(crate) const XASR_FEATURE_DIM_KEY: &str = "xasr.feature_dim";
pub(crate) const XASR_DECODE_CHUNK_LEN_KEY: &str = "xasr.decode_chunk_len";
pub(crate) const XASR_JOINER_DIM_KEY: &str = "xasr.joiner_dim";
pub(crate) const XASR_DECODER_CONTEXT_SIZE_KEY: &str = "xasr.decoder_context_size";
pub(crate) const XASR_VOCAB_SIZE_KEY: &str = "xasr.vocab_size";
pub(crate) const XASR_BLANK_ID_KEY: &str = "xasr.blank_id";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XasrZipformerExecutionMetadata {
    pub num_stacks: usize,
    pub num_encoder_layers: Vec<usize>,
    pub encoder_dims: Vec<usize>,
    pub query_head_dims: Vec<usize>,
    pub value_head_dims: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub cnn_module_kernels: Vec<usize>,
    pub left_context_len: Vec<usize>,
    pub downsampling_factors: Vec<usize>,
    pub feature_dim: usize,
    pub decode_chunk_len: usize,
    pub joiner_dim: usize,
    pub decoder_context_size: usize,
    pub vocab_size: usize,
    pub blank_id: u32,
}

impl XasrZipformerExecutionMetadata {
    pub(crate) fn total_encoder_layers(&self) -> usize {
        self.num_encoder_layers.iter().sum()
    }

    pub(crate) fn decoder_dim(&self) -> usize {
        self.joiner_dim
    }

    pub(crate) fn encoder_output_dim(&self) -> usize {
        self.encoder_dims
            .iter()
            .copied()
            .max()
            .unwrap_or(self.joiner_dim)
    }
}

pub(crate) fn parse_xasr_zipformer_execution_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<XasrZipformerExecutionMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let u32_key = |key: &'static str| -> Result<u32, MetadataContractError> {
        u64_to_u32(required_u64_scalar(metadata, key)?, key)
    };

    let num_stacks = usize_key(XASR_NUM_STACKS_KEY)?;
    validate_positive_usize(num_stacks, XASR_NUM_STACKS_KEY)?;

    let num_encoder_layers =
        required_usize_list(metadata, XASR_NUM_ENCODER_LAYERS_KEY, num_stacks)?;
    let encoder_dims = required_usize_list(metadata, XASR_ENCODER_DIMS_KEY, num_stacks)?;
    let query_head_dims = required_usize_list(metadata, XASR_QUERY_HEAD_DIMS_KEY, num_stacks)?;
    let value_head_dims = required_usize_list(metadata, XASR_VALUE_HEAD_DIMS_KEY, num_stacks)?;
    let num_heads = required_usize_list(metadata, XASR_NUM_HEADS_KEY, num_stacks)?;
    let cnn_module_kernels =
        required_usize_list(metadata, XASR_CNN_MODULE_KERNELS_KEY, num_stacks)?;
    let left_context_len = required_usize_list(metadata, XASR_LEFT_CONTEXT_LEN_KEY, num_stacks)?;
    let downsampling_factors =
        required_usize_list(metadata, XASR_DOWNSAMPLING_FACTORS_KEY, num_stacks)?;

    let feature_dim = usize_key(XASR_FEATURE_DIM_KEY)?;
    let decode_chunk_len = usize_key(XASR_DECODE_CHUNK_LEN_KEY)?;
    let joiner_dim = usize_key(XASR_JOINER_DIM_KEY)?;
    let decoder_context_size = usize_key(XASR_DECODER_CONTEXT_SIZE_KEY)?;
    let vocab_size = usize_key(XASR_VOCAB_SIZE_KEY)?;
    let blank_id = u32_key(XASR_BLANK_ID_KEY)?;

    for (key, value) in [
        (XASR_FEATURE_DIM_KEY, feature_dim),
        (XASR_DECODE_CHUNK_LEN_KEY, decode_chunk_len),
        (XASR_JOINER_DIM_KEY, joiner_dim),
        (XASR_DECODER_CONTEXT_SIZE_KEY, decoder_context_size),
        (XASR_VOCAB_SIZE_KEY, vocab_size),
    ] {
        validate_positive_usize(value, key)?;
    }
    if blank_id as usize >= vocab_size {
        return Err(MetadataContractError::InvalidValue {
            key: XASR_BLANK_ID_KEY,
            reason: format!("blank_id {blank_id} out of range for vocab_size {vocab_size}"),
        });
    }
    if decoder_context_size != 2 {
        return Err(MetadataContractError::InvalidValue {
            key: XASR_DECODER_CONTEXT_SIZE_KEY,
            reason: format!(
                "stateless X-ASR predictor expects context_size=2, got {decoder_context_size}"
            ),
        });
    }
    for (stack, ((heads, q_dim), v_dim)) in num_heads
        .iter()
        .zip(query_head_dims.iter())
        .zip(value_head_dims.iter())
        .enumerate()
    {
        validate_positive_usize(*heads, XASR_NUM_HEADS_KEY)?;
        validate_positive_usize(*q_dim, XASR_QUERY_HEAD_DIMS_KEY)?;
        validate_positive_usize(*v_dim, XASR_VALUE_HEAD_DIMS_KEY)?;
        let attn_dim =
            heads
                .checked_mul(*q_dim)
                .ok_or_else(|| MetadataContractError::InvalidValue {
                    key: XASR_QUERY_HEAD_DIMS_KEY,
                    reason: format!("stack {stack} attention dim overflows"),
                })?;
        if attn_dim == 0 {
            return Err(MetadataContractError::InvalidValue {
                key: XASR_QUERY_HEAD_DIMS_KEY,
                reason: format!("stack {stack} attention dim must be > 0"),
            });
        }
    }

    Ok(XasrZipformerExecutionMetadata {
        num_stacks,
        num_encoder_layers,
        encoder_dims,
        query_head_dims,
        value_head_dims,
        num_heads,
        cnn_module_kernels,
        left_context_len,
        downsampling_factors,
        feature_dim,
        decode_chunk_len,
        joiner_dim,
        decoder_context_size,
        vocab_size,
        blank_id,
    })
}

fn required_usize_list<M: ScalarMetadataView>(
    metadata: &M,
    key: &'static str,
    expected_len: usize,
) -> Result<Vec<usize>, MetadataContractError> {
    let raw = required_string_scalar(metadata, key)?;
    let values = raw
        .split(',')
        .map(str::trim)
        .enumerate()
        .map(|(index, item)| {
            if item.is_empty() {
                return Err(MetadataContractError::InvalidValue {
                    key,
                    reason: format!("entry {index} is empty"),
                });
            }
            let value =
                item.parse::<usize>()
                    .map_err(|source| MetadataContractError::InvalidValue {
                        key,
                        reason: format!("entry {index} '{item}' is not usize: {source}"),
                    })?;
            validate_positive_usize(value, key)?;
            Ok(value)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != expected_len {
        return Err(MetadataContractError::InvalidValue {
            key,
            reason: format!("expected {expected_len} entries, got {}", values.len()),
        });
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn metadata() -> BTreeMap<String, String> {
        [
            (XASR_NUM_STACKS_KEY, "6"),
            (XASR_NUM_ENCODER_LAYERS_KEY, "2,2,4,5,4,2"),
            (XASR_ENCODER_DIMS_KEY, "192,256,512,768,512,256"),
            (XASR_QUERY_HEAD_DIMS_KEY, "32,32,32,32,32,32"),
            (XASR_VALUE_HEAD_DIMS_KEY, "12,12,12,12,12,12"),
            (XASR_NUM_HEADS_KEY, "4,4,4,8,4,4"),
            (XASR_CNN_MODULE_KERNELS_KEY, "31,31,15,15,15,31"),
            (XASR_LEFT_CONTEXT_LEN_KEY, "256,128,64,32,64,128"),
            (XASR_DOWNSAMPLING_FACTORS_KEY, "1,2,4,8,4,2"),
            (XASR_FEATURE_DIM_KEY, "80"),
            (XASR_DECODE_CHUNK_LEN_KEY, "48"),
            (XASR_JOINER_DIM_KEY, "512"),
            (XASR_DECODER_CONTEXT_SIZE_KEY, "2"),
            (XASR_VOCAB_SIZE_KEY, "5000"),
            (XASR_BLANK_ID_KEY, "0"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    #[test]
    fn parses_xasr_metadata() {
        let parsed = parse_xasr_zipformer_execution_metadata(&metadata()).expect("parse");
        assert_eq!(parsed.num_stacks, 6);
        assert_eq!(parsed.total_encoder_layers(), 19);
        assert_eq!(parsed.encoder_output_dim(), 768);
        assert_eq!(parsed.decoder_dim(), 512);
        assert_eq!(parsed.left_context_len, vec![256, 128, 64, 32, 64, 128]);
        assert_eq!(parsed.decode_chunk_len, 48);
        assert_eq!(parsed.blank_id, 0);
    }

    #[test]
    fn rejects_list_length_drift() {
        let mut metadata = metadata();
        metadata.insert(XASR_ENCODER_DIMS_KEY.to_string(), "192,256".to_string());
        assert!(parse_xasr_zipformer_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_unsupported_context_size() {
        let mut metadata = metadata();
        metadata.insert(XASR_DECODER_CONTEXT_SIZE_KEY.to_string(), "3".to_string());
        assert!(parse_xasr_zipformer_execution_metadata(&metadata).is_err());
    }
}
