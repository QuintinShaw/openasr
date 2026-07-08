//! FireRedPunc execution metadata parsed from the `.oasr` GGUF header.
//!
//! The pull-time contract and the runtime both read the pack geometry through
//! [`parse_firered_punc_execution_metadata`]; it validates the architecture tag
//! and the chinese-lert-base geometry before any weights are materialised, so a
//! malformed or mis-converted pack fails closed.

use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_string_scalar, required_u64_scalar,
    u64_to_usize, validate_positive_usize,
};

use super::config::{
    FIRERED_PUNC_ARCHITECTURE_VALUE, FIRERED_PUNC_ATTENTION_HEAD_COUNT_KEY,
    FIRERED_PUNC_BLOCK_COUNT_KEY, FIRERED_PUNC_CONTEXT_LENGTH_KEY,
    FIRERED_PUNC_EMBEDDING_LENGTH_KEY, FIRERED_PUNC_FEED_FORWARD_LENGTH_KEY,
    FIRERED_PUNC_LABEL_COUNT_KEY, FIRERED_PUNC_VOCAB_SIZE_KEY, FireRedPuncConfigError,
    FireRedPuncExecutionMetadata,
};
use crate::arch::GENERAL_ARCHITECTURE_KEY;

/// Returns `true` when the pack declares the FireRedPunc architecture. The
/// pull-time dispatch (`crate::models::aux_pack_registry`) matches the same
/// `general.architecture` value directly against
/// [`FIRERED_PUNC_ARCHITECTURE_VALUE`] rather than calling this helper; it
/// stays as the geometry-parsing internal check and its own unit tests below.
pub(crate) fn metadata_declares_firered_punc<M: ScalarMetadataView>(metadata: &M) -> bool {
    metadata
        .get_string_scalar(GENERAL_ARCHITECTURE_KEY)
        .map(|arch| arch.trim() == FIRERED_PUNC_ARCHITECTURE_VALUE)
        .unwrap_or(false)
}

pub(crate) fn parse_firered_punc_execution_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<FireRedPuncExecutionMetadata, MetadataContractError> {
    let architecture = required_string_scalar(metadata, GENERAL_ARCHITECTURE_KEY)?;
    if architecture != FIRERED_PUNC_ARCHITECTURE_VALUE {
        return Err(MetadataContractError::InvalidValue {
            key: GENERAL_ARCHITECTURE_KEY,
            reason: format!("expected '{FIRERED_PUNC_ARCHITECTURE_VALUE}', got '{architecture}'"),
        });
    }

    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let layers = usize_key(FIRERED_PUNC_BLOCK_COUNT_KEY)?;
    let d_model = usize_key(FIRERED_PUNC_EMBEDDING_LENGTH_KEY)?;
    let ffn_dim = usize_key(FIRERED_PUNC_FEED_FORWARD_LENGTH_KEY)?;
    let heads = usize_key(FIRERED_PUNC_ATTENTION_HEAD_COUNT_KEY)?;
    let max_positions = usize_key(FIRERED_PUNC_CONTEXT_LENGTH_KEY)?;
    let vocab_size = usize_key(FIRERED_PUNC_VOCAB_SIZE_KEY)?;
    let label_count = usize_key(FIRERED_PUNC_LABEL_COUNT_KEY)?;

    for (key, value) in [
        (FIRERED_PUNC_BLOCK_COUNT_KEY, layers),
        (FIRERED_PUNC_EMBEDDING_LENGTH_KEY, d_model),
        (FIRERED_PUNC_FEED_FORWARD_LENGTH_KEY, ffn_dim),
        (FIRERED_PUNC_ATTENTION_HEAD_COUNT_KEY, heads),
        (FIRERED_PUNC_CONTEXT_LENGTH_KEY, max_positions),
        (FIRERED_PUNC_VOCAB_SIZE_KEY, vocab_size),
        (FIRERED_PUNC_LABEL_COUNT_KEY, label_count),
    ] {
        validate_positive_usize(value, key)?;
    }
    if !d_model.is_multiple_of(heads) {
        return Err(MetadataContractError::InvalidValue {
            key: FIRERED_PUNC_ATTENTION_HEAD_COUNT_KEY,
            reason: format!("head_count {heads} does not divide embedding_length {d_model}"),
        });
    }

    Ok(FireRedPuncExecutionMetadata {
        layers,
        d_model,
        ffn_dim,
        heads,
        head_dim: d_model / heads,
        vocab_size,
        max_positions,
        label_count,
    })
}

/// Parse and additionally assert the single published chinese-lert-base
/// geometry (see [`FireRedPuncExecutionMetadata::assert_expected_chinese_lert_base`]).
pub(crate) fn parse_and_validate_firered_punc_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<FireRedPuncExecutionMetadata, FireRedPuncConfigError> {
    let parsed = parse_firered_punc_execution_metadata(metadata).map_err(|error| match error {
        MetadataContractError::MissingRequiredKey { key } => {
            FireRedPuncConfigError::MissingMetadata(key.to_string())
        }
        MetadataContractError::InvalidValue { key, .. } => FireRedPuncConfigError::MetadataType {
            key: key.to_string(),
        },
    })?;
    parsed.assert_expected_chinese_lert_base()?;
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::firered_punc::config::{
        FIRERED_PUNC_EXPECTED_D_MODEL, FIRERED_PUNC_EXPECTED_FFN_DIM, FIRERED_PUNC_EXPECTED_HEADS,
        FIRERED_PUNC_EXPECTED_LAYERS, FIRERED_PUNC_EXPECTED_MAX_POSITIONS,
        FIRERED_PUNC_EXPECTED_VOCAB_SIZE, FIRERED_PUNC_LABEL_COUNT,
    };
    use std::collections::BTreeMap;

    fn base_metadata() -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        m.insert(
            GENERAL_ARCHITECTURE_KEY.to_string(),
            FIRERED_PUNC_ARCHITECTURE_VALUE.to_string(),
        );
        m.insert(
            FIRERED_PUNC_BLOCK_COUNT_KEY.to_string(),
            FIRERED_PUNC_EXPECTED_LAYERS.to_string(),
        );
        m.insert(
            FIRERED_PUNC_EMBEDDING_LENGTH_KEY.to_string(),
            FIRERED_PUNC_EXPECTED_D_MODEL.to_string(),
        );
        m.insert(
            FIRERED_PUNC_FEED_FORWARD_LENGTH_KEY.to_string(),
            FIRERED_PUNC_EXPECTED_FFN_DIM.to_string(),
        );
        m.insert(
            FIRERED_PUNC_ATTENTION_HEAD_COUNT_KEY.to_string(),
            FIRERED_PUNC_EXPECTED_HEADS.to_string(),
        );
        m.insert(
            FIRERED_PUNC_CONTEXT_LENGTH_KEY.to_string(),
            FIRERED_PUNC_EXPECTED_MAX_POSITIONS.to_string(),
        );
        m.insert(
            FIRERED_PUNC_VOCAB_SIZE_KEY.to_string(),
            FIRERED_PUNC_EXPECTED_VOCAB_SIZE.to_string(),
        );
        m.insert(
            FIRERED_PUNC_LABEL_COUNT_KEY.to_string(),
            FIRERED_PUNC_LABEL_COUNT.to_string(),
        );
        m
    }

    #[test]
    fn parses_chinese_lert_base_geometry() {
        let meta = base_metadata();
        assert!(metadata_declares_firered_punc(&meta));
        let parsed = parse_and_validate_firered_punc_metadata(&meta).expect("valid metadata");
        assert_eq!(parsed.layers, FIRERED_PUNC_EXPECTED_LAYERS);
        assert_eq!(parsed.d_model, FIRERED_PUNC_EXPECTED_D_MODEL);
        assert_eq!(
            parsed.head_dim,
            FIRERED_PUNC_EXPECTED_D_MODEL / FIRERED_PUNC_EXPECTED_HEADS
        );
        assert_eq!(parsed.label_count, FIRERED_PUNC_LABEL_COUNT);
    }

    #[test]
    fn wrong_architecture_is_rejected() {
        let mut meta = base_metadata();
        meta.insert(GENERAL_ARCHITECTURE_KEY.to_string(), "bert".to_string());
        assert!(!metadata_declares_firered_punc(&meta));
        assert!(parse_firered_punc_execution_metadata(&meta).is_err());
    }

    #[test]
    fn missing_key_fails_closed() {
        let mut meta = base_metadata();
        meta.remove(FIRERED_PUNC_BLOCK_COUNT_KEY);
        let err = parse_firered_punc_execution_metadata(&meta).unwrap_err();
        assert_eq!(
            err,
            MetadataContractError::MissingRequiredKey {
                key: FIRERED_PUNC_BLOCK_COUNT_KEY
            }
        );
    }

    #[test]
    fn off_geometry_pack_is_rejected() {
        let mut meta = base_metadata();
        meta.insert(FIRERED_PUNC_BLOCK_COUNT_KEY.to_string(), "6".to_string());
        // Parse succeeds structurally, but the expected-geometry assert rejects
        // a non chinese-lert-base layer count.
        assert!(parse_firered_punc_execution_metadata(&meta).is_ok());
        assert!(parse_and_validate_firered_punc_metadata(&meta).is_err());
    }

    #[test]
    fn indivisible_head_count_is_rejected() {
        let mut meta = base_metadata();
        meta.insert(
            FIRERED_PUNC_ATTENTION_HEAD_COUNT_KEY.to_string(),
            "5".to_string(),
        );
        assert!(parse_firered_punc_execution_metadata(&meta).is_err());
    }
}
