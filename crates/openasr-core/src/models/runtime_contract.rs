use std::collections::BTreeMap;
use std::sync::Arc;

use thiserror::Error;

use crate::GgufMetadata;

pub(crate) trait ScalarMetadataView {
    fn get_string_scalar(&self, key: &str) -> Option<&str>;
    fn get_u64_scalar(&self, key: &str) -> Option<u64>;
}

impl ScalarMetadataView for GgufMetadata {
    fn get_string_scalar(&self, key: &str) -> Option<&str> {
        self.get_string(key)
    }

    fn get_u64_scalar(&self, key: &str) -> Option<u64> {
        self.get_u64(key)
            .or_else(|| self.get_u32(key).map(u64::from))
    }
}

/// Forwarding impl so call sites can pass `&Arc<GgufMetadata>` (as stored on
/// `GgmlAsrRuntimeSourcePreflight::metadata`) directly, without an explicit
/// deref at every one of the ~25 read sites across model families.
impl<T: ScalarMetadataView + ?Sized> ScalarMetadataView for Arc<T> {
    fn get_string_scalar(&self, key: &str) -> Option<&str> {
        T::get_string_scalar(self, key)
    }

    fn get_u64_scalar(&self, key: &str) -> Option<u64> {
        T::get_u64_scalar(self, key)
    }
}

impl ScalarMetadataView for BTreeMap<String, String> {
    fn get_string_scalar(&self, key: &str) -> Option<&str> {
        self.get(key).map(String::as_str)
    }

    fn get_u64_scalar(&self, _key: &str) -> Option<u64> {
        None
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum MetadataContractError {
    #[error("missing required metadata key '{key}'")]
    MissingRequiredKey { key: &'static str },
    #[error("metadata '{key}' is invalid: {reason}")]
    InvalidValue { key: &'static str, reason: String },
}

pub(crate) fn required_string_scalar<'a, M>(
    metadata: &'a M,
    key: &'static str,
) -> Result<&'a str, MetadataContractError>
where
    M: ScalarMetadataView,
{
    let Some(value) = metadata.get_string_scalar(key) else {
        return Err(MetadataContractError::MissingRequiredKey { key });
    };
    let normalized = value.trim();
    if normalized.is_empty() {
        return Err(MetadataContractError::InvalidValue {
            key,
            reason: "value must be non-empty".to_string(),
        });
    }
    Ok(normalized)
}

pub(crate) fn required_u64_scalar<M>(
    metadata: &M,
    key: &'static str,
) -> Result<u64, MetadataContractError>
where
    M: ScalarMetadataView,
{
    if let Some(value) = metadata.get_u64_scalar(key) {
        return Ok(value);
    }
    if let Some(value) = metadata.get_string_scalar(key) {
        let trimmed = value.trim();
        let parsed =
            trimmed
                .parse::<u64>()
                .map_err(|error| MetadataContractError::InvalidValue {
                    key,
                    reason: format!("cannot parse '{trimmed}' as u64: {error}"),
                })?;
        return Ok(parsed);
    }
    Err(MetadataContractError::MissingRequiredKey { key })
}

pub(crate) fn optional_u64_scalar<M>(
    metadata: &M,
    key: &'static str,
) -> Result<Option<u64>, MetadataContractError>
where
    M: ScalarMetadataView,
{
    if let Some(value) = metadata.get_u64_scalar(key) {
        return Ok(Some(value));
    }
    let Some(value) = metadata.get_string_scalar(key) else {
        return Ok(None);
    };
    let trimmed = value.trim();
    let parsed = trimmed
        .parse::<u64>()
        .map_err(|error| MetadataContractError::InvalidValue {
            key,
            reason: format!("cannot parse '{trimmed}' as u64: {error}"),
        })?;
    Ok(Some(parsed))
}

pub(crate) fn u64_to_usize(value: u64, key: &'static str) -> Result<usize, MetadataContractError> {
    usize::try_from(value).map_err(|_| MetadataContractError::InvalidValue {
        key,
        reason: "value does not fit target usize".to_string(),
    })
}

pub(crate) fn u64_to_u32(value: u64, key: &'static str) -> Result<u32, MetadataContractError> {
    u32::try_from(value).map_err(|_| MetadataContractError::InvalidValue {
        key,
        reason: "value does not fit target u32".to_string(),
    })
}

pub(crate) fn validate_positive_usize(
    value: usize,
    key: &'static str,
) -> Result<(), MetadataContractError> {
    if value > 0 {
        return Ok(());
    }
    Err(MetadataContractError::InvalidValue {
        key,
        reason: "value must be greater than 0".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{GgufMetadata, GgufMetadataValue};

    use super::*;

    #[test]
    fn required_u64_scalar_reads_native_gguf_u64() {
        let mut values = BTreeMap::new();
        values.insert("n".to_string(), GgufMetadataValue::U64(28));
        let metadata = GgufMetadata::from_values_for_test(values);
        let value = required_u64_scalar(&metadata, "n").expect("must parse");
        assert_eq!(value, 28);
    }

    #[test]
    fn required_u64_scalar_reads_native_gguf_u32() {
        let mut values = BTreeMap::new();
        values.insert("n".to_string(), GgufMetadataValue::U32(28));
        let metadata = GgufMetadata::from_values_for_test(values);
        let value = required_u64_scalar(&metadata, "n").expect("must parse");
        assert_eq!(value, 28);
    }
}
