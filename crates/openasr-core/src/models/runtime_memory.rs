//! Count-only helpers for Rust-owned model runtime memory.
//!
//! These functions mirror the capacity of the containers that family
//! materializers actually build. They deliberately do not price ggml/native
//! allocations: those are quoted by the selected backend and admitted in the
//! corresponding physical memory domain.

use crate::ggml_runtime::{GgufMetadata, GgufTensorIndex};

use super::system_memory_owner::SystemMemoryOwnerError;

/// Checked count-only simulation of a materializer's Rust-owned heap lifetime.
///
/// `stable_bytes` is the heap that survives the most recently completed build
/// step. `peak_bytes` is the largest simultaneously-live heap observed so far.
/// A caller may therefore model a transient dequantization batch without
/// pretending that transient payloads from different layers coexist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ConstructionMemoryPlan {
    family: &'static str,
    stable_bytes: u64,
    peak_bytes: u64,
}

impl ConstructionMemoryPlan {
    pub(crate) const fn new(family: &'static str) -> Self {
        Self {
            family,
            stable_bytes: 0,
            peak_bytes: 0,
        }
    }

    /// Add bytes that become resident directly and survive later build steps.
    pub(crate) fn retain(&mut self, bytes: u64, label: &str) -> Result<(), SystemMemoryOwnerError> {
        self.materialize_then_retain(bytes, bytes, label)
    }

    /// Model one atomic materialization batch.
    ///
    /// `materialized_bytes` is the whole batch while its transient payloads are
    /// live. `retained_bytes` is what remains after that same batch completes.
    /// Neither value includes the stable heap from prior steps.
    pub(crate) fn materialize_then_retain(
        &mut self,
        materialized_bytes: u64,
        retained_bytes: u64,
        label: &str,
    ) -> Result<(), SystemMemoryOwnerError> {
        if retained_bytes > materialized_bytes {
            return Err(capacity_error(
                self.family,
                format!(
                    "{label} retains {retained_bytes} bytes after materializing only {materialized_bytes}"
                ),
            ));
        }
        let materialized_peak = self
            .stable_bytes
            .checked_add(materialized_bytes)
            .ok_or_else(|| capacity_error(self.family, format!("{label} peak overflowed")))?;
        self.peak_bytes = self.peak_bytes.max(materialized_peak);
        self.stable_bytes = self
            .stable_bytes
            .checked_add(retained_bytes)
            .ok_or_else(|| capacity_error(self.family, format!("{label} retained overflowed")))?;
        Ok(())
    }

    pub(crate) const fn stable_bytes(self) -> u64 {
        self.stable_bytes
    }

    pub(crate) const fn peak_bytes(self) -> u64 {
        self.peak_bytes
    }
}

pub(crate) fn tokenizer_btree_quote_bytes(
    metadata: &GgufMetadata,
    family: &str,
) -> Result<u64, SystemMemoryOwnerError> {
    let tokens = metadata
        .get_string_array("tokenizer.ggml.tokens")
        .ok_or_else(|| capacity_error(family, "tokenizer metadata is missing"))?;
    let token_count = tokens.len();
    let text_bytes = tokens.iter().try_fold(0_u64, |total, token| {
        u64::try_from(token.len())
            .ok()
            .and_then(|bytes| total.checked_add(bytes))
    });
    let Some(text_bytes) = text_bytes else {
        return Err(capacity_error(
            family,
            "tokenizer text byte count overflowed",
        ));
    };
    checked_sum(
        [
            element_bytes::<String>(token_count, family, "tokenizer token table")?,
            element_bytes::<(String, u32)>(token_count, family, "tokenizer reverse map")?,
            text_bytes
                .checked_mul(2)
                .ok_or_else(|| capacity_error(family, "tokenizer text bytes overflowed"))?,
        ],
        family,
        "tokenizer retained bytes",
    )
}

pub(crate) fn named_f32_tensor_quote_bytes(
    tensor_index: &GgufTensorIndex,
    name: &str,
    retain_values: bool,
    family: &str,
) -> Result<u64, SystemMemoryOwnerError> {
    let tensor = tensor_index
        .get(name)
        .ok_or_else(|| capacity_error(family, format!("required tensor '{name}' is missing")))?;
    let mut values = vec![
        u64::try_from(name.len())
            .map_err(|_| capacity_error(family, format!("tensor '{name}' name overflowed")))?,
        element_bytes::<usize>(tensor.dims.len(), family, "tensor dims")?,
    ];
    if retain_values {
        values.push(tensor_f32_bytes(tensor_index, name, family)?);
    }
    checked_sum(values, family, "named tensor bytes")
}

pub(crate) fn tensor_f32_bytes(
    tensor_index: &GgufTensorIndex,
    name: &str,
    family: &str,
) -> Result<u64, SystemMemoryOwnerError> {
    let tensor = tensor_index
        .get(name)
        .ok_or_else(|| capacity_error(family, format!("required tensor '{name}' is missing")))?;
    let elements = tensor
        .num_elements()
        .ok_or_else(|| capacity_error(family, format!("tensor '{name}' element overflowed")))?;
    elements
        .checked_mul(4)
        .ok_or_else(|| capacity_error(family, format!("tensor '{name}' f32 byte count overflowed")))
}

pub(crate) fn element_bytes<T>(
    count: usize,
    family: &str,
    label: &str,
) -> Result<u64, SystemMemoryOwnerError> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| capacity_error(family, format!("{label} byte count overflowed")))?;
    u64::try_from(bytes).map_err(|_| capacity_error(family, format!("{label} does not fit u64")))
}

pub(crate) fn checked_sum(
    values: impl IntoIterator<Item = u64>,
    family: &str,
    label: &str,
) -> Result<u64, SystemMemoryOwnerError> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(|| capacity_error(family, format!("{label} overflowed")))
    })
}

fn capacity_error(family: &str, reason: impl Into<String>) -> SystemMemoryOwnerError {
    SystemMemoryOwnerError::capacity_failure(
        "model_runtime_memory",
        format!("{family}: {}", reason.into()),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::ggml_runtime::{GgufMetadata, GgufMetadataValue};

    use super::*;

    #[test]
    fn tokenizer_quote_matches_declared_container_capacities() {
        let tokens = vec!["a".to_string(), "bc".to_string(), "def".to_string()];
        let text_bytes = tokens.iter().map(String::len).sum::<usize>() as u64;
        let mut values = BTreeMap::new();
        values.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetadataValue::StringArray(tokens),
        );
        let metadata = GgufMetadata::from_values_for_test(values);

        let expected = (3 * std::mem::size_of::<String>()) as u64
            + (3 * std::mem::size_of::<(String, u32)>()) as u64
            + (2 * text_bytes);
        assert_eq!(
            tokenizer_btree_quote_bytes(&metadata, "test").expect("quote"),
            expected
        );
    }

    #[test]
    fn tokenizer_quote_fails_closed_when_metadata_is_absent() {
        let error = tokenizer_btree_quote_bytes(&GgufMetadata::default(), "test")
            .expect_err("missing tokenizer metadata must fail");
        assert!(error.to_string().contains("tokenizer metadata is missing"));
    }

    #[test]
    fn checked_sum_and_element_bytes_reject_overflow() {
        assert!(checked_sum([u64::MAX, 1], "test", "sum").is_err());
        assert!(element_bytes::<u64>(usize::MAX, "test", "elements").is_err());
    }

    #[test]
    fn construction_plan_keeps_transient_batches_disjoint() {
        let mut plan = ConstructionMemoryPlan::new("test");
        plan.retain(10, "descriptor storage").expect("retain");
        plan.materialize_then_retain(100, 20, "layer zero")
            .expect("layer zero");
        plan.materialize_then_retain(80, 30, "layer one")
            .expect("layer one");

        assert_eq!(plan.stable_bytes(), 60);
        assert_eq!(plan.peak_bytes(), 110);
    }

    #[test]
    fn construction_plan_rejects_impossible_or_overflowing_lifetimes() {
        let mut impossible = ConstructionMemoryPlan::new("test");
        assert!(
            impossible
                .materialize_then_retain(4, 5, "impossible")
                .is_err()
        );

        let mut overflow = ConstructionMemoryPlan::new("test");
        overflow.retain(u64::MAX, "all memory").expect("first step");
        assert!(overflow.retain(1, "overflow").is_err());
    }
}
