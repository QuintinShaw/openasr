//! Dolphin `small.cn` execution metadata parsed from the `.oasr` GGUF header.
//!
//! This is the required-metadata contract the install gate (`native.rs`)
//! dispatches to for the Dolphin architecture, so a pack missing a runtime
//! scalar fails closed at `openasr pull` install time rather than at first load.
//! The key set mirrors [`crate::arch::hparams::DOLPHIN_HPARAM_SCHEMA`], except
//! `dolphin.{encoder,decoder}.max_ctx`: those two are conditionally required --
//! see [`resolve_position_table_max_ctx`] -- because a pack's baked position
//! table tensor, when present, is authoritative over the scalar (this is what
//! lets the originally published `dolphin-cn-dialect-small` pack, which
//! predates the `max_ctx` metadata key, keep loading).

#![allow(dead_code)]

use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, optional_u64_scalar, required_u64_scalar,
    u64_to_u32, u64_to_usize, validate_positive_usize,
};

pub(crate) const DOLPHIN_ENCODER_N_LAYERS_KEY: &str = "dolphin.encoder.n_layers";
pub(crate) const DOLPHIN_ENCODER_D_MODEL_KEY: &str = "dolphin.encoder.d_model";
pub(crate) const DOLPHIN_ENCODER_N_HEADS_KEY: &str = "dolphin.encoder.n_heads";
pub(crate) const DOLPHIN_ENCODER_HEAD_DIM_KEY: &str = "dolphin.encoder.head_dim";
pub(crate) const DOLPHIN_ENCODER_FFN_DIM_KEY: &str = "dolphin.encoder.ffn_dim";
pub(crate) const DOLPHIN_ENCODER_CGMLP_UNITS_KEY: &str = "dolphin.encoder.cgmlp_units";
pub(crate) const DOLPHIN_ENCODER_CGMLP_KERNEL_KEY: &str = "dolphin.encoder.cgmlp_kernel";
pub(crate) const DOLPHIN_ENCODER_MERGE_KERNEL_KEY: &str = "dolphin.encoder.merge_kernel";
pub(crate) const DOLPHIN_ENCODER_FEATURE_DIM_KEY: &str = "dolphin.encoder.feature_dim";
pub(crate) const DOLPHIN_ENCODER_MAX_CTX_KEY: &str = "dolphin.encoder.max_ctx";
pub(crate) const DOLPHIN_DECODER_N_LAYERS_KEY: &str = "dolphin.decoder.n_layers";
pub(crate) const DOLPHIN_DECODER_N_HEADS_KEY: &str = "dolphin.decoder.n_heads";
pub(crate) const DOLPHIN_DECODER_FFN_DIM_KEY: &str = "dolphin.decoder.ffn_dim";
pub(crate) const DOLPHIN_DECODER_MAX_CTX_KEY: &str = "dolphin.decoder.max_ctx";
pub(crate) const DOLPHIN_VOCAB_SIZE_KEY: &str = "dolphin.vocab_size";
pub(crate) const DOLPHIN_SOS_TOKEN_ID_KEY: &str = "dolphin.sos_token_id";
pub(crate) const DOLPHIN_EOS_TOKEN_ID_KEY: &str = "dolphin.eos_token_id";
pub(crate) const DOLPHIN_CTC_BLANK_TOKEN_ID_KEY: &str = "ctc.blank_token_id";

/// Baked sinusoidal position-table tensor names (see
/// `package_import::sinusoidal_pos_table_max_ctx`). When a pack bakes one of
/// these, its shape's `max_ctx` dimension is authoritative over the
/// corresponding `dolphin.{encoder,decoder}.max_ctx` metadata scalar -- this is
/// what lets the originally published `dolphin-cn-dialect-small` pack (which
/// predates the `max_ctx` metadata key entirely) keep loading under the
/// generalized runtime contract.
pub(crate) const DOLPHIN_ENCODER_POS_TABLE_TENSOR: &str = "encoder.embed.pos_enc.pe";
pub(crate) const DOLPHIN_DECODER_POS_TABLE_TENSOR: &str = "decoder.embed.1.pe";

/// Source of a pack's baked position-table tensor sizes, abstracted so the
/// runtime contract can resolve `max_ctx` from either a `GgufTensorIndex`
/// (cheap shape-only probe, used at install-gate time before any weight is
/// loaded) or already-loaded [`DolphinRuntimeWeights`](super::executor::DolphinRuntimeWeights)
/// (the serving path, which has already paid to dequantize the tensor).
pub(crate) trait DolphinPositionTableSource {
    /// Total element count of the named tensor if the pack bakes it, else
    /// `None`.
    fn tensor_element_count(&self, name: &str) -> Option<usize>;
}

/// No baked table available (used by the runtime-contract unit tests below to
/// exercise the "metadata scalar is the only source" branch).
impl DolphinPositionTableSource for () {
    fn tensor_element_count(&self, _name: &str) -> Option<usize> {
        None
    }
}

impl DolphinPositionTableSource for crate::ggml_runtime::GgufTensorIndex {
    fn tensor_element_count(&self, name: &str) -> Option<usize> {
        self.get(name)
            .and_then(|tensor| tensor.num_elements())
            .map(|elements| elements as usize)
    }
}

/// Parsed, validated Dolphin runtime scalars (encoder + decoder + CTC head).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DolphinExecutionMetadata {
    pub encoder_n_layers: usize,
    pub encoder_d_model: usize,
    pub encoder_n_heads: usize,
    pub encoder_head_dim: usize,
    pub encoder_ffn_dim: usize,
    pub encoder_cgmlp_units: usize,
    pub encoder_cgmlp_kernel: usize,
    pub encoder_merge_kernel: usize,
    pub feature_dim: usize,
    /// Length of the sinusoidal position table baked into
    /// `encoder.embed.pos_enc.pe` (independent of the decoder's own
    /// `decoder.embed.1.pe` table; both happen to be 5000 on every checkpoint
    /// observed so far, but are tracked separately since nothing ties them).
    pub encoder_max_ctx: usize,
    pub decoder_n_layers: usize,
    pub decoder_n_heads: usize,
    pub decoder_ffn_dim: usize,
    pub decoder_max_ctx: usize,
    pub vocab_size: usize,
    pub sos_token_id: u32,
    pub eos_token_id: u32,
    pub blank_token_id: u32,
}

pub(crate) fn parse_dolphin_execution_metadata<M, P>(
    metadata: &M,
    position_tables: &P,
) -> Result<DolphinExecutionMetadata, MetadataContractError>
where
    M: ScalarMetadataView,
    P: DolphinPositionTableSource,
{
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let u32_key = |key: &'static str| -> Result<u32, MetadataContractError> {
        u64_to_u32(required_u64_scalar(metadata, key)?, key)
    };

    let encoder_n_layers = usize_key(DOLPHIN_ENCODER_N_LAYERS_KEY)?;
    let encoder_d_model = usize_key(DOLPHIN_ENCODER_D_MODEL_KEY)?;
    validate_positive_usize(encoder_d_model, DOLPHIN_ENCODER_D_MODEL_KEY)?;
    let encoder_n_heads = usize_key(DOLPHIN_ENCODER_N_HEADS_KEY)?;
    let encoder_head_dim = usize_key(DOLPHIN_ENCODER_HEAD_DIM_KEY)?;
    let encoder_ffn_dim = usize_key(DOLPHIN_ENCODER_FFN_DIM_KEY)?;
    let encoder_cgmlp_units = usize_key(DOLPHIN_ENCODER_CGMLP_UNITS_KEY)?;
    let encoder_cgmlp_kernel = usize_key(DOLPHIN_ENCODER_CGMLP_KERNEL_KEY)?;
    let encoder_merge_kernel = usize_key(DOLPHIN_ENCODER_MERGE_KERNEL_KEY)?;
    let feature_dim = usize_key(DOLPHIN_ENCODER_FEATURE_DIM_KEY)?;
    // `max_ctx` resolution: the baked position-table tensor (when present) is
    // authoritative over the metadata scalar; see
    // `resolve_position_table_max_ctx`. The decoder table shares the encoder's
    // `d_model` (decoder_graph.rs reuses `encoder_d_model` -- the architecture
    // never tracks a separate decoder width).
    let encoder_max_ctx = resolve_position_table_max_ctx(
        metadata,
        position_tables,
        DOLPHIN_ENCODER_POS_TABLE_TENSOR,
        DOLPHIN_ENCODER_MAX_CTX_KEY,
        encoder_d_model,
    )?;
    let decoder_n_layers = usize_key(DOLPHIN_DECODER_N_LAYERS_KEY)?;
    let decoder_n_heads = usize_key(DOLPHIN_DECODER_N_HEADS_KEY)?;
    let decoder_ffn_dim = usize_key(DOLPHIN_DECODER_FFN_DIM_KEY)?;
    let decoder_max_ctx = resolve_position_table_max_ctx(
        metadata,
        position_tables,
        DOLPHIN_DECODER_POS_TABLE_TENSOR,
        DOLPHIN_DECODER_MAX_CTX_KEY,
        encoder_d_model,
    )?;
    let vocab_size = usize_key(DOLPHIN_VOCAB_SIZE_KEY)?;
    let sos_token_id = u32_key(DOLPHIN_SOS_TOKEN_ID_KEY)?;
    let eos_token_id = u32_key(DOLPHIN_EOS_TOKEN_ID_KEY)?;
    let blank_token_id = u32_key(DOLPHIN_CTC_BLANK_TOKEN_ID_KEY)?;

    for (key, value) in [
        (DOLPHIN_ENCODER_N_LAYERS_KEY, encoder_n_layers),
        (DOLPHIN_ENCODER_D_MODEL_KEY, encoder_d_model),
        (DOLPHIN_ENCODER_N_HEADS_KEY, encoder_n_heads),
        (DOLPHIN_ENCODER_HEAD_DIM_KEY, encoder_head_dim),
        (DOLPHIN_ENCODER_FFN_DIM_KEY, encoder_ffn_dim),
        (DOLPHIN_ENCODER_CGMLP_UNITS_KEY, encoder_cgmlp_units),
        (DOLPHIN_ENCODER_CGMLP_KERNEL_KEY, encoder_cgmlp_kernel),
        (DOLPHIN_ENCODER_MERGE_KERNEL_KEY, encoder_merge_kernel),
        (DOLPHIN_ENCODER_FEATURE_DIM_KEY, feature_dim),
        (DOLPHIN_ENCODER_MAX_CTX_KEY, encoder_max_ctx),
        (DOLPHIN_DECODER_N_LAYERS_KEY, decoder_n_layers),
        (DOLPHIN_DECODER_N_HEADS_KEY, decoder_n_heads),
        (DOLPHIN_DECODER_FFN_DIM_KEY, decoder_ffn_dim),
        (DOLPHIN_DECODER_MAX_CTX_KEY, decoder_max_ctx),
        (DOLPHIN_VOCAB_SIZE_KEY, vocab_size),
    ] {
        validate_positive_usize(value, key)?;
    }

    if encoder_head_dim * encoder_n_heads != encoder_d_model {
        return Err(MetadataContractError::InvalidValue {
            key: DOLPHIN_ENCODER_HEAD_DIM_KEY,
            reason: format!(
                "head_dim {encoder_head_dim} * n_heads {encoder_n_heads} != d_model {encoder_d_model}"
            ),
        });
    }
    // The cgMLP channel-split gate halves `cgmlp_units`, so an odd value would
    // split unevenly.
    if !encoder_cgmlp_units.is_multiple_of(2) {
        return Err(MetadataContractError::InvalidValue {
            key: DOLPHIN_ENCODER_CGMLP_UNITS_KEY,
            reason: format!("cgmlp_units {encoder_cgmlp_units} must be even for the CSGU split"),
        });
    }
    for (key, value) in [
        (DOLPHIN_ENCODER_CGMLP_KERNEL_KEY, encoder_cgmlp_kernel),
        (DOLPHIN_ENCODER_MERGE_KERNEL_KEY, encoder_merge_kernel),
    ] {
        // Depthwise convs use symmetric `(k - 1) / 2` padding, which is only an
        // integer round-trip for an odd kernel.
        if value == 0 || value.is_multiple_of(2) {
            return Err(MetadataContractError::InvalidValue {
                key,
                reason: format!("depthwise conv kernel {value} must be odd for symmetric padding"),
            });
        }
    }
    for (label, token) in [
        (DOLPHIN_CTC_BLANK_TOKEN_ID_KEY, blank_token_id),
        (DOLPHIN_SOS_TOKEN_ID_KEY, sos_token_id),
        (DOLPHIN_EOS_TOKEN_ID_KEY, eos_token_id),
    ] {
        if (token as usize) >= vocab_size {
            return Err(MetadataContractError::InvalidValue {
                key: label,
                reason: format!("token id {token} out of range for vocab_size {vocab_size}"),
            });
        }
    }

    Ok(DolphinExecutionMetadata {
        encoder_n_layers,
        encoder_d_model,
        encoder_n_heads,
        encoder_head_dim,
        encoder_ffn_dim,
        encoder_cgmlp_units,
        encoder_cgmlp_kernel,
        encoder_merge_kernel,
        feature_dim,
        encoder_max_ctx,
        decoder_n_layers,
        decoder_n_heads,
        decoder_ffn_dim,
        decoder_max_ctx,
        vocab_size,
        sos_token_id,
        eos_token_id,
        blank_token_id,
    })
}

/// Resolve a `dolphin.{encoder,decoder}.max_ctx` value.
///
/// Priority order (fail-closed at every step):
/// 1. If `position_tables` reports the baked `tensor_name` tensor's element
///    count, that is authoritative -- divide by `d_model` to get `max_ctx`.
///    If a `metadata_key` scalar is *also* present, it must agree, else this
///    is a typed, clearly-worded error rather than a silent mismatch.
/// 2. Else (no baked table -- the ESPnet-synthesized-at-import-time path),
///    the `metadata_key` scalar is required.
/// 3. Neither present: fail closed with the missing-key error.
///
/// This is the compatibility seam for packs published before the `max_ctx`
/// metadata key existed (the originally shipped `dolphin-cn-dialect-small`):
/// their baked position table still carries the true length, so they resolve
/// via branch 1 without ever having written the scalar.
fn resolve_position_table_max_ctx<M, P>(
    metadata: &M,
    position_tables: &P,
    tensor_name: &'static str,
    metadata_key: &'static str,
    d_model: usize,
) -> Result<usize, MetadataContractError>
where
    M: ScalarMetadataView,
    P: DolphinPositionTableSource,
{
    let table_max_ctx = match position_tables.tensor_element_count(tensor_name) {
        Some(elements) => {
            if d_model == 0 || !elements.is_multiple_of(d_model) {
                return Err(MetadataContractError::InvalidValue {
                    key: metadata_key,
                    reason: format!(
                        "baked position table '{tensor_name}' has {elements} elements, not a multiple of d_model {d_model}"
                    ),
                });
            }
            Some(elements / d_model)
        }
        None => None,
    };
    let metadata_max_ctx = optional_u64_scalar(metadata, metadata_key)?
        .map(|value| u64_to_usize(value, metadata_key))
        .transpose()?;

    match (table_max_ctx, metadata_max_ctx) {
        (Some(table_max_ctx), Some(metadata_max_ctx)) if table_max_ctx != metadata_max_ctx => {
            Err(MetadataContractError::InvalidValue {
                key: metadata_key,
                reason: format!(
                    "baked position table '{tensor_name}' implies max_ctx {table_max_ctx}, \
                     metadata scalar says {metadata_max_ctx}"
                ),
            })
        }
        (Some(table_max_ctx), _) => Ok(table_max_ctx),
        (None, Some(metadata_max_ctx)) => Ok(metadata_max_ctx),
        (None, None) => Err(MetadataContractError::MissingRequiredKey { key: metadata_key }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::hparams::DOLPHIN_HPARAM_SCHEMA;
    use std::collections::BTreeMap;

    fn dolphin_metadata() -> BTreeMap<String, String> {
        [
            (DOLPHIN_ENCODER_N_LAYERS_KEY, "12"),
            (DOLPHIN_ENCODER_D_MODEL_KEY, "768"),
            (DOLPHIN_ENCODER_N_HEADS_KEY, "12"),
            (DOLPHIN_ENCODER_HEAD_DIM_KEY, "64"),
            (DOLPHIN_ENCODER_FFN_DIM_KEY, "3072"),
            (DOLPHIN_ENCODER_CGMLP_UNITS_KEY, "3072"),
            (DOLPHIN_ENCODER_CGMLP_KERNEL_KEY, "31"),
            (DOLPHIN_ENCODER_MERGE_KERNEL_KEY, "31"),
            (DOLPHIN_ENCODER_FEATURE_DIM_KEY, "80"),
            (DOLPHIN_ENCODER_MAX_CTX_KEY, "5000"),
            (DOLPHIN_DECODER_N_LAYERS_KEY, "12"),
            (DOLPHIN_DECODER_N_HEADS_KEY, "12"),
            (DOLPHIN_DECODER_FFN_DIM_KEY, "3072"),
            (DOLPHIN_DECODER_MAX_CTX_KEY, "5000"),
            (DOLPHIN_VOCAB_SIZE_KEY, "18173"),
            (DOLPHIN_SOS_TOKEN_ID_KEY, "2"),
            (DOLPHIN_EOS_TOKEN_ID_KEY, "3"),
            (DOLPHIN_CTC_BLANK_TOKEN_ID_KEY, "0"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    /// A fake baked position-table source for unit tests: reports a fixed
    /// element count for whichever tensor names are registered, `None`
    /// otherwise (modeling a pack that never baked that tensor).
    #[derive(Default)]
    struct FakePositionTables(BTreeMap<&'static str, usize>);

    impl FakePositionTables {
        fn with(tensor_name: &'static str, elements: usize) -> Self {
            let mut map = BTreeMap::new();
            map.insert(tensor_name, elements);
            Self(map)
        }
    }

    impl DolphinPositionTableSource for FakePositionTables {
        fn tensor_element_count(&self, name: &str) -> Option<usize> {
            self.0.get(name).copied()
        }
    }

    #[test]
    fn parses_dolphin_small_cn_metadata() {
        // No baked table (`()`): every value comes from the metadata scalars,
        // exercising the "ESPnet-synthesized, no baked tensor" branch.
        let parsed = parse_dolphin_execution_metadata(&dolphin_metadata(), &()).expect("parse");
        assert_eq!(parsed.encoder_n_layers, 12);
        assert_eq!(parsed.encoder_d_model, 768);
        assert_eq!(parsed.encoder_head_dim, 64);
        assert_eq!(parsed.decoder_n_layers, 12);
        assert_eq!(parsed.encoder_max_ctx, 5000);
        assert_eq!(parsed.decoder_max_ctx, 5000);
        assert_eq!(parsed.vocab_size, 18173);
        assert_eq!(parsed.sos_token_id, 2);
        assert_eq!(parsed.eos_token_id, 3);
        assert_eq!(parsed.blank_token_id, 0);
    }

    /// The compatibility case this fix exists for: a pack (like the
    /// originally published `dolphin-cn-dialect-small`) that never wrote
    /// `dolphin.encoder.max_ctx` / `dolphin.decoder.max_ctx` at all, but does
    /// bake the sinusoidal position table -- the tensor's own shape resolves
    /// `max_ctx` instead of fail-closing on the missing scalar.
    #[test]
    fn resolves_max_ctx_from_baked_table_when_metadata_key_is_absent() {
        let mut metadata = dolphin_metadata();
        metadata.remove(DOLPHIN_ENCODER_MAX_CTX_KEY);
        metadata.remove(DOLPHIN_DECODER_MAX_CTX_KEY);
        let mut tables = FakePositionTables::with(DOLPHIN_ENCODER_POS_TABLE_TENSOR, 5000 * 768);
        tables
            .0
            .insert(DOLPHIN_DECODER_POS_TABLE_TENSOR, 4096 * 768);
        let parsed = parse_dolphin_execution_metadata(&metadata, &tables).expect("parse");
        assert_eq!(parsed.encoder_max_ctx, 5000);
        assert_eq!(parsed.decoder_max_ctx, 4096);
    }

    /// A checkpoint whose export path never bakes the table (ESPnet
    /// multilingual): no baked tensor and no metadata scalar must fail
    /// closed, not silently default.
    #[test]
    fn rejects_missing_max_ctx_when_neither_table_nor_metadata_key_present() {
        let mut metadata = dolphin_metadata();
        metadata.remove(DOLPHIN_ENCODER_MAX_CTX_KEY);
        assert!(matches!(
            parse_dolphin_execution_metadata(&metadata, &()),
            Err(MetadataContractError::MissingRequiredKey {
                key: DOLPHIN_ENCODER_MAX_CTX_KEY
            })
        ));
    }

    /// A baked table and a present-but-disagreeing metadata scalar must fail
    /// closed with a typed, specific error rather than silently trusting
    /// either side.
    #[test]
    fn rejects_baked_table_metadata_disagreement() {
        let metadata = dolphin_metadata(); // DOLPHIN_ENCODER_MAX_CTX_KEY = "5000"
        let tables = FakePositionTables::with(DOLPHIN_ENCODER_POS_TABLE_TENSOR, 4096 * 768);
        assert!(matches!(
            parse_dolphin_execution_metadata(&metadata, &tables),
            Err(MetadataContractError::InvalidValue {
                key: DOLPHIN_ENCODER_MAX_CTX_KEY,
                ..
            })
        ));
    }

    #[test]
    fn rejects_inconsistent_head_dim() {
        let mut metadata = dolphin_metadata();
        metadata.insert(DOLPHIN_ENCODER_HEAD_DIM_KEY.to_string(), "100".to_string());
        assert!(parse_dolphin_execution_metadata(&metadata, &()).is_err());
    }

    #[test]
    fn rejects_blank_out_of_vocab() {
        let mut metadata = dolphin_metadata();
        metadata.insert(DOLPHIN_VOCAB_SIZE_KEY.to_string(), "2".to_string());
        assert!(parse_dolphin_execution_metadata(&metadata, &()).is_err());
    }

    #[test]
    fn rejects_missing_required_key() {
        let mut metadata = dolphin_metadata();
        metadata.remove(DOLPHIN_DECODER_N_LAYERS_KEY);
        assert!(matches!(
            parse_dolphin_execution_metadata(&metadata, &()),
            Err(MetadataContractError::MissingRequiredKey {
                key: DOLPHIN_DECODER_N_LAYERS_KEY
            })
        ));
    }

    /// The runtime contract's required scalar keys must be exactly the arch
    /// hparam schema (drift here would let a pack pass install but miss a key the
    /// executor needs).
    #[test]
    fn required_keys_match_arch_hparam_schema() {
        let mut contract_keys = [
            DOLPHIN_ENCODER_N_LAYERS_KEY,
            DOLPHIN_ENCODER_D_MODEL_KEY,
            DOLPHIN_ENCODER_N_HEADS_KEY,
            DOLPHIN_ENCODER_HEAD_DIM_KEY,
            DOLPHIN_ENCODER_FFN_DIM_KEY,
            DOLPHIN_ENCODER_CGMLP_UNITS_KEY,
            DOLPHIN_ENCODER_CGMLP_KERNEL_KEY,
            DOLPHIN_ENCODER_MERGE_KERNEL_KEY,
            DOLPHIN_ENCODER_FEATURE_DIM_KEY,
            DOLPHIN_ENCODER_MAX_CTX_KEY,
            DOLPHIN_DECODER_N_LAYERS_KEY,
            DOLPHIN_DECODER_N_HEADS_KEY,
            DOLPHIN_DECODER_FFN_DIM_KEY,
            DOLPHIN_DECODER_MAX_CTX_KEY,
            DOLPHIN_VOCAB_SIZE_KEY,
            DOLPHIN_SOS_TOKEN_ID_KEY,
            DOLPHIN_EOS_TOKEN_ID_KEY,
            DOLPHIN_CTC_BLANK_TOKEN_ID_KEY,
        ]
        .to_vec();
        contract_keys.sort_unstable();
        let mut schema_keys = DOLPHIN_HPARAM_SCHEMA.to_vec();
        schema_keys.sort_unstable();
        assert_eq!(contract_keys, schema_keys);
    }
}
