//! Dolphin `small.cn` execution metadata parsed from the `.oasr` GGUF header,
//! plus the admission-time runtime tensor contract that proves the pack carries
//! every tensor the runtime will load (metadata-derived element counts checked
//! against the tensor index) before the pack is admitted.
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
//!
//! Tensor-contract note: Dolphin stores rank-2 `.weight` matrices with reversed
//! dims when block-quantized (the contiguous `in` axis lands on ne0; see
//! `package_import`), and the runtime re-declares every graph tensor and consumes
//! each weight only by element count. The admission contract therefore checks
//! element counts -- not stored dims -- which is invariant to the
//! quantized-orientation reversal and matches exactly what the loader enforces.

#![allow(dead_code)]

use thiserror::Error;

use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, optional_u64_scalar, required_u64_scalar,
    u64_to_u32, u64_to_usize, validate_positive_usize,
};
use crate::{GgufTensorIndex, GgufTensorMetadata};

use super::encoder_graph::subsample_width;
use super::package_import::DolphinLanguageScheme;

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

/// Selects the decode-prefix builder, audio frontend, and encoder rel-pos
/// attention scheme (see `executor::parse_dolphin_language_scheme`). Absent on
/// packs predating the key, which default to the cn-dialect scheme; a present
/// but unrecognized value fails closed. Validated here at admission so a corrupt
/// or future-versioned pack is rejected before any decode work.
pub(crate) const DOLPHIN_LANGUAGE_SCHEME_KEY: &str = "dolphin.language.scheme";
/// The char/SentencePiece vocab, stamped by the importer as a string array. Its
/// length must agree with `dolphin.vocab_size`; the runtime detokenizes through
/// it and fails on a missing entry.
pub(crate) const DOLPHIN_TOKENIZER_TOKENS_KEY: &str = "tokenizer.ggml.tokens";

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

/// Parse the pack's `dolphin.language.scheme` metadata into the typed
/// [`DolphinLanguageScheme`]. A **missing** key is an intentional
/// backward-compat default to `CnDialect` (every pack baked before the key
/// existed is cn-dialect); a **present** but unrecognized value fails closed so
/// a corrupt or future-versioned pack is never silently misdispatched to the
/// wrong frontend/attention scheme. This is the single language-scheme parser
/// shared by the admission validator and the executor.
pub(crate) fn parse_dolphin_language_scheme_value(
    value: Option<&str>,
) -> Result<DolphinLanguageScheme, DolphinTensorContractError> {
    match value {
        None => Ok(DolphinLanguageScheme::CnDialect),
        Some("cn_dialect") => Ok(DolphinLanguageScheme::CnDialect),
        Some("multilingual") => Ok(DolphinLanguageScheme::Multilingual),
        Some(other) => Err(DolphinTensorContractError::LanguageScheme {
            reason: format!(
                "unrecognized '{DOLPHIN_LANGUAGE_SCHEME_KEY}' value {other:?} \
                 (expected 'cn_dialect' or 'multilingual')"
            ),
        }),
    }
}

/// Admission-time tensor-contract errors for the Dolphin runtime set.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum DolphinTensorContractError {
    #[error("dolphin missing required runtime tensor '{name}'")]
    MissingTensor { name: String },
    #[error("dolphin runtime tensor '{name}' has {actual} elements, expected {expected}")]
    TensorElementCount {
        name: String,
        expected: u64,
        actual: u64,
    },
    #[error("dolphin language scheme metadata is invalid: {reason}")]
    LanguageScheme { reason: String },
    #[error("dolphin pack is missing the '{DOLPHIN_TOKENIZER_TOKENS_KEY}' vocab")]
    MissingTokenizer,
    #[error(
        "dolphin '{DOLPHIN_TOKENIZER_TOKENS_KEY}' has {actual} tokens, expected {expected} (vocab_size)"
    )]
    TokenizerVocabSize { expected: usize, actual: usize },
}

/// The required Dolphin runtime tensor set as `(name, expected_element_count)`,
/// derived entirely from the parsed execution metadata and the language scheme.
/// Element counts (not stored dims) are the contract: Dolphin reverses rank-2
/// weight dims when block-quantizing and the runtime consumes every weight by
/// element count, so this enumeration is invariant to the stored orientation and
/// is the single source of truth shared by the admission validator and the
/// runtime-ready test fixture. The optional hotword `context_module.*` tensors
/// are deliberately absent -- packs without a trained context module are valid
/// and simply do not advertise phrase bias.
pub(crate) fn dolphin_runtime_tensor_element_counts(
    metadata: &DolphinExecutionMetadata,
    language_scheme: DolphinLanguageScheme,
) -> Vec<(String, u64)> {
    let d = metadata.encoder_d_model as u64;
    let enc_ffn = metadata.encoder_ffn_dim as u64;
    let cg = metadata.encoder_cgmlp_units as u64;
    let cg_half = cg / 2;
    let ck = metadata.encoder_cgmlp_kernel as u64;
    let mk = metadata.encoder_merge_kernel as u64;
    let feature_dim = metadata.feature_dim as u64;
    let vocab = metadata.vocab_size as u64;
    let dec_ffn = metadata.decoder_ffn_dim as u64;
    // The decoder reuses the encoder width (the importer fails closed on a
    // disagreement), so the decoder model dim is `d` as well.
    let flat = d * subsample_width(metadata.feature_dim) as u64;

    let mut out: Vec<(String, u64)> = Vec::new();
    let mut push = |name: &str, elements: u64| out.push((name.to_string(), elements));

    // Encoder subsampling embed (Conv2d x2 + linear reshape).
    push("encoder.embed.conv.0.weight", 3 * 3 * d);
    push("encoder.embed.conv.0.bias", d);
    push("encoder.embed.conv.2.weight", 3 * 3 * d * d);
    push("encoder.embed.conv.2.bias", d);
    push("encoder.embed.out.0.weight", flat * d);
    push("encoder.embed.out.0.bias", d);

    // E-Branchformer encoder blocks.
    for index in 0..metadata.encoder_n_layers {
        let p = |suffix: &str| format!("encoder.encoders.{index}.{suffix}");
        push(&p("norm_ff_macaron.weight"), d);
        push(&p("norm_ff_macaron.bias"), d);
        push(&p("feed_forward_macaron.w_1.weight"), d * enc_ffn);
        push(&p("feed_forward_macaron.w_1.bias"), enc_ffn);
        push(&p("feed_forward_macaron.w_2.weight"), enc_ffn * d);
        push(&p("feed_forward_macaron.w_2.bias"), d);
        push(&p("norm_mha.weight"), d);
        push(&p("norm_mha.bias"), d);
        for proj in ["linear_q", "linear_k", "linear_v", "linear_out"] {
            push(&p(&format!("attn.{proj}.weight")), d * d);
            push(&p(&format!("attn.{proj}.bias")), d);
        }
        push(&p("attn.linear_pos.weight"), d * d);
        push(&p("attn.pos_bias_u"), d);
        push(&p("attn.pos_bias_v"), d);
        push(&p("norm_mlp.weight"), d);
        push(&p("norm_mlp.bias"), d);
        push(&p("cgmlp.channel_proj1.0.weight"), d * cg);
        push(&p("cgmlp.channel_proj1.0.bias"), cg);
        push(&p("cgmlp.csgu.norm.weight"), cg_half);
        push(&p("cgmlp.csgu.norm.bias"), cg_half);
        push(&p("cgmlp.csgu.conv.weight"), ck * cg_half);
        push(&p("cgmlp.csgu.conv.bias"), cg_half);
        push(&p("cgmlp.channel_proj2.weight"), cg_half * d);
        push(&p("cgmlp.channel_proj2.bias"), d);
        push(&p("depthwise_conv_fusion.weight"), mk * 2 * d);
        push(&p("depthwise_conv_fusion.bias"), 2 * d);
        push(&p("merge_proj.weight"), 2 * d * d);
        push(&p("merge_proj.bias"), d);
        push(&p("norm_ff.weight"), d);
        push(&p("norm_ff.bias"), d);
        push(&p("feed_forward.w_1.weight"), d * enc_ffn);
        push(&p("feed_forward.w_1.bias"), enc_ffn);
        push(&p("feed_forward.w_2.weight"), enc_ffn * d);
        push(&p("feed_forward.w_2.bias"), d);
        push(&p("norm_final.weight"), d);
        push(&p("norm_final.bias"), d);
    }

    // Encoder tail + global CMVN.
    push("encoder.after_norm.weight", d);
    push("encoder.after_norm.bias", d);
    push("encoder.global_cmvn.mean", feature_dim);
    push("encoder.global_cmvn.istd", feature_dim);

    // The cn-dialect encoder attention consumes the baked sinusoidal table; the
    // multilingual scheme computes a centered table per request instead, so its
    // packs legitimately omit this tensor (mirrors the executor's sentinel set).
    if language_scheme == DolphinLanguageScheme::CnDialect {
        push(
            "encoder.embed.pos_enc.pe",
            d * metadata.encoder_max_ctx as u64,
        );
    }

    // CTC head.
    push("ctc.ctc_lo.weight", vocab * d);
    push("ctc.ctc_lo.bias", vocab);

    // Transformer rescore decoder.
    push("decoder.embed.0.weight", d * vocab);
    push("decoder.embed.1.pe", d * metadata.decoder_max_ctx as u64);
    for index in 0..metadata.decoder_n_layers {
        let p = |suffix: &str| format!("decoder.decoders.{index}.{suffix}");
        for norm in ["norm1", "norm2", "norm3"] {
            push(&p(&format!("{norm}.weight")), d);
            push(&p(&format!("{norm}.bias")), d);
        }
        for attn in ["self_attn", "src_attn"] {
            for proj in ["linear_q", "linear_k", "linear_v", "linear_out"] {
                push(&p(&format!("{attn}.{proj}.weight")), d * d);
                push(&p(&format!("{attn}.{proj}.bias")), d);
            }
        }
        push(&p("feed_forward.w_1.weight"), d * dec_ffn);
        push(&p("feed_forward.w_1.bias"), dec_ffn);
        push(&p("feed_forward.w_2.weight"), dec_ffn * d);
        push(&p("feed_forward.w_2.bias"), d);
    }
    push("decoder.after_norm.weight", d);
    push("decoder.after_norm.bias", d);
    push("decoder.output_layer.weight", d * vocab);
    push("decoder.output_layer.bias", vocab);

    out
}

/// Validate the pack's runtime tensor set against the tensor index: every tensor
/// the runtime loads must be present with a metadata-consistent element count.
/// A truncated, reshaped, or mis-converted pack fails closed at admission with
/// the offending tensor named, instead of passing a metadata-only check and
/// failing later inside the executor.
pub(crate) fn validate_dolphin_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    metadata: &DolphinExecutionMetadata,
    language_scheme: DolphinLanguageScheme,
) -> Result<(), DolphinTensorContractError> {
    for (name, expected) in dolphin_runtime_tensor_element_counts(metadata, language_scheme) {
        let tensor: &GgufTensorMetadata = index
            .get(&name)
            .ok_or_else(|| DolphinTensorContractError::MissingTensor { name: name.clone() })?;
        let actual = tensor.num_elements().ok_or_else(|| {
            DolphinTensorContractError::TensorElementCount {
                name: name.clone(),
                expected,
                actual: 0,
            }
        })?;
        if actual != expected {
            return Err(DolphinTensorContractError::TensorElementCount {
                name,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_runtime_pack_contract(
    preflight: &crate::GgufRuntimeSourcePreflight,
) -> Result<(), String> {
    let metadata = parse_dolphin_execution_metadata(preflight.metadata(), preflight.tensor_index())
        .map_err(|error| {
            crate::models::runtime_pack_contract::metadata_validation_error("dolphin", error)
        })?;
    // The language scheme selects the frontend, attention scheme, and the
    // required tensor set; fail closed on an unrecognized value before any
    // tensor or decode work.
    let language_scheme = parse_dolphin_language_scheme_value(
        preflight.metadata().get_string(DOLPHIN_LANGUAGE_SCHEME_KEY),
    )
    .map_err(|error| {
        crate::models::runtime_pack_contract::metadata_validation_error("dolphin", error)
    })?;
    // The runtime detokenizes through the stamped vocab; its length must agree
    // with the metadata vocab_size.
    let tokens = preflight
        .metadata()
        .get_string_array(DOLPHIN_TOKENIZER_TOKENS_KEY)
        .ok_or_else(|| {
            crate::models::runtime_pack_contract::tensor_validation_error(
                DolphinTensorContractError::MissingTokenizer,
            )
        })?;
    if tokens.len() != metadata.vocab_size {
        return Err(
            crate::models::runtime_pack_contract::tensor_validation_error(
                DolphinTensorContractError::TokenizerVocabSize {
                    expected: metadata.vocab_size,
                    actual: tokens.len(),
                },
            ),
        );
    }
    validate_dolphin_runtime_tensors_with_index(
        preflight.tensor_index(),
        &metadata,
        language_scheme,
    )
    .map_err(crate::models::runtime_pack_contract::tensor_validation_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::hparams::DOLPHIN_HPARAM_SCHEMA;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

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

    /// Small internally-consistent metadata for the tensor-contract tests (the
    /// same geometry the runtime-ready fixture stamps).
    fn small_dolphin_metadata() -> BTreeMap<String, String> {
        [
            (DOLPHIN_ENCODER_N_LAYERS_KEY, "1"),
            (DOLPHIN_ENCODER_D_MODEL_KEY, "8"),
            (DOLPHIN_ENCODER_N_HEADS_KEY, "2"),
            (DOLPHIN_ENCODER_HEAD_DIM_KEY, "4"),
            (DOLPHIN_ENCODER_FFN_DIM_KEY, "16"),
            (DOLPHIN_ENCODER_CGMLP_UNITS_KEY, "16"),
            (DOLPHIN_ENCODER_CGMLP_KERNEL_KEY, "3"),
            (DOLPHIN_ENCODER_MERGE_KERNEL_KEY, "3"),
            (DOLPHIN_ENCODER_FEATURE_DIM_KEY, "16"),
            (DOLPHIN_ENCODER_MAX_CTX_KEY, "8"),
            (DOLPHIN_DECODER_N_LAYERS_KEY, "1"),
            (DOLPHIN_DECODER_N_HEADS_KEY, "2"),
            (DOLPHIN_DECODER_FFN_DIM_KEY, "16"),
            (DOLPHIN_DECODER_MAX_CTX_KEY, "8"),
            (DOLPHIN_VOCAB_SIZE_KEY, "12"),
            (DOLPHIN_SOS_TOKEN_ID_KEY, "2"),
            (DOLPHIN_EOS_TOKEN_ID_KEY, "3"),
            (DOLPHIN_CTC_BLANK_TOKEN_ID_KEY, "0"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    fn small_execution_metadata() -> DolphinExecutionMetadata {
        parse_dolphin_execution_metadata(&small_dolphin_metadata(), &()).expect("small parse")
    }

    /// Build a tensor index whose entries are 1-D tensors of the given element
    /// counts (the validator checks element counts, not stored orientation).
    fn tensor_index_from_counts(entries: &[(String, u64)]) -> crate::GgufTensorIndex {
        let tensors: Vec<crate::GgufTensorMetadata> = entries
            .iter()
            .map(|(name, elements)| crate::GgufTensorMetadata {
                name: name.clone(),
                dims: vec![*elements],
                ggml_type: 0,
                type_name: "f32".to_string(),
                size_bytes: 0,
                offset_bytes: 0,
            })
            .collect();
        crate::GgufTensorIndex::from_snapshot(crate::ggml_runtime::GgufTensorIndexSnapshot {
            path: PathBuf::from("/tmp/dolphin-contract.oasr"),
            data_section_offset_bytes: 0,
            tensors,
        })
        .expect("unique tensor names")
    }

    fn full_runtime_index(scheme: DolphinLanguageScheme) -> crate::GgufTensorIndex {
        let metadata = small_execution_metadata();
        let entries = dolphin_runtime_tensor_element_counts(&metadata, scheme);
        tensor_index_from_counts(&entries)
    }

    #[test]
    fn full_runtime_tensor_set_passes_the_contract() {
        let metadata = small_execution_metadata();
        for scheme in [
            DolphinLanguageScheme::CnDialect,
            DolphinLanguageScheme::Multilingual,
        ] {
            let index = full_runtime_index(scheme);
            validate_dolphin_runtime_tensors_with_index(&index, &metadata, scheme)
                .expect("complete tensor set must pass");
        }
    }

    #[test]
    fn missing_required_tensor_fails_closed() {
        let metadata = small_execution_metadata();
        let mut entries =
            dolphin_runtime_tensor_element_counts(&metadata, DolphinLanguageScheme::CnDialect);
        // Drop a tensor every scheme requires.
        entries.retain(|(name, _)| name != "ctc.ctc_lo.weight");
        let index = tensor_index_from_counts(&entries);
        let error = validate_dolphin_runtime_tensors_with_index(
            &index,
            &metadata,
            DolphinLanguageScheme::CnDialect,
        )
        .expect_err("a missing required tensor must fail closed");
        assert!(matches!(
            error,
            DolphinTensorContractError::MissingTensor { ref name } if name == "ctc.ctc_lo.weight"
        ));
    }

    #[test]
    fn wrong_element_count_fails_closed() {
        let metadata = small_execution_metadata();
        let mut entries =
            dolphin_runtime_tensor_element_counts(&metadata, DolphinLanguageScheme::CnDialect);
        for (name, elements) in entries.iter_mut() {
            if name == "ctc.ctc_lo.bias" {
                *elements += 1;
            }
        }
        let index = tensor_index_from_counts(&entries);
        let error = validate_dolphin_runtime_tensors_with_index(
            &index,
            &metadata,
            DolphinLanguageScheme::CnDialect,
        )
        .expect_err("a mis-shaped tensor must fail closed");
        assert!(matches!(
            error,
            DolphinTensorContractError::TensorElementCount { ref name, .. }
                if name == "ctc.ctc_lo.bias"
        ));
    }

    /// The cn-dialect encoder consumes the baked sinusoidal position table; the
    /// multilingual scheme computes it per request, so its packs legitimately omit
    /// the tensor and the contract must not require it there.
    #[test]
    fn encoder_position_table_is_required_only_for_cn_dialect() {
        let metadata = small_execution_metadata();
        let cn_entries =
            dolphin_runtime_tensor_element_counts(&metadata, DolphinLanguageScheme::CnDialect);
        assert!(
            cn_entries
                .iter()
                .any(|(name, _)| name == "encoder.embed.pos_enc.pe"),
            "cn-dialect contract must require the baked encoder position table"
        );
        let multilingual_entries =
            dolphin_runtime_tensor_element_counts(&metadata, DolphinLanguageScheme::Multilingual);
        assert!(
            !multilingual_entries
                .iter()
                .any(|(name, _)| name == "encoder.embed.pos_enc.pe"),
            "multilingual contract must not require the baked encoder position table"
        );

        // A multilingual pack without the table passes the multilingual contract
        // but fails the cn-dialect contract.
        let index_without = tensor_index_from_counts(&multilingual_entries);
        validate_dolphin_runtime_tensors_with_index(
            &index_without,
            &metadata,
            DolphinLanguageScheme::Multilingual,
        )
        .expect("multilingual pack without the baked table is valid");
        assert!(
            validate_dolphin_runtime_tensors_with_index(
                &index_without,
                &metadata,
                DolphinLanguageScheme::CnDialect,
            )
            .is_err()
        );
    }

    /// The required-tensor enumeration is a pure function of the metadata and the
    /// language scheme; pin a couple of counts so accidental formula drift is loud.
    #[test]
    fn tensor_element_counts_follow_the_metadata_geometry() {
        let metadata = small_execution_metadata();
        let entries =
            dolphin_runtime_tensor_element_counts(&metadata, DolphinLanguageScheme::CnDialect);
        let lookup: std::collections::BTreeMap<&str, u64> = entries
            .iter()
            .map(|(name, elements)| (name.as_str(), *elements))
            .collect();
        let d = metadata.encoder_d_model as u64; // 8
        let vocab = metadata.vocab_size as u64; // 12
        assert_eq!(lookup["encoder.encoders.0.attn.linear_q.weight"], d * d);
        assert_eq!(lookup["ctc.ctc_lo.weight"], vocab * d);
        assert_eq!(lookup["decoder.embed.0.weight"], d * vocab);
        assert_eq!(
            lookup["encoder.embed.pos_enc.pe"],
            d * metadata.encoder_max_ctx as u64
        );
    }
}
