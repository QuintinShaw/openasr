//! Host-neutral MOSS-TD assets with owner-bound SystemMemory admission.
//!
//! Native runners, schedulers, uploaded arenas, and reusable KV graphs remain
//! lane-keyed actor state. This object owns only immutable metadata, tokenizer
//! state, host adaptor/encoder weights, and the metadata-only Qwen decoder
//! plan. Keeping that boundary explicit lets CPU/GPU candidate retries share
//! the expensive host materialization without sharing a device owner.

use std::sync::Arc;

use thiserror::Error;

use crate::models::prepared_runtime_cache::{
    HostNeutralPreparedRuntime, PreparedRuntimeQuoteBuilder, PreparedRuntimeQuoteContext,
    SystemMemoryMaterialization,
};
use crate::models::qwen::{
    Qwen3AsrLlmLogitsHead, Qwen3AsrTokenEmbeddingTable, QwenDecoderTail, QwenDecoderTailLoadError,
    QwenWholeDecoderPlan, load_qwen_decoder_tail_from_contract,
};
use crate::models::system_memory_owner::{SystemMemoryAllocationQuote, SystemMemoryOwnerError};

use super::adaptor_graph::{MossAdaptorWeights, load_moss_adaptor_weights_from_reader};
use super::encoder_graph::{
    MossEncoderConfig, MossEncoderWeights, load_moss_encoder_weights_from_reader,
};
use super::runtime_contract::{
    MOSS_TD_ADAPTOR_NORM_EPSILON, MOSS_TD_RMS_NORM_EPSILON, MossTdAdaptorMetadata,
    MossTdDecoderMetadata, MossTdEncoderMetadata, moss_td_qwen_decoder_profile,
    parse_adaptor_metadata, parse_decoder_metadata, parse_encoder_metadata,
};
use super::tokenizer::MossTdTokenizer;

#[derive(Clone)]
pub(crate) struct MossTdPreparedRuntime {
    pub(crate) encoder_metadata: MossTdEncoderMetadata,
    pub(crate) adaptor_metadata: MossTdAdaptorMetadata,
    pub(crate) decoder_metadata: MossTdDecoderMetadata,
    pub(crate) tokenizer: MossTdTokenizer,
    pub(crate) adaptor_weights: Arc<MossAdaptorWeights>,
    pub(crate) encoder_weights: Arc<MossEncoderWeights>,
    pub(crate) decoder_plan: Arc<QwenWholeDecoderPlan>,
    pub(crate) token_embedding: Arc<Qwen3AsrTokenEmbeddingTable>,
    pub(crate) logits_head: Arc<Qwen3AsrLlmLogitsHead>,
}

impl MossTdPreparedRuntime {
    fn encoder_config(encoder: MossTdEncoderMetadata) -> MossEncoderConfig {
        MossEncoderConfig {
            n_layers: encoder.n_layers,
            d_model: encoder.d_model,
            n_heads: encoder.n_heads,
            n_mels: encoder.n_mels,
            max_source_positions: encoder.max_source_positions,
        }
    }

    fn add_decoder_host_quote(
        quote: &mut PreparedRuntimeQuoteBuilder,
        context: PreparedRuntimeQuoteContext<'_>,
        decoder: MossTdDecoderMetadata,
    ) -> Result<(), SystemMemoryOwnerError> {
        // Tail names and tied-embedding policy come only from the family profile
        // (output_weight=None => logits share token_embd). Do not re-state MOSS
        // tensor spellings or tied-head shape here.
        let profile = moss_td_qwen_decoder_profile();
        let tail = profile.tail();
        let embd_name = tail.token_embd;
        let norm_name = tail.output_norm;
        debug_assert!(
            tail.output_weight.is_none(),
            "MOSS profile must encode tied embeddings via output_weight=None"
        );

        let plan_bytes = QwenWholeDecoderPlan::quoted_retained_system_memory_bytes_for_family(
            decoder.n_layers,
            profile.names_for_layer(),
        )
        .map_err(|reason| {
            SystemMemoryOwnerError::capacity_failure("prepared_runtime_quote", reason)
        })?;
        quote.add_structural_bytes(plan_bytes, "moss decoder metadata plan")?;

        let embedding = context.tensor_index.get(embd_name).ok_or_else(|| {
            SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                format!("required tensor '{embd_name}' is missing"),
            )
        })?;
        let canonical_dims = [decoder.d_model as u64, decoder.vocab_size as u64];
        if embedding.ggml_type == 0 || embedding.ggml_type == 1 || embedding.dims == canonical_dims
        {
            quote.add_owned_tensor_payload_metadata(context.tensor_index, embd_name)?;
        } else {
            quote.add_tensor_f32(context.tensor_index, embd_name)?;
        }

        quote.add_tensor_f32(context.tensor_index, norm_name)?;
        // Tied head: quote logits against the same embd mapping (no separate output weight).
        if crate::models::qwen::logits_head_ggml_enabled(context.backend)
            && embedding.dims == canonical_dims
        {
            quote.add_owned_tensor_payload_metadata(context.tensor_index, embd_name)?;
            quote.add_owned_elements::<usize>(
                u64::try_from(embedding.dims.len()).map_err(|_| {
                    SystemMemoryOwnerError::capacity_failure(
                        "prepared_runtime_quote",
                        "moss logits rank does not fit u64",
                    )
                })?,
                "moss logits raw dims",
            )?;
        } else {
            quote.add_tensor_f32(context.tensor_index, embd_name)?;
        }
        Ok(())
    }

    pub(crate) fn system_memory_quote(
        context: PreparedRuntimeQuoteContext<'_>,
        pack_content_id: &str,
    ) -> Result<SystemMemoryAllocationQuote, SystemMemoryOwnerError> {
        let encoder = parse_encoder_metadata(context.metadata).map_err(|error| {
            SystemMemoryOwnerError::capacity_failure("prepared_runtime_quote", error.to_string())
        })?;
        let decoder = parse_decoder_metadata(context.metadata).map_err(|error| {
            SystemMemoryOwnerError::capacity_failure("prepared_runtime_quote", error.to_string())
        })?;
        parse_adaptor_metadata(context.metadata).map_err(|error| {
            SystemMemoryOwnerError::capacity_failure("prepared_runtime_quote", error.to_string())
        })?;

        let mut quote = PreparedRuntimeQuoteBuilder::new::<Self>(pack_content_id);
        quote.add_structural_bytes(
            MossTdTokenizer::quoted_retained_system_memory_bytes(context.metadata)?,
            "moss tokenizer",
        )?;
        MossAdaptorWeights::add_system_memory_quote(&mut quote, context.tensor_index)?;
        MossEncoderWeights::add_system_memory_quote(
            &mut quote,
            context.tensor_index,
            Self::encoder_config(encoder),
        )?;
        Self::add_decoder_host_quote(&mut quote, context, decoder)?;
        quote.finish()
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add(
            self.tokenizer.retained_system_memory_bytes()?,
            "moss prepared tokenizer",
        )?;
        bytes.add(
            self.adaptor_weights.retained_system_memory_bytes()?,
            "moss prepared adaptor",
        )?;
        bytes.add(
            self.encoder_weights.retained_system_memory_bytes()?,
            "moss prepared encoder weights",
        )?;
        bytes.add(
            self.decoder_plan.retained_system_memory_bytes()?,
            "moss prepared decoder plan",
        )?;
        bytes.add(
            self.token_embedding.retained_system_memory_bytes()?,
            "moss prepared token embedding",
        )?;
        bytes.add(
            self.logits_head.retained_system_memory_bytes()?,
            "moss prepared logits head",
        )?;
        Ok(bytes.finish())
    }
}

impl SystemMemoryMaterialization for MossTdPreparedRuntime {
    fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        Self::retained_system_memory_bytes(self)
    }
}

impl HostNeutralPreparedRuntime for MossTdPreparedRuntime {
    fn system_memory_quote(
        context: PreparedRuntimeQuoteContext<'_>,
        pack_content_id: &str,
    ) -> Result<SystemMemoryAllocationQuote, SystemMemoryOwnerError> {
        Self::system_memory_quote(context, pack_content_id)
    }
}

#[derive(Debug, Error)]
pub(crate) enum MossTdPreparedRuntimeError {
    #[error("moss prepared runtime contract failed: {reason}")]
    Contract { reason: String },
    #[error("moss prepared tokenizer failed: {reason}")]
    Tokenizer { reason: String },
    #[error("moss prepared tensor reader failed: {reason}")]
    TensorReader { reason: String },
    #[error("moss prepared adaptor failed: {reason}")]
    Adaptor { reason: String },
    #[error("moss prepared encoder weights failed: {reason}")]
    Encoder { reason: String },
    #[error("moss prepared decoder plan failed: {reason}")]
    DecoderPlan { reason: String },
    #[error("moss prepared token embedding failed: {reason}")]
    TokenEmbedding { reason: String },
    #[error("moss prepared logits head failed: {reason}")]
    LogitsHead { reason: String },
}

pub(crate) fn build_moss_td_prepared_runtime(
    preflight: &crate::GgufRuntimeSourcePreflight,
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
) -> Result<MossTdPreparedRuntime, MossTdPreparedRuntimeError> {
    let encoder_metadata = parse_encoder_metadata(&preflight.metadata).map_err(|error| {
        MossTdPreparedRuntimeError::Contract {
            reason: error.to_string(),
        }
    })?;
    let adaptor_metadata = parse_adaptor_metadata(&preflight.metadata).map_err(|error| {
        MossTdPreparedRuntimeError::Contract {
            reason: error.to_string(),
        }
    })?;
    let decoder_metadata = parse_decoder_metadata(&preflight.metadata).map_err(|error| {
        MossTdPreparedRuntimeError::Contract {
            reason: error.to_string(),
        }
    })?;
    let tokenizer = MossTdTokenizer::from_gguf_metadata(&preflight.metadata).map_err(|error| {
        MossTdPreparedRuntimeError::Tokenizer {
            reason: error.to_string(),
        }
    })?;
    let reader =
        crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight(preflight)
            .map_err(|error| MossTdPreparedRuntimeError::TensorReader {
                reason: error.to_string(),
            })?;
    let adaptor_weights = load_moss_adaptor_weights_from_reader(
        &reader,
        encoder_metadata.d_model,
        adaptor_metadata.merge_size,
        decoder_metadata.d_model,
        MOSS_TD_ADAPTOR_NORM_EPSILON,
    )
    .map_err(|error| MossTdPreparedRuntimeError::Adaptor {
        reason: error.to_string(),
    })?;
    let encoder_weights = load_moss_encoder_weights_from_reader(
        &reader,
        MossTdPreparedRuntime::encoder_config(encoder_metadata),
    )
    .map_err(|error| MossTdPreparedRuntimeError::Encoder {
        reason: error.to_string(),
    })?;
    let contract = super::runtime_contract::moss_td_qwen_decoder_contract(&decoder_metadata)
        .map_err(|error| MossTdPreparedRuntimeError::DecoderPlan {
            reason: error.to_string(),
        })?;
    let decoder_plan =
        QwenWholeDecoderPlan::for_qwen_family(&reader, contract).map_err(|error| {
            MossTdPreparedRuntimeError::DecoderPlan {
                reason: error.to_string(),
            }
        })?;
    let QwenDecoderTail {
        logits_head,
        token_embedding,
    } = load_qwen_decoder_tail_from_contract(&reader, contract, MOSS_TD_RMS_NORM_EPSILON, backend)
        .map_err(|error| match error {
            QwenDecoderTailLoadError::TokenEmbedding(error) => {
                MossTdPreparedRuntimeError::TokenEmbedding {
                    reason: error.to_string(),
                }
            }
            other => MossTdPreparedRuntimeError::LogitsHead {
                reason: other.to_string(),
            },
        })?;
    Ok(MossTdPreparedRuntime {
        encoder_metadata,
        adaptor_metadata,
        decoder_metadata,
        tokenizer,
        adaptor_weights: Arc::new(adaptor_weights),
        encoder_weights: Arc::new(encoder_weights),
        decoder_plan: Arc::new(decoder_plan),
        token_embedding: Arc::new(token_embedding),
        logits_head: Arc::new(logits_head),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_runtime_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MossTdPreparedRuntime>();
    }
}
