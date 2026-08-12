//! Host-neutral MOSS-TD assets with owner-bound SystemMemory admission.
//!
//! Native runners, schedulers, uploaded arenas, and reusable KV graphs remain
//! lane-keyed actor state. This object owns only immutable metadata, tokenizer
//! state, host encoder weights, and the metadata-only Qwen decoder
//! plan. Keeping that boundary explicit lets CPU/GPU candidate retries share
//! the expensive host materialization without sharing a device owner.

use std::sync::Arc;

use thiserror::Error;

use crate::models::mapped_token_embedding::MappedTokenEmbeddingTable;
use crate::models::prepared_runtime_cache::{
    HostNeutralPreparedRuntime, PreparedRuntimeQuoteBuilder, PreparedRuntimeQuoteContext,
    SystemMemoryMaterialization,
};
use crate::models::qwen::{
    Qwen3AsrLlmLogitsHead, QwenDecoderContract, QwenDecoderTail, QwenDecoderTailLoadError,
    QwenWholeDecoderPlan, add_qwen_decoder_prepared_runtime_quote,
    load_qwen_decoder_tail_from_contract,
};
use crate::models::system_memory_owner::{SystemMemoryAllocationQuote, SystemMemoryOwnerError};

use super::encoder_graph::{
    MossEncoderConfig, MossEncoderWeights, load_moss_encoder_weights_from_reader,
};
use super::runtime_contract::{
    MOSS_TD_RMS_NORM_EPSILON, MossTdAdaptorMetadata, MossTdDecoderMetadata, MossTdEncoderMetadata,
    moss_td_qwen_decoder_contract, parse_adaptor_metadata, parse_decoder_metadata,
    parse_encoder_metadata,
};
use super::tokenizer::MossTdTokenizer;

#[derive(Clone)]
pub(crate) struct MossTdPreparedRuntime {
    pub(crate) encoder_metadata: MossTdEncoderMetadata,
    pub(crate) adaptor_metadata: MossTdAdaptorMetadata,
    pub(crate) decoder_metadata: MossTdDecoderMetadata,
    pub(crate) tokenizer: MossTdTokenizer,
    pub(crate) encoder_weights: Arc<MossEncoderWeights>,
    pub(crate) decoder_plan: Arc<QwenWholeDecoderPlan>,
    pub(crate) token_embedding: Arc<MappedTokenEmbeddingTable>,
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
        contract: &QwenDecoderContract,
    ) -> Result<(), SystemMemoryOwnerError> {
        add_qwen_decoder_prepared_runtime_quote(quote, context, contract)
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
        let decoder_contract = moss_td_qwen_decoder_contract(&decoder).map_err(|error| {
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
        MossEncoderWeights::add_system_memory_quote(
            &mut quote,
            context.tensor_index,
            Self::encoder_config(encoder),
        )?;
        Self::add_decoder_host_quote(&mut quote, context, &decoder_contract)?;
        quote.finish()
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add(
            self.tokenizer.retained_system_memory_bytes()?,
            "moss prepared tokenizer",
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
    let encoder_weights = load_moss_encoder_weights_from_reader(
        &reader,
        MossTdPreparedRuntime::encoder_config(encoder_metadata),
    )
    .map_err(|error| MossTdPreparedRuntimeError::Encoder {
        reason: error.to_string(),
    })?;
    let contract = moss_td_qwen_decoder_contract(&decoder_metadata).map_err(|error| {
        MossTdPreparedRuntimeError::DecoderPlan {
            reason: error.to_string(),
        }
    })?;
    let decoder_plan =
        QwenWholeDecoderPlan::for_qwen_family(&reader, &contract).map_err(|error| {
            MossTdPreparedRuntimeError::DecoderPlan {
                reason: error.to_string(),
            }
        })?;
    let QwenDecoderTail {
        logits_head,
        token_embedding,
    } = load_qwen_decoder_tail_from_contract(&reader, &contract, MOSS_TD_RMS_NORM_EPSILON, backend)
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
