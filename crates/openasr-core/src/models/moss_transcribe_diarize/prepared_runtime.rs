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
    Qwen3AsrLlmLogitsHead, Qwen3AsrTokenEmbeddingTable, QwenFamilyLlmLayerTensorNames,
    QwenWholeDecoderPlan, load_llm_logits_head_from_reader_with_tensor_names,
    load_token_embedding_table_from_reader_with_tensor_name,
};
use crate::models::system_memory_owner::{SystemMemoryAllocationQuote, SystemMemoryOwnerError};

use super::adaptor_graph::{MossAdaptorWeights, load_moss_adaptor_weights_from_reader};
use super::encoder_graph::{
    MossEncoderConfig, MossEncoderWeights, load_moss_encoder_weights_from_reader,
};
use super::runtime_contract::{
    MOSS_TD_ADAPTOR_NORM_EPSILON, MOSS_TD_RMS_NORM_EPSILON, MossTdAdaptorMetadata,
    MossTdDecoderMetadata, MossTdEncoderMetadata, parse_adaptor_metadata, parse_decoder_metadata,
    parse_encoder_metadata,
};
use super::tensor_names::{
    LLM_OUTPUT_NORM_WEIGHT, LLM_TOKEN_EMBD_WEIGHT, moss_llm_layer_tensor_names,
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

    fn decoder_names(layer_index: usize) -> QwenFamilyLlmLayerTensorNames {
        let names = moss_llm_layer_tensor_names(layer_index);
        QwenFamilyLlmLayerTensorNames {
            attn_norm_name: names.attn_norm_weight,
            attn_q_name: names.attn_q_weight,
            attn_k_name: names.attn_k_weight,
            attn_v_name: names.attn_v_weight,
            attn_output_name: names.attn_output_weight,
            q_norm_name: Some(names.attn_q_norm_weight),
            k_norm_name: Some(names.attn_k_norm_weight),
            q_bias_name: None,
            k_bias_name: None,
            v_bias_name: None,
            ffn_norm_name: names.ffn_norm_weight,
            ffn_gate_name: names.ffn_gate_weight,
            ffn_up_name: names.ffn_up_weight,
            ffn_down_name: names.ffn_down_weight,
        }
    }

    fn add_decoder_host_quote(
        quote: &mut PreparedRuntimeQuoteBuilder,
        context: PreparedRuntimeQuoteContext<'_>,
        decoder: MossTdDecoderMetadata,
    ) -> Result<(), SystemMemoryOwnerError> {
        let plan_bytes = QwenWholeDecoderPlan::quoted_retained_system_memory_bytes_for_family(
            decoder.n_layers,
            Self::decoder_names,
        )
        .map_err(|reason| {
            SystemMemoryOwnerError::capacity_failure("prepared_runtime_quote", reason)
        })?;
        quote.add_structural_bytes(plan_bytes, "moss decoder metadata plan")?;

        let embedding = context
            .tensor_index
            .get(LLM_TOKEN_EMBD_WEIGHT)
            .ok_or_else(|| {
                SystemMemoryOwnerError::capacity_failure(
                    "prepared_runtime_quote",
                    format!("required tensor '{LLM_TOKEN_EMBD_WEIGHT}' is missing"),
                )
            })?;
        let canonical_dims = [decoder.d_model as u64, decoder.vocab_size as u64];
        if embedding.ggml_type == 0 || embedding.ggml_type == 1 || embedding.dims == canonical_dims
        {
            quote.add_owned_tensor_payload_metadata(context.tensor_index, LLM_TOKEN_EMBD_WEIGHT)?;
        } else {
            quote.add_tensor_f32(context.tensor_index, LLM_TOKEN_EMBD_WEIGHT)?;
        }

        quote.add_tensor_f32(context.tensor_index, LLM_OUTPUT_NORM_WEIGHT)?;
        if crate::models::qwen::logits_head_ggml_enabled(context.backend)
            && embedding.dims == canonical_dims
        {
            quote.add_owned_tensor_payload_metadata(context.tensor_index, LLM_TOKEN_EMBD_WEIGHT)?;
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
            quote.add_tensor_f32(context.tensor_index, LLM_TOKEN_EMBD_WEIGHT)?;
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
    preflight: &crate::GgmlAsrRuntimeSourcePreflight,
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
    let decoder_plan = QwenWholeDecoderPlan::for_qwen_family(
        &reader,
        decoder_metadata.n_layers,
        decoder_metadata.d_model,
        decoder_metadata.n_heads,
        decoder_metadata.n_kv_heads,
        decoder_metadata.head_dim,
        MossTdPreparedRuntime::decoder_names,
    )
    .map_err(|error| MossTdPreparedRuntimeError::DecoderPlan {
        reason: error.to_string(),
    })?;
    let logits_head = load_llm_logits_head_from_reader_with_tensor_names(
        &reader,
        &preflight.runtime_source,
        decoder_metadata.d_model,
        decoder_metadata.vocab_size,
        LLM_OUTPUT_NORM_WEIGHT,
        LLM_TOKEN_EMBD_WEIGHT,
        MOSS_TD_RMS_NORM_EPSILON,
        backend,
    )
    .map_err(|error| MossTdPreparedRuntimeError::LogitsHead {
        reason: error.to_string(),
    })?;
    let token_embedding = load_token_embedding_table_from_reader_with_tensor_name(
        &reader,
        LLM_TOKEN_EMBD_WEIGHT,
        decoder_metadata.d_model,
        decoder_metadata.vocab_size,
    )
    .map_err(|error| MossTdPreparedRuntimeError::TokenEmbedding {
        reason: error.to_string(),
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
