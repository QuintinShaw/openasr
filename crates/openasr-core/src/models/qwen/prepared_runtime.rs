use std::sync::Arc;
use thiserror::Error;

use super::audio_encoder::Qwen3AsrAudioEncoderWeights;
use super::frontend::Qwen3AsrMelFrontendPlan;
use super::llm_transformer::QwenWholeDecoderPlan;
use super::logits_head::Qwen3AsrLlmLogitsHead;
use super::runtime_contract::Qwen3AsrExecutionMetadata;
use super::token_embedding::Qwen3AsrTokenEmbeddingTable;
use super::tokenizer::Qwen3AsrTokenizer;
use crate::QWEN3_ASR_GGML_ARCHITECTURE_ID;
use crate::models::ggml_asr_executor::GgmlAsrRuntimeSourcePreflight;
use crate::models::runtime_component_bootstrap::{
    BuiltinRuntimeComponentBootstrap, BuiltinRuntimeComponentBootstrapError,
    BuiltinTokenizerMaterializationMode, build_builtin_runtime_component_bootstrap,
};
use crate::models::runtime_weight_component_registry::{
    BuiltinRuntimeWeightComponentRegistryError, materialize_builtin_runtime_weight_components,
};

#[derive(Debug, Clone)]
pub(crate) struct Qwen3AsrPreparedRuntime {
    pub metadata: Qwen3AsrExecutionMetadata,
    pub tokenizer: Option<Qwen3AsrTokenizer>,
    pub mel_frontend_plan: Qwen3AsrMelFrontendPlan,
    pub audio_encoder_weights: Qwen3AsrAudioEncoderWeights,
    pub token_embedding_table: Arc<Qwen3AsrTokenEmbeddingTable>,
    pub logits_head: Arc<Qwen3AsrLlmLogitsHead>,
    pub decoder_plan: Arc<QwenWholeDecoderPlan>,
}

impl Qwen3AsrPreparedRuntime {
    pub(crate) fn system_memory_quote(
        context: crate::models::prepared_runtime_cache::PreparedRuntimeQuoteContext<'_>,
        pack_content_id: &str,
    ) -> Result<
        crate::models::system_memory_owner::SystemMemoryAllocationQuote,
        crate::models::system_memory_owner::SystemMemoryOwnerError,
    > {
        use super::tensor_names::{OUTPUT_WEIGHT, TOKEN_EMBD_WEIGHT, llm_layer_tensor_names};
        use crate::models::system_memory_owner::SystemMemoryOwnerError;

        let execution = super::runtime_contract::parse_qwen3_execution_metadata(context.metadata)
            .map_err(|error| {
            SystemMemoryOwnerError::capacity_failure("prepared_runtime_quote", error.to_string())
        })?;
        let mut quote = crate::models::prepared_runtime_cache::PreparedRuntimeQuoteBuilder::new::<
            Self,
        >(pack_content_id);
        quote.add_tokenizer_metadata(context.metadata, true)?;
        let mut decoder_tensor_names = std::collections::HashSet::new();
        for layer_index in 0..execution.llm_layers {
            let names = llm_layer_tensor_names(layer_index);
            decoder_tensor_names.extend([
                names.attn_norm_weight,
                names.attn_q_weight,
                names.attn_k_weight,
                names.attn_v_weight,
                names.attn_output_weight,
                names.attn_q_norm_weight,
                names.attn_k_norm_weight,
                names.ffn_norm_weight,
                names.ffn_gate_weight,
                names.ffn_up_weight,
                names.ffn_down_weight,
            ]);
        }
        for tensor in context.tensor_index.tensors() {
            if tensor.name == TOKEN_EMBD_WEIGHT || tensor.name == OUTPUT_WEIGHT {
                continue;
            }
            if decoder_tensor_names.contains(&tensor.name) {
                quote.add_tensor_metadata(context.tensor_index, &tensor.name)?;
                continue;
            }
            quote.add_tensor_f32_or_raw_upper_bound(context.tensor_index, &tensor.name)?;
            quote.add_tensor_metadata(context.tensor_index, &tensor.name)?;
        }

        let embedding = context.tensor_index.get(TOKEN_EMBD_WEIGHT).ok_or_else(|| {
            SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                format!("required tensor '{TOKEN_EMBD_WEIGHT}' is missing"),
            )
        })?;
        if embedding.ggml_type == 0
            || embedding.ggml_type == 1
            || embedding.dims == [execution.llm_d_model as u64, execution.vocab_size as u64]
        {
            // F32/F16 in either orientation and token-major quantized tables
            // are retained as owning views into the already-open mmap. Only
            // the payload metadata is heap-owned.
            quote.add_owned_tensor_payload_metadata(context.tensor_index, TOKEN_EMBD_WEIGHT)?;
        } else {
            // A quantized hidden-major table cannot be gathered by ggml row;
            // this rare compatibility representation is transposed to f32.
            quote.add_tensor_f32(context.tensor_index, TOKEN_EMBD_WEIGHT)?;
        }

        let output = context.tensor_index.get(OUTPUT_WEIGHT).ok_or_else(|| {
            SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                format!("required tensor '{OUTPUT_WEIGHT}' is missing"),
            )
        })?;
        if super::logits_head::logits_head_ggml_enabled(context.backend)
            && output.dims == [execution.llm_d_model as u64, execution.vocab_size as u64]
        {
            // The direct graph path keeps an owning mmap view. Its GGUF bytes
            // are file-backed and shared with a tied token table, not copied
            // into a second resident Vec.
            quote.add_owned_tensor_payload_metadata(context.tensor_index, OUTPUT_WEIGHT)?;
            quote.add_owned_elements::<usize>(
                u64::try_from(output.dims.len()).map_err(|_| {
                    SystemMemoryOwnerError::capacity_failure(
                        "prepared_runtime_quote",
                        "qwen logits rank does not fit u64",
                    )
                })?,
                "qwen logits raw dims",
            )?;
        } else {
            quote.add_tensor_f32(context.tensor_index, OUTPUT_WEIGHT)?;
        }

        // The whole decoder is retained as names/shapes/types only. Account
        // its owned tensor-name strings a second time (the generic metadata
        // pass above accounts the index's copy); no decoder weight payload is
        // retained in the prepared runtime.
        for layer_index in 0..execution.llm_layers {
            let names = llm_layer_tensor_names(layer_index);
            for name in [
                names.attn_norm_weight,
                names.attn_q_weight,
                names.attn_k_weight,
                names.attn_v_weight,
                names.attn_output_weight,
                names.attn_q_norm_weight,
                names.attn_k_norm_weight,
                names.ffn_norm_weight,
                names.ffn_gate_weight,
                names.ffn_up_weight,
                names.ffn_down_weight,
            ] {
                quote.add_owned_bytes(
                    u64::try_from(name.len()).map_err(|_| {
                        SystemMemoryOwnerError::capacity_failure(
                            "prepared_runtime_quote",
                            "qwen decoder tensor-name length does not fit u64",
                        )
                    })?,
                    "qwen decoder-plan tensor name",
                )?;
            }
        }
        quote.finish()
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add(
            self.mel_frontend_plan.retained_system_memory_bytes()?,
            "qwen prepared frontend",
        )?;
        if let Some(tokenizer) = &self.tokenizer {
            bytes.add(
                tokenizer.retained_system_memory_bytes()?,
                "qwen prepared tokenizer",
            )?;
        }
        bytes.add(
            self.audio_encoder_weights.retained_system_memory_bytes()?,
            "qwen prepared audio weights",
        )?;
        bytes.add(
            self.token_embedding_table.retained_system_memory_bytes()?,
            "qwen prepared token embedding",
        )?;
        bytes.add(
            self.logits_head.retained_system_memory_bytes()?,
            "qwen prepared logits",
        )?;
        bytes.add(
            self.decoder_plan.retained_system_memory_bytes()?,
            "qwen prepared decoder plan",
        )?;
        Ok(bytes.finish())
    }
}

#[derive(Debug, Error)]
pub(crate) enum Qwen3AsrPreparedRuntimeError {
    #[error("qwen3-asr runtime contract check failed: {reason}")]
    RuntimeContractViolation { reason: String },
    #[error("qwen3-asr runtime metadata read failed: {reason}")]
    RuntimeMetadataReadFailed { reason: String },
    #[error("qwen3-asr mel frontend failed: {reason}")]
    MelFrontendFailed { reason: String },
    #[error("qwen3-asr audio encoder failed: {reason}")]
    AudioEncoderFailed { reason: String },
    #[error("qwen3-asr token embedding prefill failed: {reason}")]
    TokenEmbeddingPrefillFailed { reason: String },
    #[error("qwen3-asr llm logits head failed: {reason}")]
    LlmLogitsHeadFailed { reason: String },
    #[error("qwen3-asr llm transformer decode step failed: {reason}")]
    LlmTransformerDecodeStepFailed { reason: String },
}

pub(crate) fn build_qwen_prepared_runtime(
    preflight: &GgmlAsrRuntimeSourcePreflight,
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
) -> Result<Qwen3AsrPreparedRuntime, Qwen3AsrPreparedRuntimeError> {
    let components = build_builtin_runtime_component_bootstrap(
        QWEN3_ASR_GGML_ARCHITECTURE_ID,
        preflight,
        BuiltinTokenizerMaterializationMode::Optional,
    )
    .map_err(map_runtime_component_bootstrap_error)?;
    build_qwen_prepared_runtime_from_components(components, &preflight.runtime_source, backend)
}

pub(crate) fn build_qwen_prepared_runtime_from_components(
    components: BuiltinRuntimeComponentBootstrap,
    runtime_source: &crate::GgmlRuntimeSource,
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
) -> Result<Qwen3AsrPreparedRuntime, Qwen3AsrPreparedRuntimeError> {
    let runtime_metadata = components.metadata;
    let metadata = runtime_metadata
        .into_qwen3_asr()
        .expect("qwen component bootstrap must carry qwen metadata");
    let tokenizer = components
        .tokenizer
        .and_then(|tokenizer| tokenizer.into_qwen3_asr());
    let tensor_reader = components.tensor_reader;
    let mel_frontend_plan = components
        .audio_frontend
        .into_qwen3_asr()
        .expect("qwen component bootstrap must return qwen frontend plan");
    let (audio_encoder_weights, token_embedding_table, logits_head, decoder_plan) =
        materialize_builtin_runtime_weight_components(
            QWEN3_ASR_GGML_ARCHITECTURE_ID,
            &tensor_reader,
            runtime_source,
            runtime_metadata,
            backend,
        )
        .map_err(map_runtime_weight_component_error)?
        .into_qwen3_asr()
        .expect("qwen weight registry must return qwen weights");
    if decoder_plan.layer_count() == 0 {
        return Err(Qwen3AsrPreparedRuntimeError::RuntimeContractViolation {
            reason: "qwen3-asr runtime exposes zero llm layers; at least 1 is required".to_string(),
        });
    }
    Ok(Qwen3AsrPreparedRuntime {
        metadata,
        tokenizer,
        mel_frontend_plan,
        audio_encoder_weights,
        token_embedding_table: Arc::new(token_embedding_table),
        logits_head: Arc::new(logits_head),
        decoder_plan: Arc::new(decoder_plan),
    })
}

fn map_runtime_weight_component_error(
    error: BuiltinRuntimeWeightComponentRegistryError,
) -> Qwen3AsrPreparedRuntimeError {
    match error {
        BuiltinRuntimeWeightComponentRegistryError::MaterializationFailed {
            component: "qwen3-asr.audio-encoder-weights",
            reason,
        } => Qwen3AsrPreparedRuntimeError::AudioEncoderFailed { reason },
        BuiltinRuntimeWeightComponentRegistryError::MaterializationFailed {
            component: "qwen3-asr.token-embedding",
            reason,
        } => Qwen3AsrPreparedRuntimeError::TokenEmbeddingPrefillFailed { reason },
        BuiltinRuntimeWeightComponentRegistryError::MaterializationFailed {
            component: "qwen3-asr.logits-head",
            reason,
        } => Qwen3AsrPreparedRuntimeError::LlmLogitsHeadFailed { reason },
        BuiltinRuntimeWeightComponentRegistryError::MaterializationFailed { reason, .. } => {
            Qwen3AsrPreparedRuntimeError::LlmTransformerDecodeStepFailed { reason }
        }
        other => Qwen3AsrPreparedRuntimeError::RuntimeContractViolation {
            reason: other.to_string(),
        },
    }
}

fn map_runtime_component_bootstrap_error(
    error: BuiltinRuntimeComponentBootstrapError,
) -> Qwen3AsrPreparedRuntimeError {
    match error {
        BuiltinRuntimeComponentBootstrapError::RuntimeAssetBootstrap { source } => match source {
            crate::models::runtime_asset_bootstrap::BuiltinRuntimeAssetBootstrapError::RuntimeContractPreflight { source } => {
                Qwen3AsrPreparedRuntimeError::RuntimeContractViolation {
                    reason: source.to_string(),
                }
            }
            crate::models::runtime_asset_bootstrap::BuiltinRuntimeAssetBootstrapError::TensorReaderBuild { source } => {
                Qwen3AsrPreparedRuntimeError::RuntimeMetadataReadFailed {
                    reason: source.to_string(),
                }
            }
        },
        BuiltinRuntimeComponentBootstrapError::TokenizerMaterialization { source } => {
            Qwen3AsrPreparedRuntimeError::RuntimeMetadataReadFailed {
                reason: source.to_string(),
            }
        }
        BuiltinRuntimeComponentBootstrapError::AudioFrontendMaterialization { source } => {
            Qwen3AsrPreparedRuntimeError::MelFrontendFailed {
                reason: source.to_string(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_prepared_runtime_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Qwen3AsrPreparedRuntime>();
    }
}
