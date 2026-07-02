use thiserror::Error;

use super::audio_encoder::Qwen3AsrAudioEncoderWeights;
use super::frontend::Qwen3AsrMelFrontendPlan;
use super::llm_transformer::Qwen3AsrLlmLayerAttentionProjection;
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
    pub token_embedding_table: Qwen3AsrTokenEmbeddingTable,
    pub logits_head: Qwen3AsrLlmLogitsHead,
    pub layer_attention_projections: Vec<Qwen3AsrLlmLayerAttentionProjection>,
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
) -> Result<Qwen3AsrPreparedRuntime, Qwen3AsrPreparedRuntimeError> {
    let components = build_builtin_runtime_component_bootstrap(
        QWEN3_ASR_GGML_ARCHITECTURE_ID,
        preflight,
        BuiltinTokenizerMaterializationMode::Optional,
    )
    .map_err(map_runtime_component_bootstrap_error)?;
    build_qwen_prepared_runtime_from_components(components)
}

pub(crate) fn build_qwen_prepared_runtime_from_components(
    components: BuiltinRuntimeComponentBootstrap,
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
    let (audio_encoder_weights, token_embedding_table, logits_head, layer_attention_projections) =
        materialize_builtin_runtime_weight_components(
            QWEN3_ASR_GGML_ARCHITECTURE_ID,
            &tensor_reader,
            runtime_metadata,
        )
        .map_err(map_runtime_weight_component_error)?
        .into_qwen3_asr()
        .expect("qwen weight registry must return qwen weights");
    if layer_attention_projections.is_empty() {
        return Err(Qwen3AsrPreparedRuntimeError::RuntimeContractViolation {
            reason: "qwen3-asr runtime exposes zero llm layers; at least 1 is required".to_string(),
        });
    }
    Ok(Qwen3AsrPreparedRuntime {
        metadata,
        tokenizer,
        mel_frontend_plan,
        audio_encoder_weights,
        token_embedding_table,
        logits_head,
        layer_attention_projections,
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
