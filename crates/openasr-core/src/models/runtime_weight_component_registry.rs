//! Runtime weight-bundle materialization for the **data-driven composer**
//! families only (Cohere Transcribe + Qwen3-ASR). Called solely from those two
//! families' `prepared_runtime` paths — never generically across architectures.
//!
//! The dedicated-executor families (Whisper, Moonshine, Parakeet-CTC, wav2vec2/
//! data2vec-CTC) materialize their weights in their own family modules and never
//! reach this enum; that boundary is enforced at the frontend chokepoint (see
//! [`super::frontend_component_registry`]). An unrecognized architecture here is
//! therefore a programming error and fails closed via `UnknownArchitecture`.

use thiserror::Error;

use crate::GgufTensorDataReader;
use crate::models::qwen::QWEN3_ASR_MODEL_FAMILY;

use super::cohere::{
    CohereTranscribeDecoderWeights, CohereTranscribeEncoderWeights,
    load_cohere_transcribe_decoder_weights_for_runtime_from_reader,
    load_cohere_transcribe_encoder_weights_from_reader,
};
use super::qwen::{
    Qwen3AsrAudioEncoderWeights, Qwen3AsrLlmLayerAttentionProjection, Qwen3AsrLlmLogitsHead,
    Qwen3AsrTokenEmbeddingTable, load_qwen3_audio_encoder_weights_from_reader,
    load_qwen3_llm_attention_projections_from_reader, load_qwen3_llm_logits_head_from_reader,
    load_qwen3_token_embedding_table_from_reader,
};
use super::runtime_tensor_contract_registry::RuntimeTensorContractMetadata;

// Per-family weight bundles differ in size (qwen carries audio-encoder + LLM
// layer projections); this enum is materialized once and held behind an `Arc`,
// so the variant-size delta never lands on the stack — boxing would only add an
// indirection for no benefit.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub(crate) enum BuiltinRuntimeWeightComponents {
    CohereTranscribe {
        encoder_weights: CohereTranscribeEncoderWeights,
        decoder_weights: CohereTranscribeDecoderWeights,
    },
    Qwen3Asr {
        audio_encoder_weights: Qwen3AsrAudioEncoderWeights,
        token_embedding_table: Qwen3AsrTokenEmbeddingTable,
        logits_head: Qwen3AsrLlmLogitsHead,
        layer_attention_projections: Vec<Qwen3AsrLlmLayerAttentionProjection>,
    },
}

impl BuiltinRuntimeWeightComponents {
    pub(crate) fn into_cohere_transcribe(
        self,
    ) -> Option<(
        CohereTranscribeEncoderWeights,
        CohereTranscribeDecoderWeights,
    )> {
        match self {
            Self::CohereTranscribe {
                encoder_weights,
                decoder_weights,
            } => Some((encoder_weights, decoder_weights)),
            _ => None,
        }
    }

    pub(crate) fn into_qwen3_asr(
        self,
    ) -> Option<(
        Qwen3AsrAudioEncoderWeights,
        Qwen3AsrTokenEmbeddingTable,
        Qwen3AsrLlmLogitsHead,
        Vec<Qwen3AsrLlmLayerAttentionProjection>,
    )> {
        match self {
            Self::Qwen3Asr {
                audio_encoder_weights,
                token_embedding_table,
                logits_head,
                layer_attention_projections,
            } => Some((
                audio_encoder_weights,
                token_embedding_table,
                logits_head,
                layer_attention_projections,
            )),
            _ => None,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum BuiltinRuntimeWeightComponentRegistryError {
    #[error("unknown builtin model architecture '{model_architecture}'")]
    UnknownArchitecture { model_architecture: String },
    #[error("builtin runtime weights expected metadata for '{expected_kind}', got '{found_kind}'")]
    MetadataKindMismatch {
        expected_kind: &'static str,
        found_kind: &'static str,
    },
    #[error("builtin runtime weights materialization failed for '{component}': {reason}")]
    MaterializationFailed {
        component: &'static str,
        reason: String,
    },
}

pub(crate) fn materialize_builtin_runtime_weight_components(
    model_architecture: &str,
    reader: &GgufTensorDataReader,
    metadata: RuntimeTensorContractMetadata,
) -> Result<BuiltinRuntimeWeightComponents, BuiltinRuntimeWeightComponentRegistryError> {
    match (model_architecture, metadata) {
        (
            crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            RuntimeTensorContractMetadata::CohereTranscribe(metadata),
        ) => {
            let encoder_weights =
                load_cohere_transcribe_encoder_weights_from_reader(reader, metadata).map_err(
                    |error| BuiltinRuntimeWeightComponentRegistryError::MaterializationFailed {
                        component: "cohere-transcribe.encoder-weights",
                        reason: error.to_string(),
                    },
                )?;
            let decoder_weights =
                load_cohere_transcribe_decoder_weights_for_runtime_from_reader(reader, metadata)
                    .map_err(|error| {
                        BuiltinRuntimeWeightComponentRegistryError::MaterializationFailed {
                            component: "cohere-transcribe.decoder-weights",
                            reason: error.to_string(),
                        }
                    })?;
            Ok(BuiltinRuntimeWeightComponents::CohereTranscribe {
                encoder_weights,
                decoder_weights,
            })
        }
        (
            crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            RuntimeTensorContractMetadata::Qwen3Asr(metadata),
        ) => {
            let audio_encoder_weights =
                load_qwen3_audio_encoder_weights_from_reader(reader, metadata).map_err(
                    |error| BuiltinRuntimeWeightComponentRegistryError::MaterializationFailed {
                        component: "qwen3-asr.audio-encoder-weights",
                        reason: error.to_string(),
                    },
                )?;
            let token_embedding_table =
                load_qwen3_token_embedding_table_from_reader(reader, metadata).map_err(
                    |error| BuiltinRuntimeWeightComponentRegistryError::MaterializationFailed {
                        component: "qwen3-asr.token-embedding",
                        reason: error.to_string(),
                    },
                )?;
            let logits_head =
                load_qwen3_llm_logits_head_from_reader(reader, metadata).map_err(|error| {
                    BuiltinRuntimeWeightComponentRegistryError::MaterializationFailed {
                        component: "qwen3-asr.logits-head",
                        reason: error.to_string(),
                    }
                })?;
            let layer_attention_projections =
                load_qwen3_llm_attention_projections_from_reader(reader, metadata).map_err(
                    |error| BuiltinRuntimeWeightComponentRegistryError::MaterializationFailed {
                        component: "qwen3-asr.layer-attention-projections",
                        reason: error.to_string(),
                    },
                )?;
            Ok(BuiltinRuntimeWeightComponents::Qwen3Asr {
                audio_encoder_weights,
                token_embedding_table,
                logits_head,
                layer_attention_projections,
            })
        }
        (crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID, metadata) => Err(
            BuiltinRuntimeWeightComponentRegistryError::MetadataKindMismatch {
                expected_kind: "cohere-transcribe",
                found_kind: metadata_kind_label(metadata),
            },
        ),
        (crate::QWEN3_ASR_GGML_ARCHITECTURE_ID, metadata) => Err(
            BuiltinRuntimeWeightComponentRegistryError::MetadataKindMismatch {
                expected_kind: QWEN3_ASR_MODEL_FAMILY,
                found_kind: metadata_kind_label(metadata),
            },
        ),
        _ => Err(
            BuiltinRuntimeWeightComponentRegistryError::UnknownArchitecture {
                model_architecture: model_architecture.to_string(),
            },
        ),
    }
}

fn metadata_kind_label(metadata: RuntimeTensorContractMetadata) -> &'static str {
    match metadata {
        RuntimeTensorContractMetadata::CohereTranscribe(_) => "cohere-transcribe",
        RuntimeTensorContractMetadata::Qwen3Asr(_) => QWEN3_ASR_MODEL_FAMILY,
    }
}
