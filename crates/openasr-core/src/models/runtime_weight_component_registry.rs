//! Runtime weight-bundle materialization for the **data-driven composer**
//! family only (Qwen3-ASR). Called solely from that family's
//! `prepared_runtime` path — never generically across architectures.
//!
//! The dedicated-executor families (Whisper, Moonshine, Parakeet-CTC, wav2vec2/
//! data2vec-CTC) materialize their weights in their own family modules and never
//! reach this enum; that boundary is enforced at the frontend chokepoint (see
//! [`super::frontend_component_registry`]). Callers pass the typed reusable
//! component strategy selected from the family inventory; this registry never
//! switches on family identity itself.

use thiserror::Error;

use crate::GgufTensorDataReader;
use crate::arch::OpenAsrPreparedRuntimeStrategy;
use crate::models::mapped_token_embedding::MappedTokenEmbeddingTable;

use super::qwen::{
    DEFAULT_RMS_NORM_EPSILON, Qwen3AsrAudioEncoderWeights, Qwen3AsrLlmLogitsHead, QwenDecoderTail,
    QwenDecoderTailLoadError, QwenWholeDecoderPlan, load_qwen_decoder_tail_from_contract,
    load_qwen3_audio_encoder_weights_from_reader,
};
use super::runtime_tensor_contract_registry::RuntimeTensorContractMetadata;

// Per-family weight bundles differ in size (qwen carries audio-encoder + LLM
// layer projections); this enum is materialized once and held behind an `Arc`,
// so the variant-size delta never lands on the stack — boxing would only add an
// indirection for no benefit.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub(crate) enum BuiltinRuntimeWeightComponents {
    Qwen3Asr {
        audio_encoder_weights: Qwen3AsrAudioEncoderWeights,
        token_embedding_table: MappedTokenEmbeddingTable,
        logits_head: Qwen3AsrLlmLogitsHead,
        decoder_plan: QwenWholeDecoderPlan,
    },
}

impl BuiltinRuntimeWeightComponents {
    pub(crate) fn into_qwen3_asr(
        self,
    ) -> Option<(
        Qwen3AsrAudioEncoderWeights,
        MappedTokenEmbeddingTable,
        Qwen3AsrLlmLogitsHead,
        QwenWholeDecoderPlan,
    )> {
        match self {
            Self::Qwen3Asr {
                audio_encoder_weights,
                token_embedding_table,
                logits_head,
                decoder_plan,
            } => Some((
                audio_encoder_weights,
                token_embedding_table,
                logits_head,
                decoder_plan,
            )),
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum BuiltinRuntimeWeightComponentRegistryError {
    #[error("a family-owned runtime cannot enter the shared composer registry")]
    FamilyOwnedRuntime,
    #[error("builtin runtime weights materialization failed for '{component}': {reason}")]
    MaterializationFailed {
        component: &'static str,
        reason: String,
    },
}

pub(crate) fn materialize_builtin_runtime_weight_components(
    strategy: OpenAsrPreparedRuntimeStrategy,
    reader: &GgufTensorDataReader,
    _runtime_source: &crate::GgmlRuntimeSource,
    metadata: RuntimeTensorContractMetadata,
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
) -> Result<BuiltinRuntimeWeightComponents, BuiltinRuntimeWeightComponentRegistryError> {
    match (strategy, metadata) {
        (
            OpenAsrPreparedRuntimeStrategy::SharedQwen3AsrV1,
            RuntimeTensorContractMetadata::Qwen3Asr {
                metadata,
                decoder_contract,
            },
        ) => {
            let audio_encoder_weights =
                load_qwen3_audio_encoder_weights_from_reader(reader, metadata).map_err(
                    |error| BuiltinRuntimeWeightComponentRegistryError::MaterializationFailed {
                        component: "qwen3-asr.audio-encoder-weights",
                        reason: error.to_string(),
                    },
                )?;
            // Weight materialization here is cached per (architecture, pack)
            // and reused across every request regardless of which request's
            // resolved backend triggered the (first, cold) build -- so
            // `backend` is the resolved value of whichever request happened
            // to populate this cache slot, threaded down explicitly from
            // that request's own `resolved_runtime`, never re-derived here.
            let QwenDecoderTail {
                logits_head,
                token_embedding: token_embedding_table,
            } = load_qwen_decoder_tail_from_contract(
                reader,
                &decoder_contract,
                DEFAULT_RMS_NORM_EPSILON,
                backend,
            )
            .map_err(|error| {
                let component = match &error {
                    QwenDecoderTailLoadError::TokenEmbedding(_) => "qwen3-asr.token-embedding",
                    QwenDecoderTailLoadError::LogitsHead(_) => "qwen3-asr.logits-head",
                    QwenDecoderTailLoadError::Contract { .. } => "qwen3-asr.decoder-tail",
                };
                BuiltinRuntimeWeightComponentRegistryError::MaterializationFailed {
                    component,
                    reason: error.to_string(),
                }
            })?;
            let decoder_plan = QwenWholeDecoderPlan::for_qwen_family(reader, &decoder_contract)
                .map_err(|error| {
                    BuiltinRuntimeWeightComponentRegistryError::MaterializationFailed {
                        component: "qwen3-asr.layer-attention-projections",
                        reason: error.to_string(),
                    }
                })?;
            Ok(BuiltinRuntimeWeightComponents::Qwen3Asr {
                audio_encoder_weights,
                token_embedding_table,
                logits_head,
                decoder_plan,
            })
        }
        (OpenAsrPreparedRuntimeStrategy::FamilyOwned, _) => {
            Err(BuiltinRuntimeWeightComponentRegistryError::FamilyOwnedRuntime)
        }
        (_, _) => Err(BuiltinRuntimeWeightComponentRegistryError::FamilyOwnedRuntime),
    }
}
