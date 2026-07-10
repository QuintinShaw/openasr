use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

use super::CohereTranscribeFrontendPlan;
use super::decoder_weights::CohereTranscribeDecoderWeights;
use super::encoder_weights::CohereTranscribeEncoderWeights;
use super::prompt::{
    CohereTranscribeDecodePrompt, CohereTranscribeDecodePromptError,
    build_cohere_transcribe_decode_prompt,
};
use super::runtime_contract::CohereTranscribeExecutionMetadata;
use super::tokenizer::CohereTranscribeTokenizer;
use crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID;
use crate::models::ggml_asr_executor::{GgmlAsrExecutionOptions, GgmlAsrRuntimeSourcePreflight};
use crate::models::runtime_component_bootstrap::{
    BuiltinRuntimeComponentBootstrap, BuiltinRuntimeComponentBootstrapError,
    BuiltinTokenizerMaterializationMode, build_builtin_runtime_component_bootstrap,
};
use crate::models::runtime_weight_component_registry::{
    BuiltinRuntimeWeightComponentRegistryError, materialize_builtin_runtime_weight_components,
};

#[derive(Debug, Clone)]
pub(crate) struct CoherePreparedRuntime {
    pub metadata: CohereTranscribeExecutionMetadata,
    pub tokenizer: Arc<CohereTranscribeTokenizer>,
    pub frontend_plan: CohereTranscribeFrontendPlan,
    pub encoder_weights: Arc<CohereTranscribeEncoderWeights>,
    pub decoder_weights: Arc<CohereTranscribeDecoderWeights>,
}

impl CoherePreparedRuntime {
    pub(crate) fn decode_prompt(
        &self,
        language: Option<&str>,
        options: &GgmlAsrExecutionOptions,
    ) -> Result<CohereTranscribeDecodePrompt, CohereTranscribeDecodePromptError> {
        build_cohere_transcribe_decode_prompt(
            &self.tokenizer,
            self.metadata.decoder_start_token_id,
            language,
            options,
        )
    }
}

#[derive(Debug, Error)]
pub(crate) enum CoherePreparedRuntimeError {
    #[error("cohere-transcribe runtime contract check failed: {reason}")]
    RuntimeContractViolation { reason: String },
    #[error("cohere-transcribe runtime tensor reader build failed: {reason}")]
    TensorReaderBuildFailed { reason: String },
    #[error("cohere-transcribe tokenizer materialization failed: {reason}")]
    TokenizerBuildFailed { reason: String },
    #[error("cohere-transcribe frontend plan build failed: {reason}")]
    FrontendPlanBuildFailed { reason: String },
    #[error("cohere-transcribe encoder weight build failed: {reason}")]
    EncoderWeightsBuildFailed { reason: String },
    #[error("cohere-transcribe decoder weight build failed: {reason}")]
    DecoderWeightsBuildFailed { reason: String },
}

pub(crate) fn build_cohere_prepared_runtime(
    preflight: &GgmlAsrRuntimeSourcePreflight,
) -> Result<CoherePreparedRuntime, CoherePreparedRuntimeError> {
    let components = build_builtin_runtime_component_bootstrap(
        COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
        preflight,
        BuiltinTokenizerMaterializationMode::Required,
    )
    .map_err(map_runtime_component_bootstrap_error)?;
    build_cohere_prepared_runtime_from_components(components)
}

pub(crate) fn build_cohere_prepared_runtime_from_components(
    components: BuiltinRuntimeComponentBootstrap,
) -> Result<CoherePreparedRuntime, CoherePreparedRuntimeError> {
    let debug_timings = std::env::var_os("OPENASR_COHERE_DEBUG_TIMINGS").is_some();
    let runtime_metadata = components.metadata;
    let metadata = runtime_metadata
        .into_cohere_transcribe()
        .expect("cohere bootstrap must carry cohere metadata");
    let tensor_reader = components.tensor_reader;
    let tokenizer_start = Instant::now();
    let tokenizer = components
        .tokenizer
        .expect("cohere component bootstrap must materialize tokenizer")
        .into_cohere_transcribe()
        .expect("cohere component bootstrap must return cohere tokenizer");
    if debug_timings {
        eprintln!(
            "openasr cohere prepared-runtime: stage=tokenizer elapsed_ms={:.2}",
            tokenizer_start.elapsed().as_secs_f64() * 1000.0
        );
    }
    let frontend_start = Instant::now();
    let frontend_plan = components
        .audio_frontend
        .into_cohere_transcribe()
        .expect("cohere component bootstrap must return cohere frontend plan");
    if debug_timings {
        eprintln!(
            "openasr cohere prepared-runtime: stage=frontend_plan elapsed_ms={:.2}",
            frontend_start.elapsed().as_secs_f64() * 1000.0
        );
    }
    let weights_start = Instant::now();
    let (encoder_weights, decoder_weights) = materialize_builtin_runtime_weight_components(
        COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
        &tensor_reader,
        runtime_metadata,
    )
    .map_err(map_runtime_weight_component_error)?
    .into_cohere_transcribe()
    .expect("cohere weight registry must return cohere weights");
    if debug_timings {
        eprintln!(
            "openasr cohere prepared-runtime: stage=weights elapsed_ms={:.2}",
            weights_start.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(CoherePreparedRuntime {
        metadata,
        tokenizer: Arc::new(tokenizer),
        frontend_plan,
        encoder_weights: Arc::new(encoder_weights),
        decoder_weights: Arc::new(decoder_weights),
    })
}

fn map_runtime_weight_component_error(
    error: BuiltinRuntimeWeightComponentRegistryError,
) -> CoherePreparedRuntimeError {
    match error {
        BuiltinRuntimeWeightComponentRegistryError::MaterializationFailed {
            component: "cohere-transcribe.decoder-weights",
            reason,
        } => CoherePreparedRuntimeError::DecoderWeightsBuildFailed { reason },
        BuiltinRuntimeWeightComponentRegistryError::MaterializationFailed { reason, .. } => {
            CoherePreparedRuntimeError::EncoderWeightsBuildFailed { reason }
        }
        other => CoherePreparedRuntimeError::RuntimeContractViolation {
            reason: other.to_string(),
        },
    }
}

fn map_runtime_component_bootstrap_error(
    error: BuiltinRuntimeComponentBootstrapError,
) -> CoherePreparedRuntimeError {
    match error {
        BuiltinRuntimeComponentBootstrapError::RuntimeAssetBootstrap { source } => match source {
            crate::models::runtime_asset_bootstrap::BuiltinRuntimeAssetBootstrapError::RuntimeContractPreflight { source } => {
            CoherePreparedRuntimeError::RuntimeContractViolation {
                reason: source.to_string(),
            }
        }
            crate::models::runtime_asset_bootstrap::BuiltinRuntimeAssetBootstrapError::TensorReaderBuild { source } => {
            CoherePreparedRuntimeError::TensorReaderBuildFailed {
                reason: source.to_string(),
            }
        }
        },
        BuiltinRuntimeComponentBootstrapError::TokenizerMaterialization { source } => {
            CoherePreparedRuntimeError::TokenizerBuildFailed {
                reason: source.to_string(),
            }
        }
        BuiltinRuntimeComponentBootstrapError::AudioFrontendMaterialization { source } => {
            CoherePreparedRuntimeError::FrontendPlanBuildFailed {
                reason: source.to_string(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ggml_asr_executor::GgmlAsrExecutionOptions;
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
    use crate::validate_ggml_runtime_source_path;
    use crate::{
        read_gguf_metadata_from_runtime_source, read_gguf_tensor_index_from_runtime_source,
    };
    use std::sync::Arc;
    use tempfile::{NamedTempFile, TempPath};

    fn write_runtime_ready_preflight() -> (TempPath, GgmlAsrRuntimeSourcePreflight) {
        let file = NamedTempFile::new().expect("temp file");
        let persisted = file.into_temp_path();
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");

        let runtime_source =
            validate_ggml_runtime_source_path(&persisted).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        (
            persisted,
            GgmlAsrRuntimeSourcePreflight {
                runtime_source,
                metadata: Arc::new(metadata),
                tensor_index: Arc::new(tensor_index),
            },
        )
    }

    #[test]
    fn builds_runtime_ready_assets_from_preflight() {
        let (_runtime_path, preflight) = write_runtime_ready_preflight();
        let runtime = build_cohere_prepared_runtime(&preflight).expect("prepared runtime");

        assert_eq!(runtime.metadata.encoder_layers, 2);
        assert_eq!(runtime.frontend_plan.n_mels, 32);
        assert_eq!(runtime.frontend_plan.window.len(), 400);
        assert_eq!(runtime.frontend_plan.mel_filters.len(), 32 * 201);
        assert_eq!(runtime.encoder_weights.layers.len(), 2);
        assert_eq!(runtime.decoder_weights.layers.len(), 2);
        assert_eq!(
            runtime.tokenizer.token_id_by_content("<|endoftext|>"),
            Some(8)
        );
    }

    #[test]
    fn prepared_runtime_builds_default_decode_prompt() {
        let (_runtime_path, preflight) = write_runtime_ready_preflight();
        let runtime = build_cohere_prepared_runtime(&preflight).expect("prepared runtime");
        let prompt = runtime
            .decode_prompt(Some("en"), &GgmlAsrExecutionOptions::default())
            .expect("prompt");

        assert_eq!(prompt.token_ids, vec![0, 1, 2, 3, 3, 4, 5, 6, 7]);
        assert_eq!(prompt.eos_token_id, Some(8));
    }

    #[test]
    fn prepared_runtime_rejects_runtime_contract_violation() {
        let file = NamedTempFile::new().expect("temp file");
        let persisted = file.into_temp_path();
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture")
            .without_tensor("enc.proj.bias");
        write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");

        let runtime_source =
            validate_ggml_runtime_source_path(&persisted).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        let preflight = GgmlAsrRuntimeSourcePreflight {
            runtime_source,
            metadata: Arc::new(metadata),
            tensor_index: Arc::new(tensor_index),
        };

        let error = build_cohere_prepared_runtime(&preflight)
            .expect_err("invalid runtime must fail closed");
        assert!(matches!(
            error,
            CoherePreparedRuntimeError::RuntimeContractViolation { .. }
        ));
    }
}
