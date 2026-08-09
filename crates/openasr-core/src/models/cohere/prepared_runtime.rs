use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

use super::decoder_weights::CohereTranscribeDecoderWeights;
use super::encoder_weights::CohereTranscribeEncoderWeights;
use super::frontend::CohereTranscribeFrontendPlan;
use super::prompt::{
    CohereTranscribeDecodePrompt, CohereTranscribeDecodePromptError,
    build_cohere_transcribe_decode_prompt,
};
use super::runtime_contract::CohereTranscribeExecutionMetadata;
use super::tokenizer::CohereTranscribeTokenizer;
use crate::ggml_runtime::GgufRuntimeSourcePreflight;
use crate::models::ggml_asr_executor::GgmlAsrExecutionOptions;
use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;

use super::decoder_weights::load_cohere_transcribe_decoder_weights_for_runtime_from_reader;
use super::encoder_weights::load_cohere_transcribe_encoder_weights_from_reader;
use super::frontend::load_cohere_transcribe_frontend_plan_from_reader;
use super::runtime_contract::{
    parse_cohere_transcribe_execution_metadata,
    validate_cohere_transcribe_runtime_tensors_with_index,
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
    pub(crate) fn system_memory_quote(
        context: crate::models::prepared_runtime_cache::PreparedRuntimeQuoteContext<'_>,
        pack_content_id: &str,
    ) -> Result<
        crate::models::system_memory_owner::SystemMemoryAllocationQuote,
        crate::models::system_memory_owner::SystemMemoryOwnerError,
    > {
        let mut quote = crate::models::prepared_runtime_cache::PreparedRuntimeQuoteBuilder::new::<
            Self,
        >(pack_content_id);
        quote.add_tokenizer_metadata(context.metadata, true)?;
        for tensor in context.tensor_index.tensors() {
            // Cohere's runtime-ready matrices/vectors retain mmap-backed owned
            // payload descriptors, not copied tensor bytes. Encoder frontend,
            // bias/norm/statistic tensors and the few non-runtime projections
            // are the explicit f32 materialization set.
            add_cohere_owned_payload_metadata(&mut quote, tensor)?;
            if cohere_tensor_materializes_f32(&tensor.name) {
                quote.add_tensor_f32(context.tensor_index, &tensor.name)?;
            }
        }
        quote.finish()
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add(
            self.tokenizer.retained_system_memory_bytes()?,
            "cohere prepared tokenizer",
        )?;
        bytes.add(
            self.frontend_plan.retained_system_memory_bytes()?,
            "cohere prepared frontend",
        )?;
        bytes.add(
            self.encoder_weights.retained_system_memory_bytes()?,
            "cohere prepared encoder weights",
        )?;
        bytes.add(
            self.decoder_weights.retained_system_memory_bytes()?,
            "cohere prepared decoder weights",
        )?;
        Ok(bytes.finish())
    }
}

fn add_cohere_owned_payload_metadata(
    quote: &mut crate::models::prepared_runtime_cache::PreparedRuntimeQuoteBuilder,
    tensor: &crate::GgufTensorMetadata,
) -> Result<(), crate::models::system_memory_owner::SystemMemoryOwnerError> {
    use crate::models::system_memory_owner::SystemMemoryOwnerError;

    let name_bytes = u64::try_from(tensor.name.len()).map_err(|_| {
        SystemMemoryOwnerError::capacity_failure(
            "prepared_runtime_quote",
            "cohere tensor name length does not fit u64",
        )
    })?;
    let type_name_bytes = u64::try_from(tensor.type_name.len()).map_err(|_| {
        SystemMemoryOwnerError::capacity_failure(
            "prepared_runtime_quote",
            "cohere tensor type-name length does not fit u64",
        )
    })?;
    let rank = u64::try_from(tensor.dims.len()).map_err(|_| {
        SystemMemoryOwnerError::capacity_failure(
            "prepared_runtime_quote",
            "cohere tensor rank does not fit u64",
        )
    })?;
    // Weight name + owned payload metadata name/type-name. The weight's own
    // dims plus the two payload dim vectors account for every explicitly
    // requested descriptor Vec; mmap tensor bytes are not copied.
    quote.add_owned_bytes(
        name_bytes.checked_mul(2).ok_or_else(|| {
            SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                "cohere owned tensor-name bytes overflowed",
            )
        })?,
        "cohere owned tensor names",
    )?;
    quote.add_owned_bytes(type_name_bytes, "cohere owned tensor type name")?;
    quote.add_owned_elements::<usize>(rank, "cohere weight dims")?;
    quote.add_owned_elements::<usize>(rank, "cohere payload dims")?;
    quote.add_owned_elements::<u64>(rank, "cohere payload metadata dims")
}

fn cohere_tensor_materializes_f32(name: &str) -> bool {
    use super::tensor_names::{
        ENC_PRE_OUT_BIAS, ENC_PRE_OUT_WEIGHT, ENC_PROJ_BIAS, FE_MEL_FB, FE_WINDOW,
    };

    if matches!(
        name,
        FE_MEL_FB | FE_WINDOW | ENC_PRE_OUT_WEIGHT | ENC_PRE_OUT_BIAS | ENC_PROJ_BIAS
    ) {
        return true;
    }
    if name.starts_with("enc.pre.conv.") {
        return name.ends_with(".bias");
    }
    name.starts_with("enc.blk.")
        && (name.contains(".norm.")
            || name.ends_with(".bias")
            || name.ends_with(".mean")
            || name.ends_with(".var")
            || name.ends_with("attn.pos.weight")
            || name.ends_with("attn.pos_bias_u")
            || name.ends_with("attn.pos_bias_v")
            || name.ends_with("conv.dw.weight"))
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
    preflight: &GgufRuntimeSourcePreflight,
    _backend: crate::ggml_runtime::GgmlCpuGraphBackend,
) -> Result<CoherePreparedRuntime, CoherePreparedRuntimeError> {
    let metadata =
        parse_cohere_transcribe_execution_metadata(&preflight.metadata).map_err(|error| {
            CoherePreparedRuntimeError::RuntimeContractViolation {
                reason: error.to_string(),
            }
        })?;
    validate_cohere_transcribe_runtime_tensors_with_index(&preflight.tensor_index, metadata)
        .map_err(
            |error| CoherePreparedRuntimeError::RuntimeContractViolation {
                reason: error.to_string(),
            },
        )?;
    let tensor_reader = build_runtime_tensor_reader_from_preflight(preflight).map_err(|error| {
        CoherePreparedRuntimeError::TensorReaderBuildFailed {
            reason: error.to_string(),
        }
    })?;
    let debug_timings = std::env::var_os("OPENASR_COHERE_DEBUG_TIMINGS").is_some();
    let tokenizer_start = Instant::now();
    let tokenizer =
        CohereTranscribeTokenizer::from_gguf_metadata(&preflight.metadata).map_err(|error| {
            CoherePreparedRuntimeError::TokenizerBuildFailed {
                reason: error.to_string(),
            }
        })?;
    if debug_timings {
        eprintln!(
            "openasr cohere prepared-runtime: stage=tokenizer elapsed_ms={:.2}",
            tokenizer_start.elapsed().as_secs_f64() * 1000.0
        );
    }
    let frontend_start = Instant::now();
    let frontend_plan = load_cohere_transcribe_frontend_plan_from_reader(&tensor_reader, metadata)
        .map_err(
            |error| CoherePreparedRuntimeError::FrontendPlanBuildFailed {
                reason: error.to_string(),
            },
        )?;
    if debug_timings {
        eprintln!(
            "openasr cohere prepared-runtime: stage=frontend_plan elapsed_ms={:.2}",
            frontend_start.elapsed().as_secs_f64() * 1000.0
        );
    }
    let weights_start = Instant::now();
    let encoder_weights =
        load_cohere_transcribe_encoder_weights_from_reader(&tensor_reader, metadata).map_err(
            |error| CoherePreparedRuntimeError::EncoderWeightsBuildFailed {
                reason: error.to_string(),
            },
        )?;
    let decoder_weights =
        load_cohere_transcribe_decoder_weights_for_runtime_from_reader(&tensor_reader, metadata)
            .map_err(
                |error| CoherePreparedRuntimeError::DecoderWeightsBuildFailed {
                    reason: error.to_string(),
                },
            )?;
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

    fn write_runtime_ready_preflight() -> (TempPath, GgufRuntimeSourcePreflight) {
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
            GgufRuntimeSourcePreflight {
                runtime_source,
                metadata: Arc::new(metadata),
                tensor_index: Arc::new(tensor_index),
            },
        )
    }

    #[test]
    fn builds_runtime_ready_assets_from_preflight() {
        let (_runtime_path, preflight) = write_runtime_ready_preflight();
        let runtime = build_cohere_prepared_runtime(
            &preflight,
            crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
        )
        .expect("prepared runtime");

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
        let runtime = build_cohere_prepared_runtime(
            &preflight,
            crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
        )
        .expect("prepared runtime");
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
        let preflight = GgufRuntimeSourcePreflight {
            runtime_source,
            metadata: Arc::new(metadata),
            tensor_index: Arc::new(tensor_index),
        };

        let error = build_cohere_prepared_runtime(
            &preflight,
            crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
        )
        .expect_err("invalid runtime must fail closed");
        assert!(matches!(
            error,
            CoherePreparedRuntimeError::RuntimeContractViolation { .. }
        ));
    }
}
