//! firered-aed dedicated executor (Stage 4): fbank+CMVN [`frontend`] -> the
//! parity-verified Conformer [`encoder_graph`] -> greedy attention
//! [`decoder_graph`] -> char+SPM [`tokenizer`] detokenize. No CTC branch, no
//! phrase bias (pure autoregressive attention decode), single-segment plain
//! transcription. The executor fails closed with typed errors on a bad pack
//! and never fabricates a transcript.
//!
//! [`frontend`]: super::frontend
//! [`encoder_graph`]: super::encoder_graph
//! [`decoder_graph`]: super::decoder_graph
//! [`tokenizer`]: super::tokenizer

#![allow(dead_code)]

use thiserror::Error;

use crate::NativeAsrSession;
use crate::api::backend::{Segment, Transcription};
use crate::arch::FIRERED_AED_GGML_ADAPTER_ID;
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutionResult, GgmlAsrExecutor,
    GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT, build_seq2seq_streaming_session,
};
use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;

use super::decoder_graph::run_firered_aed_decoder_greedy;
use super::encoder_graph::encode_firered_aed_audio_embeddings;
use super::frontend::{FireRedFbankFrontend, apply_cmvn};
use super::runtime_contract::parse_firered_aed_execution_metadata;
use super::tokenizer::FireRedTokenizer;

const FIRERED_AED_EXECUTOR_ID: &str = "firered-aed-ggml-executor-v1";
const FIRERED_AED_STREAMING_EXECUTOR_ID: &str = "firered-aed-ggml-snapshot-streaming-executor-v1";
const CMVN_NEG_MEAN_TENSOR: &str = "frontend.cmvn.neg_mean";
const CMVN_INV_STDDEV_TENSOR: &str = "frontend.cmvn.inv_stddev";
const TOKENIZER_TOKENS_KEY: &str = "tokenizer.ggml.tokens";

#[derive(Debug, Error)]
enum FireRedAedExecutorError {
    #[error("firered-aed executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("firered-aed executor runtime preflight failed: {reason}")]
    RuntimePreflightFailed { reason: String },
    #[error("firered-aed runtime metadata contract failed: {reason}")]
    RuntimeContractViolation { reason: String },
    #[error("firered-aed tokenizer materialization failed: {reason}")]
    TokenizerBuildFailed { reason: String },
    #[error("firered-aed cmvn vectors failed: {reason}")]
    CmvnBuildFailed { reason: String },
    #[error("firered-aed frontend failed: {reason}")]
    FrontendFailed { reason: String },
    #[error("firered-aed encoder failed: {reason}")]
    EncoderFailed { reason: String },
    #[error("firered-aed decoder failed: {reason}")]
    DecoderFailed { reason: String },
}

#[derive(Debug, Default, Clone)]
pub(crate) struct FireRedAedGgmlExecutor;

impl FireRedAedGgmlExecutor {
    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, FireRedAedExecutorError> {
        if request.selected_family.adapter_id != FIRERED_AED_GGML_ADAPTER_ID {
            return Err(FireRedAedExecutorError::AdapterMismatch {
                expected: FIRERED_AED_GGML_ADAPTER_ID,
                found: request.selected_family.adapter_id.to_string(),
            });
        }
        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(|error| FireRedAedExecutorError::RuntimePreflightFailed {
                reason: error.to_string(),
            })?;
        let metadata =
            parse_firered_aed_execution_metadata(&preflight.metadata).map_err(|error| {
                FireRedAedExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let tokens = preflight
            .metadata
            .get_string_array(TOKENIZER_TOKENS_KEY)
            .ok_or_else(|| FireRedAedExecutorError::TokenizerBuildFailed {
                reason: "pack missing tokenizer.ggml.tokens".to_string(),
            })?
            .to_vec();
        let tokenizer = FireRedTokenizer::new(tokens);

        let reader = build_runtime_tensor_reader_from_preflight(&preflight).map_err(|error| {
            FireRedAedExecutorError::CmvnBuildFailed {
                reason: error.to_string(),
            }
        })?;
        let feature_dim_shape = [metadata.feature_dim as u64];
        let neg_mean = reader
            .host_tensor_f32_copy_dequantized_by_name(CMVN_NEG_MEAN_TENSOR, &feature_dim_shape)
            .map_err(|error| FireRedAedExecutorError::CmvnBuildFailed {
                reason: error.to_string(),
            })?;
        let inv_stddev = reader
            .host_tensor_f32_copy_dequantized_by_name(CMVN_INV_STDDEV_TENSOR, &feature_dim_shape)
            .map_err(|error| FireRedAedExecutorError::CmvnBuildFailed {
                reason: error.to_string(),
            })?;

        let samples = &request.prepared_audio.samples_f32;
        let frontend = FireRedFbankFrontend::new();
        let mut features =
            frontend
                .compute(samples)
                .map_err(|error| FireRedAedExecutorError::FrontendFailed {
                    reason: error.to_string(),
                })?;
        apply_cmvn(&mut features.data, features.n_mels, &neg_mean, &inv_stddev).map_err(
            |error| FireRedAedExecutorError::FrontendFailed {
                reason: error.to_string(),
            },
        )?;

        let runtime_path = preflight.runtime_source.path();
        let encoder_output = encode_firered_aed_audio_embeddings(
            runtime_path,
            metadata,
            &features.data,
            features.n_frames,
        )
        .map_err(|error| FireRedAedExecutorError::EncoderFailed {
            reason: error.to_string(),
        })?;

        let decode = run_firered_aed_decoder_greedy(
            runtime_path,
            metadata,
            &encoder_output.rows,
            encoder_output.frame_count,
            |ids| tokenizer.decode(ids).map_err(|error| error.to_string()),
        )
        .map_err(|error| FireRedAedExecutorError::DecoderFailed {
            reason: error.to_string(),
        })?;

        let audio_duration_seconds =
            samples.len() as f32 / request.prepared_audio.sample_rate_hz.max(1) as f32;
        let text = decode.text.trim().to_string();
        let transcription = Transcription {
            segments: vec![Segment {
                start: 0.0,
                end: audio_duration_seconds.max(0.0),
                text: text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            }],
            text,
            longform: None,
            language: None,
        };
        Ok(GgmlAsrExecutionResult {
            transcription,
            carry_context: None,
        })
    }
}

impl GgmlAsrExecutor for FireRedAedGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        FIRERED_AED_EXECUTOR_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        false
    }

    fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request)
            .map_err(|error| GgmlAsrExecutionError::ExecutorFailed {
                executor_id: GgmlAsrExecutor::executor_id(self),
                adapter_id: request.selected_family.adapter_id,
                reason: error.to_string(),
            })
    }
}

impl GgmlAsrStreamingExecutor for FireRedAedGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        FIRERED_AED_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        build_seq2seq_streaming_session(
            self.clone(),
            FIRERED_AED_STREAMING_EXECUTOR_ID,
            FIRERED_AED_GGML_ADAPTER_ID,
            "firered-aed",
            request,
            STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT,
            FireRedAedGgmlExecutor::execute,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::models::ggml_asr_executor::{GgmlAsrBackendPreference, GgmlAsrPreparedAudio};
    use crate::models::ggml_family_registry::firered_aed_runtime_descriptor_v1;

    use super::*;

    // Pinned to the reference PyTorch decode captured by the dev-only
    // `tmp/firered-ref-src` harness (see the Stage 1-2 module docs); the
    // fp16 pack itself is a private, non-committed dev artifact.
    const GOLDEN_JFK_TEXT: &str = "AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO \
         FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY";

    fn dev_pack_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/firered-out/firered-aed-l-fp16.oasr")
    }

    fn jfk_wav_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav")
    }

    #[test]
    #[ignore = "requires the private dev-only firered-aed-l-fp16.oasr pack; see module docs"]
    fn golden_diff_end_to_end_transcribe_matches_reference_pytorch_decode_on_jfk_wav() {
        let pack_path = dev_pack_path();
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return;
        }
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            jfk_wav_path(),
            "firered-aed golden test",
            "firered-aed golden test",
        )
        .expect("load jfk.wav");

        let request = GgmlAsrExecutionRequest {
            runtime_source_path: pack_path,
            runtime_source_preflight: None,
            selected_family: firered_aed_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(samples),
            request_options: Default::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
        };

        let executor = FireRedAedGgmlExecutor;
        let result = executor.execute(&request).expect("firered-aed transcribe");
        assert_eq!(result.transcription.text, GOLDEN_JFK_TEXT);
    }
}
