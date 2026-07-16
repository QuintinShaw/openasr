//! Whisper GGUF execution runtime on top of `GgmlAsrExecutor`.
//!
//! Hands-off: single-responsibility ggml graph transcription, guarded by
//! golden/parity tests. Do not split this module for "tidiness" -- the tensor
//! wiring is validated as a whole and refactoring here risks silent numeric
//! drift.
//!
//! Current fail-closed boundary:
//! - Family descriptor selection (`openasr.*`) proves adapter routing only.
//! - Real Whisper graph lowering still needs Whisper-specific GGUF metadata
//!   (`whisper.encoder.*`, `whisper.decoder.*`, `general.architecture`) and
//!   tensor-name coverage checks.
//! - Encoder prelude has a real planning/build seam (mel input -> conv/positional
//!   prelude graph) with explicit unsupported-primitive failure.
//! - Encoder graph builder lowers Whisper encoder structure
//!   (attn norm -> qkv -> attention -> mlp -> final norm) into a typed plan.
//! - Full Whisper encoder/decoder execution is wired through the decoder graph
//!   greedy step loop and fails closed on decoder/tokenizer boundary errors.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlLoadedTensor, GgmlLoadedWeightContext, GgmlStaticTensor,
    GgmlStaticTensorArena,
};
use crate::models::ggml_asr_executor::GgmlAsrCarryContext;
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_WHISPER_SEQ2SEQ, build_seq2seq_streaming_session,
};
use crate::models::prepared_runtime_cache::PreparedRuntimeCache;
use crate::models::runtime_contract::MetadataContractError;
use crate::models::thread_local_runtime_cache::{
    UnloadGenerationGated, canonical_runtime_cache_path,
};
use crate::models::tokenizer_component_registry::materialize_builtin_tokenizer_for_architecture;
use crate::nn::attn::{
    AttentionHeadLayout, AttentionReshapeSteps, AttentionValueMergeSteps,
    STANDARD_HEAD_PERMUTE_AXES, attention_context_from_probs,
    reshape_projection_to_attention_heads,
};
use crate::nn::conv::{
    Conv1dParams, ConvActivation, ConvBlockSteps, apply_conv_1d_bias_activation,
};
use crate::nn::decoder::{Seq2SeqReusableDecodeGraph, reusable_decode_graph_supported_for_runner};
use crate::nn::half::{f16_bits_to_f32, f32_to_f16_bits};
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};
use crate::{
    GgmlAsrExecutionError, GgmlAsrExecutionOptions, GgmlAsrExecutionRequest,
    GgmlAsrExecutionResult, GgmlAsrExecutor, GgmlAsrPreparedAudio, GgmlAsrStreamingExecutor,
    GgmlAsrStreamingSessionRequest, GgmlFamilyAdapterDescriptor, GgmlRuntimeSource, GgufMetadata,
    GgufTensorDataReadError, GgufTensorDataReader, GgufTensorIndex, NativeAsrSession, Segment,
    Transcription, WHISPER_GGML_ADAPTER_ID,
};
#[cfg(test)]
use crate::{GgufTensorIndexReadError, read_gguf_tensor_index_from_runtime_source};

use super::batched_decode::{
    WhisperServeBatchConfig, WhisperServeBatchConfigFromEnv, WhisperServeBatchJob,
    submit_whisper_serve_batch_job, whisper_serve_batch_decode_config,
};
use super::execution_policy::{
    whisper_decoder_cross_flash_attention_enabled, whisper_decoder_self_flash_attention_enabled,
    whisper_encoder_flash_attention_enabled, whisper_parallel_encoder_and_decoder_static_enabled,
};
use super::execution_trace::{
    OPENASR_WHISPER_GGML_TRACE_ENV, WHISPER_GGML_TRACE_DECODE_STEP_INTERVAL, WhisperGgmlTrace,
};
use super::ggml_decoder_graph::{
    WhisperDecoderExecutionTensorCache, WhisperDecoderGraphExecutionConfig,
    WhisperDecoderGraphExecutionError, WhisperDecoderGraphExecutionInput,
    WhisperDecoderGraphInputShape, WhisperDecoderGraphMetadata, WhisperDecoderGraphPlan,
    WhisperDecoderGraphPlanError, WhisperDecoderGraphTensorRef, WhisperDecoderHiddenStateLayout,
    WhisperDecoderLayerTensorBinding, WhisperDecoderPersistentWeightCache,
    WhisperDecoderSelfKvCacheState, WhisperDecoderTensorBindingSeam,
    WhisperDecoderTensorMaterializationSeam, WhisperDecoderTensorSource,
    build_whisper_decoder_graph_plan,
    run_whisper_decoder_greedy_step_with_cache_and_runner_ggml_v0,
    run_whisper_decoder_reused_incremental_step_ggml_v0,
};
use super::ggml_decoder_weights::{
    WhisperDecoderWeightBundle, WhisperDecoderWeightMaterializationError,
    materialize_whisper_decoder_weight_bundle,
};
use super::ggml_encoder_graph::{
    WhisperEncoderGraphInputShape, WhisperEncoderGraphMetadata, WhisperEncoderGraphPlan,
    WhisperEncoderGraphPlanError, WhisperEncoderGraphTensorRef, WhisperEncoderLayerTensorBinding,
    WhisperEncoderLinearProjectionPlan, WhisperEncoderLinearWeightLayout, WhisperEncoderNormPlan,
    WhisperEncoderTensorBindingSeam, WhisperEncoderTensorMaterializationSeam,
    build_whisper_encoder_graph_plan,
};
use super::ggml_encoder_prelude::{
    WhisperEncoderPreludeConv1dPlan, WhisperEncoderPreludeConv1dWeightLayout,
    WhisperEncoderPreludeInputShape, WhisperEncoderPreludePlan, WhisperEncoderPreludePlanError,
    build_whisper_encoder_prelude_plan,
};
use super::ggml_encoder_weights::{
    WhisperEncoderWeightBundle, WhisperEncoderWeightMaterializationError,
    WhisperMaterializedTensor, WhisperMaterializedTensorPayload,
    materialize_whisper_encoder_weight_bundle,
};
use super::ggml_tensor_binding::{
    WhisperGgufDecoderLayerTensorBindings, WhisperGgufDecoderTensorBindings,
    WhisperGgufTensorBinding, WhisperGgufTensorBindingContext, WhisperGgufTensorBindingError,
    WhisperGgufTensorBindings, bind_whisper_gguf_tensors,
};
use super::graph_config::{
    whisper_decoder_graph_config, whisper_encoder_prelude_graph_config,
    whisper_runtime_graph_config,
};
use super::mel::{
    WHISPER_CHANNELS, WHISPER_SAMPLE_RATE_HZ, whisper_mel_features_from_prepared_audio_v0,
};
use super::runtime_contract::{WhisperGgmlExecutionMetadata, validate_whisper_execution_metadata};
use super::{
    WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT,
    greedy_decode::{WhisperGreedyDecodeError, run_whisper_greedy_decode_loop},
    tokenizer::{WhisperPrefixError, WhisperPrefixSpec, WhisperTokenizer},
};
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicySeq2SeqTextPostprocessKind, BuiltinSeq2SeqDecodePolicyConfigInput,
};
use crate::models::decode_token_history::{
    build_longform_token_history_carry, context_window_budget, trim_prompt_token_tail,
};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeStepExecutor, Seq2SeqGreedyDecodeStepInput,
    Seq2SeqGreedyDecodeStepLogitsOutput,
};
use crate::models::seq2seq_word_timestamps::{
    Seq2SeqTokenTime, seq2seq_word_timestamps_from_generated_tokens,
    seq2seq_word_timestamps_from_token_times,
};

const WHISPER_DEFAULT_DECODE_MAX_GENERATED_TOKENS_CAP: usize = 256;
const WHISPER_STREAMING_EXECUTOR_ID: &str = "whisper-ggml-snapshot-streaming-executor-v1";
/// Largest vocab of an English-only (`.en`) Whisper checkpoint. The canonical
/// Whisper rule (matching whisper.cpp `vocab.is_multilingual()`) is that any
/// checkpoint with a strictly larger vocab carries the language-token block as
/// decode-time prompt state and must be prompted multilingually
/// (`<|sot|> <|en|> <|transcribe|> <|notimestamps|>`). `.en` checkpoints
/// (vocab == this value) keep the bare `<|sot|> <|notimestamps|>` prompt.
pub(crate) const WHISPER_ENGLISH_ONLY_MAX_VOCAB_SIZE: usize = 51_864;
const WHISPER_DECODER_PERSISTENT_SESSION_POOL_CAPACITY: usize = 8;
const WHISPER_ENCODER_LAYER_NORM_EPSILON: f32 = 1.0e-5;

fn whisper_can_use_serve_batch(
    graph_config: GgmlCpuGraphConfig,
    _request_options: &GgmlAsrExecutionOptions,
    _allow_persistent_session_reuse: bool,
) -> bool {
    graph_config.backend.is_gpu_class() && !graph_config.use_scheduler
}

#[derive(Debug, Error)]
pub enum WhisperGgmlExecutorError {
    #[error("whisper ggml executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("whisper ggml executor missing required GGUF metadata key '{key}'")]
    MissingRequiredMetadata { key: &'static str },
    #[error("whisper ggml executor metadata '{key}' is invalid: {reason}")]
    InvalidMetadataValue { key: &'static str, reason: String },
    #[error("whisper ggml executor mel/input preparation seam failed: {reason}")]
    MelFeatureInputPreparationFailed { reason: String },
    #[error("whisper ggml executor mel feature extraction failed: {reason}")]
    MelFeatureExtractionFailed { reason: String },
    #[cfg(test)]
    #[error("whisper ggml executor could not read GGUF tensor index: {source}")]
    TensorIndexRead { source: GgufTensorIndexReadError },
    #[error("whisper ggml executor tensor materialization failed: {reason}")]
    TensorMaterializationFailed { reason: String },
    #[error("whisper ggml executor missing required GGUF tensor '{name}'")]
    MissingRequiredTensor { name: String },
    #[error("whisper ggml executor tensor '{name}' failed binding validation: {reason}")]
    InvalidRequiredTensor { name: String, reason: String },
    #[error(
        "whisper ggml executor encoder prelude primitive '{primitive}' is unsupported: {reason}"
    )]
    EncoderPreludePrimitiveUnsupported {
        primitive: &'static str,
        reason: String,
    },
    #[error("whisper ggml executor encoder prelude graph execution failed: {reason}")]
    EncoderPreludeExecutionFailed { reason: String },
    #[error("whisper ggml executor encoder graph binding seam is unsupported: {reason}")]
    EncoderGraphBindingUnsupported { reason: String },
    #[error("whisper ggml executor encoder graph primitive '{primitive}' is unsupported: {reason}")]
    EncoderGraphPrimitiveUnsupported {
        primitive: &'static str,
        reason: String,
    },
    #[error("whisper ggml executor encoder graph execution failed: {reason}")]
    EncoderGraphExecutionFailed { reason: String },
    #[error("whisper ggml executor tokenizer is missing: {reason}")]
    TokenizerMissing { reason: String },
    #[error("whisper ggml executor cannot honor request option '{option}': {reason}")]
    UnsupportedRequestOption {
        option: &'static str,
        reason: String,
    },
    #[error("whisper ggml executor decoder weights are missing: {reason}")]
    DecoderWeightsMissing { reason: String },
    #[error("whisper ggml executor decoder graph is unsupported: {reason}")]
    DecoderGraphUnsupported { reason: String },
    #[error("whisper ggml executor decoder graph execution failed: {reason}")]
    DecoderGraphExecutionFailed { reason: String },
    #[error(
        "whisper ggml executor decoder loop reached max_generated_tokens={max_generated_tokens} before EOT"
    )]
    DecoderNoEotBeforeMaxTokens { max_generated_tokens: usize },
    #[error("whisper ggml executor decoder token->text decode failed: {reason}")]
    DecoderInvalidTokenDecode { reason: String },
    /// Carries a transient serve-batch failure (queue full / owner gone / reply
    /// timeout) through to the `execute` trait boundary so it can become a
    /// retryable HTTP status instead of a generic 500.
    #[error("{reason}")]
    ServeBatchUnavailable { reason: String, retryable: bool },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WhisperGgmlWeightIndex {
    tensor_index: Arc<GgufTensorIndex>,
    bindings: WhisperGgufTensorBindings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WhisperGgmlTensorBinding {
    weights: WhisperGgmlWeightIndex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WhisperMelFeatureInputShape {
    mel_bins: usize,
    mel_frames: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct WhisperMelFeatureInput {
    source_label: &'static str,
    shape: WhisperMelFeatureInputShape,
    values_f32: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
enum WhisperEncoderPreludeSeamResult {
    GraphExecuted {
        runner_id: &'static str,
        output_frames: usize,
        output_hidden_size: usize,
        output_hidden_f32: Vec<f32>,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum WhisperEncoderGraphSeamResult {
    GraphExecuted {
        runner_id: &'static str,
        layer_count: usize,
        output_frames: usize,
        output_hidden_size: usize,
        output_hidden_f32: Vec<f32>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct WhisperDecoderWeightSeam {
    pub(super) graph_binding: WhisperDecoderTensorBindingSeam,
    pub(super) graph_materialization: WhisperDecoderTensorMaterializationSeam,
    pub(super) tensor_source: WhisperDecoderMaterializedTensorSource,
}

#[derive(Debug, Clone)]
struct WhisperPreparedRuntime {
    execution: WhisperGgmlExecutionMetadata,
    tensor_binding: WhisperGgmlTensorBinding,
    encoder_weights: WhisperEncoderWeightBundle,
    encoder_materialization: WhisperEncoderTensorMaterializationSeam,
    encoder_binding: WhisperEncoderTensorBindingSeam,
    decoder_weights: WhisperDecoderWeightSeam,
    tokenizer: WhisperTokenizer,
}

#[derive(Debug)]
pub(super) struct WhisperExecutionOutput {
    pub(super) text: String,
    pub(super) segments: Vec<Segment>,
    pub(super) carry_prompt_token_ids: Option<Vec<u32>>,
    /// Whisper LID result for an `auto` request on a multilingual pack; `None`
    /// for English-only packs, explicit-language requests, or when detection
    /// failed (fail-open).
    pub(super) detected_language: Option<String>,
}

struct WhisperEncoderPersistentStaticSession {
    runner: GgmlCpuGraphRunner,
    resident_weights: Option<WhisperEncoderResidentWeightCache>,
    graph_config: GgmlCpuGraphConfig,
    encoder_layers: usize,
    encoder_hidden_size: usize,
}

struct WhisperDecoderPersistentStaticSession {
    runner: GgmlCpuGraphRunner,
    cache: WhisperDecoderPersistentWeightCache,
    reuse: Option<Seq2SeqReusableDecodeGraph>,
    graph_config: GgmlCpuGraphConfig,
    plan: WhisperDecoderGraphPlan,
}

type WhisperEncoderPersistentSessionKey = (PathBuf, GgmlCpuGraphBackend);
type WhisperDecoderPersistentSessionKey = (PathBuf, GgmlCpuGraphBackend);

thread_local! {
    // Gated on the idle-unload generation: these sessions pin the resident
    // encoder/decoder weight caches in the TLS of a reused spawn_blocking
    // thread, where the idle-unload reaper cannot drop them, so every access
    // goes through `synced()` and discards pre-unload sessions on the owning
    // thread instead of reusing them.
    static WHISPER_ENCODER_PERSISTENT_SESSION_BY_KEY:
        RefCell<UnloadGenerationGated<HashMap<WhisperEncoderPersistentSessionKey, WhisperEncoderPersistentStaticSession>>> =
            RefCell::new(UnloadGenerationGated::new());
    static WHISPER_DECODER_PERSISTENT_SESSION_BY_KEY:
        RefCell<UnloadGenerationGated<HashMap<WhisperDecoderPersistentSessionKey, Vec<WhisperDecoderPersistentStaticSession>>>> =
            RefCell::new(UnloadGenerationGated::new());
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WhisperDecoderStepSeamInput {
    encoder_frames: usize,
    encoder_hidden_size: usize,
    step_index: usize,
    position_offset: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct WhisperDecoderMaterializedTensorSource {
    tensors_f32_by_name: HashMap<String, Arc<[f32]>>,
    tensors_f16_bits_by_name: HashMap<String, Arc<[u16]>>,
    tensors_quantized_by_name: HashMap<String, (i32, Arc<[u8]>)>,
}

impl WhisperDecoderTensorSource for WhisperDecoderMaterializedTensorSource {
    fn materialize_tensor_f32(
        &self,
        tensor: &WhisperDecoderGraphTensorRef,
    ) -> Result<Vec<f32>, WhisperDecoderGraphExecutionError> {
        let Some(values) = self.tensors_f32_by_name.get(&tensor.tensor_name) else {
            let Some(values) = self.tensors_f16_bits_by_name.get(&tensor.tensor_name) else {
                return Err(
                    WhisperDecoderGraphExecutionError::MissingMaterializedTensor {
                        tensor_name: tensor.tensor_name.clone(),
                        reason: "tensor is absent from decoder materialization seam".to_string(),
                    },
                );
            };
            return Ok(values.iter().map(|bits| f16_bits_to_f32(*bits)).collect());
        };
        Ok(values.to_vec())
    }

    fn materialize_tensor_f32_arc(
        &self,
        tensor: &WhisperDecoderGraphTensorRef,
    ) -> Result<Arc<[f32]>, WhisperDecoderGraphExecutionError> {
        let Some(values) = self.tensors_f32_by_name.get(&tensor.tensor_name) else {
            let Some(values) = self.tensors_f16_bits_by_name.get(&tensor.tensor_name) else {
                return Err(
                    WhisperDecoderGraphExecutionError::MissingMaterializedTensor {
                        tensor_name: tensor.tensor_name.clone(),
                        reason: "tensor is absent from decoder materialization seam".to_string(),
                    },
                );
            };
            return Ok(Arc::<[f32]>::from(
                values
                    .iter()
                    .map(|bits| f16_bits_to_f32(*bits))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ));
        };
        Ok(Arc::clone(values))
    }

    fn materialize_tensor_f16_bits(
        &self,
        tensor: &WhisperDecoderGraphTensorRef,
    ) -> Result<Option<Vec<u16>>, WhisperDecoderGraphExecutionError> {
        Ok(self
            .tensors_f16_bits_by_name
            .get(&tensor.tensor_name)
            .map(|values| values.to_vec()))
    }

    fn materialize_tensor_f16_bits_arc(
        &self,
        tensor: &WhisperDecoderGraphTensorRef,
    ) -> Result<Option<Arc<[u16]>>, WhisperDecoderGraphExecutionError> {
        Ok(self
            .tensors_f16_bits_by_name
            .get(&tensor.tensor_name)
            .map(Arc::clone))
    }

    fn materialize_tensor_quantized(
        &self,
        tensor: &WhisperDecoderGraphTensorRef,
    ) -> Result<Option<(i32, Vec<u8>)>, WhisperDecoderGraphExecutionError> {
        Ok(self
            .tensors_quantized_by_name
            .get(&tensor.tensor_name)
            .map(|(ggml_type, values)| (*ggml_type, values.to_vec())))
    }

    fn materialize_tensor_quantized_arc(
        &self,
        tensor: &WhisperDecoderGraphTensorRef,
    ) -> Result<Option<(i32, Arc<[u8]>)>, WhisperDecoderGraphExecutionError> {
        Ok(self
            .tensors_quantized_by_name
            .get(&tensor.tensor_name)
            .map(|(ggml_type, values)| (*ggml_type, Arc::clone(values))))
    }
}

trait WhisperEncoderPreludeRunner: Send + Sync {
    fn runner_id(&self) -> &'static str;
    fn run_encoder_prelude(
        &self,
        _runtime_source: &GgmlRuntimeSource,
        encoder_weights: &WhisperEncoderWeightBundle,
        plan: &WhisperEncoderPreludePlan,
        mel_input: &WhisperMelFeatureInput,
    ) -> Result<WhisperEncoderPreludeSeamResult, WhisperGgmlExecutorError>;
}

trait WhisperEncoderGraphRunner: Send + Sync {
    fn runner_id(&self) -> &'static str;
    fn run_encoder_graph(
        &self,
        _runtime_source: &GgmlRuntimeSource,
        execution: &WhisperGgmlExecutionMetadata,
        encoder_weights: &WhisperEncoderWeightBundle,
        plan: &WhisperEncoderGraphPlan,
        encoder_hidden_input_f32: &[f32],
    ) -> Result<WhisperEncoderGraphSeamResult, WhisperGgmlExecutorError>;
}

trait WhisperMelFeatureInputProvider: Send + Sync {
    fn provider_id(&self) -> &'static str;
    fn prepare_mel_feature_input(
        &self,
        execution: &WhisperGgmlExecutionMetadata,
        prepared_audio: &GgmlAsrPreparedAudio,
    ) -> Result<WhisperMelFeatureInput, WhisperGgmlExecutorError>;
}

trait WhisperDecoderLoopRunner: Send + Sync {
    fn runner_id(&self) -> &'static str;
    #[allow(clippy::too_many_arguments)]
    fn step_logits(
        &self,
        runtime_source: &GgmlRuntimeSource,
        execution: &WhisperGgmlExecutionMetadata,
        decoder_weights: &WhisperDecoderWeightSeam,
        plan: &WhisperDecoderGraphPlan,
        graph_input: &WhisperDecoderGraphExecutionInput,
        graph_config: WhisperDecoderGraphExecutionConfig,
        graph_runner: &mut GgmlCpuGraphRunner,
        persistent_weights: Option<&WhisperDecoderPersistentWeightCache>,
        self_kv_state: Option<&WhisperDecoderSelfKvCacheState>,
        tensor_cache: &mut WhisperDecoderExecutionTensorCache,
        decode_input: &WhisperDecoderStepSeamInput,
    ) -> Result<WhisperDecoderStepLogits, WhisperGgmlExecutorError>;
}

trait WhisperTokenizerProvider: Send + Sync {
    fn provider_id(&self) -> &'static str;
    fn load_tokenizer(
        &self,
        _runtime_source: &GgmlRuntimeSource,
        _metadata: &GgufMetadata,
    ) -> Result<WhisperTokenizer, WhisperGgmlExecutorError>;
}

#[derive(Debug, Default, Clone, Copy)]
struct WhisperCpuEncoderPreludeComputeRunnerV0;

#[derive(Debug, Default, Clone, Copy)]
struct WhisperCpuEncoderGraphComputeRunnerV0;

#[derive(Debug, Default, Clone, Copy)]
struct WhisperMelFeatureInputProviderFrontendV0;

#[derive(Debug, Default, Clone, Copy)]
struct WhisperDecoderGraphRunnerGgmlV0;

#[derive(Debug, Default, Clone, Copy)]
struct WhisperTokenizerProviderGgufV0;

#[derive(Debug, Clone, PartialEq)]
struct WhisperDecoderStepLogits {
    logits: Vec<f32>,
    greedy_token_hint: Option<u32>,
    last_token_cross_attention_frame_probs: Option<Vec<f32>>,
    decoder_graph_run_ms: u128,
    logits_ms: u128,
}

#[derive(Debug, Clone, PartialEq)]
struct WhisperGeneratedTokenAlignment {
    token_id: u32,
    frame_probs: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhisperDecoderStepPlanCacheStatus {
    Hit,
    Miss,
}

impl WhisperDecoderStepPlanCacheStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Miss => "miss",
        }
    }
}

#[derive(Debug, Clone)]
struct WhisperDecoderStepPlanLookup {
    plan: Arc<WhisperDecoderGraphPlan>,
    plan_cache_status: WhisperDecoderStepPlanCacheStatus,
    plan_build_ms: u128,
}

impl WhisperEncoderPreludeRunner for WhisperCpuEncoderPreludeComputeRunnerV0 {
    fn runner_id(&self) -> &'static str {
        "whisper-cpu-encoder-prelude-ggml-v0"
    }

    fn run_encoder_prelude(
        &self,
        _runtime_source: &GgmlRuntimeSource,
        encoder_weights: &WhisperEncoderWeightBundle,
        plan: &WhisperEncoderPreludePlan,
        mel_input: &WhisperMelFeatureInput,
    ) -> Result<WhisperEncoderPreludeSeamResult, WhisperGgmlExecutorError> {
        if mel_input.shape.mel_bins != plan.input_shape.mel_bins
            || mel_input.shape.mel_frames != plan.input_shape.mel_frames
        {
            return Err(WhisperGgmlExecutorError::MelFeatureInputPreparationFailed {
                reason: format!(
                    "mel shape mismatch from source '{}': got ({}, {}), expected ({}, {})",
                    mel_input.source_label,
                    mel_input.shape.mel_frames,
                    mel_input.shape.mel_bins,
                    plan.input_shape.mel_frames,
                    plan.input_shape.mel_bins
                ),
            });
        }
        let expected_mel_values = plan.input_shape.mel_frames * plan.input_shape.mel_bins;
        if mel_input.values_f32.len() != expected_mel_values {
            return Err(WhisperGgmlExecutorError::MelFeatureInputPreparationFailed {
                reason: format!(
                    "mel value count mismatch from source '{}': got {}, expected {}",
                    mel_input.source_label,
                    mel_input.values_f32.len(),
                    expected_mel_values
                ),
            });
        }
        if plan.output_frames > plan.positional_embedding.max_positions {
            return Err(
                WhisperGgmlExecutorError::EncoderPreludePrimitiveUnsupported {
                    primitive: "encoder.positional_embedding.slice",
                    reason: format!(
                        "projected frames {} exceed positional capacity {}",
                        plan.output_frames, plan.positional_embedding.max_positions
                    ),
                },
            );
        }
        let mut runner = GgmlCpuGraphRunner::new(whisper_encoder_prelude_cpu_graph_config())
            .map_err(
                |error| WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
                    reason: format!("could not initialize ggml cpu graph runner: {error}"),
                },
            )?;
        // Conv-stem weight bytes are constant per pack; resolve and encode them
        // up front so they can live in a WEIGHTS-usage arena (below) instead of
        // per-call graph-input leaves. ggml's scheduler only offloads a conv/
        // matmul when its weight `src` lives in a WEIGHTS buffer, so the two
        // conv_1d ops used to pin the prelude to the CPU even on a Metal backend.
        // Mel and the run-length positional slice stay genuine graph inputs.
        let encoder_tensor_index = build_encoder_tensor_index(encoder_weights);
        let conv1_weight =
            lookup_encoder_tensor_for_prelude(&encoder_tensor_index, &plan.conv1.weight_name)?;
        let conv1_bias =
            lookup_encoder_tensor_for_prelude(&encoder_tensor_index, &plan.conv1.bias_name)?;
        let conv2_weight =
            lookup_encoder_tensor_for_prelude(&encoder_tensor_index, &plan.conv2.weight_name)?;
        let conv2_bias =
            lookup_encoder_tensor_for_prelude(&encoder_tensor_index, &plan.conv2.bias_name)?;
        let positional_embedding = lookup_encoder_tensor_for_prelude(
            &encoder_tensor_index,
            &plan.positional_embedding.tensor_name,
        )?;

        let conv1_weight_bits = encode_prelude_conv_weight_f16_bits(conv1_weight, &plan.conv1)?;
        let conv1_bias_f32 = encoder_tensor_tail_f32_values(conv1_bias, plan.conv1.out_channels)
            .map_err(|reason| WhisperGgmlExecutorError::EncoderPreludeExecutionFailed { reason })?;
        let conv2_weight_bits = encode_prelude_conv_weight_f16_bits(conv2_weight, &plan.conv2)?;
        let conv2_bias_f32 = encoder_tensor_tail_f32_values(conv2_bias, plan.conv2.out_channels)
            .map_err(|reason| WhisperGgmlExecutorError::EncoderPreludeExecutionFailed { reason })?;
        let positional_f32 = slice_encoder_positional_embedding_for_prelude(
            positional_embedding,
            plan.output_frames,
            plan.output_hidden_size,
        )?;

        // Conv-stem weights resident in the arena's WEIGHTS-usage backend buffer
        // (mirrors the dolphin/cohere encoders). Allocate then upload once; the
        // uploaded bytes are identical to the previous per-call graph inputs, so
        // the prelude output is unchanged -- only the buffer each conv op reads
        // its weight from moves off the compute graph.
        let mut arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::metadata_context_bytes(8))
            .map_err(|error| map_graph_error("static_tensor_arena", error))?;
        let conv1_w_static = arena
            .new_tensor_3d_f16(
                plan.conv1.kernel_size,
                plan.conv1.in_channels,
                plan.conv1.out_channels,
                "conv1_w",
            )
            .map_err(|error| map_graph_error("ggml_new_tensor_3d_f16(conv1_w)", error))?;
        let conv1_b_static = arena
            .new_tensor_2d_f32(1, plan.conv1.out_channels, "conv1_b")
            .map_err(|error| map_graph_error("ggml_new_tensor_2d(conv1_b)", error))?;
        let conv2_w_static = arena
            .new_tensor_3d_f16(
                plan.conv2.kernel_size,
                plan.conv2.in_channels,
                plan.conv2.out_channels,
                "conv2_w",
            )
            .map_err(|error| map_graph_error("ggml_new_tensor_3d_f16(conv2_w)", error))?;
        let conv2_b_static = arena
            .new_tensor_2d_f32(1, plan.conv2.out_channels, "conv2_b")
            .map_err(|error| map_graph_error("ggml_new_tensor_2d(conv2_b)", error))?;
        arena
            .set_f16_bits_slice(conv1_w_static, conv1_weight_bits.as_ref(), "conv1_w")
            .map_err(
                |error| WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
                    reason: format!(
                        "could not upload conv1 weight '{}' into prelude arena: {error}",
                        plan.conv1.weight_name
                    ),
                },
            )?;
        arena
            .set_f32_slice(conv1_b_static, conv1_bias_f32.as_ref(), "conv1_b")
            .map_err(
                |error| WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
                    reason: format!(
                        "could not upload conv1 bias '{}' into prelude arena: {error}",
                        plan.conv1.bias_name
                    ),
                },
            )?;
        arena
            .set_f16_bits_slice(conv2_w_static, conv2_weight_bits.as_ref(), "conv2_w")
            .map_err(
                |error| WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
                    reason: format!(
                        "could not upload conv2 weight '{}' into prelude arena: {error}",
                        plan.conv2.weight_name
                    ),
                },
            )?;
        arena
            .set_f32_slice(conv2_b_static, conv2_bias_f32.as_ref(), "conv2_b")
            .map_err(
                |error| WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
                    reason: format!(
                        "could not upload conv2 bias '{}' into prelude arena: {error}",
                        plan.conv2.bias_name
                    ),
                },
            )?;

        let mut graph = runner.start_graph();

        let mel = graph
            .new_tensor_2d_f32(
                plan.input_shape.mel_frames,
                plan.input_shape.mel_bins,
                "mel",
            )
            .map_err(|error| map_graph_error("ggml_new_tensor_2d(mel)", error))?;
        let positional = graph
            .new_tensor_2d_f32(
                plan.output_hidden_size,
                plan.output_frames,
                "encoder_positional",
            )
            .map_err(|error| map_graph_error("ggml_new_tensor_2d(encoder_positional)", error))?;

        graph
            .set_input(mel)
            .map_err(|error| map_graph_error("ggml_set_input(mel)", error))?;
        graph
            .set_input(positional)
            .map_err(|error| map_graph_error("ggml_set_input(encoder_positional)", error))?;

        let conv1_w = arena.graph_tensor(conv1_w_static);
        let conv1_b = arena.graph_tensor(conv1_b_static);
        let conv2_w = arena.graph_tensor(conv2_w_static);
        let conv2_b = arena.graph_tensor(conv2_b_static);

        let conv1 = apply_conv_1d_bias_activation(
            &graph,
            conv1_w,
            mel,
            conv1_b,
            Conv1dParams {
                stride: plan.conv1.stride,
                padding: plan.conv1.padding,
                dilation: plan.conv1.dilation,
            },
            ConvActivation::Gelu,
            ConvBlockSteps {
                conv: "ggml_conv_1d(conv1)",
                bias: "ggml_add(conv1_bias)",
                activation: "ggml_gelu(conv1)",
            },
            map_graph_error,
        )?;

        let conv2 = apply_conv_1d_bias_activation(
            &graph,
            conv2_w,
            conv1,
            conv2_b,
            Conv1dParams {
                stride: plan.conv2.stride,
                padding: plan.conv2.padding,
                dilation: plan.conv2.dilation,
            },
            ConvActivation::Gelu,
            ConvBlockSteps {
                conv: "ggml_conv_1d(conv2)",
                bias: "ggml_add(conv2_bias)",
                activation: "ggml_gelu(conv2)",
            },
            map_graph_error,
        )?;
        let conv2 = graph
            .permute(conv2, 1, 0, 2, 3)
            .map_err(|error| map_graph_error("ggml_transpose(conv2)", error))?;
        let conv2 = graph
            .cont(conv2)
            .map_err(|error| map_graph_error("ggml_cont(conv2_transposed)", error))?;
        let prelude_output = graph
            .add(conv2, positional)
            .map_err(|error| map_graph_error("ggml_add(encoder_positional)", error))?;
        graph
            .set_output(prelude_output)
            .map_err(|error| map_graph_error("ggml_set_output(encoder_prelude)", error))?;

        // Only the genuine per-call inputs are uploaded into the compute graph;
        // the conv-stem weights already reside in the arena's WEIGHTS buffer.
        graph
            .set_f32_slice(mel, &mel_input.values_f32, "mel")
            .map_err(
                |error| WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
                    reason: format!("could not upload mel feature input: {error}"),
                },
            )?;
        graph
            .set_f32_slice(positional, positional_f32.as_ref(), "encoder_positional")
            .map_err(
                |error| WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
                    reason: format!(
                        "could not upload positional embedding '{}' for prelude compute: {error}",
                        plan.positional_embedding.tensor_name
                    ),
                },
            )?;

        if std::env::var_os("OPENASR_WHISPER_GGML_TRACE_PRELUDE").is_some() {
            let conv2_probe = graph
                .compute_output_f32(conv2, plan.output_frames * plan.output_hidden_size)
                .map_err(
                    |error| WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
                        reason: format!("encoder prelude conv2 probe compute failed: {error}"),
                    },
                )?;
            emit_tensor_probe_trace(
                "prelude_probe",
                "conv2_transposed",
                &conv2_probe,
                plan.output_frames,
                plan.output_hidden_size,
            );
        }

        let hidden_by_seq = graph
            .compute_output_f32(prelude_output, plan.output_frames * plan.output_hidden_size)
            .map_err(
                |error| WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
                    reason: format!("encoder prelude graph compute failed: {error}"),
                },
            )?;

        Ok(WhisperEncoderPreludeSeamResult::GraphExecuted {
            runner_id: self.runner_id(),
            output_frames: plan.output_frames,
            output_hidden_size: plan.output_hidden_size,
            output_hidden_f32: hidden_by_seq,
        })
    }
}

impl WhisperEncoderGraphRunner for WhisperCpuEncoderGraphComputeRunnerV0 {
    fn runner_id(&self) -> &'static str {
        "whisper-cpu-encoder-graph-ggml-v0"
    }

    fn run_encoder_graph(
        &self,
        runtime_source: &GgmlRuntimeSource,
        execution: &WhisperGgmlExecutionMetadata,
        encoder_weights: &WhisperEncoderWeightBundle,
        plan: &WhisperEncoderGraphPlan,
        encoder_hidden_input_f32: &[f32],
    ) -> Result<WhisperEncoderGraphSeamResult, WhisperGgmlExecutorError> {
        let graph_config = whisper_encoder_graph_config();
        let mut session = take_or_build_whisper_encoder_persistent_static_session(
            runtime_source,
            execution,
            encoder_weights,
            plan,
            graph_config,
        )?;
        let result = run_encoder_graph_with_runner(
            self.runner_id(),
            graph_config,
            execution,
            encoder_weights,
            plan,
            encoder_hidden_input_f32,
            &mut session.runner,
            session.resident_weights.as_ref(),
        );
        store_whisper_encoder_persistent_static_session(runtime_source.path(), session);
        result
    }
}

fn run_encoder_graph_with_runner(
    runner_id: &'static str,
    graph_config: GgmlCpuGraphConfig,
    execution: &WhisperGgmlExecutionMetadata,
    encoder_weights: &WhisperEncoderWeightBundle,
    plan: &WhisperEncoderGraphPlan,
    encoder_hidden_input_f32: &[f32],
    runner: &mut GgmlCpuGraphRunner,
    resident_weights: Option<&WhisperEncoderResidentWeightCache>,
) -> Result<WhisperEncoderGraphSeamResult, WhisperGgmlExecutorError> {
    if execution.encoder_attention_heads == 0 {
        return Err(WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: "encoder_attention_heads must be > 0".to_string(),
        });
    }
    if !plan
        .output_hidden_size
        .is_multiple_of(execution.encoder_attention_heads)
    {
        return Err(WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: format!(
                "encoder hidden size {} is not divisible by attention heads {}",
                plan.output_hidden_size, execution.encoder_attention_heads
            ),
        });
    }
    let expected_hidden_values = plan.output_frames * plan.output_hidden_size;
    if encoder_hidden_input_f32.len() != expected_hidden_values {
        return Err(WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: format!(
                "encoder hidden input length mismatch: got {}, expected {}",
                encoder_hidden_input_f32.len(),
                expected_hidden_values
            ),
        });
    }
    if encoder_hidden_input_f32
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: "encoder hidden input contains non-finite values".to_string(),
        });
    }
    if encoder_weights.layers.len() != plan.layers.len() {
        return Err(WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: format!(
                "encoder weight layer count mismatch: weights={} plan={}",
                encoder_weights.layers.len(),
                plan.layers.len()
            ),
        });
    }

    let graph_build_start = Instant::now();
    let encoder_tensor_index = build_encoder_tensor_index(encoder_weights);
    let mut graph = runner.start_graph();
    let mut uploads: Vec<WhisperEncoderGraphUpload<'_>> = Vec::new();
    let hidden = graph
        .new_tensor_2d_f32(
            plan.output_hidden_size,
            plan.output_frames,
            "encoder_hidden_input",
        )
        .map_err(|error| map_encoder_graph_error("ggml_new_tensor_2d(hidden)", error))?;
    graph
        .set_input(hidden)
        .map_err(|error| map_encoder_graph_error("ggml_set_input(hidden)", error))?;
    uploads.push(WhisperEncoderGraphUpload::f32_borrowed(
        hidden,
        encoder_hidden_input_f32,
        "encoder_hidden_input",
    ));
    let trace_encoder_layer0 =
        std::env::var_os("OPENASR_WHISPER_GGML_TRACE_ENCODER_LAYER0").is_some();
    let mut probe_tensors: Vec<(&'static str, GgmlCpuTensor<'_>)> = Vec::new();

    let mut state = hidden;
    for layer_plan in &plan.layers {
        let layer_weights = encoder_weights
            .layers
            .get(layer_plan.layer_idx)
            .ok_or_else(|| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
                reason: format!(
                    "missing encoder materialized layer {}",
                    layer_plan.layer_idx
                ),
            })?;
        let attn_norm = apply_encoder_affine_layer_norm(
            &mut graph,
            &mut uploads,
            &encoder_tensor_index,
            resident_weights,
            state,
            WHISPER_ENCODER_LAYER_NORM_EPSILON,
            &layer_plan.self_attn_norm,
        )?;
        if trace_encoder_layer0 && layer_plan.layer_idx == 0 {
            probe_tensors.push(("layer0_attn_norm", attn_norm));
        }
        let mut q = apply_encoder_linear_projection(
            &mut graph,
            &mut uploads,
            &encoder_tensor_index,
            resident_weights,
            attn_norm,
            &layer_plan.self_attn_q,
        )?;
        q = add_encoder_bias_tensor(
            &mut graph,
            &mut uploads,
            resident_weights,
            q,
            &layer_weights.self_attn_q_bias,
            layer_plan.self_attn_q.output_dim,
            "encoder_self_attn_q_bias",
        )?;
        if trace_encoder_layer0 && layer_plan.layer_idx == 0 {
            probe_tensors.push(("layer0_q", q));
        }
        let k = apply_encoder_linear_projection(
            &mut graph,
            &mut uploads,
            &encoder_tensor_index,
            resident_weights,
            attn_norm,
            &layer_plan.self_attn_k,
        )?;
        let mut v = apply_encoder_linear_projection(
            &mut graph,
            &mut uploads,
            &encoder_tensor_index,
            resident_weights,
            attn_norm,
            &layer_plan.self_attn_v,
        )?;
        v = add_encoder_bias_tensor(
            &mut graph,
            &mut uploads,
            resident_weights,
            v,
            &layer_weights.self_attn_v_bias,
            layer_plan.self_attn_v.output_dim,
            "encoder_self_attn_v_bias",
        )?;

        let head_dim = plan.output_hidden_size / execution.encoder_attention_heads;
        let attention_scale = 1.0f32 / (head_dim as f32).sqrt();
        let attn_context = if whisper_encoder_flash_attention_enabled() {
            let use_strided_views = graph_config.backend.is_gpu_class();
            let q = reshape_encoder_projection_to_heads_for_flash(
                &mut graph,
                q,
                head_dim,
                plan.output_frames,
                execution.encoder_attention_heads,
                "attn_q_heads",
                use_strided_views,
            )?;
            let k = reshape_encoder_projection_to_heads_for_flash(
                &mut graph,
                k,
                head_dim,
                plan.output_frames,
                execution.encoder_attention_heads,
                "attn_k_heads",
                use_strided_views,
            )?;
            let v = reshape_encoder_projection_to_heads_for_flash(
                &mut graph,
                v,
                head_dim,
                plan.output_frames,
                execution.encoder_attention_heads,
                "attn_v_heads",
                use_strided_views,
            )?;
            let flash = graph
                .flash_attn_ext(q, k, v, None, attention_scale, 0.0, 0.0)
                .map_err(|error| map_encoder_graph_error("ggml_flash_attn_ext(attn)", error))?;
            graph
                .reshape_2d(flash, plan.output_hidden_size, plan.output_frames)
                .map_err(|error| {
                    map_encoder_graph_error("ggml_reshape_2d(attn_flash_merge)", error)
                })?
        } else {
            let attention_layout = AttentionHeadLayout {
                head_dim,
                attention_heads: execution.encoder_attention_heads,
                sequence_len: plan.output_frames,
            };
            let q = reshape_encoder_projection_to_heads(
                &mut graph,
                q,
                head_dim,
                plan.output_frames,
                execution.encoder_attention_heads,
                "attn_q_heads",
            )?;
            let k = reshape_encoder_projection_to_heads(
                &mut graph,
                k,
                head_dim,
                plan.output_frames,
                execution.encoder_attention_heads,
                "attn_k_heads",
            )?;
            let v = reshape_encoder_projection_to_heads(
                &mut graph,
                v,
                head_dim,
                plan.output_frames,
                execution.encoder_attention_heads,
                "attn_v_heads",
            )?;
            let attn_scores = graph
                .mul_mat(k, q)
                .map_err(|error| map_encoder_graph_error("ggml_mul_mat(attn_qk)", error))?;
            let attn_scores = graph
                .cont(attn_scores)
                .map_err(|error| map_encoder_graph_error("ggml_cont(attn_qk)", error))?;
            let attn_probs = graph
                .soft_max_ext(attn_scores, None, attention_scale, 0.0)
                .map_err(|error| {
                    map_encoder_graph_error("ggml_soft_max_ext(attn_qk_probs)", error)
                })?;
            attention_context_from_probs(
                &graph,
                v,
                attn_probs,
                attention_layout,
                AttentionValueMergeSteps {
                    value_permute: "ggml_permute(attn_v_t)",
                    value_cont: "ggml_cont(attn_v_t)",
                    context_mul: "ggml_mul_mat(attn_av)",
                    context_merge_permute: "ggml_permute(attn_merge)",
                    context_merge_cont: "ggml_cont(attn_merge)",
                    context_merge_reshape: "ggml_reshape_2d(attn_merge)",
                },
                map_encoder_graph_error,
            )?
        };

        let mut attn_out = apply_encoder_linear_projection(
            &mut graph,
            &mut uploads,
            &encoder_tensor_index,
            resident_weights,
            attn_context,
            &layer_plan.self_attn_out,
        )?;
        attn_out = add_encoder_bias_tensor(
            &mut graph,
            &mut uploads,
            resident_weights,
            attn_out,
            &layer_weights.self_attn_out_bias,
            layer_plan.self_attn_out.output_dim,
            "encoder_self_attn_out_bias",
        )?;
        state = graph
            .add(attn_out, state)
            .map_err(|error| map_encoder_graph_error("ggml_add(attn_residual)", error))?;

        let mlp_norm = apply_encoder_affine_layer_norm(
            &mut graph,
            &mut uploads,
            &encoder_tensor_index,
            resident_weights,
            state,
            WHISPER_ENCODER_LAYER_NORM_EPSILON,
            &layer_plan.mlp_norm,
        )?;
        let mut mlp_fc1 = apply_encoder_linear_projection(
            &mut graph,
            &mut uploads,
            &encoder_tensor_index,
            resident_weights,
            mlp_norm,
            &layer_plan.mlp_fc1,
        )?;
        mlp_fc1 = add_encoder_bias_tensor(
            &mut graph,
            &mut uploads,
            resident_weights,
            mlp_fc1,
            &layer_weights.fc1_bias,
            layer_plan.mlp_fc1.output_dim,
            "encoder_mlp_fc1_bias",
        )?;
        let mlp_fc1 = graph
            .gelu(mlp_fc1)
            .map_err(|error| map_encoder_graph_error("ggml_gelu(mlp_fc1)", error))?;
        let mut mlp_fc2 = apply_encoder_linear_projection(
            &mut graph,
            &mut uploads,
            &encoder_tensor_index,
            resident_weights,
            mlp_fc1,
            &layer_plan.mlp_fc2,
        )?;
        mlp_fc2 = add_encoder_bias_tensor(
            &mut graph,
            &mut uploads,
            resident_weights,
            mlp_fc2,
            &layer_weights.fc2_bias,
            layer_plan.mlp_fc2.output_dim,
            "encoder_mlp_fc2_bias",
        )?;
        state = graph
            .add(mlp_fc2, state)
            .map_err(|error| map_encoder_graph_error("ggml_add(mlp_residual)", error))?;
    }

    state = apply_encoder_affine_layer_norm(
        &mut graph,
        &mut uploads,
        &encoder_tensor_index,
        resident_weights,
        state,
        WHISPER_ENCODER_LAYER_NORM_EPSILON,
        &plan.final_norm,
    )?;

    graph
        .set_output(state)
        .map_err(|error| map_encoder_graph_error("ggml_set_output(state)", error))?;
    let graph_build_ms = graph_build_start.elapsed().as_millis();
    let buffer_alloc_start = Instant::now();
    graph
        .prepare_outputs_for_upload(&[state])
        .map_err(|error| map_encoder_graph_error("ggml_prepare_outputs_for_upload", error))?;
    let buffer_alloc_ms = buffer_alloc_start.elapsed().as_millis();
    let tensor_set_start = Instant::now();
    let upload_stats = upload_encoder_graph_inputs(&mut graph, uploads)?;
    let tensor_set_ms = tensor_set_start.elapsed().as_millis();
    let upload_ms = buffer_alloc_ms.saturating_add(tensor_set_ms);
    for (event, tensor) in probe_tensors {
        let probe = graph
            .compute_output_f32(tensor, plan.output_frames * plan.output_hidden_size)
            .map_err(
                |error| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
                    reason: format!("encoder graph probe '{event}' compute failed: {error}"),
                },
            )?;
        emit_tensor_probe_trace(
            "encoder_layer0_probe",
            event,
            &probe,
            plan.output_frames,
            plan.output_hidden_size,
        );
    }
    let compute_start = Instant::now();
    let hidden_by_seq = graph
        .compute_output_f32(state, plan.output_frames * plan.output_hidden_size)
        .map_err(
            |error| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
                reason: format!("encoder graph compute failed: {error}"),
            },
        )?;
    let compute_ms = compute_start.elapsed().as_millis();
    emit_encoder_graph_detail_trace(
        upload_stats.count,
        upload_stats.bytes,
        graph_build_ms,
        upload_ms,
        buffer_alloc_ms,
        tensor_set_ms,
        compute_ms,
        graph_build_start.elapsed().as_millis(),
    );

    Ok(WhisperEncoderGraphSeamResult::GraphExecuted {
        runner_id,
        layer_count: plan.layers.len(),
        output_frames: plan.output_frames,
        output_hidden_size: plan.output_hidden_size,
        output_hidden_f32: hidden_by_seq,
    })
}

fn reshape_encoder_projection_to_heads<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    sequence_len: usize,
    attention_heads: usize,
    label: &'static str,
) -> Result<GgmlCpuTensor<'a>, WhisperGgmlExecutorError> {
    reshape_projection_to_attention_heads(
        graph,
        projection,
        AttentionHeadLayout {
            head_dim,
            attention_heads,
            sequence_len,
        },
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        AttentionReshapeSteps {
            reshape: "ggml_reshape_3d(attn_heads)",
            permute: "ggml_permute(attn_heads)",
            cont: label,
        },
        map_encoder_graph_error,
    )
}

fn reshape_encoder_projection_to_heads_for_flash<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    sequence_len: usize,
    attention_heads: usize,
    label: &'static str,
    use_strided_views: bool,
) -> Result<GgmlCpuTensor<'a>, WhisperGgmlExecutorError> {
    if use_strided_views {
        reshape_encoder_projection_to_heads_view(
            graph,
            projection,
            head_dim,
            sequence_len,
            attention_heads,
        )
    } else {
        reshape_encoder_projection_to_heads(
            graph,
            projection,
            head_dim,
            sequence_len,
            attention_heads,
            label,
        )
    }
}

fn reshape_encoder_projection_to_heads_view<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    projection: GgmlCpuTensor<'a>,
    head_dim: usize,
    sequence_len: usize,
    attention_heads: usize,
) -> Result<GgmlCpuTensor<'a>, WhisperGgmlExecutorError> {
    let reshaped = graph
        .reshape_3d(projection, head_dim, attention_heads, sequence_len)
        .map_err(|error| map_encoder_graph_error("ggml_reshape_3d(attn_heads)", error))?;
    graph
        .permute(reshaped, 0, 2, 1, 3)
        .map_err(|error| map_encoder_graph_error("ggml_permute(attn_heads)", error))
}

#[derive(Debug)]
enum WhisperEncoderGraphUploadPayload<'a> {
    F32Owned(Vec<f32>),
    F32Borrowed(&'a [f32]),
    F16Bits(Vec<u16>),
    Bytes(Vec<u8>),
}

#[derive(Debug)]
struct WhisperEncoderGraphUpload<'a> {
    tensor: GgmlCpuTensor<'a>,
    label: &'static str,
    payload: WhisperEncoderGraphUploadPayload<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WhisperEncoderGraphUploadStats {
    count: usize,
    bytes: usize,
}

struct WhisperEncoderResidentWeightCache {
    arena: GgmlStaticTensorArena,
    tensors_by_name: HashMap<String, GgmlStaticTensor>,
    // Zero-copy weights bound directly to the mmap'd runtime pack (no host copy,
    // no arena upload). Only the large quantized linear weights that are consumed
    // byte-for-byte as-is (input-output layout) are bound here; everything else
    // stays on the resident arena upload path. `_loaded` owns the mmap + ggml
    // context that `loaded_tensors_by_name` points into and must outlive the graph.
    _loaded: Option<GgmlLoadedWeightContext>,
    loaded_tensors_by_name: HashMap<String, GgmlLoadedTensor>,
    upload_stats: WhisperEncoderGraphUploadStats,
}

#[derive(Debug)]
enum WhisperEncoderResidentWeightUpload<'a> {
    F32 {
        tensor: GgmlStaticTensor,
        values: Vec<f32>,
    },
    F16BitsBorrowed {
        tensor: GgmlStaticTensor,
        values: &'a [u16],
    },
    F16BitsOwned {
        tensor: GgmlStaticTensor,
        values: Vec<u16>,
    },
    QuantizedBytesBorrowed {
        tensor: GgmlStaticTensor,
        values: &'a [u8],
    },
}

impl<'a> WhisperEncoderGraphUpload<'a> {
    fn f32_owned(tensor: GgmlCpuTensor<'a>, values: Vec<f32>, label: &'static str) -> Self {
        Self {
            tensor,
            label,
            payload: WhisperEncoderGraphUploadPayload::F32Owned(values),
        }
    }

    fn f32_borrowed(tensor: GgmlCpuTensor<'a>, values: &'a [f32], label: &'static str) -> Self {
        Self {
            tensor,
            label,
            payload: WhisperEncoderGraphUploadPayload::F32Borrowed(values),
        }
    }

    fn f16_bits(tensor: GgmlCpuTensor<'a>, values: Vec<u16>, label: &'static str) -> Self {
        Self {
            tensor,
            label,
            payload: WhisperEncoderGraphUploadPayload::F16Bits(values),
        }
    }

    fn bytes(tensor: GgmlCpuTensor<'a>, values: Vec<u8>, label: &'static str) -> Self {
        Self {
            tensor,
            label,
            payload: WhisperEncoderGraphUploadPayload::Bytes(values),
        }
    }
}

impl WhisperEncoderResidentWeightCache {
    fn graph_tensor<'a>(&self, tensor_name: &str) -> Option<GgmlCpuTensor<'a>> {
        if let Some(loaded) = self.loaded_tensors_by_name.get(tensor_name) {
            return Some(loaded.as_graph_tensor());
        }
        self.tensors_by_name
            .get(tensor_name)
            .map(|tensor| self.arena.graph_tensor(*tensor))
    }
}

fn build_encoder_tensor_index(
    encoder_weights: &WhisperEncoderWeightBundle,
) -> HashMap<&str, &WhisperMaterializedTensor> {
    let mut by_name = HashMap::with_capacity(encoder_weights.materialized_tensor_count());
    by_name.insert(
        encoder_weights.prelude.conv1_weight.tensor_name.as_str(),
        &encoder_weights.prelude.conv1_weight,
    );
    by_name.insert(
        encoder_weights.prelude.conv1_bias.tensor_name.as_str(),
        &encoder_weights.prelude.conv1_bias,
    );
    by_name.insert(
        encoder_weights.prelude.conv2_weight.tensor_name.as_str(),
        &encoder_weights.prelude.conv2_weight,
    );
    by_name.insert(
        encoder_weights.prelude.conv2_bias.tensor_name.as_str(),
        &encoder_weights.prelude.conv2_bias,
    );
    by_name.insert(
        encoder_weights
            .prelude
            .positional_embedding
            .tensor_name
            .as_str(),
        &encoder_weights.prelude.positional_embedding,
    );
    for layer in &encoder_weights.layers {
        by_name.insert(
            layer.self_attn_layer_norm_weight.tensor_name.as_str(),
            &layer.self_attn_layer_norm_weight,
        );
        by_name.insert(
            layer.self_attn_layer_norm_bias.tensor_name.as_str(),
            &layer.self_attn_layer_norm_bias,
        );
        by_name.insert(
            layer.self_attn_q_weight.tensor_name.as_str(),
            &layer.self_attn_q_weight,
        );
        by_name.insert(
            layer.self_attn_q_bias.tensor_name.as_str(),
            &layer.self_attn_q_bias,
        );
        by_name.insert(
            layer.self_attn_k_weight.tensor_name.as_str(),
            &layer.self_attn_k_weight,
        );
        by_name.insert(
            layer.self_attn_v_weight.tensor_name.as_str(),
            &layer.self_attn_v_weight,
        );
        by_name.insert(
            layer.self_attn_v_bias.tensor_name.as_str(),
            &layer.self_attn_v_bias,
        );
        by_name.insert(
            layer.self_attn_out_weight.tensor_name.as_str(),
            &layer.self_attn_out_weight,
        );
        by_name.insert(
            layer.self_attn_out_bias.tensor_name.as_str(),
            &layer.self_attn_out_bias,
        );
        by_name.insert(
            layer.mlp_norm_weight.tensor_name.as_str(),
            &layer.mlp_norm_weight,
        );
        by_name.insert(
            layer.mlp_norm_bias.tensor_name.as_str(),
            &layer.mlp_norm_bias,
        );
        by_name.insert(layer.fc1_weight.tensor_name.as_str(), &layer.fc1_weight);
        by_name.insert(layer.fc1_bias.tensor_name.as_str(), &layer.fc1_bias);
        by_name.insert(layer.fc2_weight.tensor_name.as_str(), &layer.fc2_weight);
        by_name.insert(layer.fc2_bias.tensor_name.as_str(), &layer.fc2_bias);
    }
    by_name.insert(
        encoder_weights.final_norm.weight.tensor_name.as_str(),
        &encoder_weights.final_norm.weight,
    );
    by_name.insert(
        encoder_weights.final_norm.bias.tensor_name.as_str(),
        &encoder_weights.final_norm.bias,
    );
    by_name
}

fn lookup_encoder_tensor_for_prelude<'a>(
    encoder_tensors: &'a HashMap<&str, &'a WhisperMaterializedTensor>,
    tensor_name: &str,
) -> Result<&'a WhisperMaterializedTensor, WhisperGgmlExecutorError> {
    encoder_tensors.get(tensor_name).copied().ok_or_else(|| {
        WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
            reason: format!(
                "prelude tensor '{}' is missing from materialized encoder weights",
                tensor_name
            ),
        }
    })
}

fn lookup_encoder_tensor_for_graph<'a>(
    encoder_tensors: &'a HashMap<&str, &'a WhisperMaterializedTensor>,
    tensor_name: &str,
) -> Result<&'a WhisperMaterializedTensor, WhisperGgmlExecutorError> {
    encoder_tensors.get(tensor_name).copied().ok_or_else(|| {
        WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: format!(
                "encoder graph tensor '{}' is missing from materialized encoder weights",
                tensor_name
            ),
        }
    })
}

fn lookup_encoder_tensor_for_resident<'weights>(
    encoder_tensors: &HashMap<&str, &'weights WhisperMaterializedTensor>,
    tensor_name: &str,
) -> Result<&'weights WhisperMaterializedTensor, WhisperGgmlExecutorError> {
    encoder_tensors.get(tensor_name).copied().ok_or_else(|| {
        WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: format!(
                "encoder graph tensor '{}' is missing from materialized encoder weights",
                tensor_name
            ),
        }
    })
}

fn encode_prelude_conv_weight_f16_bits<'a>(
    tensor: &'a WhisperMaterializedTensor,
    plan: &WhisperEncoderPreludeConv1dPlan,
) -> Result<Cow<'a, [u16]>, WhisperGgmlExecutorError> {
    let source_bits = match &tensor.payload {
        WhisperMaterializedTensorPayload::F16Bits(values) => Cow::Borrowed(values.as_slice()),
        WhisperMaterializedTensorPayload::F32(values) => values
            .iter()
            .map(|value| f32_to_f16_bits(*value))
            .collect::<Vec<_>>()
            .into(),
        WhisperMaterializedTensorPayload::Quantized { ggml_type, .. } => {
            return Err(WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
                reason: format!(
                    "prelude tensor '{}' has quantized ggml type {ggml_type}, expected f16/f32",
                    tensor.tensor_name
                ),
            });
        }
    };
    if source_bits.len() != tensor.num_elements {
        return Err(WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
            reason: format!(
                "prelude tensor '{}' materialized {} values but metadata expects {}",
                tensor.tensor_name,
                source_bits.len(),
                tensor.num_elements
            ),
        });
    }

    let expected = plan
        .kernel_size
        .checked_mul(plan.in_channels)
        .and_then(|value| value.checked_mul(plan.out_channels))
        .ok_or_else(|| WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
            reason: format!(
                "conv weight '{}' shape overflow for [{}x{}x{}]",
                tensor.tensor_name, plan.kernel_size, plan.in_channels, plan.out_channels
            ),
        })?;
    if source_bits.len() != expected {
        return Err(WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
            reason: format!(
                "conv weight '{}' has {} values but expected {} for [{}x{}x{}]",
                tensor.tensor_name,
                source_bits.len(),
                expected,
                plan.kernel_size,
                plan.in_channels,
                plan.out_channels
            ),
        });
    }

    match plan.layout {
        WhisperEncoderPreludeConv1dWeightLayout::KernelInOut => Ok(source_bits),
        WhisperEncoderPreludeConv1dWeightLayout::OutInKernel => {
            let mut reordered = vec![0_u16; source_bits.len()];
            let kernel = plan.kernel_size;
            let input = plan.in_channels;
            let output = plan.out_channels;
            for out_idx in 0..output {
                for in_idx in 0..input {
                    for kernel_idx in 0..kernel {
                        let src = kernel_idx + kernel * (in_idx + input * out_idx);
                        let dst = kernel_idx + kernel * (in_idx + input * out_idx);
                        reordered[dst] = source_bits[src];
                    }
                }
            }
            Ok(reordered.into())
        }
    }
}

fn slice_encoder_positional_embedding_for_prelude<'a>(
    tensor: &'a WhisperMaterializedTensor,
    output_frames: usize,
    output_hidden_size: usize,
) -> Result<Cow<'a, [f32]>, WhisperGgmlExecutorError> {
    let values = encoder_tensor_values_f32(tensor)
        .map_err(|reason| WhisperGgmlExecutorError::EncoderPreludeExecutionFailed { reason })?;
    if tensor.dims.len() != 2 {
        return Err(WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
            reason: format!(
                "positional tensor '{}' must be rank-2, got dims {:?}",
                tensor.tensor_name, tensor.dims
            ),
        });
    }
    let dim0 = usize::try_from(tensor.dims[0]).map_err(|_| {
        WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
            reason: format!(
                "positional tensor '{}' dim0 does not fit usize",
                tensor.tensor_name
            ),
        }
    })?;
    let dim1 = usize::try_from(tensor.dims[1]).map_err(|_| {
        WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
            reason: format!(
                "positional tensor '{}' dim1 does not fit usize",
                tensor.tensor_name
            ),
        }
    })?;
    if dim1 == output_hidden_size {
        if dim0 < output_frames {
            return Err(WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
                reason: format!(
                    "positional tensor '{}' has {} positions but prelude requires {}",
                    tensor.tensor_name, dim0, output_frames
                ),
            });
        }
        if output_frames == dim0 {
            return Ok(values);
        }
        let row_len = output_hidden_size;
        return Ok(values[..output_frames * row_len].to_vec().into());
    }
    if dim0 == output_hidden_size {
        if dim1 < output_frames {
            return Err(WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
                reason: format!(
                    "positional tensor '{}' has {} positions but prelude requires {}",
                    tensor.tensor_name, dim1, output_frames
                ),
            });
        }
        if output_frames == dim1 {
            return Ok(values);
        }
        let row_len = output_hidden_size;
        return Ok(values[..output_frames * row_len].to_vec().into());
    }
    Err(WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
        reason: format!(
            "positional tensor '{}' dims {:?} do not match expected hidden_size={}",
            tensor.tensor_name, tensor.dims, output_hidden_size
        ),
    })
}

fn encoder_tensor_tail_f32_values<'a>(
    tensor: &'a WhisperMaterializedTensor,
    expected_len: usize,
) -> Result<Cow<'a, [f32]>, String> {
    let values = encoder_tensor_values_f32(tensor)?;
    if values.len() < expected_len {
        return Err(format!(
            "tensor '{}' has {} values but expected at least {}",
            tensor.tensor_name,
            values.len(),
            expected_len
        ));
    }
    let start = values.len() - expected_len;
    let tail = if start == 0 {
        values
    } else {
        Cow::Owned(values[start..].to_vec())
    };
    if tail.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "tensor '{}' contains non-finite values in required tail slice",
            tensor.tensor_name
        ));
    }
    Ok(tail)
}

fn encoder_tensor_values_f32<'a>(
    tensor: &'a WhisperMaterializedTensor,
) -> Result<Cow<'a, [f32]>, String> {
    let values = match &tensor.payload {
        WhisperMaterializedTensorPayload::F32(values) => Cow::Borrowed(values.as_slice()),
        WhisperMaterializedTensorPayload::F16Bits(values) => values
            .iter()
            .map(|bits| f16_bits_to_f32(*bits))
            .collect::<Vec<_>>()
            .into(),
        WhisperMaterializedTensorPayload::Quantized { ggml_type, .. } => {
            return Err(format!(
                "encoder tensor '{}' is quantized (ggml type {ggml_type}); f32 materialization is not available in this path",
                tensor.tensor_name
            ));
        }
    };
    if values.len() != tensor.num_elements {
        return Err(format!(
            "encoder tensor '{}' materialized {} values but metadata expects {}",
            tensor.tensor_name,
            values.len(),
            tensor.num_elements
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "encoder tensor '{}' materialized non-finite values",
            tensor.tensor_name
        ));
    }
    Ok(values)
}

fn prepare_encoder_runtime_weight_payloads(
    weights: &mut WhisperEncoderWeightBundle,
) -> Result<(), WhisperGgmlExecutorError> {
    prepare_encoder_weight_tensor_f16(&mut weights.prelude.conv1_weight)?;
    prepare_encoder_weight_tensor_f16(&mut weights.prelude.conv2_weight)?;
    for layer in &mut weights.layers {
        prepare_encoder_layer_runtime_weight_payloads(layer)?;
    }
    Ok(())
}

fn prepare_encoder_layer_runtime_weight_payloads(
    layer: &mut super::ggml_encoder_weights::WhisperEncoderLayerWeightBundle,
) -> Result<(), WhisperGgmlExecutorError> {
    let hidden = layer.self_attn_q_bias.num_elements;
    let ffn = layer.fc1_bias.num_elements;
    prepare_encoder_linear_weight_tensor_input_output_f16(
        &mut layer.self_attn_q_weight,
        hidden,
        hidden,
    )?;
    prepare_encoder_linear_weight_tensor_input_output_f16(
        &mut layer.self_attn_k_weight,
        hidden,
        hidden,
    )?;
    prepare_encoder_linear_weight_tensor_input_output_f16(
        &mut layer.self_attn_v_weight,
        hidden,
        hidden,
    )?;
    prepare_encoder_linear_weight_tensor_input_output_f16(
        &mut layer.self_attn_out_weight,
        hidden,
        hidden,
    )?;
    prepare_encoder_linear_weight_tensor_input_output_f16(&mut layer.fc1_weight, hidden, ffn)?;
    prepare_encoder_linear_weight_tensor_input_output_f16(&mut layer.fc2_weight, ffn, hidden)?;
    Ok(())
}

fn prepare_decoder_runtime_weight_payloads(
    weights: &mut WhisperDecoderWeightBundle,
) -> Result<(), WhisperGgmlExecutorError> {
    prepare_decoder_weight_tensor_f16(&mut weights.token_embedding)?;
    if let Some(output_projection_weight) = weights.output_projection_weight.as_mut() {
        prepare_decoder_weight_tensor_f16(output_projection_weight)?;
    }
    for layer in &mut weights.layers {
        prepare_decoder_weight_tensor_f16(&mut layer.self_attn_q_weight)?;
        prepare_decoder_weight_tensor_f16(&mut layer.self_attn_k_weight)?;
        prepare_decoder_weight_tensor_f16(&mut layer.self_attn_v_weight)?;
        prepare_decoder_weight_tensor_f16(&mut layer.self_attn_out_weight)?;
        prepare_decoder_weight_tensor_f16(&mut layer.cross_attn_q_weight)?;
        prepare_decoder_weight_tensor_f16(&mut layer.cross_attn_k_weight)?;
        prepare_decoder_weight_tensor_f16(&mut layer.cross_attn_v_weight)?;
        prepare_decoder_weight_tensor_f16(&mut layer.cross_attn_out_weight)?;
        prepare_decoder_weight_tensor_f16(&mut layer.fc1_weight)?;
        prepare_decoder_weight_tensor_f16(&mut layer.fc2_weight)?;
    }
    Ok(())
}

fn prepare_encoder_weight_tensor_f16(
    tensor: &mut WhisperMaterializedTensor,
) -> Result<(), WhisperGgmlExecutorError> {
    let WhisperMaterializedTensorPayload::F32(values) = &tensor.payload else {
        return Ok(());
    };
    let mut prepared = Vec::with_capacity(values.len());
    for value in values.iter().copied() {
        if !value.is_finite() {
            return Err(WhisperGgmlExecutorError::TensorMaterializationFailed {
                reason: format!(
                    "encoder tensor '{}' contains non-finite values before f16 runtime preparation",
                    tensor.tensor_name
                ),
            });
        }
        prepared.push(f32_to_f16_bits(value));
    }
    tensor.payload = WhisperMaterializedTensorPayload::F16Bits(prepared);
    Ok(())
}

fn prepare_encoder_linear_weight_tensor_input_output_f16(
    tensor: &mut WhisperMaterializedTensor,
    expected_input_dim: usize,
    expected_output_dim: usize,
) -> Result<(), WhisperGgmlExecutorError> {
    if let WhisperMaterializedTensorPayload::Quantized { ggml_type, .. } = &tensor.payload {
        if tensor.dims.len() != 2 {
            return Err(WhisperGgmlExecutorError::TensorMaterializationFailed {
                reason: format!(
                    "encoder quantized linear tensor '{}' must be rank-2, got {:?}",
                    tensor.tensor_name, tensor.dims
                ),
            });
        }
        let dim0 = usize::try_from(tensor.dims[0]).map_err(|_| {
            WhisperGgmlExecutorError::TensorMaterializationFailed {
                reason: format!(
                    "encoder quantized linear tensor '{}' dimension 0 does not fit usize: {}",
                    tensor.tensor_name, tensor.dims[0]
                ),
            }
        })?;
        let dim1 = usize::try_from(tensor.dims[1]).map_err(|_| {
            WhisperGgmlExecutorError::TensorMaterializationFailed {
                reason: format!(
                    "encoder quantized linear tensor '{}' dimension 1 does not fit usize: {}",
                    tensor.tensor_name, tensor.dims[1]
                ),
            }
        })?;
        if dim0 != expected_input_dim || dim1 != expected_output_dim {
            return Err(WhisperGgmlExecutorError::TensorMaterializationFailed {
                reason: format!(
                    "encoder quantized linear tensor '{}' shape {:?} must be input-output [{}, {}] for ggml type {}",
                    tensor.tensor_name,
                    tensor.dims,
                    expected_input_dim,
                    expected_output_dim,
                    ggml_type
                ),
            });
        }
        return Ok(());
    }
    prepare_encoder_weight_tensor_f16(tensor)?;
    if tensor.dims.len() != 2 {
        return Err(WhisperGgmlExecutorError::TensorMaterializationFailed {
            reason: format!(
                "encoder linear tensor '{}' must be rank-2 before runtime layout preparation, got {:?}",
                tensor.tensor_name, tensor.dims
            ),
        });
    }
    let dim0 = usize::try_from(tensor.dims[0]).map_err(|_| {
        WhisperGgmlExecutorError::TensorMaterializationFailed {
            reason: format!(
                "encoder linear tensor '{}' dimension 0 does not fit usize: {}",
                tensor.tensor_name, tensor.dims[0]
            ),
        }
    })?;
    let dim1 = usize::try_from(tensor.dims[1]).map_err(|_| {
        WhisperGgmlExecutorError::TensorMaterializationFailed {
            reason: format!(
                "encoder linear tensor '{}' dimension 1 does not fit usize: {}",
                tensor.tensor_name, tensor.dims[1]
            ),
        }
    })?;
    let expected = expected_input_dim
        .checked_mul(expected_output_dim)
        .ok_or_else(|| WhisperGgmlExecutorError::TensorMaterializationFailed {
            reason: format!(
                "encoder linear tensor '{}' dimensions overflow: {}x{}",
                tensor.tensor_name, expected_output_dim, expected_input_dim
            ),
        })?;
    let source_layout = if dim0 == expected_input_dim && dim1 == expected_output_dim {
        WhisperEncoderLinearWeightLayout::InputOutput
    } else if dim0 == expected_output_dim && dim1 == expected_input_dim {
        WhisperEncoderLinearWeightLayout::OutputInput
    } else {
        return Err(WhisperGgmlExecutorError::TensorMaterializationFailed {
            reason: format!(
                "encoder linear tensor '{}' shape {:?} matches neither input-output [{}, {}] nor output-input [{}, {}]",
                tensor.tensor_name,
                tensor.dims,
                expected_input_dim,
                expected_output_dim,
                expected_output_dim,
                expected_input_dim
            ),
        });
    };
    let WhisperMaterializedTensorPayload::F16Bits(values) = &mut tensor.payload else {
        return Err(WhisperGgmlExecutorError::TensorMaterializationFailed {
            reason: format!(
                "encoder linear tensor '{}' was not prepared as f16",
                tensor.tensor_name
            ),
        });
    };
    if values.len() != expected {
        return Err(WhisperGgmlExecutorError::TensorMaterializationFailed {
            reason: format!(
                "encoder linear tensor '{}' has {} values but expected {}",
                tensor.tensor_name,
                values.len(),
                expected
            ),
        });
    }
    if source_layout == WhisperEncoderLinearWeightLayout::OutputInput {
        *values = transpose_linear_weight_output_input_to_input_output_u16(
            values,
            expected_input_dim,
            expected_output_dim,
        )?;
    }
    tensor.dims = vec![expected_input_dim as u64, expected_output_dim as u64];
    Ok(())
}

fn prepare_decoder_weight_tensor_f16(
    tensor: &mut WhisperMaterializedTensor,
) -> Result<(), WhisperGgmlExecutorError> {
    let WhisperMaterializedTensorPayload::F32(values) = &tensor.payload else {
        return Ok(());
    };
    let mut prepared = Vec::with_capacity(values.len());
    for value in values.iter().copied() {
        if !value.is_finite() {
            return Err(WhisperGgmlExecutorError::TensorMaterializationFailed {
                reason: format!(
                    "decoder tensor '{}' contains non-finite values before f16 runtime preparation",
                    tensor.tensor_name
                ),
            });
        }
        prepared.push(f32_to_f16_bits(value));
    }
    tensor.payload = WhisperMaterializedTensorPayload::F16Bits(prepared);
    Ok(())
}

fn upload_encoder_graph_inputs<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    uploads: Vec<WhisperEncoderGraphUpload<'a>>,
) -> Result<WhisperEncoderGraphUploadStats, WhisperGgmlExecutorError> {
    let mut stats = WhisperEncoderGraphUploadStats { count: 0, bytes: 0 };
    for upload in uploads {
        stats.count = stats.count.saturating_add(1);
        match upload.payload {
            WhisperEncoderGraphUploadPayload::F32Owned(values) => {
                stats.bytes = stats
                    .bytes
                    .saturating_add(values.len().saturating_mul(std::mem::size_of::<f32>()));
                graph
                    .set_f32_slice(upload.tensor, &values, upload.label)
                    .map_err(
                        |error| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
                            reason: format!("could not upload tensor '{}': {error}", upload.label),
                        },
                    )?
            }
            WhisperEncoderGraphUploadPayload::F32Borrowed(values) => {
                stats.bytes = stats
                    .bytes
                    .saturating_add(values.len().saturating_mul(std::mem::size_of::<f32>()));
                graph
                    .set_f32_slice(upload.tensor, values, upload.label)
                    .map_err(
                        |error| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
                            reason: format!("could not upload tensor '{}': {error}", upload.label),
                        },
                    )?
            }
            WhisperEncoderGraphUploadPayload::F16Bits(values) => {
                stats.bytes = stats
                    .bytes
                    .saturating_add(values.len().saturating_mul(std::mem::size_of::<u16>()));
                graph
                    .set_f16_bits_slice(upload.tensor, &values, upload.label)
                    .map_err(
                        |error| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
                            reason: format!("could not upload tensor '{}': {error}", upload.label),
                        },
                    )?
            }
            WhisperEncoderGraphUploadPayload::Bytes(values) => {
                stats.bytes = stats.bytes.saturating_add(values.len());
                graph
                    .set_bytes_slice(upload.tensor, &values, upload.label)
                    .map_err(
                        |error| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
                            reason: format!("could not upload tensor '{}': {error}", upload.label),
                        },
                    )?
            }
        }
    }
    Ok(stats)
}

fn build_encoder_resident_weight_cache<'weights>(
    runner: &GgmlCpuGraphRunner,
    context_bytes: usize,
    source_tensors: &HashMap<&str, &'weights WhisperMaterializedTensor>,
    encoder_weights: &'weights WhisperEncoderWeightBundle,
    plan: &WhisperEncoderGraphPlan,
    runtime_path: Option<&Path>,
) -> Result<WhisperEncoderResidentWeightCache, WhisperGgmlExecutorError> {
    let mut arena = runner
        .start_static_tensor_arena(context_bytes)
        .map_err(|error| map_encoder_graph_error("ggml_static_tensor_arena", error))?;
    // Bind large quantized linear weights zero-copy to the mmap'd pack (no host
    // copy, no arena upload). Falls back to the arena path when unavailable.
    let loaded_weights = runtime_path.and_then(|path| runner.load_gguf_weight_context(path).ok());
    let mut tensors_by_name = HashMap::with_capacity(source_tensors.len());
    let mut loaded_tensors_by_name = HashMap::new();
    let mut uploads: Vec<WhisperEncoderResidentWeightUpload<'weights>> = Vec::new();

    for layer_plan in &plan.layers {
        let layer_weights = encoder_weights
            .layers
            .get(layer_plan.layer_idx)
            .ok_or_else(|| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
                reason: format!(
                    "missing encoder materialized layer {} for resident weights",
                    layer_plan.layer_idx
                ),
            })?;
        add_resident_encoder_norm(
            &arena,
            source_tensors,
            &mut tensors_by_name,
            &mut uploads,
            &layer_plan.self_attn_norm,
            "resident_self_attn_norm",
        )?;
        add_resident_encoder_linear(
            &arena,
            loaded_weights.as_ref(),
            &mut loaded_tensors_by_name,
            source_tensors,
            &mut tensors_by_name,
            &mut uploads,
            &layer_plan.self_attn_q,
            "resident_self_attn_q",
        )?;
        add_resident_encoder_bias(
            &arena,
            &mut tensors_by_name,
            &mut uploads,
            &layer_weights.self_attn_q_bias,
            layer_plan.self_attn_q.output_dim,
            "resident_self_attn_q_bias",
        )?;
        add_resident_encoder_linear(
            &arena,
            loaded_weights.as_ref(),
            &mut loaded_tensors_by_name,
            source_tensors,
            &mut tensors_by_name,
            &mut uploads,
            &layer_plan.self_attn_k,
            "resident_self_attn_k",
        )?;
        add_resident_encoder_linear(
            &arena,
            loaded_weights.as_ref(),
            &mut loaded_tensors_by_name,
            source_tensors,
            &mut tensors_by_name,
            &mut uploads,
            &layer_plan.self_attn_v,
            "resident_self_attn_v",
        )?;
        add_resident_encoder_bias(
            &arena,
            &mut tensors_by_name,
            &mut uploads,
            &layer_weights.self_attn_v_bias,
            layer_plan.self_attn_v.output_dim,
            "resident_self_attn_v_bias",
        )?;
        add_resident_encoder_linear(
            &arena,
            loaded_weights.as_ref(),
            &mut loaded_tensors_by_name,
            source_tensors,
            &mut tensors_by_name,
            &mut uploads,
            &layer_plan.self_attn_out,
            "resident_self_attn_out",
        )?;
        add_resident_encoder_bias(
            &arena,
            &mut tensors_by_name,
            &mut uploads,
            &layer_weights.self_attn_out_bias,
            layer_plan.self_attn_out.output_dim,
            "resident_self_attn_out_bias",
        )?;
        add_resident_encoder_norm(
            &arena,
            source_tensors,
            &mut tensors_by_name,
            &mut uploads,
            &layer_plan.mlp_norm,
            "resident_mlp_norm",
        )?;
        add_resident_encoder_linear(
            &arena,
            loaded_weights.as_ref(),
            &mut loaded_tensors_by_name,
            source_tensors,
            &mut tensors_by_name,
            &mut uploads,
            &layer_plan.mlp_fc1,
            "resident_mlp_fc1",
        )?;
        add_resident_encoder_bias(
            &arena,
            &mut tensors_by_name,
            &mut uploads,
            &layer_weights.fc1_bias,
            layer_plan.mlp_fc1.output_dim,
            "resident_mlp_fc1_bias",
        )?;
        add_resident_encoder_linear(
            &arena,
            loaded_weights.as_ref(),
            &mut loaded_tensors_by_name,
            source_tensors,
            &mut tensors_by_name,
            &mut uploads,
            &layer_plan.mlp_fc2,
            "resident_mlp_fc2",
        )?;
        add_resident_encoder_bias(
            &arena,
            &mut tensors_by_name,
            &mut uploads,
            &layer_weights.fc2_bias,
            layer_plan.mlp_fc2.output_dim,
            "resident_mlp_fc2_bias",
        )?;
    }
    add_resident_encoder_norm(
        &arena,
        source_tensors,
        &mut tensors_by_name,
        &mut uploads,
        &plan.final_norm,
        "resident_final_norm",
    )?;

    let mut stats = WhisperEncoderGraphUploadStats { count: 0, bytes: 0 };
    for upload in uploads {
        match upload {
            WhisperEncoderResidentWeightUpload::F32 { tensor, values } => {
                stats.count = stats.count.saturating_add(1);
                stats.bytes = stats
                    .bytes
                    .saturating_add(values.len().saturating_mul(std::mem::size_of::<f32>()));
                arena
                    .set_f32_slice(tensor, &values, "resident_encoder_weight")
                    .map_err(|error| map_encoder_graph_error("resident_encoder_weight", error))?;
            }
            WhisperEncoderResidentWeightUpload::F16BitsBorrowed { tensor, values } => {
                stats.count = stats.count.saturating_add(1);
                stats.bytes = stats
                    .bytes
                    .saturating_add(values.len().saturating_mul(std::mem::size_of::<u16>()));
                arena
                    .set_f16_bits_slice(tensor, values, "resident_encoder_weight")
                    .map_err(|error| map_encoder_graph_error("resident_encoder_weight", error))?;
            }
            WhisperEncoderResidentWeightUpload::F16BitsOwned { tensor, values } => {
                stats.count = stats.count.saturating_add(1);
                stats.bytes = stats
                    .bytes
                    .saturating_add(values.len().saturating_mul(std::mem::size_of::<u16>()));
                arena
                    .set_f16_bits_slice(tensor, &values, "resident_encoder_weight")
                    .map_err(|error| map_encoder_graph_error("resident_encoder_weight", error))?;
            }
            WhisperEncoderResidentWeightUpload::QuantizedBytesBorrowed { tensor, values } => {
                stats.count = stats.count.saturating_add(1);
                stats.bytes = stats.bytes.saturating_add(values.len());
                arena
                    .set_bytes_slice(tensor, values, "resident_encoder_weight")
                    .map_err(|error| map_encoder_graph_error("resident_encoder_weight", error))?;
            }
        }
    }

    Ok(WhisperEncoderResidentWeightCache {
        arena,
        tensors_by_name,
        _loaded: loaded_weights,
        loaded_tensors_by_name,
        upload_stats: stats,
    })
}

fn add_resident_encoder_norm<'weights>(
    arena: &GgmlStaticTensorArena,
    source_tensors: &HashMap<&str, &'weights WhisperMaterializedTensor>,
    tensors_by_name: &mut HashMap<String, GgmlStaticTensor>,
    uploads: &mut Vec<WhisperEncoderResidentWeightUpload<'weights>>,
    norm: &WhisperEncoderNormPlan,
    label: &'static str,
) -> Result<(), WhisperGgmlExecutorError> {
    let hidden = usize::try_from(*norm.weight.dims.last().ok_or_else(|| {
        WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: format!("norm tensor '{}' is missing dims", norm.weight.tensor_name),
        }
    })?)
    .map_err(|_| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
        reason: format!(
            "norm tensor '{}' hidden dimension does not fit usize",
            norm.weight.tensor_name
        ),
    })?;
    add_resident_encoder_f32_vector(
        arena,
        source_tensors,
        tensors_by_name,
        uploads,
        &norm.weight,
        hidden,
        label,
    )?;
    add_resident_encoder_f32_vector(
        arena,
        source_tensors,
        tensors_by_name,
        uploads,
        &norm.bias,
        hidden,
        label,
    )
}

fn add_resident_encoder_bias<'weights>(
    arena: &GgmlStaticTensorArena,
    tensors_by_name: &mut HashMap<String, GgmlStaticTensor>,
    uploads: &mut Vec<WhisperEncoderResidentWeightUpload<'weights>>,
    tensor: &'weights WhisperMaterializedTensor,
    expected_len: usize,
    label: &'static str,
) -> Result<(), WhisperGgmlExecutorError> {
    add_resident_encoder_materialized_f32_vector(
        arena,
        tensors_by_name,
        uploads,
        tensor,
        expected_len,
        label,
    )
}

fn add_resident_encoder_f32_vector<'weights>(
    arena: &GgmlStaticTensorArena,
    source_tensors: &HashMap<&str, &'weights WhisperMaterializedTensor>,
    tensors_by_name: &mut HashMap<String, GgmlStaticTensor>,
    uploads: &mut Vec<WhisperEncoderResidentWeightUpload<'weights>>,
    tensor_ref: &WhisperEncoderGraphTensorRef,
    expected_len: usize,
    label: &'static str,
) -> Result<(), WhisperGgmlExecutorError> {
    let tensor = lookup_encoder_tensor_for_resident(source_tensors, &tensor_ref.tensor_name)?;
    add_resident_encoder_materialized_f32_vector(
        arena,
        tensors_by_name,
        uploads,
        tensor,
        expected_len,
        label,
    )
}

fn add_resident_encoder_materialized_f32_vector<'weights>(
    arena: &GgmlStaticTensorArena,
    tensors_by_name: &mut HashMap<String, GgmlStaticTensor>,
    uploads: &mut Vec<WhisperEncoderResidentWeightUpload<'weights>>,
    tensor: &'weights WhisperMaterializedTensor,
    expected_len: usize,
    label: &'static str,
) -> Result<(), WhisperGgmlExecutorError> {
    if tensors_by_name.contains_key(&tensor.tensor_name) {
        return Ok(());
    }
    let values = encoder_tensor_tail_f32_values(tensor, expected_len)
        .map_err(|reason| WhisperGgmlExecutorError::EncoderGraphExecutionFailed { reason })?;
    let static_tensor = arena
        .new_tensor_1d_f32(expected_len, label)
        .map_err(|error| map_encoder_graph_error("ggml_new_static_tensor_1d(f32)", error))?;
    tensors_by_name.insert(tensor.tensor_name.clone(), static_tensor);
    uploads.push(WhisperEncoderResidentWeightUpload::F32 {
        tensor: static_tensor,
        values: values.into_owned(),
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn add_resident_encoder_linear<'weights>(
    arena: &GgmlStaticTensorArena,
    loaded_weights: Option<&GgmlLoadedWeightContext>,
    loaded_tensors_by_name: &mut HashMap<String, GgmlLoadedTensor>,
    source_tensors: &HashMap<&str, &'weights WhisperMaterializedTensor>,
    tensors_by_name: &mut HashMap<String, GgmlStaticTensor>,
    uploads: &mut Vec<WhisperEncoderResidentWeightUpload<'weights>>,
    projection: &WhisperEncoderLinearProjectionPlan,
    label: &'static str,
) -> Result<(), WhisperGgmlExecutorError> {
    if tensors_by_name.contains_key(&projection.weight.tensor_name)
        || loaded_tensors_by_name.contains_key(&projection.weight.tensor_name)
    {
        return Ok(());
    }
    let tensor =
        lookup_encoder_tensor_for_resident(source_tensors, &projection.weight.tensor_name)?;
    let expected_len = projection
        .input_dim
        .checked_mul(projection.output_dim)
        .ok_or_else(|| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: format!(
                "linear projection '{}' dimensions overflow: {}x{}",
                projection.weight.tensor_name, projection.input_dim, projection.output_dim
            ),
        })?;
    if tensor.num_elements != expected_len {
        return Err(WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: format!(
                "linear projection '{}' has {} values but expected {}",
                projection.weight.tensor_name, tensor.num_elements, expected_len
            ),
        });
    }
    // Zero-copy bind: a quantized input-output linear weight is uploaded to the
    // arena verbatim (no transpose/dequant), so the mmap'd pack bytes are
    // bit-identical to what the arena would hold. Bind it directly when the
    // loaded context exposes a tensor of the same on-disk name.
    if let WhisperMaterializedTensorPayload::Quantized { .. } = &tensor.payload
        && projection.weight_layout == WhisperEncoderLinearWeightLayout::InputOutput
        && let Some(loaded) =
            loaded_weights.and_then(|ctx| ctx.tensor(&projection.weight.tensor_name))
    {
        loaded_tensors_by_name.insert(projection.weight.tensor_name.clone(), loaded);
        return Ok(());
    }
    let static_tensor = match &tensor.payload {
        WhisperMaterializedTensorPayload::Quantized { ggml_type, .. } => arena
            .new_matmul_weight_2d_typed(
                projection.input_dim,
                projection.output_dim,
                *ggml_type,
                label,
            )
            .map_err(|error| {
                map_encoder_graph_error("ggml_new_static_tensor_2d(quantized)", error)
            })?,
        _ => arena
            .new_tensor_2d_f16(projection.input_dim, projection.output_dim, label)
            .map_err(|error| map_encoder_graph_error("ggml_new_static_tensor_2d(f16)", error))?,
    };
    tensors_by_name.insert(projection.weight.tensor_name.clone(), static_tensor);
    match &tensor.payload {
        WhisperMaterializedTensorPayload::Quantized {
            ggml_type: _,
            bytes,
        } => {
            if projection.weight_layout != WhisperEncoderLinearWeightLayout::InputOutput {
                return Err(WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
                    reason: format!(
                        "quantized linear projection '{}' requires input-output layout",
                        projection.weight.tensor_name
                    ),
                });
            }
            uploads.push(WhisperEncoderResidentWeightUpload::QuantizedBytesBorrowed {
                tensor: static_tensor,
                values: bytes.as_slice(),
            });
        }
        WhisperMaterializedTensorPayload::F16Bits(values)
            if projection.weight_layout == WhisperEncoderLinearWeightLayout::InputOutput =>
        {
            uploads.push(WhisperEncoderResidentWeightUpload::F16BitsBorrowed {
                tensor: static_tensor,
                values,
            });
        }
        _ => {
            let mut values = encoder_tensor_values_f16_bits_lossy(tensor).map_err(|reason| {
                WhisperGgmlExecutorError::EncoderGraphExecutionFailed { reason }
            })?;
            if projection.weight_layout == WhisperEncoderLinearWeightLayout::OutputInput {
                values = transpose_linear_weight_output_input_to_input_output_u16(
                    &values,
                    projection.input_dim,
                    projection.output_dim,
                )?;
            }
            uploads.push(WhisperEncoderResidentWeightUpload::F16BitsOwned {
                tensor: static_tensor,
                values,
            });
        }
    }
    Ok(())
}

fn emit_encoder_graph_detail_trace(
    upload_count: usize,
    upload_bytes: usize,
    graph_build_ms: u128,
    upload_ms: u128,
    buffer_alloc_ms: u128,
    tensor_set_ms: u128,
    compute_ms: u128,
    total_ms: u128,
) {
    if std::env::var_os(OPENASR_WHISPER_GGML_TRACE_ENV).is_none() {
        return;
    }
    eprintln!(
        "openasr_whisper_ggml_trace stage=encoder_graph event=detail status=ok upload_count={upload_count} upload_bytes={upload_bytes} graph_build_ms={graph_build_ms} upload_ms={upload_ms} buffer_alloc_ms={buffer_alloc_ms} tensor_set_ms={tensor_set_ms} compute_ms={compute_ms} total_ms={total_ms}"
    );
}

fn emit_encoder_resident_weight_trace(upload_count: usize, upload_bytes: usize, total_ms: u128) {
    if std::env::var_os(OPENASR_WHISPER_GGML_TRACE_ENV).is_none() {
        return;
    }
    eprintln!(
        "openasr_whisper_ggml_trace stage=encoder_resident_weights event=detail status=ok upload_count={upload_count} upload_bytes={upload_bytes} total_ms={total_ms}"
    );
}

fn emit_encoder_resident_weight_cache_reuse_trace() {
    if std::env::var_os(OPENASR_WHISPER_GGML_TRACE_ENV).is_none() {
        return;
    }
    eprintln!(
        "openasr_whisper_ggml_trace stage=encoder_resident_weights event=detail status=reused upload_count=0 upload_bytes=0 total_ms=0"
    );
}

fn whisper_encoder_resident_weights_enabled() -> bool {
    // Keep the opt-out as a correctness/perf escape hatch while resident
    // encoder weights are the default fast path.
    !matches!(
        std::env::var("OPENASR_WHISPER_GGML_RESIDENT_ENCODER_WEIGHTS")
            .ok()
            .as_deref()
            .map(str::trim),
        Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("off")
    )
}

fn whisper_encoder_graph_config() -> GgmlCpuGraphConfig {
    whisper_runtime_graph_config()
}

fn apply_encoder_affine_layer_norm<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    uploads: &mut Vec<WhisperEncoderGraphUpload<'a>>,
    encoder_tensors: &HashMap<&str, &WhisperMaterializedTensor>,
    resident_weights: Option<&WhisperEncoderResidentWeightCache>,
    input_tensor: GgmlCpuTensor<'a>,
    layer_norm_epsilon: f32,
    norm: &WhisperEncoderNormPlan,
) -> Result<GgmlCpuTensor<'a>, WhisperGgmlExecutorError> {
    let hidden = usize::try_from(*norm.weight.dims.last().ok_or_else(|| {
        WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: format!("norm tensor '{}' is missing dims", norm.weight.tensor_name),
        }
    })?)
    .map_err(|_| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
        reason: format!(
            "norm tensor '{}' hidden dimension does not fit usize",
            norm.weight.tensor_name
        ),
    })?;
    let weight_tensor = if let Some(tensor) =
        resident_weights.and_then(|resident| resident.graph_tensor(&norm.weight.tensor_name))
    {
        tensor
    } else {
        let weight = lookup_encoder_tensor_for_graph(encoder_tensors, &norm.weight.tensor_name)?;
        let weight_f32 = encoder_tensor_tail_f32_values(weight, hidden)
            .map_err(|reason| WhisperGgmlExecutorError::EncoderGraphExecutionFailed { reason })?;
        let weight_tensor = graph
            .new_tensor_1d_f32(hidden, "encoder_norm_weight")
            .map_err(|error| map_encoder_graph_error("ggml_new_tensor_1d(norm_weight)", error))?;
        graph
            .set_input(weight_tensor)
            .map_err(|error| map_encoder_graph_error("ggml_set_input(norm_weight)", error))?;
        uploads.push(WhisperEncoderGraphUpload::f32_owned(
            weight_tensor,
            weight_f32.into_owned(),
            "encoder_norm_weight",
        ));
        weight_tensor
    };

    let bias_tensor = if let Some(tensor) =
        resident_weights.and_then(|resident| resident.graph_tensor(&norm.bias.tensor_name))
    {
        tensor
    } else {
        let bias = lookup_encoder_tensor_for_graph(encoder_tensors, &norm.bias.tensor_name)?;
        let bias_f32 = encoder_tensor_tail_f32_values(bias, hidden)
            .map_err(|reason| WhisperGgmlExecutorError::EncoderGraphExecutionFailed { reason })?;
        let bias_tensor = graph
            .new_tensor_1d_f32(hidden, "encoder_norm_bias")
            .map_err(|error| map_encoder_graph_error("ggml_new_tensor_1d(norm_bias)", error))?;
        graph
            .set_input(bias_tensor)
            .map_err(|error| map_encoder_graph_error("ggml_set_input(norm_bias)", error))?;
        uploads.push(WhisperEncoderGraphUpload::f32_owned(
            bias_tensor,
            bias_f32.into_owned(),
            "encoder_norm_bias",
        ));
        bias_tensor
    };
    apply_affine_layer_norm(
        graph,
        input_tensor,
        layer_norm_epsilon,
        weight_tensor,
        bias_tensor,
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "ggml_mul(norm_weight)",
            bias: "ggml_add(norm_bias)",
        },
        map_encoder_graph_error,
    )
}

fn apply_encoder_linear_projection<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    uploads: &mut Vec<WhisperEncoderGraphUpload<'a>>,
    encoder_tensors: &HashMap<&str, &WhisperMaterializedTensor>,
    resident_weights: Option<&WhisperEncoderResidentWeightCache>,
    input_tensor: GgmlCpuTensor<'a>,
    projection: &WhisperEncoderLinearProjectionPlan,
) -> Result<GgmlCpuTensor<'a>, WhisperGgmlExecutorError> {
    if let Some(weight_tensor) =
        resident_weights.and_then(|resident| resident.graph_tensor(&projection.weight.tensor_name))
    {
        return graph
            .mul_mat(weight_tensor, input_tensor)
            .map_err(|error| map_encoder_graph_error("ggml_mul_mat(linear)", error));
    }
    let weight = lookup_encoder_tensor_for_graph(encoder_tensors, &projection.weight.tensor_name)?;
    if let WhisperMaterializedTensorPayload::Quantized { ggml_type, bytes } = &weight.payload {
        if projection.weight_layout != WhisperEncoderLinearWeightLayout::InputOutput {
            return Err(WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
                reason: format!(
                    "quantized linear projection '{}' requires input-output layout",
                    projection.weight.tensor_name
                ),
            });
        }
        let weight_tensor = graph
            .new_matmul_weight_2d_typed(
                projection.input_dim,
                projection.output_dim,
                *ggml_type,
                "encoder_linear_weight",
            )
            .map_err(|error| {
                map_encoder_graph_error("ggml_new_tensor_2d(linear_weight_quant)", error)
            })?;
        graph.set_input(weight_tensor).map_err(|error| {
            map_encoder_graph_error("ggml_set_input(linear_weight_quant)", error)
        })?;
        uploads.push(WhisperEncoderGraphUpload::bytes(
            weight_tensor,
            bytes.to_vec(),
            "encoder_linear_weight_quant",
        ));
        return graph
            .mul_mat(weight_tensor, input_tensor)
            .map_err(|error| map_encoder_graph_error("ggml_mul_mat(linear)", error));
    }
    let mut weight_f16_bits = encoder_tensor_values_f16_bits_lossy(weight)
        .map_err(|reason| WhisperGgmlExecutorError::EncoderGraphExecutionFailed { reason })?;
    let expected_len = projection
        .input_dim
        .checked_mul(projection.output_dim)
        .ok_or_else(|| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: format!(
                "linear projection '{}' dimensions overflow: {}x{}",
                projection.weight.tensor_name, projection.input_dim, projection.output_dim
            ),
        })?;
    if weight_f16_bits.len() != expected_len {
        return Err(WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: format!(
                "linear projection '{}' has {} values but expected {}",
                projection.weight.tensor_name,
                weight_f16_bits.len(),
                expected_len
            ),
        });
    }
    if projection.weight_layout == WhisperEncoderLinearWeightLayout::OutputInput {
        weight_f16_bits = transpose_linear_weight_output_input_to_input_output_u16(
            &weight_f16_bits,
            projection.input_dim,
            projection.output_dim,
        )?;
    }
    let weight_tensor = graph
        .new_tensor_2d_f16(
            projection.input_dim,
            projection.output_dim,
            "encoder_linear_weight",
        )
        .map_err(|error| map_encoder_graph_error("ggml_new_tensor_2d(linear_weight)", error))?;
    graph
        .set_input(weight_tensor)
        .map_err(|error| map_encoder_graph_error("ggml_set_input(linear_weight)", error))?;
    uploads.push(WhisperEncoderGraphUpload::f16_bits(
        weight_tensor,
        weight_f16_bits,
        "encoder_linear_weight",
    ));
    graph
        .mul_mat(weight_tensor, input_tensor)
        .map_err(|error| map_encoder_graph_error("ggml_mul_mat(linear)", error))
}

fn add_encoder_bias_tensor<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    uploads: &mut Vec<WhisperEncoderGraphUpload<'a>>,
    resident_weights: Option<&WhisperEncoderResidentWeightCache>,
    input_tensor: GgmlCpuTensor<'a>,
    bias_tensor: &WhisperMaterializedTensor,
    expected_len: usize,
    label: &'static str,
) -> Result<GgmlCpuTensor<'a>, WhisperGgmlExecutorError> {
    if let Some(bias) =
        resident_weights.and_then(|resident| resident.graph_tensor(&bias_tensor.tensor_name))
    {
        return graph
            .add(input_tensor, bias)
            .map_err(|error| map_encoder_graph_error("ggml_add(linear_bias)", error));
    }
    let bias_f32 = encoder_tensor_tail_f32_values(bias_tensor, expected_len)
        .map_err(|reason| WhisperGgmlExecutorError::EncoderGraphExecutionFailed { reason })?;
    let bias = graph
        .new_tensor_1d_f32(expected_len, label)
        .map_err(|error| map_encoder_graph_error("ggml_new_tensor_1d(linear_bias)", error))?;
    graph
        .set_input(bias)
        .map_err(|error| map_encoder_graph_error("ggml_set_input(linear_bias)", error))?;
    uploads.push(WhisperEncoderGraphUpload::f32_owned(
        bias,
        bias_f32.into_owned(),
        label,
    ));
    graph
        .add(input_tensor, bias)
        .map_err(|error| map_encoder_graph_error("ggml_add(linear_bias)", error))
}

fn encoder_tensor_values_f16_bits_lossy(
    tensor: &WhisperMaterializedTensor,
) -> Result<Vec<u16>, String> {
    let values = match &tensor.payload {
        WhisperMaterializedTensorPayload::F16Bits(values) => values.clone(),
        WhisperMaterializedTensorPayload::F32(values) => {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "encoder tensor '{}' materialized non-finite values",
                    tensor.tensor_name
                ));
            }
            values.iter().map(|value| f32_to_f16_bits(*value)).collect()
        }
        WhisperMaterializedTensorPayload::Quantized { ggml_type, .. } => {
            return Err(format!(
                "encoder tensor '{}' is quantized (ggml type {ggml_type}); f16 lossy conversion path is disabled",
                tensor.tensor_name
            ));
        }
    };
    if values.len() != tensor.num_elements {
        return Err(format!(
            "encoder tensor '{}' materialized {} values but metadata expects {}",
            tensor.tensor_name,
            values.len(),
            tensor.num_elements
        ));
    }
    Ok(values)
}

fn transpose_linear_weight_output_input_to_input_output_u16(
    source: &[u16],
    input_dim: usize,
    output_dim: usize,
) -> Result<Vec<u16>, WhisperGgmlExecutorError> {
    if source.len() != input_dim * output_dim {
        return Err(WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: format!(
                "cannot transpose linear weight with {} values for {}x{}",
                source.len(),
                output_dim,
                input_dim
            ),
        });
    }
    let mut transposed = vec![0_u16; source.len()];
    for out_idx in 0..output_dim {
        for in_idx in 0..input_dim {
            let src = in_idx + out_idx * input_dim;
            let dst = in_idx + out_idx * input_dim;
            transposed[dst] = source[src];
        }
    }
    Ok(transposed)
}

fn transpose_sequence_hidden_to_hidden_sequence(
    input: &[f32],
    frames: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for frame_idx in 0..frames {
        for hidden_idx in 0..hidden {
            let src = frame_idx * hidden + hidden_idx;
            let dst = hidden_idx * frames + frame_idx;
            output[dst] = input[src];
        }
    }
    output
}

impl WhisperMelFeatureInputProvider for WhisperMelFeatureInputProviderFrontendV0 {
    fn provider_id(&self) -> &'static str {
        "whisper-mel-feature-input-frontend-v0"
    }

    fn prepare_mel_feature_input(
        &self,
        execution: &WhisperGgmlExecutionMetadata,
        prepared_audio: &GgmlAsrPreparedAudio,
    ) -> Result<WhisperMelFeatureInput, WhisperGgmlExecutorError> {
        if prepared_audio.sample_rate_hz != WHISPER_SAMPLE_RATE_HZ {
            return Err(WhisperGgmlExecutorError::MelFeatureInputPreparationFailed {
                reason: format!(
                    "sample_rate_hz={} (expected {WHISPER_SAMPLE_RATE_HZ})",
                    prepared_audio.sample_rate_hz
                ),
            });
        }
        if prepared_audio.channels != WHISPER_CHANNELS {
            return Err(WhisperGgmlExecutorError::MelFeatureInputPreparationFailed {
                reason: format!(
                    "channels={} (expected {WHISPER_CHANNELS})",
                    prepared_audio.channels
                ),
            });
        }
        if prepared_audio.samples_f32.is_empty() {
            return Err(WhisperGgmlExecutorError::MelFeatureInputPreparationFailed {
                reason: "samples_f32 is empty".to_string(),
            });
        }
        if prepared_audio
            .samples_f32
            .iter()
            .any(|sample| !sample.is_finite())
        {
            return Err(WhisperGgmlExecutorError::MelFeatureInputPreparationFailed {
                reason: "samples_f32 contains non-finite values".to_string(),
            });
        }
        let target_frames = execution
            .encoder_context_length
            .checked_mul(2)
            .ok_or_else(
                || WhisperGgmlExecutorError::MelFeatureInputPreparationFailed {
                    reason: format!(
                        "encoder_context_length={} overflows target mel frame inference",
                        execution.encoder_context_length
                    ),
                },
            )?;
        let mel = whisper_mel_features_from_prepared_audio_v0(
            prepared_audio,
            execution.encoder_mels_count,
            target_frames,
        )
        .map_err(|error| WhisperGgmlExecutorError::MelFeatureExtractionFailed {
            reason: format!(
                "source='wav-mono-f32-16khz' provider='{}' sample_count={} mels={} target_frames={} frontend_error={error}",
                self.provider_id(),
                prepared_audio.samples_f32.len(),
                execution.encoder_mels_count,
                target_frames
            ),
        })?;
        if mel.n_mels != execution.encoder_mels_count {
            return Err(WhisperGgmlExecutorError::MelFeatureExtractionFailed {
                reason: format!(
                    "source='wav-mono-f32-16khz' provider='{}' returned n_mels={} but metadata requires {}",
                    self.provider_id(),
                    mel.n_mels,
                    execution.encoder_mels_count
                ),
            });
        }
        if mel.n_frames != target_frames {
            return Err(WhisperGgmlExecutorError::MelFeatureExtractionFailed {
                reason: format!(
                    "source='wav-mono-f32-16khz' provider='{}' returned n_frames={} but expected {}",
                    self.provider_id(),
                    mel.n_frames,
                    target_frames
                ),
            });
        }

        Ok(WhisperMelFeatureInput {
            source_label: self.provider_id(),
            shape: WhisperMelFeatureInputShape {
                mel_bins: mel.n_mels,
                mel_frames: mel.n_frames,
            },
            values_f32: mel.data,
        })
    }
}

impl WhisperDecoderLoopRunner for WhisperDecoderGraphRunnerGgmlV0 {
    fn runner_id(&self) -> &'static str {
        "whisper-decoder-graph-ggml-v0"
    }

    fn step_logits(
        &self,
        _runtime_source: &GgmlRuntimeSource,
        execution: &WhisperGgmlExecutionMetadata,
        decoder_weights: &WhisperDecoderWeightSeam,
        plan: &WhisperDecoderGraphPlan,
        graph_input: &WhisperDecoderGraphExecutionInput,
        graph_config: WhisperDecoderGraphExecutionConfig,
        graph_runner: &mut GgmlCpuGraphRunner,
        persistent_weights: Option<&WhisperDecoderPersistentWeightCache>,
        self_kv_state: Option<&WhisperDecoderSelfKvCacheState>,
        tensor_cache: &mut WhisperDecoderExecutionTensorCache,
        decode_input: &WhisperDecoderStepSeamInput,
    ) -> Result<WhisperDecoderStepLogits, WhisperGgmlExecutorError> {
        let token_count = graph_input.decoder_prefix_tokens.len();
        if token_count == 0 {
            return Err(WhisperGgmlExecutorError::DecoderGraphUnsupported {
                reason: "decoder prefix token_count must be > 0".to_string(),
            });
        }

        let graph_run_start = Instant::now();
        let output = run_whisper_decoder_greedy_step_with_cache_and_runner_ggml_v0(
            graph_runner,
            persistent_weights,
            self_kv_state,
            decode_input.position_offset,
            plan,
            graph_input,
            &decoder_weights.tensor_source,
            graph_config,
            tensor_cache,
        )
        .map_err(|error| {
            map_decoder_graph_execution_error(
                self.runner_id(),
                decode_input.step_index,
                token_count,
                error,
            )
        })?;
        let decoder_graph_run_ms = graph_run_start.elapsed().as_millis();
        let logits_start = Instant::now();
        if output.logits.len() != execution.vocab_size {
            return Err(WhisperGgmlExecutorError::DecoderGraphExecutionFailed {
                reason: format!(
                    "runner '{}' returned logits width mismatch at step {}: got {}, expected {}",
                    self.runner_id(),
                    decode_input.step_index,
                    output.logits.len(),
                    execution.vocab_size
                ),
            });
        }
        Ok(WhisperDecoderStepLogits {
            logits: output.logits,
            greedy_token_hint: Some(output.greedy_token),
            last_token_cross_attention_frame_probs: output.last_token_cross_attention_frame_probs,
            decoder_graph_run_ms,
            logits_ms: logits_start.elapsed().as_millis(),
        })
    }
}

impl WhisperTokenizerProvider for WhisperTokenizerProviderGgufV0 {
    fn provider_id(&self) -> &'static str {
        "whisper-tokenizer-gguf-v0"
    }

    fn load_tokenizer(
        &self,
        _runtime_source: &GgmlRuntimeSource,
        metadata: &GgufMetadata,
    ) -> Result<WhisperTokenizer, WhisperGgmlExecutorError> {
        materialize_builtin_tokenizer_for_architecture(crate::WHISPER_GGML_ARCHITECTURE_ID, metadata)
            .map_err(|error| WhisperGgmlExecutorError::TokenizerMissing {
                reason: format!(
                    "provider '{}' could not materialize tokenizer from preflight GGUF metadata: {error}",
                    self.provider_id()
                ),
            })?
            .into_whisper()
            .ok_or_else(|| WhisperGgmlExecutorError::TokenizerMissing {
                reason: format!(
                    "provider '{}' resolved non-whisper tokenizer component for whisper architecture",
                    self.provider_id()
                ),
            })
    }
}

#[derive(Clone)]
pub(crate) struct WhisperGgmlExecutor {
    mel_feature_input_provider: Arc<dyn WhisperMelFeatureInputProvider>,
    encoder_prelude_runner: Arc<dyn WhisperEncoderPreludeRunner>,
    encoder_graph_runner: Arc<dyn WhisperEncoderGraphRunner>,
    decoder_runner: Arc<dyn WhisperDecoderLoopRunner>,
    tokenizer_provider: Arc<dyn WhisperTokenizerProvider>,
    runtime_cache_by_path: PreparedRuntimeCache<WhisperPreparedRuntime>,
}

impl Default for WhisperGgmlExecutor {
    fn default() -> Self {
        Self {
            mel_feature_input_provider: Arc::new(WhisperMelFeatureInputProviderFrontendV0),
            encoder_prelude_runner: Arc::new(WhisperCpuEncoderPreludeComputeRunnerV0),
            encoder_graph_runner: Arc::new(WhisperCpuEncoderGraphComputeRunnerV0),
            decoder_runner: Arc::new(WhisperDecoderGraphRunnerGgmlV0),
            tokenizer_provider: Arc::new(WhisperTokenizerProviderGgufV0),
            runtime_cache_by_path: PreparedRuntimeCache::default(),
        }
    }
}

impl WhisperGgmlExecutor {
    fn prepared_runtime_for_preflight(
        &self,
        preflight: &crate::GgmlAsrRuntimeSourcePreflight,
    ) -> Result<Arc<WhisperPreparedRuntime>, WhisperGgmlExecutorError> {
        let runtime_path = preflight.runtime_source.path();
        self.runtime_cache_by_path.get_or_try_insert_with(
            runtime_path,
            || {
                build_whisper_prepared_runtime(
                    &preflight.runtime_source,
                    &preflight.metadata,
                    &preflight.tensor_index,
                    self.tokenizer_provider.as_ref(),
                )
            },
            whisper_runtime_cache_slot_unavailable,
        )
    }
}

// Covers both a genuinely poisoned slot mutex (a prior caller panicked while
// holding it -- extremely unlikely, see `PreparedRuntimeCache::get_or_try_insert_with`)
// and a build attempt that panicked and was caught (mutex stays unpoisoned,
// slot stays empty, retryable). Either way the cache could not deliver a
// prepared runtime for this attempt; the caller's next request retries clean.
fn whisper_runtime_cache_slot_unavailable() -> WhisperGgmlExecutorError {
    WhisperGgmlExecutorError::TensorMaterializationFailed {
        reason:
            "whisper runtime cache slot unavailable (poisoned lock or a caught build panic); retry"
                .to_string(),
    }
}

impl GgmlAsrExecutor for WhisperGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        "whisper-ggml-executor-v1"
    }

    fn supports_phrase_bias(&self) -> bool {
        true
    }

    fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        // Offline decode: batch worker allowed.
        self.execute_whisper_inner(request, false)
    }

    fn unload_idle_state(&self) {
        self.runtime_cache_by_path.clear();
    }
}

impl WhisperGgmlExecutor {
    /// Streaming decode bypasses the batch worker so live sessions stay on the
    /// direct greedy loop. The FINAL transcript remains byte-identical to `execute`.
    pub(crate) fn execute_streaming(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_whisper_inner(request, true)
    }

    fn execute_whisper_inner(
        &self,
        request: &GgmlAsrExecutionRequest,
        skip_serve_batch: bool,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(|error| GgmlAsrExecutionError::ExecutorFailed {
                executor_id: GgmlAsrExecutor::executor_id(self),
                adapter_id: request.selected_family.adapter_id,
                reason: error.to_string(),
            })?;
        let reuse_runtime_state = request.request_options.longform.is_some();
        let output = if reuse_runtime_state {
            let prepared_runtime = self
                .prepared_runtime_for_preflight(preflight.as_ref())
                .map_err(|error| GgmlAsrExecutionError::ExecutorFailed {
                    executor_id: GgmlAsrExecutor::executor_id(self),
                    adapter_id: request.selected_family.adapter_id,
                    reason: error.to_string(),
                })?;
            execute_whisper_with_prepared_runtime(
                &request.selected_family,
                &preflight.runtime_source,
                &request.prepared_audio,
                prepared_runtime.as_ref(),
                &request.request_options,
                self.mel_feature_input_provider.as_ref(),
                self.encoder_prelude_runner.as_ref(),
                self.encoder_graph_runner.as_ref(),
                self.decoder_runner.as_ref(),
                true,
                skip_serve_batch,
            )
        } else {
            let runtime = build_whisper_prepared_runtime(
                &preflight.runtime_source,
                &preflight.metadata,
                &preflight.tensor_index,
                self.tokenizer_provider.as_ref(),
            );
            runtime.and_then(|runtime| {
                execute_whisper_with_prepared_runtime(
                    &request.selected_family,
                    &preflight.runtime_source,
                    &request.prepared_audio,
                    &runtime,
                    &request.request_options,
                    self.mel_feature_input_provider.as_ref(),
                    self.encoder_prelude_runner.as_ref(),
                    self.encoder_graph_runner.as_ref(),
                    self.decoder_runner.as_ref(),
                    false,
                    skip_serve_batch,
                )
            })
        }
        .map_err(|error| match error {
            WhisperGgmlExecutorError::ServeBatchUnavailable { reason, retryable } => {
                GgmlAsrExecutionError::ServeBatchUnavailable { reason, retryable }
            }
            error => GgmlAsrExecutionError::ExecutorFailed {
                executor_id: GgmlAsrExecutor::executor_id(self),
                adapter_id: request.selected_family.adapter_id,
                reason: error.to_string(),
            },
        })?;

        Ok(GgmlAsrExecutionResult {
            transcription: Transcription {
                text: output.text,
                segments: output.segments,
                longform: None,
                language: output.detected_language,
            },
            carry_context: output.carry_prompt_token_ids.map(|prompt_token_ids| {
                GgmlAsrCarryContext {
                    prompt_text: None,
                    prompt_token_ids: Some(prompt_token_ids),
                }
            }),
        })
    }
}

impl GgmlAsrStreamingExecutor for WhisperGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        WHISPER_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        build_seq2seq_streaming_session(
            self.clone(),
            WHISPER_STREAMING_EXECUTOR_ID,
            WHISPER_GGML_ADAPTER_ID,
            "whisper",
            request,
            STREAMING_PARTIAL_TUNING_WHISPER_SEQ2SEQ,
            WhisperGgmlExecutor::execute_streaming,
        )
    }

    fn unload_idle_state(&self) {
        self.runtime_cache_by_path.clear();
    }
}

fn build_whisper_prepared_runtime(
    runtime_source: &GgmlRuntimeSource,
    metadata: &GgufMetadata,
    tensor_index: &GgufTensorIndex,
    tokenizer_provider: &dyn WhisperTokenizerProvider,
) -> Result<WhisperPreparedRuntime, WhisperGgmlExecutorError> {
    let execution =
        validate_whisper_execution_metadata(metadata).map_err(map_metadata_contract_error)?;
    let tensor_binding = bind_whisper_required_tensors(tensor_index, &execution)?;
    let tensor_reader = GgufTensorDataReader::from_tensor_index_shared(Arc::clone(
        &tensor_binding.weights.tensor_index,
    ))
    .map_err(map_tensor_materialization_error)?;
    let mut encoder_weights =
        materialize_whisper_encoder_weights_from_reader(&tensor_binding, &tensor_reader)?;
    prepare_encoder_runtime_weight_payloads(&mut encoder_weights)?;
    let encoder_materialization = materialize_whisper_encoder_tensor_seam(&encoder_weights);
    let encoder_binding = build_encoder_graph_binding_seam(&encoder_weights, &execution)?;
    let decoder_weights =
        build_decoder_weight_seam(&tensor_reader, &tensor_binding.weights.bindings)?;
    let tokenizer = tokenizer_provider.load_tokenizer(runtime_source, metadata)?;
    Ok(WhisperPreparedRuntime {
        execution,
        tensor_binding,
        encoder_weights,
        encoder_materialization,
        encoder_binding,
        decoder_weights,
        tokenizer,
    })
}

fn whisper_prefix_error_to_executor_error(error: WhisperPrefixError) -> WhisperGgmlExecutorError {
    match error {
        WhisperPrefixError::LanguageTokenMissing { language } => {
            WhisperGgmlExecutorError::UnsupportedRequestOption {
                option: "language",
                reason: format!("this whisper pack has no <|{language}|> language token"),
            }
        }
        WhisperPrefixError::TranslateTokenMissing => {
            WhisperGgmlExecutorError::UnsupportedRequestOption {
                option: "task",
                reason: "this whisper pack has no <|translate|> task token".to_string(),
            }
        }
    }
}

fn build_whisper_initial_prompt_tokens(
    execution: &WhisperGgmlExecutionMetadata,
    tokenizer: &WhisperTokenizer,
    request_options: &GgmlAsrExecutionOptions,
    // When set (whisper LID detected a language for an `auto` request), it takes
    // precedence over the request language for prefix construction. `Some("en")`
    // is byte-identical to the unset path, so detecting English is a no-op.
    override_language: Option<&str>,
) -> Result<Vec<u32>, WhisperGgmlExecutorError> {
    let decoder_start_token_id = tokenizer
        .start_of_transcript_token_id()
        .unwrap_or(execution.decoder_start_token_id);
    let is_multilingual = execution.vocab_size > WHISPER_ENGLISH_ONLY_MAX_VOCAB_SIZE;
    let prefix_spec = WhisperPrefixSpec {
        language: override_language.or(request_options.language.as_deref()),
        task: request_options.task,
        is_multilingual,
    };
    let prompt_init_tokens = tokenizer
        .decoder_prefix(decoder_start_token_id, &prefix_spec)
        .map_err(whisper_prefix_error_to_executor_error)?;
    if prompt_init_tokens.is_empty() {
        return Err(WhisperGgmlExecutorError::TokenizerMissing {
            reason: "whisper tokenizer returned empty initial prompt tokens".to_string(),
        });
    }

    let mut prompt_tokens = if let Some(token_ids) = request_options.prompt_token_ids.as_ref() {
        token_ids.clone()
    } else {
        let Some(prompt) = request_options.prompt.as_deref().map(str::trim) else {
            return Ok(prompt_init_tokens);
        };
        if prompt.is_empty() {
            return Ok(prompt_init_tokens);
        }
        tokenizer.encode_prompt_text(prompt).map_err(|error| {
            WhisperGgmlExecutorError::TokenizerMissing {
                reason: format!("could not encode whisper request prompt: {error}"),
            }
        })?
    };
    if prompt_tokens.is_empty() {
        return Ok(prompt_init_tokens);
    }
    let prev_token = if request_options.longform.is_some() {
        tokenizer.token_id_by_content("<|startofprev|>")
    } else {
        None
    };
    let max_prompt_tokens = execution
        .max_target_positions
        .saturating_sub(prompt_init_tokens.len())
        .saturating_sub(usize::from(prev_token.is_some()))
        .saturating_sub(1);
    if max_prompt_tokens == 0 {
        return Err(WhisperGgmlExecutorError::DecoderGraphUnsupported {
            reason: format!(
                "whisper initial prompt prefix len {} leaves no generation budget in max_target_positions {}",
                prompt_init_tokens.len(),
                execution.max_target_positions
            ),
        });
    }
    prompt_tokens = trim_prompt_token_tail(
        prompt_tokens,
        max_prompt_tokens,
        request_options.longform.is_some(),
        WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT,
    );
    let mut initial_prompt_tokens = Vec::with_capacity(
        prompt_init_tokens.len() + prompt_tokens.len() + usize::from(prev_token.is_some()),
    );
    if let Some(prev_token) = prev_token {
        initial_prompt_tokens.push(prev_token);
        initial_prompt_tokens.extend(prompt_tokens);
        initial_prompt_tokens.extend(prompt_init_tokens);
    } else {
        initial_prompt_tokens.extend(prompt_init_tokens);
        initial_prompt_tokens.extend(prompt_tokens);
    }
    Ok(initial_prompt_tokens)
}

fn build_whisper_carry_prompt_token_ids(
    tokenizer: &WhisperTokenizer,
    request_options: &GgmlAsrExecutionOptions,
    generated_tokens: &[u32],
) -> Result<Option<Vec<u32>>, WhisperGgmlExecutorError> {
    let Some(carry_tokens) = build_whisper_carry_prompt_seed_token_ids(tokenizer, request_options)?
    else {
        return Ok(None);
    };

    Ok(build_longform_token_history_carry(
        true,
        carry_tokens,
        generated_tokens,
        WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT,
    ))
}

fn build_whisper_carry_prompt_seed_token_ids(
    tokenizer: &WhisperTokenizer,
    request_options: &GgmlAsrExecutionOptions,
) -> Result<Option<Vec<u32>>, WhisperGgmlExecutorError> {
    if request_options.longform.is_none() {
        return Ok(None);
    }

    if let Some(token_ids) = request_options.prompt_token_ids.as_ref() {
        Ok(Some(token_ids.clone()))
    } else if let Some(prompt) = request_options.prompt.as_deref().map(str::trim) {
        if prompt.is_empty() {
            Ok(Some(Vec::new()))
        } else {
            tokenizer
                .encode_prompt_text(prompt)
                .map(Some)
                .map_err(|error| WhisperGgmlExecutorError::TokenizerMissing {
                    reason: format!("could not encode whisper carry prompt: {error}"),
                })
        }
    } else {
        Ok(Some(Vec::new()))
    }
}

fn take_whisper_encoder_persistent_static_session(
    runtime_path: &Path,
    backend: GgmlCpuGraphBackend,
) -> Option<WhisperEncoderPersistentStaticSession> {
    WHISPER_ENCODER_PERSISTENT_SESSION_BY_KEY.with(|sessions| {
        sessions
            .borrow_mut()
            .synced()
            .remove(&(canonical_runtime_cache_path(runtime_path), backend))
    })
}

fn store_whisper_encoder_persistent_static_session(
    runtime_path: &Path,
    session: WhisperEncoderPersistentStaticSession,
) {
    WHISPER_ENCODER_PERSISTENT_SESSION_BY_KEY.with(|sessions| {
        let key = (
            canonical_runtime_cache_path(runtime_path),
            session.graph_config.backend,
        );
        sessions.borrow_mut().synced().insert(key, session);
    });
}

fn encoder_persistent_session_matches_runtime(
    session: &WhisperEncoderPersistentStaticSession,
    execution: &WhisperGgmlExecutionMetadata,
    plan: &WhisperEncoderGraphPlan,
    graph_config: GgmlCpuGraphConfig,
) -> bool {
    session.graph_config == graph_config
        && session.encoder_layers == plan.layers.len()
        && session.encoder_hidden_size == execution.encoder_hidden_size
        && plan.output_hidden_size == execution.encoder_hidden_size
        && execution.encoder_attention_heads > 0
}

fn build_whisper_encoder_persistent_static_session(
    runtime_source: &GgmlRuntimeSource,
    execution: &WhisperGgmlExecutionMetadata,
    encoder_weights: &WhisperEncoderWeightBundle,
    plan: &WhisperEncoderGraphPlan,
    graph_config: GgmlCpuGraphConfig,
) -> Result<WhisperEncoderPersistentStaticSession, WhisperGgmlExecutorError> {
    let runner = GgmlCpuGraphRunner::new(graph_config).map_err(|error| {
        WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: format!("could not initialize ggml cpu graph runner: {error}"),
        }
    })?;
    let resident_weights = if whisper_encoder_resident_weights_enabled() {
        let resident_start = Instant::now();
        let encoder_tensor_index = build_encoder_tensor_index(encoder_weights);
        let cache = build_encoder_resident_weight_cache(
            &runner,
            graph_config.context_bytes,
            &encoder_tensor_index,
            encoder_weights,
            plan,
            Some(runtime_source.path()),
        )?;
        emit_encoder_resident_weight_trace(
            cache.upload_stats.count,
            cache.upload_stats.bytes,
            resident_start.elapsed().as_millis(),
        );
        Some(cache)
    } else {
        None
    };
    Ok(WhisperEncoderPersistentStaticSession {
        runner,
        resident_weights,
        graph_config,
        encoder_layers: plan.layers.len(),
        encoder_hidden_size: execution.encoder_hidden_size,
    })
}

fn take_or_build_whisper_encoder_persistent_static_session(
    runtime_source: &GgmlRuntimeSource,
    execution: &WhisperGgmlExecutionMetadata,
    encoder_weights: &WhisperEncoderWeightBundle,
    plan: &WhisperEncoderGraphPlan,
    graph_config: GgmlCpuGraphConfig,
) -> Result<WhisperEncoderPersistentStaticSession, WhisperGgmlExecutorError> {
    let runtime_path = runtime_source.path();
    if let Some(session) =
        take_whisper_encoder_persistent_static_session(runtime_path, graph_config.backend)
        && encoder_persistent_session_matches_runtime(&session, execution, plan, graph_config)
    {
        emit_encoder_resident_weight_cache_reuse_trace();
        return Ok(session);
    }
    build_whisper_encoder_persistent_static_session(
        runtime_source,
        execution,
        encoder_weights,
        plan,
        graph_config,
    )
}

fn decoder_persistent_session_matches_runtime(
    session: &WhisperDecoderPersistentStaticSession,
    execution: &WhisperGgmlExecutionMetadata,
    prelude_plan: &WhisperEncoderPreludePlan,
    initial_prompt_token_count: usize,
    graph_config: GgmlCpuGraphConfig,
) -> bool {
    session.graph_config == graph_config
        && session.plan.input_shape.token_count == initial_prompt_token_count
        && session.plan.input_shape.encoder_frames == prelude_plan.output_frames
        && session.plan.input_shape.hidden_size == execution.encoder_hidden_size
        && session.plan.layers.len() == execution.decoder_layers
        && session.plan.decoder_attention_heads == execution.decoder_attention_heads
        && session.plan.output_projection.vocab_size == execution.vocab_size
        && session.plan.input_shape.token_count <= execution.max_target_positions
}

fn build_whisper_decoder_persistent_static_session(
    runtime_source: &GgmlRuntimeSource,
    runtime: &WhisperPreparedRuntime,
    prelude_plan: &WhisperEncoderPreludePlan,
    initial_prompt_token_count: usize,
    trace: &WhisperGgmlTrace,
) -> Result<WhisperDecoderPersistentStaticSession, WhisperGgmlExecutorError> {
    let graph_config = whisper_decoder_graph_config();
    let plan = build_whisper_decoder_graph_plan(
        WhisperDecoderGraphMetadata {
            decoder_layers: runtime.execution.decoder_layers,
            decoder_hidden_size: runtime.execution.decoder_hidden_size,
            decoder_attention_heads: runtime.execution.decoder_attention_heads,
            vocab_size: runtime.execution.vocab_size,
            max_target_positions: runtime.execution.max_target_positions,
        },
        &runtime.decoder_weights.graph_binding,
        &runtime.decoder_weights.graph_materialization,
        WhisperDecoderGraphInputShape {
            token_count: initial_prompt_token_count,
            encoder_frames: prelude_plan.output_frames,
            hidden_size: runtime.execution.encoder_hidden_size,
        },
    )
    .map_err(map_decoder_graph_plan_error)
    .map_err(
        |error| WhisperGgmlExecutorError::DecoderGraphExecutionFailed {
            reason: error.to_string(),
        },
    )?;
    let mut runner = GgmlCpuGraphRunner::new(graph_config).map_err(|error| {
        WhisperGgmlExecutorError::DecoderGraphExecutionFailed {
            reason: format!("could not initialize static decoder cache runner: {error}"),
        }
    })?;
    let cache = trace
        .run_stage("decoder_persistent_cache_static", || {
            let mut persistent_weight_tensor_cache = WhisperDecoderExecutionTensorCache::default();
            WhisperDecoderPersistentWeightCache::build_static_stage(
                &mut runner,
                &plan,
                &runtime.decoder_weights.tensor_source,
                &mut persistent_weight_tensor_cache,
                runtime.execution.max_target_positions,
                Some(runtime_source.path()),
            )
        })
        .map_err(
            |error| WhisperGgmlExecutorError::DecoderGraphExecutionFailed {
                reason: error.to_string(),
            },
        )?;
    Ok(WhisperDecoderPersistentStaticSession {
        runner,
        cache,
        reuse: None,
        graph_config,
        plan,
    })
}

fn take_or_build_whisper_decoder_persistent_static_session(
    runtime_source: &GgmlRuntimeSource,
    runtime: &WhisperPreparedRuntime,
    prelude_plan: &WhisperEncoderPreludePlan,
    initial_prompt_token_count: usize,
    trace: &WhisperGgmlTrace,
) -> Result<WhisperDecoderPersistentStaticSession, WhisperGgmlExecutorError> {
    let runtime_path = runtime_source.path();
    let graph_config = whisper_decoder_graph_config();
    let key = (
        canonical_runtime_cache_path(runtime_path),
        graph_config.backend,
    );
    WHISPER_DECODER_PERSISTENT_SESSION_BY_KEY.with(|sessions| {
        let mut sessions = sessions.borrow_mut();
        let sessions = sessions.synced();
        if let Some(pool) = sessions.get_mut(&key)
            && let Some(index) = pool.iter().position(|session| {
                decoder_persistent_session_matches_runtime(
                    session,
                    &runtime.execution,
                    prelude_plan,
                    initial_prompt_token_count,
                    graph_config,
                )
            })
        {
            let session = pool.swap_remove(index);
            if pool.is_empty() {
                sessions.remove(&key);
            }
            return Ok(session);
        }
        build_whisper_decoder_persistent_static_session(
            runtime_source,
            runtime,
            prelude_plan,
            initial_prompt_token_count,
            trace,
        )
    })
}

fn store_whisper_decoder_persistent_static_session(
    runtime_path: &Path,
    session: WhisperDecoderPersistentStaticSession,
) {
    WHISPER_DECODER_PERSISTENT_SESSION_BY_KEY.with(|sessions| {
        let mut sessions = sessions.borrow_mut();
        let sessions = sessions.synced();
        let key = (
            canonical_runtime_cache_path(runtime_path),
            session.graph_config.backend,
        );
        let pool = sessions.entry(key).or_default();
        if let Some(index) = pool.iter().position(|existing| {
            existing.graph_config == session.graph_config
                && existing.plan.input_shape == session.plan.input_shape
                && existing.plan.layers.len() == session.plan.layers.len()
                && existing.plan.decoder_attention_heads == session.plan.decoder_attention_heads
                && existing.plan.output_projection.vocab_size
                    == session.plan.output_projection.vocab_size
        }) {
            pool.swap_remove(index);
        }
        pool.push(session);
        if pool.len() > WHISPER_DECODER_PERSISTENT_SESSION_POOL_CAPACITY {
            pool.remove(0);
        }
    });
}

fn execute_whisper_with_prepared_runtime(
    adapter: &GgmlFamilyAdapterDescriptor,
    runtime_source: &GgmlRuntimeSource,
    prepared_audio: &GgmlAsrPreparedAudio,
    runtime: &WhisperPreparedRuntime,
    request_options: &GgmlAsrExecutionOptions,
    mel_feature_input_provider: &dyn WhisperMelFeatureInputProvider,
    prelude_runner: &dyn WhisperEncoderPreludeRunner,
    encoder_graph_runner: &dyn WhisperEncoderGraphRunner,
    decoder_runner: &dyn WhisperDecoderLoopRunner,
    allow_persistent_session_reuse: bool,
    skip_serve_batch: bool,
) -> Result<WhisperExecutionOutput, WhisperGgmlExecutorError> {
    let trace = WhisperGgmlTrace::from_env();
    if adapter.adapter_id != WHISPER_GGML_ADAPTER_ID {
        return Err(WhisperGgmlExecutorError::AdapterMismatch {
            expected: WHISPER_GGML_ADAPTER_ID,
            found: adapter.adapter_id.to_string(),
        });
    }
    let initial_prompt_tokens = build_whisper_initial_prompt_tokens(
        &runtime.execution,
        &runtime.tokenizer,
        request_options,
        None,
    )?;
    let mel_input = std::thread::scope(|scope| {
        let mel_trace = trace.clone();
        let mel_execution = &runtime.execution;
        let mel_prepared_audio = prepared_audio;
        let mel_handle = scope.spawn(move || {
            mel_trace.run_stage("mel", || {
                prepare_mel_feature_input_seam(
                    mel_feature_input_provider,
                    mel_execution,
                    mel_prepared_audio,
                )
            })
        });
        mel_handle.join().map_err(|_| {
            WhisperGgmlExecutorError::MelFeatureInputPreparationFailed {
                reason: "mel feature preparation worker panicked".to_string(),
            }
        })?
    })?;
    let prelude_input_shape = infer_encoder_prelude_input_shape_from_mel_input(&mel_input)?;
    let prelude_plan = trace.run_stage("prelude_plan", || {
        build_whisper_encoder_prelude_plan(
            &runtime.tensor_binding.weights.bindings,
            prelude_input_shape,
            runtime.execution.encoder_hidden_size,
            runtime.execution.encoder_mels_count,
        )
        .map_err(map_prelude_plan_error)
    })?;
    let prelude_result = trace.run_stage("prelude_run", || {
        run_encoder_prelude_seam(
            runtime_source,
            &runtime.encoder_weights,
            &prelude_plan,
            &mel_input,
            prelude_runner,
        )
    })?;
    if std::env::var_os("OPENASR_WHISPER_GGML_TRACE_PRELUDE").is_some() {
        let WhisperEncoderPreludeSeamResult::GraphExecuted {
            output_frames,
            output_hidden_size,
            output_hidden_f32,
            ..
        } = &prelude_result;
        emit_tensor_probe_trace(
            "prelude_probe",
            "post_pos",
            output_hidden_f32,
            *output_frames,
            *output_hidden_size,
        );
    }
    let encoder_plan = trace.run_stage("encoder_plan", || {
        build_whisper_encoder_graph_plan(
            WhisperEncoderGraphMetadata {
                encoder_layers: runtime.execution.encoder_layers,
                encoder_hidden_size: runtime.execution.encoder_hidden_size,
            },
            &runtime.encoder_binding,
            &runtime.encoder_materialization,
            WhisperEncoderGraphInputShape {
                frames: prelude_plan.output_frames,
                hidden_size: prelude_plan.output_hidden_size,
            },
        )
        .map_err(map_encoder_graph_plan_error)
    })?;
    let prelude_hidden_output = match &prelude_result {
        WhisperEncoderPreludeSeamResult::GraphExecuted {
            output_hidden_f32, ..
        } => output_hidden_f32.as_slice(),
    };
    let serve_batch_config = WhisperServeBatchConfig::from_env().map_err(|error| {
        WhisperGgmlExecutorError::DecoderGraphExecutionFailed {
            reason: error.to_string(),
        }
    })?;
    let decoder_graph_config = whisper_decoder_graph_config();
    let can_use_serve_batch = !skip_serve_batch
        && whisper_can_use_serve_batch(
            decoder_graph_config,
            request_options,
            allow_persistent_session_reuse,
        );
    if let Some(serve_batch_config) = serve_batch_config.filter(|_| can_use_serve_batch) {
        let encoder_result = trace.run_stage("encoder_run", || {
            run_encoder_graph_seam(
                runtime_source,
                &runtime.execution,
                &runtime.encoder_weights,
                &encoder_plan,
                prelude_hidden_output,
                encoder_graph_runner,
            )
        })?;
        let WhisperEncoderGraphSeamResult::GraphExecuted {
            output_frames,
            output_hidden_size,
            output_hidden_f32,
            ..
        } = encoder_result;
        emit_encoder_hidden_probe_trace(&output_hidden_f32, output_frames, output_hidden_size);
        let eot_token_id = runtime
            .tokenizer
            .end_of_text_token_id()
            .unwrap_or(runtime.execution.eos_token_id);
        let max_generated_tokens = decode_generated_token_step_cap(
            runtime.execution.max_target_positions,
            initial_prompt_tokens.len(),
        )?;
        let decode_config = whisper_serve_batch_decode_config(
            initial_prompt_tokens,
            eot_token_id,
            runtime.execution.vocab_size,
            max_generated_tokens,
            &runtime.tokenizer,
            request_options.phrase_bias.as_ref(),
        )
        .map_err(
            |error| WhisperGgmlExecutorError::DecoderGraphExecutionFailed {
                reason: error.to_string(),
            },
        )?;
        return submit_whisper_serve_batch_job(
            serve_batch_config,
            WhisperServeBatchJob {
                runtime_cache_path: canonical_runtime_cache_path(runtime_source.path()),
                backend: decoder_graph_config.backend,
                uses_scheduler: decoder_graph_config.use_scheduler,
                execution: runtime.execution.clone(),
                decoder_weights: runtime.decoder_weights.clone(),
                tokenizer: runtime.tokenizer.clone(),
                encoder_frames: output_frames,
                encoder_hidden_size: output_hidden_size,
                encoder_hidden_f32: output_hidden_f32,
                decode_config,
                word_timestamps: request_options.word_timestamps,
                audio_duration_seconds: audio_duration_seconds(prepared_audio),
                carry_prompt_seed_token_ids: build_whisper_carry_prompt_seed_token_ids(
                    &runtime.tokenizer,
                    request_options,
                )?,
            },
        )
        .map_err(|error| match error.unavailable_retryable() {
            Some(retryable) => WhisperGgmlExecutorError::ServeBatchUnavailable {
                reason: error.to_string(),
                retryable,
            },
            None => WhisperGgmlExecutorError::DecoderGraphExecutionFailed {
                reason: error.to_string(),
            },
        });
    }
    let mut decoder_persistent_static = if allow_persistent_session_reuse {
        take_or_build_whisper_decoder_persistent_static_session(
            runtime_source,
            runtime,
            &prelude_plan,
            initial_prompt_tokens.len(),
            &trace,
        )?
    } else {
        build_whisper_decoder_persistent_static_session(
            runtime_source,
            runtime,
            &prelude_plan,
            initial_prompt_tokens.len(),
            &trace,
        )?
    };
    let longform_backend = whisper_decoder_graph_config().backend;
    let (encoder_result, decoder_persistent_cache_populated) =
        if whisper_parallel_encoder_and_decoder_static_enabled(
            longform_backend,
            allow_persistent_session_reuse,
        ) {
            std::thread::scope(|parallel_scope| {
                let execution_ref = &runtime.execution;
                let runtime_source_ref = runtime_source;
                let encoder_weights_ref = &runtime.encoder_weights;
                let encoder_plan_ref = &encoder_plan;
                let encoder_trace = trace.clone();
                let encoder_handle = parallel_scope.spawn(move || {
                    encoder_trace.run_stage("encoder_run", || {
                        run_encoder_graph_seam(
                            runtime_source_ref,
                            execution_ref,
                            encoder_weights_ref,
                            encoder_plan_ref,
                            prelude_hidden_output,
                            encoder_graph_runner,
                        )
                    })
                });
                let mut prepared_cross_attention_stage = if decoder_persistent_static
                    .cache
                    .supports_cross_attention_for_plan(&decoder_persistent_static.plan)
                {
                    let prepared_stage = trace
                        .run_stage("decoder_persistent_cache_prepare", || {
                            decoder_persistent_static
                                .cache
                                .prepare_cross_attention_stage(
                                    &mut decoder_persistent_static.runner,
                                    &decoder_persistent_static.plan,
                                )
                        })
                        .map_err(
                            |error| WhisperGgmlExecutorError::DecoderGraphExecutionFailed {
                                reason: error.to_string(),
                            },
                        )?;
                    Some(prepared_stage)
                } else {
                    None
                };
                let encoder_result = encoder_handle
                    .join()
                    .map_err(|_| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
                        reason: "encoder runner worker panicked".to_string(),
                    })?
                    .map_err(
                        |error| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
                            reason: error.to_string(),
                        },
                    )?;
                let mut decoder_persistent_cache_populated = false;
                if let Some(prepared_stage) = prepared_cross_attention_stage.take() {
                    let encoder_hidden_f32 = match &encoder_result {
                        WhisperEncoderGraphSeamResult::GraphExecuted {
                            output_hidden_f32, ..
                        } => output_hidden_f32.as_slice(),
                    };
                    trace
                        .run_stage("decoder_persistent_cache", || {
                            decoder_persistent_static
                                .cache
                                .populate_cross_attention_stage_with_prepared(
                                    prepared_stage,
                                    &decoder_persistent_static.plan,
                                    encoder_hidden_f32,
                                    WhisperDecoderHiddenStateLayout::SequenceHidden,
                                )
                        })
                        .map_err(
                            |error| WhisperGgmlExecutorError::DecoderGraphExecutionFailed {
                                reason: error.to_string(),
                            },
                        )?;
                    decoder_persistent_cache_populated = true;
                }
                Ok::<_, WhisperGgmlExecutorError>((
                    encoder_result,
                    decoder_persistent_cache_populated,
                ))
            })?
        } else {
            let encoder_result = trace.run_stage("encoder_run", || {
                run_encoder_graph_seam(
                    runtime_source,
                    &runtime.execution,
                    &runtime.encoder_weights,
                    &encoder_plan,
                    prelude_hidden_output,
                    encoder_graph_runner,
                )
            })?;
            let mut decoder_persistent_cache_populated = false;
            if decoder_persistent_static
                .cache
                .supports_cross_attention_for_plan(&decoder_persistent_static.plan)
            {
                let prepared_stage = trace
                    .run_stage("decoder_persistent_cache_prepare", || {
                        decoder_persistent_static
                            .cache
                            .prepare_cross_attention_stage(
                                &mut decoder_persistent_static.runner,
                                &decoder_persistent_static.plan,
                            )
                    })
                    .map_err(
                        |error| WhisperGgmlExecutorError::DecoderGraphExecutionFailed {
                            reason: error.to_string(),
                        },
                    )?;
                let encoder_hidden_f32 = match &encoder_result {
                    WhisperEncoderGraphSeamResult::GraphExecuted {
                        output_hidden_f32, ..
                    } => output_hidden_f32.as_slice(),
                };
                trace
                    .run_stage("decoder_persistent_cache", || {
                        decoder_persistent_static
                            .cache
                            .populate_cross_attention_stage_with_prepared(
                                prepared_stage,
                                &decoder_persistent_static.plan,
                                encoder_hidden_f32,
                                WhisperDecoderHiddenStateLayout::SequenceHidden,
                            )
                    })
                    .map_err(
                        |error| WhisperGgmlExecutorError::DecoderGraphExecutionFailed {
                            reason: error.to_string(),
                        },
                    )?;
                decoder_persistent_cache_populated = true;
            }
            (encoder_result, decoder_persistent_cache_populated)
        };
    let decode_result = run_whisper_decode_loop(
        runtime_source,
        &runtime.execution,
        &mut decoder_persistent_static,
        &runtime.decoder_weights,
        (&runtime.tokenizer, initial_prompt_tokens.as_slice()),
        request_options,
        &prelude_result,
        &encoder_result,
        audio_duration_seconds(prepared_audio),
        decoder_persistent_cache_populated,
        decoder_runner,
        &trace,
    );
    if allow_persistent_session_reuse {
        store_whisper_decoder_persistent_static_session(
            runtime_source.path(),
            decoder_persistent_static,
        );
    }
    decode_result
}

#[cfg(test)]
fn execute_whisper_ggml_non_streaming_cpu(
    adapter: &GgmlFamilyAdapterDescriptor,
    runtime_source: &GgmlRuntimeSource,
    metadata: &GgufMetadata,
    tensor_index: &GgufTensorIndex,
    prepared_audio: &GgmlAsrPreparedAudio,
    mel_feature_input_provider: &dyn WhisperMelFeatureInputProvider,
    prelude_runner: &dyn WhisperEncoderPreludeRunner,
    encoder_graph_runner: &dyn WhisperEncoderGraphRunner,
    decoder_runner: &dyn WhisperDecoderLoopRunner,
    tokenizer_provider: &dyn WhisperTokenizerProvider,
) -> Result<String, WhisperGgmlExecutorError> {
    let runtime =
        build_whisper_prepared_runtime(runtime_source, metadata, tensor_index, tokenizer_provider)?;
    execute_whisper_with_prepared_runtime(
        adapter,
        runtime_source,
        prepared_audio,
        &runtime,
        &GgmlAsrExecutionOptions::default(),
        mel_feature_input_provider,
        prelude_runner,
        encoder_graph_runner,
        decoder_runner,
        false,
        false,
    )
    .map(|output| output.text)
}

fn prepare_mel_feature_input_seam(
    provider: &dyn WhisperMelFeatureInputProvider,
    execution: &WhisperGgmlExecutionMetadata,
    prepared_audio: &GgmlAsrPreparedAudio,
) -> Result<WhisperMelFeatureInput, WhisperGgmlExecutorError> {
    provider.prepare_mel_feature_input(execution, prepared_audio)
}

fn infer_encoder_prelude_input_shape_from_mel_input(
    mel_input: &WhisperMelFeatureInput,
) -> Result<WhisperEncoderPreludeInputShape, WhisperGgmlExecutorError> {
    if mel_input.shape.mel_bins == 0 || mel_input.shape.mel_frames == 0 {
        return Err(WhisperGgmlExecutorError::MelFeatureInputPreparationFailed {
            reason: format!(
                "mel input shape from '{}' must be > 0, got ({}, {})",
                mel_input.source_label, mel_input.shape.mel_frames, mel_input.shape.mel_bins
            ),
        });
    }
    let expected_values = mel_input.shape.mel_bins * mel_input.shape.mel_frames;
    if mel_input.values_f32.len() != expected_values {
        return Err(WhisperGgmlExecutorError::MelFeatureInputPreparationFailed {
            reason: format!(
                "mel input value count from '{}' is {}, expected {}",
                mel_input.source_label,
                mel_input.values_f32.len(),
                expected_values
            ),
        });
    }
    Ok(WhisperEncoderPreludeInputShape {
        mel_bins: mel_input.shape.mel_bins,
        mel_frames: mel_input.shape.mel_frames,
    })
}

#[cfg(test)]
fn load_whisper_tensor_index(
    runtime_source: &GgmlRuntimeSource,
) -> Result<GgufTensorIndex, WhisperGgmlExecutorError> {
    read_gguf_tensor_index_from_runtime_source(runtime_source)
        .map_err(|source| WhisperGgmlExecutorError::TensorIndexRead { source })
}

fn bind_whisper_required_tensors(
    tensor_index: &GgufTensorIndex,
    execution: &WhisperGgmlExecutionMetadata,
) -> Result<WhisperGgmlTensorBinding, WhisperGgmlExecutorError> {
    let bindings = bind_whisper_gguf_tensors(
        &WhisperGgufTensorBindingContext {
            n_audio_layer: execution.encoder_layers,
            n_audio_state: execution.encoder_hidden_size,
            n_audio_head: execution.encoder_attention_heads,
            n_mels: execution.encoder_mels_count,
            n_audio_ctx: execution.encoder_context_length,
            n_text_layer: execution.decoder_layers,
            n_text_state: execution.decoder_hidden_size,
            n_text_head: execution.decoder_attention_heads,
            n_text_ctx: execution.max_target_positions,
            n_vocab: execution.vocab_size,
        },
        tensor_index,
    )
    .map_err(map_tensor_binding_error)?;
    let weights = WhisperGgmlWeightIndex {
        tensor_index: Arc::new(tensor_index.clone()),
        bindings,
    };
    Ok(WhisperGgmlTensorBinding { weights })
}

#[cfg(test)]
fn materialize_whisper_encoder_weights(
    tensor_binding: &WhisperGgmlTensorBinding,
) -> Result<WhisperEncoderWeightBundle, WhisperGgmlExecutorError> {
    let reader = GgufTensorDataReader::from_tensor_index_shared(Arc::clone(
        &tensor_binding.weights.tensor_index,
    ))
    .map_err(map_tensor_materialization_error)?;
    materialize_whisper_encoder_weights_from_reader(tensor_binding, &reader)
}

fn materialize_whisper_encoder_weights_from_reader(
    tensor_binding: &WhisperGgmlTensorBinding,
    reader: &GgufTensorDataReader,
) -> Result<WhisperEncoderWeightBundle, WhisperGgmlExecutorError> {
    materialize_whisper_encoder_weight_bundle(&tensor_binding.weights.bindings, reader)
        .map_err(map_encoder_weight_materialization_error)
}

fn materialize_whisper_encoder_tensor_seam(
    encoder_weights: &WhisperEncoderWeightBundle,
) -> WhisperEncoderTensorMaterializationSeam {
    WhisperEncoderTensorMaterializationSeam {
        source_label: "gguf-tensor-data-reader-v0",
        materialized_tensor_count: encoder_weights.materialized_tensor_count(),
    }
}

fn run_encoder_prelude_seam(
    runtime_source: &GgmlRuntimeSource,
    encoder_weights: &WhisperEncoderWeightBundle,
    prelude_plan: &WhisperEncoderPreludePlan,
    mel_input: &WhisperMelFeatureInput,
    prelude_runner: &dyn WhisperEncoderPreludeRunner,
) -> Result<WhisperEncoderPreludeSeamResult, WhisperGgmlExecutorError> {
    prelude_runner.run_encoder_prelude(runtime_source, encoder_weights, prelude_plan, mel_input)
}

fn run_encoder_graph_seam(
    runtime_source: &GgmlRuntimeSource,
    execution: &WhisperGgmlExecutionMetadata,
    encoder_weights: &WhisperEncoderWeightBundle,
    encoder_plan: &WhisperEncoderGraphPlan,
    encoder_hidden_input_f32: &[f32],
    encoder_graph_runner: &dyn WhisperEncoderGraphRunner,
) -> Result<WhisperEncoderGraphSeamResult, WhisperGgmlExecutorError> {
    encoder_graph_runner.run_encoder_graph(
        runtime_source,
        execution,
        encoder_weights,
        encoder_plan,
        encoder_hidden_input_f32,
    )
}

fn build_encoder_graph_binding_seam(
    encoder_weights: &WhisperEncoderWeightBundle,
    execution: &WhisperGgmlExecutionMetadata,
) -> Result<WhisperEncoderTensorBindingSeam, WhisperGgmlExecutorError> {
    if encoder_weights.layers.len() != execution.encoder_layers {
        return Err(WhisperGgmlExecutorError::EncoderGraphBindingUnsupported {
            reason: format!(
                "encoder layer count mismatch after materialization (metadata={}, materialized={})",
                execution.encoder_layers,
                encoder_weights.layers.len()
            ),
        });
    }
    let layers = encoder_weights
        .layers
        .iter()
        .map(|layer| WhisperEncoderLayerTensorBinding {
            self_attn_norm_weight: Some(materialized_tensor_ref(
                &layer.self_attn_layer_norm_weight,
            )),
            self_attn_norm_bias: Some(materialized_tensor_ref(&layer.self_attn_layer_norm_bias)),
            self_attn_q_weight: Some(materialized_tensor_ref(&layer.self_attn_q_weight)),
            self_attn_k_weight: Some(materialized_tensor_ref(&layer.self_attn_k_weight)),
            self_attn_v_weight: Some(materialized_tensor_ref(&layer.self_attn_v_weight)),
            self_attn_out_weight: Some(materialized_tensor_ref(&layer.self_attn_out_weight)),
            mlp_norm_weight: Some(materialized_tensor_ref(&layer.mlp_norm_weight)),
            mlp_norm_bias: Some(materialized_tensor_ref(&layer.mlp_norm_bias)),
            mlp_fc1_weight: Some(materialized_tensor_ref(&layer.fc1_weight)),
            mlp_fc2_weight: Some(materialized_tensor_ref(&layer.fc2_weight)),
        })
        .collect::<Vec<_>>();

    Ok(WhisperEncoderTensorBindingSeam {
        layers,
        final_norm_weight: Some(materialized_tensor_ref(&encoder_weights.final_norm.weight)),
        final_norm_bias: Some(materialized_tensor_ref(&encoder_weights.final_norm.bias)),
    })
}

fn materialized_tensor_ref(tensor: &WhisperMaterializedTensor) -> WhisperEncoderGraphTensorRef {
    WhisperEncoderGraphTensorRef {
        tensor_name: tensor.tensor_name.clone(),
        tensor_num_elements: tensor.num_elements,
        dims: tensor.dims.clone(),
        runtime_linear_weight_layout: encoder_prepared_linear_weight_layout(&tensor.tensor_name),
    }
}

fn encoder_prepared_linear_weight_layout(
    tensor_name: &str,
) -> Option<WhisperEncoderLinearWeightLayout> {
    let is_encoder_linear_weight = tensor_name.starts_with("model.encoder.layers.")
        && tensor_name.ends_with(".weight")
        && !tensor_name.ends_with("layer_norm.weight");
    is_encoder_linear_weight.then_some(WhisperEncoderLinearWeightLayout::InputOutput)
}

pub(super) fn build_decoder_weight_seam(
    tensor_reader: &GgufTensorDataReader,
    tensor_bindings: &WhisperGgufTensorBindings,
) -> Result<WhisperDecoderWeightSeam, WhisperGgmlExecutorError> {
    let mut bundle = materialize_whisper_decoder_weight_bundle(tensor_bindings, tensor_reader)
        .map_err(map_decoder_weight_materialization_error)?;
    prepare_decoder_runtime_weight_payloads(&mut bundle)?;
    let decoder = tensor_bindings.decoder();
    if decoder.layers.is_empty() {
        return Err(WhisperGgmlExecutorError::DecoderWeightsMissing {
            reason: "decoder.layers is empty after GGUF binding".to_string(),
        });
    }
    let materialized_tensor_count = bundle.materialized_tensor_count();
    if materialized_tensor_count == 0 {
        return Err(WhisperGgmlExecutorError::DecoderWeightsMissing {
            reason: "decoder typed materialization produced zero tensors".to_string(),
        });
    }

    let graph_binding = build_decoder_graph_binding_seam(decoder)?;
    let tensor_source = build_decoder_materialized_tensor_source(bundle)?;
    Ok(WhisperDecoderWeightSeam {
        graph_binding,
        graph_materialization: WhisperDecoderTensorMaterializationSeam {
            source_label: "gguf-decoder-weights-v0",
            materialized_tensor_count,
        },
        tensor_source,
    })
}

fn build_decoder_materialized_tensor_source(
    bundle: WhisperDecoderWeightBundle,
) -> Result<WhisperDecoderMaterializedTensorSource, WhisperGgmlExecutorError> {
    let tensor_count = bundle.materialized_tensor_count();
    let mut tensors_f32_by_name = HashMap::with_capacity(tensor_count);
    let mut tensors_f16_bits_by_name = HashMap::with_capacity(tensor_count);
    let mut tensors_quantized_by_name = HashMap::with_capacity(tensor_count);
    insert_decoder_tensor_owned(
        &mut tensors_f32_by_name,
        &mut tensors_f16_bits_by_name,
        &mut tensors_quantized_by_name,
        bundle.token_embedding,
    )?;
    insert_decoder_tensor_owned(
        &mut tensors_f32_by_name,
        &mut tensors_f16_bits_by_name,
        &mut tensors_quantized_by_name,
        bundle.positional_embedding,
    )?;
    insert_decoder_tensor_owned(
        &mut tensors_f32_by_name,
        &mut tensors_f16_bits_by_name,
        &mut tensors_quantized_by_name,
        bundle.final_layer_norm_weight,
    )?;
    insert_decoder_tensor_owned(
        &mut tensors_f32_by_name,
        &mut tensors_f16_bits_by_name,
        &mut tensors_quantized_by_name,
        bundle.final_layer_norm_bias,
    )?;
    if let Some(output_projection_weight) = bundle.output_projection_weight {
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            output_projection_weight,
        )?;
    }
    for layer in bundle.layers {
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.self_attn_layer_norm_weight,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.self_attn_layer_norm_bias,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.self_attn_q_weight,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.self_attn_q_bias,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.self_attn_k_weight,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.self_attn_v_weight,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.self_attn_v_bias,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.self_attn_out_weight,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.self_attn_out_bias,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.cross_attn_layer_norm_weight,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.cross_attn_layer_norm_bias,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.cross_attn_q_weight,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.cross_attn_q_bias,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.cross_attn_k_weight,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.cross_attn_v_weight,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.cross_attn_v_bias,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.cross_attn_out_weight,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.cross_attn_out_bias,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.mlp_norm_weight,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.mlp_norm_bias,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.fc1_weight,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.fc1_bias,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.fc2_weight,
        )?;
        insert_decoder_tensor_owned(
            &mut tensors_f32_by_name,
            &mut tensors_f16_bits_by_name,
            &mut tensors_quantized_by_name,
            layer.fc2_bias,
        )?;
    }
    Ok(WhisperDecoderMaterializedTensorSource {
        tensors_f32_by_name,
        tensors_f16_bits_by_name,
        tensors_quantized_by_name,
    })
}

fn insert_decoder_tensor_owned(
    target: &mut HashMap<String, Arc<[f32]>>,
    f16_target: &mut HashMap<String, Arc<[u16]>>,
    quantized_target: &mut HashMap<String, (i32, Arc<[u8]>)>,
    tensor: WhisperMaterializedTensor,
) -> Result<(), WhisperGgmlExecutorError> {
    match tensor.payload {
        WhisperMaterializedTensorPayload::F32(values) => {
            if values.len() != tensor.num_elements {
                return Err(WhisperGgmlExecutorError::DecoderWeightsMissing {
                    reason: format!(
                        "decoder tensor '{}' materialized {} f32 values but metadata expects {}",
                        tensor.tensor_name,
                        values.len(),
                        tensor.num_elements
                    ),
                });
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(WhisperGgmlExecutorError::DecoderWeightsMissing {
                    reason: format!(
                        "decoder tensor '{}' materialized non-finite values",
                        tensor.tensor_name
                    ),
                });
            }
            target.insert(
                tensor.tensor_name,
                Arc::<[f32]>::from(values.into_boxed_slice()),
            );
        }
        WhisperMaterializedTensorPayload::F16Bits(values) => {
            if values.len() != tensor.num_elements {
                return Err(WhisperGgmlExecutorError::DecoderWeightsMissing {
                    reason: format!(
                        "decoder tensor '{}' materialized {} f16 values but metadata expects {}",
                        tensor.tensor_name,
                        values.len(),
                        tensor.num_elements
                    ),
                });
            }
            f16_target.insert(
                tensor.tensor_name,
                Arc::<[u16]>::from(values.into_boxed_slice()),
            );
        }
        WhisperMaterializedTensorPayload::Quantized { ggml_type, bytes } => {
            if bytes.is_empty() {
                return Err(WhisperGgmlExecutorError::DecoderWeightsMissing {
                    reason: format!(
                        "decoder tensor '{}' materialized quantized type {} with empty bytes",
                        tensor.tensor_name, ggml_type
                    ),
                });
            }
            quantized_target.insert(
                tensor.tensor_name,
                (ggml_type, Arc::<[u8]>::from(bytes.into_boxed_slice())),
            );
        }
    }
    Ok(())
}

fn build_decoder_graph_binding_seam(
    decoder: &WhisperGgufDecoderTensorBindings,
) -> Result<WhisperDecoderTensorBindingSeam, WhisperGgmlExecutorError> {
    let layers = decoder
        .layers
        .iter()
        .map(build_decoder_layer_binding)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(WhisperDecoderTensorBindingSeam {
        token_embedding_weight: Some(decoder_tensor_ref(&decoder.token_embedding)?),
        position_embedding_weight: Some(decoder_tensor_ref(&decoder.positional_embedding)?),
        final_norm_weight: Some(decoder_tensor_ref(&decoder.final_layer_norm_weight)?),
        final_norm_bias: Some(decoder_tensor_ref(&decoder.final_layer_norm_bias)?),
        output_projection_weight: Some(decoder_tensor_ref(&decoder.output_projection_weight)?),
        output_projection_bias: None,
        layers,
    })
}

fn build_decoder_layer_binding(
    layer: &WhisperGgufDecoderLayerTensorBindings,
) -> Result<WhisperDecoderLayerTensorBinding, WhisperGgmlExecutorError> {
    Ok(WhisperDecoderLayerTensorBinding {
        self_attn_norm_weight: Some(decoder_tensor_ref(&layer.self_attn_layer_norm_weight)?),
        self_attn_norm_bias: Some(decoder_tensor_ref(&layer.self_attn_layer_norm_bias)?),
        self_attn_q_weight: Some(decoder_tensor_ref(&layer.self_attn_q_weight)?),
        self_attn_q_bias: Some(decoder_tensor_ref(&layer.self_attn_q_bias)?),
        self_attn_k_weight: Some(decoder_tensor_ref(&layer.self_attn_k_weight)?),
        self_attn_v_weight: Some(decoder_tensor_ref(&layer.self_attn_v_weight)?),
        self_attn_v_bias: Some(decoder_tensor_ref(&layer.self_attn_v_bias)?),
        self_attn_out_weight: Some(decoder_tensor_ref(&layer.self_attn_out_weight)?),
        self_attn_out_bias: Some(decoder_tensor_ref(&layer.self_attn_out_bias)?),
        cross_attn_norm_weight: Some(decoder_tensor_ref(&layer.cross_attn_layer_norm_weight)?),
        cross_attn_norm_bias: Some(decoder_tensor_ref(&layer.cross_attn_layer_norm_bias)?),
        cross_attn_q_weight: Some(decoder_tensor_ref(&layer.cross_attn_q_weight)?),
        cross_attn_q_bias: Some(decoder_tensor_ref(&layer.cross_attn_q_bias)?),
        cross_attn_k_weight: Some(decoder_tensor_ref(&layer.cross_attn_k_weight)?),
        cross_attn_v_weight: Some(decoder_tensor_ref(&layer.cross_attn_v_weight)?),
        cross_attn_v_bias: Some(decoder_tensor_ref(&layer.cross_attn_v_bias)?),
        cross_attn_out_weight: Some(decoder_tensor_ref(&layer.cross_attn_out_weight)?),
        cross_attn_out_bias: Some(decoder_tensor_ref(&layer.cross_attn_out_bias)?),
        mlp_norm_weight: Some(decoder_tensor_ref(&layer.mlp_norm_weight)?),
        mlp_norm_bias: Some(decoder_tensor_ref(&layer.mlp_norm_bias)?),
        mlp_fc1_weight: Some(decoder_tensor_ref(&layer.fc1_weight)?),
        mlp_fc1_bias: Some(decoder_tensor_ref(&layer.fc1_bias)?),
        mlp_fc2_weight: Some(decoder_tensor_ref(&layer.fc2_weight)?),
        mlp_fc2_bias: Some(decoder_tensor_ref(&layer.fc2_bias)?),
    })
}

fn decoder_tensor_ref(
    tensor: &WhisperGgufTensorBinding,
) -> Result<WhisperDecoderGraphTensorRef, WhisperGgmlExecutorError> {
    let tensor_num_elements = tensor.metadata.num_elements().ok_or_else(|| {
        WhisperGgmlExecutorError::DecoderWeightsMissing {
            reason: format!(
                "decoder tensor '{}' has overflowing element count for dims {:?}",
                tensor.resolved_name, tensor.metadata.dims
            ),
        }
    })?;
    let tensor_num_elements = usize::try_from(tensor_num_elements).map_err(|_| {
        WhisperGgmlExecutorError::DecoderWeightsMissing {
            reason: format!(
                "decoder tensor '{}' element count {} does not fit usize",
                tensor.resolved_name, tensor_num_elements
            ),
        }
    })?;
    Ok(WhisperDecoderGraphTensorRef {
        tensor_name: tensor.resolved_name.clone(),
        tensor_num_elements,
        dims: tensor.metadata.dims.clone(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WhisperDecoderStepPlanCacheBase {
    metadata: WhisperDecoderGraphMetadata,
    encoder_frames: usize,
    encoder_hidden_size: usize,
}

impl WhisperDecoderStepPlanCacheBase {
    fn input_shape(&self, token_count: usize) -> WhisperDecoderGraphInputShape {
        WhisperDecoderGraphInputShape {
            token_count,
            encoder_frames: self.encoder_frames,
            hidden_size: self.encoder_hidden_size,
        }
    }
}

fn emit_encoder_hidden_probe_trace(encoder_hidden: &[f32], frames: usize, hidden: usize) {
    if std::env::var_os("OPENASR_WHISPER_GGML_TRACE_ENCODER").is_none() {
        return;
    }
    emit_tensor_probe_trace("encoder_probe", "hidden", encoder_hidden, frames, hidden);
}

fn emit_tensor_probe_trace(
    stage: &str,
    event: &str,
    sequence_hidden: &[f32],
    frames: usize,
    hidden: usize,
) {
    let sequence_items = sequence_hidden
        .iter()
        .take(12)
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    let hidden_by_seq =
        transpose_sequence_hidden_to_hidden_sequence(sequence_hidden, frames, hidden);
    let hidden_items = hidden_by_seq
        .iter()
        .take(12)
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    let (min, max, sum_abs) = hidden_by_seq.iter().copied().fold(
        (f32::INFINITY, f32::NEG_INFINITY, 0.0_f32),
        |(min, max, sum_abs), value| (min.min(value), max.max(value), sum_abs + value.abs()),
    );
    let mean_abs = if hidden_by_seq.is_empty() {
        0.0
    } else {
        sum_abs / hidden_by_seq.len() as f32
    };
    eprintln!(
        "openasr_whisper_ggml_trace stage={stage} event={event} status=ok frames={frames} hidden={hidden} first_sequence_major={sequence_items} first_hidden_major={hidden_items} min={min:.6} max={max:.6} mean_abs={mean_abs:.6}"
    );
}

fn decode_generated_token_step_cap(
    max_target_positions: usize,
    initial_prompt_len: usize,
) -> Result<usize, WhisperGgmlExecutorError> {
    context_window_budget(max_target_positions, initial_prompt_len)
        .ok_or_else(|| WhisperGgmlExecutorError::DecoderGraphUnsupported {
            reason: format!(
                "decoder initial prompt len {initial_prompt_len} exhausts max_target_positions {max_target_positions}"
            ),
        })
        .map(|budget| budget.min(WHISPER_DEFAULT_DECODE_MAX_GENERATED_TOKENS_CAP))
}

fn audio_duration_seconds(prepared_audio: &GgmlAsrPreparedAudio) -> f32 {
    prepared_audio.samples_f32.len() as f32 / prepared_audio.sample_rate_hz.max(1) as f32
}

/// How a whisper decode derives word timestamps for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhisperWordTimestampMode {
    /// No word timestamps requested.
    Off,
    /// User-requested word timestamps: collect per-token cross-attention
    /// during decode (higher fidelity, but switches the decode path — cross
    /// flash attention off, cross-attention collection on — so the transcript
    /// can differ from a plain run via FP accumulation differences).
    CrossAttention,
    /// Word timestamps forced on solely as diarization anchors: keep the
    /// decode path byte-identical to a non-diarized run and derive word
    /// anchors post hoc from the generated tokens (the same path the whisper
    /// serve-batch decode always uses).
    PostHocAnchors,
}

fn whisper_word_timestamp_mode(
    request_options: &GgmlAsrExecutionOptions,
) -> WhisperWordTimestampMode {
    if !request_options.word_timestamps {
        WhisperWordTimestampMode::Off
    } else if request_options.word_timestamps_forced_for_diarization {
        WhisperWordTimestampMode::PostHocAnchors
    } else {
        WhisperWordTimestampMode::CrossAttention
    }
}

/// Decoder-graph `(use_cross_flash_attention, collect_cross_attention)` flags
/// for a request. Only user-requested word timestamps (`CrossAttention`) may
/// alter the decode path; diarization-forced anchors must leave both flags
/// exactly as a request without word timestamps would.
fn whisper_decoder_cross_attention_flags(
    cross_flash_attention_enabled: bool,
    request_options: &GgmlAsrExecutionOptions,
) -> (bool, bool) {
    let collect_cross_attention =
        whisper_word_timestamp_mode(request_options) == WhisperWordTimestampMode::CrossAttention;
    (
        cross_flash_attention_enabled && !collect_cross_attention,
        collect_cross_attention,
    )
}

fn whisper_cross_attention_word_timestamps(
    tokenizer: &WhisperTokenizer,
    token_alignments: &[WhisperGeneratedTokenAlignment],
    generated_probabilities: &[f32],
    audio_duration_seconds: f32,
) -> Result<Vec<crate::WordTimestamp>, WhisperGgmlExecutorError> {
    if token_alignments.is_empty() {
        return Ok(Vec::new());
    }
    // Alignments are recorded one per generated token; a step that yielded no
    // cross-attention probs breaks that parity, in which case confidence is
    // withheld rather than misattributed by position.
    let probabilities_aligned = generated_probabilities.len() == token_alignments.len();
    let duration = audio_duration_seconds.max(0.0);
    let token_times = token_alignments
        .iter()
        .enumerate()
        .map(|(index, alignment)| {
            Ok(Seq2SeqTokenTime {
                token_id: alignment.token_id,
                center_seconds: cross_attention_center_seconds(&alignment.frame_probs, duration)?,
                probability: probabilities_aligned.then(|| generated_probabilities[index]),
            })
        })
        .collect::<Result<Vec<_>, WhisperGgmlExecutorError>>()?;
    seq2seq_word_timestamps_from_token_times(
        &token_times,
        0.0,
        duration,
        BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        &|token_ids| tokenizer.decode_text_token_ids(token_ids),
    )
    .map_err(
        |error| WhisperGgmlExecutorError::DecoderInvalidTokenDecode {
            reason: format!("whisper cross-attention word timestamp token decode failed: {error}"),
        },
    )
}

fn cross_attention_center_seconds(
    frame_probs: &[f32],
    audio_duration_seconds: f32,
) -> Result<f32, WhisperGgmlExecutorError> {
    if frame_probs.is_empty() || audio_duration_seconds <= 0.0 {
        return Ok(0.0);
    }
    let mut weighted_frame = 0.0_f32;
    let mut total = 0.0_f32;
    for (frame_index, prob) in frame_probs.iter().copied().enumerate() {
        if !prob.is_finite() {
            return Err(WhisperGgmlExecutorError::DecoderGraphExecutionFailed {
                reason:
                    "whisper cross-attention word timestamp probabilities contain non-finite values"
                        .to_string(),
            });
        }
        let prob = prob.max(0.0);
        weighted_frame += (frame_index as f32 + 0.5) * prob;
        total += prob;
    }
    if total <= 0.0 || !total.is_finite() {
        return Ok(0.0);
    }
    let center_frame = weighted_frame / total;
    Ok(
        (center_frame / frame_probs.len() as f32 * audio_duration_seconds)
            .clamp(0.0, audio_duration_seconds),
    )
}

struct WhisperGreedyDecodeStepRunnerAdapter<'a> {
    runtime_source: &'a GgmlRuntimeSource,
    execution: &'a WhisperGgmlExecutionMetadata,
    decoder_weights: &'a WhisperDecoderWeightSeam,
    decoder_runner: &'a dyn WhisperDecoderLoopRunner,
    trace: &'a WhisperGgmlTrace,
    decode_loop_start: Instant,
    decode_steps_completed: usize,
    plan_cache_base: WhisperDecoderStepPlanCacheBase,
    decoder_graph_config: WhisperDecoderGraphExecutionConfig,
    decoder_persistent_weights: &'a WhisperDecoderPersistentWeightCache,
    decoder_self_kv_state: WhisperDecoderSelfKvCacheState,
    decoder_reuse: &'a mut Option<Seq2SeqReusableDecodeGraph>,
    decoder_graph_runner: &'a mut GgmlCpuGraphRunner,
    decoder_graph_input: WhisperDecoderGraphExecutionInput,
    decoder_step_input: WhisperDecoderStepSeamInput,
    decoder_tensor_cache: WhisperDecoderExecutionTensorCache,
    plan_by_token_count: BTreeMap<usize, Arc<WhisperDecoderGraphPlan>>,
    token_alignments: Vec<WhisperGeneratedTokenAlignment>,
}

impl WhisperGreedyDecodeStepRunnerAdapter<'_> {
    fn plan_for_token_count(
        &mut self,
        token_count: usize,
    ) -> Result<WhisperDecoderStepPlanLookup, WhisperGreedyDecodeError> {
        // Without decoder KV cache, token_count grows each step, so most plans are single-use.
        // This cache still avoids rebuild churn for repeated prefixes (e.g., retries/replays).
        if let Some(plan) = self.plan_by_token_count.get(&token_count) {
            return Ok(WhisperDecoderStepPlanLookup {
                plan: Arc::clone(plan),
                plan_cache_status: WhisperDecoderStepPlanCacheStatus::Hit,
                plan_build_ms: 0,
            });
        }
        let plan_build_start = Instant::now();
        let plan = build_whisper_decoder_graph_plan(
            self.plan_cache_base.metadata,
            &self.decoder_weights.graph_binding,
            &self.decoder_weights.graph_materialization,
            self.plan_cache_base.input_shape(token_count),
        )
        .map_err(map_decoder_graph_plan_error)
        .map_err(|error| WhisperGreedyDecodeError::DecoderStepFailed {
            reason: error.to_string(),
        })?;
        let plan = Arc::new(plan);
        let plan_build_ms = plan_build_start.elapsed().as_millis();
        self.plan_by_token_count
            .insert(token_count, Arc::clone(&plan));
        Ok(WhisperDecoderStepPlanLookup {
            plan,
            plan_cache_status: WhisperDecoderStepPlanCacheStatus::Miss,
            plan_build_ms,
        })
    }
}

impl Seq2SeqGreedyDecodeStepExecutor for WhisperGreedyDecodeStepRunnerAdapter<'_> {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
        let step_start = Instant::now();
        let full_token_count = input
            .initial_prompt_tokens
            .len()
            .checked_add(input.generated_tokens.len())
            .ok_or_else(|| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: "decoder token_count overflows usize".to_string(),
            })?;
        self.decoder_graph_input.decoder_prefix_tokens.clear();
        let position_offset = if input.generated_tokens.is_empty() {
            self.decoder_graph_input
                .decoder_prefix_tokens
                .extend_from_slice(input.initial_prompt_tokens);
            0
        } else {
            let token = *input.generated_tokens.last().ok_or_else(|| {
                Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: "decoder generated token list is unexpectedly empty".to_string(),
                }
            })?;
            self.decoder_graph_input.decoder_prefix_tokens.push(token);
            full_token_count.checked_sub(1).ok_or_else(|| {
                Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: "decoder position offset underflows".to_string(),
                }
            })?
        };
        let graph_token_count = self.decoder_graph_input.decoder_prefix_tokens.len();
        self.decoder_step_input.step_index = input.step_index;
        self.decoder_step_input.position_offset = position_offset;
        if input.step_index == 0
            || input
                .step_index
                .is_multiple_of(WHISPER_GGML_TRACE_DECODE_STEP_INTERVAL)
        {
            self.trace.emit_decode_step_progress(
                "step_begin",
                input.step_index,
                full_token_count,
                self.decode_steps_completed,
                self.decode_loop_start,
            );
        }
        let plan_lookup_start = Instant::now();
        let plan_lookup = match self.plan_for_token_count(graph_token_count) {
            Ok(plan_lookup) => plan_lookup,
            Err(error) => {
                self.trace.emit_decode_step_metrics(
                    "err",
                    input.step_index,
                    full_token_count,
                    WhisperDecoderStepPlanCacheStatus::Miss.as_str(),
                    false,
                    plan_lookup_start.elapsed().as_millis(),
                    0,
                    0,
                    step_start.elapsed().as_millis(),
                    self.decode_loop_start,
                );
                return Err(Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                });
            }
        };
        if self.decoder_self_kv_state.next_position() != position_offset {
            return Err(Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: format!(
                    "decoder self KV state mismatch: next_position={} position_offset={position_offset}",
                    self.decoder_self_kv_state.next_position()
                ),
            });
        }
        let logits_start = Instant::now();
        let use_reusable_graph = !input.generated_tokens.is_empty()
            && !self.decoder_graph_config.collect_cross_attention
            && reusable_decode_graph_supported_for_runner(self.decoder_graph_runner);
        let step_logits = if use_reusable_graph {
            let token_id = *self
                .decoder_graph_input
                .decoder_prefix_tokens
                .first()
                .ok_or_else(|| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: "decoder reusable step missing token".to_string(),
                })?;
            let graph_run_start = Instant::now();
            let output = run_whisper_decoder_reused_incremental_step_ggml_v0(
                self.decoder_reuse,
                self.decoder_graph_runner,
                self.decoder_persistent_weights,
                &self.decoder_self_kv_state,
                position_offset,
                plan_lookup.plan.as_ref(),
                token_id,
                &self.decoder_weights.tensor_source,
                self.decoder_graph_config,
                &mut self.decoder_tensor_cache,
            )
            .map_err(|error| {
                map_decoder_graph_execution_error(
                    self.decoder_runner.runner_id(),
                    self.decoder_step_input.step_index,
                    graph_token_count,
                    error,
                )
            })
            .map_err(|error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            })?;
            let decoder_graph_run_ms = graph_run_start.elapsed().as_millis();
            if output.logits.len() != self.execution.vocab_size {
                return Err(Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: format!(
                        "runner '{}' returned reusable logits width mismatch at step {}: got {}, expected {}",
                        self.decoder_runner.runner_id(),
                        self.decoder_step_input.step_index,
                        output.logits.len(),
                        self.execution.vocab_size
                    ),
                });
            }
            Ok(WhisperDecoderStepLogits {
                logits: output.logits,
                greedy_token_hint: Some(output.greedy_token),
                last_token_cross_attention_frame_probs: None,
                decoder_graph_run_ms,
                logits_ms: logits_start.elapsed().as_millis(),
            })
        } else {
            self.decoder_runner
                .step_logits(
                    self.runtime_source,
                    self.execution,
                    self.decoder_weights,
                    plan_lookup.plan.as_ref(),
                    &self.decoder_graph_input,
                    self.decoder_graph_config,
                    self.decoder_graph_runner,
                    Some(self.decoder_persistent_weights),
                    Some(&self.decoder_self_kv_state),
                    &mut self.decoder_tensor_cache,
                    &self.decoder_step_input,
                )
                .map_err(|error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                })
        };
        let step_logits = match step_logits {
            Ok(step_logits) => step_logits,
            Err(error) => {
                let logits_ms = logits_start.elapsed().as_millis();
                self.trace.emit_decode_step_metrics(
                    "err",
                    input.step_index,
                    full_token_count,
                    plan_lookup.plan_cache_status.as_str(),
                    plan_lookup.plan_cache_status == WhisperDecoderStepPlanCacheStatus::Hit,
                    plan_lookup.plan_build_ms,
                    logits_ms,
                    logits_ms,
                    step_start.elapsed().as_millis(),
                    self.decode_loop_start,
                );
                return Err(Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                });
            }
        };
        if let (Some(token_id), Some(frame_probs)) = (
            input.generated_tokens.last().copied(),
            step_logits.last_token_cross_attention_frame_probs.clone(),
        ) {
            self.token_alignments.push(WhisperGeneratedTokenAlignment {
                token_id,
                frame_probs,
            });
        }
        self.decoder_self_kv_state.advance(graph_token_count);
        self.decode_steps_completed = self.decode_steps_completed.saturating_add(1);
        self.trace.emit_decode_step_metrics(
            "ok",
            input.step_index,
            full_token_count,
            plan_lookup.plan_cache_status.as_str(),
            plan_lookup.plan_cache_status == WhisperDecoderStepPlanCacheStatus::Hit,
            plan_lookup.plan_build_ms,
            step_logits.decoder_graph_run_ms,
            step_logits.logits_ms,
            step_start.elapsed().as_millis(),
            self.decode_loop_start,
        );
        if self
            .decode_steps_completed
            .is_multiple_of(WHISPER_GGML_TRACE_DECODE_STEP_INTERVAL)
        {
            self.trace.emit_decode_step_progress(
                "step_progress",
                input.step_index,
                full_token_count,
                self.decode_steps_completed,
                self.decode_loop_start,
            );
        }
        Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
            logits: step_logits.logits,
            greedy_token_hint: step_logits.greedy_token_hint,
        })
    }
}

fn run_whisper_decode_loop(
    runtime_source: &GgmlRuntimeSource,
    execution: &WhisperGgmlExecutionMetadata,
    decoder_persistent_static: &mut WhisperDecoderPersistentStaticSession,
    decoder_weights: &WhisperDecoderWeightSeam,
    tokenizer_and_initial_prompt: (&WhisperTokenizer, &[u32]),
    request_options: &GgmlAsrExecutionOptions,
    prelude_result: &WhisperEncoderPreludeSeamResult,
    encoder_result: &WhisperEncoderGraphSeamResult,
    audio_duration_seconds: f32,
    decoder_persistent_cache_populated: bool,
    decoder_runner: &dyn WhisperDecoderLoopRunner,
    trace: &WhisperGgmlTrace,
) -> Result<WhisperExecutionOutput, WhisperGgmlExecutorError> {
    let prelude_summary = match prelude_result {
        WhisperEncoderPreludeSeamResult::GraphExecuted {
            runner_id,
            output_frames,
            output_hidden_size,
            ..
        } => format!(
            "encoder prelude graph executed via runner '{runner_id}' (frames={output_frames}, hidden={output_hidden_size})"
        ),
    };
    let encoder_summary = match encoder_result {
        WhisperEncoderGraphSeamResult::GraphExecuted {
            runner_id,
            layer_count,
            output_frames,
            output_hidden_size,
            ..
        } => format!(
            "encoder graph executed via runner '{runner_id}' (layers={layer_count}, frames={output_frames}, hidden={output_hidden_size})"
        ),
    };

    let (tokenizer, initial_prompt_tokens) = tokenizer_and_initial_prompt;
    let initial_prompt_tokens = initial_prompt_tokens.to_vec();
    let decoder_persistent_weights = &decoder_persistent_static.cache;
    let persistent_weight_plan = &decoder_persistent_static.plan;
    let (encoder_frames, encoder_hidden_size, encoder_hidden_f32) = match encoder_result {
        WhisperEncoderGraphSeamResult::GraphExecuted {
            output_frames,
            output_hidden_size,
            output_hidden_f32,
            ..
        } => (
            *output_frames,
            *output_hidden_size,
            output_hidden_f32.as_slice(),
        ),
    };
    emit_encoder_hidden_probe_trace(encoder_hidden_f32, encoder_frames, encoder_hidden_size);
    let max_generated_tokens = decode_generated_token_step_cap(
        execution.max_target_positions,
        initial_prompt_tokens.len(),
    )
    .map_err(|error| decorate_decoder_boundary_error(error, &prelude_summary, &encoder_summary))?;
    let decode_loop_span = trace.start_stage("decode_loop");
    let decode_loop_start = Instant::now();
    let plan_cache_base = WhisperDecoderStepPlanCacheBase {
        metadata: WhisperDecoderGraphMetadata {
            decoder_layers: execution.decoder_layers,
            decoder_hidden_size: execution.decoder_hidden_size,
            decoder_attention_heads: execution.decoder_attention_heads,
            vocab_size: execution.vocab_size,
            max_target_positions: execution.max_target_positions,
        },
        encoder_frames,
        encoder_hidden_size,
    };
    if !decoder_persistent_cache_populated {
        trace
            .run_stage("decoder_persistent_cache", || {
                decoder_persistent_weights.populate_cross_attention_stage(
                    &mut decoder_persistent_static.runner,
                    persistent_weight_plan,
                    encoder_hidden_f32,
                    WhisperDecoderHiddenStateLayout::SequenceHidden,
                )
            })
            .map_err(|error| {
                decorate_decoder_boundary_error(
                    map_decoder_graph_execution_error(
                        decoder_runner.runner_id(),
                        0,
                        initial_prompt_tokens.len(),
                        error,
                    ),
                    &prelude_summary,
                    &encoder_summary,
                )
            })?;
    }
    let eot_token_id = tokenizer
        .end_of_text_token_id()
        .unwrap_or(execution.eos_token_id);
    let needs_encoder_hidden_in_step =
        !decoder_persistent_weights.supports_cross_attention_for_plan(persistent_weight_plan);
    let word_timestamp_mode = whisper_word_timestamp_mode(request_options);
    let (decoder_cross_flash_attention, decoder_collect_cross_attention) =
        whisper_decoder_cross_attention_flags(
            whisper_decoder_cross_flash_attention_enabled(),
            request_options,
        );
    // Whisper language auto-detection (LID): only when the request language is
    // auto (unset) and the pack is multilingual. Runs one decoder step over
    // `[<sot>]`, reusing the encoder + the cross-attention cache populated above
    // (no second encoder run). Fail-open: any failure leaves the language unset.
    let detected_language: Option<String> = if request_options
        .language
        .as_deref()
        .map(str::trim)
        .filter(|code| !code.is_empty())
        .is_none()
        && execution.vocab_size > WHISPER_ENGLISH_ONLY_MAX_VOCAB_SIZE
    {
        let detect_config = WhisperDecoderGraphExecutionConfig {
            attention_heads: execution.decoder_attention_heads,
            use_self_flash_attention: whisper_decoder_self_flash_attention_enabled(),
            use_cross_flash_attention: decoder_cross_flash_attention,
            collect_cross_attention: false,
            layer_norm_epsilon: 1.0e-5_f32,
        };
        build_whisper_decoder_graph_plan(
            plan_cache_base.metadata,
            &decoder_weights.graph_binding,
            &decoder_weights.graph_materialization,
            plan_cache_base.input_shape(1),
        )
        .ok()
        .and_then(|detect_plan| {
            super::lid::detect_whisper_language_sot_step(
                &mut decoder_persistent_static.runner,
                decoder_persistent_weights,
                &detect_plan,
                &decoder_weights.tensor_source,
                detect_config,
                tokenizer,
                encoder_hidden_f32,
                execution.vocab_size,
            )
        })
    } else {
        None
    };
    // Rebuild the prefix with the detected language. Detecting "en" yields a
    // byte-identical prefix to the unset path (so English audio is unchanged);
    // a missing `<|code|>` token fails open to the unset prefix.
    let initial_prompt_tokens = match detected_language.as_deref() {
        Some(code) => {
            build_whisper_initial_prompt_tokens(execution, tokenizer, request_options, Some(code))
                .unwrap_or(initial_prompt_tokens)
        }
        None => initial_prompt_tokens,
    };
    let mut step_runner = WhisperGreedyDecodeStepRunnerAdapter {
        runtime_source,
        execution,
        decoder_weights,
        decoder_runner,
        trace,
        decode_loop_start,
        decode_steps_completed: 0,
        plan_cache_base,
        decoder_graph_config: WhisperDecoderGraphExecutionConfig {
            attention_heads: execution.decoder_attention_heads,
            use_self_flash_attention: whisper_decoder_self_flash_attention_enabled(),
            use_cross_flash_attention: decoder_cross_flash_attention,
            collect_cross_attention: decoder_collect_cross_attention,
            layer_norm_epsilon: 1.0e-5_f32,
        },
        decoder_persistent_weights,
        decoder_self_kv_state: WhisperDecoderSelfKvCacheState::new(),
        decoder_reuse: &mut decoder_persistent_static.reuse,
        decoder_graph_runner: &mut decoder_persistent_static.runner,
        decoder_graph_input: WhisperDecoderGraphExecutionInput {
            decoder_prefix_tokens: Vec::with_capacity(
                initial_prompt_tokens
                    .len()
                    .saturating_add(max_generated_tokens),
            ),
            encoder_hidden_state: if needs_encoder_hidden_in_step {
                encoder_hidden_f32.to_vec()
            } else {
                Vec::new()
            },
            encoder_layout: WhisperDecoderHiddenStateLayout::SequenceHidden,
        },
        decoder_step_input: WhisperDecoderStepSeamInput {
            encoder_frames,
            encoder_hidden_size,
            step_index: 0,
            position_offset: 0,
        },
        decoder_tensor_cache: WhisperDecoderExecutionTensorCache::default(),
        plan_by_token_count: BTreeMap::new(),
        token_alignments: Vec::new(),
    };
    let decode_text_token_ids = |token_ids: &[u32]| {
        tokenizer.decode_text_token_ids(token_ids).map_err(|error| {
            WhisperGreedyDecodeError::TokenizerDecodeFailed {
                reason: error.to_string(),
            }
        })
    };
    let config = BuiltinSeq2SeqDecodePolicyConfigInput {
        initial_prompt_tokens,
        eot_token_id,
        vocab_size: execution.vocab_size,
        max_generated_tokens,
    };
    let decode = match run_whisper_greedy_decode_loop(
        &config,
        tokenizer,
        request_options.phrase_bias.as_ref(),
        &mut step_runner,
        &decode_text_token_ids,
    ) {
        Ok(decode) => {
            decode_loop_span.finish_with_extra(
                "ok",
                &format!(
                    "steps_executed={} generated_tokens={} max_generated_tokens={}",
                    step_runner.decode_steps_completed,
                    decode.generated_tokens.len(),
                    config.max_generated_tokens
                ),
            );
            decode
        }
        Err(error) => {
            decode_loop_span.finish_with_extra(
                "err",
                &format!(
                    "steps_executed={} max_generated_tokens={}",
                    step_runner.decode_steps_completed, config.max_generated_tokens
                ),
            );
            return Err(decorate_decoder_boundary_error(
                map_greedy_decode_error(error),
                &prelude_summary,
                &encoder_summary,
            ));
        }
    };
    if decode.text.trim().is_empty() {
        return Err(WhisperGgmlExecutorError::DecoderInvalidTokenDecode {
            reason: format!(
                "{prelude_summary}; {encoder_summary}; tokenizer decode produced empty text"
            ),
        });
    }
    let text = decode.text.trim().to_string();
    let words = match word_timestamp_mode {
        WhisperWordTimestampMode::Off => Vec::new(),
        WhisperWordTimestampMode::CrossAttention => whisper_cross_attention_word_timestamps(
            tokenizer,
            &step_runner.token_alignments,
            &decode.generated_probabilities,
            audio_duration_seconds,
        )?,
        WhisperWordTimestampMode::PostHocAnchors => seq2seq_word_timestamps_from_generated_tokens(
            &decode.generated_tokens,
            &decode.generated_probabilities,
            0.0,
            audio_duration_seconds.max(0.0),
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
            &|token_ids| tokenizer.decode_text_token_ids(token_ids),
        )
        .map_err(
            |error| WhisperGgmlExecutorError::DecoderInvalidTokenDecode {
                reason: format!("whisper post-hoc word anchor token decode failed: {error}"),
            },
        )?,
    };
    let segments = if words.is_empty() || text.is_empty() {
        Vec::new()
    } else {
        vec![Segment {
            start: 0.0,
            end: audio_duration_seconds.max(0.0),
            text: text.clone(),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            words,
        }]
    };
    let carry_prompt_token_ids =
        build_whisper_carry_prompt_token_ids(tokenizer, request_options, &decode.generated_tokens)?;
    Ok(WhisperExecutionOutput {
        text,
        segments,
        carry_prompt_token_ids,
        detected_language,
    })
}

fn whisper_encoder_prelude_cpu_graph_config() -> GgmlCpuGraphConfig {
    whisper_encoder_prelude_graph_config()
}

fn map_greedy_decode_error(error: WhisperGreedyDecodeError) -> WhisperGgmlExecutorError {
    match error {
        WhisperGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            max_generated_tokens,
        } => WhisperGgmlExecutorError::DecoderNoEotBeforeMaxTokens {
            max_generated_tokens,
        },
        WhisperGreedyDecodeError::TokenizerDecodeFailed { .. }
        | WhisperGreedyDecodeError::SelectedTokenOutOfVocab { .. }
        | WhisperGreedyDecodeError::EmptyInitialPrompt
        | WhisperGreedyDecodeError::EmptyVocab
        | WhisperGreedyDecodeError::EmptyMaxGeneratedTokens
        | WhisperGreedyDecodeError::EmptyStepLogits { .. }
        | WhisperGreedyDecodeError::StepLogitsVocabMismatch { .. }
        | WhisperGreedyDecodeError::NonFiniteStepLogits { .. } => {
            WhisperGgmlExecutorError::DecoderInvalidTokenDecode {
                reason: error.to_string(),
            }
        }
        WhisperGreedyDecodeError::DecoderStepFailed { reason } => {
            if reason.contains("decoder weights are missing") {
                WhisperGgmlExecutorError::DecoderWeightsMissing { reason }
            } else if reason.contains("decoder graph is unsupported")
                || reason.contains("decoder graph unsupported")
            {
                WhisperGgmlExecutorError::DecoderGraphUnsupported { reason }
            } else {
                WhisperGgmlExecutorError::DecoderGraphExecutionFailed { reason }
            }
        }
    }
}

fn decorate_decoder_boundary_error(
    error: WhisperGgmlExecutorError,
    prelude_summary: &str,
    encoder_summary: &str,
) -> WhisperGgmlExecutorError {
    match error {
        WhisperGgmlExecutorError::TokenizerMissing { reason } => {
            WhisperGgmlExecutorError::TokenizerMissing {
                reason: format!("{prelude_summary}; {encoder_summary}; {reason}"),
            }
        }
        WhisperGgmlExecutorError::DecoderWeightsMissing { reason } => {
            WhisperGgmlExecutorError::DecoderWeightsMissing {
                reason: format!("{prelude_summary}; {encoder_summary}; {reason}"),
            }
        }
        WhisperGgmlExecutorError::DecoderGraphUnsupported { reason } => {
            WhisperGgmlExecutorError::DecoderGraphUnsupported {
                reason: format!("{prelude_summary}; {encoder_summary}; {reason}"),
            }
        }
        WhisperGgmlExecutorError::DecoderGraphExecutionFailed { reason } => {
            WhisperGgmlExecutorError::DecoderGraphExecutionFailed {
                reason: format!("{prelude_summary}; {encoder_summary}; {reason}"),
            }
        }
        other => other,
    }
}

fn map_metadata_contract_error(error: MetadataContractError) -> WhisperGgmlExecutorError {
    match error {
        MetadataContractError::MissingRequiredKey { key } => {
            WhisperGgmlExecutorError::MissingRequiredMetadata { key }
        }
        MetadataContractError::InvalidValue { key, reason } => {
            WhisperGgmlExecutorError::InvalidMetadataValue { key, reason }
        }
    }
}

fn map_tensor_binding_error(error: WhisperGgufTensorBindingError) -> WhisperGgmlExecutorError {
    match error {
        WhisperGgufTensorBindingError::InvalidContext { field, reason } => {
            WhisperGgmlExecutorError::InvalidMetadataValue { key: field, reason }
        }
        WhisperGgufTensorBindingError::MissingRequiredTensor { aliases, .. } => {
            let name = aliases
                .first()
                .cloned()
                .unwrap_or_else(|| "unknown.whisper.tensor".to_string());
            WhisperGgmlExecutorError::MissingRequiredTensor { name }
        }
        WhisperGgufTensorBindingError::TensorTypeMismatch {
            tensor_name,
            found_type,
            expected,
            ..
        } => WhisperGgmlExecutorError::InvalidRequiredTensor {
            name: tensor_name,
            reason: format!("type '{found_type}' does not satisfy expected {expected}"),
        },
        WhisperGgufTensorBindingError::TensorShapeMismatch {
            tensor_name,
            found_shape,
            expected,
            ..
        } => WhisperGgmlExecutorError::InvalidRequiredTensor {
            name: tensor_name,
            reason: format!("shape={found_shape:?} (expected {expected})"),
        },
        WhisperGgufTensorBindingError::EncoderLayerInvariant { layer_idx, reason } => {
            WhisperGgmlExecutorError::InvalidRequiredTensor {
                name: format!("model.encoder.layers.{layer_idx}"),
                reason,
            }
        }
        WhisperGgufTensorBindingError::DecoderLayerInvariant { layer_idx, reason } => {
            WhisperGgmlExecutorError::InvalidRequiredTensor {
                name: format!("model.decoder.layers.{layer_idx}"),
                reason,
            }
        }
        WhisperGgufTensorBindingError::DecoderInvariant { reason } => {
            WhisperGgmlExecutorError::InvalidRequiredTensor {
                name: "model.decoder".to_string(),
                reason,
            }
        }
    }
}

fn map_prelude_plan_error(error: WhisperEncoderPreludePlanError) -> WhisperGgmlExecutorError {
    match error {
        WhisperEncoderPreludePlanError::InvalidInputShape { reason } => {
            WhisperGgmlExecutorError::MelFeatureInputPreparationFailed { reason }
        }
        WhisperEncoderPreludePlanError::TensorShapeMismatch {
            tensor_name,
            reason,
            ..
        } => WhisperGgmlExecutorError::InvalidRequiredTensor {
            name: tensor_name,
            reason,
        },
        WhisperEncoderPreludePlanError::UnsupportedPrimitive { primitive, reason } => {
            WhisperGgmlExecutorError::EncoderPreludePrimitiveUnsupported { primitive, reason }
        }
    }
}

fn map_encoder_graph_plan_error(error: WhisperEncoderGraphPlanError) -> WhisperGgmlExecutorError {
    match error {
        WhisperEncoderGraphPlanError::InvalidInputShape { reason } => {
            WhisperGgmlExecutorError::MelFeatureInputPreparationFailed { reason }
        }
        WhisperEncoderGraphPlanError::LayerCountMismatch {
            metadata_layers,
            binding_layers,
        } => WhisperGgmlExecutorError::EncoderGraphBindingUnsupported {
            reason: format!(
                "encoder layer count mismatch (metadata={metadata_layers}, binding={binding_layers})"
            ),
        },
        WhisperEncoderGraphPlanError::MissingLayerBinding { layer_idx } => {
            WhisperGgmlExecutorError::EncoderGraphBindingUnsupported {
                reason: format!("encoder binding is missing layer {layer_idx}"),
            }
        }
        WhisperEncoderGraphPlanError::MissingTensorBinding { scope, slot } => {
            WhisperGgmlExecutorError::EncoderGraphBindingUnsupported {
                reason: format!("{scope} missing required tensor '{slot}'"),
            }
        }
        WhisperEncoderGraphPlanError::TensorShapeMismatch {
            scope,
            slot,
            tensor_name,
            reason,
            ..
        } => WhisperGgmlExecutorError::InvalidRequiredTensor {
            name: tensor_name,
            reason: format!("{scope} tensor '{slot}' failed shape validation: {reason}"),
        },
        WhisperEncoderGraphPlanError::UnsupportedEncoderPrimitive { primitive, reason } => {
            WhisperGgmlExecutorError::EncoderGraphPrimitiveUnsupported { primitive, reason }
        }
    }
}

fn map_decoder_graph_plan_error(error: WhisperDecoderGraphPlanError) -> WhisperGgmlExecutorError {
    match error {
        WhisperDecoderGraphPlanError::InvalidInputShape { reason } => {
            WhisperGgmlExecutorError::DecoderGraphUnsupported { reason }
        }
        WhisperDecoderGraphPlanError::LayerCountMismatch {
            metadata_layers,
            binding_layers,
        } => WhisperGgmlExecutorError::DecoderGraphUnsupported {
            reason: format!(
                "decoder layer count mismatch (metadata={metadata_layers}, binding={binding_layers})"
            ),
        },
        WhisperDecoderGraphPlanError::MissingLayerBinding { layer_idx } => {
            WhisperGgmlExecutorError::DecoderGraphUnsupported {
                reason: format!("decoder binding is missing layer {layer_idx}"),
            }
        }
        WhisperDecoderGraphPlanError::MissingTensorBinding { scope, slot } => {
            WhisperGgmlExecutorError::DecoderWeightsMissing {
                reason: format!("{scope} missing required tensor '{slot}'"),
            }
        }
        WhisperDecoderGraphPlanError::TensorShapeMismatch {
            scope,
            slot,
            tensor_name,
            reason,
            ..
        } => WhisperGgmlExecutorError::InvalidRequiredTensor {
            name: tensor_name,
            reason: format!("{scope} tensor '{slot}' failed shape validation: {reason}"),
        },
        WhisperDecoderGraphPlanError::UnsupportedDecoderPrimitive { primitive, reason } => {
            WhisperGgmlExecutorError::DecoderGraphUnsupported {
                reason: format!("{primitive}: {reason}"),
            }
        }
    }
}

fn map_decoder_graph_execution_error(
    runner_id: &str,
    step_index: usize,
    token_count: usize,
    error: WhisperDecoderGraphExecutionError,
) -> WhisperGgmlExecutorError {
    let reason = format!(
        "runner '{}' step {} token_count {}: {}",
        runner_id, step_index, token_count, error
    );
    match error {
        WhisperDecoderGraphExecutionError::MissingMaterializedTensor { .. }
        | WhisperDecoderGraphExecutionError::TensorMaterializationFailed { .. } => {
            WhisperGgmlExecutorError::DecoderWeightsMissing { reason }
        }
        WhisperDecoderGraphExecutionError::UnsupportedDecoderPrimitive { .. } => {
            WhisperGgmlExecutorError::DecoderGraphUnsupported { reason }
        }
        WhisperDecoderGraphExecutionError::InvalidInput { .. }
        | WhisperDecoderGraphExecutionError::GraphExecutionFailed { .. } => {
            WhisperGgmlExecutorError::DecoderGraphExecutionFailed { reason }
        }
    }
}

fn map_tensor_materialization_error(error: GgufTensorDataReadError) -> WhisperGgmlExecutorError {
    WhisperGgmlExecutorError::TensorMaterializationFailed {
        reason: error.to_string(),
    }
}

fn map_decoder_weight_materialization_error(
    error: WhisperDecoderWeightMaterializationError,
) -> WhisperGgmlExecutorError {
    match error {
        WhisperDecoderWeightMaterializationError::BindingInvariant { reason } => {
            WhisperGgmlExecutorError::DecoderWeightsMissing { reason }
        }
        WhisperDecoderWeightMaterializationError::BindingTypeMismatch {
            tensor_name,
            expected_type,
            actual_type,
            ..
        } => WhisperGgmlExecutorError::InvalidRequiredTensor {
            name: tensor_name,
            reason: format!(
                "decoder materialization type mismatch: expected ggml_type={expected_type}, actual={actual_type}"
            ),
        },
        WhisperDecoderWeightMaterializationError::BindingShapeMismatch {
            tensor_name,
            expected_shape,
            actual_shape,
            ..
        } => WhisperGgmlExecutorError::InvalidRequiredTensor {
            name: tensor_name,
            reason: format!(
                "decoder materialization shape mismatch: expected={expected_shape:?}, actual={actual_shape:?}"
            ),
        },
        WhisperDecoderWeightMaterializationError::UnsupportedTensorType {
            tensor_name,
            ggml_type,
            type_name,
            ..
        } => WhisperGgmlExecutorError::InvalidRequiredTensor {
            name: tensor_name,
            reason: format!(
                "decoder materialization has unsupported ggml type {ggml_type} ({type_name})"
            ),
        },
        WhisperDecoderWeightMaterializationError::TensorRead {
            slot,
            tensor_name,
            source,
        } => WhisperGgmlExecutorError::TensorMaterializationFailed {
            reason: format!(
                "decoder slot '{slot}' tensor '{tensor_name}' failed to materialize: {source}"
            ),
        },
    }
}

fn map_encoder_weight_materialization_error(
    error: WhisperEncoderWeightMaterializationError,
) -> WhisperGgmlExecutorError {
    match error {
        WhisperEncoderWeightMaterializationError::BindingInvariant { reason } => {
            WhisperGgmlExecutorError::EncoderGraphBindingUnsupported { reason }
        }
        WhisperEncoderWeightMaterializationError::BindingTypeMismatch {
            tensor_name,
            expected_type,
            actual_type,
            ..
        } => WhisperGgmlExecutorError::InvalidRequiredTensor {
            name: tensor_name,
            reason: format!(
                "materialization type mismatch: expected ggml_type={expected_type}, actual={actual_type}"
            ),
        },
        WhisperEncoderWeightMaterializationError::BindingShapeMismatch {
            tensor_name,
            expected_shape,
            actual_shape,
            ..
        } => WhisperGgmlExecutorError::InvalidRequiredTensor {
            name: tensor_name,
            reason: format!(
                "materialization shape mismatch: expected={expected_shape:?}, actual={actual_shape:?}"
            ),
        },
        WhisperEncoderWeightMaterializationError::UnsupportedTensorType {
            tensor_name,
            ggml_type,
            type_name,
            ..
        } => WhisperGgmlExecutorError::InvalidRequiredTensor {
            name: tensor_name,
            reason: format!("unsupported materialized ggml tensor type {ggml_type} ({type_name})"),
        },
        WhisperEncoderWeightMaterializationError::TensorRead {
            slot,
            tensor_name,
            source,
        } => WhisperGgmlExecutorError::TensorMaterializationFailed {
            reason: format!("slot '{slot}' tensor '{tensor_name}' failed to materialize: {source}"),
        },
    }
}

fn map_graph_error(primitive: &'static str, error: GgmlCpuGraphError) -> WhisperGgmlExecutorError {
    match error {
        GgmlCpuGraphError::UnsupportedOperation { .. }
        | GgmlCpuGraphError::UnsupportedInputs { .. }
        | GgmlCpuGraphError::GraphBuildFailed { .. } => {
            WhisperGgmlExecutorError::EncoderPreludePrimitiveUnsupported {
                primitive,
                reason: error.to_string(),
            }
        }
        _ => WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
            reason: format!("{primitive} failed: {error}"),
        },
    }
}

fn map_encoder_graph_error(
    primitive: &'static str,
    error: GgmlCpuGraphError,
) -> WhisperGgmlExecutorError {
    match error {
        GgmlCpuGraphError::UnsupportedOperation { .. }
        | GgmlCpuGraphError::UnsupportedInputs { .. }
        | GgmlCpuGraphError::GraphBuildFailed { .. } => {
            WhisperGgmlExecutorError::EncoderGraphPrimitiveUnsupported {
                primitive,
                reason: error.to_string(),
            }
        }
        _ => WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: format!("{primitive} failed: {error}"),
        },
    }
}

#[cfg(test)]
mod tests;
