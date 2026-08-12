//! Whisper GGUF execution runtime on top of `GgmlAsrViewExecutor`.
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
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use thiserror::Error;

use crate::capacity::topology::StateKind;
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlLoadedTensor, GgmlLoadedWeightContext, GgmlStaticTensor,
    GgmlStaticTensorArena, GgufRuntimeSourcePreflight,
};
use crate::models::admitted_pinned_runtime_actor_pool::{
    AdmittedPinnedRuntimeActorCheckoutPool, AdmittedPinnedRuntimeActorCheckoutPoolLimits,
    PinnedRuntimeActorCheckout, PinnedRuntimeActorError, call_checked_out_actor_mut_async,
};
use crate::models::ggml_asr_executor::GgmlAsrCarryContext;
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_WHISPER_SEQ2SEQ, build_seq2seq_streaming_session,
};
use crate::models::native_execution_services::{ExecutionLaneKey, current_execution_lane_key};
use crate::models::prepared_runtime_cache::{
    HostNeutralPreparedRuntime, PreparedRuntimeCache, PreparedRuntimeHandle,
    PreparedRuntimeQuoteBuilder, PreparedRuntimeQuoteContext, SystemMemoryMaterialization,
};
use crate::models::runtime_cache_coordinator::{PackContentKey, canonical_runtime_cache_path};
use crate::models::runtime_contract::MetadataContractError;
use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;
#[cfg(test)]
use crate::models::seq2seq_decoder_state::Seq2SeqStateAxis;
use crate::models::seq2seq_decoder_state::{Seq2SeqDecoderState, Seq2SeqResidentCapacity};
use crate::models::system_memory_owner::SystemMemoryOwner;
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
    GgmlAsrExecutionError, GgmlAsrExecutionOptions, GgmlAsrExecutionResult,
    GgmlAsrExecutionViewRequest, GgmlAsrPreparedAudioView, GgmlAsrStreamingExecutor,
    GgmlAsrStreamingSessionRequest, GgmlAsrViewExecutor, GgmlFamilyAdapterDescriptor,
    GgmlRuntimeSource, GgufMetadata, GgufTensorDataReader, GgufTensorIndex, NativeAsrSession,
    Segment, Transcription, WHISPER_GGML_ADAPTER_ID,
};
#[cfg(test)]
use crate::{GgufTensorIndexReadError, read_gguf_tensor_index_from_runtime_source};

use super::batched_decode::{
    WhisperServeBatchEngineRegistry, WhisperServeBatchJob, shutdown_whisper_serve_batch_engines,
    submit_whisper_serve_batch_job, whisper_serve_batch_config_from_server_policy,
    whisper_serve_batch_decode_config,
};
#[cfg(test)]
use super::capacity::WHISPER_MAX_GENERATED_TOKENS as WHISPER_DEFAULT_DECODE_MAX_GENERATED_TOKENS_CAP;
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
    build_whisper_decoder_graph_plan, persistent_cross_attention_layer_stride_frames,
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
#[cfg(test)]
use super::tokenizer::WhisperPrefixSpec;
use super::{
    WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT,
    greedy_decode::{
        WhisperGreedyDecodeError, WhisperGreedyDecodeResult, run_whisper_greedy_decode_loop,
    },
    prompt::{
        WhisperPromptError,
        build_whisper_initial_prompt_tokens as build_whisper_initial_prompt_tokens_shared,
    },
    tokenizer::WhisperTokenizer,
};
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicySeq2SeqTextPostprocessKind, BuiltinSeq2SeqDecodePolicyConfigInput,
};
use crate::models::decode_token_history::{
    build_longform_token_history_carry, context_window_budget,
};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeStepExecutor, Seq2SeqGreedyDecodeStepInput,
    Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeStopReason,
};
use crate::models::seq2seq_word_timestamps::{
    Seq2SeqTokenTime, seq2seq_word_timestamps_from_generated_tokens,
    seq2seq_word_timestamps_from_token_times,
};

const WHISPER_STREAMING_EXECUTOR_ID: &str = "whisper-ggml-snapshot-streaming-executor-v1";
/// Largest vocab of an English-only (`.en`) Whisper checkpoint. The canonical
/// Whisper rule (matching whisper.cpp `vocab.is_multilingual()`) is that any
/// checkpoint with a strictly larger vocab carries the language-token block as
/// decode-time prompt state and must be prompted multilingually
/// (`<|sot|> <|en|> <|transcribe|> <|notimestamps|>`). `.en` checkpoints
/// (vocab == this value) keep the bare `<|sot|> <|notimestamps|>` prompt.
pub(crate) const WHISPER_ENGLISH_ONLY_MAX_VOCAB_SIZE: usize = 51_864;
const WHISPER_DECODER_PERSISTENT_SESSION_POOL_CAPACITY: usize = 8;
const WHISPER_RUNTIME_ACTOR_MAX_IDLE_ENTRIES: usize = 8;
const WHISPER_ENCODER_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY: usize = 2;
const WHISPER_DECODER_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY: usize =
    WHISPER_DECODER_PERSISTENT_SESSION_POOL_CAPACITY;
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
    #[error("whisper ggml executor {stage} runtime ownership failed: {reason}")]
    RuntimeOwnershipFailed { stage: &'static str, reason: String },
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
pub(super) struct WhisperPreparedRuntime {
    pub(super) execution: WhisperGgmlExecutionMetadata,
    tensor_binding: WhisperGgmlTensorBinding,
    encoder_weights: WhisperEncoderWeightBundle,
    encoder_materialization: WhisperEncoderTensorMaterializationSeam,
    encoder_binding: WhisperEncoderTensorBindingSeam,
    decoder_weights: WhisperDecoderWeightSeam,
    pub(super) tokenizer: WhisperTokenizer,
}

impl SystemMemoryMaterialization for WhisperPreparedRuntime {
    fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add(
            self.encoder_weights.retained_system_memory_bytes()?,
            "whisper prepared encoder weights",
        )?;
        bytes.add(
            self.decoder_weights
                .tensor_source
                .retained_system_memory_bytes()?,
            "whisper prepared decoder weights",
        )?;
        bytes.add(
            self.tokenizer.retained_system_memory_bytes()?,
            "whisper prepared tokenizer",
        )?;
        Ok(bytes.finish())
    }
}

impl HostNeutralPreparedRuntime for WhisperPreparedRuntime {
    fn system_memory_quote(
        context: PreparedRuntimeQuoteContext<'_>,
        pack_content_id: &str,
    ) -> Result<
        crate::models::system_memory_owner::SystemMemoryAllocationQuote,
        crate::models::system_memory_owner::SystemMemoryOwnerError,
    > {
        let mut quote = PreparedRuntimeQuoteBuilder::new::<Self>(pack_content_id);
        quote.add_tokenizer_metadata(context.metadata, true)?;
        for tensor in context.tensor_index.tensors() {
            match tensor.ggml_type {
                0 => quote.add_tensor_f32(context.tensor_index, &tensor.name)?,
                1 => quote.add_tensor_f16(context.tensor_index, &tensor.name)?,
                _ => quote.add_tensor_raw(context.tensor_index, &tensor.name)?,
            }
        }
        quote.finish()
    }
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
    /// How the shared driver ended this decode, so a cut-short window is not
    /// handed back as a completed one.
    pub(super) stop_reason: Seq2SeqGreedyDecodeStopReason,
}

struct WhisperEncoderPersistentStaticSession {
    runner: GgmlCpuGraphRunner,
    resident_weights: Option<WhisperEncoderResidentWeightCache>,
    graph_config: GgmlCpuGraphConfig,
    encoder_layers: usize,
    encoder_hidden_size: usize,
}

impl WhisperEncoderPersistentStaticSession {
    fn release_transient_compute_memory(&mut self) -> Result<(), WhisperGgmlExecutorError> {
        self.runner
            .release_transient_scheduler_working_set()
            .map_err(
                |error| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
                    reason: format!("could not release transient encoder working set: {error}"),
                },
            )
    }
}

struct WhisperDecoderPersistentStaticSession {
    runner: GgmlCpuGraphRunner,
    cache: WhisperDecoderPersistentWeightCache,
    reuse: Option<Seq2SeqReusableDecodeGraph>,
    graph_config: GgmlCpuGraphConfig,
    plan: WhisperDecoderGraphPlan,
    decoder_state: Seq2SeqDecoderState,
}

/// (pack content id, backend). The content id
/// ([`PackContentKey::for_runtime_source`]) keeps an in-place pack
/// replacement at the same path from reusing a persistent session whose
/// resident weights came from the old bytes.
type WhisperEncoderPersistentSessionKey = (PackContentKey, ExecutionLaneKey);
type WhisperDecoderPersistentSessionKey =
    (PackContentKey, ExecutionLaneKey, Seq2SeqResidentCapacity);

type WhisperEncoderRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<
    WhisperEncoderPersistentSessionKey,
    WhisperEncoderRuntimeActorState,
>;
type WhisperDecoderRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<
    WhisperDecoderPersistentSessionKey,
    WhisperDecoderRuntimeActorState,
>;
type WhisperEncoderRuntimeActor =
    PinnedRuntimeActorCheckout<WhisperEncoderPersistentSessionKey, WhisperEncoderRuntimeActorState>;
type WhisperDecoderRuntimeActor =
    PinnedRuntimeActorCheckout<WhisperDecoderPersistentSessionKey, WhisperDecoderRuntimeActorState>;

const WHISPER_DECODER_GRAPH_RUNNER_ID: &str = "whisper-decoder-graph-ggml-v0";
const WHISPER_ENCODER_PRELUDE_RUNNER_ID: &str = "whisper-cpu-encoder-prelude-ggml-v0";

struct WhisperEncoderRuntimeActorState {
    prelude: Option<WhisperEncoderPreludeCachedRuntime>,
    session: Option<WhisperEncoderPersistentStaticSession>,
    runner: Arc<dyn WhisperEncoderGraphRunner>,
    _prepared_owner: PreparedRuntimeHandle<WhisperPreparedRuntime>,
}

/// Owner-thread resident Whisper encoder prelude.
///
/// The four convolution tensors and the run-length positional prefix are
/// immutable for a `(pack, execution lane, prelude plan)`. Keeping them in one
/// WEIGHTS-usage arena avoids rebuilding the host tensor index, converting the
/// same F16 payload, allocating the same positional prefix, and uploading all
/// five tensors for every request. Only the mel input remains request-local.
/// Field order keeps the arena (whose backend buffer is owned by `runner`)
/// ahead of the runner during drop.
struct WhisperEncoderPreludeCachedRuntime {
    plan: WhisperEncoderPreludePlan,
    graph_config: GgmlCpuGraphConfig,
    conv1_weight: GgmlStaticTensor,
    conv1_bias: GgmlStaticTensor,
    conv2_weight: GgmlStaticTensor,
    conv2_bias: GgmlStaticTensor,
    positional: GgmlStaticTensor,
    arena: GgmlStaticTensorArena,
    runner: GgmlCpuGraphRunner,
}

impl WhisperEncoderPreludeCachedRuntime {
    fn build(
        encoder_weights: &WhisperEncoderWeightBundle,
        plan: &WhisperEncoderPreludePlan,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, WhisperGgmlExecutorError> {
        let graph_config = whisper_encoder_prelude_cpu_graph_config(backend);
        let runner = GgmlCpuGraphRunner::new(graph_config).map_err(|error| {
            WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
                reason: format!("could not initialize cached prelude graph runner: {error}"),
            }
        })?;
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

        let mut arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::metadata_context_bytes(5))
            .map_err(|error| map_graph_error("cached_prelude_static_tensor_arena", error))?;
        let conv1_weight_static = arena
            .new_tensor_3d_f16(
                plan.conv1.kernel_size,
                plan.conv1.in_channels,
                plan.conv1.out_channels,
                "cached_conv1_w",
            )
            .map_err(|error| map_graph_error("ggml_new_tensor_3d_f16(cached_conv1_w)", error))?;
        let conv1_bias_static = arena
            .new_tensor_2d_f32(1, plan.conv1.out_channels, "cached_conv1_b")
            .map_err(|error| map_graph_error("ggml_new_tensor_2d(cached_conv1_b)", error))?;
        let conv2_weight_static = arena
            .new_tensor_3d_f16(
                plan.conv2.kernel_size,
                plan.conv2.in_channels,
                plan.conv2.out_channels,
                "cached_conv2_w",
            )
            .map_err(|error| map_graph_error("ggml_new_tensor_3d_f16(cached_conv2_w)", error))?;
        let conv2_bias_static = arena
            .new_tensor_2d_f32(1, plan.conv2.out_channels, "cached_conv2_b")
            .map_err(|error| map_graph_error("ggml_new_tensor_2d(cached_conv2_b)", error))?;
        let positional_static = arena
            .new_tensor_2d_f32(
                plan.output_hidden_size,
                plan.output_frames,
                "cached_encoder_positional",
            )
            .map_err(|error| {
                map_graph_error("ggml_new_tensor_2d(cached_encoder_positional)", error)
            })?;

        arena
            .set_f16_bits_slice(
                conv1_weight_static,
                conv1_weight_bits.as_ref(),
                "cached_conv1_w",
            )
            .map_err(|error| map_graph_error("upload cached_conv1_w", error))?;
        arena
            .set_f32_slice(conv1_bias_static, conv1_bias_f32.as_ref(), "cached_conv1_b")
            .map_err(|error| map_graph_error("upload cached_conv1_b", error))?;
        arena
            .set_f16_bits_slice(
                conv2_weight_static,
                conv2_weight_bits.as_ref(),
                "cached_conv2_w",
            )
            .map_err(|error| map_graph_error("upload cached_conv2_w", error))?;
        arena
            .set_f32_slice(conv2_bias_static, conv2_bias_f32.as_ref(), "cached_conv2_b")
            .map_err(|error| map_graph_error("upload cached_conv2_b", error))?;
        arena
            .set_f32_slice(
                positional_static,
                positional_f32.as_ref(),
                "cached_encoder_positional",
            )
            .map_err(|error| map_graph_error("upload cached_encoder_positional", error))?;

        Ok(Self {
            plan: plan.clone(),
            graph_config,
            conv1_weight: conv1_weight_static,
            conv1_bias: conv1_bias_static,
            conv2_weight: conv2_weight_static,
            conv2_bias: conv2_bias_static,
            positional: positional_static,
            arena,
            runner,
        })
    }

    fn matches(&self, plan: &WhisperEncoderPreludePlan, backend: GgmlCpuGraphBackend) -> bool {
        self.plan == *plan && self.graph_config == whisper_encoder_prelude_cpu_graph_config(backend)
    }

    fn run(
        &mut self,
        mel_input: &WhisperMelFeatureInput,
    ) -> Result<WhisperEncoderPreludeSeamResult, WhisperGgmlExecutorError> {
        let plan = &self.plan;
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

        let mut graph = self.runner.start_graph();
        let mel = graph
            .new_tensor_2d_f32(
                plan.input_shape.mel_frames,
                plan.input_shape.mel_bins,
                "mel",
            )
            .map_err(|error| map_graph_error("ggml_new_tensor_2d(mel)", error))?;
        graph
            .set_input(mel)
            .map_err(|error| map_graph_error("ggml_set_input(mel)", error))?;

        let conv1 = apply_conv_1d_bias_activation(
            &graph,
            self.arena.graph_tensor(self.conv1_weight),
            mel,
            self.arena.graph_tensor(self.conv1_bias),
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
            self.arena.graph_tensor(self.conv2_weight),
            conv1,
            self.arena.graph_tensor(self.conv2_bias),
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
            .and_then(|tensor| graph.cont(tensor))
            .map_err(|error| map_graph_error("ggml_cont(conv2_transposed)", error))?;
        let prelude_output = graph
            .add(conv2, self.arena.graph_tensor(self.positional))
            .map_err(|error| map_graph_error("ggml_add(encoder_positional)", error))?;
        graph
            .set_output(prelude_output)
            .map_err(|error| map_graph_error("ggml_set_output(encoder_prelude)", error))?;
        graph
            .set_f32_slice(mel, &mel_input.values_f32, "mel")
            .map_err(
                |error| WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
                    reason: format!("could not upload mel feature input: {error}"),
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
        let output_hidden_f32 = graph
            .compute_output_f32(prelude_output, plan.output_frames * plan.output_hidden_size)
            .map_err(
                |error| WhisperGgmlExecutorError::EncoderPreludeExecutionFailed {
                    reason: format!("encoder prelude graph compute failed: {error}"),
                },
            )?;
        Ok(WhisperEncoderPreludeSeamResult::GraphExecuted {
            runner_id: WHISPER_ENCODER_PRELUDE_RUNNER_ID,
            output_frames: plan.output_frames,
            output_hidden_size: plan.output_hidden_size,
            output_hidden_f32,
        })
    }
}

struct WhisperDecoderRuntimeActorState {
    session: Option<WhisperDecoderPersistentStaticSession>,
    _prepared_owner: PreparedRuntimeHandle<WhisperPreparedRuntime>,
}

enum WhisperEncoderResultDelivery {
    Ready(Result<WhisperEncoderGraphSeamResult, WhisperGgmlExecutorError>),
    Pending(
        std::sync::mpsc::Receiver<Result<WhisperEncoderGraphSeamResult, WhisperGgmlExecutorError>>,
    ),
}

impl WhisperEncoderResultDelivery {
    fn receive(self) -> Result<WhisperEncoderGraphSeamResult, WhisperGgmlExecutorError> {
        match self {
            Self::Ready(result) => result,
            Self::Pending(receiver) => {
                receiver
                    .recv()
                    .map_err(|_| WhisperGgmlExecutorError::RuntimeOwnershipFailed {
                        stage: "decoder",
                        reason: "encoder result channel closed before delivery".to_string(),
                    })?
            }
        }
    }
}

struct WhisperDecoderActorJob {
    runtime_preflight: GgufRuntimeSourcePreflight,
    prepared: PreparedRuntimeHandle<WhisperPreparedRuntime>,
    prelude_plan: WhisperEncoderPreludePlan,
    initial_prompt_tokens: Vec<u32>,
    request_options: GgmlAsrExecutionOptions,
    trace: WhisperGgmlTrace,
    prelude_result: WhisperEncoderPreludeSeamResult,
    decoder_state: Seq2SeqDecoderState,
    audio_duration: f32,
    allow_persistent_session_reuse: bool,
    backend: GgmlCpuGraphBackend,
    control: Arc<crate::api::backend::TranscriptionControl>,
    decode_work_progress: Option<crate::api::backend::WorkProgressObserver>,
}

impl WhisperDecoderActorJob {
    fn run(
        self,
        state: &mut WhisperDecoderRuntimeActorState,
        encoder_delivery: WhisperEncoderResultDelivery,
    ) -> Result<WhisperExecutionOutput, WhisperGgmlExecutorError> {
        let result = (|| {
            let graph_config = whisper_decoder_graph_config(self.backend);
            let needs_build = !self.allow_persistent_session_reuse
                || state.session.as_ref().is_none_or(|session| {
                    !decoder_persistent_session_matches_runtime(
                        session,
                        &self.prepared.execution,
                        &self.prelude_plan,
                        self.initial_prompt_tokens.len(),
                        self.decoder_state,
                        graph_config,
                    )
                });
            if needs_build {
                state.session = Some(build_whisper_decoder_persistent_static_session(
                    &self.runtime_preflight,
                    &self.prepared,
                    &self.prelude_plan,
                    self.initial_prompt_tokens.len(),
                    self.decoder_state,
                    &self.trace,
                    self.backend,
                )?);
            }
            let session = state.session.as_mut().ok_or_else(|| {
                WhisperGgmlExecutorError::RuntimeOwnershipFailed {
                    stage: "decoder",
                    reason: "actor session was not initialized".to_string(),
                }
            })?;

            // Build the cross-cache graph while the encoder actor computes its
            // hidden state. The prepared stage borrows this actor's runner and
            // therefore never crosses an owner-thread boundary.
            let prepared_stage = if session
                .cache
                .supports_cross_attention_for_plan(&session.plan)
            {
                Some(
                    self.trace
                        .run_stage("decoder_persistent_cache_prepare", || {
                            session
                                .cache
                                .prepare_cross_attention_stage(&mut session.runner, &session.plan)
                        })
                        .map_err(
                            |error| WhisperGgmlExecutorError::DecoderGraphExecutionFailed {
                                reason: error.to_string(),
                            },
                        )?,
                )
            } else {
                None
            };

            let encoder_result = encoder_delivery.receive()?;
            let actual_encoder_frames = match &encoder_result {
                WhisperEncoderGraphSeamResult::GraphExecuted { output_frames, .. } => {
                    *output_frames
                }
            };
            validate_whisper_decoder_state(
                &self.prepared.execution,
                self.decoder_state,
                actual_encoder_frames,
            )?;

            let mut decoder_persistent_cache_populated = false;
            if let Some(prepared_stage) = prepared_stage {
                let encoder_hidden_f32 = match &encoder_result {
                    WhisperEncoderGraphSeamResult::GraphExecuted {
                        output_hidden_f32, ..
                    } => output_hidden_f32.as_slice(),
                };
                self.trace
                    .run_stage("decoder_persistent_cache", || {
                        session.cache.populate_cross_attention_stage_with_prepared(
                            prepared_stage,
                            &session.plan,
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

            run_whisper_decode_loop(
                &self.prepared.execution,
                session,
                &self.prepared.decoder_weights,
                (
                    &self.prepared.tokenizer,
                    self.initial_prompt_tokens.as_slice(),
                ),
                &self.request_options,
                &self.prelude_result,
                &encoder_result,
                self.audio_duration,
                decoder_persistent_cache_populated,
                &self.trace,
                &self.control,
                self.decode_work_progress.as_ref(),
            )
        })();
        if !self.allow_persistent_session_reuse {
            // Request-scoped graph/KV state must be destroyed on the owner
            // thread on success and on every early error path.
            state.session = None;
        }
        result
    }
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

impl WhisperDecoderMaterializedTensorSource {
    fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        for (name, values) in &self.tensors_f32_by_name {
            bytes.add_string(name, "whisper decoder f32 tensor name")?;
            bytes.add_usize(
                values
                    .len()
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        "whisper decoder f32 tensor payload byte count overflowed".to_string()
                    })?,
                "whisper decoder f32 tensor payload",
            )?;
        }
        for (name, values) in &self.tensors_f16_bits_by_name {
            bytes.add_string(name, "whisper decoder f16 tensor name")?;
            bytes.add_usize(
                values
                    .len()
                    .checked_mul(std::mem::size_of::<u16>())
                    .ok_or_else(|| {
                        "whisper decoder f16 tensor payload byte count overflowed".to_string()
                    })?,
                "whisper decoder f16 tensor payload",
            )?;
        }
        for (name, (_, values)) in &self.tensors_quantized_by_name {
            bytes.add_string(name, "whisper decoder quantized tensor name")?;
            bytes.add_usize(values.len(), "whisper decoder quantized tensor payload")?;
        }
        Ok(bytes.finish())
    }
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
    fn supports_owner_thread_cached_runtime(&self) -> bool {
        false
    }
    fn run_encoder_prelude(
        &self,
        _runtime_source: &GgmlRuntimeSource,
        encoder_weights: &WhisperEncoderWeightBundle,
        plan: &WhisperEncoderPreludePlan,
        mel_input: &WhisperMelFeatureInput,
        backend: GgmlCpuGraphBackend,
    ) -> Result<WhisperEncoderPreludeSeamResult, WhisperGgmlExecutorError>;
}

/// The resolved input one encoder graph compute call runs against: which
/// pack, which architecture metadata, which materialized weights, the
/// planned graph shape, the prelude's hidden-state output, and the backend
/// this request resolved to. Grouped because they always travel together
/// from `execute_whisper_with_prepared_runtime` through the seam and into
/// the owner-thread runner.
struct WhisperEncoderGraphInput<'a> {
    execution: &'a WhisperGgmlExecutionMetadata,
    encoder_weights: &'a WhisperEncoderWeightBundle,
    plan: &'a WhisperEncoderGraphPlan,
    encoder_hidden_input_f32: &'a [f32],
    backend: GgmlCpuGraphBackend,
}

trait WhisperEncoderGraphRunner: Send + Sync {
    fn runner_id(&self) -> &'static str;
    fn run_encoder_graph(
        &self,
        input: WhisperEncoderGraphInput<'_>,
        session: &mut WhisperEncoderPersistentStaticSession,
    ) -> Result<WhisperEncoderGraphSeamResult, WhisperGgmlExecutorError>;
}

trait WhisperMelFeatureInputProvider: Send + Sync {
    fn provider_id(&self) -> &'static str;
    fn prepare_mel_feature_input(
        &self,
        execution: &WhisperGgmlExecutionMetadata,
        prepared_audio: &GgmlAsrPreparedAudioView,
    ) -> Result<WhisperMelFeatureInput, WhisperGgmlExecutorError>;
}

#[derive(Debug, Default, Clone, Copy)]
struct WhisperCpuEncoderPreludeComputeRunnerV0;

#[derive(Debug, Default, Clone, Copy)]
struct WhisperCpuEncoderGraphComputeRunnerV0;

#[derive(Debug, Default, Clone, Copy)]
struct WhisperMelFeatureInputProviderFrontendV0;

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
        WHISPER_ENCODER_PRELUDE_RUNNER_ID
    }

    fn supports_owner_thread_cached_runtime(&self) -> bool {
        true
    }

    fn run_encoder_prelude(
        &self,
        _runtime_source: &GgmlRuntimeSource,
        encoder_weights: &WhisperEncoderWeightBundle,
        plan: &WhisperEncoderPreludePlan,
        mel_input: &WhisperMelFeatureInput,
        backend: GgmlCpuGraphBackend,
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
        let mut runner = GgmlCpuGraphRunner::new(whisper_encoder_prelude_cpu_graph_config(backend))
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
            .start_static_tensor_arena(GgmlCpuGraphConfig::metadata_context_bytes(4))
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
        input: WhisperEncoderGraphInput<'_>,
        session: &mut WhisperEncoderPersistentStaticSession,
    ) -> Result<WhisperEncoderGraphSeamResult, WhisperGgmlExecutorError> {
        let graph_config = whisper_encoder_graph_config(input.backend);
        run_encoder_graph_with_runner(
            self.runner_id(),
            graph_config,
            input.execution,
            input.encoder_weights,
            input.plan,
            input.encoder_hidden_input_f32,
            &mut session.runner,
            session.resident_weights.as_ref(),
        )
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
        let use_flash_attention = whisper_encoder_flash_attention_enabled()
            && graph.supports_flash_attn_ext_head_dim(head_dim);
        let attn_context = if use_flash_attention {
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
    source_tensors: &HashMap<&str, &'weights WhisperMaterializedTensor>,
    encoder_weights: &'weights WhisperEncoderWeightBundle,
    plan: &WhisperEncoderGraphPlan,
    runtime_preflight: &GgufRuntimeSourcePreflight,
) -> Result<WhisperEncoderResidentWeightCache, WhisperGgmlExecutorError> {
    // Worst case: 15 resident handles per layer plus final norm weight/bias.
    // Eligible quantized linears bind zero-copy and use fewer handles, but the
    // topology upper bound keeps sizing independent of pack materialization.
    let arena_tensor_capacity = plan
        .layers
        .len()
        .checked_mul(15)
        .and_then(|count| count.checked_add(2))
        .ok_or_else(|| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
            reason: "encoder resident weight tensor count overflows usize".to_string(),
        })?;
    let mut arena = runner
        .start_static_tensor_arena(GgmlCpuGraphConfig::metadata_context_bytes(
            arena_tensor_capacity,
        ))
        .map_err(|error| map_encoder_graph_error("ggml_static_tensor_arena", error))?;
    // Bind large quantized linear weights zero-copy to the mmap'd pack (no host
    // copy, no arena upload). Falls back to the arena path when unavailable.
    let loaded_weights = Some(
        runner
            .load_gguf_weight_context_from_preflight(runtime_preflight)
            .map_err(
                |source| WhisperGgmlExecutorError::EncoderGraphExecutionFailed {
                    reason: format!("could not load encoder weight context: {source}"),
                },
            )?,
    );
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

fn whisper_encoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    whisper_runtime_graph_config(backend)
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
        prepared_audio: &GgmlAsrPreparedAudioView,
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

#[allow(clippy::too_many_arguments)]
fn run_whisper_decoder_step_ggml_v0(
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
            WHISPER_DECODER_GRAPH_RUNNER_ID,
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
                "runner '{WHISPER_DECODER_GRAPH_RUNNER_ID}' returned logits width mismatch at step {}: got {}, expected {}",
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

fn load_whisper_tokenizer(
    metadata: &GgufMetadata,
) -> Result<WhisperTokenizer, WhisperGgmlExecutorError> {
    materialize_builtin_tokenizer_for_architecture(crate::WHISPER_GGML_ARCHITECTURE_ID, metadata)
        .map_err(|error| WhisperGgmlExecutorError::TokenizerMissing {
            reason: format!(
                "could not materialize tokenizer from preflight GGUF metadata: {error}"
            ),
        })?
        .into_whisper()
        .ok_or_else(|| WhisperGgmlExecutorError::TokenizerMissing {
            reason: "resolved non-whisper tokenizer component for whisper architecture".to_string(),
        })
}

#[derive(Clone)]
pub(crate) struct WhisperGgmlExecutor {
    mel_feature_input_provider: Arc<dyn WhisperMelFeatureInputProvider>,
    encoder_prelude_runner: Arc<dyn WhisperEncoderPreludeRunner>,
    encoder_graph_runner: Arc<dyn WhisperEncoderGraphRunner>,
    runtime_cache_by_path: PreparedRuntimeCache<WhisperPreparedRuntime>,
    serve_batch_engines: WhisperServeBatchEngineRegistry,
    encoder_runtimes: Arc<WhisperEncoderRuntimePool>,
    decoder_runtimes: Arc<WhisperDecoderRuntimePool>,
}

impl std::fmt::Debug for WhisperGgmlExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperGgmlExecutor")
            .finish_non_exhaustive()
    }
}

impl Default for WhisperGgmlExecutor {
    fn default() -> Self {
        let max_committed_requested_bytes =
            crate::host::host_available_memory_bytes().unwrap_or(u64::MAX);
        let encoder_limits = AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
            WHISPER_RUNTIME_ACTOR_MAX_IDLE_ENTRIES,
            max_committed_requested_bytes,
            WHISPER_ENCODER_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY,
        );
        let decoder_limits = AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
            WHISPER_RUNTIME_ACTOR_MAX_IDLE_ENTRIES,
            max_committed_requested_bytes,
            WHISPER_DECODER_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY,
        );
        Self {
            mel_feature_input_provider: Arc::new(WhisperMelFeatureInputProviderFrontendV0),
            encoder_prelude_runner: Arc::new(WhisperCpuEncoderPreludeComputeRunnerV0),
            encoder_graph_runner: Arc::new(WhisperCpuEncoderGraphComputeRunnerV0),
            runtime_cache_by_path: PreparedRuntimeCache::default(),
            serve_batch_engines: WhisperServeBatchEngineRegistry::default(),
            encoder_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-whisper-encoder-owner",
                encoder_limits,
            )),
            decoder_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-whisper-decoder-owner",
                decoder_limits,
            )),
        }
    }
}

fn map_whisper_actor_error(
    stage: &'static str,
    error: PinnedRuntimeActorError,
) -> WhisperGgmlExecutorError {
    WhisperGgmlExecutorError::RuntimeOwnershipFailed {
        stage,
        reason: error.to_string(),
    }
}

fn checkout_whisper_encoder_runtime(
    pool: &WhisperEncoderRuntimePool,
    runtime_source: &GgmlRuntimeSource,
    prepared: PreparedRuntimeHandle<WhisperPreparedRuntime>,
    runner: Arc<dyn WhisperEncoderGraphRunner>,
    backend: GgmlCpuGraphBackend,
) -> Result<WhisperEncoderRuntimeActor, WhisperGgmlExecutorError> {
    let key = (
        PackContentKey::for_runtime_source(runtime_source),
        current_execution_lane_key(whisper_encoder_graph_config(backend).backend),
    );
    pool.checkout_or_try_build_with(
        key,
        move || Ok((0, (prepared, runner))),
        move |(prepared, runner)| {
            // The actor state has no separately retained host payload to quote.
            // Any cached prelude metadata context owns its exact SystemMemory
            // lease, and its static backend arena owns a broker-admitted native
            // allocation, so charging either here would double-account it.
            Ok(SystemMemoryOwner::without_allocation(
                WhisperEncoderRuntimeActorState {
                    prelude: None,
                    session: None,
                    runner,
                    _prepared_owner: prepared,
                },
            ))
        },
        |error| map_whisper_actor_error("encoder", error),
    )
}

fn checkout_whisper_decoder_runtime(
    pool: &WhisperDecoderRuntimePool,
    runtime_source: &GgmlRuntimeSource,
    prepared: PreparedRuntimeHandle<WhisperPreparedRuntime>,
    decoder_state: Seq2SeqDecoderState,
    backend: GgmlCpuGraphBackend,
) -> Result<WhisperDecoderRuntimeActor, WhisperGgmlExecutorError> {
    let key = (
        PackContentKey::for_runtime_source(runtime_source),
        current_execution_lane_key(whisper_decoder_graph_config(backend).backend),
        decoder_state.resident_capacity(),
    );
    pool.checkout_or_try_build_with(
        key,
        move || Ok((0, prepared)),
        move |prepared| {
            Ok(SystemMemoryOwner::without_allocation(
                WhisperDecoderRuntimeActorState {
                    session: None,
                    _prepared_owner: prepared,
                },
            ))
        },
        |error| map_whisper_actor_error("decoder", error),
    )
}

fn run_whisper_encoder_prelude_actor(
    actor: &WhisperEncoderRuntimeActor,
    prepared: PreparedRuntimeHandle<WhisperPreparedRuntime>,
    plan: WhisperEncoderPreludePlan,
    mel_input: WhisperMelFeatureInput,
    backend: GgmlCpuGraphBackend,
) -> Result<WhisperEncoderPreludeSeamResult, WhisperGgmlExecutorError> {
    actor
        .call_mut_fallible(move |state| {
            if !state
                .prelude
                .as_ref()
                .is_some_and(|runtime| runtime.matches(&plan, backend))
            {
                state.prelude = Some(WhisperEncoderPreludeCachedRuntime::build(
                    &prepared.encoder_weights,
                    &plan,
                    backend,
                )?);
            }
            state
                .prelude
                .as_mut()
                .ok_or_else(|| WhisperGgmlExecutorError::RuntimeOwnershipFailed {
                    stage: "encoder_prelude",
                    reason: "actor prelude runtime was not initialized".to_string(),
                })?
                .run(&mel_input)
        })
        .map_err(|error| map_whisper_actor_error("encoder_prelude", error))?
}

fn run_whisper_encoder_actor(
    actor: WhisperEncoderRuntimeActor,
    runtime_preflight: GgufRuntimeSourcePreflight,
    prepared: PreparedRuntimeHandle<WhisperPreparedRuntime>,
    execution: WhisperGgmlExecutionMetadata,
    plan: WhisperEncoderGraphPlan,
    encoder_hidden_input: Vec<f32>,
    allow_persistent_session_reuse: bool,
    backend: GgmlCpuGraphBackend,
    trace: WhisperGgmlTrace,
) -> Result<WhisperEncoderGraphSeamResult, WhisperGgmlExecutorError> {
    actor
        .call_mut(move |state| {
            let graph_config = whisper_encoder_graph_config(backend);
            let can_reuse = allow_persistent_session_reuse
                && state.session.as_ref().is_some_and(|session| {
                    encoder_persistent_session_matches_runtime(
                        session,
                        &execution,
                        &plan,
                        graph_config,
                    )
                });
            if !can_reuse {
                state.session = Some(build_whisper_encoder_persistent_static_session(
                    &runtime_preflight,
                    &execution,
                    &prepared.encoder_weights,
                    &plan,
                    graph_config,
                )?);
            } else {
                emit_encoder_resident_weight_cache_reuse_trace();
            }
            let session = state.session.as_mut().ok_or_else(|| {
                WhisperGgmlExecutorError::RuntimeOwnershipFailed {
                    stage: "encoder",
                    reason: "actor session was not initialized".to_string(),
                }
            })?;
            let result = trace.run_stage("encoder_run", || {
                run_encoder_graph_seam(
                    WhisperEncoderGraphInput {
                        execution: &execution,
                        encoder_weights: &prepared.encoder_weights,
                        plan: &plan,
                        encoder_hidden_input_f32: &encoder_hidden_input,
                        backend,
                    },
                    session,
                    state.runner.as_ref(),
                )
            });
            let release_result = if allow_persistent_session_reuse {
                session.release_transient_compute_memory()
            } else {
                Ok(())
            };
            if !allow_persistent_session_reuse {
                // Short-form requests do not promise cross-request graph
                // reuse. Drop the session while still on its owner thread;
                // the actor itself remains cacheable for future ownership
                // and prepared-runtime reuse.
                state.session = None;
            }
            match (result, release_result) {
                (Ok(output), Ok(())) => Ok(output),
                (Err(error), _) => Err(error),
                (Ok(_), Err(error)) => Err(error),
            }
        })
        .map_err(|error| map_whisper_actor_error("encoder", error))?
}

impl WhisperGgmlExecutor {
    fn prepared_runtime_for_preflight(
        &self,
        preflight: &crate::GgufRuntimeSourcePreflight,
        backend: GgmlCpuGraphBackend,
    ) -> Result<PreparedRuntimeHandle<WhisperPreparedRuntime>, WhisperGgmlExecutorError> {
        self.runtime_cache_by_path.get_or_try_insert_with(
            &preflight.runtime_source,
            PreparedRuntimeQuoteContext {
                model_architecture: crate::WHISPER_GGML_ARCHITECTURE_ID,
                metadata: &preflight.metadata,
                tensor_index: &preflight.tensor_index,
                backend,
            },
            || build_whisper_prepared_runtime(preflight),
            whisper_runtime_cache_slot_unavailable,
            |error| WhisperGgmlExecutorError::TensorMaterializationFailed {
                reason: error.to_string(),
            },
        )
    }

    /// Evicts exactly `pack_content_id`'s cached prepared runtime, releasing
    /// resident state left over from a since-replaced pack without touching
    /// any other content identity. Reached through
    /// [`crate::NativeExecutionServices::evict_prepared_runtime_content_id`].
    pub(crate) fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        self.encoder_runtimes.clear();
        self.decoder_runtimes.clear();
        self.runtime_cache_by_path.evict_content_id(pack_content_id);
        shutdown_whisper_serve_batch_engines(&self.serve_batch_engines);
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

impl GgmlAsrViewExecutor for WhisperGgmlExecutor {
    fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        WhisperGgmlExecutor::evict_prepared_runtime_content_id(self, pack_content_id);
    }

    fn executor_id(&self) -> &'static str {
        crate::arch::WHISPER_EXECUTOR_COMPONENT_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        true
    }

    fn decoder_state_contract(
        &self,
        _selected_family: &GgmlFamilyAdapterDescriptor,
    ) -> Result<crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract, GgmlAsrExecutionError>
    {
        Ok(
            crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract::planned(
                super::capacity::plan_whisper_decoder_state,
                super::capacity::WHISPER_DECODER_STATE_STREAMS,
            ),
        )
    }

    fn replan_streaming_decoder_state(
        &self,
        selected_family: &GgmlFamilyAdapterDescriptor,
        input: &crate::models::ggml_asr_executor::GgmlAsrDecoderStatePlanningInput<'_>,
    ) -> Result<crate::models::ggml_asr_executor::GgmlAsrDecoderState, GgmlAsrExecutionError> {
        if let Some(prepared) = self
            .runtime_cache_by_path
            .ready(&input.preflight.runtime_source)
        {
            let plan = super::capacity::plan_whisper_decoder_state_with_prepared_runtime(
                input,
                prepared.as_ref(),
            )?;
            return Ok(
                crate::models::ggml_asr_executor::GgmlAsrDecoderState::planned(
                    plan,
                    input.envelope,
                ),
            );
        }
        self.decoder_state_contract(selected_family)?
            .plan(input)
            .map_err(Into::into)
    }

    fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        // Offline decode: batch worker allowed.
        self.execute_whisper_inner(request, false)
    }

    fn unload_idle_state(&self) {
        shutdown_whisper_serve_batch_engines(&self.serve_batch_engines);
        self.encoder_runtimes.clear();
        self.decoder_runtimes.clear();
        self.runtime_cache_by_path.clear();
    }
}

impl WhisperGgmlExecutor {
    /// Streaming decode bypasses the batch worker so live sessions stay on the
    /// direct greedy loop. The FINAL transcript remains byte-identical to `execute`.
    pub(crate) fn execute_streaming(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_whisper_inner(request, true)
    }

    fn execute_whisper_inner(
        &self,
        request: &GgmlAsrExecutionViewRequest,
        skip_serve_batch: bool,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        let decoder_state = Seq2SeqDecoderState::from_request_state(
            &request.decoder_state,
            super::capacity::WHISPER_DECODER_STATE_IDS,
        )
        .map_err(|error| GgmlAsrExecutionError::ExecutorFailed {
            executor_id: GgmlAsrViewExecutor::executor_id(self),
            adapter_id: request.selected_family.adapter_id,
            reason: error.to_string(),
        })?;
        let preflight = request.runtime_source_preflight();
        let reuse_runtime_state = request.request_options.longform_mode_enabled();
        let prepared_runtime = self
            .prepared_runtime_for_preflight(preflight, request.resolved_runtime.backend())
            .map_err(|error| GgmlAsrExecutionError::ExecutorFailed {
                executor_id: GgmlAsrViewExecutor::executor_id(self),
                adapter_id: request.selected_family.adapter_id,
                reason: error.to_string(),
            })?;
        let output = execute_whisper_with_prepared_runtime(
            &request.selected_family,
            preflight,
            &request.prepared_audio,
            prepared_runtime,
            decoder_state,
            &request.request_options,
            &self.serve_batch_engines,
            &self.encoder_runtimes,
            &self.decoder_runtimes,
            self.mel_feature_input_provider.as_ref(),
            self.encoder_prelude_runner.as_ref(),
            Arc::clone(&self.encoder_graph_runner),
            reuse_runtime_state,
            skip_serve_batch,
            &request.execution_context,
            request.resolved_runtime.backend(),
        )
        .map_err(|error| match error {
            WhisperGgmlExecutorError::ServeBatchUnavailable { reason, retryable } => {
                GgmlAsrExecutionError::ServeBatchUnavailable { reason, retryable }
            }
            error => GgmlAsrExecutionError::ExecutorFailed {
                executor_id: GgmlAsrViewExecutor::executor_id(self),
                adapter_id: request.selected_family.adapter_id,
                reason: error.to_string(),
            },
        })?;

        let stop_reason = output.stop_reason;
        Ok(GgmlAsrExecutionResult {
            transcription: Transcription {
                truncated_decodes: Vec::new(),
                unnamed_speakers: Vec::new(),
                text: output.text,
                segments: output.segments,
                longform: None,
                language: output.detected_language,
                ..Default::default()
            },
            carry_context: output.carry_prompt_token_ids.map(|prompt_token_ids| {
                GgmlAsrCarryContext {
                    prompt_text: None,
                    prompt_token_ids: Some(prompt_token_ids),
                }
            }),
            // This executor emits one whole-window span rather than per-utterance
            // timestamps, so there is no honest second to anchor the cut to; see
            // `DecodeTruncation::transcript_covers_up_to_seconds`.
            decode_truncation: stop_reason.into_decode_truncation(None),
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
        shutdown_whisper_serve_batch_engines(&self.serve_batch_engines);
        self.encoder_runtimes.clear();
        self.decoder_runtimes.clear();
        self.runtime_cache_by_path.clear();
    }
}

fn build_whisper_prepared_runtime(
    preflight: &GgufRuntimeSourcePreflight,
) -> Result<WhisperPreparedRuntime, WhisperGgmlExecutorError> {
    let execution = validate_whisper_execution_metadata(&preflight.metadata)
        .map_err(map_metadata_contract_error)?;
    let tensor_binding = bind_whisper_required_tensors(&preflight.tensor_index, &execution)?;
    let tensor_reader = build_runtime_tensor_reader_from_preflight(preflight).map_err(|error| {
        WhisperGgmlExecutorError::TensorMaterializationFailed {
            reason: error.to_string(),
        }
    })?;
    let mut encoder_weights =
        materialize_whisper_encoder_weights_from_reader(&tensor_binding, &tensor_reader)?;
    prepare_encoder_runtime_weight_payloads(&mut encoder_weights)?;
    let encoder_materialization = materialize_whisper_encoder_tensor_seam(&encoder_weights);
    let encoder_binding = build_encoder_graph_binding_seam(&encoder_weights, &execution)?;
    let decoder_weights =
        build_decoder_weight_seam(&tensor_reader, &tensor_binding.weights.bindings)?;
    let tokenizer = load_whisper_tokenizer(&preflight.metadata)?;
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

fn whisper_prompt_error_to_executor_error(error: WhisperPromptError) -> WhisperGgmlExecutorError {
    match error {
        WhisperPromptError::LanguageTokenMissing { language } => {
            WhisperGgmlExecutorError::UnsupportedRequestOption {
                option: "language",
                reason: format!("this whisper pack has no <|{language}|> language token"),
            }
        }
        WhisperPromptError::TranslateTokenMissing => {
            WhisperGgmlExecutorError::UnsupportedRequestOption {
                option: "task",
                reason: "this whisper pack has no <|translate|> task token".to_string(),
            }
        }
        WhisperPromptError::EmptyDecoderPrefix
        | WhisperPromptError::PromptEncodingFailed { .. } => {
            WhisperGgmlExecutorError::TokenizerMissing {
                reason: error.to_string(),
            }
        }
        WhisperPromptError::PromptExhaustsContext { .. } | WhisperPromptError::PositionOverflow => {
            WhisperGgmlExecutorError::DecoderGraphUnsupported {
                reason: error.to_string(),
            }
        }
    }
}

fn build_whisper_initial_prompt_tokens(
    execution: &WhisperGgmlExecutionMetadata,
    tokenizer: &WhisperTokenizer,
    request_options: &GgmlAsrExecutionOptions,
    override_language: Option<&str>,
) -> Result<Vec<u32>, WhisperGgmlExecutorError> {
    build_whisper_initial_prompt_tokens_shared(
        execution,
        tokenizer,
        request_options,
        override_language,
    )
    .map_err(whisper_prompt_error_to_executor_error)
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
    if !request_options.longform_prompt_carry_enabled() {
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
    runtime_preflight: &GgufRuntimeSourcePreflight,
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
            &encoder_tensor_index,
            encoder_weights,
            plan,
            runtime_preflight,
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

fn decoder_persistent_session_matches_runtime(
    session: &WhisperDecoderPersistentStaticSession,
    execution: &WhisperGgmlExecutionMetadata,
    prelude_plan: &WhisperEncoderPreludePlan,
    initial_prompt_token_count: usize,
    decoder_state: Seq2SeqDecoderState,
    graph_config: GgmlCpuGraphConfig,
) -> bool {
    let current_schedule_fits =
        decode_generated_token_step_cap(execution.max_target_positions, initial_prompt_token_count)
            .ok()
            .and_then(|generated| {
                crate::capacity::decode_schedule::greedy_self_kv_positions(
                    initial_prompt_token_count,
                    generated,
                )
                .ok()
            })
            .is_some_and(|required| required <= decoder_state.self_attention.logical_positions);
    session.graph_config == graph_config
        && session.decoder_state == decoder_state
        && session.plan.input_shape.encoder_frames == prelude_plan.output_frames
        && session.plan.input_shape.hidden_size == execution.encoder_hidden_size
        && session.plan.layers.len() == execution.decoder_layers
        && session.plan.decoder_attention_heads == execution.decoder_attention_heads
        && session.plan.output_projection.vocab_size == execution.vocab_size
        && session.plan.position_embedding.vocab_size == execution.max_target_positions
        && current_schedule_fits
}

fn validate_whisper_decoder_state(
    execution: &WhisperGgmlExecutionMetadata,
    decoder_state: Seq2SeqDecoderState,
    encoder_frames: usize,
) -> Result<(), WhisperGgmlExecutorError> {
    decoder_state
        .validate()
        .and_then(|()| {
            decoder_state.self_attention.validate_runtime_ceiling(
                StateKind::SelfAttentionKv,
                execution.max_target_positions,
            )
        })
        .and_then(|()| {
            decoder_state
                .cross_attention
                .validate_exact_shape(StateKind::CrossAttentionKv, encoder_frames)
        })
        .and_then(|()| {
            decoder_state.cross_attention.validate_resident_shape(
                StateKind::CrossAttentionKv,
                persistent_cross_attention_layer_stride_frames(encoder_frames),
            )
        })
        .map_err(
            |error| WhisperGgmlExecutorError::DecoderGraphExecutionFailed {
                reason: error.to_string(),
            },
        )
}

fn build_whisper_decoder_session_plan(
    runtime: &WhisperPreparedRuntime,
    prelude_plan: &WhisperEncoderPreludePlan,
    initial_prompt_token_count: usize,
) -> Result<WhisperDecoderGraphPlan, WhisperGgmlExecutorError> {
    build_whisper_decoder_graph_plan(
        WhisperDecoderGraphMetadata {
            decoder_layers: runtime.execution.decoder_layers,
            decoder_hidden_size: runtime.execution.decoder_hidden_size,
            decoder_attention_heads: runtime.execution.decoder_attention_heads,
            vocab_size: runtime.execution.vocab_size,
            semantic_context_positions: runtime.execution.max_target_positions,
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
    )
}

fn build_whisper_decoder_persistent_static_session(
    runtime_preflight: &GgufRuntimeSourcePreflight,
    runtime: &WhisperPreparedRuntime,
    prelude_plan: &WhisperEncoderPreludePlan,
    initial_prompt_token_count: usize,
    decoder_state: Seq2SeqDecoderState,
    trace: &WhisperGgmlTrace,
    backend: GgmlCpuGraphBackend,
) -> Result<WhisperDecoderPersistentStaticSession, WhisperGgmlExecutorError> {
    let graph_config = whisper_decoder_graph_config(backend);
    validate_whisper_decoder_state(
        &runtime.execution,
        decoder_state,
        prelude_plan.output_frames,
    )?;
    let plan =
        build_whisper_decoder_session_plan(runtime, prelude_plan, initial_prompt_token_count)?;
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
                decoder_state.self_attention.resident_positions,
                runtime_preflight,
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
        decoder_state,
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_whisper_with_prepared_runtime(
    adapter: &GgmlFamilyAdapterDescriptor,
    preflight: &GgufRuntimeSourcePreflight,
    prepared_audio: &GgmlAsrPreparedAudioView,
    prepared_owner: PreparedRuntimeHandle<WhisperPreparedRuntime>,
    decoder_state: Seq2SeqDecoderState,
    request_options: &GgmlAsrExecutionOptions,
    serve_batch_engines: &WhisperServeBatchEngineRegistry,
    encoder_runtimes: &WhisperEncoderRuntimePool,
    decoder_runtimes: &WhisperDecoderRuntimePool,
    mel_feature_input_provider: &dyn WhisperMelFeatureInputProvider,
    prelude_runner: &dyn WhisperEncoderPreludeRunner,
    encoder_graph_runner: Arc<dyn WhisperEncoderGraphRunner>,
    allow_persistent_session_reuse: bool,
    skip_serve_batch: bool,
    execution_context: &std::sync::Arc<crate::RequestExecutionContext>,
    resolved_backend: GgmlCpuGraphBackend,
) -> Result<WhisperExecutionOutput, WhisperGgmlExecutorError> {
    let runtime_source = &preflight.runtime_source;
    let runtime = prepared_owner.as_ref();
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
    validate_whisper_decoder_state(
        &runtime.execution,
        decoder_state,
        prelude_plan.output_frames,
    )?;
    let encoder_actor = checkout_whisper_encoder_runtime(
        encoder_runtimes,
        runtime_source,
        Arc::clone(&prepared_owner),
        Arc::clone(&encoder_graph_runner),
        resolved_backend,
    )?;
    let prelude_result = trace.run_stage("prelude_run", || {
        if prelude_runner.supports_owner_thread_cached_runtime() {
            run_whisper_encoder_prelude_actor(
                &encoder_actor,
                Arc::clone(&prepared_owner),
                prelude_plan.clone(),
                mel_input,
                resolved_backend,
            )
        } else {
            run_encoder_prelude_seam(
                runtime_source,
                &runtime.encoder_weights,
                &prelude_plan,
                &mel_input,
                prelude_runner,
                resolved_backend,
            )
        }
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
        } => output_hidden_f32.clone(),
    };
    let audio_duration = audio_duration_seconds(prepared_audio);
    let serve_batch_config =
        whisper_serve_batch_config_from_server_policy(request_options.serve_batch);
    let decoder_graph_config = whisper_decoder_graph_config(resolved_backend);
    let can_use_serve_batch = !skip_serve_batch
        && whisper_can_use_serve_batch(
            decoder_graph_config,
            request_options,
            allow_persistent_session_reuse,
        );
    if let Some(serve_batch_config) = serve_batch_config.filter(|_| can_use_serve_batch) {
        let encoder_result = run_whisper_encoder_actor(
            encoder_actor,
            preflight.clone(),
            Arc::clone(&prepared_owner),
            runtime.execution.clone(),
            encoder_plan.clone(),
            prelude_hidden_output.clone(),
            allow_persistent_session_reuse,
            resolved_backend,
            trace.clone(),
        )?;
        let WhisperEncoderGraphSeamResult::GraphExecuted {
            output_frames,
            output_hidden_size,
            output_hidden_f32,
            ..
        } = encoder_result;
        validate_whisper_decoder_state(&runtime.execution, decoder_state, output_frames)?;
        emit_encoder_hidden_probe_trace(&output_hidden_f32, output_frames, output_hidden_size);
        let eot_token_id = runtime
            .tokenizer
            .end_of_text_token_id()
            .unwrap_or(runtime.execution.eos_token_id);
        let max_generated_tokens = decode_generated_token_step_cap(
            runtime.execution.max_target_positions,
            initial_prompt_tokens.len(),
        )?;
        validate_whisper_self_kv_schedule(
            decoder_state,
            initial_prompt_tokens.len(),
            max_generated_tokens,
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
            serve_batch_engines,
            serve_batch_config,
            WhisperServeBatchJob {
                runtime_cache_path: canonical_runtime_cache_path(runtime_source.path()),
                runtime_preflight: preflight.clone(),
                build_identity:
                    crate::models::ggml_asr_executor::serve_batch_build_identity_for_request(
                        request_options,
                        "whisper",
                        decoder_graph_config.backend,
                        runtime_source,
                    ),
                backend: decoder_graph_config.backend,
                uses_scheduler: decoder_graph_config.use_scheduler,
                execution: runtime.execution.clone(),
                decoder_weights: runtime.decoder_weights.clone(),
                tokenizer: runtime.tokenizer.clone(),
                decoder_state,
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
                execution_context: std::sync::Arc::clone(execution_context),
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
    let decoder_actor = checkout_whisper_decoder_runtime(
        decoder_runtimes,
        runtime_source,
        Arc::clone(&prepared_owner),
        decoder_state,
        resolved_backend,
    )?;

    let decoder_job = WhisperDecoderActorJob {
        runtime_preflight: preflight.clone(),
        prepared: Arc::clone(&prepared_owner),
        prelude_plan: prelude_plan.clone(),
        initial_prompt_tokens,
        request_options: request_options.clone(),
        trace: trace.clone(),
        prelude_result,
        decoder_state,
        audio_duration,
        allow_persistent_session_reuse,
        backend: resolved_backend,
        control: Arc::clone(&execution_context.control),
        decode_work_progress: execution_context.decode_work_progress_observer().cloned(),
    };
    let run_encoder = || {
        run_whisper_encoder_actor(
            encoder_actor,
            preflight.clone(),
            Arc::clone(&prepared_owner),
            runtime.execution.clone(),
            encoder_plan,
            prelude_hidden_output,
            allow_persistent_session_reuse,
            resolved_backend,
            trace,
        )
    };

    if whisper_parallel_encoder_and_decoder_static_enabled(
        resolved_backend,
        allow_persistent_session_reuse,
    ) {
        let (encoder_tx, encoder_rx) = std::sync::mpsc::sync_channel(1);
        let pending_decoder = call_checked_out_actor_mut_async(decoder_actor, move |state| {
            decoder_job.run(state, WhisperEncoderResultDelivery::Pending(encoder_rx))
        })
        .map_err(|error| map_whisper_actor_error("decoder", error))?;
        // The decoder actor is now constructing its static session/cross-cache
        // graph while this thread drives the independent encoder actor.
        let encoder_outcome = run_encoder();
        let _ = encoder_tx.send(encoder_outcome);
        pending_decoder
            .join()
            .map_err(|error| map_whisper_actor_error("decoder", error))?
    } else {
        let encoder_result = run_encoder()?;
        decoder_actor
            .call_mut(move |state| {
                decoder_job.run(
                    state,
                    WhisperEncoderResultDelivery::Ready(Ok(encoder_result)),
                )
            })
            .map_err(|error| map_whisper_actor_error("decoder", error))?
    }
}

#[cfg(test)]
fn whisper_decoder_state_for_execution(
    execution: &WhisperGgmlExecutionMetadata,
) -> Seq2SeqDecoderState {
    let cross_resident =
        persistent_cross_attention_layer_stride_frames(execution.encoder_context_length);
    let self_positions = execution.max_target_positions - 1;
    Seq2SeqDecoderState {
        self_attention: Seq2SeqStateAxis {
            logical_positions: self_positions,
            resident_positions: self_positions,
            hard_position_cap: execution.max_target_positions,
        },
        cross_attention: Seq2SeqStateAxis {
            logical_positions: execution.encoder_context_length,
            resident_positions: cross_resident,
            hard_position_cap: cross_resident,
        },
    }
}

#[cfg(test)]
fn execute_whisper_ggml_non_streaming_cpu(
    adapter: &GgmlFamilyAdapterDescriptor,
    runtime_source: &GgmlRuntimeSource,
    metadata: &GgufMetadata,
    tensor_index: &GgufTensorIndex,
    prepared_audio: &GgmlAsrPreparedAudioView,
    mel_feature_input_provider: &dyn WhisperMelFeatureInputProvider,
    prelude_runner: &dyn WhisperEncoderPreludeRunner,
    encoder_graph_runner: Arc<dyn WhisperEncoderGraphRunner>,
) -> Result<String, WhisperGgmlExecutorError> {
    let preflight = GgufRuntimeSourcePreflight {
        runtime_source: runtime_source.clone(),
        metadata: Arc::new(metadata.clone()),
        tensor_index: Arc::new(tensor_index.clone()),
    };
    let runtime = build_whisper_prepared_runtime(&preflight)?;
    let decoder_state = whisper_decoder_state_for_execution(&runtime.execution);
    let prepared_owner = Arc::new(SystemMemoryOwner::without_allocation(runtime));
    let executor = WhisperGgmlExecutor::default();
    let serve_batch_engines = WhisperServeBatchEngineRegistry::default();
    execute_whisper_with_prepared_runtime(
        adapter,
        &preflight,
        prepared_audio,
        prepared_owner,
        decoder_state,
        &GgmlAsrExecutionOptions::default(),
        &serve_batch_engines,
        &executor.encoder_runtimes,
        &executor.decoder_runtimes,
        mel_feature_input_provider,
        prelude_runner,
        encoder_graph_runner,
        false,
        false,
        &std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
            "test-only non-streaming CPU decode helper",
        )),
        GgmlCpuGraphBackend::Cpu,
    )
    .map(|output| output.text)
}

fn prepare_mel_feature_input_seam(
    provider: &dyn WhisperMelFeatureInputProvider,
    execution: &WhisperGgmlExecutionMetadata,
    prepared_audio: &GgmlAsrPreparedAudioView,
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
    runtime_source: &GgmlRuntimeSource,
    tensor_binding: &WhisperGgmlTensorBinding,
) -> Result<WhisperEncoderWeightBundle, WhisperGgmlExecutorError> {
    let reader = GgufTensorDataReader::from_runtime_source(runtime_source).map_err(|error| {
        WhisperGgmlExecutorError::TensorMaterializationFailed {
            reason: error.to_string(),
        }
    })?;
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
    backend: GgmlCpuGraphBackend,
) -> Result<WhisperEncoderPreludeSeamResult, WhisperGgmlExecutorError> {
    prelude_runner.run_encoder_prelude(
        runtime_source,
        encoder_weights,
        prelude_plan,
        mel_input,
        backend,
    )
}

fn run_encoder_graph_seam(
    input: WhisperEncoderGraphInput<'_>,
    session: &mut WhisperEncoderPersistentStaticSession,
    encoder_graph_runner: &dyn WhisperEncoderGraphRunner,
) -> Result<WhisperEncoderGraphSeamResult, WhisperGgmlExecutorError> {
    encoder_graph_runner.run_encoder_graph(input, session)
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
        .map(|budget| budget.min(super::capacity::WHISPER_MAX_GENERATED_TOKENS))
}

fn validate_whisper_self_kv_schedule(
    decoder_state: Seq2SeqDecoderState,
    initial_prompt_len: usize,
    max_generated_tokens: usize,
) -> Result<(), WhisperGgmlExecutorError> {
    let required = crate::capacity::decode_schedule::greedy_self_kv_positions(
        initial_prompt_len,
        max_generated_tokens,
    )
    .map_err(|error| WhisperGgmlExecutorError::DecoderGraphUnsupported {
        reason: format!("whisper decode schedule is invalid: {error}"),
    })?;
    if required > decoder_state.self_attention.logical_positions {
        return Err(WhisperGgmlExecutorError::DecoderGraphUnsupported {
            reason: format!(
                "whisper greedy schedule requires {required} self-KV rows, planner supplied {}",
                decoder_state.self_attention.logical_positions
            ),
        });
    }
    Ok(())
}

fn audio_duration_seconds(prepared_audio: &GgmlAsrPreparedAudioView) -> f32 {
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
    execution: &'a WhisperGgmlExecutionMetadata,
    decoder_weights: &'a WhisperDecoderWeightSeam,
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
                    WHISPER_DECODER_GRAPH_RUNNER_ID,
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
                        "runner '{WHISPER_DECODER_GRAPH_RUNNER_ID}' returned reusable logits width mismatch at step {}: got {}, expected {}",
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
            run_whisper_decoder_step_ggml_v0(
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

#[allow(clippy::too_many_arguments)]
fn run_whisper_decode_loop(
    execution: &WhisperGgmlExecutionMetadata,
    decoder_persistent_static: &mut WhisperDecoderPersistentStaticSession,
    decoder_weights: &WhisperDecoderWeightSeam,
    tokenizer_and_initial_prompt: (&WhisperTokenizer, &[u32]),
    request_options: &GgmlAsrExecutionOptions,
    prelude_result: &WhisperEncoderPreludeSeamResult,
    encoder_result: &WhisperEncoderGraphSeamResult,
    audio_duration_seconds: f32,
    decoder_persistent_cache_populated: bool,
    trace: &WhisperGgmlTrace,
    control: &std::sync::Arc<crate::api::backend::TranscriptionControl>,
    decode_work_progress: Option<&crate::api::backend::WorkProgressObserver>,
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
    validate_whisper_self_kv_schedule(
        decoder_persistent_static.decoder_state,
        initial_prompt_tokens.len(),
        max_generated_tokens,
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
            semantic_context_positions: execution.max_target_positions,
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
                        WHISPER_DECODER_GRAPH_RUNNER_ID,
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
        execution,
        decoder_weights,
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
        control,
        decode_work_progress,
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
        // Hitting the token budget without EOT degrades to the generated
        // prefix (mirrors cohere/moonshine/qwen, see
        // `qwen::ggml_executor::Qwen3AsrGgmlExecutorError` and
        // `moonshine::decoder_graph::run_moonshine_greedy_decode_loop`'s
        // callers) instead of failing the whole call. This case is reached
        // by ill-conditioned OOD decode (e.g. a language mismatch driving a
        // non-terminating greedy trajectory) more easily on one ggml backend
        // than another -- both backends are equally exposed to it in
        // principle, it is just easier to trigger on Metal in practice (see
        // the platform-audit doc for why: near-tied argmax logits are prone
        // to ULP-level cross-backend flips). It is a genuine decode-quality
        // problem (the model ran out of budget still uncertain what to say
        // next), not a backend bug, so the honest response is to hand back
        // whatever was actually transcribed rather than raise a hard,
        // fail-closed transcription error for an otherwise-successful
        // decode run. The "degraded" trace tag (distinct from both "ok" and
        // "err") keeps this outcome visible rather than silently folding it
        // into a normal completion.
        Err(WhisperGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            generated_tokens,
            generated_probabilities,
            max_generated_tokens,
        }) => {
            decode_loop_span.finish_with_extra(
                "degraded-max-tokens-cap",
                &format!(
                    "steps_executed={} generated_tokens={} max_generated_tokens={max_generated_tokens}",
                    step_runner.decode_steps_completed,
                    generated_tokens.len(),
                ),
            );
            // Same visibility contract as the shared degenerate-ngram guard's
            // own `eprintln!` (`seq2seq_greedy_decode::run_seq2seq_greedy_decode_loop_v0`):
            // a real field occurrence should be observable in stderr, not
            // silently folded into a normal completion.
            eprintln!(
                "openasr_whisper_ggml_executor stage=decode_loop event=max_tokens_cap_degraded status=partial-returned max_generated_tokens={max_generated_tokens} generated_tokens={}",
                generated_tokens.len(),
            );
            let text = decode_text_token_ids(&generated_tokens)
                .map_err(map_greedy_decode_error)
                .map_err(|error| {
                    decorate_decoder_boundary_error(error, &prelude_summary, &encoder_summary)
                })?;
            WhisperGreedyDecodeResult {
                text,
                generated_tokens,
                generated_probabilities,
                // Salvaging the prefix is not the same as completing the
                // decode; the trace tag above says so and so must the result.
                stop_reason: Seq2SeqGreedyDecodeStopReason::BudgetExhausted,
            }
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
            speaker_person_id: None,
            speaker_snapshot_label: None,
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
        stop_reason: decode.stop_reason,
    })
}

fn whisper_encoder_prelude_cpu_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    whisper_encoder_prelude_graph_config(backend)
}

fn map_greedy_decode_error(error: WhisperGreedyDecodeError) -> WhisperGgmlExecutorError {
    match error {
        WhisperGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            max_generated_tokens,
            ..
        } => WhisperGgmlExecutorError::DecoderNoEotBeforeMaxTokens {
            max_generated_tokens,
        },
        // Keep the stable cancel marker in the reason string so
        // `dispatch_error_to_backend` can rewrite to TranscriptionCanceled.
        WhisperGreedyDecodeError::Canceled => WhisperGgmlExecutorError::DecoderInvalidTokenDecode {
            reason: WhisperGreedyDecodeError::Canceled.to_string(),
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
