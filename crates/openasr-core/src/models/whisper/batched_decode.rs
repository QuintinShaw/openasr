use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;

use super::WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT;
use super::execution_policy::{
    whisper_decoder_cross_flash_attention_enabled, whisper_decoder_self_flash_attention_enabled,
};
use super::ggml_decoder_graph::{
    WhisperDecoderExecutionTensorCache, WhisperDecoderGraphExecutionConfig,
    WhisperDecoderGraphExecutionError, WhisperDecoderGraphInputShape, WhisperDecoderGraphMetadata,
    WhisperDecoderGraphPlan, WhisperDecoderHiddenStateLayout, WhisperDecoderPersistentWeightCache,
    WhisperDecoderSelfKvCacheState, build_whisper_decoder_graph_plan,
    persistent_cross_attention_layer_stride_frames,
    run_whisper_decoder_batched_prefill_step_ggml_v0,
    run_whisper_decoder_reused_batched_incremental_step_ggml_v0,
    run_whisper_decoder_reused_incremental_step_ggml_v0,
};
use super::ggml_executor::{WhisperDecoderWeightSeam, WhisperExecutionOutput};
use super::graph_config::whisper_runtime_graph_config;
#[cfg(test)]
use super::graph_config::{WhisperDecoderPlacementPolicy, whisper_decoder_graph_config};
use super::runtime_contract::WhisperGgmlExecutionMetadata;
use super::tokenizer::WhisperTokenizer;
use crate::PhraseBiasConfig;
use crate::Segment;
use crate::capacity::topology::StateKind;
use crate::ggml_runtime::GgufRuntimeSourcePreflight;
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphRunner, GgmlDecodeReuseMode,
};
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicySeq2SeqTextPostprocessKind, BuiltinSeq2SeqDecodePolicyConfigInput,
    build_builtin_seq2seq_decode_policy_config, resolve_builtin_decode_policy,
};
use crate::models::decode_token_history::build_longform_token_history_carry;
use crate::models::seq2seq_decoder_state::Seq2SeqDecoderState;
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeConfig, Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeStopReason,
    build_seq2seq_greedy_stop_token_ids,
};
#[cfg(test)]
use crate::models::seq2seq_serve_batch::{Envelope, OwnerThreadState};
use crate::models::seq2seq_serve_batch::{
    SERVE_BATCH_CANCEL_REASON, Seq2SeqServeBatchFamily, Seq2SeqServeRuntime, ServeBatchConfig,
    ServeBatchEngineRegistry,
};
use crate::models::seq2seq_word_timestamps::seq2seq_word_timestamps_from_generated_tokens;
use crate::models::serve_batch_env::{
    ServeBatchPolicy, serve_batch_estimate_seq2seq_slot_bytes,
    serve_batch_select_and_apply_greedy_step,
};

const WHISPER_SERVE_BATCH_MAX_BATCH_LIMIT: usize = 8;

/// Family-local name for the shared `ServeBatchConfig`.
pub(super) type WhisperServeBatchConfig = ServeBatchConfig;

/// Resolve the shared server policy while keeping the family marker private to
/// the batched-decode module.
pub(super) fn whisper_serve_batch_config_from_server_policy(
    policy: ServeBatchPolicy,
) -> Option<WhisperServeBatchConfig> {
    ServeBatchConfig::from_policy::<WhisperFamily>(policy)
}

#[derive(Debug, Clone)]
pub(crate) struct WhisperServeBatchJob {
    pub runtime_cache_path: PathBuf,
    /// The same already-open, already-validated source the submitting
    /// thread's preflight resolved. Cloning it is a refcount bump on its
    /// `Arc<Mmap>`, not a reopen -- the owner thread that actually builds the
    /// persistent decoder weight cache binds resident weights from this same
    /// mapping instead of a fresh `File::open`/`load_gguf_weight_context` by
    /// path, so identity and weight bytes come from one open even across
    /// this thread boundary.
    pub runtime_preflight: GgufRuntimeSourcePreflight,
    pub build_identity: crate::RuntimeBuildIdentity,
    pub backend: GgmlCpuGraphBackend,
    pub uses_scheduler: bool,
    pub reuse_mode: GgmlDecodeReuseMode,
    pub execution: WhisperGgmlExecutionMetadata,
    pub decoder_weights: WhisperDecoderWeightSeam,
    pub tokenizer: WhisperTokenizer,
    pub decoder_state: Seq2SeqDecoderState,
    pub encoder_frames: usize,
    pub encoder_hidden_size: usize,
    pub encoder_hidden_f32: Vec<f32>,
    pub decode_config: Seq2SeqGreedyDecodeConfig,
    pub word_timestamps: bool,
    pub audio_duration_seconds: f32,
    pub carry_prompt_seed_token_ids: Option<Vec<u32>>,
    /// Explicit cancel/pause/resume context for this job -- never a
    /// thread-local. See [`crate::RequestExecutionContext`].
    pub execution_context: std::sync::Arc<crate::RequestExecutionContext>,
}

#[derive(Debug, Error)]
pub(crate) enum WhisperServeBatchError {
    #[cfg(test)]
    #[error("whisper serve batch test env {env} must be an integer in 0..={max}, got '{raw}'")]
    InvalidEnv {
        env: &'static str,
        raw: String,
        max: usize,
    },
    #[error("whisper serve batch requires max batch >= 2 when enabled, got {max_batch}")]
    InvalidEnabledBatch { max_batch: usize },
    #[error("whisper serve batch supports only gpu-class direct ggml backends, got {backend:?}")]
    UnsupportedBackend { backend: GgmlCpuGraphBackend },
    #[error("whisper serve batch engine registry mutex is poisoned")]
    RegistryPoisoned,
    #[error("whisper serve batch owner thread spawn failed: {reason}")]
    ThreadSpawnFailed { reason: String },
    #[error("whisper serve batch queue is full")]
    QueueFull,
    #[error("whisper serve batch owner thread is disconnected")]
    OwnerDisconnected,
    #[error("whisper serve batch owner reply timed out")]
    ReplyTimedOut,
    #[error("whisper serve batch owner failed: {reason}")]
    OwnerFailed { reason: String },
    #[error("whisper serve batch decode failed: {reason}")]
    DecodeFailed { reason: String },
}

impl WhisperServeBatchError {
    /// Classifies the transient serve-batch failures that should surface as a
    /// retryable HTTP status. `Some(true)` => queue saturation (429 backpressure);
    /// `Some(false)` => owner gone / GPU step hung (503); `None` => every other
    /// variant keeps its existing (non-retryable) mapping.
    pub(crate) fn unavailable_retryable(&self) -> Option<bool> {
        match self {
            Self::QueueFull => Some(true),
            Self::OwnerDisconnected | Self::ReplyTimedOut => Some(false),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct WhisperServeBatchEngineKey {
    build_identity: crate::RuntimeBuildIdentity,
    lane: crate::models::native_execution_services::ExecutionLaneKey,
    resident_self_positions: usize,
    resident_cross_positions: usize,
    hidden_size: usize,
    uses_scheduler: bool,
    max_batch: usize,
}

pub(super) type WhisperServeBatchEngineRegistry = ServeBatchEngineRegistry<WhisperFamily>;

/// The whisper serve-batch ZST family wiring (`Seq2SeqServeBatchFamily`) that
/// drives the generic `OwnerThreadState` + generic `ServeBatchEngine`. Whisper
/// is the superset: it overrides the self-KV reset, the Vulkan->serial cap, the
/// non-flash batched execution config, the reset+incremental serial path, and
/// the longform carry-prompt finish.
pub(super) struct WhisperFamily;

#[cfg(test)]
type WhisperServeBatchEnvelope = Envelope<WhisperFamily>;

pub(super) struct WhisperServeDecoderRuntime {
    runner: GgmlCpuGraphRunner,
    persistent_weights: WhisperDecoderPersistentWeightCache,
    reuse: Option<crate::nn::decoder::Seq2SeqReusableDecodeGraph>,
    plan: WhisperDecoderGraphPlan,
    decoder_weights: WhisperDecoderWeightSeam,
    tensor_cache: WhisperDecoderExecutionTensorCache,
    config: WhisperDecoderGraphExecutionConfig,
    self_kv_state: WhisperDecoderSelfKvCacheState,
    decoder_state: Seq2SeqDecoderState,
}

pub(crate) struct WhisperBatchSlot {
    job: WhisperServeBatchJob,
    stop_token_ids: Vec<u32>,
    generated_tokens: Vec<u32>,
    /// Per-token softmax probability, parallel to `generated_tokens`.
    generated_probabilities: Vec<f32>,
    /// How this slot's decode ended, `None` while it is still running.
    /// Mirrors the single-utterance driver so a guard-truncated slot is
    /// distinguishable from one that reached its stop token.
    stop_reason: Option<Seq2SeqGreedyDecodeStopReason>,
}

pub(super) fn submit_whisper_serve_batch_job(
    registry: &WhisperServeBatchEngineRegistry,
    config: WhisperServeBatchConfig,
    job: WhisperServeBatchJob,
) -> Result<WhisperExecutionOutput, WhisperServeBatchError> {
    let config = config.validate_for_job::<WhisperFamily>(&job)?;
    let key = WhisperFamily::engine_key(&job, config.max_batch);
    let engine = registry.engine_for_key(key.clone(), config)?;
    let result = engine.submit(job);
    registry.evict_after_candidate_failure(&key, &engine);
    result
}

pub(super) fn shutdown_whisper_serve_batch_engines(registry: &WhisperServeBatchEngineRegistry) {
    registry.shutdown();
}

fn whisper_serve_batch_vram_slot_bytes(job: &WhisperServeBatchJob) -> usize {
    serve_batch_estimate_seq2seq_slot_bytes(
        job.execution.decoder_layers,
        job.decoder_state.self_attention.resident_positions,
        job.execution.decoder_hidden_size,
        job.decoder_state.cross_attention.resident_positions,
        job.encoder_hidden_size,
        std::mem::size_of::<u16>(),
        std::mem::size_of::<u16>(),
    )
}

fn validate_whisper_serve_decoder_state(
    job: &WhisperServeBatchJob,
) -> Result<(), WhisperServeBatchError> {
    let semantic_positions = job
        .decode_config
        .initial_prompt_tokens
        .len()
        .checked_add(job.decode_config.max_generated_tokens)
        .ok_or_else(|| WhisperServeBatchError::DecodeFailed {
            reason: "whisper semantic decoder positions overflowed".to_string(),
        })?;
    if semantic_positions > job.execution.max_target_positions {
        return Err(WhisperServeBatchError::DecodeFailed {
            reason: format!(
                "whisper prompt plus generation budget {} exceeds semantic context {}",
                semantic_positions, job.execution.max_target_positions
            ),
        });
    }
    let required_self_positions = crate::capacity::decode_schedule::greedy_self_kv_positions(
        job.decode_config.initial_prompt_tokens.len(),
        job.decode_config.max_generated_tokens,
    )
    .map_err(|error| WhisperServeBatchError::DecodeFailed {
        reason: format!("whisper decode schedule is invalid: {error}"),
    })?;
    if required_self_positions > job.decoder_state.self_attention.logical_positions {
        return Err(WhisperServeBatchError::DecodeFailed {
            reason: format!(
                "whisper greedy schedule requires {} self-KV rows, planner supplied {}",
                required_self_positions, job.decoder_state.self_attention.logical_positions
            ),
        });
    }
    let cross_resident = persistent_cross_attention_layer_stride_frames(job.encoder_frames);
    job.decoder_state
        .validate()
        .and_then(|()| {
            job.decoder_state.self_attention.validate_runtime_ceiling(
                StateKind::SelfAttentionKv,
                job.execution.max_target_positions,
            )
        })
        .and_then(|()| {
            job.decoder_state
                .cross_attention
                .validate_exact_shape(StateKind::CrossAttentionKv, job.encoder_frames)
        })
        .and_then(|()| {
            job.decoder_state
                .cross_attention
                .validate_runtime_ceiling(StateKind::CrossAttentionKv, cross_resident)
        })
        .and_then(|()| {
            job.decoder_state
                .cross_attention
                .validate_resident_shape(StateKind::CrossAttentionKv, cross_resident)
        })
        .map_err(|error| WhisperServeBatchError::DecodeFailed {
            reason: error.to_string(),
        })
}

impl WhisperServeDecoderRuntime {
    fn new(job: &WhisperServeBatchJob, n_seq: usize) -> Result<Self, WhisperServeBatchError> {
        validate_whisper_serve_decoder_state(job)?;
        // This owner runs after the submitting request's Exact route TLS has
        // ended. Keep its graph placement and scheduler as the submitting
        // thread's resolved snapshot; re-running decoder policy here could
        // otherwise change a small CUDA/Vulkan decoder back to a scheduler
        // graph on the owner thread.
        let mut graph_config = whisper_runtime_graph_config(job.backend);
        graph_config.use_scheduler = job.uses_scheduler;
        let plan = build_whisper_decoder_graph_plan(
            WhisperDecoderGraphMetadata {
                decoder_layers: job.execution.decoder_layers,
                decoder_hidden_size: job.execution.decoder_hidden_size,
                decoder_attention_heads: job.execution.decoder_attention_heads,
                vocab_size: job.execution.vocab_size,
                semantic_context_positions: job.execution.max_target_positions,
            },
            &job.decoder_weights.graph_binding,
            &job.decoder_weights.graph_materialization,
            WhisperDecoderGraphInputShape {
                token_count: job.decode_config.initial_prompt_tokens.len().max(1),
                encoder_frames: job.encoder_frames,
                hidden_size: job.execution.encoder_hidden_size,
            },
        )
        .map_err(map_decoder_plan_error)?;
        let mut runner = GgmlCpuGraphRunner::new(graph_config).map_err(|error| {
            WhisperServeBatchError::DecodeFailed {
                reason: format!("could not initialize whisper serve-batch decoder runner: {error}"),
            }
        })?;
        let mut tensor_cache = WhisperDecoderExecutionTensorCache::default();
        let persistent_weights =
            WhisperDecoderPersistentWeightCache::build_static_stage_with_n_seq(
                &mut runner,
                &plan,
                &job.decoder_weights.tensor_source,
                &mut tensor_cache,
                job.decoder_state.self_attention.resident_positions,
                &job.runtime_preflight,
                n_seq,
            )
            .map_err(map_decoder_error)?;
        Ok(Self {
            runner,
            persistent_weights,
            reuse: None,
            plan,
            decoder_weights: job.decoder_weights.clone(),
            tensor_cache,
            config: whisper_serve_batch_decoder_graph_execution_config(
                job.execution.decoder_attention_heads,
                n_seq,
            ),
            self_kv_state: WhisperDecoderSelfKvCacheState::new(),
            decoder_state: job.decoder_state,
        })
    }

    fn reset_self_kv_state(&mut self) {
        self.self_kv_state = WhisperDecoderSelfKvCacheState::new();
    }

    fn populate_cross_attention_cache_slot(
        &mut self,
        slot_index: usize,
        job: &WhisperServeBatchJob,
    ) -> Result<(), WhisperServeBatchError> {
        if job.decoder_state != self.decoder_state {
            return Err(WhisperServeBatchError::DecodeFailed {
                reason: format!(
                    "whisper serve-batch resident runtime state mismatch: built={:?}, requested={:?}",
                    self.decoder_state, job.decoder_state
                ),
            });
        }
        validate_whisper_serve_decoder_state(job)?;
        self.persistent_weights
            .populate_cross_attention_stage_slot(
                &mut self.runner,
                &self.plan,
                &job.encoder_hidden_f32,
                WhisperDecoderHiddenStateLayout::SequenceHidden,
                slot_index,
            )
            .map_err(map_decoder_error)
    }

    fn compute_reused_step_logits(
        &mut self,
        token_id: u32,
        position: usize,
    ) -> Result<Vec<f32>, WhisperServeBatchError> {
        let output = run_whisper_decoder_reused_incremental_step_ggml_v0(
            &mut self.reuse,
            &mut self.runner,
            &self.persistent_weights,
            &self.self_kv_state,
            position,
            &self.plan,
            token_id,
            &self.decoder_weights.tensor_source,
            self.config,
            &mut self.tensor_cache,
        )
        .map_err(map_decoder_error)?;
        self.self_kv_state.advance(1);
        Ok(output.logits)
    }

    fn compute_reused_batched_step_logits(
        &mut self,
        token_ids: &[u32],
        positions: &[usize],
        totals: &[usize],
    ) -> Result<Vec<f32>, WhisperServeBatchError> {
        let output = run_whisper_decoder_reused_batched_incremental_step_ggml_v0(
            &mut self.reuse,
            &mut self.runner,
            &self.persistent_weights,
            &self.plan,
            token_ids,
            positions,
            totals,
            &self.decoder_weights.tensor_source,
            self.config,
            &mut self.tensor_cache,
        )
        .map_err(map_decoder_error)?;
        Ok(output.logits)
    }

    fn compute_batched_prefill_logits(
        &mut self,
        prompt_tokens: &[u32],
    ) -> Result<Vec<f32>, WhisperServeBatchError> {
        self.reuse = None;
        let output = run_whisper_decoder_batched_prefill_step_ggml_v0(
            &mut self.runner,
            &self.persistent_weights,
            &self.plan,
            prompt_tokens,
            &self.decoder_weights.tensor_source,
            self.config,
            &mut self.tensor_cache,
        )
        .map_err(map_decoder_error)?;
        Ok(output.logits)
    }
}

impl Seq2SeqServeRuntime for WhisperServeDecoderRuntime {
    type Job = WhisperServeBatchJob;
    type Error = WhisperServeBatchError;

    fn build_serial(job: &Self::Job) -> Result<Self, Self::Error> {
        // n_seq == 1 builds the serial flash-policy runtime (the non-flash
        // batched config only applies for n_seq > 1; see
        // `whisper_serve_batch_decoder_graph_execution_config`).
        WhisperServeDecoderRuntime::new(job, 1)
    }

    fn build_batched(job: &Self::Job, n_seq: usize) -> Result<Self, Self::Error> {
        // The n_seq > 1 NON-FLASH execution config is selected inside `new` via
        // `whisper_serve_batch_decoder_graph_execution_config(.., n_seq)`; the
        // policy stays inside runtime construction, not a loop hook.
        WhisperServeDecoderRuntime::new(job, n_seq)
    }

    fn configure_for_job(&mut self, job: &Self::Job) -> Result<(), Self::Error> {
        validate_whisper_serve_decoder_state(job)?;
        if job.decoder_state != self.decoder_state {
            return Err(WhisperServeBatchError::DecodeFailed {
                reason: format!(
                    "whisper serve-batch resident runtime state mismatch: built={:?}, requested={:?}",
                    self.decoder_state, job.decoder_state
                ),
            });
        }
        Ok(())
    }

    fn reset_self_kv_state(&mut self) {
        WhisperServeDecoderRuntime::reset_self_kv_state(self);
    }

    fn populate_cross_attention_cache_serial(
        &mut self,
        job: &Self::Job,
    ) -> Result<(), Self::Error> {
        // Whisper's serial path drives slot 0 of the resident runtime; the
        // generic owner never calls this for whisper because `decode_serial`
        // is overridden, but the contract requires it.
        self.populate_cross_attention_cache_slot(0, job)
    }

    fn populate_cross_attention_cache_slot(
        &mut self,
        slot_index: usize,
        job: &Self::Job,
    ) -> Result<(), Self::Error> {
        WhisperServeDecoderRuntime::populate_cross_attention_cache_slot(self, slot_index, job)
    }

    fn compute_batched_prefill_logits(
        &mut self,
        prompt_tokens: &[u32],
    ) -> Result<Vec<f32>, Self::Error> {
        WhisperServeDecoderRuntime::compute_batched_prefill_logits(self, prompt_tokens)
    }

    fn compute_reused_batched_step_logits(
        &mut self,
        token_ids: &[u32],
        positions: &[usize],
        totals: &[usize],
    ) -> Result<Vec<f32>, Self::Error> {
        WhisperServeDecoderRuntime::compute_reused_batched_step_logits(
            self, token_ids, positions, totals,
        )
    }
}

impl Seq2SeqServeBatchFamily for WhisperFamily {
    type Runtime = WhisperServeDecoderRuntime;
    type Job = WhisperServeBatchJob;
    type Slot = WhisperBatchSlot;
    type Output = WhisperExecutionOutput;
    type Error = WhisperServeBatchError;
    type EngineKey = WhisperServeBatchEngineKey;

    const THREAD_NAME_PREFIX: &'static str = "whisper";
    const MAX_BATCH_LIMIT: usize = WHISPER_SERVE_BATCH_MAX_BATCH_LIMIT;

    fn engine_key(job: &Self::Job, max_batch: usize) -> Self::EngineKey {
        WhisperServeBatchEngineKey {
            build_identity: job.build_identity.clone(),
            lane: crate::models::native_execution_services::current_execution_lane_key(job.backend),
            resident_self_positions: job.decoder_state.self_attention.resident_positions,
            resident_cross_positions: job.decoder_state.cross_attention.resident_positions,
            hidden_size: job.encoder_hidden_size,
            uses_scheduler: job.uses_scheduler,
            max_batch,
        }
    }

    fn engine_key_backend(key: &Self::EngineKey) -> GgmlCpuGraphBackend {
        key.lane.backend()
    }

    fn can_batch_with(a: &Self::Job, b: &Self::Job) -> bool {
        a.can_batch_with(b)
    }

    fn vram_slot_bytes(job: &Self::Job) -> usize {
        whisper_serve_batch_vram_slot_bytes(job)
    }

    fn backend(job: &Self::Job) -> GgmlCpuGraphBackend {
        job.backend
    }

    fn uses_scheduler(job: &Self::Job) -> bool {
        job.uses_scheduler
    }

    fn reuse_mode(job: &Self::Job) -> GgmlDecodeReuseMode {
        job.reuse_mode
    }

    fn effective_max_batch_after_vram_cap(
        capped_max_batch: usize,
        job: &Self::Job,
    ) -> Result<usize, Self::Error> {
        // Applied AFTER the generic VRAM cap (order is load-bearing: it affects
        // the engine key). Shared runtime code resolves the concrete provider
        // spelling into typed facts; family code never parses backend names.
        // Cohere / moonshine keep the default identity hook and therefore do
        // not initialize a backend guard here.
        let capabilities = GgmlCpuGraphConfig::resolve_backend_capabilities_for(job.backend)
            .map_err(|error| WhisperServeBatchError::DecodeFailed {
                reason: format!(
                    "could not resolve whisper serve-batch backend capabilities: {error}"
                ),
            })?;
        Ok(whisper_serve_batch_effective_max_batch_for_capabilities(
            capped_max_batch,
            capabilities,
        ))
    }

    fn initial_prompt_tokens(job: &Self::Job) -> &[u32] {
        &job.decode_config.initial_prompt_tokens
    }

    fn vocab_size(job: &Self::Job) -> usize {
        job.decode_config.vocab_size
    }

    fn max_generated_tokens(job: &Self::Job) -> usize {
        job.decode_config.max_generated_tokens
    }

    fn decoder_max_context(job: &Self::Job) -> usize {
        job.decoder_state.self_attention.logical_positions
    }

    fn slot_new(job: Self::Job) -> Result<Self::Slot, Self::Error> {
        WhisperBatchSlot::new(job)
    }

    fn slot_job(slot: &Self::Slot) -> &Self::Job {
        &slot.job
    }

    fn slot_generated(slot: &Self::Slot) -> &[u32] {
        &slot.generated_tokens
    }

    fn slot_done(slot: &Self::Slot) -> bool {
        slot.is_done()
    }

    fn slot_select_next_token(slot: &mut Self::Slot, logits: Vec<f32>) -> Result<(), Self::Error> {
        slot.select_next_token_from_logits(logits)
    }

    fn slot_finish(slot: Self::Slot) -> Result<Self::Output, Self::Error> {
        slot.finish()
    }

    fn decode_serial(
        serial_runtime: &mut Option<Self::Runtime>,
        job: Self::Job,
    ) -> Result<Self::Output, Self::Error> {
        // Ported VERBATIM from the previous `WhisperOwnerThreadState::decode_serial_job`
        // (reset_self_kv_state + populate cross-KV slot 0 + per-position
        // compute_reused_step_logits prompt prefill + greedy loop + slot.finish).
        if serial_runtime.is_none() {
            *serial_runtime = Some(WhisperServeDecoderRuntime::new(&job, 1)?);
        }
        let runtime =
            serial_runtime
                .as_mut()
                .ok_or_else(|| WhisperServeBatchError::OwnerFailed {
                    reason: "whisper serve batch serial runtime cache is unexpectedly empty"
                        .to_string(),
                })?;
        runtime.reset_self_kv_state();
        runtime.populate_cross_attention_cache_slot(0, &job)?;
        let mut slot = WhisperBatchSlot::new(job)?;
        let prompt_len = slot.job.decode_config.initial_prompt_tokens.len();
        let mut logits = Vec::new();
        for position in 0..prompt_len {
            logits = runtime.compute_reused_step_logits(
                slot.job.decode_config.initial_prompt_tokens[position],
                position,
            )?;
        }
        slot.select_next_token_from_logits(logits)?;

        loop {
            if slot.stop_reason.is_none()
                && slot.generated_tokens.len() >= slot.job.decode_config.max_generated_tokens
            {
                slot.stop_reason = Some(Seq2SeqGreedyDecodeStopReason::BudgetExhausted);
            }
            if slot.is_done() {
                break;
            }
            // Cooperative cancel at each token step, mirroring the shared
            // greedy driver's L1 check: this serial path bypasses that driver
            // (see the module-level note on the ported-verbatim loop below),
            // so it needs its own check against the job's own context.
            if slot.job.execution_context.control.is_canceled() {
                return Err(WhisperServeBatchError::DecodeFailed {
                    reason: SERVE_BATCH_CANCEL_REASON.to_string(),
                });
            }
            let token_id = *slot.generated_tokens.last().ok_or_else(|| {
                WhisperServeBatchError::DecodeFailed {
                    reason: "whisper serve batch generated token history is empty".to_string(),
                }
            })?;
            let total_tokens = prompt_len
                .checked_add(slot.generated_tokens.len())
                .ok_or_else(|| WhisperServeBatchError::DecodeFailed {
                    reason: "whisper serve batch token count overflowed".to_string(),
                })?;
            let position = total_tokens.checked_sub(1).ok_or_else(|| {
                WhisperServeBatchError::DecodeFailed {
                    reason: "whisper serve batch position underflowed".to_string(),
                }
            })?;
            let logits = runtime.compute_reused_step_logits(token_id, position)?;
            slot.select_next_token_from_logits(logits)?;
        }
        slot.finish()
    }

    fn decode_failed(reason: String) -> Self::Error {
        WhisperServeBatchError::DecodeFailed { reason }
    }

    fn owner_failed(reason: String) -> Self::Error {
        WhisperServeBatchError::OwnerFailed { reason }
    }

    fn job_execution_context(job: &Self::Job) -> &Arc<crate::RequestExecutionContext> {
        &job.execution_context
    }

    #[cfg(test)]
    fn invalid_test_env(env: &'static str, raw: String, max: usize) -> Self::Error {
        WhisperServeBatchError::InvalidEnv { env, raw, max }
    }

    fn invalid_enabled_batch(max_batch: usize) -> Self::Error {
        WhisperServeBatchError::InvalidEnabledBatch { max_batch }
    }

    fn unsupported_backend(backend: GgmlCpuGraphBackend) -> Self::Error {
        WhisperServeBatchError::UnsupportedBackend { backend }
    }

    fn registry_poisoned() -> Self::Error {
        WhisperServeBatchError::RegistryPoisoned
    }

    fn thread_spawn_failed(reason: String) -> Self::Error {
        WhisperServeBatchError::ThreadSpawnFailed { reason }
    }

    fn queue_full() -> Self::Error {
        WhisperServeBatchError::QueueFull
    }

    fn owner_disconnected() -> Self::Error {
        WhisperServeBatchError::OwnerDisconnected
    }

    fn reply_timed_out() -> Self::Error {
        WhisperServeBatchError::ReplyTimedOut
    }
}

impl WhisperServeBatchJob {
    fn can_batch_with(&self, other: &Self) -> bool {
        self.build_identity == other.build_identity
            && self.runtime_cache_path == other.runtime_cache_path
            && self.backend == other.backend
            && self.uses_scheduler == other.uses_scheduler
            && self.reuse_mode == other.reuse_mode
            && whisper_serve_decode_configs_can_share_fixed_bucket(
                &self.decode_config,
                &other.decode_config,
            )
            && self.execution == other.execution
            && self.decoder_state == other.decoder_state
    }
}

fn whisper_serve_decode_configs_can_share_fixed_bucket(
    left: &Seq2SeqGreedyDecodeConfig,
    right: &Seq2SeqGreedyDecodeConfig,
) -> bool {
    left.initial_prompt_tokens == right.initial_prompt_tokens
        && left.eot_token_id == right.eot_token_id
        && left.vocab_size == right.vocab_size
}

fn whisper_serve_batch_decoder_graph_execution_config(
    attention_heads: usize,
    n_seq: usize,
) -> WhisperDecoderGraphExecutionConfig {
    let batched_slots = n_seq > 1;
    WhisperDecoderGraphExecutionConfig {
        attention_heads,
        // HIP real-server N=2 validation showed dim-3 decoder flash attention
        // can drift to comma-only text; keep batched serve decode on the
        // conservative matmul path until backend parity proves flash safe.
        use_self_flash_attention: !batched_slots && whisper_decoder_self_flash_attention_enabled(),
        use_cross_flash_attention: !batched_slots
            && whisper_decoder_cross_flash_attention_enabled(),
        collect_cross_attention: false,
        layer_norm_epsilon: 1.0e-5_f32,
    }
}

fn whisper_serve_batch_effective_max_batch_for_capabilities(
    configured_max_batch: usize,
    capabilities: crate::ggml_runtime::GgmlBackendCapabilities,
) -> usize {
    if capabilities.is_vulkan() {
        // Real server validation on Windows/RDNA showed whisper N=2 Vulkan
        // serve-batch can select empty text while serial Vulkan remains
        // correct. Keep the owner path but cap Vulkan to serial slots until
        // a backend parity harness proves the multi-sequence graph safe.
        1
    } else {
        configured_max_batch
    }
}

impl WhisperBatchSlot {
    fn new(job: WhisperServeBatchJob) -> Result<Self, WhisperServeBatchError> {
        if job.decode_config.initial_prompt_tokens.is_empty() {
            return Err(WhisperServeBatchError::DecodeFailed {
                reason: "whisper serve batch requires at least one prompt token".to_string(),
            });
        }
        if job.decode_config.vocab_size == 0 {
            return Err(WhisperServeBatchError::DecodeFailed {
                reason: "whisper serve batch requires vocab_size > 0".to_string(),
            });
        }
        if job.decode_config.max_generated_tokens == 0 {
            return Err(WhisperServeBatchError::DecodeFailed {
                reason: "whisper serve batch requires max_generated_tokens > 0".to_string(),
            });
        }
        let stop_token_ids = build_seq2seq_greedy_stop_token_ids(&job.decode_config);
        Ok(Self {
            job,
            stop_token_ids,
            generated_tokens: Vec::new(),
            generated_probabilities: Vec::new(),
            stop_reason: None,
        })
    }

    /// Whether this slot's decode has ended, for any reason. Callers that
    /// need to know WHY (a guard cut vs a real stop token) read
    /// `stop_reason` directly.
    fn is_done(&self) -> bool {
        self.stop_reason.is_some()
    }

    fn select_next_token_from_logits(
        &mut self,
        logits: Vec<f32>,
    ) -> Result<(), WhisperServeBatchError> {
        serve_batch_select_and_apply_greedy_step(
            &self.job.decode_config,
            &mut self.generated_tokens,
            &mut self.generated_probabilities,
            &mut self.stop_reason,
            self.stop_token_ids.as_slice(),
            logits,
        )
        .map_err(map_greedy_error)
    }

    fn finish(self) -> Result<WhisperExecutionOutput, WhisperServeBatchError> {
        let WhisperBatchSlot {
            job,
            generated_tokens,
            generated_probabilities,
            stop_reason,
            ..
        } = self;
        finish_whisper_serve_batch_output(
            &job.tokenizer,
            generated_tokens,
            generated_probabilities,
            job.word_timestamps,
            job.audio_duration_seconds,
            job.carry_prompt_seed_token_ids,
            stop_reason.unwrap_or(Seq2SeqGreedyDecodeStopReason::StopToken),
        )
    }
}

fn finish_whisper_serve_batch_output(
    tokenizer: &WhisperTokenizer,
    generated_tokens: Vec<u32>,
    generated_probabilities: Vec<f32>,
    word_timestamps: bool,
    audio_duration_seconds: f32,
    carry_prompt_seed_token_ids: Option<Vec<u32>>,
    stop_reason: Seq2SeqGreedyDecodeStopReason,
) -> Result<WhisperExecutionOutput, WhisperServeBatchError> {
    let text = tokenizer
        .decode_text_token_ids(&generated_tokens)
        .map_err(|error| WhisperServeBatchError::DecodeFailed {
            reason: error.to_string(),
        })?
        .trim()
        .to_string();
    if text.is_empty() {
        return Err(WhisperServeBatchError::DecodeFailed {
            reason: "tokenizer decode produced empty text".to_string(),
        });
    }
    let words = if word_timestamps {
        seq2seq_word_timestamps_from_generated_tokens(
            &generated_tokens,
            &generated_probabilities,
            0.0,
            audio_duration_seconds.max(0.0),
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
            &|token_ids| tokenizer.decode_text_token_ids(token_ids),
        )
        .map_err(|error| WhisperServeBatchError::DecodeFailed {
            reason: error.to_string(),
        })?
    } else {
        Vec::new()
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
    let carry_prompt_token_ids = carry_prompt_seed_token_ids.and_then(|seed| {
        build_longform_token_history_carry(
            true,
            seed,
            &generated_tokens,
            WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT,
        )
    });
    Ok(WhisperExecutionOutput {
        text,
        segments,
        carry_prompt_token_ids,
        // The batched decode path does not run language ID yet; an `auto` request
        // falls back to the unset-language decode, exactly as before.
        detected_language: None,
        stop_reason,
    })
}

pub(super) fn whisper_serve_batch_decode_config(
    initial_prompt_tokens: Vec<u32>,
    eot_token_id: u32,
    vocab_size: usize,
    max_generated_tokens: usize,
    tokenizer: &WhisperTokenizer,
    phrase_bias: Option<&PhraseBiasConfig>,
) -> Result<Seq2SeqGreedyDecodeConfig, WhisperServeBatchError> {
    let descriptor =
        resolve_builtin_decode_policy(crate::WHISPER_DECODE_POLICY_ID).map_err(|error| {
            WhisperServeBatchError::DecodeFailed {
                reason: error.to_string(),
            }
        })?;
    build_builtin_seq2seq_decode_policy_config(
        descriptor,
        &BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens,
            eot_token_id,
            vocab_size,
            max_generated_tokens,
        },
        tokenizer,
        phrase_bias,
    )
    .map_err(|error| WhisperServeBatchError::DecodeFailed {
        reason: error.to_string(),
    })
}

fn map_decoder_plan_error(
    error: super::ggml_decoder_graph::WhisperDecoderGraphPlanError,
) -> WhisperServeBatchError {
    WhisperServeBatchError::DecodeFailed {
        reason: error.to_string(),
    }
}

fn map_decoder_error(error: WhisperDecoderGraphExecutionError) -> WhisperServeBatchError {
    WhisperServeBatchError::DecodeFailed {
        reason: error.to_string(),
    }
}

fn map_greedy_error(error: Seq2SeqGreedyDecodeError) -> WhisperServeBatchError {
    WhisperServeBatchError::DecodeFailed {
        reason: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::whisper::ggml_executor::WHISPER_ENGLISH_ONLY_MAX_VOCAB_SIZE;
    use crate::models::whisper::tokenizer::WhisperPrefixSpec;

    #[test]
    fn serve_batch_error_classifies_transient_failures() {
        assert_eq!(
            WhisperServeBatchError::QueueFull.unavailable_retryable(),
            Some(true)
        );
        assert_eq!(
            WhisperServeBatchError::OwnerDisconnected.unavailable_retryable(),
            Some(false)
        );
        assert_eq!(
            WhisperServeBatchError::ReplyTimedOut.unavailable_retryable(),
            Some(false)
        );
        assert_eq!(
            WhisperServeBatchError::DecodeFailed {
                reason: "boom".to_string()
            }
            .unavailable_retryable(),
            None
        );
        assert_eq!(
            WhisperServeBatchError::OwnerFailed {
                reason: "boom".to_string()
            }
            .unavailable_retryable(),
            None
        );
    }

    #[test]
    fn serve_batch_engine_key_separates_scheduler_modes_on_the_same_backend() {
        let lane = crate::models::native_execution_services::current_execution_lane_key(
            GgmlCpuGraphBackend::Cpu,
        );
        let direct = WhisperServeBatchEngineKey {
            build_identity: crate::RuntimeBuildIdentity::new(
                "test-pack",
                "whisper:test",
                "adapter=none",
            ),
            lane: lane.clone(),
            resident_self_positions: 448,
            resident_cross_positions: 1500,
            hidden_size: 512,
            uses_scheduler: false,
            max_batch: 2,
        };
        let scheduled = WhisperServeBatchEngineKey {
            uses_scheduler: true,
            ..direct.clone()
        };

        assert_eq!(
            WhisperFamily::engine_key_backend(&direct),
            WhisperFamily::engine_key_backend(&scheduled)
        );
        assert_ne!(direct, scheduled);
    }

    #[test]
    fn serve_batch_engine_key_is_shared_across_output_plans() {
        use crate::ggml_runtime::{
            AutoGpuPolicy, GgmlDecodeOutputContract, GgmlDecodeOutputPlan,
            RequestBackendPreference, ResolvedFamilyRuntimeInput,
        };

        let full = ResolvedFamilyRuntimeInput::resolve_with_output_contract(
            Some(RequestBackendPreference::CpuOnly),
            AutoGpuPolicy::AllBackends,
            GgmlDecodeOutputContract::FullLogits,
        );
        let compact = ResolvedFamilyRuntimeInput::resolve_with_output_contract(
            Some(RequestBackendPreference::CpuOnly),
            AutoGpuPolicy::AllBackends,
            GgmlDecodeOutputContract::NativeFirstMaxTokenOrFullLogits,
        );
        assert_eq!(full.output_plan(), GgmlDecodeOutputPlan::FullLogits);
        assert_eq!(
            compact.output_plan(),
            GgmlDecodeOutputPlan::NativeFirstMaxToken
        );

        let lane = crate::models::native_execution_services::current_execution_lane_key(
            GgmlCpuGraphBackend::Cpu,
        );
        let engine_key = |_plan: GgmlDecodeOutputPlan| WhisperServeBatchEngineKey {
            build_identity: crate::RuntimeBuildIdentity::new(
                "test-pack",
                "whisper:test",
                "adapter=none",
            ),
            lane: lane.clone(),
            resident_self_positions: 448,
            resident_cross_positions: 1500,
            hidden_size: 512,
            uses_scheduler: false,
            max_batch: 2,
        };
        assert_eq!(
            engine_key(full.output_plan()),
            engine_key(compact.output_plan()),
            "whisper serve-batch engine must serve both plans with one retained graph"
        );
    }
    use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;
    use crate::models::serve_batch_env::OPENASR_SERVE_BATCH_ENV;
    use crate::models::whisper::runtime_contract::validate_whisper_execution_metadata;
    use crate::models::whisper::tokenizer::{
        TOKENIZER_GGML_EOT_TOKEN_ID_KEY, TOKENIZER_GGML_MERGES_KEY, TOKENIZER_GGML_MODEL_KEY,
        TOKENIZER_GGML_MODEL_VALUE_GPT2, TOKENIZER_GGML_NO_TIMESTAMPS_TOKEN_ID_KEY,
        TOKENIZER_GGML_SOT_TOKEN_ID_KEY, TOKENIZER_GGML_SPECIAL_TOKEN_IDS_KEY,
        TOKENIZER_GGML_TOKENS_KEY, TOKENIZER_GGML_TRANSCRIBE_TOKEN_ID_KEY,
    };
    use crate::{
        GgufMetadata, GgufMetadataValue, GgufRuntimeSourcePreflight,
        read_gguf_metadata_from_runtime_source, read_gguf_tensor_index_from_runtime_source,
        validate_ggml_runtime_source_path,
    };
    use std::ffi::OsString;
    use std::path::Path;
    use std::sync::mpsc;
    use std::time::Duration;

    const WHISPER_SERVE_BATCH_REAL_PACK_ENV: &str = "OPENASR_WHISPER_SERVE_BATCH_REAL_PACK";

    fn with_serve_batch_env<T>(value: Option<&str>, run: impl FnOnce() -> T) -> T {
        crate::test_process_env::with_test_process_env(
            [(OPENASR_SERVE_BATCH_ENV, value.map(OsString::from))],
            run,
        )
    }

    /// Test-only serial decode driver: runs the family serial path against a
    /// caller-owned lazily-built serial runtime (mirrors the previous
    /// `WhisperOwnerThreadState::decode_serial_job`, which is now the generic
    /// owner's private hook).
    fn decode_serial_job(
        serial_runtime: &mut Option<WhisperServeDecoderRuntime>,
        job: WhisperServeBatchJob,
    ) -> Result<WhisperExecutionOutput, WhisperServeBatchError> {
        WhisperFamily::decode_serial(serial_runtime, job)
    }

    fn with_whisper_decoder_flash_disabled_for_test<T>(run: impl FnOnce() -> T) -> T {
        crate::test_process_env::with_test_process_env(
            [
                (
                    "OPENASR_WHISPER_GGML_DISABLE_DECODER_SELF_FLASH_ATTN",
                    Some(OsString::from("1")),
                ),
                (
                    "OPENASR_WHISPER_GGML_DISABLE_DECODER_CROSS_FLASH_ATTN",
                    Some(OsString::from("1")),
                ),
            ],
            run,
        )
    }

    fn read_runtime_source_preflight(runtime_path: &Path) -> GgufRuntimeSourcePreflight {
        let runtime_source =
            validate_ggml_runtime_source_path(runtime_path).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        GgufRuntimeSourcePreflight {
            runtime_source,
            metadata: Arc::new(metadata),
            tensor_index: Arc::new(tensor_index),
        }
    }

    fn real_pack_tensor_binding_context(
        execution: &WhisperGgmlExecutionMetadata,
    ) -> super::super::ggml_tensor_binding::WhisperGgufTensorBindingContext {
        super::super::ggml_tensor_binding::WhisperGgufTensorBindingContext {
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
        }
    }

    fn load_real_pack_decoder_components(
        preflight: &GgufRuntimeSourcePreflight,
    ) -> (
        WhisperGgmlExecutionMetadata,
        WhisperTokenizer,
        WhisperDecoderWeightSeam,
    ) {
        let execution =
            validate_whisper_execution_metadata(&preflight.metadata).expect("execution metadata");
        let tokenizer = WhisperTokenizer::from_gguf_metadata(&preflight.metadata)
            .expect("load whisper tokenizer");
        let tensor_bindings = super::super::ggml_tensor_binding::bind_whisper_gguf_tensors(
            &real_pack_tensor_binding_context(&execution),
            &preflight.tensor_index,
        )
        .expect("bind whisper tensors");
        let reader = build_runtime_tensor_reader_from_preflight(preflight).expect("tensor reader");
        let decoder_weights =
            super::super::ggml_executor::build_decoder_weight_seam(&reader, &tensor_bindings)
                .expect("decoder weights");
        (execution, tokenizer, decoder_weights)
    }

    fn sample_encoder_hidden(
        execution: &WhisperGgmlExecutionMetadata,
        phase: f32,
    ) -> (usize, usize, Vec<f32>) {
        let frame_count = execution.encoder_context_length;
        let hidden_size = execution.encoder_hidden_size;
        let mut rows = Vec::with_capacity(frame_count * hidden_size);
        for frame_idx in 0..frame_count {
            for hidden_idx in 0..hidden_size {
                rows.push(
                    (((frame_idx * hidden_size + hidden_idx) as f32 * 0.03125) + phase).sin(),
                );
            }
        }
        (frame_count, hidden_size, rows)
    }

    fn decoder_state_fixture(
        execution: &WhisperGgmlExecutionMetadata,
        encoder_frames: usize,
    ) -> Seq2SeqDecoderState {
        use crate::models::seq2seq_decoder_state::Seq2SeqStateAxis;

        let cross_resident = persistent_cross_attention_layer_stride_frames(encoder_frames);
        let self_positions = execution.max_target_positions - 1;
        Seq2SeqDecoderState {
            self_attention: Seq2SeqStateAxis {
                logical_positions: self_positions,
                resident_positions: self_positions,
                hard_position_cap: execution.max_target_positions,
            },
            cross_attention: Seq2SeqStateAxis {
                logical_positions: encoder_frames,
                resident_positions: cross_resident,
                hard_position_cap: cross_resident,
            },
        }
    }

    /// Structural proof that `WhisperServeBatchJob::execution_context` is
    /// required, not optional: this only compiles because the field's type
    /// is the concrete `Arc<RequestExecutionContext>`. Never called; exists
    /// purely so `cargo check`/`clippy` re-verify the contract on every build.
    #[allow(dead_code)]
    fn require_concrete_execution_context(_: Arc<crate::RequestExecutionContext>) {}

    #[allow(dead_code)]
    fn assert_whisper_serve_batch_job_requires_execution_context(job: WhisperServeBatchJob) {
        let WhisperServeBatchJob {
            execution_context, ..
        } = job;
        require_concrete_execution_context(execution_context);
    }

    fn real_pack_batch_job(
        runtime_path: &Path,
        backend: GgmlCpuGraphBackend,
        uses_scheduler: bool,
        execution: WhisperGgmlExecutionMetadata,
        decoder_weights: WhisperDecoderWeightSeam,
        tokenizer: WhisperTokenizer,
        encoder_phase: f32,
    ) -> WhisperServeBatchJob {
        real_pack_batch_job_with_max_generated_tokens(
            runtime_path,
            backend,
            uses_scheduler,
            execution,
            decoder_weights,
            tokenizer,
            encoder_phase,
            4,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn real_pack_batch_job_with_max_generated_tokens(
        runtime_path: &Path,
        backend: GgmlCpuGraphBackend,
        uses_scheduler: bool,
        execution: WhisperGgmlExecutionMetadata,
        decoder_weights: WhisperDecoderWeightSeam,
        tokenizer: WhisperTokenizer,
        encoder_phase: f32,
        max_generated_tokens: usize,
    ) -> WhisperServeBatchJob {
        let decoder_start_token_id = tokenizer
            .start_of_transcript_token_id()
            .unwrap_or(execution.decoder_start_token_id);
        // Mirror the production multilingual decision (vocab size), not token
        // presence: `.en` packs also carry <|translate|>/<|xx|> tokens, so the
        // vocab-size gate is the single source of truth.
        let is_multilingual = execution.vocab_size > WHISPER_ENGLISH_ONLY_MAX_VOCAB_SIZE;
        let initial_prompt_tokens = tokenizer
            .decoder_prefix(
                decoder_start_token_id,
                &WhisperPrefixSpec::transcribe(is_multilingual),
            )
            .expect("default prefix");
        let eot_token_id = tokenizer
            .end_of_text_token_id()
            .unwrap_or(execution.eos_token_id);
        let decode_config = whisper_serve_batch_decode_config(
            initial_prompt_tokens,
            eot_token_id,
            execution.vocab_size,
            max_generated_tokens,
            &tokenizer,
            None,
        )
        .expect("decode config");
        let (encoder_frames, encoder_hidden_size, encoder_hidden_f32) =
            sample_encoder_hidden(&execution, encoder_phase);
        let decoder_state = decoder_state_fixture(&execution, encoder_frames);
        let runtime_preflight = read_runtime_source_preflight(runtime_path);
        let runtime_source = runtime_preflight.runtime_source.clone();
        WhisperServeBatchJob {
            runtime_cache_path: runtime_path.to_path_buf(),
            build_identity: crate::RuntimeBuildIdentity::resolve_for_request(
                None,
                "whisper:test",
                "adapter=none",
                runtime_source.content_id(),
            ),
            runtime_preflight,
            backend,
            uses_scheduler,
            reuse_mode: GgmlDecodeReuseMode::FreshGraph,
            execution,
            decoder_weights,
            tokenizer,
            decoder_state,
            encoder_frames,
            encoder_hidden_size,
            encoder_hidden_f32,
            decode_config,
            word_timestamps: false,
            audio_duration_seconds: 1.0,
            carry_prompt_seed_token_ids: None,
            execution_context: Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        }
    }

    fn whisper_execution_and_tokenizer_fixture() -> (WhisperGgmlExecutionMetadata, WhisperTokenizer)
    {
        let mut values = std::collections::BTreeMap::new();
        values.insert(
            "general.architecture".to_string(),
            GgufMetadataValue::String("whisper".to_string()),
        );
        values.insert(
            "whisper.encoder.block_count".to_string(),
            GgufMetadataValue::U32(1),
        );
        values.insert(
            "whisper.encoder.embedding_length".to_string(),
            GgufMetadataValue::U32(4),
        );
        values.insert(
            "whisper.encoder.attention.head_count".to_string(),
            GgufMetadataValue::U32(2),
        );
        values.insert(
            "whisper.encoder.context_length".to_string(),
            GgufMetadataValue::U32(1500),
        );
        values.insert(
            "whisper.encoder.mels_count".to_string(),
            GgufMetadataValue::U32(80),
        );
        values.insert(
            "whisper.decoder.block_count".to_string(),
            GgufMetadataValue::U32(1),
        );
        values.insert(
            "whisper.decoder.embedding_length".to_string(),
            GgufMetadataValue::U32(4),
        );
        values.insert(
            "whisper.decoder.attention.head_count".to_string(),
            GgufMetadataValue::U32(2),
        );
        values.insert(
            "whisper.decoder.context_length".to_string(),
            GgufMetadataValue::U32(32),
        );
        values.insert("whisper.vocab_size".to_string(), GgufMetadataValue::U32(14));
        values.insert(
            TOKENIZER_GGML_MODEL_KEY.to_string(),
            GgufMetadataValue::String(TOKENIZER_GGML_MODEL_VALUE_GPT2.to_string()),
        );
        values.insert(
            TOKENIZER_GGML_TOKENS_KEY.to_string(),
            GgufMetadataValue::StringArray(vec![
                "\u{0120}".to_string(),
                "h".to_string(),
                "e".to_string(),
                "l".to_string(),
                "o".to_string(),
                "w".to_string(),
                "r".to_string(),
                "d".to_string(),
                "<|endoftext|>".to_string(),
                "<|startoftranscript|>".to_string(),
                "<|transcribe|>".to_string(),
                "<|notimestamps|>".to_string(),
                "<|startofprev|>".to_string(),
                "\u{010A}".to_string(),
            ]),
        );
        values.insert(
            TOKENIZER_GGML_MERGES_KEY.to_string(),
            GgufMetadataValue::StringArray(vec!["x y".to_string()]),
        );
        values.insert(
            TOKENIZER_GGML_SPECIAL_TOKEN_IDS_KEY.to_string(),
            GgufMetadataValue::U32Array(vec![8, 9, 10, 11, 12]),
        );
        values.insert(
            TOKENIZER_GGML_SOT_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(9),
        );
        values.insert(
            TOKENIZER_GGML_EOT_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(8),
        );
        values.insert(
            TOKENIZER_GGML_TRANSCRIBE_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(10),
        );
        values.insert(
            TOKENIZER_GGML_NO_TIMESTAMPS_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(11),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        let execution =
            validate_whisper_execution_metadata(&metadata).expect("validate whisper metadata");
        let tokenizer = WhisperTokenizer::from_gguf_metadata(&metadata).expect("load tokenizer");
        (execution, tokenizer)
    }

    #[test]
    fn whisper_serve_batch_env_defaults_off() {
        with_serve_batch_env(None, || {
            assert!(
                ServeBatchConfig::from_test_env::<WhisperFamily>()
                    .unwrap()
                    .is_none()
            );
        });
    }

    #[test]
    fn whisper_serve_batch_env_one_keeps_default_path() {
        with_serve_batch_env(Some("1"), || {
            assert!(
                ServeBatchConfig::from_test_env::<WhisperFamily>()
                    .unwrap()
                    .is_none()
            );
        });
    }

    #[test]
    fn whisper_serve_batch_env_rejects_above_limit() {
        with_serve_batch_env(Some("9"), || {
            let error = ServeBatchConfig::from_test_env::<WhisperFamily>().unwrap_err();
            assert!(error.to_string().contains("0..=8"));
        });
    }

    #[test]
    fn whisper_serve_batch_decode_config_uses_whisper_policy() {
        let (execution, tokenizer) = whisper_execution_and_tokenizer_fixture();
        let config = whisper_serve_batch_decode_config(
            vec![9, 10, 11],
            execution.eos_token_id,
            execution.vocab_size,
            4,
            &tokenizer,
            None,
        )
        .expect("decode config");

        assert_eq!(config.eot_token_id, 8);
        assert_eq!(config.suppress_first_step_token_ids, vec![8]);
        assert_eq!(config.suppress_token_ids, vec![9, 10, 11, 12]);
    }

    #[test]
    fn whisper_serve_batch_decoder_config_keeps_serial_flash_policy() {
        let config = whisper_serve_batch_decoder_graph_execution_config(2, 1);

        assert_eq!(config.attention_heads, 2);
        assert_eq!(
            config.use_self_flash_attention,
            whisper_decoder_self_flash_attention_enabled()
        );
        assert_eq!(
            config.use_cross_flash_attention,
            whisper_decoder_cross_flash_attention_enabled()
        );
        assert!(!config.collect_cross_attention);
    }

    #[test]
    fn whisper_serve_batch_decoder_config_disables_flash_for_batched_slots() {
        let config = whisper_serve_batch_decoder_graph_execution_config(2, 2);

        assert_eq!(config.attention_heads, 2);
        assert!(!config.use_self_flash_attention);
        assert!(!config.use_cross_flash_attention);
        assert!(!config.collect_cross_attention);
    }

    #[test]
    fn whisper_serve_batch_effective_max_batch_caps_vulkan_to_serial() {
        assert_eq!(
            whisper_serve_batch_effective_max_batch_for_capabilities(
                8,
                crate::ggml_runtime::GgmlBackendCapabilities::from_backend_for_test(
                    GgmlCpuGraphBackend::Gpu,
                    "Vulkan0",
                ),
            ),
            1
        );
        assert_eq!(
            whisper_serve_batch_effective_max_batch_for_capabilities(
                2,
                crate::ggml_runtime::GgmlBackendCapabilities::from_backend_for_test(
                    GgmlCpuGraphBackend::Gpu,
                    "ggml-vulkan",
                ),
            ),
            1
        );
    }

    #[test]
    fn whisper_serve_batch_effective_max_batch_keeps_other_backends() {
        for backend_name in ["HIP0", "ROCm0", "CUDA0", "Metal", "CPU"] {
            assert_eq!(
                whisper_serve_batch_effective_max_batch_for_capabilities(
                    8,
                    crate::ggml_runtime::GgmlBackendCapabilities::from_backend_for_test(
                        if matches!(backend_name, "Metal") {
                            GgmlCpuGraphBackend::Metal
                        } else if matches!(backend_name, "CPU") {
                            GgmlCpuGraphBackend::Cpu
                        } else {
                            GgmlCpuGraphBackend::Gpu
                        },
                        backend_name,
                    ),
                ),
                8,
                "backend_name={backend_name}"
            );
        }
    }

    #[test]
    fn whisper_serve_batch_finish_can_emit_approx_word_timestamps() {
        let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
        let generated_tokens = vec![1, 2, 3, 3, 4];

        let without_words = finish_whisper_serve_batch_output(
            &tokenizer,
            generated_tokens.clone(),
            Vec::new(),
            false,
            2.0,
            None,
            Seq2SeqGreedyDecodeStopReason::StopToken,
        )
        .expect("finish without words");
        assert!(without_words.segments.is_empty());

        let with_words = finish_whisper_serve_batch_output(
            &tokenizer,
            generated_tokens,
            vec![0.5; 5],
            true,
            2.0,
            None,
            Seq2SeqGreedyDecodeStopReason::StopToken,
        )
        .expect("finish with words");
        assert_eq!(with_words.text, "hello");
        assert_eq!(with_words.segments.len(), 1);
        assert_eq!(with_words.segments[0].text, "hello");
        assert_eq!(with_words.segments[0].words.len(), 1);
        assert_eq!(with_words.segments[0].words[0].word, "hello");
        assert_eq!(with_words.segments[0].words[0].start, 0.0);
        assert_eq!(with_words.segments[0].words[0].end, 2.0);
    }

    #[test]
    fn whisper_serve_batch_finish_can_emit_longform_carry_prompt_tokens() {
        let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
        let generated_tokens = vec![1, 2, 3, 3, 4];
        let output = finish_whisper_serve_batch_output(
            &tokenizer,
            generated_tokens.clone(),
            Vec::new(),
            false,
            2.0,
            Some((1..=40).collect()),
            Seq2SeqGreedyDecodeStopReason::StopToken,
        )
        .expect("finish with carry tokens");

        let carry = output
            .carry_prompt_token_ids
            .expect("longform carry prompt tokens");
        let mut expected = (14..=40).collect::<Vec<_>>();
        expected.extend(generated_tokens);
        assert_eq!(carry, expected);
    }

    #[test]
    fn whisper_serve_batch_bucket_compatibility_allows_mixed_token_caps() {
        let (execution, tokenizer) = whisper_execution_and_tokenizer_fixture();
        let short_cap = whisper_serve_batch_decode_config(
            vec![9, 10, 11],
            execution.eos_token_id,
            execution.vocab_size,
            1,
            &tokenizer,
            None,
        )
        .expect("short decode config");
        let long_cap = whisper_serve_batch_decode_config(
            vec![9, 10, 11],
            execution.eos_token_id,
            execution.vocab_size,
            4,
            &tokenizer,
            None,
        )
        .expect("long decode config");

        assert!(whisper_serve_decode_configs_can_share_fixed_bucket(
            &short_cap, &long_cap
        ));

        let different_prompt = whisper_serve_batch_decode_config(
            vec![9, 10],
            execution.eos_token_id,
            execution.vocab_size,
            4,
            &tokenizer,
            None,
        )
        .expect("different prompt decode config");
        assert!(!whisper_serve_decode_configs_can_share_fixed_bucket(
            &short_cap,
            &different_prompt
        ));
    }

    #[test]
    #[ignore = "manual real-pack: set OPENASR_WHISPER_SERVE_BATCH_REAL_PACK to a MULTILINGUAL whisper .oasr"]
    fn whisper_multilingual_pack_resolves_translate_and_language_tokens() {
        let runtime_path = std::env::var_os(WHISPER_SERVE_BATCH_REAL_PACK_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!(
                    "{WHISPER_SERVE_BATCH_REAL_PACK_ENV} must point to a whisper .oasr model pack"
                )
            });
        let preflight = read_runtime_source_preflight(&runtime_path);
        let (execution, tokenizer, _decoder_weights) =
            load_real_pack_decoder_components(&preflight);
        assert!(
            execution.vocab_size > WHISPER_ENGLISH_ONLY_MAX_VOCAB_SIZE,
            "expected a MULTILINGUAL whisper pack (vocab {} <= {})",
            execution.vocab_size,
            WHISPER_ENGLISH_ONLY_MAX_VOCAB_SIZE
        );
        let translate = tokenizer
            .translate_token_id()
            .expect("<|translate|> must resolve on a real multilingual pack");
        let transcribe = tokenizer
            .transcribe_token_id()
            .expect("<|transcribe|> must resolve");
        // Canonical Whisper layout invariant: transcribe == translate + 1
        // (holds on tiny.en/small/large-v3); strengthens the .is_some() check.
        assert_eq!(
            transcribe,
            translate + 1,
            "transcribe should be translate + 1"
        );
        let fr = tokenizer
            .token_id_by_content("<|fr|>")
            .expect("<|fr|> language token must resolve");
        let sot = tokenizer.start_of_transcript_token_id().expect("sot");
        let notimestamps = tokenizer.no_timestamps_token_id().expect("notimestamps");
        // The explicit non-English translate prefix resolves to real pack ids.
        let prefix = tokenizer
            .decoder_prefix(
                sot,
                &WhisperPrefixSpec {
                    language: Some("fr"),
                    task: crate::TranscriptionTask::Translate,
                    is_multilingual: true,
                },
            )
            .expect("prefix");
        assert_eq!(prefix, vec![sot, fr, translate, notimestamps]);
    }

    #[test]
    #[ignore = "manual real-pack backend harness: set OPENASR_WHISPER_SERVE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=cpu, hip, or vulkan"]
    fn whisper_owner_thread_decodes_static_real_pack_selected_backend_batch() {
        let runtime_path = std::env::var_os(WHISPER_SERVE_BATCH_REAL_PACK_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!(
                    "{WHISPER_SERVE_BATCH_REAL_PACK_ENV} must point to a whisper .oasr model pack"
                )
            });
        let preflight = read_runtime_source_preflight(&runtime_path);
        let (execution, tokenizer, decoder_weights) = load_real_pack_decoder_components(&preflight);
        let runtime_config = whisper_decoder_graph_config(
            crate::ggml_runtime::GgmlCpuGraphConfig::runtime_default().backend,
            WhisperDecoderPlacementPolicy::resolve(),
        );
        assert!(
            runtime_config.backend == GgmlCpuGraphBackend::Cpu || !runtime_config.use_scheduler,
            "whisper static batch fixture validates direct graph execution, got scheduler-backed {:?}",
            runtime_config.backend
        );

        let phases = [0.0_f32, 0.25, 0.5];
        let expected_texts = with_whisper_decoder_flash_disabled_for_test(|| {
            phases
                .iter()
                .map(|&encoder_phase| {
                    let mut serial_runtime: Option<WhisperServeDecoderRuntime> = None;
                    decode_serial_job(
                        &mut serial_runtime,
                        real_pack_batch_job(
                            &runtime_path,
                            runtime_config.backend,
                            runtime_config.use_scheduler,
                            execution.clone(),
                            decoder_weights.clone(),
                            tokenizer.clone(),
                            encoder_phase,
                        ),
                    )
                    .expect("serial decode output")
                    .text
                })
                .collect::<Vec<_>>()
        });
        let envelope_for_phase = |encoder_phase: f32| {
            let (reply, reply_rx) = mpsc::channel();
            let job = real_pack_batch_job(
                &runtime_path,
                runtime_config.backend,
                runtime_config.use_scheduler,
                execution.clone(),
                decoder_weights.clone(),
                tokenizer.clone(),
                encoder_phase,
            );
            let context = Arc::clone(&job.execution_context);
            (
                WhisperServeBatchEnvelope {
                    job,
                    context,
                    native_execution_context: None,
                    reply,
                },
                reply_rx,
            )
        };
        let (slot_0, slot_0_rx) = envelope_for_phase(phases[0]);
        let (slot_1, slot_1_rx) = envelope_for_phase(phases[1]);
        let (slot_2, slot_2_rx) = envelope_for_phase(phases[2]);
        let batch = vec![slot_0, slot_1, slot_2];
        let (_queue_tx, queue_rx) = mpsc::sync_channel(0);
        let mut state = OwnerThreadState::<WhisperFamily>::new();
        let deferred = state.run_batch(batch, &queue_rx, 4, false);
        assert!(deferred.is_empty());

        let outputs = [slot_0_rx, slot_1_rx, slot_2_rx]
            .into_iter()
            .map(|reply_rx| {
                reply_rx
                    .recv_timeout(Duration::from_secs(30))
                    .expect("reply sent")
                    .expect("decode output")
            })
            .collect::<Vec<_>>();
        assert_eq!(outputs.len(), 3);
        assert_eq!(
            outputs
                .iter()
                .map(|output| output.text.clone())
                .collect::<Vec<_>>(),
            expected_texts
        );
        assert!(outputs.iter().all(|output| output.segments.is_empty()));
        assert!(
            outputs
                .iter()
                .all(|output| output.carry_prompt_token_ids.is_none())
        );
    }

    #[test]
    #[ignore = "manual real-pack backend harness: set OPENASR_WHISPER_SERVE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=cpu, hip, or vulkan"]
    fn whisper_owner_thread_refills_free_static_real_pack_selected_backend_batch() {
        let runtime_path = std::env::var_os(WHISPER_SERVE_BATCH_REAL_PACK_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!(
                    "{WHISPER_SERVE_BATCH_REAL_PACK_ENV} must point to a whisper .oasr model pack"
                )
            });
        let preflight = read_runtime_source_preflight(&runtime_path);
        let (execution, tokenizer, decoder_weights) = load_real_pack_decoder_components(&preflight);
        let runtime_config = whisper_decoder_graph_config(
            crate::ggml_runtime::GgmlCpuGraphConfig::runtime_default().backend,
            WhisperDecoderPlacementPolicy::resolve(),
        );
        assert!(
            runtime_config.backend == GgmlCpuGraphBackend::Cpu || !runtime_config.use_scheduler,
            "whisper refill fixture validates direct graph execution, got scheduler-backed {:?}",
            runtime_config.backend
        );

        let envelope = |encoder_phase: f32, max_generated_tokens: usize| {
            let job = real_pack_batch_job_with_max_generated_tokens(
                &runtime_path,
                runtime_config.backend,
                runtime_config.use_scheduler,
                execution.clone(),
                decoder_weights.clone(),
                tokenizer.clone(),
                encoder_phase,
                max_generated_tokens,
            );
            let (reply, reply_rx) = mpsc::channel();
            let context = Arc::clone(&job.execution_context);
            (
                WhisperServeBatchEnvelope {
                    job,
                    context,
                    native_execution_context: None,
                    reply,
                },
                reply_rx,
            )
        };

        let (initial_fast, initial_fast_rx) = envelope(0.0, 1);
        let (initial_long, initial_long_rx) = envelope(0.25, 3);
        let (queued_refill, queued_refill_rx) = envelope(0.5, 1);
        let (queued_tx, queued_rx) = mpsc::sync_channel(1);
        queued_tx.send(queued_refill).expect("queue refill job");

        let mut state = OwnerThreadState::<WhisperFamily>::new();
        let deferred = state.run_batch(vec![initial_fast, initial_long], &queued_rx, 2, false);
        assert!(deferred.is_empty());

        let outputs = [initial_fast_rx, initial_long_rx, queued_refill_rx]
            .into_iter()
            .map(|reply_rx| {
                reply_rx
                    .recv_timeout(Duration::from_secs(30))
                    .expect("reply sent")
                    .expect("decode output")
            })
            .collect::<Vec<_>>();
        assert_eq!(outputs.len(), 3);
        assert!(outputs.iter().all(|output| output.segments.is_empty()));
        assert!(
            outputs
                .iter()
                .all(|output| output.carry_prompt_token_ids.is_none())
        );
    }

    #[test]
    #[ignore = "manual real-pack backend harness: set OPENASR_WHISPER_SERVE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=cpu, hip, or vulkan"]
    fn whisper_owner_thread_rebuckets_full_static_real_pack_selected_backend_batch() {
        let runtime_path = std::env::var_os(WHISPER_SERVE_BATCH_REAL_PACK_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!(
                    "{WHISPER_SERVE_BATCH_REAL_PACK_ENV} must point to a whisper .oasr model pack"
                )
            });
        let preflight = read_runtime_source_preflight(&runtime_path);
        let (execution, tokenizer, decoder_weights) = load_real_pack_decoder_components(&preflight);
        let runtime_config = whisper_decoder_graph_config(
            crate::ggml_runtime::GgmlCpuGraphConfig::runtime_default().backend,
            WhisperDecoderPlacementPolicy::resolve(),
        );
        assert!(
            runtime_config.backend == GgmlCpuGraphBackend::Cpu || !runtime_config.use_scheduler,
            "whisper rebucket fixture validates direct graph execution, got scheduler-backed {:?}",
            runtime_config.backend
        );

        let envelope = |encoder_phase: f32, max_generated_tokens: usize| {
            let job = real_pack_batch_job_with_max_generated_tokens(
                &runtime_path,
                runtime_config.backend,
                runtime_config.use_scheduler,
                execution.clone(),
                decoder_weights.clone(),
                tokenizer.clone(),
                encoder_phase,
                max_generated_tokens,
            );
            let (reply, reply_rx) = mpsc::channel();
            let context = Arc::clone(&job.execution_context);
            (
                WhisperServeBatchEnvelope {
                    job,
                    context,
                    native_execution_context: None,
                    reply,
                },
                reply_rx,
            )
        };

        let (initial_long_a, initial_long_a_rx) = envelope(0.0, 3);
        let (initial_long_b, initial_long_b_rx) = envelope(0.25, 3);
        let (queued_refill_a, queued_refill_a_rx) = envelope(0.5, 1);
        let (queued_refill_b, queued_refill_b_rx) = envelope(0.75, 1);
        let (queued_tx, queued_rx) = mpsc::sync_channel(2);
        queued_tx.send(queued_refill_a).expect("queue refill a");
        queued_tx.send(queued_refill_b).expect("queue refill b");

        let mut state = OwnerThreadState::<WhisperFamily>::new();
        let deferred = state.run_batch(vec![initial_long_a, initial_long_b], &queued_rx, 4, false);
        assert!(deferred.is_empty());

        let outputs = [
            initial_long_a_rx,
            initial_long_b_rx,
            queued_refill_a_rx,
            queued_refill_b_rx,
        ]
        .into_iter()
        .map(|reply_rx| {
            reply_rx
                .recv_timeout(Duration::from_secs(30))
                .expect("reply sent")
                .expect("decode output")
        })
        .collect::<Vec<_>>();
        assert_eq!(outputs.len(), 4);
        assert!(outputs.iter().all(|output| output.segments.is_empty()));
        assert!(
            outputs
                .iter()
                .all(|output| output.carry_prompt_token_ids.is_none())
        );
        assert!(state.batched_runtimes.contains_key(&2));
        assert!(state.batched_runtimes.contains_key(&4));
    }

    #[test]
    #[ignore = "manual real-pack backend harness: set OPENASR_WHISPER_SERVE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=cpu, hip, or vulkan"]
    fn whisper_owner_thread_shrinks_tail_static_real_pack_selected_backend_batch() {
        let runtime_path = std::env::var_os(WHISPER_SERVE_BATCH_REAL_PACK_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!(
                    "{WHISPER_SERVE_BATCH_REAL_PACK_ENV} must point to a whisper .oasr model pack"
                )
            });
        let preflight = read_runtime_source_preflight(&runtime_path);
        let (execution, tokenizer, decoder_weights) = load_real_pack_decoder_components(&preflight);
        let runtime_config = whisper_decoder_graph_config(
            crate::ggml_runtime::GgmlCpuGraphConfig::runtime_default().backend,
            WhisperDecoderPlacementPolicy::resolve(),
        );
        assert!(
            runtime_config.backend == GgmlCpuGraphBackend::Cpu || !runtime_config.use_scheduler,
            "whisper shrink fixture validates direct graph execution, got scheduler-backed {:?}",
            runtime_config.backend
        );

        let envelope = |encoder_phase: f32, max_generated_tokens: usize| {
            let job = real_pack_batch_job_with_max_generated_tokens(
                &runtime_path,
                runtime_config.backend,
                runtime_config.use_scheduler,
                execution.clone(),
                decoder_weights.clone(),
                tokenizer.clone(),
                encoder_phase,
                max_generated_tokens,
            );
            let (reply, reply_rx) = mpsc::channel();
            let context = Arc::clone(&job.execution_context);
            (
                WhisperServeBatchEnvelope {
                    job,
                    context,
                    native_execution_context: None,
                    reply,
                },
                reply_rx,
            )
        };

        let (initial_fast_a, initial_fast_a_rx) = envelope(0.0, 1);
        let (initial_fast_b, initial_fast_b_rx) = envelope(0.25, 1);
        let (initial_long, initial_long_rx) = envelope(0.5, 3);
        let (_queued_tx, queued_rx) = mpsc::sync_channel(1);

        let mut state = OwnerThreadState::<WhisperFamily>::new();
        let deferred = state.run_batch(
            vec![initial_fast_a, initial_fast_b, initial_long],
            &queued_rx,
            4,
            false,
        );
        assert!(deferred.is_empty());

        let outputs = [initial_fast_a_rx, initial_fast_b_rx, initial_long_rx]
            .into_iter()
            .map(|reply_rx| {
                reply_rx
                    .recv_timeout(Duration::from_secs(30))
                    .expect("reply sent")
                    .expect("decode output")
            })
            .collect::<Vec<_>>();
        assert_eq!(outputs.len(), 3);
        assert!(outputs.iter().all(|output| output.segments.is_empty()));
        assert!(
            outputs
                .iter()
                .all(|output| output.carry_prompt_token_ids.is_none())
        );
        assert!(state.batched_runtimes.contains_key(&4));
        assert!(state.batched_runtimes.contains_key(&2));
    }
}
