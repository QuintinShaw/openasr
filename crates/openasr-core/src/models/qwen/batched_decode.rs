use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use thiserror::Error;

#[cfg(test)]
use super::graph_config::qwen_runtime_graph_config;
use super::kv_cache::{Qwen3AsrHostKvCacheOwner, Qwen3AsrHostKvMode, Qwen3AsrKvCacheCapacity};
use super::llm_prefill::Qwen3AsrLlmPrefillInput;
#[cfg(test)]
use super::llm_transformer::Qwen3AsrLlmLayerAttentionProjection;
use super::llm_transformer::{
    Qwen3AsrLlmWholeDecoderGraphExecutor, QwenQkvExecutionMode, QwenWholeDecoderPlan,
    resolve_qwen_family_production_kv_cache_policy,
};
use super::logits_head::{Qwen3AsrLlmLogitsHead, Qwen3AsrLlmLogitsHeadRuntime};
use super::prompt_embedding::Qwen3AsrPromptTokenInput;
use super::runtime_contract::Qwen3AsrExecutionMetadata;
use super::tokenizer::Qwen3AsrTokenizer;
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlDecodeReuseMode, GgmlNativeGqaCapability, ResolvedFamilyRuntimeInput,
};
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicySeq2SeqTextPostprocessKind, apply_seq2seq_text_postprocess,
};
use crate::models::mapped_token_embedding::MappedTokenEmbeddingTable;
use crate::models::native_execution_services::{
    current_execution_lane, current_native_execution_context, current_runtime_receipts,
    install_native_execution_context,
};
use crate::models::prepared_runtime_cache::PreparedRuntimeHandle;
use crate::models::runtime_prepared_registry::BuiltinPreparedRuntime;
use crate::models::runtime_receipts::{
    RuntimeOwnerGuard, RuntimeReceiptCollector, RuntimeResourceGuard,
};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeConfig, Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeStopReason,
    build_seq2seq_greedy_stop_token_ids,
};
use crate::models::seq2seq_serve_batch::ServeBatchActiveRegistry;
use crate::models::seq2seq_word_timestamps::seq2seq_word_timestamps_from_generated_tokens;
use crate::models::serve_batch_env::{
    OwnerAliveGuard, SERVE_BATCH_COLLECT_WINDOW, ServeBatchPolicy, serve_batch_bucket_width,
    serve_batch_compact_active_slots, serve_batch_drain_compatible_batch,
    serve_batch_estimate_llm_kv_slot_bytes, serve_batch_owner_alive,
    serve_batch_select_and_apply_greedy_step, serve_batch_submit_with_timeout,
    serve_batch_trace_enabled, serve_batch_vram_capped_max_batch,
};
use crate::{GgmlAsrExecutionResult, Segment, Transcription};

const QWEN_SERVE_BATCH_MAX_BATCH_LIMIT: usize = 8;
const QWEN_SERVE_BATCH_SEND_TIMEOUT: Duration = Duration::from_secs(1);
const QWEN_SERVE_BATCH_REPLY_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const QWEN_ROPE_THETA: f32 = 1_000_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Qwen3AsrServeBatchConfig {
    pub max_batch: usize,
    queue_capacity: usize,
    collect_window: Duration,
    send_timeout: Duration,
    reply_timeout: Duration,
    trace_batches: bool,
}

#[derive(Debug, Clone)]
pub(super) enum Qwen3AsrServeBatchPromptInput {
    Host(Qwen3AsrLlmPrefillInput),
    TokenIds(Qwen3AsrPromptTokenInput),
}

impl Qwen3AsrServeBatchPromptInput {
    fn token_count(&self) -> usize {
        match self {
            Self::Host(input) => input.token_count,
            Self::TokenIds(input) => input.token_ids.len(),
        }
    }

    fn hidden_size(&self, metadata: Qwen3AsrExecutionMetadata) -> usize {
        match self {
            Self::Host(input) => input.hidden_size,
            Self::TokenIds(_) => metadata.llm_d_model,
        }
    }

    fn host(&self) -> Result<&Qwen3AsrLlmPrefillInput, Qwen3AsrServeBatchError> {
        match self {
            Self::Host(input) => Ok(input),
            Self::TokenIds(_) => Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prompt rows were not materialized on device".to_string(),
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct Qwen3AsrServeBatchJob {
    pub runtime_cache_path: PathBuf,
    /// The exact preflight provenance unit used by the submitting executor.
    /// The owner thread must use this instead of reparsing a path.
    pub runtime_source_preflight: crate::GgufRuntimeSourcePreflight,
    pub build_identity: crate::RuntimeBuildIdentity,
    /// Backend plus typed native-GQA proof resolved on the submitting thread.
    /// The batch owner must never re-derive provider capability from its own
    /// thread-local state or a backend label.
    pub resolved_runtime: ResolvedFamilyRuntimeInput,
    /// Effective native-GQA capability frozen on the submitting thread after
    /// applying the process opt-out. Owner threads must not reread the env.
    pub native_gqa: GgmlNativeGqaCapability,
    pub metadata: Qwen3AsrExecutionMetadata,
    /// Owner-bound prepared assets. The job crosses to the batch owner thread,
    /// so it carries the admission lease itself rather than cloning large
    /// token/logits/decoder payloads out of the cache entry.
    pub prepared_assets: Qwen3AsrServeBatchPreparedAssets,
    pub prompt_input: Qwen3AsrServeBatchPromptInput,
    /// Current invocation's exact host span plus the stable resident envelope.
    /// The resident half is part of the engine key, so one owner never mixes
    /// incompatible reusable graph shapes.
    pub kv_capacity: Qwen3AsrKvCacheCapacity,
    pub decode_config: Seq2SeqGreedyDecodeConfig,
    pub text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind,
    pub word_timestamps: bool,
    pub audio_duration_seconds: f32,
    /// Explicit cancel/pause/resume context for this job -- never a
    /// thread-local. Serve-batch prefill runs on the owner thread, which is
    /// not the thread that submitted the request, so this `Arc` (captured at
    /// submit time and carried on the job itself) is the only way
    /// chunk-boundary polls can observe the same cancel flag the HTTP cancel
    /// handler flips. See [`crate::RequestExecutionContext`].
    pub execution_context: Arc<crate::RequestExecutionContext>,
}

#[derive(Debug, Clone)]
pub(super) enum Qwen3AsrServeBatchPreparedAssets {
    /// The only production representation: admission and materialized state
    /// have one shared lifetime even after the job crosses to its owner thread.
    Admitted(PreparedRuntimeHandle<BuiltinPreparedRuntime>),
    /// A deliberately test-only seam for tiny decoder fixtures that do not
    /// contain the audio/frontend tensors required to construct the complete
    /// prepared runtime. This variant does not exist in production binaries.
    #[cfg(test)]
    Fixture {
        tokenizer: Option<Qwen3AsrTokenizer>,
        token_embedding_table: Arc<MappedTokenEmbeddingTable>,
        logits_head: Arc<Qwen3AsrLlmLogitsHead>,
        decoder_plan: Arc<QwenWholeDecoderPlan>,
    },
}

struct Qwen3AsrServeBatchPreparedRef<'a> {
    tokenizer: Option<&'a Qwen3AsrTokenizer>,
    token_embedding_table: &'a Arc<MappedTokenEmbeddingTable>,
    logits_head: &'a Arc<Qwen3AsrLlmLogitsHead>,
    decoder_plan: &'a Arc<QwenWholeDecoderPlan>,
}

impl Qwen3AsrServeBatchJob {
    fn backend(&self) -> GgmlCpuGraphBackend {
        self.resolved_runtime.backend()
    }

    fn prepared_runtime(
        &self,
    ) -> Result<Qwen3AsrServeBatchPreparedRef<'_>, Qwen3AsrServeBatchError> {
        match &self.prepared_assets {
            Qwen3AsrServeBatchPreparedAssets::Admitted(owner) => {
                let runtime = owner.as_ref().as_qwen3_asr().ok_or_else(|| {
                    Qwen3AsrServeBatchError::OwnerFailed {
                        reason: "qwen serve batch job carries a non-qwen prepared runtime"
                            .to_string(),
                    }
                })?;
                Ok(Qwen3AsrServeBatchPreparedRef {
                    tokenizer: runtime.tokenizer.as_ref(),
                    token_embedding_table: &runtime.token_embedding_table,
                    logits_head: &runtime.logits_head,
                    decoder_plan: &runtime.decoder_plan,
                })
            }
            #[cfg(test)]
            Qwen3AsrServeBatchPreparedAssets::Fixture {
                tokenizer,
                token_embedding_table,
                logits_head,
                decoder_plan,
            } => Ok(Qwen3AsrServeBatchPreparedRef {
                tokenizer: tokenizer.as_ref(),
                token_embedding_table,
                logits_head,
                decoder_plan,
            }),
        }
    }
}

#[derive(Debug, Error)]
pub(super) enum Qwen3AsrServeBatchError {
    #[cfg(test)]
    #[error("qwen serve batch test env {env} must be an integer in 0..={max}, got '{raw}'")]
    InvalidEnv {
        env: &'static str,
        raw: String,
        max: usize,
    },
    #[error("qwen serve batch requires max batch >= 2 when enabled, got {max_batch}")]
    InvalidEnabledBatch { max_batch: usize },
    #[error("qwen serve batch supports only gpu-class direct ggml backends, got {backend:?}")]
    UnsupportedBackend { backend: GgmlCpuGraphBackend },
    #[error("qwen serve batch engine registry mutex is poisoned")]
    RegistryPoisoned,
    #[error("qwen serve batch owner thread spawn failed: {reason}")]
    ThreadSpawnFailed { reason: String },
    #[error("qwen serve batch queue is full")]
    QueueFull,
    #[error("qwen serve batch owner thread is disconnected")]
    OwnerDisconnected,
    #[error("qwen serve batch owner reply timed out")]
    ReplyTimedOut,
    #[error("qwen serve batch owner failed: {reason}")]
    OwnerFailed { reason: String },
    #[error("qwen serve batch decode failed: {reason}")]
    DecodeFailed { reason: String },
    /// Cooperative cancel observed between prefill chunks. Display text carries
    /// the stable "canceled by transcription control" marker so
    /// `dispatch_error_to_backend` rewrites to `TranscriptionCanceled`.
    #[error("qwen serve batch canceled by transcription control")]
    Canceled,
}

impl Qwen3AsrServeBatchError {
    /// Classifies the transient serve-batch failures that should surface as a
    /// retryable HTTP status. `Some(true)` => queue saturation (429 backpressure);
    /// `Some(false)` => owner gone / GPU step hung (503); `None` => every other
    /// variant keeps its existing (non-retryable) mapping.
    pub(super) fn unavailable_retryable(&self) -> Option<bool> {
        match self {
            Self::QueueFull => Some(true),
            Self::OwnerDisconnected | Self::ReplyTimedOut => Some(false),
            _ => None,
        }
    }
}

/// Poll a job-carried execution context at a serve-batch prefill chunk
/// boundary. Typed `Canceled` keeps cancel off the generic decode-failed path.
///
/// Must read the `Arc` snapped into [`Qwen3AsrServeBatchJob::execution_context`]
/// at submit time -- the owner thread never installs a thread-local for the
/// submitting request, so this explicit context is the only production
/// signal.
fn ensure_serve_batch_prefill_not_canceled(
    context: &crate::RequestExecutionContext,
) -> Result<(), Qwen3AsrServeBatchError> {
    if context.is_canceled() {
        return Err(Qwen3AsrServeBatchError::Canceled);
    }
    Ok(())
}

fn map_serve_batch_graph_error(
    error: crate::ggml_runtime::GgmlCpuGraphError,
) -> Qwen3AsrServeBatchError {
    match error {
        crate::ggml_runtime::GgmlCpuGraphError::Canceled => Qwen3AsrServeBatchError::Canceled,
        other => Qwen3AsrServeBatchError::DecodeFailed {
            reason: other.to_string(),
        },
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Qwen3AsrServeBatchEngineKey {
    build_identity: crate::RuntimeBuildIdentity,
    lane: crate::models::native_execution_services::ExecutionLaneKey,
    resident_positions: usize,
    max_batch: usize,
    native_gqa: GgmlNativeGqaCapability,
}

/// Executor-owned qwen serve-batch owners. Clones of one executor share the
/// same map; independently constructed service roots do not. This makes owner
/// thread lifetime a normal part of `NativeExecutionServices` instead of a
/// process singleton partitioned by ambient scope ids.
#[derive(Clone, Default)]
pub(super) struct Qwen3AsrServeBatchEngineRegistry {
    engines: Arc<ServeBatchActiveRegistry<Qwen3AsrServeBatchEngineKey, Qwen3AsrServeBatchEngine>>,
}

impl std::fmt::Debug for Qwen3AsrServeBatchEngineRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Qwen3AsrServeBatchEngineRegistry")
            .finish_non_exhaustive()
    }
}

struct QwenReuseState {
    latest_attempt: Option<crate::models::native_execution_services::ExecutionCacheAttemptId>,
    pending: bool,
    closed: bool,
}

struct QwenReuseSignal {
    state: Mutex<QwenReuseState>,
    collector: Option<RuntimeReceiptCollector>,
}

impl QwenReuseSignal {
    fn record(
        &self,
        attempt: Option<crate::models::native_execution_services::ExecutionCacheAttemptId>,
    ) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if state.closed {
            if let Some(collector) = self.collector.as_ref() {
                collector.record_notification_coalesced();
            }
            return;
        }
        if state.pending
            && let Some(collector) = self.collector.as_ref()
        {
            collector.record_notification_coalesced();
        }
        state.latest_attempt = attempt;
        state.pending = true;
    }

    fn consume(
        &self,
    ) -> Option<Option<crate::models::native_execution_services::ExecutionCacheAttemptId>> {
        let Ok(mut state) = self.state.lock() else {
            return None;
        };
        if !state.pending {
            return None;
        }
        state.pending = false;
        Some(state.latest_attempt.take())
    }

    fn close_and_drop(&self) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        state.closed = true;
        if state.pending {
            state.pending = false;
            state.latest_attempt.take();
            if let Some(collector) = self.collector.as_ref() {
                collector.record_notification_coalesced();
            }
        }
    }
}

struct Qwen3AsrServeBatchEngine {
    sender: Mutex<Option<SyncSender<Qwen3AsrServeBatchEnvelope>>>,
    reuse: Arc<QwenReuseSignal>,
    worker: Mutex<Option<thread::JoinHandle<()>>>,
    worker_thread_id: thread::ThreadId,
    config: Qwen3AsrServeBatchConfig,
    is_alive: Arc<AtomicBool>,
    #[cfg(test)]
    owner_ready: Arc<AtomicBool>,
}

struct Qwen3AsrServeBatchEnvelope {
    job: Qwen3AsrServeBatchJob,
    native_execution_context:
        Option<crate::models::native_execution_services::NativeExecutionContext>,
    reply: mpsc::Sender<Result<GgmlAsrExecutionResult, Qwen3AsrServeBatchError>>,
}

struct Qwen3AsrOwnerThreadState {
    decoder: Option<Qwen3AsrLlmWholeDecoderGraphExecutor>,
    logits_runtime: Option<Qwen3AsrLlmLogitsHeadRuntime>,
    runtime_receipts: Vec<RuntimeResourceGuard>,
    receipt_context: Option<ServeBatchReceiptContext>,
}

#[derive(Clone)]
struct ServeBatchReceiptContext {
    collector: RuntimeReceiptCollector,
    owner_id: crate::models::runtime_receipts::RuntimeOwnerId,
}

struct Qwen3AsrActiveBatchSlot {
    slot: Qwen3AsrBatchSlot,
    native_execution_context:
        Option<crate::models::native_execution_services::NativeExecutionContext>,
    reply: mpsc::Sender<Result<GgmlAsrExecutionResult, Qwen3AsrServeBatchError>>,
}

struct Qwen3AsrPendingRefillSlot {
    slot_index: usize,
    slot: Qwen3AsrBatchSlot,
    native_execution_context:
        Option<crate::models::native_execution_services::NativeExecutionContext>,
    reply: mpsc::Sender<Result<GgmlAsrExecutionResult, Qwen3AsrServeBatchError>>,
}

struct Qwen3AsrPrefillSlotRef<'a> {
    slot_index: usize,
    slot: &'a mut Qwen3AsrBatchSlot,
}

struct Qwen3AsrBatchSlot {
    job: Qwen3AsrServeBatchJob,
    layer_kv_caches: Qwen3AsrHostKvCacheOwner,
    stop_token_ids: Vec<u32>,
    generated_tokens: Vec<u32>,
    /// Per-token softmax probability, parallel to `generated_tokens`.
    generated_probabilities: Vec<f32>,
    cache_prompt_tokens: usize,
    prefill_logits: Option<Vec<f32>>,
    /// How this slot's decode ended, `None` while it is still running.
    /// Mirrors the single-utterance driver so a guard-truncated slot is
    /// distinguishable from one that reached its stop token.
    stop_reason: Option<Seq2SeqGreedyDecodeStopReason>,
}

impl Qwen3AsrServeBatchConfig {
    pub(super) fn from_policy(policy: ServeBatchPolicy) -> Option<Self> {
        policy.enabled().then_some(Self {
            max_batch: policy
                .max_native_sessions
                .min(QWEN_SERVE_BATCH_MAX_BATCH_LIMIT),
            queue_capacity: policy.max_native_sessions,
            collect_window: SERVE_BATCH_COLLECT_WINDOW,
            send_timeout: QWEN_SERVE_BATCH_SEND_TIMEOUT,
            reply_timeout: QWEN_SERVE_BATCH_REPLY_TIMEOUT,
            trace_batches: serve_batch_trace_enabled(),
        })
    }

    #[cfg(test)]
    pub(super) fn from_env() -> Result<Option<Self>, Qwen3AsrServeBatchError> {
        let Some(max_batch) = crate::models::serve_batch_env::serve_batch_max_from_env(
            QWEN_SERVE_BATCH_MAX_BATCH_LIMIT,
        )
        .map_err(|error| Qwen3AsrServeBatchError::InvalidEnv {
            env: error.env,
            raw: error.raw,
            max: error.max,
        })?
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            max_batch,
            queue_capacity: max_batch,
            collect_window: SERVE_BATCH_COLLECT_WINDOW,
            send_timeout: QWEN_SERVE_BATCH_SEND_TIMEOUT,
            reply_timeout: QWEN_SERVE_BATCH_REPLY_TIMEOUT,
            trace_batches: false,
        }))
    }

    fn validate_for_job(
        self,
        job: &Qwen3AsrServeBatchJob,
    ) -> Result<Self, Qwen3AsrServeBatchError> {
        if self.max_batch < 2 {
            return Err(Qwen3AsrServeBatchError::InvalidEnabledBatch {
                max_batch: self.max_batch,
            });
        }
        let backend = job.backend();
        if !job.native_gqa.is_validated() {
            return Err(Qwen3AsrServeBatchError::UnsupportedBackend { backend });
        }
        if job.resolved_runtime.reuse_mode() != GgmlDecodeReuseMode::ReusableGraph {
            return Err(Qwen3AsrServeBatchError::UnsupportedBackend { backend });
        }
        let lane = current_execution_lane();
        let max_batch = serve_batch_vram_capped_max_batch(
            self.max_batch,
            backend,
            lane.as_ref(),
            qwen_serve_batch_vram_slot_bytes(job),
        );
        Ok(Self { max_batch, ..self })
    }
}

pub(super) fn shutdown_qwen_serve_batch_engines(registry: &Qwen3AsrServeBatchEngineRegistry) {
    registry.engines.shutdown();
}

pub(super) fn submit_qwen_serve_batch_job(
    registry: &Qwen3AsrServeBatchEngineRegistry,
    config: Qwen3AsrServeBatchConfig,
    job: Qwen3AsrServeBatchJob,
) -> Result<GgmlAsrExecutionResult, Qwen3AsrServeBatchError> {
    let config = config.validate_for_job(&job)?;
    let key = Qwen3AsrServeBatchEngineKey {
        build_identity: job.build_identity.clone(),
        lane: crate::models::native_execution_services::current_execution_lane_key(job.backend()),
        resident_positions: job.kv_capacity.resident_positions(),
        max_batch: config.max_batch,
        native_gqa: job.native_gqa,
    };
    let engine = qwen_serve_batch_engine_for_key(registry, key.clone(), config)?;
    let result = engine.submit(job);
    if crate::models::native_execution_services::current_execution_candidate_failure().is_some() {
        evict_qwen_serve_batch_engine(registry, &key, &engine);
    }
    result
}

fn qwen_serve_batch_vram_slot_bytes(job: &Qwen3AsrServeBatchJob) -> usize {
    serve_batch_estimate_llm_kv_slot_bytes(
        job.metadata.llm_layers,
        job.kv_capacity.resident_positions(),
        job.metadata.llm_kv_heads,
        job.metadata.llm_head_dim,
        std::mem::size_of::<f32>(),
    )
}

fn qwen_serve_batch_engine_for_key(
    registry: &Qwen3AsrServeBatchEngineRegistry,
    key: Qwen3AsrServeBatchEngineKey,
    config: Qwen3AsrServeBatchConfig,
) -> Result<Arc<Qwen3AsrServeBatchEngine>, Qwen3AsrServeBatchError> {
    let build_key = key.clone();
    let build = move || Qwen3AsrServeBatchEngine::spawn(build_key.clone(), config).map(Arc::new);
    registry.engines.lookup_or_build(
        key,
        build,
        |engine| serve_batch_owner_alive(&engine.is_alive),
        |engine| engine.record_receipt_reuse(),
        |engine| engine.shutdown_owner(),
        || Qwen3AsrServeBatchError::RegistryPoisoned,
    )
}

fn evict_qwen_serve_batch_engine(
    registry: &Qwen3AsrServeBatchEngineRegistry,
    key: &Qwen3AsrServeBatchEngineKey,
    engine: &Arc<Qwen3AsrServeBatchEngine>,
) {
    registry.engines.evict_exact(key, engine);
}

fn qwen_receipt_descriptor(
    key: &Qwen3AsrServeBatchEngineKey,
) -> Option<crate::models::runtime_receipts::RuntimeOwnerDescriptor> {
    let collector = current_runtime_receipts()?;
    let lane = key.lane.receipt_projection(&collector);
    collector.owner_descriptor(
        "serve-batch.qwen.serialized-actor",
        Some("qwen"),
        Some(&format!("{key:?}")),
        lane,
    )
}

impl Qwen3AsrServeBatchEngine {
    fn shutdown_owner(&self) {
        let sender = self
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        drop(sender);
        let worker = self
            .worker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(worker) = worker {
            if thread::current().id() == self.worker_thread_id {
                drop(worker);
            } else {
                let _ = worker.join();
            }
        }
    }

    fn record_receipt_reuse(&self) {
        self.reuse
            .record(crate::models::native_execution_services::current_execution_cache_attempt_id());
    }

    fn spawn(
        key: Qwen3AsrServeBatchEngineKey,
        config: Qwen3AsrServeBatchConfig,
    ) -> Result<Self, Qwen3AsrServeBatchError> {
        let (sender, receiver) = mpsc::sync_channel(config.queue_capacity);
        let receipt_collector = current_runtime_receipts();
        let reuse = Arc::new(QwenReuseSignal {
            state: Mutex::new(QwenReuseState {
                latest_attempt: None,
                pending: false,
                closed: false,
            }),
            collector: receipt_collector.clone(),
        });
        let worker_reuse = Arc::clone(&reuse);
        let owner_ready = Arc::new(AtomicBool::new(false));
        let worker_ready = Arc::clone(&owner_ready);
        let (is_alive, alive_guard) = OwnerAliveGuard::new();
        let owner_context = current_native_execution_context();
        let receipt_descriptor = qwen_receipt_descriptor(&key);
        let receipt_attempt_id =
            crate::models::native_execution_services::current_execution_cache_attempt_id();
        let worker = thread::Builder::new()
            .name(format!(
                "openasr-qwen-serve-batch-{:?}-{}",
                key.lane, key.max_batch
            ))
            .spawn(move || {
                let _alive_guard = alive_guard;
                let _context = owner_context.map(install_native_execution_context);
                let receipt_owner = receipt_descriptor.and_then(|descriptor| {
                    current_runtime_receipts()
                        .map(|collector| collector.start_owner(descriptor, receipt_attempt_id))
                });
                let receipt_context = receipt_owner.as_ref().and_then(|owner| {
                    owner.owner_id().and_then(|owner_id| {
                        current_runtime_receipts().map(|collector| ServeBatchReceiptContext {
                            collector,
                            owner_id,
                        })
                    })
                });
                qwen_owner_thread_loop(
                    receiver,
                    config,
                    worker_reuse,
                    worker_ready,
                    receipt_context,
                    receipt_owner,
                )
            })
            .map_err(|error| Qwen3AsrServeBatchError::ThreadSpawnFailed {
                reason: error.to_string(),
            })?;
        let worker_thread_id = worker.thread().id();
        Ok(Self {
            sender: Mutex::new(Some(sender)),
            reuse,
            worker: Mutex::new(Some(worker)),
            worker_thread_id,
            config,
            is_alive,
            #[cfg(test)]
            owner_ready,
        })
    }

    fn submit(
        &self,
        job: Qwen3AsrServeBatchJob,
    ) -> Result<GgmlAsrExecutionResult, Qwen3AsrServeBatchError> {
        let (reply, reply_rx) = mpsc::channel();
        let sender = self
            .sender
            .lock()
            .map_err(|_| Qwen3AsrServeBatchError::OwnerDisconnected)?
            .as_ref()
            .cloned()
            .ok_or(Qwen3AsrServeBatchError::OwnerDisconnected)?;
        serve_batch_submit_with_timeout(
            &sender,
            Qwen3AsrServeBatchEnvelope {
                job,
                native_execution_context: current_native_execution_context(),
                reply,
            },
            reply_rx,
            self.config.send_timeout,
            self.config.reply_timeout,
            || Qwen3AsrServeBatchError::QueueFull,
            || Qwen3AsrServeBatchError::OwnerDisconnected,
            || Qwen3AsrServeBatchError::ReplyTimedOut,
        )
    }
}

impl Drop for Qwen3AsrServeBatchEngine {
    fn drop(&mut self) {
        self.shutdown_owner();
    }
}

fn qwen_owner_thread_loop(
    receiver: Receiver<Qwen3AsrServeBatchEnvelope>,
    config: Qwen3AsrServeBatchConfig,
    reuse: Arc<QwenReuseSignal>,
    ready: Arc<AtomicBool>,
    receipt_context: Option<ServeBatchReceiptContext>,
    receipt_owner: Option<RuntimeOwnerGuard>,
) {
    ready.store(true, std::sync::atomic::Ordering::Release);
    let receipt_owner = receipt_owner;
    let mut state = Qwen3AsrOwnerThreadState {
        decoder: None,
        logits_runtime: None,
        runtime_receipts: Vec::new(),
        receipt_context,
    };
    let mut deferred = VecDeque::new();
    loop {
        if let Some(attempt) = reuse.consume()
            && let Some(owner) = receipt_owner.as_ref()
        {
            owner.record_reuse(attempt);
        }
        let Some(batch) = serve_batch_drain_compatible_batch(
            &mut deferred,
            &receiver,
            config.max_batch,
            config.collect_window,
            |first, next| {
                first.job.build_identity == next.job.build_identity
                    && first.job.runtime_cache_path == next.job.runtime_cache_path
            },
        ) else {
            reuse.close_and_drop();
            break;
        };
        if config.trace_batches {
            eprintln!(
                "openasr qwen serve batch: drained {} request(s)",
                batch.len()
            );
        }
        deferred.extend(state.run_batch(batch, &receiver, config));
    }
}

impl Qwen3AsrOwnerThreadState {
    fn shared_native_execution_context<'a>(
        contexts: impl IntoIterator<
            Item = &'a Option<crate::models::native_execution_services::NativeExecutionContext>,
        >,
    ) -> Result<
        Option<crate::models::native_execution_services::NativeExecutionContext>,
        Qwen3AsrServeBatchError,
    > {
        let contexts = contexts
            .into_iter()
            .filter_map(Option::as_ref)
            .cloned()
            .collect::<Vec<_>>();
        crate::models::native_execution_services::NativeExecutionContext::shared_lane(&contexts)
            .map_err(|error| Qwen3AsrServeBatchError::OwnerFailed {
                reason: format!("qwen serve-batch execution lanes are incompatible: {error}"),
            })
    }

    fn active_native_execution_context(
        slots: &[Option<Qwen3AsrActiveBatchSlot>],
    ) -> Result<
        Option<crate::models::native_execution_services::NativeExecutionContext>,
        Qwen3AsrServeBatchError,
    > {
        Self::shared_native_execution_context(
            slots
                .iter()
                .filter_map(Option::as_ref)
                .map(|active| &active.native_execution_context),
        )
    }

    fn run_batch(
        &mut self,
        batch: Vec<Qwen3AsrServeBatchEnvelope>,
        receiver: &Receiver<Qwen3AsrServeBatchEnvelope>,
        config: Qwen3AsrServeBatchConfig,
    ) -> VecDeque<Qwen3AsrServeBatchEnvelope> {
        self.decode_continuous_batch(batch, receiver, config)
    }

    fn decode_continuous_batch(
        &mut self,
        batch: Vec<Qwen3AsrServeBatchEnvelope>,
        receiver: &Receiver<Qwen3AsrServeBatchEnvelope>,
        config: Qwen3AsrServeBatchConfig,
    ) -> VecDeque<Qwen3AsrServeBatchEnvelope> {
        let mut deferred = VecDeque::new();
        if batch.is_empty() {
            return deferred;
        }

        let max_positions = batch[0].job.kv_capacity.resident_positions();
        let mut prepared = Vec::with_capacity(batch.len());
        for envelope in batch {
            let required_positions =
                Qwen3AsrBatchSlot::required_max_positions_for_job(&envelope.job);
            prepared.push((envelope, required_positions));
        }

        let mut slots = Vec::with_capacity(prepared.len());
        for (envelope, required_positions) in prepared {
            let _native_execution = envelope
                .native_execution_context
                .clone()
                .map(crate::models::native_execution_services::install_native_execution_context);
            match required_positions
                .and_then(|_| Qwen3AsrBatchSlot::new(envelope.job, max_positions))
            {
                Ok(slot) => slots.push(Some(Qwen3AsrActiveBatchSlot {
                    slot,
                    native_execution_context: envelope.native_execution_context,
                    reply: envelope.reply,
                })),
                Err(error) => {
                    let _ = envelope.reply.send(Err(error));
                }
            }
        }
        slots.retain(Option::is_some);
        if slots.is_empty() {
            return deferred;
        }
        let active_count = slots.iter().filter(|slot| slot.is_some()).count();
        let bucket_width = serve_batch_bucket_width(active_count, config.max_batch);
        if bucket_width > slots.len() {
            slots.resize_with(bucket_width, || None);
        }

        let native_execution_context = match Self::active_native_execution_context(&slots) {
            Ok(context) => context,
            Err(error) => {
                Self::fail_all_slots(&mut slots, error);
                return deferred;
            }
        };
        let _native_execution = native_execution_context
            .map(crate::models::native_execution_services::install_native_execution_context);
        let decoder_and_logits_result = {
            let decoder_slot = slots
                .iter()
                .find_map(|slot| slot.as_ref().map(|active| &active.slot))
                .expect("active slot count checked above");
            self.decoder_for(decoder_slot)
        };
        let (decoder, logits_runtime) = match decoder_and_logits_result {
            Ok(runtimes) => runtimes,
            Err(error) => {
                let reason = error.to_string();
                for active in slots.into_iter().flatten() {
                    let _ = active.reply.send(Err(Qwen3AsrServeBatchError::OwnerFailed {
                        reason: reason.clone(),
                    }));
                }
                return deferred;
            }
        };

        let mut prefill_entries: Vec<Qwen3AsrPrefillSlotRef<'_>> = slots
            .iter_mut()
            .enumerate()
            .filter_map(|(slot_index, active)| {
                active.as_mut().map(|active| Qwen3AsrPrefillSlotRef {
                    slot_index,
                    slot: &mut active.slot,
                })
            })
            .collect();
        let prefill_errors =
            Self::prefill_and_select_slot_entries(decoder, logits_runtime, &mut prefill_entries);
        drop(prefill_entries);
        for (slot_index, error) in prefill_errors {
            Self::fail_slot(&mut slots, slot_index, decoder, max_positions, false, error);
        }
        for slot_index in 0..slots.len() {
            if slots[slot_index]
                .as_ref()
                .map(|active| active.slot.is_done())
                .unwrap_or(false)
            {
                Self::finish_slot(&mut slots, slot_index, decoder, max_positions, false);
            }
        }
        if !slots.iter().any(Option::is_some) {
            return deferred;
        }

        // Initial owner construction + prefill is complete. Later refills and
        // each decode iteration install a fresh aggregate context so typed
        // failures fan out to exactly the requests participating in that
        // operation.
        drop(_native_execution);

        let mut graph_initialized = false;
        loop {
            if graph_initialized {
                Self::refill_free_slots(
                    &mut slots,
                    decoder,
                    logits_runtime,
                    max_positions,
                    receiver,
                    &mut deferred,
                    config.trace_batches,
                );
                if let Err(error) = Self::try_rebucket_active_slots(
                    &mut slots,
                    decoder,
                    logits_runtime,
                    max_positions,
                    receiver,
                    &mut deferred,
                    config.max_batch,
                    config.trace_batches,
                ) {
                    Self::fail_all_slots(&mut slots, error);
                    break;
                }
            }

            let native_execution_context = match Self::active_native_execution_context(&slots) {
                Ok(context) => context,
                Err(error) => {
                    Self::fail_all_slots(&mut slots, error);
                    break;
                }
            };
            let _native_execution = native_execution_context
                .map(crate::models::native_execution_services::install_native_execution_context);

            for slot_index in 0..slots.len() {
                let max_tokens_error = slots[slot_index].as_ref().and_then(|active| {
                    if active.slot.generated_tokens.len()
                        >= active.slot.job.decode_config.max_generated_tokens
                    {
                        Some(Qwen3AsrServeBatchError::DecodeFailed {
                            reason: Seq2SeqGreedyDecodeError::EotNotReachedBeforeMaxTokens {
                                max_generated_tokens: active
                                    .slot
                                    .job
                                    .decode_config
                                    .max_generated_tokens,
                                generated_tokens: active.slot.generated_tokens.clone(),
                                // Display-only construction: the message does
                                // not render probabilities.
                                generated_probabilities: Vec::new(),
                            }
                            .to_string(),
                        })
                    } else {
                        None
                    }
                });
                if let Some(error) = max_tokens_error {
                    Self::fail_slot(
                        &mut slots,
                        slot_index,
                        decoder,
                        max_positions,
                        graph_initialized,
                        error,
                    );
                }
            }
            if !slots.iter().any(Option::is_some) {
                break;
            }
            if graph_initialized
                && let Err(error) = Self::try_shrink_active_slots(
                    &mut slots,
                    decoder,
                    max_positions,
                    config.max_batch,
                    config.trace_batches,
                )
            {
                Self::fail_all_slots(&mut slots, error);
                break;
            }
            if !slots.iter().any(Option::is_some) {
                break;
            }

            let Some(d_model) = slots.iter().find_map(|slot| {
                slot.as_ref()
                    .map(|active| active.slot.job.metadata.llm_d_model)
            }) else {
                break;
            };
            let n_seq = slots.len();
            let mut token_ids = Vec::with_capacity(n_seq);
            let mut cache_positions = Vec::with_capacity(n_seq);
            let mut pack_errors = Vec::new();
            for (slot_index, active) in slots.iter().enumerate() {
                if let Some(active) = active {
                    match (
                        active.slot.last_generated_token_id(),
                        active.slot.next_cache_position(),
                    ) {
                        (Ok(token_id), Ok(cache_position)) => {
                            token_ids.push(token_id);
                            cache_positions.push(cache_position);
                        }
                        (Err(error), _) | (_, Err(error)) => {
                            pack_errors.push((slot_index, error));
                            token_ids.push(0);
                            cache_positions.push(0);
                        }
                    }
                } else {
                    token_ids.push(0);
                    cache_positions.push(if graph_initialized {
                        max_positions.saturating_sub(1)
                    } else {
                        0
                    });
                }
            }
            if !pack_errors.is_empty() {
                for (slot_index, error) in pack_errors {
                    Self::fail_slot(
                        &mut slots,
                        slot_index,
                        decoder,
                        max_positions,
                        graph_initialized,
                        error,
                    );
                }
                continue;
            }

            let step = if graph_initialized {
                decoder.run_token_step_reused_batched(
                    &token_ids,
                    &cache_positions,
                    QWEN_ROPE_THETA,
                    max_positions,
                )
            } else {
                let dummy_seed_layers =
                    Self::dummy_seed_layers_for_inactive_slots(&slots, max_positions);
                let dummy_seed_layers = match dummy_seed_layers {
                    Ok(dummy_seed_layers) => dummy_seed_layers,
                    Err(error) => {
                        Self::fail_all_slots(&mut slots, error);
                        break;
                    }
                };
                let seed_layers = slots
                    .iter()
                    .enumerate()
                    .map(|(slot_index, slot)| {
                        slot.as_ref()
                            .map(|active| active.slot.layer_kv_caches.as_slice())
                            .or_else(|| {
                                dummy_seed_layers[slot_index]
                                    .as_ref()
                                    .map(|owner| owner.as_slice())
                            })
                            .ok_or_else(|| Qwen3AsrServeBatchError::OwnerFailed {
                                reason: "qwen serve batch cannot seed an empty initial slot"
                                    .to_string(),
                            })
                    })
                    .collect::<Result<Vec<_>, _>>();
                let seed_layers = match seed_layers {
                    Ok(seed_layers) => seed_layers,
                    Err(error) => {
                        Self::fail_all_slots(&mut slots, error);
                        break;
                    }
                };
                decoder.run_token_step_reused_batched_seeded(
                    &token_ids,
                    &cache_positions,
                    &seed_layers,
                    QWEN_ROPE_THETA,
                    max_positions,
                )
            };
            let step = match step {
                Ok(step) => {
                    graph_initialized = true;
                    step
                }
                Err(error) => {
                    Self::fail_all_slots(
                        &mut slots,
                        Qwen3AsrServeBatchError::DecodeFailed {
                            reason: error.to_string(),
                        },
                    );
                    break;
                }
            };

            for slot_index in 0..slots.len() {
                let scatter_result = (|| {
                    let Some(active) = slots[slot_index].as_mut() else {
                        return Ok(());
                    };
                    let start = slot_index.checked_mul(d_model).ok_or_else(|| {
                        Qwen3AsrServeBatchError::DecodeFailed {
                            reason: "qwen serve batch hidden scatter offset overflowed".to_string(),
                        }
                    })?;
                    let end = start.checked_add(d_model).ok_or_else(|| {
                        Qwen3AsrServeBatchError::DecodeFailed {
                            reason: "qwen serve batch hidden scatter end overflowed".to_string(),
                        }
                    })?;
                    let hidden_for_slot = step.hidden.get(start..end).ok_or_else(|| {
                        Qwen3AsrServeBatchError::DecodeFailed {
                            reason: "qwen serve batch hidden scatter out of bounds".to_string(),
                        }
                    })?;
                    let prepared_runtime = active.slot.job.prepared_runtime()?;
                    let logits = logits_runtime
                        .compute_logits_for_last_hidden(
                            prepared_runtime.logits_head,
                            hidden_for_slot,
                        )
                        .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
                            reason: error.to_string(),
                        })?;
                    active.slot.select_next_token_from_logits(logits)
                })();
                match scatter_result {
                    Ok(()) => {
                        if slots[slot_index]
                            .as_ref()
                            .map(|active| active.slot.is_done())
                            .unwrap_or(false)
                        {
                            Self::finish_slot(
                                &mut slots,
                                slot_index,
                                decoder,
                                max_positions,
                                graph_initialized,
                            );
                        }
                    }
                    Err(error) => {
                        Self::fail_slot(
                            &mut slots,
                            slot_index,
                            decoder,
                            max_positions,
                            graph_initialized,
                            error,
                        );
                    }
                }
            }
        }

        deferred
    }

    fn dummy_seed_layers_for_inactive_slots(
        slots: &[Option<Qwen3AsrActiveBatchSlot>],
        max_positions: usize,
    ) -> Result<Vec<Option<Qwen3AsrHostKvCacheOwner>>, Qwen3AsrServeBatchError> {
        let (template, backend) = slots
            .iter()
            .find_map(|slot| {
                slot.as_ref()
                    .map(|active| (active.slot.job.metadata, active.slot.job.backend()))
            })
            .ok_or_else(|| Qwen3AsrServeBatchError::OwnerFailed {
                reason: "qwen serve batch cannot build dummy seed without an active slot"
                    .to_string(),
            })?;
        slots
            .iter()
            .map(|slot| {
                if slot.is_some() {
                    Ok(None)
                } else {
                    Qwen3AsrBatchSlot::zero_seed_layer_kv_caches(template, backend, max_positions)
                        .map(Some)
                }
            })
            .collect()
    }

    fn prefill_and_select_slot_entries(
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        logits_runtime: &mut Qwen3AsrLlmLogitsHeadRuntime,
        entries: &mut [Qwen3AsrPrefillSlotRef<'_>],
    ) -> Vec<(usize, Qwen3AsrServeBatchError)> {
        let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut failures = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for entry_index in 0..entries.len() {
            if let Err(error) = entries[entry_index]
                .slot
                .ensure_device_prompt_materialized(decoder)
            {
                failures.push((entries[entry_index].slot_index, error));
                continue;
            }
            let token_count = entries[entry_index].slot.job.prompt_input.token_count();
            if let Some((_, group)) = groups
                .iter_mut()
                .find(|(group_token_count, _)| *group_token_count == token_count)
            {
                group.push(entry_index);
            } else {
                groups.push((token_count, vec![entry_index]));
            }
        }

        for (group_token_count, group) in groups {
            if group.len() > 1 {
                if let Some(chunk_size) =
                    decoder.safe_multi_query_prefill_chunk_size_for(group_token_count)
                {
                    if let Err(error) = Self::prefill_and_select_batched_group(
                        decoder,
                        logits_runtime,
                        entries,
                        &group,
                        chunk_size,
                    ) {
                        failures.extend(group.into_iter().map(|entry_index| {
                            let mapped = match &error {
                                // Preserve typed cancel so dispatch keeps the
                                // TranscriptionCanceled path (not a generic
                                // decode failure) for every group member that
                                // shared the multi-query prefill.
                                Qwen3AsrServeBatchError::Canceled => {
                                    Qwen3AsrServeBatchError::Canceled
                                }
                                other => Qwen3AsrServeBatchError::DecodeFailed {
                                    reason: other.to_string(),
                                },
                            };
                            (entries[entry_index].slot_index, mapped)
                        }));
                    }
                } else {
                    for entry_index in group {
                        if let Err(error) = entries[entry_index]
                            .slot
                            .run_prefill_and_select(decoder, logits_runtime)
                        {
                            failures.push((entries[entry_index].slot_index, error));
                        }
                    }
                }
            } else {
                let entry_index = group[0];
                if let Err(error) = entries[entry_index]
                    .slot
                    .run_prefill_and_select(decoder, logits_runtime)
                {
                    failures.push((entries[entry_index].slot_index, error));
                }
            }
        }
        failures
    }

    fn prefill_and_select_batched_group(
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        logits_runtime: &mut Qwen3AsrLlmLogitsHeadRuntime,
        entries: &mut [Qwen3AsrPrefillSlotRef<'_>],
        group: &[usize],
        chunk_size: usize,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        if chunk_size == 0 {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prefill chunk size is zero".to_string(),
            });
        }
        let first = group[0];
        let token_count = entries[first].slot.job.prompt_input.token_count();
        let hidden_size = entries[first]
            .slot
            .job
            .prompt_input
            .hidden_size(entries[first].slot.job.metadata);
        for &entry_index in group {
            let slot = &entries[entry_index].slot;
            if slot.job.prompt_input.token_count() != token_count
                || slot.job.prompt_input.hidden_size(slot.job.metadata) != hidden_size
            {
                return Err(Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch grouped prefill shape mismatch".to_string(),
                });
            }
            if decoder.layer_count() != slot.layer_kv_caches.len() {
                return Err(Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch decoder/cache layer count mismatch".to_string(),
                });
            }
        }

        let n_seq = group.len();
        let require_even_chunks = decoder.prefill_chunks_require_even_width();
        let mut position_offset = 0usize;
        let mut final_hidden_by_sequence = vec![None; n_seq];
        while position_offset < token_count {
            // L1.2 cooperative cancel between multi-query prefill chunks.
            // Multi-query groups share one graph step, so any member cancel
            // aborts the shared chunk (existing all-or-nothing group model).
            for &entry_index in group {
                ensure_serve_batch_prefill_not_canceled(
                    &entries[entry_index].slot.job.execution_context,
                )?;
            }
            let remaining = token_count - position_offset;
            let chunk_len = if require_even_chunks {
                super::even_prefill_chunk_len(remaining, chunk_size)
            } else {
                remaining.min(chunk_size)
            };
            let hidden_start = position_offset.checked_mul(hidden_size).ok_or_else(|| {
                Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch grouped prefill hidden offset overflowed".to_string(),
                }
            })?;
            let hidden_len = chunk_len.checked_mul(hidden_size).ok_or_else(|| {
                Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch grouped prefill hidden width overflowed".to_string(),
                }
            })?;
            let hidden_end = hidden_start.checked_add(hidden_len).ok_or_else(|| {
                Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch grouped prefill hidden end overflowed".to_string(),
                }
            })?;
            let total_token_count = position_offset.checked_add(chunk_len).ok_or_else(|| {
                Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch grouped prefill span overflowed".to_string(),
                }
            })?;
            let mut hidden = Vec::with_capacity(hidden_len.saturating_mul(n_seq));
            for &entry_index in group {
                let input = entries[entry_index].slot.job.prompt_input.host()?;
                hidden.extend_from_slice(
                    input
                        .token_major_embeddings
                        .get(hidden_start..hidden_end)
                        .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                            reason: "qwen serve batch grouped prefill hidden slice out of bounds"
                                .to_string(),
                        })?,
                );
            }
            let step = {
                let layer_cache_refs = group
                    .iter()
                    .map(|&entry_index| entries[entry_index].slot.layer_kv_caches.as_slice())
                    .collect::<Vec<_>>();
                decoder
                    .run_prefill_batched_chunk(
                        &hidden,
                        chunk_len,
                        n_seq,
                        position_offset,
                        total_token_count,
                        &layer_cache_refs,
                        QWEN_ROPE_THETA,
                    )
                    .map_err(map_serve_batch_graph_error)?
            };
            for (sequence_index, &entry_index) in group.iter().enumerate() {
                let final_hidden = entries[entry_index]
                    .slot
                    .write_batched_prefill_chunk_outputs(
                        sequence_index,
                        n_seq,
                        position_offset,
                        chunk_len,
                        &step,
                    )?;
                final_hidden_by_sequence[sequence_index] = Some(final_hidden);
            }
            position_offset = total_token_count;
        }

        for (sequence_index, &entry_index) in group.iter().enumerate() {
            let final_hidden =
                final_hidden_by_sequence[sequence_index]
                    .take()
                    .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                        reason: "qwen serve batch grouped prefill produced no hidden state"
                            .to_string(),
                    })?;
            let prepared_runtime = entries[entry_index].slot.job.prepared_runtime()?;
            let logits = logits_runtime
                .compute_logits_for_last_hidden(prepared_runtime.logits_head, &final_hidden)
                .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
                    reason: error.to_string(),
                })?;
            entries[entry_index].slot.cache_prompt_tokens = token_count;
            entries[entry_index].slot.prefill_logits = Some(logits);
            let logits = entries[entry_index]
                .slot
                .prefill_logits
                .take()
                .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch prefill produced no logits".to_string(),
                })?;
            entries[entry_index]
                .slot
                .select_next_token_from_logits(logits)?;
        }
        Ok(())
    }

    fn refill_free_slots(
        slots: &mut [Option<Qwen3AsrActiveBatchSlot>],
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        logits_runtime: &mut Qwen3AsrLlmLogitsHeadRuntime,
        max_positions: usize,
        receiver: &Receiver<Qwen3AsrServeBatchEnvelope>,
        deferred: &mut VecDeque<Qwen3AsrServeBatchEnvelope>,
        trace_batches: bool,
    ) {
        let mut pending_refills = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for slot_index in 0..slots.len() {
            if slots[slot_index].is_some() {
                continue;
            }
            let Some(envelope) = Self::pop_refill_candidate(deferred, receiver) else {
                break;
            };
            let required_positions =
                match Qwen3AsrBatchSlot::required_max_positions_for_job(&envelope.job) {
                    Ok(required_positions) => required_positions,
                    Err(error) => {
                        let _ = envelope.reply.send(Err(error));
                        continue;
                    }
                };
            if required_positions > max_positions {
                deferred.push_front(envelope);
                break;
            }

            let Qwen3AsrServeBatchEnvelope {
                job,
                native_execution_context,
                reply,
            } = envelope;
            let _native_execution = native_execution_context
                .clone()
                .map(crate::models::native_execution_services::install_native_execution_context);
            let slot = match Qwen3AsrBatchSlot::new(job, max_positions) {
                Ok(slot) => slot,
                Err(error) => {
                    let _ = reply.send(Err(error));
                    continue;
                }
            };
            pending_refills.push(Qwen3AsrPendingRefillSlot {
                slot_index,
                slot,
                native_execution_context,
                reply,
            });
        }
        if pending_refills.is_empty() {
            return;
        }

        let native_execution_context = match Self::shared_native_execution_context(
            slots
                .iter()
                .filter_map(Option::as_ref)
                .map(|active| &active.native_execution_context)
                .chain(
                    pending_refills
                        .iter()
                        .map(|pending| &pending.native_execution_context),
                ),
        ) {
            Ok(context) => context,
            Err(error) => {
                let reason = error.to_string();
                for pending in pending_refills {
                    let _ = pending
                        .reply
                        .send(Err(Qwen3AsrServeBatchError::OwnerFailed {
                            reason: reason.clone(),
                        }));
                }
                Self::fail_all_slots(slots, error);
                return;
            }
        };
        let _native_execution = native_execution_context
            .map(crate::models::native_execution_services::install_native_execution_context);

        let mut prefill_entries = pending_refills
            .iter_mut()
            .map(|pending| Qwen3AsrPrefillSlotRef {
                slot_index: pending.slot_index,
                slot: &mut pending.slot,
            })
            .collect::<Vec<_>>();
        let prefill_errors =
            Self::prefill_and_select_slot_entries(decoder, logits_runtime, &mut prefill_entries);
        drop(prefill_entries);

        for pending in pending_refills {
            let Qwen3AsrPendingRefillSlot {
                slot_index,
                slot,
                native_execution_context,
                reply,
            } = pending;
            if let Some((_, error)) = prefill_errors
                .iter()
                .find(|(failed_slot_index, _)| *failed_slot_index == slot_index)
            {
                let _ = reply.send(Err(Qwen3AsrServeBatchError::DecodeFailed {
                    reason: error.to_string(),
                }));
                continue;
            }
            if slot.is_done() {
                let _ = reply.send(slot.finish());
                continue;
            }
            if let Err(error) = decoder.zero_reused_batched_slot(slot_index, max_positions) {
                let _ = reply.send(Err(Qwen3AsrServeBatchError::DecodeFailed {
                    reason: error.to_string(),
                }));
                continue;
            }
            if let Err(error) = decoder.seed_reused_batched_slot(
                slot_index,
                slot.cache_prompt_tokens,
                &slot.layer_kv_caches,
                max_positions,
            ) {
                let _ = reply.send(Err(Qwen3AsrServeBatchError::DecodeFailed {
                    reason: error.to_string(),
                }));
                continue;
            }
            slots[slot_index] = Some(Qwen3AsrActiveBatchSlot {
                slot,
                native_execution_context,
                reply,
            });
            if trace_batches {
                eprintln!("openasr qwen serve batch: refilled slot {slot_index}");
            }
        }
    }

    fn try_rebucket_active_slots(
        slots: &mut Vec<Option<Qwen3AsrActiveBatchSlot>>,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        logits_runtime: &mut Qwen3AsrLlmLogitsHeadRuntime,
        max_positions: usize,
        receiver: &Receiver<Qwen3AsrServeBatchEnvelope>,
        deferred: &mut VecDeque<Qwen3AsrServeBatchEnvelope>,
        max_batch: usize,
        trace_batches: bool,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        let active_count = slots.iter().filter(|slot| slot.is_some()).count();
        if active_count == 0 || active_count != slots.len() || slots.len() >= max_batch {
            return Ok(());
        }
        let candidate_limit = max_batch.saturating_sub(active_count);
        let mut pending = Vec::new();
        while pending.len() < candidate_limit {
            let Some(envelope) = Self::pop_refill_candidate(deferred, receiver) else {
                break;
            };
            let required_positions =
                match Qwen3AsrBatchSlot::required_max_positions_for_job(&envelope.job) {
                    Ok(required_positions) => required_positions,
                    Err(error) => {
                        let _ = envelope.reply.send(Err(error));
                        continue;
                    }
                };
            if required_positions > max_positions {
                deferred.push_front(envelope);
                break;
            }

            let Qwen3AsrServeBatchEnvelope {
                job,
                native_execution_context,
                reply,
            } = envelope;
            let _native_execution = native_execution_context
                .clone()
                .map(crate::models::native_execution_services::install_native_execution_context);
            match Qwen3AsrBatchSlot::new(job, max_positions) {
                Ok(slot) => pending.push((slot, native_execution_context, reply)),
                Err(error) => {
                    let _ = reply.send(Err(error));
                }
            }
        }
        if pending.is_empty() {
            return Ok(());
        }

        let native_execution_context = Self::shared_native_execution_context(
            slots
                .iter()
                .filter_map(Option::as_ref)
                .map(|active| &active.native_execution_context)
                .chain(pending.iter().map(|(_, context, _)| context)),
        )?;
        let _native_execution = native_execution_context
            .map(crate::models::native_execution_services::install_native_execution_context);

        let previous_width = slots.len();
        let target_active = active_count.checked_add(pending.len()).ok_or_else(|| {
            Qwen3AsrServeBatchError::OwnerFailed {
                reason: "qwen serve batch rebucket active count overflowed".to_string(),
            }
        })?;
        let mut bucket_width = serve_batch_bucket_width(target_active, max_batch);
        if bucket_width <= previous_width {
            for (slot, native_execution_context, reply) in pending.into_iter().rev() {
                deferred.push_front(Qwen3AsrServeBatchEnvelope {
                    job: slot.job,
                    native_execution_context,
                    reply,
                });
            }
            return Ok(());
        }

        let mut prefill_entries = pending
            .iter_mut()
            .enumerate()
            .map(|(pending_index, (slot, _, _))| Qwen3AsrPrefillSlotRef {
                slot_index: previous_width + pending_index,
                slot,
            })
            .collect::<Vec<_>>();
        let prefill_errors =
            Self::prefill_and_select_slot_entries(decoder, logits_runtime, &mut prefill_entries);
        drop(prefill_entries);

        let mut admitted = Vec::new();
        for (pending_index, (slot, native_execution_context, reply)) in
            pending.into_iter().enumerate()
        {
            let slot_index = previous_width + pending_index;
            if let Some((_, error)) = prefill_errors
                .iter()
                .find(|(failed_slot_index, _)| *failed_slot_index == slot_index)
            {
                let _ = reply.send(Err(Qwen3AsrServeBatchError::DecodeFailed {
                    reason: error.to_string(),
                }));
                continue;
            }
            if slot.is_done() {
                let _ = reply.send(slot.finish());
                continue;
            }
            admitted.push(Qwen3AsrActiveBatchSlot {
                slot,
                native_execution_context,
                reply,
            });
        }
        if admitted.is_empty() {
            return Ok(());
        }
        bucket_width = serve_batch_bucket_width(active_count + admitted.len(), max_batch);
        if bucket_width <= previous_width {
            for active in admitted.into_iter().rev() {
                deferred.push_front(Qwen3AsrServeBatchEnvelope {
                    job: active.slot.job,
                    native_execution_context: active.native_execution_context,
                    reply: active.reply,
                });
            }
            return Ok(());
        }

        for active in admitted {
            slots.push(Some(active));
        }
        if bucket_width > slots.len() {
            slots.resize_with(bucket_width, || None);
        }
        Self::reseed_rebucketed_slots(slots, decoder, max_positions)?;
        if trace_batches {
            eprintln!(
                "openasr qwen serve batch: rebucketed {previous_width}->{bucket_width} slot(s)"
            );
        }
        Ok(())
    }

    fn try_shrink_active_slots(
        slots: &mut Vec<Option<Qwen3AsrActiveBatchSlot>>,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        max_positions: usize,
        max_batch: usize,
        trace_batches: bool,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        let active_count = slots.iter().filter(|slot| slot.is_some()).count();
        if active_count == 0 || active_count == slots.len() {
            return Ok(());
        }
        let bucket_width = serve_batch_bucket_width(active_count, max_batch);
        if bucket_width >= slots.len() {
            return Ok(());
        }

        let previous_width = slots.len();
        serve_batch_compact_active_slots(slots, bucket_width);
        Self::reseed_rebucketed_slots(slots, decoder, max_positions)?;
        if trace_batches {
            eprintln!("openasr qwen serve batch: shrank {previous_width}->{bucket_width} slot(s)");
        }
        Ok(())
    }

    fn reseed_rebucketed_slots(
        slots: &mut [Option<Qwen3AsrActiveBatchSlot>],
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        max_positions: usize,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        let dummy_seed_layers = Self::dummy_seed_layers_for_inactive_slots(slots, max_positions)?;
        for active in slots.iter_mut().filter_map(Option::as_mut) {
            active.slot.ensure_generated_host_kv_replayed(decoder)?;
        }
        let seed_layers = slots
            .iter()
            .enumerate()
            .map(|(slot_index, slot)| {
                slot.as_ref()
                    .map(|active| active.slot.layer_kv_caches.as_slice())
                    .or_else(|| {
                        dummy_seed_layers[slot_index]
                            .as_ref()
                            .map(|owner| owner.as_slice())
                    })
                    .ok_or_else(|| Qwen3AsrServeBatchError::OwnerFailed {
                        reason: "qwen serve batch cannot seed an empty rebucketed slot".to_string(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        decoder
            .reset_reused_batched_seeded(&seed_layers, QWEN_ROPE_THETA, max_positions)
            .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
                reason: error.to_string(),
            })
    }

    fn pop_refill_candidate(
        deferred: &mut VecDeque<Qwen3AsrServeBatchEnvelope>,
        receiver: &Receiver<Qwen3AsrServeBatchEnvelope>,
    ) -> Option<Qwen3AsrServeBatchEnvelope> {
        if let Some(envelope) = deferred.pop_front() {
            return Some(envelope);
        }
        receiver.try_recv().ok()
    }

    fn finish_slot(
        slots: &mut [Option<Qwen3AsrActiveBatchSlot>],
        slot_index: usize,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        max_positions: usize,
        graph_initialized: bool,
    ) {
        let Some(active) = slots[slot_index].take() else {
            return;
        };
        let Qwen3AsrActiveBatchSlot {
            slot,
            native_execution_context: _,
            reply,
        } = active;
        Self::send_result_after_optional_zero(
            reply,
            decoder,
            slot_index,
            max_positions,
            graph_initialized,
            slot.finish(),
        );
    }

    fn fail_slot(
        slots: &mut [Option<Qwen3AsrActiveBatchSlot>],
        slot_index: usize,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        max_positions: usize,
        graph_initialized: bool,
        error: Qwen3AsrServeBatchError,
    ) {
        let Some(active) = slots[slot_index].take() else {
            return;
        };
        Self::send_result_after_optional_zero(
            active.reply,
            decoder,
            slot_index,
            max_positions,
            graph_initialized,
            Err(error),
        );
    }

    fn send_result_after_optional_zero(
        reply: mpsc::Sender<Result<GgmlAsrExecutionResult, Qwen3AsrServeBatchError>>,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        slot_index: usize,
        max_positions: usize,
        graph_initialized: bool,
        mut result: Result<GgmlAsrExecutionResult, Qwen3AsrServeBatchError>,
    ) {
        if graph_initialized
            && let Err(error) = decoder.zero_reused_batched_slot(slot_index, max_positions)
        {
            result = Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: error.to_string(),
            });
        }
        let _ = reply.send(result);
    }

    fn fail_all_slots(
        slots: &mut [Option<Qwen3AsrActiveBatchSlot>],
        error: Qwen3AsrServeBatchError,
    ) {
        let reason = error.to_string();
        for active in slots.iter_mut().filter_map(Option::take) {
            let _ = active
                .reply
                .send(Err(Qwen3AsrServeBatchError::DecodeFailed {
                    reason: reason.clone(),
                }));
        }
    }

    fn record_runtime_receipts(&mut self) {
        if !self.runtime_receipts.is_empty() {
            return;
        }
        let Some(receipt) = self.receipt_context.as_ref() else {
            return;
        };
        for kind in [
            "serve-batch.qwen.decoder-runtime",
            "serve-batch.qwen.logits-runtime",
        ] {
            if let Some(descriptor) = receipt.collector.no_broker_resource_descriptor(kind)
                && let Some(resource) = receipt
                    .collector
                    .acquire_resource(receipt.owner_id, descriptor)
            {
                self.runtime_receipts.push(resource);
            }
        }
    }

    fn decoder_for(
        &mut self,
        slot: &Qwen3AsrBatchSlot,
    ) -> Result<
        (
            &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
            &mut Qwen3AsrLlmLogitsHeadRuntime,
        ),
        Qwen3AsrServeBatchError,
    > {
        if self.decoder.is_none() && self.logits_runtime.is_none() {
            let prepared_runtime = slot.job.prepared_runtime()?;
            let token_embedding = prepared_runtime
                .token_embedding_table
                .device_graph_spec()
                .ok_or_else(|| Qwen3AsrServeBatchError::OwnerFailed {
                    reason: "qwen serve batch requires canonical device token embedding"
                        .to_string(),
                })?;
            let decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new_from_plan_with_preflight_and_token_embedding_for_qwen(
                prepared_runtime.decoder_plan.as_ref(),
                &slot.job.runtime_source_preflight,
                token_embedding,
                slot.job.resolved_runtime,
                QwenQkvExecutionMode::FusedArena,
            )
            .map_err(|error| Qwen3AsrServeBatchError::OwnerFailed {
                reason: format!("qwen whole-decoder init failed: {error}"),
            })?;
            let logits_runtime = prepared_runtime
                .logits_head
                .new_runtime(slot.job.backend())
                .map_err(|error| Qwen3AsrServeBatchError::OwnerFailed {
                    reason: format!("qwen logits-head runtime init failed: {error}"),
                })?;
            self.decoder = Some(decoder);
            self.logits_runtime = Some(logits_runtime);
            self.record_runtime_receipts();
        } else if self.decoder.is_none() || self.logits_runtime.is_none() {
            return Err(Qwen3AsrServeBatchError::OwnerFailed {
                reason: "qwen serve batch decoder/logits runtime cache is inconsistent".to_string(),
            });
        }
        let decoder =
            self.decoder
                .as_mut()
                .ok_or_else(|| Qwen3AsrServeBatchError::OwnerFailed {
                    reason: "qwen serve batch decoder cache is unexpectedly empty".to_string(),
                })?;
        let logits_runtime =
            self.logits_runtime
                .as_mut()
                .ok_or_else(|| Qwen3AsrServeBatchError::OwnerFailed {
                    reason: "qwen serve batch logits runtime cache is unexpectedly empty"
                        .to_string(),
                })?;
        Ok((decoder, logits_runtime))
    }
}

impl Qwen3AsrBatchSlot {
    fn required_max_positions_for_job(
        job: &Qwen3AsrServeBatchJob,
    ) -> Result<usize, Qwen3AsrServeBatchError> {
        let measured = crate::capacity::topology::causal_prefix_positions_with_context_cap(
            super::capacity::QWEN3_SELF_KV_STATE_ID,
            job.decode_config.initial_prompt_tokens.len(),
            job.decode_config.max_generated_tokens,
            job.metadata.llm_max_positions,
        )
        .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
            reason: format!("qwen serve batch max-position calculation failed: {error}"),
        })?;
        job.kv_capacity
            .validate_measured_logical_positions(measured)
            .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
                reason: error.to_string(),
            })?;
        Ok(measured)
    }

    fn new(
        job: Qwen3AsrServeBatchJob,
        max_positions: usize,
    ) -> Result<Self, Qwen3AsrServeBatchError> {
        if job.metadata.llm_layers == 0 {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch requires at least one llm layer".to_string(),
            });
        }
        if max_positions == 0 {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch requires a positive decode span".to_string(),
            });
        }
        let required_positions = Self::required_max_positions_for_job(&job)?;
        if max_positions != job.kv_capacity.resident_positions() {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch resident span does not match the job's stable reserve"
                    .to_string(),
            });
        }
        if max_positions < required_positions {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch shared max-position span is smaller than this slot"
                    .to_string(),
            });
        }
        // `job.resolved_runtime` was materialized on the submitting thread;
        // this constructor
        // may run on a different worker thread with no request-backend
        // override installed, so re-resolving here would silently diverge
        // from the backend the submitter actually decided on.
        let host = resolve_qwen_family_production_kv_cache_policy(
            job.backend(),
            job.metadata.llm_head_dim,
        )
        .to_spec()
        .host;
        let layer_kv_caches = Qwen3AsrHostKvCacheOwner::try_new(
            "qwen3-asr.serve-batch.slot.self-kv.host",
            job.metadata.llm_layers,
            job.kv_capacity,
            job.metadata.llm_kv_heads,
            job.metadata.llm_head_dim,
            host,
            Qwen3AsrHostKvMode::Materialized,
        )
        .map_err(|reason| Qwen3AsrServeBatchError::DecodeFailed { reason })?;
        let stop_token_ids = build_seq2seq_greedy_stop_token_ids(&job.decode_config);
        Ok(Self {
            job,
            layer_kv_caches,
            stop_token_ids,
            generated_tokens: Vec::new(),
            generated_probabilities: Vec::new(),
            cache_prompt_tokens: 0,
            prefill_logits: None,
            stop_reason: None,
        })
    }

    /// `backend` must come from an active slot's own resolved runtime (already
    /// materialized on that job's submitting thread) -- this constructor may
    /// run on a worker thread with no request-backend override installed, so
    /// re-resolving here would silently diverge from what the batch is
    /// actually decoding on.
    fn zero_seed_layer_kv_caches(
        metadata: Qwen3AsrExecutionMetadata,
        backend: GgmlCpuGraphBackend,
        resident_positions: usize,
    ) -> Result<Qwen3AsrHostKvCacheOwner, Qwen3AsrServeBatchError> {
        if metadata.llm_layers == 0 {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch dummy seed requires at least one llm layer".to_string(),
            });
        }
        let row_width = metadata
            .llm_kv_heads
            .checked_mul(metadata.llm_head_dim)
            .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch dummy seed row width overflowed".to_string(),
            })?;
        let zero_row = vec![0.0_f32; row_width];
        let host = resolve_qwen_family_production_kv_cache_policy(backend, metadata.llm_head_dim)
            .to_spec()
            .host;
        if resident_positions == 0 {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch dummy seed requires a positive resident span".to_string(),
            });
        }
        // Inactive slots seed one admitted zero row, not the resident span.
        let capacity = Qwen3AsrKvCacheCapacity::new(1, resident_positions).map_err(|error| {
            Qwen3AsrServeBatchError::DecodeFailed {
                reason: error.to_string(),
            }
        })?;
        let mut layers = Qwen3AsrHostKvCacheOwner::try_new(
            "qwen3-asr.serve-batch.dummy-seed.self-kv.host",
            metadata.llm_layers,
            capacity,
            metadata.llm_kv_heads,
            metadata.llm_head_dim,
            host,
            Qwen3AsrHostKvMode::Materialized,
        )
        .map_err(|reason| Qwen3AsrServeBatchError::DecodeFailed { reason })?;
        for cache in layers.iter_mut() {
            cache
                .write(0, &zero_row, &zero_row)
                .map_err(|reason| Qwen3AsrServeBatchError::DecodeFailed { reason })?;
        }
        Ok(layers)
    }

    fn ensure_device_prompt_materialized(
        &mut self,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        let Qwen3AsrServeBatchPromptInput::TokenIds(input) = &self.job.prompt_input else {
            return Ok(());
        };
        let hidden_size = self.job.metadata.llm_d_model;
        let token_count = input.token_ids.len();
        let token_major_embeddings = decoder
            .materialize_token_prompt_on_device(
                &input.token_ids,
                &input.audio_rows,
                &input.audio_positions,
            )
            .map_err(map_serve_batch_graph_error)?
            .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch GPU lane has no device token embedding".to_string(),
            })?;
        self.job.prompt_input = Qwen3AsrServeBatchPromptInput::Host(Qwen3AsrLlmPrefillInput {
            token_count,
            hidden_size,
            token_major_embeddings,
        });
        Ok(())
    }

    fn run_prefill(
        &mut self,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        logits_runtime: &mut Qwen3AsrLlmLogitsHeadRuntime,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        self.ensure_device_prompt_materialized(decoder)?;
        let token_count = self.job.prompt_input.token_count();
        if token_count == 0 {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prefill token count is zero".to_string(),
            });
        }
        if decoder.layer_count() != self.layer_kv_caches.len() {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch decoder/cache layer count mismatch".to_string(),
            });
        }
        let Some(chunk_size) = decoder.safe_multi_query_prefill_chunk_size_for(token_count) else {
            return self.run_prefill_serial(decoder, logits_runtime);
        };
        self.run_prefill_chunked(decoder, logits_runtime, chunk_size)
    }

    fn run_prefill_chunked(
        &mut self,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        logits_runtime: &mut Qwen3AsrLlmLogitsHeadRuntime,
        chunk_size: usize,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        if chunk_size == 0 {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prefill chunk size is zero".to_string(),
            });
        }
        let token_count = self.job.prompt_input.token_count();
        if token_count <= chunk_size {
            let input = self.job.prompt_input.host()?;
            let step = decoder
                .run_prefill(&input.token_major_embeddings, token_count, QWEN_ROPE_THETA)
                .map_err(map_serve_batch_graph_error)?;
            return self.write_prefill_step_outputs(token_count, step, logits_runtime);
        }
        let hidden_size = self.job.prompt_input.hidden_size(self.job.metadata);
        let require_even_chunks = decoder.prefill_chunks_require_even_width();
        let mut position_offset = 0usize;
        let mut final_hidden = None;
        while position_offset < token_count {
            // L1.2 cooperative cancel between single-slot host-cache chunks.
            ensure_serve_batch_prefill_not_canceled(&self.job.execution_context)?;
            let remaining = token_count - position_offset;
            let chunk_len = if require_even_chunks {
                super::even_prefill_chunk_len(remaining, chunk_size)
            } else {
                remaining.min(chunk_size)
            };
            let hidden_start = position_offset.checked_mul(hidden_size).ok_or_else(|| {
                Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch prefill chunk hidden offset overflowed".to_string(),
                }
            })?;
            let hidden_len = chunk_len.checked_mul(hidden_size).ok_or_else(|| {
                Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch prefill chunk hidden width overflowed".to_string(),
                }
            })?;
            let hidden_end = hidden_start.checked_add(hidden_len).ok_or_else(|| {
                Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch prefill chunk hidden end overflowed".to_string(),
                }
            })?;
            let total_token_count = position_offset.checked_add(chunk_len).ok_or_else(|| {
                Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch prefill chunk span overflowed".to_string(),
                }
            })?;
            let step = decoder
                .run_prefill_chunk(
                    &self.job.prompt_input.host()?.token_major_embeddings[hidden_start..hidden_end],
                    chunk_len,
                    position_offset,
                    total_token_count,
                    &self.layer_kv_caches,
                    QWEN_ROPE_THETA,
                )
                .map_err(map_serve_batch_graph_error)?;
            final_hidden =
                Some(self.write_prefill_chunk_outputs(position_offset, chunk_len, step)?);
            position_offset = total_token_count;
        }
        self.cache_prompt_tokens = token_count;
        let final_hidden = final_hidden.ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
            reason: "qwen serve batch prefill produced no hidden state".to_string(),
        })?;
        let prepared_runtime = self.job.prepared_runtime()?;
        let logits = logits_runtime
            .compute_logits_for_last_hidden(prepared_runtime.logits_head, &final_hidden)
            .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
                reason: error.to_string(),
            })?;
        self.prefill_logits = Some(logits);
        Ok(())
    }

    fn run_prefill_serial(
        &mut self,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        logits_runtime: &mut Qwen3AsrLlmLogitsHeadRuntime,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        let token_count = self.job.prompt_input.token_count();
        let mut final_hidden = None;
        for token_position in 0..token_count {
            // Serial host-step prefill is the chunk-size-1 fallback; poll the
            // same cancel boundary so cancel does not wait for the full prompt.
            ensure_serve_batch_prefill_not_canceled(&self.job.execution_context)?;
            let hidden = self.prefill_prompt_hidden_at(token_position)?;
            let step = decoder
                .run_step(
                    &hidden,
                    token_position,
                    &self.layer_kv_caches,
                    QWEN_ROPE_THETA,
                )
                .map_err(map_serve_batch_graph_error)?;
            for (layer_index, (projected_k, projected_v)) in step.layer_kv.iter().enumerate() {
                self.layer_kv_caches[layer_index]
                    .write(token_position, projected_k, projected_v)
                    .map_err(|reason| Qwen3AsrServeBatchError::DecodeFailed { reason })?;
            }
            final_hidden = Some(step.hidden);
        }
        self.cache_prompt_tokens = token_count;
        let final_hidden = final_hidden.ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
            reason: "qwen serve batch prefill produced no hidden state".to_string(),
        })?;
        let prepared_runtime = self.job.prepared_runtime()?;
        let logits = logits_runtime
            .compute_logits_for_last_hidden(prepared_runtime.logits_head, &final_hidden)
            .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
                reason: error.to_string(),
            })?;
        self.prefill_logits = Some(logits);
        Ok(())
    }

    fn write_prefill_step_outputs(
        &mut self,
        token_count: usize,
        step: super::llm_transformer::Qwen3AsrLlmWholeStepOutput,
        logits_runtime: &mut Qwen3AsrLlmLogitsHeadRuntime,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        let final_hidden = self.write_prefill_chunk_outputs(0, token_count, step)?;
        self.cache_prompt_tokens = token_count;
        let prepared_runtime = self.job.prepared_runtime()?;
        let logits = logits_runtime
            .compute_logits_for_last_hidden(prepared_runtime.logits_head, &final_hidden)
            .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
                reason: error.to_string(),
            })?;
        self.prefill_logits = Some(logits);
        Ok(())
    }

    fn write_prefill_chunk_outputs(
        &mut self,
        position_offset: usize,
        token_count: usize,
        step: super::llm_transformer::Qwen3AsrLlmWholeStepOutput,
    ) -> Result<Vec<f32>, Qwen3AsrServeBatchError> {
        self.write_batched_prefill_chunk_outputs(0, 1, position_offset, token_count, &step)
    }

    fn write_batched_prefill_chunk_outputs(
        &mut self,
        sequence_index: usize,
        n_seq: usize,
        position_offset: usize,
        token_count: usize,
        step: &super::llm_transformer::Qwen3AsrLlmWholeStepOutput,
    ) -> Result<Vec<f32>, Qwen3AsrServeBatchError> {
        if sequence_index >= n_seq {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prefill sequence index out of bounds".to_string(),
            });
        }
        if step.layer_kv.len() != self.layer_kv_caches.len() {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prefill layer-KV count mismatch".to_string(),
            });
        }
        let output_tokens = token_count.checked_mul(n_seq).ok_or_else(|| {
            Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prefill token/sequence count overflowed".to_string(),
            }
        })?;
        let kv_row_width = self
            .job
            .metadata
            .llm_kv_heads
            .checked_mul(self.job.metadata.llm_head_dim)
            .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prefill KV row width overflowed".to_string(),
            })?;
        for token_position in 0..token_count {
            let absolute_position =
                position_offset.checked_add(token_position).ok_or_else(|| {
                    Qwen3AsrServeBatchError::DecodeFailed {
                        reason: "qwen serve batch prefill absolute row overflowed".to_string(),
                    }
                })?;
            let output_index = sequence_index
                .checked_mul(token_count)
                .and_then(|base| base.checked_add(token_position))
                .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch prefill output row overflowed".to_string(),
                })?;
            let row_start = output_index.checked_mul(kv_row_width).ok_or_else(|| {
                Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch prefill KV row offset overflowed".to_string(),
                }
            })?;
            let row_end = row_start.checked_add(kv_row_width).ok_or_else(|| {
                Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch prefill KV row end overflowed".to_string(),
                }
            })?;
            for (layer_index, (projected_k, projected_v)) in step.layer_kv.iter().enumerate() {
                let key_row = projected_k.get(row_start..row_end).ok_or_else(|| {
                    Qwen3AsrServeBatchError::DecodeFailed {
                        reason: "qwen serve batch prefill K row out of bounds".to_string(),
                    }
                })?;
                let value_row = projected_v.get(row_start..row_end).ok_or_else(|| {
                    Qwen3AsrServeBatchError::DecodeFailed {
                        reason: "qwen serve batch prefill V row out of bounds".to_string(),
                    }
                })?;
                self.layer_kv_caches[layer_index]
                    .write(absolute_position, key_row, value_row)
                    .map_err(|reason| Qwen3AsrServeBatchError::DecodeFailed { reason })?;
            }
        }
        let hidden_size = self.job.prompt_input.hidden_size(self.job.metadata);
        let final_output_index = sequence_index
            .checked_mul(token_count)
            .and_then(|base| base.checked_add(token_count.checked_sub(1)?))
            .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prefill final-hidden index overflowed".to_string(),
            })?;
        if final_output_index >= output_tokens {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prefill final-hidden index out of bounds".to_string(),
            });
        }
        let final_hidden_start = final_output_index.checked_mul(hidden_size).ok_or_else(|| {
            Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prefill final-hidden offset overflowed".to_string(),
            }
        })?;
        let final_hidden_end = final_hidden_start.checked_add(hidden_size).ok_or_else(|| {
            Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prefill final-hidden end overflowed".to_string(),
            }
        })?;
        let final_hidden = step
            .hidden
            .get(final_hidden_start..final_hidden_end)
            .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prefill final hidden out of bounds".to_string(),
            })?
            .to_vec();
        Ok(final_hidden)
    }

    fn prefill_prompt_hidden_at(
        &self,
        token_position: usize,
    ) -> Result<Vec<f32>, Qwen3AsrServeBatchError> {
        let hidden_size = self.job.prompt_input.hidden_size(self.job.metadata);
        let start = token_position.checked_mul(hidden_size).ok_or_else(|| {
            Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prefill hidden indexing overflowed".to_string(),
            }
        })?;
        let end = start.checked_add(hidden_size).ok_or_else(|| {
            Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prefill hidden indexing overflowed".to_string(),
            }
        })?;
        self.job
            .prompt_input
            .host()?
            .token_major_embeddings
            .get(start..end)
            .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prefill hidden slice out of bounds".to_string(),
            })
            .map(<[f32]>::to_vec)
    }

    fn run_prefill_and_select(
        &mut self,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        logits_runtime: &mut Qwen3AsrLlmLogitsHeadRuntime,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        self.run_prefill(decoder, logits_runtime)?;
        let logits =
            self.prefill_logits
                .take()
                .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch prefill produced no logits".to_string(),
                })?;
        self.select_next_token_from_logits(logits)
    }

    fn last_generated_token_id(&self) -> Result<u32, Qwen3AsrServeBatchError> {
        self.generated_tokens
            .last()
            .copied()
            .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch generated token history is unexpectedly empty"
                    .to_string(),
            })
    }

    fn ensure_generated_host_kv_replayed(
        &mut self,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        if decoder.layer_count() != self.layer_kv_caches.len() {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch decoder/cache layer count mismatch".to_string(),
            });
        }
        let target_prefix = self.reseed_host_kv_target_prefix()?;
        let mut written_prefix = self.host_kv_written_prefix()?;
        if written_prefix > target_prefix {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch host KV replay prefix moved backwards".to_string(),
            });
        }
        while written_prefix < target_prefix {
            let generated_index = written_prefix
                .checked_sub(self.cache_prompt_tokens)
                .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch host KV replay index underflowed".to_string(),
                })?;
            let token_id = *self.generated_tokens.get(generated_index).ok_or_else(|| {
                Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch host KV replay token index out of bounds".to_string(),
                }
            })?;
            let hidden = self
                .job
                .prepared_runtime()?
                .token_embedding_table
                .gather_rows(&[token_id])
                .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
                    reason: error.to_string(),
                })?;
            let step = decoder
                .run_step(
                    &hidden,
                    written_prefix,
                    &self.layer_kv_caches,
                    QWEN_ROPE_THETA,
                )
                .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
                    reason: error.to_string(),
                })?;
            self.write_replayed_host_kv_row(written_prefix, &step.layer_kv)?;
            written_prefix = written_prefix.checked_add(1).ok_or_else(|| {
                Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch host KV replay prefix overflowed".to_string(),
                }
            })?;
        }
        Ok(())
    }

    fn reseed_host_kv_target_prefix(&self) -> Result<usize, Qwen3AsrServeBatchError> {
        let replayed_generated = self.generated_tokens.len().saturating_sub(1);
        self.cache_prompt_tokens
            .checked_add(replayed_generated)
            .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch host KV replay target prefix overflowed".to_string(),
            })
    }

    fn host_kv_written_prefix(&self) -> Result<usize, Qwen3AsrServeBatchError> {
        let mut prefix = None;
        for cache in self.layer_kv_caches.iter() {
            let history = cache.full_history_storage().map_err(|reason| {
                Qwen3AsrServeBatchError::DecodeFailed {
                    reason: format!("qwen serve batch host KV replay cache invalid: {reason}"),
                }
            })?;
            match prefix {
                Some(expected) if expected != history.written_positions => {
                    return Err(Qwen3AsrServeBatchError::DecodeFailed {
                        reason: "qwen serve batch host KV replay layer prefix mismatch".to_string(),
                    });
                }
                Some(_) => {}
                None => prefix = Some(history.written_positions),
            }
        }
        prefix.ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
            reason: "qwen serve batch host KV replay has no layers".to_string(),
        })
    }

    fn write_replayed_host_kv_row(
        &mut self,
        position: usize,
        layer_kv: &[(Vec<f32>, Vec<f32>)],
    ) -> Result<(), Qwen3AsrServeBatchError> {
        if layer_kv.len() != self.layer_kv_caches.len() {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch host KV replay layer count mismatch".to_string(),
            });
        }
        for (layer_index, (projected_k, projected_v)) in layer_kv.iter().enumerate() {
            self.layer_kv_caches[layer_index]
                .write(position, projected_k, projected_v)
                .map_err(|reason| Qwen3AsrServeBatchError::DecodeFailed { reason })?;
        }
        Ok(())
    }

    fn next_cache_position(&self) -> Result<usize, Qwen3AsrServeBatchError> {
        self.cache_prompt_tokens
            .checked_add(self.generated_tokens.len())
            .and_then(|total| total.checked_sub(1))
            .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch cache position underflowed".to_string(),
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
    ) -> Result<(), Qwen3AsrServeBatchError> {
        serve_batch_select_and_apply_greedy_step(
            &self.job.decode_config,
            &mut self.generated_tokens,
            &mut self.generated_probabilities,
            &mut self.stop_reason,
            self.stop_token_ids.as_slice(),
            logits,
        )
        .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
            reason: error.to_string(),
        })
    }

    fn finish(self) -> Result<GgmlAsrExecutionResult, Qwen3AsrServeBatchError> {
        // A slot that never ran a step (admitted then drained) has no stop
        // reason; treat that as a normal completion rather than inventing a
        // truncation the decode never reported.
        let slot_stop_reason = self
            .stop_reason
            .unwrap_or(Seq2SeqGreedyDecodeStopReason::StopToken);
        let raw_text = self.decode_text_token_ids(&self.generated_tokens)?;
        let text = apply_seq2seq_text_postprocess(self.job.text_postprocess_kind, &raw_text)
            .trim()
            .to_string();
        let words = if self.job.word_timestamps {
            seq2seq_word_timestamps_from_generated_tokens(
                &self.generated_tokens,
                &self.generated_probabilities,
                0.0,
                self.job.audio_duration_seconds,
                self.job.text_postprocess_kind,
                &|token_ids| self.decode_text_token_ids(token_ids),
            )
            .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
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
                end: self.job.audio_duration_seconds,
                text: text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words,
            }]
        };
        Ok(GgmlAsrExecutionResult {
            transcription: Transcription {
                truncated_decodes: Vec::new(),
                unnamed_speakers: Vec::new(),
                text,
                segments,
                longform: None,
                language: None,
                ..Default::default()
            },
            carry_context: None,
            // Same reasoning as the single-utterance executor: no
            // intra-decode timestamps, so no honest second to anchor to.
            decode_truncation: slot_stop_reason.into_decode_truncation(None),
        })
    }

    fn decode_text_token_ids(&self, token_ids: &[u32]) -> Result<String, Qwen3AsrServeBatchError> {
        if let Some(tokenizer) = self.job.prepared_runtime()?.tokenizer.as_ref() {
            return tokenizer.decode_text_token_ids(token_ids).map_err(|error| {
                Qwen3AsrServeBatchError::DecodeFailed {
                    reason: error.to_string(),
                }
            });
        }
        Ok(token_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{GgmlCpuGraphConfig, GgufTensorDataReader};
    use crate::models::qwen::runtime_contract::parse_qwen3_execution_metadata;
    use crate::models::qwen::tensor_names::{
        OUTPUT_NORM_WEIGHT, OUTPUT_WEIGHT, TOKEN_EMBD_WEIGHT, llm_layer_tensor_names,
    };
    use crate::models::qwen::{
        load_qwen3_llm_attention_projections_from_reader, load_qwen3_llm_logits_head_from_reader,
        load_qwen3_token_embedding_table_from_reader,
    };
    use crate::models::serve_batch_env::OPENASR_SERVE_BATCH_ENV;
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
    use crate::validate_ggml_runtime_source_path;
    use std::{collections::BTreeMap, ffi::OsString};

    const QWEN_SERVE_BATCH_REAL_PACK_ENV: &str = "OPENASR_QWEN_SERVE_BATCH_REAL_PACK";

    #[test]
    fn qwen_pending_reuse_shutdown_marks_dropped_and_incomplete() {
        let services = crate::models::native_execution_services::test_native_execution_services();
        let _guard = crate::models::native_execution_services::install_native_execution_services(
            services.as_ref(),
        );
        let key = Qwen3AsrServeBatchEngineKey {
            build_identity: crate::RuntimeBuildIdentity::resolve_for_request(
                None,
                "qwen:test-shutdown",
                "adapter=none",
                "qwen-shutdown-receipt-test",
            ),
            lane: crate::models::native_execution_services::current_execution_lane_key(
                GgmlCpuGraphBackend::Cpu,
            ),
            resident_positions: 8,
            max_batch: 2,
            native_gqa: GgmlNativeGqaCapability::Validated,
        };
        let config = Qwen3AsrServeBatchConfig {
            max_batch: 2,
            queue_capacity: 2,
            collect_window: Duration::ZERO,
            send_timeout: Duration::from_secs(1),
            reply_timeout: Duration::from_secs(1),
            trace_batches: false,
        };
        let engine = Qwen3AsrServeBatchEngine::spawn(key, config).expect("qwen owner");
        while !engine
            .owner_ready
            .load(std::sync::atomic::Ordering::Acquire)
        {
            std::thread::yield_now();
        }
        engine.record_receipt_reuse();
        engine.record_receipt_reuse();
        engine.shutdown_owner();
        let completeness = services.runtime_receipts().summary().completeness;
        assert!(completeness.dropped_notifications > 0);
        assert!(!completeness.complete);
    }

    #[test]
    fn qwen_serve_batch_concurrent_same_key_has_one_owner_receipt() {
        let services = crate::models::native_execution_services::test_native_execution_services();
        let registry = Arc::new(Qwen3AsrServeBatchEngineRegistry::default());
        let key = Qwen3AsrServeBatchEngineKey {
            build_identity: crate::RuntimeBuildIdentity::resolve_for_request(
                None,
                "qwen:test",
                "adapter=none",
                "qwen-receipt-test",
            ),
            lane: crate::models::native_execution_services::current_execution_lane_key(
                GgmlCpuGraphBackend::Cpu,
            ),
            resident_positions: 8,
            max_batch: 2,
            native_gqa: GgmlNativeGqaCapability::Validated,
        };
        let config = Qwen3AsrServeBatchConfig {
            max_batch: 2,
            queue_capacity: 2,
            collect_window: Duration::ZERO,
            send_timeout: Duration::from_secs(1),
            reply_timeout: Duration::from_secs(1),
            trace_batches: false,
        };
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let first_registry = Arc::clone(&registry);
        let first_services = Arc::clone(&services);
        let first_barrier = Arc::clone(&barrier);
        let first_key = key.clone();
        let first = std::thread::spawn(move || {
            let _guard =
                crate::models::native_execution_services::install_native_execution_services(
                    first_services.as_ref(),
                );
            first_barrier.wait();
            qwen_serve_batch_engine_for_key(&first_registry, first_key, config)
                .expect("qwen first owner")
        });
        let second_registry = Arc::clone(&registry);
        let second_services = Arc::clone(&services);
        let second_barrier = Arc::clone(&barrier);
        let second_key = key;
        let second = std::thread::spawn(move || {
            let _guard =
                crate::models::native_execution_services::install_native_execution_services(
                    second_services.as_ref(),
                );
            second_barrier.wait();
            qwen_serve_batch_engine_for_key(&second_registry, second_key, config)
                .expect("qwen second owner")
        });
        barrier.wait();
        let first_engine = first.join().expect("qwen first lookup");
        let second_engine = second.join().expect("qwen second lookup");
        assert!(Arc::ptr_eq(&first_engine, &second_engine));
        first_engine.shutdown_owner();
        drop(second_engine);
        drop(first_engine);
        registry.engines.shutdown();
        let snapshot = services.runtime_receipts().snapshot();
        assert_eq!(
            snapshot
                .events
                .iter()
                .filter(|event| {
                    matches!(
                        event,
                        crate::models::runtime_receipts::RuntimeReceiptEvent::OwnerCreated { .. }
                    )
                })
                .count(),
            1
        );
    }

    #[test]
    fn serve_batch_error_classifies_transient_failures() {
        assert_eq!(
            Qwen3AsrServeBatchError::QueueFull.unavailable_retryable(),
            Some(true)
        );
        assert_eq!(
            Qwen3AsrServeBatchError::OwnerDisconnected.unavailable_retryable(),
            Some(false)
        );
        assert_eq!(
            Qwen3AsrServeBatchError::ReplyTimedOut.unavailable_retryable(),
            Some(false)
        );
        assert_eq!(
            Qwen3AsrServeBatchError::DecodeFailed {
                reason: "boom".to_string()
            }
            .unavailable_retryable(),
            None
        );
        assert_eq!(
            Qwen3AsrServeBatchError::OwnerFailed {
                reason: "boom".to_string()
            }
            .unavailable_retryable(),
            None
        );
    }
    /// Structural proof that the job carries the cancellation context across
    /// the owner-thread boundary. Host-KV memory is owned by the slot itself.
    #[allow(dead_code)]
    fn require_concrete_execution_context(_: Arc<crate::RequestExecutionContext>) {}

    #[allow(dead_code)]
    fn assert_qwen_serve_batch_job_requires_execution_context(job: Qwen3AsrServeBatchJob) {
        let Qwen3AsrServeBatchJob {
            execution_context, ..
        } = job;
        require_concrete_execution_context(execution_context);
    }

    const QWEN_PREFILL_REAL_PACK_ENV: &str = "OPENASR_QWEN_PREFILL_REAL_PACK";

    struct Qwen3AsrServeBatchFixture {
        runtime_path: PathBuf,
        runtime_source: crate::GgmlRuntimeSource,
        runtime_source_preflight: crate::GgufRuntimeSourcePreflight,
        metadata: Qwen3AsrExecutionMetadata,
        token_embedding_table: MappedTokenEmbeddingTable,
        logits_head: Qwen3AsrLlmLogitsHead,
        decoder_plan: Arc<QwenWholeDecoderPlan>,
        layer_attention_projections: Arc<Vec<Qwen3AsrLlmLayerAttentionProjection>>,
        prompt_tokens: Vec<u32>,
    }

    fn with_serve_batch_env<T>(value: Option<&str>, run: impl FnOnce() -> T) -> T {
        crate::test_process_env::with_test_process_env(
            [(OPENASR_SERVE_BATCH_ENV, value.map(OsString::from))],
            run,
        )
    }

    fn tiny_metadata() -> Qwen3AsrExecutionMetadata {
        Qwen3AsrExecutionMetadata {
            sample_rate_hz: 16_000,
            n_mels: 8,
            n_fft: 400,
            win_length: 400,
            hop_length: 160,
            audio_layers: 1,
            audio_d_model: 8,
            audio_heads: 1,
            llm_layers: 2,
            llm_d_model: 8,
            llm_heads: 1,
            llm_kv_heads: 1,
            llm_head_dim: 4,
            vocab_size: 16,
            llm_max_positions: 8,
            audio_start_token_id: 1,
            audio_end_token_id: 2,
            audio_pad_token_id: 3,
            eos_token_id: 0,
            pad_token_id: 4,
        }
    }

    fn qwen_serve_batch_real_pack_path() -> PathBuf {
        std::env::var_os(QWEN_SERVE_BATCH_REAL_PACK_ENV)
            .or_else(|| std::env::var_os(QWEN_PREFILL_REAL_PACK_ENV))
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!(
                    "{QWEN_SERVE_BATCH_REAL_PACK_ENV} or {QWEN_PREFILL_REAL_PACK_ENV} must point to a qwen .oasr model pack"
                )
            })
    }

    fn load_qwen_serve_batch_fixture_from_path(runtime_path: PathBuf) -> Qwen3AsrServeBatchFixture {
        let runtime_source =
            validate_ggml_runtime_source_path(&runtime_path).expect("valid qwen runtime source");
        let runtime_source_preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index_from_source(
                &runtime_source,
            )
            .expect("qwen runtime preflight");
        let metadata = runtime_source_preflight.metadata.as_ref();
        let metadata = parse_qwen3_execution_metadata(metadata).expect("parse qwen metadata");
        let reader =
            GgufTensorDataReader::from_runtime_source(&runtime_source).expect("qwen tensor reader");
        let token_embedding_table = load_qwen3_token_embedding_table_from_reader(&reader, metadata)
            .expect("qwen token embeddings");
        let logits_head = load_qwen3_llm_logits_head_from_reader(
            &reader,
            &runtime_source,
            metadata,
            GgmlCpuGraphConfig::runtime_default().backend,
        )
        .expect("qwen logits head");
        let decoder_plan = Arc::new(
            QwenWholeDecoderPlan::for_qwen3_asr(&reader, metadata)
                .expect("qwen whole-decoder plan"),
        );
        let layer_attention_projections = Arc::new(
            load_qwen3_llm_attention_projections_from_reader(&reader, metadata)
                .expect("qwen llm layers"),
        );
        let prompt_tokens = vec![
            metadata.audio_start_token_id,
            metadata.audio_pad_token_id,
            metadata.audio_end_token_id,
            metadata.pad_token_id,
        ];
        for &token_id in &prompt_tokens {
            assert!(
                usize::try_from(token_id)
                    .ok()
                    .is_some_and(|idx| idx < metadata.vocab_size),
                "qwen prompt token {token_id} must be in vocab_size={}",
                metadata.vocab_size
            );
        }
        Qwen3AsrServeBatchFixture {
            runtime_path,
            runtime_source,
            runtime_source_preflight,
            metadata,
            token_embedding_table,
            logits_head,
            decoder_plan,
            layer_attention_projections,
            prompt_tokens,
        }
    }

    fn load_qwen_serve_batch_real_pack_fixture() -> Qwen3AsrServeBatchFixture {
        load_qwen_serve_batch_fixture_from_path(qwen_serve_batch_real_pack_path())
    }

    fn qwen_tiny_metadata_with_llm_layers(llm_layers: usize) -> BTreeMap<String, String> {
        let mut metadata = BTreeMap::new();
        metadata.insert("general.architecture".to_string(), "qwen3-asr".to_string());
        metadata.insert("qwen3-asr.sample_rate".to_string(), "16000".to_string());
        metadata.insert("qwen3-asr.n_mels".to_string(), "8".to_string());
        metadata.insert("qwen3-asr.n_fft".to_string(), "400".to_string());
        metadata.insert("qwen3-asr.win_length".to_string(), "400".to_string());
        metadata.insert("qwen3-asr.hop_length".to_string(), "160".to_string());
        metadata.insert("qwen3-asr.audio.n_layers".to_string(), "1".to_string());
        metadata.insert("qwen3-asr.audio.d_model".to_string(), "16".to_string());
        metadata.insert("qwen3-asr.audio.n_heads".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.d_model".to_string(), "16".to_string());
        metadata.insert("qwen3-asr.llm.n_heads".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.n_kv_heads".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.head_dim".to_string(), "8".to_string());
        metadata.insert("qwen3-asr.llm.n_layers".to_string(), llm_layers.to_string());
        metadata.insert("qwen3-asr.llm.vocab_size".to_string(), "32".to_string());
        metadata.insert("qwen3-asr.llm.max_pos".to_string(), "256".to_string());
        metadata.insert(
            "qwen3-asr.audio_start_token_id".to_string(),
            "2".to_string(),
        );
        metadata.insert("qwen3-asr.audio_end_token_id".to_string(), "3".to_string());
        metadata.insert("qwen3-asr.audio_pad_token_id".to_string(), "4".to_string());
        metadata.insert("qwen3-asr.eos_token_id".to_string(), "0".to_string());
        metadata.insert("qwen3-asr.pad_token_id".to_string(), "6".to_string());
        metadata
    }

    fn add_qwen_tiny_llm_layer_shapes(
        spec: TinyGgufFixtureSpec,
        layer_idx: usize,
    ) -> TinyGgufFixtureSpec {
        let names = llm_layer_tensor_names(layer_idx);
        spec.with_tensor_shape(names.attn_norm_weight, [16_u64])
            .with_tensor_shape(names.attn_q_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_k_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_v_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_output_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_q_norm_weight, [8_u64])
            .with_tensor_shape(names.attn_k_norm_weight, [8_u64])
            .with_tensor_shape(names.ffn_norm_weight, [16_u64])
            // ggml [in, out]: gate/up = [d_model, ffn_dim], down = [ffn_dim, d_model]
            .with_tensor_shape(names.ffn_gate_weight, [16_u64, 32_u64])
            .with_tensor_shape(names.ffn_up_weight, [16_u64, 32_u64])
            .with_tensor_shape(names.ffn_down_weight, [32_u64, 16_u64])
    }

    fn qwen_tiny_serve_batch_fixture_spec(llm_layers: usize) -> TinyGgufFixtureSpec {
        let mut spec = TinyGgufFixtureSpec::new(qwen_tiny_metadata_with_llm_layers(llm_layers))
            .with_tensor_shape(TOKEN_EMBD_WEIGHT, [16_u64, 32_u64])
            .with_tensor_shape(OUTPUT_WEIGHT, [16_u64, 32_u64])
            .with_tensor_shape(OUTPUT_NORM_WEIGHT, [16_u64]);
        for layer_idx in 0..llm_layers {
            spec = add_qwen_tiny_llm_layer_shapes(spec, layer_idx);
        }
        spec
    }

    fn write_qwen_tiny_serve_batch_fixture() -> (tempfile::TempDir, Qwen3AsrServeBatchFixture) {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen3-asr-tiny.gguf");
        let fixture_spec = qwen_tiny_serve_batch_fixture_spec(2);
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).expect("write qwen fixture");
        let fixture = load_qwen_serve_batch_fixture_from_path(runtime_path);
        (temp, fixture)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn qwen_tiny_device_token_decoder(
        fixture: &Qwen3AsrServeBatchFixture,
    ) -> Qwen3AsrLlmWholeDecoderGraphExecutor {
        let token_embedding = fixture
            .token_embedding_table
            .device_graph_spec()
            .expect("tiny qwen fixture should expose canonical device token embeddings");
        Qwen3AsrLlmWholeDecoderGraphExecutor::new_from_plan_with_preflight_and_token_embedding(
            &fixture.decoder_plan,
            &fixture.runtime_source_preflight,
            token_embedding,
            crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                Some(crate::ggml_runtime::RequestBackendPreference::Accelerated),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
        )
        .expect("tiny qwen Metal decoder should compile")
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn assert_qwen_hidden_close(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        let max_abs = actual
            .iter()
            .zip(expected)
            .map(|(&actual, &expected)| {
                assert!(
                    actual.is_finite() && expected.is_finite(),
                    "{label} produced non-finite output"
                );
                (actual - expected).abs()
            })
            .fold(0.0_f32, f32::max);
        assert!(max_abs <= 1.0e-5, "{label} diverged: max_abs={max_abs:.9}");
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn qwen_device_token_lookup_matches_host_gather_and_preserves_kv_across_input_modes() {
        let (_temp, fixture) = write_qwen_tiny_serve_batch_fixture();
        let first_token = fixture.prompt_tokens[0];
        let second_token = fixture.prompt_tokens[1];
        let first_hidden = fixture
            .token_embedding_table
            .gather_rows(&[first_token])
            .expect("first host token embedding");
        let second_hidden = fixture
            .token_embedding_table
            .gather_rows(&[second_token])
            .expect("second host token embedding");
        let max_positions = 8;

        let mut hidden_reference = qwen_tiny_device_token_decoder(&fixture);
        let reference_first = hidden_reference
            .run_step_reused_batched(&first_hidden, &[0], QWEN_ROPE_THETA, max_positions)
            .expect("reference hidden step zero");
        let reference_second = hidden_reference
            .run_step_reused_batched(&second_hidden, &[1], QWEN_ROPE_THETA, max_positions)
            .expect("reference hidden step one");

        let mut hidden_to_token = qwen_tiny_device_token_decoder(&fixture);
        let switched_first = hidden_to_token
            .run_step_reused_batched(&first_hidden, &[0], QWEN_ROPE_THETA, max_positions)
            .expect("hidden-to-token hidden step");
        let switched_second = hidden_to_token
            .run_token_step_reused_batched(&[second_token], &[1], QWEN_ROPE_THETA, max_positions)
            .expect("hidden-to-token token step");
        assert_qwen_hidden_close(
            "hidden-to-token first step",
            &switched_first.hidden,
            &reference_first.hidden,
        );
        assert_qwen_hidden_close(
            "hidden-to-token second step",
            &switched_second.hidden,
            &reference_second.hidden,
        );
        assert_eq!(hidden_to_token.reused_batch_width_for_test(), Some(1));

        let mut token_to_hidden = qwen_tiny_device_token_decoder(&fixture);
        let reverse_first = token_to_hidden
            .run_token_step_reused_batched(&[first_token], &[0], QWEN_ROPE_THETA, max_positions)
            .expect("token-to-hidden token step");
        let reverse_second = token_to_hidden
            .run_step_reused_batched(&second_hidden, &[1], QWEN_ROPE_THETA, max_positions)
            .expect("token-to-hidden hidden step");
        assert_qwen_hidden_close(
            "token-to-hidden first step",
            &reverse_first.hidden,
            &reference_first.hidden,
        );
        assert_qwen_hidden_close(
            "token-to-hidden second step",
            &reverse_second.hidden,
            &reference_second.hidden,
        );
        assert_eq!(token_to_hidden.reused_batch_width_for_test(), Some(1));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn qwen_device_prompt_materialization_matches_sparse_host_splice() {
        let (_temp, fixture) = write_qwen_tiny_serve_batch_fixture();
        assert!(fixture.prompt_tokens.len() >= 2);
        let d_model = fixture.metadata.llm_d_model;
        let audio_positions = vec![0, fixture.prompt_tokens.len() - 1];
        let mut audio_rows = vec![0.0_f32; audio_positions.len() * d_model];
        for (index, value) in audio_rows.iter_mut().enumerate() {
            *value = 0.25 + index as f32 * 0.03125;
        }
        let host_rows = fixture
            .token_embedding_table
            .gather_rows(&fixture.prompt_tokens)
            .expect("host prompt rows");
        let expected =
            super::super::prompt_embedding::build_qwen3_prompt_embeddings_with_audio_positions(
                fixture.prompt_tokens.len(),
                &audio_positions,
                d_model,
                host_rows,
                &audio_rows,
            )
            .expect("host sparse splice");
        let mut decoder = qwen_tiny_device_token_decoder(&fixture);
        let actual = decoder
            .materialize_token_prompt_on_device(
                &fixture.prompt_tokens,
                &audio_rows,
                &audio_positions,
            )
            .expect("device prompt graph")
            .expect("Metal prompt materialization");
        assert_qwen_hidden_close(
            "device prompt sparse splice",
            &actual,
            &expected.token_major_values,
        );
    }

    fn with_qwen_direct_cpu_backend_for_test<T>(run: impl FnOnce() -> T) -> T {
        // Flattened into one multi-key override rather than nesting
        // `with_forced_cpu_backend_for_test` inside a second env guard: the
        // process env lock is not reentrant, so two nested guards on the same
        // thread would self-deadlock on the second `lock()` call.
        crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_GGML_BACKEND", Some(OsString::from("cpu"))),
                (
                    GgmlCpuGraphConfig::USE_SCHEDULER_ENV,
                    Some(OsString::from("0")),
                ),
            ],
            run,
        )
    }

    fn qwen_real_pack_prefill_input(
        token_embedding_table: &MappedTokenEmbeddingTable,
        prompt_tokens: &[u32],
    ) -> Qwen3AsrLlmPrefillInput {
        let token_count = prompt_tokens.len();
        let hidden_size = token_embedding_table.d_model();
        let token_major_embeddings = token_embedding_table
            .gather_rows(prompt_tokens)
            .expect("qwen prompt embeddings");
        Qwen3AsrLlmPrefillInput {
            token_count,
            hidden_size,
            token_major_embeddings,
        }
    }

    fn qwen_fixture_job(
        fixture: &Qwen3AsrServeBatchFixture,
        max_generated_tokens: usize,
    ) -> Qwen3AsrServeBatchJob {
        let logical_positions = crate::capacity::decode_schedule::greedy_self_kv_positions(
            fixture.prompt_tokens.len(),
            max_generated_tokens,
        )
        .expect("fixture decode schedule");
        qwen_fixture_job_with_resident(fixture, max_generated_tokens, logical_positions)
    }

    fn qwen_fixture_job_with_resident(
        fixture: &Qwen3AsrServeBatchFixture,
        max_generated_tokens: usize,
        resident_positions: usize,
    ) -> Qwen3AsrServeBatchJob {
        let runtime_source = crate::validate_ggml_runtime_source_path(&fixture.runtime_path)
            .expect("valid runtime source path");
        let logical_positions = crate::capacity::decode_schedule::greedy_self_kv_positions(
            fixture.prompt_tokens.len(),
            max_generated_tokens,
        )
        .expect("fixture decode schedule");
        Qwen3AsrServeBatchJob {
            runtime_cache_path: fixture.runtime_path.clone(),
            build_identity: crate::RuntimeBuildIdentity::resolve_for_request(
                None,
                "qwen:test",
                "adapter=none",
                runtime_source.content_id(),
            ),
            runtime_source_preflight: fixture.runtime_source_preflight.clone(),
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                None,
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            native_gqa: GgmlNativeGqaCapability::Validated,
            metadata: fixture.metadata,
            prepared_assets: Qwen3AsrServeBatchPreparedAssets::Fixture {
                tokenizer: None,
                token_embedding_table: Arc::new(fixture.token_embedding_table.clone()),
                logits_head: Arc::new(fixture.logits_head.clone()),
                decoder_plan: Arc::clone(&fixture.decoder_plan),
            },
            prompt_input: Qwen3AsrServeBatchPromptInput::Host(qwen_real_pack_prefill_input(
                &fixture.token_embedding_table,
                &fixture.prompt_tokens,
            )),
            kv_capacity: Qwen3AsrKvCacheCapacity::new(logical_positions, resident_positions)
                .expect("fixture KV capacity"),
            decode_config: Seq2SeqGreedyDecodeConfig {
                initial_prompt_tokens: fixture.prompt_tokens.clone(),
                eot_token_id: u32::MAX,
                stop_token_ids: Vec::new(),
                vocab_size: fixture.metadata.vocab_size,
                max_generated_tokens,
                suppress_first_step_token_ids: Vec::new(),
                suppress_token_ids: Vec::new(),
                phrase_biases: Vec::new(),
            },
            text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
            word_timestamps: false,
            audio_duration_seconds: 1.0,
            execution_context: Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        }
    }

    fn qwen_fixture_envelope(
        fixture: &Qwen3AsrServeBatchFixture,
        max_generated_tokens: usize,
    ) -> (
        Qwen3AsrServeBatchEnvelope,
        mpsc::Receiver<Result<GgmlAsrExecutionResult, Qwen3AsrServeBatchError>>,
    ) {
        let job = qwen_fixture_job(fixture, max_generated_tokens);
        let (reply, reply_rx) = mpsc::channel();
        (
            Qwen3AsrServeBatchEnvelope {
                job,
                native_execution_context: None,
                reply,
            },
            reply_rx,
        )
    }

    fn qwen_fixture_envelope_with_resident(
        fixture: &Qwen3AsrServeBatchFixture,
        max_generated_tokens: usize,
        resident_positions: usize,
    ) -> (
        Qwen3AsrServeBatchEnvelope,
        mpsc::Receiver<Result<GgmlAsrExecutionResult, Qwen3AsrServeBatchError>>,
    ) {
        let job = qwen_fixture_job_with_resident(fixture, max_generated_tokens, resident_positions);
        let (reply, reply_rx) = mpsc::channel();
        (
            Qwen3AsrServeBatchEnvelope {
                job,
                native_execution_context: None,
                reply,
            },
            reply_rx,
        )
    }

    fn assert_qwen_selected_backend_direct_for_real_pack_harness() {
        // Manual harness: not run through the dispatch, so resolve the
        // backend directly here, using qwen's own (AllBackends) declared
        // policy with no request-level override.
        let backend = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
            None,
            crate::ggml_runtime::AutoGpuPolicy::AllBackends,
        )
        .backend();
        let runtime_config = qwen_runtime_graph_config(backend);
        assert!(
            runtime_config.backend.is_gpu_class() && !runtime_config.use_scheduler,
            "qwen owner rebucket/shrink real-pack harness validates the direct GPU reusable graph, got backend={:?} use_scheduler={}",
            runtime_config.backend,
            runtime_config.use_scheduler
        );
    }

    fn qwen_prefilled_active_slot(
        fixture: &Qwen3AsrServeBatchFixture,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        logits_runtime: &mut Qwen3AsrLlmLogitsHeadRuntime,
        max_positions: usize,
    ) -> Qwen3AsrActiveBatchSlot {
        qwen_prefilled_active_slot_with_token_cap(
            fixture,
            decoder,
            logits_runtime,
            max_positions,
            4,
        )
    }

    fn qwen_prefilled_active_slot_with_token_cap(
        fixture: &Qwen3AsrServeBatchFixture,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        logits_runtime: &mut Qwen3AsrLlmLogitsHeadRuntime,
        max_positions: usize,
        max_generated_tokens: usize,
    ) -> Qwen3AsrActiveBatchSlot {
        qwen_prefilled_active_slot_with_token_cap_and_resident(
            fixture,
            decoder,
            logits_runtime,
            max_generated_tokens,
            max_positions,
        )
    }

    fn qwen_prefilled_active_slot_with_token_cap_and_resident(
        fixture: &Qwen3AsrServeBatchFixture,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        logits_runtime: &mut Qwen3AsrLlmLogitsHeadRuntime,
        max_generated_tokens: usize,
        resident_positions: usize,
    ) -> Qwen3AsrActiveBatchSlot {
        let mut slot = Qwen3AsrBatchSlot::new(
            qwen_fixture_job_with_resident(fixture, max_generated_tokens, resident_positions),
            resident_positions,
        )
        .expect("qwen slot");
        slot.run_prefill_and_select(decoder, logits_runtime)
            .expect("qwen slot prefill");
        let (reply, _reply_rx) = mpsc::channel();
        Qwen3AsrActiveBatchSlot {
            slot,
            native_execution_context: None,
            reply,
        }
    }

    fn assert_qwen_rebucket_migration(fixture: &Qwen3AsrServeBatchFixture) {
        let max_positions = crate::capacity::decode_schedule::greedy_self_kv_positions(
            fixture.prompt_tokens.len(),
            4,
        )
        .expect("fixture decode schedule");
        let mut decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new(
            fixture.layer_attention_projections.as_slice(),
            Some(&fixture.runtime_source),
            GgmlCpuGraphConfig::runtime_default().backend,
        )
        .expect("qwen decoder");
        let mut logits_runtime = fixture
            .logits_head
            .new_runtime(GgmlCpuGraphConfig::runtime_default().backend)
            .expect("qwen logits runtime");
        let mut slots = vec![
            Some(qwen_prefilled_active_slot(
                fixture,
                &mut decoder,
                &mut logits_runtime,
                max_positions,
            )),
            Some(qwen_prefilled_active_slot(
                fixture,
                &mut decoder,
                &mut logits_runtime,
                max_positions,
            )),
        ];
        Qwen3AsrOwnerThreadState::reseed_rebucketed_slots(&mut slots, &mut decoder, max_positions)
            .expect("initial qwen seed");
        assert_eq!(decoder.reused_batch_width_for_test(), Some(2));

        // Rebucket exercises migration of live slots. A one-token budget is
        // already complete after prefill selects its first token, so use the
        // resident four-token budget to keep both refill slots active.
        let (queued_live_a, _queued_live_a_rx) = qwen_fixture_envelope(fixture, 4);
        let (queued_live_b, _queued_live_b_rx) = qwen_fixture_envelope(fixture, 4);
        let (queued_tx, queued_rx) = mpsc::sync_channel(2);
        queued_tx.send(queued_live_a).expect("queue qwen refill a");
        queued_tx.send(queued_live_b).expect("queue qwen refill b");
        let mut deferred = VecDeque::new();
        Qwen3AsrOwnerThreadState::try_rebucket_active_slots(
            &mut slots,
            &mut decoder,
            &mut logits_runtime,
            max_positions,
            &queued_rx,
            &mut deferred,
            4,
            false,
        )
        .expect("qwen rebucket");
        assert!(deferred.is_empty());
        assert_eq!(slots.len(), 4);
        assert_eq!(slots.iter().filter(|slot| slot.is_some()).count(), 4);
        assert_eq!(decoder.reused_batch_width_for_test(), Some(4));

        slots[2] = None;
        slots[3] = None;
        Qwen3AsrOwnerThreadState::try_shrink_active_slots(
            &mut slots,
            &mut decoder,
            max_positions,
            4,
            false,
        )
        .expect("qwen shrink after rebucket");
        assert_eq!(slots.len(), 2);
        assert_eq!(slots.iter().filter(|slot| slot.is_some()).count(), 2);
        assert_eq!(decoder.reused_batch_width_for_test(), Some(2));
    }

    fn assert_qwen_tail_shrink_migration(fixture: &Qwen3AsrServeBatchFixture) {
        let max_positions = crate::capacity::decode_schedule::greedy_self_kv_positions(
            fixture.prompt_tokens.len(),
            4,
        )
        .expect("fixture decode schedule");
        let mut decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new(
            fixture.layer_attention_projections.as_slice(),
            Some(&fixture.runtime_source),
            GgmlCpuGraphConfig::runtime_default().backend,
        )
        .expect("qwen decoder");
        let mut logits_runtime = fixture
            .logits_head
            .new_runtime(GgmlCpuGraphConfig::runtime_default().backend)
            .expect("qwen logits runtime");
        let mut slots = vec![
            Some(qwen_prefilled_active_slot(
                fixture,
                &mut decoder,
                &mut logits_runtime,
                max_positions,
            )),
            Some(qwen_prefilled_active_slot(
                fixture,
                &mut decoder,
                &mut logits_runtime,
                max_positions,
            )),
            Some(qwen_prefilled_active_slot(
                fixture,
                &mut decoder,
                &mut logits_runtime,
                max_positions,
            )),
            None,
        ];
        Qwen3AsrOwnerThreadState::reseed_rebucketed_slots(&mut slots, &mut decoder, max_positions)
            .expect("initial qwen padded seed");
        assert_eq!(decoder.reused_batch_width_for_test(), Some(4));

        slots[0] = None;
        slots[1] = None;
        Qwen3AsrOwnerThreadState::try_shrink_active_slots(
            &mut slots,
            &mut decoder,
            max_positions,
            4,
            false,
        )
        .expect("qwen shrink");
        assert_eq!(slots.len(), 1);
        assert_eq!(slots.iter().filter(|slot| slot.is_some()).count(), 1);
        assert_eq!(decoder.reused_batch_width_for_test(), Some(1));
    }

    fn assert_qwen_stable_resident_span_accepts_distinct_logical_spans(
        fixture: &Qwen3AsrServeBatchFixture,
    ) {
        let short_logical_positions = crate::capacity::decode_schedule::greedy_self_kv_positions(
            fixture.prompt_tokens.len(),
            2,
        )
        .expect("fixture decode schedule");
        let resident_positions = crate::capacity::decode_schedule::greedy_self_kv_positions(
            fixture.prompt_tokens.len(),
            4,
        )
        .expect("fixture decode schedule");
        let mut decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new(
            fixture.layer_attention_projections.as_slice(),
            Some(&fixture.runtime_source),
            GgmlCpuGraphConfig::runtime_default().backend,
        )
        .expect("qwen decoder");
        let mut logits_runtime = fixture
            .logits_head
            .new_runtime(GgmlCpuGraphConfig::runtime_default().backend)
            .expect("qwen logits runtime");
        let mut slots = vec![
            Some(qwen_prefilled_active_slot_with_token_cap(
                fixture,
                &mut decoder,
                &mut logits_runtime,
                resident_positions,
                2,
            )),
            Some(qwen_prefilled_active_slot_with_token_cap(
                fixture,
                &mut decoder,
                &mut logits_runtime,
                resident_positions,
                2,
            )),
        ];
        Qwen3AsrOwnerThreadState::reseed_rebucketed_slots(
            &mut slots,
            &mut decoder,
            resident_positions,
        )
        .expect("initial qwen seed");
        assert_eq!(decoder.reused_batch_width_for_test(), Some(2));
        for active in slots.iter().flatten() {
            assert_eq!(
                active.slot.layer_kv_caches[0].max_positions(),
                short_logical_positions
            );
        }

        let (queued_long_a, _queued_long_a_rx) =
            qwen_fixture_envelope_with_resident(fixture, 4, resident_positions);
        let (queued_long_b, _queued_long_b_rx) =
            qwen_fixture_envelope_with_resident(fixture, 4, resident_positions);
        let (queued_tx, queued_rx) = mpsc::sync_channel(2);
        queued_tx.send(queued_long_a).expect("queue long qwen a");
        queued_tx.send(queued_long_b).expect("queue long qwen b");
        let mut deferred = VecDeque::new();
        Qwen3AsrOwnerThreadState::try_rebucket_active_slots(
            &mut slots,
            &mut decoder,
            &mut logits_runtime,
            resident_positions,
            &queued_rx,
            &mut deferred,
            4,
            false,
        )
        .expect("qwen rebucket within stable resident span");
        assert!(deferred.is_empty());
        assert_eq!(slots.len(), 4);
        assert_eq!(slots.iter().filter(|slot| slot.is_some()).count(), 4);
        assert_eq!(decoder.reused_batch_width_for_test(), Some(4));
        assert_eq!(
            slots[0].as_ref().expect("short slot").slot.layer_kv_caches[0].max_positions(),
            short_logical_positions
        );
        assert_eq!(
            slots[2].as_ref().expect("long slot").slot.layer_kv_caches[0].max_positions(),
            resident_positions
        );
    }

    fn assert_qwen_generated_host_kv_replay(fixture: &Qwen3AsrServeBatchFixture) {
        let max_positions = crate::capacity::decode_schedule::greedy_self_kv_positions(
            fixture.prompt_tokens.len(),
            4,
        )
        .expect("fixture decode schedule");
        let mut decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new(
            fixture.layer_attention_projections.as_slice(),
            Some(&fixture.runtime_source),
            GgmlCpuGraphConfig::runtime_default().backend,
        )
        .expect("qwen decoder");
        let mut logits_runtime = fixture
            .logits_head
            .new_runtime(GgmlCpuGraphConfig::runtime_default().backend)
            .expect("qwen logits runtime");
        let mut slot =
            Qwen3AsrBatchSlot::new(qwen_fixture_job(fixture, 4), max_positions).expect("qwen slot");
        slot.run_prefill_and_select(&mut decoder, &mut logits_runtime)
            .expect("qwen slot prefill");
        assert_eq!(
            slot.host_kv_written_prefix().expect("written prefix"),
            fixture.prompt_tokens.len()
        );
        let first_generated = *slot
            .generated_tokens
            .first()
            .expect("prefill should select one generated token");
        slot.generated_tokens.push(first_generated);

        slot.ensure_generated_host_kv_replayed(&mut decoder)
            .expect("qwen generated host KV replay");
        assert_eq!(
            slot.host_kv_written_prefix().expect("written prefix"),
            crate::capacity::decode_schedule::greedy_self_kv_positions(
                fixture.prompt_tokens.len(),
                2,
            )
            .expect("fixture decode schedule")
        );
    }

    #[test]
    fn qwen_dummy_seed_layers_initialize_zero_prefix_for_padded_slots() {
        let layers = Qwen3AsrBatchSlot::zero_seed_layer_kv_caches(
            tiny_metadata(),
            GgmlCpuGraphBackend::Cpu,
            8,
        )
        .expect("dummy seed");
        assert_eq!(layers.len(), 2);
        for layer in layers.iter() {
            let snapshot = layer.snapshot_written().expect("snapshot");
            assert_eq!(snapshot.written_positions, 1);
            assert_eq!(snapshot.key_width, 4);
            assert_eq!(snapshot.value_width, 4);
            let history = layer.full_history_storage().expect("history");
            assert_eq!(history.written_positions, 1);
            let keys = history.keys_f32.expect("f32 keys");
            let values = history.values_f32.expect("f32 values");
            assert!(keys.iter().all(|&value| value == 0.0));
            assert!(values.iter().all(|&value| value == 0.0));
        }
    }

    #[test]
    fn qwen_serve_batch_env_defaults_off() {
        with_serve_batch_env(None, || {
            assert!(Qwen3AsrServeBatchConfig::from_env().unwrap().is_none());
        });
    }

    #[test]
    fn qwen_serve_batch_env_one_keeps_default_path() {
        with_serve_batch_env(Some("1"), || {
            assert!(Qwen3AsrServeBatchConfig::from_env().unwrap().is_none());
        });
    }

    #[test]
    fn qwen_serve_batch_env_accepts_two_to_eight() {
        with_serve_batch_env(Some("4"), || {
            let config = Qwen3AsrServeBatchConfig::from_env()
                .unwrap()
                .expect("enabled");
            assert_eq!(config.max_batch, 4);
            // Queue capacity tracks the admission/policy source (here the env
            // width), not a separate fixed valve.
            assert_eq!(config.queue_capacity, 4);
        });
    }

    #[test]
    fn qwen_policy_derives_width_and_queue_from_admission_limit() {
        assert!(Qwen3AsrServeBatchConfig::from_policy(ServeBatchPolicy::serial()).is_none());
        let cfg = Qwen3AsrServeBatchConfig::from_policy(ServeBatchPolicy {
            max_native_sessions: 12,
        })
        .expect("enabled");
        assert_eq!(cfg.max_batch, QWEN_SERVE_BATCH_MAX_BATCH_LIMIT);
        assert_eq!(cfg.queue_capacity, 12);
        assert_eq!(
            cfg.collect_window,
            crate::models::serve_batch_env::SERVE_BATCH_COLLECT_WINDOW
        );
    }

    #[test]
    fn qwen_serve_batch_env_rejects_oversized_batch() {
        with_serve_batch_env(Some("9"), || {
            let error = Qwen3AsrServeBatchConfig::from_env().expect_err("oversized");
            assert!(matches!(error, Qwen3AsrServeBatchError::InvalidEnv { .. }));
        });
    }

    #[test]
    fn qwen_owner_thread_rebuckets_full_static_tiny_cpu_batch() {
        with_qwen_direct_cpu_backend_for_test(|| {
            let (_temp, fixture) = write_qwen_tiny_serve_batch_fixture();
            assert_qwen_rebucket_migration(&fixture);
        });
    }

    #[test]
    fn qwen_owner_thread_shrinks_tail_static_tiny_cpu_batch() {
        with_qwen_direct_cpu_backend_for_test(|| {
            let (_temp, fixture) = write_qwen_tiny_serve_batch_fixture();
            assert_qwen_tail_shrink_migration(&fixture);
        });
    }

    #[test]
    fn qwen_owner_thread_keeps_stable_span_across_logical_lengths_tiny_cpu_batch() {
        with_qwen_direct_cpu_backend_for_test(|| {
            let (_temp, fixture) = write_qwen_tiny_serve_batch_fixture();
            assert_qwen_stable_resident_span_accepts_distinct_logical_spans(&fixture);
        });
    }

    #[test]
    fn qwen_batch_slot_replays_generated_host_kv_tiny_cpu_batch() {
        with_qwen_direct_cpu_backend_for_test(|| {
            let (_temp, fixture) = write_qwen_tiny_serve_batch_fixture();
            assert_qwen_generated_host_kv_replay(&fixture);
        });
    }

    #[test]
    #[ignore = "manual real-pack backend harness: set OPENASR_QWEN_SERVE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=hip or vulkan"]
    fn qwen_owner_thread_rebuckets_full_static_real_pack_selected_backend_batch() {
        assert_qwen_selected_backend_direct_for_real_pack_harness();
        let fixture = load_qwen_serve_batch_real_pack_fixture();
        assert_qwen_rebucket_migration(&fixture);
    }

    #[test]
    #[ignore = "manual real-pack backend harness: set OPENASR_QWEN_SERVE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=hip or vulkan"]
    fn qwen_owner_thread_shrinks_tail_static_real_pack_selected_backend_batch() {
        assert_qwen_selected_backend_direct_for_real_pack_harness();
        let fixture = load_qwen_serve_batch_real_pack_fixture();
        assert_qwen_tail_shrink_migration(&fixture);
    }

    #[test]
    #[ignore = "manual real-pack backend harness: set OPENASR_QWEN_SERVE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=hip or vulkan"]
    fn qwen_owner_thread_keeps_stable_span_real_pack_selected_backend_batch() {
        assert_qwen_selected_backend_direct_for_real_pack_harness();
        let fixture = load_qwen_serve_batch_real_pack_fixture();
        assert_qwen_stable_resident_span_accepts_distinct_logical_spans(&fixture);
    }

    #[test]
    #[ignore = "manual real-pack backend harness: set OPENASR_QWEN_SERVE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=hip or vulkan"]
    fn qwen_batch_slot_replays_generated_host_kv_real_pack_selected_backend() {
        assert_qwen_selected_backend_direct_for_real_pack_harness();
        let fixture = load_qwen_serve_batch_real_pack_fixture();
        assert_qwen_generated_host_kv_replay(&fixture);
    }

    #[test]
    fn serve_batch_prefill_cancel_poll_returns_typed_canceled() {
        use crate::RequestExecutionContext;
        use crate::ggml_runtime::GgmlCpuGraphError;

        let context = RequestExecutionContext::uncancellable("test fixture");
        assert!(super::ensure_serve_batch_prefill_not_canceled(&context).is_ok());
        context.control.request_cancel();
        assert!(matches!(
            super::ensure_serve_batch_prefill_not_canceled(&context),
            Err(Qwen3AsrServeBatchError::Canceled)
        ));
        assert!(matches!(
            super::map_serve_batch_graph_error(GgmlCpuGraphError::Canceled),
            Qwen3AsrServeBatchError::Canceled
        ));
        assert!(
            Qwen3AsrServeBatchError::Canceled
                .to_string()
                .contains("canceled by transcription control")
        );
        // Cancel is not a transient serve-batch unavailable class.
        assert_eq!(
            Qwen3AsrServeBatchError::Canceled.unavailable_retryable(),
            None
        );
    }

    #[test]
    fn serve_batch_prefill_chunk_loop_harness_stops_between_chunks_on_cancel() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use crate::RequestExecutionContext;

        let context = RequestExecutionContext::uncancellable("test fixture");
        let chunks_run = AtomicUsize::new(0);
        let token_count = 10usize;
        let chunk_size = 3usize;
        let mut position_offset = 0usize;
        let mut canceled = false;
        while position_offset < token_count {
            if let Err(error) = super::ensure_serve_batch_prefill_not_canceled(&context) {
                assert!(matches!(error, Qwen3AsrServeBatchError::Canceled));
                canceled = true;
                break;
            }
            let chunk_len = (token_count - position_offset).min(chunk_size);
            let seen = chunks_run.fetch_add(1, Ordering::SeqCst) + 1;
            if seen == 2 {
                context.control.request_cancel();
            }
            position_offset = position_offset.saturating_add(chunk_len);
        }
        assert!(canceled);
        assert_eq!(chunks_run.load(Ordering::SeqCst), 2);
        assert!(position_offset < token_count);
    }

    /// Production cancel must travel via the job-carried `Arc`: the owner
    /// thread never installs anything for the submitting request -- the
    /// execution context is captured once at submit time and carried
    /// explicitly on the job, so a cancel flipped from any thread (the HTTP
    /// handler's thread here) is visible the moment the owner thread reads
    /// the same `Arc`.
    #[test]
    fn serve_batch_job_control_cancel_visible_on_owner_thread() {
        use std::sync::Arc;
        use std::thread;

        use crate::RequestExecutionContext;
        use crate::api::backend::TranscriptionControl;

        let control = Arc::new(TranscriptionControl::new());
        let job_context = Arc::new(RequestExecutionContext::new(
            Some("job-1".to_string()),
            Arc::clone(&control),
        ));

        // Submit-side handler cancels from its own thread.
        control.request_cancel();

        // Owner thread: no thread-local install of any kind; only the
        // job-carried `Arc<RequestExecutionContext>` is readable.
        let owner_context = Arc::clone(&job_context);
        let owner = thread::spawn(move || {
            assert!(
                matches!(
                    super::ensure_serve_batch_prefill_not_canceled(&owner_context),
                    Err(Qwen3AsrServeBatchError::Canceled)
                ),
                "job-carried execution context must surface cancel across threads"
            );
        });
        owner.join().expect("owner thread");
    }
}
