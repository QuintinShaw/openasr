use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use thiserror::Error;

use super::graph_config::qwen_runtime_graph_config;
use super::kv_cache::Qwen3AsrLayerKvCacheState;
use super::llm_prefill::Qwen3AsrLlmPrefillInput;
use super::llm_transformer::{
    Qwen3AsrLlmLayerAttentionProjection, Qwen3AsrLlmWholeDecoderGraphExecutor,
};
use super::logits_head::Qwen3AsrLlmLogitsHead;
use super::runtime_contract::Qwen3AsrExecutionMetadata;
use super::token_embedding::Qwen3AsrTokenEmbeddingTable;
use super::tokenizer::Qwen3AsrTokenizer;
use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicySeq2SeqTextPostprocessKind, apply_seq2seq_text_postprocess,
};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeConfig, Seq2SeqGreedyDecodeError, build_seq2seq_greedy_stop_token_ids,
};
use crate::models::seq2seq_word_timestamps::seq2seq_word_timestamps_from_generated_tokens;
use crate::models::serve_batch_env::{
    OwnerAliveGuard, ServeBatchEnvError, serve_batch_bucket_width,
    serve_batch_collect_window_from_env, serve_batch_compact_active_slots,
    serve_batch_drain_compatible_batch, serve_batch_estimate_llm_kv_slot_bytes,
    serve_batch_max_from_env, serve_batch_owner_alive, serve_batch_select_and_apply_greedy_step,
    serve_batch_submit_with_timeout, serve_batch_trace_enabled, serve_batch_vram_capped_max_batch,
};
use crate::nn::decoder::reusable_decode_graph_supported;
use crate::{GgmlAsrExecutionResult, Segment, Transcription};

const QWEN_SERVE_BATCH_MAX_BATCH_LIMIT: usize = 8;
const QWEN_SERVE_BATCH_QUEUE_CAPACITY: usize = 4;
const QWEN_SERVE_BATCH_COLLECT_WINDOW: Duration = Duration::from_millis(2);
const QWEN_SERVE_BATCH_SEND_TIMEOUT: Duration = Duration::from_secs(1);
const QWEN_SERVE_BATCH_REPLY_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const QWEN_ROPE_THETA: f32 = 1_000_000.0;

static QWEN_SERVE_BATCH_ENGINES: OnceLock<
    Mutex<HashMap<Qwen3AsrServeBatchEngineKey, Arc<Qwen3AsrServeBatchEngine>>>,
> = OnceLock::new();

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
pub(super) struct Qwen3AsrServeBatchJob {
    pub runtime_source_path: PathBuf,
    pub runtime_cache_path: PathBuf,
    pub backend: GgmlCpuGraphBackend,
    pub metadata: Qwen3AsrExecutionMetadata,
    pub tokenizer: Option<Qwen3AsrTokenizer>,
    pub token_embedding_table: Qwen3AsrTokenEmbeddingTable,
    pub logits_head: Qwen3AsrLlmLogitsHead,
    pub layer_attention_projections: Arc<Vec<Qwen3AsrLlmLayerAttentionProjection>>,
    pub llm_prefill_input: Qwen3AsrLlmPrefillInput,
    pub decode_config: Seq2SeqGreedyDecodeConfig,
    pub text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind,
    pub word_timestamps: bool,
    pub audio_duration_seconds: f32,
}

#[derive(Debug, Error)]
pub(super) enum Qwen3AsrServeBatchError {
    #[error("qwen serve batch env {env} must be an integer in 0..={max}, got '{raw}'")]
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Qwen3AsrServeBatchEngineKey {
    runtime_cache_path: PathBuf,
    backend: GgmlCpuGraphBackend,
    max_batch: usize,
}

struct Qwen3AsrServeBatchEngine {
    sender: SyncSender<Qwen3AsrServeBatchEnvelope>,
    config: Qwen3AsrServeBatchConfig,
    is_alive: Arc<AtomicBool>,
}

struct Qwen3AsrServeBatchEnvelope {
    job: Qwen3AsrServeBatchJob,
    reply: mpsc::Sender<Result<GgmlAsrExecutionResult, Qwen3AsrServeBatchError>>,
}

struct Qwen3AsrOwnerThreadState {
    decoder: Option<Qwen3AsrLlmWholeDecoderGraphExecutor>,
}

struct Qwen3AsrActiveBatchSlot {
    slot: Qwen3AsrBatchSlot,
    reply: mpsc::Sender<Result<GgmlAsrExecutionResult, Qwen3AsrServeBatchError>>,
}

struct Qwen3AsrPendingRefillSlot {
    slot_index: usize,
    slot: Qwen3AsrBatchSlot,
    reply: mpsc::Sender<Result<GgmlAsrExecutionResult, Qwen3AsrServeBatchError>>,
}

struct Qwen3AsrPrefillSlotRef<'a> {
    slot_index: usize,
    slot: &'a mut Qwen3AsrBatchSlot,
}

struct Qwen3AsrBatchSlot {
    job: Qwen3AsrServeBatchJob,
    layer_kv_caches: Vec<Qwen3AsrLayerKvCacheState>,
    stop_token_ids: Vec<u32>,
    generated_tokens: Vec<u32>,
    /// Per-token softmax probability, parallel to `generated_tokens`.
    generated_probabilities: Vec<f32>,
    cache_prompt_tokens: usize,
    prefill_logits: Option<Vec<f32>>,
    done: bool,
}

impl Qwen3AsrServeBatchConfig {
    pub(super) fn from_env() -> Result<Option<Self>, Qwen3AsrServeBatchError> {
        let Some(max_batch) = serve_batch_max_from_env(QWEN_SERVE_BATCH_MAX_BATCH_LIMIT)
            .map_err(Qwen3AsrServeBatchError::from)?
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            max_batch,
            queue_capacity: QWEN_SERVE_BATCH_QUEUE_CAPACITY,
            collect_window: serve_batch_collect_window_from_env(QWEN_SERVE_BATCH_COLLECT_WINDOW)
                .map_err(Qwen3AsrServeBatchError::from)?,
            send_timeout: QWEN_SERVE_BATCH_SEND_TIMEOUT,
            reply_timeout: QWEN_SERVE_BATCH_REPLY_TIMEOUT,
            trace_batches: serve_batch_trace_enabled(),
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
        let backend = job.backend;
        if !reusable_decode_graph_supported(backend, qwen_runtime_graph_config().use_scheduler) {
            return Err(Qwen3AsrServeBatchError::UnsupportedBackend { backend });
        }
        let max_batch = serve_batch_vram_capped_max_batch(
            self.max_batch,
            backend,
            qwen_serve_batch_vram_slot_bytes(job),
        )
        .map_err(Qwen3AsrServeBatchError::from)?;
        Ok(Self { max_batch, ..self })
    }
}

impl From<ServeBatchEnvError> for Qwen3AsrServeBatchError {
    fn from(error: ServeBatchEnvError) -> Self {
        Self::InvalidEnv {
            env: error.env,
            raw: error.raw,
            max: error.max,
        }
    }
}

pub(super) fn submit_qwen_serve_batch_job(
    config: Qwen3AsrServeBatchConfig,
    job: Qwen3AsrServeBatchJob,
) -> Result<GgmlAsrExecutionResult, Qwen3AsrServeBatchError> {
    let config = config.validate_for_job(&job)?;
    let key = Qwen3AsrServeBatchEngineKey {
        runtime_cache_path: job.runtime_cache_path.clone(),
        backend: job.backend,
        max_batch: config.max_batch,
    };
    let engine = qwen_serve_batch_engine_for_key(key, config)?;
    engine.submit(job)
}

fn qwen_serve_batch_vram_slot_bytes(job: &Qwen3AsrServeBatchJob) -> usize {
    serve_batch_estimate_llm_kv_slot_bytes(
        job.metadata.llm_layers,
        job.metadata.llm_max_positions,
        job.metadata.llm_kv_heads,
        job.metadata.llm_head_dim,
        std::mem::size_of::<f32>(),
    )
}

fn qwen_serve_batch_engine_for_key(
    key: Qwen3AsrServeBatchEngineKey,
    config: Qwen3AsrServeBatchConfig,
) -> Result<Arc<Qwen3AsrServeBatchEngine>, Qwen3AsrServeBatchError> {
    let registry = QWEN_SERVE_BATCH_ENGINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut engines = registry
        .lock()
        .map_err(|_| Qwen3AsrServeBatchError::RegistryPoisoned)?;
    if let Some(engine) = engines.get(&key) {
        if serve_batch_owner_alive(&engine.is_alive) {
            return Ok(Arc::clone(engine));
        }
        // The cached owner thread exited (normal or panic); drop the stale
        // engine and respawn a fresh one with clean ggml state.
        engines.remove(&key);
    }
    let engine = Arc::new(Qwen3AsrServeBatchEngine::spawn(key.clone(), config)?);
    engines.insert(key, Arc::clone(&engine));
    Ok(engine)
}

impl Qwen3AsrServeBatchEngine {
    fn spawn(
        key: Qwen3AsrServeBatchEngineKey,
        config: Qwen3AsrServeBatchConfig,
    ) -> Result<Self, Qwen3AsrServeBatchError> {
        let (sender, receiver) = mpsc::sync_channel(config.queue_capacity);
        let (is_alive, alive_guard) = OwnerAliveGuard::new();
        thread::Builder::new()
            .name(format!(
                "openasr-qwen-serve-batch-{:?}-{}",
                key.backend, key.max_batch
            ))
            .spawn(move || {
                let _alive_guard = alive_guard;
                qwen_owner_thread_loop(receiver, config)
            })
            .map_err(|error| Qwen3AsrServeBatchError::ThreadSpawnFailed {
                reason: error.to_string(),
            })?;
        Ok(Self {
            sender,
            config,
            is_alive,
        })
    }

    fn submit(
        &self,
        job: Qwen3AsrServeBatchJob,
    ) -> Result<GgmlAsrExecutionResult, Qwen3AsrServeBatchError> {
        let (reply, reply_rx) = mpsc::channel();
        serve_batch_submit_with_timeout(
            &self.sender,
            Qwen3AsrServeBatchEnvelope { job, reply },
            reply_rx,
            self.config.send_timeout,
            self.config.reply_timeout,
            || Qwen3AsrServeBatchError::QueueFull,
            || Qwen3AsrServeBatchError::OwnerDisconnected,
            || Qwen3AsrServeBatchError::ReplyTimedOut,
        )
    }
}

fn qwen_owner_thread_loop(
    receiver: Receiver<Qwen3AsrServeBatchEnvelope>,
    config: Qwen3AsrServeBatchConfig,
) {
    let mut state = Qwen3AsrOwnerThreadState { decoder: None };
    let mut deferred = VecDeque::new();
    loop {
        let Some(batch) = serve_batch_drain_compatible_batch(
            &mut deferred,
            &receiver,
            config.max_batch,
            config.collect_window,
            |_, _| true,
        ) else {
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

        let mut prepared = Vec::with_capacity(batch.len());
        let mut batch_max_positions: Option<usize> = None;
        for envelope in batch {
            let required_positions =
                Qwen3AsrBatchSlot::required_max_positions_for_job(&envelope.job);
            if let Ok(required_positions) = required_positions {
                batch_max_positions = Some(
                    batch_max_positions
                        .map(|current| current.max(required_positions))
                        .unwrap_or(required_positions),
                );
            }
            prepared.push((envelope, required_positions));
        }

        let Some(mut max_positions) = batch_max_positions else {
            for (envelope, required_positions) in prepared {
                let error = required_positions.err().unwrap_or_else(|| {
                    Qwen3AsrServeBatchError::OwnerFailed {
                        reason: "qwen serve batch max-position calculation produced no valid slots"
                            .to_string(),
                    }
                });
                let _ = envelope.reply.send(Err(error));
            }
            return deferred;
        };

        let mut slots = Vec::with_capacity(prepared.len());
        for (envelope, required_positions) in prepared {
            match required_positions
                .and_then(|_| Qwen3AsrBatchSlot::new(envelope.job, max_positions))
            {
                Ok(slot) => slots.push(Some(Qwen3AsrActiveBatchSlot {
                    slot,
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

        let decoder_result = {
            let decoder_slot = slots
                .iter()
                .find_map(|slot| slot.as_ref().map(|active| &active.slot))
                .expect("active slot count checked above");
            self.decoder_for(decoder_slot)
        };
        let decoder = match decoder_result {
            Ok(decoder) => decoder,
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
        let prefill_errors = Self::prefill_and_select_slot_entries(decoder, &mut prefill_entries);
        drop(prefill_entries);
        for (slot_index, error) in prefill_errors {
            Self::fail_slot(&mut slots, slot_index, decoder, max_positions, false, error);
        }
        for slot_index in 0..slots.len() {
            if slots[slot_index]
                .as_ref()
                .map(|active| active.slot.done)
                .unwrap_or(false)
            {
                Self::finish_slot(&mut slots, slot_index, decoder, max_positions, false);
            }
        }
        if !slots.iter().any(Option::is_some) {
            return deferred;
        }

        let mut graph_initialized = false;
        loop {
            if graph_initialized {
                Self::refill_free_slots(
                    &mut slots,
                    decoder,
                    max_positions,
                    receiver,
                    &mut deferred,
                    config.trace_batches,
                );
                if let Err(error) = Self::try_expand_max_positions_for_next_candidate(
                    &mut slots,
                    decoder,
                    &mut max_positions,
                    receiver,
                    &mut deferred,
                    config.max_batch,
                    config.trace_batches,
                ) {
                    Self::fail_all_slots(&mut slots, error);
                    break;
                }
                Self::refill_free_slots(
                    &mut slots,
                    decoder,
                    max_positions,
                    receiver,
                    &mut deferred,
                    config.trace_batches,
                );
                if let Err(error) = Self::try_rebucket_active_slots(
                    &mut slots,
                    decoder,
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
            let mut hidden = Vec::with_capacity(d_model.saturating_mul(n_seq));
            let mut cache_positions = Vec::with_capacity(n_seq);
            let mut pack_errors = Vec::new();
            for (slot_index, active) in slots.iter().enumerate() {
                if let Some(active) = active {
                    match (
                        active.slot.gather_last_generated_token_hidden(),
                        active.slot.next_cache_position(),
                    ) {
                        (Ok(slot_hidden), Ok(cache_position)) => {
                            hidden.extend(slot_hidden);
                            cache_positions.push(cache_position);
                        }
                        (Err(error), _) | (_, Err(error)) => {
                            pack_errors.push((slot_index, error));
                            hidden.extend(std::iter::repeat_n(0.0_f32, d_model));
                            cache_positions.push(0);
                        }
                    }
                } else {
                    hidden.extend(std::iter::repeat_n(0.0_f32, d_model));
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
                decoder.run_step_reused_batched(
                    &hidden,
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
                            .or_else(|| dummy_seed_layers[slot_index].as_deref())
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
                decoder.run_step_reused_batched_seeded(
                    &hidden,
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
                    let logits = active
                        .slot
                        .job
                        .logits_head
                        .compute_logits_for_last_hidden(hidden_for_slot)
                        .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
                            reason: error.to_string(),
                        })?;
                    active.slot.select_next_token_from_logits(logits)
                })();
                match scatter_result {
                    Ok(()) => {
                        if slots[slot_index]
                            .as_ref()
                            .map(|active| active.slot.done)
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
    ) -> Result<Vec<Option<Vec<Qwen3AsrLayerKvCacheState>>>, Qwen3AsrServeBatchError> {
        let template = slots
            .iter()
            .find_map(|slot| slot.as_ref().map(|active| &active.slot.job.metadata))
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
                    Qwen3AsrBatchSlot::zero_seed_layer_kv_caches(*template, max_positions).map(Some)
                }
            })
            .collect()
    }

    fn prefill_and_select_slot_entries(
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        entries: &mut [Qwen3AsrPrefillSlotRef<'_>],
    ) -> Vec<(usize, Qwen3AsrServeBatchError)> {
        let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for entry_index in 0..entries.len() {
            let token_count = entries[entry_index].slot.job.llm_prefill_input.token_count;
            if let Some((_, group)) = groups
                .iter_mut()
                .find(|(group_token_count, _)| *group_token_count == token_count)
            {
                group.push(entry_index);
            } else {
                groups.push((token_count, vec![entry_index]));
            }
        }

        let mut failures = Vec::new();
        for (group_token_count, group) in groups {
            if group.len() > 1 {
                if let Some(chunk_size) =
                    decoder.safe_multi_query_prefill_chunk_size_for(group_token_count)
                {
                    if let Err(error) =
                        Self::prefill_and_select_batched_group(decoder, entries, &group, chunk_size)
                    {
                        let reason = error.to_string();
                        failures.extend(group.into_iter().map(|entry_index| {
                            (
                                entries[entry_index].slot_index,
                                Qwen3AsrServeBatchError::DecodeFailed {
                                    reason: reason.clone(),
                                },
                            )
                        }));
                    }
                } else {
                    for entry_index in group {
                        if let Err(error) =
                            entries[entry_index].slot.run_prefill_and_select(decoder)
                        {
                            failures.push((entries[entry_index].slot_index, error));
                        }
                    }
                }
            } else {
                let entry_index = group[0];
                if let Err(error) = entries[entry_index].slot.run_prefill_and_select(decoder) {
                    failures.push((entries[entry_index].slot_index, error));
                }
            }
        }
        failures
    }

    fn prefill_and_select_batched_group(
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
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
        let token_count = entries[first].slot.job.llm_prefill_input.token_count;
        let hidden_size = entries[first].slot.job.llm_prefill_input.hidden_size;
        for &entry_index in group {
            let slot = &entries[entry_index].slot;
            if slot.job.llm_prefill_input.token_count != token_count
                || slot.job.llm_prefill_input.hidden_size != hidden_size
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
                let input = &entries[entry_index].slot.job.llm_prefill_input;
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
                    .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
                        reason: error.to_string(),
                    })?
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
            let logits = entries[entry_index]
                .slot
                .job
                .logits_head
                .compute_logits_for_last_hidden(&final_hidden)
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

            let Qwen3AsrServeBatchEnvelope { job, reply } = envelope;
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
                reply,
            });
        }
        if pending_refills.is_empty() {
            return;
        }

        let mut prefill_entries = pending_refills
            .iter_mut()
            .map(|pending| Qwen3AsrPrefillSlotRef {
                slot_index: pending.slot_index,
                slot: &mut pending.slot,
            })
            .collect::<Vec<_>>();
        let prefill_errors = Self::prefill_and_select_slot_entries(decoder, &mut prefill_entries);
        drop(prefill_entries);

        for pending in pending_refills {
            let Qwen3AsrPendingRefillSlot {
                slot_index,
                slot,
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
            if slot.done {
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
            slots[slot_index] = Some(Qwen3AsrActiveBatchSlot { slot, reply });
            if trace_batches {
                eprintln!("openasr qwen serve batch: refilled slot {slot_index}");
            }
        }
    }

    fn try_rebucket_active_slots(
        slots: &mut Vec<Option<Qwen3AsrActiveBatchSlot>>,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
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

            let Qwen3AsrServeBatchEnvelope { job, reply } = envelope;
            match Qwen3AsrBatchSlot::new(job, max_positions) {
                Ok(slot) => pending.push((slot, reply)),
                Err(error) => {
                    let _ = reply.send(Err(error));
                }
            }
        }
        if pending.is_empty() {
            return Ok(());
        }

        let previous_width = slots.len();
        let target_active = active_count.checked_add(pending.len()).ok_or_else(|| {
            Qwen3AsrServeBatchError::OwnerFailed {
                reason: "qwen serve batch rebucket active count overflowed".to_string(),
            }
        })?;
        let mut bucket_width = serve_batch_bucket_width(target_active, max_batch);
        if bucket_width <= previous_width {
            for (slot, reply) in pending.into_iter().rev() {
                deferred.push_front(Qwen3AsrServeBatchEnvelope {
                    job: slot.job,
                    reply,
                });
            }
            return Ok(());
        }

        let mut prefill_entries = pending
            .iter_mut()
            .enumerate()
            .map(|(pending_index, (slot, _))| Qwen3AsrPrefillSlotRef {
                slot_index: previous_width + pending_index,
                slot,
            })
            .collect::<Vec<_>>();
        let prefill_errors = Self::prefill_and_select_slot_entries(decoder, &mut prefill_entries);
        drop(prefill_entries);

        let mut admitted = Vec::new();
        for (pending_index, (slot, reply)) in pending.into_iter().enumerate() {
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
            if slot.done {
                let _ = reply.send(slot.finish());
                continue;
            }
            admitted.push(Qwen3AsrActiveBatchSlot { slot, reply });
        }
        if admitted.is_empty() {
            return Ok(());
        }
        bucket_width = serve_batch_bucket_width(active_count + admitted.len(), max_batch);
        if bucket_width <= previous_width {
            for active in admitted.into_iter().rev() {
                deferred.push_front(Qwen3AsrServeBatchEnvelope {
                    job: active.slot.job,
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

    fn try_expand_max_positions_for_next_candidate(
        slots: &mut [Option<Qwen3AsrActiveBatchSlot>],
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        max_positions: &mut usize,
        receiver: &Receiver<Qwen3AsrServeBatchEnvelope>,
        deferred: &mut VecDeque<Qwen3AsrServeBatchEnvelope>,
        max_batch: usize,
        trace_batches: bool,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        let active_count = slots.iter().filter(|slot| slot.is_some()).count();
        if active_count == 0 {
            return Ok(());
        }
        let Some(envelope) = Self::pop_refill_candidate(deferred, receiver) else {
            return Ok(());
        };
        let required_positions =
            match Qwen3AsrBatchSlot::required_max_positions_for_job(&envelope.job) {
                Ok(required_positions) => required_positions,
                Err(error) => {
                    let _ = envelope.reply.send(Err(error));
                    return Ok(());
                }
            };
        deferred.push_front(envelope);
        if required_positions <= *max_positions {
            return Ok(());
        }

        let has_free_slot = active_count < slots.len();
        let can_grow_width = active_count == slots.len() && slots.len() < max_batch;
        if !has_free_slot && !can_grow_width {
            return Ok(());
        }

        let previous_max_positions = *max_positions;
        for active in slots.iter_mut().filter_map(Option::as_mut) {
            active.slot.ensure_generated_host_kv_replayed(decoder)?;
            active.slot.resize_max_positions(required_positions)?;
        }
        *max_positions = required_positions;
        Self::reseed_rebucketed_slots(slots, decoder, *max_positions)?;
        if trace_batches {
            eprintln!(
                "openasr qwen serve batch: expanded span {previous_max_positions}->{required_positions}"
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
                    .or_else(|| dummy_seed_layers[slot_index].as_deref())
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
        let Qwen3AsrActiveBatchSlot { slot, reply } = active;
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

    fn decoder_for(
        &mut self,
        slot: &Qwen3AsrBatchSlot,
    ) -> Result<&mut Qwen3AsrLlmWholeDecoderGraphExecutor, Qwen3AsrServeBatchError> {
        if self.decoder.is_none() {
            self.decoder = Some(
                Qwen3AsrLlmWholeDecoderGraphExecutor::new(
                    slot.job.layer_attention_projections.as_slice(),
                    Some(slot.job.runtime_source_path.as_path()),
                )
                .map_err(|error| Qwen3AsrServeBatchError::OwnerFailed {
                    reason: format!("qwen whole-decoder init failed: {error}"),
                })?,
            );
        }
        self.decoder
            .as_mut()
            .ok_or_else(|| Qwen3AsrServeBatchError::OwnerFailed {
                reason: "qwen serve batch decoder cache is unexpectedly empty".to_string(),
            })
    }
}

impl Qwen3AsrBatchSlot {
    fn required_max_positions_for_job(
        job: &Qwen3AsrServeBatchJob,
    ) -> Result<usize, Qwen3AsrServeBatchError> {
        job.decode_config
            .initial_prompt_tokens
            .len()
            .checked_add(job.decode_config.max_generated_tokens)
            .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch max-position calculation overflowed".to_string(),
            })
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
        if max_positions < required_positions {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch shared max-position span is smaller than this slot"
                    .to_string(),
            });
        }
        let layer_kv_caches = (0..job.metadata.llm_layers)
            .map(|_| {
                Qwen3AsrLayerKvCacheState::new(
                    max_positions,
                    job.metadata.llm_kv_heads,
                    job.metadata.llm_head_dim,
                )
            })
            .collect();
        let stop_token_ids = build_seq2seq_greedy_stop_token_ids(&job.decode_config);
        Ok(Self {
            job,
            layer_kv_caches,
            stop_token_ids,
            generated_tokens: Vec::new(),
            generated_probabilities: Vec::new(),
            cache_prompt_tokens: 0,
            prefill_logits: None,
            done: false,
        })
    }

    fn zero_seed_layer_kv_caches(
        metadata: Qwen3AsrExecutionMetadata,
        max_positions: usize,
    ) -> Result<Vec<Qwen3AsrLayerKvCacheState>, Qwen3AsrServeBatchError> {
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
        let mut layers = Vec::with_capacity(metadata.llm_layers);
        for _ in 0..metadata.llm_layers {
            let mut cache = Qwen3AsrLayerKvCacheState::new(
                max_positions,
                metadata.llm_kv_heads,
                metadata.llm_head_dim,
            );
            cache
                .write(0, &zero_row, &zero_row)
                .map_err(|reason| Qwen3AsrServeBatchError::DecodeFailed { reason })?;
            layers.push(cache);
        }
        Ok(layers)
    }

    fn resize_max_positions(
        &mut self,
        max_positions: usize,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        let required_positions = Self::required_max_positions_for_job(&self.job)?;
        if max_positions < required_positions {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch resize span is smaller than this slot".to_string(),
            });
        }
        for cache in &mut self.layer_kv_caches {
            cache
                .resize_max_positions(max_positions)
                .map_err(|reason| Qwen3AsrServeBatchError::DecodeFailed { reason })?;
        }
        Ok(())
    }

    fn run_prefill(
        &mut self,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        let token_count = self.job.llm_prefill_input.token_count;
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
            return self.run_prefill_serial(decoder);
        };
        self.run_prefill_chunked(decoder, chunk_size)
    }

    fn run_prefill_chunked(
        &mut self,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        chunk_size: usize,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        if chunk_size == 0 {
            return Err(Qwen3AsrServeBatchError::DecodeFailed {
                reason: "qwen serve batch prefill chunk size is zero".to_string(),
            });
        }
        let token_count = self.job.llm_prefill_input.token_count;
        if token_count <= chunk_size {
            let step = decoder
                .run_prefill(
                    &self.job.llm_prefill_input.token_major_embeddings,
                    token_count,
                    QWEN_ROPE_THETA,
                )
                .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
                    reason: error.to_string(),
                })?;
            return self.write_prefill_step_outputs(token_count, step);
        }
        let hidden_size = self.job.llm_prefill_input.hidden_size;
        let require_even_chunks = decoder.prefill_chunks_require_even_width();
        let mut position_offset = 0usize;
        let mut final_hidden = None;
        while position_offset < token_count {
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
                    &self.job.llm_prefill_input.token_major_embeddings[hidden_start..hidden_end],
                    chunk_len,
                    position_offset,
                    total_token_count,
                    &self.layer_kv_caches,
                    QWEN_ROPE_THETA,
                )
                .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
                    reason: error.to_string(),
                })?;
            final_hidden =
                Some(self.write_prefill_chunk_outputs(position_offset, chunk_len, step)?);
            position_offset = total_token_count;
        }
        self.cache_prompt_tokens = token_count;
        let logits = self
            .job
            .logits_head
            .compute_logits_for_last_hidden(&final_hidden.ok_or_else(|| {
                Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch prefill produced no hidden state".to_string(),
                }
            })?)
            .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
                reason: error.to_string(),
            })?;
        self.prefill_logits = Some(logits);
        Ok(())
    }

    fn run_prefill_serial(
        &mut self,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        let token_count = self.job.llm_prefill_input.token_count;
        let mut final_hidden = None;
        for token_position in 0..token_count {
            let hidden = self.prefill_prompt_hidden_at(token_position)?;
            let step = decoder
                .run_step(
                    &hidden,
                    token_position,
                    &self.layer_kv_caches,
                    QWEN_ROPE_THETA,
                )
                .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
                    reason: error.to_string(),
                })?;
            for (layer_index, (projected_k, projected_v)) in step.layer_kv.iter().enumerate() {
                self.layer_kv_caches[layer_index]
                    .write(token_position, projected_k, projected_v)
                    .map_err(|reason| Qwen3AsrServeBatchError::DecodeFailed { reason })?;
            }
            final_hidden = Some(step.hidden);
        }
        self.cache_prompt_tokens = token_count;
        let logits = self
            .job
            .logits_head
            .compute_logits_for_last_hidden(&final_hidden.ok_or_else(|| {
                Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch prefill produced no hidden state".to_string(),
                }
            })?)
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
    ) -> Result<(), Qwen3AsrServeBatchError> {
        let final_hidden = self.write_prefill_chunk_outputs(0, token_count, step)?;
        self.cache_prompt_tokens = token_count;
        let logits = self
            .job
            .logits_head
            .compute_logits_for_last_hidden(&final_hidden)
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
        let hidden_size = self.job.llm_prefill_input.hidden_size;
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
        let hidden_size = self.job.llm_prefill_input.hidden_size;
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
            .llm_prefill_input
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
    ) -> Result<(), Qwen3AsrServeBatchError> {
        self.run_prefill(decoder)?;
        let logits =
            self.prefill_logits
                .take()
                .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch prefill produced no logits".to_string(),
                })?;
        self.select_next_token_from_logits(logits)
    }

    fn gather_last_generated_token_hidden(&self) -> Result<Vec<f32>, Qwen3AsrServeBatchError> {
        let last_token =
            *self
                .generated_tokens
                .last()
                .ok_or_else(|| Qwen3AsrServeBatchError::DecodeFailed {
                    reason: "qwen serve batch generated token history is unexpectedly empty"
                        .to_string(),
                })?;
        self.job
            .token_embedding_table
            .gather_rows(&[last_token])
            .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
                reason: error.to_string(),
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
        for cache in &self.layer_kv_caches {
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

    fn select_next_token_from_logits(
        &mut self,
        logits: Vec<f32>,
    ) -> Result<(), Qwen3AsrServeBatchError> {
        serve_batch_select_and_apply_greedy_step(
            &self.job.decode_config,
            &mut self.generated_tokens,
            &mut self.generated_probabilities,
            &mut self.done,
            self.stop_token_ids.as_slice(),
            logits,
        )
        .map_err(|error| Qwen3AsrServeBatchError::DecodeFailed {
            reason: error.to_string(),
        })
    }

    fn finish(self) -> Result<GgmlAsrExecutionResult, Qwen3AsrServeBatchError> {
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
                speaker_profile_id: None,
                words,
            }]
        };
        Ok(GgmlAsrExecutionResult {
            transcription: Transcription {
                text,
                segments,
                longform: None,
                language: None,
            },
            carry_context: None,
        })
    }

    fn decode_text_token_ids(&self, token_ids: &[u32]) -> Result<String, Qwen3AsrServeBatchError> {
        if let Some(tokenizer) = self.job.tokenizer.as_ref() {
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
    use crate::testing::{
        TinyGgufFixtureSpec, with_forced_cpu_backend_for_test, write_tiny_gguf_runtime_source,
    };
    use crate::{read_gguf_metadata_from_runtime_source, validate_ggml_runtime_source_path};
    use std::{collections::BTreeMap, ffi::OsString};

    const QWEN_SERVE_BATCH_REAL_PACK_ENV: &str = "OPENASR_QWEN_SERVE_BATCH_REAL_PACK";

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
    const QWEN_PREFILL_REAL_PACK_ENV: &str = "OPENASR_QWEN_PREFILL_REAL_PACK";

    struct Qwen3AsrServeBatchFixture {
        runtime_path: PathBuf,
        metadata: Qwen3AsrExecutionMetadata,
        token_embedding_table: Qwen3AsrTokenEmbeddingTable,
        logits_head: Qwen3AsrLlmLogitsHead,
        layer_attention_projections: Arc<Vec<Qwen3AsrLlmLayerAttentionProjection>>,
        prompt_tokens: Vec<u32>,
    }

    fn with_serve_batch_env<T>(value: Option<&str>, run: impl FnOnce() -> T) -> T {
        crate::models::serve_batch_env::with_serve_batch_env_lock(|| {
            let previous = std::env::var_os(OPENASR_SERVE_BATCH_ENV);
            set_serve_batch_env(value.map(OsString::from));
            let result = run();
            set_serve_batch_env(previous);
            result
        })
    }

    fn set_serve_batch_env(value: Option<OsString>) {
        match value {
            Some(value) => {
                #[expect(unsafe_code, reason = "test-only process env override")]
                unsafe {
                    std::env::set_var(OPENASR_SERVE_BATCH_ENV, value);
                }
            }
            None => {
                #[expect(unsafe_code, reason = "test-only process env override")]
                unsafe {
                    std::env::remove_var(OPENASR_SERVE_BATCH_ENV);
                }
            }
        }
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
        let metadata = read_gguf_metadata_from_runtime_source(&runtime_source)
            .expect("read qwen runtime metadata");
        let metadata = parse_qwen3_execution_metadata(&metadata).expect("parse qwen metadata");
        let reader =
            GgufTensorDataReader::from_path(runtime_source.path()).expect("qwen tensor reader");
        let token_embedding_table = load_qwen3_token_embedding_table_from_reader(&reader, metadata)
            .expect("qwen token embeddings");
        let logits_head =
            load_qwen3_llm_logits_head_from_reader(&reader, metadata).expect("qwen logits head");
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
            metadata,
            token_embedding_table,
            logits_head,
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
            .with_tensor_shape(names.ffn_gate_weight, [32_u64, 16_u64])
            .with_tensor_shape(names.ffn_up_weight, [32_u64, 16_u64])
            .with_tensor_shape(names.ffn_down_weight, [16_u64, 32_u64])
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

    fn with_qwen_direct_cpu_backend_for_test<T>(run: impl FnOnce() -> T) -> T {
        with_forced_cpu_backend_for_test(|| {
            let previous = std::env::var_os(GgmlCpuGraphConfig::USE_SCHEDULER_ENV);
            #[expect(unsafe_code, reason = "test-only process env override")]
            unsafe {
                std::env::set_var(GgmlCpuGraphConfig::USE_SCHEDULER_ENV, "0");
            }
            let result = run();
            match previous {
                Some(value) => {
                    #[expect(unsafe_code, reason = "test-only process env restore")]
                    unsafe {
                        std::env::set_var(GgmlCpuGraphConfig::USE_SCHEDULER_ENV, value);
                    }
                }
                None => {
                    #[expect(unsafe_code, reason = "test-only process env restore")]
                    unsafe {
                        std::env::remove_var(GgmlCpuGraphConfig::USE_SCHEDULER_ENV);
                    }
                }
            }
            result
        })
    }

    fn qwen_real_pack_prefill_input(
        token_embedding_table: &Qwen3AsrTokenEmbeddingTable,
        prompt_tokens: &[u32],
    ) -> Qwen3AsrLlmPrefillInput {
        let token_count = prompt_tokens.len();
        let hidden_size = token_embedding_table.d_model();
        let token_major_embeddings = token_embedding_table
            .gather_rows(prompt_tokens)
            .expect("qwen prompt embeddings");
        let position_ids = (0..token_count)
            .map(|idx| i32::try_from(idx).expect("position id"))
            .collect::<Vec<_>>();
        let mut causal_mask = vec![-1.0e30_f32; token_count * token_count];
        for query_idx in 0..token_count {
            for key_idx in 0..=query_idx {
                causal_mask[query_idx * token_count + key_idx] = 0.0;
            }
        }
        Qwen3AsrLlmPrefillInput {
            token_count,
            hidden_size,
            token_major_embeddings,
            position_ids,
            causal_mask,
        }
    }

    fn qwen_fixture_job(
        fixture: &Qwen3AsrServeBatchFixture,
        max_generated_tokens: usize,
    ) -> Qwen3AsrServeBatchJob {
        Qwen3AsrServeBatchJob {
            runtime_source_path: fixture.runtime_path.clone(),
            runtime_cache_path: fixture.runtime_path.clone(),
            backend: GgmlCpuGraphConfig::resolve_runtime_backend(),
            metadata: fixture.metadata,
            tokenizer: None,
            token_embedding_table: fixture.token_embedding_table.clone(),
            logits_head: fixture.logits_head.clone(),
            layer_attention_projections: Arc::clone(&fixture.layer_attention_projections),
            llm_prefill_input: qwen_real_pack_prefill_input(
                &fixture.token_embedding_table,
                &fixture.prompt_tokens,
            ),
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
        (Qwen3AsrServeBatchEnvelope { job, reply }, reply_rx)
    }

    fn assert_qwen_selected_backend_direct_for_real_pack_harness() {
        let runtime_config = qwen_runtime_graph_config();
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
        max_positions: usize,
    ) -> Qwen3AsrActiveBatchSlot {
        qwen_prefilled_active_slot_with_token_cap(fixture, decoder, max_positions, 4)
    }

    fn qwen_prefilled_active_slot_with_token_cap(
        fixture: &Qwen3AsrServeBatchFixture,
        decoder: &mut Qwen3AsrLlmWholeDecoderGraphExecutor,
        max_positions: usize,
        max_generated_tokens: usize,
    ) -> Qwen3AsrActiveBatchSlot {
        let mut slot = Qwen3AsrBatchSlot::new(
            qwen_fixture_job(fixture, max_generated_tokens),
            max_positions,
        )
        .expect("qwen slot");
        slot.run_prefill_and_select(decoder)
            .expect("qwen slot prefill");
        let (reply, _reply_rx) = mpsc::channel();
        Qwen3AsrActiveBatchSlot { slot, reply }
    }

    fn assert_qwen_rebucket_migration(fixture: &Qwen3AsrServeBatchFixture) {
        let max_positions = fixture.prompt_tokens.len() + 4;
        let mut decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new(
            fixture.layer_attention_projections.as_slice(),
            Some(fixture.runtime_path.as_path()),
        )
        .expect("qwen decoder");
        let mut slots = vec![
            Some(qwen_prefilled_active_slot(
                fixture,
                &mut decoder,
                max_positions,
            )),
            Some(qwen_prefilled_active_slot(
                fixture,
                &mut decoder,
                max_positions,
            )),
        ];
        Qwen3AsrOwnerThreadState::reseed_rebucketed_slots(&mut slots, &mut decoder, max_positions)
            .expect("initial qwen seed");
        assert_eq!(decoder.reused_batch_width_for_test(), Some(2));

        let (queued_fast_a, _queued_fast_a_rx) = qwen_fixture_envelope(fixture, 1);
        let (queued_fast_b, _queued_fast_b_rx) = qwen_fixture_envelope(fixture, 1);
        let (queued_tx, queued_rx) = mpsc::sync_channel(2);
        queued_tx.send(queued_fast_a).expect("queue qwen refill a");
        queued_tx.send(queued_fast_b).expect("queue qwen refill b");
        let mut deferred = VecDeque::new();
        Qwen3AsrOwnerThreadState::try_rebucket_active_slots(
            &mut slots,
            &mut decoder,
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
        let max_positions = fixture.prompt_tokens.len() + 4;
        let mut decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new(
            fixture.layer_attention_projections.as_slice(),
            Some(fixture.runtime_path.as_path()),
        )
        .expect("qwen decoder");
        let mut slots = vec![
            Some(qwen_prefilled_active_slot(
                fixture,
                &mut decoder,
                max_positions,
            )),
            Some(qwen_prefilled_active_slot(
                fixture,
                &mut decoder,
                max_positions,
            )),
            Some(qwen_prefilled_active_slot(
                fixture,
                &mut decoder,
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

    fn assert_qwen_span_expansion_migration(fixture: &Qwen3AsrServeBatchFixture) {
        let initial_max_positions = fixture.prompt_tokens.len() + 2;
        let expanded_max_positions = fixture.prompt_tokens.len() + 4;
        let mut max_positions = initial_max_positions;
        let mut decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new(
            fixture.layer_attention_projections.as_slice(),
            Some(fixture.runtime_path.as_path()),
        )
        .expect("qwen decoder");
        let mut slots = vec![
            Some(qwen_prefilled_active_slot_with_token_cap(
                fixture,
                &mut decoder,
                max_positions,
                2,
            )),
            Some(qwen_prefilled_active_slot_with_token_cap(
                fixture,
                &mut decoder,
                max_positions,
                2,
            )),
        ];
        Qwen3AsrOwnerThreadState::reseed_rebucketed_slots(&mut slots, &mut decoder, max_positions)
            .expect("initial qwen seed");
        assert_eq!(decoder.reused_batch_width_for_test(), Some(2));

        let (queued_long_a, _queued_long_a_rx) = qwen_fixture_envelope(fixture, 4);
        let (queued_long_b, _queued_long_b_rx) = qwen_fixture_envelope(fixture, 4);
        let (queued_tx, queued_rx) = mpsc::sync_channel(2);
        queued_tx.send(queued_long_a).expect("queue long qwen a");
        queued_tx.send(queued_long_b).expect("queue long qwen b");
        let mut deferred = VecDeque::new();
        Qwen3AsrOwnerThreadState::try_expand_max_positions_for_next_candidate(
            &mut slots,
            &mut decoder,
            &mut max_positions,
            &queued_rx,
            &mut deferred,
            4,
            false,
        )
        .expect("qwen span expansion");
        assert_eq!(max_positions, expanded_max_positions);
        assert_eq!(decoder.reused_batch_width_for_test(), Some(2));
        for active in slots.iter().flatten() {
            assert_eq!(
                active.slot.layer_kv_caches[0].max_positions(),
                expanded_max_positions
            );
        }

        Qwen3AsrOwnerThreadState::try_rebucket_active_slots(
            &mut slots,
            &mut decoder,
            max_positions,
            &queued_rx,
            &mut deferred,
            4,
            false,
        )
        .expect("qwen rebucket after span expansion");
        assert!(deferred.is_empty());
        assert_eq!(slots.len(), 4);
        assert_eq!(slots.iter().filter(|slot| slot.is_some()).count(), 4);
        assert_eq!(decoder.reused_batch_width_for_test(), Some(4));
    }

    fn assert_qwen_generated_host_kv_replay(fixture: &Qwen3AsrServeBatchFixture) {
        let max_positions = fixture.prompt_tokens.len() + 4;
        let mut decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new(
            fixture.layer_attention_projections.as_slice(),
            Some(fixture.runtime_path.as_path()),
        )
        .expect("qwen decoder");
        let mut slot =
            Qwen3AsrBatchSlot::new(qwen_fixture_job(fixture, 4), max_positions).expect("qwen slot");
        slot.run_prefill_and_select(&mut decoder)
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
            fixture.prompt_tokens.len() + 1
        );
    }

    #[test]
    fn qwen_dummy_seed_layers_initialize_zero_prefix_for_padded_slots() {
        let layers =
            Qwen3AsrBatchSlot::zero_seed_layer_kv_caches(tiny_metadata(), 8).expect("dummy seed");
        assert_eq!(layers.len(), 2);
        for layer in layers {
            let snapshot = layer.snapshot_written().expect("snapshot");
            assert_eq!(snapshot.written_positions, 1);
            assert_eq!(snapshot.key_width, 4);
            assert_eq!(snapshot.value_width, 4);
            let history = layer.full_history_storage().expect("history");
            assert_eq!(history.written_positions, 1);
            assert!(history.keys.iter().all(|&value| value == 0.0));
            assert!(history.values.iter().all(|&value| value == 0.0));
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
            assert_eq!(config.queue_capacity, QWEN_SERVE_BATCH_QUEUE_CAPACITY);
        });
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
    fn qwen_owner_thread_expands_span_before_rebucket_tiny_cpu_batch() {
        with_qwen_direct_cpu_backend_for_test(|| {
            let (_temp, fixture) = write_qwen_tiny_serve_batch_fixture();
            assert_qwen_span_expansion_migration(&fixture);
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
    fn qwen_owner_thread_expands_span_real_pack_selected_backend_batch() {
        assert_qwen_selected_backend_direct_for_real_pack_harness();
        let fixture = load_qwen_serve_batch_real_pack_fixture();
        assert_qwen_span_expansion_migration(&fixture);
    }

    #[test]
    #[ignore = "manual real-pack backend harness: set OPENASR_QWEN_SERVE_BATCH_REAL_PACK and OPENASR_GGML_BACKEND=hip or vulkan"]
    fn qwen_batch_slot_replays_generated_host_kv_real_pack_selected_backend() {
        assert_qwen_selected_backend_direct_for_real_pack_harness();
        let fixture = load_qwen_serve_batch_real_pack_fixture();
        assert_qwen_generated_host_kv_replay(&fixture);
    }
}
