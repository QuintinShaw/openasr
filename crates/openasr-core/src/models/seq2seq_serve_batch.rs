//! Generic seq2seq serve-batch owner.
//!
//! This module hosts the family-agnostic continuous-batching owner loop that
//! the seq2seq serve-batch owners (whisper / cohere / moonshine / qwen) share.
//! The control flow is ported VERBATIM from the cohere owner
//! (`models/cohere/batched_decode.rs`, the cleanest no-special-casing
//! baseline), with concrete types replaced by the `Seq2SeqServeBatchFamily`
//! associated types and concrete method calls replaced by trait hooks.
//!
//! All four families (cohere / moonshine / whisper / qwen) are wired onto this
//! generic owner.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread;
use std::time::Duration;

use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlDecodeReuseMode};
#[cfg(not(test))]
use crate::models::native_execution_services::current_execution_cache_attempt_id;
use crate::models::native_execution_services::{
    NativeExecutionContext, current_execution_lane, current_native_execution_context,
    current_runtime_receipts, install_native_execution_context,
};
use crate::models::runtime_receipts::{
    RuntimeOwnerGuard, RuntimeReceiptCollector, RuntimeResourceGuard,
};
use crate::models::serve_batch_env::{
    OwnerAliveGuard, SERVE_BATCH_COLLECT_WINDOW, ServeBatchPolicy, serve_batch_bucket_width,
    serve_batch_compact_active_slots, serve_batch_drain_compatible_batch, serve_batch_owner_alive,
    serve_batch_submit_with_timeout, serve_batch_trace_enabled, serve_batch_vram_capped_max_batch,
};
use crate::nn::decoder::reusable_decode_graph_supported;

fn rebucket_dummy_position(
    physical_self_kv_positions: usize,
    prompt_len: usize,
    replay_steps: usize,
) -> Result<Option<usize>, &'static str> {
    if replay_steps == 0 {
        return Ok(None);
    }
    let position = physical_self_kv_positions
        .checked_sub(1)
        .ok_or("seq2seq serve batch rebucket requires non-empty context")?;
    if position < prompt_len {
        return Err("seq2seq serve batch rebucket has no dummy KV row");
    }
    Ok(Some(position))
}

/// Per-family decoder-runtime contract. Bound to `WhisperServeDecoderRuntime` /
/// `CohereDecoderGraphRuntime` / `MoonshineDecoderGraphRuntime`. The generic
/// owner only ever touches a runtime through these methods.
pub(crate) trait Seq2SeqServeRuntime: Sized {
    type Job;
    type Error;
    fn build_serial(job: &Self::Job) -> Result<Self, Self::Error>; // n_seq == 1
    fn build_batched(job: &Self::Job, n_seq: usize) -> Result<Self, Self::Error>;
    /// Select the current invocation's logical views inside a stable resident
    /// capacity. Implementations must not allocate or resize here.
    fn configure_for_job(&mut self, _job: &Self::Job) -> Result<(), Self::Error> {
        Ok(())
    }
    // Part of the per-family runtime contract (the serial path drives slot 0 of
    // the resident runtime), but the generic owner never calls it: `decode_serial`
    // is a family hook that owns the serial flow. Kept on the trait so each
    // family's serial implementation stays named and discoverable.
    #[allow(dead_code)]
    fn populate_cross_attention_cache_serial(&mut self, job: &Self::Job)
    -> Result<(), Self::Error>;
    fn populate_cross_attention_cache_slot(
        &mut self,
        slot_index: usize,
        job: &Self::Job,
    ) -> Result<(), Self::Error>;
    fn compute_batched_prefill_logits(
        &mut self,
        prompt_tokens: &[u32],
    ) -> Result<Vec<f32>, Self::Error>;
    fn compute_reused_batched_step_logits(
        &mut self,
        token_ids: &[u32],
        positions: &[usize],
        totals: &[usize],
    ) -> Result<Vec<f32>, Self::Error>;
    /// whisper resets its resident self-KV cursor before replay; cohere /
    /// moonshine use a fresh per-width cached runtime so this is a NO-OP
    /// default.
    fn reset_self_kv_state(&mut self) {}
}

/// Per-family identity/config/output seam. Bound once per family (a ZST).
pub(crate) trait Seq2SeqServeBatchFamily: Sized + 'static {
    type Runtime: Seq2SeqServeRuntime<Job = Self::Job, Error = Self::Error>;
    type Job: Clone;
    type Slot;
    type Output;
    // `Display` lets the generic owner reproduce cohere's `error.to_string()`
    // re-wrapping (`fail_all_active_slots` / refill seed failure) where one
    // error is stringified and cloned into per-slot `DecodeFailed` replies.
    // All family error enums derive thiserror `Error` (hence `Display`).
    type Error: std::fmt::Display;
    type EngineKey: Clone + Eq + std::hash::Hash;

    const THREAD_NAME_PREFIX: &'static str;
    /// Upper bound for eligible family batch width (all four families use 8).
    /// Consumed by `ServeBatchConfig::from_policy`.
    const MAX_BATCH_LIMIT: usize;
    fn engine_key(job: &Self::Job, max_batch: usize) -> Self::EngineKey;
    /// The backend recorded in an engine key, used only to reproduce the owner
    /// thread name `openasr-<prefix>-serve-batch-<Backend>-<max_batch>`.
    fn engine_key_backend(key: &Self::EngineKey) -> GgmlCpuGraphBackend;
    fn can_batch_with(a: &Self::Job, b: &Self::Job) -> bool;

    fn vram_slot_bytes(job: &Self::Job) -> usize;
    fn backend(job: &Self::Job) -> GgmlCpuGraphBackend;
    fn uses_scheduler(job: &Self::Job) -> bool;
    fn reuse_mode(job: &Self::Job) -> GgmlDecodeReuseMode;
    /// Applied AFTER the VRAM cap in `validate_for_job`. The default is the
    /// identity (cohere / moonshine never resolve a backend name); whisper
    /// overrides this to resolve typed backend capabilities at the shared
    /// runtime boundary and apply its Vulkan->serial cap. Keeping capability
    /// resolution behind this hook preserves cohere/moonshine behavior exactly
    /// (they never initialize a backend guard inside validation).
    fn effective_max_batch_after_vram_cap(
        capped_max_batch: usize,
        _job: &Self::Job,
    ) -> Result<usize, Self::Error> {
        Ok(capped_max_batch)
    }
    fn shrink_floor() -> usize {
        2
    }

    fn initial_prompt_tokens(job: &Self::Job) -> &[u32];
    fn vocab_size(job: &Self::Job) -> usize;
    fn max_generated_tokens(job: &Self::Job) -> usize;
    fn decoder_max_context(job: &Self::Job) -> usize; // reseed dummy_position

    fn slot_new(job: Self::Job) -> Result<Self::Slot, Self::Error>;
    fn slot_job(slot: &Self::Slot) -> &Self::Job;
    fn slot_generated(slot: &Self::Slot) -> &[u32];
    fn slot_done(slot: &Self::Slot) -> bool;
    fn slot_select_next_token(slot: &mut Self::Slot, logits: Vec<f32>) -> Result<(), Self::Error>;
    fn slot_finish(slot: Self::Slot) -> Result<Self::Output, Self::Error>;

    /// The genuinely-different serial path (whisper: reset+incremental
    /// advance(1); cohere: recompute full prefix; moonshine: incremental
    /// w/o reset).
    fn decode_serial(
        serial_runtime: &mut Option<Self::Runtime>,
        job: Self::Job,
    ) -> Result<Self::Output, Self::Error>;

    fn decode_failed(reason: String) -> Self::Error; // map_decoder_error / inline DecodeFailed
    fn owner_failed(reason: String) -> Self::Error;

    /// The explicit cancel/pause/resume context this job was submitted with.
    /// Never read through a thread-local -- see [`crate::RequestExecutionContext`].
    fn job_execution_context(job: &Self::Job) -> &Arc<crate::RequestExecutionContext>;

    // Engine/registry/config error constructors (Wave B). Each family binds these
    // to its existing `*ServeBatchError` variant constructors; no new error enum.
    #[cfg(test)]
    fn invalid_test_env(env: &'static str, raw: String, max: usize) -> Self::Error;
    fn invalid_enabled_batch(max_batch: usize) -> Self::Error;
    fn unsupported_backend(backend: GgmlCpuGraphBackend) -> Self::Error;
    fn registry_poisoned() -> Self::Error;
    fn thread_spawn_failed(reason: String) -> Self::Error;
    #[cfg_attr(test, allow(dead_code))]
    fn candidate_attempt_required() -> Self::Error {
        Self::thread_spawn_failed(
            "serve-batch publication requires an active candidate activation attempt".to_string(),
        )
    }
    fn queue_full() -> Self::Error;
    fn owner_disconnected() -> Self::Error;
    fn reply_timed_out() -> Self::Error;
}

/// The stable cancel marker embedded in a canceled slot's `DecodeFailed`
/// reason string. Matches the marker every other decode-cancel surface in
/// this crate uses (see `native_transcribe::is_cooperative_cancel_reason`)
/// so a canceled serve-batch slot's error still rewrites to
/// `BackendError::TranscriptionCanceled` at the native dispatch boundary.
pub(crate) const SERVE_BATCH_CANCEL_REASON: &str =
    "seq2seq serve batch slot canceled by transcription control";

/// A queued serve-batch request: the family job, its explicit execution
/// context (request id + cancel/pause/resume control -- never a
/// thread-local), and the reply channel the owner thread sends the decode
/// result back through.
pub(crate) struct Envelope<F: Seq2SeqServeBatchFamily> {
    pub job: F::Job,
    pub context: Arc<crate::RequestExecutionContext>,
    pub native_execution_context: Option<NativeExecutionContext>,
    pub reply: mpsc::Sender<Result<F::Output, F::Error>>,
}

/// A slot currently occupying a batch lane, pairing the family slot state with
/// its execution context and the reply channel that owns its result.
struct ActiveBatchSlot<F: Seq2SeqServeBatchFamily> {
    slot: F::Slot,
    context: Arc<crate::RequestExecutionContext>,
    native_execution_context: Option<NativeExecutionContext>,
    reply: mpsc::Sender<Result<F::Output, F::Error>>,
}

/// Transient state for a slot that has been seeded for refill but not yet
/// committed back into the active slot vector.
struct PendingRefillSlot<F: Seq2SeqServeBatchFamily> {
    slot_index: usize,
    slot: F::Slot,
    context: Arc<crate::RequestExecutionContext>,
    native_execution_context: Option<NativeExecutionContext>,
    reply: mpsc::Sender<Result<F::Output, F::Error>>,
}

/// The owner-thread decode state: a lazily-built serial runtime and a cache of
/// per-width batched runtimes keyed by `n_seq`.
///
/// Runtime receipt guards intentionally live after the native runtimes in field
/// order. Rust therefore destroys each runtime before releasing its independent
/// component/resource receipt, while the enclosing owner receipt is released
/// only after this state has been dropped.
pub(crate) struct OwnerThreadState<F: Seq2SeqServeBatchFamily> {
    serial_runtime: Option<F::Runtime>,
    pub(crate) batched_runtimes: HashMap<usize, F::Runtime>,
    serial_runtime_receipt: Option<RuntimeResourceGuard>,
    batched_runtime_receipts: HashMap<usize, RuntimeResourceGuard>,
    receipt_context: Option<ServeBatchReceiptContext>,
}

#[derive(Clone)]
struct ServeBatchReceiptContext {
    collector: RuntimeReceiptCollector,
    owner_id: crate::models::runtime_receipts::RuntimeOwnerId,
}

impl<F: Seq2SeqServeBatchFamily> OwnerThreadState<F> {
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::with_receipt_context(None)
    }

    fn with_receipt_context(receipt_context: Option<ServeBatchReceiptContext>) -> Self {
        Self {
            serial_runtime: None,
            batched_runtimes: HashMap::new(),
            serial_runtime_receipt: None,
            batched_runtime_receipts: HashMap::new(),
            receipt_context,
        }
    }

    fn acquire_semantic_resource(&self, kind: String) -> Option<RuntimeResourceGuard> {
        let receipt = self.receipt_context.as_ref()?;
        let descriptor = receipt.collector.no_broker_resource_descriptor(&kind)?;
        receipt
            .collector
            .acquire_resource(receipt.owner_id, descriptor)
    }

    fn record_serial_runtime_receipt(&mut self) {
        if self.serial_runtime.is_some() && self.serial_runtime_receipt.is_none() {
            self.serial_runtime_receipt =
                self.acquire_semantic_resource("serve-batch.serial-runtime".to_string());
        }
    }

    fn record_batched_runtime_receipt(&mut self, width: usize) {
        if !self.batched_runtime_receipts.contains_key(&width)
            && let Some(resource) =
                self.acquire_semantic_resource(format!("serve-batch.batch-runtime-width={width}"))
        {
            self.batched_runtime_receipts.insert(width, resource);
        }
    }

    fn record_session_receipt(
        &self,
        active_slots: usize,
        max_batch: usize,
    ) -> Option<RuntimeResourceGuard> {
        self.acquire_semantic_resource(format!(
            "serve-batch.session-state:active-slots={active_slots}:max-batch={max_batch}"
        ))
    }

    pub(crate) fn run_batch(
        &mut self,
        batch: Vec<Envelope<F>>,
        receiver: &Receiver<Envelope<F>>,
        max_batch: usize,
        trace_batches: bool,
    ) -> VecDeque<Envelope<F>> {
        let _session_receipt = self.record_session_receipt(batch.len(), max_batch);
        if batch.len() <= 1 {
            for envelope in batch {
                let Envelope {
                    job,
                    native_execution_context,
                    reply,
                    ..
                } = envelope;
                let _native_execution =
                    native_execution_context.map(install_native_execution_context);
                let result = self.decode_serial_job(job);
                let _ = reply.send(result);
            }
            return VecDeque::new();
        }

        self.decode_continuous_batch(batch, receiver, max_batch, trace_batches)
    }

    fn decode_continuous_batch(
        &mut self,
        batch: Vec<Envelope<F>>,
        receiver: &Receiver<Envelope<F>>,
        max_batch: usize,
        trace_batches: bool,
    ) -> VecDeque<Envelope<F>> {
        let mut deferred = VecDeque::new();
        if batch.is_empty() {
            return deferred;
        }

        let mut contexts_and_replies = Vec::with_capacity(batch.len());
        let mut slots = Vec::with_capacity(batch.len());
        for envelope in batch {
            let Envelope {
                job,
                context,
                native_execution_context,
                reply,
            } = envelope;
            contexts_and_replies.push((context, native_execution_context, reply));
            match F::slot_new(job) {
                Ok(slot) => slots.push(slot),
                Err(error) => {
                    let (_, _, reply) = contexts_and_replies
                        .pop()
                        .expect("context/reply pushed before slot build");
                    let _ = reply.send(Err(error));
                }
            }
        }
        let mut slots = slots
            .into_iter()
            .zip(contexts_and_replies)
            .map(|(slot, (context, native_execution_context, reply))| {
                Some(ActiveBatchSlot::<F> {
                    slot,
                    context,
                    native_execution_context,
                    reply,
                })
            })
            .collect::<Vec<_>>();
        if slots.is_empty() {
            return deferred;
        }
        // A slot canceled before this batch even started decoding (submitted
        // already-canceled, or canceled while queued waiting to be drained)
        // must never enter the shared prefill below.
        Self::finish_canceled_active_slots(&mut slots);
        if !slots.iter().any(Option::is_some) {
            return deferred;
        }
        let active_count = slots.iter().filter(|slot| slot.is_some()).count();
        if active_count <= 1 {
            for active in slots.into_iter().flatten() {
                let ActiveBatchSlot {
                    slot,
                    native_execution_context,
                    reply,
                    ..
                } = active;
                let _native_execution =
                    native_execution_context.map(install_native_execution_context);
                let result = self.decode_serial_job(F::slot_job(&slot).clone());
                let _ = reply.send(result);
            }
            return deferred;
        }
        let bucket_width = serve_batch_bucket_width(active_count, max_batch);
        if bucket_width > slots.len() {
            slots.resize_with(bucket_width, || None);
        }

        let Some(first_job) = Self::first_active_job(&slots).cloned() else {
            return deferred;
        };
        let prompt_len = F::initial_prompt_tokens(&first_job).len();
        if prompt_len == 0 {
            Self::fail_all_active_slots(
                &mut slots,
                F::decode_failed("seq2seq serve batch prompt is empty".to_string()),
            );
            return deferred;
        }
        let prompt_tokens = F::initial_prompt_tokens(&first_job).to_vec();
        {
            let native_execution_context = match Self::active_native_execution_context(&slots) {
                Ok(context) => context,
                Err(error) => {
                    Self::fail_all_active_slots(&mut slots, error);
                    return deferred;
                }
            };
            let runtime_result = {
                let _native_execution =
                    native_execution_context.map(install_native_execution_context);
                self.batched_runtime_for(&first_job, slots.len())
            };
            let runtime = match runtime_result {
                Ok(runtime) => runtime,
                Err(error) => {
                    Self::fail_all_active_slots(&mut slots, error);
                    return deferred;
                }
            };
            for slot_index in 0..slots.len() {
                let Some(active) = slots[slot_index].as_ref() else {
                    continue;
                };
                let job = F::slot_job(&active.slot).clone();
                let native_execution_context = match Self::active_native_execution_context(&slots) {
                    Ok(context) => context,
                    Err(error) => {
                        Self::fail_all_active_slots(&mut slots, error);
                        return deferred;
                    }
                };
                let populate_result = {
                    let _native_execution =
                        native_execution_context.map(install_native_execution_context);
                    runtime.populate_cross_attention_cache_slot(slot_index, &job)
                };
                if let Err(error) = populate_result {
                    Self::fail_active_slot(
                        &mut slots,
                        slot_index,
                        F::decode_failed(format_error::<F>(error)),
                    );
                }
            }
            if !slots.iter().any(Option::is_some) {
                return deferred;
            }

            let native_execution_context = match Self::active_native_execution_context(&slots) {
                Ok(context) => context,
                Err(error) => {
                    Self::fail_all_active_slots(&mut slots, error);
                    return deferred;
                }
            };
            let seed_result = {
                let _native_execution =
                    native_execution_context.map(install_native_execution_context);
                Self::seed_initial_batch_prompt(&mut slots, runtime, &prompt_tokens)
            };
            match seed_result {
                Ok(()) => Self::finish_done_active_slots(&mut slots),
                Err(error) => {
                    Self::fail_all_active_slots(&mut slots, error);
                    return deferred;
                }
            }
        }
        // Safe boundary right after the initial prefill: a slot canceled
        // while that prefill ran is pulled out here, before it can be
        // admitted into the per-step batched loop below.
        Self::finish_canceled_active_slots(&mut slots);

        loop {
            // Safe boundary at the top of every iteration, before refill,
            // rebucket, shrink, or the next batched token-step compute: a
            // canceled slot is finished here and only here -- the shared
            // runtime and every healthy sibling slot continue unaffected.
            Self::finish_canceled_active_slots(&mut slots);
            Self::finish_maxed_active_slots(&mut slots);
            if let Some(first_job) = Self::first_active_job(&slots).cloned() {
                let native_execution_context = match Self::active_native_execution_context(&slots) {
                    Ok(context) => context,
                    Err(error) => {
                        Self::fail_all_active_slots(&mut slots, error);
                        break;
                    }
                };
                let runtime_result = {
                    let _native_execution =
                        native_execution_context.map(install_native_execution_context);
                    self.batched_runtime_for(&first_job, slots.len())
                };
                let runtime = match runtime_result {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        Self::fail_all_active_slots(&mut slots, error);
                        break;
                    }
                };
                Self::refill_free_slots(
                    &mut slots,
                    runtime,
                    prompt_len,
                    receiver,
                    &mut deferred,
                    trace_batches,
                );
            }
            Self::finish_maxed_active_slots(&mut slots);
            if !slots.iter().any(Option::is_some) {
                break;
            }
            if let Err(error) = self.try_rebucket_active_slots(
                &mut slots,
                receiver,
                &mut deferred,
                max_batch,
                prompt_len,
                trace_batches,
            ) {
                Self::fail_all_active_slots(&mut slots, error);
                break;
            }
            Self::finish_done_active_slots(&mut slots);
            Self::finish_maxed_active_slots(&mut slots);
            if !slots.iter().any(Option::is_some) {
                break;
            }
            if let Err(error) =
                self.try_shrink_active_slots(&mut slots, max_batch, prompt_len, trace_batches)
            {
                Self::fail_all_active_slots(&mut slots, error);
                break;
            }
            if !slots.iter().any(Option::is_some) {
                break;
            }

            let step_inputs = Self::step_inputs_for_active_slots(&slots, prompt_len);
            let (token_ids, positions, totals) = match step_inputs {
                Ok(inputs) => inputs,
                Err(error) => {
                    Self::fail_all_active_slots(&mut slots, error);
                    break;
                }
            };
            let Some(first_job) = Self::first_active_job(&slots).cloned() else {
                break;
            };
            let native_execution_context = match Self::active_native_execution_context(&slots) {
                Ok(context) => context,
                Err(error) => {
                    Self::fail_all_active_slots(&mut slots, error);
                    break;
                }
            };
            let logits = {
                let _native_execution =
                    native_execution_context.map(install_native_execution_context);
                let runtime = match self.batched_runtime_for(&first_job, slots.len()) {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        Self::fail_all_active_slots(&mut slots, error);
                        break;
                    }
                };
                match runtime.compute_reused_batched_step_logits(&token_ids, &positions, &totals) {
                    Ok(logits) => logits,
                    Err(error) => {
                        Self::fail_all_active_slots(
                            &mut slots,
                            F::decode_failed(format_error::<F>(error)),
                        );
                        break;
                    }
                }
            };
            match Self::scatter_and_select_active_slots(&mut slots, &logits) {
                Ok(()) => Self::finish_done_active_slots(&mut slots),
                Err(error) => {
                    Self::fail_all_active_slots(&mut slots, error);
                    break;
                }
            }
        }

        deferred
    }

    fn try_rebucket_active_slots(
        &mut self,
        slots: &mut Vec<Option<ActiveBatchSlot<F>>>,
        receiver: &Receiver<Envelope<F>>,
        deferred: &mut VecDeque<Envelope<F>>,
        max_batch: usize,
        prompt_len: usize,
        trace_batches: bool,
    ) -> Result<(), F::Error> {
        let active_count = slots.iter().filter(|slot| slot.is_some()).count();
        if active_count == 0
            || active_count != slots.len()
            || slots.len() >= max_batch
            || prompt_len == 0
        {
            return Ok(());
        }
        let Some(template) = Self::first_active_job(slots) else {
            return Ok(());
        };
        let template = template.clone();
        let candidate_limit = max_batch.saturating_sub(active_count);
        let mut pending = Vec::new();
        while pending.len() < candidate_limit {
            let Some(envelope) =
                Self::pop_compatible_refill_candidate(deferred, receiver, &template)
            else {
                break;
            };
            let Envelope {
                job,
                context,
                native_execution_context,
                reply,
            } = envelope;
            // Already-canceled before it ever occupied a lane: reply canceled
            // now rather than spending a prefill on a request no one is
            // waiting on.
            if context.is_canceled() {
                let _ = reply.send(Err(F::decode_failed(SERVE_BATCH_CANCEL_REASON.to_string())));
                continue;
            }
            if let Err(error) = Self::native_execution_context_for_options(
                slots
                    .iter()
                    .filter_map(Option::as_ref)
                    .map(|active| &active.native_execution_context)
                    .chain(
                        pending
                            .iter()
                            .map(|(_, _, native_execution_context, _)| native_execution_context),
                    )
                    .chain(std::iter::once(&native_execution_context)),
            ) {
                let _ = reply.send(Err(error));
                continue;
            }
            match F::slot_new(job) {
                Ok(slot) => pending.push((slot, context, native_execution_context, reply)),
                Err(error) => {
                    let _ = reply.send(Err(error));
                }
            }
        }
        if pending.is_empty() {
            return Ok(());
        }

        let target_active = active_count.checked_add(pending.len()).ok_or_else(|| {
            F::owner_failed("seq2seq serve batch rebucket active count overflowed".to_string())
        })?;
        let bucket_width = serve_batch_bucket_width(target_active, max_batch);
        if bucket_width <= slots.len() {
            for (slot, context, native_execution_context, reply) in pending.into_iter().rev() {
                deferred.push_front(Envelope {
                    job: F::slot_job(&slot).clone(),
                    context,
                    native_execution_context,
                    reply,
                });
            }
            return Ok(());
        }

        let previous_width = slots.len();
        for (slot, context, native_execution_context, reply) in pending {
            slots.push(Some(ActiveBatchSlot::<F> {
                slot,
                context,
                native_execution_context,
                reply,
            }));
        }
        if bucket_width > slots.len() {
            slots.resize_with(bucket_width, || None);
        }
        self.reseed_rebucketed_slots(slots, prompt_len)?;
        if trace_batches {
            eprintln!(
                "openasr {} serve batch: rebucketed {previous_width}->{bucket_width} slot(s)",
                F::THREAD_NAME_PREFIX
            );
        }
        Ok(())
    }

    fn try_shrink_active_slots(
        &mut self,
        slots: &mut Vec<Option<ActiveBatchSlot<F>>>,
        max_batch: usize,
        prompt_len: usize,
        trace_batches: bool,
    ) -> Result<(), F::Error> {
        let active_count = slots.iter().filter(|slot| slot.is_some()).count();
        if active_count == 0 || active_count == slots.len() || prompt_len == 0 {
            return Ok(());
        }
        let bucket_width = serve_batch_bucket_width(active_count.max(F::shrink_floor()), max_batch);
        if bucket_width >= slots.len() {
            return Ok(());
        }

        let previous_width = slots.len();
        serve_batch_compact_active_slots(slots, bucket_width);
        self.reseed_rebucketed_slots(slots, prompt_len)?;
        if trace_batches {
            eprintln!(
                "openasr {} serve batch: shrank {previous_width}->{bucket_width} slot(s)",
                F::THREAD_NAME_PREFIX
            );
        }
        Ok(())
    }

    fn reseed_rebucketed_slots(
        &mut self,
        slots: &mut [Option<ActiveBatchSlot<F>>],
        prompt_len: usize,
    ) -> Result<(), F::Error> {
        let native_execution_context = Self::active_native_execution_context(slots)?;
        let _native_execution = native_execution_context.map(install_native_execution_context);
        let first_job = Self::first_active_job(slots).cloned().ok_or_else(|| {
            F::owner_failed("seq2seq serve batch rebucket has no active slots".to_string())
        })?;
        let prompt_tokens = F::initial_prompt_tokens(&first_job).to_vec();
        if prompt_tokens.len() != prompt_len {
            return Err(F::decode_failed(
                "seq2seq serve batch rebucket prompt length changed".to_string(),
            ));
        }
        let runtime = self.batched_runtime_for(&first_job, slots.len())?;
        runtime.reset_self_kv_state();
        #[allow(clippy::needless_range_loop)]
        for slot_index in 0..slots.len() {
            let Some(active) = slots[slot_index].as_ref() else {
                continue;
            };
            runtime
                .populate_cross_attention_cache_slot(slot_index, F::slot_job(&active.slot))
                .map_err(map_decoder_error::<F>)?;
        }

        let logits = runtime
            .compute_batched_prefill_logits(&prompt_tokens)
            .map_err(map_decoder_error::<F>)?;
        let n_seq = slots.len();
        #[allow(clippy::needless_range_loop)]
        for slot_index in 0..slots.len() {
            let Some(active) = slots[slot_index].as_mut() else {
                continue;
            };
            if F::slot_generated(&active.slot).is_empty() {
                Self::select_slot_from_batched_logits(
                    &mut active.slot,
                    &logits,
                    slot_index,
                    n_seq,
                )?;
            }
        }

        let replay_steps = slots
            .iter()
            .filter_map(|active| {
                active
                    .as_ref()
                    .map(|active| F::slot_generated(&active.slot).len().saturating_sub(1))
            })
            .max()
            .unwrap_or(0);
        // A one-token generation performs no incremental replay at all, so it
        // needs no dummy row beyond the prompt. Only materialize and validate
        // the inactive-lane row when a rebucket actually replays steps. For
        // G >= 2, the exact greedy arena K = P + G - 1 guarantees K - 1 >= P.
        let dummy_position =
            rebucket_dummy_position(F::decoder_max_context(&first_job), prompt_len, replay_steps)
                .map_err(|reason| F::decode_failed(reason.to_string()))?
                .unwrap_or(0);
        for generated_index in 0..replay_steps {
            let mut token_ids = Vec::with_capacity(slots.len());
            let mut positions = Vec::with_capacity(slots.len());
            let mut totals = Vec::with_capacity(slots.len());
            for active in slots.iter() {
                let Some(active) = active else {
                    token_ids.push(0);
                    positions.push(dummy_position);
                    totals.push(1);
                    continue;
                };
                let generated = F::slot_generated(&active.slot);
                if generated_index + 1 < generated.len() {
                    let position = prompt_len.checked_add(generated_index).ok_or_else(|| {
                        F::decode_failed(
                            "seq2seq serve batch rebucket position overflowed".to_string(),
                        )
                    })?;
                    token_ids.push(generated[generated_index]);
                    positions.push(position);
                    totals.push(position.checked_add(1).ok_or_else(|| {
                        F::decode_failed(
                            "seq2seq serve batch rebucket total overflowed".to_string(),
                        )
                    })?);
                } else {
                    token_ids.push(0);
                    positions.push(dummy_position);
                    totals.push(1);
                }
            }
            runtime
                .compute_reused_batched_step_logits(&token_ids, &positions, &totals)
                .map_err(map_decoder_error::<F>)?;
        }
        Ok(())
    }

    fn seed_initial_batch_prompt(
        slots: &mut [Option<ActiveBatchSlot<F>>],
        runtime: &mut F::Runtime,
        prompt_tokens: &[u32],
    ) -> Result<(), F::Error> {
        let logits = runtime
            .compute_batched_prefill_logits(prompt_tokens)
            .map_err(map_decoder_error::<F>)?;
        Self::scatter_and_select_active_slots(slots, &logits)
    }

    fn refill_free_slots(
        slots: &mut [Option<ActiveBatchSlot<F>>],
        runtime: &mut F::Runtime,
        prompt_len: usize,
        receiver: &Receiver<Envelope<F>>,
        deferred: &mut VecDeque<Envelope<F>>,
        trace_batches: bool,
    ) {
        let mut pending_refills = Vec::new();
        for slot_index in 0..slots.len() {
            while slots[slot_index].is_none() {
                let Some(template) = Self::first_active_job(slots) else {
                    return;
                };
                let template = template.clone();
                let Some(envelope) =
                    Self::pop_compatible_refill_candidate(deferred, receiver, &template)
                else {
                    break;
                };
                let Envelope {
                    job,
                    context,
                    native_execution_context,
                    reply,
                } = envelope;
                // Already-canceled before it ever occupied a lane: reply
                // canceled now rather than spending a prefill on a request no
                // one is waiting on.
                if context.is_canceled() {
                    let _ =
                        reply.send(Err(F::decode_failed(SERVE_BATCH_CANCEL_REASON.to_string())));
                    continue;
                }
                let slot = match F::slot_new(job) {
                    Ok(slot) => slot,
                    Err(error) => {
                        let _ = reply.send(Err(error));
                        continue;
                    }
                };
                let prospective_native_execution_context =
                    match Self::native_execution_context_for_options(
                        slots
                            .iter()
                            .filter_map(Option::as_ref)
                            .map(|active| &active.native_execution_context)
                            .chain(
                                pending_refills
                                    .iter()
                                    .map(|pending: &PendingRefillSlot<F>| {
                                        &pending.native_execution_context
                                    }),
                            )
                            .chain(std::iter::once(&native_execution_context)),
                    ) {
                        Ok(context) => context,
                        Err(error) => {
                            let _ = reply.send(Err(error));
                            continue;
                        }
                    };
                let populate_result = {
                    let _native_execution =
                        prospective_native_execution_context.map(install_native_execution_context);
                    runtime.populate_cross_attention_cache_slot(slot_index, F::slot_job(&slot))
                };
                if let Err(error) = populate_result {
                    let _ = reply.send(Err(F::decode_failed(format_error::<F>(error))));
                    continue;
                }
                pending_refills.push(PendingRefillSlot::<F> {
                    slot_index,
                    slot,
                    context,
                    native_execution_context,
                    reply,
                });
                break;
            }
        }
        if pending_refills.is_empty() {
            return;
        }

        if let Err(error) =
            Self::seed_refill_slots_prompt(slots, runtime, &mut pending_refills, prompt_len)
        {
            let reason = format_error::<F>(error);
            for pending in pending_refills {
                let _ = pending.reply.send(Err(F::decode_failed(reason.clone())));
            }
            return;
        }
        for pending in pending_refills {
            let PendingRefillSlot {
                slot_index,
                slot,
                context,
                native_execution_context,
                reply,
            } = pending;
            if F::slot_done(&slot) {
                let _ = reply.send(F::slot_finish(slot));
                continue;
            }
            slots[slot_index] = Some(ActiveBatchSlot::<F> {
                slot,
                context,
                native_execution_context,
                reply,
            });
            if trace_batches {
                eprintln!(
                    "openasr {} serve batch: refilled slot {slot_index}",
                    F::THREAD_NAME_PREFIX
                );
            }
        }
    }

    fn pop_compatible_refill_candidate(
        deferred: &mut VecDeque<Envelope<F>>,
        receiver: &Receiver<Envelope<F>>,
        template: &F::Job,
    ) -> Option<Envelope<F>> {
        let deferred_len = deferred.len();
        for _ in 0..deferred_len {
            let envelope = deferred
                .pop_front()
                .expect("bounded by deferred_len captured above");
            if F::can_batch_with(template, &envelope.job) {
                return Some(envelope);
            }
            deferred.push_back(envelope);
        }
        match receiver.try_recv() {
            Ok(envelope) if F::can_batch_with(template, &envelope.job) => Some(envelope),
            Ok(envelope) => {
                deferred.push_back(envelope);
                None
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => None,
        }
    }

    fn seed_refill_slots_prompt(
        slots: &[Option<ActiveBatchSlot<F>>],
        runtime: &mut F::Runtime,
        pending_refills: &mut [PendingRefillSlot<F>],
        prompt_len: usize,
    ) -> Result<(), F::Error> {
        if pending_refills.is_empty() {
            return Ok(());
        }
        let native_execution_context = Self::native_execution_context_for_options(
            slots
                .iter()
                .filter_map(Option::as_ref)
                .map(|active| &active.native_execution_context)
                .chain(
                    pending_refills
                        .iter()
                        .map(|pending| &pending.native_execution_context),
                ),
        )?;
        let _native_execution = native_execution_context.map(install_native_execution_context);
        let n_seq = slots.len();
        let prompt_tokens =
            F::initial_prompt_tokens(F::slot_job(&pending_refills[0].slot)).to_vec();
        if prompt_tokens.len() != prompt_len {
            return Err(F::decode_failed(
                "seq2seq serve batch refill prompt length changed during seed".to_string(),
            ));
        }
        let logits = runtime
            .compute_batched_prefill_logits(&prompt_tokens)
            .map_err(map_decoder_error::<F>)?;
        for pending in pending_refills {
            Self::select_slot_from_batched_logits(
                &mut pending.slot,
                &logits,
                pending.slot_index,
                n_seq,
            )?;
        }
        Ok(())
    }

    fn step_inputs_for_active_slots(
        slots: &[Option<ActiveBatchSlot<F>>],
        prompt_len: usize,
    ) -> Result<(Vec<u32>, Vec<usize>, Vec<usize>), F::Error> {
        let mut token_ids = Vec::with_capacity(slots.len());
        let mut positions = Vec::with_capacity(slots.len());
        let mut totals = Vec::with_capacity(slots.len());
        for active in slots {
            let Some(active) = active else {
                token_ids.push(0);
                positions.push(0);
                totals.push(1);
                continue;
            };
            let token_id = *F::slot_generated(&active.slot).last().ok_or_else(|| {
                F::decode_failed("seq2seq serve batch generated token history is empty".to_string())
            })?;
            let total_tokens = prompt_len
                .checked_add(F::slot_generated(&active.slot).len())
                .ok_or_else(|| {
                    F::decode_failed("seq2seq serve batch token count overflowed".to_string())
                })?;
            let position = total_tokens.checked_sub(1).ok_or_else(|| {
                F::decode_failed("seq2seq serve batch position underflowed".to_string())
            })?;
            token_ids.push(token_id);
            positions.push(position);
            totals.push(total_tokens);
        }
        Ok((token_ids, positions, totals))
    }

    fn scatter_and_select_active_slots(
        slots: &mut [Option<ActiveBatchSlot<F>>],
        logits: &[f32],
    ) -> Result<(), F::Error> {
        let n_seq = slots.len();
        for (slot_index, active) in slots.iter_mut().enumerate() {
            let Some(active) = active else {
                continue;
            };
            Self::select_slot_from_batched_logits(&mut active.slot, logits, slot_index, n_seq)?;
        }
        Ok(())
    }

    fn select_slot_from_batched_logits(
        slot: &mut F::Slot,
        logits: &[f32],
        slot_index: usize,
        n_seq: usize,
    ) -> Result<(), F::Error> {
        let vocab_size = F::vocab_size(F::slot_job(slot));
        let expected = vocab_size.checked_mul(n_seq).ok_or_else(|| {
            F::decode_failed("seq2seq serve batch logits length overflowed".to_string())
        })?;
        if logits.len() != expected {
            return Err(F::decode_failed(format!(
                "seq2seq serve batch logits width mismatch: got {}, expected {}",
                logits.len(),
                expected
            )));
        }
        let start = slot_index.checked_mul(vocab_size).ok_or_else(|| {
            F::decode_failed("seq2seq serve batch logits offset overflowed".to_string())
        })?;
        let end = start.checked_add(vocab_size).ok_or_else(|| {
            F::decode_failed("seq2seq serve batch logits end overflowed".to_string())
        })?;
        let slot_logits = logits.get(start..end).ok_or_else(|| {
            F::decode_failed("seq2seq serve batch logits slice out of bounds".to_string())
        })?;
        F::slot_select_next_token(slot, slot_logits.to_vec())
    }

    fn finish_maxed_active_slots(slots: &mut [Option<ActiveBatchSlot<F>>]) {
        for slot_index in 0..slots.len() {
            let should_finish = slots[slot_index]
                .as_ref()
                .map(|active| {
                    F::slot_generated(&active.slot).len()
                        >= F::max_generated_tokens(F::slot_job(&active.slot))
                })
                .unwrap_or(false);
            if should_finish {
                Self::finish_active_slot(slots, slot_index);
            }
        }
    }

    fn finish_done_active_slots(slots: &mut [Option<ActiveBatchSlot<F>>]) {
        for slot_index in 0..slots.len() {
            if slots[slot_index]
                .as_ref()
                .map(|active| F::slot_done(&active.slot))
                .unwrap_or(false)
            {
                Self::finish_active_slot(slots, slot_index);
            }
        }
    }

    fn finish_active_slot(slots: &mut [Option<ActiveBatchSlot<F>>], slot_index: usize) {
        let Some(active) = slots[slot_index].take() else {
            return;
        };
        let ActiveBatchSlot { slot, reply, .. } = active;
        let _ = reply.send(F::slot_finish(slot));
    }

    /// Per-slot cancel check for the safe boundaries between batched graph
    /// calls (initial prefill, each token step, refill, rebucket, shrink).
    /// Finishes exactly the slots whose own execution context has an active
    /// cancel request with a `DecodeFailed(SERVE_BATCH_CANCEL_REASON)` reply
    /// -- every healthy sibling slot is left untouched and the shared
    /// batched runtime is never aborted. Canceling one request must never
    /// fail the whole batch.
    fn finish_canceled_active_slots(slots: &mut [Option<ActiveBatchSlot<F>>]) {
        for slot_index in 0..slots.len() {
            let is_canceled = slots[slot_index]
                .as_ref()
                .is_some_and(|active| active.context.is_canceled());
            if is_canceled {
                Self::fail_active_slot(
                    slots,
                    slot_index,
                    F::decode_failed(SERVE_BATCH_CANCEL_REASON.to_string()),
                );
            }
        }
    }

    fn fail_active_slot(
        slots: &mut [Option<ActiveBatchSlot<F>>],
        slot_index: usize,
        error: F::Error,
    ) {
        let Some(active) = slots[slot_index].take() else {
            return;
        };
        let _ = active.reply.send(Err(error));
    }

    fn fail_all_active_slots(slots: &mut [Option<ActiveBatchSlot<F>>], error: F::Error) {
        let reason = format_error::<F>(error);
        for active in slots.iter_mut().filter_map(Option::take) {
            let _ = active.reply.send(Err(F::decode_failed(reason.clone())));
        }
    }

    fn first_active_job(slots: &[Option<ActiveBatchSlot<F>>]) -> Option<&F::Job> {
        slots
            .iter()
            .find_map(|active| active.as_ref().map(|active| F::slot_job(&active.slot)))
    }

    fn active_native_execution_context(
        slots: &[Option<ActiveBatchSlot<F>>],
    ) -> Result<Option<NativeExecutionContext>, F::Error> {
        Self::native_execution_context_for_options(
            slots
                .iter()
                .filter_map(Option::as_ref)
                .map(|active| &active.native_execution_context),
        )
    }

    fn native_execution_context_for_options<'a>(
        contexts: impl IntoIterator<Item = &'a Option<NativeExecutionContext>>,
    ) -> Result<Option<NativeExecutionContext>, F::Error> {
        let contexts = contexts.into_iter().collect::<Vec<_>>();
        let present = contexts
            .iter()
            .filter_map(|context| context.as_ref().cloned())
            .collect::<Vec<_>>();
        if present.is_empty() {
            return Ok(None);
        }
        if present.len() != contexts.len() {
            return Err(F::owner_failed(
                "seq2seq serve batch cannot mix requests with and without a native execution context"
                    .to_string(),
            ));
        }
        NativeExecutionContext::shared_lane(&present)
            .map_err(|error| F::owner_failed(error.to_string()))
    }

    fn decode_serial_job(&mut self, job: F::Job) -> Result<F::Output, F::Error> {
        let result = F::decode_serial(&mut self.serial_runtime, job);
        self.record_serial_runtime_receipt();
        result
    }

    pub(crate) fn batched_runtime_for(
        &mut self,
        job: &F::Job,
        n_seq: usize,
    ) -> Result<&mut F::Runtime, F::Error> {
        if let std::collections::hash_map::Entry::Vacant(e) = self.batched_runtimes.entry(n_seq) {
            let runtime = F::Runtime::build_batched(job, n_seq)?;
            e.insert(runtime);
            self.record_batched_runtime_receipt(n_seq);
        }
        let runtime = self.batched_runtimes.get_mut(&n_seq).ok_or_else(|| {
            F::owner_failed("seq2seq serve batch runtime cache is unexpectedly empty".to_string())
        })?;
        runtime.configure_for_job(job)?;
        Ok(runtime)
    }
}

/// Cohere's `map_decoder_error`: a decoder/runtime error is always normalized
/// into the family's `DecodeFailed` variant via `decode_failed(reason)`. In the
/// generic owner the runtime methods already return `F::Error` (the runtime's
/// associated `Error` is bound equal to `F::Error`), so the error is first
/// stringified and then re-wrapped through the family hook -- matching cohere's
/// `CohereServeBatchError::DecodeFailed { reason: error.to_string() }`.
fn map_decoder_error<F: Seq2SeqServeBatchFamily>(error: F::Error) -> F::Error {
    F::decode_failed(error.to_string())
}

/// Renders any `F::Error` to its `String` reason for cohere-faithful re-wrapping
/// (`fail_all_active_slots` and the refill-seed failure clone one reason into
/// every affected slot's `DecodeFailed` reply).
fn format_error<F: Seq2SeqServeBatchFamily>(error: F::Error) -> String {
    error.to_string()
}

// ===========================================================================
// Generic serve-batch engine layer (Wave B).
//
// The original three families' `*ServeBatchConfig` structs were field-identical, and
// their engine/spawn/submit/owner-loop/registry-lookup bodies were near-clones
// differing only in error-variant constructors and the per-family thread-name
// prefix. They are collapsed here into a generic `ServeBatchConfig` +
// `ServeBatchEngine<F>` driven by the `Seq2SeqServeBatchFamily` hooks. Each
// family keeps only its `static *_SERVE_BATCH_ENGINES` registry (Rust has no
// generic-over-`F` static) plus a thin `submit_*_serve_batch_job`.
// ===========================================================================

const SERVE_BATCH_SEND_TIMEOUT: Duration = Duration::from_secs(1);
const SERVE_BATCH_REPLY_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// The serve-batch owner-thread tuning shared by all participating families. Field
/// names match the previous per-family `*ServeBatchConfig` structs so the
/// per-family type aliases and their struct-literal construction in tests keep
/// compiling unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ServeBatchConfig {
    pub max_batch: usize,
    pub(crate) queue_capacity: usize,
    pub(crate) collect_window: Duration,
    pub(crate) send_timeout: Duration,
    pub(crate) reply_timeout: Duration,
    pub(crate) trace_batches: bool,
}

impl ServeBatchConfig {
    /// Resolves an eligible family's internal batch width from the server's
    /// admission limit. The limit is also the queue capacity: permits remain
    /// held from admission until the owner replies, so every admitted engine
    /// job always has a queue slot even when `N > family_batch_cap`.
    pub(crate) fn from_policy<F: Seq2SeqServeBatchFamily>(
        policy: ServeBatchPolicy,
    ) -> Option<Self> {
        policy.enabled().then_some(Self {
            max_batch: policy.max_native_sessions.min(F::MAX_BATCH_LIMIT),
            queue_capacity: policy.max_native_sessions,
            collect_window: SERVE_BATCH_COLLECT_WINDOW,
            send_timeout: SERVE_BATCH_SEND_TIMEOUT,
            reply_timeout: SERVE_BATCH_REPLY_TIMEOUT,
            trace_batches: serve_batch_trace_enabled(),
        })
    }

    #[cfg(test)]
    pub(crate) fn from_test_env<F: Seq2SeqServeBatchFamily>() -> Result<Option<Self>, F::Error> {
        let Some(max_batch) =
            crate::models::serve_batch_env::serve_batch_max_from_env(F::MAX_BATCH_LIMIT)
                .map_err(|error| F::invalid_test_env(error.env, error.raw, error.max))?
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            max_batch,
            queue_capacity: max_batch,
            collect_window: SERVE_BATCH_COLLECT_WINDOW,
            send_timeout: SERVE_BATCH_SEND_TIMEOUT,
            reply_timeout: SERVE_BATCH_REPLY_TIMEOUT,
            trace_batches: false,
        }))
    }

    /// Validates the config against a concrete job and resolves the effective
    /// max-batch: `max_batch >= 2` (else `F::invalid_enabled_batch`); the
    /// immutable planner must have authorized a reusable graph (else
    /// `F::unsupported_backend`); the VRAM cap (`F::vram_slot_bytes`); THEN
    /// `F::effective_max_batch_after_vram_cap` (whisper Vulkan->serial). The
    /// VRAM-cap-then-backend-name-cap ORDER is load-bearing -- it affects the
    /// engine key -- and is preserved exactly.
    pub(crate) fn validate_for_job<F: Seq2SeqServeBatchFamily>(
        self,
        job: &F::Job,
    ) -> Result<Self, F::Error> {
        if self.max_batch < 2 {
            return Err(F::invalid_enabled_batch(self.max_batch));
        }
        let backend = F::backend(job);
        if !reusable_decode_graph_supported(F::reuse_mode(job)) || F::uses_scheduler(job) {
            return Err(F::unsupported_backend(backend));
        }
        let lane = current_execution_lane();
        let max_batch = serve_batch_vram_capped_max_batch(
            self.max_batch,
            backend,
            lane.as_ref(),
            F::vram_slot_bytes(job),
        );
        let max_batch = F::effective_max_batch_after_vram_cap(max_batch, job)?;
        Ok(Self { max_batch, ..self })
    }
}

/// Derive an opaque owner descriptor from the family key without putting any
/// caller-owned path or model identifier into the receipt stream. The owner
/// component is the single serialized actor; retained runtimes and transient
/// session state are separate resources under it.
fn serve_batch_receipt_descriptor<F: Seq2SeqServeBatchFamily>(
    key: &F::EngineKey,
) -> Option<crate::models::runtime_receipts::RuntimeOwnerDescriptor> {
    let collector = current_runtime_receipts()?;
    let mut first = DefaultHasher::new();
    0_u8.hash(&mut first);
    key.hash(&mut first);
    let mut second = DefaultHasher::new();
    1_u8.hash(&mut second);
    key.hash(&mut second);
    let key_token = format!("{:016x}{:016x}", first.finish(), second.finish());
    let lane = crate::models::native_execution_services::current_execution_lane_key(
        F::engine_key_backend(key),
    )
    .receipt_projection(&collector);
    collector.owner_descriptor(
        "serve-batch.serialized-actor",
        Some(F::THREAD_NAME_PREFIX),
        Some(&key_token),
        lane,
    )
}

struct ServeBatchOwner<F: Seq2SeqServeBatchFamily> {
    // Field order is load-bearing: native runtimes and their resource guards
    // drop before the owner guard emits OwnerReleased.
    state: OwnerThreadState<F>,
    _receipt_owner: Option<RuntimeOwnerGuard>,
}

/// A single FIFO serialized actor for one serve-batch engine. The bounded
/// request channel is unchanged; the actor wrapper makes owner-thread lifetime
/// and deterministic shutdown explicit.
struct ServeBatchReuseState {
    latest_attempt: Option<crate::models::native_execution_services::ExecutionCacheAttemptId>,
    pending: bool,
    closed: bool,
}

struct ServeBatchReuseSignal {
    state: Mutex<ServeBatchReuseState>,
    collector: Option<RuntimeReceiptCollector>,
}

impl ServeBatchReuseSignal {
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

pub(crate) struct ServeBatchEngine<F: Seq2SeqServeBatchFamily> {
    sender: Mutex<Option<SyncSender<Envelope<F>>>>,
    reuse: Arc<ServeBatchReuseSignal>,
    worker: Mutex<Option<thread::JoinHandle<()>>>,
    worker_thread_id: thread::ThreadId,
    config: ServeBatchConfig,
    is_alive: Arc<AtomicBool>,
    #[cfg(test)]
    owner_ready: Arc<AtomicBool>,
}

impl<F: Seq2SeqServeBatchFamily> ServeBatchEngine<F> {
    fn shutdown_owner(&self) {
        // Closing the only request sender lets the owner drain accepted
        // requests and then leave its loop. Joining makes runtime/resource
        // release deterministic before a failed build wakes waiters.
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
}

impl<F: Seq2SeqServeBatchFamily> Drop for ServeBatchEngine<F> {
    fn drop(&mut self) {
        self.shutdown_owner();
    }
}

impl<F: Seq2SeqServeBatchFamily> ServeBatchEngine<F> {
    fn spawn(key: F::EngineKey, config: ServeBatchConfig) -> Result<Self, F::Error>
    where
        // The `Envelope<F>` (job + reply sender for `Result<Output, Error>`) is
        // moved into the spawned owner thread, so each crossing type must be
        // `Send`. All concrete families satisfy this.
        F::Job: Send,
        F::Output: Send,
        F::Error: Send,
    {
        let (sender, receiver) = mpsc::sync_channel(config.queue_capacity);
        let (is_alive, alive_guard) = OwnerAliveGuard::new();
        let thread_name = format!(
            "openasr-{}-serve-batch-{:?}-{}",
            F::THREAD_NAME_PREFIX,
            F::engine_key_backend(&key),
            config.max_batch
        );
        let owner_context = current_native_execution_context();
        let receipt_collector = current_runtime_receipts();
        let receipt_descriptor = serve_batch_receipt_descriptor::<F>(&key);
        let receipt_attempt_id =
            crate::models::native_execution_services::current_execution_cache_attempt_id();
        let reuse = Arc::new(ServeBatchReuseSignal {
            state: Mutex::new(ServeBatchReuseState {
                latest_attempt: None,
                pending: false,
                closed: false,
            }),
            collector: receipt_collector.clone(),
        });
        let worker_reuse = Arc::clone(&reuse);
        let owner_ready = Arc::new(AtomicBool::new(false));
        let worker_ready = Arc::clone(&owner_ready);
        let worker = thread::Builder::new()
            .name(thread_name)
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
                let owner = ServeBatchOwner {
                    state: OwnerThreadState::with_receipt_context(receipt_context),
                    _receipt_owner: receipt_owner,
                };
                owner_thread_loop::<F>(receiver, config, worker_reuse, worker_ready, owner)
            })
            .map_err(|error| F::thread_spawn_failed(error.to_string()))?;
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

    fn record_receipt_reuse(&self) {
        self.reuse
            .record(crate::models::native_execution_services::current_execution_cache_attempt_id());
    }

    pub(crate) fn submit(&self, job: F::Job) -> Result<F::Output, F::Error> {
        let context = Arc::clone(F::job_execution_context(&job));
        let native_execution_context = current_native_execution_context();
        let (reply, reply_rx) = mpsc::channel();
        let sender = self
            .sender
            .lock()
            .map_err(|_| F::owner_disconnected())?
            .as_ref()
            .cloned()
            .ok_or_else(F::owner_disconnected)?;
        serve_batch_submit_with_timeout(
            &sender,
            Envelope {
                job,
                context,
                native_execution_context,
                reply,
            },
            reply_rx,
            self.config.send_timeout,
            self.config.reply_timeout,
            F::queue_full,
            F::owner_disconnected,
            F::reply_timed_out,
        )
    }
}

impl<F: Seq2SeqServeBatchFamily> ServeBatchOwner<F> {
    fn consume_receipt_reuse(&self, reuse: &ServeBatchReuseSignal) {
        let Some(attempt) = reuse.consume() else {
            return;
        };
        if let Some(receipt_owner) = self._receipt_owner.as_ref() {
            receipt_owner.record_reuse(attempt);
        }
    }
}

fn owner_thread_loop<F: Seq2SeqServeBatchFamily>(
    receiver: Receiver<Envelope<F>>,
    config: ServeBatchConfig,
    reuse: Arc<ServeBatchReuseSignal>,
    ready: Arc<AtomicBool>,
    owner: ServeBatchOwner<F>,
) {
    ready.store(true, Ordering::Release);
    let mut owner = owner;
    let mut deferred = VecDeque::new();
    loop {
        owner.consume_receipt_reuse(&reuse);
        let Some(batch) = serve_batch_drain_compatible_batch(
            &mut deferred,
            &receiver,
            config.max_batch,
            config.collect_window,
            |first, next| F::can_batch_with(&first.job, &next.job),
        ) else {
            reuse.close_and_drop();
            break;
        };
        if config.trace_batches {
            eprintln!(
                "openasr {} serve batch: drained {} request(s)",
                F::THREAD_NAME_PREFIX,
                batch.len()
            );
        }
        deferred.extend(owner.state.run_batch(
            batch,
            &receiver,
            config.max_batch,
            config.trace_batches,
        ));
    }
}

/// Executor-owned registry for one serve-batch family.
///
/// The registry used to be a process-global `OnceLock` and smuggled a service
/// scope id into every key. That made ownership indirect: dropping one
/// `NativeExecutionServices` root did not drop its owner threads, and scoped
/// shutdown depended on ambient TLS. The registry is now an ordinary cloneable
/// value owned by the family executor. Executor clones share the same inner
/// map, while independently constructed service roots cannot observe one
/// another's engines and therefore need no scope component in their keys.
pub(crate) struct ServeBatchEngineRegistry<F: Seq2SeqServeBatchFamily> {
    engines: Arc<Mutex<BoundedServeBatchEngineCache<F::EngineKey, ServeBatchEngine<F>>>>,
    active: Arc<Mutex<HashMap<F::EngineKey, Weak<ServeBatchEngine<F>>>>>,
    pending: Arc<Mutex<HashMap<F::EngineKey, Arc<PendingServeBatchEngine<F>>>>>,
}

struct PendingServeBatchEngine<F: Seq2SeqServeBatchFamily> {
    result: Mutex<Option<Result<Arc<ServeBatchEngine<F>>, ()>>>,
    wake: Condvar,
}

impl<F: Seq2SeqServeBatchFamily> PendingServeBatchEngine<F> {
    fn new() -> Self {
        Self {
            result: Mutex::new(None),
            wake: Condvar::new(),
        }
    }

    fn complete(&self, result: Result<Arc<ServeBatchEngine<F>>, ()>) {
        let mut state = self
            .result
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *state = Some(result);
        self.wake.notify_all();
    }

    fn wait(&self) -> Result<Arc<ServeBatchEngine<F>>, ()> {
        let mut state = self
            .result
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.is_none() {
            state = self
                .wake
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state.as_ref().expect("pending result was set").clone()
    }
}

/// A small executor-local LRU for heavyweight serve-batch owner threads.
///
/// Each owner may retain a decoder, logits runtime, reusable graphs, and up to
/// `MAX_BATCH_LIMIT` batch-width variants. Keeping an unbounded map here made
/// model/content/backend churn accumulate owner threads until the service-wide
/// idle reaper ran. Four idle engines matches the family actor-pool bound. If
/// all cached engines are concurrently borrowed, a newly spawned engine still
/// serves its current request but is deliberately not retained afterwards.
pub(crate) const SERVE_BATCH_ENGINE_REGISTRY_MAX_ENTRIES: usize = 4;

pub(crate) struct BoundedServeBatchEngineCache<K, E> {
    entries: HashMap<K, Arc<E>>,
    recency: VecDeque<K>,
    generation: u64,
}

impl<K, E> Default for BoundedServeBatchEngineCache<K, E> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            recency: VecDeque::new(),
            generation: 0,
        }
    }
}

/// Shared bounded single-flight/active-owner registry used by all serve-batch
/// family implementations, including Qwen's family-specific decode loop.
pub(crate) struct ServeBatchActiveRegistry<K, E> {
    pub(crate) engines: Arc<Mutex<BoundedServeBatchEngineCache<K, E>>>,
    active: Arc<Mutex<HashMap<K, Weak<E>>>>,
    pending: Arc<Mutex<HashMap<K, Arc<PendingSingleFlight<E>>>>>,
}

struct PendingSingleFlight<E> {
    result: Mutex<Option<Result<Arc<E>, ()>>>,
    wake: Condvar,
}

impl<E> PendingSingleFlight<E> {
    fn new() -> Self {
        Self {
            result: Mutex::new(None),
            wake: Condvar::new(),
        }
    }

    fn complete(&self, result: Result<Arc<E>, ()>) {
        let mut state = self
            .result
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *state = Some(result);
        self.wake.notify_all();
    }

    fn wait(&self) -> Result<Arc<E>, ()> {
        let mut state = self
            .result
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.is_none() {
            state = self
                .wake
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state
            .as_ref()
            .expect("single-flight result was set")
            .clone()
    }
}

impl<K, E> Clone for ServeBatchActiveRegistry<K, E> {
    fn clone(&self) -> Self {
        Self {
            engines: Arc::clone(&self.engines),
            active: Arc::clone(&self.active),
            pending: Arc::clone(&self.pending),
        }
    }
}

impl<K, E> Default for ServeBatchActiveRegistry<K, E> {
    fn default() -> Self {
        Self {
            engines: Arc::new(Mutex::new(BoundedServeBatchEngineCache::default())),
            active: Arc::new(Mutex::new(HashMap::new())),
            pending: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl<K, E> ServeBatchActiveRegistry<K, E>
where
    K: Clone + Eq + std::hash::Hash + Send + 'static,
    E: Send + Sync + 'static,
{
    pub(crate) fn lookup_or_build<Err, Build, Alive, Reuse, Shutdown, ErrorFactory>(
        &self,
        key: K,
        build: Build,
        is_alive: Alive,
        on_reuse: Reuse,
        shutdown: Shutdown,
        registry_error: ErrorFactory,
    ) -> Result<Arc<E>, Err>
    where
        Build: Fn() -> Result<Arc<E>, Err> + Clone,
        Alive: Fn(&E) -> bool + Copy + Send + Sync + 'static,
        Reuse: Fn(&E) + Copy + Send + Sync + 'static,
        Shutdown: Fn(&E) + Copy + Send + Sync + 'static,
        ErrorFactory: Fn() -> Err + Copy,
    {
        #[cfg(not(test))]
        if current_execution_cache_attempt_id().is_none() {
            return Err(registry_error());
        }
        loop {
            let pending = {
                let mut engines = self.engines.lock().map_err(|_| registry_error())?;
                if let Some(engine) = engines.get(&key) {
                    if is_alive(&engine) {
                        on_reuse(&engine);
                        return Ok(engine);
                    }
                    engines.remove(&key);
                }
                let mut pending_map = self.pending.lock().map_err(|_| registry_error())?;
                if let Some(pending) = pending_map.get(&key) {
                    Arc::clone(pending)
                } else {
                    let mut active = self.active.lock().map_err(|_| registry_error())?;
                    active.retain(|_, weak| weak.strong_count() > 0);
                    if let Some(weak) = active.get(&key) {
                        if let Some(engine) = weak.upgrade()
                            && is_alive(&engine)
                        {
                            on_reuse(&engine);
                            return Ok(engine);
                        }
                        active.remove(&key);
                    }
                    let generation = engines.generation();
                    let pending = Arc::new(PendingSingleFlight::new());
                    pending_map.insert(key.clone(), Arc::clone(&pending));
                    drop(active);
                    drop(pending_map);
                    drop(engines);

                    let engine = match build() {
                        Ok(engine) => engine,
                        Err(error) => {
                            pending.complete(Err(()));
                            self.remove_pending_if(&key, &pending);
                            return Err(error);
                        }
                    };
                    let mut active = match self.active.lock() {
                        Ok(active) => active,
                        Err(_) => {
                            shutdown(&engine);
                            drop(engine);
                            pending.complete(Err(()));
                            self.remove_pending_if(&key, &pending);
                            return Err(registry_error());
                        }
                    };
                    active.retain(|_, weak| weak.strong_count() > 0);
                    active.insert(key.clone(), Arc::downgrade(&engine));
                    drop(active);

                    let rollback_registry = self.clone();
                    let rollback_key = key.clone();
                    let rollback_engine = Arc::clone(&engine);
                    let rollback_pending = Arc::clone(&pending);
                    crate::models::native_execution_services::stage_execution_cache_rollback(
                        move || {
                            rollback_registry.evict_exact(&rollback_key, &rollback_engine);
                            shutdown(&rollback_engine);
                            drop(rollback_engine);
                            rollback_pending.complete(Err(()));
                            rollback_registry.remove_pending_if(&rollback_key, &rollback_pending);
                        },
                    );
                    let commit_registry = self.clone();
                    let commit_key = key.clone();
                    let commit_engine = Arc::clone(&engine);
                    let commit_pending = Arc::clone(&pending);
                    crate::models::native_execution_services::stage_execution_cache_commit(
                        move || {
                            let Ok(mut engines) = commit_registry.engines.lock() else {
                                commit_registry.evict_active_exact(&commit_key, &commit_engine);
                                shutdown(&commit_engine);
                                drop(commit_engine);
                                commit_pending.complete(Err(()));
                                commit_registry.remove_pending_if(&commit_key, &commit_pending);
                                return;
                            };
                            if engines.generation() != generation {
                                commit_registry.evict_active_exact(&commit_key, &commit_engine);
                                shutdown(&commit_engine);
                                drop(commit_engine);
                                commit_pending.complete(Err(()));
                                commit_registry.remove_pending_if(&commit_key, &commit_pending);
                                return;
                            }
                            let _ = engines.insert_if_idle_capacity(
                                commit_key.clone(),
                                Arc::clone(&commit_engine),
                            );
                            if let Ok(mut active) = commit_registry.active.lock() {
                                active.retain(|_, weak| weak.strong_count() > 0);
                            }
                            commit_pending.complete(Ok(commit_engine));
                            commit_registry.remove_pending_if(&commit_key, &commit_pending);
                        },
                    );
                    return Ok(engine);
                }
            };
            match pending.wait() {
                Ok(engine) if is_alive(&engine) => {
                    on_reuse(&engine);
                    return Ok(engine);
                }
                Ok(_) | Err(()) => {
                    self.remove_pending_if(&key, &pending);
                }
            }
        }
    }

    fn remove_pending_if(&self, key: &K, expected: &Arc<PendingSingleFlight<E>>) {
        let Ok(mut pending) = self.pending.lock() else {
            return;
        };
        if pending
            .get(key)
            .is_some_and(|current| Arc::ptr_eq(current, expected))
        {
            pending.remove(key);
        }
    }

    fn evict_active_exact(&self, key: &K, engine: &Arc<E>) {
        let Ok(mut active) = self.active.lock() else {
            return;
        };
        if active.get(key).is_some_and(|weak| {
            weak.upgrade()
                .is_some_and(|current| Arc::ptr_eq(&current, engine))
        }) {
            active.remove(key);
        }
    }

    pub(crate) fn evict_exact(&self, key: &K, engine: &Arc<E>) {
        if let Ok(mut engines) = self.engines.lock()
            && engines.contains_ptr(key, engine)
        {
            engines.remove(key);
        }
        self.evict_active_exact(key, engine);
    }

    pub(crate) fn shutdown(&self) {
        if let Ok(mut engines) = self.engines.lock() {
            engines.clear();
        }
        if let Ok(mut active) = self.active.lock() {
            active.clear();
        }
        if let Ok(mut pending) = self.pending.lock() {
            for build in pending.values() {
                build.complete(Err(()));
            }
            pending.clear();
        }
    }
}

impl<K, E> BoundedServeBatchEngineCache<K, E>
where
    K: Clone + Eq + std::hash::Hash,
{
    pub(crate) fn get(&mut self, key: &K) -> Option<Arc<E>> {
        let engine = Arc::clone(self.entries.get(key)?);
        self.touch(key);
        Some(engine)
    }

    pub(crate) fn insert_if_idle_capacity(&mut self, key: K, engine: Arc<E>) -> bool {
        if self.entries.contains_key(&key) {
            self.remove(&key);
        }
        while self.entries.len() >= SERVE_BATCH_ENGINE_REGISTRY_MAX_ENTRIES {
            let Some(evictable) = self.recency.iter().find_map(|candidate| {
                self.entries
                    .get(candidate)
                    .filter(|cached| Arc::strong_count(cached) == 1)
                    .map(|_| candidate.clone())
            }) else {
                return false;
            };
            self.remove(&evictable);
        }
        self.entries.insert(key.clone(), engine);
        self.touch(&key);
        true
    }

    pub(crate) fn remove(&mut self, key: &K) -> Option<Arc<E>> {
        self.recency.retain(|candidate| candidate != key);
        self.entries.remove(key)
    }

    pub(crate) fn contains_ptr(&self, key: &K, engine: &Arc<E>) -> bool {
        self.entries
            .get(key)
            .is_some_and(|cached| Arc::ptr_eq(cached, engine))
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.recency.clear();
        self.generation = self.generation.wrapping_add(1);
    }

    pub(crate) const fn generation(&self) -> u64 {
        self.generation
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn touch(&mut self, key: &K) {
        self.recency.retain(|candidate| candidate != key);
        self.recency.push_back(key.clone());
    }
}

impl<F: Seq2SeqServeBatchFamily> Clone for ServeBatchEngineRegistry<F> {
    fn clone(&self) -> Self {
        Self {
            engines: Arc::clone(&self.engines),
            active: Arc::clone(&self.active),
            pending: Arc::clone(&self.pending),
        }
    }
}

impl<F: Seq2SeqServeBatchFamily> std::fmt::Debug for ServeBatchEngineRegistry<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ServeBatchEngineRegistry")
            .finish_non_exhaustive()
    }
}

impl<F: Seq2SeqServeBatchFamily> Default for ServeBatchEngineRegistry<F> {
    fn default() -> Self {
        Self {
            engines: Arc::new(Mutex::new(BoundedServeBatchEngineCache::default())),
            active: Arc::new(Mutex::new(HashMap::new())),
            pending: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl<F: Seq2SeqServeBatchFamily> ServeBatchEngineRegistry<F>
where
    F::Job: Send,
    F::Output: Send,
    F::Error: Send,
{
    /// Registry lookup with dead-owner respawn. Publication is delayed until
    /// the enclosing execution candidate succeeds; the staged closure owns a
    /// clone of this explicit registry rather than reaching a process static.
    pub(crate) fn engine_for_key(
        &self,
        key: F::EngineKey,
        config: ServeBatchConfig,
    ) -> Result<Arc<ServeBatchEngine<F>>, F::Error> {
        #[cfg(not(test))]
        if current_execution_cache_attempt_id().is_none() {
            return Err(F::candidate_attempt_required());
        }
        loop {
            let pending = {
                let mut engines = self.engines.lock().map_err(|_| F::registry_poisoned())?;
                if let Some(engine) = engines.get(&key) {
                    if serve_batch_owner_alive(&engine.is_alive) {
                        engine.record_receipt_reuse();
                        let registry = self.clone();
                        let failed_key = key.clone();
                        let failed_engine = Arc::clone(&engine);
                        crate::models::native_execution_services::stage_execution_cache_rollback(
                            move || registry.evict_exact(&failed_key, &failed_engine),
                        );
                        return Ok(engine);
                    }
                    engines.remove(&key);
                }
                let mut pending_map = self.pending.lock().map_err(|_| F::registry_poisoned())?;
                if let Some(pending) = pending_map.get(&key) {
                    Arc::clone(pending)
                } else {
                    let mut active = self.active.lock().map_err(|_| F::registry_poisoned())?;
                    active.retain(|_, weak| weak.strong_count() > 0);
                    if let Some(weak) = active.get(&key) {
                        if let Some(engine) = weak.upgrade()
                            && serve_batch_owner_alive(&engine.is_alive)
                        {
                            engine.record_receipt_reuse();
                            let registry = self.clone();
                            let failed_key = key.clone();
                            let failed_engine = Arc::clone(&engine);
                            crate::models::native_execution_services::
                                stage_execution_cache_rollback(move || {
                                    registry.evict_exact(&failed_key, &failed_engine)
                                });
                            return Ok(engine);
                        }
                        active.remove(&key);
                    }
                    let generation = engines.generation();
                    let pending = Arc::new(PendingServeBatchEngine::new());
                    pending_map.insert(key.clone(), Arc::clone(&pending));
                    drop(pending_map);
                    drop(active);
                    drop(engines);
                    return self.build_and_publish_pending(key, config, generation, pending);
                }
            };

            match pending.wait() {
                Ok(engine) if serve_batch_owner_alive(&engine.is_alive) => {
                    engine.record_receipt_reuse();
                    return Ok(engine);
                }
                Ok(_) | Err(()) => {
                    self.remove_pending_if(&key, &pending);
                    continue;
                }
            }
        }
    }

    fn build_and_publish_pending(
        &self,
        key: F::EngineKey,
        config: ServeBatchConfig,
        generation: u64,
        pending: Arc<PendingServeBatchEngine<F>>,
    ) -> Result<Arc<ServeBatchEngine<F>>, F::Error> {
        let engine = match ServeBatchEngine::<F>::spawn(key.clone(), config) {
            Ok(engine) => Arc::new(engine),
            Err(error) => {
                pending.complete(Err(()));
                self.remove_pending_if(&key, &pending);
                return Err(error);
            }
        };
        let mut active = match self.active.lock() {
            Ok(active) => active,
            Err(_) => {
                engine.shutdown_owner();
                drop(engine);
                pending.complete(Err(()));
                self.remove_pending_if(&key, &pending);
                return Err(F::registry_poisoned());
            }
        };
        active.retain(|_, weak| weak.strong_count() > 0);
        active.insert(key.clone(), Arc::downgrade(&engine));
        drop(active);

        let failed_registry = self.clone();
        let failed_key = key.clone();
        let failed_engine = Arc::clone(&engine);
        let failed_pending = Arc::clone(&pending);
        crate::models::native_execution_services::stage_execution_cache_rollback(move || {
            failed_registry.evict_exact(&failed_key, &failed_engine);
            failed_registry.remove_active_exact(&failed_key, &failed_engine);
            failed_engine.shutdown_owner();
            drop(failed_engine);
            failed_registry.complete_pending_failure(&failed_key, &failed_pending);
        });

        let registry = self.clone();
        let staged_engine = Arc::clone(&engine);
        let staged_pending = Arc::clone(&pending);
        crate::models::native_execution_services::stage_execution_cache_commit(move || {
            let Ok(mut engines) = registry.engines.lock() else {
                registry.remove_active_exact(&key, &staged_engine);
                staged_engine.shutdown_owner();
                drop(staged_engine);
                registry.complete_pending_failure(&key, &staged_pending);
                return;
            };
            if engines.generation() != generation {
                registry.remove_active_exact(&key, &staged_engine);
                staged_engine.shutdown_owner();
                drop(staged_engine);
                registry.complete_pending_failure(&key, &staged_pending);
                return;
            }
            if let Some(existing) = engines.get(&key)
                && serve_batch_owner_alive(&existing.is_alive)
            {
                if let Ok(mut active) = registry.active.lock() {
                    active.insert(key.clone(), Arc::downgrade(&existing));
                }
                staged_pending.complete(Ok(existing));
                registry.remove_pending_if(&key, &staged_pending);
                return;
            }
            engines.remove(&key);
            let _ = engines.insert_if_idle_capacity(key.clone(), Arc::clone(&staged_engine));
            if let Ok(mut active) = registry.active.lock() {
                active.retain(|_, weak| weak.strong_count() > 0);
            }
            // The new owner still serves its admitting request even when an
            // all-borrowed LRU deliberately does not retain it.
            staged_pending.complete(Ok(staged_engine));
            registry.remove_pending_if(&key, &staged_pending);
        });
        Ok(engine)
    }

    fn remove_pending_if(&self, key: &F::EngineKey, expected: &Arc<PendingServeBatchEngine<F>>) {
        let Ok(mut pending) = self.pending.lock() else {
            return;
        };
        if pending
            .get(key)
            .is_some_and(|current| Arc::ptr_eq(current, expected))
        {
            pending.remove(key);
        }
    }

    fn complete_pending_failure(
        &self,
        key: &F::EngineKey,
        pending: &Arc<PendingServeBatchEngine<F>>,
    ) {
        pending.complete(Err(()));
        self.remove_pending_if(key, pending);
    }

    fn remove_active_exact(&self, key: &F::EngineKey, engine: &Arc<ServeBatchEngine<F>>) {
        let Ok(mut active) = self.active.lock() else {
            return;
        };
        if active.get(key).is_some_and(|weak| {
            weak.upgrade()
                .is_some_and(|current| Arc::ptr_eq(&current, engine))
        }) {
            active.remove(key);
        }
    }

    fn evict_exact(&self, key: &F::EngineKey, engine: &Arc<ServeBatchEngine<F>>) {
        if let Ok(mut engines) = self.engines.lock()
            && engines.contains_ptr(key, engine)
        {
            engines.remove(key);
        }
        self.remove_active_exact(key, engine);
    }

    /// Removes the exact owner that participated in a typed candidate failure.
    /// Pointer identity prevents an older failed submitter from evicting a
    /// newer replacement published under the same key.
    pub(crate) fn evict_after_candidate_failure(
        &self,
        key: &F::EngineKey,
        engine: &Arc<ServeBatchEngine<F>>,
    ) {
        if crate::models::native_execution_services::current_execution_candidate_failure().is_none()
        {
            return;
        }
        self.evict_exact(key, engine);
    }

    /// Drops this executor's registry references. In-flight submitters retain
    /// their engine until their reply completes; after the final sender drops,
    /// the owner drains and exits through the ordinary channel lifecycle.
    pub(crate) fn shutdown(&self) {
        if let Ok(mut engines) = self.engines.lock() {
            engines.clear();
            if let Ok(mut active) = self.active.lock() {
                active.clear();
            }
            if let Ok(mut pending) = self.pending.lock() {
                for build in pending.values() {
                    build.complete(Err(()));
                }
                pending.clear();
            }
        }
    }
}

/// Shared-layer slot isolation tests.
///
/// `cohere` / `moonshine` / `whisper` / `qwen` are wired onto the exact
/// `OwnerThreadState::run_batch` / `decode_continuous_batch` code in this
/// module (see the module doc comment), so a fake, ggml-free family here
/// exercises the real per-slot cancellation logic those families share -- one
/// test proves it for all of them instead of near-identical
/// copies each needing a real model pack.
#[cfg(test)]
mod slot_isolation_tests {
    use super::*;
    use crate::RequestExecutionContext;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use thiserror::Error;

    #[derive(Debug, Error, PartialEq, Eq)]
    enum FakeError {
        #[error("fake decode failed: {0}")]
        DecodeFailed(String),
        #[error("fake owner failed: {0}")]
        OwnerFailed(String),
        #[error("fake unsupported backend")]
        UnsupportedBackend,
        #[error("fake registry poisoned")]
        RegistryPoisoned,
        #[error("fake thread spawn failed: {0}")]
        ThreadSpawnFailed(String),
        #[error("fake queue full")]
        QueueFull,
        #[error("fake owner disconnected")]
        OwnerDisconnected,
        #[error("fake reply timed out")]
        ReplyTimedOut,
        #[error("fake invalid enabled batch: {0}")]
        InvalidEnabledBatch(usize),
    }

    /// A job that always generates `max_tokens` tokens (no real vocabulary or
    /// logits -- `slot_select_next_token` below ignores the logits content
    /// entirely and just counts steps), carrying the same explicit execution
    /// context every real family job now requires.
    #[derive(Clone)]
    struct FakeJob {
        id: u32,
        max_tokens: usize,
        execution_context: Arc<RequestExecutionContext>,
        /// Set only on the job that becomes the *first* active slot (the one
        /// `build_batched` is constructed from): fires `request_cancel` on
        /// that same job's own context after this many
        /// `compute_reused_batched_step_logits` calls, simulating an
        /// independent HTTP thread canceling this request concurrently while
        /// the owner is mid-batch.
        self_cancel_after_step: Option<usize>,
    }

    struct FakeSlot {
        job: FakeJob,
        generated: Vec<u32>,
    }

    /// No real tensors: `compute_*_logits` return a correctly-sized dummy
    /// buffer (the owner validates its length against `vocab_size * n_seq`),
    /// and `slot_select_next_token` ignores its content -- decode progress is
    /// just a step counter. `build_count` proves the shared "graph" is built
    /// once and never rebuilt/aborted because a sibling slot was canceled.
    struct FakeRuntime {
        step_calls: usize,
        cancel_hook: Option<(usize, Arc<RequestExecutionContext>)>,
    }

    static FAKE_VOCAB_SIZE: usize = 4;
    static FAKE_RUNTIME_BUILD_COUNT: AtomicUsize = AtomicUsize::new(0);

    impl Seq2SeqServeRuntime for FakeRuntime {
        type Job = FakeJob;
        type Error = FakeError;

        fn build_serial(_job: &Self::Job) -> Result<Self, Self::Error> {
            FAKE_RUNTIME_BUILD_COUNT.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(Self {
                step_calls: 0,
                cancel_hook: None,
            })
        }

        fn build_batched(job: &Self::Job, _n_seq: usize) -> Result<Self, Self::Error> {
            FAKE_RUNTIME_BUILD_COUNT.fetch_add(1, AtomicOrdering::SeqCst);
            let cancel_hook = job
                .self_cancel_after_step
                .map(|after_step| (after_step, Arc::clone(&job.execution_context)));
            Ok(Self {
                step_calls: 0,
                cancel_hook,
            })
        }

        fn populate_cross_attention_cache_serial(
            &mut self,
            _job: &Self::Job,
        ) -> Result<(), Self::Error> {
            Ok(())
        }

        fn populate_cross_attention_cache_slot(
            &mut self,
            _slot_index: usize,
            _job: &Self::Job,
        ) -> Result<(), Self::Error> {
            Ok(())
        }

        fn compute_batched_prefill_logits(
            &mut self,
            _prompt_tokens: &[u32],
        ) -> Result<Vec<f32>, Self::Error> {
            Ok(vec![0.0; FAKE_VOCAB_SIZE * 2])
        }

        fn compute_reused_batched_step_logits(
            &mut self,
            _token_ids: &[u32],
            _positions: &[usize],
            _totals: &[usize],
        ) -> Result<Vec<f32>, Self::Error> {
            self.step_calls += 1;
            if let Some((after_step, context)) = &self.cancel_hook
                && self.step_calls == *after_step
            {
                context.control.request_cancel();
            }
            Ok(vec![0.0; FAKE_VOCAB_SIZE * 2])
        }
    }

    struct FakeFamily;

    impl Seq2SeqServeBatchFamily for FakeFamily {
        type Runtime = FakeRuntime;
        type Job = FakeJob;
        type Slot = FakeSlot;
        type Output = u32;
        type Error = FakeError;
        type EngineKey = u32;

        const THREAD_NAME_PREFIX: &'static str = "fake";
        const MAX_BATCH_LIMIT: usize = 8;

        fn engine_key(job: &Self::Job, _max_batch: usize) -> Self::EngineKey {
            job.id
        }

        fn engine_key_backend(_key: &Self::EngineKey) -> GgmlCpuGraphBackend {
            GgmlCpuGraphBackend::Cpu
        }

        fn can_batch_with(_a: &Self::Job, _b: &Self::Job) -> bool {
            true
        }

        fn vram_slot_bytes(_job: &Self::Job) -> usize {
            0
        }

        fn backend(_job: &Self::Job) -> GgmlCpuGraphBackend {
            GgmlCpuGraphBackend::Cpu
        }

        fn uses_scheduler(_job: &Self::Job) -> bool {
            false
        }

        fn reuse_mode(_job: &Self::Job) -> GgmlDecodeReuseMode {
            GgmlDecodeReuseMode::FreshGraph
        }

        fn initial_prompt_tokens(_job: &Self::Job) -> &[u32] {
            &[0]
        }

        fn vocab_size(_job: &Self::Job) -> usize {
            FAKE_VOCAB_SIZE
        }

        fn max_generated_tokens(job: &Self::Job) -> usize {
            job.max_tokens
        }

        fn decoder_max_context(_job: &Self::Job) -> usize {
            64
        }

        fn slot_new(job: Self::Job) -> Result<Self::Slot, Self::Error> {
            Ok(FakeSlot {
                job,
                generated: Vec::new(),
            })
        }

        fn slot_job(slot: &Self::Slot) -> &Self::Job {
            &slot.job
        }

        fn slot_generated(slot: &Self::Slot) -> &[u32] {
            &slot.generated
        }

        fn slot_done(slot: &Self::Slot) -> bool {
            slot.generated.len() >= slot.job.max_tokens
        }

        fn slot_select_next_token(
            slot: &mut Self::Slot,
            _logits: Vec<f32>,
        ) -> Result<(), Self::Error> {
            slot.generated.push(slot.generated.len() as u32);
            Ok(())
        }

        fn slot_finish(slot: Self::Slot) -> Result<Self::Output, Self::Error> {
            Ok(slot.job.id)
        }

        fn decode_serial(
            serial_runtime: &mut Option<Self::Runtime>,
            job: Self::Job,
        ) -> Result<Self::Output, Self::Error> {
            if serial_runtime.is_none() {
                *serial_runtime = Some(FakeRuntime::build_serial(&job)?);
            }
            let mut slot = FakeSlot {
                job,
                generated: Vec::new(),
            };
            while !FakeFamily::slot_done(&slot) {
                FakeFamily::slot_select_next_token(&mut slot, Vec::new())?;
            }
            FakeFamily::slot_finish(slot)
        }

        fn decode_failed(reason: String) -> Self::Error {
            FakeError::DecodeFailed(reason)
        }

        fn owner_failed(reason: String) -> Self::Error {
            FakeError::OwnerFailed(reason)
        }

        fn job_execution_context(job: &Self::Job) -> &Arc<RequestExecutionContext> {
            &job.execution_context
        }

        #[cfg(test)]
        fn invalid_test_env(_env: &'static str, _raw: String, _max: usize) -> Self::Error {
            FakeError::InvalidEnabledBatch(0)
        }

        fn invalid_enabled_batch(max_batch: usize) -> Self::Error {
            FakeError::InvalidEnabledBatch(max_batch)
        }

        fn unsupported_backend(_backend: GgmlCpuGraphBackend) -> Self::Error {
            FakeError::UnsupportedBackend
        }

        fn registry_poisoned() -> Self::Error {
            FakeError::RegistryPoisoned
        }

        fn thread_spawn_failed(reason: String) -> Self::Error {
            FakeError::ThreadSpawnFailed(reason)
        }

        fn queue_full() -> Self::Error {
            FakeError::QueueFull
        }

        fn owner_disconnected() -> Self::Error {
            FakeError::OwnerDisconnected
        }

        fn reply_timed_out() -> Self::Error {
            FakeError::ReplyTimedOut
        }
    }

    /// Capacity 2, two requests (A, B) land in the same batched owner. A is
    /// canceled mid-batch (simulating an independent HTTP thread flipping its
    /// context while the owner is between token-step graph calls); B must
    /// finish normally and the shared "graph" (`FakeRuntime`) must be built
    /// exactly once -- proving the cancellation of A never tore down or
    /// rebuilt the runtime B is still using.
    #[test]
    fn canceling_one_slot_finishes_only_that_slot_and_leaves_the_sibling_and_shared_runtime_alone()
    {
        FAKE_RUNTIME_BUILD_COUNT.store(0, AtomicOrdering::SeqCst);

        let context_a = Arc::new(RequestExecutionContext::new(
            Some("job-a".to_string()),
            Arc::new(crate::TranscriptionControl::new()),
        ));
        let context_b = Arc::new(RequestExecutionContext::new(
            Some("job-b".to_string()),
            Arc::new(crate::TranscriptionControl::new()),
        ));

        let job_a = FakeJob {
            id: 1,
            max_tokens: 4,
            execution_context: Arc::clone(&context_a),
            // A becomes the first active slot (batch order), so its own
            // cancel fires from inside the fake "graph" after the first
            // post-prefill step -- well before its own 4-token budget and
            // before B's.
            self_cancel_after_step: Some(1),
        };
        let job_b = FakeJob {
            id: 2,
            max_tokens: 4,
            execution_context: Arc::clone(&context_b),
            self_cancel_after_step: None,
        };

        let (reply_a, reply_a_rx) = mpsc::channel();
        let (reply_b, reply_b_rx) = mpsc::channel();
        let batch = vec![
            Envelope {
                job: job_a,
                context: context_a,
                native_execution_context: None,
                reply: reply_a,
            },
            Envelope {
                job: job_b,
                context: context_b,
                native_execution_context: None,
                reply: reply_b,
            },
        ];

        let (_receiver_keepalive, receiver) = mpsc::channel();
        let mut state = OwnerThreadState::<FakeFamily>::new();
        let deferred = state.run_batch(batch, &receiver, 2, false);
        assert!(deferred.is_empty());

        let result_a = reply_a_rx
            .recv()
            .expect("A's reply channel must receive a result");
        assert!(
            matches!(result_a, Err(FakeError::DecodeFailed(ref reason)) if reason.contains("canceled by transcription control")),
            "canceled slot A must fail with the stable cancel marker, got {result_a:?}"
        );

        let result_b = reply_b_rx
            .recv()
            .expect("B's reply channel must receive a result");
        assert_eq!(
            result_b,
            Ok(2),
            "healthy sibling B must finish normally despite A's cancellation"
        );

        assert_eq!(
            FAKE_RUNTIME_BUILD_COUNT.load(AtomicOrdering::SeqCst),
            1,
            "canceling A must not rebuild or abort the shared batched runtime B still uses"
        );
    }

    #[test]
    fn requests_from_different_native_execution_lanes_fail_before_runtime_build() {
        let first_services =
            crate::models::native_execution_services::test_native_execution_services();
        let second_services =
            crate::models::native_execution_services::test_native_execution_services();
        let capture =
            |services: &crate::models::native_execution_services::NativeExecutionServices| {
                let _guard =
                    crate::models::native_execution_services::install_native_execution_services(
                        services,
                    );
                current_native_execution_context().expect("installed native execution context")
            };
        let first_native = capture(first_services.as_ref());
        let second_native = capture(second_services.as_ref());
        let first_request = Arc::new(RequestExecutionContext::new(
            Some("first-lane".to_string()),
            Arc::new(crate::TranscriptionControl::new()),
        ));
        let second_request = Arc::new(RequestExecutionContext::new(
            Some("second-lane".to_string()),
            Arc::new(crate::TranscriptionControl::new()),
        ));
        let (first_reply, first_rx) = mpsc::channel();
        let (second_reply, second_rx) = mpsc::channel();
        let batch = vec![
            Envelope {
                job: FakeJob {
                    id: 1,
                    max_tokens: 2,
                    execution_context: Arc::clone(&first_request),
                    self_cancel_after_step: None,
                },
                context: first_request,
                native_execution_context: Some(first_native),
                reply: first_reply,
            },
            Envelope {
                job: FakeJob {
                    id: 2,
                    max_tokens: 2,
                    execution_context: Arc::clone(&second_request),
                    self_cancel_after_step: None,
                },
                context: second_request,
                native_execution_context: Some(second_native),
                reply: second_reply,
            },
        ];

        let (_receiver_keepalive, receiver) = mpsc::channel();
        let mut state = OwnerThreadState::<FakeFamily>::new();
        assert!(state.run_batch(batch, &receiver, 2, false).is_empty());
        for result in [first_rx.recv().unwrap(), second_rx.recv().unwrap()] {
            assert!(
                matches!(result, Err(FakeError::DecodeFailed(ref reason)) if reason.contains("does not share the batch execution scope")),
                "incompatible execution lanes must fail closed, got {result:?}"
            );
        }
        assert!(
            state.batched_runtimes.is_empty(),
            "an incompatible lane must be rejected before resident runtime allocation"
        );
    }

    #[test]
    fn serve_batch_receipts_trace_serialized_owner_runtime_width_and_release() {
        let services = crate::models::native_execution_services::test_native_execution_services();
        let services_guard =
            crate::models::native_execution_services::install_native_execution_services(
                services.as_ref(),
            );
        let config = ServeBatchConfig {
            max_batch: 2,
            queue_capacity: 2,
            // Keep the first request open until the synchronized sibling arrives.
            collect_window: Duration::from_secs(30),
            send_timeout: Duration::from_secs(1),
            reply_timeout: Duration::from_secs(5),
            trace_batches: false,
        };
        let registry = ServeBatchEngineRegistry::<FakeFamily>::default();
        let engine = registry
            .engine_for_key(7, config)
            .expect("serve-batch actor");
        let reused_engine = registry
            .engine_for_key(7, config)
            .expect("reused serve-batch actor");
        assert!(Arc::ptr_eq(&engine, &reused_engine));
        drop(reused_engine);
        drop(services_guard);

        fn receipt_job(id: u32) -> FakeJob {
            FakeJob {
                id,
                max_tokens: 2,
                execution_context: Arc::new(RequestExecutionContext::new(
                    Some(format!("receipt-job-{id}")),
                    Arc::new(crate::TranscriptionControl::new()),
                )),
                self_cancel_after_step: None,
            }
        }
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let first_engine = Arc::clone(&engine);
        let first_barrier = Arc::clone(&barrier);
        let first = std::thread::spawn(move || {
            first_barrier.wait();
            first_engine.submit(receipt_job(1))
        });
        let second_engine = Arc::clone(&engine);
        let second_barrier = Arc::clone(&barrier);
        let second = std::thread::spawn(move || {
            second_barrier.wait();
            second_engine.submit(receipt_job(2))
        });
        barrier.wait();

        assert_eq!(first.join().expect("first submitter"), Ok(1));
        assert_eq!(second.join().expect("second submitter"), Ok(2));

        let live = services.runtime_receipts().snapshot();
        assert_eq!(live.live_owners.len(), 1);
        let resource_count = live.live_owners[0].resources.len();
        // A two-job batch can keep the original width and a compacted width
        // resident; each width is one no-broker runtime receipt.
        assert!(
            (1..=2).contains(&resource_count),
            "serialized actor retains one receipt per resident runtime width, got {resource_count}"
        );
        let resource = live.live_owners[0]
            .resources
            .values()
            .next()
            .expect("retained batch resource");
        assert!(resource.descriptor.domain.is_none());
        assert_eq!(
            resource.descriptor.requested,
            crate::models::runtime_receipts::RuntimeReceiptMetric::Known(0)
        );
        assert_eq!(
            resource.descriptor.ledger_binding,
            crate::models::runtime_receipts::RuntimeResourceLedgerBinding::NoBrokerLease
        );
        let wire = serde_json::to_value(resource).expect("semantic receipt serializes");
        assert_eq!(wire["descriptor"]["domain"], serde_json::Value::Null);
        assert_eq!(
            services
                .runtime_receipts()
                .reconcile_live_leases(services.memory_broker()),
            crate::models::runtime_receipts::LeaseReceiptShadow::Matched
        );
        assert!(live.events.iter().any(|event| matches!(
            event,
            crate::models::runtime_receipts::RuntimeReceiptEvent::OwnerCreated { .. }
        )));
        assert!(live.events.iter().any(|event| matches!(
            event,
            crate::models::runtime_receipts::RuntimeReceiptEvent::ResourceAcquired { .. }
        )));

        registry.shutdown();
        drop(engine);
        let released = services.runtime_receipts().snapshot();
        assert!(released.live_owners.is_empty());
        let resource_released = released
            .events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    crate::models::runtime_receipts::RuntimeReceiptEvent::ResourceReleased { .. }
                )
            })
            .expect("resource release event");
        let owner_released = released
            .events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    crate::models::runtime_receipts::RuntimeReceiptEvent::OwnerReleased { .. }
                )
            })
            .expect("owner release event");
        assert!(resource_released < owner_released);
        assert!(released.events.iter().any(|event| matches!(
            event,
            crate::models::runtime_receipts::RuntimeReceiptEvent::OwnerReleased { .. }
        )));
    }

    #[test]
    fn generic_pending_reuse_shutdown_marks_dropped_and_incomplete() {
        let services = crate::models::native_execution_services::test_native_execution_services();
        let _guard = crate::models::native_execution_services::install_native_execution_services(
            services.as_ref(),
        );
        let config = ServeBatchConfig {
            max_batch: 2,
            queue_capacity: 2,
            collect_window: Duration::ZERO,
            send_timeout: Duration::from_secs(1),
            reply_timeout: Duration::from_secs(1),
            trace_batches: false,
        };
        let engine = ServeBatchEngine::<FakeFamily>::spawn(777, config).expect("generic owner");
        while !engine.owner_ready.load(Ordering::Acquire) {
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
    fn same_key_concurrent_lookup_builds_one_owner_and_one_engine() {
        let services = crate::models::native_execution_services::test_native_execution_services();
        let registry = Arc::new(ServeBatchEngineRegistry::<FakeFamily>::default());
        let config = ServeBatchConfig {
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
        let first = std::thread::spawn(move || {
            let _guard =
                crate::models::native_execution_services::install_native_execution_services(
                    first_services.as_ref(),
                );
            first_barrier.wait();
            first_registry.engine_for_key(99, config)
        });
        let second_registry = Arc::clone(&registry);
        let second_services = Arc::clone(&services);
        let second_barrier = Arc::clone(&barrier);
        let second = std::thread::spawn(move || {
            let _guard =
                crate::models::native_execution_services::install_native_execution_services(
                    second_services.as_ref(),
                );
            second_barrier.wait();
            second_registry.engine_for_key(99, config)
        });
        barrier.wait();

        let first_engine = first.join().expect("first lookup").expect("first engine");
        let second_engine = second
            .join()
            .expect("second lookup")
            .expect("second engine");
        assert!(Arc::ptr_eq(&first_engine, &second_engine));
        let job = FakeJob {
            id: 1,
            max_tokens: 1,
            execution_context: Arc::new(RequestExecutionContext::new(
                Some("single-flight-job".to_string()),
                Arc::new(crate::TranscriptionControl::new()),
            )),
            self_cancel_after_step: None,
        };
        assert_eq!(first_engine.submit(job).expect("single-flight decode"), 1);
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
        drop(first_engine);
        drop(second_engine);
        registry.shutdown();
        let released = services.runtime_receipts().snapshot();
        let resource_released = released
            .events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    crate::models::runtime_receipts::RuntimeReceiptEvent::ResourceReleased { .. }
                )
            })
            .expect("owner resource teardown");
        let owner_released = released
            .events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    crate::models::runtime_receipts::RuntimeReceiptEvent::OwnerReleased { .. }
                )
            })
            .expect("owner teardown");
        assert!(resource_released < owner_released);

        let rollback_registry = Arc::new(ServeBatchEngineRegistry::<FakeFamily>::default());
        let rollback_barrier = Arc::new(std::sync::Barrier::new(2));
        let baseline_events = services.runtime_receipts().snapshot().events.len();
        let rollback_registry_first = Arc::clone(&rollback_registry);
        let rollback_services_first = Arc::clone(&services);
        let rollback_barrier_first = Arc::clone(&rollback_barrier);
        let rollback_first = std::thread::spawn(move || {
            let _guard =
                crate::models::native_execution_services::install_native_execution_services(
                    rollback_services_first.as_ref(),
                );
            let candidate = crate::device::execution_policy::ExecutionCandidate {
                device: crate::device::execution_policy::ExecutionDeviceSnapshot {
                    route: crate::device::execution_route::ResolvedExecutionRoute::cpu(),
                    ggml_kind: crate::ggml_runtime::GgmlBackendKind::Cpu,
                    memory: None,
                    buffer_alignment: None,
                },
                placement: crate::device::execution_policy::ExecutionPlacement::CpuOnly,
            };
            let outcome = crate::models::native_execution_services::run_execution_candidate_attempt(
                rollback_services_first.as_ref(),
                &candidate,
                || {
                    let engine = rollback_registry_first.engine_for_key(123, config)?;
                    rollback_barrier_first.wait();
                    crate::models::native_execution_services::record_current_execution_candidate_failure(
                        crate::device::execution_policy::ExecutionCandidateFailure::capacity(
                            "serve-batch-test",
                            "forced rollback",
                        ),
                    );
                    Ok::<_, FakeError>(engine)
                },
            );
            drop(outcome.result);
        });
        let rollback_registry_second = Arc::clone(&rollback_registry);
        let rollback_services_second = Arc::clone(&services);
        let rollback_barrier_second = Arc::clone(&rollback_barrier);
        let rollback_second = std::thread::spawn(move || {
            let _guard =
                crate::models::native_execution_services::install_native_execution_services(
                    rollback_services_second.as_ref(),
                );
            rollback_barrier_second.wait();
            let engine = rollback_registry_second
                .engine_for_key(123, config)
                .expect("retry after rollback");
            assert_eq!(
                engine
                    .submit(FakeJob {
                        id: 123,
                        max_tokens: 1,
                        execution_context: Arc::new(RequestExecutionContext::new(
                            Some("rollback-retry".to_string()),
                            Arc::new(crate::TranscriptionControl::new()),
                        )),
                        self_cancel_after_step: None,
                    })
                    .expect("rollback retry decode"),
                123
            );
        });
        rollback_first.join().expect("rollback builder");
        rollback_second.join().expect("rollback waiter");
        let rollback_events = services.runtime_receipts().snapshot().events;
        let rollback_events = &rollback_events[baseline_events..];
        let first_created = rollback_events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    crate::models::runtime_receipts::RuntimeReceiptEvent::OwnerCreated { .. }
                )
            })
            .expect("rollback owner created");
        let released = rollback_events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    crate::models::runtime_receipts::RuntimeReceiptEvent::OwnerReleased { .. }
                )
            })
            .expect("rollback owner released");
        let second_created = rollback_events
            .iter()
            .enumerate()
            .find_map(|(index, event)| {
                (index > released
                    && matches!(
                        event,
                        crate::models::runtime_receipts::RuntimeReceiptEvent::OwnerCreated { .. }
                    ))
                .then_some(index)
            })
            .unwrap_or_else(|| panic!("retry owner created; events={rollback_events:?}"));
        assert!(first_created < released && released < second_created);
        rollback_registry.shutdown();
    }

    #[test]
    fn active_owner_survives_full_borrowed_lru_and_reuses_same_key() {
        let services = crate::models::native_execution_services::test_native_execution_services();
        let _guard = crate::models::native_execution_services::install_native_execution_services(
            services.as_ref(),
        );
        let registry = ServeBatchEngineRegistry::<FakeFamily>::default();
        let config = ServeBatchConfig {
            max_batch: 2,
            queue_capacity: 2,
            collect_window: Duration::ZERO,
            send_timeout: Duration::from_secs(1),
            reply_timeout: Duration::from_secs(1),
            trace_batches: false,
        };
        fn job(id: usize) -> FakeJob {
            FakeJob {
                id: id as u32,
                max_tokens: 1,
                execution_context: Arc::new(RequestExecutionContext::new(
                    Some(format!("lru-job-{id}")),
                    Arc::new(crate::TranscriptionControl::new()),
                )),
                self_cancel_after_step: None,
            }
        }
        let mut borrowed = Vec::new();
        for key in 0..SERVE_BATCH_ENGINE_REGISTRY_MAX_ENTRIES {
            borrowed.push(
                registry
                    .engine_for_key(key as u32, config)
                    .expect("borrowed LRU owner"),
            );
        }
        let first = registry
            .engine_for_key(100, config)
            .expect("active overflow owner");
        assert_eq!(
            registry.engines.lock().unwrap().len(),
            SERVE_BATCH_ENGINE_REGISTRY_MAX_ENTRIES
        );
        let second = registry
            .engine_for_key(100, config)
            .expect("active overflow reuse");
        assert!(Arc::ptr_eq(&first, &second));
        for _ in 0..10_000 {
            first.record_receipt_reuse();
        }
        let notification_summary = services.runtime_receipts().summary();
        assert!(notification_summary.completeness.dropped_notifications > 0);
        assert!(!notification_summary.completeness.complete);
        for (id, engine) in borrowed.iter().enumerate() {
            assert_eq!(
                engine.submit(job(id)).expect("borrowed owner decode"),
                id as u32
            );
        }
        assert_eq!(first.submit(job(100)).expect("active owner decode"), 100);
        assert_eq!(services.runtime_receipts().summary().live_owner_count, 5);

        drop(second);
        drop(first);
        drop(borrowed);
        for key in 200..1200_u32 {
            let engine = registry
                .engine_for_key(key, config)
                .expect("bounded churn owner");
            drop(engine);
        }
        assert!(
            registry.active.lock().unwrap().len() <= SERVE_BATCH_ENGINE_REGISTRY_MAX_ENTRIES,
            "active weak registry exceeded its bounded LRU owner footprint"
        );
        registry.shutdown();
    }

    #[test]
    fn executor_owned_registry_is_shared_by_clones_but_isolated_between_roots() {
        let config = ServeBatchConfig {
            max_batch: 2,
            queue_capacity: 2,
            collect_window: Duration::ZERO,
            send_timeout: Duration::from_secs(1),
            reply_timeout: Duration::from_secs(1),
            trace_batches: false,
        };
        let first = ServeBatchEngineRegistry::<FakeFamily>::default();
        let first_clone = first.clone();
        let second = ServeBatchEngineRegistry::<FakeFamily>::default();

        let first_engine = first
            .engine_for_key(7, config)
            .expect("first registry engine");
        let cloned_lookup = first_clone
            .engine_for_key(7, config)
            .expect("clone registry engine");
        let second_engine = second
            .engine_for_key(7, config)
            .expect("independent registry engine");

        assert!(Arc::ptr_eq(&first_engine, &cloned_lookup));
        assert!(!Arc::ptr_eq(&first_engine, &second_engine));
        assert_eq!(first.engines.lock().unwrap().len(), 1);
        assert_eq!(second.engines.lock().unwrap().len(), 1);

        first_clone.shutdown();
        assert!(first.engines.lock().unwrap().is_empty());
        assert_eq!(second.engines.lock().unwrap().len(), 1);
        second.shutdown();
    }

    #[test]
    fn bounded_engine_cache_evicts_the_least_recent_idle_owner() {
        let mut cache = BoundedServeBatchEngineCache::<usize, usize>::default();
        for key in 0..SERVE_BATCH_ENGINE_REGISTRY_MAX_ENTRIES {
            assert!(cache.insert_if_idle_capacity(key, Arc::new(key)));
        }

        drop(cache.get(&0).expect("touch newest engine"));
        assert!(cache.insert_if_idle_capacity(99, Arc::new(99)));

        assert_eq!(cache.len(), SERVE_BATCH_ENGINE_REGISTRY_MAX_ENTRIES);
        assert!(!cache.entries.contains_key(&1));
        assert!(cache.entries.contains_key(&0));
        assert!(cache.entries.contains_key(&99));
    }

    #[test]
    fn bounded_engine_cache_does_not_retain_a_new_owner_when_all_slots_are_active() {
        let mut cache = BoundedServeBatchEngineCache::<usize, usize>::default();
        let mut active = Vec::new();
        for key in 0..SERVE_BATCH_ENGINE_REGISTRY_MAX_ENTRIES {
            let engine = Arc::new(key);
            active.push(Arc::clone(&engine));
            assert!(cache.insert_if_idle_capacity(key, engine));
        }

        assert!(!cache.insert_if_idle_capacity(99, Arc::new(99)));
        assert_eq!(cache.len(), SERVE_BATCH_ENGINE_REGISTRY_MAX_ENTRIES);
        assert!(!cache.entries.contains_key(&99));
        drop(active);
    }

    #[test]
    fn clearing_engine_cache_advances_publication_generation() {
        let mut cache = BoundedServeBatchEngineCache::<usize, usize>::default();
        assert!(cache.insert_if_idle_capacity(1, Arc::new(1)));
        let stale_generation = cache.generation();
        cache.clear();
        assert!(cache.is_empty());
        assert_ne!(cache.generation(), stale_generation);
    }

    #[test]
    fn rebucket_needs_no_dummy_row_when_one_token_budget_has_no_replay() {
        assert_eq!(rebucket_dummy_position(1, 1, 0), Ok(None));
        assert_eq!(rebucket_dummy_position(4, 4, 0), Ok(None));
    }

    #[test]
    fn rebucket_uses_the_last_physical_row_only_when_incremental_replay_exists() {
        // P=4, G=2 => exact physical K=P+G-1=5. The one replay step may use
        // row K-1=4 for an inactive lane without requiring a sixth row.
        assert_eq!(rebucket_dummy_position(5, 4, 1), Ok(Some(4)));
        assert_eq!(
            rebucket_dummy_position(4, 4, 1),
            Err("seq2seq serve batch rebucket has no dummy KV row")
        );
    }

    /// Structural proof that the generic `Envelope`'s `context` is required,
    /// not optional: this only compiles because the field's type is the
    /// concrete `Arc<RequestExecutionContext>`. Never called; exists purely
    /// so `cargo check`/`clippy` re-verify the contract on every build.
    #[allow(dead_code)]
    fn require_concrete_execution_context(_: Arc<RequestExecutionContext>) {}

    #[allow(dead_code)]
    fn assert_envelope_requires_execution_context(envelope: Envelope<FakeFamily>) {
        let Envelope { context, .. } = envelope;
        require_concrete_execution_context(context);
    }
}
