//! Generic seq2seq serve-batch owner.
//!
//! This module hosts the family-agnostic continuous-batching owner loop that
//! the three seq2seq serve-batch owners (whisper / cohere / moonshine) will
//! share. The control flow is ported VERBATIM from the cohere owner
//! (`models/cohere/batched_decode.rs`, the cleanest no-special-casing
//! baseline), with concrete types replaced by the `Seq2SeqServeBatchFamily`
//! associated types and concrete method calls replaced by trait hooks.
//!
//! All three families (cohere / moonshine / whisper) are now wired onto this
//! generic owner.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::serve_batch_env::{
    OwnerAliveGuard, serve_batch_bucket_width, serve_batch_collect_window_from_env,
    serve_batch_compact_active_slots, serve_batch_drain_compatible_batch, serve_batch_max_from_env,
    serve_batch_owner_alive, serve_batch_submit_with_timeout, serve_batch_trace_enabled,
    serve_batch_vram_capped_max_batch,
};
use crate::nn::decoder::reusable_decode_graph_supported;

/// Per-family decoder-runtime contract. Bound to `WhisperServeDecoderRuntime` /
/// `CohereDecoderGraphRuntime` / `MoonshineDecoderGraphRuntime`. The generic
/// owner only ever touches a runtime through these methods.
pub(crate) trait Seq2SeqServeRuntime: Sized {
    type Job;
    type Error;
    fn build_serial(job: &Self::Job) -> Result<Self, Self::Error>; // n_seq == 1
    fn build_batched(job: &Self::Job, n_seq: usize) -> Result<Self, Self::Error>;
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
    // All three family error enums derive thiserror `Error` (hence `Display`).
    type Error: std::fmt::Display;
    type EngineKey: Clone + Eq + std::hash::Hash;

    const THREAD_NAME_PREFIX: &'static str;
    /// Upper bound for the `OPENASR_SERVE_BATCH` env max-batch (all three
    /// families use 8). Consumed by the generic `ServeBatchConfig::from_env`.
    const MAX_BATCH_LIMIT: usize;
    fn engine_key(job: &Self::Job, max_batch: usize) -> Self::EngineKey;
    /// The backend recorded in an engine key, used only to reproduce the owner
    /// thread name `openasr-<prefix>-serve-batch-<Backend>-<max_batch>`.
    fn engine_key_backend(key: &Self::EngineKey) -> GgmlCpuGraphBackend;
    fn can_batch_with(a: &Self::Job, b: &Self::Job) -> bool;

    fn vram_slot_bytes(job: &Self::Job) -> usize;
    fn backend(job: &Self::Job) -> GgmlCpuGraphBackend;
    fn uses_scheduler(job: &Self::Job) -> bool;
    fn effective_max_batch_for_backend_name(configured: usize, _backend_name: &str) -> usize {
        configured
    } // whisper Vulkan->1
    /// Applied AFTER the VRAM cap in `validate_for_job`. The default is the
    /// identity (cohere / moonshine never resolve a backend name); whisper
    /// overrides this to resolve the backend name and apply the Vulkan->serial
    /// cap via `effective_max_batch_for_backend_name`. Keeping name resolution
    /// behind this hook preserves cohere/moonshine behavior exactly (they never
    /// initialized a backend guard inside validation).
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

    // Engine/registry/config error constructors (Wave B). Each family binds these
    // to its existing `*ServeBatchError` variant constructors; no new error enum.
    fn invalid_env(env: &'static str, raw: String, max: usize) -> Self::Error;
    fn invalid_enabled_batch(max_batch: usize) -> Self::Error;
    fn unsupported_backend(backend: GgmlCpuGraphBackend) -> Self::Error;
    fn registry_poisoned() -> Self::Error;
    fn thread_spawn_failed(reason: String) -> Self::Error;
    fn queue_full() -> Self::Error;
    fn owner_disconnected() -> Self::Error;
    fn reply_timed_out() -> Self::Error;
}

/// A queued serve-batch request: the family job plus the reply channel the
/// owner thread sends the decode result back through.
pub(crate) struct Envelope<F: Seq2SeqServeBatchFamily> {
    pub job: F::Job,
    pub reply: mpsc::Sender<Result<F::Output, F::Error>>,
}

/// A slot currently occupying a batch lane, pairing the family slot state with
/// the reply channel that owns its result.
struct ActiveBatchSlot<F: Seq2SeqServeBatchFamily> {
    slot: F::Slot,
    reply: mpsc::Sender<Result<F::Output, F::Error>>,
}

/// Transient state for a slot that has been seeded for refill but not yet
/// committed back into the active slot vector.
struct PendingRefillSlot<F: Seq2SeqServeBatchFamily> {
    slot_index: usize,
    slot: F::Slot,
    reply: mpsc::Sender<Result<F::Output, F::Error>>,
}

/// The owner-thread decode state: a lazily-built serial runtime and a cache of
/// per-width batched runtimes keyed by `n_seq`.
pub(crate) struct OwnerThreadState<F: Seq2SeqServeBatchFamily> {
    serial_runtime: Option<F::Runtime>,
    pub(crate) batched_runtimes: HashMap<usize, F::Runtime>,
}

impl<F: Seq2SeqServeBatchFamily> OwnerThreadState<F> {
    pub(crate) fn new() -> Self {
        Self {
            serial_runtime: None,
            batched_runtimes: HashMap::new(),
        }
    }

    pub(crate) fn run_batch(
        &mut self,
        batch: Vec<Envelope<F>>,
        receiver: &Receiver<Envelope<F>>,
        max_batch: usize,
        trace_batches: bool,
    ) -> VecDeque<Envelope<F>> {
        if batch.len() <= 1 {
            for envelope in batch {
                let Envelope { job, reply } = envelope;
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

        let mut replies = Vec::with_capacity(batch.len());
        let mut slots = Vec::with_capacity(batch.len());
        for envelope in batch {
            replies.push(envelope.reply);
            match F::slot_new(envelope.job) {
                Ok(slot) => slots.push(slot),
                Err(error) => {
                    let _ = replies
                        .pop()
                        .expect("reply pushed before slot build")
                        .send(Err(error));
                }
            }
        }
        let mut slots = slots
            .into_iter()
            .zip(replies)
            .map(|(slot, reply)| Some(ActiveBatchSlot::<F> { slot, reply }))
            .collect::<Vec<_>>();
        if slots.is_empty() {
            return deferred;
        }
        let active_count = slots.iter().filter(|slot| slot.is_some()).count();
        if active_count <= 1 {
            for active in slots.into_iter().flatten() {
                let ActiveBatchSlot { slot, reply } = active;
                let result = self.decode_serial_job(F::slot_job(&slot).clone());
                let _ = reply.send(result);
            }
            return deferred;
        }
        let bucket_width = serve_batch_bucket_width(active_count, max_batch);
        if bucket_width > slots.len() {
            slots.resize_with(bucket_width, || None);
        }

        let Some(first_job) = Self::first_active_job(&slots) else {
            return deferred;
        };
        let prompt_len = F::initial_prompt_tokens(first_job).len();
        if prompt_len == 0 {
            Self::fail_all_active_slots(
                &mut slots,
                F::decode_failed("seq2seq serve batch prompt is empty".to_string()),
            );
            return deferred;
        }
        let prompt_tokens = F::initial_prompt_tokens(first_job).to_vec();
        {
            let runtime = match self.batched_runtime_for(first_job, slots.len()) {
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
                if let Err(error) = runtime
                    .populate_cross_attention_cache_slot(slot_index, F::slot_job(&active.slot))
                {
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

            match Self::seed_initial_batch_prompt(&mut slots, runtime, &prompt_tokens) {
                Ok(()) => Self::finish_done_active_slots(&mut slots),
                Err(error) => {
                    Self::fail_all_active_slots(&mut slots, error);
                    return deferred;
                }
            }
        }

        loop {
            Self::finish_maxed_active_slots(&mut slots);
            if let Some(first_job) = Self::first_active_job(&slots) {
                let runtime = match self.batched_runtime_for(first_job, slots.len()) {
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
            let Some(first_job) = Self::first_active_job(&slots) else {
                break;
            };
            let runtime = match self.batched_runtime_for(first_job, slots.len()) {
                Ok(runtime) => runtime,
                Err(error) => {
                    Self::fail_all_active_slots(&mut slots, error);
                    break;
                }
            };
            let logits =
                match runtime.compute_reused_batched_step_logits(&token_ids, &positions, &totals) {
                    Ok(logits) => logits,
                    Err(error) => {
                        Self::fail_all_active_slots(
                            &mut slots,
                            F::decode_failed(format_error::<F>(error)),
                        );
                        break;
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
            let Envelope { job, reply } = envelope;
            match F::slot_new(job) {
                Ok(slot) => pending.push((slot, reply)),
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
            for (slot, reply) in pending.into_iter().rev() {
                deferred.push_front(Envelope {
                    job: F::slot_job(&slot).clone(),
                    reply,
                });
            }
            return Ok(());
        }

        let previous_width = slots.len();
        for (slot, reply) in pending {
            slots.push(Some(ActiveBatchSlot::<F> { slot, reply }));
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
        let first_job = Self::first_active_job(slots).ok_or_else(|| {
            F::owner_failed("seq2seq serve batch rebucket has no active slots".to_string())
        })?;
        let prompt_tokens = F::initial_prompt_tokens(first_job).to_vec();
        if prompt_tokens.len() != prompt_len {
            return Err(F::decode_failed(
                "seq2seq serve batch rebucket prompt length changed".to_string(),
            ));
        }
        let dummy_position = F::decoder_max_context(first_job)
            .checked_sub(1)
            .ok_or_else(|| {
                F::decode_failed(
                    "seq2seq serve batch rebucket requires non-empty context".to_string(),
                )
            })?;
        if dummy_position < prompt_len {
            return Err(F::decode_failed(
                "seq2seq serve batch rebucket has no dummy KV row".to_string(),
            ));
        }

        let runtime = self.batched_runtime_for(first_job, slots.len())?;
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
                let Envelope { job, reply } = envelope;
                let slot = match F::slot_new(job) {
                    Ok(slot) => slot,
                    Err(error) => {
                        let _ = reply.send(Err(error));
                        continue;
                    }
                };
                if let Err(error) =
                    runtime.populate_cross_attention_cache_slot(slot_index, F::slot_job(&slot))
                {
                    let _ = reply.send(Err(F::decode_failed(format_error::<F>(error))));
                    continue;
                }
                pending_refills.push(PendingRefillSlot::<F> {
                    slot_index,
                    slot,
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
                reply,
            } = pending;
            if F::slot_done(&slot) {
                let _ = reply.send(F::slot_finish(slot));
                continue;
            }
            slots[slot_index] = Some(ActiveBatchSlot::<F> { slot, reply });
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
        let ActiveBatchSlot { slot, reply } = active;
        let _ = reply.send(F::slot_finish(slot));
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

    fn decode_serial_job(&mut self, job: F::Job) -> Result<F::Output, F::Error> {
        F::decode_serial(&mut self.serial_runtime, job)
    }

    pub(crate) fn batched_runtime_for(
        &mut self,
        job: &F::Job,
        n_seq: usize,
    ) -> Result<&mut F::Runtime, F::Error> {
        if let std::collections::hash_map::Entry::Vacant(e) = self.batched_runtimes.entry(n_seq) {
            let runtime = F::Runtime::build_batched(job, n_seq)?;
            e.insert(runtime);
        }
        self.batched_runtimes.get_mut(&n_seq).ok_or_else(|| {
            F::owner_failed("seq2seq serve batch runtime cache is unexpectedly empty".to_string())
        })
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
// The three families' `*ServeBatchConfig` structs were field-identical, and
// their engine/spawn/submit/owner-loop/registry-lookup bodies were near-clones
// differing only in error-variant constructors and the per-family thread-name
// prefix. They are collapsed here into a generic `ServeBatchConfig` +
// `ServeBatchEngine<F>` driven by the `Seq2SeqServeBatchFamily` hooks. Each
// family keeps only its `static *_SERVE_BATCH_ENGINES` registry (Rust has no
// generic-over-`F` static) plus a thin `submit_*_serve_batch_job`.
// ===========================================================================

const SERVE_BATCH_QUEUE_CAPACITY: usize = 4;
const SERVE_BATCH_COLLECT_WINDOW: Duration = Duration::from_millis(2);
const SERVE_BATCH_SEND_TIMEOUT: Duration = Duration::from_secs(1);
const SERVE_BATCH_REPLY_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// The serve-batch owner-thread tuning shared by all three families. Field
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
    /// Reads `OPENASR_SERVE_BATCH` (+ collect-window / trace env). Returns
    /// `Ok(None)` when serve-batch is disabled (unset / `<= 1`), mirroring the
    /// previous per-family `from_env`. The env max-batch limit comes from
    /// `F::MAX_BATCH_LIMIT`; env-parse failures map through `F::invalid_env`.
    /// Deliberately NOT named `from_env`: each family exposes a thin
    /// `*ServeBatchConfig::from_env()` (a per-module extension trait) that calls
    /// this, and an inherent `from_env` here would shadow that trait method and
    /// break type inference at the `ggml_executor` call sites.
    pub(crate) fn read_env<F: Seq2SeqServeBatchFamily>() -> Result<Option<Self>, F::Error> {
        let Some(max_batch) =
            serve_batch_max_from_env(F::MAX_BATCH_LIMIT).map_err(map_env_error::<F>)?
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            max_batch,
            queue_capacity: SERVE_BATCH_QUEUE_CAPACITY,
            collect_window: serve_batch_collect_window_from_env(SERVE_BATCH_COLLECT_WINDOW)
                .map_err(map_env_error::<F>)?,
            send_timeout: SERVE_BATCH_SEND_TIMEOUT,
            reply_timeout: SERVE_BATCH_REPLY_TIMEOUT,
            trace_batches: serve_batch_trace_enabled(),
        }))
    }

    /// Validates the config against a concrete job and resolves the effective
    /// max-batch: `max_batch >= 2` (else `F::invalid_enabled_batch`); gpu-class
    /// backend && !scheduler (else `F::unsupported_backend`); the VRAM cap
    /// (`F::vram_slot_bytes`); THEN `F::effective_max_batch_after_vram_cap`
    /// (whisper Vulkan->serial). The VRAM-cap-then-backend-name-cap ORDER is
    /// load-bearing -- it affects the engine key -- and is preserved exactly.
    pub(crate) fn validate_for_job<F: Seq2SeqServeBatchFamily>(
        self,
        job: &F::Job,
    ) -> Result<Self, F::Error> {
        if self.max_batch < 2 {
            return Err(F::invalid_enabled_batch(self.max_batch));
        }
        let backend = F::backend(job);
        if !reusable_decode_graph_supported(backend, F::uses_scheduler(job)) {
            return Err(F::unsupported_backend(backend));
        }
        let max_batch =
            serve_batch_vram_capped_max_batch(self.max_batch, backend, F::vram_slot_bytes(job))
                .map_err(map_env_error::<F>)?;
        let max_batch = F::effective_max_batch_after_vram_cap(max_batch, job)?;
        Ok(Self { max_batch, ..self })
    }
}

/// A serve-batch engine: the owner-thread send channel, the resolved config,
/// and the owner liveness flag (used to respawn after a dead/panicked owner).
pub(crate) struct ServeBatchEngine<F: Seq2SeqServeBatchFamily> {
    sender: SyncSender<Envelope<F>>,
    config: ServeBatchConfig,
    is_alive: Arc<AtomicBool>,
}

impl<F: Seq2SeqServeBatchFamily> ServeBatchEngine<F> {
    fn spawn(key: F::EngineKey, config: ServeBatchConfig) -> Result<Self, F::Error>
    where
        // The `Envelope<F>` (job + reply sender for `Result<Output, Error>`) is
        // moved into the spawned owner thread, so each crossing type must be
        // `Send`. All three concrete families satisfy this.
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
        thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                let _alive_guard = alive_guard;
                owner_thread_loop::<F>(receiver, config)
            })
            .map_err(|error| F::thread_spawn_failed(error.to_string()))?;
        Ok(Self {
            sender,
            config,
            is_alive,
        })
    }

    pub(crate) fn submit(&self, job: F::Job) -> Result<F::Output, F::Error> {
        let (reply, reply_rx) = mpsc::channel();
        serve_batch_submit_with_timeout(
            &self.sender,
            Envelope { job, reply },
            reply_rx,
            self.config.send_timeout,
            self.config.reply_timeout,
            F::queue_full,
            F::owner_disconnected,
            F::reply_timed_out,
        )
    }
}

fn owner_thread_loop<F: Seq2SeqServeBatchFamily>(
    receiver: Receiver<Envelope<F>>,
    config: ServeBatchConfig,
) {
    let mut state = OwnerThreadState::<F>::new();
    let mut deferred = VecDeque::new();
    loop {
        let Some(batch) = serve_batch_drain_compatible_batch(
            &mut deferred,
            &receiver,
            config.max_batch,
            config.collect_window,
            |first, next| F::can_batch_with(&first.job, &next.job),
        ) else {
            break;
        };
        if config.trace_batches {
            eprintln!(
                "openasr {} serve batch: drained {} request(s)",
                F::THREAD_NAME_PREFIX,
                batch.len()
            );
        }
        deferred.extend(state.run_batch(batch, &receiver, config.max_batch, config.trace_batches));
    }
}

/// Registry lookup with dead-owner respawn: if a cached engine's owner thread
/// has exited (normal or panic), drop the stale engine and spawn a fresh one
/// with clean ggml state. The `registry` is passed in because Rust has no
/// generic-over-`F` static; each family owns its own `static`.
pub(crate) fn serve_batch_engine_for_key<F: Seq2SeqServeBatchFamily>(
    registry: &OnceLock<Mutex<HashMap<F::EngineKey, Arc<ServeBatchEngine<F>>>>>,
    key: F::EngineKey,
    config: ServeBatchConfig,
) -> Result<Arc<ServeBatchEngine<F>>, F::Error>
where
    F::Job: Send,
    F::Output: Send,
    F::Error: Send,
{
    let registry = registry.get_or_init(|| Mutex::new(HashMap::new()));
    let mut engines = registry.lock().map_err(|_| F::registry_poisoned())?;
    if let Some(engine) = engines.get(&key) {
        if serve_batch_owner_alive(&engine.is_alive) {
            return Ok(Arc::clone(engine));
        }
        // The cached owner thread exited (normal or panic); drop the stale
        // engine and respawn a fresh one with clean ggml state.
        engines.remove(&key);
    }
    let engine = Arc::new(ServeBatchEngine::<F>::spawn(key.clone(), config)?);
    engines.insert(key, Arc::clone(&engine));
    Ok(engine)
}

fn map_env_error<F: Seq2SeqServeBatchFamily>(
    error: crate::models::serve_batch_env::ServeBatchEnvError,
) -> F::Error {
    F::invalid_env(error.env, error.raw, error.max)
}
