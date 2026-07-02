use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::thread;
use std::time::{Duration, Instant};

use crate::ggml_runtime::{
    GgmlBackendKind, GgmlCpuGraphBackend, GgmlDeviceMemory, ggml_available_devices,
};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeConfig, Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeStepLogitsOutput,
    select_seq2seq_greedy_step_token,
};

pub(crate) const OPENASR_SERVE_BATCH_ENV: &str = "OPENASR_SERVE_BATCH";
pub(crate) const OPENASR_SERVE_BATCH_TRACE_ENV: &str = "OPENASR_SERVE_BATCH_TRACE";
pub(crate) const OPENASR_SERVE_BATCH_COLLECT_MS_ENV: &str = "OPENASR_SERVE_BATCH_COLLECT_MS";
pub(crate) const OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_ENV: &str =
    "OPENASR_SERVE_BATCH_VRAM_RESERVE_MB";

const OPENASR_SERVE_BATCH_COLLECT_MS_LIMIT: usize = 100;
const OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_DEFAULT: usize = 1024;
const OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_LIMIT: usize = 1024 * 1024;
const MIB_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ServeBatchEnvError {
    pub env: &'static str,
    pub raw: String,
    pub max: usize,
}

/// Liveness flag for a serve-batch owner thread. Each engine keeps a clone of
/// the returned `Arc<AtomicBool>` and the owner thread holds the paired
/// `OwnerAliveGuard`; the guard flips the flag to `false` on ANY owner-thread
/// exit -- a normal return OR a panic unwind -- so a cached engine whose owner
/// has died can be detected at the next registry lookup and respawned with
/// clean state.
///
/// We deliberately do NOT `catch_unwind` the decode loop: the owner state holds
/// `!UnwindSafe` ggml pointers (decoders/runtimes backed by C memory), so
/// resuming a panicked owner could propagate a poisoned mutex or a
/// half-written arena. Respawn-on-dead recovers safely without ever reusing
/// corrupted state.
pub(crate) struct OwnerAliveGuard {
    alive: Arc<AtomicBool>,
}

impl OwnerAliveGuard {
    /// Returns `(flag, guard)`: store `flag` in the engine, move `guard` into
    /// the owner thread closure so its `Drop` marks the owner dead on exit.
    pub(crate) fn new() -> (Arc<AtomicBool>, Self) {
        let alive = Arc::new(AtomicBool::new(true));
        (Arc::clone(&alive), Self { alive })
    }
}

impl Drop for OwnerAliveGuard {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Release);
    }
}

/// True while the owner thread paired with this flag is still running.
pub(crate) fn serve_batch_owner_alive(alive: &Arc<AtomicBool>) -> bool {
    alive.load(Ordering::Acquire)
}

/// Outcome of one greedy serve-batch decode step.
pub(crate) enum ServeBatchStepOutcome {
    /// The sequence emitted its end-of-text token; the slot is done.
    ReachedEot,
    /// A new token to append to the slot's generated history, with the
    /// softmax probability of that token over the step's logit row.
    Token { token_id: u32, probability: f32 },
}

/// Shared greedy step-token selection for serve-batch slots. Runs
/// `select_seq2seq_greedy_step_token` over the slot's decode config, generated
/// history and stop tokens, and reports whether EOT was reached or which token
/// to append. Every family's `select_next_token_from_logits` was a byte-for-byte
/// copy of this body differing only in the error type, so it lives here once;
/// callers map the `Seq2SeqGreedyDecodeError` to their own error.
pub(crate) fn serve_batch_select_greedy_step(
    decode_config: &Seq2SeqGreedyDecodeConfig,
    generated_tokens: &[u32],
    stop_token_ids: &[u32],
    logits: Vec<f32>,
) -> Result<ServeBatchStepOutcome, Seq2SeqGreedyDecodeError> {
    let step_index = generated_tokens.len();
    let mut no_topk_trace = |_: usize, _: &[f32]| {};
    let selection = select_seq2seq_greedy_step_token(
        decode_config,
        generated_tokens,
        step_index,
        Seq2SeqGreedyDecodeStepLogitsOutput {
            logits,
            greedy_token_hint: None,
        },
        stop_token_ids,
        &mut no_topk_trace,
    )?;
    Ok(if selection.reached_eot {
        ServeBatchStepOutcome::ReachedEot
    } else {
        ServeBatchStepOutcome::Token {
            token_id: selection.token_id,
            probability: selection.probability,
        }
    })
}

pub(crate) fn serve_batch_max_from_env(
    max_limit: usize,
) -> Result<Option<usize>, ServeBatchEnvError> {
    let Some(raw) = std::env::var_os(OPENASR_SERVE_BATCH_ENV) else {
        return Ok(None);
    };
    let raw = raw.to_string_lossy().trim().to_string();
    if raw.is_empty() {
        return Ok(None);
    }
    let max_batch = raw.parse::<usize>().map_err(|_| ServeBatchEnvError {
        env: OPENASR_SERVE_BATCH_ENV,
        raw: raw.clone(),
        max: max_limit,
    })?;
    if max_batch <= 1 {
        return Ok(None);
    }
    if max_batch > max_limit {
        return Err(ServeBatchEnvError {
            env: OPENASR_SERVE_BATCH_ENV,
            raw,
            max: max_limit,
        });
    }
    Ok(Some(max_batch))
}

pub(crate) fn serve_batch_collect_window_from_env(
    default: Duration,
) -> Result<Duration, ServeBatchEnvError> {
    let Some(raw) = std::env::var_os(OPENASR_SERVE_BATCH_COLLECT_MS_ENV) else {
        return Ok(default);
    };
    let raw = raw.to_string_lossy().trim().to_string();
    if raw.is_empty() {
        return Ok(default);
    }
    let collect_ms = raw.parse::<usize>().map_err(|_| ServeBatchEnvError {
        env: OPENASR_SERVE_BATCH_COLLECT_MS_ENV,
        raw: raw.clone(),
        max: OPENASR_SERVE_BATCH_COLLECT_MS_LIMIT,
    })?;
    if collect_ms > OPENASR_SERVE_BATCH_COLLECT_MS_LIMIT {
        return Err(ServeBatchEnvError {
            env: OPENASR_SERVE_BATCH_COLLECT_MS_ENV,
            raw,
            max: OPENASR_SERVE_BATCH_COLLECT_MS_LIMIT,
        });
    }
    Ok(Duration::from_millis(collect_ms as u64))
}

pub(crate) fn serve_batch_trace_enabled() -> bool {
    std::env::var_os(OPENASR_SERVE_BATCH_TRACE_ENV)
        .map(|value| {
            let value = value.to_string_lossy();
            !(value.is_empty() || value == "0" || value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(false)
}

pub(crate) fn serve_batch_vram_capped_max_batch(
    requested_max_batch: usize,
    backend: GgmlCpuGraphBackend,
    estimated_slot_bytes: usize,
) -> Result<usize, ServeBatchEnvError> {
    if !backend.is_gpu_class() || requested_max_batch <= 2 || estimated_slot_bytes == 0 {
        return Ok(requested_max_batch);
    }
    let reserve_mb = serve_batch_vram_reserve_mb_from_env()?;
    let Some(sample) = selected_gpu_memory_sample() else {
        trace_serve_batch_vram_cap_unavailable(backend, requested_max_batch, estimated_slot_bytes);
        return Ok(requested_max_batch);
    };
    let decision = serve_batch_vram_cap_decision_for_memory(
        requested_max_batch,
        estimated_slot_bytes,
        sample.memory.free_bytes,
        reserve_mb.saturating_mul(MIB_BYTES),
    );
    trace_serve_batch_vram_cap_decision(backend, &sample, &decision);
    Ok(decision.capped_max_batch)
}

pub(crate) fn serve_batch_bucket_width(active_count: usize, max_batch: usize) -> usize {
    if active_count <= 1 || max_batch <= active_count {
        return active_count;
    }
    active_count
        .checked_next_power_of_two()
        .unwrap_or(max_batch)
        .min(max_batch)
        .max(active_count)
}

pub(crate) fn serve_batch_compact_active_slots<T>(slots: &mut Vec<Option<T>>, target_width: usize) {
    let mut compacted = Vec::with_capacity(target_width.max(slots.len()));
    for active in slots.drain(..).flatten() {
        compacted.push(Some(active));
    }
    if target_width > compacted.len() {
        compacted.resize_with(target_width, || None);
    }
    *slots = compacted;
}

pub(crate) fn serve_batch_drain_compatible_batch<Envelope>(
    deferred: &mut VecDeque<Envelope>,
    receiver: &Receiver<Envelope>,
    max_batch: usize,
    collect_window: Duration,
    mut can_batch_with_first: impl FnMut(&Envelope, &Envelope) -> bool,
) -> Option<Vec<Envelope>> {
    let first = match deferred.pop_front() {
        Some(envelope) => envelope,
        None => receiver.recv().ok()?,
    };
    let mut batch = Vec::with_capacity(max_batch.max(1));
    batch.push(first);

    let deferred_len = deferred.len();
    for _ in 0..deferred_len {
        if batch.len() >= max_batch {
            break;
        }
        let Some(envelope) = deferred.pop_front() else {
            break;
        };
        if can_batch_with_first(&batch[0], &envelope) {
            batch.push(envelope);
        } else {
            deferred.push_back(envelope);
        }
    }

    let deadline = Instant::now() + collect_window;
    while batch.len() < max_batch {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        match receiver.recv_timeout(deadline - now) {
            Ok(envelope) => {
                if can_batch_with_first(&batch[0], &envelope) {
                    batch.push(envelope);
                } else {
                    deferred.push_back(envelope);
                }
            }
            Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
        }
    }

    Some(batch)
}

pub(crate) fn serve_batch_submit_with_timeout<Envelope, Reply, Error>(
    sender: &SyncSender<Envelope>,
    mut envelope: Envelope,
    reply_rx: Receiver<Result<Reply, Error>>,
    send_timeout: Duration,
    reply_timeout: Duration,
    queue_full: impl Fn() -> Error,
    owner_disconnected: impl Fn() -> Error,
    reply_timed_out: impl Fn() -> Error,
) -> Result<Reply, Error> {
    let deadline = Instant::now() + send_timeout;
    loop {
        match sender.try_send(envelope) {
            Ok(()) => break,
            Err(TrySendError::Full(returned)) => {
                if Instant::now() >= deadline {
                    return Err(queue_full());
                }
                envelope = returned;
                thread::sleep(Duration::from_millis(1));
            }
            Err(TrySendError::Disconnected(_)) => return Err(owner_disconnected()),
        }
    }
    reply_rx
        .recv_timeout(reply_timeout)
        .map_err(|error| match error {
            RecvTimeoutError::Timeout => reply_timed_out(),
            RecvTimeoutError::Disconnected => owner_disconnected(),
        })?
}

pub(crate) fn serve_batch_estimate_llm_kv_slot_bytes(
    layers: usize,
    max_positions: usize,
    kv_heads: usize,
    head_dim: usize,
    element_bytes: usize,
) -> usize {
    saturating_product(&[layers, 2, max_positions, kv_heads, head_dim, element_bytes])
}

pub(crate) fn serve_batch_estimate_seq2seq_slot_bytes(
    decoder_layers: usize,
    max_positions: usize,
    decoder_hidden_size: usize,
    encoder_frames: usize,
    encoder_hidden_size: usize,
    self_kv_element_bytes: usize,
    cross_kv_element_bytes: usize,
) -> usize {
    let self_kv = saturating_product(&[
        decoder_layers,
        2,
        max_positions,
        decoder_hidden_size,
        self_kv_element_bytes,
    ]);
    let cross_kv = saturating_product(&[
        decoder_layers,
        2,
        encoder_frames,
        encoder_hidden_size,
        cross_kv_element_bytes,
    ]);
    self_kv.saturating_add(cross_kv)
}

fn serve_batch_vram_reserve_mb_from_env() -> Result<usize, ServeBatchEnvError> {
    let Some(raw) = std::env::var_os(OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_ENV) else {
        return Ok(OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_DEFAULT);
    };
    let raw = raw.to_string_lossy().trim().to_string();
    if raw.is_empty() {
        return Ok(OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_DEFAULT);
    }
    let reserve_mb = raw.parse::<usize>().map_err(|_| ServeBatchEnvError {
        env: OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_ENV,
        raw: raw.clone(),
        max: OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_LIMIT,
    })?;
    if reserve_mb > OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_LIMIT {
        return Err(ServeBatchEnvError {
            env: OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_ENV,
            raw,
            max: OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_LIMIT,
        });
    }
    Ok(reserve_mb)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ServeBatchGpuMemorySample {
    device_name: String,
    device_kind: GgmlBackendKind,
    memory: GgmlDeviceMemory,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ServeBatchVramCapDecision {
    requested_max_batch: usize,
    capped_max_batch: usize,
    estimated_slot_bytes: usize,
    free_bytes: usize,
    reserve_bytes: usize,
    usable_bytes: usize,
}

fn selected_gpu_memory_sample() -> Option<ServeBatchGpuMemorySample> {
    let devices = ggml_available_devices()
        .into_iter()
        .map(|device| (device.name, device.kind, device.memory))
        .collect::<Vec<_>>();
    selected_gpu_memory_sample_from_device_infos(&devices)
}

fn selected_gpu_memory_sample_from_device_infos(
    devices: &[(String, GgmlBackendKind, Option<GgmlDeviceMemory>)],
) -> Option<ServeBatchGpuMemorySample> {
    let (name, kind, memory) = devices.iter().find(|(_, kind, _)| kind.is_gpu())?;
    memory.map(|memory| ServeBatchGpuMemorySample {
        device_name: name.clone(),
        device_kind: *kind,
        memory,
    })
}

fn serve_batch_vram_cap_decision_for_memory(
    requested_max_batch: usize,
    estimated_slot_bytes: usize,
    free_bytes: usize,
    reserve_bytes: usize,
) -> ServeBatchVramCapDecision {
    let usable_bytes = free_bytes.saturating_sub(reserve_bytes);
    let capped_max_batch = if requested_max_batch <= 2 || estimated_slot_bytes == 0 {
        requested_max_batch
    } else {
        let slots = usable_bytes / estimated_slot_bytes;
        requested_max_batch.min(slots.max(2))
    };
    ServeBatchVramCapDecision {
        requested_max_batch,
        capped_max_batch,
        estimated_slot_bytes,
        free_bytes,
        reserve_bytes,
        usable_bytes,
    }
}

#[cfg(test)]
fn serve_batch_vram_capped_max_batch_for_memory(
    requested_max_batch: usize,
    estimated_slot_bytes: usize,
    free_bytes: usize,
    reserve_bytes: usize,
) -> usize {
    serve_batch_vram_cap_decision_for_memory(
        requested_max_batch,
        estimated_slot_bytes,
        free_bytes,
        reserve_bytes,
    )
    .capped_max_batch
}

fn trace_serve_batch_vram_cap_decision(
    backend: GgmlCpuGraphBackend,
    sample: &ServeBatchGpuMemorySample,
    decision: &ServeBatchVramCapDecision,
) {
    if !serve_batch_trace_enabled() {
        return;
    }
    let status = if decision.capped_max_batch < decision.requested_max_batch {
        "capped"
    } else {
        "kept"
    };
    eprintln!(
        "openasr serve batch: vram cap {status} backend={backend:?} device={} kind={:?} requested={} capped={} slot_mib={} free_mib={} total_mib={} reserve_mib={} usable_mib={}",
        sample.device_name,
        sample.device_kind,
        decision.requested_max_batch,
        decision.capped_max_batch,
        bytes_to_mib(decision.estimated_slot_bytes),
        bytes_to_mib(decision.free_bytes),
        bytes_to_mib(sample.memory.total_bytes),
        bytes_to_mib(decision.reserve_bytes),
        bytes_to_mib(decision.usable_bytes),
    );
}

fn trace_serve_batch_vram_cap_unavailable(
    backend: GgmlCpuGraphBackend,
    requested_max_batch: usize,
    estimated_slot_bytes: usize,
) {
    if serve_batch_trace_enabled() {
        eprintln!(
            "openasr serve batch: vram cap skipped backend={backend:?} requested={requested_max_batch} slot_mib={} reason=no-gpu-memory-sample",
            bytes_to_mib(estimated_slot_bytes),
        );
    }
}

fn saturating_product(values: &[usize]) -> usize {
    values
        .iter()
        .copied()
        .fold(1usize, |acc, value| acc.saturating_mul(value))
}

fn bytes_to_mib(bytes: usize) -> usize {
    bytes / MIB_BYTES
}

#[cfg(test)]
pub(crate) fn with_serve_batch_env_lock<T>(run: impl FnOnce() -> T) -> T {
    use std::sync::{Mutex, OnceLock};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let lock = ENV_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().expect("serve batch env lock");
    run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn owner_alive_guard_marks_dead_on_panic() {
        let (alive, guard) = OwnerAliveGuard::new();
        assert!(serve_batch_owner_alive(&alive));
        let handle = std::thread::spawn(move || {
            let _guard = guard;
            panic!("simulated owner-thread panic");
        });
        assert!(
            handle.join().is_err(),
            "the owner thread should have panicked"
        );
        assert!(
            !serve_batch_owner_alive(&alive),
            "guard must mark the owner dead after a panic so the next lookup respawns"
        );
    }

    #[test]
    fn owner_alive_guard_marks_dead_on_normal_exit() {
        let (alive, guard) = OwnerAliveGuard::new();
        std::thread::spawn(move || {
            let _guard = guard;
        })
        .join()
        .expect("owner thread joins cleanly");
        assert!(!serve_batch_owner_alive(&alive));
    }

    fn with_env<T>(
        batch: Option<&str>,
        collect_ms: Option<&str>,
        trace: Option<&str>,
        vram_reserve_mb: Option<&str>,
        run: impl FnOnce() -> T,
    ) -> T {
        with_serve_batch_env_lock(|| {
            let previous_batch = std::env::var_os(OPENASR_SERVE_BATCH_ENV);
            let previous_collect_ms = std::env::var_os(OPENASR_SERVE_BATCH_COLLECT_MS_ENV);
            let previous_trace = std::env::var_os(OPENASR_SERVE_BATCH_TRACE_ENV);
            let previous_vram_reserve = std::env::var_os(OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_ENV);
            set_env(OPENASR_SERVE_BATCH_ENV, batch.map(OsString::from));
            set_env(
                OPENASR_SERVE_BATCH_COLLECT_MS_ENV,
                collect_ms.map(OsString::from),
            );
            set_env(OPENASR_SERVE_BATCH_TRACE_ENV, trace.map(OsString::from));
            set_env(
                OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_ENV,
                vram_reserve_mb.map(OsString::from),
            );
            let result = run();
            set_env(OPENASR_SERVE_BATCH_ENV, previous_batch);
            set_env(OPENASR_SERVE_BATCH_COLLECT_MS_ENV, previous_collect_ms);
            set_env(OPENASR_SERVE_BATCH_TRACE_ENV, previous_trace);
            set_env(
                OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_ENV,
                previous_vram_reserve,
            );
            result
        })
    }

    fn set_env(env: &'static str, value: Option<OsString>) {
        match value {
            Some(value) => {
                #[expect(unsafe_code, reason = "test-only process env override")]
                unsafe {
                    std::env::set_var(env, value);
                }
            }
            None => {
                #[expect(unsafe_code, reason = "test-only process env override")]
                unsafe {
                    std::env::remove_var(env);
                }
            }
        }
    }

    #[test]
    fn serve_batch_max_defaults_off() {
        with_env(None, None, None, None, || {
            assert_eq!(serve_batch_max_from_env(8).unwrap(), None);
        });
    }

    #[test]
    fn serve_batch_max_one_keeps_default_path() {
        with_env(Some("1"), None, None, None, || {
            assert_eq!(serve_batch_max_from_env(8).unwrap(), None);
        });
    }

    #[test]
    fn serve_batch_max_accepts_within_limit() {
        with_env(Some("4"), None, None, None, || {
            assert_eq!(serve_batch_max_from_env(8).unwrap(), Some(4));
        });
    }

    #[test]
    fn serve_batch_max_rejects_out_of_range() {
        with_env(Some("9"), None, None, None, || {
            let error = serve_batch_max_from_env(8).unwrap_err();
            assert_eq!(error.env, OPENASR_SERVE_BATCH_ENV);
            assert_eq!(error.max, 8);
        });
    }

    #[test]
    fn serve_batch_collect_window_defaults_when_unset_or_empty() {
        with_env(Some("2"), None, None, None, || {
            assert_eq!(
                serve_batch_collect_window_from_env(Duration::from_millis(2)).unwrap(),
                Duration::from_millis(2)
            );
        });
        with_env(Some("2"), Some(""), None, None, || {
            assert_eq!(
                serve_batch_collect_window_from_env(Duration::from_millis(2)).unwrap(),
                Duration::from_millis(2)
            );
        });
    }

    #[test]
    fn serve_batch_collect_window_accepts_zero_to_limit() {
        with_env(Some("2"), Some("0"), None, None, || {
            assert_eq!(
                serve_batch_collect_window_from_env(Duration::from_millis(2)).unwrap(),
                Duration::ZERO
            );
        });
        with_env(Some("2"), Some("100"), None, None, || {
            assert_eq!(
                serve_batch_collect_window_from_env(Duration::from_millis(2)).unwrap(),
                Duration::from_millis(100)
            );
        });
    }

    #[test]
    fn serve_batch_collect_window_rejects_out_of_range() {
        with_env(Some("2"), Some("101"), None, None, || {
            let error = serve_batch_collect_window_from_env(Duration::from_millis(2)).unwrap_err();
            assert_eq!(error.env, OPENASR_SERVE_BATCH_COLLECT_MS_ENV);
            assert_eq!(error.max, OPENASR_SERVE_BATCH_COLLECT_MS_LIMIT);
        });
    }

    #[test]
    fn serve_batch_trace_is_falsey_only_for_empty_zero_or_false() {
        with_env(None, None, None, None, || {
            assert!(!serve_batch_trace_enabled())
        });
        with_env(None, None, Some("0"), None, || {
            assert!(!serve_batch_trace_enabled())
        });
        with_env(None, None, Some("false"), None, || {
            assert!(!serve_batch_trace_enabled())
        });
        with_env(None, None, Some("1"), None, || {
            assert!(serve_batch_trace_enabled())
        });
    }

    #[test]
    fn serve_batch_vram_reserve_defaults_and_rejects_out_of_range() {
        with_env(None, None, None, None, || {
            assert_eq!(
                serve_batch_vram_reserve_mb_from_env().unwrap(),
                OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_DEFAULT
            );
        });
        with_env(None, None, None, Some(""), || {
            assert_eq!(
                serve_batch_vram_reserve_mb_from_env().unwrap(),
                OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_DEFAULT
            );
        });
        with_env(None, None, None, Some("2048"), || {
            assert_eq!(serve_batch_vram_reserve_mb_from_env().unwrap(), 2048);
        });
        with_env(None, None, None, Some("1048577"), || {
            let error = serve_batch_vram_reserve_mb_from_env().unwrap_err();
            assert_eq!(error.env, OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_ENV);
            assert_eq!(error.max, OPENASR_SERVE_BATCH_VRAM_RESERVE_MB_LIMIT);
        });
    }

    #[test]
    fn serve_batch_vram_cap_preserves_minimum_enabled_bucket() {
        assert_eq!(
            serve_batch_vram_capped_max_batch_for_memory(8, 512, 4096, 1024),
            6
        );
        assert_eq!(
            serve_batch_vram_capped_max_batch_for_memory(8, 512, 1500, 1024),
            2
        );
        assert_eq!(
            serve_batch_vram_capped_max_batch_for_memory(2, 512, 1500, 1024),
            2
        );
    }

    #[test]
    fn serve_batch_vram_cap_decision_records_memory_inputs() {
        let decision = serve_batch_vram_cap_decision_for_memory(
            8,
            512 * MIB_BYTES,
            3 * 1024 * MIB_BYTES,
            1024 * MIB_BYTES,
        );

        assert_eq!(decision.requested_max_batch, 8);
        assert_eq!(decision.capped_max_batch, 4);
        assert_eq!(decision.estimated_slot_bytes, 512 * MIB_BYTES);
        assert_eq!(decision.free_bytes, 3 * 1024 * MIB_BYTES);
        assert_eq!(decision.reserve_bytes, 1024 * MIB_BYTES);
        assert_eq!(decision.usable_bytes, 2 * 1024 * MIB_BYTES);
    }

    #[test]
    fn serve_batch_vram_sample_uses_first_gpu_device_not_largest_gpu() {
        let devices = vec![
            (
                "cpu".to_string(),
                GgmlBackendKind::Cpu,
                Some(GgmlDeviceMemory {
                    free_bytes: 32 * MIB_BYTES,
                    total_bytes: 64 * MIB_BYTES,
                }),
            ),
            (
                "first-gpu".to_string(),
                GgmlBackendKind::Gpu,
                Some(GgmlDeviceMemory {
                    free_bytes: 4 * MIB_BYTES,
                    total_bytes: 8 * MIB_BYTES,
                }),
            ),
            (
                "larger-second-gpu".to_string(),
                GgmlBackendKind::Gpu,
                Some(GgmlDeviceMemory {
                    free_bytes: 16 * MIB_BYTES,
                    total_bytes: 24 * MIB_BYTES,
                }),
            ),
        ];

        let sample =
            selected_gpu_memory_sample_from_device_infos(&devices).expect("gpu memory sample");

        assert_eq!(sample.device_name, "first-gpu");
        assert_eq!(sample.memory.free_bytes, 4 * MIB_BYTES);
    }

    #[test]
    fn serve_batch_vram_sample_requires_selected_gpu_memory() {
        let devices = vec![
            ("first-gpu".to_string(), GgmlBackendKind::Gpu, None),
            (
                "larger-second-gpu".to_string(),
                GgmlBackendKind::Gpu,
                Some(GgmlDeviceMemory {
                    free_bytes: 16 * MIB_BYTES,
                    total_bytes: 24 * MIB_BYTES,
                }),
            ),
        ];

        assert!(selected_gpu_memory_sample_from_device_infos(&devices).is_none());
    }

    #[test]
    fn serve_batch_bucket_width_rounds_active_batches_without_touching_singletons() {
        assert_eq!(serve_batch_bucket_width(0, 8), 0);
        assert_eq!(serve_batch_bucket_width(1, 8), 1);
        assert_eq!(serve_batch_bucket_width(2, 8), 2);
        assert_eq!(serve_batch_bucket_width(3, 8), 4);
        assert_eq!(serve_batch_bucket_width(5, 8), 8);
        assert_eq!(serve_batch_bucket_width(3, 3), 3);
        assert_eq!(serve_batch_bucket_width(4, 3), 4);
    }

    #[test]
    fn serve_batch_compact_active_slots_preserves_order_and_pads_target_width() {
        let mut slots = vec![None, Some("a"), None, Some("b"), Some("c"), None];

        serve_batch_compact_active_slots(&mut slots, 5);

        assert_eq!(slots, vec![Some("a"), Some("b"), Some("c"), None, None]);
    }

    #[test]
    fn serve_batch_compact_active_slots_never_drops_active_slots() {
        let mut slots = vec![Some(1), None, Some(2), Some(3)];

        serve_batch_compact_active_slots(&mut slots, 2);

        assert_eq!(slots, vec![Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn serve_batch_drain_compatible_batch_scans_deferred_once() {
        let (_sender, receiver) = std::sync::mpsc::channel::<i32>();
        let mut deferred = VecDeque::from([1, 2, 3, 4]);

        let batch = serve_batch_drain_compatible_batch(
            &mut deferred,
            &receiver,
            4,
            Duration::ZERO,
            |first, next| first % 2 == next % 2,
        )
        .expect("drained batch");

        assert_eq!(batch, vec![1, 3]);
        assert_eq!(deferred.into_iter().collect::<Vec<_>>(), vec![2, 4]);
    }

    #[test]
    fn serve_batch_drain_compatible_batch_collects_receiver_until_cap() {
        let (sender, receiver) = std::sync::mpsc::channel();
        sender.send(1).expect("first");
        sender.send(2).expect("second");
        sender.send(3).expect("third");
        let mut deferred = VecDeque::new();

        let batch = serve_batch_drain_compatible_batch(
            &mut deferred,
            &receiver,
            2,
            Duration::from_millis(1),
            |_, _| true,
        )
        .expect("drained batch");

        assert_eq!(batch, vec![1, 2]);
        assert!(deferred.is_empty());
        assert_eq!(receiver.try_recv().expect("leftover receiver item"), 3);
    }

    #[test]
    fn serve_batch_drain_compatible_batch_defers_receiver_mismatch() {
        let (sender, receiver) = std::sync::mpsc::channel();
        sender.send(1).expect("first");
        sender.send(2).expect("second");
        sender.send(3).expect("third");
        drop(sender);
        let mut deferred = VecDeque::new();

        let batch = serve_batch_drain_compatible_batch(
            &mut deferred,
            &receiver,
            3,
            Duration::from_millis(1),
            |first, next| first % 2 == next % 2,
        )
        .expect("drained batch");

        assert_eq!(batch, vec![1, 3]);
        assert_eq!(deferred.into_iter().collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn serve_batch_submit_with_timeout_returns_reply() {
        let (sender, _receiver) = std::sync::mpsc::sync_channel(1);
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        reply_tx.send(Ok::<_, &'static str>("ok")).expect("reply");

        let reply = serve_batch_submit_with_timeout(
            &sender,
            7,
            reply_rx,
            Duration::ZERO,
            Duration::from_millis(1),
            || "full",
            || "disconnected",
            || "timeout",
        )
        .expect("submit reply");

        assert_eq!(reply, "ok");
    }

    #[test]
    fn serve_batch_submit_with_timeout_reports_queue_full() {
        let (sender, _receiver) = std::sync::mpsc::sync_channel(0);
        let (_reply_tx, reply_rx) = std::sync::mpsc::channel::<Result<(), &'static str>>();

        let error = serve_batch_submit_with_timeout(
            &sender,
            7,
            reply_rx,
            Duration::ZERO,
            Duration::from_millis(1),
            || "full",
            || "disconnected",
            || "timeout",
        )
        .expect_err("zero-capacity queue should be full");

        assert_eq!(error, "full");
    }

    #[test]
    fn serve_batch_submit_with_timeout_reports_owner_disconnected_on_send() {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        drop(receiver);
        let (_reply_tx, reply_rx) = std::sync::mpsc::channel::<Result<(), &'static str>>();

        let error = serve_batch_submit_with_timeout(
            &sender,
            7,
            reply_rx,
            Duration::ZERO,
            Duration::from_millis(1),
            || "full",
            || "disconnected",
            || "timeout",
        )
        .expect_err("disconnected owner should fail");

        assert_eq!(error, "disconnected");
    }

    #[test]
    fn serve_batch_submit_with_timeout_reports_reply_timeout() {
        let (sender, _receiver) = std::sync::mpsc::sync_channel(1);
        let (_reply_tx, reply_rx) = std::sync::mpsc::channel::<Result<(), &'static str>>();

        let error = serve_batch_submit_with_timeout(
            &sender,
            7,
            reply_rx,
            Duration::ZERO,
            Duration::ZERO,
            || "full",
            || "disconnected",
            || "timeout",
        )
        .expect_err("missing reply should time out");

        assert_eq!(error, "timeout");
    }

    #[test]
    fn serve_batch_slot_byte_estimators_are_saturating() {
        assert_eq!(
            serve_batch_estimate_llm_kv_slot_bytes(2, 3, 4, 5, 6),
            2 * 2 * 3 * 4 * 5 * 6
        );
        assert_eq!(
            serve_batch_estimate_seq2seq_slot_bytes(2, 3, 4, 5, 6, 2, 4),
            (2 * 2 * 3 * 4 * 2) + (2 * 2 * 5 * 6 * 4)
        );
        assert_eq!(
            serve_batch_estimate_llm_kv_slot_bytes(usize::MAX, 2, 2, 2, 2),
            usize::MAX
        );
    }
}
