//! Typed, bounded request-local facts for native execution receipts.
//!
//! This collector is deliberately an in-memory authority. JSON projection is
//! owned by `short_audio_receipt`; no caller may reconstruct facts from a
//! backend label, environment variable, or CLI policy option after execution.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};
use thiserror::Error;

use crate::{
    RequestAttemptId,
    device::{execution_policy::ExecutionPlacement, execution_route::ExecutionProvider},
    ggml_runtime::{
        GgmlActualDeviceFacts, GgmlCpuGraphBackend, GgmlExecutionPlacementSummary,
        GgmlGraphLifecycleCollector, GgmlGraphLifecycleSnapshot, GgmlSelectionEvidenceRef,
        ResolvedFamilyRuntimeInput, diagnostic_logits_sha256,
    },
};

use super::native_execution_services::ExecutionLaneKey;

const MAX_TRACE_EVENTS: usize = 4_096;
const MAX_TRACE_TOP_K: usize = 8;
const MAX_FULL_LOGITS_ELEMENTS: usize = 32 * 1024 * 1024;
pub const GPU_CORRECTNESS_TRACE_MAX_STEPS: usize = 4_096;
pub const GPU_FULL_LOGITS_MAX_VOCAB: usize = 1_000_000;
pub const GPU_FULL_LOGITS_TRACE_SCHEMA: &str = "openasr.gpu-full-logits-trace.v1";

static PROCESS_TRACE_NONCE: OnceLock<Option<String>> = OnceLock::new();

fn process_trace_nonce() -> Option<&'static str> {
    PROCESS_TRACE_NONCE
        .get_or_init(|| {
            let mut nonce = [0_u8; 16];
            getrandom::fill(&mut nonce).ok()?;
            Some(
                nonce
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>(),
            )
        })
        .as_deref()
}

/// Complete selected-family topology captured from the live adapter/inventory
/// selection, rather than reconstructed by a receipt consumer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeExecutionTopologyFacts {
    pub family: String,
    pub model_architecture: String,
    pub adapter_id: String,
    pub decode_policy_id: String,
    pub decode_driver: String,
    pub decoder_state: String,
    pub block_stack: String,
}

/// Facts captured inside the successful native candidate boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeExecutionRequestFacts {
    pub resolved_runtime: ResolvedFamilyRuntimeInput,
    pub(crate) execution_lane: ExecutionLaneKey,
    pub selected_provider: ExecutionProvider,
    pub stable_device_id: String,
    pub backend_id: Option<String>,
    pub device_target: Option<String>,
    pub backend_driver_version: Option<String>,
    pub backend_artifact_fingerprint: Option<String>,
    pub placement: ExecutionPlacement,
    pub backend: GgmlCpuGraphBackend,
    pub topology: NativeExecutionTopologyFacts,
    pub pack_content_id: String,
    pub pack_size_bytes: u64,
    pub actual_provider: Option<ExecutionProvider>,
    pub actual_stable_device_id: Option<String>,
    pub actual_device: Option<GgmlActualDeviceFacts>,
    pub scheduler_enabled: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeExecutionTraceSnapshot {
    pub jsonl: String,
    pub full_logits_jsonl: Option<String>,
    pub overflowed: bool,
    pub invalid_binding: bool,
    pub event_count: usize,
    pub full_logits_step_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeExecutionTraceMode {
    Cold,
    Reuse,
}

impl NativeExecutionTraceMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Reuse => "reuse",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeExecutionTokenStep {
    pub step_index: usize,
    pub token_id: u32,
    pub is_eot: bool,
    pub top2_margin: Option<f32>,
    pub logits_sha256: Option<String>,
    pub(crate) compute: Option<GgmlSelectionEvidenceRef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeExecutionReceiptSnapshot {
    pub request_attempt_id: Option<RequestAttemptId>,
    pub request_attempt_conflicted: bool,
    pub phase_duration_micros: BTreeMap<RequestExecutionPhase, u64>,
    pub timing_complete: bool,
    pub terminal: Option<RequestExecutionTerminal>,
    pub timeline_conflicted: bool,
    pub facts: Option<NativeExecutionRequestFacts>,
    pub placement: GgmlExecutionPlacementSummary,
    pub trace: NativeExecutionTraceSnapshot,
    pub graph_lifecycle: GgmlGraphLifecycleSnapshot,
    pub token_steps: Vec<NativeExecutionTokenStep>,
    pub completed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RequestExecutionPhase {
    UploadIngest,
    DecodeNormalize,
    AdmissionWait,
    Compute,
    /// Internal-only attach of already-prepared samples. This is intentionally
    /// not reported as audio decode/preparation.
    PreparedSampleAttach,
}

impl RequestExecutionPhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UploadIngest => "upload-ingest",
            Self::DecodeNormalize => "decode-normalize",
            Self::AdmissionWait => "admission-wait",
            Self::Compute => "compute",
            Self::PreparedSampleAttach => "prepared-sample-attach",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestExecutionTerminal {
    Succeeded,
    Canceled,
    Failed,
}

impl RequestExecutionTerminal {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Canceled => "canceled",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NativeExecutionAttestationError {
    #[error("candidate attempt did not complete")]
    Incomplete,
    #[error("candidate attempt produced no immutable request facts")]
    MissingFacts,
    #[error("candidate attempt used a different pack content identity")]
    PackContentMismatch,
    #[error("candidate attempt used a different output or reuse plan")]
    RuntimePlanMismatch,
    #[error("candidate attempt selected a different execution lane")]
    LaneMismatch,
    #[error(
        "candidate attempt lacks matching live backend attestation: expected provider={expected_provider} stable_device_id={expected_stable_device_id}, actual provider={actual_provider:?} stable_device_id={actual_stable_device_id:?} scheduler_enabled={scheduler_enabled:?}"
    )]
    LiveBackendMismatch {
        expected_provider: ExecutionProvider,
        expected_stable_device_id: String,
        actual_provider: Option<ExecutionProvider>,
        actual_stable_device_id: Option<String>,
        scheduler_enabled: Option<bool>,
    },
}

impl NativeExecutionReceiptSnapshot {
    /// Attest that the completed request used the exact immutable activation
    /// plan and physical lane selected before owner acquisition.
    pub fn attest_activation(
        &self,
        expected_pack_content_id: &str,
        expected_runtime: ResolvedFamilyRuntimeInput,
        expected_provider: ExecutionProvider,
        expected_stable_device_id: &str,
        expected_placement: ExecutionPlacement,
    ) -> Result<(), NativeExecutionAttestationError> {
        if !self.completed {
            return Err(NativeExecutionAttestationError::Incomplete);
        }
        let facts = self
            .facts
            .as_ref()
            .ok_or(NativeExecutionAttestationError::MissingFacts)?;
        if facts.pack_content_id != expected_pack_content_id {
            return Err(NativeExecutionAttestationError::PackContentMismatch);
        }
        if facts.resolved_runtime != expected_runtime {
            return Err(NativeExecutionAttestationError::RuntimePlanMismatch);
        }
        if facts.selected_provider != expected_provider
            || facts.stable_device_id != expected_stable_device_id
            || facts.placement != expected_placement
            || facts.backend != expected_runtime.backend()
        {
            return Err(NativeExecutionAttestationError::LaneMismatch);
        }
        if facts.actual_provider != Some(expected_provider)
            || facts.actual_stable_device_id.as_deref() != Some(expected_stable_device_id)
            || facts.actual_device.is_none()
            || facts.scheduler_enabled.is_none()
        {
            return Err(NativeExecutionAttestationError::LiveBackendMismatch {
                expected_provider,
                expected_stable_device_id: expected_stable_device_id.to_string(),
                actual_provider: facts.actual_provider,
                actual_stable_device_id: facts.actual_stable_device_id.clone(),
                scheduler_enabled: facts.scheduler_enabled,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct ReceiptState {
    request_attempt_id: Option<RequestAttemptId>,
    request_attempt_conflicted: bool,
    phase_duration_micros: BTreeMap<RequestExecutionPhase, u64>,
    terminal: Option<RequestExecutionTerminal>,
    timeline_conflicted: bool,
    facts: Option<NativeExecutionRequestFacts>,
    /// Live backend handles whose runner-cached device facts were already
    /// checked during this candidate attempt.
    observed_backend_identities: BTreeSet<usize>,
    placement: GgmlExecutionPlacementSummary,
    trace_events: Vec<String>,
    token_steps: Vec<NativeExecutionTokenStep>,
    top_k_margins: BTreeMap<usize, f32>,
    logits_hashes: BTreeMap<usize, String>,
    active_decode_step: Option<ActiveDecodeStep>,
    trace_binding_invalid: bool,
    trace_mode: Option<NativeExecutionTraceMode>,
    trace_run_id: Option<String>,
    trace_process_nonce: Option<String>,
    trace_process_id: Option<u32>,
    capture_full_logits: bool,
    full_logits_steps: Vec<FullLogitsStep>,
    full_logits_elements: usize,
    next_decode_step_index: usize,
    trace_overflowed: bool,
    graph_lifecycle_checkpoint: (usize, bool),
    completed: bool,
    /// Nested `begin_candidate_attempt` must not erase the parent candidate's
    /// shipped facts. Auxiliary stages (punctuation, VAD, aligner) open an
    /// inner attempt on the same collector; restoring this stack keeps the
    /// ASR row intact after the inner attempt finishes.
    attempt_stack: Vec<ReceiptAttemptCheckpoint>,
}

#[derive(Debug, Clone)]
struct ReceiptAttemptCheckpoint {
    facts: Option<NativeExecutionRequestFacts>,
    observed_backend_identities: BTreeSet<usize>,
    placement: GgmlExecutionPlacementSummary,
    trace_events: Vec<String>,
    token_steps: Vec<NativeExecutionTokenStep>,
    top_k_margins: BTreeMap<usize, f32>,
    logits_hashes: BTreeMap<usize, String>,
    full_logits_steps: Vec<FullLogitsStep>,
    full_logits_elements: usize,
    next_decode_step_index: usize,
    active_decode_step: Option<ActiveDecodeStep>,
    trace_binding_invalid: bool,
    trace_overflowed: bool,
    graph_lifecycle_checkpoint: (usize, bool),
    completed: bool,
}

impl ReceiptState {
    fn take_attempt_checkpoint(&mut self) -> ReceiptAttemptCheckpoint {
        ReceiptAttemptCheckpoint {
            facts: self.facts.take(),
            observed_backend_identities: std::mem::take(&mut self.observed_backend_identities),
            placement: std::mem::take(&mut self.placement),
            trace_events: std::mem::take(&mut self.trace_events),
            token_steps: std::mem::take(&mut self.token_steps),
            top_k_margins: std::mem::take(&mut self.top_k_margins),
            logits_hashes: std::mem::take(&mut self.logits_hashes),
            full_logits_steps: std::mem::take(&mut self.full_logits_steps),
            full_logits_elements: std::mem::take(&mut self.full_logits_elements),
            next_decode_step_index: std::mem::take(&mut self.next_decode_step_index),
            active_decode_step: self.active_decode_step.take(),
            trace_binding_invalid: std::mem::take(&mut self.trace_binding_invalid),
            trace_overflowed: std::mem::take(&mut self.trace_overflowed),
            graph_lifecycle_checkpoint: self.graph_lifecycle_checkpoint,
            completed: std::mem::take(&mut self.completed),
        }
    }

    fn restore_attempt_checkpoint(&mut self, checkpoint: ReceiptAttemptCheckpoint) {
        self.facts = checkpoint.facts;
        self.observed_backend_identities = checkpoint.observed_backend_identities;
        self.placement = checkpoint.placement;
        self.trace_events = checkpoint.trace_events;
        self.token_steps = checkpoint.token_steps;
        self.top_k_margins = checkpoint.top_k_margins;
        self.logits_hashes = checkpoint.logits_hashes;
        self.full_logits_steps = checkpoint.full_logits_steps;
        self.full_logits_elements = checkpoint.full_logits_elements;
        self.next_decode_step_index = checkpoint.next_decode_step_index;
        self.active_decode_step = checkpoint.active_decode_step;
        self.trace_binding_invalid = checkpoint.trace_binding_invalid;
        self.trace_overflowed = checkpoint.trace_overflowed;
        self.graph_lifecycle_checkpoint = checkpoint.graph_lifecycle_checkpoint;
        self.completed = checkpoint.completed;
    }
}

#[derive(Debug, Clone, Copy)]
struct ActiveDecodeStep {
    step_index: usize,
    compute: Option<GgmlSelectionEvidenceRef>,
    top_k_recorded: bool,
    token_recorded: bool,
}

#[derive(Debug, Clone)]
struct FullLogitsStep {
    step_index: usize,
    compute: GgmlSelectionEvidenceRef,
    values: Vec<f32>,
}

#[derive(serde::Serialize)]
struct FullLogitsHeader<'a> {
    schema: &'static str,
    event: &'static str,
    run_id: &'a str,
    process_nonce: &'a str,
    process_id: u32,
    mode: &'a str,
    graph_mode: &'static str,
    provider: &'a str,
    device_target: &'a str,
    backend_id: &'a str,
    driver_version: &'a str,
    artifact_fingerprint: &'a str,
    device: &'a str,
    actual_provider: &'a str,
    actual_stable_device_id: &'a str,
    actual_device: &'a GgmlActualDeviceFacts,
    dtype: &'static str,
    encoding: &'static str,
    step_count: usize,
}

#[derive(serde::Serialize)]
struct FullLogitsArtifactStep<'a> {
    schema: &'static str,
    event: &'static str,
    step_index: usize,
    compute: GgmlSelectionEvidenceRef,
    vocab_size: usize,
    values: &'a [f32],
}

/// Cloneable request-scoped receipt collector. It is installed only by an
/// explicit caller such as the strict short-audio row producer.
#[derive(Debug, Clone, Default)]
pub struct NativeExecutionReceiptCollector {
    state: Arc<Mutex<ReceiptState>>,
    graph_lifecycle: GgmlGraphLifecycleCollector,
}

impl NativeExecutionReceiptCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Binds the request-level correlation identity once. Candidate retries
    /// share it; conflicting rebinding invalidates completion rather than
    /// choosing either value.
    pub fn bind_request_attempt(&self, attempt_id: RequestAttemptId) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match state.request_attempt_id {
            None => state.request_attempt_id = Some(attempt_id),
            Some(existing) if existing == attempt_id => {}
            Some(_) => {
                state.request_attempt_conflicted = true;
                state.completed = false;
            }
        }
    }

    pub(crate) fn request_attempt_id(&self) -> Option<RequestAttemptId> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (!state.request_attempt_conflicted)
            .then_some(state.request_attempt_id)
            .flatten()
    }

    pub fn record_phase_duration(&self, phase: RequestExecutionPhase, duration: Duration) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Ok(micros) = u64::try_from(duration.as_micros()) else {
            state.timeline_conflicted = true;
            return;
        };
        let current = state
            .phase_duration_micros
            .get(&phase)
            .copied()
            .unwrap_or(0);
        let Some(total) = current.checked_add(micros) else {
            state.timeline_conflicted = true;
            return;
        };
        state.phase_duration_micros.insert(phase, total);
    }

    pub fn record_terminal(&self, terminal: RequestExecutionTerminal) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match state.terminal {
            None => state.terminal = Some(terminal),
            Some(existing) if existing == terminal => {}
            Some(_) => state.timeline_conflicted = true,
        }
    }

    /// Candidate attempts are transactional: a failed candidate cannot leave
    /// facts or trace events that a later fallback might publish. Nested
    /// attempts push the parent row aside and restore it on finish so an
    /// auxiliary stage cannot erase the ASR candidate's shipped facts.
    pub(crate) fn begin_candidate_attempt(&self) {
        self.graph_lifecycle.begin_observation_scope();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let parent = state.take_attempt_checkpoint();
        // A nested begin happens while the outer attempt is still open.
        // Sequential warmup/measured begins happen after the previous attempt
        // finished (`completed == true`) and must start a fresh row.
        let nested = !parent.completed
            && (parent.facts.is_some()
                || !parent.token_steps.is_empty()
                || !parent.trace_events.is_empty());
        if nested {
            state.attempt_stack.push(parent);
        }
        state.trace_binding_invalid = state.trace_mode.is_some() && state.trace_run_id.is_none();
        state.graph_lifecycle_checkpoint = self.graph_lifecycle.checkpoint();
        state.completed = false;
    }

    pub(crate) fn finish_candidate_attempt(&self, committed: bool) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let committed = committed && !state.request_attempt_conflicted;
        let parent = state.attempt_stack.pop();
        let restore_parent = parent.as_ref().is_some_and(|parent| {
            !parent.completed
                && (parent.facts.is_some()
                    || !parent.token_steps.is_empty()
                    || !parent.trace_events.is_empty())
        });
        let rollback_checkpoint =
            (!committed || restore_parent).then_some(state.graph_lifecycle_checkpoint);
        if restore_parent {
            state.restore_attempt_checkpoint(parent.expect("parent presence checked"));
        } else if committed {
            state.completed = true;
        } else {
            let _ = state.take_attempt_checkpoint();
            state.completed = false;
        }
        // Lifecycle producers never run while the receipt mutex is held.
        // Preserve that lock order during rollback and scope closure too.
        drop(state);
        if let Some(checkpoint) = rollback_checkpoint {
            self.graph_lifecycle.truncate(checkpoint);
        }
        self.graph_lifecycle.end_observation_scope();
    }

    pub(crate) fn graph_lifecycle_collector(&self) -> GgmlGraphLifecycleCollector {
        self.graph_lifecycle.clone()
    }

    pub fn set_trace_mode(&self, mode: NativeExecutionTraceMode) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.active_decode_step.is_some() {
            state.trace_binding_invalid = true;
            return;
        }
        state.trace_mode = Some(mode);
        let mut nonce = [0_u8; 16];
        if getrandom::fill(&mut nonce).is_err() {
            state.trace_run_id = None;
            state.trace_binding_invalid = true;
            return;
        }
        state.trace_run_id = Some(
            nonce
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
        );
        let Some(process_nonce) = process_trace_nonce() else {
            state.trace_process_nonce = None;
            state.trace_process_id = None;
            state.trace_binding_invalid = true;
            return;
        };
        state.trace_process_nonce = Some(process_nonce.to_string());
        state.trace_process_id = Some(std::process::id());
    }

    pub fn enable_full_logits_trace(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.active_decode_step.is_some() {
            state.trace_binding_invalid = true;
            return;
        }
        state.capture_full_logits = true;
    }

    pub(crate) fn record_facts(&self, facts: NativeExecutionRequestFacts) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match &state.facts {
            Some(existing) if existing != &facts => {
                // First writer wins. A later graph in the same candidate must
                // not erase shipped output_plan/reuse_mode by disagreeing.
            }
            Some(_) => {}
            None => state.facts = Some(facts),
        }
    }

    pub(crate) fn record_backend_observation(
        &self,
        backend_identity: usize,
        provider: ExecutionProvider,
        stable_device_id: &str,
        actual_device: &GgmlActualDeviceFacts,
        scheduler_enabled: bool,
    ) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let backend_was_observed = state
            .observed_backend_identities
            .contains(&backend_identity);
        let Some(facts) = state.facts.as_mut() else {
            return;
        };
        let route_drifted = facts
            .actual_provider
            .is_some_and(|actual| actual != provider)
            || facts
                .actual_stable_device_id
                .as_deref()
                .is_some_and(|actual| actual != stable_device_id)
            || facts
                .scheduler_enabled
                .is_some_and(|actual| actual != scheduler_enabled);
        // Full device facts are immutable for one verified live backend
        // handle. Recheck a newly observed handle once, while the per-compute
        // hot path continues to attest provider, stable id, and scheduler.
        let new_backend_device_drifted = !backend_was_observed
            && facts
                .actual_device
                .as_ref()
                .is_some_and(|actual| actual != actual_device);
        if route_drifted || new_backend_device_drifted {
            // Live backend attestation is fail-closed at the candidate
            // boundary. Erasing request facts here would let transcription
            // succeed while the shipped output_plan/reuse_mode disappeared
            // from the receipt. Keep the first observation; a later handle
            // still cannot overwrite it.
            return;
        }
        if facts.actual_provider.is_none() {
            facts.actual_provider = Some(provider);
        }
        if facts.actual_stable_device_id.is_none() {
            facts.actual_stable_device_id = Some(stable_device_id.to_string());
        }
        if facts.actual_device.is_none() {
            facts.actual_device = Some(actual_device.clone());
        }
        if facts.scheduler_enabled.is_none() {
            facts.scheduler_enabled = Some(scheduler_enabled);
        }
        if !backend_was_observed {
            state.observed_backend_identities.insert(backend_identity);
        }
    }

    pub(crate) fn record_placement(&self, placement: GgmlExecutionPlacementSummary) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.placement = placement;
    }

    pub(crate) fn begin_decode_step(
        &self,
        step_index: usize,
        compute: Option<GgmlSelectionEvidenceRef>,
    ) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Self::begin_decode_step_locked(&mut state, step_index, compute);
    }

    /// Start the next request-global selection step. Dedicated decoders such
    /// as streaming RNN-T may enter the shared loop once per encoder chunk, so
    /// a caller-local `0..` counter is not a stable request identity.
    pub(crate) fn begin_next_decode_step(
        &self,
        compute: Option<GgmlSelectionEvidenceRef>,
    ) -> usize {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let step_index = state.next_decode_step_index;
        Self::begin_decode_step_locked(&mut state, step_index, compute);
        step_index
    }

    /// Bind one token under a single lock so concurrent longform slices cannot
    /// overwrite each other's `active_decode_step`. Call after the shipped
    /// decode already produced logits and the selected token.
    pub(crate) fn commit_decode_step(
        &self,
        compute: Option<GgmlSelectionEvidenceRef>,
        token_id: u32,
        is_eot: bool,
        logits: &[f32],
    ) {
        let logits_sha256 = (!logits.is_empty()).then(|| diagnostic_logits_sha256(logits));
        let margin = if logits.len() < 2 {
            None
        } else {
            let mut best = None;
            let mut second = None;
            for logit in logits.iter().copied().filter(|value| value.is_finite()) {
                match best {
                    None => best = Some(logit),
                    Some(first) if logit > first => {
                        second = best;
                        best = Some(logit);
                    }
                    Some(first) if second.is_none_or(|value| logit > value) && logit != first => {
                        second = Some(logit);
                    }
                    _ => {}
                }
            }
            best.zip(second).map(|(first, second)| first - second)
        };
        let step_index = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let step_index = state.next_decode_step_index;
            Self::begin_decode_step_locked(&mut state, step_index, compute);
            if let Some(active) = state.active_decode_step.as_mut() {
                if !logits.is_empty() {
                    active.top_k_recorded = true;
                }
                active.token_recorded = true;
            }
            if let Some(margin) = margin {
                state.top_k_margins.insert(step_index, margin);
            }
            if let Some(hash) = logits_sha256.clone() {
                state.logits_hashes.insert(step_index, hash);
            }
            state.token_steps.push(NativeExecutionTokenStep {
                step_index,
                token_id,
                is_eot,
                top2_margin: margin,
                logits_sha256,
                compute,
            });
            let requires_complete_output = state.facts.as_ref().is_some_and(|facts| {
                matches!(
                    facts.resolved_runtime.output_plan(),
                    crate::ggml_runtime::GgmlDecodeOutputPlan::FullLogits
                )
            });
            let Some(active) = state.active_decode_step.take() else {
                state.trace_binding_invalid = true;
                return;
            };
            if active.compute.is_none()
                || !active.token_recorded
                || (requires_complete_output && !logits.is_empty() && !active.top_k_recorded)
            {
                state.trace_binding_invalid = true;
            }
            match state.next_decode_step_index.checked_add(1) {
                Some(next) => state.next_decode_step_index = next,
                None => state.trace_overflowed = true,
            }
            step_index
        };
        if !logits.is_empty() {
            let mut top = Vec::<(usize, f32)>::new();
            for (token, logit) in logits.iter().copied().enumerate() {
                if !logit.is_finite() {
                    continue;
                }
                let insert_at = top
                    .iter()
                    .position(|(_, existing)| logit.total_cmp(existing).is_gt());
                if let Some(insert_at) = insert_at {
                    top.insert(insert_at, (token, logit));
                } else if top.len() < MAX_TRACE_TOP_K {
                    top.push((token, logit));
                }
                if top.len() > MAX_TRACE_TOP_K {
                    top.truncate(MAX_TRACE_TOP_K);
                }
            }
            let items = top
                .iter()
                .map(|(token, value)| serde_json::json!({"token_id": token, "value": value}))
                .collect::<Vec<_>>();
            self.record_trace_event(
                serde_json::json!({
                    "schema": "openasr.gpu-correctness-trace.v1",
                    "event": "top_k",
                    "step_index": step_index,
                    "items": items,
                    "top1_top2_margin": margin,
                    "compute": compute,
                })
                .to_string(),
            );
        }
        self.record_trace_event(
            serde_json::json!({
                "schema": "openasr.gpu-correctness-trace.v1",
                "event": "token",
                "step_index": step_index,
                "token_id": token_id,
                "is_eot": usize::from(is_eot),
                "compute": compute,
            })
            .to_string(),
        );
    }

    fn begin_decode_step_locked(
        state: &mut ReceiptState,
        step_index: usize,
        compute: Option<GgmlSelectionEvidenceRef>,
    ) {
        if state.active_decode_step.is_some()
            || compute.is_none()
            || step_index != state.next_decode_step_index
        {
            state.trace_binding_invalid = true;
        }
        state.active_decode_step = Some(ActiveDecodeStep {
            step_index,
            compute,
            top_k_recorded: false,
            token_recorded: false,
        });
    }

    pub(crate) fn finish_decode_step(&self, step_index: usize) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(active) = state.active_decode_step.take() else {
            state.trace_binding_invalid = true;
            return;
        };
        let requires_complete_output = state.facts.as_ref().is_some_and(|facts| {
            matches!(
                facts.resolved_runtime.output_plan(),
                crate::ggml_runtime::GgmlDecodeOutputPlan::FullLogits
            )
        });
        let step_identity_matches =
            active.step_index == step_index && active.step_index == state.next_decode_step_index;
        if !step_identity_matches
            || active.compute.is_none()
            || !active.token_recorded
            || (requires_complete_output && !active.top_k_recorded)
        {
            state.trace_binding_invalid = true;
        }
        if step_identity_matches {
            match state.next_decode_step_index.checked_add(1) {
                Some(next) => state.next_decode_step_index = next,
                None => state.trace_overflowed = true,
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn abort_decode_step(&self, step_index: usize) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state
            .active_decode_step
            .take()
            .is_some_and(|active| active.step_index != step_index)
        {
            state.trace_binding_invalid = true;
        }
    }

    pub fn record_token(&self, step_index: usize, token_id: u32, is_eot: bool) {
        let compute = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let top2_margin = state.top_k_margins.get(&step_index).copied();
            let logits_sha256 = state.logits_hashes.get(&step_index).cloned();
            let compute = match state.active_decode_step.as_mut() {
                Some(active) if active.step_index == step_index && !active.token_recorded => {
                    active.token_recorded = true;
                    active.compute
                }
                _ => {
                    state.trace_binding_invalid = true;
                    None
                }
            };
            if let Some(existing) = state
                .token_steps
                .iter_mut()
                .find(|step| step.step_index == step_index)
            {
                existing.token_id = token_id;
                existing.is_eot = is_eot;
                if existing.top2_margin.is_none() {
                    existing.top2_margin = top2_margin;
                }
                if existing.logits_sha256.is_none() {
                    existing.logits_sha256 = logits_sha256;
                }
                existing.compute = compute;
            } else {
                state.token_steps.push(NativeExecutionTokenStep {
                    step_index,
                    token_id,
                    is_eot,
                    top2_margin,
                    logits_sha256,
                    compute,
                });
            }
            compute
        };
        self.record_trace_event(
            serde_json::json!({
                "schema": "openasr.gpu-correctness-trace.v1",
                "event": "token",
                "step_index": step_index,
                "token_id": token_id,
                "is_eot": usize::from(is_eot),
                "compute": compute,
            })
            .to_string(),
        );
    }

    pub fn record_top_k(&self, step_index: usize, logits: &[f32]) {
        self.record_top_k_with_tie_order(step_index, logits, false);
    }

    pub(crate) fn record_top_k_last_max(&self, step_index: usize, logits: &[f32]) {
        self.record_top_k_with_tie_order(step_index, logits, true);
    }

    fn record_top_k_with_tie_order(&self, step_index: usize, logits: &[f32], last_maximum: bool) {
        let logits_sha256 = diagnostic_logits_sha256(logits);
        let mut top = Vec::<(usize, f32)>::new();
        for (token_id, logit) in logits.iter().copied().enumerate() {
            if !logit.is_finite() {
                continue;
            }
            let insert_at = top.iter().position(|(_, existing)| {
                let order = logit.total_cmp(existing);
                order.is_gt() || (last_maximum && order.is_eq())
            });
            if let Some(insert_at) = insert_at {
                top.insert(insert_at, (token_id, logit));
            } else if top.len() < MAX_TRACE_TOP_K {
                top.push((token_id, logit));
            }
            if top.len() > MAX_TRACE_TOP_K {
                top.truncate(MAX_TRACE_TOP_K);
            }
        }
        let margin = top
            .first()
            .zip(top.get(1))
            .map(|((_, first), (_, second))| first - second);
        let compute = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(margin) = margin {
                state.top_k_margins.insert(step_index, margin);
                if let Some(existing) = state
                    .token_steps
                    .iter_mut()
                    .find(|step| step.step_index == step_index)
                {
                    existing.top2_margin = Some(margin);
                }
            }
            state
                .logits_hashes
                .insert(step_index, logits_sha256.clone());
            if let Some(existing) = state
                .token_steps
                .iter_mut()
                .find(|step| step.step_index == step_index)
            {
                existing.logits_sha256 = Some(logits_sha256.clone());
            }
            let compute = match state.active_decode_step.as_mut() {
                Some(active) if active.step_index == step_index && !active.top_k_recorded => {
                    active.top_k_recorded = true;
                    active.compute
                }
                _ => {
                    state.trace_binding_invalid = true;
                    None
                }
            };
            let requires_complete_output = state.facts.as_ref().is_some_and(|facts| {
                matches!(
                    facts.resolved_runtime.output_plan(),
                    crate::ggml_runtime::GgmlDecodeOutputPlan::FullLogits
                )
            });
            if state.capture_full_logits && requires_complete_output {
                let Some(compute) = compute else {
                    state.trace_binding_invalid = true;
                    return;
                };
                let Some(next_elements) = state.full_logits_elements.checked_add(logits.len())
                else {
                    state.trace_overflowed = true;
                    return;
                };
                if logits.is_empty()
                    || logits.len() > GPU_FULL_LOGITS_MAX_VOCAB
                    || logits.iter().any(|value| !value.is_finite())
                    || next_elements > MAX_FULL_LOGITS_ELEMENTS
                {
                    state.trace_overflowed = true;
                    return;
                }
                state.full_logits_elements = next_elements;
                state.full_logits_steps.push(FullLogitsStep {
                    step_index,
                    compute,
                    values: logits.to_vec(),
                });
            }
            compute
        };
        let items = top
            .iter()
            .map(|(token_id, value)| serde_json::json!({"token_id": token_id, "value": value}))
            .collect::<Vec<_>>();
        self.record_trace_event(
            serde_json::json!({
                "schema": "openasr.gpu-correctness-trace.v1",
                "event": "top_k",
                "step_index": step_index,
                "items": items,
                "top1_top2_margin": margin,
                "compute": compute,
            })
            .to_string(),
        );
    }

    fn record_trace_event(&self, event: String) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.trace_events.len() >= MAX_TRACE_EVENTS {
            state.trace_overflowed = true;
            return;
        }
        state.trace_events.push(event);
    }

    pub fn snapshot(&self) -> NativeExecutionReceiptSnapshot {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut lines = Vec::new();
        if let Some(facts) = &state.facts
            && let (
                Some(provider),
                Some(device),
                Some(actual_device),
                Some(trace_mode),
                Some(trace_run_id),
                Some(trace_process_nonce),
                Some(trace_process_id),
            ) = (
                facts.actual_provider,
                facts.actual_stable_device_id.as_deref(),
                facts.actual_device.as_ref(),
                state.trace_mode,
                state.trace_run_id.as_deref(),
                state.trace_process_nonce.as_deref(),
                state.trace_process_id,
            )
        {
            let backend_id = facts.backend_id.as_deref().unwrap_or("unqualified");
            let device_target = facts.device_target.as_deref().unwrap_or("unqualified");
            let artifact_fingerprint = facts
                .backend_artifact_fingerprint
                .as_deref()
                .unwrap_or("unqualified");
            let driver_version = facts
                .backend_driver_version
                .as_deref()
                .unwrap_or("unqualified");
            lines.push(
                serde_json::json!({
                    "schema": "openasr.gpu-correctness-trace.v1",
                    "event": "header",
                    "run_id": trace_run_id,
                    "process_nonce": trace_process_nonce,
                    "process_id": trace_process_id,
                    "mode": trace_mode.as_str(),
                    "graph_mode": if facts.resolved_runtime.reuse_mode() == crate::ggml_runtime::GgmlDecodeReuseMode::FreshGraph { "fresh_graph" } else { "reusable_graph" },
                    "provider": provider.as_str(),
                    "device": device,
                    "device_target": device_target,
                    "backend_id": backend_id,
                    "driver_version": driver_version,
                    "artifact_fingerprint": artifact_fingerprint,
                    "actual_provider": provider.as_str(),
                    "actual_stable_device_id": device,
                    "actual_device": actual_device,
                })
                .to_string(),
            );
        }
        let graph_lifecycle = self.graph_lifecycle.snapshot();
        lines.extend(
            graph_lifecycle
                .events
                .iter()
                .filter_map(|event| serde_json::to_string(event).ok()),
        );
        lines.extend(state.trace_events.iter().cloned());
        let full_logits_jsonl = if state.capture_full_logits {
            state.facts.as_ref().and_then(|facts| {
                let provider = facts.actual_provider?;
                let device = facts.actual_stable_device_id.as_deref()?;
                let actual_device = facts.actual_device.as_ref()?;
                let trace_mode = state.trace_mode?;
                let trace_run_id = state.trace_run_id.as_deref()?;
                let trace_process_nonce = state.trace_process_nonce.as_deref()?;
                let trace_process_id = state.trace_process_id?;
                let backend_id = facts.backend_id.as_deref().unwrap_or("unqualified");
                let device_target = facts.device_target.as_deref().unwrap_or("unqualified");
                let artifact_fingerprint = facts
                    .backend_artifact_fingerprint
                    .as_deref()
                    .unwrap_or("unqualified");
                let driver_version = facts
                    .backend_driver_version
                    .as_deref()
                    .unwrap_or("unqualified");
                let mut artifact_lines = Vec::with_capacity(state.full_logits_steps.len() + 1);
                artifact_lines.push(
                    serde_json::to_string(&FullLogitsHeader {
                        schema: GPU_FULL_LOGITS_TRACE_SCHEMA,
                        event: "header",
                        run_id: trace_run_id,
                        process_nonce: trace_process_nonce,
                        process_id: trace_process_id,
                        mode: trace_mode.as_str(),
                        graph_mode: if facts.resolved_runtime.reuse_mode()
                            == crate::ggml_runtime::GgmlDecodeReuseMode::FreshGraph
                        {
                            "fresh_graph"
                        } else {
                            "reusable_graph"
                        },
                        provider: provider.as_str(),
                        device_target,
                        backend_id,
                        driver_version,
                        artifact_fingerprint,
                        device,
                        actual_provider: provider.as_str(),
                        actual_stable_device_id: device,
                        actual_device,
                        dtype: "f32",
                        encoding: "json_numbers",
                        step_count: state.full_logits_steps.len(),
                    })
                    .expect("full logits header serialization is infallible"),
                );
                artifact_lines.extend(state.full_logits_steps.iter().map(|step| {
                    serde_json::to_string(&FullLogitsArtifactStep {
                        schema: GPU_FULL_LOGITS_TRACE_SCHEMA,
                        event: "logits",
                        step_index: step.step_index,
                        compute: step.compute,
                        vocab_size: step.values.len(),
                        values: &step.values,
                    })
                    .expect("finite full logits serialization is infallible")
                }));
                Some(format!("{}\n", artifact_lines.join("\n")))
            })
        } else {
            None
        };
        NativeExecutionReceiptSnapshot {
            request_attempt_id: state.request_attempt_id,
            request_attempt_conflicted: state.request_attempt_conflicted,
            phase_duration_micros: state.phase_duration_micros.clone(),
            timing_complete: !state.timeline_conflicted
                && [
                    RequestExecutionPhase::UploadIngest,
                    RequestExecutionPhase::DecodeNormalize,
                    RequestExecutionPhase::AdmissionWait,
                    RequestExecutionPhase::Compute,
                ]
                .iter()
                .all(|phase| state.phase_duration_micros.contains_key(phase)),
            terminal: state.terminal,
            timeline_conflicted: state.timeline_conflicted,
            facts: state.facts.clone(),
            placement: state.placement.clone(),
            trace: NativeExecutionTraceSnapshot {
                jsonl: if lines.is_empty() {
                    String::new()
                } else {
                    format!("{}\n", lines.join("\n"))
                },
                full_logits_jsonl,
                overflowed: state.trace_overflowed,
                invalid_binding: state.trace_binding_invalid || state.active_decode_step.is_some(),
                event_count: state.trace_events.len(),
                full_logits_step_count: state.full_logits_steps.len(),
            },
            graph_lifecycle,
            token_steps: state.token_steps.clone(),
            completed: state.completed,
        }
    }
}

impl PartialEq for NativeExecutionReceiptCollector {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }
}

impl Eq for NativeExecutionReceiptCollector {}

/// Record the immutable family, pack, lane, and output-plan facts selected by
/// the successful candidate attempt. Offline, streaming, warm-up, and model
/// activation all call this one interface; a path that has no explicit
/// collector remains uninstrumented rather than reconstructing facts later.
pub(crate) fn record_request_execution_facts(
    receipt: Option<&NativeExecutionReceiptCollector>,
    verified_pack: &crate::models::pack_verifier::VerifiedPack,
    selected_family: &crate::GgmlFamilyAdapterDescriptor,
    resolved_runtime: ResolvedFamilyRuntimeInput,
    execution_lane: &ExecutionLaneKey,
) -> Result<(), String> {
    let Some(receipt) = receipt else {
        return Ok(());
    };
    let descriptor = crate::arch::OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(selected_family.model_architecture)
        .ok_or_else(|| {
            "selected native family is absent from the architecture inventory".to_string()
        })?;
    let decode_driver = match descriptor.topology_contract.decode_driver {
        crate::arch::OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy { .. } => {
            "shared-seq2seq-greedy"
        }
        crate::arch::OpenAsrDecodeDriverStrategy::SharedCtcGreedy { .. } => "shared-ctc-greedy",
        crate::arch::OpenAsrDecodeDriverStrategy::Dedicated { .. } => "dedicated",
    };
    let decoder_state = match descriptor.topology_contract.decoder_state_topology {
        crate::arch::OpenAsrDecoderStateTopology::None => "none",
        crate::arch::OpenAsrDecoderStateTopology::CausalSelfAttentionKv => {
            "causal-self-attention-kv"
        }
        crate::arch::OpenAsrDecoderStateTopology::EncoderDecoderSelfAndCrossAttentionKv => {
            "encoder-decoder-self-and-cross-attention-kv"
        }
        crate::arch::OpenAsrDecoderStateTopology::FamilyDefinedTokenScaledPersistent => {
            "family-defined-token-scaled-persistent"
        }
    };
    let block_stack = match descriptor.topology_contract.block_stack {
        crate::arch::OpenAsrBlockStackStrategy::Shared(_) => "shared",
        crate::arch::OpenAsrBlockStackStrategy::ArchitectureGraph { .. } => "architecture-graph",
    };
    let activated_backend = crate::ggml_runtime::activated_backend_execution_identity()
        .filter(|identity| identity.provider == execution_lane.provider());
    receipt.record_facts(NativeExecutionRequestFacts {
        resolved_runtime,
        execution_lane: execution_lane.clone(),
        selected_provider: execution_lane.provider(),
        stable_device_id: execution_lane.stable_device_id().to_string(),
        backend_id: activated_backend
            .as_ref()
            .map(|identity| identity.backend_id.clone()),
        device_target: activated_backend
            .as_ref()
            .map(|identity| identity.device_target.clone()),
        backend_driver_version: activated_backend
            .as_ref()
            .map(|identity| identity.driver_version.clone()),
        backend_artifact_fingerprint: activated_backend
            .as_ref()
            .map(|identity| identity.artifact_fingerprint.clone()),
        placement: execution_lane.placement(),
        backend: execution_lane.backend(),
        topology: NativeExecutionTopologyFacts {
            family: selected_family.model_family.to_string(),
            model_architecture: selected_family.model_architecture.to_string(),
            adapter_id: selected_family.adapter_id.to_string(),
            decode_policy_id: selected_family.decode_policy_id.to_string(),
            decode_driver: decode_driver.to_string(),
            decoder_state: decoder_state.to_string(),
            block_stack: block_stack.to_string(),
        },
        pack_content_id: verified_pack.content_id().to_string(),
        pack_size_bytes: verified_pack.preflight().runtime_source().byte_len(),
        actual_provider: None,
        actual_stable_device_id: None,
        actual_device: None,
        scheduler_enabled: None,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_actual_device() -> GgmlActualDeviceFacts {
        GgmlActualDeviceFacts {
            device_type: "cpu".to_string(),
            name: "CPU".to_string(),
            description: "test CPU".to_string(),
            provider_device_id: None,
            pci_vendor_id: None,
        }
    }

    #[test]
    fn warm_and_measured_passes_require_fresh_backend_attestation() {
        let receipt = NativeExecutionReceiptCollector::new();
        for (token_id, pass) in [(11, "warmup"), (22, "measured")] {
            let execution_lane =
                super::super::native_execution_services::current_execution_lane_key(
                    GgmlCpuGraphBackend::Cpu,
                );
            receipt.begin_candidate_attempt();
            receipt.record_facts(NativeExecutionRequestFacts {
                resolved_runtime: ResolvedFamilyRuntimeInput::resolve(
                    None,
                    crate::ggml_runtime::AutoGpuPolicy::Never,
                ),
                execution_lane: execution_lane.clone(),
                selected_provider: ExecutionProvider::Cpu,
                stable_device_id: "CPU".to_string(),
                backend_id: None,
                device_target: None,
                backend_driver_version: None,
                backend_artifact_fingerprint: None,
                placement: ExecutionPlacement::CpuOnly,
                backend: GgmlCpuGraphBackend::Cpu,
                topology: NativeExecutionTopologyFacts {
                    family: "test".to_string(),
                    model_architecture: "test".to_string(),
                    adapter_id: "test".to_string(),
                    decode_policy_id: "test".to_string(),
                    decode_driver: "test".to_string(),
                    decoder_state: "none".to_string(),
                    block_stack: "shared".to_string(),
                },
                pack_content_id: "test-pack".to_string(),
                pack_size_bytes: 1,
                actual_provider: None,
                actual_stable_device_id: None,
                actual_device: None,
                scheduler_enabled: None,
            });
            receipt.record_backend_observation(
                1,
                ExecutionProvider::Cpu,
                "CPU",
                &test_actual_device(),
                false,
            );
            receipt.record_token(0, token_id, true);
            receipt.record_top_k(0, &[2.0, 1.0]);
            receipt.finish_candidate_attempt(true);

            let snapshot = receipt.snapshot();
            let facts = snapshot.facts.expect("pass facts");
            assert_eq!(
                facts.actual_provider,
                Some(ExecutionProvider::Cpu),
                "{pass}"
            );
            assert_eq!(
                facts.actual_stable_device_id.as_deref(),
                Some("CPU"),
                "{pass}"
            );
            assert_eq!(facts.scheduler_enabled, Some(false), "{pass}");
            assert!(
                snapshot
                    .trace
                    .jsonl
                    .contains(&format!("\"token_id\":{token_id}")),
                "{pass}"
            );
            assert!(snapshot.trace.event_count > 0, "{pass}");
            assert_eq!(snapshot.token_steps.len(), 1, "{pass}");
            assert_eq!(snapshot.token_steps[0].token_id, token_id, "{pass}");
            assert_eq!(snapshot.token_steps[0].top2_margin, Some(1.0), "{pass}");
            assert!(snapshot.completed, "{pass}");
        }
    }

    #[test]
    fn failed_candidate_discards_its_trace_before_fallback_commits() {
        let receipt = NativeExecutionReceiptCollector::new();
        receipt.begin_candidate_attempt();
        receipt.record_token(0, 11, false);
        receipt.finish_candidate_attempt(false);
        receipt.begin_candidate_attempt();
        receipt.record_token(0, 22, false);
        receipt.finish_candidate_attempt(true);
        let snapshot = receipt.snapshot();
        let trace = snapshot.trace.jsonl;
        assert!(!trace.contains("\"token_id\":11"));
        assert!(trace.contains("\"token_id\":22"));
        assert_eq!(snapshot.token_steps.len(), 1);
        assert_eq!(snapshot.token_steps[0].token_id, 22);
    }

    #[test]
    fn nested_candidate_restores_parent_request_facts() {
        let receipt = NativeExecutionReceiptCollector::new();
        receipt.begin_candidate_attempt();
        receipt.record_facts(seq2seq_facts());
        receipt.record_token(0, 7, false);
        receipt.begin_candidate_attempt();
        let mut nested_facts = seq2seq_facts();
        nested_facts.pack_content_id = "nested-aux-pack".to_string();
        receipt.record_facts(nested_facts);
        receipt.record_token(0, 99, false);
        receipt.finish_candidate_attempt(true);
        receipt.finish_candidate_attempt(true);
        let snapshot = receipt.snapshot();
        let facts = snapshot.facts.expect("parent facts");
        assert_eq!(facts.pack_content_id, "test-pack");
        assert_eq!(snapshot.token_steps.len(), 1);
        assert_eq!(snapshot.token_steps[0].token_id, 7);
        assert!(snapshot.completed);
    }

    #[test]
    fn later_facts_writer_cannot_erase_first_request_facts() {
        let receipt = NativeExecutionReceiptCollector::new();
        receipt.begin_candidate_attempt();
        receipt.record_facts(seq2seq_facts());
        let mut drifted = seq2seq_facts();
        drifted.pack_content_id = "other-pack".to_string();
        receipt.record_facts(drifted);
        receipt.finish_candidate_attempt(true);
        let facts = receipt.snapshot().facts.expect("first facts");
        assert_eq!(facts.pack_content_id, "test-pack");
    }

    #[test]
    fn backend_observation_drift_does_not_erase_request_facts() {
        let receipt = NativeExecutionReceiptCollector::new();
        receipt.begin_candidate_attempt();
        receipt.record_facts(seq2seq_facts());
        receipt.record_backend_observation(
            1,
            ExecutionProvider::Cpu,
            "CPU",
            &test_actual_device(),
            false,
        );
        let mut other = test_actual_device();
        other.description = "other CPU".to_string();
        receipt.record_backend_observation(2, ExecutionProvider::Cpu, "CPU", &other, true);
        receipt.record_token(0, 7, false);
        receipt.finish_candidate_attempt(true);
        let facts = receipt.snapshot().facts.expect("request facts");
        assert_eq!(facts.pack_content_id, "test-pack");
        assert_eq!(facts.actual_provider, Some(ExecutionProvider::Cpu));
        assert_eq!(facts.scheduler_enabled, Some(false));
        assert_eq!(
            facts.actual_device.as_ref().unwrap().description,
            "test CPU"
        );
    }

    #[test]
    fn conflicting_request_attempt_binding_is_irreversible_and_cannot_complete() {
        let receipt = NativeExecutionReceiptCollector::new();
        let first = crate::RequestAttemptId::parse("00112233445566778899aabbccddeeff").unwrap();
        let second = crate::RequestAttemptId::parse("ffeeddccbbaa99887766554433221100").unwrap();
        receipt.bind_request_attempt(first);
        receipt.bind_request_attempt(second);
        receipt.bind_request_attempt(first);
        receipt.begin_candidate_attempt();
        receipt.finish_candidate_attempt(true);

        let snapshot = receipt.snapshot();
        assert_eq!(snapshot.request_attempt_id, Some(first));
        assert!(snapshot.request_attempt_conflicted);
        assert!(!snapshot.completed);
        assert_eq!(receipt.request_attempt_id(), None);
    }

    fn seq2seq_facts() -> NativeExecutionRequestFacts {
        let execution_lane = super::super::native_execution_services::current_execution_lane_key(
            GgmlCpuGraphBackend::Cpu,
        );
        NativeExecutionRequestFacts {
            resolved_runtime: ResolvedFamilyRuntimeInput::resolve(
                None,
                crate::ggml_runtime::AutoGpuPolicy::Never,
            ),
            execution_lane,
            selected_provider: ExecutionProvider::Cpu,
            stable_device_id: "CPU".to_string(),
            backend_id: None,
            device_target: None,
            backend_driver_version: None,
            backend_artifact_fingerprint: None,
            placement: ExecutionPlacement::CpuOnly,
            backend: GgmlCpuGraphBackend::Cpu,
            topology: NativeExecutionTopologyFacts {
                family: "test".to_string(),
                model_architecture: "test".to_string(),
                adapter_id: "test".to_string(),
                decode_policy_id: "test".to_string(),
                decode_driver: "shared-seq2seq-greedy".to_string(),
                decoder_state: "none".to_string(),
                block_stack: "shared".to_string(),
            },
            pack_content_id: "test-pack".to_string(),
            pack_size_bytes: 1,
            actual_provider: None,
            actual_stable_device_id: None,
            actual_device: None,
            scheduler_enabled: None,
        }
    }

    #[test]
    fn complete_logits_trace_binds_runtime_row_to_successful_output_read() {
        let receipt = NativeExecutionReceiptCollector::new();
        receipt.set_trace_mode(NativeExecutionTraceMode::Cold);
        receipt.enable_full_logits_trace();
        receipt.begin_candidate_attempt();
        let mut facts = seq2seq_facts();
        facts.resolved_runtime =
            ResolvedFamilyRuntimeInput::resolve_with_output_contract_and_consumers(
                Some(crate::ggml_runtime::RequestBackendPreference::CpuOnly),
                crate::ggml_runtime::AutoGpuPolicy::Never,
                crate::ggml_runtime::GgmlDecodeOutputContract::NativeFirstMaxTokenOrFullLogits,
                crate::ggml_runtime::GgmlDecodeLogitsConsumers::none().with_debug_logits(true),
            );
        receipt.record_facts(facts);
        receipt.record_backend_observation(
            1,
            ExecutionProvider::Cpu,
            "CPU",
            &test_actual_device(),
            false,
        );

        let lifecycle = receipt.graph_lifecycle_collector();
        let _guard = lifecycle.install();
        let mut runner = crate::ggml_runtime::GgmlCpuGraphRunner::new(
            crate::ggml_runtime::GgmlCpuGraphConfig::conservative_default(),
        )
        .expect("CPU graph runner");
        let mut graph = runner.start_graph();
        let input = graph.new_tensor_1d_f32(3, "trace_input").expect("input");
        graph.set_input(input).expect("set input");
        graph.set_output(input).expect("set output");
        graph
            .set_f32_slice(input, &[0.0, 2.0, 1.0], "trace_input")
            .expect("input upload");
        let output = graph
            .compute_output_f32_with_evidence(input, 3)
            .expect("output read");
        let (logits, compute) = output.into_parts();
        drop(graph);

        receipt.begin_decode_step(0, compute);
        receipt.record_top_k(0, &logits);
        receipt.record_token(0, 1, false);
        receipt.finish_decode_step(0);
        receipt.finish_candidate_attempt(true);
        let snapshot = receipt.snapshot();
        assert!(!snapshot.trace.invalid_binding);
        assert!(!snapshot.trace.overflowed);
        assert_eq!(snapshot.trace.full_logits_step_count, 1);
        let diagnostics = crate::decode_diagnostics_from_shipped_runtime(None, Some(&snapshot))
            .expect("observed graph lifecycle projects decode diagnostics");
        assert!(diagnostics.steps[0].graph_rebuilt);
        let artifact = snapshot
            .trace
            .full_logits_jsonl
            .expect("complete logits artifact");
        let token_header: serde_json::Value = serde_json::from_str(
            snapshot
                .trace
                .jsonl
                .lines()
                .next()
                .expect("token trace header"),
        )
        .expect("valid token trace header");
        let logits_header: serde_json::Value =
            serde_json::from_str(artifact.lines().next().expect("full logits header"))
                .expect("valid full logits header");
        assert_eq!(token_header["run_id"], logits_header["run_id"]);
        assert_eq!(
            token_header["process_nonce"],
            logits_header["process_nonce"]
        );
        assert_eq!(token_header["process_id"], logits_header["process_id"]);
        for field in [
            "mode",
            "graph_mode",
            "provider",
            "device_target",
            "backend_id",
            "driver_version",
            "artifact_fingerprint",
            "device",
            "actual_provider",
            "actual_stable_device_id",
            "actual_device",
        ] {
            assert_eq!(token_header[field], logits_header[field], "{field}");
        }
        assert_eq!(
            token_header["process_id"].as_u64(),
            Some(u64::from(std::process::id()))
        );
        assert_eq!(
            token_header["process_nonce"]
                .as_str()
                .expect("process nonce")
                .len(),
            32
        );
        assert!(artifact.contains(GPU_FULL_LOGITS_TRACE_SCHEMA));
        assert!(artifact.contains("\"values\":[0.0,2.0,1.0]"));
        assert!(snapshot.trace.jsonl.contains("\"compute\":"));
    }

    #[test]
    fn activation_attestation_requires_exact_plan_lane_and_live_backend() {
        let receipt = NativeExecutionReceiptCollector::new();
        let facts = seq2seq_facts();
        let expected_runtime = facts.resolved_runtime;
        receipt.begin_candidate_attempt();
        receipt.record_facts(facts);
        receipt.record_backend_observation(
            1,
            ExecutionProvider::Cpu,
            "CPU",
            &test_actual_device(),
            false,
        );
        receipt.finish_candidate_attempt(true);
        let snapshot = receipt.snapshot();

        snapshot
            .attest_activation(
                "test-pack",
                expected_runtime,
                ExecutionProvider::Cpu,
                "CPU",
                ExecutionPlacement::CpuOnly,
            )
            .expect("matching activation receipt must attest");
        assert_eq!(
            snapshot.attest_activation(
                "test-pack",
                expected_runtime,
                ExecutionProvider::Cpu,
                "CPU-other",
                ExecutionPlacement::CpuOnly,
            ),
            Err(NativeExecutionAttestationError::LaneMismatch)
        );

        let missing_live = NativeExecutionReceiptCollector::new();
        missing_live.begin_candidate_attempt();
        missing_live.record_facts(seq2seq_facts());
        missing_live.finish_candidate_attempt(true);
        assert!(matches!(
            missing_live.snapshot().attest_activation(
                "test-pack",
                expected_runtime,
                ExecutionProvider::Cpu,
                "CPU",
                ExecutionPlacement::CpuOnly,
            ),
            Err(NativeExecutionAttestationError::LiveBackendMismatch {
                expected_provider: ExecutionProvider::Cpu,
                actual_provider: None,
                scheduler_enabled: None,
                ..
            })
        ));
    }

    #[test]
    fn seq2seq_receipt_fails_closed_without_token_steps() {
        let receipt = NativeExecutionReceiptCollector::new();
        receipt.begin_candidate_attempt();
        receipt.record_facts(seq2seq_facts());
        receipt.record_backend_observation(
            1,
            ExecutionProvider::Cpu,
            "CPU",
            &test_actual_device(),
            false,
        );
        receipt.finish_candidate_attempt(true);
        let snapshot = receipt.snapshot();
        assert!(snapshot.completed);
        assert_eq!(
            snapshot.facts.as_ref().unwrap().scheduler_enabled,
            Some(false)
        );
        assert!(snapshot.token_steps.is_empty());
        let error = crate::decode_diagnostics_from_shipped_runtime(None, Some(&snapshot))
            .expect_err("seq2seq native receipt without tokens must fail closed");
        assert_eq!(
            error,
            crate::ShortAudioReceiptError::NativeSeq2SeqTokenStepsMissing
        );
    }

    #[test]
    fn seq2seq_receipt_rejects_token_steps_without_native_compute_witness() {
        let receipt = NativeExecutionReceiptCollector::new();
        receipt.begin_candidate_attempt();
        receipt.record_facts(seq2seq_facts());
        receipt.record_backend_observation(
            1,
            ExecutionProvider::Cpu,
            "CPU",
            &test_actual_device(),
            false,
        );
        receipt.record_token(0, 11, false);
        receipt.record_top_k(0, &[4.0, 1.5]);
        receipt.finish_candidate_attempt(true);
        let snapshot = receipt.snapshot();
        let error = crate::decode_diagnostics_from_shipped_runtime(None, Some(&snapshot))
            .expect_err("a caller-recorded token is not graph evidence");
        assert!(matches!(
            error,
            crate::ShortAudioReceiptError::InvalidEvidenceField {
                field: "decode_diagnostics.steps.graph_rebuilt",
                ..
            }
        ));
        assert!(snapshot.completed);
        assert!(snapshot.trace.event_count > 0);
        assert_eq!(
            snapshot.facts.as_ref().unwrap().scheduler_enabled,
            Some(false)
        );
    }
}
