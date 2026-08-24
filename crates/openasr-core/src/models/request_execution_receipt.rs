//! Typed, bounded request-local facts for native execution receipts.
//!
//! This collector is deliberately an in-memory authority. JSON projection is
//! owned by `short_audio_receipt`; no caller may reconstruct facts from a
//! backend label, environment variable, or CLI policy option after execution.

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Duration,
};
use thiserror::Error;

use crate::{
    RequestAttemptId,
    device::{execution_policy::ExecutionPlacement, execution_route::ExecutionProvider},
    ggml_runtime::{
        GgmlCpuGraphBackend, GgmlExecutionPlacementSummary, ResolvedFamilyRuntimeInput,
        diagnostic_logits_sha256,
    },
};

use super::native_execution_services::ExecutionLaneKey;

const MAX_TRACE_EVENTS: usize = 4_096;
const MAX_TRACE_TOP_K: usize = 8;

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
    pub scheduler_enabled: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeExecutionTraceSnapshot {
    pub jsonl: String,
    pub overflowed: bool,
    pub event_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeExecutionTokenStep {
    pub step_index: usize,
    pub token_id: u32,
    pub is_eot: bool,
    pub top2_margin: Option<f32>,
    pub logits_sha256: Option<String>,
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
    placement: GgmlExecutionPlacementSummary,
    trace_events: Vec<String>,
    token_steps: Vec<NativeExecutionTokenStep>,
    top_k_margins: BTreeMap<usize, f32>,
    logits_hashes: BTreeMap<usize, String>,
    trace_overflowed: bool,
    completed: bool,
}

/// Cloneable request-scoped receipt collector. It is installed only by an
/// explicit caller such as the strict short-audio row producer.
#[derive(Debug, Clone, Default)]
pub struct NativeExecutionReceiptCollector {
    state: Arc<Mutex<ReceiptState>>,
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
    /// facts or trace events that a later fallback might publish.
    pub(crate) fn begin_candidate_attempt(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.facts = None;
        state.placement = GgmlExecutionPlacementSummary::default();
        state.trace_events.clear();
        state.token_steps.clear();
        state.top_k_margins.clear();
        state.logits_hashes.clear();
        state.trace_overflowed = false;
        state.completed = false;
    }

    pub(crate) fn finish_candidate_attempt(&self, committed: bool) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if committed && !state.request_attempt_conflicted {
            state.completed = true;
        } else {
            state.facts = None;
            state.placement = GgmlExecutionPlacementSummary::default();
            state.trace_events.clear();
            state.token_steps.clear();
            state.top_k_margins.clear();
            state.logits_hashes.clear();
            state.trace_overflowed = false;
            state.completed = false;
        }
    }

    pub(crate) fn record_facts(&self, facts: NativeExecutionRequestFacts) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match &state.facts {
            Some(existing) if existing != &facts => {
                // The same candidate may build several graphs, but immutable
                // request facts must not drift within one receipt row.
                state.facts = None;
            }
            Some(_) => {}
            None => state.facts = Some(facts),
        }
    }

    pub(crate) fn record_backend_observation(
        &self,
        provider: ExecutionProvider,
        stable_device_id: &str,
        scheduler_enabled: bool,
    ) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(facts) = state.facts.as_mut() else {
            return;
        };
        if facts
            .actual_provider
            .is_some_and(|actual| actual != provider)
            || facts
                .actual_stable_device_id
                .as_deref()
                .is_some_and(|actual| actual != stable_device_id)
            || facts
                .scheduler_enabled
                .is_some_and(|actual| actual != scheduler_enabled)
        {
            state.facts = None;
            return;
        }
        facts.actual_provider = Some(provider);
        facts.actual_stable_device_id = Some(stable_device_id.to_string());
        facts.scheduler_enabled = Some(scheduler_enabled);
    }

    pub(crate) fn record_placement(&self, placement: GgmlExecutionPlacementSummary) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.placement = placement;
    }

    pub fn record_token(&self, step_index: usize, token_id: u32, is_eot: bool) {
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let top2_margin = state.top_k_margins.get(&step_index).copied();
            let logits_sha256 = state.logits_hashes.get(&step_index).cloned();
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
            } else {
                state.token_steps.push(NativeExecutionTokenStep {
                    step_index,
                    token_id,
                    is_eot,
                    top2_margin,
                    logits_sha256,
                });
            }
        }
        self.record_trace_event(format!(
            "{{\"schema\":\"openasr.gpu-correctness-trace.v1\",\"event\":\"token\",\"step_index\":{step_index},\"token_id\":{token_id},\"is_eot\":{}}}",
            usize::from(is_eot)
        ));
    }

    pub fn record_top_k(&self, step_index: usize, logits: &[f32]) {
        let logits_sha256 = diagnostic_logits_sha256(logits);
        let non_finite_count = logits.iter().filter(|value| !value.is_finite()).count();
        self.record_trace_event(format!(
            "{{\"schema\":\"openasr.gpu-correctness-trace.v1\",\"event\":\"logits_digest\",\"step_index\":{step_index},\"element_count\":{},\"sha256\":\"{logits_sha256}\",\"non_finite_count\":{non_finite_count}}}",
            logits.len(),
        ));
        let mut top = Vec::<(usize, f32)>::new();
        for (token_id, logit) in logits.iter().copied().enumerate() {
            if !logit.is_finite() {
                continue;
            }
            let insert_at = top
                .iter()
                .position(|(_, existing)| logit.total_cmp(existing).is_gt());
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
        if let Some(margin) = margin {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.top_k_margins.insert(step_index, margin);
            if let Some(existing) = state
                .token_steps
                .iter_mut()
                .find(|step| step.step_index == step_index)
            {
                existing.top2_margin = Some(margin);
            }
        }
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
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
        }
        let items = top
            .iter()
            .map(|(token_id, logit)| format!("{{\"token_id\":{token_id},\"value\":{logit:.6}}}"))
            .collect::<Vec<_>>()
            .join(",");
        self.record_trace_event(format!(
            "{{\"schema\":\"openasr.gpu-correctness-trace.v1\",\"event\":\"top_k\",\"step_index\":{step_index},\"items\":[{items}],\"top1_top2_margin\":{}}}",
            margin
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "null".to_string()),
        ));
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
        if let Some(facts) = &state.facts {
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
            lines.push(format!(
                "{{\"schema\":\"openasr.gpu-correctness-trace.v1\",\"event\":\"header\",\"graph_mode\":\"{}\",\"provider\":\"{}\",\"device_target\":\"{}\",\"backend_id\":\"{}\",\"driver_version\":\"{}\",\"artifact_fingerprint\":\"{}\",\"device\":\"{}\"}}",
                if facts.resolved_runtime.reuse_mode() == crate::ggml_runtime::GgmlDecodeReuseMode::FreshGraph { "fresh_graph" } else { "reusable_graph" },
                facts.selected_provider.as_str(),
                device_target,
                backend_id,
                driver_version,
                artifact_fingerprint,
                facts.stable_device_id,
            ));
        }
        lines.extend(state.trace_events.iter().cloned());
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
                overflowed: state.trace_overflowed,
                event_count: state.trace_events.len(),
            },
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
        scheduler_enabled: None,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
                scheduler_enabled: None,
            });
            receipt.record_backend_observation(ExecutionProvider::Cpu, "CPU", false);
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
            scheduler_enabled: None,
        }
    }

    #[test]
    fn activation_attestation_requires_exact_plan_lane_and_live_backend() {
        let receipt = NativeExecutionReceiptCollector::new();
        let facts = seq2seq_facts();
        let expected_runtime = facts.resolved_runtime;
        receipt.begin_candidate_attempt();
        receipt.record_facts(facts);
        receipt.record_backend_observation(ExecutionProvider::Cpu, "CPU", false);
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
        receipt.record_backend_observation(ExecutionProvider::Cpu, "CPU", false);
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
    fn seq2seq_receipt_projects_token_steps_from_record_token() {
        let receipt = NativeExecutionReceiptCollector::new();
        receipt.begin_candidate_attempt();
        receipt.record_facts(seq2seq_facts());
        receipt.record_backend_observation(ExecutionProvider::Cpu, "CPU", false);
        receipt.record_token(0, 11, false);
        receipt.record_top_k(0, &[4.0, 1.5]);
        receipt.finish_candidate_attempt(true);
        let snapshot = receipt.snapshot();
        let diagnostics = crate::decode_diagnostics_from_shipped_runtime(None, Some(&snapshot))
            .expect("seq2seq token steps must project");
        assert_eq!(diagnostics.steps.len(), 1);
        assert_eq!(diagnostics.steps[0].token_id, Some(11));
        assert_eq!(diagnostics.steps[0].top2_margin, Some(2.5));
        assert!(snapshot.completed);
        assert!(snapshot.trace.event_count > 0);
        assert_eq!(
            snapshot.facts.as_ref().unwrap().scheduler_enabled,
            Some(false)
        );
    }
}
