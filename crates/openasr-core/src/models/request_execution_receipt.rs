//! Typed, bounded request-local facts for native execution receipts.
//!
//! This collector is deliberately an in-memory authority. JSON projection is
//! owned by `short_audio_receipt`; no caller may reconstruct facts from a
//! backend label, environment variable, or CLI policy option after execution.

use std::sync::{Arc, Mutex};

use crate::{
    device::{execution_policy::ExecutionPlacement, execution_route::ExecutionProvider},
    ggml_runtime::{
        GgmlCpuGraphBackend, GgmlExecutionPlacementSummary, ResolvedFamilyRuntimeInput,
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
pub struct NativeExecutionReceiptSnapshot {
    pub facts: Option<NativeExecutionRequestFacts>,
    pub placement: GgmlExecutionPlacementSummary,
    pub trace: NativeExecutionTraceSnapshot,
    pub completed: bool,
}

#[derive(Debug, Default)]
struct ReceiptState {
    facts: Option<NativeExecutionRequestFacts>,
    placement: GgmlExecutionPlacementSummary,
    trace_events: Vec<String>,
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
        state.trace_overflowed = false;
        state.completed = false;
    }

    pub(crate) fn finish_candidate_attempt(&self, committed: bool) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if committed {
            state.completed = true;
        } else {
            state.facts = None;
            state.placement = GgmlExecutionPlacementSummary::default();
            state.trace_events.clear();
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

    pub(crate) fn record_token(&self, step_index: usize, token_id: u32, is_eot: bool) {
        self.record_trace_event(format!(
            "{{\"schema\":\"openasr.gpu-correctness-trace.v1\",\"event\":\"token\",\"step_index\":{step_index},\"token_id\":{token_id},\"is_eot\":{}}}",
            usize::from(is_eot)
        ));
    }

    pub(crate) fn record_top_k(&self, step_index: usize, logits: &[f32]) {
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
            lines.push(format!(
                "{{\"schema\":\"openasr.gpu-correctness-trace.v1\",\"event\":\"header\",\"mode\":\"{}\",\"provider\":\"{}\",\"device\":\"{}\"}}",
                if facts.resolved_runtime.reuse_mode() == crate::ggml_runtime::GgmlDecodeReuseMode::FreshGraph { "cold" } else { "reuse" },
                facts.selected_provider.as_str(),
                facts.stable_device_id,
            ));
        }
        lines.extend(state.trace_events.iter().cloned());
        NativeExecutionReceiptSnapshot {
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
            completed: state.completed,
        }
    }
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
        let trace = receipt.snapshot().trace.jsonl;
        assert!(!trace.contains("\"token_id\":11"));
        assert!(trace.contains("\"token_id\":22"));
    }
}
