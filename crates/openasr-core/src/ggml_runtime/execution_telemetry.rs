//! Request-scoped observation of where ggml graph nodes actually execute.
//!
//! The collector is opt-in and dynamically scoped. Native request contexts
//! explicitly propagate it to owner/worker threads; production requests that
//! do not install a collector pay only one TLS lookup per graph compute.

use std::{
    cell::RefCell,
    collections::{BTreeMap, HashSet},
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};

/// Aggregated execution placement observed from actual graph computes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgmlExecutionPlacementSummary {
    pub direct_graph_computes: u64,
    pub scheduler_graph_computes: u64,
    /// Prepared-graph operation nodes observed on each runtime backend.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub observed_nodes_by_backend: BTreeMap<String, u64>,
    /// Observed nodes that require a backend kernel, excluding metadata-only
    /// view operations such as reshape/view/permute/transpose.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub observed_compute_nodes_by_backend: BTreeMap<String, u64>,
    /// Sum of observed node output-tensor bytes grouped by runtime backend.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub observed_node_output_bytes_by_backend: BTreeMap<String, u64>,
    /// Bounded examples of compute nodes assigned away from the requested
    /// backend. Metadata-only view nodes are intentionally excluded.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fallback_node_samples_by_backend: BTreeMap<String, Vec<GgmlExecutionNodeSample>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgmlExecutionNodeSample {
    pub name: String,
    pub op: String,
    pub output_bytes: u64,
}

#[derive(Default)]
struct TelemetryTargetState {
    summary: GgmlExecutionPlacementSummary,
    observed_graph_ids: HashSet<u64>,
}

type TelemetryTarget = Arc<Mutex<TelemetryTargetState>>;

/// Cloneable sink installed around one request or benchmark receipt.
#[derive(Clone)]
pub struct GgmlExecutionTelemetryCollector {
    targets: Arc<[TelemetryTarget]>,
}

impl Default for GgmlExecutionTelemetryCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl GgmlExecutionTelemetryCollector {
    pub fn new() -> Self {
        Self {
            targets: Arc::from([Arc::new(Mutex::new(TelemetryTargetState::default()))]),
        }
    }

    /// Install this collector on the current thread until the guard drops.
    pub fn install(&self) -> GgmlExecutionTelemetryGuard {
        install_execution_telemetry_collector(Some(self.clone()))
    }

    pub fn snapshot(&self) -> GgmlExecutionPlacementSummary {
        let mut merged = GgmlExecutionPlacementSummary::default();
        for target in self.targets.iter() {
            let snapshot = target
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            merged.merge_from(&snapshot.summary);
        }
        merged
    }

    pub(crate) fn fanout<'a>(
        collectors: impl IntoIterator<Item = &'a GgmlExecutionTelemetryCollector>,
    ) -> Option<Self> {
        let mut targets = Vec::<TelemetryTarget>::new();
        for collector in collectors {
            for target in collector.targets.iter() {
                if targets.iter().any(|existing| Arc::ptr_eq(existing, target)) {
                    continue;
                }
                targets.push(Arc::clone(target));
            }
        }
        (!targets.is_empty()).then(|| Self {
            targets: targets.into(),
        })
    }

    pub(crate) fn record_graph_compute(&self, scheduler: bool) {
        for target in self.targets.iter() {
            let mut summary = target
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if scheduler {
                summary.summary.scheduler_graph_computes =
                    summary.summary.scheduler_graph_computes.saturating_add(1);
            } else {
                summary.summary.direct_graph_computes =
                    summary.summary.direct_graph_computes.saturating_add(1);
            }
        }
    }

    pub(crate) fn record_observed_graph(
        &self,
        graph_id: u64,
        by_backend: &BTreeMap<String, (u64, u64)>,
        compute_nodes_by_backend: &BTreeMap<String, u64>,
        fallback_samples: &BTreeMap<String, Vec<GgmlExecutionNodeSample>>,
    ) {
        for target in self.targets.iter() {
            let mut summary = target
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !summary.observed_graph_ids.insert(graph_id) {
                continue;
            }
            let summary = &mut summary.summary;
            for (backend, (nodes, bytes)) in by_backend {
                let node_total = summary
                    .observed_nodes_by_backend
                    .entry(backend.clone())
                    .or_default();
                *node_total = node_total.saturating_add(*nodes);
                let byte_total = summary
                    .observed_node_output_bytes_by_backend
                    .entry(backend.clone())
                    .or_default();
                *byte_total = byte_total.saturating_add(*bytes);
            }
            for (backend, nodes) in compute_nodes_by_backend {
                let total = summary
                    .observed_compute_nodes_by_backend
                    .entry(backend.clone())
                    .or_default();
                *total = total.saturating_add(*nodes);
            }
            for (backend, samples) in fallback_samples {
                let retained = summary
                    .fallback_node_samples_by_backend
                    .entry(backend.clone())
                    .or_default();
                for sample in samples {
                    if retained.len() >= 16 {
                        break;
                    }
                    if !retained.contains(sample) {
                        retained.push(sample.clone());
                    }
                }
            }
        }
    }
}

impl GgmlExecutionPlacementSummary {
    pub fn is_empty(&self) -> bool {
        self.direct_graph_computes == 0
            && self.scheduler_graph_computes == 0
            && self.observed_nodes_by_backend.is_empty()
            && self.observed_compute_nodes_by_backend.is_empty()
            && self.observed_node_output_bytes_by_backend.is_empty()
            && self.fallback_node_samples_by_backend.is_empty()
    }

    fn merge_from(&mut self, other: &Self) {
        self.direct_graph_computes = self
            .direct_graph_computes
            .saturating_add(other.direct_graph_computes);
        self.scheduler_graph_computes = self
            .scheduler_graph_computes
            .saturating_add(other.scheduler_graph_computes);
        for (backend, nodes) in &other.observed_nodes_by_backend {
            let total = self
                .observed_nodes_by_backend
                .entry(backend.clone())
                .or_default();
            *total = total.saturating_add(*nodes);
        }
        for (backend, nodes) in &other.observed_compute_nodes_by_backend {
            let total = self
                .observed_compute_nodes_by_backend
                .entry(backend.clone())
                .or_default();
            *total = total.saturating_add(*nodes);
        }
        for (backend, bytes) in &other.observed_node_output_bytes_by_backend {
            let total = self
                .observed_node_output_bytes_by_backend
                .entry(backend.clone())
                .or_default();
            *total = total.saturating_add(*bytes);
        }
        for (backend, samples) in &other.fallback_node_samples_by_backend {
            let retained = self
                .fallback_node_samples_by_backend
                .entry(backend.clone())
                .or_default();
            for sample in samples {
                if retained.len() >= 16 {
                    break;
                }
                if !retained.contains(sample) {
                    retained.push(sample.clone());
                }
            }
        }
    }
}

thread_local! {
    static CURRENT_EXECUTION_TELEMETRY_COLLECTOR:
        RefCell<Option<GgmlExecutionTelemetryCollector>> = const { RefCell::new(None) };
}

pub(crate) fn current_execution_telemetry_collector() -> Option<GgmlExecutionTelemetryCollector> {
    CURRENT_EXECUTION_TELEMETRY_COLLECTOR.with(|current| current.borrow().clone())
}

pub(crate) fn install_execution_telemetry_collector(
    collector: Option<GgmlExecutionTelemetryCollector>,
) -> GgmlExecutionTelemetryGuard {
    let previous = CURRENT_EXECUTION_TELEMETRY_COLLECTOR.with(|current| current.replace(collector));
    GgmlExecutionTelemetryGuard { previous }
}

/// Restores the prior request collector on drop.
pub struct GgmlExecutionTelemetryGuard {
    previous: Option<GgmlExecutionTelemetryCollector>,
}

impl Drop for GgmlExecutionTelemetryGuard {
    fn drop(&mut self) {
        CURRENT_EXECUTION_TELEMETRY_COLLECTOR.with(|current| current.replace(self.previous.take()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fanout_deduplicates_targets_and_preserves_totals() {
        let collector = GgmlExecutionTelemetryCollector::new();
        let fanout = GgmlExecutionTelemetryCollector::fanout([&collector, &collector]).unwrap();
        fanout.record_graph_compute(false);
        fanout.record_observed_graph(
            1,
            &BTreeMap::from([("MTL0".to_string(), (2, 64))]),
            &BTreeMap::from([("MTL0".to_string(), 2)]),
            &BTreeMap::new(),
        );
        assert_eq!(
            collector.snapshot(),
            GgmlExecutionPlacementSummary {
                direct_graph_computes: 1,
                scheduler_graph_computes: 0,
                observed_nodes_by_backend: BTreeMap::from([("MTL0".to_string(), 2)]),
                observed_compute_nodes_by_backend: BTreeMap::from([("MTL0".to_string(), 2)]),
                observed_node_output_bytes_by_backend: BTreeMap::from([("MTL0".to_string(), 64,)]),
                fallback_node_samples_by_backend: BTreeMap::new(),
            }
        );
    }

    #[test]
    fn graph_observation_is_deduplicated_per_target_but_replayed_to_new_collectors() {
        let first = GgmlExecutionTelemetryCollector::new();
        let second = GgmlExecutionTelemetryCollector::new();
        let by_backend = BTreeMap::from([("MTL0".to_string(), (3, 96))]);
        let compute = BTreeMap::from([("MTL0".to_string(), 2)]);

        first.record_observed_graph(7, &by_backend, &compute, &BTreeMap::new());
        first.record_observed_graph(7, &by_backend, &compute, &BTreeMap::new());
        second.record_observed_graph(7, &by_backend, &compute, &BTreeMap::new());

        assert_eq!(
            first.snapshot().observed_nodes_by_backend,
            BTreeMap::from([("MTL0".to_string(), 3)])
        );
        assert_eq!(first.snapshot(), second.snapshot());
    }

    #[test]
    fn guard_restores_outer_collector() {
        let outer = GgmlExecutionTelemetryCollector::new();
        let inner = GgmlExecutionTelemetryCollector::new();
        let _outer_guard = outer.install();
        {
            let _inner_guard = inner.install();
            assert_eq!(
                current_execution_telemetry_collector().unwrap().snapshot(),
                inner.snapshot()
            );
        }
        assert_eq!(
            current_execution_telemetry_collector().unwrap().snapshot(),
            outer.snapshot()
        );
    }
}
