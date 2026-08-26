//! Request-scoped observation of real ggml graph lifecycle operations.
//!
//! Every identity in this module is minted at the operation boundary that owns
//! it. None is reconstructed from a planner decision, a provider label, or the
//! presence of an optional native pointer. Values are opaque outside the
//! current process and must never be compared across process starts.

use std::{
    cell::RefCell,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

use serde::{Deserialize, Serialize};

pub const GGML_GRAPH_LIFECYCLE_SCHEMA: &str = "openasr.ggml-graph-lifecycle.v1";
/// Bound for one request observation. FreshGraph seq2seq families mint a
/// decoder graph and a logits-head graph per token; unsupported-capture
/// backends still emit one capture observation per graph. Longform multi-slice
/// receipts accumulate those events across every slice of one request, so
/// 262,144 covers an 8-slice 2,048-token FreshGraph decode without dropping
/// the start/complete/readback events a short-audio receipt still binds.
pub const MAX_GRAPH_LIFECYCLE_EVENTS: usize = 262_144;

/// Exact JSON field contract for one serialized lifecycle event. Serde's
/// flattened tagged enum cannot use `deny_unknown_fields`, so every artifact
/// parser must call this before deserializing untrusted JSON.
pub fn ggml_graph_lifecycle_json_shape_is_strict(value: &serde_json::Value) -> bool {
    const COMMON: &[&str] = &[
        "schema",
        "sequence",
        "provider",
        "device",
        "graph_instance",
        "graph_generation",
        "event",
    ];
    if value.get("schema").and_then(serde_json::Value::as_str) != Some(GGML_GRAPH_LIFECYCLE_SCHEMA)
    {
        return false;
    }
    let Some(event) = value.get("event").and_then(serde_json::Value::as_str) else {
        return false;
    };
    let (required, optional): (&[&str], &[&str]) = match event {
        "created" => (&["scheduler_enabled"], &[]),
        "existing_graph_observed" => (&["scheduler_enabled"], &["prepare_generation"]),
        "prepared" => (&["prepare_generation"], &[]),
        "input_write" => (&["input_generation", "bytes"], &[]),
        "compute_started" => (
            &["compute_sequence"],
            &[
                "prepare_generation",
                "input_generation_consumed",
                "capture_executable_generation",
            ],
        ),
        "compute_completed" => (&["compute_sequence", "output_generation"], &[]),
        "output_read" => (
            &["compute_sequence", "output_generation_consumed", "bytes"],
            &[],
        ),
        "kv_write_committed" => (&["compute_sequence", "kv_write_generation"], &[]),
        "rebuilt" => (&["previous_graph_generation", "reason"], &[]),
        "poisoned" => (&["reason"], &[]),
        "dropped" => (&[], &[]),
        "capture_state_observed" => (
            &[
                "phase",
                "capture_supported",
                "graph_tracked",
                "executable_present",
            ],
            &["capture_enabled"],
        ),
        "capture_executable_observed" => (&["capture_executable_generation", "last_change"], &[]),
        "capture_executable_created" => (&["capture_executable_generation", "change"], &[]),
        _ => return false,
    };
    let Some(object) = value.as_object() else {
        return false;
    };
    COMMON
        .iter()
        .chain(required)
        .all(|field| object.contains_key(*field))
        && object.keys().all(|field| {
            COMMON.contains(&field.as_str())
                || required.contains(&field.as_str())
                || optional.contains(&field.as_str())
        })
}

/// Bounded facts read from the live ggml device handle that initialized the
/// runner. `provider_device_id` preserves ggml's spelling (normally a PCI BDF)
/// and must not be reinterpreted as a Vulkan `VkPhysicalDeviceProperties`
/// device id. Missing optional backend facts remain absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GgmlActualDeviceFacts {
    #[serde(rename = "type")]
    pub device_type: String,
    pub name: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_device_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pci_vendor_id: Option<u32>,
}

static NEXT_OPAQUE_ID: AtomicU64 = AtomicU64::new(1);

#[cfg(test)]
thread_local! {
    static TEST_OPAQUE_ID_MINT_COUNT: std::cell::Cell<u64> = const {
        std::cell::Cell::new(0)
    };
}

pub(crate) fn mint_opaque_graph_id() -> u64 {
    #[cfg(test)]
    TEST_OPAQUE_ID_MINT_COUNT.with(|count| count.set(count.get().saturating_add(1)));
    NEXT_OPAQUE_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
pub(crate) fn test_opaque_graph_id_mint_count() -> u64 {
    TEST_OPAQUE_ID_MINT_COUNT.with(std::cell::Cell::get)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GgmlGraphRebuildReason {
    FreshStep,
    TopologyChanged,
    PoisonRecovery,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GgmlGraphPoisonReason {
    ComputeFailed,
    ReadbackFailed,
    MemoryCommitFailed,
    CaptureObservationFailed,
    Explicit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GgmlCaptureExecutableChange {
    Instantiated,
    Updated,
    Replaced,
}

/// Exact side of a measured native graph compute at which capture state was
/// read from the backend ABI.  This is deliberately an observed phase rather
/// than a planner policy: a capture-enabled lane must prove both observations
/// around every successful compute it wants to use as evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GgmlCaptureObservationPhase {
    BeforeCompute,
    AfterCompute,
}

/// Opaque, process-local proof that a concrete graph compute completed and
/// that one of its outputs was successfully read back. The fields stay
/// private so model-family code can carry this value but cannot manufacture
/// one from planner state, a provider label, or a cached graph handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GgmlComputeEvidenceRef {
    graph_instance: u64,
    graph_generation: u64,
    compute_sequence: u64,
    output_generation: u64,
}

impl GgmlComputeEvidenceRef {
    pub(in crate::ggml_runtime) fn new(
        graph_instance: u64,
        graph_generation: u64,
        compute_sequence: u64,
        output_generation: u64,
    ) -> Self {
        Self {
            graph_instance,
            graph_generation,
            compute_sequence,
            output_generation,
        }
    }

    pub(in crate::ggml_runtime) fn selection(
        self,
        output_index: usize,
        output_count: usize,
    ) -> Option<GgmlSelectionEvidenceRef> {
        if output_count == 0 || output_index >= output_count {
            return None;
        }
        Some(GgmlSelectionEvidenceRef {
            graph_instance: self.graph_instance,
            graph_generation: self.graph_generation,
            compute_sequence: self.compute_sequence,
            output_generation: self.output_generation,
            output_index,
            output_count,
        })
    }
}

/// Opaque proof for one logical row of a successfully read native output.
/// Only the ggml readback wrapper can partition a concrete buffer and mint
/// these values. It is serializable for evidence artifacts but deliberately
/// not deserializable back into a runtime witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GgmlSelectionEvidenceRef {
    graph_instance: u64,
    graph_generation: u64,
    compute_sequence: u64,
    output_generation: u64,
    output_index: usize,
    output_count: usize,
}

impl GgmlSelectionEvidenceRef {
    pub(crate) fn compute_identity(&self) -> (u64, u64, u64, u64) {
        (
            self.graph_instance,
            self.graph_generation,
            self.compute_sequence,
            self.output_generation,
        )
    }

    #[cfg(test)]
    pub(crate) fn identity_tuple(&self) -> (u64, u64, u64, u64, usize, usize) {
        (
            self.graph_instance,
            self.graph_generation,
            self.compute_sequence,
            self.output_generation,
            self.output_index,
            self.output_count,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgmlGraphLifecycleEvent {
    pub schema: String,
    pub sequence: u64,
    pub provider: String,
    pub device: String,
    pub graph_instance: u64,
    pub graph_generation: u64,
    #[serde(flatten)]
    pub kind: GgmlGraphLifecycleEventKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum GgmlGraphLifecycleEventKind {
    Created {
        scheduler_enabled: bool,
    },
    /// A request-local collector attached to a persistent graph that was
    /// created and prepared by an earlier request in the same process.
    ExistingGraphObserved {
        scheduler_enabled: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        prepare_generation: Option<u64>,
    },
    Prepared {
        prepare_generation: u64,
    },
    InputWrite {
        input_generation: u64,
        bytes: u64,
    },
    ComputeStarted {
        compute_sequence: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        prepare_generation: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        input_generation_consumed: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        capture_executable_generation: Option<u64>,
    },
    ComputeCompleted {
        compute_sequence: u64,
        output_generation: u64,
    },
    OutputRead {
        compute_sequence: u64,
        output_generation_consumed: u64,
        bytes: u64,
    },
    KvWriteCommitted {
        compute_sequence: u64,
        kv_write_generation: u64,
    },
    Rebuilt {
        previous_graph_generation: u64,
        reason: GgmlGraphRebuildReason,
    },
    Poisoned {
        reason: GgmlGraphPoisonReason,
    },
    Dropped,
    /// Backend-native graph tracking, capture support, and enablement observed
    /// immediately before or after a real graph compute. This is emitted from
    /// the optional backend ABI, never inferred from a build option, provider
    /// label, or planner policy.
    CaptureStateObserved {
        phase: GgmlCaptureObservationPhase,
        capture_supported: bool,
        graph_tracked: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        capture_enabled: Option<bool>,
        executable_present: bool,
    },
    /// An executable already existed before the measured compute. The event
    /// binds the upcoming compute to that native generation without claiming
    /// that the current Rust graph lifecycle created it.
    CaptureExecutableObserved {
        capture_executable_generation: u64,
        last_change: GgmlCaptureExecutableChange,
    },
    /// Emitted only when before/after observations from the backend API prove
    /// that the measured compute advanced the native executable generation.
    /// The Rust graph layer never emits it from a provider label or build
    /// option.
    CaptureExecutableCreated {
        capture_executable_generation: u64,
        change: GgmlCaptureExecutableChange,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GgmlGraphLifecycleSnapshot {
    pub events: Vec<GgmlGraphLifecycleEvent>,
    pub overflowed: bool,
}

#[derive(Debug, Default)]
struct LifecycleState {
    events: Vec<GgmlGraphLifecycleEvent>,
    overflowed: bool,
    next_sequence: u64,
    observation_scope: u64,
    observation_scope_open: bool,
    /// Concurrent longform slices share one request collector. Begin/end must
    /// be refcounted so a sibling finishing its candidate cannot close the
    /// scope and drop the other worker's create/compute/read events.
    observation_scope_refs: u32,
}

#[derive(Debug, Clone)]
pub struct GgmlGraphLifecycleCollector {
    state: Arc<Mutex<LifecycleState>>,
}

impl Default for GgmlGraphLifecycleCollector {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(LifecycleState {
                observation_scope_open: true,
                ..LifecycleState::default()
            })),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GgmlGraphLifecycleGeneration {
    collector_state: Weak<Mutex<LifecycleState>>,
    observation_scope: Option<u64>,
    graph_generation: u64,
}

impl GgmlGraphLifecycleGeneration {
    pub(crate) fn generation_for(&self, collector: &GgmlGraphLifecycleCollector) -> Option<u64> {
        let current = Arc::downgrade(&collector.state);
        (self.collector_state.strong_count() > 0
            && Weak::ptr_eq(&self.collector_state, &current)
            && self.observation_scope.is_some()
            && collector.observation_scope() == self.observation_scope)
            .then_some(self.graph_generation)
    }
}

impl GgmlGraphLifecycleCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn install(&self) -> GgmlGraphLifecycleGuard {
        install_graph_lifecycle_collector(Some(self.clone()))
    }

    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }

    /// Start one transactional request/candidate observation scope. A graph
    /// may survive a failed candidate or a later request while this collector
    /// remains allocated, so pointer identity alone cannot decide whether its
    /// attachment and native capture facts have already been emitted.
    ///
    /// Concurrent candidate attempts keep the collector open until the last
    /// matching end. Each begin still mints a scope id so a later sequential
    /// candidate can re-observe native capture; ending one sibling must not
    /// drop the others' create/compute/read events.
    pub(crate) fn begin_observation_scope(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.observation_scope = mint_opaque_graph_id();
        state.observation_scope_refs = state.observation_scope_refs.saturating_add(1);
        state.observation_scope_open = true;
    }

    pub(crate) fn end_observation_scope(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.observation_scope_refs = state.observation_scope_refs.saturating_sub(1);
        if state.observation_scope_refs == 0 {
            state.observation_scope_open = false;
        }
    }

    pub(crate) fn observation_scope(&self) -> Option<u64> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state
            .observation_scope_open
            .then_some(state.observation_scope)
    }

    pub fn snapshot(&self) -> GgmlGraphLifecycleSnapshot {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        GgmlGraphLifecycleSnapshot {
            events: state.events.clone(),
            overflowed: state.overflowed,
        }
    }

    pub(crate) fn observed_generation(
        &self,
        graph_generation: u64,
    ) -> GgmlGraphLifecycleGeneration {
        GgmlGraphLifecycleGeneration {
            collector_state: Arc::downgrade(&self.state),
            observation_scope: self.observation_scope(),
            graph_generation,
        }
    }

    pub(crate) fn checkpoint(&self) -> (usize, bool) {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (state.events.len(), state.overflowed)
    }

    pub(crate) fn truncate(&self, checkpoint: (usize, bool)) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.events.truncate(checkpoint.0);
        state.overflowed = checkpoint.1;
    }

    pub(crate) fn record(
        &self,
        provider: &str,
        device: &str,
        graph_instance: u64,
        graph_generation: u64,
        kind: GgmlGraphLifecycleEventKind,
    ) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !state.observation_scope_open {
            return;
        }
        if state.events.len() >= MAX_GRAPH_LIFECYCLE_EVENTS {
            state.overflowed = true;
            return;
        }
        state.next_sequence = state.next_sequence.saturating_add(1);
        let sequence = state.next_sequence;
        state.events.push(GgmlGraphLifecycleEvent {
            schema: GGML_GRAPH_LIFECYCLE_SCHEMA.to_string(),
            sequence,
            provider: provider.to_string(),
            device: device.to_string(),
            graph_instance,
            graph_generation,
            kind,
        });
    }
}

thread_local! {
    static CURRENT_GRAPH_LIFECYCLE_COLLECTOR:
        RefCell<Option<GgmlGraphLifecycleCollector>> = const { RefCell::new(None) };
}

pub(crate) fn current_graph_lifecycle_collector() -> Option<GgmlGraphLifecycleCollector> {
    CURRENT_GRAPH_LIFECYCLE_COLLECTOR.with(|current| current.borrow().clone())
}

pub(crate) fn install_graph_lifecycle_collector(
    collector: Option<GgmlGraphLifecycleCollector>,
) -> GgmlGraphLifecycleGuard {
    let previous = CURRENT_GRAPH_LIFECYCLE_COLLECTOR.with(|current| current.replace(collector));
    GgmlGraphLifecycleGuard { previous }
}

pub struct GgmlGraphLifecycleGuard {
    previous: Option<GgmlGraphLifecycleCollector>,
}

impl Drop for GgmlGraphLifecycleGuard {
    fn drop(&mut self) {
        CURRENT_GRAPH_LIFECYCLE_COLLECTOR.with(|current| {
            *current.borrow_mut() = self.previous.take();
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NN_DECODER: &str = include_str!("../nn/decoder.rs");
    const COHERE_DECODER: &str = include_str!("../models/cohere/decoder_graph.rs");
    const FIRERED_DECODER: &str = include_str!("../models/firered_aed/decoder_graph.rs");
    const GRANITE_DECODER: &str = include_str!("../models/granite_speech/decode_session.rs");
    const MOONSHINE_DECODER: &str = include_str!("../models/moonshine/decoder_graph.rs");
    const WHISPER_DECODER: &str = include_str!("../models/whisper/ggml_decoder_graph.rs");
    const XASR_ENCODER: &str = include_str!("../models/xasr_zipformer/encoder_graph.rs");
    const XASR_HEAD: &str = include_str!("../models/xasr_zipformer/device_head_graph.rs");
    const PARAKEET_TDT_DECODER: &str =
        include_str!("../models/parakeet_tdt/device_decoder_graph.rs");

    #[test]
    fn rollback_discards_failed_attempt_events_without_reusing_sequence() {
        let collector = GgmlGraphLifecycleCollector::new();
        let checkpoint = collector.checkpoint();
        collector.record(
            "cpu",
            "CPU",
            1,
            2,
            GgmlGraphLifecycleEventKind::Created {
                scheduler_enabled: false,
            },
        );
        collector.truncate(checkpoint);
        collector.record(
            "cpu",
            "CPU",
            3,
            4,
            GgmlGraphLifecycleEventKind::Created {
                scheduler_enabled: false,
            },
        );
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.events.len(), 1);
        assert_eq!(snapshot.events[0].sequence, 2);
        assert_eq!(snapshot.events[0].graph_instance, 3);
    }

    #[test]
    fn graph_generation_witness_is_valid_only_inside_its_open_observation_scope() {
        let collector = GgmlGraphLifecycleCollector::new();
        let generation = collector.observed_generation(7);
        assert_eq!(generation.generation_for(&collector), Some(7));

        collector.begin_observation_scope();
        assert_eq!(generation.generation_for(&collector), None);
        let next_generation = collector.observed_generation(8);
        assert_eq!(next_generation.generation_for(&collector), Some(8));

        collector.end_observation_scope();
        assert_eq!(next_generation.generation_for(&collector), None);
    }

    #[test]
    fn concurrent_observation_scopes_stay_open_until_last_end() {
        let collector = GgmlGraphLifecycleCollector::new();
        collector.begin_observation_scope();
        collector.begin_observation_scope();
        collector.end_observation_scope();
        collector.record(
            "hip",
            "ROCm0",
            1,
            2,
            GgmlGraphLifecycleEventKind::Created {
                scheduler_enabled: false,
            },
        );
        assert_eq!(collector.snapshot().events.len(), 1);
        collector.end_observation_scope();
        collector.record("hip", "ROCm0", 1, 2, GgmlGraphLifecycleEventKind::Dropped);
        assert_eq!(collector.snapshot().events.len(), 1);
    }

    #[test]
    fn closed_observation_scope_discards_late_events_without_consuming_sequence() {
        let collector = GgmlGraphLifecycleCollector::new();
        collector.begin_observation_scope();
        collector.record(
            "hip",
            "ROCm0",
            1,
            2,
            GgmlGraphLifecycleEventKind::Created {
                scheduler_enabled: false,
            },
        );
        collector.end_observation_scope();
        collector.record("hip", "ROCm0", 1, 2, GgmlGraphLifecycleEventKind::Dropped);
        collector.begin_observation_scope();
        collector.record(
            "hip",
            "ROCm0",
            3,
            4,
            GgmlGraphLifecycleEventKind::Created {
                scheduler_enabled: false,
            },
        );

        let events = collector.snapshot().events;
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].sequence, 1);
        assert_eq!(events[1].sequence, 2);
        assert!(
            events
                .iter()
                .all(|event| !matches!(event.kind, GgmlGraphLifecycleEventKind::Dropped))
        );
    }

    #[test]
    fn native_capture_events_serialize_observation_phase_without_policy_inference() {
        let collector = GgmlGraphLifecycleCollector::new();
        collector.record(
            "hip",
            "ROCm0",
            1,
            2,
            GgmlGraphLifecycleEventKind::CaptureStateObserved {
                phase: GgmlCaptureObservationPhase::BeforeCompute,
                capture_supported: true,
                graph_tracked: false,
                capture_enabled: None,
                executable_present: false,
            },
        );
        collector.record(
            "hip",
            "ROCm0",
            1,
            2,
            GgmlGraphLifecycleEventKind::CaptureStateObserved {
                phase: GgmlCaptureObservationPhase::AfterCompute,
                capture_supported: true,
                graph_tracked: true,
                capture_enabled: Some(true),
                executable_present: true,
            },
        );
        collector.record(
            "hip",
            "ROCm0",
            1,
            2,
            GgmlGraphLifecycleEventKind::CaptureExecutableObserved {
                capture_executable_generation: 7,
                last_change: GgmlCaptureExecutableChange::Instantiated,
            },
        );
        collector.record(
            "hip",
            "ROCm0",
            1,
            2,
            GgmlGraphLifecycleEventKind::CaptureExecutableCreated {
                capture_executable_generation: 8,
                change: GgmlCaptureExecutableChange::Updated,
            },
        );

        let values = collector
            .snapshot()
            .events
            .iter()
            .map(|event| serde_json::to_value(event).expect("serialize lifecycle event"))
            .collect::<Vec<_>>();
        assert_eq!(values[0]["event"], "capture_state_observed");
        assert_eq!(values[0]["phase"], "before_compute");
        assert_eq!(values[0]["graph_tracked"], false);
        assert!(values[0].get("capture_enabled").is_none());
        assert_eq!(values[1]["event"], "capture_state_observed");
        assert_eq!(values[1]["phase"], "after_compute");
        assert_eq!(values[1]["capture_enabled"], true);
        assert_eq!(values[2]["event"], "capture_executable_observed");
        assert_eq!(values[2]["last_change"], "instantiated");
        assert!(values.iter().all(ggml_graph_lifecycle_json_shape_is_strict));
        let mut unknown = values[0].clone();
        unknown
            .as_object_mut()
            .expect("lifecycle event object")
            .insert("activation_mode".to_string(), serde_json::json!("auto"));
        assert!(!ggml_graph_lifecycle_json_shape_is_strict(&unknown));
        assert_eq!(values[3]["event"], "capture_executable_created");
        assert_eq!(values[3]["change"], "updated");
    }

    #[test]
    fn production_kv_writes_use_typed_lifecycle_registration() {
        let sources = [
            ("nn decoder", NN_DECODER),
            ("cohere decoder", COHERE_DECODER),
            ("firered decoder", FIRERED_DECODER),
            ("granite decoder", GRANITE_DECODER),
            ("moonshine decoder", MOONSHINE_DECODER),
            ("whisper decoder", WHISPER_DECODER),
            ("xasr encoder", XASR_ENCODER),
        ];
        let compact = sources
            .iter()
            .map(|(label, source)| (*label, source.split_whitespace().collect::<String>()))
            .collect::<Vec<_>>();
        for (label, source) in &compact {
            for forbidden in [
                "add_side_effect_root(write_key)",
                "add_side_effect_root(write_value)",
                "add_side_effect_root(k_write)",
                "add_side_effect_root(v_write)",
                "add_side_effect_root(key_write)",
                "add_side_effect_root(value_write)",
                "add_side_effect_root(k_seed)",
                "add_side_effect_root(v_seed)",
            ] {
                assert!(
                    !source.contains(forbidden),
                    "{label} routes a named KV write through the generic side-effect API: {forbidden}"
                );
            }
        }

        let required = [
            ("nn decoder", "add_kv_write_root(write)"),
            ("nn decoder", "set_kv_rows(self_kv.key"),
            ("nn decoder", "set_kv_rows(kv.key_history"),
            ("cohere decoder", "add_kv_write_root(write_key)"),
            ("firered decoder", "add_kv_write_root(write_key)"),
            ("granite decoder", "set_kv_rows(arena_k"),
            ("granite decoder", "add_kv_write_root(k_seed)"),
            ("moonshine decoder", "add_kv_write_root(write_key)"),
            ("moonshine decoder", "set_kv_rows(self_k_cache"),
            ("whisper decoder", "set_kv_rows(k_layer"),
            ("whisper decoder", "add_kv_write_root(k_write)"),
            ("whisper decoder", "add_kv_write_root(key_write)"),
            (
                "xasr encoder",
                "resident_kv_cache_side_effect\",graph.add_kv_write_root(write)",
            ),
        ];
        for (label, pattern) in required {
            let source = compact
                .iter()
                .find_map(|(candidate, source)| (*candidate == label).then_some(source))
                .expect("audited source label exists");
            assert!(
                source.contains(pattern),
                "{label} is missing typed KV lifecycle registration: {pattern}"
            );
        }

        for (label, source) in [
            ("xasr device head", XASR_HEAD),
            ("parakeet TDT recurrent decoder", PARAKEET_TDT_DECODER),
        ] {
            assert!(
                !source.contains("add_kv_write_root") && !source.contains("set_kv_rows"),
                "{label} ordinary persistent state must not be mislabeled as KV"
            );
        }
    }
}
