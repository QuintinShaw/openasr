//! Machine-readable short-audio audit receipt (`openasr.short-audio-receipt.v0`).
//!
//! Binds the exact core commit, pack bytes, audio fixture, backend/device/OS,
//! command, warmup/cache state, transcript, and optional RTF samples so a
//! later full WER/CER claim can be compared against a frozen short-audio gate.
//!
//! This is data-only evidence for tooling. It is not an execution capability
//! and does not replace [`crate::ModelPackPreflightReceipt`] (pack install
//! sealing).

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::Path,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::RequestAttemptId;
use crate::ggml_runtime::{
    GgmlActualDeviceFacts, GgmlExecutionPlacementSummary, GgmlGraphLifecycleSnapshot,
    ResolvedFamilyRuntimeInput,
};
use crate::models::request_execution_receipt::{
    NativeExecutionReceiptSnapshot, NativeExecutionTokenStep,
};
use crate::models::runtime_receipts::{
    LeaseReceiptShadow, LeaseReceiptShadowIncomparable, RuntimeOwnerPlacement, RuntimeReceiptEvent,
    RuntimeReceiptSnapshot, SafeExecutionLaneProjection, SafeMemoryDomainKind,
    SafeMemoryDomainProjection,
};

pub use crate::ggml_runtime::{
    DecodeFirstDivergenceClass, EncoderDecoderSplitLane, EncoderDecoderSplitProbeRecord,
    SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS, ShortAudioReceiptDecodeDiagnostics,
    ShortAudioReceiptDecodeStep, ShortAudioReceiptOutputPlan, ShortAudioReceiptReuseMode,
};

/// Stable schema id for the short-audio receipt MVP.
pub const SHORT_AUDIO_RECEIPT_SCHEMA: &str = "openasr.short-audio-receipt.v0";

/// Default product scope for the short-audio quality/perf gate.
pub const SHORT_AUDIO_RECEIPT_DEFAULT_SCOPE: &str = "short-audio-gate";

/// How wall time was converted into RTF samples.
pub const SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK: &str = "wall_clock_process_elapsed";

/// Validation failures for a receipt document or its required bindings.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ShortAudioReceiptError {
    #[error("short-audio receipt schema must be {expected}, got {actual:?}")]
    SchemaMismatch {
        expected: &'static str,
        actual: String,
    },
    #[error("short-audio receipt field `{field}` must be non-empty")]
    EmptyField { field: &'static str },
    #[error(
        "short-audio receipt pack.content_sha256 must be 64 lowercase hex chars, got {actual:?}"
    )]
    InvalidContentSha256 { actual: String },
    #[error("short-audio receipt audio.sha256 must be 64 lowercase hex chars, got {actual:?}")]
    InvalidAudioSha256 { actual: String },
    #[error(
        "short-audio receipt transcript.text_sha256 must be 64 lowercase hex chars, got {actual:?}"
    )]
    InvalidTranscriptSha256 { actual: String },
    #[error(
        "short-audio receipt correctness evidence schema must be openasr.short-audio-receipt.evidence.v1, got {actual:?}"
    )]
    EvidenceSchemaMismatch { actual: String },
    #[error("short-audio receipt correctness field `{field}` is invalid: {actual:?}")]
    InvalidEvidenceField { field: &'static str, actual: String },
    #[error(
        "short-audio receipt correctness digest `{field}` must be 64 lowercase hex chars, got {actual:?}"
    )]
    InvalidEvidenceDigest { field: &'static str, actual: String },
    #[error(
        "short-audio receipt correctness evidence class `{evidence_class}` did not pass: {result}"
    )]
    EvidenceNotPassing {
        evidence_class: String,
        result: String,
    },
    #[error("short-audio receipt placement evidence requires observed_placement")]
    PlacementEvidenceMissing,
    #[error("short-audio receipt token-transcript evidence is incomplete")]
    TokenEvidenceIncomplete,
    #[error("short-audio receipt token top-k or margin summary is invalid")]
    InvalidTopKSummary,
    #[error("short-audio receipt output plan and family oracle do not match")]
    OutputPlanOracleMismatch,
    #[error("short-audio receipt evidence binding does not match the outer receipt")]
    EvidenceBindingMismatch,
    #[error("short-audio receipt core_commit must be a 40-hex git sha, got {actual:?}")]
    InvalidCoreCommit { actual: String },
    #[error("short-audio receipt rtf_median requires non-empty rtf_samples when present")]
    MedianWithoutSamples,
    #[error("short-audio receipt rtf_median {median} does not match samples median {expected}")]
    MedianMismatch { median: String, expected: String },
    #[error("short-audio receipt metric `{field}` must be finite and non-negative, got {actual}")]
    InvalidMetric { field: &'static str, actual: String },
    #[error(
        "short-audio receipt RTF samples require measurement_method={expected}, got {actual:?}"
    )]
    InvalidMeasurementMethod {
        expected: &'static str,
        actual: Option<String>,
    },
    #[error("could not hash path {path}: {reason}")]
    HashIo { path: String, reason: String },
    #[error(
        "short-audio receipt decode_diagnostics is required and must bind output_plan and reuse_mode"
    )]
    DecodeDiagnosticsMissing,
    #[error("short-audio receipt native seq2seq decode produced no token steps")]
    NativeSeq2SeqTokenStepsMissing,
    #[error("short-audio receipt decode diagnostics exceed {max} steps, got {actual}")]
    DecodeStepsUnbounded { max: usize, actual: usize },
    #[error(
        "short-audio receipt decode diagnostics exceed {max} encoder/decoder splits, got {actual}"
    )]
    EncoderDecoderSplitsUnbounded { max: usize, actual: usize },
    #[error(
        "short-audio receipt decode diagnostics field `{field}` must be 64 lowercase hex chars, got {actual:?}"
    )]
    InvalidDiagnosticSha256 { field: &'static str, actual: String },
    #[error("short-audio receipt execution projection is internally inconsistent: {reason}")]
    InvalidExecutionProjection { reason: &'static str },
    #[error("short-audio receipt privacy-safe field `{field}` is invalid: {actual:?}")]
    InvalidPrivacyProjection { field: &'static str, actual: String },
    #[error("short-audio receipt is not qualification-eligible: {reason}")]
    QualificationIneligible { reason: &'static str },
    #[error("short-audio formal correctness evidence cannot contain free-form notes")]
    FormalEvidenceNotesNotAllowed,
    #[error("short-audio real-family evidence is incomplete: {reason}")]
    RealFamilyEvidenceIncomplete { reason: &'static str },
}

/// Top-level short-audio receipt document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioReceipt {
    pub schema: String,
    /// 40-hex git commit of the openasr core that produced the transcript.
    pub core_commit: String,
    pub pack: ShortAudioReceiptPack,
    pub audio: ShortAudioReceiptAudio,
    pub run: ShortAudioReceiptRun,
    pub metrics: ShortAudioReceiptMetrics,
    pub transcript: ShortAudioReceiptTranscript,
    /// Reported weight/compute placement label (requested device in v0 when
    /// runtime placement is not introspected).
    pub placement: String,
    /// Actual graph-node placement observed at compute time. Older receipts
    /// and non-ggml backends may omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_placement: Option<GgmlExecutionPlacementSummary>,
    /// Real graph lifecycle operations observed by the shared ggml runtime.
    /// IDs are opaque within this producer process and are not stable receipt
    /// identities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_lifecycle: Option<GgmlGraphLifecycleSnapshot>,
    /// Final live backend identity. These three fields are all-or-none and are
    /// never reconstructed from `run.device` or the requested placement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_stable_device_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_device: Option<GgmlActualDeviceFacts>,
    /// Optional versioned evidence. Its class determines which release gate it
    /// can satisfy; old receipts omit this field and remain readable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<ShortAudioReceiptEvidence>,
    /// Optional projection of the existing request/runtime receipt authorities.
    /// Older v0 documents omit it and remain readable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<ShortAudioExecutionProjection>,
    /// Gate scope, typically [`SHORT_AUDIO_RECEIPT_DEFAULT_SCOPE`].
    pub scope: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    /// Fail-closed decode-correctness diagnostics. Dual-output or four-quadrant
    /// agreement recorded here is not production compact-path authorization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_diagnostics: Option<ShortAudioReceiptDecodeDiagnostics>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioExecutionProjection {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_attempt_id: Option<RequestAttemptId>,
    pub request_attempt_conflicted: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidate_attempt_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lanes: Vec<ShortAudioExecutionLane>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memory_domains: Vec<ShortAudioExecutionDomain>,
    pub live_lease_reconciliation: ShortAudioLeaseReconciliation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reconciliation_reason: Option<String>,
    pub live_state_complete: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub live_state_reason: Option<String>,
    pub event_history_complete: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_history_reason: Option<String>,
    pub dropped_events: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub phase_duration_micros: BTreeMap<String, u64>,
    pub timing_complete: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<String>,
    pub request_receipt_complete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioExecutionLane {
    pub provider: String,
    pub placement: String,
    pub backend: String,
    pub device: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioExecutionDomain {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heap: Option<u32>,
    pub join_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShortAudioLeaseReconciliation {
    Matched,
    Mismatch,
    Incomparable,
}

impl ShortAudioExecutionProjection {
    pub fn from_receipts(
        request: &NativeExecutionReceiptSnapshot,
        runtime: &RuntimeReceiptSnapshot,
        reconciliation: &LeaseReceiptShadow,
    ) -> Self {
        let expected_request_attempt = request.request_attempt_id;
        let mut candidate_attempt_ids = BTreeSet::new();
        let mut owner_ids = BTreeSet::new();
        let mut lanes = BTreeSet::new();
        let mut memory_domains = BTreeSet::new();

        for event in &runtime.events {
            let (event_request_attempt, candidate_attempt, owner_id) = match event {
                RuntimeReceiptEvent::OwnerCreated {
                    owner_id,
                    descriptor,
                    attempt_id,
                    request_attempt_id,
                } => {
                    if *request_attempt_id == expected_request_attempt
                        && let RuntimeOwnerPlacement::LaneBound(lane) = descriptor.placement
                    {
                        lanes.insert(execution_lane_projection(lane));
                    }
                    (*request_attempt_id, *attempt_id, *owner_id)
                }
                RuntimeReceiptEvent::OwnerReused {
                    owner_id,
                    attempt_id,
                    request_attempt_id,
                }
                | RuntimeReceiptEvent::OwnerReleased {
                    owner_id,
                    attempt_id,
                    request_attempt_id,
                }
                | RuntimeReceiptEvent::ResourceReleased {
                    owner_id,
                    attempt_id,
                    request_attempt_id,
                    ..
                } => (*request_attempt_id, *attempt_id, *owner_id),
                RuntimeReceiptEvent::ResourceAcquired {
                    owner_id,
                    descriptor,
                    attempt_id,
                    request_attempt_id,
                    ..
                }
                | RuntimeReceiptEvent::ResourceStateChanged {
                    owner_id,
                    descriptor,
                    attempt_id,
                    request_attempt_id,
                    ..
                } => {
                    if *request_attempt_id == expected_request_attempt
                        && let Some(domain) = descriptor.domain
                    {
                        memory_domains.insert(execution_domain_projection(domain));
                    }
                    (*request_attempt_id, *attempt_id, *owner_id)
                }
            };
            if event_request_attempt != expected_request_attempt {
                continue;
            }
            owner_ids.insert(owner_id);
            if let Some(attempt) = candidate_attempt {
                candidate_attempt_ids.insert(format!("attempt-{}", attempt.ordinal()));
            }
        }

        for owner in runtime
            .live_owners
            .iter()
            .filter(|owner| owner_ids.contains(&owner.id))
        {
            if let RuntimeOwnerPlacement::LaneBound(lane) = owner.descriptor.placement {
                lanes.insert(execution_lane_projection(lane));
            }
            for resource in owner.resources.values() {
                if let Some(domain) = resource.descriptor.domain {
                    memory_domains.insert(execution_domain_projection(domain));
                }
            }
        }

        let (live_lease_reconciliation, reconciliation_reason) =
            short_audio_reconciliation_projection(reconciliation);
        Self {
            request_attempt_id: expected_request_attempt,
            request_attempt_conflicted: request.request_attempt_conflicted,
            candidate_attempt_ids: candidate_attempt_ids.into_iter().collect(),
            lanes: lanes.into_iter().collect(),
            memory_domains: memory_domains.into_iter().collect(),
            live_lease_reconciliation,
            reconciliation_reason,
            live_state_complete: runtime.completeness.live_state_complete,
            live_state_reason: runtime
                .completeness
                .live_state_reason
                .map(|reason| reason.as_str().to_string()),
            event_history_complete: runtime.completeness.event_history_complete,
            event_history_reason: runtime
                .completeness
                .event_history_reason
                .map(|reason| reason.as_str().to_string()),
            dropped_events: runtime.completeness.dropped_events,
            phase_duration_micros: request
                .phase_duration_micros
                .iter()
                .map(|(phase, duration)| (phase.as_str().to_string(), *duration))
                .collect(),
            timing_complete: request.timing_complete,
            terminal: request
                .terminal
                .map(|terminal| terminal.as_str().to_string()),
            request_receipt_complete: request.completed
                && !request.request_attempt_conflicted
                && !request.timeline_conflicted,
        }
    }
}

fn execution_lane_projection(lane: SafeExecutionLaneProjection) -> ShortAudioExecutionLane {
    ShortAudioExecutionLane {
        provider: lane.provider.as_str().to_string(),
        placement: lane.placement.as_str().to_string(),
        backend: lane.backend.as_str().to_string(),
        device: lane.device.to_hex(),
    }
}

fn execution_domain_projection(domain: SafeMemoryDomainProjection) -> ShortAudioExecutionDomain {
    ShortAudioExecutionDomain {
        kind: match domain.kind {
            SafeMemoryDomainKind::SystemMemory => "system-memory",
            SafeMemoryDomainKind::DedicatedDevice => "dedicated-device",
        }
        .to_string(),
        heap: domain.heap,
        join_id: domain.join_id.to_hex(),
    }
}

fn short_audio_reconciliation_projection(
    reconciliation: &LeaseReceiptShadow,
) -> (ShortAudioLeaseReconciliation, Option<String>) {
    match reconciliation {
        LeaseReceiptShadow::Matched => (ShortAudioLeaseReconciliation::Matched, None),
        LeaseReceiptShadow::Mismatch(_) => (
            ShortAudioLeaseReconciliation::Mismatch,
            Some("lane-domain-byte-mismatch".to_string()),
        ),
        LeaseReceiptShadow::Incomparable { reason } => (
            ShortAudioLeaseReconciliation::Incomparable,
            Some(
                match reason {
                    LeaseReceiptShadowIncomparable::ReceiptsUnavailable => "receipts-unavailable",
                    LeaseReceiptShadowIncomparable::ReceiptsIncomplete(reason) => reason.as_str(),
                    LeaseReceiptShadowIncomparable::UnpricedLiveResource => {
                        "unpriced-live-resource"
                    }
                    LeaseReceiptShadowIncomparable::OwnerPlacementUnknown => {
                        "owner-placement-unknown"
                    }
                    LeaseReceiptShadowIncomparable::ResourcePlacementUnknown => {
                        "resource-placement-unknown"
                    }
                    LeaseReceiptShadowIncomparable::ResourceOwnerPlacementMismatch => {
                        "resource-owner-placement-mismatch"
                    }
                    LeaseReceiptShadowIncomparable::LedgerPlacementUnknown => {
                        "ledger-placement-unknown"
                    }
                    LeaseReceiptShadowIncomparable::InvalidLiveLifecycle => {
                        "invalid-live-lifecycle"
                    }
                    LeaseReceiptShadowIncomparable::SnapshotChanged => "snapshot-changed",
                }
                .to_string(),
            ),
        ),
    }
}

/// Pack identity bound into the receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioReceiptPack {
    pub model_id: String,
    /// Lowercase hex sha256 of the exact pack bytes (no `sha256:` prefix).
    pub content_sha256: String,
    pub size_bytes: u64,
    pub quant: String,
}

/// Audio fixture bound into the receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioReceiptAudio {
    pub path_or_label: String,
    /// Lowercase hex sha256 of the exact audio file bytes.
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_s: Option<f64>,
}

/// Run environment and command binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioReceiptRun {
    /// CLI backend kind (`native` / `mock`).
    pub backend: String,
    /// Device label requested for the run (`cpu` / `metal` / `cuda` / `auto` / ...).
    pub device: String,
    /// Host OS id: `darwin`, `linux`, or `windows`.
    pub os: String,
    /// Effective command argv that produced the receipt (tooling-facing).
    pub command: Vec<String>,
    /// Small allowlisted environment snapshot (never a full env dump).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env_allowlist: BTreeMap<String, String>,
    /// `cold` or `warm` relative to process / model-cache state.
    pub warmup: String,
    /// `empty` or `populated` cache state at the timed runs.
    pub cache_state: String,
}

/// Optional metrics. Empty RTF lists are valid for transcript-only receipts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioReceiptMetrics {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wer_or_cer: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rtf_samples: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rtf_median: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_rss_bytes: Option<u64>,
    /// Process peak RSS captured immediately before the first model run. The
    /// difference to `peak_rss_bytes` isolates additional high-water created
    /// by model execution from CLI/audio preparation startup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_rss_before_model_bytes: Option<u64>,
    /// Process RSS after audio preparation but before the first model run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rss_before_model_bytes: Option<u64>,
    /// Process RSS after all warmup and measured runs, while resident runtime
    /// caches still reflect the product's warm state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rss_after_model_bytes: Option<u64>,
    /// Darwin process physical footprint before the first model run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phys_footprint_before_model_bytes: Option<u64>,
    /// Darwin process physical footprint after all model runs while warm
    /// runtimes remain resident.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phys_footprint_after_model_bytes: Option<u64>,
    /// Darwin lifetime maximum physical footprint before the first model run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_phys_footprint_before_model_bytes: Option<u64>,
    /// Darwin lifetime maximum physical footprint after all model runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_phys_footprint_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_vram_bytes: Option<u64>,
    /// How RTF was measured. v0 uses wall-clock process elapsed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurement_method: Option<String>,
}

impl ShortAudioReceiptMetrics {
    fn validate(&self) -> Result<(), ShortAudioReceiptError> {
        for sample in &self.rtf_samples {
            validate_finite_non_negative_metric("metrics.rtf_samples", *sample)?;
        }
        for (field, value) in [
            ("metrics.rtf_median", self.rtf_median),
            ("metrics.wer_or_cer", self.wer_or_cer),
            ("metrics.ttft_s", self.ttft_s),
        ] {
            if let Some(value) = value {
                validate_finite_non_negative_metric(field, value)?;
            }
        }
        if self
            .measurement_method
            .as_deref()
            .is_some_and(|method| method != SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK)
            || (!self.rtf_samples.is_empty() && self.measurement_method.is_none())
        {
            return Err(ShortAudioReceiptError::InvalidMeasurementMethod {
                expected: SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK,
                actual: self.measurement_method.clone(),
            });
        }
        Ok(())
    }
}

/// Transcript payload and content hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioReceiptTranscript {
    pub text: String,
    /// Lowercase hex sha256 of the UTF-8 transcript bytes.
    pub text_sha256: String,
}

impl ShortAudioReceipt {
    /// Build a receipt and validate required bindings.
    pub fn try_new(mut receipt: ShortAudioReceipt) -> Result<Self, ShortAudioReceiptError> {
        if receipt.metrics.measurement_method.is_none() && !receipt.metrics.rtf_samples.is_empty() {
            receipt.metrics.measurement_method =
                Some(SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK.to_string());
        }
        if receipt.metrics.rtf_median.is_none() && !receipt.metrics.rtf_samples.is_empty() {
            receipt.metrics.rtf_median = median_f64(&receipt.metrics.rtf_samples);
        }
        receipt.validate()?;
        Ok(receipt)
    }

    /// Fail-closed field checks for tooling that loads a receipt from disk.
    pub fn validate(&self) -> Result<(), ShortAudioReceiptError> {
        self.validate_legacy_compatible()?;
        self.validate_privacy_safe_projection()
    }

    /// Common structural validation retained for historical v0 input. Legacy
    /// documents may contain paths and OPENASR_HOME, but no caller may turn
    /// them back into a newly serialized or qualification-eligible receipt.
    fn validate_legacy_compatible(&self) -> Result<(), ShortAudioReceiptError> {
        if self.schema != SHORT_AUDIO_RECEIPT_SCHEMA {
            return Err(ShortAudioReceiptError::SchemaMismatch {
                expected: SHORT_AUDIO_RECEIPT_SCHEMA,
                actual: self.schema.clone(),
            });
        }
        require_non_empty("core_commit", &self.core_commit)?;
        validate_core_commit(&self.core_commit)?;
        require_non_empty("pack.model_id", &self.pack.model_id)?;
        require_non_empty("pack.quant", &self.pack.quant)?;
        validate_sha256_hex("pack.content_sha256", &self.pack.content_sha256)
            .map_err(|actual| ShortAudioReceiptError::InvalidContentSha256 { actual })?;
        require_non_empty("audio.path_or_label", &self.audio.path_or_label)?;
        validate_sha256_hex("audio.sha256", &self.audio.sha256)
            .map_err(|actual| ShortAudioReceiptError::InvalidAudioSha256 { actual })?;
        if let Some(duration_s) = self.audio.duration_s {
            validate_finite_non_negative_metric("audio.duration_s", duration_s)?;
        }
        require_non_empty("run.backend", &self.run.backend)?;
        require_non_empty("run.device", &self.run.device)?;
        require_non_empty("run.os", &self.run.os)?;
        require_non_empty("run.warmup", &self.run.warmup)?;
        require_non_empty("run.cache_state", &self.run.cache_state)?;
        if self.run.command.is_empty() || self.run.command.iter().any(|part| part.trim().is_empty())
        {
            return Err(ShortAudioReceiptError::EmptyField {
                field: "run.command",
            });
        }
        require_non_empty("placement", &self.placement)?;
        require_non_empty("scope", &self.scope)?;
        validate_sha256_hex("transcript.text_sha256", &self.transcript.text_sha256)
            .map_err(|actual| ShortAudioReceiptError::InvalidTranscriptSha256 { actual })?;
        let expected_text_sha = sha256_hex_bytes(self.transcript.text.as_bytes());
        if self.transcript.text_sha256 != expected_text_sha {
            return Err(ShortAudioReceiptError::InvalidTranscriptSha256 {
                actual: self.transcript.text_sha256.clone(),
            });
        }

        self.metrics.validate()?;
        if let Some(evidence) = &self.evidence {
            if !self.notes.is_empty() {
                return Err(ShortAudioReceiptError::FormalEvidenceNotesNotAllowed);
            }
            evidence.validate(self.observed_placement.as_ref())?;
            if evidence.core_commit != self.core_commit
                || format!("{}:{}", evidence.model_id, evidence.quant) != self.pack.model_id
                || evidence.quant != self.pack.quant
            {
                return Err(ShortAudioReceiptError::EvidenceBindingMismatch);
            }
            if let Some(execution) = &evidence.execution {
                let expected_run_state = match execution.mode {
                    ShortAudioReuseMode::Cold => ("cold", "empty"),
                    ShortAudioReuseMode::Reuse => ("warm", "populated"),
                };
                if (self.run.warmup.as_str(), self.run.cache_state.as_str()) != expected_run_state {
                    return Err(ShortAudioReceiptError::InvalidEvidenceField {
                        field: "correctness.execution.mode",
                        actual: format!(
                            "{:?} contradicts run.warmup={}/cache_state={}",
                            execution.mode, self.run.warmup, self.run.cache_state
                        ),
                    });
                }
            }
        }

        if let Some(execution) = &self.execution {
            if execution.request_attempt_id.is_none() || execution.request_attempt_conflicted {
                return Err(ShortAudioReceiptError::InvalidExecutionProjection {
                    reason: "request attempt identity is missing or conflicted",
                });
            }
            if execution.event_history_complete != execution.event_history_reason.is_none()
                || (execution.event_history_complete && execution.dropped_events != 0)
            {
                return Err(ShortAudioReceiptError::InvalidExecutionProjection {
                    reason: "event-history completeness contradicts its reason or drop count",
                });
            }
            if execution.live_state_complete != execution.live_state_reason.is_none() {
                return Err(ShortAudioReceiptError::InvalidExecutionProjection {
                    reason: "live-state completeness contradicts its reason",
                });
            }
            if execution.live_lease_reconciliation == ShortAudioLeaseReconciliation::Matched
                && !execution.live_state_complete
            {
                return Err(ShortAudioReceiptError::InvalidExecutionProjection {
                    reason: "matched lease reconciliation requires complete live state",
                });
            }
            if execution.request_receipt_complete
                && execution.terminal.as_deref() != Some("succeeded")
            {
                return Err(ShortAudioReceiptError::InvalidExecutionProjection {
                    reason: "complete request receipt requires a succeeded terminal",
                });
            }
        }
        if let Some(lifecycle) = &self.graph_lifecycle {
            if lifecycle.overflowed || lifecycle.events.is_empty() {
                return Err(ShortAudioReceiptError::InvalidEvidenceField {
                    field: "graph_lifecycle",
                    actual: "empty or overflowed lifecycle evidence".to_string(),
                });
            }
            if lifecycle.events.iter().any(|event| {
                event.schema.as_ref() != crate::ggml_runtime::GGML_GRAPH_LIFECYCLE_SCHEMA
                    || event.provider.is_empty()
                    || event.device.is_empty()
            }) {
                return Err(ShortAudioReceiptError::InvalidEvidenceField {
                    field: "graph_lifecycle.events",
                    actual: "unversioned or unbound lifecycle event".to_string(),
                });
            }
        }
        match (
            self.actual_provider.as_deref(),
            self.actual_stable_device_id.as_deref(),
            self.actual_device.as_ref(),
        ) {
            (None, None, None) => {}
            (Some(provider), Some(stable_id), Some(device)) => {
                validate_actual_device_facts("actual_device", provider, stable_id, device)?;
                if let Some(lifecycle) = &self.graph_lifecycle {
                    let mut drifted = BTreeSet::new();
                    for event in &lifecycle.events {
                        if !lifecycle_event_proves_compute_route(&event.kind) {
                            continue;
                        }
                        if !lifecycle_route_matches_attested_or_hybrid_host(
                            event.provider.as_ref(),
                            event.device.as_ref(),
                            provider,
                            stable_id,
                        ) {
                            drifted.insert(format!("{}:{}", event.provider, event.device));
                        }
                    }
                    if !drifted.is_empty() {
                        return Err(ShortAudioReceiptError::InvalidEvidenceField {
                            field: "actual_device",
                            actual: format!(
                                "lifecycle route differs from final live backend {provider}:{stable_id}; observed {}",
                                drifted.into_iter().collect::<Vec<_>>().join(", ")
                            ),
                        });
                    }
                }
            }
            _ => {
                return Err(ShortAudioReceiptError::InvalidEvidenceField {
                    field: "actual_device",
                    actual: "actual provider/device facts must be complete and consistent"
                        .to_string(),
                });
            }
        }

        match (self.metrics.rtf_median, self.metrics.rtf_samples.is_empty()) {
            (Some(_), true) => return Err(ShortAudioReceiptError::MedianWithoutSamples),
            (Some(median), false) => {
                let expected = median_f64(&self.metrics.rtf_samples)
                    .ok_or(ShortAudioReceiptError::MedianWithoutSamples)?;
                if !approx_eq(median, expected) {
                    return Err(ShortAudioReceiptError::MedianMismatch {
                        median: format!("{median}"),
                        expected: format!("{expected}"),
                    });
                }
            }
            (None, _) => {}
        }
        let diagnostics = self
            .decode_diagnostics
            .as_ref()
            .ok_or(ShortAudioReceiptError::DecodeDiagnosticsMissing)?;
        validate_decode_diagnostics(diagnostics)?;
        if self.evidence.is_some() && diagnostics.capability_evidence_revision.is_none() {
            return Err(ShortAudioReceiptError::InvalidEvidenceField {
                field: "decode_diagnostics.capability_evidence_revision",
                actual: "missing from release-bound evidence".to_string(),
            });
        }
        Ok(())
    }

    fn validate_privacy_safe_projection(&self) -> Result<(), ShortAudioReceiptError> {
        validate_safe_receipt_label("audio.path_or_label", &self.audio.path_or_label)?;
        if let Some(audio_sha) = self.audio.path_or_label.strip_prefix("audio-sha256:")
            && audio_sha != self.audio.sha256
        {
            return Err(ShortAudioReceiptError::InvalidPrivacyProjection {
                field: "audio.path_or_label",
                actual: self.audio.path_or_label.clone(),
            });
        }
        validate_safe_receipt_scope(&self.scope)?;
        validate_safe_receipt_label("placement", &self.placement)?;
        validate_semantic_command(&self.run.command, &self.audio.path_or_label)?;
        validate_safe_run_vocabulary(&self.run)?;
        validate_safe_environment(&self.run.env_allowlist, &self.core_commit)?;
        for note in &self.notes {
            if note.len() > 512 || note.contains(['\n', '\r']) || note.contains("OPENASR_HOME") {
                return Err(ShortAudioReceiptError::InvalidPrivacyProjection {
                    field: "notes",
                    actual: note.clone(),
                });
            }
        }
        Ok(())
    }

    /// Stronger gate predicate than document validity. Runtime evidence needs
    /// a complete request join, exact live reconciliation, and intact bounded
    /// history. A legacy or overflowed receipt remains readable, but cannot
    /// close placement/token qualification cells.
    pub fn validate_qualification_eligibility(&self) -> Result<(), ShortAudioReceiptError> {
        self.validate()?;
        let Some(evidence) = self.evidence.as_ref() else {
            return Err(ShortAudioReceiptError::QualificationIneligible {
                reason: "correctness evidence is missing",
            });
        };
        if matches!(
            evidence.evidence_class,
            ShortAudioEvidenceClass::BuildPackaging
        ) {
            return Ok(());
        }
        let execution =
            self.execution
                .as_ref()
                .ok_or(ShortAudioReceiptError::QualificationIneligible {
                    reason: "runtime evidence has no execution projection",
                })?;
        if !execution.live_state_complete
            || execution.live_lease_reconciliation != ShortAudioLeaseReconciliation::Matched
        {
            return Err(ShortAudioReceiptError::QualificationIneligible {
                reason: "live owner and broker state did not reconcile",
            });
        }
        if !execution.event_history_complete || execution.dropped_events != 0 {
            return Err(ShortAudioReceiptError::QualificationIneligible {
                reason: "runtime event history is incomplete",
            });
        }
        if !execution.request_receipt_complete || !execution.timing_complete {
            return Err(ShortAudioReceiptError::QualificationIneligible {
                reason: "request receipt or four-phase timing is incomplete",
            });
        }
        Ok(())
    }

    /// Serialize as pretty JSON.
    pub fn to_pretty_json(&self) -> Result<String, ShortAudioReceiptSerializeError> {
        self.validate()?;
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Parse JSON and validate required fields.
    pub fn from_json_str(raw: &str) -> Result<Self, ShortAudioReceiptLoadError> {
        let value: serde_json::Value = serde_json::from_str(raw)?;
        validate_serialized_graph_lifecycle(&value)?;
        let receipt: Self = serde_json::from_value(value)?;
        receipt.validate_legacy_compatible()?;
        Ok(receipt)
    }
}

fn validate_serialized_graph_lifecycle(
    receipt: &serde_json::Value,
) -> Result<(), ShortAudioReceiptError> {
    let Some(lifecycle) = receipt.get("graph_lifecycle") else {
        return Ok(());
    };
    let Some(events) = lifecycle
        .as_object()
        .and_then(|object| object.get("events"))
        .and_then(serde_json::Value::as_array)
    else {
        return Ok(());
    };
    if events
        .iter()
        .any(|event| !crate::ggml_runtime::ggml_graph_lifecycle_json_shape_is_strict(event))
    {
        return Err(ShortAudioReceiptError::InvalidEvidenceField {
            field: "graph_lifecycle.events",
            actual: "unknown or missing serialized lifecycle fields".to_string(),
        });
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum ShortAudioReceiptSerializeError {
    #[error(transparent)]
    Validate(#[from] ShortAudioReceiptError),
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
}

/// Load-time errors (serde or validation).
#[derive(Debug, Error)]
pub enum ShortAudioReceiptLoadError {
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
    #[error(transparent)]
    Validate(#[from] ShortAudioReceiptError),
}

impl ShortAudioReceiptTranscript {
    pub fn from_text(text: impl Into<String>) -> Self {
        let text = text.into();
        let text_sha256 = sha256_hex_bytes(text.as_bytes());
        Self { text, text_sha256 }
    }
}

/// Versioned correctness evidence nested in the v0 receipt.
pub const SHORT_AUDIO_RECEIPT_EVIDENCE_SCHEMA: &str = "openasr.short-audio-receipt.evidence.v1";

/// stricter than the legacy v0 receipt so a hand-written partial JSON object
/// cannot become a release proof.
pub const SHORT_AUDIO_RECEIPT_ARTIFACT_CONTRACT: &str = "openasr.gpu-correctness-artifact.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShortAudioEvidenceClass {
    BuildPackaging,
    PlacementResource,
    TokenTranscript,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShortAudioReuseMode {
    Cold,
    Reuse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShortAudioCaptureMode {
    Disabled,
    Enabled,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShortAudioSchedulerMode {
    Disabled,
    Enabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShortAudioTiePolicy {
    FirstMaximum,
    LastMaximum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShortAudioOutputPlanKind {
    FullLogits,
    CompleteScores,
    NativeFirstMaxToken,
}

/// Evidence classes are intentionally disjoint. A passing placement receipt
/// cannot be consumed as token correctness, and a packaging receipt cannot
/// authorize runtime placement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioReceiptEvidence {
    pub schema: String,
    pub contract: String,
    pub evidence_class: ShortAudioEvidenceClass,
    pub matrix_sha256: String,
    pub candidate_release_subject: String,
    pub core_commit: String,
    pub catalog_digests: ShortAudioCatalogDigests,
    pub family: String,
    pub model_id: String,
    pub quant: String,
    pub topology: String,
    pub provider: String,
    /// Exact provider compilation/device target or an explicitly named,
    /// reviewed equivalence class. Provider-only evidence is never eligible.
    pub device_target: String,
    /// Exact signed backend candidate that executed this receipt.
    pub backend_id: String,
    /// Exact driver version observed by the signed backend probe.
    pub driver_version: String,
    /// Canonical fingerprint of the complete signed backend entry, including
    /// plugin and vendor runtime bytes.
    pub artifact_fingerprint: String,
    /// Stable bounded device label or opaque identity; never a local path.
    pub device: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_stable_device_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_device: Option<GgmlActualDeviceFacts>,
    pub placement: String,
    pub capture_mode: ShortAudioCaptureMode,
    pub scheduler_mode: ShortAudioSchedulerMode,
    pub result: String,
    pub artifacts: ShortAudioReceiptArtifacts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_plan: Option<ShortAudioOutputPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family_oracle: Option<ShortAudioFamilyOracle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<ShortAudioExecutionMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace: Option<ShortAudioTraceSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioCatalogDigests {
    pub inventory_sha256: String,
    pub model_catalog_sha256: String,
    pub backend_catalog_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioReceiptArtifacts {
    pub binary: ShortAudioArtifactIdentity,
    pub plugin: ShortAudioArtifactIdentity,
    pub pack: ShortAudioArtifactIdentity,
    pub fixture: ShortAudioArtifactIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioArtifactIdentity {
    pub label: String,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioOutputPlan {
    pub kind: ShortAudioOutputPlanKind,
    pub requires_complete_output: bool,
    pub tie_policy: ShortAudioTiePolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioFamilyOracle {
    pub family: String,
    pub tie_policy: ShortAudioTiePolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioExecutionMode {
    pub mode: ShortAudioReuseMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_rebuild_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioTraceSummary {
    pub token_trace: ShortAudioArtifactIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logits: Option<ShortAudioArtifactIdentity>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_k: Vec<ShortAudioTopKSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top1_top2_margin: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioTopKSummary {
    pub token_id: u32,
    pub value: f64,
}

impl ShortAudioReceiptEvidence {
    fn validate(
        &self,
        observed_placement: Option<&GgmlExecutionPlacementSummary>,
    ) -> Result<(), ShortAudioReceiptError> {
        if self.schema != SHORT_AUDIO_RECEIPT_EVIDENCE_SCHEMA {
            return Err(ShortAudioReceiptError::EvidenceSchemaMismatch {
                actual: self.schema.clone(),
            });
        }
        if self.contract != SHORT_AUDIO_RECEIPT_ARTIFACT_CONTRACT {
            return Err(ShortAudioReceiptError::InvalidEvidenceField {
                field: "correctness.contract",
                actual: self.contract.clone(),
            });
        }
        for (field, value) in [
            ("correctness.matrix_sha256", self.matrix_sha256.as_str()),
            (
                "correctness.candidate_release_subject",
                self.candidate_release_subject.as_str(),
            ),
            ("correctness.core_commit", self.core_commit.as_str()),
            ("correctness.family", self.family.as_str()),
            ("correctness.model_id", self.model_id.as_str()),
            ("correctness.quant", self.quant.as_str()),
            ("correctness.topology", self.topology.as_str()),
            ("correctness.provider", self.provider.as_str()),
            ("correctness.device_target", self.device_target.as_str()),
            ("correctness.backend_id", self.backend_id.as_str()),
            ("correctness.driver_version", self.driver_version.as_str()),
            (
                "correctness.artifact_fingerprint",
                self.artifact_fingerprint.as_str(),
            ),
            ("correctness.device", self.device.as_str()),
            ("correctness.placement", self.placement.as_str()),
            ("correctness.result", self.result.as_str()),
        ] {
            require_non_empty(field, value)?;
            if value.len() > 256 || value.contains(['\n', '\r']) {
                return Err(ShortAudioReceiptError::InvalidEvidenceField {
                    field,
                    actual: value.to_string(),
                });
            }
        }
        validate_sha256_hex("correctness.matrix_sha256", &self.matrix_sha256).map_err(
            |actual| ShortAudioReceiptError::InvalidEvidenceDigest {
                field: "correctness.matrix_sha256",
                actual,
            },
        )?;
        validate_sha256_hex(
            "correctness.artifact_fingerprint",
            &self.artifact_fingerprint,
        )
        .map_err(|actual| ShortAudioReceiptError::InvalidEvidenceDigest {
            field: "correctness.artifact_fingerprint",
            actual,
        })?;
        if self.driver_version.len() > 64
            || !self
                .driver_version
                .split('.')
                .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
        {
            return Err(ShortAudioReceiptError::InvalidEvidenceField {
                field: "correctness.driver_version",
                actual: self.driver_version.clone(),
            });
        }
        validate_core_commit(&self.core_commit)?;
        self.catalog_digests.validate()?;
        if self.result != "pass" {
            return Err(ShortAudioReceiptError::EvidenceNotPassing {
                evidence_class: format!("{:?}", self.evidence_class),
                result: self.result.clone(),
            });
        }
        self.artifacts.validate()?;
        match self.evidence_class {
            ShortAudioEvidenceClass::BuildPackaging => {}
            ShortAudioEvidenceClass::PlacementResource => {
                self.validate_actual_device()?;
                if observed_placement.is_none() {
                    return Err(ShortAudioReceiptError::PlacementEvidenceMissing);
                }
            }
            ShortAudioEvidenceClass::TokenTranscript => {
                self.validate_actual_device()?;
                if self.output_plan.is_none()
                    || self.family_oracle.is_none()
                    || self.execution.is_none()
                    || self.trace.is_none()
                {
                    return Err(ShortAudioReceiptError::TokenEvidenceIncomplete);
                }
                let plan = self.output_plan.as_ref().expect("checked above");
                let oracle = self.family_oracle.as_ref().expect("checked above");
                if oracle.family != self.family || oracle.tie_policy != plan.tie_policy {
                    return Err(ShortAudioReceiptError::OutputPlanOracleMismatch);
                }
                if matches!(plan.kind, ShortAudioOutputPlanKind::NativeFirstMaxToken)
                    && plan.requires_complete_output
                {
                    return Err(ShortAudioReceiptError::InvalidEvidenceField {
                        field: "correctness.output_plan.requires_complete_output",
                        actual: "true for native compact token plan".to_string(),
                    });
                }
                if !matches!(plan.kind, ShortAudioOutputPlanKind::NativeFirstMaxToken)
                    && !plan.requires_complete_output
                {
                    return Err(ShortAudioReceiptError::InvalidEvidenceField {
                        field: "correctness.output_plan.requires_complete_output",
                        actual: "false for complete output plan".to_string(),
                    });
                }
                let execution = self.execution.as_ref().expect("checked above");
                if execution.mode != ShortAudioReuseMode::Cold
                    && execution.mode != ShortAudioReuseMode::Reuse
                {
                    return Err(ShortAudioReceiptError::InvalidEvidenceField {
                        field: "correctness.execution.mode",
                        actual: format!("{:?}", execution.mode),
                    });
                }
                let trace = self.trace.as_ref().expect("checked above");
                trace.token_trace.validate()?;
                match (plan.requires_complete_output, trace.logits.as_ref()) {
                    (true, Some(logits)) => logits.validate()?,
                    (true, None) => {
                        return Err(ShortAudioReceiptError::InvalidEvidenceField {
                            field: "correctness.trace.logits",
                            actual: "missing for a complete-output plan".to_string(),
                        });
                    }
                    (false, Some(_)) => {
                        return Err(ShortAudioReceiptError::InvalidEvidenceField {
                            field: "correctness.trace.logits",
                            actual: "present for a compact token-only plan".to_string(),
                        });
                    }
                    (false, None) => {}
                }
                if trace.top_k.is_empty()
                    || trace.top_k.len() > 32
                    || trace.top_k.iter().any(|item| !item.value.is_finite())
                {
                    return Err(ShortAudioReceiptError::InvalidTopKSummary);
                }
                if trace
                    .top1_top2_margin
                    .is_some_and(|margin| !margin.is_finite() || margin < 0.0)
                {
                    return Err(ShortAudioReceiptError::InvalidTopKSummary);
                }
            }
        }
        Ok(())
    }

    fn validate_actual_device(&self) -> Result<(), ShortAudioReceiptError> {
        let (Some(actual_provider), Some(actual_stable_device_id), Some(actual_device)) = (
            self.actual_provider.as_deref(),
            self.actual_stable_device_id.as_deref(),
            self.actual_device.as_ref(),
        ) else {
            return Err(ShortAudioReceiptError::InvalidEvidenceField {
                field: "correctness.actual_device",
                actual: "missing final live backend facts".to_string(),
            });
        };
        if actual_provider != self.provider || actual_stable_device_id != self.device {
            return Err(ShortAudioReceiptError::InvalidEvidenceField {
                field: "correctness.actual_device",
                actual: "final live backend identity differs from the evidence lane".to_string(),
            });
        }
        validate_actual_device_facts(
            "correctness.actual_device",
            actual_provider,
            actual_stable_device_id,
            actual_device,
        )
    }
}

impl ShortAudioCatalogDigests {
    fn validate(&self) -> Result<(), ShortAudioReceiptError> {
        for (field, value) in [
            (
                "correctness.catalog_digests.inventory_sha256",
                &self.inventory_sha256,
            ),
            (
                "correctness.catalog_digests.model_catalog_sha256",
                &self.model_catalog_sha256,
            ),
            (
                "correctness.catalog_digests.backend_catalog_sha256",
                &self.backend_catalog_sha256,
            ),
        ] {
            validate_sha256_hex(field, value).map_err(|actual| {
                ShortAudioReceiptError::InvalidEvidenceDigest {
                    field: "correctness.catalog_digests",
                    actual,
                }
            })?;
        }
        Ok(())
    }
}

impl ShortAudioArtifactIdentity {
    fn validate(&self) -> Result<(), ShortAudioReceiptError> {
        require_non_empty("correctness.artifacts.label", &self.label)?;
        if self.label.len() > 128 || self.label.contains(['\n', '\r', '/', '\\']) {
            return Err(ShortAudioReceiptError::InvalidEvidenceField {
                field: "correctness.artifacts.label",
                actual: self.label.clone(),
            });
        }
        validate_sha256_hex("correctness.artifacts.sha256", &self.sha256).map_err(|actual| {
            ShortAudioReceiptError::InvalidEvidenceDigest {
                field: "correctness.artifacts.sha256",
                actual,
            }
        })
    }
}

impl ShortAudioReceiptArtifacts {
    fn validate(&self) -> Result<(), ShortAudioReceiptError> {
        for identity in [&self.binary, &self.plugin, &self.pack, &self.fixture] {
            identity.validate()?;
        }
        Ok(())
    }
}

/// Lowercase hex sha256 of `bytes`.
pub fn sha256_hex_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex_lower(digest)
}

/// Stream-hash a file without loading it entirely into memory.
pub fn sha256_file(path: impl AsRef<Path>) -> Result<(u64, String), ShortAudioReceiptError> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|source| ShortAudioReceiptError::HashIo {
        path: path.display().to_string(),
        reason: source.to_string(),
    })?;
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 1024 * 64];
    let mut total = 0_u64;
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|source| ShortAudioReceiptError::HashIo {
                path: path.display().to_string(),
                reason: source.to_string(),
            })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total = total.saturating_add(n as u64);
    }
    Ok((total, hex_lower(hasher.finalize())))
}

fn validate_finite_non_negative_metric(
    field: &'static str,
    value: f64,
) -> Result<(), ShortAudioReceiptError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(ShortAudioReceiptError::InvalidMetric {
            field,
            actual: value.to_string(),
        })
    }
}

/// Median of finite f64 samples. Even counts use a scaled mean to avoid
/// overflowing when two large finite values are added.
pub fn median_f64(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() || samples.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[mid])
    } else {
        Some(sorted[mid - 1] * 0.5 + sorted[mid] * 0.5)
    }
}

/// Compact host OS id used in receipts: `darwin` / `linux` / `windows`.
pub fn receipt_os_id() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "darwin"
    }
    #[cfg(target_os = "linux")]
    {
        "linux"
    }
    #[cfg(windows)]
    {
        "windows"
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
    {
        "unknown"
    }
}

/// Validate a 40-hex git commit sha (lowercase or uppercase accepted; stored as-is).
pub fn validate_core_commit(value: &str) -> Result<(), ShortAudioReceiptError> {
    let ok = value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit());
    if ok {
        Ok(())
    } else {
        Err(ShortAudioReceiptError::InvalidCoreCommit {
            actual: value.to_string(),
        })
    }
}

/// Resolve core commit from explicit value, then `OPENASR_BUILD_COMMIT`, then
/// `git rev-parse HEAD` in `git_cwd` when provided.
pub fn resolve_core_commit(
    explicit: Option<&str>,
    git_cwd: Option<&Path>,
) -> Result<String, ShortAudioReceiptError> {
    if let Some(value) = explicit.map(str::trim).filter(|v| !v.is_empty()) {
        validate_core_commit(value)?;
        return Ok(value.to_ascii_lowercase());
    }
    if let Ok(value) = std::env::var(crate::ggml_runtime::BUILD_COMMIT_ENV) {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            validate_core_commit(trimmed)?;
            return Ok(trimmed.to_ascii_lowercase());
        }
    }
    if let Some(cwd) = git_cwd
        && let Some(sha) = git_rev_parse_head(cwd)
    {
        validate_core_commit(&sha)?;
        return Ok(sha.to_ascii_lowercase());
    }
    Err(ShortAudioReceiptError::InvalidCoreCommit {
        actual: String::new(),
    })
}

fn git_rev_parse_head(cwd: &Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(cwd)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let sha = String::from_utf8(output.stdout).ok()?;
    let sha = sha.trim();
    if sha.len() == 40 {
        Some(sha.to_string())
    } else {
        None
    }
}

fn require_non_empty(field: &'static str, value: &str) -> Result<(), ShortAudioReceiptError> {
    if value.trim().is_empty() {
        Err(ShortAudioReceiptError::EmptyField { field })
    } else {
        Ok(())
    }
}

fn validate_safe_receipt_label(
    field: &'static str,
    value: &str,
) -> Result<(), ShortAudioReceiptError> {
    let bytes = value.as_bytes();
    let valid = !bytes.is_empty()
        && bytes.len() <= 256
        && bytes[0].is_ascii_alphanumeric()
        && bytes.iter().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'-' | b'_' | b'.' | b':' | b'+' | b'@' | b'=')
        })
        && !looks_like_windows_drive_path(value)
        && !value.eq_ignore_ascii_case("OPENASR_HOME");
    if valid {
        Ok(())
    } else {
        Err(ShortAudioReceiptError::InvalidPrivacyProjection {
            field,
            actual: value.to_string(),
        })
    }
}

fn validate_safe_receipt_scope(value: &str) -> Result<(), ShortAudioReceiptError> {
    if validate_safe_receipt_label("scope", value).is_ok() {
        return Ok(());
    }
    let mut segments = value.split('/');
    let Some(base) = segments.next() else {
        unreachable!("split always yields one segment");
    };
    let nonce = segments.next();
    let valid = segments.next().is_none()
        && validate_safe_receipt_label("scope", base).is_ok()
        && nonce.is_some_and(|nonce| {
            nonce.len() == 32
                && nonce
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        });
    if valid {
        Ok(())
    } else {
        Err(ShortAudioReceiptError::InvalidPrivacyProjection {
            field: "scope",
            actual: value.to_string(),
        })
    }
}

fn looks_like_windows_drive_path(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

fn validate_semantic_command(
    command: &[String],
    audio_label: &str,
) -> Result<(), ShortAudioReceiptError> {
    if command.first().map(String::as_str) != Some("openasr") || command.len() > 64 {
        return Err(ShortAudioReceiptError::InvalidPrivacyProjection {
            field: "run.command",
            actual: command.join(" "),
        });
    }
    for (index, part) in command.iter().enumerate() {
        if index > 0 && command[index - 1] == "--scope" {
            validate_safe_receipt_scope(part)?;
            continue;
        }
        validate_safe_command_part(part)?;
        if part.eq_ignore_ascii_case("--openasr-home")
            || part.eq_ignore_ascii_case("OPENASR_HOME")
            || part.to_ascii_lowercase().starts_with("file:")
        {
            return Err(ShortAudioReceiptError::InvalidPrivacyProjection {
                field: "run.command",
                actual: part.clone(),
            });
        }
    }
    for (flag, expected_prefix, exact_value) in [
        ("--audio", Some("audio-sha256:"), Some(audio_label)),
        ("--out", None, Some("receipt-output")),
        ("--model-pack", Some("pack-content-sha256:"), None),
        ("--trace-out", None, Some("runtime-trace-output")),
        ("--logits-out", None, Some("full-logits-output")),
    ] {
        for index in command
            .iter()
            .enumerate()
            .filter_map(|(index, part)| (part == flag).then_some(index))
        {
            let value = command.get(index + 1).ok_or_else(|| {
                ShortAudioReceiptError::InvalidPrivacyProjection {
                    field: "run.command",
                    actual: format!("{flag} has no semantic value"),
                }
            })?;
            let valid = exact_value.is_none_or(|expected| value == expected)
                && expected_prefix.is_none_or(|prefix| value.starts_with(prefix));
            if !valid {
                return Err(ShortAudioReceiptError::InvalidPrivacyProjection {
                    field: "run.command",
                    actual: format!("{flag} {value}"),
                });
            }
        }
    }
    Ok(())
}

fn validate_safe_command_part(value: &str) -> Result<(), ShortAudioReceiptError> {
    let bytes = value.as_bytes();
    let valid = !bytes.is_empty()
        && bytes.len() <= 256
        && bytes.iter().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'-' | b'_' | b'.' | b':' | b'+' | b'@' | b'=')
        })
        && !looks_like_windows_drive_path(value)
        && !value.contains(['/', '\\', '~']);
    if valid {
        Ok(())
    } else {
        Err(ShortAudioReceiptError::InvalidPrivacyProjection {
            field: "run.command",
            actual: value.to_string(),
        })
    }
}

fn validate_safe_run_vocabulary(run: &ShortAudioReceiptRun) -> Result<(), ShortAudioReceiptError> {
    let valid = matches!(run.backend.as_str(), "native" | "mock")
        && matches!(
            run.device.as_str(),
            "cpu" | "metal" | "cuda" | "hip" | "vulkan" | "gpu" | "accelerated" | "auto"
        )
        && matches!(run.os.as_str(), "darwin" | "linux" | "windows")
        && matches!(run.warmup.as_str(), "cold" | "warm")
        && matches!(run.cache_state.as_str(), "empty" | "populated");
    if valid {
        Ok(())
    } else {
        Err(ShortAudioReceiptError::InvalidPrivacyProjection {
            field: "run",
            actual: format!(
                "backend={},device={},os={},warmup={},cache_state={}",
                run.backend, run.device, run.os, run.warmup, run.cache_state
            ),
        })
    }
}

fn validate_safe_environment(
    env: &BTreeMap<String, String>,
    core_commit: &str,
) -> Result<(), ShortAudioReceiptError> {
    for (key, value) in env {
        let valid = match key.as_str() {
            "OPENASR_GGML_BACKEND" => {
                matches!(
                    value.as_str(),
                    "cpu" | "metal" | "gpu" | "cuda" | "hip" | "vulkan"
                )
            }
            "OPENASR_BUILD_COMMIT" => value == core_commit && validate_core_commit(value).is_ok(),
            "OPENASR_OFFLINE" => value == "true",
            _ => false,
        };
        if !valid {
            return Err(ShortAudioReceiptError::InvalidPrivacyProjection {
                field: "run.env_allowlist",
                actual: format!("{key}={value}"),
            });
        }
    }
    Ok(())
}

fn lifecycle_event_proves_compute_route(
    kind: &crate::ggml_runtime::GgmlGraphLifecycleEventKind,
) -> bool {
    matches!(
        kind,
        crate::ggml_runtime::GgmlGraphLifecycleEventKind::ComputeStarted { .. }
            | crate::ggml_runtime::GgmlGraphLifecycleEventKind::ComputeCompleted { .. }
            | crate::ggml_runtime::GgmlGraphLifecycleEventKind::OutputRead { .. }
            | crate::ggml_runtime::GgmlGraphLifecycleEventKind::KvWriteCommitted { .. }
            | crate::ggml_runtime::GgmlGraphLifecycleEventKind::CaptureStateObserved { .. }
            | crate::ggml_runtime::GgmlGraphLifecycleEventKind::CaptureExecutableObserved { .. }
            | crate::ggml_runtime::GgmlGraphLifecycleEventKind::CaptureExecutableCreated { .. }
    )
}

fn lifecycle_route_matches_attested_or_hybrid_host(
    event_provider: &str,
    event_device: &str,
    attested_provider: &str,
    attested_stable_id: &str,
) -> bool {
    if event_provider == attested_provider && event_device == attested_stable_id {
        return true;
    }
    // Product Hybrid (Moonshine Vulkan default) keeps the decoder on CPU
    // while the attested live backend is the encoder accelerator. CPU
    // events are the host half of that plan, not a second GPU route.
    attested_provider != crate::device::execution_route::ExecutionProvider::Cpu.as_str()
        && event_provider == crate::device::execution_route::ExecutionProvider::Cpu.as_str()
        && event_device == "CPU"
}

fn validate_actual_device_facts(
    field: &'static str,
    provider: &str,
    stable_device_id: &str,
    device: &GgmlActualDeviceFacts,
) -> Result<(), ShortAudioReceiptError> {
    let bounded = |value: &str, max_chars: usize| {
        !value.trim().is_empty()
            && value.chars().count() <= max_chars
            && !value.contains(['\n', '\r'])
    };
    let provider_device_id_valid = device
        .provider_device_id
        .as_deref()
        .is_none_or(|value| bounded(value, 128));
    if !bounded(provider, 64)
        || !bounded(stable_device_id, 128)
        || stable_device_id != device.name
        || !bounded(&device.device_type, 64)
        || !bounded(&device.name, 128)
        || !bounded(&device.description, 256)
        || !provider_device_id_valid
        || device.pci_vendor_id == Some(0)
    {
        return Err(ShortAudioReceiptError::InvalidEvidenceField {
            field,
            actual: "live backend facts are empty, unbounded, or inconsistent".to_string(),
        });
    }
    Ok(())
}

const MAX_ENCODER_DECODER_SPLITS: usize = 8;
const NATIVE_RECEIPT_SEQ2SEQ_DECODE_DRIVER: &str = "shared-seq2seq-greedy";

/// Project fail-closed decode diagnostics from the shipped runtime that ran.
///
/// `resolved` is the mock-receipt planner input. Native receipts pass the
/// request-local collector snapshot instead of reconstructing plan/reuse from
/// CLI flags. Seq2seq native receipts without token steps fail closed.
pub fn decode_diagnostics_from_shipped_runtime(
    resolved: Option<&ResolvedFamilyRuntimeInput>,
    snapshot: Option<&NativeExecutionReceiptSnapshot>,
) -> Result<ShortAudioReceiptDecodeDiagnostics, ShortAudioReceiptError> {
    let resolved = snapshot
        .and_then(|snapshot| snapshot.facts.as_ref())
        .map(|facts| &facts.resolved_runtime)
        .or(resolved)
        .ok_or(ShortAudioReceiptError::DecodeDiagnosticsMissing)?;
    let token_steps = snapshot
        .map(|snapshot| snapshot.token_steps.as_slice())
        .unwrap_or(&[]);
    if snapshot
        .and_then(|snapshot| snapshot.facts.as_ref())
        .is_some_and(|facts| facts.topology.decode_driver == NATIVE_RECEIPT_SEQ2SEQ_DECODE_DRIVER)
        && token_steps.is_empty()
    {
        return Err(ShortAudioReceiptError::NativeSeq2SeqTokenStepsMissing);
    }
    if token_steps.len() > SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS {
        return Err(ShortAudioReceiptError::DecodeStepsUnbounded {
            max: SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS,
            actual: token_steps.len(),
        });
    }
    let mut steps = Vec::with_capacity(token_steps.len());
    for step in token_steps {
        steps.push(decode_step_from_native_token(
            step,
            snapshot.map(|snapshot| &snapshot.graph_lifecycle),
        )?);
    }
    Ok(ShortAudioReceiptDecodeDiagnostics {
        output_plan: ShortAudioReceiptOutputPlan::from(resolved.output_plan()),
        reuse_mode: ShortAudioReceiptReuseMode::from(resolved.reuse_mode()),
        capability_evidence_revision: Some(resolved.evidence_revision()),
        steps,
        first_divergence: None,
        encoder_decoder_splits: Vec::new(),
    })
}

fn decode_step_from_native_token(
    step: &NativeExecutionTokenStep,
    lifecycle: Option<&GgmlGraphLifecycleSnapshot>,
) -> Result<ShortAudioReceiptDecodeStep, ShortAudioReceiptError> {
    let step_index = u32::try_from(step.step_index).map_err(|_| {
        ShortAudioReceiptError::DecodeStepsUnbounded {
            max: SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS,
            actual: step.step_index,
        }
    })?;
    let token_id =
        i32::try_from(step.token_id).map_err(|_| ShortAudioReceiptError::InvalidEvidenceField {
            field: "decode_diagnostics.steps.token_id",
            actual: step.token_id.to_string(),
        })?;
    let graph_rebuilt = observed_graph_built_for_compute(step, lifecycle)?;
    Ok(ShortAudioReceiptDecodeStep {
        step: step_index,
        token_id: Some(token_id),
        logits_sha256: step.logits_sha256.clone(),
        top2_margin: step.top2_margin,
        graph_rebuilt,
    })
}

fn observed_graph_built_for_compute(
    step: &NativeExecutionTokenStep,
    lifecycle: Option<&GgmlGraphLifecycleSnapshot>,
) -> Result<bool, ShortAudioReceiptError> {
    let Some(compute) = step.compute.as_ref() else {
        return Err(ShortAudioReceiptError::InvalidEvidenceField {
            field: "decode_diagnostics.steps.graph_rebuilt",
            actual: "token step has no native compute witness".to_string(),
        });
    };
    let Some(lifecycle) = lifecycle else {
        return Err(ShortAudioReceiptError::InvalidEvidenceField {
            field: "decode_diagnostics.steps.graph_rebuilt",
            actual: "native token step has no graph lifecycle snapshot".to_string(),
        });
    };
    let (graph_instance, graph_generation, compute_sequence, output_generation) =
        compute.compute_identity();
    let mut last_origin_created = None::<bool>;
    let mut first_compute_after_origin = None::<u64>;
    let mut origin_at_this_compute = None::<bool>;
    let mut first_compute_at_this_compute = None::<u64>;
    let mut started = false;
    let mut completed = false;
    let mut read = false;
    for event in lifecycle.events.iter().filter(|event| {
        event.graph_instance == graph_instance && event.graph_generation == graph_generation
    }) {
        match event.kind {
            crate::ggml_runtime::GgmlGraphLifecycleEventKind::Created { .. } => {
                last_origin_created = Some(true);
                first_compute_after_origin = None;
            }
            crate::ggml_runtime::GgmlGraphLifecycleEventKind::ExistingGraphObserved { .. } => {
                last_origin_created = Some(false);
                first_compute_after_origin = None;
            }
            crate::ggml_runtime::GgmlGraphLifecycleEventKind::ComputeStarted {
                compute_sequence: observed,
                ..
            } => {
                if first_compute_after_origin.is_none() {
                    first_compute_after_origin = Some(observed);
                }
                if observed == compute_sequence {
                    started = true;
                    origin_at_this_compute = last_origin_created;
                    first_compute_at_this_compute = first_compute_after_origin;
                }
            }
            crate::ggml_runtime::GgmlGraphLifecycleEventKind::ComputeCompleted {
                compute_sequence: observed,
                output_generation: output,
            } => {
                completed |= observed == compute_sequence && output == output_generation;
            }
            crate::ggml_runtime::GgmlGraphLifecycleEventKind::OutputRead {
                compute_sequence: observed,
                output_generation_consumed: output,
                ..
            } => {
                read |= observed == compute_sequence && output == output_generation;
            }
            _ => {}
        }
    }
    if origin_at_this_compute.is_some() && started {
        // ComputeCompleted/OutputRead may be omitted on the capture-unsupported
        // hot path. Origin plus ComputeStarted is enough to bind graph_rebuilt.
        return Ok(origin_at_this_compute == Some(true)
            && first_compute_at_this_compute == Some(compute_sequence));
    }
    // HIP/Vulkan capture-unsupported graphs stop appending per-compute events
    // after the first few sequences so Zipformer/longform receipts do not
    // overflow. Later token steps still consume the same persistent graph:
    // Created/ExistingGraphObserved plus an earlier attested compute is
    // enough to project graph_rebuilt=false.
    if !lifecycle.overflowed
        && last_origin_created.is_some()
        && first_compute_after_origin.is_some_and(|first| compute_sequence > first)
    {
        return Ok(false);
    }
    let reason = match (
        lifecycle.overflowed,
        origin_at_this_compute,
        started,
        completed,
        read,
    ) {
        (true, _, _, _, _) => {
            "native graph lifecycle overflowed before this compute could be bound"
        }
        (_, None, _, _, _) => {
            "native compute witness has no created or existing-graph origin before compute_started"
        }
        (_, _, false, _, _) => "native compute witness has no matching compute_started event",
        (_, _, _, false, _) => "native compute witness has no matching compute_completed event",
        _ => "native compute witness has no matching output_read event",
    };
    Err(ShortAudioReceiptError::InvalidEvidenceField {
        field: "decode_diagnostics.steps.graph_rebuilt",
        actual: reason.to_string(),
    })
}

fn validate_decode_diagnostics(
    diagnostics: &ShortAudioReceiptDecodeDiagnostics,
) -> Result<(), ShortAudioReceiptError> {
    if diagnostics.capability_evidence_revision == Some(0) {
        return Err(ShortAudioReceiptError::InvalidEvidenceField {
            field: "decode_diagnostics.capability_evidence_revision",
            actual: "0".to_string(),
        });
    }
    if diagnostics.steps.len() > SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS {
        return Err(ShortAudioReceiptError::DecodeStepsUnbounded {
            max: SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS,
            actual: diagnostics.steps.len(),
        });
    }
    if diagnostics.encoder_decoder_splits.len() > MAX_ENCODER_DECODER_SPLITS {
        return Err(ShortAudioReceiptError::EncoderDecoderSplitsUnbounded {
            max: MAX_ENCODER_DECODER_SPLITS,
            actual: diagnostics.encoder_decoder_splits.len(),
        });
    }
    for step in &diagnostics.steps {
        if let Some(hash) = &step.logits_sha256 {
            validate_sha256_hex("decode_diagnostics.steps.logits_sha256", hash).map_err(
                |actual| ShortAudioReceiptError::InvalidDiagnosticSha256 {
                    field: "decode_diagnostics.steps.logits_sha256",
                    actual,
                },
            )?;
        }
    }
    for split in &diagnostics.encoder_decoder_splits {
        if split.step_logits_hashes.len() > SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS {
            return Err(ShortAudioReceiptError::DecodeStepsUnbounded {
                max: SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS,
                actual: split.step_logits_hashes.len(),
            });
        }
        if let Some(hash) = &split.encoder_checksum {
            validate_sha256_hex("decode_diagnostics.encoder_checksum", hash).map_err(|actual| {
                ShortAudioReceiptError::InvalidDiagnosticSha256 {
                    field: "decode_diagnostics.encoder_checksum",
                    actual,
                }
            })?;
        }
        if let Some(hash) = &split.cross_kv_checksum {
            validate_sha256_hex("decode_diagnostics.cross_kv_checksum", hash).map_err(
                |actual| ShortAudioReceiptError::InvalidDiagnosticSha256 {
                    field: "decode_diagnostics.cross_kv_checksum",
                    actual,
                },
            )?;
        }
        for hash in &split.step_logits_hashes {
            validate_sha256_hex("decode_diagnostics.step_logits_hashes", hash).map_err(
                |actual| ShortAudioReceiptError::InvalidDiagnosticSha256 {
                    field: "decode_diagnostics.step_logits_hashes",
                    actual,
                },
            )?;
        }
        for hash in &split.mask_hashes {
            validate_sha256_hex("decode_diagnostics.mask_hashes", hash).map_err(|actual| {
                ShortAudioReceiptError::InvalidDiagnosticSha256 {
                    field: "decode_diagnostics.mask_hashes",
                    actual,
                }
            })?;
        }
    }
    Ok(())
}

fn validate_sha256_hex(_field: &str, value: &str) -> Result<(), String> {
    let ok = value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    if ok { Ok(()) } else { Err(value.to_string()) }
}

fn hex_lower(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn approx_eq(a: f64, b: f64) -> bool {
    if a == b {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() <= scale * 1e-9
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn sample_receipt() -> ShortAudioReceipt {
        let transcript = ShortAudioReceiptTranscript::from_text("hello world");
        ShortAudioReceipt {
            schema: SHORT_AUDIO_RECEIPT_SCHEMA.to_string(),
            core_commit: "0123456789abcdef0123456789abcdef01234567".to_string(),
            pack: ShortAudioReceiptPack {
                model_id: "funasr-nano:q4_k".to_string(),
                content_sha256: "a".repeat(64),
                size_bytes: 12,
                quant: "q4_k".to_string(),
            },
            audio: ShortAudioReceiptAudio {
                path_or_label: format!("audio-sha256:{}", "b".repeat(64)),
                sha256: "b".repeat(64),
                duration_s: Some(1.5),
            },
            run: ShortAudioReceiptRun {
                backend: "native".to_string(),
                device: "cpu".to_string(),
                os: "darwin".to_string(),
                command: vec![
                    "openasr".to_string(),
                    "bench-receipt".to_string(),
                    "short-audio".to_string(),
                ],
                env_allowlist: BTreeMap::from([
                    ("OPENASR_GGML_BACKEND".to_string(), "cpu".to_string()),
                    (
                        "OPENASR_BUILD_COMMIT".to_string(),
                        "0123456789abcdef0123456789abcdef01234567".to_string(),
                    ),
                    ("OPENASR_OFFLINE".to_string(), "true".to_string()),
                ]),
                warmup: "cold".to_string(),
                cache_state: "empty".to_string(),
            },
            metrics: ShortAudioReceiptMetrics {
                wer_or_cer: None,
                rtf_samples: vec![0.4, 0.5, 0.6],
                rtf_median: Some(0.5),
                ttft_s: None,
                peak_rss_bytes: Some(1024),
                peak_rss_before_model_bytes: Some(640),
                rss_before_model_bytes: Some(512),
                rss_after_model_bytes: Some(768),
                phys_footprint_before_model_bytes: Some(448),
                phys_footprint_after_model_bytes: Some(704),
                peak_phys_footprint_before_model_bytes: Some(576),
                peak_phys_footprint_bytes: Some(896),
                peak_vram_bytes: None,
                measurement_method: Some(SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK.to_string()),
            },
            transcript,
            placement: "cpu".to_string(),
            observed_placement: Some(GgmlExecutionPlacementSummary {
                direct_graph_computes: 1,
                scheduler_graph_computes: 0,
                observed_nodes_by_backend: BTreeMap::from([("CPU".to_string(), 12)]),
                observed_compute_nodes_by_backend: BTreeMap::from([("CPU".to_string(), 10)]),
                observed_node_output_bytes_by_backend: BTreeMap::from([("CPU".to_string(), 4096)]),
                fallback_node_samples_by_backend: BTreeMap::new(),
            }),
            graph_lifecycle: None,
            actual_provider: None,
            actual_stable_device_id: None,
            actual_device: None,
            evidence: None,
            execution: None,
            scope: SHORT_AUDIO_RECEIPT_DEFAULT_SCOPE.to_string(),
            notes: Vec::new(),
            decode_diagnostics: Some(sample_decode_diagnostics()),
        }
    }

    fn sample_decode_diagnostics() -> ShortAudioReceiptDecodeDiagnostics {
        ShortAudioReceiptDecodeDiagnostics {
            output_plan: ShortAudioReceiptOutputPlan::FullLogits,
            reuse_mode: ShortAudioReceiptReuseMode::FreshGraph,
            capability_evidence_revision: Some(1),
            steps: Vec::new(),
            first_divergence: None,
            encoder_decoder_splits: Vec::new(),
        }
    }

    fn sample_artifacts() -> ShortAudioReceiptArtifacts {
        let identity = |label: &str, sha256: &str| ShortAudioArtifactIdentity {
            label: label.to_string(),
            sha256: sha256.repeat(64),
            size_bytes: Some(10),
        };
        ShortAudioReceiptArtifacts {
            binary: identity("openasr-test-binary", "c"),
            plugin: identity("cuda-plugin", "d"),
            pack: identity("fixture-pack", "e"),
            fixture: identity("jfk-short", "f"),
        }
    }

    fn sample_token_evidence(mode: ShortAudioReuseMode) -> ShortAudioReceiptEvidence {
        ShortAudioReceiptEvidence {
            schema: SHORT_AUDIO_RECEIPT_EVIDENCE_SCHEMA.to_string(),
            contract: SHORT_AUDIO_RECEIPT_ARTIFACT_CONTRACT.to_string(),
            evidence_class: ShortAudioEvidenceClass::TokenTranscript,
            matrix_sha256: "a".repeat(64),
            candidate_release_subject: "v0.1.36-test".to_string(),
            core_commit: "0123456789abcdef0123456789abcdef01234567".to_string(),
            catalog_digests: ShortAudioCatalogDigests {
                inventory_sha256: "1".repeat(64),
                model_catalog_sha256: "2".repeat(64),
                backend_catalog_sha256: "3".repeat(64),
            },
            family: "qwen".to_string(),
            model_id: "funasr-nano".to_string(),
            quant: "q4_k".to_string(),
            topology: "causal-self-attention-kv".to_string(),
            provider: "cuda".to_string(),
            device_target: "sm_89".to_string(),
            backend_id: "cuda-windows-x86_64-test-sm_89".to_string(),
            driver_version: "12.7.0".to_string(),
            artifact_fingerprint: "9".repeat(64),
            device: "cuda0".to_string(),
            actual_provider: Some("cuda".to_string()),
            actual_stable_device_id: Some("cuda0".to_string()),
            actual_device: Some(GgmlActualDeviceFacts {
                device_type: "gpu".to_string(),
                name: "cuda0".to_string(),
                description: "test cuda device".to_string(),
                provider_device_id: Some("0000:01:00.0".to_string()),
                pci_vendor_id: Some(0x10de),
            }),
            placement: "full_device".to_string(),
            capture_mode: ShortAudioCaptureMode::Disabled,
            scheduler_mode: ShortAudioSchedulerMode::Disabled,
            result: "pass".to_string(),
            artifacts: sample_artifacts(),
            output_plan: Some(ShortAudioOutputPlan {
                kind: ShortAudioOutputPlanKind::FullLogits,
                requires_complete_output: true,
                tie_policy: ShortAudioTiePolicy::FirstMaximum,
            }),
            family_oracle: Some(ShortAudioFamilyOracle {
                family: "qwen".to_string(),
                tie_policy: ShortAudioTiePolicy::FirstMaximum,
            }),
            execution: Some(ShortAudioExecutionMode {
                mode,
                graph_rebuild_reason: None,
            }),
            trace: Some(ShortAudioTraceSummary {
                token_trace: ShortAudioArtifactIdentity {
                    label: "token-trace.jsonl".to_string(),
                    sha256: "4".repeat(64),
                    size_bytes: Some(12),
                },
                logits: Some(ShortAudioArtifactIdentity {
                    label: "logits.jsonl".to_string(),
                    sha256: "5".repeat(64),
                    size_bytes: Some(20),
                }),
                top_k: vec![ShortAudioTopKSummary {
                    token_id: 7,
                    value: 1.25,
                }],
                top1_top2_margin: Some(0.5),
            }),
        }
    }

    #[test]
    fn token_evidence_validates_as_a_separate_class() {
        let mut receipt = sample_receipt();
        receipt.evidence = Some(sample_token_evidence(ShortAudioReuseMode::Cold));
        ShortAudioReceipt::try_new(receipt).expect("complete token evidence should validate");
    }

    #[test]
    fn correctness_process_mode_must_match_warmup_and_cache_state() {
        let mut receipt = sample_receipt();
        receipt.evidence = Some(sample_token_evidence(ShortAudioReuseMode::Reuse));
        assert!(matches!(
            receipt.validate(),
            Err(ShortAudioReceiptError::InvalidEvidenceField {
                field: "correctness.execution.mode",
                ..
            })
        ));
        receipt.run.warmup = "warm".to_string();
        receipt.run.cache_state = "populated".to_string();
        ShortAudioReceipt::try_new(receipt)
            .expect("reuse evidence with one warmup and a populated cache should validate");
    }

    #[test]
    fn token_evidence_rejects_missing_trace() {
        let mut receipt = sample_receipt();
        let mut evidence = sample_token_evidence(ShortAudioReuseMode::Reuse);
        evidence.trace = None;
        receipt.evidence = Some(evidence);
        assert!(matches!(
            receipt.validate(),
            Err(ShortAudioReceiptError::TokenEvidenceIncomplete)
        ));
    }

    #[test]
    fn complete_output_evidence_requires_a_full_logits_artifact() {
        let mut receipt = sample_receipt();
        let mut evidence = sample_token_evidence(ShortAudioReuseMode::Cold);
        evidence.trace.as_mut().expect("trace").logits = None;
        receipt.evidence = Some(evidence);
        assert!(matches!(
            receipt.validate_qualification_eligibility(),
            Err(ShortAudioReceiptError::InvalidEvidenceField {
                field: "correctness.trace.logits",
                ..
            })
        ));
    }

    #[test]
    fn formal_evidence_rejects_all_free_form_notes() {
        let mut receipt = sample_receipt();
        receipt.notes.push("reviewed locally".to_string());
        receipt.evidence = Some(sample_token_evidence(ShortAudioReuseMode::Cold));
        assert_eq!(
            receipt.validate(),
            Err(ShortAudioReceiptError::FormalEvidenceNotesNotAllowed)
        );
    }

    #[test]
    fn placement_evidence_cannot_approve_without_observed_placement() {
        let mut receipt = sample_receipt();
        let mut evidence = sample_token_evidence(ShortAudioReuseMode::Cold);
        evidence.evidence_class = ShortAudioEvidenceClass::PlacementResource;
        evidence.output_plan = None;
        evidence.family_oracle = None;
        evidence.execution = None;
        evidence.trace = None;
        receipt.observed_placement = None;
        receipt.evidence = Some(evidence);
        assert!(matches!(
            receipt.validate(),
            Err(ShortAudioReceiptError::PlacementEvidenceMissing)
        ));
    }
    #[test]
    fn roundtrip_json_preserves_receipt() {
        let receipt = ShortAudioReceipt::try_new(sample_receipt()).unwrap();
        let json = receipt.to_pretty_json().unwrap();
        let loaded = ShortAudioReceipt::from_json_str(&json).unwrap();
        assert_eq!(loaded.schema, SHORT_AUDIO_RECEIPT_SCHEMA);
        assert_eq!(loaded.pack.model_id, "funasr-nano:q4_k");
        assert_eq!(loaded.metrics.rtf_median, Some(0.5));
        assert_eq!(loaded.transcript.text, "hello world");
        assert_eq!(
            loaded.transcript.text_sha256,
            sha256_hex_bytes(b"hello world")
        );
        assert!(loaded.execution.is_none());
        assert!(!json.contains("\"execution\""));
    }

    #[test]
    fn receipt_wire_objects_reject_unknown_fields_before_qualification() {
        let mut receipt = sample_receipt();
        receipt.evidence = Some(sample_token_evidence(ShortAudioReuseMode::Cold));
        let value = serde_json::to_value(receipt).expect("serialize fixture");
        for pointer in [
            "",
            "/pack",
            "/audio",
            "/run",
            "/metrics",
            "/transcript",
            "/observed_placement",
            "/evidence",
            "/evidence/catalog_digests",
            "/evidence/artifacts",
            "/evidence/artifacts/binary",
            "/evidence/actual_device",
            "/evidence/output_plan",
            "/evidence/family_oracle",
            "/evidence/execution",
            "/evidence/trace",
            "/evidence/trace/top_k/0",
            "/decode_diagnostics",
        ] {
            let mut candidate = value.clone();
            candidate
                .pointer_mut(pointer)
                .and_then(serde_json::Value::as_object_mut)
                .expect("fixture object")
                .insert(
                    "private_local_path".to_string(),
                    serde_json::json!("/home/alice/private"),
                );
            let error = ShortAudioReceipt::from_json_str(
                &serde_json::to_string(&candidate).expect("serialize candidate"),
            )
            .expect_err("unknown field must fail closed");
            assert!(
                error.to_string().contains("unknown field"),
                "pointer {pointer} was not rejected by strict deserialization: {error}"
            );
        }
    }

    #[test]
    fn receipt_loader_rejects_unknown_graph_lifecycle_event_fields() {
        let mut value = serde_json::to_value(sample_receipt()).expect("serialize fixture");
        value["graph_lifecycle"] = serde_json::json!({
            "events": [{
                "schema": crate::ggml_runtime::GGML_GRAPH_LIFECYCLE_SCHEMA,
                "sequence": 1,
                "provider": "cpu",
                "device": "CPU",
                "graph_instance": 1,
                "graph_generation": 2,
                "event": "created",
                "scheduler_enabled": false,
                "private_local_path": "/home/alice/private"
            }],
            "overflowed": false
        });
        let error = ShortAudioReceipt::from_json_str(
            &serde_json::to_string(&value).expect("serialize candidate"),
        )
        .expect_err("unknown lifecycle field must fail closed");
        assert!(
            error
                .to_string()
                .contains("unknown or missing serialized lifecycle fields")
        );
    }

    #[test]
    fn event_overflow_is_truthful_without_invalidating_live_reconciliation() {
        let attempt = RequestAttemptId::parse("00112233445566778899aabbccddeeff").unwrap();
        let request_receipt =
            crate::models::request_execution_receipt::NativeExecutionReceiptCollector::new();
        request_receipt.bind_request_attempt(attempt);
        request_receipt.record_terminal(crate::RequestExecutionTerminal::Succeeded);
        let _request =
            crate::models::native_execution_services::install_execution_receipt_collector(Some(
                request_receipt.clone(),
            ));
        let runtime = crate::models::runtime_receipts::RuntimeReceiptCollector::new_for_test(
            crate::models::native_execution_services::NativeExecutionScopeId::next(),
            1,
        )
        .unwrap();
        let descriptor = runtime
            .host_neutral_owner_descriptor("overflow-owner", None, None)
            .unwrap();
        let owner = runtime.start_owner(descriptor, None);
        drop(owner);
        let runtime_snapshot = runtime.snapshot();
        let broker = crate::device::execution_memory::DeviceMemoryBrokerSet::new(
            crate::device::execution_memory::DeviceMemoryPolicy::default(),
        );
        let reconciliation = runtime.reconcile_live_leases(&broker);
        assert_eq!(reconciliation, LeaseReceiptShadow::Matched);

        let execution = ShortAudioExecutionProjection::from_receipts(
            &request_receipt.snapshot(),
            &runtime_snapshot,
            &reconciliation,
        );
        assert!(execution.live_state_complete);
        assert!(!execution.event_history_complete);
        assert_eq!(
            execution.event_history_reason.as_deref(),
            Some("event-capacity-exceeded")
        );
        assert!(execution.dropped_events > 0);
        assert_eq!(
            execution.live_lease_reconciliation,
            ShortAudioLeaseReconciliation::Matched
        );

        let mut receipt = sample_receipt();
        receipt.execution = Some(execution);
        let receipt = ShortAudioReceipt::try_new(receipt).unwrap();
        let reloaded =
            ShortAudioReceipt::from_json_str(&receipt.to_pretty_json().unwrap()).unwrap();
        assert!(!reloaded.execution.unwrap().event_history_complete);

        let mut qualification = sample_receipt();
        qualification.evidence = Some(sample_token_evidence(ShortAudioReuseMode::Cold));
        let mut execution = ShortAudioExecutionProjection::from_receipts(
            &request_receipt.snapshot(),
            &runtime_snapshot,
            &reconciliation,
        );
        execution.timing_complete = true;
        execution.request_receipt_complete = true;
        execution.phase_duration_micros = BTreeMap::from([
            ("upload-ingest".to_string(), 1),
            ("decode-normalize".to_string(), 1),
            ("admission-wait".to_string(), 1),
            ("compute".to_string(), 1),
        ]);
        qualification.execution = Some(execution);
        assert_eq!(
            qualification.validate_qualification_eligibility(),
            Err(ShortAudioReceiptError::QualificationIneligible {
                reason: "runtime event history is incomplete",
            })
        );
    }

    #[test]
    fn legacy_paths_are_readable_but_cannot_be_republished_or_qualified() {
        let mut legacy = sample_receipt();
        legacy.audio.path_or_label = "C:\\Users\\alice\\fixture.wav".to_string();
        legacy.run.command = vec![
            "openasr".to_string(),
            "bench-receipt".to_string(),
            "short-audio".to_string(),
            "--audio".to_string(),
            "/home/alice/fixture.wav".to_string(),
        ];
        legacy.run.env_allowlist = BTreeMap::from([(
            "OPENASR_HOME".to_string(),
            "/home/alice/.openasr".to_string(),
        )]);
        let raw = serde_json::to_string(&legacy).unwrap();
        let loaded = ShortAudioReceipt::from_json_str(&raw).expect("legacy v0 remains readable");
        assert!(matches!(
            loaded.validate(),
            Err(ShortAudioReceiptError::InvalidPrivacyProjection { .. })
        ));
        assert!(loaded.to_pretty_json().is_err());
        assert!(loaded.validate_qualification_eligibility().is_err());
    }

    #[test]
    fn new_receipts_reject_posix_windows_unc_paths_and_untyped_environment_values() {
        for raw_path in [
            "/home/alice/fixture.wav",
            "C:\\Users\\alice\\fixture.wav",
            "\\\\server\\share\\fixture.wav",
        ] {
            let mut receipt = sample_receipt();
            receipt.audio.path_or_label = raw_path.to_string();
            assert!(matches!(
                receipt.validate(),
                Err(ShortAudioReceiptError::InvalidPrivacyProjection {
                    field: "audio.path_or_label",
                    ..
                })
            ));

            let mut receipt = sample_receipt();
            receipt
                .run
                .command
                .extend(["--audio".to_string(), raw_path.to_string()]);
            assert!(matches!(
                receipt.validate(),
                Err(ShortAudioReceiptError::InvalidPrivacyProjection {
                    field: "run.command",
                    ..
                })
            ));

            let mut receipt = sample_receipt();
            receipt.scope = raw_path.to_string();
            assert!(matches!(
                receipt.validate(),
                Err(ShortAudioReceiptError::InvalidPrivacyProjection { field: "scope", .. })
            ));
        }

        let mut receipt = sample_receipt();
        receipt.run.env_allowlist.insert(
            "OPENASR_GGML_BACKEND".to_string(),
            "/home/alice/private".to_string(),
        );
        assert!(matches!(
            receipt.validate(),
            Err(ShortAudioReceiptError::InvalidPrivacyProjection {
                field: "run.env_allowlist",
                ..
            })
        ));
    }

    #[test]
    fn hardware_runner_scope_accepts_one_nonce_segment_but_not_path_traversal() {
        let mut receipt = sample_receipt();
        receipt.scope = format!("{}/{}", SHORT_AUDIO_RECEIPT_DEFAULT_SCOPE, "a".repeat(32));
        receipt.validate().expect("runner nonce scope");

        for scope in [
            "../0123456789abcdef0123456789abcdef",
            "/home/alice",
            "C:\\Users\\alice",
            "\\\\server\\share",
            "scope/not-a-nonce",
            "scope/0123456789abcdef0123456789abcdef/extra",
        ] {
            let mut receipt = sample_receipt();
            receipt.scope = scope.to_string();
            assert!(matches!(
                receipt.validate(),
                Err(ShortAudioReceiptError::InvalidPrivacyProjection { field: "scope", .. })
            ));
        }
    }

    #[test]
    fn hybrid_cpu_decoder_lifecycle_is_not_route_drift_from_vulkan_encoder() {
        use crate::ggml_runtime::{
            GGML_GRAPH_LIFECYCLE_SCHEMA, GgmlGraphLifecycleEvent, GgmlGraphLifecycleEventKind,
            GgmlGraphLifecycleSnapshot,
        };
        let compute = |provider: &str, device: &str, sequence: u64| GgmlGraphLifecycleEvent {
            schema: GGML_GRAPH_LIFECYCLE_SCHEMA.into(),
            sequence,
            provider: provider.into(),
            device: device.into(),
            graph_instance: 1,
            graph_generation: 1,
            kind: GgmlGraphLifecycleEventKind::ComputeStarted {
                compute_sequence: sequence,
                prepare_generation: None,
                input_generation_consumed: None,
                capture_executable_generation: None,
            },
        };
        let mut receipt = sample_receipt();
        receipt.placement = "vulkan".to_string();
        receipt.actual_provider = Some("vulkan".to_string());
        receipt.actual_stable_device_id = Some("Vulkan0".to_string());
        receipt.actual_device = Some(GgmlActualDeviceFacts {
            device_type: "gpu".to_string(),
            name: "Vulkan0".to_string(),
            description: "test vulkan device".to_string(),
            provider_device_id: Some("0000:03:00.0".to_string()),
            pci_vendor_id: Some(0x1002),
        });
        receipt.graph_lifecycle = Some(GgmlGraphLifecycleSnapshot {
            events: vec![compute("vulkan", "Vulkan0", 1), compute("cpu", "CPU", 2)],
            overflowed: false,
        });
        ShortAudioReceipt::try_new(receipt).expect("hybrid CPU decoder events are the host half");

        let mut drifted = sample_receipt();
        drifted.placement = "vulkan".to_string();
        drifted.actual_provider = Some("vulkan".to_string());
        drifted.actual_stable_device_id = Some("Vulkan0".to_string());
        drifted.actual_device = Some(GgmlActualDeviceFacts {
            device_type: "gpu".to_string(),
            name: "Vulkan0".to_string(),
            description: "test vulkan device".to_string(),
            provider_device_id: Some("0000:03:00.0".to_string()),
            pci_vendor_id: Some(0x1002),
        });
        drifted.graph_lifecycle = Some(GgmlGraphLifecycleSnapshot {
            events: vec![compute("cuda", "CUDA0", 1)],
            overflowed: false,
        });
        assert!(matches!(
            ShortAudioReceipt::try_new(drifted),
            Err(ShortAudioReceiptError::InvalidEvidenceField {
                field: "actual_device",
                ..
            })
        ));
    }

    #[test]
    fn actual_device_facts_are_bounded_and_route_consistent() {
        let mut receipt = sample_receipt();
        receipt.actual_provider = Some("cpu".to_string());
        receipt.actual_stable_device_id = Some("CPU".to_string());
        receipt.actual_device = Some(GgmlActualDeviceFacts {
            device_type: "cpu".to_string(),
            name: "CPU".to_string(),
            description: "x".repeat(257),
            provider_device_id: None,
            pci_vendor_id: None,
        });
        assert!(matches!(
            ShortAudioReceipt::try_new(receipt),
            Err(ShortAudioReceiptError::InvalidEvidenceField {
                field: "actual_device",
                ..
            })
        ));
    }

    #[test]
    fn missing_required_field_fails_validation() {
        let mut receipt = sample_receipt();
        receipt.core_commit.clear();
        let err = receipt.validate().unwrap_err();
        assert!(matches!(
            err,
            ShortAudioReceiptError::EmptyField {
                field: "core_commit"
            }
        ));
    }

    #[test]
    fn wrong_schema_fails_validation() {
        let mut receipt = sample_receipt();
        receipt.schema = "openasr.model-pack-preflight.v1".to_string();
        let err = receipt.validate().unwrap_err();
        assert!(matches!(err, ShortAudioReceiptError::SchemaMismatch { .. }));
    }

    #[test]
    fn transcript_sha_mismatch_fails() {
        let mut receipt = sample_receipt();
        receipt.transcript.text_sha256 = "c".repeat(64);
        let err = receipt.validate().unwrap_err();
        assert!(matches!(
            err,
            ShortAudioReceiptError::InvalidTranscriptSha256 { .. }
        ));
    }

    #[test]
    fn empty_rtf_samples_are_allowed() {
        let mut receipt = sample_receipt();
        receipt.metrics.rtf_samples.clear();
        receipt.metrics.rtf_median = None;
        ShortAudioReceipt::try_new(receipt).unwrap();
    }

    #[test]
    fn median_without_samples_fails() {
        let mut receipt = sample_receipt();
        receipt.metrics.rtf_samples.clear();
        receipt.metrics.rtf_median = Some(0.5);
        let err = receipt.validate().unwrap_err();
        assert!(matches!(err, ShortAudioReceiptError::MedianWithoutSamples));
    }

    #[test]
    fn loaded_rtf_samples_require_the_fixed_measurement_method() {
        for method in [None, Some("process_cpu_time".to_string())] {
            let mut receipt = sample_receipt();
            receipt.metrics.measurement_method = method.clone();
            let raw = serde_json::to_string(&receipt).expect("serialize fixture");
            assert!(matches!(
                ShortAudioReceipt::from_json_str(&raw),
                Err(ShortAudioReceiptLoadError::Validate(
                    ShortAudioReceiptError::InvalidMeasurementMethod { actual, .. }
                )) if actual == method
            ));
        }
    }

    #[test]
    fn receipt_metrics_must_be_finite_and_non_negative() {
        let mut invalid = Vec::new();
        for value in [-1.0, f64::NAN, f64::INFINITY] {
            let mut receipt = sample_receipt();
            receipt.metrics.rtf_samples = vec![value];
            receipt.metrics.rtf_median = Some(value);
            invalid.push(receipt);

            let mut receipt = sample_receipt();
            receipt.metrics.wer_or_cer = Some(value);
            invalid.push(receipt);

            let mut receipt = sample_receipt();
            receipt.metrics.ttft_s = Some(value);
            invalid.push(receipt);

            let mut receipt = sample_receipt();
            receipt.audio.duration_s = Some(value);
            invalid.push(receipt);
        }
        for receipt in invalid {
            assert!(matches!(
                receipt.validate(),
                Err(ShortAudioReceiptError::InvalidMetric { .. })
            ));
        }
        assert_eq!(median_f64(&[1.0, f64::NAN]), None);
        assert_eq!(median_f64(&[f64::MAX, f64::MAX]), Some(f64::MAX));
    }

    #[test]
    fn median_helper_handles_even_and_odd() {
        assert_eq!(median_f64(&[]), None);
        assert_eq!(median_f64(&[3.0]), Some(3.0));
        assert_eq!(median_f64(&[1.0, 3.0, 2.0]), Some(2.0));
        assert_eq!(median_f64(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
    }

    #[test]
    fn sha256_file_matches_bytes_helper() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"short-audio-receipt").unwrap();
        tmp.flush().unwrap();
        let (size, hex) = sha256_file(tmp.path()).unwrap();
        assert_eq!(size, b"short-audio-receipt".len() as u64);
        assert_eq!(hex, sha256_hex_bytes(b"short-audio-receipt"));
    }

    #[test]
    fn resolve_core_commit_accepts_explicit_sha() {
        let sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        assert_eq!(resolve_core_commit(Some(sha), None).unwrap(), sha);
    }

    #[test]
    fn resolve_core_commit_rejects_short_sha() {
        let err = resolve_core_commit(Some("abc"), None).unwrap_err();
        assert!(matches!(
            err,
            ShortAudioReceiptError::InvalidCoreCommit { .. }
        ));
    }

    #[test]
    fn try_new_fills_median_and_measurement_method() {
        let mut receipt = sample_receipt();
        receipt.metrics.rtf_median = None;
        receipt.metrics.measurement_method = None;
        let built = ShortAudioReceipt::try_new(receipt).unwrap();
        assert_eq!(built.metrics.rtf_median, Some(0.5));
        assert_eq!(
            built.metrics.measurement_method.as_deref(),
            Some(SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK)
        );
    }

    #[test]
    fn decode_diagnostics_are_required_fail_closed() {
        let mut receipt = sample_receipt();
        receipt.decode_diagnostics = None;
        assert!(matches!(
            receipt.validate(),
            Err(ShortAudioReceiptError::DecodeDiagnosticsMissing)
        ));
        let json = serde_json::to_string(&receipt).unwrap();
        assert!(
            !json.contains("decode_diagnostics"),
            "absent diagnostics must not serialize as a placeholder"
        );
        assert!(ShortAudioReceipt::from_json_str(&json).is_err());
    }

    #[test]
    fn decode_diagnostics_bind_output_plan_and_reuse_mode() {
        let receipt = ShortAudioReceipt::try_new(sample_receipt()).unwrap();
        let diagnostics = receipt.decode_diagnostics.as_ref().expect("required");
        assert_eq!(
            diagnostics.output_plan,
            ShortAudioReceiptOutputPlan::FullLogits
        );
        assert_eq!(
            diagnostics.reuse_mode,
            ShortAudioReceiptReuseMode::FreshGraph
        );
        let json = receipt.to_pretty_json().unwrap();
        assert!(json.contains("\"output_plan\": \"full_logits\""));
        assert!(json.contains("\"reuse_mode\": \"fresh_graph\""));
        assert!(!json.contains("raw_audio"));
        assert!(!json.contains("weights"));
        assert!(!json.contains("secret"));
    }

    #[test]
    fn placement_evidence_is_not_token_transcript_proof() {
        let mut receipt = sample_receipt();
        let mut evidence = sample_token_evidence(ShortAudioReuseMode::Cold);
        evidence.evidence_class = ShortAudioEvidenceClass::PlacementResource;
        evidence.output_plan = None;
        evidence.family_oracle = None;
        evidence.execution = Some(ShortAudioExecutionMode {
            mode: ShortAudioReuseMode::Cold,
            graph_rebuild_reason: None,
        });
        evidence.trace = None;
        receipt.evidence = Some(evidence);
        ShortAudioReceipt::try_new(receipt).expect("placement evidence may omit token fields");
    }

    fn cpu_resolved_runtime() -> ResolvedFamilyRuntimeInput {
        ResolvedFamilyRuntimeInput::resolve(
            Some(crate::ggml_runtime::RequestBackendPreference::CpuOnly),
            crate::ggml_runtime::AutoGpuPolicy::Never,
        )
    }

    #[test]
    fn decode_step_bound_admits_xasr_device_head_frame_count() {
        let diagnostics = crate::ggml_runtime::ShortAudioReceiptDecodeDiagnostics {
            output_plan: crate::ggml_runtime::ShortAudioReceiptOutputPlan::FullLogits,
            reuse_mode: crate::ggml_runtime::ShortAudioReceiptReuseMode::ReusableGraph,
            capability_evidence_revision: Some(2),
            steps: (0..329)
                .map(|step| crate::ggml_runtime::ShortAudioReceiptDecodeStep {
                    step,
                    token_id: Some(0),
                    logits_sha256: None,
                    top2_margin: None,
                    graph_rebuilt: false,
                })
                .collect(),
            first_divergence: None,
            encoder_decoder_splits: Vec::new(),
        };
        super::validate_decode_diagnostics(&diagnostics)
            .expect("329 X-ASR device-head frames must fit the receipt bound");
        const {
            assert!(
                SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS >= 2048,
                "longform CTC/RNN-T frames on the 69s fixture must stay in-bound"
            );
        }
    }

    #[test]
    fn shipped_emitter_projects_observed_created_and_existing_graphs() {
        let resolved = cpu_resolved_runtime();
        let first = crate::NativeExecutionReceiptCollector::new();
        first.begin_candidate_attempt();
        let first_lifecycle = first.graph_lifecycle_collector();
        let first_scope = first_lifecycle.install();
        let mut runner = crate::ggml_runtime::GgmlCpuGraphRunner::new(
            crate::ggml_runtime::GgmlCpuGraphConfig::conservative_default(),
        )
        .expect("CPU graph runner");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_1d_f32(3, "receipt_graph_input")
            .expect("input");
        graph.set_input(input).expect("set input");
        graph.set_output(input).expect("set output");
        graph
            .set_f32_slice(input, &[0.0, 2.0, 1.0], "receipt_graph_input")
            .expect("first input upload");
        let (_, first_compute) = graph
            .compute_output_f32_with_evidence(input, 3)
            .expect("first output read")
            .into_parts();
        first.begin_decode_step(0, first_compute);
        first.record_token(0, 1, false);
        first.finish_decode_step(0);
        drop(first_scope);
        first.finish_candidate_attempt(true);
        let first_snapshot = first.snapshot();
        let first_diagnostics =
            decode_diagnostics_from_shipped_runtime(Some(&resolved), Some(&first_snapshot))
                .expect("created graph projects from its compute lifecycle");
        assert!(first_diagnostics.steps[0].graph_rebuilt);

        let reuse = crate::NativeExecutionReceiptCollector::new();
        reuse.begin_candidate_attempt();
        let reuse_lifecycle = reuse.graph_lifecycle_collector();
        let reuse_scope = reuse_lifecycle.install();
        graph
            .set_f32_slice(input, &[3.0, 1.0, 2.0], "receipt_graph_input")
            .expect("reuse input upload");
        let (_, reuse_compute) = graph
            .compute_output_f32_with_evidence(input, 3)
            .expect("reuse output read")
            .into_parts();
        reuse.begin_decode_step(0, reuse_compute);
        reuse.record_token(0, 0, false);
        reuse.finish_decode_step(0);
        drop(reuse_scope);
        reuse.finish_candidate_attempt(true);
        let reuse_snapshot = reuse.snapshot();
        let reuse_diagnostics =
            decode_diagnostics_from_shipped_runtime(Some(&resolved), Some(&reuse_snapshot))
                .expect("existing graph projects from its request-local lifecycle");
        assert!(!reuse_diagnostics.steps[0].graph_rebuilt);
    }

    #[test]
    fn shipped_emitter_binds_compute_to_origin_in_effect_not_snapshot_xor() {
        let resolved = cpu_resolved_runtime();
        let collector = crate::NativeExecutionReceiptCollector::new();
        collector.begin_candidate_attempt();
        let lifecycle = collector.graph_lifecycle_collector();
        let scope = lifecycle.install();
        let mut runner = crate::ggml_runtime::GgmlCpuGraphRunner::new(
            crate::ggml_runtime::GgmlCpuGraphConfig::conservative_default(),
        )
        .expect("CPU graph runner");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_1d_f32(3, "receipt_graph_input")
            .expect("input");
        graph.set_input(input).expect("set input");
        graph.set_output(input).expect("set output");
        graph
            .set_f32_slice(input, &[0.0, 2.0, 1.0], "receipt_graph_input")
            .expect("first input upload");
        graph
            .compute_output_f32_with_evidence(input, 3)
            .expect("created-graph compute");
        collector.finish_candidate_attempt(true);
        collector.begin_candidate_attempt();
        graph
            .set_f32_slice(input, &[3.0, 1.0, 2.0], "receipt_graph_input")
            .expect("re-attached input upload");
        let (_, compute) = graph
            .compute_output_f32_with_evidence(input, 3)
            .expect("existing-graph compute")
            .into_parts();
        collector.begin_decode_step(0, compute);
        collector.record_token(0, 1, false);
        collector.finish_decode_step(0);
        drop(scope);
        collector.finish_candidate_attempt(true);
        let snapshot = collector.snapshot();
        let created = snapshot.graph_lifecycle.events.iter().any(|event| {
            matches!(
                event.kind,
                crate::ggml_runtime::GgmlGraphLifecycleEventKind::Created { .. }
            )
        });
        let attached = snapshot.graph_lifecycle.events.iter().any(|event| {
            matches!(
                event.kind,
                crate::ggml_runtime::GgmlGraphLifecycleEventKind::ExistingGraphObserved { .. }
            )
        });
        assert!(created && attached);
        let diagnostics = decode_diagnostics_from_shipped_runtime(Some(&resolved), Some(&snapshot))
            .expect("compute binds to the origin in effect at that compute");
        assert!(!diagnostics.steps[0].graph_rebuilt);
    }

    #[test]
    fn throttled_hot_path_computes_project_as_reused_not_missing_origin() {
        use std::sync::Arc;

        use crate::ggml_runtime::{
            GgmlGraphLifecycleEvent, GgmlGraphLifecycleEventKind, GgmlGraphLifecycleSnapshot,
            GgmlSelectionEvidenceRef,
        };

        let lifecycle = GgmlGraphLifecycleSnapshot {
            events: vec![
                GgmlGraphLifecycleEvent {
                    schema: Arc::from("openasr.graph-lifecycle.v1"),
                    sequence: 1,
                    provider: Arc::from("hip"),
                    device: Arc::from("HIP0"),
                    graph_instance: 11,
                    graph_generation: 22,
                    kind: GgmlGraphLifecycleEventKind::ExistingGraphObserved {
                        scheduler_enabled: false,
                        prepare_generation: None,
                    },
                },
                GgmlGraphLifecycleEvent {
                    schema: Arc::from("openasr.graph-lifecycle.v1"),
                    sequence: 2,
                    provider: Arc::from("hip"),
                    device: Arc::from("HIP0"),
                    graph_instance: 11,
                    graph_generation: 22,
                    kind: GgmlGraphLifecycleEventKind::ComputeStarted {
                        compute_sequence: 1,
                        prepare_generation: None,
                        input_generation_consumed: None,
                        capture_executable_generation: None,
                    },
                },
                GgmlGraphLifecycleEvent {
                    schema: Arc::from("openasr.graph-lifecycle.v1"),
                    sequence: 3,
                    provider: Arc::from("hip"),
                    device: Arc::from("HIP0"),
                    graph_instance: 11,
                    graph_generation: 22,
                    kind: GgmlGraphLifecycleEventKind::ComputeCompleted {
                        compute_sequence: 1,
                        output_generation: 7,
                    },
                },
                GgmlGraphLifecycleEvent {
                    schema: Arc::from("openasr.graph-lifecycle.v1"),
                    sequence: 4,
                    provider: Arc::from("hip"),
                    device: Arc::from("HIP0"),
                    graph_instance: 11,
                    graph_generation: 22,
                    kind: GgmlGraphLifecycleEventKind::OutputRead {
                        compute_sequence: 1,
                        output_generation_consumed: 7,
                        bytes: 16,
                    },
                },
            ],
            overflowed: false,
        };
        let step = crate::NativeExecutionTokenStep {
            step_index: 8,
            token_id: 3,
            is_eot: false,
            top2_margin: None,
            logits_sha256: None,
            compute: Some(GgmlSelectionEvidenceRef::from_parts_for_test(11, 22, 8, 99)),
        };
        assert!(
            !super::observed_graph_built_for_compute(&step, Some(&lifecycle))
                .expect("later HIP/Vulkan token steps must bind as reuse after hot-path throttle")
        );
    }

    #[test]
    fn shipped_emitter_rejects_tokens_without_native_compute_witness() {
        let resolved = cpu_resolved_runtime();
        let collector = crate::NativeExecutionReceiptCollector::new();
        collector.record_top_k(0, &[2.0, 1.0]);
        collector.record_token(0, 11, false);
        collector.record_token(1, 7, true);
        let snapshot = collector.snapshot();
        let error = decode_diagnostics_from_shipped_runtime(Some(&resolved), Some(&snapshot))
            .expect_err("caller-recorded tokens cannot synthesize graph lifecycle evidence");
        assert!(matches!(
            error,
            ShortAudioReceiptError::InvalidEvidenceField {
                field: "decode_diagnostics.steps.graph_rebuilt",
                ..
            }
        ));
    }

    #[test]
    fn shipped_emitter_fail_closed_without_resolved_runtime() {
        let error = decode_diagnostics_from_shipped_runtime(None, None)
            .expect_err("missing shipped plan/reuse must not emit");
        assert_eq!(error, ShortAudioReceiptError::DecodeDiagnosticsMissing);
    }
}
