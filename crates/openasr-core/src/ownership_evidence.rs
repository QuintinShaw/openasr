//! Artifact-bound release evidence over ordered runtime ownership snapshots.
//!
//! openasr.runtime-ownership-receipt.v1 remains the production diagnostic
//! snapshot. This module binds hashes of those snapshots and adjacent request /
//! activation / pressure-helper receipts to immutable release artifacts and a
//! causal phase sequence. Admission and runtime policy never consume it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const OWNERSHIP_EVIDENCE_SCHEMA: &str = "openasr.runtime-ownership-evidence.v1";
pub const OWNERSHIP_ACTIVATION_RECEIPT_SCHEMA: &str = "openasr.model-activation-receipt.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OwnershipEvidenceScenario {
    ColdWarmLifecycle,
    DeterministicPressureRace,
    RealHostPressureRollback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OwnershipEvidencePhaseKind {
    BaselineAdmissible,
    OldRuntimeActive,
    ColdRequestCompleted,
    WarmRequestCompleted,
    ForecastSucceeded,
    FactsChanged,
    PressureReady,
    ActivationRejected,
    OldRuntimeTranscribed,
    Reconciled,
    OwnerReleased,
    PressureReleased,
    Recovered,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OwnershipLeaseReconciliationStatus {
    Matched,
    Mismatched,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnershipEvidenceArtifact {
    pub label: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnershipReleaseBinding {
    pub release_subject: String,
    pub core_commit: String,
    pub host_abi_fingerprint: String,
    pub binary_sha256: String,
    pub plugin_sha256: String,
    pub pack_sha256: String,
    pub catalog_sha256: String,
    pub catalog_signature_sha256: String,
    pub capability_matrix_sha256: String,
    pub capability_epoch: u64,
    pub capability_cell_sha256: String,
    pub family: String,
    pub model_id: String,
    pub quant: String,
    pub topology: String,
    pub provider: String,
    pub device_target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approved_target_set_sha256: Option<String>,
    pub placement: String,
    pub output_plan: String,
    pub reuse_mode: String,
    pub capture_mode: String,
    pub scheduler_mode: String,
    pub evidence_revision: u64,
    pub activation_mode: String,
}

/// Bounded production fact emitted by the shared model-activation transaction.
/// The ownership envelope hashes this artifact; release tooling validates it
/// against the exact candidate and immutable release binding. It contains no
/// model path or raw runtime/device identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnershipActivationReceipt {
    pub schema: String,
    pub result: String,
    pub daemon_start_identity: OwnershipDaemonStartIdentity,
    pub release_subject: String,
    pub core_commit: String,
    pub pack_sha256: String,
    pub capability_matrix_sha256: String,
    pub capability_epoch: u64,
    pub provider: String,
    pub device_target: String,
    pub failure_stage: String,
    pub fresh_reserve: OwnershipAdmissionObservation,
    pub durable_selection_before_sha256: String,
    pub durable_selection_after_sha256: String,
    pub live_runtime_before_sha256: String,
    pub live_runtime_after_sha256: String,
    pub staged_owner_cleanup: String,
}

impl OwnershipActivationReceipt {
    pub fn try_new(mut receipt: Self) -> Result<Self, OwnershipEvidenceError> {
        receipt.schema = OWNERSHIP_ACTIVATION_RECEIPT_SCHEMA.to_string();
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn validate(&self) -> Result<(), OwnershipEvidenceError> {
        if self.schema != OWNERSHIP_ACTIVATION_RECEIPT_SCHEMA {
            return Err(OwnershipEvidenceError::ActivationReceiptSchemaMismatch);
        }
        if self.result != "rejected" || self.failure_stage != "fresh_reserve" {
            return Err(OwnershipEvidenceError::ActivationReceiptNotFreshReserveRejection);
        }
        if self.daemon_start_identity.pid == 0
            || self.daemon_start_identity.started_at_unix_secs == 0
        {
            return Err(OwnershipEvidenceError::InvalidDaemonIdentity);
        }
        require_lower_hex(
            "activation.daemon_start_identity.nonce",
            &self.daemon_start_identity.nonce,
            32,
        )?;
        require_non_empty("activation.release_subject", &self.release_subject)?;
        require_lower_hex("activation.core_commit", &self.core_commit, 40)?;
        for (field, value) in [
            ("activation.pack_sha256", self.pack_sha256.as_str()),
            (
                "activation.capability_matrix_sha256",
                self.capability_matrix_sha256.as_str(),
            ),
            (
                "activation.durable_selection_before_sha256",
                self.durable_selection_before_sha256.as_str(),
            ),
            (
                "activation.durable_selection_after_sha256",
                self.durable_selection_after_sha256.as_str(),
            ),
            (
                "activation.live_runtime_before_sha256",
                self.live_runtime_before_sha256.as_str(),
            ),
            (
                "activation.live_runtime_after_sha256",
                self.live_runtime_after_sha256.as_str(),
            ),
        ] {
            require_lower_hex(field, value, 64)?;
        }
        if self.capability_epoch == 0 {
            return Err(OwnershipEvidenceError::InvalidField {
                field: "activation.capability_epoch",
            });
        }
        require_one_of(
            "activation.provider",
            &self.provider,
            &["cpu", "metal", "cuda", "hip", "vulkan"],
        )?;
        require_non_empty("activation.device_target", &self.device_target)?;
        self.fresh_reserve.validate()?;
        if !self.fresh_reserve.crosses_rejection_threshold() {
            return Err(OwnershipEvidenceError::ActivationReceiptNotFreshReserveRejection);
        }
        if self.durable_selection_before_sha256 != self.durable_selection_after_sha256
            || self.live_runtime_before_sha256 != self.live_runtime_after_sha256
            || self.staged_owner_cleanup != "released"
        {
            return Err(OwnershipEvidenceError::ActivationReceiptDidNotPreserveOldState);
        }
        Ok(())
    }

    pub fn from_json_str(raw: &str) -> Result<Self, OwnershipActivationReceiptLoadError> {
        let receipt: Self = serde_json::from_str(raw)?;
        receipt.validate()?;
        Ok(receipt)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnershipDaemonStartIdentity {
    pub pid: u32,
    pub nonce: String,
    pub started_at_unix_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnershipAdmissionObservation {
    /// Digest of the exact pack/lane/artifact/capability cell being attempted.
    pub candidate_sha256: String,
    /// Full policy-ledger charge for the exact candidate.
    pub policy_requested_bytes: u64,
    /// Remaining policy capacity after live pending/committed/unreclaimable owners.
    pub policy_remaining_bytes: u64,
    /// Bytes that must fit in the fresh native/OS observation. This may be
    /// smaller than the policy charge for reclaimable file-backed residency.
    pub observed_requested_bytes: u64,
    /// Fresh native/OS capacity remaining at the actual reserve attempt.
    pub observed_remaining_bytes: u64,
}

impl OwnershipAdmissionObservation {
    pub fn is_admissible(&self) -> bool {
        self.policy_requested_bytes <= self.policy_remaining_bytes
            && self.observed_requested_bytes <= self.observed_remaining_bytes
    }

    pub fn crosses_rejection_threshold(&self) -> bool {
        self.policy_requested_bytes > self.policy_remaining_bytes
            || self.observed_requested_bytes > self.observed_remaining_bytes
    }

    fn validate(&self) -> Result<(), OwnershipEvidenceError> {
        require_lower_hex("admission.candidate_sha256", &self.candidate_sha256, 64)?;
        if self.policy_requested_bytes == 0 {
            return Err(OwnershipEvidenceError::InvalidField {
                field: "admission.policy_requested_bytes",
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnershipCandidateObservation {
    pub admission: OwnershipAdmissionObservation,
    pub safety_floor_bytes: u64,
    pub helper_committed_bytes: u64,
    pub helper_touched_bytes: u64,
}

impl OwnershipCandidateObservation {
    pub fn is_admissible(&self) -> bool {
        self.admission.is_admissible()
    }

    pub fn crosses_rejection_threshold(&self) -> bool {
        self.admission.crosses_rejection_threshold()
    }

    pub fn crosses_observed_rejection_threshold(&self) -> bool {
        self.admission.observed_requested_bytes > self.admission.observed_remaining_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnershipEvidencePhase {
    pub ordinal: u32,
    pub kind: OwnershipEvidencePhaseKind,
    pub daemon_start_identity: OwnershipDaemonStartIdentity,
    pub runtime_snapshot: OwnershipEvidenceArtifact,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_receipt: Option<OwnershipEvidenceArtifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_receipt: Option<OwnershipEvidenceArtifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pressure_helper_receipt: Option<OwnershipEvidenceArtifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observation: Option<OwnershipCandidateObservation>,
    pub lease_reconciliation: OwnershipLeaseReconciliationStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnershipEvidenceEnvelope {
    pub schema: String,
    pub scenario: OwnershipEvidenceScenario,
    pub result: String,
    pub release: OwnershipReleaseBinding,
    pub phases: Vec<OwnershipEvidencePhase>,
}

impl OwnershipEvidenceEnvelope {
    pub fn try_new(mut envelope: Self) -> Result<Self, OwnershipEvidenceError> {
        envelope.schema = OWNERSHIP_EVIDENCE_SCHEMA.to_string();
        envelope.validate()?;
        Ok(envelope)
    }

    pub fn validate(&self) -> Result<(), OwnershipEvidenceError> {
        if self.schema != OWNERSHIP_EVIDENCE_SCHEMA {
            return Err(OwnershipEvidenceError::SchemaMismatch);
        }
        if self.result != "pass" {
            return Err(OwnershipEvidenceError::NonPassingResult);
        }
        self.release.validate()?;
        if self.phases.is_empty() {
            return Err(OwnershipEvidenceError::MissingPhases);
        }
        for (index, phase) in self.phases.iter().enumerate() {
            if phase.ordinal != index as u32 {
                return Err(OwnershipEvidenceError::NonContiguousPhaseOrder);
            }
            phase.validate()?;
        }
        if self.phases.iter().any(|phase| {
            phase.observation.as_ref().is_some_and(|observation| {
                observation.admission.candidate_sha256 != self.release.capability_cell_sha256
            })
        }) {
            return Err(OwnershipEvidenceError::CandidateIdentityChanged);
        }
        self.validate_artifact_bindings()?;
        match self.scenario {
            OwnershipEvidenceScenario::ColdWarmLifecycle => self.validate_cold_warm_lifecycle(),
            OwnershipEvidenceScenario::DeterministicPressureRace => {
                self.validate_deterministic_pressure_race()
            }
            OwnershipEvidenceScenario::RealHostPressureRollback => {
                self.validate_real_host_pressure()
            }
        }
    }

    pub fn to_pretty_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json_str(raw: &str) -> Result<Self, OwnershipEvidenceLoadError> {
        let envelope: Self = serde_json::from_str(raw)?;
        envelope.validate()?;
        Ok(envelope)
    }

    /// All external artifacts referenced by this envelope, deduplicated by
    /// their safe release-asset label. The envelope remains data-only; release
    /// tooling rehashes these files before accepting the evidence bundle.
    pub fn artifact_bindings(&self) -> Vec<&OwnershipEvidenceArtifact> {
        let mut artifacts = BTreeMap::<&str, &OwnershipEvidenceArtifact>::new();
        for phase in &self.phases {
            for artifact in [
                Some(&phase.runtime_snapshot),
                phase.request_receipt.as_ref(),
                phase.activation_receipt.as_ref(),
                phase.pressure_helper_receipt.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                artifacts.entry(artifact.label.as_str()).or_insert(artifact);
            }
        }
        artifacts.into_values().collect()
    }

    fn validate_artifact_bindings(&self) -> Result<(), OwnershipEvidenceError> {
        let mut digests = BTreeMap::<&str, &str>::new();
        for phase in &self.phases {
            for artifact in [
                Some(&phase.runtime_snapshot),
                phase.request_receipt.as_ref(),
                phase.activation_receipt.as_ref(),
                phase.pressure_helper_receipt.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                if let Some(previous) = digests.insert(&artifact.label, &artifact.sha256)
                    && previous != artifact.sha256
                {
                    return Err(OwnershipEvidenceError::ArtifactDigestConflict);
                }
            }
        }
        Ok(())
    }

    fn validate_cold_warm_lifecycle(&self) -> Result<(), OwnershipEvidenceError> {
        self.require_phase_sequence(&[
            OwnershipEvidencePhaseKind::BaselineAdmissible,
            OwnershipEvidencePhaseKind::ColdRequestCompleted,
            OwnershipEvidencePhaseKind::WarmRequestCompleted,
            OwnershipEvidencePhaseKind::OwnerReleased,
            OwnershipEvidencePhaseKind::Reconciled,
        ])?;
        self.require_one_daemon()?;
        let baseline = self.require_observation(OwnershipEvidencePhaseKind::BaselineAdmissible)?;
        if !baseline.is_admissible() {
            return Err(OwnershipEvidenceError::BaselineNotAdmissible);
        }
        for kind in [
            OwnershipEvidencePhaseKind::ColdRequestCompleted,
            OwnershipEvidencePhaseKind::WarmRequestCompleted,
        ] {
            let phase = self.phase(kind)?;
            if phase.request_receipt.is_none() {
                return Err(OwnershipEvidenceError::MissingRequestReceipt { phase: kind });
            }
        }
        self.require_all_leases_matched()
    }

    fn validate_deterministic_pressure_race(&self) -> Result<(), OwnershipEvidenceError> {
        self.require_phase_sequence(&[
            OwnershipEvidencePhaseKind::BaselineAdmissible,
            OwnershipEvidencePhaseKind::ForecastSucceeded,
            OwnershipEvidencePhaseKind::FactsChanged,
            OwnershipEvidencePhaseKind::ActivationRejected,
            OwnershipEvidencePhaseKind::OldRuntimeTranscribed,
            OwnershipEvidencePhaseKind::Reconciled,
            OwnershipEvidencePhaseKind::Recovered,
        ])?;
        self.require_one_daemon()?;
        let baseline = self.require_observation(OwnershipEvidencePhaseKind::BaselineAdmissible)?;
        let changed = self.require_observation(OwnershipEvidencePhaseKind::FactsChanged)?;
        let recovered = self.require_observation(OwnershipEvidencePhaseKind::Recovered)?;
        self.require_same_candidate([baseline, changed, recovered])?;
        if !baseline.is_admissible() {
            return Err(OwnershipEvidenceError::BaselineNotAdmissible);
        }
        if !changed.crosses_rejection_threshold() {
            return Err(OwnershipEvidenceError::PressureDidNotCrossThreshold);
        }
        if !recovered.is_admissible() {
            return Err(OwnershipEvidenceError::ObservationDidNotRecover);
        }
        self.require_rejection_and_old_runtime_proof()?;
        self.require_all_leases_matched()
    }

    fn validate_real_host_pressure(&self) -> Result<(), OwnershipEvidenceError> {
        self.require_phase_sequence(&[
            OwnershipEvidencePhaseKind::BaselineAdmissible,
            OwnershipEvidencePhaseKind::OldRuntimeActive,
            OwnershipEvidencePhaseKind::PressureReady,
            OwnershipEvidencePhaseKind::ActivationRejected,
            OwnershipEvidencePhaseKind::OldRuntimeTranscribed,
            OwnershipEvidencePhaseKind::Reconciled,
            OwnershipEvidencePhaseKind::PressureReleased,
            OwnershipEvidencePhaseKind::Recovered,
        ])?;
        self.require_one_daemon()?;
        let baseline = self.require_observation(OwnershipEvidencePhaseKind::BaselineAdmissible)?;
        let pressured = self.require_observation(OwnershipEvidencePhaseKind::PressureReady)?;
        let recovered = self.require_observation(OwnershipEvidencePhaseKind::Recovered)?;
        self.require_same_candidate([baseline, pressured, recovered])?;
        if !baseline.is_admissible() {
            return Err(OwnershipEvidenceError::BaselineNotAdmissible);
        }
        if !pressured.crosses_observed_rejection_threshold()
            || pressured.helper_committed_bytes == 0
            || pressured.helper_touched_bytes == 0
        {
            return Err(OwnershipEvidenceError::PressureDidNotCrossThreshold);
        }
        if pressured.admission.observed_remaining_bytes < pressured.safety_floor_bytes {
            return Err(OwnershipEvidenceError::SafetyFloorViolated);
        }
        if !recovered.is_admissible()
            || recovered.helper_committed_bytes != 0
            || recovered.helper_touched_bytes != 0
        {
            return Err(OwnershipEvidenceError::ObservationDidNotRecover);
        }
        for kind in [
            OwnershipEvidencePhaseKind::PressureReady,
            OwnershipEvidencePhaseKind::PressureReleased,
        ] {
            if self.phase(kind)?.pressure_helper_receipt.is_none() {
                return Err(OwnershipEvidenceError::MissingPressureHelperReceipt { phase: kind });
            }
        }
        self.require_rejection_and_old_runtime_proof()?;
        self.require_all_leases_matched()
    }

    fn require_rejection_and_old_runtime_proof(&self) -> Result<(), OwnershipEvidenceError> {
        if self
            .phase(OwnershipEvidencePhaseKind::ActivationRejected)?
            .activation_receipt
            .is_none()
        {
            return Err(OwnershipEvidenceError::MissingActivationReceipt);
        }
        if self
            .phase(OwnershipEvidencePhaseKind::OldRuntimeTranscribed)?
            .request_receipt
            .is_none()
        {
            return Err(OwnershipEvidenceError::MissingRequestReceipt {
                phase: OwnershipEvidencePhaseKind::OldRuntimeTranscribed,
            });
        }
        Ok(())
    }

    fn require_phase_sequence(
        &self,
        expected: &[OwnershipEvidencePhaseKind],
    ) -> Result<(), OwnershipEvidenceError> {
        let actual = self
            .phases
            .iter()
            .map(|phase| phase.kind)
            .collect::<Vec<_>>();
        if actual != expected {
            return Err(OwnershipEvidenceError::WrongPhaseSequence);
        }
        Ok(())
    }

    fn require_one_daemon(&self) -> Result<(), OwnershipEvidenceError> {
        let first = &self.phases[0].daemon_start_identity;
        if self
            .phases
            .iter()
            .any(|phase| &phase.daemon_start_identity != first)
        {
            return Err(OwnershipEvidenceError::DaemonIdentityChanged);
        }
        Ok(())
    }

    fn phase(
        &self,
        kind: OwnershipEvidencePhaseKind,
    ) -> Result<&OwnershipEvidencePhase, OwnershipEvidenceError> {
        self.phases
            .iter()
            .find(|phase| phase.kind == kind)
            .ok_or(OwnershipEvidenceError::MissingPhase { phase: kind })
    }

    fn require_observation(
        &self,
        kind: OwnershipEvidencePhaseKind,
    ) -> Result<&OwnershipCandidateObservation, OwnershipEvidenceError> {
        self.phase(kind)?
            .observation
            .as_ref()
            .ok_or(OwnershipEvidenceError::MissingObservation { phase: kind })
    }

    fn require_same_candidate<const N: usize>(
        &self,
        observations: [&OwnershipCandidateObservation; N],
    ) -> Result<(), OwnershipEvidenceError> {
        let first = &observations[0].admission.candidate_sha256;
        if observations
            .iter()
            .any(|observation| &observation.admission.candidate_sha256 != first)
        {
            return Err(OwnershipEvidenceError::CandidateIdentityChanged);
        }
        Ok(())
    }

    fn require_all_leases_matched(&self) -> Result<(), OwnershipEvidenceError> {
        if self
            .phases
            .iter()
            .any(|phase| phase.lease_reconciliation != OwnershipLeaseReconciliationStatus::Matched)
        {
            return Err(OwnershipEvidenceError::LeaseReconciliationNotMatched);
        }
        Ok(())
    }
}

impl OwnershipReleaseBinding {
    fn validate(&self) -> Result<(), OwnershipEvidenceError> {
        require_non_empty("release.release_subject", &self.release_subject)?;
        require_lower_hex("release.core_commit", &self.core_commit, 40)?;
        for (field, value) in [
            (
                "release.host_abi_fingerprint",
                self.host_abi_fingerprint.as_str(),
            ),
            ("release.binary_sha256", self.binary_sha256.as_str()),
            ("release.plugin_sha256", self.plugin_sha256.as_str()),
            ("release.pack_sha256", self.pack_sha256.as_str()),
            ("release.catalog_sha256", self.catalog_sha256.as_str()),
            (
                "release.catalog_signature_sha256",
                self.catalog_signature_sha256.as_str(),
            ),
            (
                "release.capability_matrix_sha256",
                self.capability_matrix_sha256.as_str(),
            ),
            (
                "release.capability_cell_sha256",
                self.capability_cell_sha256.as_str(),
            ),
        ] {
            require_lower_hex(field, value, 64)?;
        }
        if self.capability_epoch == 0 {
            return Err(OwnershipEvidenceError::InvalidField {
                field: "release.capability_epoch",
            });
        }
        if self.evidence_revision == 0 {
            return Err(OwnershipEvidenceError::InvalidField {
                field: "release.evidence_revision",
            });
        }
        for (field, value) in [
            ("release.family", self.family.as_str()),
            ("release.model_id", self.model_id.as_str()),
            ("release.quant", self.quant.as_str()),
            ("release.topology", self.topology.as_str()),
            ("release.device_target", self.device_target.as_str()),
        ] {
            require_non_empty(field, value)?;
        }
        if let Some(digest) = &self.approved_target_set_sha256 {
            require_lower_hex("release.approved_target_set_sha256", digest, 64)?;
        }
        require_one_of(
            "release.provider",
            &self.provider,
            &["cpu", "metal", "cuda", "hip", "vulkan"],
        )?;
        require_one_of(
            "release.placement",
            &self.placement,
            &["cpu_only", "full_device", "hybrid"],
        )?;
        require_one_of(
            "release.output_plan",
            &self.output_plan,
            &["full_logits", "complete_scores", "native_first_max_token"],
        )?;
        require_one_of(
            "release.reuse_mode",
            &self.reuse_mode,
            &["fresh_graph", "reusable_graph"],
        )?;
        require_one_of(
            "release.capture_mode",
            &self.capture_mode,
            &["disabled", "enabled", "unsupported"],
        )?;
        require_one_of(
            "release.scheduler_mode",
            &self.scheduler_mode,
            &["disabled", "enabled"],
        )?;
        require_one_of(
            "release.activation_mode",
            &self.activation_mode,
            &["auto", "explicit"],
        )
    }
}

impl OwnershipEvidencePhase {
    fn validate(&self) -> Result<(), OwnershipEvidenceError> {
        if self.daemon_start_identity.pid == 0
            || self.daemon_start_identity.started_at_unix_secs == 0
        {
            return Err(OwnershipEvidenceError::InvalidDaemonIdentity);
        }
        require_lower_hex(
            "phase.daemon_start_identity.nonce",
            &self.daemon_start_identity.nonce,
            32,
        )?;
        self.runtime_snapshot.validate()?;
        for artifact in [
            self.request_receipt.as_ref(),
            self.activation_receipt.as_ref(),
            self.pressure_helper_receipt.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            artifact.validate()?;
        }
        if let Some(observation) = &self.observation {
            observation.admission.validate()?;
        }
        Ok(())
    }
}

impl OwnershipEvidenceArtifact {
    fn validate(&self) -> Result<(), OwnershipEvidenceError> {
        require_safe_artifact_label(&self.label)?;
        require_lower_hex("artifact.sha256", &self.sha256, 64)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OwnershipEvidenceError {
    #[error("ownership evidence schema mismatch")]
    SchemaMismatch,
    #[error("ownership evidence result is not pass")]
    NonPassingResult,
    #[error("ownership evidence field '{field}' is invalid")]
    InvalidField { field: &'static str },
    #[error("ownership evidence contains no phases")]
    MissingPhases,
    #[error("ownership evidence phase ordinals are not contiguous")]
    NonContiguousPhaseOrder,
    #[error("ownership evidence phase sequence does not match its scenario")]
    WrongPhaseSequence,
    #[error("ownership evidence is missing phase {phase:?}")]
    MissingPhase { phase: OwnershipEvidencePhaseKind },
    #[error("ownership evidence daemon identity changed within one scenario")]
    DaemonIdentityChanged,
    #[error("ownership evidence daemon start identity is invalid")]
    InvalidDaemonIdentity,
    #[error("ownership baseline was not admissible")]
    BaselineNotAdmissible,
    #[error("pressure did not cause the same candidate to cross the rejection threshold")]
    PressureDidNotCrossThreshold,
    #[error("pressure helper violated the configured available-memory floor")]
    SafetyFloorViolated,
    #[error("available-memory observation did not recover to an admissible state")]
    ObservationDidNotRecover,
    #[error("ownership evidence candidate identity changed across phases")]
    CandidateIdentityChanged,
    #[error("ownership phase {phase:?} has no candidate observation")]
    MissingObservation { phase: OwnershipEvidencePhaseKind },
    #[error("ownership phase {phase:?} has no request receipt")]
    MissingRequestReceipt { phase: OwnershipEvidencePhaseKind },
    #[error("activation rejection phase has no activation receipt")]
    MissingActivationReceipt,
    #[error("pressure phase {phase:?} has no pressure-helper receipt")]
    MissingPressureHelperReceipt { phase: OwnershipEvidencePhaseKind },
    #[error("one or more ownership phases did not reconcile to the broker ledger")]
    LeaseReconciliationNotMatched,
    #[error("ownership evidence artifact label is unsafe")]
    UnsafeArtifactLabel,
    #[error("one ownership artifact label is bound to more than one digest")]
    ArtifactDigestConflict,
    #[error("model activation receipt schema mismatch")]
    ActivationReceiptSchemaMismatch,
    #[error("model activation receipt is not a fresh-reserve rejection")]
    ActivationReceiptNotFreshReserveRejection,
    #[error(
        "model activation receipt did not preserve old durable/live state and release staged owners"
    )]
    ActivationReceiptDidNotPreserveOldState,
}

#[derive(Debug, Error)]
pub enum OwnershipEvidenceLoadError {
    #[error("could not parse ownership evidence: {0}")]
    Parse(#[from] serde_json::Error),
    #[error(transparent)]
    Validate(#[from] OwnershipEvidenceError),
}

#[derive(Debug, Error)]
pub enum OwnershipActivationReceiptLoadError {
    #[error("could not parse model activation receipt: {0}")]
    Parse(#[from] serde_json::Error),
    #[error(transparent)]
    Validate(#[from] OwnershipEvidenceError),
}

fn require_non_empty(field: &'static str, value: &str) -> Result<(), OwnershipEvidenceError> {
    if value.trim().is_empty()
        || value.trim() != value
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        Err(OwnershipEvidenceError::InvalidField { field })
    } else {
        Ok(())
    }
}

fn require_one_of(
    field: &'static str,
    value: &str,
    allowed: &[&str],
) -> Result<(), OwnershipEvidenceError> {
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(OwnershipEvidenceError::InvalidField { field })
    }
}

fn require_lower_hex(
    field: &'static str,
    value: &str,
    length: usize,
) -> Result<(), OwnershipEvidenceError> {
    if value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(OwnershipEvidenceError::InvalidField { field })
    }
}

fn require_safe_artifact_label(value: &str) -> Result<(), OwnershipEvidenceError> {
    if value.is_empty()
        || value.len() > 160
        || matches!(value, "." | "..")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        Err(OwnershipEvidenceError::UnsafeArtifactLabel)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn binding() -> OwnershipReleaseBinding {
        OwnershipReleaseBinding {
            release_subject: "openasr-v0.1.36-windows-x86_64.zip".to_string(),
            core_commit: "1234567890123456789012345678901234567890".to_string(),
            host_abi_fingerprint: SHA_A.to_string(),
            binary_sha256: SHA_A.to_string(),
            plugin_sha256: SHA_B.to_string(),
            pack_sha256: SHA_A.to_string(),
            catalog_sha256: SHA_B.to_string(),
            catalog_signature_sha256: SHA_A.to_string(),
            capability_matrix_sha256: SHA_B.to_string(),
            capability_epoch: 3,
            capability_cell_sha256: SHA_B.to_string(),
            family: "qwen3_asr".to_string(),
            model_id: "qwen3-asr-0.6b".to_string(),
            quant: "q8_0".to_string(),
            topology: "discrete".to_string(),
            provider: "hip".to_string(),
            device_target: "gfx1200".to_string(),
            approved_target_set_sha256: None,
            placement: "full_device".to_string(),
            output_plan: "full_logits".to_string(),
            reuse_mode: "fresh_graph".to_string(),
            capture_mode: "enabled".to_string(),
            scheduler_mode: "disabled".to_string(),
            evidence_revision: 1,
            activation_mode: "explicit".to_string(),
        }
    }

    fn daemon() -> OwnershipDaemonStartIdentity {
        OwnershipDaemonStartIdentity {
            pid: 42,
            nonce: "0123456789abcdef0123456789abcdef".to_string(),
            started_at_unix_secs: 1_700_000_000,
        }
    }

    fn artifact(label: &str) -> OwnershipEvidenceArtifact {
        OwnershipEvidenceArtifact {
            label: label.to_string(),
            sha256: SHA_A.to_string(),
        }
    }

    fn activation_receipt() -> OwnershipActivationReceipt {
        OwnershipActivationReceipt {
            schema: OWNERSHIP_ACTIVATION_RECEIPT_SCHEMA.to_string(),
            result: "rejected".to_string(),
            daemon_start_identity: daemon(),
            release_subject: binding().release_subject,
            core_commit: binding().core_commit,
            pack_sha256: binding().pack_sha256,
            capability_matrix_sha256: binding().capability_matrix_sha256,
            capability_epoch: binding().capability_epoch,
            provider: binding().provider,
            device_target: binding().device_target,
            failure_stage: "fresh_reserve".to_string(),
            fresh_reserve: admission(300),
            durable_selection_before_sha256: SHA_A.to_string(),
            durable_selection_after_sha256: SHA_A.to_string(),
            live_runtime_before_sha256: SHA_B.to_string(),
            live_runtime_after_sha256: SHA_B.to_string(),
            staged_owner_cleanup: "released".to_string(),
        }
    }

    fn admission(available: u64) -> OwnershipAdmissionObservation {
        OwnershipAdmissionObservation {
            candidate_sha256: SHA_B.to_string(),
            policy_requested_bytes: 500,
            policy_remaining_bytes: 1_000,
            observed_requested_bytes: 500,
            observed_remaining_bytes: available,
        }
    }

    fn observation(available: u64, helper: u64) -> OwnershipCandidateObservation {
        OwnershipCandidateObservation {
            admission: admission(available),
            safety_floor_bytes: 200,
            helper_committed_bytes: helper,
            helper_touched_bytes: helper,
        }
    }

    fn phase(
        ordinal: u32,
        kind: OwnershipEvidencePhaseKind,
        observation: Option<OwnershipCandidateObservation>,
    ) -> OwnershipEvidencePhase {
        OwnershipEvidencePhase {
            ordinal,
            kind,
            daemon_start_identity: daemon(),
            runtime_snapshot: artifact(&format!("snapshot-{ordinal}.json")),
            request_receipt: (kind == OwnershipEvidencePhaseKind::OldRuntimeTranscribed)
                .then(|| artifact("old-runtime-request.json")),
            activation_receipt: (kind == OwnershipEvidencePhaseKind::ActivationRejected)
                .then(|| artifact("activation-rejected.json")),
            pressure_helper_receipt: matches!(
                kind,
                OwnershipEvidencePhaseKind::PressureReady
                    | OwnershipEvidencePhaseKind::PressureReleased
            )
            .then(|| artifact("pressure-helper.json")),
            observation,
            lease_reconciliation: OwnershipLeaseReconciliationStatus::Matched,
        }
    }

    fn real_pressure_envelope() -> OwnershipEvidenceEnvelope {
        let kinds = [
            OwnershipEvidencePhaseKind::BaselineAdmissible,
            OwnershipEvidencePhaseKind::OldRuntimeActive,
            OwnershipEvidencePhaseKind::PressureReady,
            OwnershipEvidencePhaseKind::ActivationRejected,
            OwnershipEvidencePhaseKind::OldRuntimeTranscribed,
            OwnershipEvidencePhaseKind::Reconciled,
            OwnershipEvidencePhaseKind::PressureReleased,
            OwnershipEvidencePhaseKind::Recovered,
        ];
        let phases = kinds
            .into_iter()
            .enumerate()
            .map(|(index, kind)| {
                let observation = match kind {
                    OwnershipEvidencePhaseKind::BaselineAdmissible => Some(observation(800, 0)),
                    OwnershipEvidencePhaseKind::PressureReady => Some(observation(300, 600)),
                    OwnershipEvidencePhaseKind::Recovered => Some(observation(700, 0)),
                    _ => None,
                };
                phase(index as u32, kind, observation)
            })
            .collect();
        OwnershipEvidenceEnvelope {
            schema: OWNERSHIP_EVIDENCE_SCHEMA.to_string(),
            scenario: OwnershipEvidenceScenario::RealHostPressureRollback,
            result: "pass".to_string(),
            release: binding(),
            phases,
        }
    }

    fn deterministic_pressure_envelope() -> OwnershipEvidenceEnvelope {
        let kinds = [
            OwnershipEvidencePhaseKind::BaselineAdmissible,
            OwnershipEvidencePhaseKind::ForecastSucceeded,
            OwnershipEvidencePhaseKind::FactsChanged,
            OwnershipEvidencePhaseKind::ActivationRejected,
            OwnershipEvidencePhaseKind::OldRuntimeTranscribed,
            OwnershipEvidencePhaseKind::Reconciled,
            OwnershipEvidencePhaseKind::Recovered,
        ];
        let phases = kinds
            .into_iter()
            .enumerate()
            .map(|(index, kind)| {
                let observation = match kind {
                    OwnershipEvidencePhaseKind::BaselineAdmissible => Some(observation(800, 0)),
                    OwnershipEvidencePhaseKind::FactsChanged => Some(observation(300, 0)),
                    OwnershipEvidencePhaseKind::Recovered => Some(observation(700, 0)),
                    _ => None,
                };
                phase(index as u32, kind, observation)
            })
            .collect();
        OwnershipEvidenceEnvelope {
            schema: OWNERSHIP_EVIDENCE_SCHEMA.to_string(),
            scenario: OwnershipEvidenceScenario::DeterministicPressureRace,
            result: "pass".to_string(),
            release: binding(),
            phases,
        }
    }

    fn cold_warm_envelope() -> OwnershipEvidenceEnvelope {
        let kinds = [
            OwnershipEvidencePhaseKind::BaselineAdmissible,
            OwnershipEvidencePhaseKind::ColdRequestCompleted,
            OwnershipEvidencePhaseKind::WarmRequestCompleted,
            OwnershipEvidencePhaseKind::OwnerReleased,
            OwnershipEvidencePhaseKind::Reconciled,
        ];
        let phases = kinds
            .into_iter()
            .enumerate()
            .map(|(index, kind)| {
                let mut phase = phase(
                    index as u32,
                    kind,
                    (kind == OwnershipEvidencePhaseKind::BaselineAdmissible)
                        .then(|| observation(800, 0)),
                );
                if matches!(
                    kind,
                    OwnershipEvidencePhaseKind::ColdRequestCompleted
                        | OwnershipEvidencePhaseKind::WarmRequestCompleted
                ) {
                    phase.request_receipt = Some(artifact(&format!("request-{index}.json")));
                }
                phase
            })
            .collect();
        OwnershipEvidenceEnvelope {
            schema: OWNERSHIP_EVIDENCE_SCHEMA.to_string(),
            scenario: OwnershipEvidenceScenario::ColdWarmLifecycle,
            result: "pass".to_string(),
            release: binding(),
            phases,
        }
    }

    #[test]
    fn real_pressure_requires_a_causal_state_flip_and_recovery() {
        OwnershipEvidenceEnvelope::try_new(real_pressure_envelope()).unwrap();
    }

    #[test]
    fn deterministic_race_requires_forecast_then_fresh_rejection() {
        OwnershipEvidenceEnvelope::try_new(deterministic_pressure_envelope()).unwrap();
    }

    #[test]
    fn deterministic_race_accepts_a_policy_ledger_state_flip() {
        let mut envelope = deterministic_pressure_envelope();
        let changed = envelope.phases[2].observation.as_mut().unwrap();
        changed.admission.observed_remaining_bytes = 800;
        changed.admission.policy_remaining_bytes = 300;
        OwnershipEvidenceEnvelope::try_new(envelope).unwrap();
    }

    #[test]
    fn real_host_pressure_must_cross_the_native_observation_axis() {
        let mut envelope = real_pressure_envelope();
        let pressured = envelope.phases[2].observation.as_mut().unwrap();
        pressured.admission.observed_remaining_bytes = 800;
        pressured.admission.policy_remaining_bytes = 300;
        assert_eq!(
            envelope.validate().unwrap_err(),
            OwnershipEvidenceError::PressureDidNotCrossThreshold
        );
    }

    #[test]
    fn cold_warm_lifecycle_requires_request_and_release_phases() {
        OwnershipEvidenceEnvelope::try_new(cold_warm_envelope()).unwrap();
    }

    #[test]
    fn baseline_failure_is_not_a_pressure_pass() {
        let mut envelope = real_pressure_envelope();
        envelope.phases[0]
            .observation
            .as_mut()
            .unwrap()
            .admission
            .observed_remaining_bytes = 300;
        assert_eq!(
            envelope.validate().unwrap_err(),
            OwnershipEvidenceError::BaselineNotAdmissible
        );
    }

    #[test]
    fn helper_without_threshold_crossing_is_not_a_pass() {
        let mut envelope = real_pressure_envelope();
        envelope.phases[2]
            .observation
            .as_mut()
            .unwrap()
            .admission
            .observed_remaining_bytes = 700;
        assert_eq!(
            envelope.validate().unwrap_err(),
            OwnershipEvidenceError::PressureDidNotCrossThreshold
        );
    }

    #[test]
    fn pressure_cannot_violate_the_safety_floor() {
        let mut envelope = real_pressure_envelope();
        envelope.phases[2]
            .observation
            .as_mut()
            .unwrap()
            .safety_floor_bytes = 400;
        assert_eq!(
            envelope.validate().unwrap_err(),
            OwnershipEvidenceError::SafetyFloorViolated
        );
    }

    #[test]
    fn recovery_must_make_the_same_candidate_admissible_again() {
        let mut envelope = real_pressure_envelope();
        envelope.phases[7]
            .observation
            .as_mut()
            .unwrap()
            .admission
            .observed_remaining_bytes = 400;
        assert_eq!(
            envelope.validate().unwrap_err(),
            OwnershipEvidenceError::ObservationDidNotRecover
        );
    }

    #[test]
    fn pressure_phases_must_keep_the_exact_candidate_identity() {
        let mut envelope = real_pressure_envelope();
        envelope.phases[2]
            .observation
            .as_mut()
            .unwrap()
            .admission
            .candidate_sha256 = SHA_A.to_string();
        assert_eq!(
            envelope.validate().unwrap_err(),
            OwnershipEvidenceError::CandidateIdentityChanged
        );
    }

    #[test]
    fn every_observation_must_bind_the_release_capability_cell() {
        let mut envelope = real_pressure_envelope();
        envelope.release.capability_cell_sha256 = SHA_A.to_string();
        assert_eq!(
            envelope.validate().unwrap_err(),
            OwnershipEvidenceError::CandidateIdentityChanged
        );
    }

    #[test]
    fn every_phase_must_reconcile_to_the_broker_ledger() {
        let mut envelope = real_pressure_envelope();
        envelope.phases[5].lease_reconciliation = OwnershipLeaseReconciliationStatus::Mismatched;
        assert_eq!(
            envelope.validate().unwrap_err(),
            OwnershipEvidenceError::LeaseReconciliationNotMatched
        );
    }

    #[test]
    fn redacted_runtime_ids_are_not_part_of_cross_phase_identity() {
        let json = real_pressure_envelope().to_pretty_json().unwrap();
        assert!(!json.contains("owner_id"));
        assert!(!json.contains("join_id"));
        OwnershipEvidenceEnvelope::from_json_str(&json).unwrap();
    }

    #[test]
    fn artifact_labels_are_release_safe_basenames() {
        let mut envelope = real_pressure_envelope();
        envelope.phases[0].runtime_snapshot.label = "../snapshot.json".to_string();
        assert_eq!(
            envelope.validate().unwrap_err(),
            OwnershipEvidenceError::UnsafeArtifactLabel
        );
    }

    #[test]
    fn one_artifact_label_cannot_name_different_bytes() {
        let mut envelope = real_pressure_envelope();
        envelope.phases[1].runtime_snapshot.label =
            envelope.phases[0].runtime_snapshot.label.clone();
        envelope.phases[1].runtime_snapshot.sha256 = SHA_B.to_string();
        assert_eq!(
            envelope.validate().unwrap_err(),
            OwnershipEvidenceError::ArtifactDigestConflict
        );
    }

    #[test]
    fn artifact_bindings_are_deduplicated_by_label() {
        let mut envelope = real_pressure_envelope();
        envelope.phases[1].runtime_snapshot = envelope.phases[0].runtime_snapshot.clone();
        envelope.validate().unwrap();
        let labels = envelope
            .artifact_bindings()
            .into_iter()
            .map(|artifact| artifact.label.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            labels
                .iter()
                .filter(|label| **label == "snapshot-0.json")
                .count(),
            1
        );
    }

    #[test]
    fn activation_receipt_proves_fresh_rejection_and_old_state_preservation() {
        let receipt = OwnershipActivationReceipt::try_new(activation_receipt()).unwrap();
        let json = serde_json::to_string(&receipt).unwrap();
        OwnershipActivationReceipt::from_json_str(&json).unwrap();
    }

    #[test]
    fn activation_receipt_rejects_durable_or_live_state_drift() {
        let mut receipt = activation_receipt();
        receipt.fresh_reserve.observed_remaining_bytes = 1_000;
        assert_eq!(
            receipt.validate().unwrap_err(),
            OwnershipEvidenceError::ActivationReceiptNotFreshReserveRejection
        );
        let mut receipt = activation_receipt();
        receipt.durable_selection_after_sha256 = SHA_B.to_string();
        assert_eq!(
            receipt.validate().unwrap_err(),
            OwnershipEvidenceError::ActivationReceiptDidNotPreserveOldState
        );
        let mut receipt = activation_receipt();
        receipt.staged_owner_cleanup = "quarantined".to_string();
        assert_eq!(
            receipt.validate().unwrap_err(),
            OwnershipEvidenceError::ActivationReceiptDidNotPreserveOldState
        );
    }
}
