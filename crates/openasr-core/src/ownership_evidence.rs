//! Artifact-bound release evidence over ordered runtime ownership snapshots.
//!
//! openasr.runtime-ownership-receipt.v1 remains the production diagnostic
//! snapshot. This module binds hashes of those snapshots and adjacent request /
//! activation / pressure-helper receipts to immutable release artifacts and a
//! causal phase sequence. Admission and runtime policy never consume it.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const OWNERSHIP_EVIDENCE_SCHEMA: &str = "openasr.runtime-ownership-evidence.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OwnershipEvidenceScenario {
    ColdWarmLifecycle,
    DeterministicPressureRace,
    RealHostPressureRollback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    pub provider: String,
    pub device_target: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnershipDaemonStartIdentity {
    pub pid: u32,
    pub nonce: String,
    pub started_at_unix_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnershipCandidateObservation {
    /// Digest of the exact pack/lane/artifact/capability cell being attempted.
    pub candidate_sha256: String,
    pub requested_bytes: u64,
    pub policy_remainder_bytes: u64,
    pub observed_available_bytes: u64,
    pub safety_floor_bytes: u64,
    pub helper_committed_bytes: u64,
    pub helper_touched_bytes: u64,
}

impl OwnershipCandidateObservation {
    pub fn is_admissible(&self) -> bool {
        self.requested_bytes <= self.policy_remainder_bytes
            && self.requested_bytes <= self.observed_available_bytes
    }

    pub fn crosses_observed_rejection_threshold(&self) -> bool {
        self.requested_bytes > self.observed_available_bytes
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
        if !changed.crosses_observed_rejection_threshold() {
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
        if pressured.observed_available_bytes < pressured.safety_floor_bytes {
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
        let first = &observations[0].candidate_sha256;
        if observations
            .iter()
            .any(|observation| &observation.candidate_sha256 != first)
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
        ] {
            require_lower_hex(field, value, 64)?;
        }
        if self.capability_epoch == 0 {
            return Err(OwnershipEvidenceError::InvalidField {
                field: "release.capability_epoch",
            });
        }
        require_non_empty("release.provider", &self.provider)?;
        require_non_empty("release.device_target", &self.device_target)
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
            require_lower_hex(
                "phase.observation.candidate_sha256",
                &observation.candidate_sha256,
                64,
            )?;
            if observation.requested_bytes == 0 {
                return Err(OwnershipEvidenceError::InvalidField {
                    field: "phase.observation.requested_bytes",
                });
            }
        }
        Ok(())
    }
}

impl OwnershipEvidenceArtifact {
    fn validate(&self) -> Result<(), OwnershipEvidenceError> {
        require_non_empty("artifact.label", &self.label)?;
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
}

#[derive(Debug, Error)]
pub enum OwnershipEvidenceLoadError {
    #[error("could not parse ownership evidence: {0}")]
    Parse(#[from] serde_json::Error),
    #[error(transparent)]
    Validate(#[from] OwnershipEvidenceError),
}

fn require_non_empty(field: &'static str, value: &str) -> Result<(), OwnershipEvidenceError> {
    if value.trim().is_empty() || value.trim() != value {
        Err(OwnershipEvidenceError::InvalidField { field })
    } else {
        Ok(())
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
            provider: "hip".to_string(),
            device_target: "gfx1200".to_string(),
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

    fn observation(available: u64, helper: u64) -> OwnershipCandidateObservation {
        OwnershipCandidateObservation {
            candidate_sha256: SHA_B.to_string(),
            requested_bytes: 500,
            policy_remainder_bytes: 1_000,
            observed_available_bytes: available,
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
            .observed_available_bytes = 300;
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
            .observed_available_bytes = 700;
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
            .observed_available_bytes = 400;
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
            .candidate_sha256 = SHA_A.to_string();
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
}
