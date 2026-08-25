//! Release-only verifier for artifact-bound runtime ownership evidence.
//!
//! The semantic envelope and activation receipt are owned by openasr-core.
//! This module supplies the filesystem boundary: release-safe direct-child
//! lookup, regular-file checks, content hashing, and validation of the runtime
//! HTTP snapshots/request receipts/helper trace referenced by the envelope.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use openasr_core::{
    OWNERSHIP_EVIDENCE_SCHEMA, OwnershipActivationReceipt, OwnershipDaemonStartIdentity,
    OwnershipEvidenceArtifact, OwnershipEvidenceEnvelope, OwnershipEvidencePhaseKind,
    OwnershipEvidenceScenario, OwnershipReleaseBinding, ShortAudioCaptureMode,
    ShortAudioEvidenceClass, ShortAudioOutputPlanKind, ShortAudioReceipt, ShortAudioSchedulerMode,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const BUNDLE_RESULT_SCHEMA: &str = "openasr.runtime-ownership-evidence-bundle-result.v1";
const RUNTIME_SNAPSHOT_SCHEMA: &str = "openasr.runtime-ownership-receipt.v1";
const PRESSURE_HELPER_SCHEMA: &str = "openasr.windows-memory-pressure-helper.v1";
const MAX_OWNERSHIP_ARTIFACT_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
struct RuntimeSnapshotFacts {
    live_owners: Value,
    live_owner_count: usize,
    retained_bytes: u64,
    has_release_provider: bool,
}

fn value_u64(value: &Value, field: &str) -> Result<u64> {
    value
        .as_u64()
        .with_context(|| format!("{field} must be an unsigned integer"))
}

fn value_str<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .as_str()
        .with_context(|| format!("{field} must be a string"))
}

fn read_regular_file(path: &Path, label: &str) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("ownership artifact is missing: {label}"))?;
    if !metadata.file_type().is_file() {
        bail!("ownership artifact must be a regular non-symlink file: {label}");
    }
    if metadata.len() > MAX_OWNERSHIP_ARTIFACT_BYTES {
        bail!("ownership artifact exceeds the bounded verifier limit: {label}");
    }
    fs::read(path).with_context(|| format!("could not read ownership artifact: {label}"))
}

fn read_bound_artifact(
    artifact_dir: &Path,
    binding: &OwnershipEvidenceArtifact,
) -> Result<Vec<u8>> {
    let path = artifact_dir.join(&binding.label);
    let bytes = read_regular_file(&path, &binding.label)?;
    let actual_sha256 = format!("{:x}", Sha256::digest(&bytes));
    if actual_sha256 != binding.sha256 {
        bail!("ownership artifact hash mismatch: {}", binding.label);
    }
    Ok(bytes)
}

fn daemon_identity_matches(value: &Value, expected: &OwnershipDaemonStartIdentity) -> Result<()> {
    if value_u64(&value["pid"], "daemon_start_identity.pid")? != u64::from(expected.pid)
        || value_str(&value["nonce"], "daemon_start_identity.nonce")? != expected.nonce
        || value_u64(
            &value["started_at_unix_secs"],
            "daemon_start_identity.started_at_unix_secs",
        )? != expected.started_at_unix_secs
    {
        bail!("runtime artifact daemon start identity does not match its envelope phase");
    }
    Ok(())
}

fn known_retained_bytes(resource: &Value) -> Result<u64> {
    let retained = &resource["retained"];
    if retained["status"] != "known" {
        bail!("live runtime resource has unknown or unavailable retained bytes");
    }
    value_u64(&retained["bytes"], "live_owners.resources.retained.bytes")
}

fn validate_runtime_snapshot(
    bytes: &[u8],
    expected_daemon: &OwnershipDaemonStartIdentity,
    release_provider: &str,
) -> Result<RuntimeSnapshotFacts> {
    let value: Value = serde_json::from_slice(bytes).context("runtime snapshot is not JSON")?;
    if value["schema"] != RUNTIME_SNAPSHOT_SCHEMA
        || value["availability"] != "available"
        || value["snapshot_completeness"]["live_state_complete"] != true
        || value["lease_reconciliation"]["status"] != "matched"
    {
        bail!("runtime snapshot is unavailable, live-incomplete, or ledger-mismatched");
    }
    daemon_identity_matches(&value["daemon_start_identity"], expected_daemon)?;
    let live_owners = value["live_owners"]
        .as_array()
        .context("runtime snapshot live_owners must be an array")?;
    let mut retained_bytes = 0_u64;
    let mut has_release_provider = false;
    for owner in live_owners {
        if owner["placement"] == "unknown" {
            bail!("runtime snapshot contains an owner with unknown placement");
        }
        if owner["placement"] == "lane-bound" {
            let provider = value_str(&owner["lane"]["provider"], "live_owners.lane.provider")?;
            has_release_provider |= provider.eq_ignore_ascii_case(release_provider);
        }
        let resources = owner["resources"]
            .as_array()
            .context("runtime owner resources must be an array")?;
        for resource in resources {
            if resource["ledger_binding"] == "unknown" {
                bail!("runtime snapshot contains an unpriced live resource");
            }
            retained_bytes = retained_bytes
                .checked_add(known_retained_bytes(resource)?)
                .context("runtime retained-byte total overflowed")?;
        }
    }
    Ok(RuntimeSnapshotFacts {
        live_owners: value["live_owners"].clone(),
        live_owner_count: live_owners.len(),
        retained_bytes,
        has_release_provider,
    })
}

fn validate_request_receipt(
    bytes: &[u8],
    release: &OwnershipReleaseBinding,
    phase: OwnershipEvidencePhaseKind,
) -> Result<()> {
    let raw = std::str::from_utf8(bytes).context("request receipt is not UTF-8")?;
    let receipt = ShortAudioReceipt::from_json_str(raw)
        .context("request artifact is not a valid short-audio receipt")?;
    let evidence = receipt
        .evidence
        .as_ref()
        .context("ownership request receipt has no formal evidence.v1 binding")?;
    let output_plan_matches = evidence.output_plan.as_ref().is_some_and(|output| {
        matches!(
            (output.kind, release.output_plan.as_str()),
            (ShortAudioOutputPlanKind::FullLogits, "full_logits")
                | (ShortAudioOutputPlanKind::CompleteScores, "complete_scores")
                | (
                    ShortAudioOutputPlanKind::NativeFirstMaxToken,
                    "native_first_max_token"
                )
        )
    });
    let capture_matches = matches!(
        (evidence.capture_mode, release.capture_mode.as_str()),
        (ShortAudioCaptureMode::Disabled, "disabled")
            | (ShortAudioCaptureMode::Enabled, "enabled")
            | (ShortAudioCaptureMode::Unsupported, "unsupported")
    );
    let scheduler_matches = matches!(
        (evidence.scheduler_mode, release.scheduler_mode.as_str()),
        (ShortAudioSchedulerMode::Disabled, "disabled")
            | (ShortAudioSchedulerMode::Enabled, "enabled")
    );
    let pack_model_matches = receipt.pack.model_id == release.model_id
        || receipt.pack.model_id == format!("{}:{}", release.model_id, release.quant);
    if receipt.core_commit != release.core_commit
        || receipt.pack.content_sha256 != release.pack_sha256
        || !pack_model_matches
        || receipt.pack.quant != release.quant
        || evidence.candidate_release_subject != release.release_subject
        || evidence.core_commit != release.core_commit
        || evidence.matrix_sha256 != release.capability_matrix_sha256
        || evidence.evidence_class != ShortAudioEvidenceClass::TokenTranscript
        || evidence.family != release.family
        || evidence.model_id != release.model_id
        || evidence.quant != release.quant
        || evidence.topology != release.topology
        || !evidence.provider.eq_ignore_ascii_case(&release.provider)
        || evidence.placement != release.placement
        || !output_plan_matches
        || !capture_matches
        || !scheduler_matches
        || evidence.result != "pass"
        || evidence.artifacts.binary.sha256 != release.binary_sha256
        || evidence.artifacts.plugin.sha256 != release.plugin_sha256
        || evidence.artifacts.pack.sha256 != release.pack_sha256
    {
        bail!("ownership request receipt does not bind the exact passing release cell");
    }
    match phase {
        OwnershipEvidencePhaseKind::ColdRequestCompleted if receipt.run.warmup != "cold" => {
            bail!("cold ownership request receipt is not a cold run")
        }
        OwnershipEvidencePhaseKind::WarmRequestCompleted if receipt.run.warmup != "warm" => {
            bail!("warm ownership request receipt is not a same-process warm run")
        }
        _ => {}
    }
    Ok(())
}

fn validate_activation_receipt(
    bytes: &[u8],
    envelope: &OwnershipEvidenceEnvelope,
    expected_observation: &openasr_core::OwnershipCandidateObservation,
    expected_daemon: &OwnershipDaemonStartIdentity,
) -> Result<()> {
    let raw = std::str::from_utf8(bytes).context("activation receipt is not UTF-8")?;
    let receipt = OwnershipActivationReceipt::from_json_str(raw)
        .context("activation artifact is not a valid activation receipt")?;
    let release = &envelope.release;
    if &receipt.daemon_start_identity != expected_daemon
        || receipt.release_subject != release.release_subject
        || receipt.core_commit != release.core_commit
        || receipt.pack_sha256 != release.pack_sha256
        || receipt.capability_matrix_sha256 != release.capability_matrix_sha256
        || receipt.capability_epoch != release.capability_epoch
        || !receipt.provider.eq_ignore_ascii_case(&release.provider)
        || receipt.device_target != release.device_target
        || receipt.fresh_reserve != expected_observation.admission
    {
        bail!("activation rejection receipt does not bind the exact envelope candidate");
    }
    Ok(())
}

fn parse_helper_event(line: &str, expected_event: &str, expected_result: &str) -> Result<Value> {
    let event: Value =
        serde_json::from_str(line).context("pressure helper trace line is not JSON")?;
    if event["schema"] != PRESSURE_HELPER_SCHEMA
        || event["event"] != expected_event
        || event["result"] != expected_result
        || event["job_kill_on_close"] != true
        || event["parent_death_cleanup"] != true
        || event["page_locking"] != false
    {
        bail!("pressure helper trace lacks its safety contract");
    }
    Ok(event)
}

fn validate_pressure_helper_trace(
    bytes: &[u8],
    observation: &openasr_core::OwnershipCandidateObservation,
) -> Result<()> {
    let raw = std::str::from_utf8(bytes).context("pressure helper trace is not UTF-8")?;
    let lines = raw
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    if lines.len() != 2 {
        bail!("passing pressure helper trace must contain exactly ready and released events");
    }
    let ready = parse_helper_event(lines[0], "ready", "holding")?;
    let released = parse_helper_event(lines[1], "released", "pass")?;
    for field in [
        "helper_pid",
        "parent_pid",
        "candidate_required_bytes",
        "safety_floor_bytes",
    ] {
        if ready[field] != released[field] {
            bail!("pressure helper identity or guardrail changed while holding memory");
        }
    }
    let requested = value_u64(
        &ready["candidate_required_bytes"],
        "candidate_required_bytes",
    )?;
    let ready_available = value_u64(
        &ready["observed_available_bytes"],
        "observed_available_bytes",
    )?;
    let released_available = value_u64(
        &released["observed_available_bytes"],
        "released.observed_available_bytes",
    )?;
    let lowest = value_u64(
        &released["lowest_available_bytes"],
        "lowest_available_bytes",
    )?;
    let floor = value_u64(&ready["safety_floor_bytes"], "safety_floor_bytes")?;
    let committed = value_u64(&ready["committed_bytes"], "committed_bytes")?;
    let touched = value_u64(&ready["touched_bytes"], "touched_bytes")?;
    let timeout = value_u64(&ready["timeout_seconds"], "timeout_seconds")?;
    if requested != observation.admission.observed_requested_bytes
        || floor != observation.safety_floor_bytes
        || committed != observation.helper_committed_bytes
        || touched != observation.helper_touched_bytes
        || committed == 0
        || touched == 0
        || ready_available >= requested
        || released_available < requested
        || lowest < floor
        || timeout > 120
    {
        bail!("pressure helper trace does not prove the envelope's safe causal state flip");
    }
    Ok(())
}

fn phase_facts(
    facts: &BTreeMap<OwnershipEvidencePhaseKind, RuntimeSnapshotFacts>,
    kind: OwnershipEvidencePhaseKind,
) -> Result<&RuntimeSnapshotFacts> {
    facts
        .get(&kind)
        .with_context(|| format!("validated runtime snapshot is missing phase {kind:?}"))
}

fn require_same_live_state(
    facts: &BTreeMap<OwnershipEvidencePhaseKind, RuntimeSnapshotFacts>,
    kinds: &[OwnershipEvidencePhaseKind],
) -> Result<()> {
    let first = phase_facts(facts, kinds[0])?;
    for kind in kinds.iter().skip(1) {
        if phase_facts(facts, *kind)?.live_owners != first.live_owners {
            bail!("old runtime live-owner state changed across the protected activation sequence");
        }
    }
    Ok(())
}

fn validate_snapshot_transitions(
    envelope: &OwnershipEvidenceEnvelope,
    facts: &BTreeMap<OwnershipEvidencePhaseKind, RuntimeSnapshotFacts>,
) -> Result<()> {
    match envelope.scenario {
        OwnershipEvidenceScenario::ColdWarmLifecycle => {
            let cold = phase_facts(facts, OwnershipEvidencePhaseKind::ColdRequestCompleted)?;
            let warm = phase_facts(facts, OwnershipEvidencePhaseKind::WarmRequestCompleted)?;
            let released = phase_facts(facts, OwnershipEvidencePhaseKind::OwnerReleased)?;
            let reconciled = phase_facts(facts, OwnershipEvidencePhaseKind::Reconciled)?;
            if cold.live_owners != warm.live_owners
                || !cold.has_release_provider
                || !warm.has_release_provider
                || (released.live_owner_count >= cold.live_owner_count
                    && released.retained_bytes >= cold.retained_bytes)
                || released.live_owners != reconciled.live_owners
            {
                bail!(
                    "cold/warm ownership snapshots do not prove stable reuse, release, and reconciliation"
                );
            }
        }
        OwnershipEvidenceScenario::DeterministicPressureRace => require_same_live_state(
            facts,
            &[
                OwnershipEvidencePhaseKind::BaselineAdmissible,
                OwnershipEvidencePhaseKind::ForecastSucceeded,
                OwnershipEvidencePhaseKind::FactsChanged,
                OwnershipEvidencePhaseKind::ActivationRejected,
                OwnershipEvidencePhaseKind::OldRuntimeTranscribed,
                OwnershipEvidencePhaseKind::Reconciled,
                OwnershipEvidencePhaseKind::Recovered,
            ],
        )?,
        OwnershipEvidenceScenario::RealHostPressureRollback => require_same_live_state(
            facts,
            &[
                OwnershipEvidencePhaseKind::OldRuntimeActive,
                OwnershipEvidencePhaseKind::PressureReady,
                OwnershipEvidencePhaseKind::ActivationRejected,
                OwnershipEvidencePhaseKind::OldRuntimeTranscribed,
                OwnershipEvidencePhaseKind::Reconciled,
                OwnershipEvidencePhaseKind::PressureReleased,
                OwnershipEvidencePhaseKind::Recovered,
            ],
        )?,
    }
    if matches!(
        envelope.scenario,
        OwnershipEvidenceScenario::DeterministicPressureRace
            | OwnershipEvidenceScenario::RealHostPressureRollback
    ) && !phase_facts(facts, OwnershipEvidencePhaseKind::OldRuntimeTranscribed)?
        .has_release_provider
    {
        bail!("protected old runtime snapshot has no lane-bound release provider owner");
    }
    Ok(())
}

fn validate_envelope_artifacts(
    artifact_dir: &Path,
    envelope: &OwnershipEvidenceEnvelope,
) -> Result<usize> {
    let mut bytes_by_label = BTreeMap::<String, Vec<u8>>::new();
    for binding in envelope.artifact_bindings() {
        bytes_by_label.insert(
            binding.label.clone(),
            read_bound_artifact(artifact_dir, binding)?,
        );
    }
    let mut snapshot_facts = BTreeMap::new();
    for phase in &envelope.phases {
        let snapshot_bytes = bytes_by_label
            .get(&phase.runtime_snapshot.label)
            .context("runtime snapshot binding was not loaded")?;
        snapshot_facts.insert(
            phase.kind,
            validate_runtime_snapshot(
                snapshot_bytes,
                &phase.daemon_start_identity,
                &envelope.release.provider,
            )?,
        );
        if let Some(request) = &phase.request_receipt {
            validate_request_receipt(
                bytes_by_label
                    .get(&request.label)
                    .context("request receipt binding was not loaded")?,
                &envelope.release,
                phase.kind,
            )?;
        }
    }

    if let Some(phase) = envelope
        .phases
        .iter()
        .find(|phase| phase.kind == OwnershipEvidencePhaseKind::ActivationRejected)
    {
        let activation = phase
            .activation_receipt
            .as_ref()
            .context("activation rejection phase has no receipt binding")?;
        let candidate = match envelope.scenario {
            OwnershipEvidenceScenario::DeterministicPressureRace => envelope
                .phases
                .iter()
                .find(|phase| phase.kind == OwnershipEvidencePhaseKind::FactsChanged),
            OwnershipEvidenceScenario::RealHostPressureRollback => envelope
                .phases
                .iter()
                .find(|phase| phase.kind == OwnershipEvidencePhaseKind::PressureReady),
            OwnershipEvidenceScenario::ColdWarmLifecycle => None,
        }
        .and_then(|phase| phase.observation.as_ref())
        .context("pressure scenario has no rejection candidate observation")?;
        validate_activation_receipt(
            bytes_by_label
                .get(&activation.label)
                .context("activation receipt binding was not loaded")?,
            envelope,
            candidate,
            &phase.daemon_start_identity,
        )?;
    }

    if envelope.scenario == OwnershipEvidenceScenario::RealHostPressureRollback {
        let ready = envelope
            .phases
            .iter()
            .find(|phase| phase.kind == OwnershipEvidencePhaseKind::PressureReady)
            .context("real pressure envelope has no ready phase")?;
        let released = envelope
            .phases
            .iter()
            .find(|phase| phase.kind == OwnershipEvidencePhaseKind::PressureReleased)
            .context("real pressure envelope has no released phase")?;
        let ready_binding = ready
            .pressure_helper_receipt
            .as_ref()
            .context("pressure ready phase has no helper receipt")?;
        let released_binding = released
            .pressure_helper_receipt
            .as_ref()
            .context("pressure released phase has no helper receipt")?;
        if ready_binding != released_binding {
            bail!("pressure ready/released phases do not bind the same helper trace");
        }
        validate_pressure_helper_trace(
            bytes_by_label
                .get(&ready_binding.label)
                .context("pressure helper trace binding was not loaded")?,
            ready
                .observation
                .as_ref()
                .context("pressure ready phase has no candidate observation")?,
        )?;
    }
    validate_snapshot_transitions(envelope, &snapshot_facts)?;
    Ok(bytes_by_label.len())
}

fn load_envelope(artifact_dir: &Path, path: &Path) -> Result<OwnershipEvidenceEnvelope> {
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("ownership envelope has no safe UTF-8 filename")?;
    if path != artifact_dir.join(filename) {
        bail!("ownership envelope must be a direct child of the artifact directory");
    }
    let bytes = read_regular_file(path, filename)?;
    let raw = std::str::from_utf8(&bytes).context("ownership envelope is not UTF-8")?;
    OwnershipEvidenceEnvelope::from_json_str(raw)
        .context("ownership envelope failed semantic validation")
}

pub(crate) fn validate_bundle(artifact_dir: &Path, envelope_paths: &[PathBuf]) -> Result<()> {
    let metadata =
        fs::symlink_metadata(artifact_dir).context("ownership artifact directory is missing")?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        bail!("ownership artifact directory must be a real directory");
    }
    if envelope_paths.is_empty() || envelope_paths.len() > 3 {
        bail!(
            "ownership bundle requires one diagnostic envelope or exactly three scenario envelopes"
        );
    }
    let mut envelopes = Vec::with_capacity(envelope_paths.len());
    for path in envelope_paths {
        envelopes.push(load_envelope(artifact_dir, path)?);
    }
    let release = envelopes[0].release.clone();
    if envelopes.iter().any(|envelope| envelope.release != release) {
        bail!("ownership scenario envelopes do not bind one exact release cell");
    }
    if envelope_paths.len() == 3 {
        let scenarios = envelopes
            .iter()
            .map(|envelope| envelope.scenario)
            .collect::<BTreeSet<_>>();
        let expected = BTreeSet::from([
            OwnershipEvidenceScenario::ColdWarmLifecycle,
            OwnershipEvidenceScenario::DeterministicPressureRace,
            OwnershipEvidenceScenario::RealHostPressureRollback,
        ]);
        if scenarios != expected {
            bail!("ownership bundle must contain one envelope for every required scenario");
        }
    }
    let mut artifact_count = 0_usize;
    for envelope in &envelopes {
        artifact_count += validate_envelope_artifacts(artifact_dir, envelope)?;
    }
    println!(
        "{}",
        serde_json::to_string(&json!({
            "schema": BUNDLE_RESULT_SCHEMA,
            "result": "pass",
            "envelope_schema": OWNERSHIP_EVIDENCE_SCHEMA,
            "release_subject": release.release_subject,
            "core_commit": release.core_commit,
            "provider": release.provider,
            "device_target": release.device_target,
            "capability_epoch": release.capability_epoch,
            "capability_cell_sha256": release.capability_cell_sha256,
            "scenario_count": envelopes.len(),
            "artifact_reference_count": artifact_count,
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn daemon() -> OwnershipDaemonStartIdentity {
        OwnershipDaemonStartIdentity {
            pid: 42,
            nonce: "0123456789abcdef0123456789abcdef".to_string(),
            started_at_unix_secs: 1_700_000_000,
        }
    }

    fn snapshot(event_history_complete: bool) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema": RUNTIME_SNAPSHOT_SCHEMA,
            "daemon_start_identity": {
                "pid": 42,
                "nonce": "0123456789abcdef0123456789abcdef",
                "started_at_unix_secs": 1_700_000_000_u64,
            },
            "availability": "available",
            "snapshot_completeness": {
                "complete": event_history_complete,
                "live_state_complete": true,
                "event_history_complete": event_history_complete,
            },
            "lease_reconciliation": {"status": "matched"},
            "live_owners": [{
                "id": "owner-1",
                "placement": "lane-bound",
                "lane": {"provider": "Hip", "device": "abcd"},
                "resources": [{
                    "id": "resource-1",
                    "ledger_binding": "brokered",
                    "retained": {"status": "known", "bytes": 4096},
                }],
            }],
        }))
        .unwrap()
    }

    #[test]
    fn runtime_snapshot_accepts_event_history_overflow_but_not_live_incompleteness() {
        let facts = validate_runtime_snapshot(&snapshot(false), &daemon(), "hip").unwrap();
        assert!(facts.has_release_provider);
        assert_eq!(facts.retained_bytes, 4096);

        let mut value: Value = serde_json::from_slice(&snapshot(true)).unwrap();
        value["snapshot_completeness"]["live_state_complete"] = json!(false);
        assert!(
            validate_runtime_snapshot(&serde_json::to_vec(&value).unwrap(), &daemon(), "hip")
                .is_err()
        );
    }

    #[test]
    fn helper_trace_requires_safe_crossing_and_recovery() {
        let event = |name: &str, result: &str, available: u64| {
            json!({
                "schema": PRESSURE_HELPER_SCHEMA,
                "event": name,
                "result": result,
                "helper_pid": 7,
                "parent_pid": 42,
                "candidate_required_bytes": 500,
                "safety_floor_bytes": 200,
                "observed_available_bytes": available,
                "lowest_available_bytes": 300,
                "committed_bytes": 600,
                "touched_bytes": 600,
                "timeout_seconds": 60,
                "job_kill_on_close": true,
                "parent_death_cleanup": true,
                "page_locking": false,
            })
        };
        let trace = format!(
            "{}\n{}\n",
            event("ready", "holding", 300),
            event("released", "pass", 700)
        );
        let observation = openasr_core::OwnershipCandidateObservation {
            admission: openasr_core::OwnershipAdmissionObservation {
                candidate_sha256: "a".repeat(64),
                policy_requested_bytes: 500,
                policy_remaining_bytes: 1_000,
                observed_requested_bytes: 500,
                observed_remaining_bytes: 300,
            },
            safety_floor_bytes: 200,
            helper_committed_bytes: 600,
            helper_touched_bytes: 600,
        };
        validate_pressure_helper_trace(trace.as_bytes(), &observation).unwrap();
    }
}
