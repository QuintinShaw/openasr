//! Typed exact-cell capability approval.
//!
//! Catalog parsing and signature verification happen before this layer. The
//! resulting immutable snapshot is attested once against the running release
//! artifacts, then request dispatch performs an O(1) lookup. Family code never
//! parses provider names, catalogs, matrix JSON, or receipt text.

use std::{collections::HashMap, sync::Arc};

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    device::{
        execution_policy::{ExecutionCandidate, ExecutionPlacement},
        execution_route::ExecutionProvider,
    },
    ggml_runtime::{GgmlDecodeOutputPlan, GgmlDecodeReuseMode},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapabilityActivationMode {
    Auto,
    Explicit,
}

impl CapabilityActivationMode {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Explicit => "explicit",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapabilityCaptureMode {
    Disabled,
    Enabled,
    Unsupported,
}

impl CapabilityCaptureMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Enabled => "enabled",
            Self::Unsupported => "unsupported",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapabilitySchedulerMode {
    Disabled,
    Enabled,
}

impl CapabilitySchedulerMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Enabled => "enabled",
        }
    }
}

/// Release-wide identity shared by every exact approval cell in one snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityArtifactBinding {
    pub release_subject: String,
    pub core_commit: String,
    pub host_abi_fingerprint: String,
    pub binary_sha256: String,
    pub plugin_sha256: String,
    pub matrix_sha256: String,
    pub capability_epoch: u64,
}

impl CapabilityArtifactBinding {
    fn validate(&self) -> Result<(), CapabilityApprovalError> {
        require_non_empty("release_subject", &self.release_subject)?;
        require_lower_hex("core_commit", &self.core_commit, 40)?;
        for (field, value) in [
            ("host_abi_fingerprint", self.host_abi_fingerprint.as_str()),
            ("binary_sha256", self.binary_sha256.as_str()),
            ("plugin_sha256", self.plugin_sha256.as_str()),
            ("matrix_sha256", self.matrix_sha256.as_str()),
        ] {
            require_lower_hex(field, value, 64)?;
        }
        if self.capability_epoch == 0 {
            return Err(CapabilityApprovalError::InvalidField {
                field: "capability_epoch",
            });
        }
        Ok(())
    }
}

/// Artifact facts measured from the running process before a snapshot becomes
/// usable on the request hot path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeCapabilityArtifactIdentity {
    pub release_subject: String,
    pub core_commit: String,
    pub host_abi_fingerprint: String,
    pub binary_sha256: String,
    pub plugin_sha256: String,
    pub matrix_sha256: String,
    pub capability_epoch: u64,
}

impl From<RuntimeCapabilityArtifactIdentity> for CapabilityArtifactBinding {
    fn from(value: RuntimeCapabilityArtifactIdentity) -> Self {
        Self {
            release_subject: value.release_subject,
            core_commit: value.core_commit,
            host_abi_fingerprint: value.host_abi_fingerprint,
            binary_sha256: value.binary_sha256,
            plugin_sha256: value.plugin_sha256,
            matrix_sha256: value.matrix_sha256,
            capability_epoch: value.capability_epoch,
        }
    }
}

/// Request facts that are independent of device/provider selection. Provider
/// and placement are derived from the immutable `ExecutionCandidate` so callers
/// cannot present a CUDA approval while executing another lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityCellContext {
    pub pack_content_sha256: String,
    pub family: String,
    pub model_id: String,
    pub quant: String,
    pub topology: String,
    pub device_target: String,
    pub approved_target_set_sha256: Option<String>,
    pub output_plan: GgmlDecodeOutputPlan,
    pub reuse_mode: GgmlDecodeReuseMode,
    pub capture_mode: CapabilityCaptureMode,
    pub scheduler_mode: CapabilitySchedulerMode,
    pub evidence_revision: u64,
    pub activation_mode: CapabilityActivationMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct CapabilityCellKey {
    pack_content_sha256: String,
    family: String,
    model_id: String,
    quant: String,
    topology: String,
    provider: ExecutionProvider,
    device_target: String,
    approved_target_set_sha256: Option<String>,
    placement: ExecutionPlacement,
    output_plan: GgmlDecodeOutputPlan,
    reuse_mode: GgmlDecodeReuseMode,
    capture_mode: CapabilityCaptureMode,
    scheduler_mode: CapabilitySchedulerMode,
    evidence_revision: u64,
    activation_mode: CapabilityActivationMode,
}

impl CapabilityCellKey {
    pub(crate) fn from_candidate(
        candidate: &ExecutionCandidate,
        context: CapabilityCellContext,
    ) -> Result<Self, CapabilityApprovalError> {
        Self::from_approval(
            candidate.device.route.provider,
            candidate.placement,
            context,
        )
    }

    pub(crate) fn from_approval(
        provider: ExecutionProvider,
        placement: ExecutionPlacement,
        context: CapabilityCellContext,
    ) -> Result<Self, CapabilityApprovalError> {
        let key = Self {
            pack_content_sha256: context.pack_content_sha256,
            family: context.family,
            model_id: context.model_id,
            quant: context.quant,
            topology: context.topology,
            provider,
            device_target: context.device_target,
            approved_target_set_sha256: context.approved_target_set_sha256,
            placement,
            output_plan: context.output_plan,
            reuse_mode: context.reuse_mode,
            capture_mode: context.capture_mode,
            scheduler_mode: context.scheduler_mode,
            evidence_revision: context.evidence_revision,
            activation_mode: context.activation_mode,
        };
        key.validate()?;
        Ok(key)
    }

    fn validate(&self) -> Result<(), CapabilityApprovalError> {
        require_lower_hex("pack_content_sha256", &self.pack_content_sha256, 64)?;
        for (field, value) in [
            ("family", self.family.as_str()),
            ("model_id", self.model_id.as_str()),
            ("quant", self.quant.as_str()),
            ("topology", self.topology.as_str()),
            ("device_target", self.device_target.as_str()),
        ] {
            require_non_empty(field, value)?;
        }
        if let Some(digest) = &self.approved_target_set_sha256 {
            require_lower_hex("approved_target_set_sha256", digest, 64)?;
        }
        if self.evidence_revision == 0 {
            return Err(CapabilityApprovalError::InvalidField {
                field: "evidence_revision",
            });
        }
        let placement_matches = match self.placement {
            ExecutionPlacement::CpuOnly => self.provider == ExecutionProvider::Cpu,
            ExecutionPlacement::FullDevice | ExecutionPlacement::Hybrid => {
                matches!(
                    self.provider,
                    ExecutionProvider::Metal
                        | ExecutionProvider::Cuda
                        | ExecutionProvider::Hip
                        | ExecutionProvider::Vulkan
                )
            }
        };
        if !placement_matches {
            return Err(CapabilityApprovalError::CandidatePlacementMismatch);
        }
        Ok(())
    }

    fn digest(&self) -> String {
        let mut digest = Sha256::new();
        for value in [
            self.pack_content_sha256.as_str(),
            self.family.as_str(),
            self.model_id.as_str(),
            self.quant.as_str(),
            self.topology.as_str(),
            self.provider.as_str(),
            self.device_target.as_str(),
            self.approved_target_set_sha256.as_deref().unwrap_or(""),
            placement_label(self.placement),
            output_plan_label(self.output_plan),
            reuse_mode_label(self.reuse_mode),
            self.capture_mode.as_str(),
            self.scheduler_mode.as_str(),
            self.activation_mode.as_str(),
        ] {
            digest.update((value.len() as u64).to_le_bytes());
            digest.update(value.as_bytes());
        }
        digest.update(self.evidence_revision.to_le_bytes());
        format!("{:x}", digest.finalize())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CapabilityDecision {
    Activatable,
    Revoked { tombstone_sha256: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CapabilityApprovalRecord {
    pub key: CapabilityCellKey,
    pub decision: CapabilityDecision,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum StoredCapabilityDecision {
    Activatable,
    Revoked { tombstone_sha256: String },
}

#[derive(Debug)]
struct CapabilityApprovalSnapshotInner {
    binding: CapabilityArtifactBinding,
    decisions: HashMap<CapabilityCellKey, StoredCapabilityDecision>,
}

/// Immutable, already signature-verified capability state. Construction remains
/// crate-private so untrusted callers cannot mint approvals from arbitrary JSON.
#[derive(Debug, Clone)]
pub struct CapabilityApprovalSnapshot {
    inner: Arc<CapabilityApprovalSnapshotInner>,
}

impl CapabilityApprovalSnapshot {
    pub(crate) fn from_verified_records(
        binding: CapabilityArtifactBinding,
        records: impl IntoIterator<Item = CapabilityApprovalRecord>,
    ) -> Result<Self, CapabilityApprovalError> {
        binding.validate()?;
        let mut decisions = HashMap::new();
        for record in records {
            record.key.validate()?;
            let incoming = match record.decision {
                CapabilityDecision::Activatable => StoredCapabilityDecision::Activatable,
                CapabilityDecision::Revoked { tombstone_sha256 } => {
                    require_lower_hex("tombstone_sha256", &tombstone_sha256, 64)?;
                    StoredCapabilityDecision::Revoked { tombstone_sha256 }
                }
            };
            match (decisions.get(&record.key), &incoming) {
                (None, _) => {
                    decisions.insert(record.key, incoming);
                }
                (
                    Some(StoredCapabilityDecision::Activatable),
                    StoredCapabilityDecision::Revoked { .. },
                ) => {
                    decisions.insert(record.key, incoming);
                }
                (
                    Some(StoredCapabilityDecision::Revoked { .. }),
                    StoredCapabilityDecision::Activatable,
                ) => {
                    // A signed tombstone is monotonic inside one snapshot and
                    // wins regardless of record ordering.
                }
                _ => return Err(CapabilityApprovalError::DuplicateDecision),
            }
        }
        Ok(Self {
            inner: Arc::new(CapabilityApprovalSnapshotInner { binding, decisions }),
        })
    }

    pub fn attest_runtime(
        &self,
        runtime: &RuntimeCapabilityArtifactIdentity,
    ) -> Result<AttestedCapabilityApprovalSnapshot, CapabilityApprovalError> {
        let runtime = CapabilityArtifactBinding::from(runtime.clone());
        runtime.validate()?;
        for (field, matches) in [
            (
                "release_subject",
                self.inner.binding.release_subject == runtime.release_subject,
            ),
            (
                "core_commit",
                self.inner.binding.core_commit == runtime.core_commit,
            ),
            (
                "host_abi_fingerprint",
                self.inner.binding.host_abi_fingerprint == runtime.host_abi_fingerprint,
            ),
            (
                "binary_sha256",
                self.inner.binding.binary_sha256 == runtime.binary_sha256,
            ),
            (
                "plugin_sha256",
                self.inner.binding.plugin_sha256 == runtime.plugin_sha256,
            ),
            (
                "matrix_sha256",
                self.inner.binding.matrix_sha256 == runtime.matrix_sha256,
            ),
            (
                "capability_epoch",
                self.inner.binding.capability_epoch == runtime.capability_epoch,
            ),
        ] {
            if !matches {
                return Err(CapabilityApprovalError::ArtifactMismatch { field });
            }
        }
        Ok(AttestedCapabilityApprovalSnapshot {
            inner: Arc::clone(&self.inner),
        })
    }
}

/// Snapshot typestate available only after release artifact attestation.
#[derive(Debug, Clone)]
pub struct AttestedCapabilityApprovalSnapshot {
    inner: Arc<CapabilityApprovalSnapshotInner>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CapabilityApprovalIdentity {
    pub capability_epoch: u64,
    pub matrix_sha256: String,
    pub cell_sha256: String,
    pub evidence_revision: u64,
    pub activation_mode: CapabilityActivationMode,
}

/// The only candidate type allowed to cross the capability gate. Its fields are
/// private so policy, family, and product code cannot construct one directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovedExecutionCandidate {
    candidate: ExecutionCandidate,
    approval: CapabilityApprovalIdentity,
}

impl ApprovedExecutionCandidate {
    pub fn candidate(&self) -> &ExecutionCandidate {
        &self.candidate
    }

    pub fn approval(&self) -> &CapabilityApprovalIdentity {
        &self.approval
    }

    pub fn into_parts(self) -> (ExecutionCandidate, CapabilityApprovalIdentity) {
        (self.candidate, self.approval)
    }
}

#[derive(Debug, Clone)]
pub struct CapabilityApprovalResolver {
    snapshot: AttestedCapabilityApprovalSnapshot,
}

impl CapabilityApprovalResolver {
    pub fn new(snapshot: AttestedCapabilityApprovalSnapshot) -> Self {
        Self { snapshot }
    }

    pub fn approve(
        &self,
        candidate: ExecutionCandidate,
        context: CapabilityCellContext,
    ) -> Result<ApprovedExecutionCandidate, CapabilityApprovalError> {
        let key = CapabilityCellKey::from_candidate(&candidate, context)?;
        match self.snapshot.inner.decisions.get(&key) {
            Some(StoredCapabilityDecision::Activatable) => Ok(ApprovedExecutionCandidate {
                approval: CapabilityApprovalIdentity {
                    capability_epoch: self.snapshot.inner.binding.capability_epoch,
                    matrix_sha256: self.snapshot.inner.binding.matrix_sha256.clone(),
                    cell_sha256: key.digest(),
                    evidence_revision: key.evidence_revision,
                    activation_mode: key.activation_mode,
                },
                candidate,
            }),
            Some(StoredCapabilityDecision::Revoked { tombstone_sha256 }) => {
                Err(CapabilityApprovalError::Revoked {
                    tombstone_sha256: tombstone_sha256.clone(),
                })
            }
            None => Err(CapabilityApprovalError::MissingCell),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CapabilityApprovalError {
    #[error("capability approval field '{field}' is invalid")]
    InvalidField { field: &'static str },
    #[error("runtime capability artifact field '{field}' does not match the signed snapshot")]
    ArtifactMismatch { field: &'static str },
    #[error("execution candidate provider and placement are incompatible")]
    CandidatePlacementMismatch,
    #[error("capability snapshot contains a duplicate decision")]
    DuplicateDecision,
    #[error("no approved capability cell matches this exact execution candidate")]
    MissingCell,
    #[error("the exact capability cell is revoked by tombstone {tombstone_sha256}")]
    Revoked { tombstone_sha256: String },
}

fn require_non_empty(field: &'static str, value: &str) -> Result<(), CapabilityApprovalError> {
    if value.trim().is_empty() || value.trim() != value {
        Err(CapabilityApprovalError::InvalidField { field })
    } else {
        Ok(())
    }
}

fn require_lower_hex(
    field: &'static str,
    value: &str,
    length: usize,
) -> Result<(), CapabilityApprovalError> {
    if value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(CapabilityApprovalError::InvalidField { field })
    }
}

const fn placement_label(value: ExecutionPlacement) -> &'static str {
    match value {
        ExecutionPlacement::CpuOnly => "cpu_only",
        ExecutionPlacement::FullDevice => "full_device",
        ExecutionPlacement::Hybrid => "hybrid",
    }
}

const fn output_plan_label(value: GgmlDecodeOutputPlan) -> &'static str {
    match value {
        GgmlDecodeOutputPlan::FullLogits => "full_logits",
        GgmlDecodeOutputPlan::CompleteScores => "complete_scores",
        GgmlDecodeOutputPlan::NativeFirstMaxToken => "native_first_max_token",
    }
}

const fn reuse_mode_label(value: GgmlDecodeReuseMode) -> &'static str {
    match value {
        GgmlDecodeReuseMode::FreshGraph => "fresh_graph",
        GgmlDecodeReuseMode::ReusableGraph => "reusable_graph",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        device::{
            execution_policy::{ExecutionDeviceSnapshot, ExecutionPlacement},
            execution_route::{
                DeviceAddressability, PhysicalResourceKey, ResolvedExecutionRoute, RouteDeviceKind,
            },
        },
        ggml_runtime::GgmlBackendKind,
    };

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn binding() -> CapabilityArtifactBinding {
        CapabilityArtifactBinding {
            release_subject: "openasr-v0.1.36-windows-x86_64.zip".to_string(),
            core_commit: "1234567890123456789012345678901234567890".to_string(),
            host_abi_fingerprint: SHA_A.to_string(),
            binary_sha256: SHA_B.to_string(),
            plugin_sha256: SHA_C.to_string(),
            matrix_sha256: SHA_A.to_string(),
            capability_epoch: 7,
        }
    }

    fn runtime_binding() -> RuntimeCapabilityArtifactIdentity {
        let binding = binding();
        RuntimeCapabilityArtifactIdentity {
            release_subject: binding.release_subject,
            core_commit: binding.core_commit,
            host_abi_fingerprint: binding.host_abi_fingerprint,
            binary_sha256: binding.binary_sha256,
            plugin_sha256: binding.plugin_sha256,
            matrix_sha256: binding.matrix_sha256,
            capability_epoch: binding.capability_epoch,
        }
    }

    fn cuda_candidate() -> ExecutionCandidate {
        ExecutionCandidate {
            device: ExecutionDeviceSnapshot {
                route: ResolvedExecutionRoute {
                    provider: ExecutionProvider::Cuda,
                    stable_id: "CUDA0".to_string(),
                    registry_ordinal: 0,
                    kind: RouteDeviceKind::Accelerated,
                    addressability: DeviceAddressability::ExactlyAddressable {
                        physical_key: PhysicalResourceKey::new("0000:01:00.0").unwrap(),
                    },
                },
                ggml_kind: GgmlBackendKind::Gpu,
                memory: None,
                buffer_alignment: None,
            },
            placement: ExecutionPlacement::FullDevice,
        }
    }

    fn context() -> CapabilityCellContext {
        CapabilityCellContext {
            pack_content_sha256: SHA_B.to_string(),
            family: "firered-aed".to_string(),
            model_id: "firered-aed-l-v2".to_string(),
            quant: "fp16".to_string(),
            topology: "firered-aed-decoder-v1".to_string(),
            device_target: "sm_89".to_string(),
            approved_target_set_sha256: None,
            output_plan: GgmlDecodeOutputPlan::FullLogits,
            reuse_mode: GgmlDecodeReuseMode::FreshGraph,
            capture_mode: CapabilityCaptureMode::Disabled,
            scheduler_mode: CapabilitySchedulerMode::Disabled,
            evidence_revision: 1,
            activation_mode: CapabilityActivationMode::Explicit,
        }
    }

    fn key() -> CapabilityCellKey {
        CapabilityCellKey::from_candidate(&cuda_candidate(), context()).unwrap()
    }

    fn resolver(records: Vec<CapabilityApprovalRecord>) -> CapabilityApprovalResolver {
        let snapshot = CapabilityApprovalSnapshot::from_verified_records(binding(), records)
            .expect("verified snapshot");
        CapabilityApprovalResolver::new(
            snapshot
                .attest_runtime(&runtime_binding())
                .expect("runtime attestation"),
        )
    }

    #[test]
    fn exact_cell_returns_unforgeable_approved_candidate() {
        let candidate = cuda_candidate();
        let approved = resolver(vec![CapabilityApprovalRecord {
            key: key(),
            decision: CapabilityDecision::Activatable,
        }])
        .approve(candidate.clone(), context())
        .unwrap();

        assert_eq!(approved.candidate(), &candidate);
        assert_eq!(approved.approval().capability_epoch, 7);
        assert_eq!(approved.approval().matrix_sha256, SHA_A);
        assert_eq!(approved.approval().cell_sha256.len(), 64);
    }

    #[test]
    fn any_exact_cell_drift_fails_closed() {
        let resolver = resolver(vec![CapabilityApprovalRecord {
            key: key(),
            decision: CapabilityDecision::Activatable,
        }]);
        let mut changed = context();
        changed.device_target = "sm_86".to_string();
        assert_eq!(
            resolver.approve(cuda_candidate(), changed).unwrap_err(),
            CapabilityApprovalError::MissingCell
        );

        let mut changed = context();
        changed.reuse_mode = GgmlDecodeReuseMode::ReusableGraph;
        assert_eq!(
            resolver.approve(cuda_candidate(), changed).unwrap_err(),
            CapabilityApprovalError::MissingCell
        );

        let mut changed = context();
        changed.activation_mode = CapabilityActivationMode::Auto;
        assert_eq!(
            resolver.approve(cuda_candidate(), changed).unwrap_err(),
            CapabilityApprovalError::MissingCell
        );
    }

    #[test]
    fn tombstone_wins_regardless_of_record_order() {
        for records in [
            vec![
                CapabilityApprovalRecord {
                    key: key(),
                    decision: CapabilityDecision::Activatable,
                },
                CapabilityApprovalRecord {
                    key: key(),
                    decision: CapabilityDecision::Revoked {
                        tombstone_sha256: SHA_C.to_string(),
                    },
                },
            ],
            vec![
                CapabilityApprovalRecord {
                    key: key(),
                    decision: CapabilityDecision::Revoked {
                        tombstone_sha256: SHA_C.to_string(),
                    },
                },
                CapabilityApprovalRecord {
                    key: key(),
                    decision: CapabilityDecision::Activatable,
                },
            ],
        ] {
            assert_eq!(
                resolver(records)
                    .approve(cuda_candidate(), context())
                    .unwrap_err(),
                CapabilityApprovalError::Revoked {
                    tombstone_sha256: SHA_C.to_string(),
                }
            );
        }
    }

    #[test]
    fn runtime_artifact_mismatch_blocks_resolver_construction() {
        let snapshot = CapabilityApprovalSnapshot::from_verified_records(
            binding(),
            [CapabilityApprovalRecord {
                key: key(),
                decision: CapabilityDecision::Activatable,
            }],
        )
        .unwrap();
        let mut runtime = runtime_binding();
        runtime.plugin_sha256 = SHA_A.to_string();

        assert_eq!(
            snapshot.attest_runtime(&runtime).unwrap_err(),
            CapabilityApprovalError::ArtifactMismatch {
                field: "plugin_sha256",
            }
        );
    }

    #[test]
    fn invalid_candidate_placement_cannot_form_a_lookup_key() {
        let mut candidate = cuda_candidate();
        candidate.placement = ExecutionPlacement::CpuOnly;
        assert_eq!(
            CapabilityCellKey::from_candidate(&candidate, context()).unwrap_err(),
            CapabilityApprovalError::CandidatePlacementMismatch
        );
    }
}
