//! Signed exact-cell approval rows carried by the ordinary model catalog.
//!
//! Qualification manifests never enter this module. Rows here are already in
//! the normal runtime policy authority and therefore must be activatable or
//! revoked, structurally bound to catalog model/backend artifacts, and projected
//! through the typed capability resolver.

use serde::{Deserialize, Serialize};

use super::{
    CatalogBackendFileRole, CatalogBackendVendor, CatalogError, ModelCatalog,
    resolve_catalog_backend_pull,
};
use crate::{
    CapabilityActivationMode, CapabilityApprovalSnapshot, CapabilityArtifactBinding,
    CapabilityCaptureMode, CapabilityCellContext, CapabilitySchedulerMode,
    capability_approval::{CapabilityApprovalRecord, CapabilityCellKey, CapabilityDecision},
    device::{execution_policy::ExecutionPlacement, execution_route::ExecutionProvider},
    ggml_runtime::{GgmlDecodeOutputPlan, GgmlDecodeReuseMode},
};

pub const CATALOG_EXECUTION_APPROVAL_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogExecutionApprovalSet {
    pub schema_version: u32,
    pub release_subject: String,
    pub core_commit: String,
    pub binary_sha256: String,
    pub matrix_sha256: String,
    pub capability_epoch: u64,
    pub cells: Vec<CatalogExecutionApprovalCell>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogExecutionApprovalCell {
    pub pack_content_sha256: String,
    pub family: String,
    pub model_id: String,
    pub quant: String,
    pub topology: String,
    pub provider: CatalogExecutionProvider,
    pub device_target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approved_target_set_sha256: Option<String>,
    pub placement: CatalogExecutionPlacement,
    pub output_plan: CatalogExecutionOutputPlan,
    pub reuse_mode: CatalogExecutionReuseMode,
    pub capture_mode: CatalogExecutionCaptureMode,
    pub scheduler_mode: CatalogExecutionSchedulerMode,
    pub evidence_revision: u64,
    pub activation_modes: Vec<CatalogExecutionActivationMode>,
    pub plugin_sha256: String,
    pub decision: CatalogExecutionApprovalDecision,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tombstone_sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogExecutionProvider {
    Cpu,
    Metal,
    Cuda,
    Hip,
    Vulkan,
    #[serde(other)]
    Unknown,
}

impl CatalogExecutionProvider {
    fn runtime(self) -> Option<ExecutionProvider> {
        match self {
            Self::Cpu => Some(ExecutionProvider::Cpu),
            Self::Metal => Some(ExecutionProvider::Metal),
            Self::Cuda => Some(ExecutionProvider::Cuda),
            Self::Hip => Some(ExecutionProvider::Hip),
            Self::Vulkan => Some(ExecutionProvider::Vulkan),
            Self::Unknown => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogExecutionPlacement {
    CpuOnly,
    FullDevice,
    Hybrid,
    #[serde(other)]
    Unknown,
}

impl CatalogExecutionPlacement {
    fn runtime(self) -> Option<ExecutionPlacement> {
        match self {
            Self::CpuOnly => Some(ExecutionPlacement::CpuOnly),
            Self::FullDevice => Some(ExecutionPlacement::FullDevice),
            Self::Hybrid => Some(ExecutionPlacement::Hybrid),
            Self::Unknown => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogExecutionOutputPlan {
    FullLogits,
    CompleteScores,
    NativeFirstMaxToken,
    #[serde(other)]
    Unknown,
}

impl CatalogExecutionOutputPlan {
    fn runtime(self) -> Option<GgmlDecodeOutputPlan> {
        match self {
            Self::FullLogits => Some(GgmlDecodeOutputPlan::FullLogits),
            Self::CompleteScores => Some(GgmlDecodeOutputPlan::CompleteScores),
            Self::NativeFirstMaxToken => Some(GgmlDecodeOutputPlan::NativeFirstMaxToken),
            Self::Unknown => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogExecutionReuseMode {
    FreshGraph,
    ReusableGraph,
    #[serde(other)]
    Unknown,
}

impl CatalogExecutionReuseMode {
    fn runtime(self) -> Option<GgmlDecodeReuseMode> {
        match self {
            Self::FreshGraph => Some(GgmlDecodeReuseMode::FreshGraph),
            Self::ReusableGraph => Some(GgmlDecodeReuseMode::ReusableGraph),
            Self::Unknown => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogExecutionCaptureMode {
    Disabled,
    Enabled,
    Unsupported,
    #[serde(other)]
    Unknown,
}

impl CatalogExecutionCaptureMode {
    fn runtime(self) -> Option<CapabilityCaptureMode> {
        match self {
            Self::Disabled => Some(CapabilityCaptureMode::Disabled),
            Self::Enabled => Some(CapabilityCaptureMode::Enabled),
            Self::Unsupported => Some(CapabilityCaptureMode::Unsupported),
            Self::Unknown => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogExecutionSchedulerMode {
    Disabled,
    Enabled,
    #[serde(other)]
    Unknown,
}

impl CatalogExecutionSchedulerMode {
    fn runtime(self) -> Option<CapabilitySchedulerMode> {
        match self {
            Self::Disabled => Some(CapabilitySchedulerMode::Disabled),
            Self::Enabled => Some(CapabilitySchedulerMode::Enabled),
            Self::Unknown => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogExecutionActivationMode {
    Auto,
    Explicit,
    #[serde(other)]
    Unknown,
}

impl CatalogExecutionActivationMode {
    fn runtime(self) -> Option<CapabilityActivationMode> {
        match self {
            Self::Auto => Some(CapabilityActivationMode::Auto),
            Self::Explicit => Some(CapabilityActivationMode::Explicit),
            Self::Unknown => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogExecutionApprovalDecision {
    Activatable,
    Revoked,
    QualificationOnly,
    #[serde(other)]
    Unknown,
}

fn catalog_backend_provider(vendor: CatalogBackendVendor) -> Option<ExecutionProvider> {
    match vendor {
        CatalogBackendVendor::Cpu => Some(ExecutionProvider::Cpu),
        CatalogBackendVendor::Vulkan => Some(ExecutionProvider::Vulkan),
        CatalogBackendVendor::Hip => Some(ExecutionProvider::Hip),
        CatalogBackendVendor::Cuda => Some(ExecutionProvider::Cuda),
        CatalogBackendVendor::Unknown => None,
    }
}

impl ModelCatalog {
    /// Project the signed exact-cell rows for one installed/activated backend
    /// into the immutable runtime snapshot. An absent approval set is a normal
    /// fail-closed state for qualification-only or CPU-only catalogs.
    pub fn capability_approval_snapshot_for_backend(
        &self,
        backend_id: &str,
    ) -> Result<Option<CapabilityApprovalSnapshot>, CatalogError> {
        let Some(approvals) = &self.execution_approvals else {
            return Ok(None);
        };
        validate_catalog_execution_approvals(self, approvals)?;
        let backend = resolve_catalog_backend_pull(self, backend_id)
            .map_err(|error| CatalogError::InvalidCatalog(error.to_string()))?;
        let plugin = backend
            .files
            .iter()
            .find(|file| file.role == CatalogBackendFileRole::Plugin)
            .ok_or_else(|| {
                CatalogError::InvalidCatalog(format!(
                    "backend '{}' has no plugin artifact",
                    backend.backend_id
                ))
            })?;
        let provider = catalog_backend_provider(backend.vendor).ok_or_else(|| {
            CatalogError::InvalidCatalog(format!(
                "backend '{}' has no runtime provider",
                backend.backend_id
            ))
        })?;
        let binding = CapabilityArtifactBinding {
            release_subject: approvals.release_subject.clone(),
            core_commit: approvals.core_commit.clone(),
            host_abi_fingerprint: backend.host_abi.fingerprint.clone(),
            binary_sha256: approvals.binary_sha256.clone(),
            plugin_sha256: plugin.sha256.clone(),
            matrix_sha256: approvals.matrix_sha256.clone(),
            capability_epoch: approvals.capability_epoch,
        };
        let mut records = Vec::new();
        for cell in approvals.cells.iter().filter(|cell| {
            cell.provider.runtime() == Some(provider) && cell.plugin_sha256 == plugin.sha256
        }) {
            let placement = cell.placement.runtime().ok_or_else(|| {
                CatalogError::InvalidCatalog("execution approval has unknown placement".to_string())
            })?;
            let output_plan = cell.output_plan.runtime().ok_or_else(|| {
                CatalogError::InvalidCatalog(
                    "execution approval has unknown output plan".to_string(),
                )
            })?;
            let reuse_mode = cell.reuse_mode.runtime().ok_or_else(|| {
                CatalogError::InvalidCatalog(
                    "execution approval has unknown reuse mode".to_string(),
                )
            })?;
            let capture_mode = cell.capture_mode.runtime().ok_or_else(|| {
                CatalogError::InvalidCatalog(
                    "execution approval has unknown capture mode".to_string(),
                )
            })?;
            let scheduler_mode = cell.scheduler_mode.runtime().ok_or_else(|| {
                CatalogError::InvalidCatalog(
                    "execution approval has unknown scheduler mode".to_string(),
                )
            })?;
            for activation_mode in &cell.activation_modes {
                let activation_mode = activation_mode.runtime().ok_or_else(|| {
                    CatalogError::InvalidCatalog(
                        "execution approval has unknown activation mode".to_string(),
                    )
                })?;
                let key = CapabilityCellKey::from_approval(
                    provider,
                    placement,
                    CapabilityCellContext {
                        pack_content_sha256: cell.pack_content_sha256.clone(),
                        family: cell.family.clone(),
                        model_id: cell.model_id.clone(),
                        quant: cell.quant.clone(),
                        topology: cell.topology.clone(),
                        device_target: cell.device_target.clone(),
                        approved_target_set_sha256: cell.approved_target_set_sha256.clone(),
                        output_plan,
                        reuse_mode,
                        capture_mode,
                        scheduler_mode,
                        evidence_revision: cell.evidence_revision,
                        activation_mode,
                    },
                )
                .map_err(|error| CatalogError::InvalidCatalog(error.to_string()))?;
                let decision = match cell.decision {
                    CatalogExecutionApprovalDecision::Activatable => {
                        CapabilityDecision::Activatable
                    }
                    CatalogExecutionApprovalDecision::Revoked => CapabilityDecision::Revoked {
                        tombstone_sha256: cell.tombstone_sha256.clone().ok_or_else(|| {
                            CatalogError::InvalidCatalog(
                                "revoked execution approval has no tombstone digest".to_string(),
                            )
                        })?,
                    },
                    CatalogExecutionApprovalDecision::QualificationOnly => {
                        return Err(CatalogError::InvalidCatalog(
                            "qualification_only cannot enter ordinary execution approvals"
                                .to_string(),
                        ));
                    }
                    CatalogExecutionApprovalDecision::Unknown => {
                        return Err(CatalogError::InvalidCatalog(
                            "execution approval has unknown decision".to_string(),
                        ));
                    }
                };
                records.push(CapabilityApprovalRecord { key, decision });
            }
        }
        CapabilityApprovalSnapshot::from_verified_records(binding, records)
            .map(Some)
            .map_err(|error| CatalogError::InvalidCatalog(error.to_string()))
    }
}

pub(super) fn validate_catalog_execution_approvals(
    catalog: &ModelCatalog,
    approvals: &CatalogExecutionApprovalSet,
) -> Result<(), CatalogError> {
    let invalid = |reason: String| CatalogError::InvalidCatalog(reason);
    if approvals.schema_version != CATALOG_EXECUTION_APPROVAL_SCHEMA_VERSION {
        return Err(invalid(format!(
            "execution approvals schema {} is unsupported",
            approvals.schema_version
        )));
    }
    if approvals.release_subject.trim().is_empty()
        || approvals.release_subject.trim() != approvals.release_subject
        || approvals.capability_epoch == 0
        || approvals.cells.is_empty()
    {
        return Err(invalid(
            "execution approvals release subject, epoch, and cells must be present".to_string(),
        ));
    }
    if approvals.core_commit.len() != 40
        || !approvals
            .core_commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(
            "execution approvals core_commit must be lowercase 40-hex".to_string(),
        ));
    }
    for (field, value) in [
        ("binary_sha256", approvals.binary_sha256.as_str()),
        ("matrix_sha256", approvals.matrix_sha256.as_str()),
    ] {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(invalid(format!(
                "execution approvals {field} must be lowercase 64-hex"
            )));
        }
    }
    for cell in &approvals.cells {
        let provider = cell
            .provider
            .runtime()
            .ok_or_else(|| invalid("execution approval has unknown provider".to_string()))?;
        let placement = cell
            .placement
            .runtime()
            .ok_or_else(|| invalid("execution approval has unknown placement".to_string()))?;
        let output_plan = cell
            .output_plan
            .runtime()
            .ok_or_else(|| invalid("execution approval has unknown output plan".to_string()))?;
        let reuse_mode = cell
            .reuse_mode
            .runtime()
            .ok_or_else(|| invalid("execution approval has unknown reuse mode".to_string()))?;
        let capture_mode = cell
            .capture_mode
            .runtime()
            .ok_or_else(|| invalid("execution approval has unknown capture mode".to_string()))?;
        let scheduler_mode = cell
            .scheduler_mode
            .runtime()
            .ok_or_else(|| invalid("execution approval has unknown scheduler mode".to_string()))?;
        if matches!(
            cell.decision,
            CatalogExecutionApprovalDecision::QualificationOnly
                | CatalogExecutionApprovalDecision::Unknown
        ) {
            return Err(invalid(
                "qualification-only or unknown decisions cannot enter ordinary execution approvals"
                    .to_string(),
            ));
        }
        match (&cell.decision, &cell.tombstone_sha256) {
            (CatalogExecutionApprovalDecision::Activatable, None) => {}
            (CatalogExecutionApprovalDecision::Revoked, Some(digest))
                if digest.len() == 64
                    && digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) => {}
            _ => {
                return Err(invalid(
                    "execution approval decision/tombstone fields are inconsistent".to_string(),
                ));
            }
        }
        if cell.plugin_sha256.len() != 64
            || !cell
                .plugin_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(invalid(
                "execution approval plugin_sha256 must be lowercase 64-hex".to_string(),
            ));
        }
        let model = catalog
            .models
            .iter()
            .find(|model| model.id == cell.model_id && model.family == cell.family)
            .ok_or_else(|| {
                invalid(format!(
                    "execution approval references unknown model/family '{}'/ '{}'",
                    cell.model_id, cell.family
                ))
            })?;
        let quant = model
            .quants
            .iter()
            .find(|quant| quant.quant == cell.quant)
            .ok_or_else(|| {
                invalid(format!(
                    "execution approval references unknown quant '{}:{}'",
                    cell.model_id, cell.quant
                ))
            })?;
        if quant.sha256 != cell.pack_content_sha256 {
            return Err(invalid(format!(
                "execution approval pack digest does not match '{}:{}'",
                cell.model_id, cell.quant
            )));
        }
        let plugin_matches = match provider {
            ExecutionProvider::Cpu | ExecutionProvider::Metal => {
                cell.plugin_sha256 == approvals.binary_sha256
            }
            ExecutionProvider::Cuda | ExecutionProvider::Hip | ExecutionProvider::Vulkan => {
                catalog.backends.iter().any(|backend| {
                    catalog_backend_provider(backend.vendor) == Some(provider)
                        && backend.files.iter().any(|file| {
                            file.role == CatalogBackendFileRole::Plugin
                                && file.sha256 == cell.plugin_sha256
                        })
                })
            }
            ExecutionProvider::Accelerator | ExecutionProvider::Unknown => false,
        };
        if !plugin_matches {
            return Err(invalid(format!(
                "execution approval plugin does not match provider '{}'",
                provider.as_str()
            )));
        }
        if cell.activation_modes.is_empty() {
            return Err(invalid(
                "execution approval must contain at least one activation mode".to_string(),
            ));
        }
        let mut seen_modes = std::collections::BTreeSet::new();
        for activation_mode in &cell.activation_modes {
            let activation_mode = activation_mode.runtime().ok_or_else(|| {
                invalid("execution approval has unknown activation mode".to_string())
            })?;
            if !seen_modes.insert(activation_mode.as_str()) {
                return Err(invalid(
                    "execution approval has duplicate activation mode".to_string(),
                ));
            }
            CapabilityCellKey::from_approval(
                provider,
                placement,
                CapabilityCellContext {
                    pack_content_sha256: cell.pack_content_sha256.clone(),
                    family: cell.family.clone(),
                    model_id: cell.model_id.clone(),
                    quant: cell.quant.clone(),
                    topology: cell.topology.clone(),
                    device_target: cell.device_target.clone(),
                    approved_target_set_sha256: cell.approved_target_set_sha256.clone(),
                    output_plan,
                    reuse_mode,
                    capture_mode,
                    scheduler_mode,
                    evidence_revision: cell.evidence_revision,
                    activation_mode,
                },
            )
            .map_err(|error| invalid(error.to_string()))?;
        }
    }
    Ok(())
}
