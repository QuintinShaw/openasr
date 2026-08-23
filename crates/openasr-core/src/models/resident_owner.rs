//! Backend-neutral identity and checkout seams for resident owners.
//!
//! `ResidentKey` and `OwnerInstanceKey` are intentionally proof-gated. Their
//! production construction starts from a `VerifiedResidentComponent` emitted by
//! the architecture descriptor's topology builder; arbitrary paths, component
//! strings, slots, and generations cannot create an owner identity.
#![allow(dead_code)]

use crate::arch::runtime_footprint::{
    ResidentDecoderCapacity, ResidentPartition, ResidentRepresentation, ResidentSessionEnvelope,
    SessionFootprintSpec, VerifiedResidentComponent,
};
use crate::ggml_runtime::{
    ResidentDeviceCopyCapability, ResidentHostImportCapability, StrongFileIdentity,
};
use crate::models::native_execution_services::{ExecutionLaneKey, NativeExecutionScopeId};

/// Stable identity of an adapter or LoRA binding. Construction is private so a
/// future verified adapter seam, rather than a caller's string, owns its value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct AdapterLoraFingerprint(String);

impl AdapterLoraFingerprint {
    fn new_verified(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }

    #[cfg(test)]
    pub(crate) fn test() -> Self {
        Self::new_verified("adapter:test")
    }
}

/// The single backend-adapter output accepted by resident owners. Its fields
/// are private so a caller cannot split the proof from the observed outcome.
#[derive(Debug)]
pub(crate) struct MaterializedResidentComponent<'a> {
    component: &'a VerifiedResidentComponent<'a>,
    outcome: MaterializedResidentOutcome,
}

#[derive(Debug)]
enum MaterializedResidentOutcome {
    HostImported(ResidentHostImportCapability),
    DeviceCopied(ResidentDeviceCopyCapability),
}

impl<'a> MaterializedResidentComponent<'a> {
    pub(crate) fn from_host_import(
        component: &'a VerifiedResidentComponent<'a>,
        capability: ResidentHostImportCapability,
    ) -> Result<Self, ResidentKeyError> {
        if !component
            .spec()
            .representations()
            .contains(&ResidentRepresentation::HostImportedBinding)
        {
            return Err(ResidentKeyError::ContractMismatch);
        }
        if !capability.proves_preflight(component.preflight()) {
            return Err(ResidentKeyError::GenerationMismatch);
        }
        Ok(Self {
            component,
            outcome: MaterializedResidentOutcome::HostImported(capability),
        })
    }

    pub(crate) fn from_device_copy(
        component: &'a VerifiedResidentComponent<'a>,
        capability: ResidentDeviceCopyCapability,
    ) -> Result<Self, ResidentKeyError> {
        if !component
            .spec()
            .representations()
            .contains(&ResidentRepresentation::DeviceCopiedBinding)
        {
            return Err(ResidentKeyError::ContractMismatch);
        }
        Ok(Self {
            component,
            outcome: MaterializedResidentOutcome::DeviceCopied(capability),
        })
    }
}

/// Identity of one resident checkout. All fields are private; production code
/// receives this only from a materialized component token.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ResidentKey {
    content: String,
    source_identity: StrongFileIdentity,
    architecture: &'static str,
    component: &'static str,
    variant: &'static str,
    placement_variant: crate::arch::runtime_footprint::ResidentPlacementVariant,
    adapter_lora_fingerprint: AdapterLoraFingerprint,
    partition: ResidentPartition,
    lane: Option<ExecutionLaneKey>,
    decoder_capacity: ResidentDecoderCapacity,
    session: ResidentSessionEnvelope,
    component_session: Option<SessionFootprintSpec>,
    scope: NativeExecutionScopeId,
}

impl ResidentKey {
    fn from_materialized(
        materialized: MaterializedResidentComponent<'_>,
        adapter_lora_fingerprint: AdapterLoraFingerprint,
        scope: NativeExecutionScopeId,
    ) -> Result<Self, ResidentKeyError> {
        let component = materialized.component;
        let (partition, lane, source_identity) = match materialized.outcome {
            MaterializedResidentOutcome::HostImported(capability) => {
                if !capability.proves_preflight(component.preflight()) {
                    return Err(ResidentKeyError::GenerationMismatch);
                }
                (
                    ResidentPartition::HostNeutral,
                    None,
                    component
                        .preflight()
                        .runtime_source()
                        .strong_file_identity(),
                )
            }
            MaterializedResidentOutcome::DeviceCopied(capability) => (
                ResidentPartition::DeviceOwning,
                Some(capability.lane().clone()),
                component
                    .preflight()
                    .runtime_source()
                    .strong_file_identity(),
            ),
        };
        Ok(Self {
            content: component.verified_pack().content_id().to_string(),
            source_identity,
            architecture: component.architecture(),
            component: component.spec().component(),
            variant: component.spec().variant(),
            placement_variant: component.resolved_variant().placement(),
            adapter_lora_fingerprint,
            partition,
            lane,
            decoder_capacity: component.session().decoder_capacity(),
            session: component.session(),
            component_session: component.component_session(),
            scope,
        })
    }

    #[cfg(test)]
    pub(crate) fn lane(&self) -> Option<&ExecutionLaneKey> {
        self.lane.as_ref()
    }

    fn validate_partition_lane(&self) -> Result<(), ResidentKeyError> {
        match (self.partition, self.lane.is_some()) {
            (ResidentPartition::DeviceOwning, false) => Err(ResidentKeyError::DeviceRequiresLane),
            (ResidentPartition::HostNeutral, true) => Err(ResidentKeyError::HostForbidsLane),
            _ => Ok(()),
        }
    }

    #[cfg(test)]
    fn test_key(
        partition: ResidentPartition,
        lane: Option<ExecutionLaneKey>,
        scope: NativeExecutionScopeId,
    ) -> Self {
        Self {
            content: "sha256:test-content".to_string(),
            source_identity: StrongFileIdentity::test_fixture(7),
            architecture: "test-architecture",
            component: "execution-state",
            variant: "v1",
            placement_variant: crate::arch::runtime_footprint::ResidentPlacementVariant::Unified,
            adapter_lora_fingerprint: AdapterLoraFingerprint::new_verified("adapter:v1"),
            partition,
            lane,
            decoder_capacity: ResidentDecoderCapacity::TokenPositions {
                self_attention: 32,
                cross_attention: 16,
            },
            session: ResidentSessionEnvelope::test_default(),
            component_session: None,
            scope,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentKeyError {
    DeviceRequiresLane,
    HostForbidsLane,
    ContractMismatch,
    GenerationMismatch,
}

/// An actual materialization binding. The only production constructor accepts
/// the indivisible backend materialization token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResidentBinding {
    HostImportedBinding { key: ResidentKey },
    DeviceCopiedBinding { key: ResidentKey },
}

impl ResidentBinding {
    pub(crate) fn from_materialized(
        materialized: MaterializedResidentComponent<'_>,
        adapter_lora_fingerprint: AdapterLoraFingerprint,
        scope: NativeExecutionScopeId,
    ) -> Result<Self, ResidentKeyError> {
        let is_device = matches!(
            &materialized.outcome,
            MaterializedResidentOutcome::DeviceCopied(_)
        );
        let key = ResidentKey::from_materialized(materialized, adapter_lora_fingerprint, scope)?;
        if is_device {
            Ok(Self::DeviceCopiedBinding { key })
        } else {
            Ok(Self::HostImportedBinding { key })
        }
    }

    fn key(&self) -> &ResidentKey {
        match self {
            Self::HostImportedBinding { key } | Self::DeviceCopiedBinding { key } => key,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::runtime_footprint::ResidentPartition;
    use crate::ggml_runtime::GgmlCpuGraphBackend;
    use crate::models::native_execution_services::{
        NativeExecutionServices, current_execution_lane_key,
    };

    fn test_scope_id() -> NativeExecutionScopeId {
        NativeExecutionServices::for_local_process()
            .expect("local execution services")
            .scope_id()
    }

    #[test]
    fn resident_key_identity_cannot_drop_contract_dimensions() {
        let lane = current_execution_lane_key(GgmlCpuGraphBackend::Cpu);
        let scope = test_scope_id();
        let base =
            ResidentKey::test_key(ResidentPartition::DeviceOwning, Some(lane.clone()), scope);

        let mut changed = base.clone();
        changed.content = "sha256:other".to_string();
        assert_ne!(base, changed);
        let mut changed = base.clone();
        changed.source_identity = StrongFileIdentity::test_fixture(8);
        assert_ne!(base, changed);
        let mut changed = base.clone();
        changed.architecture = "other-architecture";
        assert_ne!(base, changed);
        let mut changed = base.clone();
        changed.component = "other-component";
        assert_ne!(base, changed);
        let mut changed = base.clone();
        changed.variant = "v2";
        assert_ne!(base, changed);
        let mut changed = base.clone();
        changed.placement_variant = crate::arch::runtime_footprint::ResidentPlacementVariant::Split;
        assert_ne!(base, changed);
        let mut changed = base.clone();
        changed.session = ResidentSessionEnvelope::test_cohere();
        assert_ne!(base, changed);
        let mut changed = base.clone();
        changed.adapter_lora_fingerprint = AdapterLoraFingerprint::new_verified("adapter:v2");
        assert_ne!(base, changed);
        let mut changed = base.clone();
        changed.partition = ResidentPartition::HostNeutral;
        changed.lane = None;
        assert_ne!(base, changed);
        let mut changed = base.clone();
        changed.lane = None;
        assert_ne!(base, changed);
        let mut changed = base.clone();
        changed.decoder_capacity = ResidentDecoderCapacity::TokenPositions {
            self_attention: 64,
            cross_attention: 16,
        };
        assert_ne!(base, changed);
        let mut changed = base.clone();
        changed.scope = test_scope_id();
        assert_ne!(base, changed);
        assert!(base.validate_partition_lane().is_ok());
    }

    #[test]
    fn partition_rejects_lane_escaping_and_binding_requires_lane_for_copy() {
        let lane = current_execution_lane_key(GgmlCpuGraphBackend::Cpu);
        let scope = test_scope_id();
        let invalid_device = ResidentKey::test_key(ResidentPartition::DeviceOwning, None, scope);
        assert_eq!(
            invalid_device.validate_partition_lane(),
            Err(ResidentKeyError::DeviceRequiresLane)
        );
        let invalid_host = ResidentKey::test_key(ResidentPartition::HostNeutral, Some(lane), scope);
        assert_eq!(
            invalid_host.validate_partition_lane(),
            Err(ResidentKeyError::HostForbidsLane)
        );
    }

    #[test]
    fn binding_enum_keeps_host_import_lane_free_and_device_copy_lane_bound() {
        // The constructors are proof-gated in production; this test locks the
        // resulting shape without introducing a second runtime path.
        let host = ResidentKey::test_key(ResidentPartition::HostNeutral, None, test_scope_id());
        let device = ResidentKey::test_key(
            ResidentPartition::DeviceOwning,
            Some(current_execution_lane_key(GgmlCpuGraphBackend::Cpu)),
            test_scope_id(),
        );
        let host_binding = ResidentBinding::HostImportedBinding { key: host };
        let device_binding = ResidentBinding::DeviceCopiedBinding { key: device };
        assert!(host_binding.key().lane.is_none());
        assert!(device_binding.key().lane.is_some());
    }
}
