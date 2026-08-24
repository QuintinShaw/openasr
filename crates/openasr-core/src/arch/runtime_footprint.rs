//! Backend-neutral resident-memory ownership contracts.
//!
//! The facet describes semantic resident topology only. It never prices native
//! allocations, names a provider, or chooses a platform path. A descriptor-owned
//! builder turns one verified source and execution candidate into a topology;
//! backend code later materializes that topology as a host import or a device
//! copy.
#![allow(dead_code)]

use crate::device::execution_policy::{ExecutionCandidate, ExecutionIntent};
use crate::ggml_runtime::GgufRuntimeSourcePreflight;
use crate::models::pack_verifier::VerifiedPack;

/// The semantic placement partition of a resident component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ResidentPartition {
    /// The component is not tied to a physical execution lane.
    HostNeutral,
    /// The component owns state on one physical execution lane.
    DeviceOwning,
}

/// The owner boundary required by a resident component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ResidentOwner {
    HostScope,
    DeviceLane,
}

impl ResidentPartition {
    pub(crate) const fn required_owner(self) -> ResidentOwner {
        match self {
            Self::HostNeutral => ResidentOwner::HostScope,
            Self::DeviceOwning => ResidentOwner::DeviceLane,
        }
    }
}

/// Backend-neutral representation alternatives declared by a component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ResidentRepresentation {
    HostImportedBinding,
    DeviceCopiedBinding,
}

/// Whether a component has one unified resident or a split resident variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ResidentPlacementVariant {
    Unified,
    Split,
}

/// Lifecycle phase in which a resident component is materialized or used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ResidentPhase {
    Load,
    Prepare,
    Execute,
    Stream,
}

/// Lifetime boundary of a resident component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ResidentLifetime {
    Process,
    ExecutionScope,
    Request,
    Session,
}

/// Checkout discipline for an owner-backed resident component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ResidentCheckout {
    Serialized { max_instances: u16 },
    Bounded { max_instances: u16 },
}

impl ResidentCheckout {
    pub(crate) const fn serialized() -> Self {
        Self::Serialized { max_instances: 1 }
    }

    pub(crate) const fn bounded(max_instances: u16) -> Self {
        Self::Bounded { max_instances }
    }

    pub(crate) const fn max_instances(self) -> u16 {
        match self {
            Self::Serialized { max_instances } | Self::Bounded { max_instances } => max_instances,
        }
    }

    pub(crate) const fn is_serialized(self) -> bool {
        matches!(self, Self::Serialized { .. })
    }
}

/// One retained serve-batch topology variant. A variant owns a graph width and
/// the number of simultaneously live slots for that width; keeping the pair
/// together prevents a width list and a slot limit from drifting apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ServeBatchVariant {
    pub(crate) width: u16,
    pub(crate) slots: u16,
}

impl ServeBatchVariant {
    const fn new(width: u16, slots: u16) -> Self {
        Self { width, slots }
    }
}

/// Retained runtime variants and active session limits for serve-batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ServeBatchSpec {
    pub(crate) retained_variants: &'static [ServeBatchVariant],
    pub(crate) active_slots: u16,
    pub(crate) max_sessions: u16,
}

impl ServeBatchSpec {
    const fn new(
        retained_variants: &'static [ServeBatchVariant],
        active_slots: u16,
        max_sessions: u16,
    ) -> Self {
        Self {
            retained_variants,
            active_slots,
            max_sessions,
        }
    }

    fn validate(self) -> Result<(), ResidentFootprintValidationError> {
        if self.retained_variants.is_empty() {
            return Err(ResidentFootprintValidationError::EmptyServeBatchVariants);
        }
        if self.active_slots == 0 || self.max_sessions == 0 {
            return Err(ResidentFootprintValidationError::ZeroSessionLimit);
        }
        if self.active_slots > self.max_sessions {
            return Err(ResidentFootprintValidationError::ActiveSlotsExceedSessions);
        }
        for (index, variant) in self.retained_variants.iter().enumerate() {
            if variant.width == 0 {
                return Err(ResidentFootprintValidationError::ZeroServeBatchWidth);
            }
            if variant.slots == 0 {
                return Err(ResidentFootprintValidationError::ZeroServeBatchSlots);
            }
            if variant.slots > self.active_slots {
                return Err(ResidentFootprintValidationError::VariantSlotsExceedActiveSlots);
            }
            if self.retained_variants[..index]
                .iter()
                .any(|previous| previous.width == variant.width)
            {
                return Err(ResidentFootprintValidationError::DuplicateServeBatchWidth);
            }
        }
        Ok(())
    }
}

/// Request/session-scoped semantic state. It deliberately contains no byte
/// quote; shape oracles and the selected backend own physical pricing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SessionFootprintSpec {
    decoder_capacity: ResidentDecoderCapacity,
    active_slots: u16,
    max_sessions: u16,
    serve_batch: Option<ServeBatchSpec>,
}

impl SessionFootprintSpec {
    const fn new(
        decoder_capacity: ResidentDecoderCapacity,
        active_slots: u16,
        max_sessions: u16,
        serve_batch: Option<ServeBatchSpec>,
    ) -> Self {
        Self {
            decoder_capacity,
            active_slots,
            max_sessions,
            serve_batch,
        }
    }

    fn validate(self) -> Result<(), ResidentFootprintValidationError> {
        if self.active_slots == 0 || self.max_sessions == 0 {
            return Err(ResidentFootprintValidationError::ZeroSessionLimit);
        }
        if self.active_slots > self.max_sessions {
            return Err(ResidentFootprintValidationError::ActiveSlotsExceedSessions);
        }
        if let Some(serve_batch) = self.serve_batch {
            serve_batch.validate()?;
            if serve_batch.max_sessions > self.max_sessions {
                return Err(ResidentFootprintValidationError::ServeBatchExceedsSessions);
            }
        }
        Ok(())
    }
}

/// One descriptor-owned semantic resident component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentComponentSpec {
    component: &'static str,
    variant: &'static str,
    phase: ResidentPhase,
    lifetime: ResidentLifetime,
    dependencies: &'static [&'static str],
    representations: &'static [ResidentRepresentation],
    placement_variants: &'static [ResidentPlacementVariant],
    checkout: ResidentCheckout,
    session: Option<SessionFootprintSpec>,
}

impl ResidentComponentSpec {
    const fn new(
        component: &'static str,
        variant: &'static str,
        phase: ResidentPhase,
        lifetime: ResidentLifetime,
        dependencies: &'static [&'static str],
        representations: &'static [ResidentRepresentation],
        placement_variants: &'static [ResidentPlacementVariant],
        checkout: ResidentCheckout,
        session: Option<SessionFootprintSpec>,
    ) -> Self {
        Self {
            component,
            variant,
            phase,
            lifetime,
            dependencies,
            representations,
            placement_variants,
            checkout,
            session,
        }
    }
    pub(crate) fn component(&self) -> &'static str {
        self.component
    }

    pub(crate) fn variant(&self) -> &'static str {
        self.variant
    }

    pub(crate) fn phase(&self) -> ResidentPhase {
        self.phase
    }

    pub(crate) fn lifetime(&self) -> ResidentLifetime {
        self.lifetime
    }

    pub(crate) fn dependencies(&self) -> &'static [&'static str] {
        self.dependencies
    }

    pub(crate) fn representations(&self) -> &'static [ResidentRepresentation] {
        self.representations
    }

    pub(crate) fn placement_variants(&self) -> &'static [ResidentPlacementVariant] {
        self.placement_variants
    }

    pub(crate) fn checkout(&self) -> ResidentCheckout {
        self.checkout
    }

    pub(crate) fn session(&self) -> Option<SessionFootprintSpec> {
        self.session
    }
}

/// Architecture identity is created by the descriptor seam, not supplied as a
/// free-form component/model string by a family module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct ResidentArchitectureId(&'static str);

impl ResidentArchitectureId {
    pub(super) const fn from_descriptor(model_architecture: &'static str) -> Self {
        Self(model_architecture)
    }

    fn as_str(self) -> &'static str {
        self.0
    }
}

/// Inputs are private to the architecture module. In production the preflight
/// is always borrowed from the proof; the test-only alternate constructor exists
/// solely to prove generation mismatch rejection.
pub(super) struct ResidentTopologyInputs<'a> {
    verified_pack: &'a VerifiedPack,
    preflight: &'a GgufRuntimeSourcePreflight,
    candidate: &'a ExecutionCandidate,
    intent: &'a ExecutionIntent,
    session: &'a ResidentSessionEnvelope,
    allow_unified_runtime: bool,
}

impl<'a> ResidentTopologyInputs<'a> {
    pub(super) fn new(
        verified_pack: &'a VerifiedPack,
        candidate: &'a ExecutionCandidate,
        intent: &'a ExecutionIntent,
        session: &'a ResidentSessionEnvelope,
        allow_unified_runtime: bool,
    ) -> Self {
        Self {
            verified_pack,
            preflight: verified_pack.preflight(),
            candidate,
            intent,
            session,
            allow_unified_runtime,
        }
    }

    #[cfg(test)]
    fn with_preflight_for_test(
        verified_pack: &'a VerifiedPack,
        preflight: &'a GgufRuntimeSourcePreflight,
        candidate: &'a ExecutionCandidate,
        intent: &'a ExecutionIntent,
        session: &'a ResidentSessionEnvelope,
    ) -> Self {
        Self {
            verified_pack,
            preflight,
            candidate,
            intent,
            session,
            allow_unified_runtime: false,
        }
    }
}

/// Dynamic request envelope used only for semantic session planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ResidentSessionEnvelope {
    decoder_capacity: ResidentDecoderCapacity,
    active_slots: u16,
    max_sessions: u16,
    serve_batch: Option<ServeBatchSpec>,
    deferred_dynamic_session: bool,
}

impl ResidentSessionEnvelope {
    const fn new(
        decoder_capacity: ResidentDecoderCapacity,
        active_slots: u16,
        max_sessions: u16,
        serve_batch: Option<ServeBatchSpec>,
    ) -> Self {
        Self {
            decoder_capacity,
            active_slots,
            max_sessions,
            serve_batch,
            deferred_dynamic_session: false,
        }
    }

    pub(crate) const fn decoder_capacity(self) -> ResidentDecoderCapacity {
        self.decoder_capacity
    }

    pub(crate) const fn active_slots(self) -> u16 {
        self.active_slots
    }

    pub(crate) const fn max_sessions(self) -> u16 {
        self.max_sessions
    }

    pub(crate) const fn serve_batch(self) -> Option<ServeBatchSpec> {
        self.serve_batch
    }

    /// Envelope for Prepare/Load quoting. Request/session components are not
    /// reserved here; they stay JIT behind the prepared topology.
    pub(crate) const fn activation_prepare() -> Self {
        Self {
            decoder_capacity: ResidentDecoderCapacity::None,
            active_slots: 1,
            max_sessions: 1,
            serve_batch: None,
            deferred_dynamic_session: true,
        }
    }

    #[cfg(test)]
    pub(crate) const fn test_default() -> Self {
        Self::activation_prepare()
    }

    #[cfg(test)]
    pub(crate) const fn test_cohere() -> Self {
        Self {
            decoder_capacity: ResidentDecoderCapacity::TokenPositions {
                self_attention: 0,
                cross_attention: 0,
            },
            active_slots: 1,
            max_sessions: 4,
            serve_batch: Some(SERVE_BATCH),
            deferred_dynamic_session: false,
        }
    }

    fn validate(self) -> Result<(), ResidentTopologyError> {
        SessionFootprintSpec::new(
            self.decoder_capacity,
            self.active_slots,
            self.max_sessions,
            self.serve_batch,
        )
        .validate()
        .map_err(ResidentTopologyError::InvalidSessionEnvelope)
    }
}

/// The topology variant resolved by the descriptor from the execution shape.
/// Representation alternatives remain unresolved until a backend reports the
/// actual host-import or device-copy outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ResidentResolvedVariant {
    placement: ResidentPlacementVariant,
}

impl ResidentResolvedVariant {
    pub(crate) const fn placement(self) -> ResidentPlacementVariant {
        self.placement
    }

    fn for_placement(
        placement: crate::device::execution_policy::ExecutionPlacement,
        declared: &[ResidentPlacementVariant],
    ) -> Result<Self, ResidentTopologyError> {
        let resolved = match placement {
            crate::device::execution_policy::ExecutionPlacement::Hybrid => {
                ResidentPlacementVariant::Split
            }
            crate::device::execution_policy::ExecutionPlacement::CpuOnly
            | crate::device::execution_policy::ExecutionPlacement::FullDevice => {
                ResidentPlacementVariant::Unified
            }
        };
        if declared.contains(&resolved) {
            Ok(Self {
                placement: resolved,
            })
        } else {
            Err(ResidentTopologyError::PlacementVariantUnavailable { resolved })
        }
    }
}

/// A proof-bearing component token emitted only by the descriptor-owned builder.
/// Owner keys accept this token instead of arbitrary component/path strings.
#[derive(Debug)]
pub(crate) struct VerifiedResidentComponent<'a> {
    architecture: &'static str,
    spec: &'static ResidentComponentSpec,
    verified_pack: &'a VerifiedPack,
    preflight: &'a GgufRuntimeSourcePreflight,
    candidate: &'a ExecutionCandidate,
    intent: &'a ExecutionIntent,
    resolved_variant: ResidentResolvedVariant,
    session: ResidentSessionEnvelope,
    component_session: Option<SessionFootprintSpec>,
}

impl<'a> VerifiedResidentComponent<'a> {
    pub(crate) fn architecture(&self) -> &'static str {
        self.architecture
    }

    pub(crate) fn spec(&self) -> &'static ResidentComponentSpec {
        self.spec
    }

    pub(crate) fn verified_pack(&self) -> &'a VerifiedPack {
        self.verified_pack
    }

    pub(crate) fn preflight(&self) -> &'a GgufRuntimeSourcePreflight {
        self.preflight
    }

    pub(crate) fn candidate(&self) -> &'a ExecutionCandidate {
        self.candidate
    }

    pub(crate) fn intent(&self) -> &'a ExecutionIntent {
        self.intent
    }

    pub(crate) const fn resolved_variant(&self) -> ResidentResolvedVariant {
        self.resolved_variant
    }

    pub(crate) const fn session(&self) -> ResidentSessionEnvelope {
        self.session
    }

    pub(crate) fn dependencies(&self) -> &'static [&'static str] {
        self.spec.dependencies()
    }

    pub(crate) const fn component_session(&self) -> Option<SessionFootprintSpec> {
        self.component_session
    }
}

/// One component in a built topology, retaining its proof-bearing token and
/// the descriptor-resolved unified/split variant.
pub(crate) struct ResidentTopologyComponent<'a> {
    verified: VerifiedResidentComponent<'a>,
}

impl<'a> ResidentTopologyComponent<'a> {
    pub(crate) fn verified(&self) -> &VerifiedResidentComponent<'a> {
        &self.verified
    }
}

/// Descriptor-owned semantic topology. No native bytes or platform branches
/// are present; the later backend adapter consumes the proof-bearing components.
pub(crate) struct ResidentTopology<'a> {
    architecture: &'static str,
    components: Vec<ResidentTopologyComponent<'a>>,
    dependency_order: Vec<&'static str>,
    session: ResidentSessionEnvelope,
}

impl<'a> ResidentTopology<'a> {
    pub(crate) fn architecture(&self) -> &'static str {
        self.architecture
    }

    pub(crate) fn components(&self) -> &[ResidentTopologyComponent<'a>] {
        &self.components
    }

    pub(crate) fn dependency_order(&self) -> &[&'static str] {
        &self.dependency_order
    }

    pub(crate) const fn session(&self) -> ResidentSessionEnvelope {
        self.session
    }
}

/// Mandatory resident footprint facet of one architecture descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentFootprintFacet {
    components: &'static [ResidentComponentSpec],
}

fn resolve_dependency_order(
    components: &[&'static ResidentComponentSpec],
) -> Result<Vec<&'static str>, ResidentTopologyError> {
    let mut order = Vec::with_capacity(components.len());
    while order.len() < components.len() {
        let mut progressed = false;
        for component in components {
            if order.contains(&component.component()) {
                continue;
            }
            if component
                .dependencies()
                .iter()
                .all(|dependency| order.contains(dependency))
            {
                order.push(component.component());
                progressed = true;
            }
        }
        if !progressed {
            return Err(ResidentTopologyError::DependencyCycle);
        }
    }
    Ok(order)
}

fn validate_proof_preflight_identity(
    verified_pack: &VerifiedPack,
    preflight: &GgufRuntimeSourcePreflight,
) -> Result<(), ResidentTopologyError> {
    let verified_source = verified_pack.preflight().runtime_source();
    let supplied_source = preflight.runtime_source();
    if verified_source.strong_file_identity() != supplied_source.strong_file_identity() {
        return Err(ResidentTopologyError::GenerationMismatch);
    }
    if verified_source.content_id() != supplied_source.content_id() {
        return Err(ResidentTopologyError::ProofPreflightMismatch);
    }
    Ok(())
}

fn validate_session_envelope_for_component(
    envelope: &ResidentSessionEnvelope,
    component: SessionFootprintSpec,
) -> Result<(), ResidentTopologyError> {
    if envelope.decoder_capacity != component.decoder_capacity
        || envelope.active_slots != component.active_slots
        || envelope.max_sessions > component.max_sessions
    {
        return Err(ResidentTopologyError::SessionEnvelopeMismatch);
    }
    if envelope.serve_batch != component.serve_batch {
        return Err(ResidentTopologyError::SessionEnvelopeMismatch);
    }
    Ok(())
}

impl ResidentFootprintFacet {
    pub(super) const fn new(components: &'static [ResidentComponentSpec]) -> Self {
        Self { components }
    }

    pub(crate) fn components(&self) -> &'static [ResidentComponentSpec] {
        self.components
    }

    pub(crate) fn validate(self) -> Result<(), ResidentFootprintValidationError> {
        if self.components.is_empty() {
            return Err(ResidentFootprintValidationError::EmptyComponents);
        }
        for (index, component) in self.components.iter().enumerate() {
            if self.components[..index].iter().any(|previous| {
                previous.component == component.component && previous.variant == component.variant
            }) {
                return Err(ResidentFootprintValidationError::DuplicateComponent);
            }
            if component.component.is_empty() {
                return Err(ResidentFootprintValidationError::EmptyComponent);
            }
            if component.variant.is_empty() {
                return Err(ResidentFootprintValidationError::EmptyVariant);
            }
            if component.representations.is_empty() {
                return Err(ResidentFootprintValidationError::EmptyRepresentations);
            }
            if component.placement_variants.is_empty() {
                return Err(ResidentFootprintValidationError::EmptyPlacementVariants);
            }
            if component.checkout.max_instances() == 0 {
                return Err(ResidentFootprintValidationError::ZeroCheckoutMaximum);
            }
            if component.checkout.is_serialized() && component.checkout.max_instances() != 1 {
                return Err(ResidentFootprintValidationError::SerializedCheckoutMustBeOne);
            }
            if let Some(session) = component.session {
                session.validate()?;
            }
            for (dependency_index, dependency) in component.dependencies.iter().enumerate() {
                if *dependency == component.component {
                    return Err(ResidentFootprintValidationError::SelfDependency);
                }
                if component.dependencies[..dependency_index].contains(dependency) {
                    return Err(ResidentFootprintValidationError::DuplicateDependency);
                }
                if !self
                    .components
                    .iter()
                    .any(|candidate| candidate.component == *dependency)
                {
                    return Err(ResidentFootprintValidationError::MissingDependency);
                }
            }
        }
        Ok(())
    }

    pub(crate) const fn component_count(self) -> usize {
        self.components.len()
    }

    /// The only production construction seam for a resident topology.
    pub(super) fn build_topology<'a>(
        self,
        architecture: ResidentArchitectureId,
        inputs: &ResidentTopologyInputs<'a>,
    ) -> Result<ResidentTopology<'a>, ResidentTopologyError> {
        self.validate()
            .map_err(ResidentTopologyError::InvalidFacet)?;
        inputs.session.validate()?;
        validate_proof_preflight_identity(inputs.verified_pack, inputs.preflight)?;
        let verified_preflight = inputs.verified_pack.preflight();

        let resolved_variant = {
            let exact_backend_preference =
                matches!(inputs.intent, ExecutionIntent::Exact(_)).then(|| {
                    crate::ggml_runtime::RequestBackendPreference::Exact(
                        inputs.candidate.device.route.clone(),
                    )
                });
            let backend = match inputs.candidate.device.ggml_kind {
                crate::ggml_runtime::GgmlBackendKind::Gpu
                | crate::ggml_runtime::GgmlBackendKind::IntegratedGpu => {
                    crate::ggml_runtime::GgmlCpuGraphBackend::Gpu
                }
                _ => crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
            };
            let unified = if architecture.as_str() == "firered-llm-conformer-adapter-qwen2" {
                crate::arch::firered_llm_unified_runtime_enabled(
                    inputs.allow_unified_runtime,
                    backend,
                    exact_backend_preference.as_ref(),
                    Some(inputs.candidate.placement),
                )
            } else {
                matches!(
                    inputs.candidate.placement,
                    crate::device::execution_policy::ExecutionPlacement::CpuOnly
                        | crate::device::execution_policy::ExecutionPlacement::FullDevice
                )
            };
            ResidentResolvedVariant {
                placement: if unified {
                    ResidentPlacementVariant::Unified
                } else {
                    ResidentPlacementVariant::Split
                },
            }
        };
        let active_specs: Vec<&'static ResidentComponentSpec> = self
            .components
            .iter()
            .filter(|spec| {
                spec.placement_variants
                    .contains(&resolved_variant.placement())
            })
            .collect();
        if active_specs.is_empty() {
            return Err(ResidentTopologyError::PlacementVariantUnavailable {
                resolved: resolved_variant.placement(),
            });
        }
        let dependency_order = resolve_dependency_order(&active_specs)?;
        let mut components = Vec::with_capacity(active_specs.len());
        for spec in active_specs {
            if let Some(component_session) = spec.session
                && !inputs.session.deferred_dynamic_session
            {
                validate_session_envelope_for_component(inputs.session, component_session)?;
            }
            components.push(ResidentTopologyComponent {
                verified: VerifiedResidentComponent {
                    architecture: architecture.as_str(),
                    spec,
                    verified_pack: inputs.verified_pack,
                    preflight: verified_preflight,
                    candidate: inputs.candidate,
                    intent: inputs.intent,
                    resolved_variant,
                    session: *inputs.session,
                    component_session: spec.session(),
                },
            });
        }
        Ok(ResidentTopology {
            architecture: architecture.as_str(),
            components,
            dependency_order,
            session: *inputs.session,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentFootprintValidationError {
    EmptyComponents,
    EmptyComponent,
    EmptyVariant,
    DuplicateComponent,
    EmptyRepresentations,
    EmptyPlacementVariants,
    MissingDependency,
    DuplicateDependency,
    SelfDependency,
    ZeroCheckoutMaximum,
    SerializedCheckoutMustBeOne,
    EmptyServeBatchVariants,
    ZeroServeBatchWidth,
    ZeroServeBatchSlots,
    DuplicateServeBatchWidth,
    VariantSlotsExceedActiveSlots,
    ZeroSessionLimit,
    ActiveSlotsExceedSessions,
    ServeBatchExceedsSessions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentTopologyError {
    DependencyCycle,
    InvalidFacet(ResidentFootprintValidationError),
    InvalidSessionEnvelope(ResidentFootprintValidationError),
    GenerationMismatch,
    ProofPreflightMismatch,
    PlacementVariantUnavailable { resolved: ResidentPlacementVariant },
    SessionEnvelopeMismatch,
}

const BOTH_BINDINGS: &[ResidentRepresentation] = &[
    ResidentRepresentation::HostImportedBinding,
    ResidentRepresentation::DeviceCopiedBinding,
];
const UNIFIED_AND_SPLIT: &[ResidentPlacementVariant] = &[
    ResidentPlacementVariant::Unified,
    ResidentPlacementVariant::Split,
];
const NO_DEPENDENCIES: &[&str] = &[];
const MODEL_BINDING_DEPENDENCY: &[&str] = &["model-binding"];
const EXECUTION_STATE_DEPENDENCY: &[&str] = &["execution-state"];
const RETAINED_BATCH_VARIANTS: &[ServeBatchVariant] = &[
    ServeBatchVariant::new(1, 4),
    ServeBatchVariant::new(2, 2),
    ServeBatchVariant::new(4, 1),
];
const SERVE_BATCH: ServeBatchSpec = ServeBatchSpec::new(RETAINED_BATCH_VARIANTS, 4, 4);
const DECODER_SESSION: SessionFootprintSpec = SessionFootprintSpec::new(
    ResidentDecoderCapacity::TokenPositions {
        self_attention: 0,
        cross_attention: 0,
    },
    1,
    4,
    Some(SERVE_BATCH),
);
const STREAM_SESSION: SessionFootprintSpec =
    SessionFootprintSpec::new(ResidentDecoderCapacity::None, 1, 1, None);

const MODEL_BINDING: ResidentComponentSpec = ResidentComponentSpec::new(
    "model-binding",
    "v1",
    ResidentPhase::Prepare,
    ResidentLifetime::ExecutionScope,
    NO_DEPENDENCIES,
    BOTH_BINDINGS,
    UNIFIED_AND_SPLIT,
    ResidentCheckout::serialized(),
    None,
);
const EXECUTION_STATE: ResidentComponentSpec = ResidentComponentSpec::new(
    "execution-state",
    "v1",
    ResidentPhase::Execute,
    ResidentLifetime::ExecutionScope,
    MODEL_BINDING_DEPENDENCY,
    BOTH_BINDINGS,
    UNIFIED_AND_SPLIT,
    ResidentCheckout::bounded(2),
    None,
);
const DECODER_STATE: ResidentComponentSpec = ResidentComponentSpec::new(
    "decoder-state",
    "v1",
    ResidentPhase::Execute,
    ResidentLifetime::Session,
    EXECUTION_STATE_DEPENDENCY,
    BOTH_BINDINGS,
    UNIFIED_AND_SPLIT,
    ResidentCheckout::serialized(),
    Some(DECODER_SESSION),
);
const STREAM_STATE: ResidentComponentSpec = ResidentComponentSpec::new(
    "stream-state",
    "v1",
    ResidentPhase::Stream,
    ResidentLifetime::Session,
    EXECUTION_STATE_DEPENDENCY,
    BOTH_BINDINGS,
    UNIFIED_AND_SPLIT,
    ResidentCheckout::serialized(),
    Some(STREAM_SESSION),
);
const TEXT_STATE: ResidentComponentSpec = ResidentComponentSpec::new(
    "text-state",
    "v1",
    ResidentPhase::Execute,
    ResidentLifetime::Request,
    MODEL_BINDING_DEPENDENCY,
    BOTH_BINDINGS,
    UNIFIED_AND_SPLIT,
    ResidentCheckout::bounded(4),
    None,
);

const SPLIT_ONLY: &[ResidentPlacementVariant] = &[ResidentPlacementVariant::Split];
const UNIFIED_ONLY: &[ResidentPlacementVariant] = &[ResidentPlacementVariant::Unified];

const FIRERED_LLM_SPLIT_ENCODER: ResidentComponentSpec = ResidentComponentSpec::new(
    "split-encoder",
    "v1",
    ResidentPhase::Prepare,
    ResidentLifetime::ExecutionScope,
    NO_DEPENDENCIES,
    BOTH_BINDINGS,
    SPLIT_ONLY,
    ResidentCheckout::serialized(),
    None,
);
const FIRERED_LLM_SPLIT_ADAPTER_DEPENDENCY: &[&str] = &["split-encoder"];
const FIRERED_LLM_SPLIT_ADAPTER: ResidentComponentSpec = ResidentComponentSpec::new(
    "split-adapter",
    "v1",
    ResidentPhase::Prepare,
    ResidentLifetime::ExecutionScope,
    FIRERED_LLM_SPLIT_ADAPTER_DEPENDENCY,
    BOTH_BINDINGS,
    SPLIT_ONLY,
    ResidentCheckout::bounded(2),
    None,
);
const FIRERED_LLM_SPLIT_DECODER_DEPENDENCY: &[&str] = &["split-adapter"];
const FIRERED_LLM_SPLIT_DECODER_CHECKOUT: ResidentComponentSpec = ResidentComponentSpec::new(
    "split-decoder-checkout",
    "v1",
    ResidentPhase::Execute,
    ResidentLifetime::Session,
    FIRERED_LLM_SPLIT_DECODER_DEPENDENCY,
    BOTH_BINDINGS,
    SPLIT_ONLY,
    ResidentCheckout::bounded(2),
    Some(DECODER_SESSION),
);
const FIRERED_LLM_UNIFIED_POOL_DEPENDENCY: &[&str] = &["unified-runtime"];
const FIRERED_LLM_UNIFIED_RUNTIME: ResidentComponentSpec = ResidentComponentSpec::new(
    "unified-runtime",
    "v1",
    ResidentPhase::Prepare,
    ResidentLifetime::ExecutionScope,
    NO_DEPENDENCIES,
    BOTH_BINDINGS,
    UNIFIED_ONLY,
    ResidentCheckout::serialized(),
    None,
);
const FIRERED_LLM_UNIFIED_POOL: ResidentComponentSpec = ResidentComponentSpec::new(
    "unified-pool",
    "v1",
    ResidentPhase::Execute,
    ResidentLifetime::ExecutionScope,
    FIRERED_LLM_UNIFIED_POOL_DEPENDENCY,
    BOTH_BINDINGS,
    UNIFIED_ONLY,
    ResidentCheckout::bounded(1),
    None,
);

const FIRERED_LLM_RESIDENT_COMPONENTS: &[ResidentComponentSpec] = &[
    FIRERED_LLM_SPLIT_ENCODER,
    FIRERED_LLM_SPLIT_ADAPTER,
    FIRERED_LLM_SPLIT_DECODER_CHECKOUT,
    FIRERED_LLM_UNIFIED_RUNTIME,
    FIRERED_LLM_UNIFIED_POOL,
];

pub(crate) const COHERE_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE, DECODER_STATE]);
pub(crate) const WHISPER_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE, DECODER_STATE]);
pub(crate) const QWEN_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE, DECODER_STATE]);
pub(crate) const PARAKEET_CTC_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE, TEXT_STATE]);
pub(crate) const PARAKEET_TDT_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE, TEXT_STATE]);
pub(crate) const WAV2VEC2_CTC_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE, TEXT_STATE]);
pub(crate) const XASR_ZIPFORMER_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE, STREAM_STATE]);
pub(crate) const MOONSHINE_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE, DECODER_STATE]);
pub(crate) const DOLPHIN_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE, DECODER_STATE]);
pub(crate) const SENSEVOICE_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE, TEXT_STATE]);
pub(crate) const FIRERED_AED_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE, DECODER_STATE]);
pub(crate) const FIRERED_LLM_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(FIRERED_LLM_RESIDENT_COMPONENTS);
pub(crate) const FUNASR_NANO_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE, DECODER_STATE]);
pub(crate) const MIMO_ASR_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE, DECODER_STATE]);
pub(crate) const MOSS_TD_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE, DECODER_STATE]);
pub(crate) const GRANITE_SPEECH_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE, DECODER_STATE]);

const STREAM_VAD_EMBEDDED: ResidentComponentSpec = ResidentComponentSpec::new(
    "embedded-model",
    "v1",
    ResidentPhase::Load,
    ResidentLifetime::ExecutionScope,
    NO_DEPENDENCIES,
    BOTH_BINDINGS,
    UNIFIED_AND_SPLIT,
    ResidentCheckout::serialized(),
    None,
);
const STREAM_VAD_EMBEDDED_DEPENDENCY: &[&str] = &["embedded-model"];
const STREAM_VAD_HOST_SESSION: ResidentComponentSpec = ResidentComponentSpec::new(
    "host-session",
    "v1",
    ResidentPhase::Stream,
    ResidentLifetime::Session,
    STREAM_VAD_EMBEDDED_DEPENDENCY,
    BOTH_BINDINGS,
    UNIFIED_AND_SPLIT,
    ResidentCheckout::serialized(),
    Some(STREAM_SESSION),
);
const STREAM_VAD_ACCELERATED_ACTOR: ResidentComponentSpec = ResidentComponentSpec::new(
    "accelerated-actor",
    "v1",
    ResidentPhase::Stream,
    ResidentLifetime::ExecutionScope,
    STREAM_VAD_EMBEDDED_DEPENDENCY,
    BOTH_BINDINGS,
    UNIFIED_AND_SPLIT,
    ResidentCheckout::bounded(2),
    None,
);
pub(crate) const FIRERED_STREAM_VAD_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[
        STREAM_VAD_EMBEDDED,
        STREAM_VAD_HOST_SESSION,
        STREAM_VAD_ACCELERATED_ACTOR,
    ]);
pub(crate) const REDIMNET_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE]);
pub(crate) const PYANNOTE_SEGMENT_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE]);
pub(crate) const DIARIZEN_SEGMENT_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE]);
pub(crate) const FIRERED_PUNC_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE]);

const FORCED_ALIGNER_PREPARED_ASSETS: ResidentComponentSpec = ResidentComponentSpec::new(
    "prepared-assets",
    "v1",
    ResidentPhase::Prepare,
    ResidentLifetime::ExecutionScope,
    NO_DEPENDENCIES,
    BOTH_BINDINGS,
    UNIFIED_AND_SPLIT,
    ResidentCheckout::serialized(),
    None,
);
const FORCED_ALIGNER_ASSETS_DEPENDENCY: &[&str] = &["prepared-assets"];
const FORCED_ALIGNER_AUDIO_RUNTIME: ResidentComponentSpec = ResidentComponentSpec::new(
    "audio-runtime",
    "v1",
    ResidentPhase::Execute,
    ResidentLifetime::Request,
    FORCED_ALIGNER_ASSETS_DEPENDENCY,
    BOTH_BINDINGS,
    UNIFIED_AND_SPLIT,
    ResidentCheckout::bounded(4),
    None,
);
const FORCED_ALIGNER_DECODER_RUNTIME: ResidentComponentSpec = ResidentComponentSpec::new(
    "decoder-runtime",
    "v1",
    ResidentPhase::Execute,
    ResidentLifetime::Request,
    FORCED_ALIGNER_ASSETS_DEPENDENCY,
    BOTH_BINDINGS,
    UNIFIED_AND_SPLIT,
    ResidentCheckout::bounded(4),
    None,
);
const FORCED_ALIGNER_LOGITS_RUNTIME: ResidentComponentSpec = ResidentComponentSpec::new(
    "logits-runtime",
    "v1",
    ResidentPhase::Execute,
    ResidentLifetime::Request,
    FORCED_ALIGNER_ASSETS_DEPENDENCY,
    BOTH_BINDINGS,
    UNIFIED_AND_SPLIT,
    ResidentCheckout::bounded(4),
    None,
);
pub(crate) const QWEN_FORCED_ALIGNER_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[
        FORCED_ALIGNER_PREPARED_ASSETS,
        FORCED_ALIGNER_AUDIO_RUNTIME,
        FORCED_ALIGNER_DECODER_RUNTIME,
        FORCED_ALIGNER_LOGITS_RUNTIME,
    ]);

pub(crate) const AUXILIARY_RESIDENT_FOOTPRINTS: &[(&str, ResidentFootprintFacet)] = &[
    ("firered-stream-vad", FIRERED_STREAM_VAD_RESIDENT_FOOTPRINT),
    ("redimnet2", REDIMNET_RESIDENT_FOOTPRINT),
    ("pyannote-segmentation", PYANNOTE_SEGMENT_RESIDENT_FOOTPRINT),
    ("diarizen-segmentation", DIARIZEN_SEGMENT_RESIDENT_FOOTPRINT),
    ("firered-punc", FIRERED_PUNC_RESIDENT_FOOTPRINT),
    (
        "qwen3-forced-aligner",
        QWEN_FORCED_ALIGNER_RESIDENT_FOOTPRINT,
    ),
];

#[cfg(test)]
pub(crate) const TEST_RESIDENT_FOOTPRINT: ResidentFootprintFacet =
    ResidentFootprintFacet::new(&[MODEL_BINDING, EXECUTION_STATE]);

/// Neutral decoder/session capacity identity. It is not a native-memory quote.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ResidentDecoderCapacity {
    None,
    TokenPositions {
        self_attention: usize,
        cross_attention: usize,
    },
    FrameCount {
        frames: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proof_preflight_rejects_a_different_open_generation() {
        use std::collections::BTreeMap;

        let temp = tempfile::tempdir().expect("temp dir");
        let first_path = temp.path().join("first.gguf");
        let second_path = temp.path().join("second.gguf");
        crate::ggml_runtime::write_gguf_file_v0(&first_path, &BTreeMap::new(), &[])
            .expect("first GGUF");
        crate::ggml_runtime::write_gguf_file_v0(&second_path, &BTreeMap::new(), &[])
            .expect("second GGUF");
        let first = crate::ggml_runtime::load_runtime_source_metadata_and_tensor_index(&first_path)
            .expect("first preflight");
        let second =
            crate::ggml_runtime::load_runtime_source_metadata_and_tensor_index(&second_path)
                .expect("second preflight");
        let verified =
            VerifiedPack::from_unverified_preflight_for_test(first, "whisper-encoder-decoder");

        assert_eq!(
            validate_proof_preflight_identity(&verified, &second),
            Err(ResidentTopologyError::GenerationMismatch)
        );
    }

    #[test]
    fn materialized_outcomes_distinguish_metal_host_import_and_cpu_copy_lane() {
        use crate::device::execution_policy::{
            ExecutionCandidate, ExecutionDeviceSnapshot, ExecutionIntent, ExecutionPlacement,
        };
        use crate::ggml_runtime::GgmlCpuGraphBackend;
        use crate::ggml_runtime::{
            GgmlBackendKind, ResidentDeviceCopyCapability, ResidentHostImportCapability,
        };
        use crate::models::native_execution_services::{
            NativeExecutionServices, current_execution_lane_key,
        };
        use crate::models::resident_owner::{
            AdapterLoraFingerprint, MaterializedResidentComponent, ResidentBinding,
        };
        use crate::{DeviceAddressability, ExecutionProvider, ResolvedExecutionRoute};
        use std::collections::BTreeMap;

        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("metal-outcome.gguf");
        crate::ggml_runtime::write_gguf_file_v0(&path, &BTreeMap::new(), &[])
            .expect("minimal GGUF");
        let preflight = crate::ggml_runtime::load_runtime_source_metadata_and_tensor_index(&path)
            .expect("preflight");
        let verified =
            VerifiedPack::from_unverified_preflight_for_test(preflight, "whisper-encoder-decoder");
        let candidate = ExecutionCandidate {
            device: ExecutionDeviceSnapshot {
                route: ResolvedExecutionRoute {
                    provider: ExecutionProvider::Metal,
                    stable_id: "Metal".to_string(),
                    registry_ordinal: 0,
                    kind: crate::RouteDeviceKind::Accelerated,
                    addressability: DeviceAddressability::NotExactlyAddressable {
                        reason: "test candidate",
                    },
                },
                ggml_kind: GgmlBackendKind::Gpu,
                memory: None,
                buffer_alignment: None,
            },
            placement: ExecutionPlacement::Hybrid,
        };
        let intent = ExecutionIntent::AcceleratedOnly;
        let session = ResidentSessionEnvelope::test_cohere();
        let inputs = ResidentTopologyInputs::new(&verified, &candidate, &intent, &session, true);
        let topology = COHERE_RESIDENT_FOOTPRINT
            .build_topology(
                ResidentArchitectureId::from_descriptor("whisper-encoder-decoder"),
                &inputs,
            )
            .expect("descriptor topology");
        assert_eq!(
            topology.components()[0]
                .verified()
                .resolved_variant()
                .placement(),
            ResidentPlacementVariant::Split
        );
        assert_eq!(
            topology.dependency_order(),
            &["model-binding", "execution-state", "decoder-state"]
        );
        assert!(
            topology.components()[2]
                .verified()
                .component_session()
                .is_some()
        );
        let firered_split = FIRERED_LLM_RESIDENT_FOOTPRINT
            .build_topology(
                ResidentArchitectureId::from_descriptor("firered-llm-conformer-adapter-qwen2"),
                &inputs,
            )
            .expect("FireRed LLM split topology");
        let split_components: Vec<_> = firered_split
            .components()
            .iter()
            .map(|component| component.verified().spec().component())
            .collect();
        assert_eq!(
            split_components,
            vec!["split-encoder", "split-adapter", "split-decoder-checkout"]
        );
        assert_eq!(
            firered_split.dependency_order(),
            &["split-encoder", "split-adapter", "split-decoder-checkout"]
        );
        assert_eq!(
            firered_split
                .components()
                .iter()
                .map(|component| {
                    let spec = component.verified().spec();
                    (
                        spec.component(),
                        spec.lifetime(),
                        spec.checkout().max_instances(),
                    )
                })
                .collect::<Vec<_>>(),
            vec![
                ("split-encoder", ResidentLifetime::ExecutionScope, 1),
                ("split-adapter", ResidentLifetime::ExecutionScope, 2),
                ("split-decoder-checkout", ResidentLifetime::Session, 2),
            ]
        );
        let cpu_candidate = ExecutionCandidate {
            placement: ExecutionPlacement::CpuOnly,
            ..candidate.clone()
        };
        let cpu_inputs =
            ResidentTopologyInputs::new(&verified, &cpu_candidate, &intent, &session, true);
        let firered_cpu = FIRERED_LLM_RESIDENT_FOOTPRINT
            .build_topology(
                ResidentArchitectureId::from_descriptor("firered-llm-conformer-adapter-qwen2"),
                &cpu_inputs,
            )
            .expect("FireRed LLM CPU split topology");
        let cpu_components: Vec<_> = firered_cpu
            .components()
            .iter()
            .map(|component| component.verified().spec().component())
            .collect();
        assert_eq!(
            cpu_components,
            vec!["split-encoder", "split-adapter", "split-decoder-checkout"]
        );

        let gpu_physical_key =
            crate::device::execution_route::PhysicalResourceKey::new("0000:00:02.0")
                .expect("GPU physical key");
        let gpu_candidate = ExecutionCandidate {
            device: ExecutionDeviceSnapshot {
                route: ResolvedExecutionRoute {
                    provider: ExecutionProvider::Cuda,
                    stable_id: "CUDA:0".to_string(),
                    registry_ordinal: 0,
                    kind: crate::RouteDeviceKind::Accelerated,
                    addressability: DeviceAddressability::ExactlyAddressable {
                        physical_key: gpu_physical_key.clone(),
                    },
                },
                ggml_kind: GgmlBackendKind::Gpu,
                memory: None,
                buffer_alignment: None,
            },
            placement: ExecutionPlacement::FullDevice,
        };
        let gpu_intent = ExecutionIntent::Exact(
            crate::device::execution_route::ExactDeviceSelector::PhysicalKey(gpu_physical_key),
        );
        let gpu_inputs =
            ResidentTopologyInputs::new(&verified, &gpu_candidate, &gpu_intent, &session, true);
        let firered_gpu = FIRERED_LLM_RESIDENT_FOOTPRINT
            .build_topology(
                ResidentArchitectureId::from_descriptor("firered-llm-conformer-adapter-qwen2"),
                &gpu_inputs,
            )
            .expect("FireRed LLM exact GPU unified topology");
        let gpu_components: Vec<_> = firered_gpu
            .components()
            .iter()
            .map(|component| component.verified().spec().component())
            .collect();
        assert_eq!(gpu_components, vec!["unified-runtime", "unified-pool"]);
        assert_eq!(
            firered_gpu.dependency_order(),
            &["unified-runtime", "unified-pool"]
        );
        assert_eq!(
            firered_gpu
                .components()
                .iter()
                .map(|component| {
                    let spec = component.verified().spec();
                    (
                        spec.component(),
                        spec.lifetime(),
                        spec.checkout().max_instances(),
                    )
                })
                .collect::<Vec<_>>(),
            vec![
                ("unified-runtime", ResidentLifetime::ExecutionScope, 1),
                ("unified-pool", ResidentLifetime::ExecutionScope, 1),
            ]
        );
        let component = topology.components()[0].verified();
        let identity = component
            .preflight()
            .runtime_source()
            .strong_file_identity();
        assert!(matches!(
            MaterializedResidentComponent::from_host_import(
                component,
                ResidentHostImportCapability::for_test(
                    crate::ggml_runtime::StrongFileIdentity::test_fixture(99),
                ),
            ),
            Err(crate::models::resident_owner::ResidentKeyError::GenerationMismatch)
        ));
        let host = MaterializedResidentComponent::from_host_import(
            component,
            ResidentHostImportCapability::for_test(identity),
        )
        .expect("host import outcome");
        let host_binding = ResidentBinding::from_materialized(
            host,
            AdapterLoraFingerprint::test(),
            NativeExecutionServices::for_local_process()
                .expect("execution services")
                .scope_id(),
        )
        .expect("host binding");
        assert!(matches!(
            host_binding,
            ResidentBinding::HostImportedBinding { .. }
        ));

        let lane = current_execution_lane_key(GgmlCpuGraphBackend::Cpu);
        let copied = MaterializedResidentComponent::from_device_copy(
            component,
            ResidentDeviceCopyCapability::for_test(lane.clone()),
        )
        .expect("device-copy outcome");
        let copied_binding = ResidentBinding::from_materialized(
            copied,
            AdapterLoraFingerprint::test(),
            NativeExecutionServices::for_local_process()
                .expect("execution services")
                .scope_id(),
        )
        .expect("copied binding");
        match copied_binding {
            ResidentBinding::DeviceCopiedBinding { key } => assert_eq!(key.lane(), Some(&lane)),
            ResidentBinding::HostImportedBinding { .. } => panic!("copy became host import"),
        }
    }

    #[test]
    fn batch_session_envelope_mismatch_fails_closed() {
        let bad = ResidentSessionEnvelope::test_default();
        assert_eq!(
            validate_session_envelope_for_component(&bad, DECODER_SESSION),
            Err(ResidentTopologyError::SessionEnvelopeMismatch)
        );
        let bad_capacity = ResidentSessionEnvelope::new(
            ResidentDecoderCapacity::FrameCount { frames: 1 },
            1,
            4,
            Some(SERVE_BATCH),
        );
        assert_eq!(
            validate_session_envelope_for_component(&bad_capacity, DECODER_SESSION),
            Err(ResidentTopologyError::SessionEnvelopeMismatch)
        );
        let bad_active_slots = ResidentSessionEnvelope::new(
            ResidentDecoderCapacity::TokenPositions {
                self_attention: 0,
                cross_attention: 0,
            },
            2,
            4,
            Some(SERVE_BATCH),
        );
        assert_eq!(
            validate_session_envelope_for_component(&bad_active_slots, DECODER_SESSION),
            Err(ResidentTopologyError::SessionEnvelopeMismatch)
        );
    }

    #[test]
    fn facet_declares_alternatives_dependencies_and_serve_batch_shape() {
        assert!(COHERE_RESIDENT_FOOTPRINT.validate().is_ok());
        let decoder = COHERE_RESIDENT_FOOTPRINT
            .components
            .iter()
            .find(|component| component.component == "decoder-state")
            .expect("decoder component");
        assert_eq!(decoder.dependencies, EXECUTION_STATE_DEPENDENCY);
        assert!(
            decoder
                .representations
                .contains(&ResidentRepresentation::HostImportedBinding)
        );
        assert!(
            decoder
                .representations
                .contains(&ResidentRepresentation::DeviceCopiedBinding)
        );
        assert_eq!(decoder.placement_variants, UNIFIED_AND_SPLIT);
        let serve_batch = decoder.session.expect("serve-batch session").serve_batch;
        assert_eq!(
            serve_batch.expect("serve-batch").retained_variants,
            RETAINED_BATCH_VARIANTS
        );
    }

    #[test]
    fn session_and_checkout_validation_fails_closed() {
        const INVALID_VARIANTS: &[ServeBatchVariant] = &[ServeBatchVariant::new(1, 1)];
        assert_eq!(
            ServeBatchSpec::new(INVALID_VARIANTS, 2, 1).validate(),
            Err(ResidentFootprintValidationError::ActiveSlotsExceedSessions)
        );
        assert_eq!(
            ResidentSessionEnvelope::new(ResidentDecoderCapacity::None, 0, 1, None).validate(),
            Err(ResidentTopologyError::InvalidSessionEnvelope(
                ResidentFootprintValidationError::ZeroSessionLimit
            ))
        );
        const INVALID_SERIALIZED: ResidentComponentSpec = ResidentComponentSpec::new(
            "invalid",
            "v1",
            ResidentPhase::Prepare,
            ResidentLifetime::ExecutionScope,
            NO_DEPENDENCIES,
            BOTH_BINDINGS,
            UNIFIED_AND_SPLIT,
            ResidentCheckout::Serialized { max_instances: 2 },
            None,
        );
        assert_eq!(
            ResidentFootprintFacet::new(&[INVALID_SERIALIZED]).validate(),
            Err(ResidentFootprintValidationError::SerializedCheckoutMustBeOne)
        );
    }
}
