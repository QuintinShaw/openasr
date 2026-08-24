//! Operator-only, bounded runtime ownership diagnostics.
//!
//! This route is a read-only projection of the process-owned receipt collector.
//! It never participates in admission, policy selection, or fallback, and it
//! deliberately serializes only keyed/redacted identifiers.

use crate::*;
use axum::{Json, extract::Query};
use openasr_core::models::native_execution_services::ExecutionCacheAttemptId;
use openasr_core::runtime_receipts::{
    LeaseReceiptShadow, LiveRuntimeOwner, ReceiptCompletenessReason, RuntimeOwnerId,
    RuntimeOwnerPlacement, RuntimeReceiptAvailability, RuntimeReceiptEvent, RuntimeReceiptMetric,
    RuntimeReceiptSnapshot, RuntimeResourceId, RuntimeResourceState, SafeMemoryDomainKind,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

const DEFAULT_EVENT_LIMIT: usize = 64;
const MAX_EVENT_LIMIT: usize = 128;

#[derive(Debug, Default, Deserialize)]
pub(crate) struct RuntimeReceiptQuery {
    /// Optional safe domain kind filter. Accepted values are `system-memory`
    /// and `dedicated-device`; physical device names never cross this boundary.
    pub(crate) domain: Option<String>,
    /// Maximum number of recent events returned. The route clamps this to a
    /// small fixed bound even when the collector has a larger in-memory ring.
    pub(crate) event_limit: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DomainFilter {
    SystemMemory,
    DedicatedDevice,
}

impl DomainFilter {
    fn parse(value: &str) -> Result<Self, RuntimeReceiptQueryError> {
        match value.trim() {
            "system-memory" => Ok(Self::SystemMemory),
            "dedicated-device" => Ok(Self::DedicatedDevice),
            _ => Err(RuntimeReceiptQueryError::InvalidDomain),
        }
    }

    fn kind(self) -> SafeMemoryDomainKind {
        match self {
            Self::SystemMemory => SafeMemoryDomainKind::SystemMemory,
            Self::DedicatedDevice => SafeMemoryDomainKind::DedicatedDevice,
        }
    }

    fn matches(self, kind: SafeMemoryDomainKind) -> bool {
        matches!(
            (self, kind),
            (Self::SystemMemory, SafeMemoryDomainKind::SystemMemory)
                | (Self::DedicatedDevice, SafeMemoryDomainKind::DedicatedDevice)
        )
    }
}

#[derive(Debug)]
pub(crate) enum RuntimeReceiptQueryError {
    InvalidDomain,
}

impl IntoResponse for RuntimeReceiptQueryError {
    fn into_response(self) -> Response {
        let message = match self {
            Self::InvalidDomain => "domain must be one of: system-memory, dedicated-device.",
        };
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": message,
                    "type": "invalid_request_error"
                }
            })),
        )
            .into_response()
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct RuntimeReceiptResponse {
    schema: &'static str,
    daemon_start_identity: SafeDaemonStartIdentityView,
    availability: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    unavailable_reason: Option<&'static str>,
    snapshot_completeness: ReceiptCompletenessView,
    lease_reconciliation: LeaseReceiptShadow,
    live_owners: Vec<RuntimeOwnerView>,
    recent_events: Vec<RuntimeEventView>,
    event_limit: usize,
}

#[derive(Debug, Serialize)]
struct SafeDaemonStartIdentityView {
    pid: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    nonce: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    started_at_unix_secs: Option<u64>,
}

#[derive(Debug, Serialize)]
struct ReceiptCompletenessView {
    complete: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    live_state_complete: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    live_state_reason: Option<String>,
    event_history_complete: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    event_history_reason: Option<String>,
    dropped_events: u64,
    dropped_owners: u64,
    rejected_resources: u64,
    dropped_notifications: u64,
}

#[derive(Debug, Serialize)]
struct RuntimeOwnerView {
    id: String,
    component: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    placement: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    lane: Option<RuntimeLaneView>,
    resources: Vec<RuntimeResourceView>,
}

#[derive(Debug, Serialize)]
struct RuntimeLaneView {
    provider: String,
    placement: String,
    backend: String,
    device: String,
}

#[derive(Debug, Serialize)]
struct RuntimeResourceView {
    id: String,
    kind: String,
    placement: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    lane: Option<RuntimeLaneView>,
    #[serde(skip_serializing_if = "Option::is_none")]
    domain: Option<RuntimeDomainView>,
    ledger_binding: &'static str,
    requested: RuntimeMetricView,
    peak: RuntimeMetricView,
    retained: RuntimeMetricView,
    quote_confidence: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    observation_confidence: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", content = "bytes", rename_all = "kebab-case")]
enum RuntimeMetricView {
    Known(u64),
    Unknown,
    Unavailable,
}

#[derive(Debug, Serialize)]
struct RuntimeDomainView {
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    heap: Option<u32>,
    join_id: String,
}

#[derive(Debug, Serialize)]
struct RuntimeEventView {
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    attempt_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_attempt_id: Option<String>,
    owner_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    resource_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    owner: Option<RuntimeOwnerDescriptorView>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resource: Option<RuntimeResourceView>,
}

#[derive(Debug, Serialize)]
struct RuntimeOwnerDescriptorView {
    component: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    placement: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    lane: Option<RuntimeLaneView>,
}

pub(crate) async fn runtime_receipts(
    State(runtime): State<ServerRuntime>,
    Extension(identity): Extension<ServerStartIdentity>,
    Query(query): Query<RuntimeReceiptQuery>,
) -> Result<Json<RuntimeReceiptResponse>, RuntimeReceiptQueryError> {
    let domain = query
        .domain
        .as_deref()
        .map(DomainFilter::parse)
        .transpose()?;
    let event_limit = query
        .event_limit
        .unwrap_or(DEFAULT_EVENT_LIMIT)
        .min(MAX_EVENT_LIMIT);
    let snapshot = runtime.runtime_receipt_snapshot();
    let reconciliation = runtime.runtime_receipt_reconciliation();
    Ok(Json(RuntimeReceiptResponse::from_snapshot(
        snapshot,
        reconciliation,
        identity,
        domain,
        event_limit,
    )))
}

#[derive(Default)]
struct DomainAttribution {
    owner_domains: HashMap<RuntimeOwnerId, HashSet<SafeMemoryDomainKind>>,
    resource_domains: HashMap<RuntimeResourceId, (RuntimeOwnerId, SafeMemoryDomainKind)>,
}

impl DomainAttribution {
    fn from_events(events: &[RuntimeReceiptEvent]) -> Self {
        let mut attribution = Self::default();
        for event in events {
            if let RuntimeReceiptEvent::ResourceAcquired {
                owner_id,
                resource_id,
                descriptor,
                ..
            } = event
                && let Some(domain) = descriptor.domain
            {
                attribution
                    .owner_domains
                    .entry(*owner_id)
                    .or_default()
                    .insert(domain.kind);
                attribution
                    .resource_domains
                    .insert(*resource_id, (*owner_id, domain.kind));
            }
        }
        attribution
    }
}

impl RuntimeReceiptResponse {
    fn from_snapshot(
        snapshot: RuntimeReceiptSnapshot,
        lease_reconciliation: LeaseReceiptShadow,
        identity: ServerStartIdentity,
        domain: Option<DomainFilter>,
        event_limit: usize,
    ) -> Self {
        let attribution = DomainAttribution::from_events(&snapshot.events);
        let (live_owners, live_owners_complete) =
            filter_live_owners(&snapshot.live_owners, domain, &attribution);
        let (filtered_events, events_complete) =
            filter_events(&snapshot.events, domain, &attribution);
        let recent_events = filtered_events
            .into_iter()
            .rev()
            .take(event_limit)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(RuntimeEventView::from_event)
            .collect();
        let (availability, unavailable_reason) = match snapshot.availability {
            RuntimeReceiptAvailability::Available => ("available", None),
            RuntimeReceiptAvailability::Unavailable { reason } => (
                "unavailable",
                Some(format_receipt_unavailable_reason(reason)),
            ),
        };
        let mut snapshot_completeness = ReceiptCompletenessView::from(snapshot.completeness);
        if domain.is_some() && (!live_owners_complete || !events_complete) {
            snapshot_completeness.complete = false;
            if snapshot_completeness.reason.is_none() {
                snapshot_completeness.reason =
                    Some("domain-filter-attribution-unavailable".to_string());
            }
        }
        Self {
            schema: snapshot.schema,
            daemon_start_identity: SafeDaemonStartIdentityView::from(identity),
            availability,
            unavailable_reason,
            snapshot_completeness,
            lease_reconciliation,
            live_owners: live_owners
                .into_iter()
                .map(|owner| RuntimeOwnerView::from_owner(owner, domain))
                .collect(),
            recent_events,
            event_limit,
        }
    }
}

fn filter_live_owners<'a>(
    owners: &'a [LiveRuntimeOwner],
    domain: Option<DomainFilter>,
    attribution: &DomainAttribution,
) -> (Vec<&'a LiveRuntimeOwner>, bool) {
    let Some(domain) = domain else {
        return (owners.iter().collect(), true);
    };
    let mut complete = true;
    let filtered = owners
        .iter()
        .filter(|owner| {
            if owner.resources.is_empty() {
                let Some(domains) = attribution.owner_domains.get(&owner.id) else {
                    complete = false;
                    return false;
                };
                return domains.contains(&domain.kind());
            }
            owner.resources.values().any(|resource| {
                let Some(resource_domain) = resource.descriptor.domain else {
                    complete = false;
                    return false;
                };
                domain.matches(resource_domain.kind)
            })
        })
        .collect();
    (filtered, complete)
}

fn filter_events<'a>(
    events: &'a [RuntimeReceiptEvent],
    domain: Option<DomainFilter>,
    attribution: &DomainAttribution,
) -> (Vec<&'a RuntimeReceiptEvent>, bool) {
    let Some(domain) = domain else {
        return (events.iter().collect(), true);
    };
    let mut complete = true;
    let filtered = events
        .iter()
        .filter(|event| match event {
            RuntimeReceiptEvent::ResourceAcquired { descriptor, .. }
            | RuntimeReceiptEvent::ResourceStateChanged { descriptor, .. } => {
                let Some(resource_domain) = descriptor.domain else {
                    complete = false;
                    return false;
                };
                domain.matches(resource_domain.kind)
            }
            RuntimeReceiptEvent::ResourceReleased {
                owner_id,
                resource_id,
                ..
            } => {
                let Some((acquired_owner_id, resource_domain)) =
                    attribution.resource_domains.get(resource_id)
                else {
                    complete = false;
                    return false;
                };
                if acquired_owner_id != owner_id {
                    complete = false;
                    return false;
                }
                domain.matches(*resource_domain)
            }
            RuntimeReceiptEvent::OwnerCreated { owner_id, .. }
            | RuntimeReceiptEvent::OwnerReused { owner_id, .. }
            | RuntimeReceiptEvent::OwnerReleased { owner_id, .. } => {
                let Some(domains) = attribution.owner_domains.get(owner_id) else {
                    complete = false;
                    return false;
                };
                domains.contains(&domain.kind())
            }
        })
        .collect();
    (filtered, complete)
}

impl From<ServerStartIdentity> for SafeDaemonStartIdentityView {
    fn from(identity: ServerStartIdentity) -> Self {
        Self {
            pid: identity.pid,
            nonce: identity.nonce,
            started_at_unix_secs: identity.started_at_unix_secs,
        }
    }
}

impl RuntimeOwnerView {
    fn from_owner(owner: &LiveRuntimeOwner, domain: Option<DomainFilter>) -> Self {
        let (placement, lane) = runtime_owner_placement(owner.descriptor.placement);
        Self {
            id: owner_view_id(owner.id.ordinal),
            component: owner.descriptor.component.to_hex(),
            content: owner.descriptor.content.map(|value| value.to_hex()),
            source: owner.descriptor.source.map(|value| value.to_hex()),
            placement,
            lane,
            resources: owner
                .resources
                .values()
                .filter(|resource| match (domain, resource.descriptor.domain) {
                    (None, _) => true,
                    (Some(domain), Some(resource_domain)) => domain.matches(resource_domain.kind),
                    (Some(_), None) => false,
                })
                .map(RuntimeResourceView::from_resource)
                .collect(),
        }
    }
}

impl RuntimeOwnerDescriptorView {
    fn from_descriptor(
        descriptor: &openasr_core::runtime_receipts::RuntimeOwnerDescriptor,
    ) -> Self {
        let (placement, lane) = runtime_owner_placement(descriptor.placement);
        Self {
            component: descriptor.component.to_hex(),
            content: descriptor.content.map(|value| value.to_hex()),
            source: descriptor.source.map(|value| value.to_hex()),
            placement,
            lane,
        }
    }
}

fn runtime_owner_placement(
    placement: RuntimeOwnerPlacement,
) -> (&'static str, Option<RuntimeLaneView>) {
    match placement {
        RuntimeOwnerPlacement::HostNeutral => ("host-neutral", None),
        RuntimeOwnerPlacement::LaneBound(lane) => ("lane-bound", Some(RuntimeLaneView::from(lane))),
        RuntimeOwnerPlacement::Unknown => ("unknown", None),
    }
}

impl RuntimeResourceView {
    fn from_resource(resource: &openasr_core::runtime_receipts::LiveRuntimeResource) -> Self {
        Self::from_descriptor(resource_view_id(resource.id.ordinal), &resource.descriptor)
    }

    fn from_descriptor(
        id: String,
        descriptor: &openasr_core::runtime_receipts::RuntimeResourceDescriptor,
    ) -> Self {
        let (placement, lane) = runtime_owner_placement(descriptor.placement);
        Self {
            id,
            kind: descriptor.kind.to_hex(),
            placement,
            lane,
            domain: descriptor.domain.map(RuntimeDomainView::from),
            ledger_binding: match descriptor.ledger_binding {
                openasr_core::runtime_receipts::RuntimeResourceLedgerBinding::Brokered(_) => {
                    "brokered"
                }
                openasr_core::runtime_receipts::RuntimeResourceLedgerBinding::NoBrokerLease => {
                    "no-broker-lease"
                }
                openasr_core::runtime_receipts::RuntimeResourceLedgerBinding::Unknown => "unknown",
            },
            requested: RuntimeMetricView::from(descriptor.requested),
            peak: RuntimeMetricView::from(descriptor.peak),
            retained: RuntimeMetricView::from(descriptor.retained),
            quote_confidence: descriptor.quote_confidence.as_str().to_string(),
            observation_confidence: descriptor
                .observation_confidence
                .map(|value| value.as_str().to_string()),
        }
    }
}

impl From<RuntimeReceiptMetric> for RuntimeMetricView {
    fn from(metric: RuntimeReceiptMetric) -> Self {
        match metric {
            RuntimeReceiptMetric::Known(bytes) => Self::Known(bytes),
            RuntimeReceiptMetric::Unknown => Self::Unknown,
            RuntimeReceiptMetric::Unavailable => Self::Unavailable,
        }
    }
}

impl RuntimeDomainView {
    fn from(domain: openasr_core::runtime_receipts::SafeMemoryDomainProjection) -> Self {
        Self {
            kind: runtime_domain_kind(domain.kind).to_string(),
            heap: domain.heap,
            join_id: domain.join_id.to_hex(),
        }
    }
}

fn runtime_domain_kind(kind: SafeMemoryDomainKind) -> &'static str {
    match kind {
        SafeMemoryDomainKind::SystemMemory => "system-memory",
        SafeMemoryDomainKind::DedicatedDevice => "dedicated-device",
    }
}

impl From<openasr_core::runtime_receipts::SafeExecutionLaneProjection> for RuntimeLaneView {
    fn from(lane: openasr_core::runtime_receipts::SafeExecutionLaneProjection) -> Self {
        Self {
            provider: lane.provider.as_str().to_string(),
            placement: lane.placement.as_str().to_string(),
            backend: lane.backend.as_str().to_string(),
            device: lane.device.to_hex(),
        }
    }
}

impl RuntimeEventView {
    fn from_event(event: &RuntimeReceiptEvent) -> Self {
        match event {
            RuntimeReceiptEvent::OwnerCreated {
                owner_id,
                descriptor,
                attempt_id,
                request_attempt_id,
            } => Self {
                kind: "owner-created",
                attempt_id: (*attempt_id).map(attempt_view_id),
                request_attempt_id: request_attempt_id.map(|attempt| attempt.to_string()),
                owner_id: owner_view_id(owner_id.ordinal),
                resource_id: None,
                owner: Some(RuntimeOwnerDescriptorView::from_descriptor(descriptor)),
                resource: None,
            },
            RuntimeReceiptEvent::OwnerReused {
                owner_id,
                attempt_id,
                request_attempt_id,
            } => Self {
                kind: "owner-reused",
                attempt_id: (*attempt_id).map(attempt_view_id),
                request_attempt_id: request_attempt_id.map(|attempt| attempt.to_string()),
                owner_id: owner_view_id(owner_id.ordinal),
                resource_id: None,
                owner: None,
                resource: None,
            },
            RuntimeReceiptEvent::OwnerReleased {
                owner_id,
                attempt_id,
                request_attempt_id,
            } => Self {
                kind: "owner-released",
                attempt_id: (*attempt_id).map(attempt_view_id),
                request_attempt_id: request_attempt_id.map(|attempt| attempt.to_string()),
                owner_id: owner_view_id(owner_id.ordinal),
                resource_id: None,
                owner: None,
                resource: None,
            },
            RuntimeReceiptEvent::ResourceAcquired {
                owner_id,
                resource_id,
                descriptor,
                attempt_id,
                request_attempt_id,
            } => Self {
                kind: "resource-acquired",
                attempt_id: (*attempt_id).map(attempt_view_id),
                request_attempt_id: request_attempt_id.map(|attempt| attempt.to_string()),
                owner_id: owner_view_id(owner_id.ordinal),
                resource_id: Some(resource_view_id(resource_id.ordinal)),
                owner: None,
                resource: Some(RuntimeResourceView::from_descriptor(
                    resource_view_id(resource_id.ordinal),
                    descriptor,
                )),
            },
            RuntimeReceiptEvent::ResourceStateChanged {
                owner_id,
                resource_id,
                state,
                descriptor,
                attempt_id,
                request_attempt_id,
            } => Self {
                kind: match state {
                    RuntimeResourceState::Reserved => "resource-reserved",
                    RuntimeResourceState::Reconciled => "resource-reconciled",
                    RuntimeResourceState::Committed => "resource-committed",
                    RuntimeResourceState::Quarantined => "resource-quarantined",
                    RuntimeResourceState::Released => "resource-released",
                },
                attempt_id: (*attempt_id).map(attempt_view_id),
                request_attempt_id: request_attempt_id.map(|attempt| attempt.to_string()),
                owner_id: owner_view_id(owner_id.ordinal),
                resource_id: Some(resource_view_id(resource_id.ordinal)),
                owner: None,
                resource: Some(RuntimeResourceView::from_descriptor(
                    resource_view_id(resource_id.ordinal),
                    descriptor,
                )),
            },
            RuntimeReceiptEvent::ResourceReleased {
                owner_id,
                resource_id,
                attempt_id,
                request_attempt_id,
            } => Self {
                kind: "resource-released",
                attempt_id: (*attempt_id).map(attempt_view_id),
                request_attempt_id: request_attempt_id.map(|attempt| attempt.to_string()),
                owner_id: owner_view_id(owner_id.ordinal),
                resource_id: Some(resource_view_id(resource_id.ordinal)),
                owner: None,
                resource: None,
            },
        }
    }
}

impl From<openasr_core::runtime_receipts::ReceiptCompleteness> for ReceiptCompletenessView {
    fn from(completeness: openasr_core::runtime_receipts::ReceiptCompleteness) -> Self {
        Self {
            complete: completeness.complete,
            reason: completeness.reason.map(receipt_completeness_reason),
            live_state_complete: completeness.live_state_complete,
            live_state_reason: completeness
                .live_state_reason
                .map(receipt_completeness_reason),
            event_history_complete: completeness.event_history_complete,
            event_history_reason: completeness
                .event_history_reason
                .map(receipt_completeness_reason),
            dropped_events: completeness.dropped_events,
            dropped_owners: completeness.dropped_owners,
            rejected_resources: completeness.rejected_resources,
            dropped_notifications: completeness.dropped_notifications,
        }
    }
}

/// The attempt identifier is process-local and only meaningful alongside the
/// daemon-start identity. Prefixing its fixed-width ordinal prevents callers
/// from treating it as a durable cross-daemon numeric value.
fn attempt_view_id(attempt_id: ExecutionCacheAttemptId) -> String {
    format!("attempt-{}", attempt_id.ordinal())
}

fn receipt_completeness_reason(reason: ReceiptCompletenessReason) -> String {
    reason.as_str().to_string()
}

fn owner_view_id(ordinal: u64) -> String {
    format!("owner-{ordinal}")
}

fn resource_view_id(ordinal: u64) -> String {
    format!("resource-{ordinal}")
}

fn format_receipt_unavailable_reason(
    reason: openasr_core::runtime_receipts::RuntimeReceiptUnavailableReason,
) -> &'static str {
    match reason {
        openasr_core::runtime_receipts::RuntimeReceiptUnavailableReason::EntropyUnavailable => {
            "entropy-unavailable"
        }
        openasr_core::runtime_receipts::RuntimeReceiptUnavailableReason::IdentityExhausted => {
            "identity-exhausted"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_filter_wire_values_are_stable_and_safe() {
        assert_eq!(
            runtime_domain_kind(SafeMemoryDomainKind::SystemMemory),
            "system-memory"
        );
        assert_eq!(
            runtime_domain_kind(SafeMemoryDomainKind::DedicatedDevice),
            "dedicated-device"
        );
        assert!(DomainFilter::parse("physical-device").is_err());
    }

    #[test]
    fn unavailable_snapshot_is_explicitly_serialized() {
        let snapshot = ServerRuntime::default().runtime_receipt_snapshot();
        let snapshot = RuntimeReceiptSnapshot {
            availability: RuntimeReceiptAvailability::Unavailable {
                reason: openasr_core::runtime_receipts::RuntimeReceiptUnavailableReason::
                    EntropyUnavailable,
            },
            completeness: openasr_core::runtime_receipts::ReceiptCompleteness {
                complete: false,
                reason: Some(
                    openasr_core::runtime_receipts::ReceiptCompletenessReason::Unavailable(
                        openasr_core::runtime_receipts::RuntimeReceiptUnavailableReason::
                            EntropyUnavailable,
                    ),
                ),
                live_state_complete: false,
                live_state_reason: Some(
                    openasr_core::runtime_receipts::ReceiptCompletenessReason::Unavailable(
                        openasr_core::runtime_receipts::RuntimeReceiptUnavailableReason::
                            EntropyUnavailable,
                    ),
                ),
                event_history_complete: true,
                event_history_reason: None,
                dropped_events: 0,
                dropped_owners: 0,
                rejected_resources: 0,
                dropped_notifications: 0,
            },
            ..snapshot
        };
        let response = RuntimeReceiptResponse::from_snapshot(
            snapshot,
            LeaseReceiptShadow::Incomparable {
                reason: openasr_core::runtime_receipts::LeaseReceiptShadowIncomparable::
                    ReceiptsUnavailable,
            },
            ServerStartIdentity {
                pid: 42,
                nonce: None,
                started_at_unix_secs: None,
            },
            None,
            MAX_EVENT_LIMIT,
        );
        let json = serde_json::to_value(response).expect("serialize receipt response");
        assert_eq!(json["availability"], "unavailable");
        assert_eq!(json["unavailable_reason"], "entropy-unavailable");
        assert_eq!(json["snapshot_completeness"]["complete"], false);
        assert_eq!(json["daemon_start_identity"]["pid"], 42);
    }

    #[test]
    fn resource_events_keep_the_optional_attempt_wire_field() {
        let snapshot = ServerRuntime::default().runtime_receipt_snapshot();
        let event = RuntimeReceiptEvent::ResourceReleased {
            owner_id: RuntimeOwnerId {
                scope_id: snapshot.scope_id,
                ordinal: 3,
            },
            resource_id: RuntimeResourceId {
                scope_id: snapshot.scope_id,
                ordinal: 5,
            },
            attempt_id: None,
            request_attempt_id: None,
        };
        let json = serde_json::to_value(RuntimeEventView::from_event(&event))
            .expect("serialize resource event view");
        assert_eq!(json["kind"], "resource-released");
        assert_eq!(json["owner_id"], "owner-3");
        assert_eq!(json["resource_id"], "resource-5");
        assert!(json.get("attempt_id").is_none());

        let attempted = RuntimeEventView {
            kind: "resource-released",
            attempt_id: Some("attempt-41".to_string()),
            request_attempt_id: Some("00112233445566778899aabbccddeeff".to_string()),
            owner_id: "owner-3".to_string(),
            resource_id: Some("resource-5".to_string()),
            owner: None,
            resource: None,
        };
        let attempted_json =
            serde_json::to_value(attempted).expect("serialize attempted resource event view");
        assert_eq!(attempted_json["attempt_id"], "attempt-41");
        assert_eq!(
            attempted_json["request_attempt_id"],
            "00112233445566778899aabbccddeeff"
        );
    }

    #[test]
    fn resource_view_keeps_placement_and_lane_wire_fields() {
        let resource = RuntimeResourceView {
            id: "resource-5".to_string(),
            kind: "aabbccdd".to_string(),
            placement: "lane-bound",
            lane: Some(RuntimeLaneView {
                provider: "cuda".to_string(),
                placement: "full-device".to_string(),
                backend: "gpu".to_string(),
                device: "deadbeef".to_string(),
            }),
            domain: None,
            ledger_binding: "brokered",
            requested: RuntimeMetricView::Known(1),
            peak: RuntimeMetricView::Known(2),
            retained: RuntimeMetricView::Known(0),
            quote_confidence: "exact-committed".to_string(),
            observation_confidence: None,
        };
        let json = serde_json::to_value(resource).expect("serialize resource view");
        assert_eq!(json["placement"], "lane-bound");
        assert_eq!(json["lane"]["provider"], "cuda");
        assert_eq!(json["lane"]["device"], "deadbeef");
    }

    #[test]
    fn completeness_reason_wire_values_are_stable_kebab_case() {
        use openasr_core::runtime_receipts::RuntimeReceiptUnavailableReason;

        assert_eq!(
            receipt_completeness_reason(ReceiptCompletenessReason::Unavailable(
                RuntimeReceiptUnavailableReason::EntropyUnavailable,
            )),
            "entropy-unavailable"
        );
        assert_eq!(
            receipt_completeness_reason(ReceiptCompletenessReason::Unavailable(
                RuntimeReceiptUnavailableReason::IdentityExhausted,
            )),
            "identity-exhausted"
        );
        assert_eq!(
            receipt_completeness_reason(ReceiptCompletenessReason::IdentityExhausted),
            "identity-exhausted"
        );
        assert_eq!(
            receipt_completeness_reason(ReceiptCompletenessReason::EventCapacityExceeded),
            "event-capacity-exceeded"
        );
        assert_eq!(
            receipt_completeness_reason(ReceiptCompletenessReason::OwnerCapacityExceeded),
            "owner-capacity-exceeded"
        );
        assert_eq!(
            receipt_completeness_reason(ReceiptCompletenessReason::ResourceCapacityExceeded),
            "resource-capacity-exceeded"
        );
        assert_eq!(
            receipt_completeness_reason(ReceiptCompletenessReason::NotificationCapacityExceeded),
            "notification-capacity-exceeded"
        );
        assert_eq!(
            receipt_completeness_reason(ReceiptCompletenessReason::InvalidLifecycle),
            "invalid-lifecycle"
        );
    }

    #[test]
    fn identity_exhausted_is_projected_and_not_folded_to_known_zero() {
        let snapshot = ServerRuntime::default().runtime_receipt_snapshot();
        let snapshot = RuntimeReceiptSnapshot {
            availability: RuntimeReceiptAvailability::Unavailable {
                reason: openasr_core::runtime_receipts::RuntimeReceiptUnavailableReason::
                    IdentityExhausted,
            },
            completeness: openasr_core::runtime_receipts::ReceiptCompleteness {
                complete: false,
                reason: Some(
                    openasr_core::runtime_receipts::ReceiptCompletenessReason::IdentityExhausted,
                ),
                live_state_complete: false,
                live_state_reason: Some(
                    openasr_core::runtime_receipts::ReceiptCompletenessReason::IdentityExhausted,
                ),
                event_history_complete: true,
                event_history_reason: None,
                dropped_events: 0,
                dropped_owners: 0,
                rejected_resources: 0,
                dropped_notifications: 0,
            },
            ..snapshot
        };
        let response = RuntimeReceiptResponse::from_snapshot(
            snapshot,
            LeaseReceiptShadow::Incomparable {
                reason: openasr_core::runtime_receipts::LeaseReceiptShadowIncomparable::
                    ReceiptsUnavailable,
            },
            ServerStartIdentity {
                pid: 42,
                nonce: None,
                started_at_unix_secs: None,
            },
            None,
            MAX_EVENT_LIMIT,
        );
        let json = serde_json::to_value(response).expect("serialize receipt response");
        assert_eq!(json["availability"], "unavailable");
        assert_eq!(json["unavailable_reason"], "identity-exhausted");
        assert_eq!(json["snapshot_completeness"]["complete"], false);
        assert_eq!(
            json["snapshot_completeness"]["reason"],
            "identity-exhausted"
        );
        let rendered = json.to_string();
        assert!(!rendered.contains("\"status\":\"known\""));
        assert!(!rendered.contains("owner-0"));
        assert!(!rendered.contains("resource-0"));
        assert_ne!(json["unavailable_reason"], serde_json::json!(0));
        assert_ne!(json["unavailable_reason"], "known");
    }
}
