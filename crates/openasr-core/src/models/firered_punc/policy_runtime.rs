//! Admitted owner-thread runtime for FireRedPunc.

use thiserror::Error;

use crate::device::execution_policy::ExecutionCandidate;
use crate::models::admitted_pinned_runtime_actor_pool::PinnedRuntimeActor;
use crate::models::policy_resolved_aux_runtime::{
    AuxiliaryPinnedRuntimeCacheKey, resolved_runtime_for_auxiliary_candidate,
};
use crate::models::runtime_receipts::RuntimeOwnerDescriptor;
use crate::{NativeExecutionServices, punctuation::PunctuationError};

use super::config::FIRERED_PUNC_ARCHITECTURE_VALUE;
use super::runtime::{FireRedPuncRuntime, FireRedPuncRuntimeError};

pub(crate) type FireRedPuncActor = PinnedRuntimeActor<FireRedPuncRuntime>;
const WARMUP_TEXT: &str = "你好";

#[derive(Debug, Error)]
pub(crate) enum PolicyOwnedFireRedPuncError {
    #[error(transparent)]
    Runtime(#[from] FireRedPuncRuntimeError),
    #[error(transparent)]
    Punctuation(#[from] PunctuationError),
    #[error("FireRedPunc owner-thread runtime failed: {0}")]
    Actor(String),
    #[error("FireRedPunc pack identity changed: expected {expected}, got {actual}")]
    ContentChanged { expected: String, actual: String },
}

fn actor_receipt_descriptor(
    content_id: &str,
    candidate: &ExecutionCandidate,
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
) -> Option<RuntimeOwnerDescriptor> {
    let collector = crate::models::native_execution_services::current_runtime_receipts()?;
    let lane = collector.lane_projection(
        candidate.device.route.provider,
        &candidate.device.route.stable_id,
        candidate.placement,
        backend,
    )?;
    collector.owner_descriptor(
        "firered-punc.actor-runtime",
        Some(content_id),
        Some("firered-punc.runtime.v1"),
        Some(lane),
    )
}

pub(crate) fn load_actor(
    execution_services: &NativeExecutionServices,
    preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    expected_content_id: &str,
    candidate: &ExecutionCandidate,
) -> Result<FireRedPuncActor, PolicyOwnedFireRedPuncError> {
    if preflight.runtime_source.content_id() != expected_content_id {
        return Err(PolicyOwnedFireRedPuncError::ContentChanged {
            expected: expected_content_id.to_string(),
            actual: preflight.runtime_source.content_id().to_string(),
        });
    }
    let backend = resolved_runtime_for_auxiliary_candidate(candidate).backend();
    let key = AuxiliaryPinnedRuntimeCacheKey::for_current_lane::<FireRedPuncRuntime>(
        FIRERED_PUNC_ARCHITECTURE_VALUE,
        expected_content_id,
        "firered-punc.runtime.v1",
        backend,
    );
    let build_preflight = preflight.clone();
    let build_content_id = expected_content_id.to_string();
    let owner_descriptor = actor_receipt_descriptor(expected_content_id, candidate, backend);
    execution_services
        .firered_punc_actors()
        .get_or_try_insert_with_owner_receipt(
            key,
            owner_descriptor,
            || {
                let quote = FireRedPuncRuntime::quote_candidate_system_memory(preflight)?;
                Ok((quote.retained_bytes, quote))
            },
            move |quote| {
                let snapshot = build_preflight
                    .immutable_snapshot_matching_content_id(&build_content_id)
                    .map_err(|error| FireRedPuncRuntimeError::Read(error.to_string()))?;
                let owner = FireRedPuncRuntime::try_allocate_inside_parent_candidate(
                    quote, &snapshot, backend,
                )
                .map_err(PolicyOwnedFireRedPuncError::from)?;
                owner.punctuate(WARMUP_TEXT)?;
                Ok(owner)
            },
            |error| PolicyOwnedFireRedPuncError::Actor(error.to_string()),
        )
}

pub(crate) fn punctuate(
    actor: &FireRedPuncActor,
    text: &str,
) -> Result<String, PolicyOwnedFireRedPuncError> {
    let text = text.to_string();
    actor
        .call_mut(move |runtime| runtime.punctuate(&text))
        .map_err(|error| PolicyOwnedFireRedPuncError::Actor(error.to_string()))?
        .map_err(PolicyOwnedFireRedPuncError::from)
}
