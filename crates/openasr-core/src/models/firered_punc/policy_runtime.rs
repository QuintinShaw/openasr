//! Admitted owner-thread runtime for FireRedPunc.

use std::path::Path;

use thiserror::Error;

use crate::device::execution_policy::ExecutionCandidate;
use crate::models::admitted_pinned_runtime_actor_pool::PinnedRuntimeActor;
use crate::models::policy_resolved_aux_runtime::{
    AuxiliaryPinnedRuntimeCacheKey, resolved_runtime_for_auxiliary_candidate,
};
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

pub(crate) fn load_actor(
    execution_services: &NativeExecutionServices,
    pack_path: &Path,
    expected_content_id: &str,
    candidate: &ExecutionCandidate,
) -> Result<FireRedPuncActor, PolicyOwnedFireRedPuncError> {
    let source = crate::validate_ggml_runtime_source_path(pack_path)
        .map_err(|error| FireRedPuncRuntimeError::Read(error.to_string()))?;
    if source.content_id() != expected_content_id {
        return Err(PolicyOwnedFireRedPuncError::ContentChanged {
            expected: expected_content_id.to_string(),
            actual: source.content_id().to_string(),
        });
    }
    let backend = resolved_runtime_for_auxiliary_candidate(
        candidate,
        crate::ggml_runtime::AutoGpuPolicy::AllBackends,
    )
    .backend();
    let key = AuxiliaryPinnedRuntimeCacheKey::for_current_lane::<FireRedPuncRuntime>(
        FIRERED_PUNC_ARCHITECTURE_VALUE,
        expected_content_id,
        "firered-punc.runtime.v1",
        backend,
    );
    let build_path = pack_path.to_path_buf();
    let build_content_id = expected_content_id.to_string();
    execution_services
        .firered_punc_actors()
        .get_or_try_insert_with(
            key,
            || {
                let quote = FireRedPuncRuntime::quote_candidate_system_memory(&source)?;
                Ok((quote.retained_bytes, quote))
            },
            move |quote| {
                let source = crate::validate_ggml_runtime_source_path(&build_path)
                    .map_err(|error| FireRedPuncRuntimeError::Read(error.to_string()))?;
                if source.content_id() != build_content_id {
                    return Err(PolicyOwnedFireRedPuncError::ContentChanged {
                        expected: build_content_id,
                        actual: source.content_id().to_string(),
                    });
                }
                let owner = FireRedPuncRuntime::try_allocate_inside_parent_candidate(
                    quote, &source, backend,
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
