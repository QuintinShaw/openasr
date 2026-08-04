//! Policy-owned FireRedPunc runtime for streaming FINAL text.

use std::sync::{Arc, Mutex};

use thiserror::Error;

use crate::arch::emits_punctuation_for_model_architecture;
use crate::device::execution_policy::ExecutionIntent;
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrStreamingFinalTextProcessorSlot,
};
use crate::models::policy_resolved_aux_runtime::{
    PolicyResolvedAuxRuntime, PolicyResolvedAuxRuntimeError, resolve_auxiliary_execution_plan,
};
use crate::models::{
    aux_pack_registry::AuxPackKind,
    pack_verifier::{PackCandidate, PackRoute, PackVerifier},
};
use crate::punctuation::{PunctuationError, should_apply_punctuation};

use super::{
    config::FIRERED_PUNC_ARCHITECTURE_VALUE,
    pack::resolve_firered_punc_pack_path,
    policy_runtime::{FireRedPuncActor, load_actor, punctuate},
};

const STREAMING_PUNCTUATION_STAGE: &str = "firered-punc-streaming-stage-v1";

#[derive(Debug, Error)]
enum StreamingPunctuationRuntimeError {
    #[error("FireRedPunc runtime build failed: {0}")]
    Build(String),
    #[error("FireRedPunc inference failed: {0}")]
    Inference(#[from] PunctuationError),
}

struct StreamingPunctuationInitializer {
    execution_services: Arc<crate::NativeExecutionServices>,
    execution_plan: crate::device::execution_policy::ExecutionPlan,
    builder: Arc<
        dyn Fn(
                &crate::device::execution_policy::ExecutionCandidate,
            ) -> Result<FireRedPuncActor, StreamingPunctuationRuntimeError>
            + Send
            + Sync,
    >,
    adapter_id: &'static str,
}

/// Session-stable punctuation owner. Preparation is read-only; initialization
/// is deliberately delayed until the primary ASR session has constructed (or
/// warmed) so an optional post-processor cannot displace the user's primary
/// model from its preferred candidate.
pub(crate) struct PolicyResolvedStreamingPunctuator {
    slot: GgmlAsrStreamingFinalTextProcessorSlot,
    initializer: Mutex<Option<StreamingPunctuationInitializer>>,
}

impl PolicyResolvedStreamingPunctuator {
    pub(crate) fn prepare(
        execution_services: Arc<crate::NativeExecutionServices>,
        model_architecture: &'static str,
        adapter_id: &'static str,
        request_intent: &ExecutionIntent,
    ) -> Result<Option<Arc<Self>>, GgmlAsrExecutionError> {
        if !streaming_punctuation_stage_applies(model_architecture) {
            return Ok(None);
        }
        let Some(pack_path) = resolve_firered_punc_pack_path() else {
            return Ok(None);
        };
        let verified_pack = match PackVerifier.verify_candidate(PackCandidate::new(&pack_path)) {
            Ok(verified) => verified,
            Err(error) => {
                crate::stage_timing::log_detail_event(
                    "native_auxiliary_runtime",
                    format_args!(
                        "stage=streaming_punctuation event=disabled reason=pack-verification detail={error}"
                    ),
                );
                return Ok(None);
            }
        };
        if !matches!(
            verified_pack.route(),
            PackRoute::Aux {
                kind: AuxPackKind::Punctuation,
                ..
            }
        ) {
            crate::stage_timing::log_detail_event(
                "native_auxiliary_runtime",
                format_args!(
                    "stage=streaming_punctuation event=disabled reason=pack-route-mismatch"
                ),
            );
            return Ok(None);
        }
        let prepared_preflight = verified_pack.preflight().clone();
        let prepared_content_id = prepared_preflight.runtime_source.content_id().to_string();
        let execution_plan = resolve_auxiliary_execution_plan(
            execution_services.as_ref(),
            FIRERED_PUNC_ARCHITECTURE_VALUE,
            request_intent,
        )
        .map_err(|error| {
            GgmlAsrExecutionError::executor_failed(
                STREAMING_PUNCTUATION_STAGE,
                adapter_id,
                error.to_string(),
            )
        })?;
        let builder_services = Arc::clone(&execution_services);
        let builder = Arc::new(move |candidate: &_| {
            load_actor(
                builder_services.as_ref(),
                &prepared_preflight,
                &prepared_content_id,
                candidate,
            )
            .map_err(|error| StreamingPunctuationRuntimeError::Build(error.to_string()))
        });
        Ok(Some(Arc::new(Self {
            slot: GgmlAsrStreamingFinalTextProcessorSlot::default(),
            initializer: Mutex::new(Some(StreamingPunctuationInitializer {
                execution_services,
                execution_plan,
                builder,
                adapter_id,
            })),
        })))
    }

    pub(crate) fn slot(&self) -> GgmlAsrStreamingFinalTextProcessorSlot {
        self.slot.clone()
    }

    /// Idempotent session initialization. FireRedPunc is an automatic,
    /// optional enhancement: ordinary model errors disable it immediately,
    /// while typed resource/device errors first exhaust this stage's own
    /// candidate plan and then disable it without sacrificing ASR. Empty-plan
    /// and other internal invariant failures remain fail-closed.
    pub(crate) fn initialize(&self) -> Result<(), GgmlAsrExecutionError> {
        let mut initializer = self.initializer.lock().map_err(|_| {
            GgmlAsrExecutionError::executor_failed(
                STREAMING_PUNCTUATION_STAGE,
                "firered-punc",
                "streaming punctuation initializer is poisoned",
            )
        })?;
        let Some(inputs) = initializer.as_ref() else {
            return Ok(());
        };
        let runtime = match PolicyResolvedAuxRuntime::try_new(
            Arc::clone(&inputs.execution_services),
            inputs.execution_plan.clone(),
            STREAMING_PUNCTUATION_STAGE,
            Arc::clone(&inputs.builder),
        ) {
            Ok(runtime) => runtime,
            Err(error) if optional_punctuation_failure_disables_stage(&error) => {
                crate::stage_timing::log_detail_event(
                    "native_auxiliary_runtime",
                    format_args!("stage=streaming_punctuation event=disabled reason={error}"),
                );
                *initializer = None;
                return Ok(());
            }
            Err(error) => {
                return Err(GgmlAsrExecutionError::executor_failed(
                    STREAMING_PUNCTUATION_STAGE,
                    inputs.adapter_id,
                    error.to_string(),
                ));
            }
        };
        let runtime = Arc::new(Mutex::new(runtime));
        self.slot
            .install(Arc::new(move |text| {
                runtime
                    .lock()
                    .map_err(|_| "streaming punctuation runtime is poisoned".to_string())?
                    .invoke_replay_safe(|runtime| {
                        punctuate(runtime, text).map_err(|error| {
                            StreamingPunctuationRuntimeError::Build(error.to_string())
                        })
                    })
                    .map_err(|error| error.to_string())
            }))
            .map_err(|reason| {
                GgmlAsrExecutionError::executor_failed(
                    STREAMING_PUNCTUATION_STAGE,
                    inputs.adapter_id,
                    reason,
                )
            })?;
        *initializer = None;
        Ok(())
    }
}

fn optional_punctuation_failure_disables_stage<E>(
    error: &PolicyResolvedAuxRuntimeError<E>,
) -> bool {
    matches!(
        error,
        PolicyResolvedAuxRuntimeError::Operation(_)
            | PolicyResolvedAuxRuntimeError::CandidatesExhausted { .. }
    )
}

fn streaming_punctuation_stage_applies(model_architecture: &str) -> bool {
    should_apply_punctuation(emits_punctuation_for_model_architecture(model_architecture))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::execution_policy::ExecutionCandidateFailure;

    #[test]
    fn stage_applies_only_to_unpunctuated_architectures() {
        assert!(streaming_punctuation_stage_applies(
            crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID
        ));
        assert!(!streaming_punctuation_stage_applies(
            crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID
        ));
        assert!(!streaming_punctuation_stage_applies("no-such-architecture"));
    }

    #[test]
    fn optional_stage_disables_only_for_model_error_or_exhausted_candidates() {
        let ordinary = PolicyResolvedAuxRuntimeError::<StreamingPunctuationRuntimeError>::Operation(
            StreamingPunctuationRuntimeError::Build("invalid pack".to_string()),
        );
        let exhausted = PolicyResolvedAuxRuntimeError::<StreamingPunctuationRuntimeError>::CandidatesExhausted {
            stage: STREAMING_PUNCTUATION_STAGE,
            failure: ExecutionCandidateFailure::capacity("test", "full"),
            source: None,
        };
        let empty = PolicyResolvedAuxRuntimeError::<StreamingPunctuationRuntimeError>::EmptyPlan {
            stage: STREAMING_PUNCTUATION_STAGE,
        };
        let pinned =
            PolicyResolvedAuxRuntimeError::<StreamingPunctuationRuntimeError>::CandidateFailed {
                stage: STREAMING_PUNCTUATION_STAGE,
                failure: ExecutionCandidateFailure::device_lost("test", "lost"),
                source: None,
            };

        assert!(optional_punctuation_failure_disables_stage(&ordinary));
        assert!(optional_punctuation_failure_disables_stage(&exhausted));
        assert!(!optional_punctuation_failure_disables_stage(&empty));
        assert!(!optional_punctuation_failure_disables_stage(&pinned));
    }
}
