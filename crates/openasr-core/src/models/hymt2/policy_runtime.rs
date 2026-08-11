//! Execution-policy ownership for one realtime Hy-MT2 translation lane.
//!
//! Translation is stateful but has a narrow replay-safe frontier. Runtime
//! construction, warmup, and the first request may advance after a typed
//! candidate-local failure because no translation has been observed yet. The
//! first successful worker output permanently pins the lane; later failures
//! are surfaced instead of replaying accumulated context on another device.

use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use thiserror::Error;

use crate::{
    ExecutionTarget, NativeExecutionServices, TranslationRequest, TranslationWorkerOutput,
    device::execution_policy::{ExecutionCandidate, ExecutionIntent},
    models::{
        aux_pack_registry::AuxPackKind,
        native_execution_services::current_execution_candidate_failure,
        pack_verifier::{PackCandidate, PackRoute, PackVerificationError, PackVerifier},
        policy_resolved_aux_runtime::{
            AuxiliaryPinnedRuntimeCacheKey, PolicyResolvedAuxRuntime,
            PolicyResolvedAuxRuntimeError, PolicyResolvedStatefulAuxRuntime,
            resolve_auxiliary_execution_plan, resolved_runtime_for_auxiliary_candidate,
        },
    },
};

use super::{
    Hymt2PrefixCacheConfig, Hymt2Runtime, Hymt2RuntimeError, Hymt2TranslationSessionCache,
    config::HUNYUAN_DENSE_ARCHITECTURE_VALUE,
};

const REALTIME_TRANSLATION_STAGE: &str = "hymt2-realtime-translation-stage-v1";
const WARMUP_SOURCE_TEXT: &str = "你好";
static NEXT_TRANSLATION_ACTOR_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) struct Hymt2TranslationCandidate {
    runtime: Hymt2Runtime,
    cache: Hymt2TranslationSessionCache,
}

impl Hymt2TranslationCandidate {
    fn translate(
        &mut self,
        request: &TranslationRequest,
    ) -> Result<TranslationWorkerOutput, PolicyResolvedHymt2Error> {
        self.runtime
            .translate_request_with_cache(&mut self.cache, request)
            .map_err(PolicyResolvedHymt2Error::from)
    }
}

#[derive(Debug, Error)]
pub enum PolicyResolvedHymt2Error {
    #[error(transparent)]
    Runtime(#[from] Hymt2RuntimeError),
    #[error("Hy-MT2 execution policy failed: {reason}")]
    Policy { reason: String },
}

/// Stateful translation runtime with a one-way replay-safe -> pinned
/// transition after its first successful output.
pub struct PolicyResolvedHymt2TranslationRuntime {
    execution_services: Arc<NativeExecutionServices>,
    actor_instance_id: u64,
    runtime: PolicyResolvedStatefulAuxRuntime<
        crate::models::admitted_pinned_runtime_actor_pool::PinnedRuntimeActor<
            Hymt2TranslationCandidate,
        >,
        PolicyResolvedHymt2Error,
    >,
}

impl PolicyResolvedHymt2TranslationRuntime {
    pub fn load(
        execution_services: Arc<NativeExecutionServices>,
        pack_path: PathBuf,
        execution_target: ExecutionTarget,
        max_source_clause_chars: usize,
    ) -> Result<Self, PolicyResolvedHymt2Error> {
        let verified_pack = PackVerifier
            .verify_candidate(PackCandidate::new(&pack_path))
            .map_err(|error| {
                PolicyResolvedHymt2Error::Runtime(map_pack_verification_error(error))
            })?;
        if !matches!(
            verified_pack.route(),
            PackRoute::Aux {
                kind: AuxPackKind::Translation,
                ..
            }
        ) {
            return Err(PolicyResolvedHymt2Error::Runtime(
                Hymt2RuntimeError::Preflight {
                    reason: format!(
                        "Hy-MT2 pack route is not auxiliary translation: {:?}",
                        verified_pack.route()
                    ),
                },
            ));
        }
        let preflight = verified_pack.preflight().clone();
        let content_id = preflight.runtime_source.content_id().to_string();

        let intent = ExecutionIntent::from(execution_target);
        let actor_instance_id = NEXT_TRANSLATION_ACTOR_INSTANCE_ID.fetch_add(1, Ordering::Relaxed);
        let execution_plan = resolve_auxiliary_execution_plan(
            execution_services.as_ref(),
            HUNYUAN_DENSE_ARCHITECTURE_VALUE,
            &intent,
        )
        .map_err(|error| PolicyResolvedHymt2Error::Policy {
            reason: error.to_string(),
        })?;

        let builder_preflight = preflight;
        let builder_content_id = content_id;
        let builder_services = Arc::clone(&execution_services);
        let builder = Arc::new(move |candidate: &ExecutionCandidate| {
            build_candidate(
                builder_services.as_ref(),
                &builder_preflight,
                &builder_content_id,
                candidate,
                max_source_clause_chars,
                actor_instance_id,
            )
        });
        let runtime = PolicyResolvedAuxRuntime::try_new(
            Arc::clone(&execution_services),
            execution_plan,
            REALTIME_TRANSLATION_STAGE,
            builder,
        )
        .map_err(policy_error)?;
        Ok(Self {
            execution_services,
            actor_instance_id,
            runtime: PolicyResolvedStatefulAuxRuntime::new(runtime),
        })
    }

    pub fn translate_request(
        &mut self,
        request: &TranslationRequest,
    ) -> Result<TranslationWorkerOutput, PolicyResolvedHymt2Error> {
        let request = request.clone();
        self.runtime
            .invoke(|candidate| {
                let request = request.clone();
                candidate
                    .call_mut(move |runtime| runtime.translate(&request))
                    .map_err(|error| PolicyResolvedHymt2Error::Policy {
                        reason: error.to_string(),
                    })?
            })
            .map_err(policy_error)
    }

    pub fn output_committed(&self) -> bool {
        self.runtime.output_committed()
    }
}

impl Drop for PolicyResolvedHymt2TranslationRuntime {
    fn drop(&mut self) {
        self.execution_services
            .hymt2_translation_actors()
            .evict_where(|key| key.has_instance_id(self.actor_instance_id));
    }
}

fn map_pack_verification_error(error: PackVerificationError) -> Hymt2RuntimeError {
    match error {
        PackVerificationError::RuntimeSource { source, .. } => {
            Hymt2RuntimeError::RuntimeSourcePath { source }
        }
        other => Hymt2RuntimeError::Preflight {
            reason: other.to_string(),
        },
    }
}

fn build_candidate(
    execution_services: &NativeExecutionServices,
    preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    expected_content_id: &str,
    candidate: &ExecutionCandidate,
    max_source_clause_chars: usize,
    actor_instance_id: u64,
) -> Result<
    crate::models::admitted_pinned_runtime_actor_pool::PinnedRuntimeActor<
        Hymt2TranslationCandidate,
    >,
    PolicyResolvedHymt2Error,
> {
    let actual_content_id = preflight.runtime_source.content_id();
    if actual_content_id != expected_content_id {
        return Err(PolicyResolvedHymt2Error::Policy {
            reason: format!(
                "Hy-MT2 pack changed between planning and construction: expected {expected_content_id}, got {actual_content_id}"
            ),
        });
    }
    let backend = resolved_runtime_for_auxiliary_candidate(candidate).backend();
    let key = AuxiliaryPinnedRuntimeCacheKey::for_current_session_lane::<Hymt2TranslationCandidate>(
        HUNYUAN_DENSE_ARCHITECTURE_VALUE,
        expected_content_id,
        "hymt2.translation-candidate.v1",
        actor_instance_id,
        backend,
    );
    let build_preflight = preflight.clone();
    let build_content_id = expected_content_id.to_string();
    execution_services.hymt2_translation_actors().get_or_try_insert_with(
        key,
        || {
            let quote = Hymt2Runtime::quote_candidate_system_memory(
                preflight,
                backend,
                max_source_clause_chars,
            )?;
            Ok((quote.retained_bytes, quote))
        },
        move |quote| {
            let snapshot = build_preflight
                .immutable_snapshot_matching_content_id(&build_content_id)
                .map_err(|source| Hymt2RuntimeError::RuntimeSourcePath { source })?;
            match crate::models::system_memory_owner::SystemMemoryOwner::try_allocate_transaction(
                quote,
                || {
                    let runtime = Hymt2Runtime::from_preflight_with_clause_envelope_inside_parent_transaction(
                        &snapshot,
                        backend,
                        max_source_clause_chars,
                    )?;
                    let mut cache = runtime
                        .new_translation_session_cache(Hymt2PrefixCacheConfig::default())?;
                    runtime.translate_clause_with_cache(
                        &mut cache,
                        "warmup",
                        WARMUP_SOURCE_TEXT,
                        &[],
                        true,
                    )?;
                    cache.invalidate();
                    if let Some(failure) = current_execution_candidate_failure() {
                        return Err(PolicyResolvedHymt2Error::Policy {
                            reason: format!(
                                "Hy-MT2 warmup recorded {:?} at {}: {}",
                                failure.kind, failure.operation, failure.detail
                            ),
                        });
                    }
                    let retained = runtime
                        .retained_system_memory_bytes()
                        .and_then(|runtime_bytes| {
                            cache
                                .retained_system_memory_bytes()
                                .and_then(|cache_bytes| runtime_bytes.checked_add(cache_bytes).ok_or_else(|| {
                                    "Hy-MT2 candidate retained-byte sum overflowed".to_string()
                                }))
                        })
                        .map_err(|reason| PolicyResolvedHymt2Error::Policy { reason })?;
                    Ok::<_, PolicyResolvedHymt2Error>(
                        crate::models::system_memory_owner::SystemMemoryAllocationOutcome::new(
                            Hymt2TranslationCandidate { runtime, cache },
                            retained,
                            retained,
                        ),
                    )
                },
            ) {
                Ok(owner) => Ok(owner),
                Err(crate::models::system_memory_owner::SystemMemoryAllocationTransactionError::Allocation(error)) => Err(error),
                Err(crate::models::system_memory_owner::SystemMemoryAllocationTransactionError::Capacity(error)) => {
                    Err(PolicyResolvedHymt2Error::Policy { reason: error.to_string() })
                }
            }
        },
        |error| PolicyResolvedHymt2Error::Policy {
            reason: error.to_string(),
        },
    )
}

fn policy_error(
    error: PolicyResolvedAuxRuntimeError<PolicyResolvedHymt2Error>,
) -> PolicyResolvedHymt2Error {
    match error {
        PolicyResolvedAuxRuntimeError::Operation(error) => error,
        error => PolicyResolvedHymt2Error::Policy {
            reason: error.to_string(),
        },
    }
}
