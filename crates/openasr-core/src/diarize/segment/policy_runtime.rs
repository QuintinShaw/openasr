//! Policy-resolved ownership for recording-local activity segmenters.
//!
//! Pyannote is a Send-safe, CPU-only host owner. DiariZen owns native ggml
//! state and is therefore constructed, invoked, and destroyed on a dedicated
//! admitted actor thread. Both providers expose the same local-activity seam;
//! provider selection is frozen before materialization and never changes after
//! an inference error.

use std::sync::{Arc, Mutex};

use crate::{
    NativeExecutionServices,
    device::execution_policy::{ExecutionCandidate, ExecutionIntent},
    ggml_runtime::{AutoGpuPolicy, GgmlCpuGraphBackend},
    models::{
        admitted_pinned_runtime_actor_pool::PinnedRuntimeActor,
        policy_resolved_aux_runtime::{
            AuxiliaryPinnedRuntimeCacheKey, AuxiliaryRuntimeCacheKey, PolicyResolvedAuxRuntime,
            PolicyResolvedAuxRuntimeError, resolve_auxiliary_execution_plan,
            resolved_runtime_for_auxiliary_candidate,
        },
        system_memory_owner::{
            AdmittedHostObject, SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
            SystemMemoryAllocationTransactionError, SystemMemoryOwner,
        },
    },
};

use super::{
    DIARIZEN_GGML_ARCHITECTURE_ID, LocalActivity, LocalActivitySegmenter, PyannoteSegmenter,
    SegmentError, SegmenterProvider, diarizen,
    pack::{PreparedSegmenterSource, PreparedSelectedSegmenter},
};
use crate::diarize::embed::weights::WeightsError;
use crate::models::pyannote::PYANNOTE_GGML_ARCHITECTURE_ID;

const PYANNOTE_STAGE: &str = "pyannote-segmentation-stage-v1";
const DIARIZEN_STAGE: &str = "diarizen-segmentation-stage-v1";
const PYANNOTE_HOST_REPRESENTATION: &str = "pyannote-segmentation.f32-pure-rust.v1";
const DIARIZEN_RUNTIME_REPRESENTATION: &str = "diarizen-large-s80-v2.ggml.v1";

type SharedPyannote = AdmittedHostObject<PyannoteSegmenter>;
type DiariZenActor = PinnedRuntimeActor<diarizen::DiariZenRuntime>;

pub struct PolicyResolvedPyannoteSegmenterRuntime {
    runtime: Mutex<PolicyResolvedAuxRuntime<SharedPyannote, SegmentError>>,
}

impl PolicyResolvedPyannoteSegmenterRuntime {
    pub fn load(
        execution_services: Arc<NativeExecutionServices>,
    ) -> Result<Option<Self>, SegmentError> {
        Self::load_with_intent(execution_services, ExecutionIntent::Auto)
    }

    pub(crate) fn load_with_intent(
        execution_services: Arc<NativeExecutionServices>,
        execution_intent: ExecutionIntent,
    ) -> Result<Option<Self>, SegmentError> {
        let Some(prepared) = super::pack::pyannote_pack_path()
            .map(|_| {
                super::pack::prepare_segmenter(
                    crate::config::VoiceIdSegmenterPreference::Segmentation3_0,
                )
            })
            .transpose()?
        else {
            return Ok(None);
        };
        Self::from_prepared(execution_services, execution_intent, prepared).map(Some)
    }

    fn from_prepared(
        execution_services: Arc<NativeExecutionServices>,
        execution_intent: ExecutionIntent,
        prepared: PreparedSelectedSegmenter,
    ) -> Result<Self, SegmentError> {
        debug_assert_eq!(prepared.provider, SegmenterProvider::Segmentation3_0);
        let source = prepared.source;
        let content_id = source.content_id().to_string();
        let (retained_quote, peak_quote) = pyannote_source_quote(&source)?;

        let execution_plan = resolve_auxiliary_execution_plan(
            execution_services.as_ref(),
            PYANNOTE_GGML_ARCHITECTURE_ID,
            &execution_intent,
        )
        .map_err(|error| SegmentError::LoadFailed(error.to_string()))?;
        let services_for_builder = Arc::clone(&execution_services);
        let builder = Arc::new(move |_candidate: &ExecutionCandidate| {
            let key = AuxiliaryRuntimeCacheKey::for_current_lane::<PyannoteSegmenter>(
                PYANNOTE_GGML_ARCHITECTURE_ID,
                content_id.clone(),
                PYANNOTE_HOST_REPRESENTATION,
                GgmlCpuGraphBackend::Cpu,
            );
            services_for_builder
                .auxiliary_runtime_owners()
                .get_or_try_insert_admitted_with(
                    key,
                    retained_quote,
                    || build_admitted_pyannote(&source, &content_id, peak_quote, retained_quote),
                    |error| SegmentError::LoadFailed(error.to_string()),
                )
        });
        let runtime = PolicyResolvedAuxRuntime::try_new(
            execution_services,
            execution_plan,
            PYANNOTE_STAGE,
            builder,
        )
        .map_err(policy_error)?;
        Ok(Self {
            runtime: Mutex::new(runtime),
        })
    }
}

impl LocalActivitySegmenter for PolicyResolvedPyannoteSegmenterRuntime {
    fn segment_local_activity(
        &self,
        samples: crate::PcmSlice,
        sample_rate_hz: u32,
        canceled: &dyn Fn() -> bool,
        progress: Option<&crate::api::backend::WorkProgressObserver>,
    ) -> Result<LocalActivity, SegmentError> {
        self.runtime
            .lock()
            .map_err(|_| SegmentError::Inference("pyannote runtime lock is poisoned".into()))?
            .invoke_replay_safe(|owner| {
                owner.segment_local_activity(samples.clone(), sample_rate_hz, canceled, progress)
            })
            .map_err(policy_error)
    }
}

struct PolicyResolvedDiariZenSegmenterRuntime {
    runtime: Mutex<PolicyResolvedAuxRuntime<DiariZenActor, SegmentError>>,
}

impl PolicyResolvedDiariZenSegmenterRuntime {
    fn from_prepared(
        execution_services: Arc<NativeExecutionServices>,
        execution_intent: ExecutionIntent,
        prepared: PreparedSelectedSegmenter,
    ) -> Result<Self, SegmentError> {
        debug_assert_eq!(prepared.provider, SegmenterProvider::DiariZen);
        let execution_plan = resolve_auxiliary_execution_plan(
            execution_services.as_ref(),
            DIARIZEN_GGML_ARCHITECTURE_ID,
            &execution_intent,
        )
        .map_err(|error| SegmentError::LoadFailed(error.to_string()))?;
        let services_for_builder = Arc::clone(&execution_services);
        let (preflight, content_id) = prepared.source.into_parts();
        let builder = Arc::new(move |candidate: &ExecutionCandidate| {
            load_diarizen_actor(
                services_for_builder.as_ref(),
                &preflight,
                &content_id,
                candidate,
            )
        });
        let runtime = PolicyResolvedAuxRuntime::try_new(
            execution_services,
            execution_plan,
            DIARIZEN_STAGE,
            builder,
        )
        .map_err(policy_error)?;
        Ok(Self {
            runtime: Mutex::new(runtime),
        })
    }
}

impl LocalActivitySegmenter for PolicyResolvedDiariZenSegmenterRuntime {
    fn segment_local_activity(
        &self,
        samples: crate::PcmSlice,
        sample_rate_hz: u32,
        canceled: &dyn Fn() -> bool,
        progress: Option<&crate::api::backend::WorkProgressObserver>,
    ) -> Result<LocalActivity, SegmentError> {
        super::segment_diarizen_local_activity(
            samples,
            sample_rate_hz,
            canceled,
            progress,
            |window| {
                self.runtime
                    .lock()
                    .map_err(|_| {
                        SegmentError::Inference("DiariZen runtime lock is poisoned".to_string())
                    })?
                    .invoke_replay_safe(|actor| {
                        actor
                            .call_mut_fallible({
                                let window = window.clone();
                                move |runtime| runtime.infer(window.as_slice())
                            })
                            .map_err(|error| SegmentError::Inference(error.to_string()))?
                            .map_err(diarizen_error)
                    })
                    .map_err(policy_error)
            },
        )
    }
}

/// Selected, admitted provider for one request. The provider is frozen during
/// preflight; candidate retry may change only its execution placement.
pub struct PolicyResolvedSegmenterRuntime {
    provider: SegmenterProvider,
    adapter: Arc<dyn LocalActivitySegmenter>,
}

impl PolicyResolvedSegmenterRuntime {
    pub(crate) fn load_prepared(
        execution_services: Arc<NativeExecutionServices>,
        execution_intent: ExecutionIntent,
        prepared: PreparedSelectedSegmenter,
    ) -> Result<Self, SegmentError> {
        let provider = prepared.provider;
        let adapter: Arc<dyn LocalActivitySegmenter> = match provider {
            SegmenterProvider::Segmentation3_0 => {
                Arc::new(PolicyResolvedPyannoteSegmenterRuntime::from_prepared(
                    execution_services,
                    execution_intent,
                    prepared,
                )?)
            }
            SegmenterProvider::DiariZen => {
                Arc::new(PolicyResolvedDiariZenSegmenterRuntime::from_prepared(
                    execution_services,
                    execution_intent,
                    prepared,
                )?)
            }
        };
        Ok(Self { provider, adapter })
    }

    pub(crate) fn provider(&self) -> SegmenterProvider {
        self.provider
    }

    pub(crate) fn adapter(&self) -> &dyn LocalActivitySegmenter {
        self.adapter.as_ref()
    }
}

fn load_diarizen_actor(
    execution_services: &NativeExecutionServices,
    preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    expected_content_id: &str,
    candidate: &ExecutionCandidate,
) -> Result<DiariZenActor, SegmentError> {
    if preflight.runtime_source.content_id() != expected_content_id {
        return Err(content_changed(
            "DiariZen",
            expected_content_id,
            preflight.runtime_source.content_id(),
        ));
    }
    let backend =
        resolved_runtime_for_auxiliary_candidate(candidate, AutoGpuPolicy::AllBackends).backend();
    let key = AuxiliaryPinnedRuntimeCacheKey::for_current_lane::<diarizen::DiariZenRuntime>(
        DIARIZEN_GGML_ARCHITECTURE_ID,
        expected_content_id,
        DIARIZEN_RUNTIME_REPRESENTATION,
        backend,
    );
    let quote = diarizen::DiariZenRuntime::quote_candidate_system_memory(preflight)
        .map_err(diarizen_error)?;
    let retained_bytes = quote.retained_bytes;
    let preflight = preflight.clone();
    let content_id = expected_content_id.to_string();
    execution_services
        .diarizen_segmenter_actors()
        .get_or_try_insert_with(
            key,
            || Ok((retained_bytes, quote)),
            move |quote| {
                let snapshot = preflight
                    .immutable_snapshot_matching_content_id(&content_id)
                    .map_err(|error| SegmentError::LoadFailed(error.to_string()))?;
                let mut owner = diarizen::DiariZenRuntime::try_allocate_inside_parent_candidate(
                    quote, &snapshot, backend,
                )
                .map_err(diarizen_error)?;
                let warmup = vec![0.0_f32; diarizen::DIARIZEN_WINDOW_SAMPLES];
                owner.infer(&warmup).map_err(diarizen_error)?;
                Ok(owner)
            },
            |error| SegmentError::Inference(error.to_string()),
        )
}

fn build_admitted_pyannote(
    source: &PreparedSegmenterSource,
    expected_content_id: &str,
    peak_quote: u64,
    retained_quote: u64,
) -> Result<SharedPyannote, SegmentError> {
    let quote = SystemMemoryAllocationQuote::new(
        format!("aux.{PYANNOTE_GGML_ARCHITECTURE_ID}.{expected_content_id}.host-state"),
        peak_quote,
        retained_quote,
    )
    .map_err(|error| SegmentError::LoadFailed(error.to_string()))?;
    let transaction = SystemMemoryOwner::try_allocate_transaction(quote, || {
        let snapshot = source
            .preflight()
            .immutable_snapshot_matching_content_id(expected_content_id)
            .map_err(|error| SegmentError::LoadFailed(error.to_string()))?;
        let segmenter = PyannoteSegmenter::from_preflight(&snapshot).map_err(weights_error)?;
        let actual_retained = segmenter
            .persistent_host_commitment_bytes()
            .map_err(weights_error)?;
        Ok::<_, SegmentError>(SystemMemoryAllocationOutcome::new(
            segmenter,
            peak_quote,
            actual_retained,
        ))
    });
    let owner = match transaction {
        Ok(owner) => owner,
        Err(SystemMemoryAllocationTransactionError::Allocation(error)) => return Err(error),
        Err(SystemMemoryAllocationTransactionError::Capacity(error)) => {
            return Err(SegmentError::LoadFailed(error.to_string()));
        }
    };
    Ok(Arc::new(owner))
}

fn pyannote_source_quote(source: &PreparedSegmenterSource) -> Result<(u64, u64), SegmentError> {
    let preflight = source.preflight();
    if preflight.runtime_source.content_id() != source.content_id() {
        return Err(content_changed(
            "segmenter",
            source.content_id(),
            preflight.runtime_source.content_id(),
        ));
    }
    let retained =
        PyannoteSegmenter::quoted_persistent_host_commitment_bytes(&preflight.tensor_index)
            .map_err(weights_error)?;
    let peak = preflight
        .runtime_source
        .immutable_snapshot_construction_peak_bytes(retained)
        .map_err(|error| SegmentError::LoadFailed(error.to_string()))?;
    Ok((retained, peak))
}

fn weights_error(error: WeightsError) -> SegmentError {
    SegmentError::LoadFailed(error.to_string())
}

fn diarizen_error(error: diarizen::DiariZenSegmenterError) -> SegmentError {
    if error.is_canceled() {
        SegmentError::Canceled
    } else {
        SegmentError::Inference(error.to_string())
    }
}

fn policy_error(error: PolicyResolvedAuxRuntimeError<SegmentError>) -> SegmentError {
    match error {
        PolicyResolvedAuxRuntimeError::Operation(error) => error,
        error => SegmentError::Inference(error.to_string()),
    }
}

fn content_changed(label: &str, expected: &str, actual: &str) -> SegmentError {
    SegmentError::LoadFailed(format!(
        "{label} pack changed between preflight and construction: expected {expected}, got {actual}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diarizen_graph_cancellation_remains_typed_across_policy_boundary() {
        for source in [
            crate::ggml_runtime::GgmlCpuGraphError::Aborted,
            crate::ggml_runtime::GgmlCpuGraphError::Canceled,
        ] {
            let mapped = diarizen_error(diarizen::DiariZenSegmenterError::Graph {
                step: "fixture",
                source,
            });
            assert!(matches!(mapped, SegmentError::Canceled));
        }
    }
}
