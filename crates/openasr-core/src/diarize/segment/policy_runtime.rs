//! Policy-resolved ownership for recording-local activity segmenters.
//!
//! Pyannote uses a Send-safe host owner on CPU and a thread-pinned FullDevice
//! ggml owner for an explicit Metal route. DiariZen owns native ggml state for every
//! backend. Both providers expose the same local-activity seam; provider
//! selection is frozen before materialization and never changes after an
//! inference error.

use std::sync::{Arc, Mutex};

use crate::{
    NativeExecutionServices,
    device::execution_policy::{ExecutionCandidate, ExecutionIntent},
    ggml_runtime::GgmlCpuGraphBackend,
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
    DIARIZEN_GGML_ARCHITECTURE_ID, LocalActivity, LocalActivitySegmenter, PyannetGgmlRuntime,
    PyannoteSegmenter, SegmentError, SegmenterProvider, decode_activity, diarizen,
    pack::{PreparedSegmenterSource, PreparedSelectedSegmenter},
    segment_pyannote_local_activity_serial,
};
use crate::diarize::embed::weights::WeightsError;
use crate::models::pyannote::PYANNOTE_GGML_ARCHITECTURE_ID;

const PYANNOTE_STAGE: &str = "pyannote-segmentation-stage-v1";
const DIARIZEN_STAGE: &str = "diarizen-segmentation-stage-v1";
const PYANNOTE_HOST_REPRESENTATION: &str = "pyannote-segmentation.f32-pure-rust.v1";
const DIARIZEN_RUNTIME_REPRESENTATION: &str = "diarizen-large-s80-v2.ggml.v1";

type SharedPyannote = AdmittedHostObject<PyannoteSegmenter>;
type PyannoteActor = PinnedRuntimeActor<PyannetGgmlRuntime>;
type DiariZenActor = PinnedRuntimeActor<diarizen::DiariZenRuntime>;

enum PyannoteRuntimeOwner {
    Host(SharedPyannote),
    Accelerated(PyannoteActor),
}

pub struct PolicyResolvedPyannoteSegmenterRuntime {
    runtime: Mutex<PolicyResolvedAuxRuntime<PyannoteRuntimeOwner, SegmentError>>,
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
        let accelerated_retained_quote =
            PyannetGgmlRuntime::quoted_persistent_host_commitment_bytes();

        let execution_plan = resolve_auxiliary_execution_plan(
            execution_services.as_ref(),
            PYANNOTE_GGML_ARCHITECTURE_ID,
            &execution_intent,
        )
        .map_err(|error| SegmentError::LoadFailed(error.to_string()))?;
        let services_for_builder = Arc::clone(&execution_services);
        let builder = Arc::new(move |candidate: &ExecutionCandidate| {
            let backend = resolved_runtime_for_auxiliary_candidate(candidate).backend();
            if backend == GgmlCpuGraphBackend::Cpu {
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
                        || {
                            build_admitted_pyannote(
                                &source,
                                &content_id,
                                peak_quote,
                                retained_quote,
                            )
                        },
                        |error| SegmentError::LoadFailed(error.to_string()),
                    )
                    .map(PyannoteRuntimeOwner::Host)
            } else {
                load_pyannote_actor(
                    services_for_builder.as_ref(),
                    &source,
                    &content_id,
                    backend,
                    candidate.placement,
                    peak_quote,
                    accelerated_retained_quote,
                )
                .map(PyannoteRuntimeOwner::Accelerated)
            }
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
            .invoke_replay_safe(|owner| match owner {
                PyannoteRuntimeOwner::Host(owner) => owner.segment_local_activity(
                    samples.clone(),
                    sample_rate_hz,
                    canceled,
                    progress,
                ),
                PyannoteRuntimeOwner::Accelerated(actor) => segment_pyannote_local_activity_serial(
                    samples.clone(),
                    sample_rate_hz,
                    canceled,
                    progress,
                    |window| {
                        actor
                            .call_mut_fallible({
                                let window = window.clone();
                                move |runtime| {
                                    runtime
                                        .forward(window.as_slice())
                                        .map(|(logp, frames)| decode_activity(&logp, frames))
                                }
                            })
                            .map_err(|error| SegmentError::Inference(error.to_string()))?
                            .map_err(|error| SegmentError::Inference(error.to_string()))
                    },
                ),
            })
            .map_err(policy_error)
    }
}

fn load_pyannote_actor(
    execution_services: &NativeExecutionServices,
    source: &PreparedSegmenterSource,
    expected_content_id: &str,
    backend: GgmlCpuGraphBackend,
    placement: crate::device::execution_policy::ExecutionPlacement,
    peak_quote: u64,
    retained_quote: u64,
) -> Result<PyannoteActor, SegmentError> {
    if source.preflight().runtime_source.content_id() != expected_content_id {
        return Err(content_changed(
            "PyanNet",
            expected_content_id,
            source.preflight().runtime_source.content_id(),
        ));
    }
    let key = AuxiliaryPinnedRuntimeCacheKey::for_current_lane::<PyannetGgmlRuntime>(
        PYANNOTE_GGML_ARCHITECTURE_ID,
        expected_content_id,
        "pyannote-segmentation.full-device-ggml.v2",
        backend,
    );
    let preflight = source.preflight().clone();
    let content_id = expected_content_id.to_string();
    let quote_content_id = content_id.clone();
    execution_services
        .pyannote_segmenter_actors()
        .get_or_try_insert_with(
            key,
            move || {
                let quote = SystemMemoryAllocationQuote::new(
                    format!(
                        "aux.{PYANNOTE_GGML_ARCHITECTURE_ID}.{quote_content_id}.device-runtime-state"
                    ),
                    peak_quote,
                    retained_quote,
                )
                .map_err(|error| SegmentError::LoadFailed(error.to_string()))?;
                Ok((retained_quote, quote))
            },
            move |quote| {
                let snapshot = preflight
                    .immutable_snapshot_matching_content_id(&content_id)
                    .map_err(|error| SegmentError::LoadFailed(error.to_string()))?;
                let transaction = SystemMemoryOwner::try_allocate_transaction(quote, || {
                    let runtime = PyannetGgmlRuntime::from_preflight(&snapshot, backend, placement)
                        .map_err(|error| SegmentError::LoadFailed(error.to_string()))?;
                    let actual_retained = runtime
                        .persistent_host_commitment_bytes()
                        .map_err(|error| SegmentError::LoadFailed(error.to_string()))?;
                    Ok::<_, SegmentError>(SystemMemoryAllocationOutcome::new(
                        runtime,
                        peak_quote,
                        actual_retained,
                    ))
                });
                match transaction {
                    Ok(owner) => Ok(owner),
                    Err(SystemMemoryAllocationTransactionError::Allocation(error)) => Err(error),
                    Err(SystemMemoryAllocationTransactionError::Capacity(error)) => {
                        Err(SegmentError::LoadFailed(error.to_string()))
                    }
                }
            },
            |error| SegmentError::Inference(error.to_string()),
        )
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
    let backend = resolved_runtime_for_auxiliary_candidate(candidate).backend();
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

    #[test]
    #[ignore = "requires OPENASR_PYANNOTE_PACK and a representative Metal device"]
    fn explicit_metal_pyannote_route_matches_cpu_product_activity() {
        let pack = std::env::var_os("OPENASR_PYANNOTE_PACK")
            .expect("OPENASR_PYANNOTE_PACK must point to a verified f32 pack");
        crate::test_process_env::with_test_process_env(
            [("OPENASR_PYANNOTE_PACK", Some(pack))],
            || {
                let samples: Vec<f32> = (0..12 * 16_000)
                    .map(|index| {
                        let time = index as f32 / 16_000.0;
                        0.11 * (time * 307.0 * std::f32::consts::TAU).sin()
                            + 0.04 * (time * 881.0 * std::f32::consts::TAU).cos()
                    })
                    .collect();
                let pcm = crate::PcmBuffer::from_vec(samples);
                let run = |intent| {
                    let placement = crate::GgmlExecutionTelemetryCollector::new();
                    let _placement_guard = placement.install();
                    let services = Arc::new(
                        NativeExecutionServices::for_local_process().expect("execution services"),
                    );
                    let runtime =
                        PolicyResolvedPyannoteSegmenterRuntime::load_with_intent(services, intent)
                            .expect("load PyanNet runtime")
                            .expect("PyanNet pack must resolve");
                    let activity = runtime
                        .segment_local_activity(pcm.full_slice(), 16_000, &|| false, None)
                        .expect("segment activity");
                    (activity, placement.snapshot())
                };
                let (cpu, _) = run(ExecutionIntent::CpuOnly);
                let (metal, metal_placement) = run(ExecutionIntent::ConstrainedAcceleratedOnly(
                    crate::device::execution_policy::AcceleratedDeviceConstraint::Provider(
                        crate::device::execution_route::ExecutionProvider::Metal,
                    ),
                ));
                assert_eq!(metal.windows, cpu.windows);
                assert_eq!(metal.speaker_count, cpu.speaker_count);
                assert!(
                    !metal_placement.observed_compute_nodes_by_backend.is_empty(),
                    "explicit Metal PyanNet route must observe recurrent/classifier compute nodes"
                );
                assert!(
                    metal_placement
                        .observed_compute_nodes_by_backend
                        .keys()
                        .all(|backend| {
                            let backend = backend.to_ascii_lowercase();
                            backend.starts_with("mtl") || backend.contains("metal")
                        }),
                    "explicit Metal PyanNet route observed non-Metal compute: {:?}",
                    metal_placement.observed_compute_nodes_by_backend
                );
            },
        );
    }
}
