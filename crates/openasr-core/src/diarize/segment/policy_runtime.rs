//! Policy-resolved ownership for recording-local activity segmenters.
//!
//! Pyannote uses a Send-safe host owner on CPU, a thread-pinned FullDevice
//! ggml owner on Metal, and a verified host-SincNet/direct-GPU recurrent
//! Hybrid owner on CUDA and Vulkan. DiariZen owns native ggml
//! state for every backend. Both providers expose the same local-activity seam; provider
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
    segment_pyannote_local_activity_batched, segment_pyannote_local_activity_serial,
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
    FullDevice(PyannoteActor),
    Hybrid {
        frontend: SharedPyannote,
        recurrent: PyannoteActor,
    },
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
        let execution_plan = resolve_auxiliary_execution_plan(
            execution_services.as_ref(),
            PYANNOTE_GGML_ARCHITECTURE_ID,
            &execution_intent,
        )
        .map_err(|error| SegmentError::LoadFailed(error.to_string()))?;
        let services_for_builder = Arc::clone(&execution_services);
        let builder = Arc::new(move |candidate: &ExecutionCandidate| {
            let backend = resolved_runtime_for_auxiliary_candidate(candidate).backend();
            match candidate.placement {
                crate::device::execution_policy::ExecutionPlacement::CpuOnly => load_pyannote_host(
                    services_for_builder.as_ref(),
                    &source,
                    &content_id,
                    peak_quote,
                    retained_quote,
                )
                .map(PyannoteRuntimeOwner::Host),
                crate::device::execution_policy::ExecutionPlacement::FullDevice => {
                    load_pyannote_actor(
                        services_for_builder.as_ref(),
                        &source,
                        &content_id,
                        backend,
                        candidate.placement,
                        peak_quote,
                    )
                    .map(PyannoteRuntimeOwner::FullDevice)
                }
                crate::device::execution_policy::ExecutionPlacement::Hybrid => {
                    let frontend = load_pyannote_host(
                        services_for_builder.as_ref(),
                        &source,
                        &content_id,
                        peak_quote,
                        retained_quote,
                    )?;
                    let recurrent = load_pyannote_actor(
                        services_for_builder.as_ref(),
                        &source,
                        &content_id,
                        backend,
                        candidate.placement,
                        peak_quote,
                    )?;
                    Ok(PyannoteRuntimeOwner::Hybrid {
                        frontend,
                        recurrent,
                    })
                }
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
                PyannoteRuntimeOwner::FullDevice(actor) => segment_pyannote_local_activity_serial(
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
                PyannoteRuntimeOwner::Hybrid {
                    frontend,
                    recurrent,
                } => segment_pyannote_local_activity_batched(
                    samples.clone(),
                    sample_rate_hz,
                    canceled,
                    progress,
                    PyannetGgmlRuntime::hybrid_batch_width(),
                    |windows| {
                        let prepared = windows
                            .iter()
                            .map(|window| {
                                if canceled() {
                                    return Err(SegmentError::Canceled);
                                }
                                frontend
                                    .prepare_accelerated_features(window.as_slice())
                                    .map_err(|error| SegmentError::Inference(error.to_string()))
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let frames = prepared.first().map_or(0, |(_, frames)| *frames);
                        if prepared.iter().any(|(_, actual)| *actual != frames) {
                            return Err(SegmentError::Inference(
                                "pyannote accelerated batch frame counts differ".to_string(),
                            ));
                        }
                        if canceled() {
                            return Err(SegmentError::Canceled);
                        }
                        recurrent
                            .call_mut_fallible(move |runtime| {
                                let features = prepared
                                    .iter()
                                    .map(|(features, _)| features.as_slice())
                                    .collect::<Vec<_>>();
                                runtime
                                    .forward_features_batch(&features, frames)
                                    .map(|batch| {
                                        batch
                                            .into_iter()
                                            .map(|logp| decode_activity(&logp, frames))
                                            .collect::<Vec<_>>()
                                    })
                            })
                            .map_err(|error| SegmentError::Inference(error.to_string()))?
                            .map_err(|error| SegmentError::Inference(error.to_string()))
                    },
                ),
            })
            .map_err(policy_error)
    }
}

fn load_pyannote_host(
    execution_services: &NativeExecutionServices,
    source: &PreparedSegmenterSource,
    expected_content_id: &str,
    peak_quote: u64,
    retained_quote: u64,
) -> Result<SharedPyannote, SegmentError> {
    let key = AuxiliaryRuntimeCacheKey::host_neutral::<PyannoteSegmenter>(
        PYANNOTE_GGML_ARCHITECTURE_ID,
        expected_content_id,
        PYANNOTE_HOST_REPRESENTATION,
    );
    execution_services
        .auxiliary_runtime_owners()
        .get_or_try_insert_admitted_with(
            key,
            retained_quote,
            || build_admitted_pyannote(source, expected_content_id, peak_quote, retained_quote),
            |error| SegmentError::LoadFailed(error.to_string()),
        )
}

fn load_pyannote_actor(
    execution_services: &NativeExecutionServices,
    source: &PreparedSegmenterSource,
    expected_content_id: &str,
    backend: GgmlCpuGraphBackend,
    placement: crate::device::execution_policy::ExecutionPlacement,
    peak_quote: u64,
) -> Result<PyannoteActor, SegmentError> {
    if source.preflight().runtime_source.content_id() != expected_content_id {
        return Err(content_changed(
            "PyanNet",
            expected_content_id,
            source.preflight().runtime_source.content_id(),
        ));
    }
    let representation = match placement {
        crate::device::execution_policy::ExecutionPlacement::FullDevice => {
            "pyannote-segmentation.full-device-ggml.v2"
        }
        crate::device::execution_policy::ExecutionPlacement::Hybrid => {
            "pyannote-segmentation.hybrid-recurrent-ggml.v2"
        }
        crate::device::execution_policy::ExecutionPlacement::CpuOnly => {
            return Err(SegmentError::LoadFailed(
                "PyanNet CPU candidate cannot construct a pinned GPU actor".into(),
            ));
        }
    };
    let retained_quote = PyannetGgmlRuntime::quoted_persistent_host_commitment_bytes(placement);
    let key = AuxiliaryPinnedRuntimeCacheKey::for_current_lane::<PyannetGgmlRuntime>(
        PYANNOTE_GGML_ARCHITECTURE_ID,
        expected_content_id,
        representation,
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
                            // DiariZen owns a persistent graph whose session is
                            // poisoned on abort/compute failure and rebuilt by
                            // `ensure_healthy_graph` on the next request. Keep
                            // the owner alive so that recovery path remains
                            // reachable. A typed candidate failure is still
                            // observed by `invoke_replay_safe`, which drops the
                            // entire runtime before changing lanes.
                            .call_mut({
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

    fn diarizen_accelerated_provider() -> crate::device::execution_route::ExecutionProvider {
        match std::env::var("OPENASR_DIARIZEN_BENCH_BACKEND")
            .expect("OPENASR_DIARIZEN_BENCH_BACKEND must select cuda or vulkan")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "cuda" => crate::device::execution_route::ExecutionProvider::Cuda,
            "vulkan" => crate::device::execution_route::ExecutionProvider::Vulkan,
            backend => {
                panic!("DiariZen accelerated stress accepts only cuda or vulkan, got {backend:?}")
            }
        }
    }

    fn diarizen_activity_sha256(activity: &LocalActivity) -> String {
        crate::testing::benchmark_sha256_bytes(
            activity
                .windows
                .iter()
                .map(|window| window.frame_activity.as_slice())
                .chain(std::iter::once(activity.speaker_count.as_slice())),
        )
    }

    fn diarizen_stress_samples(sample_count: usize) -> crate::PcmBuffer {
        crate::PcmBuffer::from_vec(
            (0..sample_count)
                .map(|index| {
                    let time = index as f32 / diarizen::DIARIZEN_SAMPLE_RATE_HZ as f32;
                    0.13 * (time * 173.0 * std::f32::consts::TAU).sin()
                        + 0.05 * (time * 421.0 * std::f32::consts::TAU + 0.23).cos()
                })
                .collect(),
        )
    }

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
    #[ignore = "host-local stress: needs OPENASR_DIARIZEN_PACK, OPENASR_DIARIZEN_BENCH_BACKEND, and the requested CUDA/Vulkan device"]
    fn diarizen_accelerated_concurrency_cancel_and_recover_when_pack_present() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Barrier, mpsc};
        use std::time::{Duration, Instant};

        let requested_provider = diarizen_accelerated_provider();
        let intent = ExecutionIntent::ConstrainedAcceleratedOnly(
            crate::device::execution_policy::AcceleratedDeviceConstraint::Provider(
                requested_provider,
            ),
        );
        let services = Arc::new(
            NativeExecutionServices::for_local_process().expect("native execution services"),
        );
        let prepared =
            super::super::pack::prepare_segmenter(crate::config::VoiceIdSegmenterPreference::Auto)
                .expect("prepare installed DiariZen pack");
        assert_eq!(prepared.provider, SegmenterProvider::DiariZen);
        let observations =
            crate::models::native_execution_services::ExecutionObservationSink::new();
        let runtime = {
            let _observation_guard =
                crate::models::native_execution_services::install_execution_observation_sink(
                    observations.clone(),
                );
            Arc::new(
                PolicyResolvedSegmenterRuntime::load_prepared(services, intent, prepared)
                    .expect("load accelerated DiariZen runtime"),
            )
        };
        let observations = observations.observations();
        assert!(!observations.is_empty(), "DiariZen constructed no backend");
        let requested_route = &observations[0].requested_route;
        assert_eq!(requested_route.provider, requested_provider);
        assert!(
            observations.iter().all(|observation| {
                observation.requested_route == *requested_route
                    && observation.actual_provider == requested_provider
                    && observation.actual_stable_id == requested_route.stable_id
                    && observation.placement
                        == crate::device::execution_policy::ExecutionPlacement::FullDevice
                    && observation.backend_kind.is_gpu_class()
                    && !observation.use_scheduler
            }),
            "DiariZen did not remain on one direct FullDevice route: {observations:?}"
        );

        let short_samples = Arc::new(diarizen_stress_samples(diarizen::DIARIZEN_WINDOW_SAMPLES));
        let baseline = runtime
            .adapter()
            .segment_local_activity(short_samples.full_slice(), 16_000, &|| false, None)
            .expect("baseline DiariZen activity");
        let baseline_sha256 = diarizen_activity_sha256(&baseline);

        let barrier = Arc::new(Barrier::new(3));
        let workers = (0..2)
            .map(|_| {
                let runtime = Arc::clone(&runtime);
                let samples = Arc::clone(&short_samples);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    runtime
                        .adapter()
                        .segment_local_activity(samples.full_slice(), 16_000, &|| false, None)
                        .map(|activity| diarizen_activity_sha256(&activity))
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        for worker in workers {
            let concurrent_sha256 = worker
                .join()
                .expect("concurrent DiariZen worker panicked")
                .expect("concurrent DiariZen activity");
            assert_eq!(concurrent_sha256, baseline_sha256);
        }

        let long_sample_count =
            diarizen::DIARIZEN_WINDOW_SAMPLES + 63 * diarizen::DIARIZEN_WINDOW_STEP_SAMPLES;
        let long_samples = Arc::new(diarizen_stress_samples(long_sample_count));
        let cancel = Arc::new(AtomicBool::new(false));
        let telemetry = crate::GgmlExecutionTelemetryCollector::new();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let cancel_worker = {
            let runtime = Arc::clone(&runtime);
            let samples = Arc::clone(&long_samples);
            let cancel = Arc::clone(&cancel);
            let telemetry = telemetry.clone();
            std::thread::spawn(move || {
                let _telemetry_guard = telemetry.install();
                let previous =
                    crate::ggml_runtime::arm_thread_job_cancel_flag(Some(Arc::clone(&cancel)));
                ready_tx.send(()).expect("cancel worker readiness");
                let result = runtime.adapter().segment_local_activity(
                    samples.full_slice(),
                    16_000,
                    &|| cancel.load(Ordering::Acquire),
                    None,
                );
                assert!(
                    crate::ggml_runtime::disarm_thread_job_cancel_flag_if_current(
                        &cancel, previous
                    )
                );
                result
            })
        };
        ready_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("cancel worker did not become ready");
        let deadline = Instant::now() + Duration::from_secs(10);
        while telemetry.snapshot().direct_graph_computes == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(2));
        }
        let observed_direct_graph_entries = telemetry.snapshot().direct_graph_computes;
        cancel.store(true, Ordering::Release);
        let canceled = cancel_worker
            .join()
            .expect("cancel DiariZen worker panicked");
        assert!(
            observed_direct_graph_entries > 0,
            "DiariZen cancel gate did not enter direct GPU graph compute"
        );
        assert!(
            matches!(canceled, Err(SegmentError::Canceled)),
            "DiariZen cancellation must remain typed, got {canceled:?}"
        );

        let recovered = runtime
            .adapter()
            .segment_local_activity(short_samples.full_slice(), 16_000, &|| false, None)
            .expect("DiariZen must recover after cancellation");
        let recovered_sha256 = diarizen_activity_sha256(&recovered);
        assert_eq!(recovered_sha256, baseline_sha256);
        eprintln!(
            "DIARIZEN_ACCELERATED_STRESS provider={} stable_id={} placement=FullDevice scheduler=false concurrent_requests=2 cancel_after_direct_graph_entries={observed_direct_graph_entries} recovery_sha256={recovered_sha256}",
            requested_provider.as_str(),
            observations[0].actual_stable_id,
        );
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
