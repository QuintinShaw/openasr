//! Request-scoped realtime Stream-VAD execution.
//!
//! CPU keeps the lightweight host implementation in the caller. Accelerated
//! candidates keep the stateful frontend/cache plus ggml runtime on one
//! dedicated owner thread: Metal backend objects are thread-confined and must
//! be constructed, used, and destroyed on that same thread. The process side
//! retains only an exclusive checkout handle.

use std::{
    fmt,
    sync::{Arc, Mutex},
};

use super::{
    FireRedStreamVadError, FireRedStreamingVad, frontend::FRAME_LENGTH,
    ggml_runtime::FireRedStreamVadGgmlRuntime,
};
use crate::device::{
    execution_policy::{ExecutionCandidate, ExecutionIntent, ExecutionPlacement},
    execution_route::enumerate_compute_devices_from_ggml,
};
use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::{
    admitted_pinned_runtime_actor_pool::{PinnedRuntimeActorCheckout, PinnedRuntimeActorError},
    native_execution_services::NativeExecutionServices,
    policy_resolved_aux_runtime::{
        AuxiliaryPinnedRuntimeCacheKey, PolicyResolvedAuxRuntime, PolicyResolvedAuxRuntimeError,
        PolicyResolvedStatefulAuxRuntime, resolved_runtime_for_auxiliary_candidate,
    },
    system_memory_owner::{
        SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
        SystemMemoryAllocationTransactionError, SystemMemoryOwner,
    },
};

const REALTIME_VAD_STAGE: &str = "firered-stream-vad-realtime-v1";
const REALTIME_VAD_CONTENT_ID: &str = "firered-stream-vad-embedded-v1";
const REALTIME_VAD_REPRESENTATION: &str = "firered-stream-vad.realtime-ggml.v1";

/// Conservative family-owned Rust capacity beside the three ggml metadata
/// contexts. Backend tensor/workspace buffers are admitted independently by
/// the shared ggml backend layer and are intentionally not double-counted.
const RUNTIME_RUST_RETAINED_BYTES: u64 = 1 << 20;
const RUNTIME_CONSTRUCTION_TRANSIENT_BYTES: u64 = 1 << 20;

type FireRedRealtimeVadActor =
    PinnedRuntimeActorCheckout<AuxiliaryPinnedRuntimeCacheKey, FireRedRealtimeVadRuntime>;

enum FireRedRealtimeVadCandidate {
    Host(Mutex<FireRedStreamingVad>),
    Accelerated(FireRedRealtimeVadActor),
}

impl FireRedRealtimeVadCandidate {
    fn accept_frame(&self, samples: &[i16]) -> Result<(f32, bool), FireRedStreamVadError> {
        match self {
            Self::Host(streaming) => streaming
                .lock()
                .map_err(|_| FireRedStreamVadError::RealtimeRuntime {
                    reason: "host realtime VAD state lock is poisoned".to_string(),
                })
                .map(|mut streaming| streaming.accept_frame_with_decision(samples)),
            Self::Accelerated(actor) => {
                let samples = samples.to_vec();
                actor
                    .call_mut_fallible(move |runtime| runtime.accept_frame(&samples))
                    .map_err(map_actor_error)?
            }
        }
    }

    fn reset(&self) -> Result<(), FireRedStreamVadError> {
        match self {
            Self::Host(streaming) => {
                streaming
                    .lock()
                    .map_err(|_| FireRedStreamVadError::RealtimeRuntime {
                        reason: "host realtime VAD state lock is poisoned".to_string(),
                    })?
                    .reset();
                Ok(())
            }
            Self::Accelerated(actor) => actor
                .call_mut(|runtime| runtime.reset())
                .map_err(map_actor_error),
        }
    }

    #[cfg(test)]
    fn actor_identity_for_test(&self) -> Option<usize> {
        match self {
            Self::Host(_) => None,
            Self::Accelerated(actor) => actor
                .call_mut(|runtime| runtime as *mut FireRedRealtimeVadRuntime as usize)
                .ok(),
        }
    }
}

/// One realtime neural-VAD lane with request-local execution policy.
///
/// A typed failure may advance Auto before any successful frame. After the
/// first successful VAD decision, the lane is pinned because frontend/cache
/// state has become externally observable and cannot be replayed safely.
pub(crate) struct FireRedRealtimeVadSession {
    runtime: FireRedRealtimeVadSessionRuntime,
    expected_frame_samples: usize,
    precommit_frames: Vec<Vec<i16>>,
}

impl fmt::Debug for FireRedRealtimeVadSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let route = match &self.runtime {
            FireRedRealtimeVadSessionRuntime::Host(_) => "host",
            FireRedRealtimeVadSessionRuntime::Policy(_) => "policy-resolved",
        };
        formatter
            .debug_struct("FireRedRealtimeVadSession")
            .field("route", &route)
            .finish_non_exhaustive()
    }
}

enum FireRedRealtimeVadSessionRuntime {
    Host(FireRedStreamingVad),
    Policy(
        Box<PolicyResolvedStatefulAuxRuntime<FireRedRealtimeVadCandidate, FireRedStreamVadError>>,
    ),
}

impl FireRedRealtimeVadSession {
    pub(crate) fn host(frame_samples: usize) -> Result<Self, FireRedStreamVadError> {
        validate_frame_samples(frame_samples)?;
        let streaming = FireRedStreamingVad::shared().ok_or_else(|| {
            FireRedStreamVadError::RealtimeRuntime {
                reason: "vendored Stream-VAD weights failed to parse".to_string(),
            }
        })?;
        Ok(Self {
            runtime: FireRedRealtimeVadSessionRuntime::Host(streaming),
            expected_frame_samples: frame_samples,
            precommit_frames: Vec::new(),
        })
    }

    pub(crate) fn for_execution(
        execution_services: Arc<NativeExecutionServices>,
        execution_intent: ExecutionIntent,
        frame_samples: usize,
    ) -> Result<Self, FireRedStreamVadError> {
        if matches!(&execution_intent, ExecutionIntent::CpuOnly)
            || (matches!(&execution_intent, ExecutionIntent::Auto)
                && matches!(
                    super::AUTO_GPU_POLICY,
                    crate::ggml_runtime::AutoGpuPolicy::Never
                ))
        {
            return Self::host(frame_samples);
        }
        validate_frame_samples(frame_samples)?;
        let inventory = enumerate_compute_devices_from_ggml(&crate::ggml_available_devices());
        let execution_plan = execution_services
            .policy_resolver()
            .resolve(
                execution_intent,
                super::AUTO_GPU_POLICY,
                super::execution_capabilities(),
                &inventory,
            )
            .map_err(|error| FireRedStreamVadError::ExecutionPolicy {
                reason: error.to_string(),
            })?;
        let services_for_builder = Arc::clone(&execution_services);
        let builder = Arc::new(move |candidate: &ExecutionCandidate| {
            build_candidate(Arc::clone(&services_for_builder), candidate, frame_samples)
        });
        let runtime = PolicyResolvedAuxRuntime::try_new(
            execution_services,
            execution_plan,
            REALTIME_VAD_STAGE,
            builder,
        )
        .map_err(map_policy_error)?;
        Ok(Self {
            runtime: FireRedRealtimeVadSessionRuntime::Policy(Box::new(
                PolicyResolvedStatefulAuxRuntime::new(runtime),
            )),
            expected_frame_samples: frame_samples,
            precommit_frames: Vec::new(),
        })
    }

    pub(crate) fn accept_frame(&mut self, samples: &[i16]) -> Result<f32, FireRedStreamVadError> {
        if samples.len() != self.expected_frame_samples {
            return Err(FireRedStreamVadError::RealtimeRuntime {
                reason: format!(
                    "realtime VAD frame has {} samples, expected {}",
                    samples.len(),
                    self.expected_frame_samples
                ),
            });
        }
        match &mut self.runtime {
            FireRedRealtimeVadSessionRuntime::Host(streaming) => {
                Ok(streaming.accept_frame(samples))
            }
            FireRedRealtimeVadSessionRuntime::Policy(runtime) => {
                invoke_frame_with_precommit_replay(
                    runtime,
                    &mut self.precommit_frames,
                    samples,
                    FireRedRealtimeVadCandidate::accept_frame,
                    FireRedRealtimeVadCandidate::reset,
                )
                .map_err(map_policy_error)
            }
        }
    }

    pub(crate) fn reset(&mut self) -> Result<(), FireRedStreamVadError> {
        match &mut self.runtime {
            FireRedRealtimeVadSessionRuntime::Host(streaming) => {
                streaming.reset();
                self.precommit_frames.clear();
                Ok(())
            }
            FireRedRealtimeVadSessionRuntime::Policy(runtime) => {
                let result = runtime.invoke_with_commit(|candidate| {
                    candidate.reset()?;
                    Ok(((), false))
                });
                if result.is_ok() {
                    self.precommit_frames.clear();
                }
                result.map_err(map_policy_error)
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn actor_identity_for_test(&self) -> Option<usize> {
        match &self.runtime {
            FireRedRealtimeVadSessionRuntime::Host(_) => None,
            FireRedRealtimeVadSessionRuntime::Policy(runtime) => runtime
                .runtime_for_test()
                .and_then(FireRedRealtimeVadCandidate::actor_identity_for_test),
        }
    }
}

fn invoke_frame_with_precommit_replay<R, E>(
    runtime: &mut PolicyResolvedStatefulAuxRuntime<R, E>,
    precommit_frames: &mut Vec<Vec<i16>>,
    samples: &[i16],
    mut accept_frame: impl FnMut(&R, &[i16]) -> Result<(f32, bool), E>,
    mut reset: impl FnMut(&R) -> Result<(), E>,
) -> Result<f32, PolicyResolvedAuxRuntimeError<E>> {
    precommit_frames.push(samples.to_vec());
    let current_frame = precommit_frames
        .last()
        .expect("the current realtime frame was buffered")
        .clone();
    let replay_frames = precommit_frames.clone();
    let mut invocation = 0_usize;
    let result = runtime.invoke_with_commit(|candidate| {
        invocation = invocation.saturating_add(1);
        if invocation == 1 {
            return accept_frame(candidate, &current_frame);
        }

        // A typed failure before the first decision may switch lanes. The
        // replacement candidate has no preceding frontend/cache state, so
        // replay every frame buffered since session/reset rather than only
        // the frame that happened to trigger the first graph compute.
        reset(candidate)?;
        let mut final_probability = 0.0_f32;
        let mut produced_decision = false;
        for frame in &replay_frames {
            let (probability, produced) = accept_frame(candidate, frame)?;
            final_probability = probability;
            produced_decision |= produced;
        }
        Ok((final_probability, produced_decision))
    });
    if runtime.output_committed() {
        precommit_frames.clear();
    }
    result
}

fn validate_frame_samples(frame_samples: usize) -> Result<(), FireRedStreamVadError> {
    if frame_samples == 0 {
        return Err(FireRedStreamVadError::RealtimeRuntime {
            reason: "realtime VAD frame must contain at least one sample".to_string(),
        });
    }
    Ok(())
}

fn build_candidate(
    execution_services: Arc<NativeExecutionServices>,
    candidate: &ExecutionCandidate,
    frame_samples: usize,
) -> Result<FireRedRealtimeVadCandidate, FireRedStreamVadError> {
    let model = super::shared_model().ok_or_else(|| FireRedStreamVadError::RealtimeRuntime {
        reason: "vendored Stream-VAD weights failed to parse".to_string(),
    })?;
    let streaming = FireRedStreamingVad::from_model(model);
    if candidate.placement == ExecutionPlacement::CpuOnly {
        return Ok(FireRedRealtimeVadCandidate::Host(Mutex::new(streaming)));
    }

    let backend = resolved_runtime_for_auxiliary_candidate(candidate).backend();
    if backend == GgmlCpuGraphBackend::Cpu {
        return Err(FireRedStreamVadError::ExecutionPolicy {
            reason: "accelerated realtime VAD candidate resolved to CPU".to_string(),
        });
    }
    let key = realtime_actor_cache_key(backend, frame_samples);
    let placement = candidate.placement;
    let checkout = execution_services
        .firered_stream_vad_realtime_actors()
        .checkout_or_try_build_with(
            key,
            move || {
                let quote = FireRedRealtimeVadRuntime::system_memory_quote(frame_samples)?;
                Ok((quote.retained_bytes, quote))
            },
            move |quote| allocate_runtime_owner(quote, backend, placement),
            map_actor_error,
        )?;

    // Every checkout, including an idle reused actor, is reset and warmed for
    // this session's actual input cadence. The enclosing candidate attempt
    // observes the warm graph's real backend before user audio is accepted.
    checkout
        .call_mut_fallible(move |runtime| {
            runtime.reset();
            runtime.warm_for_frame_samples(frame_samples)?;
            runtime.reset();
            Ok::<(), FireRedStreamVadError>(())
        })
        .map_err(map_actor_error)??;
    Ok(FireRedRealtimeVadCandidate::Accelerated(checkout))
}

fn realtime_actor_cache_key(
    backend: GgmlCpuGraphBackend,
    frame_samples: usize,
) -> AuxiliaryPinnedRuntimeCacheKey {
    AuxiliaryPinnedRuntimeCacheKey::for_current_lane::<FireRedRealtimeVadRuntime>(
        REALTIME_VAD_STAGE,
        format!("{REALTIME_VAD_CONTENT_ID}:frame-samples={frame_samples}"),
        REALTIME_VAD_REPRESENTATION,
        backend,
    )
}

fn allocate_runtime_owner(
    quote: SystemMemoryAllocationQuote,
    backend: GgmlCpuGraphBackend,
    placement: ExecutionPlacement,
) -> Result<SystemMemoryOwner<FireRedRealtimeVadRuntime>, FireRedStreamVadError> {
    match SystemMemoryOwner::try_allocate_transaction(quote.clone(), || {
        let runtime = FireRedRealtimeVadRuntime::new(backend, placement)?;
        Ok::<_, FireRedStreamVadError>(SystemMemoryAllocationOutcome::new(
            runtime,
            quote.peak_bytes,
            quote.retained_bytes,
        ))
    }) {
        Ok(owner) => Ok(owner),
        Err(SystemMemoryAllocationTransactionError::Allocation(error)) => Err(error),
        Err(SystemMemoryAllocationTransactionError::Capacity(error)) => {
            Err(FireRedStreamVadError::ExecutionPolicy {
                reason: error.to_string(),
            })
        }
    }
}

fn map_actor_error(error: PinnedRuntimeActorError) -> FireRedStreamVadError {
    FireRedStreamVadError::RealtimeRuntime {
        reason: error.to_string(),
    }
}

fn map_policy_error(
    error: PolicyResolvedAuxRuntimeError<FireRedStreamVadError>,
) -> FireRedStreamVadError {
    match error {
        PolicyResolvedAuxRuntimeError::Operation(error) => error,
        other => FireRedStreamVadError::ExecutionPolicy {
            reason: other.to_string(),
        },
    }
}

/// Owner-thread state for accelerated realtime VAD.
pub(crate) struct FireRedRealtimeVadRuntime {
    streaming: FireRedStreamingVad,
    device: FireRedStreamVadGgmlRuntime,
}

impl FireRedRealtimeVadRuntime {
    fn new(
        backend: GgmlCpuGraphBackend,
        placement: ExecutionPlacement,
    ) -> Result<Self, FireRedStreamVadError> {
        let model = super::shared_model().ok_or(FireRedStreamVadError::RealtimeRuntime {
            reason: "vendored Stream-VAD weights failed to parse".to_string(),
        })?;
        let device =
            FireRedStreamVadGgmlRuntime::new(model, backend, placement).map_err(|error| {
                FireRedStreamVadError::Graph {
                    reason: error.to_string(),
                }
            })?;
        Ok(Self {
            streaming: FireRedStreamingVad::from_model(model),
            device,
        })
    }

    fn accept_frame(&mut self, samples: &[i16]) -> Result<(f32, bool), FireRedStreamVadError> {
        let float_samples = samples
            .iter()
            .map(|sample| *sample as f32 / 32_768.0)
            .collect::<Vec<_>>();
        let device = &mut self.device;
        let probabilities =
            self.streaming
                .accept_f32_chunk_with(&float_samples, |features, frames, cache| {
                    device
                        .forward_chunk(features, frames, cache)
                        .map_err(|error| FireRedStreamVadError::Graph {
                            reason: error.to_string(),
                        })
                })?;
        Ok((self.streaming.last_probability(), !probabilities.is_empty()))
    }

    fn reset(&mut self) {
        self.streaming.reset();
    }

    fn warm_for_frame_samples(
        &mut self,
        frame_samples: usize,
    ) -> Result<(), FireRedStreamVadError> {
        if frame_samples == 0 {
            return Err(FireRedStreamVadError::RealtimeRuntime {
                reason: "realtime VAD frame must contain at least one sample".to_string(),
            });
        }
        let frames_until_first_compute = FRAME_LENGTH.div_ceil(frame_samples);
        let silent = vec![0_i16; frame_samples];
        for _ in 0..=frames_until_first_compute {
            let _ = self.accept_frame(&silent)?;
        }
        Ok(())
    }

    fn system_memory_quote(
        frame_samples: usize,
    ) -> Result<SystemMemoryAllocationQuote, FireRedStreamVadError> {
        // Every ggml metadata context now owns its own shared-layer
        // SystemMemory lease. This family quote covers only Rust containers
        // and construction/front-end transients, avoiding double admission.
        let retained = RUNTIME_RUST_RETAINED_BYTES;
        let warm_samples = frame_samples
            .checked_mul(FRAME_LENGTH.div_ceil(frame_samples).saturating_add(1))
            .ok_or_else(|| FireRedStreamVadError::RealtimeRuntime {
                reason: "realtime VAD warm-up sample count overflowed".to_string(),
            })?;
        let model = super::shared_model().ok_or(FireRedStreamVadError::RealtimeRuntime {
            reason: "vendored Stream-VAD weights failed to parse".to_string(),
        })?;
        let frontend_peak = model.quoted_streaming_chunk_peak_bytes(warm_samples);
        let peak = retained
            .checked_add(frontend_peak)
            .and_then(|bytes| bytes.checked_add(RUNTIME_CONSTRUCTION_TRANSIENT_BYTES))
            .ok_or_else(|| FireRedStreamVadError::RealtimeRuntime {
                reason: "realtime VAD peak memory quote overflowed".to_string(),
            })?;
        SystemMemoryAllocationQuote::new(
            format!("aux.firered-stream-vad.realtime-runtime.{frame_samples}"),
            peak,
            retained,
        )
        .map_err(|error| FireRedStreamVadError::RealtimeRuntime {
            reason: error.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::device::{
        execution_memory::{DeviceMemoryBrokerSet, DeviceMemoryPolicy},
        execution_policy::{
            DefaultExecutionPolicyResolver, ExecutionCandidateFailure, ExecutionDeviceSnapshot,
            ExecutionPlan,
        },
        execution_route::{
            DeviceAddressability, ExecutionProvider, ResolvedExecutionRoute, RouteDeviceKind,
        },
    };
    use crate::ggml_runtime::GgmlBackendKind;
    use crate::models::native_execution_services::record_current_execution_candidate_failure;

    use super::*;

    fn candidate(provider: ExecutionProvider, stable_id: &str) -> ExecutionCandidate {
        ExecutionCandidate {
            device: ExecutionDeviceSnapshot {
                route: ResolvedExecutionRoute {
                    provider,
                    stable_id: stable_id.to_string(),
                    registry_ordinal: 0,
                    kind: if provider == ExecutionProvider::Cpu {
                        RouteDeviceKind::Cpu
                    } else {
                        RouteDeviceKind::Accelerated
                    },
                    addressability: DeviceAddressability::NotExactlyAddressable {
                        reason: "synthetic realtime VAD replay route",
                    },
                },
                ggml_kind: if provider == ExecutionProvider::Cpu {
                    GgmlBackendKind::Cpu
                } else {
                    GgmlBackendKind::Gpu
                },
                memory: None,
                buffer_alignment: None,
            },
            placement: if provider == ExecutionProvider::Cpu {
                ExecutionPlacement::CpuOnly
            } else {
                ExecutionPlacement::FullDevice
            },
        }
    }

    #[test]
    fn actor_cache_identity_includes_realtime_frame_geometry() {
        let ten_ms = realtime_actor_cache_key(GgmlCpuGraphBackend::Metal, 160);
        let twenty_ms = realtime_actor_cache_key(GgmlCpuGraphBackend::Metal, 320);
        let thirty_ms = realtime_actor_cache_key(GgmlCpuGraphBackend::Metal, 480);

        assert_ne!(ten_ms, twenty_ms);
        assert_ne!(twenty_ms, thirty_ms);
        assert_ne!(ten_ms, thirty_ms);
    }

    #[test]
    fn first_decision_fallback_replays_every_buffered_pcm_frame() {
        struct FakeLane {
            provider: ExecutionProvider,
            samples: Mutex<Vec<i16>>,
        }

        let services = Arc::new(
            NativeExecutionServices::new_with_broker(
                Arc::new(DefaultExecutionPolicyResolver),
                Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default())),
            )
            .unwrap(),
        );
        let plan = ExecutionPlan::for_test(
            ExecutionIntent::Auto,
            vec![
                candidate(ExecutionProvider::Vulkan, "gpu-0"),
                candidate(ExecutionProvider::Cpu, "cpu"),
            ],
        );
        let runtime = PolicyResolvedAuxRuntime::try_new(
            services,
            plan,
            "test-realtime-vad-precommit-replay",
            Arc::new(|candidate: &ExecutionCandidate| {
                Ok::<_, &'static str>(FakeLane {
                    provider: candidate.device.route.provider,
                    samples: Mutex::new(Vec::new()),
                })
            }),
        )
        .unwrap();
        let mut runtime = PolicyResolvedStatefulAuxRuntime::new(runtime);
        let mut precommit = Vec::new();
        let accept = |lane: &FakeLane, samples: &[i16]| {
            let mut buffered = lane.samples.lock().unwrap();
            buffered.extend_from_slice(samples);
            if lane.provider == ExecutionProvider::Vulkan && buffered.len() >= 480 {
                record_current_execution_candidate_failure(ExecutionCandidateFailure::device_lost(
                    "test-first-vad-decision",
                    "device lost after buffering two earlier frames",
                ));
                return Err("gpu lost");
            }
            let produced = buffered.len() >= 480;
            Ok((buffered.len() as f32, produced))
        };
        let reset = |lane: &FakeLane| {
            lane.samples.lock().unwrap().clear();
            Ok::<_, &'static str>(())
        };

        for value in [1_i16, 2] {
            let probability = invoke_frame_with_precommit_replay(
                &mut runtime,
                &mut precommit,
                &vec![value; 160],
                accept,
                reset,
            )
            .unwrap();
            assert_eq!(probability, if value == 1 { 160.0 } else { 320.0 });
            assert!(!runtime.output_committed());
        }
        let probability = invoke_frame_with_precommit_replay(
            &mut runtime,
            &mut precommit,
            &vec![3_i16; 160],
            accept,
            reset,
        )
        .unwrap();
        assert_eq!(probability, 480.0);
        assert!(runtime.output_committed());
        assert!(precommit.is_empty());
        let recovered = runtime.runtime_for_test().unwrap();
        assert_eq!(recovered.provider, ExecutionProvider::Cpu);
        let samples = recovered.samples.lock().unwrap();
        assert_eq!(&samples[..160], vec![1_i16; 160]);
        assert_eq!(&samples[160..320], vec![2_i16; 160]);
        assert_eq!(&samples[320..], vec![3_i16; 160]);
    }
}
