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
    execution_route::{ExecutionProvider, enumerate_compute_devices_from_ggml},
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
    Host(Box<Mutex<FireRedStreamingVad>>),
    Accelerated(Box<FireRedRealtimeVadActor>),
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
    _request_receipt_owner: Option<super::RuntimeOwnerGuard>,
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
    Host(Box<FireRedStreamingVad>),
    Policy(
        Box<PolicyResolvedStatefulAuxRuntime<FireRedRealtimeVadCandidate, FireRedStreamVadError>>,
    ),
}

impl FireRedRealtimeVadSession {
    pub(crate) fn host(frame_samples: usize) -> Result<Self, FireRedStreamVadError> {
        let execution_services = Arc::new(NativeExecutionServices::for_local_process().map_err(
            |error| FireRedStreamVadError::RealtimeRuntime {
                reason: format!("construct native execution services: {error}"),
            },
        )?);
        Self::host_with_services(execution_services, frame_samples)
    }

    fn host_with_services(
        execution_services: Arc<NativeExecutionServices>,
        frame_samples: usize,
    ) -> Result<Self, FireRedStreamVadError> {
        let _execution_scope =
            crate::models::native_execution_services::install_native_execution_services(
                &execution_services,
            );
        validate_frame_samples(frame_samples)?;
        let streaming = FireRedStreamingVad::shared().ok_or_else(|| {
            FireRedStreamVadError::RealtimeRuntime {
                reason: "vendored Stream-VAD weights failed to parse".to_string(),
            }
        })?;
        Ok(Self {
            runtime: FireRedRealtimeVadSessionRuntime::Host(Box::new(streaming)),
            expected_frame_samples: frame_samples,
            precommit_frames: Vec::new(),
            _request_receipt_owner: super::receipt_owner(
                "firered-stream-vad.realtime.request",
                Some(&format!("frame-samples={frame_samples}")),
                Some("request"),
            ),
        })
    }

    pub(crate) fn for_execution(
        execution_services: Arc<NativeExecutionServices>,
        execution_intent: ExecutionIntent,
        frame_samples: usize,
    ) -> Result<Self, FireRedStreamVadError> {
        let _execution_scope =
            crate::models::native_execution_services::install_native_execution_services(
                &execution_services,
            );
        if matches!(&execution_intent, ExecutionIntent::CpuOnly)
            || (matches!(&execution_intent, ExecutionIntent::Auto)
                && matches!(
                    super::AUTO_GPU_POLICY,
                    crate::ggml_runtime::AutoGpuPolicy::Never
                ))
        {
            return Self::host_with_services(execution_services, frame_samples);
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
        let activation_quote =
            crate::models::native_execution_services::CandidateActivationQuoteSource::Declared(
                super::FireRedStreamVadModel::system_memory_quote()
                    .map_err(|reason| FireRedStreamVadError::ExecutionPolicy { reason })?,
            );
        let runtime = PolicyResolvedAuxRuntime::try_new(
            execution_services,
            execution_plan,
            REALTIME_VAD_STAGE,
            builder,
            activation_quote,
        )
        .map_err(map_policy_error)?;
        Ok(Self {
            runtime: FireRedRealtimeVadSessionRuntime::Policy(Box::new(
                PolicyResolvedStatefulAuxRuntime::new(runtime),
            )),
            expected_frame_samples: frame_samples,
            precommit_frames: Vec::new(),
            _request_receipt_owner: super::receipt_owner(
                "firered-stream-vad.realtime.request",
                Some(&format!("frame-samples={frame_samples}")),
                Some("request"),
            ),
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
    if candidate.placement == ExecutionPlacement::CpuOnly {
        let streaming = FireRedStreamingVad::shared().ok_or_else(|| {
            FireRedStreamVadError::RealtimeRuntime {
                reason: "vendored Stream-VAD weights failed to parse".to_string(),
            }
        })?;
        return Ok(FireRedRealtimeVadCandidate::Host(Box::new(Mutex::new(
            streaming,
        ))));
    }

    let backend = resolved_runtime_for_auxiliary_candidate(candidate).backend();
    if backend == GgmlCpuGraphBackend::Cpu {
        return Err(FireRedStreamVadError::ExecutionPolicy {
            reason: "accelerated realtime VAD candidate resolved to CPU".to_string(),
        });
    }
    let key = realtime_actor_cache_key(backend, frame_samples);
    let placement = candidate.placement;
    let stable_id = candidate.device.route.stable_id.clone();
    let provider = candidate.device.route.provider;
    let checkout = execution_services
        .firered_stream_vad_realtime_actors()
        .checkout_or_try_build_with(
            key,
            move || {
                let quote = FireRedRealtimeVadRuntime::system_memory_quote(frame_samples)?;
                Ok((quote.retained_bytes, quote))
            },
            move |quote| {
                allocate_runtime_owner(
                    quote,
                    backend,
                    placement,
                    frame_samples,
                    stable_id.clone(),
                    provider,
                )
            },
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
    Ok(FireRedRealtimeVadCandidate::Accelerated(Box::new(checkout)))
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
    frame_samples: usize,
    stable_device_id: String,
    provider: ExecutionProvider,
) -> Result<SystemMemoryOwner<FireRedRealtimeVadRuntime>, FireRedStreamVadError> {
    match SystemMemoryOwner::try_allocate_transaction(quote.clone(), || {
        let runtime = FireRedRealtimeVadRuntime::new(
            backend,
            placement,
            frame_samples,
            &stable_device_id,
            provider,
        )?;
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
    _receipt_owner: Option<super::RuntimeOwnerGuard>,
}

impl FireRedRealtimeVadRuntime {
    fn new(
        backend: GgmlCpuGraphBackend,
        placement: ExecutionPlacement,
        frame_samples: usize,
        stable_device_id: &str,
        provider: ExecutionProvider,
    ) -> Result<Self, FireRedStreamVadError> {
        let model = super::shared_model().ok_or(FireRedStreamVadError::RealtimeRuntime {
            reason: "vendored Stream-VAD weights failed to parse".to_string(),
        })?;
        let device =
            FireRedStreamVadGgmlRuntime::new(&model, backend, placement).map_err(|error| {
                FireRedStreamVadError::Graph {
                    reason: error.to_string(),
                }
            })?;
        let content = format!("firered-stream-vad-embedded-v1:frame-samples={frame_samples}");
        let receipt_owner = crate::models::native_execution_services::current_runtime_receipts()
            .filter(|collector| collector.is_available())
            .and_then(|collector| {
                let lane =
                    collector.lane_projection(provider, stable_device_id, placement, backend)?;
                let descriptor = collector.owner_descriptor(
                    "firered-stream-vad.realtime.actor-runtime",
                    Some(&content),
                    Some("accelerated actor checkout"),
                    Some(lane),
                )?;
                Some(collector.start_owner(
                    descriptor,
                    crate::models::native_execution_services::current_execution_cache_attempt_id(),
                ))
            });
        Ok(Self {
            streaming: FireRedStreamingVad::from_model(model).map_err(|error| {
                FireRedStreamVadError::RealtimeRuntime {
                    reason: error.to_string(),
                }
            })?,
            device,
            _receipt_owner: receipt_owner,
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
        let _warmup_receipt_owner = super::receipt_owner(
            "firered-stream-vad.realtime.warmup",
            Some(&format!("frame-samples={frame_samples}")),
            Some("pre-output candidate validation"),
        );
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
    fn cpu_session_receipts_cover_request_frontend_cache_and_release() {
        let services = Arc::new(
            NativeExecutionServices::for_local_process()
                .expect("construct native execution services"),
        );
        let _scope =
            crate::models::native_execution_services::install_native_execution_services(&services);
        let session = FireRedRealtimeVadSession::for_execution(
            Arc::clone(&services),
            ExecutionIntent::CpuOnly,
            160,
        )
        .expect("embedded Stream-VAD model");
        let collector = services.runtime_receipts();
        let embedded_source = collector
            .host_neutral_owner_descriptor(
                "system-memory-owner",
                None,
                Some("firered-stream-vad.embedded-model"),
            )
            .unwrap()
            .source;
        let session_source = collector
            .host_neutral_owner_descriptor(
                "system-memory-owner",
                None,
                Some("firered-stream-vad.host-session"),
            )
            .unwrap()
            .source;
        let request_component = collector
            .host_neutral_owner_descriptor(
                "firered-stream-vad.realtime.request",
                Some("frame-samples=160"),
                Some("request"),
            )
            .unwrap()
            .component;
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.live_owners.len(), 3);
        assert!(
            snapshot
                .live_owners
                .iter()
                .any(|owner| owner.descriptor.source == embedded_source)
        );
        assert!(
            snapshot
                .live_owners
                .iter()
                .any(|owner| owner.descriptor.source == session_source)
        );
        assert!(
            snapshot
                .live_owners
                .iter()
                .any(|owner| owner.descriptor.component == request_component)
        );
        assert!(snapshot.events.iter().any(|event| matches!(
            event,
            crate::models::runtime_receipts::RuntimeReceiptEvent::OwnerCreated { .. }
        )));
        assert_eq!(
            collector.reconcile_live_leases(services.memory_broker()),
            crate::models::runtime_receipts::LeaseReceiptShadow::Matched
        );
        drop(session);
        assert_eq!(services.runtime_receipts().summary().live_owner_count, 1);
        services.unload_idle_native_model_runtime_caches();
        assert_eq!(services.runtime_receipts().summary().live_owner_count, 0);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn accelerated_candidate_build_has_one_actor_and_brokered_runtime_receipt() {
        let services = Arc::new(
            NativeExecutionServices::for_local_process()
                .expect("construct native execution services"),
        );
        let _scope =
            crate::models::native_execution_services::install_native_execution_services(&services);
        let actor_component = services
            .runtime_receipts()
            .owner_descriptor(
                "firered-stream-vad.realtime.actor-runtime",
                Some("firered-stream-vad-embedded-v1:frame-samples=160"),
                Some("accelerated actor checkout"),
                None,
            )
            .expect("actor receipt descriptor")
            .component;
        let runtime_source = services
            .runtime_receipts()
            .host_neutral_owner_descriptor(
                "system-memory-owner",
                None,
                Some("aux.firered-stream-vad.realtime-runtime.160"),
            )
            .expect("runtime system-memory receipt descriptor")
            .source;
        let session = FireRedRealtimeVadSession::for_execution(
            Arc::clone(&services),
            ExecutionIntent::AcceleratedOnly,
            160,
        )
        .expect("build accelerated Stream-VAD candidate");
        let snapshot = services.runtime_receipts().snapshot();
        let count_component = |component| {
            snapshot
                .events
                .iter()
                .filter(|event| {
                    matches!(
                        event,
                        crate::models::runtime_receipts::RuntimeReceiptEvent::OwnerCreated {
                            descriptor,
                            ..
                        } if descriptor.component == component
                    )
                })
                .count()
        };
        assert_eq!(count_component(actor_component), 1);
        assert_eq!(
            snapshot
                .live_owners
                .iter()
                .filter(|owner| owner.descriptor.source == runtime_source)
                .count(),
            1
        );
        assert_eq!(
            services
                .runtime_receipts()
                .reconcile_live_leases(services.memory_broker()),
            crate::models::runtime_receipts::LeaseReceiptShadow::Matched
        );
        drop(session);
        services.unload_idle_native_model_runtime_caches();
        assert_eq!(services.runtime_receipts().summary().live_owner_count, 0);
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
            crate::models::native_execution_services::CandidateActivationQuoteSource::Declared(
                crate::diarize::vad::FireRedStreamVadModel::system_memory_quote().unwrap(),
            ),
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
