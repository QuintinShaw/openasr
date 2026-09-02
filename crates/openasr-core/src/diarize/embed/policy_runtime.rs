//! Policy-resolved ownership for the ReDimNet speaker-embedding stage.
//!
//! Parsed f32 weights and frontend state are immutable, Send-safe host data and
//! live in the service-root admitted host cache. Every mutable ggml runtime is
//! constructed, used, and destroyed on a dedicated owner thread. A bounded
//! checkout pool provides up to four resident runtimes for batch parallelism;
//! no process-global or thread-local owner exists outside the injected service
//! root.

use std::ops::Deref;
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;

use crate::{
    NativeExecutionServices,
    device::execution_policy::ExecutionIntent,
    device::execution_route::ExecutionProvider,
    ggml_runtime::GgmlCpuGraphBackend,
    models::{
        admitted_pinned_runtime_actor_pool::{
            CheckedOutPinnedRuntimeActorCall, PinnedRuntimeActorError,
            call_checked_out_actor_mut_fallible_async,
        },
        aux_pack_registry::REDIMNET2_GGML_ARCHITECTURE_ID,
        aux_pack_registry::WESPEAKER_RESNET_ARCHITECTURE_ID,
        native_execution_services::current_execution_candidate_failure,
        policy_resolved_aux_runtime::{
            AuxiliaryPinnedRuntimeCacheKey, AuxiliaryRuntimeCacheKey, PolicyResolvedAuxRuntime,
            PolicyResolvedAuxRuntimeError, resolve_auxiliary_execution_plan,
            resolved_runtime_for_auxiliary_candidate,
        },
        runtime_receipts::{RuntimeOwnerDescriptor, RuntimeOwnerGuard},
        system_memory_owner::{
            AdmittedHostObject, SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
            SystemMemoryOwner,
        },
    },
};

use super::{
    EmbedError, REDIMNET_MAX_BATCH_WORKERS, RedimNet2Embedder, RedimNetResidentRuntime,
    SpeakerEmbedder, SpeakerEmbedderFamily, SpeakerEmbedderIdentity, SpeakerEmbeddingExecutionPlan,
    WeSpeakerEmbedder, WeSpeakerResidentRuntime, abort_successful_results_after_terminal_failure,
    cancel_requested,
    pack::{PreparedSelectedEmbedder, prepare_embedder},
    redimnet::config::EMBED_DIM,
    wespeaker,
};
use crate::config::VoiceIdEmbedderPreference;
use crate::diarize::{
    calibration::{REDIMNET_CALIBRATION, SpeakerCalibrationProfile, WESPEAKER_CALIBRATION},
    contract::SpeakerEmbedding,
    streaming::{StreamingDiarizer, StreamingSpeakerChangeDetector},
    voice_id::{EmbeddingSpace, PersonMatcher, load_person_matcher_for_embedder},
};

const STREAMING_SPEAKER_STAGE: &str = "redimnet2-streaming-speaker-stage-v1";
const WESPEAKER_STREAMING_SPEAKER_STAGE: &str = "wespeaker-streaming-speaker-stage-v1";
const WARMUP_SAMPLE_RATE_HZ: u32 = 16_000;
const WARMUP_SAMPLE_COUNT: usize = 40_000;
const REDIMNET_PARSED_HOST_REPRESENTATION: &str = "redimnet2.parsed-f32.v1";
const REDIMNET_RESIDENT_REPRESENTATION: &str = "redimnet2.ggml-resident.v1";
const WESPEAKER_PARSED_HOST_REPRESENTATION: &str = "wespeaker.parsed-f32.v1";
const WESPEAKER_RESIDENT_REPRESENTATION: &str = "wespeaker.ggml-resident.v1";

static REDIMNET_FRONTEND_POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();

fn redimnet_frontend_pool() -> &'static Result<rayon::ThreadPool, String> {
    REDIMNET_FRONTEND_POOL.get_or_init(|| {
        let workers = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
            .clamp(1, REDIMNET_MAX_BATCH_WORKERS);
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|index| format!("openasr-redimnet-frontend-{index}"))
            .build()
            .map_err(|error| error.to_string())
    })
}

fn redimnet_resident_worker_limit(backend: GgmlCpuGraphBackend, pool_threads: usize) -> usize {
    // The ignored Pareto harness deliberately sweeps the otherwise private
    // production limit. nextest process isolation keeps this test-only env
    // override from leaking into another request.
    #[cfg(test)]
    if std::env::var_os("OPENASR_REDIMNET_BENCH_WORKERS").is_some() {
        return super::redimnet_batch_worker_limit(pool_threads);
    }
    if backend.is_gpu_class() {
        1
    } else {
        super::redimnet_batch_worker_limit(pool_threads)
    }
}

type SharedAdmittedEmbedder = AdmittedHostObject<RedimNetParsedHost>;
type SharedAdmittedWeSpeaker = AdmittedHostObject<WeSpeakerParsedHost>;
type PendingActorBatch = CheckedOutPinnedRuntimeActorCall<
    AuxiliaryPinnedRuntimeCacheKey,
    RedimNetResidentRuntime,
    Result<Vec<(usize, Result<SpeakerEmbedding, EmbedError>)>, EmbedError>,
>;
type PendingWeSpeakerActorBatch = CheckedOutPinnedRuntimeActorCall<
    AuxiliaryPinnedRuntimeCacheKey,
    WeSpeakerResidentRuntime,
    Result<Vec<(usize, Result<SpeakerEmbedding, EmbedError>)>, EmbedError>,
>;

struct RedimNetParsedHost {
    embedder: RedimNet2Embedder,
    _receipt_owner: Option<RuntimeOwnerGuard>,
}

struct WeSpeakerParsedHost {
    embedder: WeSpeakerEmbedder,
    _receipt_owner: Option<RuntimeOwnerGuard>,
}

impl Deref for RedimNetParsedHost {
    type Target = RedimNet2Embedder;

    fn deref(&self) -> &Self::Target {
        &self.embedder
    }
}

impl Deref for WeSpeakerParsedHost {
    type Target = WeSpeakerEmbedder;

    fn deref(&self) -> &Self::Target {
        &self.embedder
    }
}

struct PolicySpeakerCandidate {
    parsed: SharedAdmittedEmbedder,
    services: Arc<NativeExecutionServices>,
    content_id: String,
    backend: GgmlCpuGraphBackend,
    placement: crate::device::execution_policy::ExecutionPlacement,
    provider: ExecutionProvider,
    stable_device_id: String,
}

fn redimnet_actor_key(
    content_id: impl Into<String>,
    backend: GgmlCpuGraphBackend,
) -> AuxiliaryPinnedRuntimeCacheKey {
    AuxiliaryPinnedRuntimeCacheKey::for_current_lane::<RedimNetResidentRuntime>(
        REDIMNET2_GGML_ARCHITECTURE_ID,
        content_id,
        REDIMNET_RESIDENT_REPRESENTATION,
        backend,
    )
}

fn redimnet_actor_receipt_descriptor(
    content_id: &str,
    backend: GgmlCpuGraphBackend,
    placement: crate::device::execution_policy::ExecutionPlacement,
    provider: ExecutionProvider,
    stable_device_id: &str,
) -> Option<RuntimeOwnerDescriptor> {
    let collector = crate::models::native_execution_services::current_runtime_receipts()?;
    let lane = collector.lane_projection(provider, stable_device_id, placement, backend)?;
    collector.owner_descriptor(
        "redimnet2.resident-runtime",
        Some(content_id),
        Some(REDIMNET_RESIDENT_REPRESENTATION),
        Some(lane),
    )
}

fn redimnet_parsed_host_receipt_owner(content_id: &str) -> Option<RuntimeOwnerGuard> {
    let collector = crate::models::native_execution_services::current_runtime_receipts()?;
    let descriptor = collector.host_neutral_owner_descriptor(
        "redimnet2.parsed-host-state",
        Some(content_id),
        Some(REDIMNET_PARSED_HOST_REPRESENTATION),
    )?;
    Some(collector.start_owner(
        descriptor,
        crate::models::native_execution_services::current_execution_cache_attempt_id(),
    ))
}

fn wespeaker_actor_key(
    content_id: impl Into<String>,
    backend: GgmlCpuGraphBackend,
) -> AuxiliaryPinnedRuntimeCacheKey {
    AuxiliaryPinnedRuntimeCacheKey::for_current_lane::<WeSpeakerResidentRuntime>(
        WESPEAKER_RESNET_ARCHITECTURE_ID,
        content_id,
        WESPEAKER_RESIDENT_REPRESENTATION,
        backend,
    )
}

fn wespeaker_actor_receipt_descriptor(
    content_id: &str,
    backend: GgmlCpuGraphBackend,
    placement: crate::device::execution_policy::ExecutionPlacement,
    provider: ExecutionProvider,
    stable_device_id: &str,
) -> Option<RuntimeOwnerDescriptor> {
    let collector = crate::models::native_execution_services::current_runtime_receipts()?;
    let lane = collector.lane_projection(provider, stable_device_id, placement, backend)?;
    collector.owner_descriptor(
        "wespeaker.resident-runtime",
        Some(content_id),
        Some(WESPEAKER_RESIDENT_REPRESENTATION),
        Some(lane),
    )
}

fn wespeaker_parsed_host_receipt_owner(content_id: &str) -> Option<RuntimeOwnerGuard> {
    let collector = crate::models::native_execution_services::current_runtime_receipts()?;
    let descriptor = collector.host_neutral_owner_descriptor(
        "wespeaker.parsed-host-state",
        Some(content_id),
        Some(WESPEAKER_PARSED_HOST_REPRESENTATION),
    )?;
    Some(collector.start_owner(
        descriptor,
        crate::models::native_execution_services::current_execution_cache_attempt_id(),
    ))
}

impl PolicySpeakerCandidate {
    fn actor_key(&self) -> AuxiliaryPinnedRuntimeCacheKey {
        redimnet_actor_key(self.content_id.clone(), self.backend)
    }

    fn checkout_actor(
        &self,
        threads: usize,
        warmup: Option<(Vec<f32>, usize)>,
    ) -> Result<
        crate::models::admitted_pinned_runtime_actor_pool::PinnedRuntimeActorCheckout<
            AuxiliaryPinnedRuntimeCacheKey,
            RedimNetResidentRuntime,
        >,
        EmbedError,
    > {
        let weights = self.parsed.shared_weights();
        let backend = self.backend;
        let placement = self.placement;
        let owner_descriptor = redimnet_actor_receipt_descriptor(
            &self.content_id,
            backend,
            placement,
            self.provider,
            &self.stable_device_id,
        );
        self.services
            .redimnet_runtime_actors()
            .checkout_or_try_build_with_owner_receipt(
                self.actor_key(),
                owner_descriptor,
                || Ok((0, (weights, threads, warmup))),
                move |(weights, threads, warmup)| {
                    let mut runtime =
                        RedimNetResidentRuntime::new(weights, Some(threads), backend, placement)
                            .map_err(map_backbone_error)?;
                    if let Some((features, frames)) = warmup {
                        runtime
                            .forward(&features, frames, Some(threads))
                            .map_err(map_backbone_error)?;
                    }
                    Ok(SystemMemoryOwner::without_allocation(runtime))
                },
                map_actor_error,
            )
    }

    fn embed(&self, samples: &[f32], sample_rate_hz: u32) -> Result<SpeakerEmbedding, EmbedError> {
        let (features, frames) = self
            .parsed
            .prepare_embedding_input(samples, sample_rate_hz)?;
        let threads = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        let actor = self.checkout_actor(threads, None)?;
        actor
            .call_mut_fallible(move |runtime| forward_one(runtime, features, frames, Some(threads)))
            .map_err(map_actor_error)??
    }

    fn embed_batch(
        &self,
        clips: &[&[f32]],
        sample_rate_hz: u32,
    ) -> Vec<Result<SpeakerEmbedding, EmbedError>> {
        if clips.is_empty() {
            return Vec::new();
        }
        let pool = match redimnet_frontend_pool() {
            Ok(pool) => pool,
            Err(reason) => {
                return repeat_error(
                    clips.len(),
                    EmbedError::Unavailable(format!(
                        "could not create bounded ReDimNet frontend pool: {reason}"
                    )),
                );
            }
        };
        let available = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        // A single full-device graph already saturates the integrated GPU;
        // duplicating resident weights/graphs across four actors multiplies
        // memory without introducing a batch dimension inside the model.
        // CPU retains the measured bounded worker fan-out.
        let max_workers = redimnet_resident_worker_limit(self.backend, pool.current_num_threads());
        let plan = SpeakerEmbeddingExecutionPlan::for_clips(clips.len(), available, max_workers);
        let inherited_cancel = crate::ggml_runtime::thread_job_cancel_flag();
        let prepared = pool.install(|| {
            clips
                .par_iter()
                .map(|samples| {
                    let _cancel = inherited_cancel
                        .as_ref()
                        .map(crate::ggml_runtime::InheritedJobCancelGuard::arm);
                    self.parsed.prepare_embedding_input(samples, sample_rate_hz)
                })
                .collect::<Vec<_>>()
        });

        let mut prepared = prepared.into_iter().map(Some).collect::<Vec<_>>();
        let mut results: Vec<Option<Result<SpeakerEmbedding, EmbedError>>> =
            std::iter::repeat_with(|| None).take(clips.len()).collect();
        let mut jobs = Vec::<PendingActorBatch>::with_capacity(plan.workers);
        for worker in 0..plan.workers {
            let range = plan.worker_range(worker, clips.len());
            let mut inputs = Vec::with_capacity(range.len());
            for index in range {
                match prepared[index]
                    .take()
                    .expect("each prepared ReDimNet input is consumed once")
                {
                    Ok((features, frames)) => inputs.push((index, features, frames)),
                    Err(error) => results[index] = Some(Err(error)),
                }
            }
            if inputs.is_empty() {
                continue;
            }
            let actor = match self.checkout_actor(plan.threads_per_runner, None) {
                Ok(actor) => actor,
                Err(error) => {
                    for (index, _, _) in inputs {
                        results[index] = Some(Err(clone_embed_error(&error)));
                    }
                    continue;
                }
            };
            let cancel = inherited_cancel.clone();
            match call_checked_out_actor_mut_fallible_async(actor, move |runtime| {
                let mut output = Vec::with_capacity(inputs.len());
                for (index, features, frames) in inputs {
                    let _cancel = cancel
                        .as_ref()
                        .map(crate::ggml_runtime::InheritedJobCancelGuard::arm);
                    if cancel.as_ref().is_some_and(cancel_requested) {
                        output.push((index, Err(EmbedError::Canceled)));
                        continue;
                    }
                    match forward_one(runtime, features, frames, Some(plan.threads_per_runner)) {
                        Ok(result) => output.push((index, result)),
                        Err(terminal) => return Err(terminal),
                    }
                }
                Ok(output)
            }) {
                Ok(job) => jobs.push(job),
                Err(error) => {
                    let error = map_actor_error(error);
                    for index in plan.worker_range(worker, clips.len()) {
                        if results[index].is_none() {
                            results[index] = Some(Err(clone_embed_error(&error)));
                        }
                    }
                }
            }
        }

        let terminal_failure = OnceLock::new();
        for job in jobs {
            match job.join().map_err(map_actor_error) {
                Ok(Ok(output)) => {
                    for (index, result) in output {
                        results[index] = Some(result);
                    }
                }
                Ok(Err(EmbedError::TerminalBackend(reason))) => {
                    let _ = terminal_failure.set(reason);
                }
                Ok(Err(error)) | Err(error) => {
                    let _ = terminal_failure.set(error.to_string());
                }
            }
        }

        let mut results = results
            .into_iter()
            .map(|result| {
                result.unwrap_or_else(|| {
                    Err(EmbedError::BatchAbortedAfterTerminalBackend(
                        terminal_failure
                            .get()
                            .cloned()
                            .unwrap_or_else(|| "ReDimNet actor returned no result".to_string()),
                    ))
                })
            })
            .collect::<Vec<_>>();
        if let Some(reason) = terminal_failure.get() {
            abort_successful_results_after_terminal_failure(&mut results, reason);
        }
        results
    }
}

struct PolicyWeSpeakerCandidate {
    parsed: SharedAdmittedWeSpeaker,
    services: Arc<NativeExecutionServices>,
    content_id: String,
    backend: GgmlCpuGraphBackend,
    placement: crate::device::execution_policy::ExecutionPlacement,
    provider: ExecutionProvider,
    stable_device_id: String,
}

impl PolicyWeSpeakerCandidate {
    fn actor_key(&self) -> AuxiliaryPinnedRuntimeCacheKey {
        wespeaker_actor_key(self.content_id.clone(), self.backend)
    }

    fn checkout_actor(
        &self,
        threads: usize,
        warmup: Option<(Vec<f32>, usize)>,
    ) -> Result<
        crate::models::admitted_pinned_runtime_actor_pool::PinnedRuntimeActorCheckout<
            AuxiliaryPinnedRuntimeCacheKey,
            WeSpeakerResidentRuntime,
        >,
        EmbedError,
    > {
        let weights = self.parsed.shared_weights();
        let config = self.parsed.config();
        let backend = self.backend;
        let placement = self.placement;
        let owner_descriptor = wespeaker_actor_receipt_descriptor(
            &self.content_id,
            backend,
            placement,
            self.provider,
            &self.stable_device_id,
        );
        self.services
            .wespeaker_runtime_actors()
            .checkout_or_try_build_with_owner_receipt(
                self.actor_key(),
                owner_descriptor,
                || Ok((0, (weights, config, threads, warmup))),
                move |(weights, config, threads, warmup)| {
                    let mut runtime = WeSpeakerResidentRuntime::new(
                        weights,
                        config,
                        Some(threads),
                        backend,
                        placement,
                    )
                    .map_err(map_wespeaker_backbone_error)?;
                    if let Some((features, frames)) = warmup {
                        runtime
                            .forward(&features, frames, Some(threads))
                            .map_err(map_wespeaker_backbone_error)?;
                    }
                    Ok(SystemMemoryOwner::without_allocation(runtime))
                },
                map_actor_error,
            )
    }

    fn embed(&self, samples: &[f32], sample_rate_hz: u32) -> Result<SpeakerEmbedding, EmbedError> {
        let (features, frames) = self
            .parsed
            .prepare_embedding_input(samples, sample_rate_hz)?;
        let threads = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        let actor = self.checkout_actor(threads, None)?;
        actor
            .call_mut_fallible(move |runtime| {
                wespeaker_forward_one(runtime, features, frames, Some(threads))
            })
            .map_err(map_actor_error)??
    }

    fn embed_batch(
        &self,
        clips: &[&[f32]],
        sample_rate_hz: u32,
    ) -> Vec<Result<SpeakerEmbedding, EmbedError>> {
        if clips.is_empty() {
            return Vec::new();
        }
        let pool = match redimnet_frontend_pool() {
            Ok(pool) => pool,
            Err(reason) => {
                return repeat_error(
                    clips.len(),
                    EmbedError::Unavailable(format!(
                        "could not create bounded WeSpeaker frontend pool: {reason}"
                    )),
                );
            }
        };
        let available = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        let max_workers = redimnet_resident_worker_limit(self.backend, pool.current_num_threads());
        let plan = SpeakerEmbeddingExecutionPlan::for_clips(clips.len(), available, max_workers);
        let inherited_cancel = crate::ggml_runtime::thread_job_cancel_flag();
        let prepared = pool.install(|| {
            clips
                .par_iter()
                .map(|samples| {
                    let _cancel = inherited_cancel
                        .as_ref()
                        .map(crate::ggml_runtime::InheritedJobCancelGuard::arm);
                    self.parsed.prepare_embedding_input(samples, sample_rate_hz)
                })
                .collect::<Vec<_>>()
        });

        let mut prepared = prepared.into_iter().map(Some).collect::<Vec<_>>();
        let mut results: Vec<Option<Result<SpeakerEmbedding, EmbedError>>> =
            std::iter::repeat_with(|| None).take(clips.len()).collect();
        let mut jobs = Vec::<PendingWeSpeakerActorBatch>::with_capacity(plan.workers);
        for worker in 0..plan.workers {
            let range = plan.worker_range(worker, clips.len());
            let mut inputs = Vec::with_capacity(range.len());
            for index in range {
                match prepared[index]
                    .take()
                    .expect("each prepared WeSpeaker input is consumed once")
                {
                    Ok((features, frames)) => inputs.push((index, features, frames)),
                    Err(error) => results[index] = Some(Err(error)),
                }
            }
            if inputs.is_empty() {
                continue;
            }
            let actor = match self.checkout_actor(plan.threads_per_runner, None) {
                Ok(actor) => actor,
                Err(error) => {
                    for (index, _, _) in inputs {
                        results[index] = Some(Err(clone_embed_error(&error)));
                    }
                    continue;
                }
            };
            let cancel = inherited_cancel.clone();
            match call_checked_out_actor_mut_fallible_async(actor, move |runtime| {
                let mut output = Vec::with_capacity(inputs.len());
                for (index, features, frames) in inputs {
                    let _cancel = cancel
                        .as_ref()
                        .map(crate::ggml_runtime::InheritedJobCancelGuard::arm);
                    if cancel.as_ref().is_some_and(cancel_requested) {
                        output.push((index, Err(EmbedError::Canceled)));
                        continue;
                    }
                    match wespeaker_forward_one(
                        runtime,
                        features,
                        frames,
                        Some(plan.threads_per_runner),
                    ) {
                        Ok(result) => output.push((index, result)),
                        Err(terminal) => return Err(terminal),
                    }
                }
                Ok(output)
            }) {
                Ok(job) => jobs.push(job),
                Err(error) => {
                    let error = map_actor_error(error);
                    for index in plan.worker_range(worker, clips.len()) {
                        if results[index].is_none() {
                            results[index] = Some(Err(clone_embed_error(&error)));
                        }
                    }
                }
            }
        }

        let terminal_failure = OnceLock::new();
        for job in jobs {
            match job.join().map_err(map_actor_error) {
                Ok(Ok(output)) => {
                    for (index, result) in output {
                        results[index] = Some(result);
                    }
                }
                Ok(Err(EmbedError::TerminalBackend(reason))) => {
                    let _ = terminal_failure.set(reason);
                }
                Ok(Err(error)) | Err(error) => {
                    let _ = terminal_failure.set(error.to_string());
                }
            }
        }

        let mut results = results
            .into_iter()
            .map(|result| {
                result.unwrap_or_else(|| {
                    Err(EmbedError::BatchAbortedAfterTerminalBackend(
                        terminal_failure
                            .get()
                            .cloned()
                            .unwrap_or_else(|| "WeSpeaker actor returned no result".to_string()),
                    ))
                })
            })
            .collect::<Vec<_>>();
        if let Some(reason) = terminal_failure.get() {
            abort_successful_results_after_terminal_failure(&mut results, reason);
        }
        results
    }
}

enum PolicySpeakerRuntimeInner {
    ReDimNet(Mutex<PolicyResolvedAuxRuntime<PolicySpeakerCandidate, EmbedError>>),
    WeSpeaker(Mutex<PolicyResolvedAuxRuntime<PolicyWeSpeakerCandidate, EmbedError>>),
}

struct PolicyResolvedSpeakerEmbedder {
    runtime: PolicySpeakerRuntimeInner,
    identity: SpeakerEmbedderIdentity,
}

impl SpeakerEmbedder for PolicyResolvedSpeakerEmbedder {
    fn embed(&self, samples: &[f32], sample_rate_hz: u32) -> Result<SpeakerEmbedding, EmbedError> {
        match &self.runtime {
            PolicySpeakerRuntimeInner::ReDimNet(runtime) => runtime
                .lock()
                .map_err(|_| {
                    EmbedError::Unavailable(
                        "policy-resolved speaker runtime lock is poisoned".to_string(),
                    )
                })?
                .invoke_replay_safe(|candidate| candidate.embed(samples, sample_rate_hz))
                .map_err(policy_runtime_error),
            PolicySpeakerRuntimeInner::WeSpeaker(runtime) => runtime
                .lock()
                .map_err(|_| {
                    EmbedError::Unavailable(
                        "policy-resolved speaker runtime lock is poisoned".to_string(),
                    )
                })?
                .invoke_replay_safe(|candidate| candidate.embed(samples, sample_rate_hz))
                .map_err(policy_runtime_error),
        }
    }

    fn embed_batch(
        &self,
        clips: &[&[f32]],
        sample_rate_hz: u32,
    ) -> Vec<Result<SpeakerEmbedding, EmbedError>> {
        match &self.runtime {
            PolicySpeakerRuntimeInner::ReDimNet(runtime) => match runtime.lock() {
                Ok(mut runtime) => runtime
                    .invoke_replay_safe(
                        |candidate| Ok(candidate.embed_batch(clips, sample_rate_hz)),
                    )
                    .unwrap_or_else(|error| repeat_error(clips.len(), policy_runtime_error(error))),
                Err(_) => repeat_error(
                    clips.len(),
                    EmbedError::Unavailable(
                        "policy-resolved speaker runtime lock is poisoned".to_string(),
                    ),
                ),
            },
            PolicySpeakerRuntimeInner::WeSpeaker(runtime) => match runtime.lock() {
                Ok(mut runtime) => runtime
                    .invoke_replay_safe(
                        |candidate| Ok(candidate.embed_batch(clips, sample_rate_hz)),
                    )
                    .unwrap_or_else(|error| repeat_error(clips.len(), policy_runtime_error(error))),
                Err(_) => repeat_error(
                    clips.len(),
                    EmbedError::Unavailable(
                        "policy-resolved speaker runtime lock is poisoned".to_string(),
                    ),
                ),
            },
        }
    }

    fn embedding_dim(&self) -> usize {
        self.identity.embedding_dim
    }

    fn calibration_profile(&self) -> SpeakerCalibrationProfile {
        match self.identity.family {
            SpeakerEmbedderFamily::ReDimNet2 => REDIMNET_CALIBRATION,
            SpeakerEmbedderFamily::WeSpeakerResNet => WESPEAKER_CALIBRATION,
        }
    }

    fn identity(&self) -> Option<SpeakerEmbedderIdentity> {
        Some(self.identity.clone())
    }
}

#[derive(Clone)]
pub struct PolicyResolvedSpeakerRuntime {
    embedder: Arc<dyn SpeakerEmbedder>,
    identity: SpeakerEmbedderIdentity,
}

impl PolicyResolvedSpeakerRuntime {
    pub fn load(
        execution_services: Arc<NativeExecutionServices>,
    ) -> Result<Option<Self>, EmbedError> {
        Self::load_with_intent(execution_services, ExecutionIntent::Auto)
    }

    pub(crate) fn load_with_intent(
        execution_services: Arc<NativeExecutionServices>,
        execution_intent: ExecutionIntent,
    ) -> Result<Option<Self>, EmbedError> {
        Self::load_with_preference(
            execution_services,
            execution_intent,
            VoiceIdEmbedderPreference::default(),
        )
    }

    pub(crate) fn load_with_preference(
        execution_services: Arc<NativeExecutionServices>,
        execution_intent: ExecutionIntent,
        preference: VoiceIdEmbedderPreference,
    ) -> Result<Option<Self>, EmbedError> {
        let Some(prepared) = prepare_embedder(preference)? else {
            return Ok(None);
        };
        match prepared.family {
            SpeakerEmbedderFamily::ReDimNet2 => {
                Self::load_redimnet(execution_services, execution_intent, prepared)
            }
            SpeakerEmbedderFamily::WeSpeakerResNet => {
                Self::load_wespeaker(execution_services, execution_intent, prepared)
            }
        }
    }

    fn load_redimnet(
        execution_services: Arc<NativeExecutionServices>,
        execution_intent: ExecutionIntent,
        prepared: PreparedSelectedEmbedder,
    ) -> Result<Option<Self>, EmbedError> {
        let (verified_pack, preflight, content_id) = prepared.source.into_parts();
        let retained_quote =
            RedimNet2Embedder::quoted_persistent_host_commitment_bytes(&preflight.tensor_index)?;
        let peak_quote = preflight
            .runtime_source
            .immutable_snapshot_construction_peak_bytes(retained_quote)
            .map_err(|error| EmbedError::Unavailable(error.to_string()))?;

        let execution_plan = resolve_auxiliary_execution_plan(
            execution_services.as_ref(),
            REDIMNET2_GGML_ARCHITECTURE_ID,
            &execution_intent,
        )
        .map_err(|error| EmbedError::Unavailable(error.to_string()))?;

        let services_for_builder = Arc::clone(&execution_services);
        let content_for_builder = content_id.clone();
        let preflight_for_builder = preflight.clone();
        let builder = Arc::new(
            move |execution_candidate: &crate::device::execution_policy::ExecutionCandidate| {
                let backend =
                    resolved_runtime_for_auxiliary_candidate(execution_candidate).backend();
                let key = AuxiliaryRuntimeCacheKey::host_neutral::<RedimNetParsedHost>(
                    REDIMNET2_GGML_ARCHITECTURE_ID,
                    content_for_builder.clone(),
                    REDIMNET_PARSED_HOST_REPRESENTATION,
                );
                let parsed = services_for_builder
                    .auxiliary_runtime_owners()
                    .get_or_try_insert_admitted_with(
                        key,
                        retained_quote,
                        || {
                            build_admitted_embedder(
                                &preflight_for_builder,
                                &content_for_builder,
                                peak_quote,
                                retained_quote,
                            )
                        },
                        |error| EmbedError::Unavailable(error.to_string()),
                    )?;
                let candidate = PolicySpeakerCandidate {
                    parsed,
                    services: Arc::clone(&services_for_builder),
                    content_id: content_for_builder.clone(),
                    backend,
                    placement: execution_candidate.placement,
                    provider: execution_candidate.device.route.provider,
                    stable_device_id: execution_candidate.device.route.stable_id.clone(),
                };
                let warmup = deterministic_warmup_audio();
                let warmup = candidate
                    .parsed
                    .prepare_embedding_input(&warmup, WARMUP_SAMPLE_RATE_HZ)?;
                drop(candidate.checkout_actor(1, Some(warmup))?);
                if let Some(failure) = current_execution_candidate_failure() {
                    return Err(EmbedError::Unavailable(format!(
                        "redimnet warmup recorded {:?} at {}: {}",
                        failure.kind, failure.operation, failure.detail
                    )));
                }
                Ok(candidate)
            },
        );
        let runtime = PolicyResolvedAuxRuntime::try_new(
            Arc::clone(&execution_services),
            execution_plan,
            STREAMING_SPEAKER_STAGE,
            builder,
            crate::models::native_execution_services::CandidateActivationQuoteSource::Pack(
                verified_pack,
            ),
        )
        .map_err(policy_runtime_error)?;
        let identity = SpeakerEmbedderIdentity {
            family: SpeakerEmbedderFamily::ReDimNet2,
            embedding_dim: EMBED_DIM,
            pack_fingerprint: content_id,
        };
        let embedder: Arc<dyn SpeakerEmbedder> = Arc::new(PolicyResolvedSpeakerEmbedder {
            runtime: PolicySpeakerRuntimeInner::ReDimNet(Mutex::new(runtime)),
            identity: identity.clone(),
        });
        Ok(Some(Self { embedder, identity }))
    }

    fn load_wespeaker(
        execution_services: Arc<NativeExecutionServices>,
        execution_intent: ExecutionIntent,
        prepared: PreparedSelectedEmbedder,
    ) -> Result<Option<Self>, EmbedError> {
        let (verified_pack, preflight, content_id) = prepared.source.into_parts();
        let retained_quote =
            WeSpeakerEmbedder::quoted_persistent_host_commitment_bytes(&preflight.tensor_index)?;
        let peak_quote = preflight
            .runtime_source
            .immutable_snapshot_construction_peak_bytes(retained_quote)
            .map_err(|error| EmbedError::Unavailable(error.to_string()))?;

        let execution_plan = resolve_auxiliary_execution_plan(
            execution_services.as_ref(),
            WESPEAKER_RESNET_ARCHITECTURE_ID,
            &execution_intent,
        )
        .map_err(|error| EmbedError::Unavailable(error.to_string()))?;

        let services_for_builder = Arc::clone(&execution_services);
        let content_for_builder = content_id.clone();
        let preflight_for_builder = preflight.clone();
        let builder = Arc::new(
            move |execution_candidate: &crate::device::execution_policy::ExecutionCandidate| {
                let backend =
                    resolved_runtime_for_auxiliary_candidate(execution_candidate).backend();
                let key = AuxiliaryRuntimeCacheKey::host_neutral::<WeSpeakerParsedHost>(
                    WESPEAKER_RESNET_ARCHITECTURE_ID,
                    content_for_builder.clone(),
                    WESPEAKER_PARSED_HOST_REPRESENTATION,
                );
                let parsed = services_for_builder
                    .auxiliary_runtime_owners()
                    .get_or_try_insert_admitted_with(
                        key,
                        retained_quote,
                        || {
                            build_admitted_wespeaker_embedder(
                                &preflight_for_builder,
                                &content_for_builder,
                                peak_quote,
                                retained_quote,
                            )
                        },
                        |error| EmbedError::Unavailable(error.to_string()),
                    )?;
                let candidate = PolicyWeSpeakerCandidate {
                    parsed,
                    services: Arc::clone(&services_for_builder),
                    content_id: content_for_builder.clone(),
                    backend,
                    placement: execution_candidate.placement,
                    provider: execution_candidate.device.route.provider,
                    stable_device_id: execution_candidate.device.route.stable_id.clone(),
                };
                let warmup = deterministic_warmup_audio();
                let warmup = candidate
                    .parsed
                    .prepare_embedding_input(&warmup, WARMUP_SAMPLE_RATE_HZ)?;
                drop(candidate.checkout_actor(1, Some(warmup))?);
                if let Some(failure) = current_execution_candidate_failure() {
                    return Err(EmbedError::Unavailable(format!(
                        "wespeaker warmup recorded {:?} at {}: {}",
                        failure.kind, failure.operation, failure.detail
                    )));
                }
                Ok(candidate)
            },
        );
        let runtime = PolicyResolvedAuxRuntime::try_new(
            Arc::clone(&execution_services),
            execution_plan,
            WESPEAKER_STREAMING_SPEAKER_STAGE,
            builder,
            crate::models::native_execution_services::CandidateActivationQuoteSource::Pack(
                verified_pack,
            ),
        )
        .map_err(policy_runtime_error)?;
        let identity = SpeakerEmbedderIdentity {
            family: SpeakerEmbedderFamily::WeSpeakerResNet,
            embedding_dim: wespeaker::config::EMBED_DIM,
            pack_fingerprint: content_id,
        };
        let embedder: Arc<dyn SpeakerEmbedder> = Arc::new(PolicyResolvedSpeakerEmbedder {
            runtime: PolicySpeakerRuntimeInner::WeSpeaker(Mutex::new(runtime)),
            identity: identity.clone(),
        });
        Ok(Some(Self { embedder, identity }))
    }

    pub fn diarizer(
        &self,
        sample_rate_hz: u32,
    ) -> Result<StreamingDiarizer, crate::diarize::voice_id::VoiceIdLibraryError> {
        let persons = load_person_matcher_for_embedder(&self.identity, self.embedder.as_ref())?;
        Ok(StreamingDiarizer::with_shared_embedder_and_persons(
            Arc::clone(&self.embedder),
            sample_rate_hz,
            persons,
        ))
    }

    /// Anonymous SPEAKER_00 clustering. Enrolled Voice ID names stay on the
    /// originating client; this matcher is intentionally empty.
    pub fn anonymous_diarizer(&self, sample_rate_hz: u32) -> StreamingDiarizer {
        let space = EmbeddingSpace::for_active_embedder(
            &self.identity,
            self.embedder.calibration_profile(),
        );
        StreamingDiarizer::with_shared_embedder_and_persons(
            Arc::clone(&self.embedder),
            sample_rate_hz,
            PersonMatcher::new(space, Vec::new(), 1.0, 0.0),
        )
    }

    pub fn speaker_change_detector(&self, sample_rate_hz: u32) -> StreamingSpeakerChangeDetector {
        StreamingSpeakerChangeDetector::with_shared_embedder(
            Arc::clone(&self.embedder),
            sample_rate_hz,
        )
    }

    pub fn identity(&self) -> &SpeakerEmbedderIdentity {
        &self.identity
    }

    pub fn embedder(&self) -> &dyn SpeakerEmbedder {
        self.embedder.as_ref()
    }

    pub(crate) fn shared_embedder(&self) -> Arc<dyn SpeakerEmbedder> {
        Arc::clone(&self.embedder)
    }
}

fn build_admitted_embedder(
    preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    expected_content_id: &str,
    peak_quote: u64,
    retained_quote: u64,
) -> Result<AdmittedHostObject<RedimNetParsedHost>, EmbedError> {
    let quote = SystemMemoryAllocationQuote::new(
        format!("aux.{REDIMNET2_GGML_ARCHITECTURE_ID}.{expected_content_id}.parsed-host-state"),
        peak_quote,
        retained_quote,
    )
    .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
    SystemMemoryOwner::try_allocate(quote, || {
        let snapshot = preflight
            .immutable_snapshot_matching_content_id(expected_content_id)
            .map_err(|error| error.to_string())?;
        let embedder =
            RedimNet2Embedder::from_preflight(&snapshot).map_err(|error| error.to_string())?;
        let actual_retained = embedder
            .persistent_host_commitment_bytes()
            .map_err(|error| error.to_string())?;
        let actual_peak = snapshot
            .runtime_source
            .immutable_snapshot_construction_peak_bytes(actual_retained)
            .map_err(|error| error.to_string())?;
        Ok(SystemMemoryAllocationOutcome::new(
            RedimNetParsedHost {
                embedder,
                _receipt_owner: redimnet_parsed_host_receipt_owner(expected_content_id),
            },
            actual_peak,
            actual_retained,
        ))
    })
    .map(Arc::new)
    .map_err(|error| EmbedError::Unavailable(error.to_string()))
}

fn build_admitted_wespeaker_embedder(
    preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    expected_content_id: &str,
    peak_quote: u64,
    retained_quote: u64,
) -> Result<AdmittedHostObject<WeSpeakerParsedHost>, EmbedError> {
    let quote = SystemMemoryAllocationQuote::new(
        format!("aux.{WESPEAKER_RESNET_ARCHITECTURE_ID}.{expected_content_id}.parsed-host-state"),
        peak_quote,
        retained_quote,
    )
    .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
    SystemMemoryOwner::try_allocate(quote, || {
        let snapshot = preflight
            .immutable_snapshot_matching_content_id(expected_content_id)
            .map_err(|error| error.to_string())?;
        let embedder =
            WeSpeakerEmbedder::from_preflight(&snapshot).map_err(|error| error.to_string())?;
        let actual_retained = embedder
            .persistent_host_commitment_bytes()
            .map_err(|error| error.to_string())?;
        let actual_peak = snapshot
            .runtime_source
            .immutable_snapshot_construction_peak_bytes(actual_retained)
            .map_err(|error| error.to_string())?;
        Ok(SystemMemoryAllocationOutcome::new(
            WeSpeakerParsedHost {
                embedder,
                _receipt_owner: wespeaker_parsed_host_receipt_owner(expected_content_id),
            },
            actual_peak,
            actual_retained,
        ))
    })
    .map(Arc::new)
    .map_err(|error| EmbedError::Unavailable(error.to_string()))
}

fn forward_one(
    runtime: &mut RedimNetResidentRuntime,
    features: Vec<f32>,
    frames: usize,
    threads: Option<usize>,
) -> Result<Result<SpeakerEmbedding, EmbedError>, EmbedError> {
    match runtime.forward(&features, frames, threads) {
        Ok(raw) => Ok(Ok(SpeakerEmbedding::l2_normalized(raw))),
        Err(error) if error.is_terminal_backend_failure() => {
            Err(EmbedError::TerminalBackend(error.to_string()))
        }
        Err(error) => Ok(Err(map_backbone_error(error))),
    }
}

fn wespeaker_forward_one(
    runtime: &mut WeSpeakerResidentRuntime,
    features: Vec<f32>,
    frames: usize,
    threads: Option<usize>,
) -> Result<Result<SpeakerEmbedding, EmbedError>, EmbedError> {
    match runtime.forward(&features, frames, threads) {
        Ok(raw) => Ok(Ok(SpeakerEmbedding::l2_normalized(raw))),
        Err(error) if error.is_terminal_backend_failure() => {
            Err(EmbedError::TerminalBackend(error.to_string()))
        }
        Err(error) => Ok(Err(map_wespeaker_backbone_error(error))),
    }
}

fn map_backbone_error(error: super::redimnet::backbone::RedimNetBackboneError) -> EmbedError {
    if error.is_canceled() {
        EmbedError::Canceled
    } else if error.is_terminal_backend_failure() {
        EmbedError::TerminalBackend(error.to_string())
    } else {
        EmbedError::Unavailable(error.to_string())
    }
}

fn map_wespeaker_backbone_error(
    error: super::wespeaker::backbone::WeSpeakerBackboneError,
) -> EmbedError {
    if error.is_canceled() {
        EmbedError::Canceled
    } else if error.is_terminal_backend_failure() {
        EmbedError::TerminalBackend(error.to_string())
    } else {
        EmbedError::Unavailable(error.to_string())
    }
}

fn map_actor_error(error: PinnedRuntimeActorError) -> EmbedError {
    EmbedError::TerminalBackend(error.to_string())
}

fn clone_embed_error(error: &EmbedError) -> EmbedError {
    match error {
        EmbedError::Unavailable(reason) => EmbedError::Unavailable(reason.clone()),
        EmbedError::TerminalBackend(reason) => EmbedError::TerminalBackend(reason.clone()),
        EmbedError::BatchAbortedAfterTerminalBackend(reason) => {
            EmbedError::BatchAbortedAfterTerminalBackend(reason.clone())
        }
        EmbedError::TooShort => EmbedError::TooShort,
        EmbedError::UnsupportedSampleRate(rate) => EmbedError::UnsupportedSampleRate(*rate),
        EmbedError::Canceled => EmbedError::Canceled,
    }
}

fn repeat_error(count: usize, error: EmbedError) -> Vec<Result<SpeakerEmbedding, EmbedError>> {
    (0..count).map(|_| Err(clone_embed_error(&error))).collect()
}

fn deterministic_warmup_audio() -> Vec<f32> {
    (0..WARMUP_SAMPLE_COUNT)
        .map(|index| if index % 200 < 100 { 0.02 } else { -0.02 })
        .collect()
}

fn policy_runtime_error(error: PolicyResolvedAuxRuntimeError<EmbedError>) -> EmbedError {
    EmbedError::Unavailable(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redimnet_host_receipt_releases_without_leaking_across_cache_lifecycle() {
        let services = Arc::new(
            NativeExecutionServices::for_local_process()
                .expect("construct native execution services"),
        );
        let _context =
            crate::models::native_execution_services::install_native_execution_services(&services);
        let owner = redimnet_parsed_host_receipt_owner("redimnet-test-content")
            .expect("host receipt owner");
        assert_eq!(services.runtime_receipts().summary().live_owner_count, 1);
        drop(owner);
        assert_eq!(services.runtime_receipts().summary().live_owner_count, 0);
    }
    #[test]
    fn redimnet_actor_key_separates_cpu_and_gpu_residents() {
        let cpu = redimnet_actor_key("same-pack", GgmlCpuGraphBackend::Cpu);
        let gpu = redimnet_actor_key("same-pack", GgmlCpuGraphBackend::Gpu);
        assert_ne!(cpu, gpu);
    }
}
