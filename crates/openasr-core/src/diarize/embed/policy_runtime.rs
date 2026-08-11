//! Policy-resolved ownership for the ReDimNet speaker-embedding stage.
//!
//! Parsed f32 weights and frontend state are immutable, Send-safe host data and
//! live in the service-root admitted host cache. Every mutable ggml runtime is
//! constructed, used, and destroyed on a dedicated owner thread. A bounded
//! checkout pool provides up to four resident runtimes for batch parallelism;
//! no process-global or thread-local owner exists outside the injected service
//! root.

use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;

use crate::{
    NativeExecutionServices,
    device::execution_policy::ExecutionIntent,
    ggml_runtime::GgmlCpuGraphBackend,
    models::{
        admitted_pinned_runtime_actor_pool::{
            CheckedOutPinnedRuntimeActorCall, PinnedRuntimeActorError,
            call_checked_out_actor_mut_fallible_async,
        },
        aux_pack_registry::AuxPackKind,
        aux_pack_registry::REDIMNET2_GGML_ARCHITECTURE_ID,
        native_execution_services::current_execution_candidate_failure,
        pack_verifier::{PackCandidate, PackRoute, PackVerifier},
        policy_resolved_aux_runtime::{
            AuxiliaryPinnedRuntimeCacheKey, AuxiliaryRuntimeCacheKey, PolicyResolvedAuxRuntime,
            PolicyResolvedAuxRuntimeError, resolve_auxiliary_execution_plan,
            resolved_runtime_for_auxiliary_candidate,
        },
        system_memory_owner::{
            AdmittedHostObject, SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
            SystemMemoryOwner,
        },
    },
};

use super::{
    EmbedError, REDIMNET_MAX_BATCH_WORKERS, RedimNet2Embedder, RedimNetResidentRuntime,
    SpeakerEmbedder, SpeakerEmbedderIdentity, SpeakerEmbeddingExecutionPlan,
    abort_successful_results_after_terminal_failure, cancel_requested, pack::redimnet_pack_path,
    redimnet::config::EMBED_DIM,
};
use crate::diarize::{
    calibration::{REDIMNET_CALIBRATION, SpeakerCalibrationProfile},
    contract::SpeakerEmbedding,
    streaming::{StreamingDiarizer, StreamingSpeakerChangeDetector},
    voice_id::load_person_matcher_for_embedder,
};

const STREAMING_SPEAKER_STAGE: &str = "redimnet2-streaming-speaker-stage-v1";
const WARMUP_SAMPLE_RATE_HZ: u32 = 16_000;
const WARMUP_SAMPLE_COUNT: usize = 40_000;
const REDIMNET_PARSED_HOST_REPRESENTATION: &str = "redimnet2.parsed-f32.v1";
const REDIMNET_RESIDENT_REPRESENTATION: &str = "redimnet2.ggml-resident.v1";

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

type SharedAdmittedEmbedder = AdmittedHostObject<RedimNet2Embedder>;
type PendingActorBatch = CheckedOutPinnedRuntimeActorCall<
    AuxiliaryPinnedRuntimeCacheKey,
    RedimNetResidentRuntime,
    Result<Vec<(usize, Result<SpeakerEmbedding, EmbedError>)>, EmbedError>,
>;

struct PolicySpeakerCandidate {
    parsed: SharedAdmittedEmbedder,
    services: Arc<NativeExecutionServices>,
    content_id: String,
    backend: GgmlCpuGraphBackend,
    placement: crate::device::execution_policy::ExecutionPlacement,
}

impl PolicySpeakerCandidate {
    fn actor_key(&self) -> AuxiliaryPinnedRuntimeCacheKey {
        AuxiliaryPinnedRuntimeCacheKey::for_current_lane::<RedimNetResidentRuntime>(
            REDIMNET2_GGML_ARCHITECTURE_ID,
            self.content_id.clone(),
            REDIMNET_RESIDENT_REPRESENTATION,
            self.backend,
        )
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
        self.services
            .redimnet_runtime_actors()
            .checkout_or_try_build_with(
                self.actor_key(),
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

struct PolicyResolvedSpeakerEmbedder {
    runtime: Mutex<PolicyResolvedAuxRuntime<PolicySpeakerCandidate, EmbedError>>,
    identity: SpeakerEmbedderIdentity,
}

impl SpeakerEmbedder for PolicyResolvedSpeakerEmbedder {
    fn embed(&self, samples: &[f32], sample_rate_hz: u32) -> Result<SpeakerEmbedding, EmbedError> {
        self.runtime
            .lock()
            .map_err(|_| {
                EmbedError::Unavailable(
                    "policy-resolved speaker runtime lock is poisoned".to_string(),
                )
            })?
            .invoke_replay_safe(|candidate| candidate.embed(samples, sample_rate_hz))
            .map_err(policy_runtime_error)
    }

    fn embed_batch(
        &self,
        clips: &[&[f32]],
        sample_rate_hz: u32,
    ) -> Vec<Result<SpeakerEmbedding, EmbedError>> {
        match self.runtime.lock() {
            Ok(mut runtime) => runtime
                .invoke_replay_safe(|candidate| Ok(candidate.embed_batch(clips, sample_rate_hz)))
                .unwrap_or_else(|error| repeat_error(clips.len(), policy_runtime_error(error))),
            Err(_) => repeat_error(
                clips.len(),
                EmbedError::Unavailable(
                    "policy-resolved speaker runtime lock is poisoned".to_string(),
                ),
            ),
        }
    }

    fn embedding_dim(&self) -> usize {
        EMBED_DIM
    }

    fn calibration_profile(&self) -> SpeakerCalibrationProfile {
        REDIMNET_CALIBRATION
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
        let Some(pack_path) = redimnet_pack_path() else {
            return Ok(None);
        };
        let verified_pack = PackVerifier
            .verify_candidate(PackCandidate::new(&pack_path))
            .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
        if !matches!(
            verified_pack.route(),
            PackRoute::Aux {
                kind: AuxPackKind::Diarization,
                ..
            }
        ) {
            return Err(EmbedError::Unavailable(format!(
                "ReDimNet pack route is not auxiliary diarization: {:?}",
                verified_pack.route()
            )));
        }
        let preflight = verified_pack.preflight().clone();
        let content_id = preflight.runtime_source.content_id().to_string();
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
                let key = AuxiliaryRuntimeCacheKey::host_neutral::<RedimNet2Embedder>(
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
        )
        .map_err(policy_runtime_error)?;
        let identity = SpeakerEmbedderIdentity {
            embedding_dim: EMBED_DIM,
            pack_fingerprint: content_id,
        };
        let embedder: Arc<dyn SpeakerEmbedder> = Arc::new(PolicyResolvedSpeakerEmbedder {
            runtime: Mutex::new(runtime),
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
) -> Result<AdmittedHostObject<RedimNet2Embedder>, EmbedError> {
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
            embedder,
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

fn map_backbone_error(error: super::redimnet::backbone::RedimNetBackboneError) -> EmbedError {
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
