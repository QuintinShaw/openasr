//! One policy family for admitted host, actor checkout, and bounded batch.
//!
//! ReDimNet2 and WeSpeaker keep typed NES actor pools. This layer implements
//! the shared host/batch protocol once and type-erases at `Arc<dyn SpeakerEmbedder>`.

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
            CheckedOutPinnedRuntimeActorCall, PinnedRuntimeActorCheckout, PinnedRuntimeActorError,
            call_checked_out_actor_mut_fallible_async,
        },
        aux_pack_registry::{REDIMNET2_GGML_ARCHITECTURE_ID, WESPEAKER_RESNET_ARCHITECTURE_ID},
        native_execution_services::current_execution_candidate_failure,
        policy_resolved_aux_runtime::{
            AuxiliaryPinnedRuntimeCacheKey, AuxiliaryRuntimeCacheKey, PolicyResolvedAuxRuntime,
            resolve_auxiliary_execution_plan, resolved_runtime_for_auxiliary_candidate,
        },
        runtime_receipts::{RuntimeOwnerDescriptor, RuntimeOwnerGuard},
        system_memory_owner::{
            AdmittedHostObject, SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
            SystemMemoryOwner,
        },
    },
};

use super::{
    EMBEDDER_MAX_BATCH_WORKERS, EmbedError, RedimNet2Embedder, RedimNetResidentRuntime,
    SpeakerEmbedder, SpeakerEmbedderFamily, SpeakerEmbedderIdentity, SpeakerEmbeddingExecutionPlan,
    WeSpeakerEmbedder, WeSpeakerResidentRuntime, abort_successful_results_after_terminal_failure,
    cancel_requested, embedder_batch_worker_limit, pack::PreparedSelectedEmbedder,
};
use crate::diarize::contract::SpeakerEmbedding;

const WARMUP_SAMPLE_RATE_HZ: u32 = 16_000;
const WARMUP_SAMPLE_COUNT: usize = 40_000;
const REDIMNET_PARSED_HOST_REPRESENTATION: &str = "redimnet2.parsed-f32.v1";
const REDIMNET_RESIDENT_REPRESENTATION: &str = "redimnet2.ggml-resident.v1";
const REDIMNET_STREAMING_SPEAKER_STAGE: &str = "redimnet2-streaming-speaker-stage-v1";
const WESPEAKER_PARSED_HOST_REPRESENTATION: &str = "wespeaker.parsed-f32.v1";
const WESPEAKER_RESIDENT_REPRESENTATION: &str = "wespeaker.ggml-resident.v1";
const WESPEAKER_STREAMING_SPEAKER_STAGE: &str = "wespeaker-streaming-speaker-stage-v1";

static EMBEDDER_FRONTEND_POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();

fn embedder_frontend_pool() -> &'static Result<rayon::ThreadPool, String> {
    EMBEDDER_FRONTEND_POOL.get_or_init(|| {
        let workers = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
            .clamp(1, EMBEDDER_MAX_BATCH_WORKERS);
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|index| format!("openasr-embedder-frontend-{index}"))
            .build()
            .map_err(|error| error.to_string())
    })
}

fn embedder_resident_worker_limit(backend: GgmlCpuGraphBackend, pool_threads: usize) -> usize {
    #[cfg(test)]
    if std::env::var_os("OPENASR_REDIMNET_BENCH_WORKERS").is_some() {
        return embedder_batch_worker_limit(pool_threads);
    }
    if backend.is_gpu_class() {
        1
    } else {
        embedder_batch_worker_limit(pool_threads)
    }
}

pub(super) trait SpeakerPolicyFamily: Sized + Send + Sync + 'static {
    type Embedder: Send + Sync + 'static;
    type ResidentRuntime: 'static;
    type BackboneError: std::fmt::Display;

    fn family() -> SpeakerEmbedderFamily;
    fn architecture_id() -> &'static str;
    fn parsed_host_representation() -> &'static str;
    fn resident_representation() -> &'static str;
    fn streaming_stage() -> &'static str;
    fn parsed_host_owner_kind() -> &'static str;
    fn resident_owner_kind() -> &'static str;

    fn identity(pack_fingerprint: String, catalog_model_id: String) -> SpeakerEmbedderIdentity;

    fn from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    ) -> Result<Self::Embedder, EmbedError>;
    fn persistent_host_commitment_bytes(embedder: &Self::Embedder) -> Result<u64, EmbedError>;
    fn quoted_persistent_host_commitment_bytes(
        tensor_index: &crate::GgufTensorIndex,
    ) -> Result<u64, EmbedError>;
    fn prepare_embedding_input(
        embedder: &Self::Embedder,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Result<(Vec<f32>, usize), EmbedError>;

    fn is_canceled(error: &Self::BackboneError) -> bool;
    fn is_terminal(error: &Self::BackboneError) -> bool;
    fn forward(
        runtime: &mut Self::ResidentRuntime,
        features: &[f32],
        frames: usize,
        threads: Option<usize>,
    ) -> Result<Vec<f32>, Self::BackboneError>;

    fn checkout_actor(
        services: &NativeExecutionServices,
        key: AuxiliaryPinnedRuntimeCacheKey,
        owner_descriptor: Option<RuntimeOwnerDescriptor>,
        embedder: &Self::Embedder,
        backend: GgmlCpuGraphBackend,
        placement: crate::device::execution_policy::ExecutionPlacement,
        threads: usize,
        warmup: Option<(Vec<f32>, usize)>,
    ) -> Result<
        PinnedRuntimeActorCheckout<AuxiliaryPinnedRuntimeCacheKey, Self::ResidentRuntime>,
        EmbedError,
    >;

    fn actor_key(
        content_id: impl Into<String>,
        backend: GgmlCpuGraphBackend,
    ) -> AuxiliaryPinnedRuntimeCacheKey {
        AuxiliaryPinnedRuntimeCacheKey::for_current_lane::<Self::ResidentRuntime>(
            Self::architecture_id(),
            content_id,
            Self::resident_representation(),
            backend,
        )
    }

    fn actor_receipt_descriptor(
        content_id: &str,
        backend: GgmlCpuGraphBackend,
        placement: crate::device::execution_policy::ExecutionPlacement,
        provider: ExecutionProvider,
        stable_device_id: &str,
    ) -> Option<RuntimeOwnerDescriptor> {
        let collector = crate::models::native_execution_services::current_runtime_receipts()?;
        let lane = collector.lane_projection(provider, stable_device_id, placement, backend)?;
        collector.owner_descriptor(
            Self::resident_owner_kind(),
            Some(content_id),
            Some(Self::resident_representation()),
            Some(lane),
        )
    }

    fn parsed_host_receipt_owner(content_id: &str) -> Option<RuntimeOwnerGuard> {
        let collector = crate::models::native_execution_services::current_runtime_receipts()?;
        let descriptor = collector.host_neutral_owner_descriptor(
            Self::parsed_host_owner_kind(),
            Some(content_id),
            Some(Self::parsed_host_representation()),
        )?;
        Some(collector.start_owner(
            descriptor,
            crate::models::native_execution_services::current_execution_cache_attempt_id(),
        ))
    }

    fn map_backbone_error(error: Self::BackboneError) -> EmbedError {
        if Self::is_canceled(&error) {
            EmbedError::Canceled
        } else if Self::is_terminal(&error) {
            EmbedError::TerminalBackend(error.to_string())
        } else {
            EmbedError::Unavailable(error.to_string())
        }
    }

    fn forward_one(
        runtime: &mut Self::ResidentRuntime,
        features: Vec<f32>,
        frames: usize,
        threads: Option<usize>,
    ) -> Result<Result<SpeakerEmbedding, EmbedError>, EmbedError> {
        match Self::forward(runtime, &features, frames, threads) {
            Ok(raw) => Ok(Ok(SpeakerEmbedding::l2_normalized(raw))),
            Err(error) if Self::is_terminal(&error) => {
                Err(EmbedError::TerminalBackend(error.to_string()))
            }
            Err(error) => Ok(Err(Self::map_backbone_error(error))),
        }
    }
}

pub(super) struct RedimNetPolicy;
pub(super) struct WeSpeakerPolicy;

impl SpeakerPolicyFamily for RedimNetPolicy {
    type Embedder = RedimNet2Embedder;
    type ResidentRuntime = RedimNetResidentRuntime;
    type BackboneError = super::redimnet::backbone::RedimNetBackboneError;

    fn family() -> SpeakerEmbedderFamily {
        SpeakerEmbedderFamily::ReDimNet2
    }
    fn architecture_id() -> &'static str {
        REDIMNET2_GGML_ARCHITECTURE_ID
    }
    fn parsed_host_representation() -> &'static str {
        REDIMNET_PARSED_HOST_REPRESENTATION
    }
    fn resident_representation() -> &'static str {
        REDIMNET_RESIDENT_REPRESENTATION
    }
    fn streaming_stage() -> &'static str {
        REDIMNET_STREAMING_SPEAKER_STAGE
    }
    fn parsed_host_owner_kind() -> &'static str {
        "redimnet2.parsed-host-state"
    }
    fn resident_owner_kind() -> &'static str {
        "redimnet2.resident-runtime"
    }

    fn identity(pack_fingerprint: String, catalog_model_id: String) -> SpeakerEmbedderIdentity {
        SpeakerEmbedderIdentity::redimnet2(pack_fingerprint, catalog_model_id)
    }

    fn from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    ) -> Result<Self::Embedder, EmbedError> {
        RedimNet2Embedder::from_preflight(preflight)
    }
    fn persistent_host_commitment_bytes(embedder: &Self::Embedder) -> Result<u64, EmbedError> {
        embedder.persistent_host_commitment_bytes()
    }
    fn quoted_persistent_host_commitment_bytes(
        tensor_index: &crate::GgufTensorIndex,
    ) -> Result<u64, EmbedError> {
        RedimNet2Embedder::quoted_persistent_host_commitment_bytes(tensor_index)
    }
    fn prepare_embedding_input(
        embedder: &Self::Embedder,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Result<(Vec<f32>, usize), EmbedError> {
        embedder.prepare_embedding_input(samples, sample_rate_hz)
    }

    fn is_canceled(error: &Self::BackboneError) -> bool {
        error.is_canceled()
    }
    fn is_terminal(error: &Self::BackboneError) -> bool {
        error.is_terminal_backend_failure()
    }
    fn forward(
        runtime: &mut Self::ResidentRuntime,
        features: &[f32],
        frames: usize,
        threads: Option<usize>,
    ) -> Result<Vec<f32>, Self::BackboneError> {
        runtime.forward(features, frames, threads)
    }

    fn checkout_actor(
        services: &NativeExecutionServices,
        key: AuxiliaryPinnedRuntimeCacheKey,
        owner_descriptor: Option<RuntimeOwnerDescriptor>,
        embedder: &Self::Embedder,
        backend: GgmlCpuGraphBackend,
        placement: crate::device::execution_policy::ExecutionPlacement,
        threads: usize,
        warmup: Option<(Vec<f32>, usize)>,
    ) -> Result<
        PinnedRuntimeActorCheckout<AuxiliaryPinnedRuntimeCacheKey, Self::ResidentRuntime>,
        EmbedError,
    > {
        let weights = embedder.shared_weights();
        services
            .redimnet_runtime_actors()
            .checkout_or_try_build_with_owner_receipt(
                key,
                owner_descriptor,
                || Ok((0, (weights, threads, warmup))),
                move |(weights, threads, warmup)| {
                    let mut runtime =
                        RedimNetResidentRuntime::new(weights, Some(threads), backend, placement)
                            .map_err(Self::map_backbone_error)?;
                    if let Some((features, frames)) = warmup {
                        runtime
                            .forward(&features, frames, Some(threads))
                            .map_err(Self::map_backbone_error)?;
                    }
                    Ok(SystemMemoryOwner::without_allocation(runtime))
                },
                map_actor_error,
            )
    }
}

impl SpeakerPolicyFamily for WeSpeakerPolicy {
    type Embedder = WeSpeakerEmbedder;
    type ResidentRuntime = WeSpeakerResidentRuntime;
    type BackboneError = super::wespeaker::backbone::WeSpeakerBackboneError;

    fn family() -> SpeakerEmbedderFamily {
        SpeakerEmbedderFamily::WeSpeakerResNet
    }
    fn architecture_id() -> &'static str {
        WESPEAKER_RESNET_ARCHITECTURE_ID
    }
    fn parsed_host_representation() -> &'static str {
        WESPEAKER_PARSED_HOST_REPRESENTATION
    }
    fn resident_representation() -> &'static str {
        WESPEAKER_RESIDENT_REPRESENTATION
    }
    fn streaming_stage() -> &'static str {
        WESPEAKER_STREAMING_SPEAKER_STAGE
    }
    fn parsed_host_owner_kind() -> &'static str {
        "wespeaker.parsed-host-state"
    }
    fn resident_owner_kind() -> &'static str {
        "wespeaker.resident-runtime"
    }

    fn identity(pack_fingerprint: String, catalog_model_id: String) -> SpeakerEmbedderIdentity {
        SpeakerEmbedderIdentity::wespeaker_resnet(pack_fingerprint, catalog_model_id)
    }

    fn from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    ) -> Result<Self::Embedder, EmbedError> {
        WeSpeakerEmbedder::from_preflight(preflight)
    }
    fn persistent_host_commitment_bytes(embedder: &Self::Embedder) -> Result<u64, EmbedError> {
        embedder.persistent_host_commitment_bytes()
    }
    fn quoted_persistent_host_commitment_bytes(
        tensor_index: &crate::GgufTensorIndex,
    ) -> Result<u64, EmbedError> {
        WeSpeakerEmbedder::quoted_persistent_host_commitment_bytes(tensor_index)
    }
    fn prepare_embedding_input(
        embedder: &Self::Embedder,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Result<(Vec<f32>, usize), EmbedError> {
        embedder.prepare_embedding_input(samples, sample_rate_hz)
    }

    fn is_canceled(error: &Self::BackboneError) -> bool {
        error.is_canceled()
    }
    fn is_terminal(error: &Self::BackboneError) -> bool {
        error.is_terminal_backend_failure()
    }
    fn forward(
        runtime: &mut Self::ResidentRuntime,
        features: &[f32],
        frames: usize,
        threads: Option<usize>,
    ) -> Result<Vec<f32>, Self::BackboneError> {
        runtime.forward(features, frames, threads)
    }

    fn checkout_actor(
        services: &NativeExecutionServices,
        key: AuxiliaryPinnedRuntimeCacheKey,
        owner_descriptor: Option<RuntimeOwnerDescriptor>,
        embedder: &Self::Embedder,
        backend: GgmlCpuGraphBackend,
        placement: crate::device::execution_policy::ExecutionPlacement,
        threads: usize,
        warmup: Option<(Vec<f32>, usize)>,
    ) -> Result<
        PinnedRuntimeActorCheckout<AuxiliaryPinnedRuntimeCacheKey, Self::ResidentRuntime>,
        EmbedError,
    > {
        let weights = embedder.shared_weights();
        let config = embedder.config();
        services
            .wespeaker_runtime_actors()
            .checkout_or_try_build_with_owner_receipt(
                key,
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
                    .map_err(Self::map_backbone_error)?;
                    if let Some((features, frames)) = warmup {
                        runtime
                            .forward(&features, frames, Some(threads))
                            .map_err(Self::map_backbone_error)?;
                    }
                    Ok(SystemMemoryOwner::without_allocation(runtime))
                },
                map_actor_error,
            )
    }
}

struct PolicyParsedHost<E> {
    embedder: E,
    _receipt_owner: Option<RuntimeOwnerGuard>,
}

impl<E> Deref for PolicyParsedHost<E> {
    type Target = E;

    fn deref(&self) -> &Self::Target {
        &self.embedder
    }
}

struct PolicySpeakerCandidate<F: SpeakerPolicyFamily> {
    parsed: AdmittedHostObject<PolicyParsedHost<F::Embedder>>,
    services: Arc<NativeExecutionServices>,
    content_id: String,
    backend: GgmlCpuGraphBackend,
    placement: crate::device::execution_policy::ExecutionPlacement,
    provider: ExecutionProvider,
    stable_device_id: String,
}

impl<F: SpeakerPolicyFamily> PolicySpeakerCandidate<F> {
    fn actor_key(&self) -> AuxiliaryPinnedRuntimeCacheKey {
        F::actor_key(self.content_id.clone(), self.backend)
    }

    fn checkout_actor(
        &self,
        threads: usize,
        warmup: Option<(Vec<f32>, usize)>,
    ) -> Result<
        PinnedRuntimeActorCheckout<AuxiliaryPinnedRuntimeCacheKey, F::ResidentRuntime>,
        EmbedError,
    > {
        F::checkout_actor(
            &self.services,
            self.actor_key(),
            F::actor_receipt_descriptor(
                &self.content_id,
                self.backend,
                self.placement,
                self.provider,
                &self.stable_device_id,
            ),
            &self.parsed,
            self.backend,
            self.placement,
            threads,
            warmup,
        )
    }

    fn embed(&self, samples: &[f32], sample_rate_hz: u32) -> Result<SpeakerEmbedding, EmbedError> {
        let (features, frames) = F::prepare_embedding_input(&self.parsed, samples, sample_rate_hz)?;
        let threads = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        let actor = self.checkout_actor(threads, None)?;
        actor
            .call_mut_fallible(move |runtime| {
                F::forward_one(runtime, features, frames, Some(threads))
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
        let family_name = F::family().display_name();
        let pool = match embedder_frontend_pool() {
            Ok(pool) => pool,
            Err(reason) => {
                return repeat_error(
                    clips.len(),
                    EmbedError::Unavailable(format!(
                        "could not create bounded {family_name} frontend pool: {reason}"
                    )),
                );
            }
        };
        let available = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        let max_workers = embedder_resident_worker_limit(self.backend, pool.current_num_threads());
        let plan = SpeakerEmbeddingExecutionPlan::for_clips(clips.len(), available, max_workers);
        let inherited_cancel = crate::ggml_runtime::thread_job_cancel_flag();
        let prepared = pool.install(|| {
            clips
                .par_iter()
                .map(|samples| {
                    let _cancel = inherited_cancel
                        .as_ref()
                        .map(crate::ggml_runtime::InheritedJobCancelGuard::arm);
                    F::prepare_embedding_input(&self.parsed, samples, sample_rate_hz)
                })
                .collect::<Vec<_>>()
        });

        let mut prepared = prepared.into_iter().map(Some).collect::<Vec<_>>();
        let mut results: Vec<Option<Result<SpeakerEmbedding, EmbedError>>> =
            std::iter::repeat_with(|| None).take(clips.len()).collect();
        let mut jobs = Vec::<
            CheckedOutPinnedRuntimeActorCall<
                AuxiliaryPinnedRuntimeCacheKey,
                F::ResidentRuntime,
                Result<Vec<(usize, Result<SpeakerEmbedding, EmbedError>)>, EmbedError>,
            >,
        >::with_capacity(plan.workers);
        for worker in 0..plan.workers {
            let range = plan.worker_range(worker, clips.len());
            let mut inputs = Vec::with_capacity(range.len());
            for index in range {
                match prepared[index]
                    .take()
                    .unwrap_or_else(|| panic!("each prepared {family_name} input is consumed once"))
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
                    match F::forward_one(runtime, features, frames, Some(plan.threads_per_runner)) {
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
                            .unwrap_or_else(|| format!("{family_name} actor returned no result")),
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

struct PolicyResolvedSpeakerEmbedder<F: SpeakerPolicyFamily> {
    runtime: Mutex<PolicyResolvedAuxRuntime<PolicySpeakerCandidate<F>, EmbedError>>,
    identity: SpeakerEmbedderIdentity,
}

impl<F: SpeakerPolicyFamily> SpeakerEmbedder for PolicyResolvedSpeakerEmbedder<F> {
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
        self.identity.embedding_dim
    }

    fn calibration_profile(&self) -> crate::diarize::calibration::SpeakerCalibrationProfile {
        match self.identity.family {
            SpeakerEmbedderFamily::ReDimNet2 => crate::diarize::calibration::REDIMNET_CALIBRATION,
            SpeakerEmbedderFamily::WeSpeakerResNet => {
                crate::diarize::calibration::WESPEAKER_CALIBRATION
            }
        }
    }

    fn identity(&self) -> Option<SpeakerEmbedderIdentity> {
        Some(self.identity.clone())
    }
}

pub(super) fn load_family<F: SpeakerPolicyFamily>(
    execution_services: Arc<NativeExecutionServices>,
    execution_intent: ExecutionIntent,
    prepared: PreparedSelectedEmbedder,
) -> Result<Option<(Arc<dyn SpeakerEmbedder>, SpeakerEmbedderIdentity)>, EmbedError> {
    let catalog_model_id = prepared.catalog_model_id.clone();
    let (verified_pack, preflight, content_id) = prepared.source.into_parts();
    let retained_quote = F::quoted_persistent_host_commitment_bytes(&preflight.tensor_index)?;
    let peak_quote = preflight
        .runtime_source
        .immutable_snapshot_construction_peak_bytes(retained_quote)
        .map_err(|error| EmbedError::Unavailable(error.to_string()))?;

    let execution_plan = resolve_auxiliary_execution_plan(
        execution_services.as_ref(),
        F::architecture_id(),
        &execution_intent,
    )
    .map_err(|error| EmbedError::Unavailable(error.to_string()))?;

    let services_for_builder = Arc::clone(&execution_services);
    let content_for_builder = content_id.clone();
    let preflight_for_builder = preflight.clone();
    let family_name = F::family().display_name();
    let builder = Arc::new(
        move |execution_candidate: &crate::device::execution_policy::ExecutionCandidate| {
            let backend = resolved_runtime_for_auxiliary_candidate(execution_candidate).backend();
            let key = AuxiliaryRuntimeCacheKey::host_neutral::<PolicyParsedHost<F::Embedder>>(
                F::architecture_id(),
                content_for_builder.clone(),
                F::parsed_host_representation(),
            );
            let parsed = services_for_builder
                .auxiliary_runtime_owners()
                .get_or_try_insert_admitted_with(
                    key,
                    retained_quote,
                    || {
                        build_admitted::<F>(
                            &preflight_for_builder,
                            &content_for_builder,
                            peak_quote,
                            retained_quote,
                        )
                    },
                    |error| EmbedError::Unavailable(error.to_string()),
                )?;
            let candidate = PolicySpeakerCandidate::<F> {
                parsed,
                services: Arc::clone(&services_for_builder),
                content_id: content_for_builder.clone(),
                backend,
                placement: execution_candidate.placement,
                provider: execution_candidate.device.route.provider,
                stable_device_id: execution_candidate.device.route.stable_id.clone(),
            };
            let warmup = deterministic_warmup_audio();
            let warmup =
                F::prepare_embedding_input(&candidate.parsed, &warmup, WARMUP_SAMPLE_RATE_HZ)?;
            drop(candidate.checkout_actor(1, Some(warmup))?);
            if let Some(failure) = current_execution_candidate_failure() {
                return Err(EmbedError::Unavailable(format!(
                    "{family_name} warmup recorded {:?} at {}: {}",
                    failure.kind, failure.operation, failure.detail
                )));
            }
            Ok(candidate)
        },
    );
    let runtime = PolicyResolvedAuxRuntime::try_new(
        Arc::clone(&execution_services),
        execution_plan,
        F::streaming_stage(),
        builder,
        crate::models::native_execution_services::CandidateActivationQuoteSource::Pack(
            verified_pack,
        ),
    )
    .map_err(policy_runtime_error)?;
    let identity = F::identity(content_id, catalog_model_id);
    let embedder: Arc<dyn SpeakerEmbedder> = Arc::new(PolicyResolvedSpeakerEmbedder::<F> {
        runtime: Mutex::new(runtime),
        identity: identity.clone(),
    });
    Ok(Some((embedder, identity)))
}

fn build_admitted<F: SpeakerPolicyFamily>(
    preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    expected_content_id: &str,
    peak_quote: u64,
    retained_quote: u64,
) -> Result<AdmittedHostObject<PolicyParsedHost<F::Embedder>>, EmbedError> {
    let quote = SystemMemoryAllocationQuote::new(
        format!(
            "aux.{}.{expected_content_id}.parsed-host-state",
            F::architecture_id()
        ),
        peak_quote,
        retained_quote,
    )
    .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
    SystemMemoryOwner::try_allocate(quote, || {
        let snapshot = preflight
            .immutable_snapshot_matching_content_id(expected_content_id)
            .map_err(|error| error.to_string())?;
        let embedder = F::from_preflight(&snapshot).map_err(|error| error.to_string())?;
        let actual_retained =
            F::persistent_host_commitment_bytes(&embedder).map_err(|error| error.to_string())?;
        let actual_peak = snapshot
            .runtime_source
            .immutable_snapshot_construction_peak_bytes(actual_retained)
            .map_err(|error| error.to_string())?;
        Ok(SystemMemoryAllocationOutcome::new(
            PolicyParsedHost {
                embedder,
                _receipt_owner: F::parsed_host_receipt_owner(expected_content_id),
            },
            actual_peak,
            actual_retained,
        ))
    })
    .map(Arc::new)
    .map_err(|error| EmbedError::Unavailable(error.to_string()))
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

fn policy_runtime_error(
    error: crate::models::policy_resolved_aux_runtime::PolicyResolvedAuxRuntimeError<EmbedError>,
) -> EmbedError {
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
        let owner = RedimNetPolicy::parsed_host_receipt_owner("redimnet-test-content")
            .expect("host receipt owner");
        assert_eq!(services.runtime_receipts().summary().live_owner_count, 1);
        drop(owner);
        assert_eq!(services.runtime_receipts().summary().live_owner_count, 0);
    }

    #[test]
    fn redimnet_actor_key_separates_cpu_and_gpu_residents() {
        let cpu = RedimNetPolicy::actor_key("same-pack", GgmlCpuGraphBackend::Cpu);
        let gpu = RedimNetPolicy::actor_key("same-pack", GgmlCpuGraphBackend::Gpu);
        assert_ne!(cpu, gpu);
        let wespeaker_cpu = WeSpeakerPolicy::actor_key("same-pack", GgmlCpuGraphBackend::Cpu);
        assert_ne!(cpu, wespeaker_cpu);
    }
}
