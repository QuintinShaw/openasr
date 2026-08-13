use std::{
    collections::BTreeMap,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    time::Instant,
};

use crate::NATIVE_RUNTIME_MODEL_ID_AUTO;
use crate::api::audio_io::load_wav_16khz_mono_f32_v0;
use crate::arch::{
    DEFAULT_ENCODER_CHUNK_SECONDS, OpenAsrArchitectureRegistry, SpeakerSegmentationSource,
    emits_punctuation_for_model_architecture,
};
use crate::device::{
    execution_policy::{
        AcceleratedDeviceConstraint, ExecutionCandidate, ExecutionCandidateFailure,
        ExecutionIntent, ExecutionPlacement, ExecutionPlan, ExecutionPolicyError,
    },
    execution_route::{ExecutionProvider, enumerate_compute_devices_from_ggml},
};
use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig, RequestBackendPreference};
#[cfg(test)]
use crate::longform::plan_longform_slices;
use crate::longform::{
    AudioSliceKind, LongFormMode, LongFormSliceError, LongFormSlicePlanningError,
    LongFormVadProvider, SegmentMergePolicy, SegmentTimeDomain, SliceTranscript,
    TranscriptAssembler, plan_longform_slices_with_materialization_gate,
};
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyLongformProfile, BuiltinDecodePolicyLongformPromptCarryMode,
    resolve_builtin_decode_policy_for_architecture,
};
use crate::models::ggml_family_adapter::GgmlFamilyAdapterSelectionError;
use crate::models::graph_runtime_config::install_request_inference_threads_override;
#[cfg(test)]
use crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index_from_source;
use crate::models::runtime_selection_metadata::selection_metadata_from_gguf;
use crate::{
    GgmlAsrBackendPreference, GgmlAsrExecutionDispatch, GgmlAsrExecutionError,
    GgmlAsrExecutionOptions, GgmlAsrExecutionResult, GgmlAsrExecutionViewRequest,
    GgmlAsrPreparedAudioView, GgmlFamilyAdapterDescriptor, GgufRuntimeSourcePreflight,
    NativeExecutionServices, OasrV1MetadataError, PcmBuffer, PcmSlice, parse_model_ref,
};

use crate::api::backend::{FailureCategory, log_failure_context, log_request_context};

use super::{BackendError, Transcription, TranscriptionRequest};
use crate::Segment;
use crate::WordTimestamp;
use crate::api::backend::{DecodeTruncation, TranscriptionLongFormMetadata, TruncatedDecode};
use crate::models::firered_punc::pack::resolve_firered_punc_pack_path;
use crate::models::firered_punc::policy_runtime::{FireRedPuncActor, load_actor, punctuate};
#[cfg(test)]
use crate::models::firered_punc::runtime::FireRedPuncRuntime;
use crate::models::policy_resolved_aux_runtime::PolicyResolvedAuxRuntimeError;
use crate::models::qwen::{
    ForcedAlignItem, Qwen3ForcedAlignerSession, forced_aligner_pack, verify_forced_aligner_pack,
};
use crate::models::{
    aux_pack_registry::AuxPackKind,
    pack_verifier::{PackCandidate, PackRoute, PackVerifier, VerifiedPack},
};
use crate::punctuation::should_apply_punctuation;

const DEFAULT_NATIVE_LONGFORM_AUTO_TRIGGER_SECONDS: f32 = 30.0;
/// Chunk-length ceiling for the decode-side `ConservativeSeq2SeqV1`
/// repetition-guard profile (issue #60: cohere-transcribe, moonshine,
/// firered-aed). Historically this was a hard-coded `10.0` with no model
/// basis -- a defensive patch from when the repetition failure mode was
/// first found, predating the structural fix (the shared greedy-decode
/// driver's degenerate-loop guard, which is the actual anti-repetition
/// mechanism and stays in place regardless of chunk length). That 10s value
/// has since been surveyed against the industry evidence backing
/// `DEFAULT_ENCODER_CHUNK_SECONDS` (Whisper/Moonshine/NeMo/FunASR/
/// Dolphin/Cohere all converge near 30s) and found to have no independent
/// justification, so it is unified with that default: the previous name
/// (`COHERE_LONGFORM_MAX_CHUNK_SECONDS`) was also misleading on both counts
/// (not 10s anymore, and not cohere-only -- moonshine and firered-aed carry
/// the same profile).
///
/// It follows the *quality* default rather than
/// `arch::DEFAULT_ENCODER_SAFE_CHUNK_SECONDS` because that is the evidence it
/// actually rests on: this cap exists to keep decode well inside the regime
/// these families transcribe reliably in, not to bound encoder memory. The
/// memory ceiling applies separately and independently
/// (`apply_encoder_attention_span_longform_safety_policy`); a family carrying
/// both gets whichever is tighter.
const CONSERVATIVE_SEQ2SEQ_LONGFORM_MAX_CHUNK_SECONDS: f32 = DEFAULT_ENCODER_CHUNK_SECONDS;
const CONSERVATIVE_SEQ2SEQ_LONGFORM_OVERLAP_SECONDS: f32 = 0.0;

fn execution_intent_from_backend_env(raw: Option<&str>) -> Option<ExecutionIntent> {
    let value = raw.map(str::trim).filter(|value| !value.is_empty())?;
    if value.eq_ignore_ascii_case("cpu") {
        return Some(ExecutionIntent::CpuOnly);
    }
    if value.eq_ignore_ascii_case("gpu") {
        return Some(ExecutionIntent::AcceleratedOnly);
    }
    let provider = if value.eq_ignore_ascii_case("metal") {
        ExecutionProvider::Metal
    } else if value.eq_ignore_ascii_case("cuda") {
        ExecutionProvider::Cuda
    } else if value.eq_ignore_ascii_case("hip") || value.eq_ignore_ascii_case("rocm") {
        ExecutionProvider::Hip
    } else if value.eq_ignore_ascii_case("vulkan") {
        ExecutionProvider::Vulkan
    } else {
        return None;
    };
    Some(ExecutionIntent::ConstrainedAcceleratedOnly(
        AcceleratedDeviceConstraint::Provider(provider),
    ))
}

/// Resolve the process-wide developer/backend override into the same typed
/// request intent consumed by the unified execution policy. The environment
/// is read exactly once at the native boundary; every main and auxiliary
/// stage then receives a clone of this immutable value.
fn request_execution_intent(target: Option<crate::ExecutionTarget>) -> ExecutionIntent {
    let backend_env = std::env::var(GgmlCpuGraphConfig::BACKEND_ENV).ok();
    request_execution_intent_with_backend_env(target, backend_env.as_deref())
}

fn request_execution_intent_with_backend_env(
    target: Option<crate::ExecutionTarget>,
    backend_env: Option<&str>,
) -> ExecutionIntent {
    match target.unwrap_or_default() {
        crate::ExecutionTarget::Cpu => ExecutionIntent::CpuOnly,
        crate::ExecutionTarget::Accelerated => {
            match execution_intent_from_backend_env(backend_env) {
                Some(intent @ ExecutionIntent::ConstrainedAcceleratedOnly(_)) => intent,
                _ => ExecutionIntent::AcceleratedOnly,
            }
        }
        crate::ExecutionTarget::Auto => {
            execution_intent_from_backend_env(backend_env).unwrap_or(ExecutionIntent::Auto)
        }
    }
}
// Stage-weighted progress for the in-flight native file transcription.
// Registry + plan + overall math live in `transcription_progress`; this
// module only owns decode-slice sub-progress and wires real pipeline events
// into the shared reporter. Without a real event the overall fraction never
// increases (no fixed fake percentages, no time-based auto-climb).

#[cfg(test)]
use super::transcription_progress::{LegacyNativeTranscriptionProgress, NativeTranscriptionPhase};
use super::transcription_progress::{
    ProgressBackendClass, ProgressPlan, ProgressPlanInput, ProgressRegistryHandle,
    ProgressReporter, ProgressSegmenterKind, TranscriptionStage, duration_weighted_fraction,
};
#[cfg(test)]
use super::transcription_progress::{
    clear_progress_registry_for_test, native_transcription_progress,
    native_transcription_progress_for_id, progress_registry_test_lock,
};

/// Decode-stage sub-progress for multi-slice long-form (and the single-pass
/// "whole file is one slice" path). Each slice is weighted by sample count;
/// reported value is the **stage** fraction (0..=1 of decode), which the
/// shared reporter folds into cost-weighted overall.
struct DecodeProgress {
    reporter: ProgressReporter,
    total_samples: u64,
    // Atomic so the concurrent slice pipeline can accumulate completed-slice
    // shares from several worker threads at once. Stage-fraction reports are
    // monotonic under the progress registry.
    decoded_samples: AtomicU64,
}

impl DecodeProgress {
    fn begin(reporter: ProgressReporter, total_samples: u64) -> Self {
        reporter.enter_stage(TranscriptionStage::Decode);
        // Stage fraction only — never publish raw PCM sample counters as
        // completed/total units (UI would show "0/957696", which is internal
        // noise to users). Windows/segments stay on report_units elsewhere.
        reporter.report_fraction(0.0);
        Self {
            reporter,
            total_samples,
            decoded_samples: AtomicU64::new(0),
        }
    }

    /// Mark one slice decoded (or skipped as silent -- silence still consumes
    /// its share of the audio timeline).
    fn complete_slice(&self, slice_samples: u64) {
        let decoded = self
            .decoded_samples
            .fetch_add(slice_samples, Ordering::Relaxed)
            .saturating_add(slice_samples);
        let total = self.total_samples.max(1);
        self.reporter
            .report_fraction((decoded as f32 / total as f32).clamp(0.0, 1.0));
    }

    /// The [start, start+span) sub-range of the decode **stage** fraction that
    /// the next slice owns. Token-level interpolation runs inside this window.
    fn slice_progress_window(&self, slice_samples: u64) -> SliceProgressWindow {
        let total = (self.total_samples.max(1)) as f32;
        let decoded = self.decoded_samples.load(Ordering::Relaxed);
        let start_ratio = (decoded as f32 / total).clamp(0.0, 1.0);
        let span_ratio = (slice_samples as f32 / total).clamp(0.0, 1.0 - start_ratio);
        SliceProgressWindow {
            start_fraction: start_ratio,
            span_fraction: span_ratio,
        }
    }

    fn report_stage_fraction(&self, fraction: f32) {
        self.reporter.report_fraction(fraction);
    }
}

/// A slice's own sub-range of the decode stage fraction (0..=1 of decode).
#[derive(Debug, Clone, Copy, PartialEq)]
struct SliceProgressWindow {
    start_fraction: f32,
    span_fraction: f32,
}

/// Cap decode-work interpolation below 1.0 of the slice window so
/// `complete_slice` still visibly closes the slice after the last work unit.
const DECODE_WORK_PROGRESS_SLICE_SHARE_CAP: f32 = 0.95;

/// Publish at most every Nth work unit, plus the first and final units.
const DECODE_WORK_PROGRESS_PUBLISH_STRIDE: usize = 4;

fn decode_work_fraction(
    window: SliceProgressWindow,
    completed_work: usize,
    total_work: usize,
) -> f32 {
    let ratio = if total_work == 0 {
        DECODE_WORK_PROGRESS_SLICE_SHARE_CAP
    } else {
        let raw = completed_work as f32 / total_work as f32;
        raw.min(DECODE_WORK_PROGRESS_SLICE_SHARE_CAP)
    };
    window.start_fraction + window.span_fraction * ratio
}

fn should_publish_decode_work(completed_work: usize, total_work: usize) -> bool {
    completed_work == 1
        || completed_work == total_work
        || completed_work.is_multiple_of(DECODE_WORK_PROGRESS_PUBLISH_STRIDE)
}

/// Whether the request is likely to run forced alignment (plan includes align
/// weight). Late Auto decisions may still skip; the bar stays monotonic.
fn request_may_need_align(request: &TranscriptionRequest) -> bool {
    request.word_timestamps_refine
        || matches!(
            request.timeline_precision,
            crate::subtitle::TimelinePrecisionPolicy::Always
        )
        || request.needs_subtitle_export
        || request.voice_id
}

fn progress_backend_class(intent: &ExecutionIntent) -> ProgressBackendClass {
    match intent {
        ExecutionIntent::CpuOnly | ExecutionIntent::Auto => ProgressBackendClass::AutoOrCpu,
        ExecutionIntent::AcceleratedOnly
        | ExecutionIntent::ConstrainedAcceleratedOnly(_)
        | ExecutionIntent::Exact(_) => ProgressBackendClass::Accelerated,
    }
}

/// Map the resolved ggml backend (after candidate selection) onto progress
/// weight class. Prefer this over intent-only classification once the runtime
/// is known so Auto->Metal/GPU pays accelerated decode weights.
fn progress_backend_class_for_resolved(backend: GgmlCpuGraphBackend) -> ProgressBackendClass {
    match backend {
        GgmlCpuGraphBackend::Cpu => ProgressBackendClass::AutoOrCpu,
        GgmlCpuGraphBackend::Metal | GgmlCpuGraphBackend::Gpu => ProgressBackendClass::Accelerated,
    }
}

/// Map the prepared external segmenter provider onto plan weights. Provisional
/// `Auto` preference is replaced once prepare pins DiariZen vs Segmentation3_0.
fn progress_segmenter_kind_for_provider(
    provider: crate::diarize::segment::SegmenterProvider,
) -> ProgressSegmenterKind {
    match provider {
        crate::diarize::segment::SegmenterProvider::DiariZen => ProgressSegmenterKind::DiariZen,
        crate::diarize::segment::SegmenterProvider::Segmentation3_0 => {
            ProgressSegmenterKind::Segmentation3_0
        }
    }
}

/// Run one `run_dispatch_once` with per-decode-work progress wired to the
/// decode stage window for `slice_samples`, then `complete_slice` on success.
#[allow(clippy::too_many_arguments)]
fn run_dispatch_once_with_progress(
    dispatch: &GgmlAsrExecutionDispatch,
    execution_services: &Arc<NativeExecutionServices>,
    verified_pack: &VerifiedPack,
    selected_family: &GgmlFamilyAdapterDescriptor,
    chunk: PcmSlice,
    request_options: GgmlAsrExecutionOptions,
    backend_preference: GgmlAsrBackendPreference,
    resolved_preference: Option<RequestBackendPreference>,
    auto_gpu_policy: crate::ggml_runtime::AutoGpuPolicy,
    execution_context: &Arc<crate::RequestExecutionContext>,
    decode_progress: &DecodeProgress,
    slice_samples: u64,
) -> Result<GgmlAsrExecutionResult, BackendError> {
    let window = decode_progress.slice_progress_window(slice_samples);
    let reporter = decode_progress.reporter.clone();
    let observer =
        crate::api::backend::WorkProgressObserver::new(move |completed_work, total_work| {
            if should_publish_decode_work(completed_work, total_work) {
                reporter.report_fraction(decode_work_fraction(window, completed_work, total_work));
            }
        });
    let execution_context =
        Arc::new(execution_context.with_decode_work_progress_observer(observer));
    let result = run_dispatch_once(
        dispatch,
        execution_services,
        verified_pack,
        selected_family,
        chunk,
        request_options,
        backend_preference,
        resolved_preference,
        auto_gpu_policy,
        &execution_context,
    )?;
    decode_progress.complete_slice(slice_samples);
    Ok(result)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SliceExecutionFallback {
    failures: Vec<(ExecutionCandidate, ExecutionCandidateFailure)>,
    selected: ExecutionCandidate,
}

/// Runs one slice through the immutable execution plan. Every attempt covers
/// decoder-state planning plus the complete family dispatch. A later candidate
/// is tried only when the failing attempt's allocator/device boundary recorded
/// a typed candidate-local failure; ordinary decode/input/model errors fail
/// closed without inspecting their text. `AcceleratedOnly` and `Exact` plans
/// contain no CPU candidate, so this loop cannot weaken those user intents.
#[allow(clippy::too_many_arguments)]
fn run_dispatch_once_with_progress_and_policy(
    dispatch: &GgmlAsrExecutionDispatch,
    execution_services: &Arc<NativeExecutionServices>,
    verified_pack: &VerifiedPack,
    selected_family: &GgmlFamilyAdapterDescriptor,
    chunk: PcmSlice,
    request_options: GgmlAsrExecutionOptions,
    execution_plan: &ExecutionPlan,
    auto_gpu_policy: crate::ggml_runtime::AutoGpuPolicy,
    execution_context: &Arc<crate::RequestExecutionContext>,
    decode_progress: &DecodeProgress,
    slice_samples: u64,
    slice_label: &str,
) -> Result<(GgmlAsrExecutionResult, Option<SliceExecutionFallback>), BackendError> {
    let mut failures = Vec::new();
    let candidates = execution_plan.candidates();
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        let backend_preference = match candidate.placement {
            ExecutionPlacement::CpuOnly => GgmlAsrBackendPreference::CpuOnly,
            ExecutionPlacement::FullDevice | ExecutionPlacement::Hybrid => {
                GgmlAsrBackendPreference::Accelerated
            }
        };
        let attempt = crate::models::native_execution_services::run_execution_candidate_attempt(
            execution_services.as_ref(),
            candidate,
            || {
                run_dispatch_once_with_progress(
                    dispatch,
                    execution_services,
                    verified_pack,
                    selected_family,
                    chunk.clone(),
                    request_options.clone(),
                    backend_preference,
                    request_backend_preference_for_candidate(candidate),
                    auto_gpu_policy,
                    execution_context,
                    decode_progress,
                    slice_samples,
                )
            },
        );
        match (attempt.result, attempt.candidate_failure) {
            (Ok(result), None) => {
                let fallback = (!failures.is_empty()).then(|| SliceExecutionFallback {
                    failures,
                    selected: candidate.clone(),
                });
                return Ok((result, fallback));
            }
            (Err(error), None) => return Err(error),
            (result, Some(failure)) => {
                let error = crate::models::native_execution_services::execution_candidate_failure_source(result)
                    .unwrap_or_else(|| BackendError::NativeFailClosed {
                        reason: format!(
                            "execution candidate reported {:?} during '{}' despite returning success",
                            failure.kind, failure.operation
                        ),
                    });
                if candidate_index + 1 == candidates.len() {
                    return Err(error);
                }
                crate::stage_timing::log_detail_event(
                    "native_transcribe",
                    format_args!(
                        "stage=execution_candidate event=retry slice={slice_label} provider={} placement={:?} failure={:?} operation={}",
                        candidate.device.route.provider,
                        candidate.placement,
                        failure.kind,
                        failure.operation,
                    ),
                );
                failures.push((candidate.clone(), failure));
            }
        }
    }
    Err(BackendError::NativeFailClosed {
        reason: "execution policy produced no candidate attempts".to_string(),
    })
}

/// Upper bound on concurrent long-audio slice workers. Kept small: the win is
/// filling encode/decode GPU bubbles (2-4 in-flight slices saturate a single
/// GPU's execution pipeline, the same admission-concurrency effect the server
/// path already relies on), not unbounded fan-out, and every extra worker costs
/// another resident decoder runtime + KV cache.
const SLICE_PIPELINE_MAX_WIDTH: usize = 4;

/// Memory head-room the concurrent slice pipeline always leaves free when
/// deciding how many workers fit, so it never claims the last of available
/// memory and pushes the host into swap thrash.
const SLICE_PIPELINE_MEMORY_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

/// Floor for the per-worker memory estimate when the runtime pack size cannot be
/// stat'd, so the capacity gate never divides available memory by an
/// unrealistically small number and over-admits workers.
const SLICE_PIPELINE_PER_WORKER_BYTES_FLOOR: u64 = 256 * 1024 * 1024;

/// Explicit slice-pipeline width override from `OPENASR_SLICE_PIPELINE_WIDTH`.
///
/// `None` when the variable is unset or unparseable -- the carry-gated default
/// in [`slice_pipeline_requested_width`] then decides. A parsed value is
/// clamped to `1..=`[`SLICE_PIPELINE_MAX_WIDTH`], so "0" and "1" both mean an
/// explicit serial pin. The override wins in both directions: it can force the
/// concurrent path onto a carry-active run (accepting the carry-light quality
/// cost) and force serial onto a carry-disabled run.
fn slice_pipeline_explicit_width() -> Option<usize> {
    std::env::var("OPENASR_SLICE_PIPELINE_WIDTH")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .map(|value| value.clamp(1, SLICE_PIPELINE_MAX_WIDTH))
}

/// Requested concurrent slice-pipeline width for one run, gated on that run's
/// normalized effective prompt-carry state ([`longform_prompt_carry_mode`],
/// which already folds the request option and the family's decode policy
/// together -- deliberately not a per-family list). The provider gate in
/// [`effective_slice_pipeline_width`] then keeps the automatic concurrent path
/// on independently addressable discrete-GPU lanes; CPU and unified-memory
/// Metal already saturate their shared compute/memory domain inside one decode
/// and default to serial slices.
///
/// - Carry `Disabled`: the serial loop threads no cross-slice prompt anyway,
///   so the carry-light concurrent path is transcript-equivalent (proven
///   byte-identical by `concurrent_slice_pipeline_equivalence`). Default to
///   [`SLICE_PIPELINE_MAX_WIDTH`] and let the capacity and slice-count gates
///   in [`effective_slice_pipeline_width`] pick what actually fits.
/// - Carry active (`Text` / `TokenHistory`): the concurrent path would drop
///   the carry and change the transcript (the short-audio audit measured
///   whole-clause deletions), so the default stays 1 -- the byte-identical
///   serial + prompt-carry path. Only an explicit
///   `OPENASR_SLICE_PIPELINE_WIDTH>=2` overrides that, and the run then
///   records the dropped carry in its provenance.
fn slice_pipeline_requested_width(carry_prompt_mode: LongformPromptCarryMode) -> usize {
    if let Some(explicit) = slice_pipeline_explicit_width() {
        return explicit;
    }
    match carry_prompt_mode {
        LongformPromptCarryMode::Disabled => SLICE_PIPELINE_MAX_WIDTH,
        LongformPromptCarryMode::Text | LongformPromptCarryMode::TokenHistory => 1,
    }
}

/// Pure capacity gate for the concurrent slice pipeline: how many workers may
/// run at once given `available_bytes` of head-room and a conservative
/// `per_worker_bytes` estimate. Returns >= 1 and never exceeds `requested_width`
/// or `decode_slice_count`. When nothing fits it falls back to 1 (serial), never
/// 0 -- the gate can only ever reduce concurrency, so it cannot OOM the host.
fn slice_pipeline_capped_width(
    requested_width: usize,
    decode_slice_count: usize,
    available_bytes: Option<u64>,
    per_worker_bytes: u64,
    reserve_bytes: u64,
) -> usize {
    let ceiling = requested_width.min(decode_slice_count);
    if ceiling <= 1 {
        return 1;
    }
    // No memory probe on this host: honor the requested width rather than
    // silently disabling it, matching the serve-batch VRAM-cap precedent
    // (`serve_batch_vram_capped_max_batch` returns the request unchanged when no
    // memory sample is available). The reserve plus the conservative per-worker
    // estimate still bound the real risk.
    let Some(available) = available_bytes else {
        return ceiling;
    };
    if per_worker_bytes == 0 {
        return ceiling;
    }
    let usable = available.saturating_sub(reserve_bytes);
    let fits = (usable / per_worker_bytes).min(ceiling as u64) as usize;
    fits.max(1)
}

/// Conservative per-worker memory estimate for the capacity gate: one runtime
/// pack's on-disk size (with a floor). The mmapped weights are actually shared
/// across workers, so charging each worker a whole pack over-estimates the true
/// marginal cost (KV cache + compute buffers) and errs toward fewer workers --
/// the safe direction for an OOM gate.
fn slice_pipeline_per_worker_bytes(runtime_preflight: &GgufRuntimeSourcePreflight) -> u64 {
    // Size the exact mapped generation already proven by preflight. Re-stating
    // the display path could observe a replacement and would also add a system
    // call to every request for information the pinned source already owns.
    let pack_bytes = runtime_preflight.runtime_source.byte_len();
    pack_bytes.max(SLICE_PIPELINE_PER_WORKER_BYTES_FLOOR)
}

/// Automatic slice concurrency is only enabled for independently addressable
/// discrete-GPU providers. CPU workers compete for the same cores, while ggml
/// Metal already uses command-buffer concurrency and every extra slice creates
/// another large runtime in the same unified-memory domain. On both routes the
/// observed result is higher RSS without a latency win, and cold candidates can
/// also make each other fall back. CUDA/HIP/Vulkan retain the bubble-filling
/// path. An explicit `OPENASR_SLICE_PIPELINE_WIDTH` remains an operator escape
/// hatch and bypasses this default provider cap.
fn slice_pipeline_default_provider_width(
    requested_width: usize,
    provider: crate::ExecutionProvider,
) -> usize {
    match provider {
        crate::ExecutionProvider::Cuda
        | crate::ExecutionProvider::Hip
        | crate::ExecutionProvider::Vulkan => requested_width,
        crate::ExecutionProvider::Cpu
        | crate::ExecutionProvider::Metal
        | crate::ExecutionProvider::Accelerator
        | crate::ExecutionProvider::Unknown => 1,
    }
}

/// Concurrent-width decision wired to the live host: caps `requested_width` by
/// swap-aware available memory ([`crate::host::host_available_memory_bytes`], the
/// capacity source) against the conservative per-worker estimate, and by the
/// slice count. Returns 1 (serial) whenever concurrency is not worth it or does
/// not fit. `slices.len()` is an upper bound on decodable slices (some may be
/// suppressed as silent at run time); the gate only ever caps downward, so the
/// bound is safe.
fn effective_slice_pipeline_width(
    requested_width: usize,
    slices: &[crate::longform::AudioSlice],
    runtime_preflight: &GgufRuntimeSourcePreflight,
    execution_plan: &ExecutionPlan,
) -> usize {
    let requested_width = if slice_pipeline_explicit_width().is_some() {
        requested_width
    } else {
        execution_plan
            .candidates()
            .first()
            .map(|candidate| {
                slice_pipeline_default_provider_width(
                    requested_width,
                    candidate.device.route.provider,
                )
            })
            .unwrap_or(1)
    };
    if requested_width <= 1 || slices.len() <= 1 {
        return 1;
    }
    slice_pipeline_capped_width(
        requested_width,
        slices.len(),
        crate::host::host_available_memory_bytes(),
        slice_pipeline_per_worker_bytes(runtime_preflight),
        SLICE_PIPELINE_MEMORY_RESERVE_BYTES,
    )
}

/// One slice's place in the concurrent pipeline: the slice itself, its sample
/// weight for progress, and whether it was suppressed as silent (silence is
/// decided once up front on the main thread, exactly as the serial loop does).
struct SlicePlanItem {
    slice: crate::longform::AudioSlice,
    slice_samples: u64,
    silent: bool,
}

/// A worker's decoded output for one slice, carried back to the main thread for
/// in-order assembly. Deliberately owns only the plain data the ordered
/// integration needs -- text, segments, truncation, GPU-fallback tag -- so
/// nothing family-specific or non-`Send` crosses the thread boundary.
struct DecodedSlice {
    text: String,
    segments: Vec<Segment>,
    truncation: Option<DecodeTruncation>,
    fallback: Option<SliceExecutionFallback>,
}

/// Borrowed context for one concurrent long-audio slice-pipeline run. Grouped
/// into a struct so the entry point stays one readable call instead of a
/// twenty-argument function.
struct ConcurrentSlicePipeline<'a> {
    width: usize,
    slices: Vec<crate::longform::AudioSlice>,
    plan_audio: &'a PcmBuffer,
    dispatch: &'a GgmlAsrExecutionDispatch,
    execution_services: &'a Arc<NativeExecutionServices>,
    verified_pack: &'a VerifiedPack,
    selected_family: &'a GgmlFamilyAdapterDescriptor,
    request_options: &'a GgmlAsrExecutionOptions,
    execution_plan: &'a ExecutionPlan,
    auto_gpu_policy: crate::ggml_runtime::AutoGpuPolicy,
    execution_context: &'a Arc<crate::RequestExecutionContext>,
    longform_options: &'a crate::LongFormOptions,
    speaker_plan: SpeakerPlan,
    decode_progress: &'a DecodeProgress,
    assembler: &'a mut TranscriptAssembler,
    ran_any_slice: &'a mut bool,
    suppressed_slice_count: &'a mut usize,
    degraded_slice_fallbacks: &'a mut Vec<(usize, SliceExecutionFallback)>,
    truncated_slices: &'a mut Vec<String>,
    truncated_decodes: &'a mut Vec<TruncatedDecode>,
    speaker_scope_count: &'a mut usize,
}

/// Long-audio slice pipeline: decode up to `width` slices concurrently and
/// assemble their results in slice order.
///
/// This is the carry-light path (see module notes on `carry_prompt_mode`): the
/// cross-slice prompt carry the serial loop threads between slices is a strict
/// serial dependency, so the concurrent path drops it -- slice N+1 no longer
/// waits on slice N's transcript. The output is otherwise assembled from the
/// same per-slice results, in the same order, so it is byte-identical to the
/// serial path except where a family's decode genuinely depended on the carried
/// prompt.
///
/// The five correctness properties the concurrent path must preserve:
/// 1. Ordered assembly: workers finish out of order, results are routed back by
///    slice position and integrated strictly in slice order.
/// 2. Cancel / pause: each worker gates on the shared control at every slice
///    boundary (pause blocks it, cancel stops it), arms the ggml abort callback
///    on its own thread for mid-graph cancel, and the shared greedy driver still
///    polls cancel per token via the job-carried control.
/// 3. Progress: `DecodeProgress` accumulates atomically and the registry clamps
///    every report upward, so concurrent completions never move the bar back.
/// 4. Memory: `width` is already capacity-gated by the caller
///    (`effective_slice_pipeline_width`).
/// 5. Errors / truncation: a worker's error and truncated-slice facts are routed
///    back and integrated in order; the first (lowest-index) error fails the run
///    closed, exactly like the serial `?`.
fn run_concurrent_slice_pipeline(pipeline: ConcurrentSlicePipeline) -> Result<(), BackendError> {
    let ConcurrentSlicePipeline {
        width,
        slices,
        plan_audio,
        dispatch,
        execution_services,
        verified_pack,
        selected_family,
        request_options,
        execution_plan,
        auto_gpu_policy,
        execution_context,
        longform_options,
        speaker_plan,
        decode_progress,
        assembler,
        ran_any_slice,
        suppressed_slice_count,
        degraded_slice_fallbacks,
        truncated_slices,
        truncated_decodes,
        speaker_scope_count,
    } = pipeline;

    // Pre-scan on the main thread: decide silence once (identical predicate to
    // the serial loop), fold each silent slice's share into progress up front,
    // and record which positions actually need a decode worker.
    let mut plan_items: Vec<SlicePlanItem> = Vec::with_capacity(slices.len());
    let mut decode_positions: Vec<usize> = Vec::new();
    for slice in slices {
        let slice_samples = slice.duration_samples() as u64;
        let relative_start = slice
            .content_start_sample
            .saturating_sub(slice.start_sample);
        let relative_end = slice
            .content_end_sample
            .saturating_sub(slice.start_sample)
            .min(slice.duration_samples());
        let chunk = &plan_audio[slice.start_sample..slice.end_sample];
        let silent = longform_options.suppress_silent_slices
            && is_effectively_silent(
                &chunk[relative_start..relative_end],
                longform_options.energy_silence_threshold_db,
            );
        if silent {
            decode_progress.complete_slice(slice_samples);
        } else {
            decode_positions.push(plan_items.len());
        }
        plan_items.push(SlicePlanItem {
            slice,
            slice_samples,
            silent,
        });
    }

    // Results routed back by slice position; silent positions stay `None`.
    let mut results: Vec<Option<Result<DecodedSlice, BackendError>>> =
        (0..plan_items.len()).map(|_| None).collect();

    if !decode_positions.is_empty() {
        let cursor = AtomicUsize::new(0);
        // Set on cancel or the first worker error so peers stop pulling new work
        // promptly instead of decoding slices whose result will be discarded.
        let stop = AtomicBool::new(false);
        let worker_count = width.min(decode_positions.len()).max(1);
        let (result_tx, result_rx) = mpsc::channel::<(usize, Result<DecodedSlice, BackendError>)>();
        let items = &plan_items;
        let decode_positions_ref = &decode_positions;
        let cursor_ref = &cursor;
        let stop_ref = &stop;
        // `thread::scope` does not inherit TLS. Capture the complete request
        // execution context once so every slice worker keeps the same broker,
        // Exact route namespace, telemetry, and transactional observation
        // parent as the request thread. Candidate attempts installed below the
        // worker then publish into that parent instead of silently losing the
        // evidence stream on concurrent long audio.
        let native_execution_context =
            crate::models::native_execution_services::current_native_execution_context();
        // The orchestration thread can sit above the per-candidate service
        // scope, so `current_native_execution_context()` is legitimately None
        // while an audit sink is still installed around the whole request.
        // Carry that parent independently; the worker's candidate transaction
        // will derive its attempt-local sink from it.
        let execution_observation_sink =
            crate::models::native_execution_services::current_execution_observation_sink();
        std::thread::scope(|scope| {
            for _ in 0..worker_count {
                let result_tx = result_tx.clone();
                let native_execution_context = native_execution_context.clone();
                let execution_observation_sink = execution_observation_sink.clone();
                scope.spawn(move || {
                    let _native_execution_context = native_execution_context.map(
                        crate::models::native_execution_services::install_native_execution_context,
                    );
                    let _execution_observation_sink = execution_observation_sink.map(
                        crate::models::native_execution_services::install_execution_observation_sink,
                    );
                    // Arm this worker thread's ggml abort callback when the
                    // request has a cancel source, so a mid-graph cancel aborts
                    // this worker too. Between-step cancel is already covered
                    // by the shared greedy driver's per-token control poll.
                    let _abort_guard = execution_context
                        .control
                        .arm_for_native_decode_if_cancellable();
                    loop {
                        if stop_ref.load(Ordering::Relaxed) {
                            break;
                        }
                        // Slice-boundary pause/cancel gate, mirroring the serial
                        // loop: pause blocks this worker here; cancel stops it.
                        if execution_context.control.wait_at_slice_boundary()
                            == super::transcription_control::SliceBoundaryControl::Canceled
                        {
                            stop_ref.store(true, Ordering::Relaxed);
                            break;
                        }
                        let next = cursor_ref.fetch_add(1, Ordering::Relaxed);
                        if next >= decode_positions_ref.len() {
                            break;
                        }
                        let pos = decode_positions_ref[next];
                        let item = &items[pos];
                        // Carry-light: no cross-slice prompt carry in the
                        // concurrent path (that is the serial dependency this
                        // path trades away for overlap).
                        let slice_options = request_options.clone();
                        let chunk =
                            plan_audio.slice(item.slice.start_sample..item.slice.end_sample);
                        let outcome = run_dispatch_once_with_progress_and_policy(
                            dispatch,
                            execution_services,
                            verified_pack,
                            selected_family,
                            chunk,
                            slice_options,
                            execution_plan,
                            auto_gpu_policy,
                            execution_context,
                            decode_progress,
                            item.slice_samples,
                            &format!("concurrent-pos={pos}"),
                        )
                        .map(|(result, fallback)| DecodedSlice {
                            text: result.transcription.text,
                            segments: result.transcription.segments,
                            truncation: result.decode_truncation,
                            fallback,
                        });
                        let is_err = outcome.is_err();
                        if result_tx.send((pos, outcome)).is_err() {
                            break;
                        }
                        if is_err {
                            // First failure stops the pipeline (property 6):
                            // peers stop pulling, and the main thread returns the
                            // lowest-index error, matching the serial `?`.
                            stop_ref.store(true, Ordering::Relaxed);
                            break;
                        }
                    }
                });
            }
            // Drop the main thread's spare sender so the receiver below ends once
            // every worker has finished and dropped its clone.
            drop(result_tx);
            for (pos, outcome) in result_rx {
                results[pos] = Some(outcome);
            }
        });
    }

    // Ordered integration on the main thread (property 1): replay the serial
    // loop's post-decode bookkeeping in slice order, so `slice_index`, the
    // provenance vectors, and the assembler see exactly the serial sequence.
    // `slice_index` is bumped only on a decoded (non-silent) slice, exactly as
    // the serial loop does, so the 1-based indices stamped into truncation and
    // GPU-fallback provenance match byte-for-byte.
    let mut slice_index = 0usize;
    let mut first_error: Option<BackendError> = None;
    for (position, item) in plan_items.into_iter().enumerate() {
        if item.silent {
            *suppressed_slice_count += 1;
            assembler.push_slice_result(SliceTranscript {
                slice: item.slice,
                text: String::new(),
                segments: Vec::new(),
                time_domain: SegmentTimeDomain::AbsoluteOriginal,
            });
            continue;
        }
        match results[position].take() {
            Some(Ok(decoded)) => {
                slice_index += 1;
                if let Some(fallback) = decoded.fallback {
                    degraded_slice_fallbacks.push((slice_index, fallback));
                }
                if let Some(truncation) = decoded.truncation {
                    truncated_slices
                        .push(format_truncated_slice_provenance(slice_index, &truncation));
                    truncated_decodes.push(TruncatedDecode {
                        slice_index: Some(slice_index),
                        truncation,
                    });
                }
                *ran_any_slice = true;
                let transcript = SliceTranscript {
                    slice: item.slice,
                    text: decoded.text,
                    segments: decoded.segments,
                    time_domain: SegmentTimeDomain::RelativeToSliceContent,
                };
                if speaker_plan == SpeakerPlan::InDecoder {
                    let scope = *speaker_scope_count;
                    *speaker_scope_count += 1;
                    assembler.push_slice_result_with_speaker_scope(transcript, scope);
                } else {
                    assembler.push_slice_result(transcript);
                }
            }
            // Keep the first (lowest-index) error so the returned failure matches
            // the serial `?`, which fails at the earliest bad slice.
            Some(Err(err)) if first_error.is_none() => {
                first_error = Some(err);
            }
            Some(Err(_)) => {}
            None => {
                // A decodable position with no result only happens when a worker
                // stopped early (a peer error already set `first_error`, or a
                // cancel, checked below). Nothing to integrate here.
            }
        }
    }

    // Cancel wins over a partial assembly (property 2 + 6): a cancel that raced
    // the workers leaves some positions undecoded, so surface it as the typed
    // cancel rather than a truncated transcript.
    if execution_context.is_canceled() {
        return Err(BackendError::TranscriptionCanceled);
    }
    if let Some(err) = first_error {
        return Err(err);
    }
    Ok(())
}

/// RAII cleanup for one native transcription's progress-registry entry:
/// removes it on normal completion, an early `?` return, or a panic, so a
/// finished run's progress is never read as still in-flight. Created once per
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LongformPromptCarryMode {
    Disabled,
    Text,
    TokenHistory,
}

#[derive(Debug, Clone, PartialEq)]
struct NativeLongformPolicyResolution {
    options: crate::LongFormOptions,
    provenance: Vec<String>,
}

/// The offline decode result plus the immutable facts selected during the one
/// model preflight. Post-processing consumes this outcome rather than opening
/// the primary model path again to rediscover a descriptor capability.
struct NativeTranscriptionOutcome {
    transcription: Transcription,
    prepared_audio: PcmBuffer,
    emits_punctuation: Option<bool>,
    speaker_finalization: SpeakerFinalizationContext,
    /// Resolved progress weights (actual backend + segmenter after prepare).
    progress_backend: ProgressBackendClass,
    progress_segmenter: ProgressSegmenterKind,
}

struct SpeakerFinalizationContext {
    attribution: SpeakerAttribution,
    embedder: Option<Arc<dyn crate::diarize::embed::SpeakerEmbedder>>,
    plan: SpeakerPlan,
    scope_by_segment: Vec<Option<usize>>,
    /// Retained for decode-path bookkeeping (`word_timestamps_forced_for_diarization`).
    /// Word stripping after projection is decided from request keep-words policy.
    #[allow(dead_code)]
    strip_forced_word_timestamps: bool,
}

impl SpeakerFinalizationContext {
    /// External Voice ID needs word anchors when a multi-speaker segment must
    /// be split for text ownership. Empty words or present-but-unreliable
    /// anchors both force FA / fail-closed; single-speaker identity alone does not.
    fn requires_word_alignment(
        &self,
        transcription: &Transcription,
        word_anchors_reliable: bool,
    ) -> bool {
        self.plan == SpeakerPlan::External
            && crate::diarize::attribution::requires_word_alignment(
                &self.attribution.timeline.turns,
                &transcription.segments,
                word_anchors_reliable,
            )
    }
}

/// Entry point for the native backend: prepares the ordinary decode/longform
/// result (`run_native_transcription_impl`), then --
/// gated on the resolved model's `emits_punctuation` capability and the
/// request's `punctuate` opt-out -- restores punctuation with the installed
/// FireRedPunc capability pack, then -- only when the request opted into
/// `--word-timestamps=aligned` (`word_timestamps_refine`), or when external
/// speaker attribution discovers a coarse multi-speaker segment -- refines
/// the finished transcript's per-word timestamps with the installed
/// Qwen3-ForcedAligner-0.6B capability pack. Kept as a thin wrapper rather
/// than threading either post-process into the (already long) decode/longform
/// function: both re-read only the finished transcript (the aligner also
/// re-reads the audio file), so neither has a dependency on any intermediate
/// state that function computes. Punctuation runs before the forced-aligner
/// refine so the aligner (and every other downstream consumer) sees the
/// punctuated text.
/// Thin wrapper over [`run_native_transcription_fallible`] that adds exactly
/// one thing: a `stage=transcribe_failure` `daemon.log` line on the `Err`
/// path, carrying the failure's category and the process's current
/// available-memory (and, if applicable, VRAM) reading. Kept as a separate
/// wrapper rather than folding the logging into the fallible function itself
/// so every one of that function's many early-return `?` sites (model
/// resolve, audio prep, capability rejection, decode dispatch, ...) is
/// covered by one log site instead of needing its own.
pub(super) fn run_native_transcription(
    request: TranscriptionRequest,
    execution_services: Arc<NativeExecutionServices>,
) -> Result<Transcription, BackendError> {
    run_native_transcription_with_intent(request, execution_services, None)
}

pub(super) fn run_native_transcription_with_intent(
    request: TranscriptionRequest,
    execution_services: Arc<NativeExecutionServices>,
    execution_intent: Option<ExecutionIntent>,
) -> Result<Transcription, BackendError> {
    run_native_transcription_fallible(request, &execution_services, execution_intent).inspect_err(
        |error| {
            log_failure_context(classify_backend_error_for_failure_log(error));
        },
    )
}

pub(super) fn run_native_transcription_with_verified_pack(
    request: TranscriptionRequest,
    execution_services: Arc<NativeExecutionServices>,
    execution_intent: Option<ExecutionIntent>,
    verified_pack: Arc<crate::models::pack_verifier::VerifiedPack>,
) -> Result<Transcription, BackendError> {
    run_native_transcription_fallible_with_input(
        request,
        &execution_services,
        execution_intent,
        NativeRuntimePackInput::Verified(verified_pack),
    )
    .inspect_err(|error| {
        log_failure_context(classify_backend_error_for_failure_log(error));
    })
}

enum NativeRuntimePackInput {
    /// Untrusted ingress used by the direct `TranscriptionBackend` interface.
    CandidatePath,
    /// Exact open generation already proven while resolving the product
    /// `NativeRuntimeModelAdapter`.
    Verified(Arc<crate::models::pack_verifier::VerifiedPack>),
}

/// Coarse [`FailureCategory`] bucket for a final `BackendError`. Candidate
/// retry decisions use the typed attempt-local failure sink and never this
/// diagnostic classification.
fn classify_backend_error_for_failure_log(error: &BackendError) -> FailureCategory {
    match error {
        BackendError::NativeUnsupportedInputFormat { .. } => FailureCategory::AudioIo,
        BackendError::NativeModelPackPathRequired
        | BackendError::NativeModelPackPathRejected { .. }
        | BackendError::NativeModelSelectionMismatch { .. } => FailureCategory::ModelResolve,
        BackendError::VoiceIdUnsupportedForRealtime { .. }
        | BackendError::DiarizationNotSupported { .. }
        | BackendError::DiarizationSegmenterUnavailable
        | BackendError::VoiceIdIdentityFailed(_)
        | BackendError::DiarizeSpeakersRequiresDiarization
        | BackendError::PhraseBiasNotSupported { .. }
        | BackendError::AdapterNotSupported { .. }
        | BackendError::PhraseBiasUnsupportedByModel { .. }
        | BackendError::RequestOptionUnsupportedByModel { .. }
        | BackendError::WordTimestampAlignmentRequiresWordTimestamps
        | BackendError::WordTimestampAlignmentPackMissing { .. }
        | BackendError::ExecutionDeviceNotFound { .. }
        | BackendError::ExecutionDeviceNotAddressable { .. }
        | BackendError::ExecutionDeviceInitFailed { .. } => FailureCategory::UnsupportedCapability,
        BackendError::TranscriptionCanceled => FailureCategory::Canceled,
        BackendError::ServeBatchUnavailable { .. } => FailureCategory::Transient,
        // A pre-emptive admission rejection of the same failure class a raw
        // ggml allocation error represents, just caught before the graph
        // build instead of during it.
        BackendError::NativeInsufficientHostMemory { .. } => FailureCategory::Alloc,
        BackendError::NativeFailClosed { .. }
        | BackendError::ExternalDiarizationFailed { .. }
        | BackendError::WordTimestampAlignmentFailed { .. } => FailureCategory::Decode,
    }
}

fn run_native_transcription_fallible(
    request: TranscriptionRequest,
    execution_services: &Arc<NativeExecutionServices>,
    execution_intent: Option<ExecutionIntent>,
) -> Result<Transcription, BackendError> {
    run_native_transcription_fallible_with_input(
        request,
        execution_services,
        execution_intent,
        NativeRuntimePackInput::CandidatePath,
    )
}

fn run_native_transcription_fallible_with_input(
    request: TranscriptionRequest,
    execution_services: &Arc<NativeExecutionServices>,
    execution_intent: Option<ExecutionIntent>,
    runtime_pack_input: NativeRuntimePackInput,
) -> Result<Transcription, BackendError> {
    if let Some(requested) = request.diarize_speakers {
        let max = crate::diarize::contract::MAX_DIARIZATION_SPEAKERS;
        if !(1..=max).contains(&requested) {
            return Err(BackendError::NativeFailClosed {
                reason: format!(
                    "The speakers hint must be between 1 and {max}, got {requested}. The request was rejected instead of silently clamping it to a different diarization workload."
                ),
            });
        }
    }
    if request.voice_id && !request.source.supports_recording_voice_id() {
        return Err(BackendError::VoiceIdUnsupportedForRealtime {
            request_source: request.source.as_log_label(),
        });
    }
    let refine = request.word_timestamps_refine;
    if refine && !request.word_timestamps {
        return Err(BackendError::WordTimestampAlignmentRequiresWordTimestamps);
    }
    // Captured before `request` is moved into `run_native_transcription_impl`
    // below: `publish_align_progress` after that call still needs this
    // request's transcription id.
    let execution_context = Arc::clone(&request.execution_context);
    // Own graph-level cancellation at the shared native-core boundary. Every
    // caller that supplies a cancellable context now publishes the request's
    // flag for synchronous graph compute on this thread; detached contexts
    // remain callback-free. Concurrent longform workers install the same flag
    // separately because TLS does not cross thread boundaries.
    let _abort_callback_guard = execution_context
        .control
        .arm_for_native_decode_if_cancellable();
    if execution_context.is_canceled() {
        return Err(BackendError::TranscriptionCanceled);
    }
    // Spans the whole run so this request's progress-registry entry is removed
    // on every exit (completion, cancel, error, panic unwind).
    let _progress_handle = ProgressRegistryHandle::new(execution_context.request_id.clone());
    let language_hint = request.language.clone();
    let punctuate = request.punctuate;
    let explicit_refine = request.word_timestamps_refine;
    let timeline_precision = request.timeline_precision;
    let needs_subtitle_export = request.needs_subtitle_export;
    let request_word_timestamps = request.word_timestamps;
    let voice_id = request.voice_id;
    let may_align = request_may_need_align(&request);
    let segmenter_kind = ProgressSegmenterKind::from_preference(request.voice_id_segmenter);
    // Every independent native model stage resolves from this same immutable
    // product intent. Each stage still owns its own capability matrix and
    // candidate transaction; no auxiliary model inherits a coarse backend or
    // re-reads process defaults after the main ASR dispatch completes.
    let request_execution_intent = execution_intent
        .clone()
        .unwrap_or_else(|| request_execution_intent(request.execution_target));
    let backend_class = progress_backend_class(&request_execution_intent);
    // Provisional plan: duration and external-diarize are refined inside impl
    // once audio is prepared and the family speaker plan is known. Stages that
    // cannot run stay off the plan so their weight never dilutes overall.
    let provisional_plan = ProgressPlan::build(ProgressPlanInput {
        audio_duration_s: 0.0,
        voice_id,
        external_diarize: voice_id, // refined to external-only after family select
        segmenter: segmenter_kind,
        punctuate: false, // refined after emits_punctuation is known
        align: may_align,
        backend: backend_class,
        persist: false,
    });
    let progress =
        ProgressReporter::install(execution_context.request_id.clone(), provisional_plan);
    // Coarse per-request stage timing: "inference" spans model resolution +
    // audio prep + decode/longform-assembly; "postprocess" covers punctuation
    // and forced-align.
    let inference_started = Instant::now();
    let NativeTranscriptionOutcome {
        transcription,
        prepared_audio,
        emits_punctuation,
        speaker_finalization,
        progress_backend: backend_class,
        progress_segmenter: segmenter_kind,
    } = run_native_transcription_impl(
        request,
        execution_services,
        Some(request_execution_intent.clone()),
        runtime_pack_input,
        &progress,
    )?;
    if execution_context.is_canceled() {
        return Err(BackendError::TranscriptionCanceled);
    }
    crate::stage_timing::log_stage(
        "native_transcribe",
        "inference",
        inference_started.elapsed(),
    );
    let postprocess_started = Instant::now();
    let will_punctuate = should_run_punctuation_stage(punctuate, emits_punctuation);
    // Rebuild plan with known duration / punctuation before postprocess so
    // overall weights match the stages that will actually run. Align weight
    // stays if `may_align`; if the decision later skips align, project simply
    // finishes the remaining bar without inventing intermediate ticks.
    // Backend class and segmenter kind are the values resolved inside impl
    // (actual accelerated device + prepared DiariZen/pyannote), not the
    // provisional request preference / intent-only Auto mapping.
    let audio_duration_s = prepared_audio.len() as f32 / 16_000.0;
    let external_diarize = speaker_finalization.plan == SpeakerPlan::External;
    let voice_id = speaker_finalization.plan != SpeakerPlan::Off;
    progress.replace_plan(ProgressPlan::build(ProgressPlanInput {
        audio_duration_s,
        voice_id,
        external_diarize,
        segmenter: segmenter_kind,
        punctuate: will_punctuate,
        align: may_align,
        backend: backend_class,
        persist: false,
    }));
    let transcription = apply_punctuation_stage_with_policy(
        transcription,
        emits_punctuation,
        punctuate,
        execution_services,
        &request_execution_intent,
        execution_context.as_ref(),
        &progress,
    )?;
    let native_validation =
        crate::subtitle::validate_word_anchors(&transcription, audio_duration_s);
    let voice_id_needs_align = speaker_finalization
        .requires_word_alignment(&transcription, native_validation.is_reliable());
    let align_decision = crate::subtitle::decide_forced_alignment(
        timeline_precision,
        explicit_refine,
        voice_id_needs_align,
        needs_subtitle_export,
        &native_validation,
    );
    let mut timeline_quality = if align_decision.native_reliable {
        crate::subtitle::TimelineQuality::NativeReliable
    } else {
        crate::subtitle::TimelineQuality::NativeApproximate
    };
    let transcription = if align_decision.need_align {
        // Forced alignment is a separate heavyweight model phase. Unload idle
        // primary ASR caches so the aligner can quote its graph against free
        // headroom. V1 realigns the whole transcript when validation fails.
        execution_services.unload_idle_native_model_runtime_caches();
        if !may_align {
            // Late decision to align: ensure plan carries align weight.
            progress.replace_plan(ProgressPlan::build(ProgressPlanInput {
                audio_duration_s,
                voice_id,
                external_diarize,
                segmenter: segmenter_kind,
                punctuate: will_punctuate,
                align: true,
                backend: backend_class,
                persist: false,
            }));
        }
        let refined = refine_transcription_word_timestamps_with_forced_aligner_policy(
            transcription,
            forced_aligner_audio_view(&prepared_audio, true)
                .expect("enabled forced alignment retains the normalized PCM view"),
            language_hint.as_deref(),
            execution_services,
            &request_execution_intent,
            execution_context.as_ref(),
            Some(&progress),
        )?;
        timeline_quality = crate::subtitle::TimelineQuality::ForcedAligned;
        refined
    } else if may_align {
        // Planned align was skipped: drop its weight so overall can finish.
        progress.replace_plan(ProgressPlan::build(ProgressPlanInput {
            audio_duration_s,
            voice_id,
            external_diarize,
            segmenter: segmenter_kind,
            punctuate: will_punctuate,
            align: false,
            backend: backend_class,
            persist: false,
        }));
        transcription
    } else {
        transcription
    };
    let result = finalize_native_transcription(
        transcription,
        &speaker_finalization,
        prepared_audio.as_slice(),
        timeline_quality,
        request_word_timestamps,
        explicit_refine,
        timeline_precision,
        &progress,
    );
    crate::stage_timing::log_stage(
        "native_transcribe",
        "postprocess",
        postprocess_started.elapsed(),
    );
    if execution_context.is_canceled() {
        Err(BackendError::TranscriptionCanceled)
    } else {
        result
    }
}

/// Whether the punctuation-restoration stage should attempt to run: the
/// request has not opted out (`punctuate`, the desktop preference toggle) AND
/// the resolved model's `emits_punctuation` capability is honestly `Some(false)`
/// (see [`should_apply_punctuation`]) -- a model that already punctuates, or
/// whose capability is unknown, is never re-punctuated.
fn should_run_punctuation_stage(punctuate: bool, emits_punctuation: Option<bool>) -> bool {
    punctuate && should_apply_punctuation(emits_punctuation)
}

/// Punctuation-restoration post-process: runs only for an ASR result the
/// catalog honestly declares unpunctuated, and only when the FireRedPunc
/// capability pack is installed. Fail-closed by design -- a missing pack, a
/// corrupt pack, or a classifier failure all leave `transcription` exactly as
/// the ASR family produced it rather than crashing the request or fabricating
/// punctuation; the native backend never downloads this pack.
#[cfg(test)]
fn apply_punctuation_stage_if_applicable(
    transcription: Transcription,
    emits_punctuation: Option<bool>,
    punctuate: bool,
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
) -> Transcription {
    if !should_run_punctuation_stage(punctuate, emits_punctuation) {
        return transcription;
    }
    let Some(punc_pack_path) = resolve_firered_punc_pack_path() else {
        return transcription;
    };
    let Ok(runtime) = FireRedPuncRuntime::from_pack(&punc_pack_path, backend) else {
        return transcription;
    };
    punctuate_transcription_segments(transcription, &runtime)
}

fn apply_punctuation_stage_with_policy(
    transcription: Transcription,
    emits_punctuation: Option<bool>,
    punctuate: bool,
    execution_services: &NativeExecutionServices,
    request_intent: &ExecutionIntent,
    execution_context: &crate::RequestExecutionContext,
    progress: &ProgressReporter,
) -> Result<Transcription, BackendError> {
    if execution_context.is_canceled() {
        return Err(BackendError::TranscriptionCanceled);
    }
    if !should_run_punctuation_stage(punctuate, emits_punctuation) {
        return Ok(transcription);
    }
    progress.enter_stage(TranscriptionStage::Punctuate);
    let Some(punc_pack_path) = resolve_firered_punc_pack_path() else {
        return Ok(transcription);
    };
    let Ok(verified_pack) = PackVerifier.verify_candidate(PackCandidate::new(&punc_pack_path))
    else {
        return Ok(transcription);
    };
    if !matches!(
        verified_pack.route(),
        PackRoute::Aux {
            kind: AuxPackKind::Punctuation,
            ..
        }
    ) {
        return Ok(transcription);
    }
    let prepared_preflight = verified_pack.preflight();
    let prepared_content_id = prepared_preflight.runtime_source.content_id().to_string();
    let execution_plan = resolve_auxiliary_execution_plan(
        execution_services,
        crate::models::firered_punc::config::FIRERED_PUNC_ARCHITECTURE_VALUE,
        request_intent,
    )?;
    let result = run_auxiliary_stage_with_policy(
        execution_services,
        &execution_plan,
        "firered-punctuation",
        |candidate| {
            // Punctuation is an optional accuracy stage: malformed/missing
            // runtime errors keep the ASR output unchanged. Candidate-local
            // allocator/device failures are still recorded by the graph
            // boundary; the stage policy sees that typed side channel and
            // retries instead of silently accepting the no-op.
            let runtime = match load_actor(
                execution_services,
                prepared_preflight,
                &prepared_content_id,
                candidate,
            ) {
                Ok(runtime) => runtime,
                Err(error)
                    if execution_context.is_canceled()
                        || is_cooperative_cancel_reason(&error.to_string()) =>
                {
                    return Err(BackendError::TranscriptionCanceled);
                }
                Err(_) => return Ok(transcription.clone()),
            };
            punctuate_transcription_segments_with_actor(
                transcription.clone(),
                &runtime,
                execution_context,
                progress,
            )
        },
    );
    let out = finish_optional_punctuation_stage(transcription, result)?;
    progress.complete_stage();
    Ok(out)
}

/// Product-policy boundary for FireRedPunc. The planner has already exhausted
/// only semantics-equivalent execution candidates when it returns
/// `CandidatesExhausted`; because punctuation is an automatic enhancement,
/// that expected resource failure (or an ordinary optional-model failure)
/// preserves the ASR result. Internal planner invariants remain fatal.
fn finish_optional_punctuation_stage(
    original: Transcription,
    result: Result<Transcription, PolicyResolvedAuxRuntimeError<BackendError>>,
) -> Result<Transcription, BackendError> {
    match result {
        Ok(punctuated) => Ok(punctuated),
        Err(error) if optional_punctuation_failure_disables_stage(&error) => {
            crate::stage_timing::log_detail_event(
                "native_transcribe",
                format_args!(
                    "stage=auxiliary_execution_candidate event=disabled auxiliary_stage=firered-punctuation reason={error}"
                ),
            );
            Ok(original)
        }
        Err(error) => Err(required_auxiliary_stage_error(error)),
    }
}

fn optional_punctuation_failure_disables_stage(
    error: &PolicyResolvedAuxRuntimeError<BackendError>,
) -> bool {
    match error {
        PolicyResolvedAuxRuntimeError::Operation(BackendError::TranscriptionCanceled) => false,
        PolicyResolvedAuxRuntimeError::Operation(_)
        | PolicyResolvedAuxRuntimeError::CandidatesExhausted { .. } => true,
        PolicyResolvedAuxRuntimeError::CandidateFailed { .. } => false,
        PolicyResolvedAuxRuntimeError::EmptyPlan { .. } => false,
    }
}

/// Restores punctuation on each finalized segment's text independently (the
/// stage's documented "finalize-only, per segment" contract -- see
/// `crate::punctuation`'s module docs) and rebuilds the top-level `text` field
/// from the punctuated segments the same way the longform assembler does
/// (trim, drop empties, join with a space), so the punctuated text and
/// segments stay consistent. A segment whose classifier call fails keeps its
/// original (unpunctuated) text -- fail-closed per segment rather than
/// aborting the whole transcript.
#[cfg(test)]
fn punctuate_transcription_segments(
    mut transcription: Transcription,
    runtime: &FireRedPuncRuntime,
) -> Transcription {
    for segment in &mut transcription.segments {
        if let Ok(punctuated) = runtime.punctuate(&segment.text) {
            segment.text = punctuated;
        }
    }
    transcription.text = transcription
        .segments
        .iter()
        .map(|segment| segment.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    transcription
}

fn punctuate_transcription_segments_with_actor(
    mut transcription: Transcription,
    runtime: &FireRedPuncActor,
    execution_context: &crate::RequestExecutionContext,
    progress: &ProgressReporter,
) -> Result<Transcription, BackendError> {
    let total = transcription.segments.len() as u64;
    for (index, segment) in transcription.segments.iter_mut().enumerate() {
        if execution_context.is_canceled() {
            return Err(BackendError::TranscriptionCanceled);
        }
        match punctuate(runtime, &segment.text) {
            Ok(punctuated) => segment.text = punctuated,
            Err(error)
                if execution_context.is_canceled()
                    || is_cooperative_cancel_reason(&error.to_string()) =>
            {
                return Err(BackendError::TranscriptionCanceled);
            }
            Err(_) => {}
        }
        if execution_context.is_canceled() {
            return Err(BackendError::TranscriptionCanceled);
        }
        progress.report_units((index as u64).saturating_add(1), total.max(1));
    }
    transcription.text = transcription
        .segments
        .iter()
        .map(|segment| segment.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    Ok(transcription)
}

/// Returns the audio at `input_path` as 16 kHz mono f32 samples, preferring
/// `prepared_samples` -- already resident in memory when
/// `prepare_audio_input`'s in-process symphonia decode path produced them
/// (see `crate::audio::PreparedAudioInput::samples`) -- over re-reading
/// `input_path` from disk. The WAV-passthrough and external ffmpeg/afconvert
/// conversion paths leave `prepared_samples` unset, so this falls back to the
/// WAV load exactly as before for those.
///
/// The result is an immutable shared PCM owner. Both an already-prepared input
/// and a WAV loaded here enter the same representation; every later consumer
/// receives a cheap range view, so a second reference can never trigger a
/// whole-recording clone.
fn resolve_prepared_audio_samples(
    input_path: &Path,
    prepared_samples: Option<Arc<Vec<f32>>>,
) -> Result<PcmBuffer, crate::NativeAsrError> {
    if let Some(samples) = prepared_samples {
        return Ok(PcmBuffer::from_shared(samples));
    }
    load_wav_16khz_mono_f32_v0(
        input_path,
        "Native ASR Core backend",
        "Native ASR Core backend",
    )
    .map(PcmBuffer::from_vec)
}

/// Post-hoc precise timeline refine for a finished transcription.
///
/// Does **not** re-run ASR. Re-validates word anchors against `prepared_audio`
/// (16 kHz mono f32). When the timeline is already precise and validates,
/// re-projects reading + subtitle views and returns without loading the
/// Forced Aligner. Otherwise runs the whole-document forced aligner, then
/// projects dual views with [`TimelineQuality::ForcedAligned`].
///
/// Speaker labels / person attribution on segments are preserved; only word
/// timestamps and the dual-view projection change. Missing Forced Aligner pack
/// fails closed with [`BackendError::WordTimestampAlignmentPackMissing`] (no
/// silent download).
pub fn refine_existing_transcription_timeline(
    transcription: Transcription,
    prepared_audio_16khz_mono: &[f32],
    execution_services: &NativeExecutionServices,
    execution_target: crate::ExecutionTarget,
    language_hint: Option<&str>,
    keep_word_timestamps: bool,
) -> Result<Transcription, BackendError> {
    if prepared_audio_16khz_mono.is_empty() {
        return Err(BackendError::WordTimestampAlignmentFailed {
            reason: "audio is empty; cannot refine timeline without PCM samples".into(),
        });
    }
    if transcription.segments.is_empty() {
        return Err(BackendError::WordTimestampAlignmentFailed {
            reason: "transcription has no timed segments to align".into(),
        });
    }
    let audio_duration_s = prepared_audio_16khz_mono.len() as f32 / 16_000.0;
    let validation = crate::subtitle::validate_word_anchors(&transcription, audio_duration_s);
    let already_precise = matches!(
        transcription.timeline_quality,
        Some(crate::subtitle::TimelineQuality::NativeReliable)
            | Some(crate::subtitle::TimelineQuality::ForcedAligned)
    );
    if already_precise && validation.is_reliable() {
        let quality = transcription
            .timeline_quality
            .unwrap_or(crate::subtitle::TimelineQuality::NativeReliable);
        return Ok(crate::subtitle::project_transcription(
            transcription,
            crate::subtitle::TimelineProjectOptions {
                timeline_quality: quality,
                strip_words: !keep_word_timestamps,
                audio_duration_s: Some(audio_duration_s),
            },
        ));
    }

    let pcm = PcmBuffer::from_vec(prepared_audio_16khz_mono.to_vec());
    let request_intent = ExecutionIntent::from(execution_target);
    let backend_class = progress_backend_class(&request_intent);
    // Post-hoc FA is an independent operation: its own progress id is not
    // available here (caller may install one later). Report through a detached
    // reporter unless the caller shares an id via thread-local in a follow-up;
    // for now install under no-id (no publish) unless we invent an id. Server
    // post-hoc path should pass progress once it has a request id -- keep the
    // align loop progress-capable via optional reporter below.
    let _progress_handle = ProgressRegistryHandle::new(None);
    let progress = ProgressReporter::install(
        None,
        ProgressPlan::post_hoc_align(audio_duration_s, backend_class),
    );
    // Align is a separate heavyweight phase; drop idle primary ASR caches so
    // the aligner can quote its graph against free headroom.
    execution_services.unload_idle_native_model_runtime_caches();
    let refined = refine_transcription_word_timestamps_with_forced_aligner_policy(
        transcription,
        pcm.full_slice(),
        language_hint,
        execution_services,
        &request_intent,
        &crate::RequestExecutionContext::uncancellable(
            "post-hoc timeline refinement has no external request control",
        ),
        Some(&progress),
    )?;
    progress.complete_stage_brief(TranscriptionStage::Project);
    Ok(crate::subtitle::project_transcription(
        refined,
        crate::subtitle::TimelineProjectOptions {
            timeline_quality: crate::subtitle::TimelineQuality::ForcedAligned,
            strip_words: !keep_word_timestamps,
            audio_duration_s: Some(audio_duration_s),
        },
    ))
}

/// Maps a resolved provider/placement pair to one measured aligner topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForcedAlignerSessionPlan {
    Uniform,
    GpuAudioHybrid,
}

fn forced_aligner_session_plan(
    placement: ExecutionPlacement,
    provider: ExecutionProvider,
) -> Result<ForcedAlignerSessionPlan, &'static str> {
    match (placement, provider) {
        (ExecutionPlacement::CpuOnly, ExecutionProvider::Cpu) => {
            Ok(ForcedAlignerSessionPlan::Uniform)
        }
        (ExecutionPlacement::FullDevice, ExecutionProvider::Metal) => {
            Ok(ForcedAlignerSessionPlan::Uniform)
        }
        (ExecutionPlacement::Hybrid, ExecutionProvider::Cuda | ExecutionProvider::Vulkan) => {
            Ok(ForcedAlignerSessionPlan::GpuAudioHybrid)
        }
        _ => Err("forced-aligner execution topology is not validated for this provider"),
    }
}

/// Reuses the exact normalized PCM backing decoded by the main request and
/// loads the installed Qwen3-ForcedAligner pack once. Each already-bounded ASR
/// segment is aligned independently, then its local spans are mapped back to
/// the recording clock. This keeps graph memory bounded by the largest decode
/// segment instead of growing with the whole meeting. Segment text and speaker
/// attribution are left untouched; only `words` changes.
///
/// When `progress` is set, stage_fraction advances by **audio duration weight**
/// (sum of segment durations), not bare segment index.
pub(crate) fn refine_transcription_word_timestamps_with_forced_aligner_policy(
    transcription: Transcription,
    prepared_audio: PcmSlice,
    language_hint: Option<&str>,
    execution_services: &NativeExecutionServices,
    request_intent: &ExecutionIntent,
    execution_context: &crate::RequestExecutionContext,
    progress: Option<&ProgressReporter>,
) -> Result<Transcription, BackendError> {
    let _abort_callback_guard = execution_context
        .control
        .arm_for_native_decode_if_cancellable();
    if execution_context.is_canceled() {
        return Err(BackendError::TranscriptionCanceled);
    }
    let pack_path = forced_aligner_pack::resolve_forced_aligner_pack_path()
        .ok_or(BackendError::WordTimestampAlignmentPackMissing { backend: "native" })?;
    let language = transcription
        .language
        .clone()
        .or_else(|| language_hint.map(str::to_string))
        .unwrap_or_else(|| "en".to_string());
    let verified_forced_aligner = verify_forced_aligner_pack(&pack_path)
        .map_err(|error| forced_alignment_error_to_backend(execution_context, error.to_string()))?;
    let execution_plan = resolve_auxiliary_execution_plan(
        execution_services,
        crate::models::qwen::QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID,
        request_intent,
    )?;
    if let Some(progress) = progress {
        progress.enter_stage(TranscriptionStage::Align);
    }
    // Total alignable duration for duration-weighted stage_fraction.
    let total_align_duration_s: f64 = transcription
        .segments
        .iter()
        .filter(|s| !s.text.trim().is_empty())
        .map(|s| (f64::from(s.end) - f64::from(s.start)).max(0.0))
        .sum();
    let result = run_auxiliary_stage_with_policy(
        execution_services,
        &execution_plan,
        "qwen3-forced-aligner",
        |candidate| {
            if execution_context.is_canceled() {
                return Err(BackendError::TranscriptionCanceled);
            }
            let backend = crate::models::policy_resolved_aux_runtime::resolved_runtime_for_auxiliary_candidate(candidate).backend();
            let session_load_started = Instant::now();
            let session_plan =
                forced_aligner_session_plan(candidate.placement, candidate.device.route.provider)
                    .map_err(|reason| BackendError::NativeFailClosed {
                    reason: format!(
                        "{reason}: provider={:?} placement={:?}",
                        candidate.device.route.provider, candidate.placement,
                    ),
                })?;
            let session = match session_plan {
                ForcedAlignerSessionPlan::Uniform => Qwen3ForcedAlignerSession::load_verified(
                    verified_forced_aligner.clone(),
                    backend,
                ),
                ForcedAlignerSessionPlan::GpuAudioHybrid => {
                    Qwen3ForcedAlignerSession::load_verified_gpu_audio_hybrid(
                        verified_forced_aligner.clone(),
                    )
                }
            }
            .map_err(|error| {
                forced_alignment_error_to_backend(execution_context, error.to_string())
            })?;
            crate::stage_timing::log_detail_stage(
                "forced_aligner",
                "session_load",
                session_load_started.elapsed(),
            );
            let mut refined = transcription.clone();
            let audio_samples = prepared_audio.as_slice().len();
            let mut completed_align_duration_s = 0.0f64;
            for (index, segment) in refined.segments.iter_mut().enumerate() {
                if execution_context.is_canceled() {
                    return Err(BackendError::TranscriptionCanceled);
                }
                if segment.text.trim().is_empty() {
                    continue;
                }
                let range = forced_alignment_segment_sample_range(segment, audio_samples)
                    .ok_or_else(|| BackendError::WordTimestampAlignmentFailed {
                        reason: format!(
                            "segment {index} has no valid audio span for non-empty text: start={} end={}",
                            segment.start, segment.end
                        ),
                    })?;
                let segment_audio_seconds = range.len() as f64 / 16_000.0;
                let segment_duration_s =
                    (f64::from(segment.end) - f64::from(segment.start)).max(0.0);
                let alignment_started = Instant::now();
                let items = if let Some(progress) = progress {
                    let completed_before_segment = completed_align_duration_s;
                    let mut report_inner = |event| {
                        let inner_fraction = forced_aligner_inner_fraction(event, backend);
                        let stage_fraction = duration_weighted_fraction(
                            completed_before_segment + segment_duration_s * inner_fraction,
                            total_align_duration_s,
                        );
                        progress.report(
                            stage_fraction,
                            None,
                            None,
                            Some(forced_aligner_progress_detail(index, event)),
                        );
                    };
                    session.align_with_progress(
                        prepared_audio.slice(range),
                        &segment.text,
                        &language,
                        &mut report_inner,
                    )
                } else {
                    session.align(prepared_audio.slice(range), &segment.text, &language)
                }
                .map_err(|error| {
                    forced_alignment_error_to_backend(
                        execution_context,
                        format!("segment {index}: {error}"),
                    )
                })?;
                if execution_context.is_canceled() {
                    return Err(BackendError::TranscriptionCanceled);
                }
                crate::stage_timing::log_detail_event(
                    "forced_aligner",
                    format_args!(
                        "stage=segment_align index={index} audio_duration_s={segment_audio_seconds:.3} words={} duration_ms={:.3}",
                        items.len(),
                        alignment_started.elapsed().as_secs_f64() * 1000.0,
                    ),
                );
                assign_local_aligned_words(segment, &items);
                completed_align_duration_s += segment_duration_s;
                if let Some(progress) = progress {
                    progress.report_fraction(duration_weighted_fraction(
                        completed_align_duration_s,
                        total_align_duration_s,
                    ));
                }
            }
            Ok(refined)
        },
    );
    let result = match result {
        Ok(result) => result,
        Err(error)
            if execution_context.is_canceled()
                || is_cooperative_cancel_reason(&error.to_string()) =>
        {
            return Err(BackendError::TranscriptionCanceled);
        }
        Err(error) => return Err(required_auxiliary_stage_error(error)),
    };
    if execution_context.is_canceled() {
        return Err(BackendError::TranscriptionCanceled);
    }
    if let Some(progress) = progress {
        progress.complete_stage();
    }
    Ok(result)
}

/// Converts real ForcedAligner execution milestones into a calibrated share of
/// one segment's work. Graph internals stay monolithic for peak memory and
/// throughput; these cumulative boundaries make the observable progress less
/// sparse without pretending to have layer-level completion signals.
fn forced_aligner_inner_fraction(
    event: crate::models::qwen::ForcedAlignerProgressEvent,
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
) -> f64 {
    use crate::models::qwen::ForcedAlignerProgressEvent;

    // Cumulative medians measured by `forced_aligner_aux_audio_benchmark` on
    // the bound Q8_0 pack and 59.712 s reference fixture. CPU: mel=.01743,
    // audio=.25288, prompt=.25467, decoder=.99134, timestamp=.99952. M1 Metal:
    // mel=.06287, audio=.39172, prompt=.39563, decoder=.90061,
    // timestamp=.99865. Rounded values avoid false precision while explicit
    // `*Started` events describe monolithic graph work in flight. Unmeasured
    // generic GPU routes retain the conservative CPU profile.
    let (mel, audio, prompt, decoder, timestamp_span) = match backend {
        crate::ggml_runtime::GgmlCpuGraphBackend::Metal => (0.063, 0.392, 0.396, 0.901, 0.098),
        crate::ggml_runtime::GgmlCpuGraphBackend::Cpu
        | crate::ggml_runtime::GgmlCpuGraphBackend::Gpu => (0.017, 0.253, 0.255, 0.991, 0.008),
    };
    match event {
        ForcedAlignerProgressEvent::MelReady | ForcedAlignerProgressEvent::AudioEncodingStarted => {
            mel
        }
        ForcedAlignerProgressEvent::AudioEncoded => audio,
        ForcedAlignerProgressEvent::PromptPrepared
        | ForcedAlignerProgressEvent::DecoderPrefillStarted => prompt,
        ForcedAlignerProgressEvent::DecoderPrefilled
        | ForcedAlignerProgressEvent::TimestampLogitsStarted { .. } => decoder,
        ForcedAlignerProgressEvent::TimestampLogits { completed, total } => {
            decoder + timestamp_span * f64::from(completed_work_fraction(completed, total))
        }
        ForcedAlignerProgressEvent::Finalized => 1.0,
    }
}

fn forced_aligner_progress_detail(
    segment_index: usize,
    event: crate::models::qwen::ForcedAlignerProgressEvent,
) -> String {
    use crate::models::qwen::ForcedAlignerProgressEvent;

    let phase = match event {
        ForcedAlignerProgressEvent::MelReady => "mel_ready".to_string(),
        ForcedAlignerProgressEvent::AudioEncodingStarted => "audio_encoding".to_string(),
        ForcedAlignerProgressEvent::AudioEncoded => "audio_encoded".to_string(),
        ForcedAlignerProgressEvent::PromptPrepared => "prompt_prepared".to_string(),
        ForcedAlignerProgressEvent::DecoderPrefillStarted => "decoder_prefill".to_string(),
        ForcedAlignerProgressEvent::DecoderPrefilled => "decoder_prefilled".to_string(),
        ForcedAlignerProgressEvent::TimestampLogitsStarted { total } => {
            format!("timestamp_logits:0/{total}")
        }
        ForcedAlignerProgressEvent::TimestampLogits { completed, total } => {
            format!("timestamp_logits:{completed}/{total}")
        }
        ForcedAlignerProgressEvent::Finalized => "finalized".to_string(),
    };
    format!("forced_aligner segment={segment_index} phase={phase}")
}

fn forced_alignment_error_to_backend(
    execution_context: &crate::RequestExecutionContext,
    reason: String,
) -> BackendError {
    if execution_context.is_canceled() || is_cooperative_cancel_reason(&reason) {
        BackendError::TranscriptionCanceled
    } else {
        BackendError::WordTimestampAlignmentFailed { reason }
    }
}

fn forced_alignment_segment_sample_range(
    segment: &Segment,
    audio_samples: usize,
) -> Option<std::ops::Range<usize>> {
    let start_s = f64::from(segment.start).max(0.0);
    let end_s = f64::from(segment.end).max(start_s);
    let start = ((start_s * 16_000.0).floor() as usize).min(audio_samples);
    let end = ((end_s * 16_000.0).ceil() as usize).min(audio_samples);
    (start < end).then_some(start..end)
}

fn assign_local_aligned_words(segment: &mut Segment, items: &[ForcedAlignItem]) {
    if items.is_empty() {
        return;
    }
    let offset = f64::from(segment.start);
    let segment_end = f64::from(segment.end);
    segment.words = items
        .iter()
        .map(|item| {
            let start = (offset + item.start_time_s).clamp(offset, segment_end);
            let end = (offset + item.end_time_s)
                .clamp(start, segment_end)
                .max(start);
            WordTimestamp {
                word: item.text.clone(),
                start: start as f32,
                end: end as f32,
                confidence: None,
            }
        })
        .collect();
}

/// Distributes forced-aligner word spans onto the (time-ordered,
/// non-overlapping) segments they fall into: each item's start time selects
/// the last segment whose own start is `<=` it (segments are sorted and cover
/// the whole file, so this always finds the enclosing segment for a
/// well-formed decode). A segment with no aligned words keeps its prior
/// (family-approximate) word list rather than being emptied -- most often
/// because there is exactly one segment and the whole item list lands in it.
#[cfg(test)]
fn assign_aligned_words_to_segments(segments: &mut [Segment], items: &[ForcedAlignItem]) {
    if segments.is_empty() || items.is_empty() {
        return;
    }
    let mut buckets: Vec<Vec<WordTimestamp>> = segments.iter().map(|_| Vec::new()).collect();
    for item in items {
        let segment_index = segments
            .iter()
            .rposition(|segment| f64::from(segment.start) <= item.start_time_s)
            .unwrap_or(0);
        buckets[segment_index].push(WordTimestamp {
            word: item.text.clone(),
            start: item.start_time_s as f32,
            end: item.end_time_s as f32,
            confidence: None,
        });
    }
    for (segment, bucket) in segments.iter_mut().zip(buckets) {
        if !bucket.is_empty() {
            segment.words = bucket;
        }
    }
}

/// Which speaker segmentation source runs for one transcription: the resolved
/// product of "did the user turn Voice ID on" and "where does this family's
/// speaker structure come from". Exactly one source runs, which is what makes
/// speaker labels single-writer -- the bug this type replaces was two derived
/// booleans that could both be live, letting an external pass overwrite labels
/// a family had already produced.
///
/// Identity is deliberately NOT part of the segmentation-source decision:
/// matching recording-local turns to known people is one source-independent
/// stage that runs afterwards (`diarize::voice_id`) and composes with either
/// source. Voice ID is default-off; once explicitly enabled, an installed but
/// unusable required embedder fails closed before speaker results escape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpeakerPlan {
    /// Voice ID off. No speaker structure reaches the caller -- including for a
    /// family that always writes its own markup, which strips it (see
    /// `models::moss_transcribe_diarize`), so the transcript is
    /// indistinguishable from one produced by a model that cannot separate
    /// speakers at all.
    Off,
    /// The family's own decode carries the turns.
    InDecoder,
    /// A separate recording-level segmenter over the same audio produces the
    /// turns, followed by speaker embedding, clustering and overlap recovery.
    External,
}

impl SpeakerPlan {
    fn resolve(voice_id: bool, source: SpeakerSegmentationSource) -> Self {
        match (voice_id, source) {
            (false, _) => Self::Off,
            (true, SpeakerSegmentationSource::InDecoder) => Self::InDecoder,
            (true, SpeakerSegmentationSource::External) => Self::External,
        }
    }
}

fn voice_id_audio_view(audio: &PcmBuffer, speaker_plan: SpeakerPlan) -> Option<PcmSlice> {
    (speaker_plan != SpeakerPlan::Off).then(|| audio.full_slice())
}

fn forced_aligner_audio_view(audio: &PcmBuffer, refine: bool) -> Option<PcmSlice> {
    refine.then(|| audio.full_slice())
}

fn run_native_transcription_impl(
    mut request: TranscriptionRequest,
    execution_services: &Arc<NativeExecutionServices>,
    execution_intent: Option<ExecutionIntent>,
    runtime_pack_input: NativeRuntimePackInput,
    progress: &ProgressReporter,
) -> Result<NativeTranscriptionOutcome, BackendError> {
    // Captured up front and threaded explicitly through the dispatch calls
    // below (never a thread-local): every cooperative cancel checkpoint in
    // this function and the shared decode driver reads this same `Arc`.
    let execution_context = Arc::clone(&request.execution_context);
    // Taken up front before `requested_model_id` borrows `request`. The value
    // is already an immutable shared owner; moving it here preserves the
    // exact backing while later ASR/Voice-ID/aligner stages clone only views.
    let prepared_samples = request.prepared_samples.take();
    progress.enter_stage_indeterminate(TranscriptionStage::LoadModel);
    let model_resolve_started = Instant::now();
    let requested_model_id = normalize_and_validate_model_id(&request)?;
    let model_pack_path = request
        .model_pack_path
        .as_deref()
        .ok_or(BackendError::NativeModelPackPathRequired)?;
    let verified_pack = match runtime_pack_input {
        NativeRuntimePackInput::CandidatePath => {
            let runtime_source =
                super::native_path::validate_local_native_runtime_source(model_pack_path)?;
            Arc::new(
                PackVerifier
                    .verify_runtime_source(runtime_source)
                    .map_err(|error| BackendError::NativeFailClosed {
                        reason: format!(
                            "runtime pack verification failed for '{}': {error}",
                            model_pack_path.display()
                        ),
                    })?,
            )
        }
        NativeRuntimePackInput::Verified(verified_pack) => {
            if verified_pack.preflight().runtime_source().path() != model_pack_path {
                return Err(BackendError::NativeFailClosed {
                    reason: format!(
                        "verified runtime pack path '{}' does not match request path '{}'",
                        verified_pack.preflight().runtime_source().path().display(),
                        model_pack_path.display()
                    ),
                });
            }
            verified_pack
        }
    };
    if !matches!(verified_pack.route(), PackRoute::Asr { .. }) {
        return Err(BackendError::NativeFailClosed {
            reason: format!(
                "runtime pack '{}' is auxiliary and cannot execute as an ASR model",
                model_pack_path.display()
            ),
        });
    }
    let runtime_preflight = verified_pack.preflight();
    let selection_metadata = selection_metadata_from_gguf(&runtime_preflight.metadata);
    let selected_family = validate_runtime_source_and_select_adapter(
        requested_model_id,
        &verified_pack,
        &selection_metadata,
    )?;
    let emits_punctuation =
        emits_punctuation_for_model_architecture(selected_family.model_architecture);
    let request_execution_intent =
        execution_intent.unwrap_or_else(|| request_execution_intent(request.execution_target));
    let execution_plan = resolve_native_execution_plan(
        execution_services.as_ref(),
        &selected_family,
        request_execution_intent.clone(),
    )?;
    let auto_gpu_policy = crate::arch::family_auto_gpu_policy_for_model_architecture(
        selected_family.model_architecture,
    );
    // Fail closed up front on task/language a non-Whisper family cannot honor,
    // rather than silently transcribing or erroring deep in the decode loop.
    let language_mode = crate::models::language::resolve_language_mode(
        selected_family.language_family_hint,
        &runtime_preflight.metadata,
    );
    crate::api::backend::reject_unsupported_task_or_language(
        selected_family.adapter_id,
        language_mode,
        request.task.unwrap_or_default(),
        request.language.as_deref(),
    )?;
    // The effective source language to stamp on the finished transcription:
    // honest per the resolved mode, and None when the model does not determine it.
    let reported_language = crate::models::language::effective_reported_language(
        language_mode,
        request.language.as_deref(),
    );
    crate::api::backend::reject_unsupported_phrase_bias_for_model(
        selected_family.adapter_id,
        selected_family.model_family,
        super::native_runtime_descriptor_supports_phrase_bias(
            &selected_family,
            Some(runtime_preflight.tensor_index.as_ref()),
        ),
        request.phrase_bias.as_ref(),
    )?;
    // Resolve the one segmentation source for this request. Exactly one runs:
    // the family's own decode, or the external segment/embed/cluster pass --
    // never both, so nothing can overwrite the other's labels downstream.
    let speaker_plan = SpeakerPlan::resolve(request.voice_id, selected_family.speaker_segmentation);
    if request.diarize_speakers.is_some() {
        // Fail closed instead of silently ignoring the clustering hint: it
        // needs Voice ID on, and only the external clustering path clusters.
        if !request.voice_id {
            return Err(BackendError::DiarizeSpeakersRequiresDiarization);
        }
        if speaker_plan == SpeakerPlan::InDecoder {
            return Err(BackendError::RequestOptionUnsupportedByModel {
                adapter: selected_family.adapter_id,
                option: "speakers hint",
                reason: "The model separates speakers in-decoder; the exact-speaker-count hint only applies to the external segment/embed/cluster path.",
            });
        }
    }
    // OPENASR_TIMING=1 detail: model-pack path validation + gguf metadata/
    // tensor-index preflight + family/adapter selection, i.e. everything
    // above this point in the request path. Nested inside the coarse
    // `inference` stage the caller (`run_native_transcription`) already logs
    // unconditionally.
    crate::stage_timing::log_detail_stage(
        "native_transcribe",
        "model_resolve",
        model_resolve_started.elapsed(),
    );
    progress.report_fraction(1.0);
    progress.complete_stage();
    progress.enter_stage_indeterminate(TranscriptionStage::Prepare);
    let audio_prep_started = Instant::now();
    let prepared_audio = resolve_prepared_audio_samples(&request.input_path, prepared_samples)
        .map_err(|error| BackendError::NativeUnsupportedInputFormat {
            reason: error.to_string(),
        })?;
    crate::stage_timing::log_stage(
        "native_transcribe",
        "audio_prep",
        audio_prep_started.elapsed(),
    );
    progress.report_fraction(1.0);
    progress.complete_stage();
    // Empty-but-valid PCM must reach the ASR family's established empty-input
    // behavior without first materializing Voice ID. No acoustic evidence can
    // produce a speaker label, so its effective speaker plan is off.
    let speaker_plan = if prepared_audio.is_empty() {
        SpeakerPlan::Off
    } else {
        speaker_plan
    };
    // Rebuild plan with known duration / speaker path so overall weights match
    // the remaining real stages (external: diarize+identify before decode;
    // in-decoder: identify after decode).
    let audio_duration_s = prepared_audio.len() as f32 / 16_000.0;
    let mut backend_class = progress_backend_class(&request_execution_intent);
    let external_diarize = speaker_plan == SpeakerPlan::External;
    let mut segmenter_kind = ProgressSegmenterKind::from_preference(request.voice_id_segmenter);
    progress.replace_plan(ProgressPlan::build(ProgressPlanInput {
        audio_duration_s,
        voice_id: speaker_plan != SpeakerPlan::Off,
        external_diarize,
        segmenter: segmenter_kind,
        punctuate: false, // postprocess rebuilds after emits_punctuation
        align: request_may_need_align(&request),
        backend: backend_class,
        persist: false,
    }));

    // Voice ID auxiliary load (embedder / segmenter packs) is not "准备中".
    // Re-enter LoadModel so the UI says "加载模型" until diarize/decode starts.
    if speaker_plan != SpeakerPlan::Off {
        progress.enter_stage_indeterminate(TranscriptionStage::LoadModel);
    }

    // Resolve the dependencies shared by every Voice ID path before probing
    // the external-only segmenter. This keeps the failure deterministic when
    // both packs are absent, avoids constructing either runtime on a known
    // incomplete stack, and still lets valid empty audio follow the family's
    // established empty-input behavior without auxiliary models.
    if speaker_plan != SpeakerPlan::Off && !crate::diarize::embed::embedder_pack_installed() {
        return Err(BackendError::DiarizationNotSupported { backend: "native" });
    }
    if speaker_plan == SpeakerPlan::External && !crate::diarize::segment::segmenter_pack_installed()
    {
        return Err(BackendError::DiarizationSegmenterUnavailable);
    }

    let external_diarizer_plan = if speaker_plan == SpeakerPlan::External {
        Some(
            crate::diarize::external::PreparedExternalDiarizer::prepare(request.voice_id_segmenter)
                .map_err(external_diarization_error_to_backend)?,
        )
    } else {
        None
    };
    // Once prepare pins the real segmenter, rebuild weights (Auto may have
    // selected DiariZen, which is heavier than the provisional Auto profile).
    if let Some(ref prepared) = external_diarizer_plan {
        let resolved = progress_segmenter_kind_for_provider(prepared.segmenter_provider());
        if resolved != segmenter_kind {
            segmenter_kind = resolved;
            progress.replace_plan(ProgressPlan::build(ProgressPlanInput {
                audio_duration_s,
                voice_id: speaker_plan != SpeakerPlan::Off,
                external_diarize,
                segmenter: segmenter_kind,
                punctuate: false,
                align: request_may_need_align(&request),
                backend: backend_class,
                persist: false,
            }));
            // replace_plan restores the open stage; keep LoadModel visible
            // while auxiliary packs are still materializing.
            if speaker_plan != SpeakerPlan::Off {
                progress.enter_stage_indeterminate(TranscriptionStage::LoadModel);
            }
        }
    }

    let audio_duration_seconds = prepared_audio.len() as f32 / 16_000.0;
    let speaker_runtime = if speaker_plan == SpeakerPlan::Off {
        None
    } else {
        Some(
            crate::diarize::embed::PolicyResolvedSpeakerRuntime::load_with_intent(
                Arc::clone(execution_services),
                request_execution_intent.clone(),
            )
            .map_err(|error| BackendError::NativeFailClosed {
                reason: format!("could not construct the admitted speaker runtime: {error}"),
            })?
            .ok_or(BackendError::DiarizationNotSupported { backend: "native" })?,
        )
    };
    let external_diarizer = if speaker_plan == SpeakerPlan::External {
        let speaker_runtime = speaker_runtime
            .as_ref()
            .expect("external speaker plan materialized speaker runtime");
        Some(
            external_diarizer_plan
                .expect("external speaker plan prepared a segmenter")
                .materialize(
                    Arc::clone(execution_services),
                    request_execution_intent.clone(),
                    speaker_runtime.shared_embedder(),
                )
                .map_err(external_diarization_error_to_backend)?,
        )
    } else {
        None
    };
    if speaker_plan != SpeakerPlan::Off {
        progress.report_fraction(1.0);
        progress.complete_stage();
    }
    let voice_id_embedder = speaker_runtime
        .as_ref()
        .map(|runtime| runtime.shared_embedder());
    // Compute speaker turns up front (independent of the transcript) so they can
    // be attributed onto whichever transcription path runs below.
    let voice_id_audio = voice_id_audio_view(&prepared_audio, speaker_plan);
    let speaker_turns = if let Some(diarizer) = external_diarizer.as_ref() {
        // External diarization runs outside the ASR candidate attempt, but its
        // invocation-local scratch still belongs to this process-wide broker.
        // Install only the service context for this phase: the scratch owner
        // below creates and drops its own reservation, while persistent
        // segmenter/embedder owners keep their independent candidate leases.
        let _memory_context =
            crate::models::native_execution_services::install_native_execution_services(
                execution_services.as_ref(),
            );
        let hint = match request.diarize_speakers {
            Some(speakers) => crate::diarize::contract::DiarizeHint::NumSpeakers(speakers),
            None => crate::diarize::contract::DiarizeHint::Auto,
        };
        compute_speaker_attribution(
            diarizer,
            voice_id_audio
                .as_ref()
                .expect("external speaker plan retains a Voice ID PCM view")
                .clone(),
            voice_id_embedder
                .as_deref()
                .expect("external speaker plan has a resolved embedder"),
            hint,
            &execution_context,
            progress,
        )?
    } else {
        SpeakerAttribution::default()
    };
    // External attribution is pure data at this point: both the timeline and
    // enrolled-person assignments have been copied out of the auxiliary
    // runtimes. Do not retain the segmenter/ReDimNet candidate leases while
    // the primary ASR candidate is admitted. In-decoder identity still needs
    // the shared embedder after decode, so only that plan carries it forward.
    let voice_id_embedder = (speaker_plan == SpeakerPlan::InDecoder)
        .then_some(voice_id_embedder)
        .flatten();
    drop(external_diarizer);
    drop(speaker_runtime);
    let dispatch = execution_services.offline_dispatch();
    let longform_resolution = resolve_native_longform_policy(
        request.longform.as_ref(),
        audio_duration_seconds,
        selected_family.model_architecture,
    );
    let longform_options = longform_resolution.options.clone();
    let run_longform = !matches!(longform_options.mode, LongFormMode::Off);
    let execution_longform =
        (!matches!(longform_options.mode, LongFormMode::Off)).then(|| longform_options.clone());
    let mut request_options = GgmlAsrExecutionOptions::from_transcription_request_with_phrase_bias(
        request.language.clone(),
        request.prompt.clone(),
        request.phrase_bias.clone(),
        execution_longform,
    );
    request_options.task = request.task.unwrap_or_default();
    request_options.inference_threads = request.inference_threads.map(usize::from);
    request_options.serve_batch = crate::models::serve_batch_env::ServeBatchPolicy {
        max_native_sessions: request.serve_batch_max_native_sessions.unwrap_or(1).max(1),
    };
    // VAD diarization needs word anchors to split multi-speaker transcript
    // segments at speaker-turn boundaries (X-ASR batch emits one monolithic
    // segment for the whole file). For most native families word timings are
    // free — pure post-processing of token emission times already captured
    // during decode — so force them on while diarizing and strip them from the
    // result below when the caller did not ask for word timestamps. Whisper is
    // the exception: user-requested word timestamps switch its decode path to
    // collect cross-attention (and disable cross flash attention), which can
    // perturb the transcript via FP accumulation differences. The
    // forced-for-diarization marker below tells whisper to keep the decode
    // path identical to a non-diarized run and derive word anchors post hoc
    // from the generated tokens instead.
    // Every family's transcript is re-segmented into subtitle-grade cues after
    // decode (see `cue_segmentation`); the splitter needs word anchors to place
    // cue boundaries. For all families except whisper these are free -- pure
    // post-processing of decode-time emission/token times already captured
    // during decode -- so force them on and strip them again if the caller did
    // not ask for them. Whisper is the exception: user-requested word timestamps
    // switch its decode path to collect cross-attention (which can perturb the
    // transcript), so it is left alone here and its cues fall back to
    // proportional splitting when a segment exceeds the caps.
    let force_word_timestamps_for_segmentation = matches!(
        selected_family.word_timestamps,
        crate::arch::OpenAsrWordTimestampStrategy::DecodeInvariant
    ) && !request.word_timestamps;
    let external_speakers = speaker_plan == SpeakerPlan::External;
    request_options.word_timestamps =
        request.word_timestamps || external_speakers || force_word_timestamps_for_segmentation;
    let strip_forced_word_timestamps =
        (external_speakers || force_word_timestamps_for_segmentation) && !request.word_timestamps;
    request_options.word_timestamps_forced_for_diarization = strip_forced_word_timestamps;
    // OADP Phase 0: the request-level adapter path rides the execution options
    // down to the family executor (env stays the server-side fallback).
    request_options.adapter_path = request.adapter_path.clone();
    // Only the in-decoder path consumes this flag; the external
    // VAD + speaker-embedder pass runs separately. `SpeakerPlan` already made
    // the two mutually exclusive, and this is where that decision reaches the
    // family executor.
    request_options.in_decoder_speakers = speaker_plan == SpeakerPlan::InDecoder;
    let primary_candidate = execution_plan
        .candidates()
        .first()
        .expect("execution policy plans are non-empty");
    let resolved_runtime_for_request =
        resolved_runtime_for_candidate(primary_candidate, auto_gpu_policy);
    // Actual device class after candidate selection: Auto may land on Metal/GPU
    // even though the intent-only provisional plan used AutoOrCpu weights.
    let resolved_backend_class =
        progress_backend_class_for_resolved(resolved_runtime_for_request.backend());
    if resolved_backend_class != backend_class {
        backend_class = resolved_backend_class;
        progress.replace_plan(ProgressPlan::build(ProgressPlanInput {
            audio_duration_s,
            voice_id: speaker_plan != SpeakerPlan::Off,
            external_diarize,
            segmenter: segmenter_kind,
            punctuate: false,
            align: request_may_need_align(&request),
            backend: backend_class,
            persist: false,
        }));
    }
    // Per-request diagnostics line (source/model/quant/backend/audio shape) --
    // logged once here, after model resolution and audio prep have both
    // succeeded and the backend label is resolvable, and before decode
    // dispatch. Deliberately excludes `request.input_path`/
    // `request.display_file_name` and any decoded/transcribed text: see
    // `request_context`'s module doc for the privacy contract.
    log_request_context(
        request.source,
        requested_model_id,
        &quant_tag_for_log(requested_model_id, runtime_preflight.runtime_source.path()),
        native_runtime_backend_label(resolved_runtime_for_request.backend()),
        audio_duration_seconds,
        request.source_container.as_deref(),
        request.source_sample_rate_hz,
        request.source_channels,
    );
    let mut longform_metadata: Option<TranscriptionLongFormMetadata> = None;
    // Decodes that stopped short of their own audio, for every exit path of
    // this function. Declared out here rather than inside the long-form block
    // because the single-pass path can truncate too -- and that is the case
    // with no long-form metadata to hide the fact in, which is exactly how a
    // short recording used to come back silently cut with a success status.
    let mut truncated_decodes: Vec<TruncatedDecode> = Vec::new();
    if run_longform {
        let vad_execution_plan = resolve_longform_vad_execution_plan(
            execution_services.as_ref(),
            &request_execution_intent,
        )?;
        let (mut plan, vad_engine_label) = run_auxiliary_stage_with_policy(
            execution_services.as_ref(),
            &vad_execution_plan,
            "longform-vad",
            |candidate| {
                let (vad_provider, vad_engine_label) =
                    resolve_longform_vad_provider(
                        &longform_options,
                        resolved_runtime_for_candidate(
                            candidate,
                            crate::diarize::vad::STREAM_VAD_OFFLINE_AUTO_GPU_POLICY,
                        )
                        .backend(),
                        candidate.placement,
                    )?;
                let plan = plan_longform_slices_with_materialization_gate(
                    &prepared_audio,
                    16_000,
                    &longform_options,
                    Some(vad_provider.as_ref()),
                    &|| execution_context.is_canceled(),
                    |packed_samples| {
                        // Packing a VAD timeline creates a second, recording-sized
                        // PCM buffer. Reject a known-impossible allocation before
                        // materializing it, while retaining broker headroom for
                        // driver and backend allocations outside OpenASR owners.
                        let packed_bytes = u64::try_from(packed_samples)
                            .unwrap_or(u64::MAX)
                            .saturating_mul(std::mem::size_of::<f32>() as u64);
                        let headroom_bytes =
                            execution_services.memory_broker().minimum_headroom_bytes();
                        let required_bytes = packed_bytes.saturating_add(headroom_bytes);
                        if let Some(available_bytes) = crate::host::host_available_memory_bytes()
                            && available_bytes < required_bytes
                        {
                            return Err(BackendError::NativeInsufficientHostMemory {
                                reason: format!(
                                    "longform packed-audio materialization needs {packed_bytes} bytes in addition to broker headroom ({headroom_bytes} bytes), but only {available_bytes} bytes are currently available"
                                ),
                            });
                        }
                        Ok(())
                    },
                )
                .map_err(longform_planning_error_to_backend)?;
                Ok((plan, vad_engine_label))
            },
        )
        .map_err(required_auxiliary_stage_error)?;
        let plan_stats = plan.stats.clone();
        let mut longform_provenance =
            combined_longform_provenance(&longform_resolution.provenance, &plan_stats.provenance);
        // Record which VAD engine actually ran, so the slice-kind label (which
        // reflects the slicing algorithm) is never mistaken for the provider.
        longform_provenance.push(format!("core.native.vad.engine:{vad_engine_label}"));
        request_options.longform_chunk_count_hint = Some(plan_stats.chunk_count);
        let multichunk_on_metal = should_prefer_cpu_decoder_for_multichunk_metal(
            selected_family.model_architecture,
            &request_execution_intent,
            plan_stats.chunk_count,
            resolved_runtime_for_request.backend(),
        );
        if multichunk_on_metal {
            request_options.auto_prefer_cpu_decoder_for_multichunk_metal = true;
        }
        if multichunk_on_metal {
            longform_provenance.push(
                "core.native.longform.policy:cohere-metal-multichunk-prefer-cpu-decoder"
                    .to_string(),
            );
        }
        let slice_kind_summary = summarize_slice_kinds(&plan.slices);
        let has_processed_audio = plan.processed_audio.is_some();
        let timeline_kind = if has_processed_audio {
            "packed"
        } else {
            "identity"
        };
        if plan.slices.is_empty() {
            return Ok(NativeTranscriptionOutcome {
                transcription: Transcription {
                    truncated_decodes: Vec::new(),
                    unnamed_speakers: Vec::new(),
                    text: String::new(),
                    segments: Vec::new(),
                    longform: Some(build_longform_metadata(
                        &longform_options,
                        plan_stats.chunk_count,
                        plan_stats.skipped_silent_chunks,
                        plan_stats.duplicate_merge_count,
                        slice_kind_summary,
                        timeline_kind,
                        &longform_provenance,
                        resolved_runtime_for_request.backend(),
                    )),
                    language: reported_language.clone(),
                    ..Default::default()
                },
                prepared_audio,
                emits_punctuation,
                speaker_finalization: SpeakerFinalizationContext {
                    attribution: speaker_turns,
                    embedder: voice_id_embedder,
                    plan: speaker_plan,
                    scope_by_segment: Vec::new(),
                    strip_forced_word_timestamps,
                },
                progress_backend: backend_class,
                progress_segmenter: segmenter_kind,
            });
        }
        if has_processed_audio || plan.slices.len() > 1 {
            let mut assembler =
                TranscriptAssembler::new(plan.timeline.clone(), SegmentMergePolicy::default());
            let mut rolling_prompt = request_options.prompt.clone().unwrap_or_default();
            let mut rolling_prompt_token_ids: Vec<u32> = Vec::new();
            let carry_prompt_mode =
                longform_prompt_carry_mode(&longform_options, selected_family.model_architecture);
            let mut ran_any_slice = false;
            let mut suppressed_slice_count = 0usize;
            // Silence packing necessarily creates different samples; move
            // that Vec into one new immutable backing. Identity plans clone
            // only the original backing handle. Every slice below is a range
            // view into whichever one applies.
            let plan_audio = plan
                .processed_audio
                .take()
                .map(PcmBuffer::from_vec)
                .unwrap_or_else(|| prepared_audio.clone());
            // Publish per-slice decode progress for the UI, weighted by each
            // slice's audio samples so the bar tracks decode time rather than slice
            // number. The forced-align refine (if any) continues the same monotonic
            // bar from the outer wrapper; the run-scoped handle removes this
            // request's registry entry on any exit. `word_timestamps_refine`
            // reserves headroom for that phase.
            let total_decode_samples: u64 = plan
                .slices
                .iter()
                .map(|slice| slice.duration_samples() as u64)
                .sum();
            let decode_progress = DecodeProgress::begin(progress.clone(), total_decode_samples);
            // In-session pause/cancel control for this in-flight transcription,
            // carried explicitly on `request.execution_context` (never a
            // thread-local). Checked at each slice boundary (L0): a cancel
            // unwinds cleanly with `TranscriptionCanceled` (dropping the
            // assembler and progress guard), and a pause blocks the worker here
            // until resume or cancel. The shared seq2seq greedy driver also
            // polls cancel at each token step (L1) so cancel does not wait for
            // the end of a long slice. A detached context (CLI / no control
            // registered) never trips either check, leaving the decode
            // byte-identical to before.
            let mut slice_index = 0usize;
            let mut degraded_slice_fallbacks: Vec<(usize, SliceExecutionFallback)> = Vec::new();
            // Slices whose decode stopped short of their own audio, rendered
            // for the provenance string channel (see
            // `format_truncated_slice_provenance`).
            let mut truncated_slices: Vec<String> = Vec::new();
            // Monotonic identity assigned to every slice that actually decoded
            // with an in-decoder speaker model. The assembler carries this
            // provenance beside each surviving segment through overlap trim and
            // de-duplication, so final identity matching never guesses a slice
            // from its timestamp.
            let mut speaker_scope_count = 0usize;
            // P1 long-audio slice pipeline: decode K slices concurrently to
            // overlap the encode/decode GPU bubbles (the admission-concurrency
            // win, applied to one file's slices). The default is gated on this
            // run's effective prompt-carry state (see
            // `slice_pipeline_requested_width`): a carry-disabled run goes
            // concurrent up to the capacity gate, a carry-active run stays on
            // the byte-identical serial + prompt-carry path in the `else`
            // below unless `OPENASR_SLICE_PIPELINE_WIDTH` overrides.
            let pipeline_width = effective_slice_pipeline_width(
                slice_pipeline_requested_width(carry_prompt_mode),
                &plan.slices,
                runtime_preflight,
                &execution_plan,
            );
            if pipeline_width > 1 {
                let carry_note = if carry_prompt_mode == LongformPromptCarryMode::Disabled {
                    "carry=disabled"
                } else {
                    // Explicit escape hatch on a carry-active run: the
                    // concurrent path is carry-light, so the cross-slice
                    // prompt carry this run would otherwise thread is dropped
                    // -- an accepted quality cost, recorded for diagnosis.
                    "carry=dropped-by-explicit-width"
                };
                longform_provenance.push(format!(
                    "core.native.longform.slice-pipeline:width={pipeline_width},{carry_note}"
                ));
                run_concurrent_slice_pipeline(ConcurrentSlicePipeline {
                    width: pipeline_width,
                    slices: plan.slices,
                    plan_audio: &plan_audio,
                    dispatch,
                    execution_services,
                    verified_pack: verified_pack.as_ref(),
                    selected_family: &selected_family,
                    request_options: &request_options,
                    execution_plan: &execution_plan,
                    auto_gpu_policy,
                    execution_context: &execution_context,
                    longform_options: &longform_options,
                    speaker_plan,
                    decode_progress: &decode_progress,
                    assembler: &mut assembler,
                    ran_any_slice: &mut ran_any_slice,
                    suppressed_slice_count: &mut suppressed_slice_count,
                    degraded_slice_fallbacks: &mut degraded_slice_fallbacks,
                    truncated_slices: &mut truncated_slices,
                    truncated_decodes: &mut truncated_decodes,
                    speaker_scope_count: &mut speaker_scope_count,
                })?;
            } else {
                for slice in plan.slices {
                    if execution_context.control.wait_at_slice_boundary()
                        == super::transcription_control::SliceBoundaryControl::Canceled
                    {
                        return Err(BackendError::TranscriptionCanceled);
                    }
                    let slice_samples = slice.duration_samples() as u64;
                    let relative_start = slice
                        .content_start_sample
                        .saturating_sub(slice.start_sample);
                    let relative_end = slice
                        .content_end_sample
                        .saturating_sub(slice.start_sample)
                        .min(slice.duration_samples());
                    let chunk = plan_audio.slice(slice.start_sample..slice.end_sample);
                    if longform_options.suppress_silent_slices
                        && is_effectively_silent(
                            &chunk[relative_start..relative_end],
                            longform_options.energy_silence_threshold_db,
                        )
                    {
                        suppressed_slice_count += 1;
                        assembler.push_slice_result(SliceTranscript {
                            slice,
                            text: String::new(),
                            segments: Vec::new(),
                            time_domain: SegmentTimeDomain::AbsoluteOriginal,
                        });
                        decode_progress.complete_slice(slice_samples);
                        continue;
                    }
                    let mut slice_options = request_options.clone();
                    match carry_prompt_mode {
                        LongformPromptCarryMode::Disabled => {}
                        LongformPromptCarryMode::Text => {
                            let trimmed = rolling_prompt.trim();
                            if !trimmed.is_empty() {
                                slice_options.prompt = Some(trimmed.to_string());
                            }
                        }
                        LongformPromptCarryMode::TokenHistory => {
                            if !rolling_prompt_token_ids.is_empty() {
                                slice_options.prompt = None;
                                slice_options.prompt_token_ids =
                                    Some(rolling_prompt_token_ids.clone());
                            }
                        }
                    }
                    slice_index += 1;
                    let slice_decode_started = Instant::now();
                    let (result, slice_execution_fallback) =
                        run_dispatch_once_with_progress_and_policy(
                            dispatch,
                            execution_services,
                            verified_pack.as_ref(),
                            &selected_family,
                            chunk,
                            slice_options,
                            &execution_plan,
                            auto_gpu_policy,
                            &execution_context,
                            &decode_progress,
                            slice_samples,
                            &format!("index={slice_index}"),
                        )?;
                    if let Some(fallback) = slice_execution_fallback {
                        degraded_slice_fallbacks.push((slice_index, fallback));
                    }
                    // OPENASR_TIMING=1 detail: per-longform-slice decode time.
                    // Coarse by default (only the whole-request `inference` stage
                    // is logged unconditionally) since a long recording can chunk
                    // into many slices -- one line per slice would be noisy for
                    // the always-on tier.
                    crate::stage_timing::log_detail_event(
                        "native_transcribe",
                        format_args!(
                            "stage=longform_slice_decode index={slice_index} samples={slice_samples} duration_ms={:.3}",
                            slice_decode_started.elapsed().as_secs_f64() * 1000.0
                        ),
                    );
                    // Destructure instead of `result.clone().into_transcription()`:
                    // the fields are consumed below and nothing needs `result`
                    // as a whole afterwards, so there is nothing left to clone.
                    let GgmlAsrExecutionResult {
                        transcription,
                        carry_context,
                        decode_truncation,
                    } = result;
                    if let Some(truncation) = decode_truncation {
                        // A slice whose decode gave up partway is a degraded
                        // result, not a normal one: the audio after this point is
                        // absent from the transcript. Carried structurally on the
                        // returned transcript (so every output format can see it)
                        // AND summarized in the same provenance channel as the
                        // other "this run did not behave like the naive default"
                        // facts, rather than left as a log line the caller never
                        // sees.
                        truncated_slices
                            .push(format_truncated_slice_provenance(slice_index, &truncation));
                        truncated_decodes.push(TruncatedDecode {
                            slice_index: Some(slice_index),
                            truncation,
                        });
                    }
                    ran_any_slice = true;
                    match carry_prompt_mode {
                        LongformPromptCarryMode::Disabled => {}
                        LongformPromptCarryMode::Text => {
                            if !transcription.text.trim().is_empty() {
                                rolling_prompt = append_context_tail(
                                    &rolling_prompt,
                                    &transcription.text,
                                    longform_options.max_context_chars,
                                );
                            }
                        }
                        LongformPromptCarryMode::TokenHistory => {
                            if let Some(prompt_token_ids) =
                                carry_context.and_then(|context| context.prompt_token_ids)
                            {
                                rolling_prompt_token_ids = prompt_token_ids;
                            }
                        }
                    }
                    let transcript = SliceTranscript {
                        slice,
                        text: transcription.text,
                        segments: transcription.segments,
                        time_domain: SegmentTimeDomain::RelativeToSliceContent,
                    };
                    if speaker_plan == SpeakerPlan::InDecoder {
                        let scope = speaker_scope_count;
                        speaker_scope_count += 1;
                        assembler.push_slice_result_with_speaker_scope(transcript, scope);
                    } else {
                        assembler.push_slice_result(transcript);
                    }
                }
            }
            // Decode stage complete; merge/resegment is short and folds into
            // later project / postprocess stages rather than a fixed ceiling.
            decode_progress.report_stage_fraction(1.0);
            progress.complete_stage();
            if !degraded_slice_fallbacks.is_empty() {
                let fallback_facts: Vec<String> = degraded_slice_fallbacks
                    .iter()
                    .map(|(index, fallback)| {
                        let failed = fallback
                            .failures
                            .iter()
                            .map(|(candidate, failure)| {
                                format!(
                                    "{}:{:?}:{:?}",
                                    candidate.device.route.provider,
                                    candidate.placement,
                                    failure.kind
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("+");
                        format!(
                            "{index}[{failed}->{}:{:?}]",
                            fallback.selected.device.route.provider, fallback.selected.placement
                        )
                    })
                    .collect();
                longform_provenance.push(format!(
                    "core.native.execution.candidate-fallback:slices={}",
                    fallback_facts.join(";")
                ));
            }
            if !truncated_slices.is_empty() {
                longform_provenance.push(format!(
                    "core.native.decode.truncated:slices={}",
                    truncated_slices.join(";")
                ));
            }
            let (assembled, assemble_stats, speaker_scope_by_segment) =
                assembler.into_parts_with_speaker_scopes();
            let run_metadata = build_longform_metadata(
                &longform_options,
                plan_stats.chunk_count,
                plan_stats
                    .skipped_silent_chunks
                    .saturating_add(assemble_stats.skipped_silent_chunks),
                plan_stats
                    .duplicate_merge_count
                    .saturating_add(assemble_stats.duplicate_merge_count),
                slice_kind_summary,
                timeline_kind,
                &longform_provenance,
                resolved_runtime_for_request.backend(),
            );
            if !ran_any_slice && suppressed_slice_count > 0 {
                let fallback_options = request_options.clone();
                let (fallback, _) = run_dispatch_once_with_progress_and_policy(
                    dispatch,
                    execution_services,
                    verified_pack.as_ref(),
                    &selected_family,
                    prepared_audio.full_slice(),
                    fallback_options,
                    &execution_plan,
                    auto_gpu_policy,
                    &execution_context,
                    &decode_progress,
                    0,
                    "suppressed-whole-file",
                )?;
                // This whole-file fallback replaces the slice results entirely,
                // so its own truncation is the only one that describes the
                // transcript being returned.
                let fallback_truncated_decodes = fallback
                    .decode_truncation
                    .map(|truncation| TruncatedDecode {
                        slice_index: None,
                        truncation,
                    })
                    .into_iter()
                    .collect();
                let transcription = prepare_native_transcription(
                    fallback.into_transcription(),
                    audio_duration_seconds,
                    Some(run_metadata),
                    reported_language.clone(),
                    fallback_truncated_decodes,
                );
                return Ok(NativeTranscriptionOutcome {
                    transcription,
                    prepared_audio,
                    emits_punctuation,
                    speaker_finalization: SpeakerFinalizationContext {
                        attribution: speaker_turns,
                        embedder: voice_id_embedder,
                        plan: speaker_plan,
                        scope_by_segment: Vec::new(),
                        strip_forced_word_timestamps,
                    },
                    progress_backend: backend_class,
                    progress_segmenter: segmenter_kind,
                });
            }
            let transcription = prepare_native_transcription(
                assembled,
                audio_duration_seconds,
                Some(run_metadata),
                reported_language.clone(),
                truncated_decodes,
            );
            return Ok(NativeTranscriptionOutcome {
                transcription,
                prepared_audio,
                emits_punctuation,
                speaker_finalization: SpeakerFinalizationContext {
                    attribution: speaker_turns,
                    embedder: voice_id_embedder,
                    plan: speaker_plan,
                    scope_by_segment: speaker_scope_by_segment,
                    strip_forced_word_timestamps,
                },
                progress_backend: backend_class,
                progress_segmenter: segmenter_kind,
            });
        }
        longform_metadata = Some(build_longform_metadata(
            &longform_options,
            plan_stats.chunk_count,
            plan_stats.skipped_silent_chunks,
            plan_stats.duplicate_merge_count,
            slice_kind_summary,
            timeline_kind,
            &longform_provenance,
            resolved_runtime_for_request.backend(),
        ));
    }

    // Short audio (no longform) and a longform run that planned down to a
    // single un-resampled slice both land here with the whole file decoded
    // in one `run_dispatch_once` call. Give that call its own one-slice
    // `DecodeProgress` (the slice's own window spans the entire decode-phase
    // fraction) instead of leaving it unreported: this used to be the exact
    // gap that left short-audio transcriptions with no progress signal at
    // all, forcing the UI onto a pure time estimate that had no way to know
    // decode had actually finished (issue: short-audio progress bar).
    let single_pass_total_samples = prepared_audio.len() as u64;
    let single_pass_decode_progress =
        DecodeProgress::begin(progress.clone(), single_pass_total_samples);
    let (transcription, single_pass_fallback) = run_dispatch_once_with_progress_and_policy(
        dispatch,
        execution_services,
        verified_pack.as_ref(),
        &selected_family,
        prepared_audio.full_slice(),
        request_options,
        &execution_plan,
        auto_gpu_policy,
        &execution_context,
        &single_pass_decode_progress,
        single_pass_total_samples,
        "single-pass",
    )?;
    if single_pass_fallback.is_some() {
        let tag = "core.native.execution.candidate-fallback:slices=single-pass";
        // No longform run at all (plain short-audio decode) leaves nowhere to
        // stamp this: the structured log line from
        // `run_dispatch_once_with_progress_and_policy` is this path's
        // only degraded-result diagnostic in that case.
        if let Some(metadata) = longform_metadata.as_mut() {
            metadata.provenance.push(tag.to_string());
        }
    }
    if let Some(truncation) = transcription.decode_truncation {
        // Unlike the GPU-fallback tag above, this one is NOT dependent on
        // long-form metadata existing: it rides on the transcript itself, so a
        // plain short-audio decode that the guard cut short still reports it.
        if let Some(metadata) = longform_metadata.as_mut() {
            metadata.provenance.push(format!(
                "core.native.decode.truncated:slices={}",
                format_truncated_slice_provenance_for_single_pass(&truncation)
            ));
        }
        truncated_decodes.push(TruncatedDecode {
            slice_index: None,
            truncation,
        });
    }
    let transcription = prepare_native_transcription(
        transcription.into_transcription(),
        audio_duration_seconds,
        longform_metadata,
        reported_language,
        truncated_decodes,
    );
    Ok(NativeTranscriptionOutcome {
        transcription,
        prepared_audio,
        emits_punctuation,
        speaker_finalization: SpeakerFinalizationContext {
            attribution: speaker_turns,
            embedder: voice_id_embedder,
            plan: speaker_plan,
            scope_by_segment: Vec::new(),
            strip_forced_word_timestamps,
        },
        progress_backend: backend_class,
        progress_segmenter: segmenter_kind,
    })
}

fn longform_planning_error_to_backend(
    error: LongFormSlicePlanningError<BackendError>,
) -> BackendError {
    match error {
        LongFormSlicePlanningError::Planning(LongFormSliceError::Canceled) => {
            BackendError::TranscriptionCanceled
        }
        LongFormSlicePlanningError::Planning(error) => BackendError::NativeFailClosed {
            reason: format!("could not build longform slice plan: {error}"),
        },
        LongFormSlicePlanningError::PackedAudioAdmission(error) => error,
    }
}

/// Render one truncated slice for the `core.native.decode.truncated`
/// provenance string: `<index>@<seconds>s:<reason>`, or `<index>@?:<reason>`
/// when the family emits no intra-decode timestamps to anchor it (see
/// [`DecodeTruncation::transcript_covers_up_to_seconds`]). Reporting `?` keeps
/// the missing anchor legible instead of substituting the clip length, which
/// would read as "nothing was lost".
fn format_truncated_slice_provenance(slice_index: usize, truncation: &DecodeTruncation) -> String {
    format!(
        "{slice_index}@{}:{}",
        format_truncation_anchor(truncation),
        truncation.reason.as_str()
    )
}

fn format_truncated_slice_provenance_for_single_pass(truncation: &DecodeTruncation) -> String {
    format!(
        "single-pass@{}:{}",
        format_truncation_anchor(truncation),
        truncation.reason.as_str()
    )
}

fn format_truncation_anchor(truncation: &DecodeTruncation) -> String {
    truncation
        .transcript_covers_up_to_seconds
        .map(|seconds| format!("{seconds:.2}s"))
        .unwrap_or_else(|| "?".to_string())
}

/// Normalize decode output before transcript-aware post-processing. Punctuation
/// and forced alignment need stable segment clocks and the reported language,
/// but speaker attribution must wait until both have finished.
fn prepare_native_transcription(
    transcription: Transcription,
    audio_duration_seconds: f32,
    longform_metadata: Option<TranscriptionLongFormMetadata>,
    reported_language: Option<String>,
    truncated_decodes: Vec<TruncatedDecode>,
) -> Transcription {
    let mut transcription = with_longform_metadata(
        normalize_transcription_segments(transcription, 0.0, audio_duration_seconds),
        longform_metadata,
    );
    debug_assert!(
        transcription.truncated_decodes.is_empty(),
        "prepare_native_transcription overwrites truncated_decodes; the incoming transcription must not already carry any"
    );
    transcription.truncated_decodes = truncated_decodes;
    with_reported_language(transcription, reported_language)
}

/// Complete speaker attribution and identity only after punctuation and any
/// required word alignment have run. This ordering is the contract: external
/// timelines may require word anchors to project a coarse ASR segment without
/// losing speaker turns. After identity, project the dual reading + subtitle
/// views from the attributed word timeline.
///
/// InDecoder progress order: Decode -> Punctuate -> Align -> Identify -> Project.
fn finalize_native_transcription(
    mut transcription: Transcription,
    speaker: &SpeakerFinalizationContext,
    prepared_audio: &[f32],
    timeline_quality: crate::subtitle::TimelineQuality,
    request_word_timestamps: bool,
    explicit_refine: bool,
    timeline_precision: crate::subtitle::TimelinePrecisionPolicy,
    progress: &ProgressReporter,
) -> Result<Transcription, BackendError> {
    if speaker.plan == SpeakerPlan::External {
        transcription = apply_speaker_attribution(transcription, &speaker.attribution)?;
    }
    match speaker.plan {
        SpeakerPlan::InDecoder => {
            // Each independently decoded slice is a label scope. The shared
            // identity stage disambiguates those local counters, gathers
            // acoustic evidence, stitches matching voices, and names enrolled
            // people. Runs after punctuate/align (plan order:
            // Decode -> Punctuate -> Align -> Identify -> Project) with real
            // batch sub-progress.
            progress.enter_stage(TranscriptionStage::IdentifySpeakers);
            let identity_progress = progress.clone();
            let identity_observer =
                crate::api::backend::WorkProgressObserver::new(move |done, total| {
                    identity_progress.report_units(done as u64, total.max(1) as u64);
                });
            let mut scopes = speaker_scopes_by_provenance(
                &mut transcription.segments,
                &speaker.scope_by_segment,
                prepared_audio,
            )?;
            let embedder =
                speaker
                    .embedder
                    .as_deref()
                    .ok_or(BackendError::VoiceIdIdentityFailed(
                        crate::diarize::voice_id::SpeakerIdentityError::EmbedderPackMissing,
                    ))?;
            transcription.unnamed_speakers =
                crate::diarize::voice_id::name_speakers_across_scopes_with_embedder_and_progress(
                    embedder,
                    &mut scopes,
                    Some(&identity_observer),
                )
                .map_err(speaker_identity_error_to_backend)?;
            progress.complete_stage();
        }
        SpeakerPlan::External => {
            // External identity was resolved directly from the canonical
            // speaker timeline before ASR attribution. Never rebuild its audio
            // evidence from transcript segments: coarse ASR segments can span
            // several speakers even when the timeline is correct.
            transcription.unnamed_speakers = speaker.attribution.unnamed_speakers.clone();
        }
        SpeakerPlan::Off => {
            transcription.unnamed_speakers.clear();
        }
    }
    // Identity runs before reading/cue projection. Besides avoiding redundant
    // embedding work over presentation-only cue fragments, this preserves the
    // exact one-to-one alignment between assembled in-decoder segments and
    // their decode-scope provenance. Cue splitting copies the resolved speaker
    // identity fields onto every child afterwards.
    //
    // Strip top-level words unless the caller asked to keep them. Words may
    // have been produced by native decode, by Auto/subtitle forced alignment,
    // or by Voice ID packing; cue start/end stay correct either way. Always /
    // explicit refine / requested word timestamps keep words on the wire.
    // Do not gate this on `strip_forced_word_timestamps` alone: that flag only
    // tracks decode-time diarization forcing and would miss Whisper/Auto+SRT
    // whole-document alignment, leaking unrequested per-word arrays.
    let keep_words = request_word_timestamps
        || explicit_refine
        || matches!(
            timeline_precision,
            crate::subtitle::TimelinePrecisionPolicy::Always
        );
    let strip_words = !keep_words;
    let audio_duration_s = prepared_audio.len() as f32 / 16_000.0;
    progress.enter_stage(TranscriptionStage::Project);
    progress.report_fraction(0.0);
    let projected = crate::subtitle::project_transcription(
        transcription,
        crate::subtitle::TimelineProjectOptions {
            timeline_quality,
            strip_words,
            audio_duration_s: Some(audio_duration_s),
        },
    );
    progress.report_fraction(1.0);
    progress.complete_stage();
    Ok(projected)
}

/// Cut time-ordered segments into the exact decode scopes that produced them.
///
/// `scope_by_segment` is emitted by [`TranscriptAssembler`] after overlap trim
/// and de-duplication, aligned one-for-one with the final segments. This is a
/// provenance contract, not a time heuristic: a segment retained from an
/// earlier overlapping slice remains in that slice's label namespace even if
/// its midpoint lies after the next slice's content start.
///
/// Every scope shares the whole recording as its `samples`: segment times are
/// already mapped to the original timeline by the assembler, so they index
/// straight into it. Empty decoded slices simply leave no group; scope numbers
/// may therefore skip but must never move backwards.
fn speaker_scopes_by_provenance<'a>(
    segments: &'a mut [Segment],
    scope_by_segment: &[Option<usize>],
    samples: &'a [f32],
) -> Result<Vec<crate::diarize::voice_id::SpeakerScope<'a>>, BackendError> {
    if scope_by_segment.is_empty() {
        return Ok(vec![crate::diarize::voice_id::SpeakerScope {
            segments,
            samples,
        }]);
    }
    if scope_by_segment.len() != segments.len() {
        return Err(BackendError::VoiceIdIdentityFailed(
            crate::diarize::voice_id::SpeakerIdentityError::InvalidScopeProvenance {
                reason: format!(
                    "{} scope entries for {} assembled segments",
                    scope_by_segment.len(),
                    segments.len()
                ),
            },
        ));
    }
    let mut lengths = Vec::new();
    let mut previous_scope = None;
    for scope in scope_by_segment {
        let scope = scope.ok_or_else(|| {
            BackendError::VoiceIdIdentityFailed(
                crate::diarize::voice_id::SpeakerIdentityError::InvalidScopeProvenance {
                    reason: "an assembled in-decoder segment has no decode scope".to_string(),
                },
            )
        })?;
        match previous_scope {
            None => lengths.push(1usize),
            Some(previous) if scope == previous => {
                *lengths.last_mut().expect("a previous scope has a length") += 1;
            }
            Some(previous) if scope > previous => lengths.push(1usize),
            Some(previous) => {
                return Err(BackendError::VoiceIdIdentityFailed(
                    crate::diarize::voice_id::SpeakerIdentityError::InvalidScopeProvenance {
                        reason: format!("decode scope moved backwards from {previous} to {scope}"),
                    },
                ));
            }
        }
        previous_scope = Some(scope);
    }
    let mut scopes = Vec::with_capacity(lengths.len());
    let mut rest = segments;
    for length in lengths {
        let (head, tail) = rest.split_at_mut(length);
        rest = tail;
        scopes.push(crate::diarize::voice_id::SpeakerScope {
            segments: head,
            samples,
        });
    }
    Ok(scopes)
}

/// Stamp the effective source language onto a finished transcription so every
/// exit path of `run_native_transcription` reports the same value (see
/// `crate::models::language::effective_reported_language`).
fn with_reported_language(
    mut transcription: Transcription,
    language: Option<String>,
) -> Transcription {
    // Prefer the request-derived language (explicit / fixed / default); fall back
    // to one the executor itself determined (whisper auto-detect sets the detected
    // code on the transcription it returns).
    let executor_detected = transcription.language.take();
    transcription.language = language.or(executor_detected);
    transcription
}

/// Recording-local speaker turns normalized from the selected segmentation
/// source plus identities resolved directly from clean timeline windows.
#[derive(Default)]
struct SpeakerAttribution {
    timeline: crate::diarize::contract::SpeakerTimeline,
    identities: BTreeMap<
        crate::diarize::contract::SpeakerId,
        crate::diarize::enrollment::SpeakerDisplayAssignment,
    >,
    unnamed_speakers: Vec<crate::diarize::voice_id::UnnamedSpeaker>,
}

/// Diarize the prepared audio into recording-local speaker turns, then match
/// enrolled people from those turns. All external protocol details stay
/// behind `ExternalDiarizer`; this layer only consumes normalized turns and
/// centroids.
fn compute_speaker_attribution(
    diarizer: &crate::diarize::external::ExternalDiarizer,
    samples: PcmSlice,
    embedder: &dyn crate::diarize::embed::SpeakerEmbedder,
    hint: crate::diarize::contract::DiarizeHint,
    execution_context: &crate::RequestExecutionContext,
    progress: &ProgressReporter,
) -> Result<SpeakerAttribution, BackendError> {
    let total_started = Instant::now();
    let diarize_debug = crate::diarize::debug::diarize_debug_enabled();
    if execution_context.is_canceled() {
        return Err(BackendError::TranscriptionCanceled);
    }
    progress.enter_stage(TranscriptionStage::Diarize);
    // External diarization has two heavyweight, independently bounded loops:
    // activity windows and ReDimNet embedding windows. Report both instead of
    // reaching 100% after segmentation and appearing stalled during embedding.
    // The split is a calibrated work share; movement within each share is
    // driven only by completed production work units.
    let segmenter = progress_segmenter_kind_for_provider(diarizer.segmenter_provider());
    let segment_share = external_diarization_segment_share(segmenter);
    let segment_progress = progress.clone();
    let segment_observer = crate::api::backend::WorkProgressObserver::new(move |done, total| {
        segment_progress.report_fraction(segment_share * completed_work_fraction(done, total));
    });
    let embedding_progress = progress.clone();
    let embedding_observer = crate::api::backend::WorkProgressObserver::new(move |done, total| {
        embedding_progress.report_fraction(external_diarization_embedding_progress(
            segment_share,
            done,
            total,
        ));
    });
    let diarization_started = Instant::now();
    let timeline = diarizer
        .diarize_with_progress(
            samples.clone(),
            16_000,
            hint,
            &|| execution_context.is_canceled(),
            crate::diarize::external::ExternalDiarizationProgress::new(
                &segment_observer,
                &embedding_observer,
            ),
        )
        .map_err(external_diarization_error_to_backend)?;
    progress.complete_stage();
    crate::stage_timing::log_detail_stage(
        "speaker_attribution",
        "diarization",
        diarization_started.elapsed(),
    );
    if execution_context.is_canceled() {
        return Err(BackendError::TranscriptionCanceled);
    }
    if diarize_debug {
        eprintln!(
            "openasr_diarize_debug stage=batch turns={} speakers={}",
            timeline.turns.len(),
            timeline.centroids.len()
        );
        for turn in &timeline.turns {
            eprintln!(
                "openasr_diarize_debug stage=batch turn start={:.2} end={:.2} speaker={} overlap={}",
                turn.range.start_s,
                turn.range.end_s,
                turn.speaker.label(),
                turn.overlap
            );
        }
    }
    progress.enter_stage(TranscriptionStage::IdentifySpeakers);
    let identity_progress = progress.clone();
    let identity_observer = crate::api::backend::WorkProgressObserver::new(move |done, total| {
        identity_progress.report_units(done as u64, total.max(1) as u64);
    });
    let identity_started = Instant::now();
    let identity =
        crate::diarize::voice_id::resolve_timeline_identities_with_embedder_and_progress(
            embedder,
            &timeline,
            samples.as_slice(),
            Some(&identity_observer),
        )
        .map_err(speaker_identity_error_to_backend)?;
    progress.complete_stage();
    crate::stage_timing::log_detail_stage(
        "speaker_attribution",
        "identity",
        identity_started.elapsed(),
    );
    crate::stage_timing::log_detail_event(
        "speaker_attribution",
        format_args!(
            "stage=complete speakers={} named={} unnamed={} duration_ms={:.3}",
            timeline.centroids.len(),
            identity
                .assignments
                .len()
                .saturating_sub(identity.unnamed_speakers.len()),
            identity.unnamed_speakers.len(),
            total_started.elapsed().as_secs_f64() * 1000.0,
        ),
    );
    Ok(SpeakerAttribution {
        timeline,
        identities: identity.assignments,
        unnamed_speakers: identity.unnamed_speakers,
    })
}

fn completed_work_fraction(completed: usize, total: usize) -> f32 {
    if total == 0 {
        1.0
    } else {
        (completed as f32 / total as f32).clamp(0.0, 1.0)
    }
}

const EXTERNAL_DIARIZATION_EMBEDDING_END: f32 = 0.98;

fn external_diarization_embedding_progress(
    segment_share: f32,
    completed: usize,
    total: usize,
) -> f32 {
    segment_share
        + (EXTERNAL_DIARIZATION_EMBEDDING_END - segment_share)
            * completed_work_fraction(completed, total)
}

fn external_diarization_segment_share(segmenter: ProgressSegmenterKind) -> f32 {
    match segmenter {
        // Fifteen-minute production-geometry measurements put Segmentation3
        // at roughly 34% of segment+embed compute and DiariZen Metal at 50%.
        // Rounded shares are deliberately stable across hosts; actual window
        // and embedding completion, not wall-clock interpolation, moves them.
        ProgressSegmenterKind::Segmentation3_0 | ProgressSegmenterKind::Auto => 0.34,
        ProgressSegmenterKind::DiariZen => 0.51,
    }
}

fn external_diarization_error_to_backend(
    error: crate::diarize::external::ExternalDiarizationError,
) -> BackendError {
    use crate::diarize::external::ExternalDiarizationError;
    use crate::diarize::segment::SegmentError;

    match error {
        ExternalDiarizationError::Canceled
        | ExternalDiarizationError::Segmenter(SegmentError::Canceled) => {
            BackendError::TranscriptionCanceled
        }
        ExternalDiarizationError::Segmenter(SegmentError::MissingPack { .. }) => {
            BackendError::DiarizationSegmenterUnavailable
        }
        error => BackendError::ExternalDiarizationFailed {
            reason: error.to_string(),
        },
    }
}

fn speaker_identity_error_to_backend(
    error: crate::diarize::voice_id::SpeakerIdentityError,
) -> BackendError {
    match error {
        crate::diarize::voice_id::SpeakerIdentityError::Canceled => {
            BackendError::TranscriptionCanceled
        }
        error => BackendError::VoiceIdIdentityFailed(error),
    }
}

/// Attribute recording-level speaker turns onto decoded segments. This is
/// deliberately separate from cue re-segmentation: identity resolution needs
/// the original assembled-segment boundaries and exact decode-scope
/// provenance, while subtitle cue splitting is presentation-only and copies
/// the resolved speaker fields afterwards.
fn apply_speaker_attribution(
    mut transcription: Transcription,
    attribution: &SpeakerAttribution,
) -> Result<Transcription, BackendError> {
    if !attribution.timeline.turns.is_empty() {
        transcription.segments = crate::diarize::attribution::assign_speakers(
            &attribution.timeline.turns,
            std::mem::take(&mut transcription.segments),
            &attribution.identities,
        )
        .map_err(|error| BackendError::WordTimestampAlignmentFailed {
            reason: error.to_string(),
        })?;
    }
    Ok(transcription)
}

/// Resolve the long-form VAD provider for this request, returning the
/// provider and a label for the engine that ran. Stream-VAD is the sole VAD
/// engine and is vendored (`include_bytes!`), so in practice this always
/// loads (a build-integrity problem otherwise); still, fail closed with a
/// typed `BackendError` on the request path instead of panicking.
fn resolve_longform_vad_provider(
    _options: &crate::LongFormOptions,
    backend: GgmlCpuGraphBackend,
    placement: crate::device::execution_policy::ExecutionPlacement,
) -> Result<(Box<dyn LongFormVadProvider>, &'static str), BackendError> {
    let provider = crate::diarize::vad::FireRedStreamVadProvider::shared_for_backend_and_placement(
        backend, placement,
    )
    .ok_or_else(|| BackendError::NativeFailClosed {
        reason: "Stream-VAD is unavailable: vendored weights failed to parse \
                         (build-integrity problem)"
            .to_string(),
    })?;
    let label = match backend {
        GgmlCpuGraphBackend::Cpu => "firered-stream-cpu",
        GgmlCpuGraphBackend::Metal => "firered-stream-metal",
        GgmlCpuGraphBackend::Gpu => "firered-stream-gpu",
    };
    Ok((Box::new(provider), label))
}

fn resolve_native_longform_policy(
    requested: Option<&crate::LongFormOptions>,
    audio_duration_seconds: f32,
    model_architecture: &str,
) -> NativeLongformPolicyResolution {
    resolve_native_longform_policy_for_backend(
        requested,
        audio_duration_seconds,
        model_architecture,
        GgmlCpuGraphConfig::runtime_default().backend,
    )
}

fn resolve_native_longform_policy_for_backend(
    requested: Option<&crate::LongFormOptions>,
    audio_duration_seconds: f32,
    model_architecture: &str,
    _backend: GgmlCpuGraphBackend,
) -> NativeLongformPolicyResolution {
    let mut options = if let Some(options) = requested {
        options.clone()
    } else if audio_duration_seconds > DEFAULT_NATIVE_LONGFORM_AUTO_TRIGGER_SECONDS {
        crate::LongFormOptions::default()
    } else {
        crate::LongFormOptions {
            mode: LongFormMode::Off,
            ..crate::LongFormOptions::default()
        }
    };
    let mut provenance = Vec::new();
    if !matches!(options.mode, LongFormMode::Off)
        && scoped_slice_recording_fits_one_decode(
            model_architecture,
            audio_duration_seconds,
            requested,
        )
    {
        options.mode = LongFormMode::Off;
        provenance.push(format!(
            "core.native.longform.policy:scoped-slices-integral,audio_seconds={audio_duration_seconds:.3}"
        ));
    }
    if !matches!(options.mode, LongFormMode::Off) {
        apply_scoped_slice_longform_window_policy(
            model_architecture,
            &mut options,
            &mut provenance,
        );
        apply_longform_safety_policy(model_architecture, &mut options, &mut provenance);
    }
    NativeLongformPolicyResolution {
        options,
        provenance,
    }
}

/// Whether this recording is short enough for a
/// [`OpenAsrLongformSliceShape::ScopedSlices`] family to decode it whole, in
/// which case slicing is skipped entirely.
///
/// For such a family slicing is a degradation rather than the normal path: the
/// in-decoder speaker numbering restarts at every seam, so cross-slice identity
/// has to be re-established from voice evidence alone, and the cut-point search
/// can clip speech. The family's `integral_seconds` is exactly how much audio
/// its decoder context can serve in one prompt, so anything at or under it is
/// decoded whole and only longer recordings fall back to slices.
///
/// An explicitly requested [`crate::LongFormOptions`] is honored as-is: a
/// caller that asked for specific slicing gets it, and this only decides the
/// automatic policy.
fn scoped_slice_recording_fits_one_decode(
    model_architecture: &str,
    audio_duration_seconds: f32,
    requested: Option<&crate::LongFormOptions>,
) -> bool {
    if requested.is_some() {
        return false;
    }
    let crate::arch::OpenAsrLongformSliceShape::ScopedSlices {
        integral_seconds, ..
    } = crate::arch::longform_slice_shape_for_model_architecture(model_architecture)
    else {
        return false;
    };
    audio_duration_seconds <= integral_seconds
}

/// Installs the slice window an
/// [`OpenAsrLongformSliceShape::ScopedSlices`] family declares.
///
/// Unlike the safety caps below this is not a clamp in one direction: the
/// declared window is the family's decoder-context fact, so it replaces the
/// shared default whether that default was wider or (as with the 30s generic
/// target) much narrower. A family that folds a whole slice into one
/// autoregressive prompt gets *worse*, not safer, when handed thirty-second
/// windows -- the prompt overhead is paid per slice and its in-decoder speaker
/// numbering restarts at every seam. The safety caps still run afterwards and
/// may narrow this further; they only ever clamp downward, so the effective
/// window stays the min of every applicable rule.
///
/// Three shared options are also pinned for this shape, all consequences of
/// "the slice audio *is* the decode unit":
/// - lead-in/lead-out padding is dropped, because such a family timestamps
///   relative to the buffer it was handed while the assembler maps slice-
///   relative times from `content_start_sample`; any padding is a straight
///   bias on every timestamp in the slice;
/// - prompt carry is disabled, because the decode prompt is a fixed
///   fine-tuned instruction, not a free-text context window;
/// - the slicing mode is pinned to the contiguous, full-coverage
///   [`LongFormMode::Energy`] planner, because `Auto` may elect a *packed*
///   layout that splices the recording's speech spans together and elides
///   everything its energy VAD read as silence. See below.
///
/// The packed layout is a legitimate optimization for a family that decodes a
/// slice as plain speech-to-text, but it is structurally wrong here on two
/// counts. It hands the decoder audio that does not exist -- turns spliced
/// end-to-end across a seam of a few zero samples -- while this family's whole
/// job is to tell speakers apart from continuous acoustic context, pauses
/// included. And its timeline map collapses each elided region to a seam, so a
/// segment whose two ends straddle one is stretched across audio the decoder
/// never saw: a real Mandarin meeting recording (speech peaking near -44 dBFS,
/// well under the pipeline's -38 dBFS `energy_silence_threshold_db`) had 47% of
/// its 360s elided, and the surviving turns were blanketed over the gaps
/// (one 5-character turn spanning 30.7s across two other speakers' lost
/// content). `enforce_coverage_dominance` could not catch that case at the
/// time: it measured "audible" against the same floor the energy VAD elides
/// by, so the guard read its own input back and always said no. That closed
/// loop has since been broken -- the guard now judges against a
/// recording-relative reference (`longform::audibility`) and does disqualify
/// this shape -- but the pin stays, because it is not a level question here:
/// splicing away the pauses is wrong for a family whose job is to tell
/// speakers apart from continuous acoustic context, at any level. This shape
/// takes the planner that cannot elide at all -- the energy planner slices
/// contiguously from the first sample to the last and only chooses *where* to
/// cut (see `plan_energy_slices_contiguous`).
fn apply_scoped_slice_longform_window_policy(
    model_architecture: &str,
    options: &mut crate::LongFormOptions,
    provenance: &mut Vec<String>,
) {
    let crate::arch::OpenAsrLongformSliceShape::ScopedSlices {
        target_seconds,
        max_seconds,
        ..
    } = crate::arch::longform_slice_shape_for_model_architecture(model_architecture)
    else {
        return;
    };
    options.mode = LongFormMode::Energy;
    options.chunk_seconds = target_seconds;
    options.max_chunk_seconds = max_seconds.max(target_seconds);
    options.min_chunk_seconds = options.min_chunk_seconds.min(target_seconds);
    options.padding_seconds = 0.0;
    options.carry_prompt_across_slices = false;
    provenance.push(format!(
        "core.native.longform.policy:scoped-slices,mode=energy,target_seconds={target_seconds},max_seconds={max_seconds}"
    ));
}

/// Applies every family-specific longform safety cap for `model_architecture`.
/// Two independent caps can apply to the same architecture (e.g.
/// firered-aed/cohere/moonshine carry both), and they are combined by never
/// letting a later cap *widen* a value an earlier cap already narrowed --
/// each helper only clamps downward, so the net effect is always the min of
/// whichever caps apply. Order does not matter for that reason; the
/// repetition-guard profile runs first only because it is the
/// longer-standing check.
fn apply_longform_safety_policy(
    model_architecture: &str,
    options: &mut crate::LongFormOptions,
    provenance: &mut Vec<String>,
) {
    apply_invocation_span_longform_policy(model_architecture, options, provenance);
    apply_conservative_seq2seq_longform_safety_policy(model_architecture, options, provenance);
    apply_encoder_attention_span_longform_safety_policy(model_architecture, options, provenance);
}

/// Enforces the family runtime's semantic maximum for one executor call.
/// This is not memory-pressure adaptation: a fixed-window frontend would
/// otherwise discard audio, while explicit-limit families would fail only
/// after slicing had already selected an invalid unit. The bound is stable
/// across execution candidates, so CPU/GPU choice cannot change transcript
/// segmentation.
fn apply_invocation_span_longform_policy(
    model_architecture: &str,
    options: &mut crate::LongFormOptions,
    provenance: &mut Vec<String>,
) {
    let Some(max_seconds) = OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(model_architecture)
        .and_then(|descriptor| descriptor.max_single_invocation_seconds())
    else {
        return;
    };
    if clamp_longform_chunks_to_ceiling(options, max_seconds) {
        provenance.push(format!(
            "core.native.longform.policy:invocation-span-cap={max_seconds}"
        ));
    }
}

/// Caps longform chunking for the decode-side `ConservativeSeq2SeqV1`
/// repetition-guard profile (issue #60): plain `<sos>`-prompted AED decoders
/// with a small effective context (cohere-transcribe, moonshine, firered-aed)
/// repeat/hallucinate on long, pause-free chunks, so prompt carry across
/// slices is disabled here. The chunk-length cap itself
/// (`CONSERVATIVE_SEQ2SEQ_LONGFORM_MAX_CHUNK_SECONDS`) is *not* the
/// repetition fix -- that is the shared greedy-decode driver's
/// degenerate-loop guard, which applies regardless of chunk length -- so
/// this cap uses the same industry-surveyed default as the encoder-memory
/// cap below rather than an arbitrarily tighter number. This is a decode
/// semantics cap, independent of the encoder-memory cap below (which caps a
/// different, larger set of architectures for a different reason); the two
/// happen to share the same default value today, but remain conceptually
/// distinct and compose by taking the min if a future override diverges them.
fn apply_conservative_seq2seq_longform_safety_policy(
    model_architecture: &str,
    options: &mut crate::LongFormOptions,
    provenance: &mut Vec<String>,
) {
    let Ok(policy) = resolve_builtin_decode_policy_for_architecture(model_architecture) else {
        return;
    };
    if policy.longform_profile != BuiltinDecodePolicyLongformProfile::ConservativeSeq2SeqV1 {
        return;
    }
    let mut changed = false;
    if options.chunk_seconds > CONSERVATIVE_SEQ2SEQ_LONGFORM_MAX_CHUNK_SECONDS {
        options.chunk_seconds = CONSERVATIVE_SEQ2SEQ_LONGFORM_MAX_CHUNK_SECONDS;
        changed = true;
    }
    if options.max_chunk_seconds > CONSERVATIVE_SEQ2SEQ_LONGFORM_MAX_CHUNK_SECONDS {
        options.max_chunk_seconds = CONSERVATIVE_SEQ2SEQ_LONGFORM_MAX_CHUNK_SECONDS;
        changed = true;
    }
    if options.min_chunk_seconds > CONSERVATIVE_SEQ2SEQ_LONGFORM_MAX_CHUNK_SECONDS {
        options.min_chunk_seconds = CONSERVATIVE_SEQ2SEQ_LONGFORM_MAX_CHUNK_SECONDS;
        changed = true;
    }
    if options.max_chunk_seconds < options.chunk_seconds {
        options.max_chunk_seconds = options.chunk_seconds;
        changed = true;
    }
    if options.min_chunk_seconds > options.chunk_seconds {
        options.min_chunk_seconds = options.chunk_seconds;
        changed = true;
    }
    if (options.overlap_seconds - CONSERVATIVE_SEQ2SEQ_LONGFORM_OVERLAP_SECONDS).abs()
        > f32::EPSILON
    {
        options.overlap_seconds = CONSERVATIVE_SEQ2SEQ_LONGFORM_OVERLAP_SECONDS;
        changed = true;
        provenance.push(format!(
            "core.native.longform.policy:conservative-seq2seq-overlap={}",
            CONSERVATIVE_SEQ2SEQ_LONGFORM_OVERLAP_SECONDS
        ));
    }
    if options.carry_prompt_across_slices {
        options.carry_prompt_across_slices = false;
        changed = true;
        provenance.push(
            "core.native.longform.policy:conservative-seq2seq-disable-prompt-carry".to_string(),
        );
    }
    if changed {
        provenance.push(format!(
            "core.native.longform.policy:conservative-seq2seq-chunk-cap={}",
            CONSERVATIVE_SEQ2SEQ_LONGFORM_MAX_CHUNK_SECONDS
        ));
    }
}

/// Caps longform chunking to the architecture's declared
/// `OpenAsrEncoderAttentionSpan::GlobalQuadratic` safety ceiling (issue #68):
/// a global-quadratic-attention encoder's activation memory grows with the
/// square of chunk length, so a long, pause-free recording that lets the
/// auto/energy/VAD slicer grow a chunk up to the (much larger)
/// `LongFormOptions::default().max_chunk_seconds` can exhaust RAM. Whisper
/// (`FixedWindow`) and zipformer (`LocalChunked`) need no cap here -- their
/// encoders do not scale with the logical chunk length -- so this is a no-op
/// for them. Only ever clamps downward, so it composes safely with
/// `apply_conservative_seq2seq_longform_safety_policy`'s tighter cap on the
/// families that carry both.
fn apply_encoder_attention_span_longform_safety_policy(
    model_architecture: &str,
    options: &mut crate::LongFormOptions,
    provenance: &mut Vec<String>,
) {
    let Some(descriptor) =
        OpenAsrArchitectureRegistry::with_builtins().find_by_model_architecture(model_architecture)
    else {
        return;
    };
    let Some(max_safe_chunk_seconds) = descriptor.longform_max_safe_chunk_seconds() else {
        return;
    };
    if clamp_longform_chunks_to_encoder_memory_ceiling(options, max_safe_chunk_seconds) {
        provenance.push(format!(
            "core.native.longform.policy:encoder-attention-span-chunk-cap={max_safe_chunk_seconds}"
        ));
    }
}

/// The clamp itself, split out from the registry lookup so it can be exercised
/// against a ceiling that differs from `LongFormOptions`' default chunk
/// length. That the two can differ is the whole point of the split described
/// on `arch::DEFAULT_ENCODER_SAFE_CHUNK_SECONDS`: this function must narrow
/// toward whatever memory ceiling it is given, never toward the chunk length
/// the slicer happens to prefer. Returns whether anything moved.
fn clamp_longform_chunks_to_encoder_memory_ceiling(
    options: &mut crate::LongFormOptions,
    max_safe_chunk_seconds: f32,
) -> bool {
    clamp_longform_chunks_to_ceiling(options, max_safe_chunk_seconds)
}

fn clamp_longform_chunks_to_ceiling(
    options: &mut crate::LongFormOptions,
    max_chunk_seconds: f32,
) -> bool {
    let mut changed = false;
    if options.chunk_seconds > max_chunk_seconds {
        options.chunk_seconds = max_chunk_seconds;
        changed = true;
    }
    if options.max_chunk_seconds > max_chunk_seconds {
        options.max_chunk_seconds = max_chunk_seconds;
        changed = true;
    }
    if options.min_chunk_seconds > max_chunk_seconds {
        options.min_chunk_seconds = max_chunk_seconds;
        changed = true;
    }
    if options.max_chunk_seconds < options.chunk_seconds {
        options.max_chunk_seconds = options.chunk_seconds;
        changed = true;
    }
    if options.min_chunk_seconds > options.chunk_seconds {
        options.min_chunk_seconds = options.chunk_seconds;
        changed = true;
    }
    changed
}

fn combined_longform_provenance(policy: &[String], plan: &[String]) -> Vec<String> {
    let mut combined = Vec::with_capacity(policy.len().saturating_add(plan.len()));
    combined.extend(policy.iter().cloned());
    combined.extend(plan.iter().cloned());
    combined
}

fn normalize_and_validate_model_id(request: &TranscriptionRequest) -> Result<&str, BackendError> {
    let requested_model_id = request.model_id.trim();
    if requested_model_id == NATIVE_RUNTIME_MODEL_ID_AUTO {
        return Ok(requested_model_id);
    }
    if let Err(error) = parse_model_ref(requested_model_id) {
        return Err(BackendError::NativeFailClosed {
            reason: format!(
                "model '{}' is not a valid model id: {error}",
                request.model_id
            ),
        });
    }
    Ok(requested_model_id)
}

fn validate_runtime_source_and_select_adapter(
    requested_model_id: &str,
    verified_pack: &VerifiedPack,
    metadata: &BTreeMap<String, String>,
) -> Result<GgmlFamilyAdapterDescriptor, BackendError> {
    let normalized_model_id =
        super::native_model_id::resolve_native_runtime_model_identity_from_string_metadata(
            metadata,
            verified_pack.preflight().runtime_source().path(),
            None,
        )
        .map_err(|error| BackendError::NativeFailClosed {
            reason: error.to_string(),
        })?
        .model_id;
    let selected = OpenAsrArchitectureRegistry::with_builtins()
        .select_ggml_adapter_from_gguf_metadata_v1(metadata)
        .map(|descriptor| descriptor.ggml_family_adapter_descriptor())
        .map_err(map_family_selection_error)?;
    if requested_model_id != NATIVE_RUNTIME_MODEL_ID_AUTO
        && !super::native_runtime_model_ref_matches_verified_pack(
            requested_model_id,
            &normalized_model_id,
            verified_pack,
            &selected,
        )
    {
        return Err(BackendError::NativeModelSelectionMismatch {
            requested: requested_model_id.to_string(),
            local: normalized_model_id,
        });
    }
    Ok(selected)
}

/// Whether a requested model ref names the same native pack as a local runtime
/// source id. This is the single tolerant matcher for the "bare id contract":
/// packs burn no quant tag into `openasr.model.id`, so a quant-pinned request
/// (`family:quant`) matches a bare runtime id (`family`) -- the
/// `(Some(_), None) => true` arm below is load-bearing. Quant tags on both
/// sides compare through `canonical_quant_tag` so catalog aliases (`q8` vs
/// `q8_0`) match. Verified-pack gates use this as their ordinary spelling
/// check before considering a narrowly content-bound published compatibility.
pub fn native_runtime_model_refs_match(requested: &str, runtime_source_id: &str) -> bool {
    let requested = requested.trim();
    let runtime_source_id = runtime_source_id.trim();
    if requested == runtime_source_id {
        return true;
    }
    let Ok(requested_ref) = parse_model_ref(requested) else {
        return false;
    };
    let Some(runtime_ref) = parse_native_runtime_source_ref(runtime_source_id) else {
        return false;
    };
    if requested_ref.family != runtime_ref.family {
        return false;
    }
    match (requested_ref.tag.as_deref(), runtime_ref.tag.as_deref()) {
        (Some(requested_quant), Some(runtime_quant)) => {
            crate::canonical_quant_tag(requested_quant) == crate::canonical_quant_tag(runtime_quant)
        }
        (Some(_), None) => true,
        _ => false,
    }
}

/// Renders a diagnostic string for a native model mismatch error: the
/// requested ref's normalized `family:canonical_quant` form (when parseable)
/// and the loaded runtime source id's normalized form, computed with the same
/// legacy-hyphen-aware parsing `native_runtime_model_refs_match` uses. Lets an
/// operator see *why* two apparently-similar ids failed to match (a genuinely
/// different family, vs. an unrecognized quant alias spelling) instead of
/// only the raw strings, which are often identical-looking after truncation
/// or already differ only in a quant suffix that a human cannot canonicalize
/// by eye.
pub fn describe_native_runtime_model_mismatch(requested: &str, runtime_source_id: &str) -> String {
    let requested_normalized = parse_model_ref(requested.trim())
        .map(|r| normalized_model_ref_display(&r))
        .unwrap_or_else(|_| requested.trim().to_string());
    let runtime_normalized = parse_native_runtime_source_ref(runtime_source_id.trim())
        .map(|r| normalized_model_ref_display(&r))
        .unwrap_or_else(|| runtime_source_id.trim().to_string());
    format!(
        "requested model normalizes to '{requested_normalized}', loaded native runtime source normalizes to '{runtime_normalized}'"
    )
}

fn normalized_model_ref_display(model_ref: &crate::registry::ModelRef) -> String {
    match &model_ref.tag {
        Some(tag) => format!("{}:{}", model_ref.family, crate::canonical_quant_tag(tag)),
        None => model_ref.family.clone(),
    }
}

/// Parses a native runtime pack's source id for matching purposes.
///
/// Prefers the standard `family:quant` colon form used everywhere else in the
/// catalog/registry contract. Falls back to splitting a legacy hyphen-joined
/// `family-quant` id when the trailing hyphen segment is a recognized quant
/// alias token (`crate::registry::is_recognized_quant_alias_token`, the same
/// table `canonical_quant_tag` uses -- no separate mapping is maintained
/// here). That hyphen form is not the catalog convention, but it is what an
/// older conversion tool (`tooling/mimo-asr/convert_mimo_asr.py`, fixed to
/// emit colon-joined ids going forward) baked into `openasr.model.id`
/// metadata for already-published packs; this keeps those packs matchable
/// without requiring every shipped asset to be reconverted and republished.
fn parse_native_runtime_source_ref(runtime_source_id: &str) -> Option<crate::registry::ModelRef> {
    let parsed = parse_model_ref(runtime_source_id).ok()?;
    if parsed.tag.is_some() {
        return Some(parsed);
    }
    if let Some((family, tag)) = parsed.family.rsplit_once('-').filter(|(family, alias)| {
        !family.is_empty() && crate::registry::is_recognized_quant_alias_token(alias)
    }) {
        return Some(crate::registry::ModelRef {
            family: family.to_string(),
            tag: Some(tag.to_string()),
        });
    }
    Some(parsed)
}

fn map_family_selection_error(error: GgmlFamilyAdapterSelectionError) -> BackendError {
    match error {
        GgmlFamilyAdapterSelectionError::InvalidMetadata(OasrV1MetadataError::MissingKey(key)) => {
            BackendError::NativeFailClosed {
                reason: format!(
                    "gguf metadata is missing required OASR v1 key '{key}' for family adapter selection"
                ),
            }
        }
        GgmlFamilyAdapterSelectionError::InvalidMetadata(OasrV1MetadataError::EmptyValue(key)) => {
            BackendError::NativeFailClosed {
                reason: format!(
                    "gguf metadata key '{key}' must be non-empty for family adapter selection"
                ),
            }
        }
        GgmlFamilyAdapterSelectionError::Ambiguous { adapter_ids } => {
            BackendError::NativeFailClosed {
                reason: format!(
                    "gguf metadata matched multiple family adapters: {}",
                    adapter_ids.join(", ")
                ),
            }
        }
        _ => BackendError::NativeFailClosed {
            reason: "gguf metadata does not match any registered family adapter".to_string(),
        },
    }
}

fn dispatch_error_to_backend(
    error: GgmlAsrExecutionError,
    execution_context: &crate::RequestExecutionContext,
) -> BackendError {
    // L1 cooperative cancel (token-loop) and L0 slice cancel both leave the
    // active control flagged. Prefer the typed cancel surface over a generic
    // fail-closed reason so CLI/native and server agree on
    // `BackendError::TranscriptionCanceled` (HTTP 409). Also recognize the
    // stable cancel marker embedded in family executor reason strings as a
    // belt-and-suspenders signal for a decode path that stringified a
    // `Canceled` variant before it reached here.
    if execution_context.is_canceled() || is_cooperative_cancel_reason(&error.to_string()) {
        return BackendError::TranscriptionCanceled;
    }
    match error {
        GgmlAsrExecutionError::ExecutorUnavailable { .. } => BackendError::NativeFailClosed {
            reason: format!(
                "{error}. Native ggml dispatch does not fall back to non-GGUF runtime paths."
            ),
        },
        GgmlAsrExecutionError::ServeBatchUnavailable { reason, retryable } => {
            BackendError::ServeBatchUnavailable { reason, retryable }
        }
        GgmlAsrExecutionError::ExecutionRoute(error) => {
            BackendError::from_execution_route_error(error)
        }
        other => {
            // Family executors historically stringify `GgmlCpuGraphError` into
            // `ExecutorFailed.reason`. Recover the typed route failure when the
            // Display text still embeds it so Exact/init failures stay
            // `ExecutionDevice*` end-to-end.
            if let Some(route_error) =
                crate::device::execution_route::ExecutionRouteError::from_embedded_message(
                    &other.to_string(),
                )
            {
                return BackendError::from_execution_route_error(route_error);
            }
            BackendError::NativeFailClosed {
                reason: other.to_string(),
            }
        }
    }
}

/// Stable substrings shared by cooperative-cancel error paths.
///
/// Matches:
/// - `Seq2SeqGreedyDecodeError::Canceled` / family greedy bridges
///   (`"... canceled by transcription control"`)
/// - `GgmlCpuGraphError::Aborted` (`"aborted by cancel request"`)
///
/// Used as a belt-and-suspenders signal when the active control handle is no
/// longer bound on this thread.
fn is_cooperative_cancel_reason(reason: &str) -> bool {
    reason.contains("canceled by transcription control")
        || reason.contains("aborted by cancel request")
}

/// Builds the request's resolved runtime from the exact candidate route passed
/// by the policy loop. Recomputing it per attempt is required because a retry
/// can change both provider and placement.
fn run_dispatch_once(
    dispatch: &GgmlAsrExecutionDispatch,
    execution_services: &Arc<NativeExecutionServices>,
    verified_pack: &VerifiedPack,
    selected_family: &GgmlFamilyAdapterDescriptor,
    samples: PcmSlice,
    request_options: GgmlAsrExecutionOptions,
    backend_preference: GgmlAsrBackendPreference,
    resolved_preference: Option<RequestBackendPreference>,
    auto_gpu_policy: crate::ggml_runtime::AutoGpuPolicy,
    execution_context: &Arc<crate::RequestExecutionContext>,
) -> Result<GgmlAsrExecutionResult, BackendError> {
    let runtime_preflight = verified_pack.preflight();
    let resolved_runtime = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
        resolved_preference,
        auto_gpu_policy,
    );
    let execution_request = GgmlAsrExecutionViewRequest {
        execution_services: Arc::clone(execution_services),
        decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
        verified_pack: verified_pack.clone(),
        selected_family: selected_family.clone(),
        prepared_audio: GgmlAsrPreparedAudioView::mono_16khz_shared(samples),
        request_options,
        backend_preference,
        resolved_runtime,
        execution_context: Arc::clone(execution_context),
    };
    let planning_input =
        crate::models::ggml_asr_executor::GgmlAsrDecoderStatePlanningInput::for_offline_view_request(
            runtime_preflight,
            &execution_request.prepared_audio,
            &execution_request.request_options,
            execution_request.resolved_runtime.backend(),
        )
        .map_err(|error| dispatch_error_to_backend(error.into(), execution_context))?;
    let decoder_state = dispatch
        .plan_decoder_state(selected_family, &planning_input)
        .map_err(|error| dispatch_error_to_backend(error, execution_context))?;
    let execution_request = GgmlAsrExecutionViewRequest {
        decoder_state,
        ..execution_request
    };
    let _thread_override = install_request_inference_threads_override(
        execution_request.request_options.inference_threads,
    );
    let result = dispatch
        .execute_view(&execution_request)
        .map_err(|error| dispatch_error_to_backend(error, execution_context))?;
    Ok(result)
}

fn resolve_native_execution_plan(
    execution_services: &NativeExecutionServices,
    selected_family: &GgmlFamilyAdapterDescriptor,
    intent: ExecutionIntent,
) -> Result<ExecutionPlan, BackendError> {
    let inventory = enumerate_compute_devices_from_ggml(&crate::ggml_available_devices());
    execution_services
        .policy_resolver()
        .resolve(
            intent,
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                selected_family.model_architecture,
            ),
            selected_family.execution_capabilities,
            &inventory,
        )
        .map_err(execution_policy_error_to_backend)
}

fn resolve_auxiliary_execution_plan(
    execution_services: &NativeExecutionServices,
    architecture_id: &'static str,
    request_intent: &ExecutionIntent,
) -> Result<ExecutionPlan, BackendError> {
    crate::models::policy_resolved_aux_runtime::resolve_auxiliary_execution_plan(
        execution_services,
        architecture_id,
        request_intent,
    )
    .map_err(|error| BackendError::NativeFailClosed {
        reason: error.to_string(),
    })
}

fn resolve_longform_vad_execution_plan(
    execution_services: &NativeExecutionServices,
    request_intent: &ExecutionIntent,
) -> Result<ExecutionPlan, BackendError> {
    let inventory = enumerate_compute_devices_from_ggml(&crate::ggml_available_devices());
    execution_services
        .policy_resolver()
        .resolve(
            request_intent.clone(),
            crate::diarize::vad::STREAM_VAD_OFFLINE_AUTO_GPU_POLICY,
            crate::diarize::vad::stream_vad_execution_capabilities(),
            &inventory,
        )
        .map_err(execution_policy_error_to_backend)
}

/// Execute one independent auxiliary model stage transactionally. A stage may
/// deliberately treat non-resource errors as a no-op (punctuation does); a
/// typed allocator/device failure still invalidates that candidate even when
/// the inner stage swallowed its ordinary error, so Auto can try its next
/// semantics-equivalent placement instead of silently dropping the stage.
fn run_auxiliary_stage_with_policy<T>(
    execution_services: &NativeExecutionServices,
    execution_plan: &ExecutionPlan,
    stage: &'static str,
    mut operation: impl FnMut(&ExecutionCandidate) -> Result<T, BackendError>,
) -> Result<T, PolicyResolvedAuxRuntimeError<BackendError>> {
    let candidates = execution_plan.candidates();
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        let attempt = crate::models::native_execution_services::run_execution_candidate_attempt(
            execution_services,
            candidate,
            || operation(candidate),
        );
        match (attempt.result, attempt.candidate_failure) {
            (Ok(value), None) => return Ok(value),
            (Err(error), None) => {
                return Err(PolicyResolvedAuxRuntimeError::Operation(error));
            }
            (result, Some(failure)) => {
                if candidate_index + 1 == candidates.len() {
                    return Err(PolicyResolvedAuxRuntimeError::CandidatesExhausted {
                        stage,
                        failure,
                        source: crate::models::native_execution_services::execution_candidate_failure_source(result),
                    });
                }
                crate::stage_timing::log_detail_event(
                    "native_transcribe",
                    format_args!(
                        "stage=auxiliary_execution_candidate event=retry auxiliary_stage={stage} provider={} placement={:?} failure={:?} operation={}",
                        candidate.device.route.provider,
                        candidate.placement,
                        failure.kind,
                        failure.operation,
                    ),
                );
                let _ =
                    crate::models::native_execution_services::execution_candidate_failure_source(
                        result,
                    );
            }
        }
    }
    Err(PolicyResolvedAuxRuntimeError::EmptyPlan { stage })
}

fn required_auxiliary_stage_error(
    error: PolicyResolvedAuxRuntimeError<BackendError>,
) -> BackendError {
    match error {
        PolicyResolvedAuxRuntimeError::Operation(error) => error,
        error => BackendError::NativeFailClosed {
            reason: error.to_string(),
        },
    }
}

fn execution_policy_error_to_backend(error: ExecutionPolicyError) -> BackendError {
    match error {
        ExecutionPolicyError::Route(error) => BackendError::from_execution_route_error(error),
        other => BackendError::NativeFailClosed {
            reason: format!("could not resolve an execution candidate: {other}"),
        },
    }
}

fn resolved_runtime_for_candidate(
    candidate: &ExecutionCandidate,
    auto_gpu_policy: crate::ggml_runtime::AutoGpuPolicy,
) -> crate::ggml_runtime::ResolvedFamilyRuntimeInput {
    crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
        request_backend_preference_for_candidate(candidate),
        auto_gpu_policy,
    )
}

fn request_backend_preference_for_candidate(
    candidate: &ExecutionCandidate,
) -> Option<RequestBackendPreference> {
    match candidate.placement {
        ExecutionPlacement::CpuOnly => Some(RequestBackendPreference::CpuOnly),
        ExecutionPlacement::FullDevice | ExecutionPlacement::Hybrid => Some(
            RequestBackendPreference::Exact(candidate.device.route.clone()),
        ),
    }
}

/// Whole-slice RMS against an absolute dBFS line. The one caller is the
/// opt-in `suppress_silent_slices` skip (default off), which is a *decision*
/// use of `energy_silence_threshold_db` -- it chooses not to decode a slice.
/// It is deliberately not the standard any plan validation measures against;
/// see `longform::audibility` for why judging an elision by the same line
/// that produced it is a closed loop.
fn is_effectively_silent(samples: &[f32], threshold_db: f32) -> bool {
    if samples.is_empty() {
        return true;
    }
    let mut sum_sq = 0.0f64;
    for sample in samples {
        let value = *sample as f64;
        sum_sq += value * value;
    }
    let rms = (sum_sq / samples.len() as f64).sqrt() as f32;
    if rms <= f32::EPSILON {
        return true;
    }
    let db = 20.0 * rms.log10();
    db <= threshold_db
}

fn append_context_tail(existing: &str, new_text: &str, max_chars: usize) -> String {
    let merged = if existing.trim().is_empty() {
        new_text.trim().to_string()
    } else {
        format!("{} {}", existing.trim(), new_text.trim())
    };
    take_tail_chars(&merged, max_chars)
}

fn take_tail_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let total = value.chars().count();
    value.chars().skip(total - max_chars).collect()
}

fn build_longform_metadata(
    options: &crate::LongFormOptions,
    chunk_count: usize,
    skipped_silent_chunks: usize,
    duplicate_merge_count: usize,
    slice_kind_summary: &'static str,
    timeline_kind: &'static str,
    extra_provenance: &[String],
    resolved_backend: GgmlCpuGraphBackend,
) -> TranscriptionLongFormMetadata {
    let mode = match options.mode {
        LongFormMode::Off => "off",
        LongFormMode::Auto => "auto",
        LongFormMode::Fixed => "fixed",
        LongFormMode::Energy => "energy",
        LongFormMode::Vad => "vad",
    };
    let mut provenance = vec![
        format!("core.longform.plan:{mode}"),
        format!("core.longform.slice-kind:{slice_kind_summary}"),
        format!("core.longform.timeline:{timeline_kind}"),
        format!(
            "core.native.backend:{}",
            native_runtime_backend_label(resolved_backend)
        ),
        "core.longform.assembler".to_string(),
        "core.native.ggml".to_string(),
    ];
    provenance.extend(extra_provenance.iter().cloned());
    TranscriptionLongFormMetadata {
        chunk_count,
        skipped_silent_chunks,
        duplicate_merge_count,
        provenance,
    }
}

fn summarize_slice_kinds(slices: &[crate::AudioSlice]) -> &'static str {
    let has_vad = slices
        .iter()
        .any(|slice| matches!(slice.kind, AudioSliceKind::Vad));
    let has_energy = slices
        .iter()
        .any(|slice| matches!(slice.kind, AudioSliceKind::Energy));
    let has_fixed = slices
        .iter()
        .any(|slice| matches!(slice.kind, AudioSliceKind::Fixed));
    let has_full = slices
        .iter()
        .any(|slice| matches!(slice.kind, AudioSliceKind::Full));
    if has_vad {
        "vad"
    } else if has_energy {
        "energy"
    } else if has_fixed {
        "fixed"
    } else if has_full {
        "full"
    } else {
        "unknown"
    }
}

fn with_longform_metadata(
    mut transcription: Transcription,
    metadata: Option<TranscriptionLongFormMetadata>,
) -> Transcription {
    transcription.longform = metadata;
    transcription
}

fn normalize_transcription_segments(
    mut transcription: Transcription,
    fallback_start_seconds: f32,
    fallback_end_seconds: f32,
) -> Transcription {
    let mut fallback_start = fallback_start_seconds.max(0.0);
    let mut fallback_end = fallback_end_seconds.max(fallback_start);
    if !fallback_start.is_finite() {
        fallback_start = 0.0;
    }
    if !fallback_end.is_finite() {
        fallback_end = fallback_start;
    }
    let trimmed_text = transcription.text.trim().to_string();
    if transcription.segments.is_empty() {
        if trimmed_text.is_empty() {
            transcription.text = String::new();
            return transcription;
        }
        transcription.text = trimmed_text.clone();
        transcription.segments = vec![Segment {
            start: fallback_start,
            end: fallback_end,
            text: trimmed_text,
            speaker: None,
            speaker_label: None,
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words: Vec::new(),
        }];
        return transcription;
    }

    let mut normalized = Vec::with_capacity(transcription.segments.len());
    let mut previous_end = fallback_start;
    for segment in transcription.segments {
        let text = segment.text.trim().to_string();
        if text.is_empty() {
            continue;
        }
        let mut start = if segment.start.is_finite() {
            segment.start.max(0.0)
        } else {
            previous_end
        };
        if start < previous_end {
            start = previous_end;
        }
        let mut end = if segment.end.is_finite() {
            segment.end.max(start)
        } else {
            start
        };
        if end < start {
            end = start;
        }
        normalized.push(Segment {
            start,
            end,
            text,
            speaker: segment.speaker,
            speaker_label: segment.speaker_label,
            speaker_person_id: segment.speaker_person_id,
            speaker_snapshot_label: segment.speaker_snapshot_label,
            words: segment.words,
        });
        previous_end = end;
    }

    if normalized.is_empty() {
        if trimmed_text.is_empty() {
            transcription.text = String::new();
            transcription.segments = Vec::new();
            return transcription;
        }
        transcription.text = trimmed_text.clone();
        transcription.segments = vec![Segment {
            start: fallback_start,
            end: fallback_end,
            text: trimmed_text,
            speaker: None,
            speaker_label: None,
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words: Vec::new(),
        }];
        return transcription;
    }

    if normalized.len() == 1
        && fallback_end > fallback_start
        && normalized[0].end.is_finite()
        && normalized[0].end < (fallback_end * 0.95)
    {
        normalized[0].start = normalized[0].start.min(fallback_start);
        normalized[0].end = fallback_end.max(normalized[0].start);
    }

    transcription.segments = normalized;
    if trimmed_text.is_empty() {
        transcription.text = transcription
            .segments
            .iter()
            .map(|segment| segment.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
    } else {
        transcription.text = trimmed_text;
    }
    transcription
}

fn longform_prompt_carry_mode(
    options: &crate::LongFormOptions,
    model_architecture: &str,
) -> LongformPromptCarryMode {
    if matches!(options.mode, LongFormMode::Off) || !options.carry_prompt_across_slices {
        return LongformPromptCarryMode::Disabled;
    }
    resolve_builtin_decode_policy_for_architecture(model_architecture)
        .map(|policy| match policy.longform_prompt_carry_mode {
            BuiltinDecodePolicyLongformPromptCarryMode::Disabled => {
                LongformPromptCarryMode::Disabled
            }
            BuiltinDecodePolicyLongformPromptCarryMode::Text => LongformPromptCarryMode::Text,
            BuiltinDecodePolicyLongformPromptCarryMode::TokenHistory => {
                LongformPromptCarryMode::TokenHistory
            }
        })
        .unwrap_or(LongformPromptCarryMode::Text)
}

fn prefers_cpu_decoder_for_multichunk_metal(model_architecture: &str) -> bool {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(model_architecture)
        .is_some_and(|descriptor| {
            descriptor
                .optimization_contract
                .prefer_cpu_decoder_for_multichunk_metal
        })
}

fn should_prefer_cpu_decoder_for_multichunk_metal(
    model_architecture: &str,
    request_intent: &ExecutionIntent,
    chunk_count: usize,
    resolved_backend: GgmlCpuGraphBackend,
) -> bool {
    // This is an Auto-mode performance hint, never a license to rewrite an
    // explicit execution contract. If the user selected any accelerated
    // target (generic or provider-constrained), every neural stage stays on
    // that resolved accelerator even when a CPU decoder benchmarked faster.
    matches!(request_intent, ExecutionIntent::Auto)
        && chunk_count > 1
        && resolved_backend == GgmlCpuGraphBackend::Metal
        && prefers_cpu_decoder_for_multichunk_metal(model_architecture)
}

/// The `core.native.backend` provenance label. Callers must pass the
/// family-resolved backend (`ResolvedFamilyRuntimeInput::resolve`, keyed by
/// this family's `auto_gpu_policy` capability declaration) -- never the
/// generic ungated resolution, which drifts from reality for any family
/// whose policy pins (or platform-scopes) Auto away from a backend, exactly
/// the bug that produced a `core.native.backend:metal` label on a dolphin
/// Auto request that in fact ran entirely on CPU (before dolphin's own gate
/// flipped to GPU-enabled).
fn native_runtime_backend_label(backend: GgmlCpuGraphBackend) -> &'static str {
    match backend {
        GgmlCpuGraphBackend::Cpu => "cpu",
        GgmlCpuGraphBackend::Metal => "metal",
        GgmlCpuGraphBackend::Gpu => "gpu",
    }
}

/// Best-effort quant tag for the `stage=request_context` log line without a
/// second GGUF/metadata read. The resolved request model ref is authoritative
/// for current content-addressed installs; their path parent is a SHA-256
/// digest, never a quant tag. The parent-directory fallback exists only for
/// legacy `<model>/<quant>/<pack>.oasr` layouts and is accepted when it is a
/// known quant token. Arbitrary path segments become `"unknown"` rather than
/// fabricated telemetry.
fn quant_tag_for_log(requested_model_id: &str, runtime_pack_path: &Path) -> String {
    let from_request_tag = parse_model_ref(requested_model_id)
        .ok()
        .and_then(|reference| reference.tag);
    let from_parent_dir = runtime_pack_path
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str());
    for candidate in [from_request_tag.as_deref(), from_parent_dir]
        .into_iter()
        .flatten()
    {
        let canonical = crate::canonical_quant_tag(candidate);
        if matches!(canonical, "f32" | "fp16" | "q8_0" | "q4_k" | "q3_k") {
            return canonical.to_string();
        }
    }
    "unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_aligner_session_plan_matches_validated_provider_topologies() {
        assert_eq!(
            forced_aligner_session_plan(ExecutionPlacement::CpuOnly, ExecutionProvider::Cpu),
            Ok(ForcedAlignerSessionPlan::Uniform),
        );
        assert_eq!(
            forced_aligner_session_plan(ExecutionPlacement::FullDevice, ExecutionProvider::Metal),
            Ok(ForcedAlignerSessionPlan::Uniform),
        );
        assert_eq!(
            forced_aligner_session_plan(ExecutionPlacement::Hybrid, ExecutionProvider::Cuda),
            Ok(ForcedAlignerSessionPlan::GpuAudioHybrid),
        );
        assert_eq!(
            forced_aligner_session_plan(ExecutionPlacement::Hybrid, ExecutionProvider::Vulkan),
            Ok(ForcedAlignerSessionPlan::GpuAudioHybrid),
        );
        for (placement, provider) in [
            (ExecutionPlacement::Hybrid, ExecutionProvider::Metal),
            (ExecutionPlacement::FullDevice, ExecutionProvider::Cuda),
            (ExecutionPlacement::FullDevice, ExecutionProvider::Vulkan),
            (ExecutionPlacement::Hybrid, ExecutionProvider::Hip),
            (ExecutionPlacement::FullDevice, ExecutionProvider::Hip),
        ] {
            assert!(
                forced_aligner_session_plan(placement, provider).is_err(),
                "provider {provider:?} must fail closed for {placement:?}",
            );
        }
    }

    #[test]
    #[ignore = "host-local: needs the installed Qwen3-ForcedAligner q8 pack and an exact CUDA or Vulkan device"]
    fn forced_aligner_exact_hybrid_q8_matches_cpu_on_jfk() {
        let provider = asr_exact_smoke_provider(
            &std::env::var("OPENASR_FORCED_ALIGNER_SMOKE_PROVIDER")
                .expect("OPENASR_FORCED_ALIGNER_SMOKE_PROVIDER must be cuda or vulkan"),
        );
        let exact_intent = asr_exact_smoke_intent(provider, None);
        let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav,
            "forced-aligner Exact Hybrid smoke",
            "forced-aligner Exact Hybrid smoke",
        )
        .expect("JFK fixture loads");
        let pcm = crate::PcmBuffer::from_vec(samples);
        let audio_seconds = pcm.len() as f32 / 16_000.0;
        let text = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";
        let input = Transcription {
            text: text.to_string(),
            language: Some("English".to_string()),
            segments: vec![segment(0.0, audio_seconds, text)],
            ..Transcription::default()
        };
        let services = native_execution_services_for_test();
        let execution_context =
            crate::RequestExecutionContext::uncancellable("forced-aligner Exact Hybrid q8 smoke");
        let cpu = refine_transcription_word_timestamps_with_forced_aligner_policy(
            input.clone(),
            pcm.full_slice(),
            Some("English"),
            services.as_ref(),
            &ExecutionIntent::CpuOnly,
            &execution_context,
            None,
        )
        .expect("CPU forced alignment");

        let telemetry = crate::GgmlExecutionTelemetryCollector::new();
        let _telemetry_guard = telemetry.install();
        let accelerated = refine_transcription_word_timestamps_with_forced_aligner_policy(
            input,
            pcm.full_slice(),
            Some("English"),
            services.as_ref(),
            &exact_intent,
            &execution_context,
            None,
        )
        .expect("Exact Hybrid forced alignment");
        let observed = telemetry.snapshot();
        assert!(
            !observed.observed_compute_nodes_by_backend.is_empty(),
            "Exact Hybrid aligner must execute backend graph nodes"
        );
        let expected_backend_fragment = match provider {
            ExecutionProvider::Cuda => "cuda",
            ExecutionProvider::Vulkan => "vulkan",
            _ => unreachable!("provider parser accepts only CUDA/Vulkan"),
        };
        let observed_target_gpu = observed
            .observed_compute_nodes_by_backend
            .iter()
            .filter(|(backend, _)| {
                backend
                    .to_ascii_lowercase()
                    .contains(expected_backend_fragment)
            })
            .map(|(_, nodes)| nodes)
            .sum::<u64>();
        let observed_cpu = observed
            .observed_compute_nodes_by_backend
            .iter()
            .filter(|(backend, _)| backend.to_ascii_lowercase().contains("cpu"))
            .map(|(_, nodes)| nodes)
            .sum::<u64>();
        assert!(
            observed_target_gpu > 0 && observed_cpu > 0,
            "Exact {provider:?} Hybrid aligner must execute both its target GPU stage and CPU stages: {:?}",
            observed.observed_compute_nodes_by_backend,
        );
        assert!(
            observed
                .observed_compute_nodes_by_backend
                .keys()
                .all(|backend| {
                    let backend = backend.to_ascii_lowercase();
                    backend.contains(expected_backend_fragment) || backend.contains("cpu")
                }),
            "Exact {provider:?} Hybrid aligner observed an unrelated backend: {:?}",
            observed.observed_compute_nodes_by_backend,
        );

        assert_eq!(cpu.segments.len(), accelerated.segments.len());
        let mut drift_ms = Vec::new();
        let mut output_bytes = Vec::new();
        for (cpu_segment, accelerated_segment) in cpu.segments.iter().zip(&accelerated.segments) {
            assert_eq!(cpu_segment.text, accelerated_segment.text);
            assert_eq!(cpu_segment.words.len(), accelerated_segment.words.len());
            for (cpu_word, accelerated_word) in
                cpu_segment.words.iter().zip(&accelerated_segment.words)
            {
                assert_eq!(cpu_word.word, accelerated_word.word);
                drift_ms.push(
                    (f64::from(cpu_word.start) - f64::from(accelerated_word.start)).abs() * 1000.0,
                );
                drift_ms.push(
                    (f64::from(cpu_word.end) - f64::from(accelerated_word.end)).abs() * 1000.0,
                );
                output_bytes.extend_from_slice(accelerated_word.word.as_bytes());
                output_bytes.push(0);
                output_bytes.extend_from_slice(&accelerated_word.start.to_le_bytes());
                output_bytes.extend_from_slice(&accelerated_word.end.to_le_bytes());
            }
        }
        assert!(!drift_ms.is_empty(), "forced aligner must emit word spans");
        drift_ms.sort_by(f64::total_cmp);
        let median_ms = drift_ms[drift_ms.len() / 2];
        let p95_ms = drift_ms[(drift_ms.len() - 1) * 95 / 100];
        let max_ms = drift_ms[drift_ms.len() - 1];
        let output_sha256 = crate::testing::benchmark_sha256_bytes([output_bytes]);
        eprintln!(
            "FORCED_ALIGNER_EXACT_HYBRID_Q8 provider={provider:?} words={} endpoints={} median_ms={median_ms:.3} p95_ms={p95_ms:.3} max_ms={max_ms:.3} output_sha256={output_sha256} observed_compute_nodes={:?}",
            drift_ms.len() / 2,
            drift_ms.len(),
            observed.observed_compute_nodes_by_backend,
        );
        assert!(median_ms < 80.0, "median drift {median_ms:.3}ms");
        assert!(p95_ms <= 160.0, "p95 drift {p95_ms:.3}ms");
        assert!(max_ms <= 320.0, "maximum drift {max_ms:.3}ms");
    }

    #[test]
    fn multichunk_cpu_decoder_hint_is_auto_only() {
        let architecture = crate::arch::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID;
        assert!(should_prefer_cpu_decoder_for_multichunk_metal(
            architecture,
            &ExecutionIntent::Auto,
            2,
            GgmlCpuGraphBackend::Metal,
        ));
        assert!(!should_prefer_cpu_decoder_for_multichunk_metal(
            architecture,
            &ExecutionIntent::AcceleratedOnly,
            2,
            GgmlCpuGraphBackend::Metal,
        ));
        assert!(!should_prefer_cpu_decoder_for_multichunk_metal(
            architecture,
            &ExecutionIntent::ConstrainedAcceleratedOnly(AcceleratedDeviceConstraint::Provider(
                crate::device::execution_route::ExecutionProvider::Metal,
            ),),
            2,
            GgmlCpuGraphBackend::Metal,
        ));
    }

    #[test]
    fn request_context_quant_prefers_model_ref_over_content_digest() {
        let object = Path::new(
            "/models/objects/sha256/0044546efb95d4d08e85f5574da2b042a5a4fb2490678c666b65404f1ac94c04/content",
        );
        assert_eq!(
            quant_tag_for_log("moss-transcribe-diarize:q4", object),
            "q4_k"
        );
        assert_eq!(
            quant_tag_for_log("moss-transcribe-diarize", object),
            "unknown"
        );
    }

    #[test]
    fn request_context_quant_accepts_only_known_legacy_parent_tags() {
        assert_eq!(
            quant_tag_for_log(
                "moss-transcribe-diarize",
                Path::new("/models/moss-transcribe-diarize/q8_0/model.oasr")
            ),
            "q8_0"
        );
        assert_eq!(
            quant_tag_for_log(
                "moss-transcribe-diarize",
                Path::new("/arbitrary/not-a-quant/model.oasr")
            ),
            "unknown"
        );
    }

    #[test]
    fn native_request_auto_honors_backend_environment_as_typed_intent() {
        assert_eq!(
            request_execution_intent_with_backend_env(None, Some("cpu")),
            ExecutionIntent::CpuOnly
        );
        assert_eq!(
            request_execution_intent_with_backend_env(
                Some(crate::ExecutionTarget::Auto),
                Some("metal")
            ),
            ExecutionIntent::ConstrainedAcceleratedOnly(AcceleratedDeviceConstraint::Provider(
                ExecutionProvider::Metal
            ))
        );
        assert_eq!(
            request_execution_intent_with_backend_env(None, Some("rocm")),
            ExecutionIntent::ConstrainedAcceleratedOnly(AcceleratedDeviceConstraint::Provider(
                ExecutionProvider::Hip
            ))
        );
        assert_eq!(
            request_execution_intent_with_backend_env(None, Some("gpu")),
            ExecutionIntent::AcceleratedOnly
        );
    }

    #[test]
    fn native_request_explicit_target_preserves_product_constraint() {
        assert_eq!(
            request_execution_intent_with_backend_env(
                Some(crate::ExecutionTarget::Cpu),
                Some("cuda")
            ),
            ExecutionIntent::CpuOnly
        );
        assert_eq!(
            request_execution_intent_with_backend_env(
                Some(crate::ExecutionTarget::Accelerated),
                Some("cpu")
            ),
            ExecutionIntent::AcceleratedOnly
        );
        assert_eq!(
            request_execution_intent_with_backend_env(
                Some(crate::ExecutionTarget::Accelerated),
                Some("vulkan")
            ),
            ExecutionIntent::ConstrainedAcceleratedOnly(AcceleratedDeviceConstraint::Provider(
                ExecutionProvider::Vulkan
            ))
        );
    }

    #[test]
    fn native_request_unknown_or_missing_backend_environment_keeps_auto() {
        assert_eq!(
            request_execution_intent_with_backend_env(None, None),
            ExecutionIntent::Auto
        );
        assert_eq!(
            request_execution_intent_with_backend_env(None, Some("")),
            ExecutionIntent::Auto
        );
        assert_eq!(
            request_execution_intent_with_backend_env(None, Some("not-a-backend")),
            ExecutionIntent::Auto
        );
    }

    #[test]
    fn canceled_longform_planning_maps_to_typed_backend_cancel() {
        let error = longform_planning_error_to_backend(LongFormSlicePlanningError::Planning(
            LongFormSliceError::Canceled,
        ));
        assert!(matches!(error, BackendError::TranscriptionCanceled));
    }

    #[test]
    fn auxiliary_execution_policy_preserves_typed_longform_cancel() {
        let services = native_execution_services_for_test();
        let plan = resolve_longform_vad_execution_plan(services.as_ref(), &ExecutionIntent::Auto)
            .expect("Auto VAD plan");
        let error =
            run_auxiliary_stage_with_policy(services.as_ref(), &plan, "longform-vad", |_| {
                Err::<(), BackendError>(BackendError::TranscriptionCanceled)
            })
            .expect_err("canceled long-form VAD must fail the auxiliary stage");
        assert!(matches!(
            required_auxiliary_stage_error(error),
            BackendError::TranscriptionCanceled
        ));
    }

    #[test]
    fn native_boundary_rejects_voice_id_before_any_realtime_model_load() {
        let services = native_execution_services_for_test();
        for source in [
            crate::RequestSource::CliLive,
            crate::RequestSource::ServerRealtime,
        ] {
            let mut request = TranscriptionRequest::new("unused.wav", "unused-model");
            request.voice_id = true;
            request.source = source;
            assert!(matches!(
                run_native_transcription_fallible(request, &services, None),
                Err(BackendError::VoiceIdUnsupportedForRealtime { request_source: label })
                    if label == source.as_log_label()
            ));
        }

        let mut file_request = TranscriptionRequest::new("unused.wav", "unused-model");
        file_request.voice_id = true;
        file_request.source = crate::RequestSource::CliTranscribe;
        assert!(matches!(
            run_native_transcription_fallible(file_request, &services, None),
            Err(BackendError::NativeModelPackPathRequired)
        ));
    }

    #[test]
    fn native_boundary_rejects_out_of_range_speaker_hints_before_model_resolution() {
        let services = native_execution_services_for_test();
        let max = crate::diarize::contract::MAX_DIARIZATION_SPEAKERS;
        for requested in [0, max + 1] {
            let mut request = TranscriptionRequest::new("unused.wav", "unused-model");
            request.diarize_speakers = Some(requested);
            let error = run_native_transcription_fallible(request, &services, None)
                .expect_err("an out-of-range hint must fail closed at the request boundary");
            assert!(matches!(
                &error,
                BackendError::NativeFailClosed { reason }
                    if reason.contains(&format!("between 1 and {max}"))
                        && reason.contains(&format!("got {requested}"))
            ));
            assert_eq!(
                classify_backend_error_for_failure_log(&error),
                FailureCategory::Decode
            );
        }

        for requested in [1, max] {
            let mut request = TranscriptionRequest::new("unused.wav", "unused-model");
            request.voice_id = true;
            request.source = crate::RequestSource::CliTranscribe;
            request.diarize_speakers = Some(requested);
            assert!(matches!(
                run_native_transcription_fallible(request, &services, None),
                Err(BackendError::NativeModelPackPathRequired)
            ));
        }
    }
    use crate::GgmlAsrViewExecutor;
    use crate::arch::DEFAULT_ENCODER_SAFE_CHUNK_SECONDS;
    use std::sync::Mutex;

    fn uncancellable_execution_context_for_test() -> Arc<crate::RequestExecutionContext> {
        Arc::new(crate::RequestExecutionContext::uncancellable(
            "test fixture",
        ))
    }

    fn native_execution_services_for_test() -> Arc<NativeExecutionServices> {
        crate::models::native_execution_services::test_native_execution_services()
    }

    const ASR_EXACT_SMOKE_PACK_ENV: &str = "OPENASR_ASR_SMOKE_PACK";
    const ASR_EXACT_SMOKE_MODEL_ENV: &str = "OPENASR_ASR_SMOKE_MODEL";
    const ASR_EXACT_SMOKE_PROVIDER_ENV: &str = "OPENASR_ASR_SMOKE_PROVIDER";
    const ASR_EXACT_SMOKE_STABLE_ID_ENV: &str = "OPENASR_ASR_SMOKE_STABLE_ID";
    const ASR_EXACT_SMOKE_FIXTURE_ENV: &str = "OPENASR_ASR_SMOKE_FIXTURE";
    const ASR_EXACT_SMOKE_PRIVATE_ENV: &str = "OPENASR_ASR_SMOKE_PRIVATE";
    const ASR_EXACT_SMOKE_AUDIO_PATH_ENV: &str = "OPENASR_ASR_SMOKE_AUDIO_PATH";
    const ASR_EXACT_SMOKE_AUDIO_LABEL_ENV: &str = "OPENASR_ASR_SMOKE_AUDIO_LABEL";
    const ASR_EXACT_SMOKE_AUDIO_SHA256_ENV: &str = "OPENASR_ASR_SMOKE_AUDIO_SHA256";
    const ASR_EXACT_SMOKE_FRESH_MODE_ENV: &str = "OPENASR_ASR_SMOKE_FRESH_MODE";

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum AsrExactSmokeFreshMode {
        CpuOnly,
        ExactAccelerated,
    }

    fn asr_exact_smoke_fresh_mode(raw: &str) -> AsrExactSmokeFreshMode {
        match raw.trim().to_ascii_lowercase().as_str() {
            "cpu" => AsrExactSmokeFreshMode::CpuOnly,
            "accelerated" => AsrExactSmokeFreshMode::ExactAccelerated,
            _ => panic!("OPENASR_ASR_SMOKE_FRESH_MODE must be cpu or accelerated"),
        }
    }

    fn render_safe_memory_bytes(value: crate::ggml_runtime::BackendMemoryBytes) -> String {
        match value {
            crate::ggml_runtime::BackendMemoryBytes::Known(bytes) => format!("known:{bytes}"),
            crate::ggml_runtime::BackendMemoryBytes::Unknown(reason) => {
                format!("unknown:{reason:?}")
            }
        }
    }

    fn emit_exact_smoke_safe_memory_receipts(
        observations: &[crate::models::native_execution_services::ExecutionBackendObservation],
    ) {
        for observation in observations {
            for receipt in &observation.memory_receipts {
                eprintln!(
                    "ASR_EXACT_MEMORY_RECEIPT provider={} placement={:?} backend_kind={:?} lifecycle={:?} domain_kind={:?} heap_index={:?} device_used_bytes={} device_free_bytes={} backend_owned_live_bytes={} backend_owned_cached_bytes={} backend_owned_workspace_bytes={} backend_owned_observed_high_water_bytes={}",
                    observation.actual_provider.as_str(),
                    observation.placement,
                    observation.backend_kind,
                    receipt.lifecycle,
                    receipt.domain_kind,
                    receipt.heap_index,
                    render_safe_memory_bytes(receipt.device_used_bytes),
                    render_safe_memory_bytes(receipt.device_free_bytes),
                    render_safe_memory_bytes(receipt.backend_owned_live_bytes),
                    render_safe_memory_bytes(receipt.backend_owned_cached_bytes),
                    render_safe_memory_bytes(receipt.backend_owned_workspace_bytes),
                    render_safe_memory_bytes(receipt.backend_owned_observed_high_water_bytes),
                );
            }
        }
    }

    struct AsrExactSmokeAudio {
        label: &'static str,
        basename: &'static str,
        oracle_tier: &'static str,
        path: std::path::PathBuf,
        sha256: String,
        samples: Arc<Vec<f32>>,
        allow_matching_truncation: bool,
        longform_mode: LongFormMode,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct AsrExactSmokeTextDriftBudget {
        max_segment_mismatches: usize,
        max_word_edits: usize,
    }

    fn required_asr_exact_smoke_env(name: &str) -> String {
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| {
                panic!("missing required host-local smoke environment variable {name}")
            })
    }

    fn asr_exact_smoke_fixture(raw: &str) -> (&'static str, std::path::PathBuf) {
        let fixture = raw.trim().to_ascii_lowercase();
        let (label, file) = match fixture.as_str() {
            "jfk" => ("jfk", "jfk.wav"),
            "jfk_repeated_15m" => ("jfk_repeated_15m", "jfk.wav"),
            "zh_sample" => ("zh_sample", "zh_sample.wav"),
            "en_zh_mixed" => ("en_zh_mixed", "en_zh_mixed.wav"),
            _ => panic!(
                "OPENASR_ASR_SMOKE_FIXTURE must be one of jfk, jfk_repeated_15m, zh_sample, en_zh_mixed"
            ),
        };
        (
            label,
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("fixtures")
                .join(file),
        )
    }

    fn private_asr_exact_smoke_audio_spec(
        raw: &str,
    ) -> (&'static str, &'static str, &'static str, bool) {
        match raw.trim().to_ascii_lowercase().as_str() {
            "sichuan_dialect_30s" => (
                "sichuan_dialect_30s",
                "sichuan_dialect_30s.wav",
                "stress_parity_only",
                false,
            ),
            "dolphin_sichuan_clip" => (
                "dolphin_sichuan_clip",
                "clip_sichuan.wav",
                "human_ground_truth_2_38s_only",
                false,
            ),
            "private_family_59s_normalized" => (
                "private_family_59s_normalized",
                "private_family_59s_16k_mono_f32.wav",
                "pinned_reference",
                false,
            ),
            "arabic_synthetic" => (
                "arabic_synthetic",
                "arabic_synthetic_16k_mono.wav",
                "routing_crash_parity_only",
                true,
            ),
            _ => panic!("OPENASR_ASR_SMOKE_AUDIO_LABEL is not an approved private-audio label"),
        }
    }

    fn parse_asr_exact_smoke_sha256(raw: &str) -> String {
        let normalized = raw.trim().to_ascii_lowercase();
        assert!(
            normalized.len() == 64 && normalized.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "OPENASR_ASR_SMOKE_AUDIO_SHA256 must be exactly 64 hexadecimal characters"
        );
        normalized
    }

    fn asr_exact_smoke_audio() -> AsrExactSmokeAudio {
        let private_path = std::env::var(ASR_EXACT_SMOKE_AUDIO_PATH_ENV)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());

        let (label, basename, oracle_tier, path, expected_sha256, allow_matching_truncation) =
            if let Some(private_path) = private_path {
                assert_eq!(
                    std::env::var(ASR_EXACT_SMOKE_PRIVATE_ENV).ok().as_deref(),
                    Some("1"),
                    "private ASR smoke audio requires OPENASR_ASR_SMOKE_PRIVATE=1"
                );
                assert!(
                    std::env::var(ASR_EXACT_SMOKE_FIXTURE_ENV)
                        .ok()
                        .is_none_or(|value| value.trim().is_empty()),
                    "private ASR smoke audio cannot be combined with a public fixture"
                );
                let raw_label = required_asr_exact_smoke_env(ASR_EXACT_SMOKE_AUDIO_LABEL_ENV);
                let (label, basename, oracle_tier, allow_matching_truncation) =
                    private_asr_exact_smoke_audio_spec(&raw_label);
                let path = std::path::PathBuf::from(private_path);
                assert!(
                    path.is_absolute(),
                    "private ASR smoke audio path must be absolute"
                );
                assert!(
                    path.file_name()
                        .and_then(|value| value.to_str())
                        .is_some_and(|value| value.eq_ignore_ascii_case(basename)),
                    "private ASR smoke audio basename does not match its approved label"
                );
                assert!(path.is_file(), "private ASR smoke audio is missing");
                let expected_sha256 = parse_asr_exact_smoke_sha256(&required_asr_exact_smoke_env(
                    ASR_EXACT_SMOKE_AUDIO_SHA256_ENV,
                ));
                (
                    label,
                    basename,
                    oracle_tier,
                    path,
                    Some(expected_sha256),
                    allow_matching_truncation,
                )
            } else {
                assert!(
                    std::env::var(ASR_EXACT_SMOKE_PRIVATE_ENV)
                        .ok()
                        .is_none_or(|value| value.trim().is_empty()),
                    "OPENASR_ASR_SMOKE_PRIVATE requires a private audio path"
                );
                let (label, path) = asr_exact_smoke_fixture(&required_asr_exact_smoke_env(
                    ASR_EXACT_SMOKE_FIXTURE_ENV,
                ));
                let basename = match label {
                    "jfk" => "jfk.wav",
                    "jfk_repeated_15m" => "jfk_repeated_15m.wav",
                    "zh_sample" => "zh_sample.wav",
                    "en_zh_mixed" => "en_zh_mixed.wav",
                    _ => unreachable!("public fixture parser returned an unknown label"),
                };
                let oracle_tier = if label == "jfk_repeated_15m" {
                    "public_fixture_longform_quality"
                } else {
                    "public_fixture_parity"
                };
                (label, basename, oracle_tier, path, None, false)
            };

        let bytes = std::fs::read(&path)
            .unwrap_or_else(|_| panic!("ASR smoke audio could not be read after identity checks"));
        let source_sha256 = crate::testing::benchmark_sha256_bytes([bytes.as_slice()]);
        if let Some(expected_sha256) = expected_sha256 {
            assert_eq!(
                source_sha256, expected_sha256,
                "private ASR smoke audio SHA-256 does not match its declared identity"
            );
        }
        let mut samples =
            crate::api::audio_io::parse_wav_16khz_mono_f32(&bytes, "ASR Exact smoke audio")
                .unwrap_or_else(|_| panic!("ASR smoke audio is not a supported 16 kHz mono WAV"));
        let sha256 = if label == "jfk_repeated_15m" {
            samples = repeat_f32_samples_to_exact_len(&samples, 15 * 60 * 16_000);
            crate::testing::benchmark_sha256_f32(&samples)
        } else {
            source_sha256
        };
        AsrExactSmokeAudio {
            label,
            basename,
            oracle_tier,
            path,
            sha256,
            samples: Arc::new(samples),
            allow_matching_truncation,
            longform_mode: asr_exact_smoke_longform_mode(label),
        }
    }

    fn asr_exact_smoke_longform_mode(label: &str) -> LongFormMode {
        match label {
            // Keep the ASR Exact seam scoped to the main model. Stream-VAD has
            // its own CUDA/Vulkan 15-minute endurance gate and publishes a
            // separate placement contract; Fixed still exercises the complete
            // 30-second window/assembler/state lifecycle without conflating
            // auxiliary observations with this ASR-only evidence stream.
            "jfk_repeated_15m" => LongFormMode::Fixed,
            _ => LongFormMode::Off,
        }
    }

    fn asr_exact_smoke_text_drift_budget(
        label: &str,
        provider: ExecutionProvider,
    ) -> AsrExactSmokeTextDriftBudget {
        if label == "jfk_repeated_15m" {
            // Floating-point reductions may change a few greedy choices across 30
            // independently decoded windows. Keep this long-audio quality gate
            // much tighter than a conventional WER threshold while the short
            // fixtures continue to require byte-identical normalized text. The
            // bounds are provider-specific because the stable CUDA reduction
            // envelope is four edits while Vulkan remains at two.
            let max_edits = match provider {
                ExecutionProvider::Cuda => 4,
                ExecutionProvider::Vulkan => 2,
                _ => 0,
            };
            AsrExactSmokeTextDriftBudget {
                max_segment_mismatches: max_edits,
                max_word_edits: max_edits,
            }
        } else {
            AsrExactSmokeTextDriftBudget {
                max_segment_mismatches: 0,
                max_word_edits: 0,
            }
        }
    }

    fn repeat_f32_samples_to_exact_len(source: &[f32], target_len: usize) -> Vec<f32> {
        assert!(!source.is_empty(), "repeat source must not be empty");
        let mut output = Vec::with_capacity(target_len);
        while output.len() < target_len {
            let remaining = target_len - output.len();
            output.extend_from_slice(&source[..source.len().min(remaining)]);
        }
        output
    }

    #[test]
    fn repeated_public_smoke_audio_has_exact_target_length_and_prefix() {
        let repeated = repeat_f32_samples_to_exact_len(&[1.0, 2.0, 3.0], 8);
        assert_eq!(repeated, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]);
        assert_eq!(
            asr_exact_smoke_longform_mode("jfk_repeated_15m"),
            LongFormMode::Fixed
        );
        assert_eq!(asr_exact_smoke_longform_mode("jfk"), LongFormMode::Off);
        assert_eq!(
            asr_exact_smoke_text_drift_budget("jfk_repeated_15m", ExecutionProvider::Cuda),
            AsrExactSmokeTextDriftBudget {
                max_segment_mismatches: 4,
                max_word_edits: 4,
            }
        );
        assert_eq!(
            asr_exact_smoke_text_drift_budget("jfk_repeated_15m", ExecutionProvider::Vulkan),
            AsrExactSmokeTextDriftBudget {
                max_segment_mismatches: 2,
                max_word_edits: 2,
            }
        );
        assert_eq!(
            asr_exact_smoke_text_drift_budget("jfk", ExecutionProvider::Cuda),
            AsrExactSmokeTextDriftBudget {
                max_segment_mismatches: 0,
                max_word_edits: 0,
            }
        );
    }

    fn asr_exact_smoke_provider(raw: &str) -> ExecutionProvider {
        if raw.eq_ignore_ascii_case("cuda") {
            ExecutionProvider::Cuda
        } else if raw.eq_ignore_ascii_case("vulkan") {
            ExecutionProvider::Vulkan
        } else {
            panic!("OPENASR_ASR_SMOKE_PROVIDER must be cuda or vulkan")
        }
    }

    fn asr_exact_smoke_route(
        provider: ExecutionProvider,
        configured_stable_id: Option<String>,
    ) -> crate::ResolvedExecutionRoute {
        let inventory = enumerate_compute_devices_from_ggml(&crate::ggml_available_devices());
        let matching = inventory
            .iter()
            .filter(|device| device.provider == provider)
            .collect::<Vec<_>>();
        let stable_id = match configured_stable_id
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
        {
            Some(stable_id) => stable_id,
            None if matching.len() == 1 => matching[0].stable_id.clone(),
            None => panic!(
                "OPENASR_ASR_SMOKE_STABLE_ID is required unless exactly one requested provider device is visible"
            ),
        };
        matching
            .into_iter()
            .find(|device| device.stable_id == stable_id)
            .unwrap_or_else(|| panic!("configured Exact ASR smoke device is not visible"))
            .to_resolved_route()
    }

    fn asr_exact_smoke_intent(
        provider: ExecutionProvider,
        configured_stable_id: Option<String>,
    ) -> ExecutionIntent {
        let route = asr_exact_smoke_route(provider, configured_stable_id);
        ExecutionIntent::Exact(crate::ExactDeviceSelector::StableId {
            provider: Some(provider),
            stable_id: route.stable_id,
        })
    }

    fn assert_exact_stress_observations(
        observations: &[crate::models::native_execution_services::ExecutionBackendObservation],
        expected_route: &crate::ResolvedExecutionRoute,
    ) {
        assert!(
            !observations.is_empty(),
            "Exact stress request constructed no observed backend"
        );
        assert!(
            observations
                .iter()
                .all(|observation| observation.requested_route == *expected_route),
            "Exact stress request changed its requested route"
        );
        let placements = observations
            .iter()
            .map(|observation| observation.placement)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            placements.len(),
            1,
            "Exact stress request published more than one committed placement"
        );
        assert!(
            observations.iter().any(|observation| {
                observation.backend_kind.is_gpu_class()
                    && observation.actual_provider == expected_route.provider
                    && observation.actual_stable_id == expected_route.stable_id
            }),
            "Exact stress request did not construct the requested accelerated device"
        );
        assert!(observations.iter().all(|observation| {
            if observation.backend_kind.is_gpu_class() {
                observation.actual_provider == expected_route.provider
                    && observation.actual_stable_id == expected_route.stable_id
                    && matches!(
                        observation.placement,
                        ExecutionPlacement::FullDevice | ExecutionPlacement::Hybrid
                    )
                    && (observation.placement != ExecutionPlacement::FullDevice
                        || !observation.use_scheduler)
            } else {
                observation.actual_provider == ExecutionProvider::Cpu
                    && observation.placement == ExecutionPlacement::Hybrid
            }
        }));
    }

    fn asr_exact_stress_decode_is_in_progress(
        stage: TranscriptionStage,
        stage_fraction: Option<f32>,
    ) -> bool {
        stage == TranscriptionStage::Decode
            && stage_fraction.is_some_and(|fraction| fraction > 0.0 && fraction < 1.0)
    }

    fn asr_exact_smoke_request(
        fixture: &std::path::Path,
        model_ref: &str,
        pack: &std::path::Path,
        prepared_samples: Arc<Vec<f32>>,
        longform_mode: LongFormMode,
    ) -> TranscriptionRequest {
        TranscriptionRequest::new(fixture, model_ref)
            .with_model_pack_path(Some(pack.to_path_buf()))
            .with_prepared_samples(Some(prepared_samples))
            .with_punctuation(false)
            .with_word_timestamps(false)
            .with_word_timestamps_refine(false)
            .with_voice_id(false)
            .with_longform(Some(crate::LongFormOptions {
                mode: longform_mode,
                ..crate::LongFormOptions::default()
            }))
    }

    fn normalized_transcription_hash(transcription: &Transcription) -> String {
        crate::testing::benchmark_sha256_bytes([
            crate::normalize_text(&transcription.text).as_bytes()
        ])
    }

    fn normalized_segment_hash(segment: &Segment) -> String {
        crate::testing::benchmark_sha256_bytes([crate::normalize_text(&segment.text).as_bytes()])
    }

    fn assert_exact_smoke_timestamps_are_valid(transcription: &Transcription, label: &str) {
        let mut previous_segment_start = 0.0_f32;
        let mut previous_segment_end = 0.0_f32;
        for (segment_index, segment) in transcription.segments.iter().enumerate() {
            assert!(
                segment.start.is_finite()
                    && segment.end.is_finite()
                    && segment.start >= 0.0
                    && segment.end >= segment.start,
                "{label} segment {segment_index} has invalid timestamps"
            );
            assert!(
                segment.start >= previous_segment_start && segment.end >= previous_segment_end,
                "{label} segment {segment_index} timestamps are not monotonic"
            );
            previous_segment_start = segment.start;
            previous_segment_end = segment.end;

            let mut previous_word_start = segment.start;
            let mut previous_word_end = segment.start;
            for (word_index, word) in segment.words.iter().enumerate() {
                assert!(
                    word.start.is_finite()
                        && word.end.is_finite()
                        && word.start >= 0.0
                        && word.end >= word.start,
                    "{label} segment {segment_index} word {word_index} has invalid timestamps"
                );
                assert!(
                    word.start >= previous_word_start && word.end >= previous_word_end,
                    "{label} segment {segment_index} word {word_index} timestamps are not monotonic"
                );
                previous_word_start = word.start;
                previous_word_end = word.end;
            }
        }
    }

    fn assert_exact_smoke_structure_parity(
        cpu: &Transcription,
        accelerated: &Transcription,
        allow_matching_truncation: bool,
        timestamp_tolerance_seconds: f32,
    ) -> (usize, Option<usize>) {
        assert_eq!(
            cpu.segments.len(),
            accelerated.segments.len(),
            "CPU/accelerated segment count mismatch"
        );
        let mut segment_text_mismatches = 0usize;
        let mut first_segment_text_mismatch = None;
        for (segment_index, (cpu_segment, accelerated_segment)) in
            cpu.segments.iter().zip(&accelerated.segments).enumerate()
        {
            if normalized_segment_hash(cpu_segment) != normalized_segment_hash(accelerated_segment)
            {
                segment_text_mismatches += 1;
                first_segment_text_mismatch.get_or_insert(segment_index);
            }
            let start_delta = (cpu_segment.start - accelerated_segment.start).abs();
            let end_delta = (cpu_segment.end - accelerated_segment.end).abs();
            assert!(
                start_delta <= timestamp_tolerance_seconds
                    && end_delta <= timestamp_tolerance_seconds,
                "CPU/accelerated segment timestamp mismatch at segment {segment_index}: \
                 cpu_start={:.6} accelerated_start={:.6} start_delta={start_delta:.6} \
                 cpu_end={:.6} accelerated_end={:.6} end_delta={end_delta:.6} tolerance={timestamp_tolerance_seconds:.6}",
                cpu_segment.start,
                accelerated_segment.start,
                cpu_segment.end,
                accelerated_segment.end,
            );
        }
        if allow_matching_truncation {
            assert_eq!(
                cpu.truncated_decodes.len(),
                accelerated.truncated_decodes.len(),
                "CPU/accelerated truncation count mismatch"
            );
            for (index, (cpu_truncation, accelerated_truncation)) in cpu
                .truncated_decodes
                .iter()
                .zip(&accelerated.truncated_decodes)
                .enumerate()
            {
                assert_eq!(
                    cpu_truncation.slice_index, accelerated_truncation.slice_index,
                    "CPU/accelerated truncation slice mismatch at index {index}"
                );
                assert_eq!(
                    cpu_truncation.truncation.reason, accelerated_truncation.truncation.reason,
                    "CPU/accelerated truncation reason mismatch at index {index}"
                );
                match (
                    cpu_truncation.truncation.transcript_covers_up_to_seconds,
                    accelerated_truncation
                        .truncation
                        .transcript_covers_up_to_seconds,
                ) {
                    (None, None) => {}
                    (Some(cpu_coverage), Some(accelerated_coverage)) => assert!(
                        cpu_coverage.is_finite()
                            && accelerated_coverage.is_finite()
                            && (cpu_coverage - accelerated_coverage).abs()
                                <= timestamp_tolerance_seconds,
                        "CPU/accelerated truncation coverage mismatch at index {index}"
                    ),
                    _ => panic!(
                        "CPU/accelerated truncation coverage presence mismatch at index {index}"
                    ),
                }
            }
        } else {
            assert!(
                cpu.truncated_decodes.is_empty() && accelerated.truncated_decodes.is_empty(),
                "ASR Exact smoke must not produce truncated decodes for this audio tier"
            );
        }
        (segment_text_mismatches, first_segment_text_mismatch)
    }

    fn exact_smoke_timestamp_tolerance_seconds(model_architecture: &str) -> f32 {
        if model_architecture == crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID {
            // MOSS emits centisecond timestamps as autoregressive text tokens.
            // Keep text and segment structure exact while allowing at most ten
            // centiseconds of backend-dependent anchor drift.
            0.100
        } else {
            0.050
        }
    }

    #[test]
    fn exact_smoke_timestamp_tolerance_is_narrowly_moss_specific() {
        assert_eq!(
            exact_smoke_timestamp_tolerance_seconds(crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID),
            0.100
        );
        for architecture in [
            crate::arch::FUNASR_NANO_GGML_ARCHITECTURE_ID,
            crate::arch::GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            "future-asr-family",
        ] {
            assert_eq!(exact_smoke_timestamp_tolerance_seconds(architecture), 0.050);
        }
    }

    /// Opt-in true-pack parity seam for every native ASR family. This test
    /// accepts only a verified local pack plus either a committed public
    /// fixture enum or a strictly allowlisted private WAV whose basename and
    /// exact file SHA-256 are supplied separately. It never downloads or
    /// serializes source text or absolute paths into test output.
    /// Long-form and auxiliary-model placement belong to their own matrix so
    /// this ASR-only seam never conflates their observations with a main
    /// model's FullDevice contract.
    #[test]
    #[ignore = "host-local true-pack CPU/CUDA-or-Vulkan Exact smoke; requires explicit environment"]
    fn asr_exact_pack_cpu_accelerated_smoke_and_parity() {
        let pack_path =
            std::path::PathBuf::from(required_asr_exact_smoke_env(ASR_EXACT_SMOKE_PACK_ENV));
        let model_ref = required_asr_exact_smoke_env(ASR_EXACT_SMOKE_MODEL_ENV);
        parse_model_ref(&model_ref).unwrap_or_else(|_| {
            panic!("OPENASR_ASR_SMOKE_MODEL must be a valid catalog model reference")
        });
        let provider =
            asr_exact_smoke_provider(&required_asr_exact_smoke_env(ASR_EXACT_SMOKE_PROVIDER_ENV));
        let configured_stable_id = std::env::var(ASR_EXACT_SMOKE_STABLE_ID_ENV).ok();
        let audio = asr_exact_smoke_audio();

        let verified_pack = Arc::new(
            PackVerifier
                .verify_candidate(PackCandidate::new(&pack_path))
                .unwrap_or_else(|_| panic!("OPENASR_ASR_SMOKE_PACK did not verify as an ASR pack")),
        );
        assert!(
            matches!(verified_pack.route(), PackRoute::Asr { .. }),
            "OPENASR_ASR_SMOKE_PACK must be an ASR pack"
        );
        let audio_seconds = audio.samples.len() as f64 / 16_000.0;

        let services = native_execution_services_for_test();
        let exact_intent = asr_exact_smoke_intent(provider, configured_stable_id);
        let selection_metadata = selection_metadata_from_gguf(&verified_pack.preflight().metadata);
        let selected_family = validate_runtime_source_and_select_adapter(
            &model_ref,
            &verified_pack,
            &selection_metadata,
        )
        .unwrap_or_else(|_| panic!("catalog model reference does not match the verified ASR pack"));
        let exact_plan = resolve_native_execution_plan(
            services.as_ref(),
            &selected_family,
            exact_intent.clone(),
        )
        .unwrap_or_else(|_| panic!("Exact accelerated plan could not be resolved"));
        let exact_candidates = exact_plan.candidates();
        assert!(
            !exact_candidates.is_empty(),
            "Exact execution must contain an accelerated candidate"
        );
        let expected_route = exact_candidates[0].device.route.clone();
        let declared_placements = exact_candidates
            .iter()
            .map(|candidate| candidate.placement)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(expected_route.provider, provider);
        assert!(
            exact_candidates.iter().all(|candidate| {
                candidate.device.route == expected_route
                    && matches!(
                        candidate.placement,
                        ExecutionPlacement::FullDevice | ExecutionPlacement::Hybrid
                    )
            }),
            "Exact execution may try multiple placements, but must never change device or append CPU fallback"
        );

        let cpu_started = Instant::now();
        let cpu = run_native_transcription_with_verified_pack(
            asr_exact_smoke_request(
                &audio.path,
                &model_ref,
                &pack_path,
                Arc::clone(&audio.samples),
                audio.longform_mode,
            ),
            Arc::clone(&services),
            Some(ExecutionIntent::CpuOnly),
            Arc::clone(&verified_pack),
        )
        .unwrap_or_else(|_| panic!("CPU baseline failed"));
        let cpu_seconds = cpu_started.elapsed().as_secs_f64();

        let observations =
            crate::models::native_execution_services::ExecutionObservationSink::new();
        let accelerated_started = Instant::now();
        let accelerated = {
            let _observation_guard =
                crate::models::native_execution_services::install_execution_observation_sink(
                    observations.clone(),
                );
            run_native_transcription_with_verified_pack(
                asr_exact_smoke_request(
                    &audio.path,
                    &model_ref,
                    &pack_path,
                    Arc::clone(&audio.samples),
                    audio.longform_mode,
                ),
                Arc::clone(&services),
                Some(exact_intent),
                Arc::clone(&verified_pack),
            )
            .unwrap_or_else(|_| panic!("Exact accelerated transcription failed"))
        };
        let accelerated_seconds = accelerated_started.elapsed().as_secs_f64();

        assert!(
            !crate::normalize_text(&cpu.text).is_empty(),
            "CPU baseline produced an empty normalized transcript"
        );
        assert!(
            !crate::normalize_text(&accelerated.text).is_empty(),
            "Exact accelerated run produced an empty normalized transcript"
        );
        assert_exact_smoke_timestamps_are_valid(&cpu, "CPU baseline");
        assert_exact_smoke_timestamps_are_valid(&accelerated, "Exact accelerated");
        let (segment_text_mismatches, first_segment_text_mismatch) =
            assert_exact_smoke_structure_parity(
                &cpu,
                &accelerated,
                audio.allow_matching_truncation,
                exact_smoke_timestamp_tolerance_seconds(selected_family.model_architecture),
            );
        let cpu_hash = normalized_transcription_hash(&cpu);
        let accelerated_hash = normalized_transcription_hash(&accelerated);
        let text_drift_budget = asr_exact_smoke_text_drift_budget(audio.label, provider);
        let normalized_word_edits = crate::metrics::wer_counts(&accelerated.text, &cpu.text).errors;
        assert!(
            segment_text_mismatches <= text_drift_budget.max_segment_mismatches
                && normalized_word_edits <= text_drift_budget.max_word_edits,
            "CPU/accelerated normalized text drift exceeded the fixture budget: \
             segment_mismatches={segment_text_mismatches} segment_budget={} \
             word_edits={normalized_word_edits} word_budget={} \
             first_segment_mismatch={first_segment_text_mismatch:?}",
            text_drift_budget.max_segment_mismatches,
            text_drift_budget.max_word_edits,
        );
        if text_drift_budget.max_word_edits == 0 {
            assert_eq!(
                cpu_hash, accelerated_hash,
                "CPU/accelerated normalized text hash mismatch"
            );
        }
        let observations = observations.observations();
        assert!(
            !observations.is_empty(),
            "Exact accelerated execution must construct an observed backend"
        );
        assert!(observations.iter().all(|observation| {
            observation.requested_route == expected_route
                && declared_placements.contains(&observation.placement)
        }));
        let observed_placements = observations
            .iter()
            .map(|observation| observation.placement)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            observed_placements.len(),
            1,
            "only the committed Exact candidate placement may publish observations"
        );
        let selected_placement = *observed_placements
            .iter()
            .next()
            .expect("non-empty observations yield one selected placement");
        let gpu_observations = observations
            .iter()
            .filter(|observation| observation.backend_kind.is_gpu_class())
            .collect::<Vec<_>>();
        assert!(
            !gpu_observations.is_empty(),
            "Exact FullDevice/Hybrid execution must construct at least one observed GPU backend"
        );
        assert!(gpu_observations.iter().all(|observation| {
            observation.actual_provider == expected_route.provider
                && observation.actual_stable_id == expected_route.stable_id
        }));
        if selected_placement == ExecutionPlacement::FullDevice {
            assert!(
                observations
                    .iter()
                    .all(|observation| observation.backend_kind.is_gpu_class()
                        && !observation.use_scheduler),
                "Exact FullDevice execution must construct only direct GPU runners"
            );
        }
        let peak_rss_bytes = crate::metrics::peak_rss_bytes().unwrap_or(0);
        eprintln!(
            "ASR_EXACT_SMOKE model={model_ref} audio_label={} audio_basename={} oracle_tier={} pack_content_id={} audio_sha256={} requested_provider={} requested_stable_id={} placement={:?} cpu_seconds={cpu_seconds:.6} cpu_rtf={:.6} accelerated_seconds={accelerated_seconds:.6} accelerated_rtf={:.6} peak_rss_bytes={peak_rss_bytes} cpu_segments={} accelerated_segments={} segment_text_mismatches={segment_text_mismatches} normalized_word_edits={normalized_word_edits} truncated_decodes={} normalized_text_sha256={accelerated_hash} observed_gpu_backends={}",
            audio.label,
            audio.basename,
            audio.oracle_tier,
            verified_pack.content_id(),
            audio.sha256,
            expected_route.provider.as_str(),
            expected_route.stable_id,
            selected_placement,
            cpu_seconds / audio_seconds.max(f64::MIN_POSITIVE),
            accelerated_seconds / audio_seconds.max(f64::MIN_POSITIVE),
            cpu.segments.len(),
            accelerated.segments.len(),
            accelerated.truncated_decodes.len(),
            gpu_observations.len(),
        );
    }

    /// Single-route memory receipt seam. Run this ignored test by exact name
    /// in a newly spawned test process for each mode; unlike the parity seam
    /// above it never initializes both CPU and an accelerator in one process.
    /// Lines prefixed with `ASR_EXACT_MEMORY_RECEIPT` contain only typed
    /// backend/device memory receipts and route-class facts. A private runner
    /// must parse those lines in memory and persist only that allowlisted
    /// projection; ordinary test/runtime diagnostics are not evidence-safe.
    #[test]
    #[ignore = "host-local fresh-process CPU-only or Exact accelerated ASR smoke"]
    fn asr_exact_pack_fresh_single_route_memory_receipt() {
        let mode = asr_exact_smoke_fresh_mode(&required_asr_exact_smoke_env(
            ASR_EXACT_SMOKE_FRESH_MODE_ENV,
        ));
        let pack_path =
            std::path::PathBuf::from(required_asr_exact_smoke_env(ASR_EXACT_SMOKE_PACK_ENV));
        let model_ref = required_asr_exact_smoke_env(ASR_EXACT_SMOKE_MODEL_ENV);
        parse_model_ref(&model_ref).unwrap_or_else(|_| {
            panic!("OPENASR_ASR_SMOKE_MODEL must be a valid catalog model reference")
        });
        let audio = asr_exact_smoke_audio();
        let verified_pack = Arc::new(
            PackVerifier
                .verify_candidate(PackCandidate::new(&pack_path))
                .unwrap_or_else(|_| panic!("OPENASR_ASR_SMOKE_PACK did not verify as an ASR pack")),
        );
        assert!(matches!(verified_pack.route(), PackRoute::Asr { .. }));
        let intent = match mode {
            AsrExactSmokeFreshMode::CpuOnly => ExecutionIntent::CpuOnly,
            AsrExactSmokeFreshMode::ExactAccelerated => {
                let provider = asr_exact_smoke_provider(&required_asr_exact_smoke_env(
                    ASR_EXACT_SMOKE_PROVIDER_ENV,
                ));
                asr_exact_smoke_intent(provider, std::env::var(ASR_EXACT_SMOKE_STABLE_ID_ENV).ok())
            }
        };
        let services = native_execution_services_for_test();
        let observations =
            crate::models::native_execution_services::ExecutionObservationSink::new();
        let transcription = {
            let _observation_guard =
                crate::models::native_execution_services::install_execution_observation_sink(
                    observations.clone(),
                );
            run_native_transcription_with_verified_pack(
                asr_exact_smoke_request(
                    &audio.path,
                    &model_ref,
                    &pack_path,
                    Arc::clone(&audio.samples),
                    audio.longform_mode,
                ),
                services,
                Some(intent),
                verified_pack,
            )
            .unwrap_or_else(|_| panic!("fresh single-route ASR transcription failed"))
        };
        assert!(
            !crate::normalize_text(&transcription.text).is_empty(),
            "fresh single-route ASR smoke produced an empty transcript"
        );
        let observations = observations.observations();
        assert!(
            !observations.is_empty(),
            "fresh single-route ASR smoke did not construct an observed backend"
        );
        assert!(
            observations
                .iter()
                .any(|observation| !observation.memory_receipts.is_empty()),
            "fresh single-route ASR smoke produced no typed memory receipt"
        );
        match mode {
            AsrExactSmokeFreshMode::CpuOnly => assert!(
                observations
                    .iter()
                    .all(|observation| observation.actual_provider == ExecutionProvider::Cpu)
            ),
            AsrExactSmokeFreshMode::ExactAccelerated => {
                let first = observations
                    .first()
                    .expect("accelerated memory receipt requires an observation");
                assert_eq!(
                    first.placement,
                    ExecutionPlacement::FullDevice,
                    "accelerated memory receipt requires committed FullDevice placement"
                );
                assert!(matches!(
                    first.actual_provider,
                    ExecutionProvider::Cuda | ExecutionProvider::Vulkan
                ));
                assert!(
                    observations.iter().all(|observation| {
                        observation.placement == ExecutionPlacement::FullDevice
                            && observation.backend_kind.is_gpu_class()
                            && !observation.use_scheduler
                            && observation.actual_provider == first.actual_provider
                            && observation.requested_route == first.requested_route
                    }),
                    "accelerated memory receipt must contain only one exact direct-GPU route"
                );
            }
        }
        emit_exact_smoke_safe_memory_receipts(&observations);
    }

    /// Fresh-process concurrency gate for one exact provider. Both requests
    /// share the production execution services and verified pack, enter the
    /// dispatcher together, and must retain independent outputs plus exact
    /// aggregate route telemetry. A warmed actor may legitimately emit no new
    /// backend-construction observation for one request.
    #[test]
    #[ignore = "host-local fresh-process two-request Exact accelerated ASR gate"]
    fn asr_exact_pack_two_concurrent_requests_match() {
        let pack_path =
            std::path::PathBuf::from(required_asr_exact_smoke_env(ASR_EXACT_SMOKE_PACK_ENV));
        let model_ref = required_asr_exact_smoke_env(ASR_EXACT_SMOKE_MODEL_ENV);
        parse_model_ref(&model_ref)
            .unwrap_or_else(|_| panic!("OPENASR_ASR_SMOKE_MODEL must be a valid model reference"));
        let provider =
            asr_exact_smoke_provider(&required_asr_exact_smoke_env(ASR_EXACT_SMOKE_PROVIDER_ENV));
        let expected_route =
            asr_exact_smoke_route(provider, std::env::var(ASR_EXACT_SMOKE_STABLE_ID_ENV).ok());
        let intent = ExecutionIntent::Exact(crate::ExactDeviceSelector::StableId {
            provider: Some(provider),
            stable_id: expected_route.stable_id.clone(),
        });
        let audio = asr_exact_smoke_audio();
        let verified_pack = Arc::new(
            PackVerifier
                .verify_candidate(PackCandidate::new(&pack_path))
                .unwrap_or_else(|_| panic!("OPENASR_ASR_SMOKE_PACK did not verify as an ASR pack")),
        );
        let services = native_execution_services_for_test();
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let (outcome_tx, outcome_rx) = std::sync::mpsc::sync_channel(2);
        let mut workers = Vec::with_capacity(2);
        for _ in 0..2 {
            let barrier = Arc::clone(&barrier);
            let outcome_tx = outcome_tx.clone();
            let services = Arc::clone(&services);
            let verified_pack = Arc::clone(&verified_pack);
            let intent = intent.clone();
            let pack_path = pack_path.clone();
            let model_ref = model_ref.clone();
            let audio_path = audio.path.clone();
            let samples = Arc::clone(&audio.samples);
            let longform_mode = audio.longform_mode;
            workers.push(std::thread::spawn(move || {
                let observations =
                    crate::models::native_execution_services::ExecutionObservationSink::new();
                barrier.wait();
                let transcription = {
                    let _observation_guard = crate::models::native_execution_services::
                        install_execution_observation_sink(observations.clone());
                    run_native_transcription_with_verified_pack(
                        asr_exact_smoke_request(
                            &audio_path,
                            &model_ref,
                            &pack_path,
                            samples,
                            longform_mode,
                        ),
                        services,
                        Some(intent),
                        verified_pack,
                    )
                };
                let _ = outcome_tx.send((transcription, observations.observations()));
            }));
        }
        drop(outcome_tx);
        barrier.wait();
        let mut outcomes = (0..2)
            .map(|_| {
                let (result, observations) = outcome_rx
                    .recv_timeout(std::time::Duration::from_secs(60))
                    .expect("concurrent Exact ASR request must not hang");
                result
                    .map(|transcription| (transcription, observations))
                    .unwrap_or_else(|_| panic!("concurrent Exact ASR request failed"))
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker
                .join()
                .unwrap_or_else(|_| panic!("concurrent Exact ASR worker panicked"));
        }
        let (first, first_observations) = outcomes.remove(0);
        let (second, second_observations) = outcomes.remove(0);
        assert_eq!(
            normalized_transcription_hash(&first),
            normalized_transcription_hash(&second),
            "concurrent Exact requests produced different normalized text"
        );
        assert_exact_smoke_structure_parity(&first, &second, false, 0.05);
        let combined_observations = first_observations
            .iter()
            .chain(&second_observations)
            .cloned()
            .collect::<Vec<_>>();
        assert_exact_stress_observations(&combined_observations, &expected_route);
        eprintln!(
            "ASR_EXACT_STRESS mode=concurrent model={model_ref} provider={} status=pass requests=2 normalized_text_sha256={}",
            provider.as_str(),
            normalized_transcription_hash(&first),
        );
    }

    /// Mid-decode cancellation and same-services recovery gate. A canceled
    /// request must return the typed terminal error promptly; the next exact
    /// request must rebuild/reuse only healthy state and complete normally.
    #[test]
    #[ignore = "host-local fresh-process mid-decode cancel/recovery Exact accelerated ASR gate"]
    fn asr_exact_pack_mid_decode_cancel_then_recovers() {
        let pack_path =
            std::path::PathBuf::from(required_asr_exact_smoke_env(ASR_EXACT_SMOKE_PACK_ENV));
        let model_ref = required_asr_exact_smoke_env(ASR_EXACT_SMOKE_MODEL_ENV);
        parse_model_ref(&model_ref)
            .unwrap_or_else(|_| panic!("OPENASR_ASR_SMOKE_MODEL must be a valid model reference"));
        let provider =
            asr_exact_smoke_provider(&required_asr_exact_smoke_env(ASR_EXACT_SMOKE_PROVIDER_ENV));
        let expected_route =
            asr_exact_smoke_route(provider, std::env::var(ASR_EXACT_SMOKE_STABLE_ID_ENV).ok());
        let intent = ExecutionIntent::Exact(crate::ExactDeviceSelector::StableId {
            provider: Some(provider),
            stable_id: expected_route.stable_id.clone(),
        });
        let audio = asr_exact_smoke_audio();
        let verified_pack = Arc::new(
            PackVerifier
                .verify_candidate(PackCandidate::new(&pack_path))
                .unwrap_or_else(|_| panic!("OPENASR_ASR_SMOKE_PACK did not verify as an ASR pack")),
        );
        let services = native_execution_services_for_test();
        let control = Arc::new(crate::api::backend::TranscriptionControl::new());
        let request_id = "exact-stress-cancel";
        let execution_context = Arc::new(crate::RequestExecutionContext::new(
            Some(request_id.to_string()),
            Arc::clone(&control),
        ));
        let (result_tx, result_rx) = std::sync::mpsc::sync_channel(1);
        let worker = {
            let services = Arc::clone(&services);
            let verified_pack = Arc::clone(&verified_pack);
            let intent = intent.clone();
            let request = asr_exact_smoke_request(
                &audio.path,
                &model_ref,
                &pack_path,
                Arc::clone(&audio.samples),
                audio.longform_mode,
            )
            .with_execution_context(execution_context);
            std::thread::spawn(move || {
                let observations =
                    crate::models::native_execution_services::ExecutionObservationSink::new();
                let result = {
                    let _observation_guard = crate::models::native_execution_services::
                        install_execution_observation_sink(observations.clone());
                    run_native_transcription_with_verified_pack(
                        request,
                        services,
                        Some(intent),
                        verified_pack,
                    )
                };
                let _ = result_tx.send((result, observations.observations()));
            })
        };
        let decode_ready_deadline = Instant::now() + std::time::Duration::from_secs(60);
        loop {
            if native_transcription_progress_for_id(request_id).is_some_and(|progress| {
                asr_exact_stress_decode_is_in_progress(progress.stage, progress.stage_fraction)
            }) {
                break;
            }
            match result_rx.try_recv() {
                Ok(_) => {
                    worker
                        .join()
                        .unwrap_or_else(|_| panic!("mid-decode Exact worker panicked"));
                    panic!("Exact request completed before publishing real decode work");
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    worker
                        .join()
                        .unwrap_or_else(|_| panic!("mid-decode Exact worker panicked"));
                    panic!("mid-decode Exact worker disconnected before publishing progress");
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            assert!(
                Instant::now() < decode_ready_deadline,
                "Exact request did not publish real decode work before cancellation timeout"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        control.request_cancel();
        let (canceled, canceled_observations) = result_rx
            .recv_timeout(std::time::Duration::from_secs(60))
            .expect("mid-decode Exact cancellation must not hang");
        worker
            .join()
            .unwrap_or_else(|_| panic!("mid-decode Exact worker panicked"));
        let canceled = canceled.expect_err("mid-decode Exact cancellation must fail closed");
        assert!(
            matches!(canceled, BackendError::TranscriptionCanceled),
            "mid-decode Exact cancellation must remain typed"
        );
        // A canceled execution candidate intentionally does not commit its
        // observation journal. If an earlier stage did commit observations,
        // they must still prove the exact route; an empty journal is the
        // expected fail-closed outcome for a cancellation inside the main
        // candidate.
        if !canceled_observations.is_empty() {
            assert_exact_stress_observations(&canceled_observations, &expected_route);
        }

        let recovered_observations =
            crate::models::native_execution_services::ExecutionObservationSink::new();
        let recovered = {
            let _observation_guard =
                crate::models::native_execution_services::install_execution_observation_sink(
                    recovered_observations.clone(),
                );
            run_native_transcription_with_verified_pack(
                asr_exact_smoke_request(
                    &audio.path,
                    &model_ref,
                    &pack_path,
                    Arc::clone(&audio.samples),
                    audio.longform_mode,
                ),
                services,
                Some(intent),
                verified_pack,
            )
            .unwrap_or_else(|_| panic!("Exact ASR request did not recover after cancellation"))
        };
        assert!(
            !crate::normalize_text(&recovered.text).is_empty(),
            "recovered Exact ASR request produced empty text"
        );
        assert_exact_stress_observations(&recovered_observations.observations(), &expected_route);
        eprintln!(
            "ASR_EXACT_STRESS mode=cancel-recover model={model_ref} provider={} status=pass normalized_text_sha256={}",
            provider.as_str(),
            normalized_transcription_hash(&recovered),
        );
    }

    #[test]
    fn exact_stress_decode_readiness_requires_strictly_in_progress_fraction() {
        assert!(!asr_exact_stress_decode_is_in_progress(
            TranscriptionStage::Decode,
            None,
        ));
        assert!(!asr_exact_stress_decode_is_in_progress(
            TranscriptionStage::Decode,
            Some(0.0),
        ));
        assert!(asr_exact_stress_decode_is_in_progress(
            TranscriptionStage::Decode,
            Some(0.5),
        ));
        assert!(!asr_exact_stress_decode_is_in_progress(
            TranscriptionStage::Decode,
            Some(1.0),
        ));
        assert!(!asr_exact_stress_decode_is_in_progress(
            TranscriptionStage::Decode,
            Some(f32::NAN),
        ));
        assert!(!asr_exact_stress_decode_is_in_progress(
            TranscriptionStage::Project,
            Some(0.5),
        ));
    }

    #[test]
    fn private_asr_exact_smoke_audio_specs_are_safe_and_unique() {
        let labels = [
            "sichuan_dialect_30s",
            "dolphin_sichuan_clip",
            "private_family_59s_normalized",
            "arabic_synthetic",
        ];
        let mut basenames = std::collections::HashSet::new();
        for label in labels {
            let (canonical_label, basename, oracle_tier, _) =
                private_asr_exact_smoke_audio_spec(label);
            assert_eq!(canonical_label, label);
            assert!(!basename.is_empty());
            assert!(!basename.contains(['/', '\\']));
            assert!(!oracle_tier.is_empty());
            assert!(basenames.insert(basename));
        }
    }

    #[test]
    fn private_asr_exact_smoke_sha256_parser_is_canonical() {
        assert_eq!(
            parse_asr_exact_smoke_sha256(
                "ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
            ),
            "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        );
    }

    /// The full user-intent x family-capability matrix, pinned because every
    /// downstream decision (which source runs, whether an embedder is
    /// required, whether the decoder is asked for speaker structure, whether
    /// word anchors are forced on) reads this one value. The load-bearing rows
    /// are the two `Off` ones: Voice ID off means no speaker structure even for
    /// a family whose decode always writes some.
    #[test]
    fn speaker_plan_picks_exactly_one_source_per_request() {
        use SpeakerSegmentationSource::{External, InDecoder};

        assert_eq!(SpeakerPlan::resolve(false, InDecoder), SpeakerPlan::Off);
        assert_eq!(SpeakerPlan::resolve(false, External), SpeakerPlan::Off);
        assert_eq!(
            SpeakerPlan::resolve(true, InDecoder),
            SpeakerPlan::InDecoder
        );
        assert_eq!(SpeakerPlan::resolve(true, External), SpeakerPlan::External);
    }

    #[test]
    fn native_asr_voice_id_and_forced_align_views_share_one_pcm_backing() {
        let prepared = PcmBuffer::from_vec((0..64).map(|sample| sample as f32).collect());
        let identity = prepared.backing_identity();

        assert!(voice_id_audio_view(&prepared, SpeakerPlan::Off).is_none());
        for plan in [SpeakerPlan::InDecoder, SpeakerPlan::External] {
            let voice_id = voice_id_audio_view(&prepared, plan)
                .expect("enabled Voice ID must borrow normalized PCM");
            assert_eq!(voice_id.backing_identity(), identity);
            assert_eq!(voice_id.as_ptr(), prepared.as_ptr());
        }

        assert!(forced_aligner_audio_view(&prepared, false).is_none());
        let align = forced_aligner_audio_view(&prepared, true)
            .expect("enabled forced aligner must borrow normalized PCM");
        let dispatch = GgmlAsrPreparedAudioView::mono_16khz_shared(prepared.slice(8..24));
        let align_request = GgmlAsrPreparedAudioView::mono_16khz_shared(align);
        assert_eq!(dispatch.samples_f32.backing_identity(), identity);
        assert_eq!(align_request.samples_f32.backing_identity(), identity);
        assert_eq!(
            dispatch.samples_f32.as_ptr(),
            prepared.as_ptr().wrapping_add(8)
        );
        assert_eq!(align_request.samples_f32.as_ptr(), prepared.as_ptr());
    }

    #[test]
    fn resolving_an_already_shared_pcm_buffer_never_copies_the_recording() {
        let prepared = Arc::new(vec![0.25; 16_000]);
        let retained_by_preparer = Arc::clone(&prepared);
        let identity = Arc::as_ptr(&prepared) as usize;
        let samples_ptr = prepared.as_ptr();

        let resolved =
            resolve_prepared_audio_samples(Path::new("must-not-be-read.wav"), Some(prepared))
                .expect("in-memory PCM bypasses the path");

        assert_eq!(resolved.backing_identity(), identity);
        assert_eq!(
            resolved.backing_identity(),
            Arc::as_ptr(&retained_by_preparer) as usize
        );
        assert_eq!(resolved.as_ptr(), samples_ptr);
    }

    /// End of the chain for a moss-shaped decode: the family descriptor picks
    /// the source, the plan turns the Voice ID switch into a decision, and the
    /// family's own normalizer honors it. With Voice ID off the transcript
    /// carries no trace of the markers the fixed decode prompt makes the model
    /// write; with it on, the same decode yields recording-local turns at the
    /// shared boundary. Uses the real reference-decode shape pinned by this
    /// family's golden fixtures, so a change to either the descriptor or the
    /// normalizer breaks it.
    #[test]
    fn a_moss_shaped_decode_honors_the_voice_id_switch_end_to_end() {
        use crate::models::moss_transcribe_diarize::speaker_segments::{
            MossTdDecodeExtent, normalize_moss_td_decode,
        };

        let descriptor = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID)
            .expect("moss-transcribe-diarize is a builtin architecture");
        assert_eq!(
            descriptor.execution_contract.speaker_segmentation,
            SpeakerSegmentationSource::InDecoder
        );
        let decoded = concat!(
            "[0.28][S01] And so, my fellow Americans,[2.32][3.22][S02] ask not what your ",
            "country can do for you,[7.71][8.12][S01] ask what you can do for your country.[10.59]",
        );

        let off = SpeakerPlan::resolve(false, descriptor.execution_contract.speaker_segmentation);
        assert_eq!(off, SpeakerPlan::Off);
        let normalized = normalize_moss_td_decode(
            decoded,
            MossTdDecodeExtent::complete(10.59),
            off == SpeakerPlan::InDecoder,
        );
        assert!(
            !normalized.text.contains('['),
            "Voice ID off must not leak markup: {:?}",
            normalized.text
        );
        assert_eq!(
            normalized.text,
            "And so, my fellow Americans, ask not what your country can do for you, \
             ask what you can do for your country."
        );
        for segment in &normalized.segments {
            assert!(!segment.text.contains("[S"));
            assert!(segment.speaker.is_none());
            assert!(segment.speaker_label.is_none());
        }

        let on = SpeakerPlan::resolve(true, descriptor.execution_contract.speaker_segmentation);
        assert_eq!(on, SpeakerPlan::InDecoder);
        let normalized = normalize_moss_td_decode(
            decoded,
            MossTdDecodeExtent::complete(10.59),
            on == SpeakerPlan::InDecoder,
        );
        assert!(!normalized.text.contains('['));
        let labels: Vec<_> = normalized
            .segments
            .iter()
            .map(|segment| segment.speaker_label.as_deref())
            .collect();
        assert_eq!(
            labels,
            vec![Some("SPEAKER_01"), Some("SPEAKER_02"), Some("SPEAKER_01")]
        );
        // Recording-local labels only: nothing here is a person yet. Naming
        // them is the separate identity stage, and it needs embeddings.
        for segment in &normalized.segments {
            assert!(segment.speaker_person_id.is_none());
        }
    }

    #[test]
    fn family_auto_gpu_policy_lookup_matches_measured_metal_gates() {
        use crate::ggml_runtime::AutoGpuPolicy;

        // Regression pin: dolphin lets Auto pick any GPU-class backend
        // (it flipped from CPU-pinned once its encoder weight-placement fix
        // let Metal truly offload and beat CPU end-to-end). xasr-zipformer and
        // moonshine are `ExceptMetal`: Auto still prefers the generic GPU lane
        // but falls back to CPU on Apple Silicon Metal for their dispatch-bound
        // graph shapes. Qwen stays explicitly `AllBackends`; a family-specific
        // platform gate must not spread to a neighboring architecture.
        assert_eq!(
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                crate::arch::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID
            ),
            AutoGpuPolicy::ExceptMetal
        );
        assert_eq!(
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                crate::arch::DOLPHIN_GGML_ARCHITECTURE_ID
            ),
            AutoGpuPolicy::AllBackends
        );
        assert_eq!(
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                crate::arch::MOONSHINE_GGML_ARCHITECTURE_ID
            ),
            AutoGpuPolicy::ExceptMetal
        );
        assert_eq!(
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID
            ),
            AutoGpuPolicy::AllBackends
        );
        // An unrecognized architecture defaults to the majority behavior
        // (Auto may use any GPU backend) rather than silently pinning an
        // unknown family to CPU.
        assert_eq!(
            crate::arch::family_auto_gpu_policy_for_model_architecture("not-a-real-architecture"),
            AutoGpuPolicy::AllBackends
        );
    }

    /// Regression for the gated-family-plus-Auto provenance mislabel: the
    /// `core.native.backend` label must resolve through the same
    /// family-aware gate the family's own executor used
    /// (`GgmlCpuGraphConfig::resolve_family_runtime_backend`), not recompute
    /// generically. Before this fix, `native_runtime_backend_label` called
    /// `GgmlCpuGraphConfig::resolve_runtime_backend()` directly, which on a
    /// host with a GPU device reports "metal" for an Auto request from a
    /// CPU-gated family (xasr-zipformer today) that in fact ran entirely on
    /// CPU; see `xasr_zipformer::graph_config::encoder_gpu_enabled`.
    #[test]
    fn native_runtime_backend_label_reflects_family_auto_gate_not_generic_resolver() {
        use crate::ggml_runtime::{
            AutoGpuPolicy, RequestBackendPreference, ResolvedFamilyRuntimeInput,
            install_request_backend_override, request_backend_override,
        };

        // `native_runtime_backend_label` itself takes an already-resolved
        // backend: resolution happens once, in
        // `ResolvedFamilyRuntimeInput::resolve`, not inside the label
        // formatter. This helper reproduces exactly that resolution step
        // from the still-live `request_backend_override()` TLS (the
        // pre-existing, unrelated per-request-override mechanism this test
        // exercises via `install_request_backend_override` below) plus a
        // family's `AutoGpuPolicy` gate, mirroring what the real call site
        // in `transcribe_native` does.
        let label_for = |policy: AutoGpuPolicy| {
            native_runtime_backend_label(
                ResolvedFamilyRuntimeInput::resolve(request_backend_override(), policy).backend(),
            )
        };

        // Auto, family gate fully disabled (`Never` shape): must report
        // "cpu" regardless of what the generic resolver would pick.
        assert_eq!(label_for(AutoGpuPolicy::Never), "cpu");

        // Auto, family gate enabled (`AllBackends` shape): reports exactly
        // what the generic resolver picks -- unchanged behavior.
        let generic_auto_label = match GgmlCpuGraphConfig::runtime_default().backend {
            GgmlCpuGraphBackend::Cpu => "cpu",
            GgmlCpuGraphBackend::Metal => "metal",
            GgmlCpuGraphBackend::Gpu => "gpu",
        };
        assert_eq!(label_for(AutoGpuPolicy::AllBackends), generic_auto_label);

        // `ExceptMetal`: reports "cpu" if and only if the generic resolver
        // would have picked Metal specifically; never touches a resolved
        // Cpu or generic Gpu (CUDA/HIP/Vulkan) pick.
        let except_metal_label = label_for(AutoGpuPolicy::ExceptMetal);
        if generic_auto_label == "metal" {
            assert_eq!(except_metal_label, "cpu");
        } else {
            assert_eq!(except_metal_label, generic_auto_label);
        }

        // An explicit accelerated request always reports the accelerated
        // backend, even for a family whose Auto default is gated to CPU --
        // the gate never overrides an explicit per-request choice.
        {
            let _guard =
                install_request_backend_override(Some(RequestBackendPreference::Accelerated));
            let label = label_for(AutoGpuPolicy::Never);
            assert!(label == "metal" || label == "gpu", "got {label}");
            assert_eq!(label, label_for(AutoGpuPolicy::AllBackends));
            assert_eq!(label, label_for(AutoGpuPolicy::ExceptMetal));
        }
    }

    fn test_plan(align: bool) -> ProgressPlan {
        ProgressPlan::build(ProgressPlanInput {
            audio_duration_s: 30.0,
            voice_id: false,
            external_diarize: false,
            segmenter: ProgressSegmenterKind::Auto,
            punctuate: false,
            align,
            backend: ProgressBackendClass::AutoOrCpu,
            persist: false,
        })
    }

    #[test]
    fn external_diarization_progress_shares_are_bounded_and_provider_specific() {
        assert_eq!(completed_work_fraction(0, 0), 1.0);
        assert_eq!(completed_work_fraction(0, 4), 0.0);
        assert_eq!(completed_work_fraction(1, 4), 0.25);
        assert_eq!(completed_work_fraction(8, 4), 1.0);

        let seg3 = external_diarization_segment_share(ProgressSegmenterKind::Segmentation3_0);
        let auto = external_diarization_segment_share(ProgressSegmenterKind::Auto);
        let diarizen = external_diarization_segment_share(ProgressSegmenterKind::DiariZen);
        assert_eq!(seg3, auto);
        assert!(seg3 > 0.0);
        assert!(diarizen > seg3);
        assert!(diarizen < EXTERNAL_DIARIZATION_EMBEDDING_END);
        for share in [seg3, diarizen] {
            assert_eq!(
                external_diarization_embedding_progress(share, 0, 4),
                share,
                "embedding starts exactly where segmentation ends"
            );
            assert_eq!(
                external_diarization_embedding_progress(share, 4, 4),
                EXTERNAL_DIARIZATION_EMBEDDING_END,
                "embedding completion leaves an explicit clustering tail"
            );
        }
    }

    #[test]
    fn forced_aligner_milestones_are_monotonic_and_fill_one_segment_window() {
        use crate::models::qwen::ForcedAlignerProgressEvent;

        let events = [
            ForcedAlignerProgressEvent::MelReady,
            ForcedAlignerProgressEvent::AudioEncodingStarted,
            ForcedAlignerProgressEvent::AudioEncoded,
            ForcedAlignerProgressEvent::PromptPrepared,
            ForcedAlignerProgressEvent::DecoderPrefillStarted,
            ForcedAlignerProgressEvent::DecoderPrefilled,
            ForcedAlignerProgressEvent::TimestampLogitsStarted { total: 4 },
            ForcedAlignerProgressEvent::TimestampLogits {
                completed: 1,
                total: 4,
            },
            ForcedAlignerProgressEvent::TimestampLogits {
                completed: 4,
                total: 4,
            },
            ForcedAlignerProgressEvent::Finalized,
        ];
        for backend in [
            crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
            crate::ggml_runtime::GgmlCpuGraphBackend::Metal,
            crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
        ] {
            let inner: Vec<f64> = events
                .iter()
                .copied()
                .map(|event| forced_aligner_inner_fraction(event, backend))
                .collect();
            assert!(inner.windows(2).all(|pair| pair[1] >= pair[0]));
            assert_eq!(inner.last().copied(), Some(1.0));

            let completed_before = 10.0;
            let segment_duration = 20.0;
            let total_duration = 50.0;
            let stage: Vec<f32> = inner
                .into_iter()
                .map(|fraction| {
                    duration_weighted_fraction(
                        completed_before + segment_duration * fraction,
                        total_duration,
                    )
                })
                .collect();
            assert!(stage.windows(2).all(|pair| pair[1] >= pair[0]));
            assert!((stage.last().copied().unwrap() - 0.6).abs() < 1e-6);
        }
        assert_eq!(
            forced_aligner_progress_detail(
                3,
                ForcedAlignerProgressEvent::TimestampLogits {
                    completed: 2,
                    total: 4,
                },
            ),
            "forced_aligner segment=3 phase=timestamp_logits:2/4"
        );
    }

    #[test]
    fn native_progress_is_monotonic_across_stages_and_clears() {
        let _serial = progress_registry_test_lock();
        let id = "monotonic-phases";
        assert_eq!(native_transcription_progress_for_id(id), None);
        {
            let _handle = ProgressRegistryHandle::new(Some(id.to_string()));
            let reporter = ProgressReporter::install(Some(id.to_string()), test_plan(true));
            let decode = DecodeProgress::begin(reporter.clone(), 1000);
            let start = native_transcription_progress_for_id(id).expect("run is active");
            assert_eq!(start.stage, TranscriptionStage::Decode);
            assert_eq!(start.phase, NativeTranscriptionPhase::Decode);

            decode.complete_slice(400);
            let mid = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(mid.stage, TranscriptionStage::Decode);
            assert!(mid.overall_fraction >= start.overall_fraction);
            assert!((mid.stage_fraction.unwrap() - 0.4).abs() < 1e-5);

            decode.complete_slice(600);
            let decoded = native_transcription_progress_for_id(id).unwrap();
            assert!(decoded.overall_fraction >= mid.overall_fraction);
            assert!((decoded.stage_fraction.unwrap() - 1.0).abs() < 1e-5);

            // Real pipeline order: Align before Project.
            reporter.enter_stage(TranscriptionStage::Align);
            let aligning = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(aligning.phase, NativeTranscriptionPhase::Align);
            assert!(aligning.overall_fraction >= decoded.overall_fraction);

            reporter.report_fraction(0.5);
            let align_mid = native_transcription_progress_for_id(id).unwrap();
            assert!(align_mid.overall_fraction >= aligning.overall_fraction);

            // Lower stage report must not regress overall.
            reporter.report_fraction(0.1);
            let after = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(after.overall_fraction, align_mid.overall_fraction);

            reporter.complete_stage_brief(TranscriptionStage::Project);
            let projected = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(projected.phase, NativeTranscriptionPhase::Assemble);
            assert!(projected.overall_fraction >= after.overall_fraction);
        }
        assert_eq!(native_transcription_progress_for_id(id), None);
    }

    #[test]
    fn native_progress_two_concurrent_requests_stay_independent_and_a_finishing_does_not_affect_b()
    {
        let _serial = progress_registry_test_lock();
        let id_a = "concurrent-a";
        let id_b = "concurrent-b";
        assert_eq!(native_transcription_progress_for_id(id_a), None);
        assert_eq!(native_transcription_progress_for_id(id_b), None);

        let handle_a = ProgressRegistryHandle::new(Some(id_a.to_string()));
        let handle_b = ProgressRegistryHandle::new(Some(id_b.to_string()));
        let ra = ProgressReporter::install(Some(id_a.to_string()), test_plan(false));
        let rb = ProgressReporter::install(Some(id_b.to_string()), test_plan(true));

        ra.enter_stage(TranscriptionStage::Decode);
        ra.report_fraction(0.4);
        rb.enter_stage(TranscriptionStage::Align);
        rb.report_fraction(0.9);

        let progress_a = native_transcription_progress_for_id(id_a).expect("A is active");
        let progress_b = native_transcription_progress_for_id(id_b).expect("B is active");
        assert_eq!(progress_a.stage, TranscriptionStage::Decode);
        assert!((progress_a.stage_fraction.unwrap() - 0.4).abs() < 1e-5);
        assert_eq!(progress_b.stage, TranscriptionStage::Align);
        assert!((progress_b.stage_fraction.unwrap() - 0.9).abs() < 1e-5);

        ra.report_fraction(0.5);
        let progress_b_after_a_advances = native_transcription_progress_for_id(id_b).unwrap();
        assert_eq!(
            progress_b_after_a_advances.overall_fraction,
            progress_b.overall_fraction
        );

        drop(handle_a);
        assert_eq!(native_transcription_progress_for_id(id_a), None);
        let progress_b_after_a_finishes =
            native_transcription_progress_for_id(id_b).expect("B must survive A finishing");
        assert_eq!(
            progress_b_after_a_finishes.overall_fraction,
            progress_b.overall_fraction
        );

        drop(handle_b);
        assert_eq!(native_transcription_progress_for_id(id_b), None);
    }

    #[test]
    fn native_progress_detached_request_never_publishes() {
        let _serial = progress_registry_test_lock();
        let _handle = ProgressRegistryHandle::new(None);
        let reporter = ProgressReporter::install(None, test_plan(false));
        let decode = DecodeProgress::begin(reporter.clone(), 1000);
        decode.complete_slice(500);
        reporter.complete_stage_brief(TranscriptionStage::Project);
        assert_eq!(
            native_transcription_progress_for_id("native-progress-detached-request-probe"),
            None
        );
    }

    #[test]
    fn native_progress_sequential_runs_reset_start_and_clear() {
        let _serial = progress_registry_test_lock();
        let id = "sequential-runs";
        assert_eq!(native_transcription_progress_for_id(id), None);

        {
            let _run1 = ProgressRegistryHandle::new(Some(id.to_string()));
            let r1 = ProgressReporter::install(Some(id.to_string()), test_plan(false));
            r1.enter_stage(TranscriptionStage::Decode);
            r1.report_fraction(0.1);
            r1.report_fraction(0.9);
            let run1_progress = native_transcription_progress_for_id(id).unwrap();
            assert!((run1_progress.stage_fraction.unwrap() - 0.9).abs() < 1e-5);
        }
        assert_eq!(native_transcription_progress_for_id(id), None);

        {
            let _run2 = ProgressRegistryHandle::new(Some(id.to_string()));
            let r2 = ProgressReporter::install(Some(id.to_string()), test_plan(false));
            r2.enter_stage(TranscriptionStage::Decode);
            r2.report_fraction(0.2);
            let run2_start = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(run2_start.stage, TranscriptionStage::Decode);
            assert!((run2_start.stage_fraction.unwrap() - 0.2).abs() < 1e-5);

            r2.report_fraction(0.05);
            let run2_after_lower = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(run2_after_lower.stage_fraction, run2_start.stage_fraction);

            r2.complete_stage_brief(TranscriptionStage::Project);
            let run2_projected = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(run2_projected.stage, TranscriptionStage::Project);
            assert!(run2_projected.overall_fraction >= run2_start.overall_fraction);
        }
        assert_eq!(native_transcription_progress_for_id(id), None);
    }

    #[test]
    fn native_transcription_progress_legacy_reports_idle_with_no_active_runs() {
        let _serial = progress_registry_test_lock();
        clear_progress_registry_for_test();
        assert_eq!(
            native_transcription_progress(),
            LegacyNativeTranscriptionProgress::Idle
        );
    }

    #[test]
    fn native_transcription_progress_legacy_reports_the_single_active_run() {
        let _serial = progress_registry_test_lock();
        clear_progress_registry_for_test();
        let id = "legacy-single-active";
        let _handle = ProgressRegistryHandle::new(Some(id.to_string()));
        let reporter = ProgressReporter::install(Some(id.to_string()), test_plan(false));
        reporter.enter_stage(TranscriptionStage::Decode);
        reporter.report_fraction(0.33);
        match native_transcription_progress() {
            LegacyNativeTranscriptionProgress::Single(p) => {
                assert_eq!(p.stage, TranscriptionStage::Decode);
                assert!((p.stage_fraction.unwrap() - 0.33).abs() < 1e-5);
                assert!((p.fraction - p.overall_fraction).abs() < 1e-9);
            }
            other => panic!("expected Single, got {other:?}"),
        }
        clear_progress_registry_for_test();
    }

    #[test]
    fn native_transcription_progress_legacy_is_ambiguous_with_more_than_one_active_run() {
        let _serial = progress_registry_test_lock();
        clear_progress_registry_for_test();
        let id_a = "legacy-ambiguous-a";
        let id_b = "legacy-ambiguous-b";
        let _handle_a = ProgressRegistryHandle::new(Some(id_a.to_string()));
        let _handle_b = ProgressRegistryHandle::new(Some(id_b.to_string()));
        let ra = ProgressReporter::install(Some(id_a.to_string()), test_plan(false));
        let rb = ProgressReporter::install(Some(id_b.to_string()), test_plan(false));
        ra.enter_stage(TranscriptionStage::Decode);
        rb.enter_stage(TranscriptionStage::Decode);
        assert_eq!(
            native_transcription_progress(),
            LegacyNativeTranscriptionProgress::Ambiguous { active_count: 2 }
        );
        clear_progress_registry_for_test();
    }

    #[test]
    fn decode_work_fraction_normalizes_completed_units_against_total() {
        let window = SliceProgressWindow {
            start_fraction: 0.0,
            span_fraction: 1.0,
        };
        assert!((decode_work_fraction(window, 1, 10) - 0.1).abs() < 1e-6);
        assert!((decode_work_fraction(window, 5, 10) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn decode_work_fraction_scales_by_the_slice_window() {
        // A slice that owns [0.2, 0.2 + 0.3) of the decode-phase fraction:
        // token progress must land inside that sub-range, not [0, 1].
        let window = SliceProgressWindow {
            start_fraction: 0.2,
            span_fraction: 0.3,
        };
        let at_start = decode_work_fraction(window, 1, 100);
        let at_half = decode_work_fraction(window, 50, 100);
        assert!((at_start - (0.2 + 0.3 * 0.01)).abs() < 1e-6);
        assert!((at_half - (0.2 + 0.3 * 0.50)).abs() < 1e-6);
        assert!(at_start >= window.start_fraction);
        assert!(at_half <= window.start_fraction + window.span_fraction);
    }

    #[test]
    fn decode_work_fraction_caps_below_the_full_slice_span() {
        // Even once completed work reaches (or blows past) total work,
        // the window's own share must stay strictly under its full span --
        // `DecodeProgress::complete_slice` owns closing out the remaining
        // sliver, not work interpolation racing ahead of it.
        let window = SliceProgressWindow {
            start_fraction: 0.0,
            span_fraction: 1.0,
        };
        let at_cap = decode_work_fraction(window, 100, 100);
        let past_cap = decode_work_fraction(window, 501, 100);
        assert!((at_cap - DECODE_WORK_PROGRESS_SLICE_SHARE_CAP).abs() < 1e-6);
        assert!((past_cap - DECODE_WORK_PROGRESS_SLICE_SHARE_CAP).abs() < 1e-6);
        assert!(at_cap < window.start_fraction + window.span_fraction);
    }

    #[test]
    fn decode_work_fraction_is_monotonic_in_completed_work() {
        let window = SliceProgressWindow {
            start_fraction: 0.1,
            span_fraction: 0.4,
        };
        let mut previous = decode_work_fraction(window, 0, 37);
        for completed_work in 1..200 {
            let current = decode_work_fraction(window, completed_work, 37);
            assert!(
                current >= previous,
                "fraction regressed at work {completed_work}: {previous} -> {current}"
            );
            previous = current;
        }
    }

    #[test]
    fn decode_work_fraction_falls_back_to_the_cap_when_total_is_zero() {
        // A zero denominator (defensive: no builtin family emits
        // max_generated_tokens=0, `Seq2SeqGreedyDecodeConfig` fails closed on
        // it) must not divide by zero or report the window as fully done --
        // the cap is the safe fallback, matching an "unknown, assume
        // in-progress" reading.
        let window = SliceProgressWindow {
            start_fraction: 0.0,
            span_fraction: 1.0,
        };
        assert!(
            (decode_work_fraction(window, 0, 0) - DECODE_WORK_PROGRESS_SLICE_SHARE_CAP).abs()
                < 1e-6
        );
    }

    #[test]
    fn slice_progress_window_places_slices_back_to_back_within_decode_stage() {
        // Windows are in stage-fraction space (0..=1 of decode), not overall.
        let _serial = progress_registry_test_lock();
        let id = "slice-window-back-to-back";
        let _handle = ProgressRegistryHandle::new(Some(id.to_string()));
        let reporter = ProgressReporter::install(Some(id.to_string()), test_plan(false));
        let decode = DecodeProgress::begin(reporter, 1000);
        let first = decode.slice_progress_window(400);
        assert!((first.start_fraction - 0.0).abs() < 1e-6);
        assert!((first.span_fraction - 0.4).abs() < 1e-6);

        decode.complete_slice(400);
        let second = decode.slice_progress_window(600);
        assert!((second.start_fraction - 0.4).abs() < 1e-6);
        assert!((second.span_fraction - 0.6).abs() < 1e-6);
        assert!((second.start_fraction + second.span_fraction - 1.0).abs() < 1e-6);
    }

    #[test]
    fn slice_progress_window_is_the_full_decode_stage_for_a_single_slice_run() {
        let _serial = progress_registry_test_lock();
        let id = "slice-window-single-slice";
        let _handle = ProgressRegistryHandle::new(Some(id.to_string()));
        let reporter = ProgressReporter::install(Some(id.to_string()), test_plan(true));
        let decode = DecodeProgress::begin(reporter, 1000);
        let window = decode.slice_progress_window(1000);
        assert!((window.start_fraction - 0.0).abs() < 1e-6);
        assert!((window.span_fraction - 1.0).abs() < 1e-6);
    }

    #[test]
    fn should_publish_decode_work_throttles_and_keeps_first_and_final_units() {
        assert!(should_publish_decode_work(1, 20));
        for completed_work in 2..DECODE_WORK_PROGRESS_PUBLISH_STRIDE {
            assert!(
                !should_publish_decode_work(completed_work, 20),
                "work unit {completed_work} should be throttled"
            );
        }
        assert!(should_publish_decode_work(
            DECODE_WORK_PROGRESS_PUBLISH_STRIDE,
            20
        ));
        assert!(should_publish_decode_work(20, 20));
    }

    /// End-to-end wiring: a request-local observer reports stage progress from
    /// a different thread, proving it follows the request rather than TLS.
    #[test]
    fn decode_work_progress_crosses_threads_and_stays_inside_its_window() {
        let _serial = progress_registry_test_lock();
        let id = "token-step-sink-window";
        assert_eq!(native_transcription_progress_for_id(id), None);

        {
            let _handle = ProgressRegistryHandle::new(Some(id.to_string()));
            let reporter = ProgressReporter::install(Some(id.to_string()), test_plan(false));
            reporter.enter_stage(TranscriptionStage::Decode);
            let window = SliceProgressWindow {
                start_fraction: 0.0,
                span_fraction: 1.0,
            };
            let observed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let observer = crate::api::backend::WorkProgressObserver::new({
                let reporter = reporter.clone();
                let observed = std::sync::Arc::clone(&observed);
                move |completed_work, total_work| {
                    if should_publish_decode_work(completed_work, total_work) {
                        let fraction = decode_work_fraction(window, completed_work, total_work);
                        observed
                            .lock()
                            .expect("progress observations")
                            .push(fraction);
                        reporter.report_fraction(fraction);
                    }
                }
            });
            let context =
                crate::RequestExecutionContext::uncancellable("cross-thread progress test")
                    .with_decode_work_progress_observer(observer);
            std::thread::spawn(move || {
                for completed_work in 1..=40 {
                    context
                        .decode_work_progress_observer()
                        .expect("observer follows context")
                        .report(completed_work, 40);
                }
            })
            .join()
            .expect("worker thread");

            let observed = observed.lock().expect("progress observations");
            assert_eq!(observed.len(), 11);
            for pair in observed.windows(2) {
                assert!(pair[0] <= pair[1], "progress regressed: {pair:?}");
            }
            assert!(observed.iter().all(|value| *value <= 1.0));
            assert!(
                (observed.last().copied().unwrap_or_default()
                    - DECODE_WORK_PROGRESS_SLICE_SHARE_CAP)
                    .abs()
                    < 1e-6
            );
            let progress =
                native_transcription_progress_for_id(id).expect("observer published progress");
            assert_eq!(progress.stage, TranscriptionStage::Decode);
        }
        assert_eq!(native_transcription_progress_for_id(id), None);
    }

    /// Real-decode regression for the short-audio / single-pass progress gap
    /// this change fixes: before it, `run_native_transcription` on audio
    /// under the longform trigger (`fixtures/jfk.wav`, ~11s) never called
    /// `publish_progress` at all -- its progress stayed unreadable for the
    /// whole decode, and the UI fell back to a pure time estimate with no
    /// relationship to real progress (see the recon this change is based
    /// on). Runs a real firered-aed decode on a background thread while
    /// polling this request's id-scoped progress from this thread, and
    /// requires at least one snapshot strictly between 0 and the decode
    /// ceiling -- proof of a genuine intermediate signal, not just an initial
    /// 0.0 immediately followed by the ceiling. Attaches a real
    /// transcription id via `with_execution_context` (unlike
    /// `TranscriptionRequest::new`'s uncancellable default) since a detached
    /// request never publishes at all under the id-scoped registry.
    #[test]
    #[ignore = "host-local: requires tmp/firered-aed-l-v2-q4_k.oasr (a real firered-aed pack)"]
    fn real_decode_short_audio_reports_intermediate_token_level_progress() {
        let pack =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/firered-aed-l-v2-q4_k.oasr");
        if !pack.exists() {
            eprintln!("skipping: pack ({}) absent", pack.display());
            return;
        }
        let wav = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");

        let id = "real-decode-short-audio";
        assert_eq!(native_transcription_progress_for_id(id), None);

        let pack = pack.canonicalize().expect("pack path must canonicalize");
        let wav = wav.canonicalize().expect("wav path must canonicalize");
        let execution_context = Arc::new(crate::RequestExecutionContext::new(
            Some(id.to_string()),
            Arc::new(crate::TranscriptionControl::new()),
        ));
        let request = TranscriptionRequest::new(wav, NATIVE_RUNTIME_MODEL_ID_AUTO)
            .with_model_pack_path(Some(pack))
            .with_execution_context(execution_context);

        let execution_services = native_execution_services_for_test();
        let decode_thread =
            std::thread::spawn(move || run_native_transcription(request, execution_services));

        let mut saw_intermediate_signal = false;
        let mut previous_fraction = 0.0_f32;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while !decode_thread.is_finished() && std::time::Instant::now() < deadline {
            if let Some(progress) = native_transcription_progress_for_id(id) {
                assert_eq!(progress.phase, NativeTranscriptionPhase::Decode);
                // Monotonic even across raw polling (no lock held across
                // reads, but the registry's own lock guarantees a reader
                // never observes a regression).
                assert!(progress.fraction >= previous_fraction);
                previous_fraction = progress.fraction;
                if progress.overall_fraction > 0.0 && progress.overall_fraction < 0.99 {
                    saw_intermediate_signal = true;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let transcription = decode_thread
            .join()
            .expect("decode thread must not panic")
            .expect("real decode must succeed");
        assert!(
            transcription.text.to_uppercase().contains("COUNTRY"),
            "unexpected transcript: {:?}",
            transcription.text
        );
        assert!(
            saw_intermediate_signal,
            "expected at least one progress snapshot strictly between 0 and the decode ceiling; \
             short-audio decode must report continuous token-level progress, not stay silent \
             until completion"
        );
        assert_eq!(native_transcription_progress_for_id(id), None);
    }

    #[test]
    fn native_runtime_model_refs_match_catalog_quant_aliases() {
        assert!(native_runtime_model_refs_match(
            "qwen3-asr-0.6b:q8",
            "qwen3-asr-0.6b:q8_0"
        ));
        assert!(native_runtime_model_refs_match(
            "qwen3-asr-0.6b:q4_k_m",
            "qwen3-asr-0.6b:q4_k"
        ));
        assert!(!native_runtime_model_refs_match(
            "qwen3-asr-0.6b",
            "qwen3-asr-0.6b:q8_0"
        ));
        // Quant-pinned request vs the BARE runtime source id (the loaded native
        // pack's openasr.model.id has no quant tag): must match — it names that
        // same single loaded pack. Regression guard for dictation / live captions,
        // which send "<id>:<quant>".
        assert!(native_runtime_model_refs_match(
            "qwen3-asr-0.6b:q8_0",
            "qwen3-asr-0.6b"
        ));
        assert!(!native_runtime_model_refs_match(
            "qwen3-asr-1.7b:q8",
            "qwen3-asr-0.6b:q8_0"
        ));
    }

    // Regression guard for the reported bug: a runtime source id whose
    // `openasr.model.id` was baked by an older mimo-asr conversion tool as
    // `family-quant` (hyphen-joined) instead of the catalog's `family:quant`
    // colon convention must still match a colon-form request naming any
    // recognized alias of that quant. Fixed forward in
    // tooling/mimo-asr/convert_mimo_asr.py, but already-published packs still
    // carry the old metadata, so the matcher must tolerate it.
    #[test]
    fn native_runtime_model_refs_match_legacy_hyphen_joined_runtime_source_id() {
        assert!(native_runtime_model_refs_match(
            "mimo-v2.5-asr:q4",
            "mimo-v2.5-asr-q4_k"
        ));
        assert!(native_runtime_model_refs_match(
            "mimo-v2.5-asr:q4_k",
            "mimo-v2.5-asr-q4_k"
        ));
        assert!(native_runtime_model_refs_match(
            "mimo-v2.5-asr:q8_0",
            "mimo-v2.5-asr-q8_0"
        ));
        // Different quant on each side: still a mismatch even through the
        // legacy hyphen fallback (fail-closed, not a blanket bare-family pass).
        assert!(!native_runtime_model_refs_match(
            "mimo-v2.5-asr:q8_0",
            "mimo-v2.5-asr-q4_k"
        ));
        // Different family: the hyphen split must not make an unrelated
        // family with a coincidentally quant-alias-shaped suffix match.
        assert!(!native_runtime_model_refs_match(
            "mimo-v2.5-asr:q4",
            "some-other-family-q4_k"
        ));
        // A genuinely single-word family with no quant suffix at all must
        // stay a bare-id match (no accidental split).
        assert!(native_runtime_model_refs_match(
            "whisper-runtime:q8_0",
            "whisper-runtime"
        ));
    }

    // The catalog product suffix for mixed Q4_K_M packs is
    // "q4km" (tooling/publish-model/scripts/_catalog.py QUANT_METADATA), which
    // is exactly what a user copies from `pull_recommended` /
    // `openasr pull translator-test:q4km`. `canonical_quant_tag` must recognize it
    // as an alias of q4_k so a request using it matches a runtime source
    // tagged with any other spelling of the same quant.
    #[test]
    fn native_runtime_model_refs_match_catalog_q4km_product_suffix_alias() {
        assert!(native_runtime_model_refs_match(
            "translator-test:q4km",
            "translator-test:q4_k"
        ));
        assert!(native_runtime_model_refs_match(
            "translator-test:q4_k_m",
            "translator-test:q4km"
        ));
    }

    #[test]
    fn implicit_native_longform_stays_off_for_short_audio() {
        let resolution =
            resolve_native_longform_policy_for_backend(None, 10.6, "", GgmlCpuGraphBackend::Cpu);
        assert_eq!(resolution.options.mode, LongFormMode::Off);
    }

    #[test]
    fn implicit_native_longform_uses_auto_for_long_audio() {
        let resolution =
            resolve_native_longform_policy_for_backend(None, 120.0, "", GgmlCpuGraphBackend::Cpu);
        assert_eq!(resolution.options.mode, LongFormMode::Auto);
    }

    /// A `ScopedSlices` family decodes a recording whole whenever its context
    /// can serve it, and only slices past that point. Slicing costs identity
    /// (every seam restarts the in-decoder speaker numbering) and can clip
    /// speech at cut points, so it must be the fallback, not the default: a
    /// recording inside `integral_seconds` has to come back with longform off,
    /// however long it is relative to the generic 30s auto-trigger.
    #[test]
    fn scoped_slice_family_decodes_a_recording_that_fits_its_context_whole() {
        let crate::arch::OpenAsrLongformSliceShape::ScopedSlices {
            integral_seconds,
            target_seconds,
            ..
        } = crate::arch::longform_slice_shape_for_model_architecture(
            crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
        )
        else {
            panic!("moss-transcribe-diarize must declare ScopedSlices");
        };

        // Well past the generic auto-trigger and past a single slice window,
        // but still inside what one prompt can serve.
        for audio_seconds in [
            DEFAULT_NATIVE_LONGFORM_AUTO_TRIGGER_SECONDS + 1.0,
            target_seconds + 1.0,
            integral_seconds,
        ] {
            let resolution = resolve_native_longform_policy_for_backend(
                None,
                audio_seconds,
                crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
                GgmlCpuGraphBackend::Cpu,
            );
            assert_eq!(
                resolution.options.mode,
                LongFormMode::Off,
                "{audio_seconds}s fits one decode and must not be sliced"
            );
        }

        // Just past it, slicing takes over rather than failing the request.
        let resolution = resolve_native_longform_policy_for_backend(
            None,
            integral_seconds + 1.0,
            crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
            GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(resolution.options.mode, LongFormMode::Energy);
        assert_eq!(resolution.options.chunk_seconds, target_seconds);
    }

    /// The integral path is an *automatic* policy decision. A caller that
    /// explicitly asked for longform options still gets them, so an explicit
    /// request is never silently overridden into a whole-recording decode its
    /// context may not survive.
    #[test]
    fn an_explicit_longform_request_still_slices_inside_the_integral_window() {
        let requested = crate::LongFormOptions::default();
        let resolution = resolve_native_longform_policy_for_backend(
            Some(&requested),
            120.0,
            crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
            GgmlCpuGraphBackend::Cpu,
        );
        assert!(!matches!(requested.mode, LongFormMode::Off));
        assert!(!matches!(resolution.options.mode, LongFormMode::Off));
    }

    /// A `ScopedSlices` family gets its declared decoder-context window rather
    /// than inheriting the shared default by accident. The current product
    /// target deliberately equals the shared 30s target, while the 60s ceiling
    /// remains family-owned and independently asserted below. The shape also
    /// carries the three options it implies (a contiguous full-coverage planner
    /// that cannot elide audio, no padding bias on in-decoder timestamps, and
    /// no free-text prompt carry across a fixed fine-tuned instruction).
    #[test]
    fn scoped_slice_family_gets_its_declared_window_instead_of_the_shared_default() {
        let crate::arch::OpenAsrLongformSliceShape::ScopedSlices {
            target_seconds,
            max_seconds,
            ..
        } = crate::arch::longform_slice_shape_for_model_architecture(
            crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
        )
        else {
            panic!("moss-transcribe-diarize must declare ScopedSlices");
        };
        assert_eq!(target_seconds, 30.0);
        assert_eq!(max_seconds, 60.0);

        let resolution = resolve_native_longform_policy_for_backend(
            None,
            600.0,
            crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
            GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(resolution.options.mode, LongFormMode::Energy);
        assert_eq!(resolution.options.chunk_seconds, target_seconds);
        assert_eq!(resolution.options.max_chunk_seconds, max_seconds);
        assert_eq!(resolution.options.padding_seconds, 0.0);
        assert!(!resolution.options.carry_prompt_across_slices);
        assert_eq!(
            longform_prompt_carry_mode(
                &resolution.options,
                crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID
            ),
            LongformPromptCarryMode::Disabled,
        );
        resolution.options.validate().expect("resolved options");
    }

    /// Deterministic stand-in for a far-field meeting recording where most of
    /// the speech sits *below* the pipeline's absolute silence floor
    /// (`energy_silence_threshold_db`, -38 dBFS): a loud talker near the mic
    /// at the top of each minute, then a long stretch of quiet talkers around
    /// -45 dBFS, then a genuinely silent tail. This is the level profile that
    /// made the auto planner elide 47% of a real 360s recording -- the energy
    /// VAD read sub-floor speech as silence, and the coverage guard read the
    /// same floor back and agreed it was safe to drop. The guard no longer
    /// depends on that floor (see `longform::audibility`), so `Auto` keeps
    /// this profile whole too; the pin below is the structural guarantee that
    /// a scoped-slice family never sees an elided plan regardless.
    fn quiet_speech_under_the_silence_floor(total_seconds: f32) -> Vec<f32> {
        const SAMPLE_RATE: usize = 16_000;
        const BLOCK_SECONDS: usize = 60;
        const LOUD_SECONDS: usize = 6;
        const QUIET_SECONDS: usize = 49;
        const LOUD_AMPLITUDE: f32 = 0.07;
        const QUIET_AMPLITUDE: f32 = 0.0056;
        const SILENCE_AMPLITUDE: f32 = 0.0001;

        let total_samples = (total_seconds * SAMPLE_RATE as f32) as usize;
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        (0..total_samples)
            .map(|index| {
                // xorshift64: a deterministic broadband carrier, so the test
                // depends on the level profile rather than on any waveform.
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let noise = (state >> 40) as f32 / 8_388_608.0 - 1.0;
                let offset = (index / SAMPLE_RATE) % BLOCK_SECONDS;
                let amplitude = if offset < LOUD_SECONDS {
                    LOUD_AMPLITUDE
                } else if offset < LOUD_SECONDS + QUIET_SECONDS {
                    QUIET_AMPLITUDE
                } else {
                    SILENCE_AMPLITUDE
                };
                noise * amplitude
            })
            .collect()
    }

    fn slice_plan_covers_every_sample(plan: &crate::longform::LongFormSlicePlan) -> bool {
        if plan.processed_audio.is_some() {
            return false;
        }
        let mut covered_to = 0usize;
        for slice in &plan.slices {
            if slice.content_start_sample > covered_to {
                return false;
            }
            covered_to = covered_to.max(slice.content_end_sample);
        }
        covered_to >= plan.total_samples
    }

    /// The invariant behind the scoped-slice mode pin: a `ScopedSlices` family
    /// never gets a plan that elides audio, so no assembled segment can span
    /// content the decoder was never given. Asserted on both level profiles,
    /// plus the `Auto` counterfactual on the packable one -- `Auto` really
    /// does elide there, so the test cannot pass on a build where the pin was
    /// deleted.
    #[test]
    fn scoped_slice_family_never_gets_a_plan_that_elides_audio() {
        let samples = quiet_speech_under_the_silence_floor(360.0);
        let resolution = resolve_native_longform_policy_for_backend(
            None,
            samples.len() as f32 / 16_000.0,
            crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
            GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(resolution.options.mode, LongFormMode::Energy);

        let plan = plan_longform_slices(&samples, 16_000, &resolution.options, None)
            .expect("scoped-slice options must plan");
        assert!(plan.slices.len() > 1, "360s must slice at the 180s target");
        assert!(
            slice_plan_covers_every_sample(&plan),
            "scoped slices must cover every sample on an identity timeline, got {:?}",
            plan.slices
        );

        // Counterfactual: `Auto` is free to elide, and on audio whose pauses
        // really are room tone it does. Without this half the test would pass
        // on a build where the mode pin was deleted and `Auto` merely happened
        // to keep the first fixture whole.
        let packable = loud_speech_with_room_tone_gaps(360.0);
        let auto_options = crate::LongFormOptions {
            mode: LongFormMode::Auto,
            ..resolution.options.clone()
        };
        let auto_plan = plan_longform_slices(&packable, 16_000, &auto_options, None)
            .expect("auto options must plan");
        assert!(
            !slice_plan_covers_every_sample(&auto_plan),
            "the Auto planner is expected to elide true room-tone gaps; if it no longer does, \
             this test has stopped proving that the mode pin is what protects coverage"
        );
        let pinned_plan = plan_longform_slices(&packable, 16_000, &resolution.options, None)
            .expect("scoped-slice options must plan");
        assert!(
            slice_plan_covers_every_sample(&pinned_plan),
            "the pinned scoped-slice planner must cover the same audio `Auto` elides"
        );
    }

    /// The other level profile a scoped-slice family must survive: normally
    /// levelled speech separated by genuine room tone, which the auto planner
    /// legitimately packs out. Speech blocks are 20s, gaps 25s.
    fn loud_speech_with_room_tone_gaps(total_seconds: f32) -> Vec<f32> {
        const SAMPLE_RATE: usize = 16_000;
        const BLOCK_SECONDS: usize = 45;
        const SPEECH_SECONDS: usize = 20;
        const SPEECH_AMPLITUDE: f32 = 0.2;
        const ROOM_TONE_AMPLITUDE: f32 = 0.0004;

        let total_samples = (total_seconds * SAMPLE_RATE as f32) as usize;
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        (0..total_samples)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let noise = (state >> 40) as f32 / 8_388_608.0 - 1.0;
                let offset = (index / SAMPLE_RATE) % BLOCK_SECONDS;
                let amplitude = if offset < SPEECH_SECONDS {
                    SPEECH_AMPLITUDE
                } else {
                    ROOM_TONE_AMPLITUDE
                };
                noise * amplitude
            })
            .collect()
    }

    /// A `SharedWindow` family is untouched by the scoped-slice rule.
    #[test]
    fn shared_window_family_keeps_the_generic_longform_window() {
        let defaults = crate::LongFormOptions::default();
        let resolution = resolve_native_longform_policy_for_backend(
            None,
            600.0,
            crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(resolution.options.chunk_seconds, defaults.chunk_seconds);
        assert_eq!(resolution.options.padding_seconds, defaults.padding_seconds);
        assert!(resolution.options.carry_prompt_across_slices);
    }

    #[test]
    fn whisper_semantic_window_caps_padded_executor_inputs_at_thirty_seconds() {
        let requested = crate::LongFormOptions {
            mode: LongFormMode::Fixed,
            chunk_seconds: 30.0,
            max_chunk_seconds: 60.0,
            padding_seconds: 0.25,
            ..crate::LongFormOptions::default()
        };
        let resolution = resolve_native_longform_policy_for_backend(
            Some(&requested),
            61.0,
            crate::WHISPER_GGML_ARCHITECTURE_ID,
            GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(resolution.options.chunk_seconds, 30.0);
        assert_eq!(resolution.options.max_chunk_seconds, 30.0);
        assert!(
            resolution
                .provenance
                .iter()
                .any(|entry| entry.contains("invocation-span-cap=30"))
        );

        let samples = vec![0.05_f32; 61 * 16_000];
        let plan = plan_longform_slices(&samples, 16_000, &resolution.options, None)
            .expect("fixed Whisper slices");
        assert!(plan.slices.len() >= 3);
        assert!(
            plan.slices
                .iter()
                .all(|slice| slice.duration_samples() <= 30 * 16_000),
            "padding must shrink inside the semantic invocation cap"
        );
    }

    /// granite-speech is `SharedWindow` + `LocalChunked` + decode-policy
    /// `Default` (not `ConservativeSeq2SeqV1`): multi-minute audio must ride the
    /// generic longform window (default 30s chunk) rather than a tighter
    /// conservative cap or a whole-recording integral window. This is the
    /// planner-side half of the long-audio degradation gate -- the pack-backed
    /// multi-slice e2e lives next to the family executor.
    #[test]
    fn granite_speech_shared_window_keeps_generic_longform_window() {
        let defaults = crate::LongFormOptions::default();
        assert_eq!(
            crate::arch::longform_slice_shape_for_model_architecture(
                crate::arch::GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
            crate::arch::OpenAsrLongformSliceShape::SharedWindow,
            "granite-speech must stay SharedWindow so multi-minute audio is sliced"
        );

        let resolution = resolve_native_longform_policy_for_backend(
            None,
            90.0,
            crate::arch::GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(resolution.options.mode, LongFormMode::Auto);
        assert_eq!(resolution.options.chunk_seconds, defaults.chunk_seconds);
        assert_eq!(
            resolution.options.max_chunk_seconds,
            defaults.max_chunk_seconds
        );
        assert!(
            resolution.options.carry_prompt_across_slices,
            "the generic window resolver preserves the requested carry switch"
        );
        assert_eq!(
            longform_prompt_carry_mode(
                &resolution.options,
                crate::arch::GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
            LongformPromptCarryMode::Disabled,
            "granite has no carry producer, so its effective policy must remain disabled"
        );
        // LocalChunked encoder + Default profile: no encoder-memory or
        // conservative-seq2seq provenance tags on the auto path.
        assert!(
            resolution.provenance.iter().all(|entry| {
                !entry.contains("conservative-seq2seq-chunk-cap")
                    && !entry.contains("encoder-attention-span")
                    && !entry.contains("scoped-slices")
            }),
            "unexpected longform safety provenance for granite-speech: {:?}",
            resolution.provenance
        );
    }

    /// Fixed-window plan for ~69s of audio under granite's resolved longform
    /// options must produce multiple slices, each bounded by the default chunk
    /// window (+ padding). This is the weight-free structural gate that the
    /// multi-slice pack e2e depends on: if the planner ever collapsed back to a
    /// single whole-recording buffer, the 256-token generation backstop would
    /// silently truncate multi-minute speech inside one decode.
    #[test]
    fn granite_speech_longform_planner_splits_beyond_default_window() {
        const SAMPLE_RATE_HZ: u32 = 16_000;
        const AUDIO_SECONDS: f32 = 69.0;
        let total_samples = (AUDIO_SECONDS * SAMPLE_RATE_HZ as f32) as usize;
        // Non-silent samples so energy/auto fallbacks do not collapse the plan.
        let samples = vec![0.05_f32; total_samples];

        let requested = crate::LongFormOptions {
            mode: LongFormMode::Fixed,
            ..crate::LongFormOptions::default()
        };
        let resolution = resolve_native_longform_policy_for_backend(
            Some(&requested),
            AUDIO_SECONDS,
            crate::arch::GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(resolution.options.mode, LongFormMode::Fixed);
        assert_eq!(
            resolution.options.chunk_seconds,
            crate::LongFormOptions::default().chunk_seconds
        );

        let plan = plan_longform_slices(&samples, SAMPLE_RATE_HZ, &resolution.options, None)
            .expect("granite SharedWindow fixed plan must build");
        assert!(
            plan.stats.chunk_count >= 3,
            "69s at the default 30s window must yield >=3 slices, got {} ({:?})",
            plan.stats.chunk_count,
            plan.slices
                .iter()
                .map(|slice| {
                    (
                        slice.content_start_sample,
                        slice.content_end_sample,
                        slice.duration_samples(),
                    )
                })
                .collect::<Vec<_>>(),
        );

        let max_allowed_samples =
            ((resolution.options.chunk_seconds + resolution.options.padding_seconds * 2.0 + 1.0)
                * SAMPLE_RATE_HZ as f32)
                .ceil() as usize;
        for (index, slice) in plan.slices.iter().enumerate() {
            assert!(
                slice.duration_samples() <= max_allowed_samples,
                "slice {index} is {} samples (>{max_allowed_samples}); granite must not hand the \
                 executor a buffer past the shared window",
                slice.duration_samples()
            );
            assert!(
                slice.content_end_sample > slice.content_start_sample,
                "slice {index} must cover content"
            );
        }
        // Content coverage: first content starts at 0, last content reaches the end.
        assert_eq!(plan.slices.first().unwrap().content_start_sample, 0);
        assert_eq!(
            plan.slices.last().unwrap().content_end_sample,
            total_samples
        );
    }

    #[test]
    fn explicit_native_longform_request_is_preserved() {
        let requested = crate::LongFormOptions {
            mode: LongFormMode::Energy,
            ..crate::LongFormOptions::default()
        };
        let resolution = resolve_native_longform_policy_for_backend(
            Some(&requested),
            10.6,
            "",
            GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(resolution.options.mode, LongFormMode::Energy);
    }

    #[test]
    fn cohere_longform_policy_caps_default_chunk_sizes() {
        let resolution = resolve_native_longform_policy_for_backend(
            None,
            120.0,
            crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            GgmlCpuGraphBackend::Metal,
        );
        assert_eq!(resolution.options.mode, LongFormMode::Auto);
        assert_eq!(
            resolution.options.chunk_seconds,
            CONSERVATIVE_SEQ2SEQ_LONGFORM_MAX_CHUNK_SECONDS
        );
        assert_eq!(
            resolution.options.max_chunk_seconds,
            CONSERVATIVE_SEQ2SEQ_LONGFORM_MAX_CHUNK_SECONDS
        );
        assert_eq!(resolution.options.min_chunk_seconds, 1.0);
        assert_eq!(
            resolution.options.overlap_seconds,
            CONSERVATIVE_SEQ2SEQ_LONGFORM_OVERLAP_SECONDS
        );
        assert!(resolution.provenance.iter().any(|entry| {
            entry.contains("core.native.longform.policy:conservative-seq2seq-chunk-cap=")
        }));
        assert!(resolution.provenance.iter().any(|entry| {
            entry.contains("core.native.longform.policy:conservative-seq2seq-overlap=")
        }));
        assert!(resolution.provenance.iter().any(|entry| {
            entry.contains("core.native.longform.policy:conservative-seq2seq-disable-prompt-carry")
        }));
    }

    #[test]
    fn cohere_longform_policy_clamps_explicit_large_chunk_request() {
        let requested = crate::LongFormOptions {
            mode: LongFormMode::Fixed,
            chunk_seconds: 45.0,
            max_chunk_seconds: 90.0,
            min_chunk_seconds: 30.0,
            overlap_seconds: 20.0,
            ..crate::LongFormOptions::default()
        };
        let resolution = resolve_native_longform_policy_for_backend(
            Some(&requested),
            120.0,
            crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(
            resolution.options.chunk_seconds,
            CONSERVATIVE_SEQ2SEQ_LONGFORM_MAX_CHUNK_SECONDS
        );
        assert_eq!(
            resolution.options.max_chunk_seconds,
            CONSERVATIVE_SEQ2SEQ_LONGFORM_MAX_CHUNK_SECONDS
        );
        assert_eq!(
            resolution.options.min_chunk_seconds,
            CONSERVATIVE_SEQ2SEQ_LONGFORM_MAX_CHUNK_SECONDS
        );
        assert_eq!(
            resolution.options.overlap_seconds,
            CONSERVATIVE_SEQ2SEQ_LONGFORM_OVERLAP_SECONDS
        );
        assert!(!resolution.options.carry_prompt_across_slices);
    }

    #[test]
    fn qwen_metal_longform_policy_keeps_default_chunk_size() {
        // qwen has no `ConservativeSeq2SeqV1` decode-side profile, so
        // `chunk_seconds` (already 30.0 by default) is untouched. But qwen's
        // audio encoder IS `GlobalQuadratic` (issue #68), so the much larger
        // `max_chunk_seconds` default (60.0) -- the true ceiling the VAD/
        // energy/auto slicer can grow a chunk to on long, pause-free audio --
        // must still be capped down to the 30s safe ceiling.
        let resolution = resolve_native_longform_policy_for_backend(
            None,
            120.0,
            crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            GgmlCpuGraphBackend::Metal,
        );
        assert_eq!(resolution.options.chunk_seconds, 30.0);
        assert_eq!(resolution.options.max_chunk_seconds, 30.0);
        assert!(resolution.provenance.iter().any(|entry| {
            entry.contains("core.native.longform.policy:encoder-attention-span-chunk-cap=30")
        }));
    }

    /// Production-path regression test for the issue #68 wiring bug: the real
    /// call site (`run_native_transcription`) resolves the longform safety
    /// cap from the `GgmlFamilyAdapterDescriptor` the same way
    /// `validate_runtime_source_and_select_adapter` builds it, and MUST key
    /// off `model_architecture` -- never `adapter_id`. The two are different
    /// strings for every builtin family (asserted below), so passing the
    /// wrong one makes `resolve_builtin_decode_policy_for_architecture` and
    /// `OpenAsrArchitectureRegistry::find_by_model_architecture` both miss,
    /// silently dropping every family-specific longform safety cap -- which
    /// is exactly how firered-aed/cohere/moonshine's `ConservativeSeq2SeqV1`
    /// cap and every `GlobalQuadratic` family's encoder-memory cap went live
    /// but never actually applied in production (chunk length stayed at the
    /// unsafe 120s default) until this fix.
    #[test]
    fn native_longform_policy_uses_selected_family_model_architecture_not_adapter_id() {
        let selected_family = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID)
            .expect("firered-aed architecture")
            .ggml_family_adapter_descriptor();
        assert_ne!(
            selected_family.adapter_id,
            selected_family.model_architecture
        );

        // Correct wiring: keying off model_architecture applies BOTH the
        // encoder-attention-span cap and the conservative seq2seq cap --
        // both now resolve to the same default (30s), so composing them
        // (taking the min) is a no-op, but both must still actually run.
        let correct = resolve_native_longform_policy_for_backend(
            None,
            120.0,
            selected_family.model_architecture,
            GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(
            correct.options.max_chunk_seconds,
            CONSERVATIVE_SEQ2SEQ_LONGFORM_MAX_CHUNK_SECONDS
        );
        assert!(correct.options.max_chunk_seconds < 120.0);

        // The bug class this guards against: keying off adapter_id finds no
        // matching architecture, so family safety policy silently no-ops. The
        // product-wide default is now 60s (not the old 120s), but it must still
        // be distinguishable from FireRed-AED's stricter family ceiling.
        let wrong = resolve_native_longform_policy_for_backend(
            None,
            120.0,
            selected_family.adapter_id,
            GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(
            wrong.options.max_chunk_seconds,
            crate::LongFormOptions::default().max_chunk_seconds
        );
        assert!(
            wrong.options.max_chunk_seconds > correct.options.max_chunk_seconds,
            "the wrong identity must demonstrably miss the stricter family cap"
        );
        assert!(wrong.provenance.is_empty());
    }

    /// The encoder memory ceiling has to be able to say something the default
    /// chunk length does not, and the only way to prove that is to give it a
    /// ceiling the two do not share.
    ///
    /// Under the old arrangement they were one symbol, so the clamp had no
    /// independent content: its `chunk_seconds` arm could not fire (the value
    /// under test *was* the ceiling), and the arm that did fire flattened the
    /// slicer's 30-120s search band onto the default. Both arms are asserted,
    /// with the old shared value restated as a local literal rather than
    /// imported -- reading a production constant here would let a later edit
    /// quietly turn this into a comparison of a number with itself.
    #[test]
    fn the_encoder_memory_ceiling_clamps_to_itself_not_to_the_default_chunk_length() {
        /// What both roles held when they were one symbol.
        const OLD_SHARED_VALUE: f32 = 30.0;
        let defaults = crate::LongFormOptions::default();

        // A ceiling above the target but below the product-wide maximum keeps
        // a non-collapsed band for cutting on a real pause.
        let mut roomy = defaults.clone();
        assert!(clamp_longform_chunks_to_encoder_memory_ceiling(
            &mut roomy, 45.0
        ));
        assert_eq!(roomy.chunk_seconds, defaults.chunk_seconds);
        assert_eq!(roomy.max_chunk_seconds, 45.0);
        assert_eq!(roomy.min_chunk_seconds, defaults.min_chunk_seconds);

        // Counterfactual: with the ceiling pinned to the default chunk length,
        // the band collapses onto it and `chunk_seconds` is never touched --
        // the clamp reports "capped" without any memory claim behind it.
        let mut shared = defaults.clone();
        assert!(clamp_longform_chunks_to_encoder_memory_ceiling(
            &mut shared,
            OLD_SHARED_VALUE
        ));
        assert_eq!(shared.chunk_seconds, defaults.chunk_seconds);
        assert_eq!(shared.max_chunk_seconds, OLD_SHARED_VALUE);

        // A host that can afford less does reach the arm the shared value made
        // unreachable.
        let mut tight = defaults.clone();
        assert!(clamp_longform_chunks_to_encoder_memory_ceiling(
            &mut tight, 12.0
        ));
        assert_eq!(tight.chunk_seconds, 12.0);
        assert_eq!(tight.max_chunk_seconds, 12.0);
        assert!(tight.chunk_seconds < defaults.chunk_seconds);
    }

    /// Data-driven production-path coverage over every builtin architecture
    /// (issue #68): a `GlobalQuadratic` encoder must never be handed a
    /// longform chunk longer than its declared safe ceiling, while
    /// encoder-memory cap. An independent semantic invocation span may still
    /// narrow the window (notably Whisper's 30s frontend). All nine
    /// `GlobalQuadratic` builtins (including firered-aed/cohere-transcribe/
    /// moonshine, which also carry the decode-side `ConservativeSeq2SeqV1`
    /// cap) declare `DEFAULT_ENCODER_SAFE_CHUNK_SECONDS`, so this asserts
    /// exact equality, not just an upper bound: the two caps stacked on the
    /// conservative-seq2seq trio must resolve to the same 30s default, not
    /// silently over-tighten to something smaller than either cap alone
    /// intends.
    #[test]
    fn encoder_attention_span_caps_every_builtin_architecture_on_the_production_path() {
        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            // Long enough to be past every family's integral window, so the
            // slicing policy actually runs for `ScopedSlices` families too --
            // a shorter recording legitimately resolves to longform off for
            // them, which would say nothing about the encoder caps under test.
            let resolution = resolve_native_longform_policy_for_backend(
                None,
                600.0,
                descriptor.identity.model_architecture,
                GgmlCpuGraphBackend::Cpu,
            );
            match descriptor.longform_max_safe_chunk_seconds() {
                Some(max_safe_chunk_seconds) => {
                    assert_eq!(
                        max_safe_chunk_seconds, DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
                        "'{}' GlobalQuadratic ceiling must be the shared default absent a cited \
                         upstream override",
                        descriptor.identity.model_architecture
                    );
                    assert_eq!(
                        resolution.options.max_chunk_seconds,
                        max_safe_chunk_seconds,
                        "'{}' must resolve max_chunk_seconds to exactly {max_safe_chunk_seconds}, got {}",
                        descriptor.identity.model_architecture,
                        resolution.options.max_chunk_seconds
                    );
                    assert!(
                        resolution.options.chunk_seconds <= max_safe_chunk_seconds,
                        "'{}' must cap chunk_seconds to <= {max_safe_chunk_seconds}, got {}",
                        descriptor.identity.model_architecture,
                        resolution.options.chunk_seconds
                    );
                }
                None => {
                    // No encoder-memory cap applies. The product slice shape
                    // supplies the base window and a semantic invocation span
                    // may independently narrow it.
                    let product_window = match descriptor.execution_contract.longform_slice_shape {
                        crate::arch::OpenAsrLongformSliceShape::ScopedSlices {
                            max_seconds,
                            ..
                        } => max_seconds,
                        crate::arch::OpenAsrLongformSliceShape::SharedWindow => {
                            crate::arch::DEFAULT_ENCODER_MAX_CHUNK_SECONDS
                        }
                    };
                    let expected = descriptor
                        .max_single_invocation_seconds()
                        .map_or(product_window, |semantic_max| {
                            product_window.min(semantic_max)
                        });
                    assert_eq!(
                        resolution.options.max_chunk_seconds, expected,
                        "'{}' must keep the min(product window, semantic invocation span)",
                        descriptor.identity.model_architecture
                    );
                }
            }
        }
    }

    #[test]
    fn longform_prompt_carry_mode_uses_whisper_token_history() {
        let options = crate::LongFormOptions::default();
        assert_eq!(
            longform_prompt_carry_mode(&options, crate::WHISPER_GGML_ARCHITECTURE_ID),
            LongformPromptCarryMode::TokenHistory,
        );
        assert_eq!(
            longform_prompt_carry_mode(&options, crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID),
            LongformPromptCarryMode::TokenHistory,
        );
        assert_eq!(
            longform_prompt_carry_mode(&options, crate::QWEN3_ASR_GGML_ARCHITECTURE_ID),
            LongformPromptCarryMode::Text,
        );
    }

    #[test]
    fn longform_prompt_carry_mode_stays_disabled_when_option_is_off() {
        let options = crate::LongFormOptions {
            carry_prompt_across_slices: false,
            ..crate::LongFormOptions::default()
        };
        assert_eq!(
            longform_prompt_carry_mode(&options, crate::WHISPER_GGML_ARCHITECTURE_ID),
            LongformPromptCarryMode::Disabled,
        );
        assert_eq!(
            longform_prompt_carry_mode(&options, crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID),
            LongformPromptCarryMode::Disabled,
        );
        assert_eq!(
            longform_prompt_carry_mode(&options, crate::QWEN3_ASR_GGML_ARCHITECTURE_ID),
            LongformPromptCarryMode::Disabled,
        );

        let disabled_mode = crate::LongFormOptions {
            mode: LongFormMode::Off,
            carry_prompt_across_slices: true,
            ..crate::LongFormOptions::default()
        };
        assert_eq!(
            longform_prompt_carry_mode(&disabled_mode, crate::WHISPER_GGML_ARCHITECTURE_ID,),
            LongformPromptCarryMode::Disabled,
        );
    }

    #[test]
    fn execution_longform_is_present_for_implicit_long_audio() {
        let resolution =
            resolve_native_longform_policy_for_backend(None, 120.0, "", GgmlCpuGraphBackend::Cpu);
        assert_eq!(resolution.options.mode, LongFormMode::Auto);
    }

    #[test]
    fn execution_longform_is_absent_for_short_audio() {
        let resolution =
            resolve_native_longform_policy_for_backend(None, 10.6, "", GgmlCpuGraphBackend::Cpu);
        assert!(matches!(resolution.options.mode, LongFormMode::Off));
    }

    #[test]
    fn native_dispatch_is_reused_within_one_service_root() {
        let services = native_execution_services_for_test();
        let first = services.offline_dispatch() as *const _;
        let second = services.offline_dispatch() as *const _;
        assert_eq!(first, second);
    }

    #[test]
    fn normalize_synthesizes_single_segment_when_model_returns_none() {
        let transcription = normalize_transcription_segments(
            Transcription {
                truncated_decodes: Vec::new(),
                unnamed_speakers: Vec::new(),
                text: "hello world".to_string(),
                segments: Vec::new(),
                longform: None,
                language: None,
                ..Default::default()
            },
            0.0,
            2.0,
        );
        assert_eq!(transcription.segments.len(), 1);
        assert_eq!(transcription.segments[0].start, 0.0);
        assert_eq!(transcription.segments[0].end, 2.0);
        assert_eq!(transcription.segments[0].text, "hello world");
    }

    #[test]
    fn normalize_keeps_segment_timestamps_monotonic() {
        let transcription = normalize_transcription_segments(
            Transcription {
                truncated_decodes: Vec::new(),
                unnamed_speakers: Vec::new(),
                text: "a b".to_string(),
                segments: vec![
                    Segment {
                        start: 0.8,
                        end: 1.0,
                        text: "a".to_string(),
                        speaker: None,
                        speaker_label: None,
                        speaker_person_id: None,
                        speaker_snapshot_label: None,
                        words: Vec::new(),
                    },
                    Segment {
                        start: 0.5,
                        end: 0.7,
                        text: "b".to_string(),
                        speaker: None,
                        speaker_label: None,
                        speaker_person_id: None,
                        speaker_snapshot_label: None,
                        words: Vec::new(),
                    },
                ],
                longform: None,
                language: None,
                ..Default::default()
            },
            0.0,
            2.0,
        );
        assert_eq!(transcription.segments.len(), 2);
        assert!(transcription.segments[1].start >= transcription.segments[0].end);
        assert!(transcription.segments[1].end >= transcription.segments[1].start);
    }

    #[test]
    fn normalize_expands_single_short_segment_to_audio_duration() {
        let transcription = normalize_transcription_segments(
            Transcription {
                truncated_decodes: Vec::new(),
                unnamed_speakers: Vec::new(),
                text: "long transcript".to_string(),
                segments: vec![Segment {
                    start: 0.0,
                    end: 1.0,
                    text: "long transcript".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_person_id: None,
                    speaker_snapshot_label: None,
                    words: Vec::new(),
                }],
                longform: None,
                language: None,
                ..Default::default()
            },
            0.0,
            120.0,
        );
        assert_eq!(transcription.segments.len(), 1);
        assert_eq!(transcription.segments[0].end, 120.0);
    }

    #[test]
    fn normalize_keeps_single_segment_when_end_is_already_near_duration() {
        let transcription = normalize_transcription_segments(
            Transcription {
                truncated_decodes: Vec::new(),
                unnamed_speakers: Vec::new(),
                text: "near full".to_string(),
                segments: vec![Segment {
                    start: 0.0,
                    end: 11.5,
                    text: "near full".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_person_id: None,
                    speaker_snapshot_label: None,
                    words: Vec::new(),
                }],
                longform: None,
                language: None,
                ..Default::default()
            },
            0.0,
            12.0,
        );
        assert_eq!(transcription.segments.len(), 1);
        assert_eq!(transcription.segments[0].end, 11.5);
    }

    /// Real-recording regression for diarization attribution granularity: the
    /// X-ASR batch path emits one monolithic transcript segment, which used to
    /// collapse a 2-speaker recording into a single SPEAKER_xx segment. The
    /// recording is the user speaking at both ends (~1.4-3.5s and ~16.0-17.8s)
    /// with a video playing in the middle (~5.8-13.9s), so verbose_json must
    /// show >=3 segments with >=2 distinct speakers in an A/B/A bookend shape.
    #[test]
    #[ignore = "host-local: requires the X-ASR q8_0 pack, the redimnet diarize pack, and tmp/diar-real-case-1781172161.wav"]
    fn real_recording_diarization_splits_monolithic_segment_into_speaker_turns() {
        let pack = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/xasr-test/out/xasr-zh-en-onnx-q8_0.oasr");
        let wav =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/diar-real-case-1781172161.wav");
        if !pack.exists() || !wav.exists() {
            eprintln!(
                "skipping: pack ({}) or wav ({}) absent",
                pack.display(),
                wav.display()
            );
            return;
        }
        if !crate::diarize::vad_diarization_available() {
            eprintln!("skipping: speaker-embedder diarize pack not installed");
            return;
        }
        let pack = pack.canonicalize().expect("pack path must canonicalize");
        let request = TranscriptionRequest::new(
            wav.canonicalize().expect("wav path must canonicalize"),
            "xasr-zh-en",
        )
        .with_model_pack_path(Some(pack))
        .with_voice_id(true);
        let transcription = run_native_transcription(request, native_execution_services_for_test())
            .expect("diarized transcription must succeed");

        let rendered = crate::format::render_transcription(
            &transcription,
            crate::format::ResponseFormat::VerboseJson,
        )
        .expect("verbose_json must render");
        let parsed: serde_json::Value =
            serde_json::from_str(&rendered).expect("verbose_json must parse");
        let segments = parsed["segments"]
            .as_array()
            .expect("segments array")
            .clone();
        assert!(
            segments.len() >= 3,
            "user/video/user bookends must yield >=3 segments, got {segments:?}"
        );

        let speakers: Vec<&str> = segments
            .iter()
            .map(|segment| segment["speaker"].as_str().expect("every segment labeled"))
            .collect();
        let distinct: std::collections::BTreeSet<&str> = speakers.iter().copied().collect();
        assert!(
            distinct.len() >= 2,
            "expected >=2 distinct speakers, got {speakers:?}"
        );

        // Bookend shape: the first and last segments are the same (user)
        // speaker, and the middle (video) speaker is someone else.
        let first = *speakers.first().expect("first segment");
        let last = *speakers.last().expect("last segment");
        assert_eq!(
            first, last,
            "the user's bookend speech must share one speaker, got {speakers:?}"
        );
        assert!(
            speakers.iter().any(|speaker| *speaker != first),
            "the video middle must be a different speaker, got {speakers:?}"
        );

        // Segments must stay ordered with no time travel and no overlap: a
        // glued punctuation word emitted late into the inter-turn gap must not
        // drag one piece's end past the next piece's start.
        let mut previous_start = f64::MIN;
        let mut previous_end = f64::MIN;
        for segment in &segments {
            let start = segment["start"].as_f64().expect("start");
            let end = segment["end"].as_f64().expect("end");
            assert!(start >= previous_start, "segments must stay ordered");
            assert!(end >= start);
            assert!(
                start >= previous_end,
                "split segments must not overlap: previous end {previous_end} > start {start}"
            );
            previous_start = start;
            previous_end = end;
        }

        // Word timestamps were forced internally for the split; the request
        // did not ask for them, so they must not leak into the output.
        for segment in &segments {
            assert!(
                segment.get("words").is_none(),
                "forced word timestamps must be stripped: {segment}"
            );
        }
    }

    // --- long-form VAD provider resolution (Stream-VAD is the sole engine) ---

    #[test]
    fn resolve_longform_vad_provider_always_resolves_stream_vad() {
        let options = crate::LongFormOptions::default();
        let (_, label) = resolve_longform_vad_provider(
            &options,
            GgmlCpuGraphBackend::Cpu,
            crate::device::execution_policy::ExecutionPlacement::CpuOnly,
        )
        .expect("Stream-VAD must resolve in tests");
        assert_eq!(label, "firered-stream-cpu");
    }

    // --- real-audio long-form slicing smoke test ---

    fn jfk_wav_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav")
    }

    fn zh_wav_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/zh_sample.wav")
    }

    fn assert_stream_vad_slices_real_audio_without_panicking(wav_path: std::path::PathBuf) {
        let samples = load_wav_16khz_mono_f32_v0(
            &wav_path,
            "longform VAD smoke test",
            "longform VAD smoke test",
        )
        .expect("load wav fixture");

        let mut options = crate::LongFormOptions {
            mode: LongFormMode::Vad,
            ..crate::LongFormOptions::default()
        };
        // Keep the fixture (11-20s) comfortably above the min chunk size so
        // `Vad` mode actually exercises slicing rather than the `total <=
        // chunk_samples` single-slice shortcut.
        options.chunk_seconds = 2.0;
        let (provider, label) = resolve_longform_vad_provider(
            &options,
            GgmlCpuGraphBackend::Cpu,
            crate::device::execution_policy::ExecutionPlacement::CpuOnly,
        )
        .expect("Stream-VAD's vendored weights must load in tests");
        assert_eq!(
            label, "firered-stream-cpu",
            "Stream-VAD's vendored weights must load in tests"
        );

        let plan = plan_longform_slices(&samples, 16_000, &options, Some(provider.as_ref()))
            .unwrap_or_else(|error| panic!("{label} produced an invalid slice plan: {error}"));
        assert!(
            !plan.slices.is_empty(),
            "{label} must produce at least one slice for {}",
            wav_path.display()
        );
        for slice in &plan.slices {
            assert!(slice.end_sample > slice.start_sample);
            assert!(slice.end_sample <= plan.total_samples);
        }
    }

    #[test]
    fn stream_vad_slices_real_jfk_audio_without_panicking() {
        assert_stream_vad_slices_real_audio_without_panicking(jfk_wav_path());
    }

    #[test]
    fn stream_vad_slices_real_zh_audio_without_panicking() {
        assert_stream_vad_slices_real_audio_without_panicking(zh_wav_path());
    }

    fn segment(start: f32, end: f32, text: &str) -> Segment {
        Segment {
            start,
            end,
            text: text.to_string(),
            speaker: None,
            speaker_label: None,
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words: vec![WordTimestamp {
                word: text.to_string(),
                start,
                end,
                confidence: Some(0.9),
            }],
        }
    }

    /// A single decode unit stays one scope, so its source's own numbering is
    /// authoritative and nothing gets renumbered.
    #[test]
    fn absent_scope_provenance_is_one_scope() {
        let mut segments = vec![segment(0.0, 1.0, "a"), segment(1.0, 2.0, "b")];
        let scopes = speaker_scopes_by_provenance(&mut segments, &[], &[]).unwrap();
        assert_eq!(scopes.len(), 1);
        assert_eq!(scopes[0].segments.len(), 2);

        let mut segments = vec![segment(0.0, 1.0, "a")];
        let scopes = speaker_scopes_by_provenance(&mut segments, &[Some(0)], &[]).unwrap();
        assert_eq!(scopes.len(), 1);
    }

    /// Each slice's segments land in that slice's scope, and every scope is a
    /// contiguous run so no segment can be assigned to two scopes.
    #[test]
    fn segments_are_cut_into_the_scope_that_decoded_them() {
        let mut segments = vec![
            segment(0.0, 10.0, "a"),
            segment(10.0, 20.0, "b"),
            segment(180.5, 190.0, "c"),
            segment(360.0, 370.0, "d"),
        ];
        let scopes =
            speaker_scopes_by_provenance(&mut segments, &[Some(0), Some(0), Some(1), Some(2)], &[])
                .unwrap();
        let sizes: Vec<usize> = scopes.iter().map(|scope| scope.segments.len()).collect();
        assert_eq!(sizes, vec![2, 1, 1]);
    }

    /// A segment retained from the earlier owner of an overlap can have a
    /// midpoint after the next slice started. Exact decode provenance keeps it
    /// in the earlier label namespace instead of guessing from that midpoint.
    #[test]
    fn overlapping_slice_midpoints_do_not_change_scope_provenance() {
        let mut segments = vec![
            segment(29.6, 29.9, "owned by the first slice"),
            segment(30.1, 30.4, "owned by the second slice"),
        ];
        let scopes = speaker_scopes_by_provenance(&mut segments, &[Some(0), Some(1)], &[]).unwrap();
        let sizes: Vec<usize> = scopes.iter().map(|scope| scope.segments.len()).collect();
        assert_eq!(sizes, vec![1, 1]);
        assert_eq!(scopes[0].segments[0].text, "owned by the first slice");
    }

    #[test]
    fn invalid_scope_provenance_fails_closed() {
        let mut missing = vec![segment(0.0, 1.0, "a")];
        assert!(
            speaker_scopes_by_provenance(&mut missing, &[None], &[]).is_err(),
            "a local speaker label without a decode scope must not be merged"
        );

        let mut backwards = vec![segment(0.0, 1.0, "a"), segment(1.0, 2.0, "b")];
        assert!(
            speaker_scopes_by_provenance(&mut backwards, &[Some(2), Some(1)], &[]).is_err(),
            "scope provenance must remain ordered with assembled segments"
        );
    }

    fn item(text: &str, start_time_s: f64, end_time_s: f64) -> ForcedAlignItem {
        ForcedAlignItem {
            text: text.to_string(),
            start_time_s,
            end_time_s,
        }
    }

    #[test]
    fn assign_aligned_words_replaces_words_within_one_segment() {
        let mut segments = vec![segment(0.0, 2.0, "hello world")];
        let items = vec![item("hello", 0.1, 0.4), item("world", 0.5, 0.9)];

        assign_aligned_words_to_segments(&mut segments, &items);

        let words = &segments[0].words;
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[0].start, 0.1);
        assert_eq!(words[0].end, 0.4);
        assert_eq!(words[0].confidence, None);
        assert_eq!(words[1].word, "world");
    }

    #[test]
    fn assign_aligned_words_distributes_across_segments_by_start_time() {
        let mut segments = vec![segment(0.0, 1.0, "hi"), segment(1.0, 2.0, "there")];
        let items = vec![item("hi", 0.1, 0.5), item("there", 1.2, 1.6)];

        assign_aligned_words_to_segments(&mut segments, &items);

        assert_eq!(segments[0].words.len(), 1);
        assert_eq!(segments[0].words[0].word, "hi");
        assert_eq!(segments[1].words.len(), 1);
        assert_eq!(segments[1].words[0].word, "there");
    }

    #[test]
    fn assign_aligned_words_leaves_segments_untouched_when_items_empty() {
        let mut segments = vec![segment(0.0, 1.0, "hi")];
        let original_words = segments[0].words.clone();

        assign_aligned_words_to_segments(&mut segments, &[]);

        assert_eq!(segments[0].words, original_words);
    }

    #[test]
    fn forced_alignment_uses_each_decode_segments_bounded_pcm_view() {
        let first = segment(0.0, 30.0, "first");
        let second = segment(30.0, 59.71, "second");
        let audio_samples = 955_360;

        assert_eq!(
            forced_alignment_segment_sample_range(&first, audio_samples),
            Some(0..480_000)
        );
        assert_eq!(
            forced_alignment_segment_sample_range(&second, audio_samples),
            Some(480_000..955_360)
        );
    }

    #[test]
    fn local_forced_alignment_is_mapped_back_to_the_recording_clock() {
        let mut target = segment(30.0, 32.0, "hello world");
        let items = vec![item("hello", 0.1, 0.4), item("world", 0.5, 2.4)];

        assign_local_aligned_words(&mut target, &items);

        assert_eq!(target.words.len(), 2);
        assert_eq!(target.words[0].start, 30.1);
        assert_eq!(target.words[0].end, 30.4);
        assert_eq!(target.words[1].start, 30.5);
        assert_eq!(target.words[1].end, 32.0);
    }

    #[test]
    fn should_run_punctuation_stage_requires_both_opt_in_and_unpunctuated_capability() {
        // The stage only runs when the request has not opted out AND the
        // model's capability is honestly `Some(false)` -- an unknown or
        // already-punctuated model is never re-punctuated, and an explicit
        // opt-out wins even for an unpunctuated model.
        assert!(should_run_punctuation_stage(true, Some(false)));
        assert!(!should_run_punctuation_stage(false, Some(false)));
        assert!(!should_run_punctuation_stage(true, Some(true)));
        assert!(!should_run_punctuation_stage(true, None));
    }

    #[test]
    fn punctuation_capability_is_derived_from_selected_architecture() {
        // Dolphin's cn-dialect training corpus is honestly unpunctuated.
        assert_eq!(
            emits_punctuation_for_model_architecture(crate::arch::DOLPHIN_GGML_ARCHITECTURE_ID),
            Some(false)
        );
        assert_eq!(
            emits_punctuation_for_model_architecture(crate::arch::WHISPER_GGML_ARCHITECTURE_ID),
            Some(true)
        );
        assert_eq!(
            emits_punctuation_for_model_architecture("unknown-architecture"),
            None
        );
    }

    #[test]
    fn apply_punctuation_stage_leaves_transcription_unchanged_when_stage_does_not_run() {
        // An unknown selected architecture (`None`) means the stage never runs,
        // regardless of the FireRedPunc pack's install state on this machine --
        // fail-closed, never fabricated punctuation.
        let transcription = Transcription {
            truncated_decodes: Vec::new(),
            unnamed_speakers: Vec::new(),
            text: "hello world".to_string(),
            segments: vec![Segment {
                start: 0.0,
                end: 1.0,
                text: "hello world".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            }],
            longform: None,
            language: None,
            ..Default::default()
        };
        let unchanged = apply_punctuation_stage_if_applicable(
            transcription.clone(),
            None,
            true,
            GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(unchanged, transcription);

        // Explicit opt-out short-circuits before any pack resolution too.
        let unchanged = apply_punctuation_stage_if_applicable(
            transcription.clone(),
            None,
            false,
            GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(unchanged, transcription);
    }

    fn tiny_whisper_preflight(dir: &Path) -> GgufRuntimeSourcePreflight {
        let pack_path = dir.join("whisper.oasr");
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            crate::arch::GENERAL_ARCHITECTURE_KEY.to_string(),
            crate::arch::WHISPER_GGML_ARCHITECTURE_ID.to_string(),
        );
        crate::testing::write_tiny_gguf_runtime_source(
            &pack_path,
            &crate::testing::TinyGgufFixtureSpec::new(metadata),
        )
        .expect("write tiny whisper fixture");
        let runtime_source = crate::ggml_runtime::validate_ggml_runtime_source_path(&pack_path)
            .expect("validate tiny fixture path");
        load_runtime_source_metadata_and_tensor_index_from_source(&runtime_source)
            .expect("load tiny fixture preflight")
    }

    fn execution_policy_test_fixture(
        dir: &Path,
        executor: std::sync::Arc<dyn GgmlAsrViewExecutor>,
    ) -> (
        GgmlAsrExecutionDispatch,
        VerifiedPack,
        GgmlFamilyAdapterDescriptor,
    ) {
        let preflight = tiny_whisper_preflight(dir);
        let verified_pack = VerifiedPack::from_unverified_preflight_for_test(
            preflight,
            crate::arch::WHISPER_GGML_ARCHITECTURE_ID,
        );
        let dispatch = GgmlAsrExecutionDispatch::default()
            .with_view_executor_for_adapter(crate::WHISPER_GGML_ADAPTER_ID, executor);
        (
            dispatch,
            verified_pack,
            crate::arch::builtin_adapter_descriptor(crate::arch::WHISPER_GGML_ARCHITECTURE_ID),
        )
    }

    struct TypedCandidateFailureStubExecutor {
        calls: Mutex<Vec<GgmlAsrBackendPreference>>,
        record_typed_failure: bool,
    }

    impl TypedCandidateFailureStubExecutor {
        fn new(record_typed_failure: bool) -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                record_typed_failure,
            }
        }

        fn calls(&self) -> Vec<GgmlAsrBackendPreference> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl GgmlAsrViewExecutor for TypedCandidateFailureStubExecutor {
        fn executor_id(&self) -> &'static str {
            "typed-candidate-failure-stub"
        }

        fn supports_phrase_bias(&self) -> bool {
            true
        }

        fn evict_prepared_runtime_content_id(&self, _pack_content_id: &str) {}

        fn decoder_state_contract(
            &self,
            _selected_family: &crate::GgmlFamilyAdapterDescriptor,
        ) -> Result<
            crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract,
            GgmlAsrExecutionError,
        > {
            Ok(crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract::NoPersistentState)
        }

        fn execute_view(
            &self,
            request: &GgmlAsrExecutionViewRequest<'_>,
        ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
            self.calls.lock().unwrap().push(request.backend_preference);
            if request.backend_preference == GgmlAsrBackendPreference::CpuOnly {
                return Ok(GgmlAsrExecutionResult {
                    transcription: Transcription {
                        truncated_decodes: Vec::new(),
                        unnamed_speakers: Vec::new(),
                        text: "cpu-success".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                        ..Default::default()
                    },
                    carry_context: None,
                    decode_truncation: None,
                });
            }
            if self.record_typed_failure {
                crate::models::native_execution_services::record_current_execution_candidate_failure(
                    ExecutionCandidateFailure::capacity(
                        "stub-allocation",
                        "typed capacity fact independent of the returned error text",
                    ),
                );
            }
            Err(GgmlAsrExecutionError::ExecutorFailed {
                executor_id: self.executor_id(),
                adapter_id: request.selected_family.adapter_id,
                reason: "opaque failure text with no allocation marker".to_string(),
            })
        }
    }

    fn policy_test_candidate(
        provider: crate::ExecutionProvider,
        stable_id: &str,
        placement: ExecutionPlacement,
    ) -> ExecutionCandidate {
        let kind = if placement == ExecutionPlacement::CpuOnly {
            crate::RouteDeviceKind::Cpu
        } else {
            crate::RouteDeviceKind::Accelerated
        };
        ExecutionCandidate {
            device: crate::device::execution_policy::ExecutionDeviceSnapshot {
                route: crate::ResolvedExecutionRoute {
                    provider,
                    stable_id: stable_id.to_string(),
                    registry_ordinal: 0,
                    kind,
                    addressability: crate::DeviceAddressability::NotExactlyAddressable {
                        reason: "test candidate",
                    },
                },
                ggml_kind: if placement == ExecutionPlacement::CpuOnly {
                    crate::GgmlBackendKind::Cpu
                } else {
                    crate::GgmlBackendKind::Gpu
                },
                memory: None,
                buffer_alignment: None,
            },
            placement,
        }
    }

    fn typed_fallback_test_plan() -> ExecutionPlan {
        ExecutionPlan::for_test(
            ExecutionIntent::Auto,
            vec![
                policy_test_candidate(
                    crate::ExecutionProvider::Vulkan,
                    "VulkanTest0",
                    ExecutionPlacement::Hybrid,
                ),
                policy_test_candidate(
                    crate::ExecutionProvider::Cpu,
                    "CPU",
                    ExecutionPlacement::CpuOnly,
                ),
            ],
        )
    }

    fn optional_punctuation_test_transcription() -> Transcription {
        Transcription {
            truncated_decodes: Vec::new(),
            unnamed_speakers: Vec::new(),
            text: "raw transcript".to_string(),
            segments: Vec::new(),
            longform: None,
            language: None,
            ..Default::default()
        }
    }

    #[test]
    fn optional_punctuation_preserves_asr_after_typed_candidates_are_exhausted() {
        let original = optional_punctuation_test_transcription();
        let error = PolicyResolvedAuxRuntimeError::CandidatesExhausted {
            stage: "firered-punctuation",
            failure: ExecutionCandidateFailure::capacity("test-punctuation", "full"),
            source: None,
        };

        let resolved = finish_optional_punctuation_stage(original.clone(), Err(error))
            .expect("optional punctuation exhaustion must preserve ASR");

        assert_eq!(resolved, original);
    }

    #[test]
    fn optional_punctuation_does_not_hide_empty_plan_invariant() {
        let error = PolicyResolvedAuxRuntimeError::EmptyPlan {
            stage: "firered-punctuation",
        };

        let result = finish_optional_punctuation_stage(
            optional_punctuation_test_transcription(),
            Err(error),
        );

        assert!(matches!(result, Err(BackendError::NativeFailClosed { .. })));
    }

    #[test]
    fn optional_punctuation_never_swallows_typed_cancellation() {
        let error = PolicyResolvedAuxRuntimeError::Operation(BackendError::TranscriptionCanceled);

        let result = finish_optional_punctuation_stage(
            optional_punctuation_test_transcription(),
            Err(error),
        );

        assert!(matches!(result, Err(BackendError::TranscriptionCanceled)));
    }

    #[test]
    #[ignore = "host-local: needs OPENASR_FIRERED_PUNC_PACK and OPENASR_AUX_BENCH_TEXT"]
    fn firered_punctuation_actor_fifteen_minute_and_cancel_endurance() {
        crate::testing::external_test_fixture_path(
            "OPENASR_FIRERED_PUNC_PACK",
            "FireRedPunc actor endurance pack",
        )
        .expect("OPENASR_FIRERED_PUNC_PACK");
        let text_path = crate::testing::external_test_fixture_path(
            "OPENASR_AUX_BENCH_TEXT",
            "private auxiliary-model benchmark transcript",
        )
        .expect("OPENASR_AUX_BENCH_TEXT");
        let segment_text = std::fs::read_to_string(text_path).expect("read benchmark transcript");
        let segment_text = segment_text.trim().to_string();
        assert!(!segment_text.is_empty());

        let make_transcription = || {
            let segments = (0..60)
                .map(|index| {
                    segment(
                        index as f32 * 15.0,
                        (index + 1) as f32 * 15.0,
                        &segment_text,
                    )
                })
                .collect::<Vec<_>>();
            Transcription {
                text: segments
                    .iter()
                    .map(|segment| segment.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
                segments,
                ..Default::default()
            }
        };
        let progress_for = || {
            ProgressReporter::install(
                None,
                ProgressPlan::build(ProgressPlanInput {
                    audio_duration_s: 900.0,
                    voice_id: false,
                    external_diarize: false,
                    segmenter: ProgressSegmenterKind::Auto,
                    punctuate: true,
                    align: false,
                    backend: ProgressBackendClass::AutoOrCpu,
                    persist: false,
                }),
            )
        };
        let services = native_execution_services_for_test();
        let backend_label =
            std::env::var("OPENASR_AUX_BENCH_BACKEND").unwrap_or_else(|_| "cpu".to_string());
        let request_intent = execution_intent_from_backend_env(Some(&backend_label))
            .expect("OPENASR_AUX_BENCH_BACKEND must name a supported backend");
        let detached =
            crate::RequestExecutionContext::uncancellable("FireRedPunc host-local endurance run");
        let progress = progress_for();
        let execution_placement = crate::GgmlExecutionTelemetryCollector::new();
        let _execution_placement_guard = execution_placement.install();
        let started = Instant::now();
        let punctuated = apply_punctuation_stage_with_policy(
            make_transcription(),
            Some(false),
            true,
            services.as_ref(),
            &request_intent,
            &detached,
            &progress,
        )
        .expect("punctuate fifteen-minute-equivalent transcript");
        let elapsed_seconds = started.elapsed().as_secs_f64();
        let observed = execution_placement.snapshot();
        let output_sha256 = crate::testing::benchmark_sha256_bytes(
            punctuated
                .segments
                .iter()
                .map(|segment| segment.text.as_bytes()),
        );
        let memory = crate::metrics::process_memory_snapshot();
        let peak_rss_bytes = memory.peak_rss_bytes.unwrap_or(0);
        let current_rss_bytes = memory.current_rss_bytes.unwrap_or(0);
        let phys_footprint_bytes = memory.current_phys_footprint_bytes.unwrap_or(0);
        let peak_phys_footprint_bytes = memory.peak_phys_footprint_bytes.unwrap_or(0);
        eprintln!(
            "AUX_MODEL_ENDURANCE model=fireredpunc backend={backend_label} represented_audio_seconds=900.000000 elapsed_seconds={elapsed_seconds:.6} peak_rss_bytes={peak_rss_bytes} current_rss_bytes={current_rss_bytes} phys_footprint_bytes={phys_footprint_bytes} peak_phys_footprint_bytes={peak_phys_footprint_bytes} observed_compute_nodes={:?} segments={} output_sha256={output_sha256}",
            observed.observed_compute_nodes_by_backend,
            punctuated.segments.len(),
        );
        assert_eq!(punctuated.segments.len(), 60);
        if backend_label.eq_ignore_ascii_case("metal") {
            assert!(
                !observed.observed_compute_nodes_by_backend.is_empty()
                    && observed
                        .observed_compute_nodes_by_backend
                        .keys()
                        .all(|backend| {
                            let backend = backend.to_ascii_lowercase();
                            backend.starts_with("mtl") || backend.contains("metal")
                        }),
                "explicit Metal FireRedPunc product route observed non-Metal compute: {:?}",
                observed.observed_compute_nodes_by_backend
            );
        }

        // Reuse the now-warm actor and cancel while its first long segment is
        // in flight. The actor republishes this request's ggml cancel flag on
        // its owner thread; the optional-stage policy must preserve the typed
        // terminal status instead of silently returning raw ASR text.
        let control = Arc::new(crate::api::backend::TranscriptionControl::new());
        let execution_context = Arc::new(crate::RequestExecutionContext::new(
            None,
            Arc::clone(&control),
        ));
        let cancel_services = Arc::clone(&services);
        let cancel_context = Arc::clone(&execution_context);
        let cancel_input = make_transcription();
        let (result_tx, result_rx) = mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let _abort_guard = cancel_context
                .control
                .arm_for_native_decode_if_cancellable();
            let progress = progress_for();
            let result = apply_punctuation_stage_with_policy(
                cancel_input,
                Some(false),
                true,
                cancel_services.as_ref(),
                &ExecutionIntent::CpuOnly,
                cancel_context.as_ref(),
                &progress,
            );
            let _ = result_tx.send(result);
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        control.request_cancel();
        let error = result_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("canceled punctuation actor must not hang")
            .expect_err("canceled punctuation actor must fail closed");
        assert!(matches!(error, BackendError::TranscriptionCanceled));
    }

    #[test]
    fn forced_alignment_graph_abort_maps_to_typed_cancellation() {
        let control = Arc::new(crate::api::backend::TranscriptionControl::new());
        let execution_context = crate::RequestExecutionContext::new(None, Arc::clone(&control));
        control.request_cancel();

        assert!(matches!(
            forced_alignment_error_to_backend(
                &execution_context,
                "segment 2: arbitrary graph failure".to_string(),
            ),
            BackendError::TranscriptionCanceled
        ));

        let detached =
            crate::RequestExecutionContext::uncancellable("cooperative-cancel reason mapping test");
        assert!(matches!(
            forced_alignment_error_to_backend(
                &detached,
                "segment 2: ggml graph compute aborted by cancel request".to_string(),
            ),
            BackendError::TranscriptionCanceled
        ));
        assert!(matches!(
            forced_alignment_error_to_backend(
                &detached,
                "segment 2: malformed timestamp head".to_string(),
            ),
            BackendError::WordTimestampAlignmentFailed { .. }
        ));
    }

    #[test]
    fn every_required_auxiliary_stage_fails_closed_after_typed_exhaustion() {
        for stage in [
            "qwen3-forced-aligner",
            "speaker-attribution",
            "longform-vad",
            "speaker-identity",
        ] {
            let calls = std::sync::atomic::AtomicUsize::new(0);
            let error = run_auxiliary_stage_with_policy(
                native_execution_services_for_test().as_ref(),
                &typed_fallback_test_plan(),
                stage,
                |_| {
                    calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    crate::models::native_execution_services::record_current_execution_candidate_failure(
                        ExecutionCandidateFailure::capacity("test-required-aux", "full"),
                    );
                    Ok::<(), BackendError>(())
                },
            )
            .expect_err("required auxiliary stage must retain typed exhaustion");
            assert!(matches!(
                error,
                PolicyResolvedAuxRuntimeError::CandidatesExhausted { .. }
            ));

            let error = required_auxiliary_stage_error(error);
            let BackendError::NativeFailClosed { reason } = error else {
                panic!("{stage} typed exhaustion must fail closed");
            };
            assert!(reason.contains(stage), "{stage}: {reason}");
            assert_eq!(
                calls.load(std::sync::atomic::Ordering::SeqCst),
                2,
                "{stage} must exhaust both approved candidates before failing"
            );
        }
    }

    #[test]
    fn typed_candidate_failure_retries_without_parsing_error_text() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(TypedCandidateFailureStubExecutor::new(true));
        let (dispatch, verified_pack, family) =
            execution_policy_test_fixture(dir.path(), executor.clone());
        let services = native_execution_services_for_test();
        let progress =
            DecodeProgress::begin(ProgressReporter::install(None, test_plan(false)), 160);
        let (result, fallback) = run_dispatch_once_with_progress_and_policy(
            &dispatch,
            &services,
            &verified_pack,
            &family,
            vec![0.0; 160].into(),
            GgmlAsrExecutionOptions::default(),
            &typed_fallback_test_plan(),
            crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            &uncancellable_execution_context_for_test(),
            &progress,
            160,
            "typed-fallback-test",
        )
        .expect("typed capacity failure should advance to CPU under Auto");
        assert_eq!(result.transcription.text, "cpu-success");
        assert!(fallback.is_some());
        assert_eq!(
            executor.calls(),
            vec![
                GgmlAsrBackendPreference::Accelerated,
                GgmlAsrBackendPreference::CpuOnly,
            ]
        );
    }

    #[test]
    fn identical_error_without_typed_failure_never_retries() {
        let dir = tempfile::tempdir().unwrap();
        let executor = Arc::new(TypedCandidateFailureStubExecutor::new(false));
        let (dispatch, verified_pack, family) =
            execution_policy_test_fixture(dir.path(), executor.clone());
        let services = native_execution_services_for_test();
        let progress =
            DecodeProgress::begin(ProgressReporter::install(None, test_plan(false)), 160);
        let error = run_dispatch_once_with_progress_and_policy(
            &dispatch,
            &services,
            &verified_pack,
            &family,
            vec![0.0; 160].into(),
            GgmlAsrExecutionOptions::default(),
            &typed_fallback_test_plan(),
            crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            &uncancellable_execution_context_for_test(),
            &progress,
            160,
            "untyped-failure-test",
        )
        .expect_err("ordinary executor failure must fail closed on its candidate");
        assert!(error.to_string().contains("opaque failure text"));
        assert_eq!(
            executor.calls(),
            vec![GgmlAsrBackendPreference::Accelerated]
        );
    }

    // ---- P1 concurrent slice pipeline ----

    #[test]
    fn capacity_gate_caps_by_slice_count_and_never_returns_zero() {
        // Plenty of memory: width is bounded only by the slice count.
        assert_eq!(
            slice_pipeline_capped_width(4, 2, Some(64 << 30), 1 << 20, 0),
            2,
            "cannot run more workers than there are slices"
        );
        // Tight memory: only one worker fits, and the gate floors at 1 (serial),
        // never 0 -- it can only reduce concurrency, so it cannot OOM.
        assert_eq!(
            slice_pipeline_capped_width(4, 8, Some(1_200 << 20), 1 << 30, 512 << 20),
            1,
            "one worker's worth of head-room -> serial, never zero"
        );
        // Enough memory for exactly three workers.
        assert_eq!(
            slice_pipeline_capped_width(4, 8, Some((3 << 30) + (512 << 20)), 1 << 30, 512 << 20),
            3
        );
        // No memory probe: honor the explicit opt-in (serve-batch precedent).
        assert_eq!(
            slice_pipeline_capped_width(3, 8, None, 1 << 30, 512 << 20),
            3
        );
        // A zero per-worker estimate cannot divide; fall back to the ceiling.
        assert_eq!(slice_pipeline_capped_width(3, 8, Some(1 << 30), 0, 0), 3);
        // A width-1 request is always serial regardless of memory.
        assert_eq!(
            slice_pipeline_capped_width(1, 8, Some(64 << 30), 1 << 20, 0),
            1
        );
    }

    #[test]
    fn automatic_slice_concurrency_is_limited_to_discrete_gpu_providers() {
        for provider in [
            crate::ExecutionProvider::Cuda,
            crate::ExecutionProvider::Hip,
            crate::ExecutionProvider::Vulkan,
        ] {
            assert_eq!(slice_pipeline_default_provider_width(4, provider), 4);
        }
        for provider in [
            crate::ExecutionProvider::Cpu,
            crate::ExecutionProvider::Metal,
            crate::ExecutionProvider::Accelerator,
            crate::ExecutionProvider::Unknown,
        ] {
            assert_eq!(slice_pipeline_default_provider_width(4, provider), 1);
        }
    }

    #[test]
    fn requested_width_default_is_gated_on_the_run_carry_state() {
        // SAFETY: nextest runs each test in its own process, so mutating this
        // process-global env var cannot race another test.
        unsafe {
            std::env::remove_var("OPENASR_SLICE_PIPELINE_WIDTH");
        }
        // Carry disabled: concurrent is transcript-equivalent, so the default
        // requests the maximum and lets the capacity gate pick K.
        assert_eq!(
            slice_pipeline_requested_width(LongformPromptCarryMode::Disabled),
            SLICE_PIPELINE_MAX_WIDTH,
            "carry-disabled run defaults to the concurrent pipeline"
        );
        // ... which still flows through the capacity gate: plenty of memory
        // admits the full width, tight memory caps it back to serial.
        assert_eq!(
            slice_pipeline_capped_width(
                slice_pipeline_requested_width(LongformPromptCarryMode::Disabled),
                8,
                Some(64 << 30),
                1 << 20,
                0,
            ),
            SLICE_PIPELINE_MAX_WIDTH,
        );
        assert_eq!(
            slice_pipeline_capped_width(
                slice_pipeline_requested_width(LongformPromptCarryMode::Disabled),
                8,
                Some(1_200 << 20),
                1 << 30,
                512 << 20,
            ),
            1,
        );
        // Carry active: the concurrent path would drop the carry, so the
        // default stays on the byte-identical serial + prompt-carry path.
        assert_eq!(
            slice_pipeline_requested_width(LongformPromptCarryMode::Text),
            1,
            "text-carry run defaults to serial"
        );
        assert_eq!(
            slice_pipeline_requested_width(LongformPromptCarryMode::TokenHistory),
            1,
            "token-history-carry run defaults to serial"
        );
    }

    #[test]
    fn requested_width_env_overrides_both_directions_and_clamps() {
        // Explicit widths override the carry-gated default in both directions:
        // ">=2" forces the carry-light concurrent path onto a carry-active
        // run, and "0"/"1" pin a carry-disabled run to serial.
        for (value, expected) in [("0", 1), ("1", 1), ("2", 2), ("4", 4), ("9", 4)] {
            // SAFETY: nextest runs each test in its own process, so mutating
            // this process-global env var cannot race another test.
            unsafe {
                std::env::set_var("OPENASR_SLICE_PIPELINE_WIDTH", value);
            }
            for carry_mode in [
                LongformPromptCarryMode::Disabled,
                LongformPromptCarryMode::Text,
                LongformPromptCarryMode::TokenHistory,
            ] {
                assert_eq!(
                    slice_pipeline_requested_width(carry_mode),
                    expected,
                    "OPENASR_SLICE_PIPELINE_WIDTH={value} carry={carry_mode:?}"
                );
            }
        }
        // An unparseable value is not an explicit choice: fall back to the
        // carry-gated default rather than guessing a width.
        unsafe {
            std::env::set_var("OPENASR_SLICE_PIPELINE_WIDTH", "junk");
        }
        assert_eq!(
            slice_pipeline_requested_width(LongformPromptCarryMode::Disabled),
            SLICE_PIPELINE_MAX_WIDTH,
        );
        assert_eq!(
            slice_pipeline_requested_width(LongformPromptCarryMode::TokenHistory),
            1,
        );
        unsafe {
            std::env::remove_var("OPENASR_SLICE_PIPELINE_WIDTH");
        }
    }

    /// Deterministic executor for the concurrent-pipeline tests: echoes the
    /// slice's audio marker (the constant its region is filled with, see
    /// [`concurrent_pipeline_slices`]) back as its transcript text, so a test
    /// can prove each slice's result is paired with the right slice and
    /// assembled in slice order. Fails on a configured set of markers to
    /// exercise error routing.
    struct ConcurrentPipelineStubExecutor {
        fail_markers: std::collections::BTreeSet<i32>,
        observed_views: Option<Arc<Mutex<Vec<(usize, std::ops::Range<usize>)>>>>,
    }

    impl ConcurrentPipelineStubExecutor {
        fn echoing() -> Self {
            Self {
                fail_markers: std::collections::BTreeSet::new(),
                observed_views: None,
            }
        }

        fn failing_on(markers: &[i32]) -> Self {
            Self {
                fail_markers: markers.iter().copied().collect(),
                observed_views: None,
            }
        }

        fn recording_views(
            observed_views: Arc<Mutex<Vec<(usize, std::ops::Range<usize>)>>>,
        ) -> Self {
            Self {
                fail_markers: std::collections::BTreeSet::new(),
                observed_views: Some(observed_views),
            }
        }

        fn marker_of(request: &GgmlAsrExecutionViewRequest) -> i32 {
            request
                .prepared_audio
                .samples_f32
                .first()
                .copied()
                .unwrap_or(0.0)
                .round() as i32
        }
    }

    impl GgmlAsrViewExecutor for ConcurrentPipelineStubExecutor {
        fn executor_id(&self) -> &'static str {
            "concurrent-pipeline-stub"
        }

        fn supports_phrase_bias(&self) -> bool {
            true
        }

        fn evict_prepared_runtime_content_id(&self, _pack_content_id: &str) {}

        fn decoder_state_contract(
            &self,
            _selected_family: &crate::GgmlFamilyAdapterDescriptor,
        ) -> Result<
            crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract,
            GgmlAsrExecutionError,
        > {
            Ok(crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract::NoPersistentState)
        }

        fn execute_view(
            &self,
            request: &GgmlAsrExecutionViewRequest,
        ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
            if let Some(observed) = self.observed_views.as_ref() {
                observed.lock().unwrap().push((
                    request.prepared_audio.samples_f32.backing_identity(),
                    request.prepared_audio.samples_f32.range(),
                ));
            }
            let marker = Self::marker_of(request);
            if self.fail_markers.contains(&marker) {
                return Err(GgmlAsrExecutionError::ExecutorFailed {
                    executor_id: "concurrent-pipeline-stub",
                    adapter_id: request.selected_family.adapter_id,
                    reason: format!("stub failure marker={marker}"),
                });
            }
            Ok(GgmlAsrExecutionResult {
                transcription: Transcription {
                    truncated_decodes: Vec::new(),
                    unnamed_speakers: Vec::new(),
                    text: format!("w{marker}"),
                    segments: Vec::new(),
                    longform: None,
                    language: None,
                    ..Default::default()
                },
                carry_context: None,
                decode_truncation: None,
            })
        }
    }

    /// `count` back-to-back 1000-sample slices; slice `i`'s audio region is
    /// filled with the constant `(i + 1)` so the stub echoes a per-slice marker.
    fn concurrent_pipeline_slices(count: usize) -> (Vec<f32>, Vec<crate::longform::AudioSlice>) {
        let slice_len = 1000usize;
        let mut audio = vec![0.0f32; count * slice_len];
        let mut slices = Vec::with_capacity(count);
        for i in 0..count {
            let start = i * slice_len;
            let end = start + slice_len;
            for sample in &mut audio[start..end] {
                *sample = (i + 1) as f32;
            }
            slices.push(crate::longform::AudioSlice {
                index: i,
                kind: AudioSliceKind::Fixed,
                start_sample: start,
                end_sample: end,
                content_start_sample: start,
                content_end_sample: end,
            });
        }
        (audio, slices)
    }

    #[derive(Debug)]
    struct ConcurrentPipelineOutcome {
        assembled: Transcription,
        ran_any_slice: bool,
        suppressed: usize,
    }

    #[allow(clippy::too_many_arguments)]
    fn run_concurrent_pipeline_for_test(
        width: usize,
        audio: &[f32],
        slices: Vec<crate::longform::AudioSlice>,
        executor: Arc<dyn GgmlAsrViewExecutor>,
        execution_context: &Arc<crate::RequestExecutionContext>,
        longform_options: &crate::LongFormOptions,
        progress_id: Option<String>,
    ) -> Result<ConcurrentPipelineOutcome, BackendError> {
        let audio = PcmBuffer::from_vec(audio.to_vec());
        let dir = tempfile::tempdir().unwrap();
        let (dispatch, verified_pack, family) = execution_policy_test_fixture(dir.path(), executor);
        let timeline = crate::longform::TimelineMap::identity();
        let mut assembler =
            TranscriptAssembler::new(timeline.clone(), SegmentMergePolicy::default());
        let total: u64 = slices.iter().map(|s| s.duration_samples() as u64).sum();
        let reporter = ProgressReporter::install(progress_id.clone(), test_plan(false));
        let decode_progress = DecodeProgress::begin(reporter, total);
        let request_options = GgmlAsrExecutionOptions::default();
        let mut ran_any_slice = false;
        let mut suppressed = 0usize;
        let mut degraded = Vec::new();
        let mut truncated_slices = Vec::new();
        let mut truncated_decodes = Vec::new();
        let mut speaker_scope_count = 0usize;
        let execution_services = native_execution_services_for_test();
        let execution_plan = resolve_native_execution_plan(
            execution_services.as_ref(),
            &family,
            ExecutionIntent::CpuOnly,
        )?;
        let auto_gpu_policy =
            crate::arch::family_auto_gpu_policy_for_model_architecture(family.model_architecture);
        run_concurrent_slice_pipeline(ConcurrentSlicePipeline {
            width,
            slices,
            plan_audio: &audio,
            dispatch: &dispatch,
            execution_services: &execution_services,
            verified_pack: &verified_pack,
            selected_family: &family,
            request_options: &request_options,
            execution_plan: &execution_plan,
            auto_gpu_policy,
            execution_context,
            longform_options,
            speaker_plan: SpeakerPlan::Off,
            decode_progress: &decode_progress,
            assembler: &mut assembler,
            ran_any_slice: &mut ran_any_slice,
            suppressed_slice_count: &mut suppressed,
            degraded_slice_fallbacks: &mut degraded,
            truncated_slices: &mut truncated_slices,
            truncated_decodes: &mut truncated_decodes,
            speaker_scope_count: &mut speaker_scope_count,
        })?;
        let (assembled, _stats) = assembler.into_parts();
        Ok(ConcurrentPipelineOutcome {
            assembled,
            ran_any_slice,
            suppressed,
        })
    }

    #[test]
    fn concurrent_pipeline_assembles_slices_in_order_and_reaches_progress_ceiling() {
        let _serial = progress_registry_test_lock();
        clear_progress_registry_for_test();
        let id = "concurrent-pipeline-ordered";
        let _handle = ProgressRegistryHandle::new(Some(id.to_string()));
        let (audio, slices) = concurrent_pipeline_slices(6);
        let outcome = run_concurrent_pipeline_for_test(
            4,
            &audio,
            slices,
            Arc::new(ConcurrentPipelineStubExecutor::echoing()),
            &uncancellable_execution_context_for_test(),
            &crate::LongFormOptions::default(),
            Some(id.to_string()),
        )
        .expect("all slices decode successfully");
        // Out-of-order worker completion, but each result is paired with its own
        // slice and integrated in slice order (property 1).
        assert_eq!(outcome.assembled.text, "w1 w2 w3 w4 w5 w6");
        assert!(outcome.ran_any_slice);
        assert_eq!(outcome.suppressed, 0);
        // Progress accumulated atomically across workers; stage_fraction of
        // decode reaches 1.0 (registry clamp keeps overall monotonic).
        let progress = native_transcription_progress_for_id(id).expect("run published progress");
        assert!(
            (progress.stage_fraction.unwrap_or(0.0) - 1.0).abs() < 1e-5,
            "decode stage_fraction should reach 1.0, got {:?}",
            progress.stage_fraction
        );
    }

    #[test]
    fn concurrent_pipeline_dispatches_range_views_from_one_pcm_backing() {
        let (audio, slices) = concurrent_pipeline_slices(6);
        let expected_ranges: Vec<_> = slices
            .iter()
            .map(|slice| slice.start_sample..slice.end_sample)
            .collect();
        let observed = Arc::new(Mutex::new(Vec::new()));
        run_concurrent_pipeline_for_test(
            4,
            &audio,
            slices,
            Arc::new(ConcurrentPipelineStubExecutor::recording_views(Arc::clone(
                &observed,
            ))),
            &uncancellable_execution_context_for_test(),
            &crate::LongFormOptions::default(),
            None,
        )
        .expect("all slices decode successfully");

        let mut observed = observed.lock().unwrap().clone();
        let identity = observed
            .first()
            .expect("every decoded slice records a view")
            .0;
        assert!(observed.iter().all(|(candidate, _)| *candidate == identity));
        observed.sort_by_key(|(_, range)| range.start);
        assert_eq!(
            observed
                .into_iter()
                .map(|(_, range)| range)
                .collect::<Vec<_>>(),
            expected_ranges
        );
    }

    #[test]
    fn concurrent_pipeline_returns_the_lowest_index_worker_error_and_fails_closed() {
        let (audio, slices) = concurrent_pipeline_slices(6);
        // Slices with markers 2 and 4 fail; the lowest-index failure (marker 2,
        // the second slice) is the one surfaced, matching the serial `?`.
        let error = run_concurrent_pipeline_for_test(
            4,
            &audio,
            slices,
            Arc::new(ConcurrentPipelineStubExecutor::failing_on(&[2, 4])),
            &uncancellable_execution_context_for_test(),
            &crate::LongFormOptions::default(),
            None,
        )
        .expect_err("a worker failure must fail the whole run closed");
        assert!(
            error.to_string().contains("marker=2"),
            "the earliest (lowest-index) slice error must be returned: {error}"
        );
    }

    #[test]
    fn concurrent_pipeline_surfaces_cancel_without_decoding() {
        let control = Arc::new(crate::api::backend::TranscriptionControl::new());
        control.request_cancel();
        let execution_context = Arc::new(crate::RequestExecutionContext::new(
            None,
            Arc::clone(&control),
        ));
        let (audio, slices) = concurrent_pipeline_slices(6);
        let error = run_concurrent_pipeline_for_test(
            4,
            &audio,
            slices,
            Arc::new(ConcurrentPipelineStubExecutor::echoing()),
            &execution_context,
            &crate::LongFormOptions::default(),
            None,
        )
        .expect_err("a pre-canceled run must stop at the slice-boundary gate");
        assert!(
            matches!(error, BackendError::TranscriptionCanceled),
            "cancel must surface as the typed TranscriptionCanceled: {error}"
        );
    }

    /// Deterministic executor that echoes each slice's marker as BOTH its text
    /// (`w{marker}`) and a single segment at a fixed slice-relative time, so a
    /// test proves the concurrent path's ordered *segment* assembly and
    /// per-slice time-domain remap -- not just the flat text -- matches the
    /// serial path. Like [`ConcurrentPipelineStubExecutor`] it reads nothing
    /// but the audio marker, so it is completely insensitive to the request
    /// prompt / cross-slice carry: the ONLY variable it can react to is which
    /// slice it was handed.
    struct SegmentEchoStubExecutor;

    impl GgmlAsrViewExecutor for SegmentEchoStubExecutor {
        fn executor_id(&self) -> &'static str {
            "segment-echo-stub"
        }

        fn supports_phrase_bias(&self) -> bool {
            true
        }

        fn evict_prepared_runtime_content_id(&self, _pack_content_id: &str) {}

        fn decoder_state_contract(
            &self,
            _selected_family: &crate::GgmlFamilyAdapterDescriptor,
        ) -> Result<
            crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract,
            GgmlAsrExecutionError,
        > {
            Ok(crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract::NoPersistentState)
        }

        fn execute_view(
            &self,
            request: &GgmlAsrExecutionViewRequest,
        ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
            let marker = ConcurrentPipelineStubExecutor::marker_of(request);
            Ok(GgmlAsrExecutionResult {
                transcription: Transcription {
                    truncated_decodes: Vec::new(),
                    unnamed_speakers: Vec::new(),
                    text: format!("w{marker}"),
                    segments: vec![segment(0.10, 0.20, &format!("w{marker}"))],
                    longform: None,
                    language: None,
                    ..Default::default()
                },
                carry_context: None,
                decode_truncation: None,
            })
        }
    }

    /// Concurrency-vs-serial equivalence with a carry-insensitive deterministic
    /// backend (supplement 1, mock tier): running the SAME slices through the
    /// real assembly code path at width 1 (a single worker pulling slices in
    /// order == the serial reference) and at widths 2/3/4 (workers finishing
    /// out of order) must produce a BYTE-IDENTICAL assembled transcription --
    /// text AND segments AND their remapped timings. Because the stub reads
    /// only the audio marker and ignores the prompt/carry entirely, the sole
    /// difference between the width-1 and width-N runs is concurrency itself,
    /// so equality isolates and proves that concurrency alone does not change
    /// the output (the carry variable that separates the production serial and
    /// carry-light paths is held constant here at "no carry").
    #[test]
    fn concurrent_pipeline_output_is_byte_identical_across_widths() {
        let (audio, slices) = concurrent_pipeline_slices(7);
        let run = |width: usize| {
            run_concurrent_pipeline_for_test(
                width,
                &audio,
                slices.clone(),
                Arc::new(SegmentEchoStubExecutor),
                &uncancellable_execution_context_for_test(),
                &crate::LongFormOptions::default(),
                None,
            )
            .expect("all slices decode")
        };

        // Width 1 == single worker, strictly serial slice order: the reference.
        let serial = run(1);
        assert_eq!(
            serial.assembled.text, "w1 w2 w3 w4 w5 w6 w7",
            "serial (width=1) reference text"
        );
        assert_eq!(
            serial.assembled.segments.len(),
            7,
            "one segment per decoded slice survives assembly"
        );
        assert!(serial.ran_any_slice);
        assert_eq!(serial.suppressed, 0);

        for width in [2usize, 3, 4] {
            let concurrent = run(width);
            assert_eq!(
                concurrent.assembled, serial.assembled,
                "width={width} concurrent output must be byte-identical to the \
                 serial (width=1) reference: text, segments, and remapped timings"
            );
            assert_eq!(concurrent.suppressed, serial.suppressed);
            assert_eq!(concurrent.ran_any_slice, serial.ran_any_slice);
        }
    }

    /// Same equivalence, but with a suppressed silent slice in the middle: the
    /// concurrent path decides silence once up front on the main thread and
    /// leaves that position empty, then integrates in slice order. Width 1 and
    /// width 4 must agree byte-for-byte on both the assembled transcript and
    /// the suppressed-slice count, proving the concurrent silence bookkeeping
    /// matches the serial loop's.
    #[test]
    fn concurrent_pipeline_silent_slice_handling_matches_across_widths() {
        let (mut audio, slices) = concurrent_pipeline_slices(6);
        // Zero slice index 2's audio region so it reads as silence (marker 0),
        // while every other slice keeps its distinct non-zero marker.
        for sample in &mut audio[2 * 1000..3 * 1000] {
            *sample = 0.0;
        }
        let longform = crate::LongFormOptions {
            suppress_silent_slices: true,
            ..crate::LongFormOptions::default()
        };
        let run = |width: usize| {
            run_concurrent_pipeline_for_test(
                width,
                &audio,
                slices.clone(),
                Arc::new(SegmentEchoStubExecutor),
                &uncancellable_execution_context_for_test(),
                &longform,
                None,
            )
            .expect("non-silent slices decode")
        };

        let serial = run(1);
        // Slice index 2 is suppressed; the rest echo their markers in order.
        assert_eq!(serial.assembled.text, "w1 w2 w4 w5 w6");
        assert_eq!(serial.suppressed, 1);

        let concurrent = run(4);
        assert_eq!(
            concurrent.assembled, serial.assembled,
            "concurrent silent-slice suppression must be byte-identical to serial"
        );
        assert_eq!(concurrent.suppressed, serial.suppressed);
    }

    /// Shared handshake between a blocking test executor and the test thread:
    /// counts how many decodes have entered `execute` (so the test can wait
    /// until a worker is genuinely mid-decode before flipping a control) and
    /// lets the test release those blocked decodes. Used only to construct
    /// deterministic in-flight timings for the cancel / pause tests.
    struct DecodeGate {
        entered: Mutex<usize>,
        entered_cv: std::sync::Condvar,
        release: Mutex<bool>,
        release_cv: std::sync::Condvar,
    }

    impl DecodeGate {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                entered: Mutex::new(0),
                entered_cv: std::sync::Condvar::new(),
                release: Mutex::new(false),
                release_cv: std::sync::Condvar::new(),
            })
        }

        fn mark_entered(&self) {
            *self.entered.lock().unwrap() += 1;
            self.entered_cv.notify_all();
        }

        fn wait_entered_at_least(&self, count: usize) {
            let mut entered = self.entered.lock().unwrap();
            while *entered < count {
                entered = self.entered_cv.wait(entered).unwrap();
            }
        }

        fn release_all(&self) {
            *self.release.lock().unwrap() = true;
            self.release_cv.notify_all();
        }

        fn wait_for_release(&self) {
            let mut released = self.release.lock().unwrap();
            while !*released {
                released = self.release_cv.wait(released).unwrap();
            }
        }
    }

    /// Executor that parks inside `execute` (a worker genuinely mid-decode)
    /// until the test releases it, then echoes the slice marker. Lets the
    /// pause/resume test place a worker in-flight before pausing.
    struct PauseGateExecutor {
        gate: Arc<DecodeGate>,
    }

    impl GgmlAsrViewExecutor for PauseGateExecutor {
        fn executor_id(&self) -> &'static str {
            "pause-gate-stub"
        }

        fn supports_phrase_bias(&self) -> bool {
            true
        }

        fn evict_prepared_runtime_content_id(&self, _pack_content_id: &str) {}

        fn decoder_state_contract(
            &self,
            _selected_family: &crate::GgmlFamilyAdapterDescriptor,
        ) -> Result<
            crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract,
            GgmlAsrExecutionError,
        > {
            Ok(crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract::NoPersistentState)
        }

        fn execute_view(
            &self,
            request: &GgmlAsrExecutionViewRequest,
        ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
            let marker = ConcurrentPipelineStubExecutor::marker_of(request);
            self.gate.mark_entered();
            self.gate.wait_for_release();
            Ok(GgmlAsrExecutionResult {
                transcription: Transcription {
                    truncated_decodes: Vec::new(),
                    unnamed_speakers: Vec::new(),
                    text: format!("w{marker}"),
                    segments: Vec::new(),
                    longform: None,
                    language: None,
                    ..Default::default()
                },
                carry_context: None,
                decode_truncation: None,
            })
        }
    }

    /// Executor that simulates a real ggml graph observing a mid-compute
    /// cancel: it blocks inside `execute` (past the slice-boundary gate, i.e.
    /// genuinely in-flight) and spins on the per-worker abort flag the pipeline
    /// arms via `arm_for_native_decode`, exactly the flag a real ggml
    /// abort_callback reads. When the cancel trips it returns an aborted error,
    /// as an aborted graph would. A 30s safety valve keeps a regression from
    /// hanging the suite forever.
    struct CancelGateExecutor {
        gate: Arc<DecodeGate>,
    }

    impl GgmlAsrViewExecutor for CancelGateExecutor {
        fn executor_id(&self) -> &'static str {
            "cancel-gate-stub"
        }

        fn supports_phrase_bias(&self) -> bool {
            true
        }

        fn evict_prepared_runtime_content_id(&self, _pack_content_id: &str) {}

        fn decoder_state_contract(
            &self,
            _selected_family: &crate::GgmlFamilyAdapterDescriptor,
        ) -> Result<
            crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract,
            GgmlAsrExecutionError,
        > {
            Ok(crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract::NoPersistentState)
        }

        fn execute_view(
            &self,
            request: &GgmlAsrExecutionViewRequest,
        ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
            self.gate.mark_entered();
            let started = Instant::now();
            loop {
                if crate::ggml_runtime::thread_job_cancel_requested() {
                    return Err(GgmlAsrExecutionError::ExecutorFailed {
                        executor_id: "cancel-gate-stub",
                        adapter_id: request.selected_family.adapter_id,
                        reason: "aborted mid-flight by cancel".to_string(),
                    });
                }
                if started.elapsed() > std::time::Duration::from_secs(30) {
                    return Err(GgmlAsrExecutionError::ExecutorFailed {
                        executor_id: "cancel-gate-stub",
                        adapter_id: request.selected_family.adapter_id,
                        reason: "cancel never observed within 30s (test safety valve)".to_string(),
                    });
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
    }

    /// Run the concurrent pipeline on a scratch thread and hand its result back
    /// through a channel so the caller can bound the wait -- a hang (deadlock,
    /// lost worker, dropped channel) surfaces as a test failure instead of a
    /// frozen suite.
    fn spawn_pipeline_bounded(
        width: usize,
        audio: Vec<f32>,
        slices: Vec<crate::longform::AudioSlice>,
        executor: Arc<dyn GgmlAsrViewExecutor>,
        execution_context: Arc<crate::RequestExecutionContext>,
        longform: crate::LongFormOptions,
    ) -> mpsc::Receiver<Result<ConcurrentPipelineOutcome, BackendError>> {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let outcome = run_concurrent_pipeline_for_test(
                width,
                &audio,
                slices,
                executor,
                &execution_context,
                &longform,
                None,
            );
            // Receiver may already be gone if the test timed out; ignore.
            let _ = tx.send(outcome);
        });
        rx
    }

    /// Supplement 2, mid-flight cancel: a cancel that arrives while workers are
    /// genuinely inside a decode (past the slice-boundary gate) must abort the
    /// in-flight workers promptly, converge every worker, and surface the typed
    /// `TranscriptionCanceled` -- without hanging or panicking a channel. The
    /// existing cancel test only covers a cancel observed *before* any decode
    /// starts (at the boundary gate); this covers the in-flight/abort path.
    #[test]
    fn concurrent_pipeline_mid_flight_cancel_aborts_in_flight_workers() {
        let control = Arc::new(crate::api::backend::TranscriptionControl::new());
        let execution_context = Arc::new(crate::RequestExecutionContext::new(
            None,
            Arc::clone(&control),
        ));
        let gate = DecodeGate::new();
        let (audio, slices) = concurrent_pipeline_slices(4);

        let rx = spawn_pipeline_bounded(
            2,
            audio,
            slices,
            Arc::new(CancelGateExecutor {
                gate: Arc::clone(&gate),
            }),
            Arc::clone(&execution_context),
            crate::LongFormOptions::default(),
        );

        // Wait until at least one worker is genuinely mid-decode, then cancel.
        gate.wait_entered_at_least(1);
        control.request_cancel();

        let outcome = rx
            .recv_timeout(std::time::Duration::from_secs(15))
            .expect("mid-flight cancel must not hang the pipeline");
        let error = outcome.expect_err("a canceled run must fail closed");
        assert!(
            matches!(error, BackendError::TranscriptionCanceled),
            "mid-flight cancel must surface as the typed TranscriptionCanceled: {error}"
        );
    }

    /// Supplement 2, pause/resume: a pause requested while the pipeline is
    /// running must park every worker at a slice boundary (the whole run
    /// suspends, no deadlock and no further slices decoded), and a later resume
    /// must let it run to completion with the correct in-order output. Pause
    /// was previously uncovered.
    #[test]
    fn concurrent_pipeline_pause_parks_workers_then_resume_completes() {
        let control = Arc::new(crate::api::backend::TranscriptionControl::new());
        let execution_context = Arc::new(crate::RequestExecutionContext::new(
            None,
            Arc::clone(&control),
        ));
        let gate = DecodeGate::new();
        let (audio, slices) = concurrent_pipeline_slices(4);

        let rx = spawn_pipeline_bounded(
            2,
            audio,
            slices,
            Arc::new(PauseGateExecutor {
                gate: Arc::clone(&gate),
            }),
            Arc::clone(&execution_context),
            crate::LongFormOptions::default(),
        );

        // A worker is mid-decode of its first slice. Request the pause now, then
        // release the in-flight decode(s): each worker finishes its current
        // slice, loops back to the boundary, and parks on the pending pause
        // instead of pulling the remaining slices.
        gate.wait_entered_at_least(1);
        control.request_pause();
        gate.release_all();

        // The run must NOT complete while paused: with width 2 at most two
        // slices could have been in flight, so slices remain and the workers are
        // parked at the boundary.
        assert!(
            matches!(
                rx.recv_timeout(std::time::Duration::from_millis(300)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ),
            "the pipeline must stay parked while paused, not complete"
        );

        // Resume: parked workers wake, drain the remaining slices, and the run
        // completes with the byte-identical in-order transcript.
        control.request_resume();
        let outcome = rx
            .recv_timeout(std::time::Duration::from_secs(15))
            .expect("resume must let the paused pipeline finish, not hang")
            .expect("a resumed run completes successfully");
        assert_eq!(outcome.assembled.text, "w1 w2 w3 w4");
        assert!(outcome.ran_any_slice);
        assert_eq!(outcome.suppressed, 0);
    }
}
