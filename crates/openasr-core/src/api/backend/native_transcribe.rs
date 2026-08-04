use std::{
    collections::BTreeMap,
    path::Path,
    sync::{
        Arc, Mutex,
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
use crate::models::qwen::{ForcedAlignItem, Qwen3ForcedAlignerSession, forced_aligner_pack};
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
// Phase-aware progress for the in-flight native file transcription, keyed by
// transcription id in a bounded per-request registry. The server's native
// path has no concurrency gate (each request's native transcription runs on
// its own `spawn_blocking` thread; see `routes/transcription.rs`), so more
// than one `run_native_transcription` can be in flight at once -- each one's
// `RequestExecutionContext::request_id` (see that type; every dispatch
// surface already carries one) scopes its progress to its own registry entry
// rather than fighting over one shared slot. A request with no id (a
// detached/uncancellable context: CLI single-shot, an internal caller that
// never registered one) has nowhere honest to publish to and simply does not
// publish -- see `publish_progress` below.
// Progress is a monotonic overall fraction (0..=1) plus a coarse phase label, so
// the UI advances smoothly across decode -> assemble -> forced-align refine
// instead of stalling once the last slice decodes. The old bare slice counter
// reached "done" at the last decode and then sat frozen through assembly/merge
// and the whole-file forced-align pass, which read to users as a bar stuck near
// the end (issue #61). Every `run_dispatch_once` call for every builtin seq2seq
// family -- long-form slices and the short single-pass / single-slice path
// alike -- also reports continuous per-token progress within its own share of
// the decode phase (see `run_dispatch_once_with_progress`, `SliceProgressWindow`),
// closing the gap where short audio used to report nothing at all and fall
// back entirely on a time-based estimate (issue: short-audio progress bar).

/// Bound on the number of transcription ids the progress registry tracks at
/// once. Ordinary operation never approaches this: each id's entry is
/// inserted by `publish_progress`'s first call for that id and removed again
/// by that request's [`ProgressRegistryHandle`] on `Drop` (completion, error,
/// or panic unwind), so the registry only ever holds *currently in-flight*
/// native transcriptions. The bound is a safety net against unbounded growth
/// if that invariant is ever violated (e.g. a future caller that leaks a
/// handle); rather than grow forever, the registry evicts its
/// longest-resident entry to make room -- the one most likely to be a leak,
/// not a genuinely long-running decode that keeps re-publishing (and so keeps
/// getting re-found, not evicted, by the lookup in `publish`).
const PROGRESS_REGISTRY_CAPACITY: usize = 64;

/// Per-id progress storage backing [`native_transcription_progress_for_id`]
/// and the legacy [`native_transcription_progress`]. An insertion-ordered
/// `Vec` rather than a `HashMap`: `PROGRESS_REGISTRY_CAPACITY` keeps this
/// small, a linear scan by id is fast enough at that size, and a `Vec` gives
/// the FIFO eviction order in `publish` for free (index 0 is always the
/// longest-resident surviving entry).
struct ProgressRegistry {
    entries: Vec<(String, NativeTranscriptionProgress)>,
}

impl ProgressRegistry {
    const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn get(&self, id: &str) -> Option<NativeTranscriptionProgress> {
        self.entries
            .iter()
            .find(|(entry_id, _)| entry_id == id)
            .map(|(_, progress)| *progress)
    }

    /// Raise `id`'s stored fraction monotonically (a later phase or a
    /// further-along report never moves the bar backward) and update its
    /// phase. Creates a fresh entry -- starting exactly at `fraction`, never
    /// maxed against anything -- if `id` has none yet, whether because this
    /// is genuinely its first report or because a previous entry under the
    /// same id was already removed (finished) or evicted.
    fn publish(&mut self, id: &str, phase: NativeTranscriptionPhase, fraction: f32) {
        if let Some((_, progress)) = self.entries.iter_mut().find(|(entry_id, _)| entry_id == id) {
            progress.phase = phase;
            progress.fraction = progress.fraction.max(fraction);
            return;
        }
        if self.entries.len() >= PROGRESS_REGISTRY_CAPACITY {
            self.entries.remove(0);
        }
        self.entries.push((
            id.to_string(),
            NativeTranscriptionProgress { phase, fraction },
        ));
    }

    fn remove(&mut self, id: &str) {
        self.entries.retain(|(entry_id, _)| entry_id != id);
    }
}

static PROGRESS_REGISTRY: Mutex<ProgressRegistry> = Mutex::new(ProgressRegistry::new());

// Heuristic phase ceilings the monotonic overall fraction climbs to at each phase
// boundary -- not measured timings. Decode (autoregressive, per-slice) dominates;
// the assembly/merge/resegment tail is short; the forced-align refine is a single
// non-autoregressive forward pass over the whole file, present only when the caller
// opted into word_timestamps=aligned. The monotonic clamp keeps the bar honest even
// when a run's real mix differs from these shares.
const DECODE_CEIL_WITH_ALIGN: f32 = 0.75;
const ASSEMBLE_CEIL_WITH_ALIGN: f32 = 0.80;
const ALIGN_CEIL: f32 = 0.92;
const DECODE_CEIL_NO_ALIGN: f32 = 0.92;
const ASSEMBLE_CEIL_NO_ALIGN: f32 = 0.97;

/// Coarse phase of the in-flight native file transcription.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeTranscriptionPhase {
    /// Decoding audio slices.
    Decode,
    /// Merging slice transcripts and re-segmenting into subtitle cues.
    Assemble,
    /// Refining per-word timestamps with the forced aligner (word_timestamps=aligned).
    Align,
}

impl NativeTranscriptionPhase {
    /// Stable lowercase label for the wire contract and the optional UI phase text.
    pub fn label(self) -> &'static str {
        match self {
            NativeTranscriptionPhase::Decode => "decode",
            NativeTranscriptionPhase::Assemble => "assemble",
            NativeTranscriptionPhase::Align => "align",
        }
    }
}

/// Snapshot of the in-flight native run: a monotonic overall `fraction` in
/// `0..=1` plus the current `phase`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NativeTranscriptionProgress {
    pub phase: NativeTranscriptionPhase,
    pub fraction: f32,
}

/// Progress of the in-flight native transcription with this `id`, or `None`
/// when no such run is currently active (finished, canceled, or never
/// existed). Every decode call -- long-form multi-slice, forced-align
/// refine, and the short single-pass / single-slice path (a "whole file is
/// one slice" `DecodeProgress`, see `run_dispatch_once_with_progress`) --
/// reports through this registry entry. Only a decode that fails before its
/// first report (e.g. model resolution) leaves no signal, and the caller
/// falls back to a time-based estimate for the gap.
pub fn native_transcription_progress_for_id(id: &str) -> Option<NativeTranscriptionProgress> {
    PROGRESS_REGISTRY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(id)
}

/// Legacy (pre-multi-request) id-less progress read, kept for HTTP clients
/// that predate transcription-id-scoped progress. Because the server places
/// no concurrency gate on native transcription, more than one run can be in
/// flight at once; unlike the old single global slot, this says so
/// explicitly rather than silently picking one owner to report as "the"
/// progress -- see `native_transcription_progress_for_id` for the id-scoped
/// read every other caller should prefer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LegacyNativeTranscriptionProgress {
    /// No native transcription is currently in flight.
    Idle,
    /// Exactly one native transcription is in flight: its progress.
    Single(NativeTranscriptionProgress),
    /// More than one native transcription is in flight; there is no honest
    /// single answer for an id-less caller.
    Ambiguous { active_count: usize },
}

pub fn native_transcription_progress() -> LegacyNativeTranscriptionProgress {
    let registry = PROGRESS_REGISTRY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match registry.entries.as_slice() {
        [] => LegacyNativeTranscriptionProgress::Idle,
        [(_, progress)] => LegacyNativeTranscriptionProgress::Single(*progress),
        entries => LegacyNativeTranscriptionProgress::Ambiguous {
            active_count: entries.len(),
        },
    }
}

/// Publish `phase` and raise `id`'s overall fraction monotonically (a later
/// phase or a further-along report never moves that id's bar backward). A
/// no-op for `id: None` -- a detached/uncancellable request has no
/// transcription id to scope its progress to, so it simply never publishes
/// rather than falling back to some shared slot a second, unrelated request
/// could misread as its own.
fn publish_progress(id: Option<&str>, phase: NativeTranscriptionPhase, fraction: f32) {
    let Some(id) = id else {
        return;
    };
    let clamped = fraction.clamp(0.0, 1.0);
    PROGRESS_REGISTRY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .publish(id, phase, clamped);
}

/// Enter the assembly/merge phase, raising `id`'s bar to that phase's ceiling.
fn publish_assemble_progress(id: Option<&str>, with_align: bool) {
    let ceil = if with_align {
        ASSEMBLE_CEIL_WITH_ALIGN
    } else {
        ASSEMBLE_CEIL_NO_ALIGN
    };
    publish_progress(id, NativeTranscriptionPhase::Assemble, ceil);
}

/// Enter the forced-align refine phase, raising `id`'s bar to the align ceiling. The
/// refine is a single opaque forward pass, so the bar holds here (with the "align"
/// phase label explaining the pause) until the run completes and its entry is removed.
fn publish_align_progress(id: Option<&str>) {
    publish_progress(id, NativeTranscriptionPhase::Align, ALIGN_CEIL);
}

/// Decode-phase progress for the multi-slice long-form path. Each slice is weighted
/// by its audio sample count (not a flat per-slice tick) so the bar tracks decode
/// time -- which scales with audio duration -- rather than slice number, which makes
/// variable-length VAD slices advance the bar unevenly.
struct DecodeProgress {
    id: Option<String>,
    total_samples: u64,
    // Atomic so the concurrent slice pipeline (see `run_concurrent_slice_pipeline`)
    // can accumulate completed-slice shares from several worker threads at once
    // without a lock. Reports remain monotonic and race-free: the
    // progress-registry already clamps every published fraction upward
    // (`ProgressRegistry::raise`), so overlapping in-flight windows from
    // concurrent slices can never move the bar backward. Under the serial path
    // (`decoded_samples` touched from one thread only) the observable sequence is
    // byte-identical to the previous plain-`u64` field.
    decoded_samples: AtomicU64,
    decode_ceil: f32,
}

impl DecodeProgress {
    fn begin(id: Option<String>, total_samples: u64, with_align: bool) -> Self {
        let decode_ceil = if with_align {
            DECODE_CEIL_WITH_ALIGN
        } else {
            DECODE_CEIL_NO_ALIGN
        };
        publish_progress(id.as_deref(), NativeTranscriptionPhase::Decode, 0.0);
        Self {
            id,
            total_samples,
            decoded_samples: AtomicU64::new(0),
            decode_ceil,
        }
    }

    /// Mark one slice decoded (or skipped as silent -- silence still consumes its
    /// share of the audio timeline), advancing the bar by that slice's sample share.
    /// `&self` (not `&mut self`) so concurrent slice workers can each fold their
    /// completed slice's share in; `fetch_add` makes the accumulation atomic.
    fn complete_slice(&self, slice_samples: u64) {
        let decoded = self
            .decoded_samples
            .fetch_add(slice_samples, Ordering::Relaxed)
            .saturating_add(slice_samples);
        let ratio = if self.total_samples == 0 {
            1.0
        } else {
            (decoded as f32 / self.total_samples as f32).clamp(0.0, 1.0)
        };
        publish_progress(
            self.id.as_deref(),
            NativeTranscriptionPhase::Decode,
            self.decode_ceil * ratio,
        );
    }

    /// The [start, start+span) sub-range of the overall decode-phase fraction
    /// that the slice about to be decoded (`slice_samples` long, not yet
    /// folded into `decoded_samples`) owns. Per-token progress during that
    /// slice's decode interpolates within this window; `complete_slice`
    /// (called once the slice actually finishes) supersedes it with the
    /// slice's full share regardless of where token interpolation left off.
    fn slice_progress_window(&self, slice_samples: u64) -> SliceProgressWindow {
        let total = (self.total_samples.max(1)) as f32;
        let decoded = self.decoded_samples.load(Ordering::Relaxed);
        let start_ratio = (decoded as f32 / total).clamp(0.0, 1.0);
        let span_ratio = (slice_samples as f32 / total).clamp(0.0, 1.0 - start_ratio);
        SliceProgressWindow {
            start_fraction: self.decode_ceil * start_ratio,
            span_fraction: self.decode_ceil * span_ratio,
        }
    }
}

/// A slice's own sub-range of the overall decode-phase fraction (see
/// `DecodeProgress::slice_progress_window`), token-level interpolation runs.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SliceProgressWindow {
    start_fraction: f32,
    span_fraction: f32,
}

/// Fraction of a decode slice's own progress-bar span (`SliceProgressWindow`)
/// considered "reached" after `step_index` (0-based) of an estimated
/// `estimated_total_tokens` steps. Capped below 1.0 so token-level
/// interpolation never completes a slice's full span before
/// `DecodeProgress::complete_slice` (called once the slice actually
/// finishes decoding) closes it out -- without the cap, a short decode
/// against a generous `max_generated_tokens` budget would already read as
/// "fully decoded" mid-stream, leaving nothing for `complete_slice` to
/// visibly add and reintroducing the old flat-then-jump behavior at a
/// smaller scale.
const TOKEN_PROGRESS_SLICE_SHARE_CAP: f32 = 0.95;

/// Publish at most every Nth generated token (plus always the first) so a
/// very fast decoder does not spend cycles on redundant atomic CAS traffic;
/// the visual granularity given up is well under the frontend's 240ms poll
/// interval, so it is not user-visible.
const TOKEN_PROGRESS_PUBLISH_STRIDE: usize = 4;

/// Pure progress math shared by every token-step sink below: how far through
/// its own window (see `SliceProgressWindow`) a slice's decode should read
/// after generating `step_index + 1` of an estimated `estimated_total_tokens`
/// tokens. `estimated_total_tokens` is deliberately the decode's configured
/// `max_generated_tokens` cap, not a measured or duration-derived estimate:
/// every builtin seq2seq family already picks that cap conservatively for its
/// own architecture (context-window budget, corpus-derived step ceiling,
/// ...), so real decodes almost always finish well under it -- using it as
/// the denominator can only under-promise (fraction climbs slower than real
/// progress), never over-promise (jump past what `complete_slice` will
/// confirm).
fn token_step_fraction(
    window: SliceProgressWindow,
    step_index: usize,
    estimated_total_tokens: usize,
) -> f32 {
    let ratio = if estimated_total_tokens == 0 {
        TOKEN_PROGRESS_SLICE_SHARE_CAP
    } else {
        // step_index is 0-based; +1 so the first generated token already
        // shows forward motion instead of reporting the window's start again.
        let raw = (step_index.saturating_add(1)) as f32 / estimated_total_tokens as f32;
        raw.min(TOKEN_PROGRESS_SLICE_SHARE_CAP)
    };
    window.start_fraction + window.span_fraction * ratio
}

/// Throttle predicate for the token-step sink: true on the first token and
/// every `TOKEN_PROGRESS_PUBLISH_STRIDE`th one after it. A pure function so
/// the stride behavior is unit-testable without a live decode.
fn should_publish_token_step(step_index: usize) -> bool {
    step_index.is_multiple_of(TOKEN_PROGRESS_PUBLISH_STRIDE)
}

/// Run one `run_dispatch_once` call with a per-token progress sink wired to
/// `decode_progress`'s window for `slice_samples`, then close the slice out
/// with `complete_slice` on success. This is the single place that turns
/// per-token decode steps into `publish_progress` calls, so every call site
/// that decodes one slice of audio -- the long-form per-slice loop and the
/// short single-pass / single-slice path, which is `DecodeProgress` for a
/// "whole file is one slice" run -- shares the same continuous signal
/// instead of the short path reporting nothing (see module docs above on why
/// short/single-slice decodes used to never call `publish_progress`).
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
    let id = execution_context.request_id.clone();
    let _token_progress_guard =
        crate::models::seq2seq_greedy_decode::install_token_step_progress_sink(
            move |step_index, max_generated_tokens| {
                if should_publish_token_step(step_index) {
                    publish_progress(
                        id.as_deref(),
                        NativeTranscriptionPhase::Decode,
                        token_step_fraction(window, step_index, max_generated_tokens),
                    );
                }
            },
        );
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
        execution_context,
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
                let error = match result {
                    Err(error) => error,
                    Ok(_) => BackendError::NativeFailClosed {
                        reason: format!(
                            "execution candidate reported {:?} during '{}' despite returning success",
                            failure.kind, failure.operation
                        ),
                    },
                };
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
        std::thread::scope(|scope| {
            for _ in 0..worker_count {
                let result_tx = result_tx.clone();
                scope.spawn(move || {
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
/// `run_native_transcription` so its lifetime spans decode, assembly, and the
/// forced-align refine.
///
/// A request with no transcription id (`id: None` -- a detached/uncancellable
/// context: the client never registered one, or an internal/test caller used
/// `RequestExecutionContext::uncancellable`) never had a registry entry to
/// begin with (`publish_progress` never writes one for a `None` id), so
/// `Drop` here is a no-op for those requests.
struct ProgressRegistryHandle {
    id: Option<String>,
}

impl ProgressRegistryHandle {
    fn new(id: Option<String>) -> Self {
        Self { id }
    }
}

impl Drop for ProgressRegistryHandle {
    fn drop(&mut self) {
        if let Some(id) = &self.id {
            PROGRESS_REGISTRY
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(id);
        }
    }
}

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
}

struct SpeakerFinalizationContext {
    attribution: SpeakerAttribution,
    embedder: Option<Arc<dyn crate::diarize::embed::SpeakerEmbedder>>,
    plan: SpeakerPlan,
    scope_by_segment: Vec<Option<usize>>,
    strip_forced_word_timestamps: bool,
}

impl SpeakerFinalizationContext {
    fn requires_word_alignment(&self, transcription: &Transcription) -> bool {
        self.plan == SpeakerPlan::External
            && crate::diarize::attribution::requires_word_alignment(
                &self.attribution.timeline.turns,
                &transcription.segments,
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
    // Spans the whole run (decode + assembly inside impl, then the punctuation
    // and forced-align post-processes below) so this request's progress-registry
    // entry is removed on every exit and the align phase advances the same
    // monotonic bar rather than running uncounted.
    let _progress = ProgressRegistryHandle::new(execution_context.request_id.clone());
    let language_hint = request.language.clone();
    let punctuate = request.punctuate;
    let explicit_refine = request.word_timestamps_refine;
    // Every independent native model stage resolves from this same immutable
    // product intent. Each stage still owns its own capability matrix and
    // candidate transaction; no auxiliary model inherits a coarse backend or
    // re-reads process defaults after the main ASR dispatch completes.
    let request_execution_intent = execution_intent
        .clone()
        .unwrap_or_else(|| request_execution_intent(request.execution_target));
    // Coarse per-request stage timing: "inference" spans model resolution +
    // audio prep (see the `audio_prep` stage logged inside `_impl` around the
    // WAV load) + decode/longform-assembly, i.e. the whole
    // `run_native_transcription_impl` call; "postprocess" covers the
    // punctuation-restoration and explicit-or-speaker-required forced-align
    // stages below.
    // Grain matches what the task asked for (per-request, not per-frame); the
    // finer `audio_prep` sub-stage nests inside `inference`'s span rather than
    // being disjoint from it, which is called out in both log lines' names.
    let inference_started = Instant::now();
    let NativeTranscriptionOutcome {
        transcription,
        prepared_audio,
        emits_punctuation,
        speaker_finalization,
    } = run_native_transcription_impl(
        request,
        execution_services,
        Some(request_execution_intent.clone()),
        runtime_pack_input,
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
    let transcription = apply_punctuation_stage_with_policy(
        transcription,
        emits_punctuation,
        punctuate,
        execution_services,
        &request_execution_intent,
    )?;
    let refine = explicit_refine || speaker_finalization.requires_word_alignment(&transcription);
    let transcription = if refine {
        // Forced alignment is a separate heavyweight model phase. The
        // finished transcript and normalized PCM are the complete boundary
        // contract, so primary-ASR and earlier auxiliary runtimes are idle and
        // must not retain their admitted host/device commitments while the
        // aligner quotes its graph. This is especially important on unified
        // memory machines, where otherwise two independently valid models can
        // never coexist inside the physical headroom even though their phases
        // are strictly sequential.
        execution_services.unload_idle_native_model_runtime_caches();
        publish_align_progress(execution_context.request_id.as_deref());
        refine_transcription_word_timestamps_with_forced_aligner_policy(
            transcription,
            forced_aligner_audio_view(&prepared_audio, refine)
                .expect("enabled forced alignment retains the normalized PCM view"),
            language_hint.as_deref(),
            execution_services,
            &request_execution_intent,
        )?
    } else {
        transcription
    };
    let result = finalize_native_transcription(
        transcription,
        &speaker_finalization,
        prepared_audio.as_slice(),
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
) -> Result<Transcription, BackendError> {
    if !should_run_punctuation_stage(punctuate, emits_punctuation) {
        return Ok(transcription);
    }
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
            let Ok(runtime) = load_actor(
                execution_services,
                prepared_preflight,
                &prepared_content_id,
                candidate,
            ) else {
                return Ok(transcription.clone());
            };
            Ok(punctuate_transcription_segments_with_actor(
                transcription.clone(),
                &runtime,
            ))
        },
    );
    finish_optional_punctuation_stage(transcription, result)
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
    matches!(
        error,
        PolicyResolvedAuxRuntimeError::Operation(_)
            | PolicyResolvedAuxRuntimeError::CandidatesExhausted { .. }
    )
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
) -> Transcription {
    for segment in &mut transcription.segments {
        if let Ok(punctuated) = punctuate(runtime, &segment.text) {
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

/// Reuses the exact normalized PCM backing decoded by the main request and
/// loads the installed Qwen3-ForcedAligner pack once. Each already-bounded ASR
/// segment is aligned independently, then its local spans are mapped back to
/// the recording clock. This keeps graph memory bounded by the largest decode
/// segment instead of growing with the whole meeting. Segment text and speaker
/// attribution are left untouched; only `words` changes.
fn refine_transcription_word_timestamps_with_forced_aligner_policy(
    transcription: Transcription,
    prepared_audio: PcmSlice,
    language_hint: Option<&str>,
    execution_services: &NativeExecutionServices,
    request_intent: &ExecutionIntent,
) -> Result<Transcription, BackendError> {
    let pack_path = forced_aligner_pack::resolve_forced_aligner_pack_path()
        .ok_or(BackendError::WordTimestampAlignmentPackMissing { backend: "native" })?;
    let language = transcription
        .language
        .clone()
        .or_else(|| language_hint.map(str::to_string))
        .unwrap_or_else(|| "en".to_string());
    let execution_plan = resolve_auxiliary_execution_plan(
        execution_services,
        crate::models::qwen::QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID,
        request_intent,
    )?;
    run_auxiliary_stage_with_policy(
        execution_services,
        &execution_plan,
        "qwen3-forced-aligner",
        |candidate| {
            let backend = resolved_runtime_for_candidate(
                candidate,
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            )
            .backend();
            let session_load_started = Instant::now();
            let session = Qwen3ForcedAlignerSession::load(&pack_path, backend).map_err(|error| {
                BackendError::WordTimestampAlignmentFailed {
                    reason: error.to_string(),
                }
            })?;
            crate::stage_timing::log_detail_stage(
                "forced_aligner",
                "session_load",
                session_load_started.elapsed(),
            );
            let mut refined = transcription.clone();
            let audio_samples = prepared_audio.as_slice().len();
            for (index, segment) in refined.segments.iter_mut().enumerate() {
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
                let alignment_started = Instant::now();
                let items = session
                    .align(
                        prepared_audio.slice(range),
                        &segment.text,
                        &language,
                    )
                    .map_err(|error| BackendError::WordTimestampAlignmentFailed {
                        reason: format!("segment {index}: {error}"),
                    })?;
                crate::stage_timing::log_detail_event(
                    "forced_aligner",
                    format_args!(
                        "stage=segment_align index={index} audio_duration_s={segment_audio_seconds:.3} words={} duration_ms={:.3}",
                        items.len(),
                        alignment_started.elapsed().as_secs_f64() * 1000.0,
                    ),
                );
                assign_local_aligned_words(segment, &items);
            }
            Ok(refined)
        },
    )
    .map_err(required_auxiliary_stage_error)
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
) -> Result<NativeTranscriptionOutcome, BackendError> {
    // Captured up front and threaded explicitly through the dispatch calls
    // below (never a thread-local): every cooperative cancel checkpoint in
    // this function and the shared decode driver reads this same `Arc`.
    let execution_context = Arc::clone(&request.execution_context);
    // Taken up front before `requested_model_id` borrows `request`. The value
    // is already an immutable shared owner; moving it here preserves the
    // exact backing while later ASR/Voice-ID/aligner stages clone only views.
    let prepared_samples = request.prepared_samples.take();
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
        runtime_preflight.runtime_source.path(),
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
    // Empty-but-valid PCM must reach the ASR family's established empty-input
    // behavior without first materializing Voice ID. No acoustic evidence can
    // produce a speaker label, so its effective speaker plan is off.
    let speaker_plan = if prepared_audio.is_empty() {
        SpeakerPlan::Off
    } else {
        speaker_plan
    };

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
        let vad_execution_plan = resolve_fixed_cpu_execution_plan(execution_services.as_ref())?;
        let (mut plan, vad_engine_label) = run_auxiliary_stage_with_policy(
            execution_services.as_ref(),
            &vad_execution_plan,
            "longform-vad",
            |_| {
                let (vad_provider, vad_engine_label) =
                    resolve_longform_vad_provider(&longform_options)?;
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
        let arch_prefers_cpu_decoder =
            prefers_cpu_decoder_for_multichunk_metal(selected_family.model_architecture);
        let multichunk_on_metal = arch_prefers_cpu_decoder
            && plan_stats.chunk_count > 1
            && matches!(
                resolved_runtime_for_request.backend(),
                GgmlCpuGraphBackend::Metal
            );
        if multichunk_on_metal {
            request_options.prefer_cpu_decoder_for_multichunk_metal = true;
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
            let with_align = request.word_timestamps_refine
                || (external_speakers
                    && selected_family.word_timestamp_source
                        == crate::arch::WordTimestampSource::ForcedAligner);
            let total_decode_samples: u64 = plan
                .slices
                .iter()
                .map(|slice| slice.duration_samples() as u64)
                .sum();
            let decode_progress = DecodeProgress::begin(
                execution_context.request_id.clone(),
                total_decode_samples,
                with_align,
            );
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
            // Decode done; the merge/resegment tail below runs uncounted otherwise,
            // which is where the bar used to sit frozen at the last slice count.
            publish_assemble_progress(execution_context.request_id.as_deref(), with_align);
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
    let single_pass_decode_progress = DecodeProgress::begin(
        execution_context.request_id.clone(),
        single_pass_total_samples,
        request.word_timestamps_refine
            || (external_speakers
                && selected_family.word_timestamp_source
                    == crate::arch::WordTimestampSource::ForcedAligner),
    );
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
/// losing speaker turns.
fn finalize_native_transcription(
    mut transcription: Transcription,
    speaker: &SpeakerFinalizationContext,
    prepared_audio: &[f32],
) -> Result<Transcription, BackendError> {
    if speaker.plan == SpeakerPlan::External {
        transcription = apply_speaker_attribution(transcription, &speaker.attribution)?;
    }
    match speaker.plan {
        SpeakerPlan::InDecoder => {
            // Each independently decoded slice is a label scope. The shared
            // identity stage disambiguates those local counters, gathers
            // acoustic evidence, stitches matching voices, and names enrolled
            // people.
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
                crate::diarize::voice_id::name_speakers_across_scopes_with_embedder(
                    embedder,
                    &mut scopes,
                )
                .map_err(speaker_identity_error_to_backend)?;
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
    // Identity runs before cue re-segmentation. Besides avoiding redundant
    // embedding work over presentation-only cue fragments, this preserves the
    // exact one-to-one alignment between assembled in-decoder segments and
    // their decode-scope provenance. Cue splitting copies the resolved speaker
    // identity fields onto every child afterwards.
    transcription = super::cue_segmentation::resegment_transcription_cues(transcription);
    if speaker.strip_forced_word_timestamps {
        for segment in &mut transcription.segments {
            segment.words.clear();
        }
    }
    Ok(transcription)
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
) -> Result<SpeakerAttribution, BackendError> {
    let total_started = Instant::now();
    let diarize_debug = crate::diarize::debug::diarize_debug_enabled();
    if execution_context.is_canceled() {
        return Err(BackendError::TranscriptionCanceled);
    }
    let diarization_started = Instant::now();
    let timeline = diarizer
        .diarize(samples.clone(), 16_000, hint, &|| {
            execution_context.is_canceled()
        })
        .map_err(external_diarization_error_to_backend)?;
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
    let identity_started = Instant::now();
    let identity = crate::diarize::voice_id::resolve_timeline_identities_with_embedder(
        embedder,
        &timeline,
        samples.as_slice(),
    )
    .map_err(speaker_identity_error_to_backend)?;
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
) -> Result<(Box<dyn LongFormVadProvider>, &'static str), BackendError> {
    let provider = crate::diarize::vad::FireRedStreamVadProvider::shared().ok_or_else(|| {
        BackendError::NativeFailClosed {
            reason: "Stream-VAD is unavailable: vendored weights failed to parse \
                         (build-integrity problem)"
                .to_string(),
        }
    })?;
    Ok((Box::new(provider), "firered-stream"))
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
    runtime_source_path: &Path,
    metadata: &BTreeMap<String, String>,
) -> Result<GgmlFamilyAdapterDescriptor, BackendError> {
    let normalized_model_id =
        super::native_model_id::resolve_native_runtime_model_identity_from_string_metadata(
            metadata,
            runtime_source_path,
            None,
        )
        .map_err(|error| BackendError::NativeFailClosed {
            reason: error.to_string(),
        })?
        .model_id;
    if requested_model_id != NATIVE_RUNTIME_MODEL_ID_AUTO
        && !native_runtime_model_refs_match(requested_model_id, &normalized_model_id)
    {
        return Err(BackendError::NativeModelSelectionMismatch {
            requested: requested_model_id.to_string(),
            local: normalized_model_id,
        });
    }

    let selected = OpenAsrArchitectureRegistry::with_builtins()
        .select_ggml_adapter_from_gguf_metadata_v1(metadata)
        .map(|descriptor| descriptor.ggml_family_adapter_descriptor())
        .map_err(map_family_selection_error)?;
    Ok(selected)
}

/// Whether a requested model ref names the same native pack as a local runtime
/// source id. This is the single tolerant matcher for the "bare id contract":
/// packs burn no quant tag into `openasr.model.id`, so a quant-pinned request
/// (`family:quant`) matches a bare runtime id (`family`) -- the
/// `(Some(_), None) => true` arm below is load-bearing. Quant tags on both
/// sides compare through `canonical_quant_tag` so catalog aliases (`q8` vs
/// `q8_0`) match. Every requested-vs-loaded-pack gate (core dispatch, server
/// request validation, CLI serve startup) must use this instead of comparing
/// strings, or catalog-resolved refs spuriously mismatch the loaded pack.
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

fn resolve_fixed_cpu_execution_plan(
    execution_services: &NativeExecutionServices,
) -> Result<ExecutionPlan, BackendError> {
    crate::models::policy_resolved_aux_runtime::resolve_fixed_cpu_execution_plan(execution_services)
        .map_err(|error| BackendError::NativeFailClosed {
            reason: error.to_string(),
        })
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
                        source: result.err(),
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
                drop(result);
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
        let plan = resolve_fixed_cpu_execution_plan(services.as_ref()).expect("CPU plan");
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
    fn family_auto_gpu_policy_lookup_matches_dolphin_and_xasr_gates() {
        use crate::ggml_runtime::AutoGpuPolicy;

        // Regression pin: dolphin lets Auto pick any GPU-class backend
        // (it flipped from CPU-pinned once its encoder weight-placement fix
        // let Metal truly offload and beat CPU end-to-end). xasr-zipformer is
        // `ExceptMetal`: Auto still prefers the generic GPU lane but falls
        // back to CPU on Apple Silicon Metal specifically per the platform
        // performance audit. qwen measured a similar Metal slowdown but is
        // deliberately left `AllBackends` pending a dedicated follow-up (see
        // `models::qwen::graph_config`).
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

        // Auto, family gate enabled (`AllBackends` shape, every builtin
        // family but the three `ExceptMetal` ones): reports exactly what the
        // generic resolver picks -- unchanged behavior.
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

    #[test]
    fn native_progress_is_monotonic_across_phases_and_clears() {
        let id = "monotonic-phases";
        // No run active for this id -> None.
        assert_eq!(native_transcription_progress_for_id(id), None);
        {
            let _handle = ProgressRegistryHandle::new(Some(id.to_string()));
            // Decode phase, weighted by sample share; a run that will forced-align
            // reserves headroom above the decode ceiling.
            let decode = DecodeProgress::begin(Some(id.to_string()), 1000, true);
            let start = native_transcription_progress_for_id(id).expect("run is active");
            assert_eq!(start.phase, NativeTranscriptionPhase::Decode);
            assert_eq!(start.fraction, 0.0);

            decode.complete_slice(400);
            let mid = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(mid.phase, NativeTranscriptionPhase::Decode);
            assert!(mid.fraction >= start.fraction);
            assert!((mid.fraction - DECODE_CEIL_WITH_ALIGN * 0.4).abs() < 1e-6);

            decode.complete_slice(600);
            let decoded = native_transcription_progress_for_id(id).unwrap();
            assert!(decoded.fraction >= mid.fraction);
            // All samples decoded -> exactly the decode ceiling.
            assert!((decoded.fraction - DECODE_CEIL_WITH_ALIGN).abs() < 1e-6);

            publish_assemble_progress(Some(id), true);
            let assembled = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(assembled.phase, NativeTranscriptionPhase::Assemble);
            assert!(assembled.fraction >= decoded.fraction);
            assert!((assembled.fraction - ASSEMBLE_CEIL_WITH_ALIGN).abs() < 1e-6);

            publish_align_progress(Some(id));
            let aligning = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(aligning.phase, NativeTranscriptionPhase::Align);
            assert!(aligning.fraction >= assembled.fraction);
            assert!(aligning.fraction <= 1.0);

            // A late lower report (e.g. an out-of-order slice) never moves the bar
            // backward; only the phase label follows the latest report.
            publish_progress(Some(id), NativeTranscriptionPhase::Decode, 0.1);
            let after = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(after.fraction, aligning.fraction);
        }
        // Handle dropped (completion / early return / panic) -> entry removed.
        assert_eq!(native_transcription_progress_for_id(id), None);
    }

    /// Requirement: two concurrent native transcriptions -- the server places
    /// no concurrency gate on native sessions -- must each get independent,
    /// monotonic progress keyed by their own transcription id, and must not
    /// see any of the other's reports. Also covers the id-scoped analogue of
    /// the old owner-clobber regression: A finishing (its registry entry
    /// removed) must never affect B's still-active, still-readable progress,
    /// and reading B afterward must never show a spurious idle gap.
    #[test]
    fn native_progress_two_concurrent_requests_stay_independent_and_a_finishing_does_not_affect_b()
    {
        let id_a = "concurrent-a";
        let id_b = "concurrent-b";
        assert_eq!(native_transcription_progress_for_id(id_a), None);
        assert_eq!(native_transcription_progress_for_id(id_b), None);

        let handle_a = ProgressRegistryHandle::new(Some(id_a.to_string()));
        let handle_b = ProgressRegistryHandle::new(Some(id_b.to_string()));

        publish_progress(Some(id_a), NativeTranscriptionPhase::Decode, 0.4);
        publish_progress(Some(id_b), NativeTranscriptionPhase::Align, 0.92);

        let progress_a = native_transcription_progress_for_id(id_a).expect("A is active");
        let progress_b = native_transcription_progress_for_id(id_b).expect("B is active");
        assert_eq!(progress_a.phase, NativeTranscriptionPhase::Decode);
        assert!((progress_a.fraction - 0.4).abs() < 1e-6);
        assert_eq!(progress_b.phase, NativeTranscriptionPhase::Align);
        assert!((progress_b.fraction - 0.92).abs() < 1e-6);

        // A further report on A alone must not move B.
        publish_progress(Some(id_a), NativeTranscriptionPhase::Decode, 0.5);
        let progress_b_after_a_advances = native_transcription_progress_for_id(id_b).unwrap();
        assert_eq!(progress_b_after_a_advances, progress_b);

        // A finishes: its own entry is gone, but B is untouched and still reads
        // its exact last-known progress -- no momentary idle in between.
        drop(handle_a);
        assert_eq!(native_transcription_progress_for_id(id_a), None);
        let progress_b_after_a_finishes =
            native_transcription_progress_for_id(id_b).expect("B must survive A finishing");
        assert_eq!(progress_b_after_a_finishes, progress_b);

        drop(handle_b);
        assert_eq!(native_transcription_progress_for_id(id_b), None);
    }

    /// Requirement: a request with no transcription id (a detached/uncancellable
    /// `RequestExecutionContext` -- the client never registered one, or an
    /// internal caller like a CLI single-shot transcribe) must never write a
    /// readable progress entry anywhere. There is no shared slot left for it
    /// to fall back to publishing into.
    #[test]
    fn native_progress_detached_request_never_publishes() {
        let _handle = ProgressRegistryHandle::new(None);
        let decode = DecodeProgress::begin(None, 1000, false);
        decode.complete_slice(500);
        publish_assemble_progress(None, false);
        publish_align_progress(None);
        publish_progress(None, NativeTranscriptionPhase::Decode, 0.5);

        assert_eq!(
            native_transcription_progress_for_id("native-progress-detached-request-probe"),
            None
        );
    }

    /// Sequential (non-overlapping) runs sharing the same transcription id:
    /// the second run's first report must reset the bar to its own starting
    /// point rather than being maxed against whatever the first run left
    /// behind, and each run's handle `Drop` must remove its entry before the
    /// next one starts.
    #[test]
    fn native_progress_sequential_runs_reset_start_and_clear() {
        let id = "sequential-runs";
        assert_eq!(native_transcription_progress_for_id(id), None);

        {
            let _run1 = ProgressRegistryHandle::new(Some(id.to_string()));
            publish_progress(Some(id), NativeTranscriptionPhase::Decode, 0.1);
            publish_progress(Some(id), NativeTranscriptionPhase::Decode, 0.9);
            let run1_progress = native_transcription_progress_for_id(id).unwrap();
            assert!((run1_progress.fraction - 0.9).abs() < 1e-6);
        }
        // run1's handle dropped -> its entry removed before run2 starts.
        assert_eq!(native_transcription_progress_for_id(id), None);

        {
            let _run2 = ProgressRegistryHandle::new(Some(id.to_string()));
            // run2's first report is lower than run1's last fraction; it must
            // become the new starting point, not be maxed against 0.9.
            publish_progress(Some(id), NativeTranscriptionPhase::Decode, 0.2);
            let run2_start = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(run2_start.phase, NativeTranscriptionPhase::Decode);
            assert!((run2_start.fraction - 0.2).abs() < 1e-6);

            // Within run2 the monotonic max still holds.
            publish_progress(Some(id), NativeTranscriptionPhase::Decode, 0.05);
            let run2_after_lower = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(run2_after_lower.fraction, run2_start.fraction);

            publish_progress(Some(id), NativeTranscriptionPhase::Assemble, 0.6);
            let run2_assembled = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(run2_assembled.phase, NativeTranscriptionPhase::Assemble);
            assert!((run2_assembled.fraction - 0.6).abs() < 1e-6);
        }
        assert_eq!(native_transcription_progress_for_id(id), None);
    }

    /// The registry never grows past `PROGRESS_REGISTRY_CAPACITY`: once full,
    /// publishing a new id evicts the longest-resident entry (index 0 of the
    /// insertion-ordered backing `Vec`) to make room, rather than growing
    /// unboundedly. This asserts the aggregate registry state directly, so
    /// (like every other test in this crate that inspects a workspace-shared
    /// resource) it depends on per-test process isolation -- see AGENTS.md's
    /// `cargo nextest` requirement.
    #[test]
    fn native_progress_registry_evicts_the_oldest_entry_once_capacity_is_exceeded() {
        let ids: Vec<String> = (0..=PROGRESS_REGISTRY_CAPACITY)
            .map(|index| format!("capacity-probe-{index}"))
            .collect();
        for id in &ids {
            publish_progress(Some(id), NativeTranscriptionPhase::Decode, 0.1);
        }

        {
            let registry = PROGRESS_REGISTRY
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            assert_eq!(registry.entries.len(), PROGRESS_REGISTRY_CAPACITY);
        }
        // The very first id inserted was evicted to make room for the last one...
        assert_eq!(native_transcription_progress_for_id(&ids[0]), None);
        // ...while every id inserted after it survived.
        for id in &ids[1..] {
            assert!(
                native_transcription_progress_for_id(id).is_some(),
                "expected {id} to still be tracked"
            );
        }

        // Leave the registry as this test found it, rather than leaking
        // `PROGRESS_REGISTRY_CAPACITY` entries into whichever test runs next
        // in this process.
        let mut registry = PROGRESS_REGISTRY
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for id in &ids[1..] {
            registry.remove(id);
        }
    }

    #[test]
    fn native_transcription_progress_legacy_reports_idle_with_no_active_runs() {
        assert_eq!(
            native_transcription_progress(),
            LegacyNativeTranscriptionProgress::Idle
        );
    }

    #[test]
    fn native_transcription_progress_legacy_reports_the_single_active_run() {
        let id = "legacy-single-active";
        let _handle = ProgressRegistryHandle::new(Some(id.to_string()));
        publish_progress(Some(id), NativeTranscriptionPhase::Decode, 0.33);
        assert_eq!(
            native_transcription_progress(),
            LegacyNativeTranscriptionProgress::Single(NativeTranscriptionProgress {
                phase: NativeTranscriptionPhase::Decode,
                fraction: 0.33,
            })
        );
    }

    /// Requirement: with more than one active run, the legacy id-less
    /// endpoint must say so explicitly rather than picking one owner to
    /// impersonate "the" global progress.
    #[test]
    fn native_transcription_progress_legacy_is_ambiguous_with_more_than_one_active_run() {
        let id_a = "legacy-ambiguous-a";
        let id_b = "legacy-ambiguous-b";
        let _handle_a = ProgressRegistryHandle::new(Some(id_a.to_string()));
        let _handle_b = ProgressRegistryHandle::new(Some(id_b.to_string()));
        publish_progress(Some(id_a), NativeTranscriptionPhase::Decode, 0.1);
        publish_progress(Some(id_b), NativeTranscriptionPhase::Decode, 0.2);
        assert_eq!(
            native_transcription_progress(),
            LegacyNativeTranscriptionProgress::Ambiguous { active_count: 2 }
        );
    }

    #[test]
    fn token_step_fraction_normalizes_step_index_against_estimated_total() {
        let window = SliceProgressWindow {
            start_fraction: 0.0,
            span_fraction: 1.0,
        };
        // step_index is 0-based, so "step 0 of 10" already reads as 1/10 of
        // the window, not 0/10 -- the first generated token must show
        // forward motion instead of reporting the window's start again.
        assert!((token_step_fraction(window, 0, 10) - 0.1).abs() < 1e-6);
        assert!((token_step_fraction(window, 4, 10) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn token_step_fraction_scales_by_the_slice_window() {
        // A slice that owns [0.2, 0.2 + 0.3) of the decode-phase fraction:
        // token progress must land inside that sub-range, not [0, 1].
        let window = SliceProgressWindow {
            start_fraction: 0.2,
            span_fraction: 0.3,
        };
        let at_start = token_step_fraction(window, 0, 100);
        let at_half = token_step_fraction(window, 49, 100);
        assert!((at_start - (0.2 + 0.3 * 0.01)).abs() < 1e-6);
        assert!((at_half - (0.2 + 0.3 * 0.50)).abs() < 1e-6);
        assert!(at_start >= window.start_fraction);
        assert!(at_half <= window.start_fraction + window.span_fraction);
    }

    #[test]
    fn token_step_fraction_caps_below_the_full_slice_span() {
        // Even once step_index reaches (or blows past) estimated_total_tokens,
        // the window's own share must stay strictly under its full span --
        // `DecodeProgress::complete_slice` owns closing out the remaining
        // sliver, not per-token interpolation racing ahead of it.
        let window = SliceProgressWindow {
            start_fraction: 0.0,
            span_fraction: 1.0,
        };
        let at_cap = token_step_fraction(window, 99, 100);
        let past_cap = token_step_fraction(window, 500, 100);
        assert!((at_cap - TOKEN_PROGRESS_SLICE_SHARE_CAP).abs() < 1e-6);
        assert!((past_cap - TOKEN_PROGRESS_SLICE_SHARE_CAP).abs() < 1e-6);
        assert!(at_cap < window.start_fraction + window.span_fraction);
    }

    #[test]
    fn token_step_fraction_is_monotonic_in_step_index() {
        let window = SliceProgressWindow {
            start_fraction: 0.1,
            span_fraction: 0.4,
        };
        let mut previous = token_step_fraction(window, 0, 37);
        for step_index in 1..200 {
            let current = token_step_fraction(window, step_index, 37);
            assert!(
                current >= previous,
                "fraction regressed at step {step_index}: {previous} -> {current}"
            );
            previous = current;
        }
    }

    #[test]
    fn token_step_fraction_falls_back_to_the_cap_when_estimate_is_zero() {
        // A zero denominator (defensive: no builtin family emits
        // max_generated_tokens=0, `Seq2SeqGreedyDecodeConfig` fails closed on
        // it) must not divide by zero or report the window as fully done --
        // the cap is the safe fallback, matching an "unknown, assume
        // in-progress" reading.
        let window = SliceProgressWindow {
            start_fraction: 0.0,
            span_fraction: 1.0,
        };
        assert!((token_step_fraction(window, 0, 0) - TOKEN_PROGRESS_SLICE_SHARE_CAP).abs() < 1e-6);
    }

    #[test]
    fn slice_progress_window_places_slices_back_to_back_within_the_decode_ceiling() {
        // `DecodeProgress::begin`/`complete_slice` publish into this id's own
        // registry entry, so -- unlike the old global-slot design -- a unique
        // id here needs no lock or guard to stay isolated from every other
        // test.
        let decode =
            DecodeProgress::begin(Some("slice-window-back-to-back".to_string()), 1000, false);
        let first = decode.slice_progress_window(400);
        assert!((first.start_fraction - 0.0).abs() < 1e-6);
        assert!((first.span_fraction - DECODE_CEIL_NO_ALIGN * 0.4).abs() < 1e-6);

        decode.complete_slice(400);
        let second = decode.slice_progress_window(600);
        // The second slice's window starts exactly where the first slice's
        // completed share left off, so token interpolation never overlaps or
        // skips ahead relative to the sample-weighted slice boundaries.
        assert!((second.start_fraction - DECODE_CEIL_NO_ALIGN * 0.4).abs() < 1e-6);
        assert!((second.span_fraction - DECODE_CEIL_NO_ALIGN * 0.6).abs() < 1e-6);
        assert!((second.start_fraction + second.span_fraction - DECODE_CEIL_NO_ALIGN).abs() < 1e-6);
    }

    #[test]
    fn slice_progress_window_is_the_full_decode_ceiling_for_a_single_slice_run() {
        // The short single-pass / single-slice path treats the whole file as
        // one slice: its window must span the entire decode phase exactly
        // like the long-form path's last slice does, not some smaller
        // fixed share -- this is what makes the two paths share one signal.
        let decode =
            DecodeProgress::begin(Some("slice-window-single-slice".to_string()), 1000, true);
        let window = decode.slice_progress_window(1000);
        assert!((window.start_fraction - 0.0).abs() < 1e-6);
        assert!((window.span_fraction - DECODE_CEIL_WITH_ALIGN).abs() < 1e-6);
    }

    #[test]
    fn should_publish_token_step_throttles_to_every_stride_and_always_the_first() {
        assert!(should_publish_token_step(0));
        for step_index in 1..TOKEN_PROGRESS_PUBLISH_STRIDE {
            assert!(
                !should_publish_token_step(step_index),
                "step {step_index} should be throttled"
            );
        }
        assert!(should_publish_token_step(TOKEN_PROGRESS_PUBLISH_STRIDE));
        assert!(should_publish_token_step(TOKEN_PROGRESS_PUBLISH_STRIDE * 5));
    }

    /// End-to-end wiring test: a `run_dispatch_once`-shaped call routed
    /// through the shared decode driver's token-step sink must land token-
    /// level `publish_progress` calls strictly inside the installed window,
    /// increasing monotonically, without needing a real model pack. Exercises
    /// `install_token_step_progress_sink` (the models-layer hook) and this
    /// module's sink closure shape together, the same composition
    /// `run_dispatch_once_with_progress` installs around a real decode.
    #[test]
    fn token_step_progress_sink_reports_monotonically_inside_its_window() {
        let id = "token-step-sink-window";
        assert_eq!(native_transcription_progress_for_id(id), None);

        {
            let _handle = ProgressRegistryHandle::new(Some(id.to_string()));
            let window = SliceProgressWindow {
                start_fraction: 0.0,
                span_fraction: DECODE_CEIL_NO_ALIGN,
            };
            let _sink_guard =
                crate::models::seq2seq_greedy_decode::install_token_step_progress_sink(
                    move |step_index, max_generated_tokens| {
                        if should_publish_token_step(step_index) {
                            publish_progress(
                                Some(id),
                                NativeTranscriptionPhase::Decode,
                                token_step_fraction(window, step_index, max_generated_tokens),
                            );
                        }
                    },
                );

            let mut previous = 0.0_f32;
            for step_index in 0..40 {
                crate::models::seq2seq_greedy_decode::report_token_step_progress(step_index, 40);
                let progress =
                    native_transcription_progress_for_id(id).expect("sink published at least once");
                assert!(progress.fraction >= previous);
                assert!(progress.fraction <= window.start_fraction + window.span_fraction);
                previous = progress.fraction;
            }
        }
        // Both guards dropped (sink first, then the registry handle) -> entry removed.
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
                if progress.fraction > 0.0 && progress.fraction < DECODE_CEIL_NO_ALIGN {
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

    // The catalog's own product suffix for Hy-MT2's mixed Q4_K_M pack is
    // "q4km" (tooling/publish-model/scripts/_catalog.py QUANT_METADATA), which
    // is exactly what a user copies from `pull_recommended` /
    // `openasr pull hymt2-1.8b:q4km`. `canonical_quant_tag` must recognize it
    // as an alias of q4_k so a request using it matches a runtime source
    // tagged with any other spelling of the same quant.
    #[test]
    fn native_runtime_model_refs_match_catalog_q4km_product_suffix_alias() {
        assert!(native_runtime_model_refs_match(
            "hymt2-1.8b:q4km",
            "hymt2-1.8b:q4_k"
        ));
        assert!(native_runtime_model_refs_match(
            "hymt2-1.8b:q4_k_m",
            "hymt2-1.8b:q4km"
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
        let (_, label) =
            resolve_longform_vad_provider(&options).expect("Stream-VAD must resolve in tests");
        assert_eq!(label, "firered-stream");
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
        let (provider, label) = resolve_longform_vad_provider(&options)
            .expect("Stream-VAD's vendored weights must load in tests");
        assert_eq!(
            label, "firered-stream",
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
        let progress = DecodeProgress::begin(None, 160, false);
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
        let progress = DecodeProgress::begin(None, 160, false);
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
        let decode_progress = DecodeProgress::begin(progress_id, total, false);
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
        // Progress accumulated atomically across workers and reached the decode
        // ceiling (property 3); the registry clamp keeps it monotonic.
        let progress = native_transcription_progress_for_id(id).expect("run published progress");
        assert!(
            (progress.fraction - DECODE_CEIL_NO_ALIGN).abs() < 1e-6,
            "decode progress should reach the ceiling, got {}",
            progress.fraction
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
