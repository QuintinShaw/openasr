use std::{
    collections::BTreeMap,
    path::Path,
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

use crate::NATIVE_RUNTIME_MODEL_ID_AUTO;
use crate::api::audio_io::load_wav_16khz_mono_f32_v0;
use crate::arch::{
    DEFAULT_ENCODER_CHUNK_SECONDS, GENERAL_ARCHITECTURE_KEY, OpenAsrArchitectureRegistry,
    SpeakerSegmentationSource, emits_punctuation_for_model_architecture,
};
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, install_request_backend_override, read_gguf_metadata,
};
use crate::longform::{
    AudioSliceKind, LongFormMode, LongFormVadProvider, SegmentMergePolicy, SegmentTimeDomain,
    SliceTranscript, TranscriptAssembler, plan_longform_slices,
};
use crate::models::builtin_execution_dispatch::build_builtin_ggml_execution_dispatch;
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyLongformProfile, BuiltinDecodePolicyLongformPromptCarryMode,
    resolve_builtin_decode_policy_for_architecture,
};
use crate::models::graph_runtime_config::install_request_inference_threads_override;
use crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index_from_source;
use crate::models::runtime_selection_metadata::selection_metadata_from_gguf;
use crate::{
    ExecutionTarget, GgmlAsrBackendPreference, GgmlAsrExecutionDispatch, GgmlAsrExecutionError,
    GgmlAsrExecutionOptions, GgmlAsrExecutionRequest, GgmlAsrExecutionResult, GgmlAsrPreparedAudio,
    GgmlAsrRuntimeSourcePreflight, GgmlFamilyAdapterDescriptor, GgmlFamilyRegistry,
    GgmlFamilyRegistrySelectionError, OasrV1MetadataError, parse_model_ref,
};

use crate::api::backend::{FailureCategory, log_failure_context, log_request_context};

use super::{BackendError, Transcription, TranscriptionRequest};
use crate::Segment;
use crate::WordTimestamp;
use crate::api::backend::{DecodeTruncation, TranscriptionLongFormMetadata, TruncatedDecode};
use crate::models::firered_punc::pack::resolve_firered_punc_pack_path;
use crate::models::firered_punc::runtime::FireRedPuncRuntime;
use crate::models::qwen::{
    ForcedAlignItem, forced_aligner_pack, refine_word_timestamps_with_forced_aligner,
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
const COHERE_LONGFORM_OVERLAP_SECONDS: f32 = 0.0;
static NATIVE_GGML_EXECUTION_DISPATCH: OnceLock<GgmlAsrExecutionDispatch> = OnceLock::new();

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
    decoded_samples: u64,
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
            decoded_samples: 0,
            decode_ceil,
        }
    }

    /// Mark one slice decoded (or skipped as silent -- silence still consumes its
    /// share of the audio timeline), advancing the bar by that slice's sample share.
    fn complete_slice(&mut self, slice_samples: u64) {
        self.decoded_samples = self.decoded_samples.saturating_add(slice_samples);
        let ratio = if self.total_samples == 0 {
            1.0
        } else {
            (self.decoded_samples as f32 / self.total_samples as f32).clamp(0.0, 1.0)
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
        let start_ratio = (self.decoded_samples as f32 / total).clamp(0.0, 1.0);
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
    runtime_preflight: &GgmlAsrRuntimeSourcePreflight,
    selected_family: &GgmlFamilyAdapterDescriptor,
    chunk: Vec<f32>,
    request_options: GgmlAsrExecutionOptions,
    backend_preference: GgmlAsrBackendPreference,
    execution_context: &Arc<crate::RequestExecutionContext>,
    decode_progress: &mut DecodeProgress,
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
        runtime_preflight,
        selected_family,
        chunk,
        request_options,
        backend_preference,
        execution_context,
    )?;
    decode_progress.complete_slice(slice_samples);
    Ok(result)
}

/// Number of consecutive slices that must each hit a GPU-class compute-buffer
/// allocation failure before the rest of the request stops even trying the
/// GPU (issue #158): one slice's allocation failing under transient VRAM
/// pressure is worth an immediate CPU retry, but if the *next* slice also
/// fails the same way, the pressure is sustained rather than a one-off blip,
/// so re-attempting the GPU on every remaining slice would just re-pay the
/// same failed-allocation cost for no benefit.
const GPU_ALLOCATION_FALLBACK_STREAK_LIMIT: usize = 2;

/// Per-request state for the generic GPU-class allocation-failure fallback.
/// Lives for the duration of one `run_native_transcription` call (one per
/// longform slice loop, or one throwaway instance for the single-pass path).
#[derive(Debug, Default)]
struct GpuAllocationFallbackTracker {
    consecutive_fallbacks: usize,
    forced_cpu_for_rest: bool,
}

impl GpuAllocationFallbackTracker {
    /// The backend preference to actually dispatch this attempt with. Forces
    /// `CpuOnly` once the streak limit has tripped; otherwise passes
    /// `requested` through unchanged so every slice still gets its own first
    /// try at the requested backend (GPU pressure may be transient) until the
    /// streak proves it is not.
    fn effective_preference(
        &self,
        requested: GgmlAsrBackendPreference,
    ) -> GgmlAsrBackendPreference {
        if self.forced_cpu_for_rest && !matches!(requested, GgmlAsrBackendPreference::CpuOnly) {
            GgmlAsrBackendPreference::CpuOnly
        } else {
            requested
        }
    }

    /// Records the outcome of one attempt at `attempted` (the *effective*
    /// preference actually dispatched, not necessarily the caller's original
    /// request). An explicit `CpuOnly` attempt never participates in the
    /// streak: there is no lower backend to fall back to, so a CPU failure is
    /// a real capacity error, not a signal that GPU is under pressure.
    fn record(&mut self, attempted: GgmlAsrBackendPreference, degraded: bool) {
        if matches!(attempted, GgmlAsrBackendPreference::CpuOnly) {
            return;
        }
        if degraded {
            self.consecutive_fallbacks += 1;
            if self.consecutive_fallbacks >= GPU_ALLOCATION_FALLBACK_STREAK_LIMIT {
                self.forced_cpu_for_rest = true;
            }
        } else {
            self.consecutive_fallbacks = 0;
        }
    }
}

/// Why a slice's result came from CPU instead of the requested GPU-class
/// backend. Threaded into the longform provenance / degraded-result
/// diagnostics rather than silently folded into a normal completion (mirrors
/// the whisper max-tokens-cap "degraded" trace tag precedent).
#[derive(Debug, Clone, PartialEq, Eq)]
enum SliceGpuFallback {
    /// This slice's own allocation failed on `original_backend` and was
    /// retried on CPU.
    AllocationFailure { original_backend: String },
    /// The GPU was skipped entirely for this slice because the previous
    /// `GPU_ALLOCATION_FALLBACK_STREAK_LIMIT` slices already fell back.
    SkippedAfterStreak,
}

impl SliceGpuFallback {
    fn log_reason(&self) -> &'static str {
        match self {
            Self::AllocationFailure { .. } => "allocation_failure",
            Self::SkippedAfterStreak => "previous_slices_exhausted_gpu",
        }
    }
}

/// Recognizes `GgmlCpuGraphError::BackendBufferAllocationFailed`'s Display
/// text after it has been flattened into a `BackendError::NativeFailClosed`
/// reason string, and extracts the backend name it names.
///
/// This matches on message text rather than downcasting a preserved error
/// source chain because most model families already flatten their internal
/// decode-pipeline errors to a plain `String` well before the error reaches
/// this generic dispatch boundary (e.g. `dolphin::executor`'s `execute` uses
/// a `fail: impl Fn(String) -> GgmlAsrExecutionError` closure fed by
/// `error.to_string()` at each internal call site) -- by the time a slice's
/// dispatch fails here, there is no live source chain left to downcast.
/// Threading a structured error through every family's internal pipeline
/// just for this one classification would be exactly the kind of invasive,
/// family-specific plumbing change `AGENTS.md` asks generic infrastructure
/// changes to avoid. The marker text comes from this crate's own
/// `#[error(...)]` message (`GgmlCpuGraphError::BackendBufferAllocationFailed`
/// in `ggml_runtime::cpu_graph`), not an external dependency, so it is stable
/// under our own control.
fn gpu_buffer_allocation_failure_backend(error: &BackendError) -> Option<&str> {
    let BackendError::NativeFailClosed { reason } = error else {
        return None;
    };
    const MARKER: &str = "compute buffer allocation failed (backend: ";
    let start = reason.find(MARKER)? + MARKER.len();
    let end = start + reason[start..].find(')')?;
    Some(&reason[start..end])
}

/// Runs one slice's decode, transparently retrying on CPU if the requested
/// GPU-class backend's compute-buffer allocation fails for that slice.
///
/// Issue #158: an ~8GB Vulkan/Metal device's small per-slice compute buffer
/// can OOM on one slice of a long-form request while every other slice
/// succeeds (e.g. concurrent VRAM pressure from resident model weights and
/// warmup buffers) -- failing the *whole* request over one slice's
/// allocation is strictly worse than falling that one slice back to CPU.
/// Later slices still try the requested backend first (the pressure may have
/// been transient), unless the fallback streak limit has tripped (see
/// [`GpuAllocationFallbackTracker`]).
///
/// An explicit `CpuOnly` request never retries: there is no lower backend
/// to fall back to, so a CPU allocation failure is a real, unrecoverable
/// capacity error and must fail closed as before. An explicit `Accelerated`
/// request *does* retry -- the caller asked for speed, but handing back a
/// slower, correct CPU result beats failing the whole transcription outright
/// when the GPU genuinely cannot fit this slice; the degraded flag returned
/// here keeps that substitution visible to the caller rather than silently
/// invisible. Only `BackendBufferAllocationFailed` triggers this; every
/// other error class fails closed exactly as before.
#[allow(clippy::too_many_arguments)]
fn run_dispatch_once_with_progress_and_gpu_fallback(
    dispatch: &GgmlAsrExecutionDispatch,
    runtime_preflight: &GgmlAsrRuntimeSourcePreflight,
    selected_family: &GgmlFamilyAdapterDescriptor,
    chunk: Vec<f32>,
    request_options: GgmlAsrExecutionOptions,
    backend_preference: GgmlAsrBackendPreference,
    execution_context: &Arc<crate::RequestExecutionContext>,
    decode_progress: &mut DecodeProgress,
    slice_samples: u64,
    slice_label: &str,
    tracker: &mut GpuAllocationFallbackTracker,
) -> Result<(GgmlAsrExecutionResult, Option<SliceGpuFallback>), BackendError> {
    let effective_preference = tracker.effective_preference(backend_preference);
    if effective_preference != backend_preference {
        let fallback = SliceGpuFallback::SkippedAfterStreak;
        crate::stage_timing::log_detail_event(
            "native_transcribe",
            format_args!(
                "stage=gpu_alloc_fallback event=skip_gpu slice={slice_label} reason={}",
                fallback.log_reason()
            ),
        );
        let result = run_dispatch_once_with_progress(
            dispatch,
            runtime_preflight,
            selected_family,
            chunk,
            request_options,
            effective_preference,
            execution_context,
            decode_progress,
            slice_samples,
        )?;
        tracker.record(effective_preference, true);
        return Ok((result, Some(fallback)));
    }

    match run_dispatch_once_with_progress(
        dispatch,
        runtime_preflight,
        selected_family,
        chunk.clone(),
        request_options.clone(),
        effective_preference,
        execution_context,
        decode_progress,
        slice_samples,
    ) {
        Ok(result) => {
            tracker.record(effective_preference, false);
            Ok((result, None))
        }
        Err(error) => {
            // An already-CPU attempt has no lower backend to retry against: a
            // `BackendBufferAllocationFailed` here is a real, unrecoverable
            // capacity error, not a signal to fall further back. Must not
            // recurse into another CPU attempt.
            if matches!(effective_preference, GgmlAsrBackendPreference::CpuOnly) {
                return Err(error);
            }
            let Some(backend) = gpu_buffer_allocation_failure_backend(&error) else {
                // Not a GPU allocation failure: fails closed exactly as before,
                // no retry, no streak bookkeeping (this attempt was never a
                // GPU-fallback-eligible outcome).
                return Err(error);
            };
            let backend = backend.to_string();
            let fallback = SliceGpuFallback::AllocationFailure {
                original_backend: backend.clone(),
            };
            crate::stage_timing::log_detail_event(
                "native_transcribe",
                format_args!(
                    "stage=gpu_alloc_fallback event=retry_on_cpu slice={slice_label} backend={backend} reason={}",
                    fallback.log_reason()
                ),
            );
            let result = run_dispatch_once_with_progress(
                dispatch,
                runtime_preflight,
                selected_family,
                chunk,
                request_options,
                GgmlAsrBackendPreference::CpuOnly,
                execution_context,
                decode_progress,
                slice_samples,
            )?;
            tracker.record(effective_preference, true);
            Ok((result, Some(fallback)))
        }
    }
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

/// Entry point for the native backend: runs the ordinary decode/longform/
/// diarization pipeline unchanged (`run_native_transcription_impl`), then --
/// gated on the resolved model's `emits_punctuation` capability and the
/// request's `punctuate` opt-out -- restores punctuation with the installed
/// FireRedPunc capability pack, then -- only when the request opted into
/// `--word-timestamps=aligned` (`word_timestamps_refine`) -- refines the
/// finished transcript's per-word timestamps with the installed
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
) -> Result<Transcription, BackendError> {
    run_native_transcription_fallible(request).inspect_err(|error| {
        log_failure_context(classify_backend_error_for_failure_log(error));
    })
}

/// Coarse [`FailureCategory`] bucket for a `BackendError`, reusing its
/// existing variants (and, for the variants that flatten internal detail
/// into a `NativeFailClosed` reason string, the same allocation-failure
/// marker-text sniffing `gpu_buffer_allocation_failure_backend` already
/// relies on above) rather than introducing a second, parallel error
/// taxonomy just for logging.
fn classify_backend_error_for_failure_log(error: &BackendError) -> FailureCategory {
    match error {
        BackendError::NativeUnsupportedInputFormat { .. } => FailureCategory::AudioIo,
        BackendError::NativeModelPackPathRequired
        | BackendError::NativeModelPackPathRejected { .. }
        | BackendError::NativeModelSelectionMismatch { .. } => FailureCategory::ModelResolve,
        BackendError::DiarizationNotSupported { .. }
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
        BackendError::NativeFailClosed { .. }
            if gpu_buffer_allocation_failure_backend(error).is_some() =>
        {
            FailureCategory::Alloc
        }
        BackendError::NativeFailClosed { .. }
        | BackendError::WordTimestampAlignmentFailed { .. } => FailureCategory::Decode,
    }
}

fn run_native_transcription_fallible(
    request: TranscriptionRequest,
) -> Result<Transcription, BackendError> {
    let refine = request.word_timestamps_refine;
    if refine && !request.word_timestamps {
        return Err(BackendError::WordTimestampAlignmentRequiresWordTimestamps);
    }
    // Captured before `request` is moved into `run_native_transcription_impl`
    // below: `publish_align_progress` after that call still needs this
    // request's transcription id.
    let execution_context = Arc::clone(&request.execution_context);
    // Spans the whole run (decode + assembly inside impl, then the punctuation
    // and forced-align post-processes below) so this request's progress-registry
    // entry is removed on every exit and the align phase advances the same
    // monotonic bar rather than running uncounted.
    let _progress = ProgressRegistryHandle::new(execution_context.request_id.clone());
    let input_path = request.input_path.clone();
    // Only clone the in-memory samples' `Arc` when the (opt-in, uncommon)
    // forced-aligner refine stage will actually need to re-read them after
    // the main decode below has consumed `request`: cloning unconditionally
    // would keep a second strong reference alive for the whole decode,
    // permanently defeating the zero-copy `Arc::try_unwrap` reclaim in
    // `resolve_prepared_audio_samples` (see its doc comment) for every
    // request, not just refine ones.
    let prepared_samples_for_refine = refine.then(|| request.prepared_samples.clone()).flatten();
    let language_hint = request.language.clone();
    let model_pack_path = request.model_pack_path.clone();
    let punctuate = request.punctuate;
    // Captured before the move below: the punctuation post-process is a
    // separate pack from the main ASR family (never carries a
    // `GgmlAsrExecutionRequest`/`resolved_runtime`), so it resolves its own
    // backend explicitly here from this request's own execution target,
    // rather than reaching for the implicit generic default.
    let punctuation_backend = execution_target_backend_preference(request.execution_target)
        .ok()
        .map(|preference| {
            crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                preference.request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            )
            .backend()
        })
        .unwrap_or(crate::ggml_runtime::GgmlCpuGraphBackend::Cpu);
    // Coarse per-request stage timing: "inference" spans model resolution +
    // audio prep (see the `audio_prep` stage logged inside `_impl` around the
    // WAV load) + decode/longform-assembly, i.e. the whole
    // `run_native_transcription_impl` call; "postprocess" covers the
    // optional punctuation-restoration and forced-align refine stages below.
    // Grain matches what the task asked for (per-request, not per-frame); the
    // finer `audio_prep` sub-stage nests inside `inference`'s span rather than
    // being disjoint from it, which is called out in both log lines' names.
    let inference_started = Instant::now();
    let transcription = run_native_transcription_impl(request)?;
    crate::stage_timing::log_stage(
        "native_transcribe",
        "inference",
        inference_started.elapsed(),
    );
    let postprocess_started = Instant::now();
    let transcription = apply_punctuation_stage_if_applicable(
        transcription,
        model_pack_path.as_deref(),
        punctuate,
        punctuation_backend,
    );
    let result = if refine {
        publish_align_progress(execution_context.request_id.as_deref());
        refine_transcription_word_timestamps_with_forced_aligner(
            transcription,
            &input_path,
            prepared_samples_for_refine,
            language_hint.as_deref(),
        )
    } else {
        Ok(transcription)
    };
    crate::stage_timing::log_stage(
        "native_transcribe",
        "postprocess",
        postprocess_started.elapsed(),
    );
    result
}

/// Whether the punctuation-restoration stage should attempt to run: the
/// request has not opted out (`punctuate`, the desktop preference toggle) AND
/// the resolved model's `emits_punctuation` capability is honestly `Some(false)`
/// (see [`should_apply_punctuation`]) -- a model that already punctuates, or
/// whose capability is unknown, is never re-punctuated.
fn should_run_punctuation_stage(punctuate: bool, emits_punctuation: Option<bool>) -> bool {
    punctuate && should_apply_punctuation(emits_punctuation)
}

/// The `general.architecture` value's `emits_punctuation` capability for the
/// pack at `model_pack_path`, or `None` when the path is absent or its
/// metadata cannot be read/does not declare a known architecture -- callers
/// treat `None` exactly like an ASR family with unknown punctuation status
/// (stage does not run), never a hard error: this is a best-effort read of
/// metadata already validated once by `run_native_transcription_impl`.
fn model_emits_punctuation(model_pack_path: Option<&Path>) -> Option<bool> {
    let path = model_pack_path?;
    let metadata = read_gguf_metadata(path).ok()?;
    let architecture = metadata.get_string(GENERAL_ARCHITECTURE_KEY)?;
    emits_punctuation_for_model_architecture(architecture)
}

/// Punctuation-restoration post-process: runs only for an ASR result the
/// catalog honestly declares unpunctuated, and only when the FireRedPunc
/// capability pack is installed. Fail-closed by design -- a missing pack, a
/// corrupt pack, or a classifier failure all leave `transcription` exactly as
/// the ASR family produced it rather than crashing the request or fabricating
/// punctuation; the native backend never downloads this pack.
fn apply_punctuation_stage_if_applicable(
    transcription: Transcription,
    model_pack_path: Option<&Path>,
    punctuate: bool,
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
) -> Transcription {
    if !should_run_punctuation_stage(punctuate, model_emits_punctuation(model_pack_path)) {
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

/// Restores punctuation on each finalized segment's text independently (the
/// stage's documented "finalize-only, per segment" contract -- see
/// `crate::punctuation`'s module docs) and rebuilds the top-level `text` field
/// from the punctuated segments the same way the longform assembler does
/// (trim, drop empties, join with a space), so the punctuated text and
/// segments stay consistent. A segment whose classifier call fails keeps its
/// original (unpunctuated) text -- fail-closed per segment rather than
/// aborting the whole transcript.
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

/// Returns the audio at `input_path` as 16 kHz mono f32 samples, preferring
/// `prepared_samples` -- already resident in memory when
/// `prepare_audio_input`'s in-process symphonia decode path produced them
/// (see `crate::audio::PreparedAudioInput::samples`) -- over re-reading
/// `input_path` from disk. The WAV-passthrough and external ffmpeg/afconvert
/// conversion paths leave `prepared_samples` unset, so this falls back to the
/// WAV load exactly as before for those.
///
/// Takes `prepared_samples` by value (not by reference) so that when the
/// caller passes its *only* remaining `Arc` handle, `Arc::try_unwrap` below
/// reclaims the underlying `Vec<f32>` with zero copy instead of cloning it
/// out from behind a shared reference -- avoiding the double-memory window
/// (the `Arc`'s buffer plus a freshly cloned `Vec` of the same audio) this
/// would otherwise cost for the whole decode. Only falls back to an actual
/// clone when another reference is still alive (the opt-in forced-aligner
/// refine path, which legitimately needs the samples a second time after the
/// main decode has consumed them -- see `run_native_transcription_fallible`).
fn resolve_prepared_audio_samples(
    input_path: &Path,
    prepared_samples: Option<Arc<Vec<f32>>>,
) -> Result<Vec<f32>, crate::NativeAsrError> {
    if let Some(samples) = prepared_samples {
        return Ok(Arc::try_unwrap(samples).unwrap_or_else(|shared| (*shared).clone()));
    }
    load_wav_16khz_mono_f32_v0(
        input_path,
        "Native ASR Core backend",
        "Native ASR Core backend",
    )
}

/// Re-decodes `input_path` (or reuses `prepared_samples` when already
/// resident in memory) and calls the installed Qwen3-ForcedAligner pack once
/// over the whole finished transcript, then reassigns each segment's `words`
/// from the aligner's own per-word spans (dropping the family's approximate
/// per-word confidence -- the aligner does not produce one; never inventing a
/// value is preferred to fabricating one). Segments/text/speaker attribution
/// from the ordinary decode path are left untouched; only `words` changes.
fn refine_transcription_word_timestamps_with_forced_aligner(
    mut transcription: Transcription,
    input_path: &Path,
    prepared_samples: Option<Arc<Vec<f32>>>,
    language_hint: Option<&str>,
) -> Result<Transcription, BackendError> {
    let pack_path = forced_aligner_pack::resolve_forced_aligner_pack_path()
        .ok_or(BackendError::WordTimestampAlignmentPackMissing { backend: "native" })?;
    let prepared_audio =
        resolve_prepared_audio_samples(input_path, prepared_samples).map_err(|error| {
            BackendError::NativeUnsupportedInputFormat {
                reason: error.to_string(),
            }
        })?;
    let language = transcription
        .language
        .clone()
        .or_else(|| language_hint.map(str::to_string))
        .unwrap_or_else(|| "en".to_string());
    let items = refine_word_timestamps_with_forced_aligner(
        &pack_path,
        &prepared_audio,
        &transcription.text,
        &language,
    )
    .map_err(|error| BackendError::WordTimestampAlignmentFailed {
        reason: error.to_string(),
    })?;
    assign_aligned_words_to_segments(&mut transcription.segments, &items);
    Ok(transcription)
}

/// Distributes forced-aligner word spans onto the (time-ordered,
/// non-overlapping) segments they fall into: each item's start time selects
/// the last segment whose own start is `<=` it (segments are sorted and cover
/// the whole file, so this always finds the enclosing segment for a
/// well-formed decode). A segment with no aligned words keeps its prior
/// (family-approximate) word list rather than being emptied -- most often
/// because there is exactly one segment and the whole item list lands in it.
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
/// Identity is deliberately NOT part of this decision: matching
/// recording-local turns to known people is one source-independent stage that
/// runs afterwards (`diarize::voice_id`), so it composes with either source and
/// its absence degrades the result instead of failing the request.
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
    /// A separate segmenter over the same audio produces the turns: today the
    /// VAD + speaker-embedder clustering path.
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

fn run_native_transcription_impl(
    mut request: TranscriptionRequest,
) -> Result<Transcription, BackendError> {
    // Captured up front and threaded explicitly through the dispatch calls
    // below (never a thread-local): every cooperative cancel checkpoint in
    // this function and the shared decode driver reads this same `Arc`.
    let execution_context = Arc::clone(&request.execution_context);
    // Taken up front (before `requested_model_id` below borrows `request` for
    // the rest of this function): leaves `request.prepared_samples` as
    // `None` and hands `resolve_prepared_audio_samples` the only `Arc`
    // strong reference in the common (non-refine) case, so it can reclaim
    // the underlying `Vec<f32>` with zero copy instead of cloning it -- see
    // that function's doc comment.
    let prepared_samples = request.prepared_samples.take();
    let model_resolve_started = Instant::now();
    let requested_model_id = normalize_and_validate_model_id(&request)?;
    let model_pack_path = request
        .model_pack_path
        .as_deref()
        .ok_or(BackendError::NativeModelPackPathRequired)?;
    let runtime_source = super::native_path::validate_local_native_runtime_source(model_pack_path)?;
    let runtime_preflight = load_runtime_source_metadata_and_tensor_index_from_source(
        &runtime_source,
    )
    .map_err(|error| BackendError::NativeFailClosed {
        reason: format!(
            "could not load runtime metadata preflight from '{}': {error}",
            runtime_source.path().display()
        ),
    })?;
    let selection_metadata = selection_metadata_from_gguf(&runtime_preflight.metadata);
    let selected_family = validate_runtime_source_and_select_adapter(
        requested_model_id,
        runtime_preflight.runtime_source.path(),
        &selection_metadata,
    )?;
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
    // the family's own decode, or the external VAD + speaker-embedder pass --
    // never both, so nothing can overwrite the other's labels downstream.
    let speaker_plan = SpeakerPlan::resolve(request.voice_id, selected_family.speaker_segmentation);
    if speaker_plan == SpeakerPlan::External
        && (crate::diarize::embed::shared_embedder().is_none()
            || crate::diarize::vad::FireRedStreamVadProvider::shared().is_none())
    {
        // Fail closed up front rather than silently returning a speaker-less
        // transcript: this family has no speaker structure of its own, so with
        // no embedder or VAD model there is no source at all. (An in-decoder
        // family never reaches this branch -- it degrades to recording-local
        // turns instead of refusing the request.)
        return Err(BackendError::DiarizationNotSupported { backend: "native" });
    }
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
                reason: "The model separates speakers in-decoder; the exact-speaker-count hint only applies to the VAD + speaker-embedder clustering path.",
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

    // Compute speaker turns up front (independent of the transcript) so they can
    // be attributed onto whichever transcription path runs below.
    let speaker_turns = if speaker_plan == SpeakerPlan::External {
        let hint = match request.diarize_speakers {
            Some(speakers) => crate::diarize::contract::DiarizeHint::NumSpeakers(speakers),
            None => crate::diarize::contract::DiarizeHint::Auto,
        };
        compute_speaker_attribution(&prepared_audio, hint)
    } else {
        SpeakerAttribution::default()
    };
    // The executor consumes its input buffer on the short-form path. Retain a
    // copy only for the in-decoder plan, whose recording-local turns need their
    // own acoustic evidence before they can be matched to known people;
    // ordinary transcriptions keep the zero-copy path.
    let in_decoder_turn_audio =
        (speaker_plan == SpeakerPlan::InDecoder).then(|| prepared_audio.clone());

    let dispatch = shared_native_ggml_execution_dispatch();
    let audio_duration_seconds = prepared_audio.len() as f32 / 16_000.0;
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
    let is_whisper_family = selected_family.adapter_id == crate::arch::WHISPER_GGML_ADAPTER_ID;
    let force_word_timestamps_for_segmentation = !is_whisper_family && !request.word_timestamps;
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
    let backend_preference = execution_target_backend_preference(request.execution_target)?;
    // Installed for the whole transcribe call: not consulted by the
    // provenance label or the longform multichunk-metal probe below (those
    // resolve from the explicit `backend_preference` value directly, via
    // `resolved_runtime_for_request` a few lines down), but still the
    // pre-existing, unrelated per-request-override channel some family
    // internals legitimately read mid-decode (e.g. firered_llm's RAM-fit
    // override check, `graph_runtime_config`'s `gpu_stage_enabled_for_backend`).
    let _backend_guard =
        install_request_backend_override(backend_preference.request_backend_override());
    // This family's Auto-mode GPU capability, so the provenance backend label
    // below resolves through the same family-aware gate the family's own
    // executor used.
    let auto_gpu_policy = crate::arch::family_auto_gpu_policy_for_model_architecture(
        selected_family.model_architecture,
    );
    // Resolved once here, from the explicit `backend_preference` value above
    // (never a thread-local read), for everything in this function that
    // needs it before dispatch runs: the longform multichunk-metal probe and
    // the provenance label. `run_dispatch_once` below resolves its own copy
    // the same way, from its own (possibly GPU-fallback-adjusted)
    // `backend_preference` parameter -- see its doc comment for why this
    // can't just be threaded down as the same precomputed value.
    let resolved_runtime_for_request = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
        backend_preference.request_backend_override(),
        auto_gpu_policy,
    );
    // Per-request diagnostics line (source/model/quant/backend/audio shape) --
    // logged once here, after model resolution and audio prep have both
    // succeeded and the backend label is resolvable, and before decode
    // dispatch. Deliberately excludes `request.input_path`/
    // `request.display_file_name` and any decoded/transcribed text: see
    // `request_context`'s module doc for the privacy contract.
    log_request_context(
        request.source,
        requested_model_id,
        &quant_tag_for_log(requested_model_id, runtime_source.path()),
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
        let (vad_provider, vad_engine_label) = resolve_longform_vad_provider(&longform_options)?;
        let plan = plan_longform_slices(
            &prepared_audio,
            16_000,
            &longform_options,
            Some(vad_provider.as_ref()),
        )
        .map_err(|error| BackendError::NativeFailClosed {
            reason: format!("could not build longform slice plan: {error}"),
        })?;
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
        let timeline_kind = if plan.processed_audio.is_some() {
            "packed"
        } else {
            "identity"
        };
        if plan.slices.is_empty() {
            return Ok(Transcription {
                truncated_decodes: Vec::new(),
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
            });
        }
        if plan.processed_audio.is_some() || plan.slices.len() > 1 {
            let mut assembler =
                TranscriptAssembler::new(plan.timeline.clone(), SegmentMergePolicy::default());
            let mut rolling_prompt = request_options.prompt.clone().unwrap_or_default();
            let mut rolling_prompt_token_ids: Vec<u32> = Vec::new();
            let carry_prompt_mode =
                longform_prompt_carry_mode(&longform_options, selected_family.model_architecture);
            let mut ran_any_slice = false;
            let mut suppressed_slice_count = 0usize;
            let plan_audio = plan
                .processed_audio
                .as_deref()
                .unwrap_or(prepared_audio.as_slice());
            // Publish per-slice decode progress for the UI, weighted by each
            // slice's audio samples so the bar tracks decode time rather than slice
            // number. The forced-align refine (if any) continues the same monotonic
            // bar from the outer wrapper; the run-scoped handle removes this
            // request's registry entry on any exit. `word_timestamps_refine`
            // reserves headroom for that phase.
            let with_align = request.word_timestamps_refine;
            let total_decode_samples: u64 = plan
                .slices
                .iter()
                .map(|slice| slice.duration_samples() as u64)
                .sum();
            let mut decode_progress = DecodeProgress::begin(
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
            // Issue #158: per-slice GPU-class compute-buffer allocation
            // fallback state, persisted across the whole slice loop (see
            // `GpuAllocationFallbackTracker` / `run_dispatch_once_with_progress_and_gpu_fallback`).
            let mut gpu_fallback_tracker = GpuAllocationFallbackTracker::default();
            let mut degraded_slice_fallbacks: Vec<(usize, SliceGpuFallback)> = Vec::new();
            // Slices whose decode stopped short of their own audio, rendered
            // for the provenance string channel (see
            // `format_truncated_slice_provenance`).
            let mut truncated_slices: Vec<String> = Vec::new();
            // Original-timeline start of every slice that actually decoded,
            // collected only for an in-decoder-diarizing family: each such
            // slice numbered its speakers from one on its own, so these are
            // the seams between independent speaker scopes (see
            // `speaker_scopes_by_start`). A family whose speakers come from the
            // one whole-recording external pass has a single scope and leaves
            // this empty.
            let mut speaker_scope_starts: Vec<f32> = Vec::new();
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
                let chunk = plan_audio[slice.start_sample..slice.end_sample].to_vec();
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
                            slice_options.prompt_token_ids = Some(rolling_prompt_token_ids.clone());
                        }
                    }
                }
                slice_index += 1;
                let slice_decode_started = Instant::now();
                let (result, slice_gpu_fallback) =
                    run_dispatch_once_with_progress_and_gpu_fallback(
                        dispatch,
                        &runtime_preflight,
                        &selected_family,
                        chunk,
                        slice_options,
                        backend_preference,
                        &execution_context,
                        &mut decode_progress,
                        slice_samples,
                        &format!("index={slice_index}"),
                        &mut gpu_fallback_tracker,
                    )?;
                if let Some(fallback) = slice_gpu_fallback {
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
                if speaker_plan == SpeakerPlan::InDecoder {
                    speaker_scope_starts.push(plan.timeline.map_processed_to_original_seconds(
                        slice.content_start_sample as f32 / 16_000.0,
                    ));
                }
                assembler.push_slice_result(SliceTranscript {
                    slice,
                    text: transcription.text,
                    segments: transcription.segments,
                    time_domain: SegmentTimeDomain::RelativeToSliceContent,
                });
            }
            // Decode done; the merge/resegment tail below runs uncounted otherwise,
            // which is where the bar used to sit frozen at the last slice count.
            publish_assemble_progress(execution_context.request_id.as_deref(), with_align);
            // Issue #158: surface any per-slice GPU-allocation-failure fallback in
            // the run's existing provenance channel (mirrors the
            // `cohere-metal-multichunk-prefer-cpu-decoder` tag above for a similar
            // "backend behaved differently than the naive default" case) rather
            // than silently returning a result whose degraded slices are invisible
            // to the caller.
            if !degraded_slice_fallbacks.is_empty() {
                let retried_indices: Vec<String> = degraded_slice_fallbacks
                    .iter()
                    .filter(|(_, fallback)| {
                        matches!(fallback, SliceGpuFallback::AllocationFailure { .. })
                    })
                    .map(|(index, _)| index.to_string())
                    .collect();
                let skipped_indices: Vec<String> = degraded_slice_fallbacks
                    .iter()
                    .filter(|(_, fallback)| {
                        matches!(fallback, SliceGpuFallback::SkippedAfterStreak)
                    })
                    .map(|(index, _)| index.to_string())
                    .collect();
                if !retried_indices.is_empty() {
                    longform_provenance.push(format!(
                        "core.native.backend.gpu-alloc-fallback:retried-on-cpu,slices={}",
                        retried_indices.join(";")
                    ));
                }
                if !skipped_indices.is_empty() {
                    longform_provenance.push(format!(
                        "core.native.backend.gpu-alloc-fallback:skipped-gpu-after-streak,slices={}",
                        skipped_indices.join(";")
                    ));
                }
            }
            if !truncated_slices.is_empty() {
                longform_provenance.push(format!(
                    "core.native.decode.truncated:slices={}",
                    truncated_slices.join(";")
                ));
            }
            let (assembled, assemble_stats) = assembler.into_parts();
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
                let fallback = run_dispatch_once(
                    dispatch,
                    &runtime_preflight,
                    &selected_family,
                    prepared_audio.clone(),
                    fallback_options,
                    backend_preference,
                    &execution_context,
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
                return finalize_native_transcription(
                    fallback.into_transcription(),
                    audio_duration_seconds,
                    Some(run_metadata),
                    &speaker_turns,
                    in_decoder_turn_audio.as_deref().unwrap_or(&[]),
                    speaker_plan,
                    &[],
                    strip_forced_word_timestamps,
                    reported_language.clone(),
                    fallback_truncated_decodes,
                );
            }
            return finalize_native_transcription(
                assembled,
                audio_duration_seconds,
                Some(run_metadata),
                &speaker_turns,
                in_decoder_turn_audio.as_deref().unwrap_or(&[]),
                speaker_plan,
                &speaker_scope_starts,
                strip_forced_word_timestamps,
                reported_language.clone(),
                truncated_decodes,
            );
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
    let mut single_pass_decode_progress = DecodeProgress::begin(
        execution_context.request_id.clone(),
        single_pass_total_samples,
        request.word_timestamps_refine,
    );
    // Issue #158: a fresh, one-shot tracker -- a single-pass request has
    // exactly one dispatch attempt, so the streak-limit skip path can never
    // trip here, but the same GPU-allocation-failure retry still applies
    // (short audio, or a longform plan that collapsed to one slice, can still
    // land on a GPU-class backend with no VRAM headroom for it).
    let mut single_pass_gpu_fallback_tracker = GpuAllocationFallbackTracker::default();
    let (transcription, single_pass_fallback) = run_dispatch_once_with_progress_and_gpu_fallback(
        dispatch,
        &runtime_preflight,
        &selected_family,
        prepared_audio,
        request_options,
        backend_preference,
        &execution_context,
        &mut single_pass_decode_progress,
        single_pass_total_samples,
        "single-pass",
        &mut single_pass_gpu_fallback_tracker,
    )?;
    if single_pass_fallback.is_some() {
        let tag = "core.native.backend.gpu-alloc-fallback:retried-on-cpu,slices=single-pass";
        // No longform run at all (plain short-audio decode) leaves nowhere to
        // stamp this: the structured log line from
        // `run_dispatch_once_with_progress_and_gpu_fallback` is this path's
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
    finalize_native_transcription(
        transcription.into_transcription(),
        audio_duration_seconds,
        longform_metadata,
        &speaker_turns,
        in_decoder_turn_audio.as_deref().unwrap_or(&[]),
        speaker_plan,
        &[],
        strip_forced_word_timestamps,
        reported_language,
        truncated_decodes,
    )
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

/// Finalize a decoded transcription for return from
/// `run_native_transcription_impl`: normalize segment timing/text (dropping
/// empty segments, filling a fallback span from the request-level audio
/// duration), stamp the longform metadata for this run, attribute + re-segment
/// speaker turns, and stamp the reported source language -- in that fixed
/// order. Every exit path of `run_native_transcription_impl` (the longform
/// all-silent fallback, the longform assembled result, and the short-form /
/// single-slice result) funnels through this single function so the order and
/// parameters of the chain cannot drift between paths; only the decoded
/// `Transcription` body and its longform metadata differ per call site. See
/// the `C1` pipeline-split roadmap: this collapses what were three
/// byte-for-byte-identical call chains into one.
fn finalize_native_transcription(
    transcription: Transcription,
    audio_duration_seconds: f32,
    longform_metadata: Option<TranscriptionLongFormMetadata>,
    speaker_turns: &SpeakerAttribution,
    prepared_audio: &[f32],
    speaker_plan: SpeakerPlan,
    speaker_scope_starts: &[f32],
    strip_forced_word_timestamps: bool,
    reported_language: Option<String>,
    truncated_decodes: Vec<TruncatedDecode>,
) -> Result<Transcription, BackendError> {
    let mut transcription = apply_speaker_turns(
        with_longform_metadata(
            normalize_transcription_segments(transcription, 0.0, audio_duration_seconds),
            longform_metadata,
        ),
        speaker_turns,
        strip_forced_word_timestamps,
    );
    if speaker_plan == SpeakerPlan::InDecoder {
        // The family already produced turns, but only *within* each decode
        // unit: a sliced run numbered its speakers from one per slice, so the
        // labels only mean something relative to the scope that produced them.
        // The source-independent identity stage is what relates the scopes,
        // and fails closed (rather than silently degrading) when it had
        // stitching or naming work to do and no embedder to do it with -- see
        // `voice_id::name_speakers_across_scopes`.
        let mut scopes = speaker_scopes_by_start(
            &mut transcription.segments,
            speaker_scope_starts,
            prepared_audio,
        );
        crate::diarize::voice_id::name_speakers_across_scopes(&mut scopes)?;
    }
    // Stamped after the body is assembled and before the transcript leaves the
    // engine, on every exit path: the per-decode results this run consumed are
    // gone by now, so this is the last point at which "the transcript is short"
    // is still knowable.
    //
    // This is an overwrite, not a merge, so it silently clobbers anything a
    // caller already set on `transcription.truncated_decodes` before handing
    // it here. Every call site is expected to pass that field in empty and
    // supply the real list via the `truncated_decodes` parameter instead --
    // catch a caller that drifts from that contract before it loses truncation
    // visibility outright.
    debug_assert!(
        transcription.truncated_decodes.is_empty(),
        "finalize_native_transcription overwrites truncated_decodes; \
         the incoming transcription must not already carry any"
    );
    transcription.truncated_decodes = truncated_decodes;
    Ok(with_reported_language(transcription, reported_language))
}

/// Cut time-ordered segments into the decode scopes they came from.
///
/// `scope_starts` holds the original-timeline start of each independently
/// decoded slice, in order; fewer than two means the whole transcription is one
/// scope. A segment belongs to the last scope that started at or before its
/// midpoint, and scope assignment is forced non-decreasing so every scope is a
/// contiguous run even if a boundary-straddling segment's midpoint lands on the
/// wrong side. The midpoint (not the start) is the anchor because the
/// assembler's overlap trim can leave a kept segment reaching slightly back
/// into the previous slice's committed span.
///
/// Every scope shares the whole recording as its `samples`: segment times are
/// already mapped to the original timeline by the assembler, so they index
/// straight into it. Scope identity here is about *label provenance*, not about
/// which audio a scope may look at.
fn speaker_scopes_by_start<'a>(
    segments: &'a mut [Segment],
    scope_starts: &[f32],
    samples: &'a [f32],
) -> Vec<crate::diarize::voice_id::SpeakerScope<'a>> {
    if scope_starts.len() < 2 {
        return vec![crate::diarize::voice_id::SpeakerScope { segments, samples }];
    }
    let mut lengths = vec![0usize; scope_starts.len()];
    let mut current = 0usize;
    for segment in segments.iter() {
        let midpoint = (segment.start + segment.end.max(segment.start)) / 2.0;
        let matched = scope_starts
            .iter()
            .rposition(|start| *start <= midpoint)
            .unwrap_or(0);
        current = current.max(matched);
        lengths[current] += 1;
    }
    let mut scopes = Vec::with_capacity(scope_starts.len());
    let mut rest = segments;
    for length in lengths {
        let (head, tail) = rest.split_at_mut(length);
        rest = tail;
        scopes.push(crate::diarize::voice_id::SpeakerScope {
            segments: head,
            samples,
        });
    }
    scopes
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

/// Speaker turns plus the optionally-matched enrolled primary-user identity.
#[derive(Default)]
struct SpeakerAttribution {
    turns: Vec<crate::diarize::contract::SpeakerTurn>,
    identities: BTreeMap<
        crate::diarize::contract::SpeakerId,
        crate::diarize::enrollment::SpeakerDisplayAssignment,
    >,
}

/// Diarize the prepared audio into speaker turns, then match the optional
/// enrolled primary user. Speech segments come from pyannote segmentation
/// (speaker-change + overlap aware) when its pack is installed, else the neural
/// VAD; the shared speaker embedder + agglomerative clustering assign global
/// speakers. Returns empty if the embedder/segmenter are unavailable.
fn compute_speaker_attribution(
    samples: &[f32],
    hint: crate::diarize::contract::DiarizeHint,
) -> SpeakerAttribution {
    use crate::diarize::clustering::AgglomerativeClusterer;
    use crate::diarize::embed::shared_embedder;
    use crate::diarize::pipeline::BatchDiarizer;

    let diarize_debug = crate::diarize::debug::diarize_debug_enabled();
    let Some(embedder) = shared_embedder() else {
        if diarize_debug {
            eprintln!("openasr_diarize_debug stage=batch decision=no-embedder");
        }
        return SpeakerAttribution::default();
    };
    let Some(speech) = crate::diarize::pipeline::resolve_diarization_regions(samples) else {
        if diarize_debug {
            eprintln!("openasr_diarize_debug stage=batch decision=no-speech-regions");
        }
        return SpeakerAttribution::default();
    };
    if diarize_debug {
        eprintln!("openasr_diarize_debug stage=batch regions={}", speech.len());
        for region in &speech {
            eprintln!(
                "openasr_diarize_debug stage=batch region start={:.2} end={:.2} local_speaker={} overlap={}",
                region.range.start_s,
                region.range.end_s,
                region
                    .local_speaker
                    .map(|speaker| speaker.label())
                    .unwrap_or_else(|| "none".to_string()),
                region.overlap
            );
        }
    }
    let clusterer = AgglomerativeClusterer::for_embedder(embedder);
    let diarization =
        BatchDiarizer::new(embedder, &clusterer).diarize_regions(samples, 16_000, &speech, hint);
    if diarize_debug {
        eprintln!(
            "openasr_diarize_debug stage=batch turns={} speakers={}",
            diarization.turns.len(),
            diarization.centroids.len()
        );
        for turn in &diarization.turns {
            eprintln!(
                "openasr_diarize_debug stage=batch turn start={:.2} end={:.2} speaker={} overlap={}",
                turn.range.start_s,
                turn.range.end_s,
                turn.speaker.label(),
                turn.overlap
            );
        }
    }
    let matcher = crate::diarize::voice_id::load_person_matcher_for_active_embedder();
    let identities: BTreeMap<
        crate::diarize::contract::SpeakerId,
        crate::diarize::enrollment::SpeakerDisplayAssignment,
    > = diarization
        .centroids
        .iter()
        .filter_map(|(speaker_id, embedding)| {
            matcher.best_match(embedding).map(|person_match| {
                let assignment = crate::diarize::voice_id::VoiceIdAssignment::from_person_match(
                    *speaker_id,
                    &person_match,
                );
                (
                    *speaker_id,
                    crate::diarize::enrollment::SpeakerDisplayAssignment::from_voice_id_assignment(
                        assignment,
                    ),
                )
            })
        })
        .collect();
    if diarize_debug {
        for (speaker_id, assignment) in &identities {
            eprintln!(
                "openasr_diarize_debug stage=batch identity speaker={} display={} person_id={}",
                speaker_id.label(),
                assignment.speaker,
                assignment.speaker_person_id.as_deref().unwrap_or("none"),
            );
        }
    }
    SpeakerAttribution {
        turns: diarization.turns,
        identities,
    }
}

/// Finalize a transcription for output: attribute speaker turns onto its
/// segments (no-op if empty, splitting segments that span multiple speakers at
/// word-snapped turn boundaries), then re-segment every (single-speaker) segment
/// into subtitle-grade cues. Re-segmentation runs after attribution so cues
/// never straddle a speaker turn, and before the strip so it can use the word
/// anchors. `strip_forced_word_timestamps` removes the anchors that were
/// force-enabled for the split when the caller did not request them.
fn apply_speaker_turns(
    mut transcription: Transcription,
    attribution: &SpeakerAttribution,
    strip_forced_word_timestamps: bool,
) -> Transcription {
    if !attribution.turns.is_empty() {
        transcription.segments = crate::diarize::attribution::assign_speakers(
            &attribution.turns,
            std::mem::take(&mut transcription.segments),
            &attribution.identities,
        );
    }
    transcription = super::cue_segmentation::resegment_transcription_cues(transcription);
    if strip_forced_word_timestamps {
        for segment in &mut transcription.segments {
            segment.words.clear();
        }
    }
    transcription
}

fn shared_native_ggml_execution_dispatch() -> &'static GgmlAsrExecutionDispatch {
    NATIVE_GGML_EXECUTION_DISPATCH.get_or_init(|| {
        build_builtin_ggml_execution_dispatch().expect("builtin native ggml dispatch must wire")
    })
}

/// Idle-unload for the offline (file-transcription) dispatch. Deliberately
/// uses `get()`, not `get_or_init` -- a daemon that never served a file
/// transcription has nothing resident here, and this must not be the thing
/// that first builds the dispatch.
pub(crate) fn unload_idle_native_offline_runtime_caches() {
    if let Some(dispatch) = NATIVE_GGML_EXECUTION_DISPATCH.get() {
        dispatch.unload_all();
    }
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
    apply_conservative_seq2seq_longform_safety_policy(model_architecture, options, provenance);
    apply_encoder_attention_span_longform_safety_policy(model_architecture, options, provenance);
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
    if (options.overlap_seconds - COHERE_LONGFORM_OVERLAP_SECONDS).abs() > f32::EPSILON {
        options.overlap_seconds = COHERE_LONGFORM_OVERLAP_SECONDS;
        changed = true;
        provenance.push(format!(
            "core.native.longform.policy:cohere-overlap={}",
            COHERE_LONGFORM_OVERLAP_SECONDS
        ));
    }
    if options.carry_prompt_across_slices {
        options.carry_prompt_across_slices = false;
        changed = true;
        provenance.push("core.native.longform.policy:cohere-disable-prompt-carry".to_string());
    }
    if changed {
        provenance.push(format!(
            "core.native.longform.policy:cohere-chunk-cap={}",
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
    let mut changed = false;
    if options.chunk_seconds > max_safe_chunk_seconds {
        options.chunk_seconds = max_safe_chunk_seconds;
        changed = true;
    }
    if options.max_chunk_seconds > max_safe_chunk_seconds {
        options.max_chunk_seconds = max_safe_chunk_seconds;
        changed = true;
    }
    if options.min_chunk_seconds > max_safe_chunk_seconds {
        options.min_chunk_seconds = max_safe_chunk_seconds;
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

    let registry = GgmlFamilyRegistry::with_builtin_adapters();
    let selected = registry
        .select_from_gguf_metadata_v1(metadata)
        .cloned()
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

fn map_family_selection_error(error: GgmlFamilyRegistrySelectionError) -> BackendError {
    match error {
        GgmlFamilyRegistrySelectionError::InvalidMetadata(OasrV1MetadataError::MissingKey(key)) => {
            BackendError::NativeFailClosed {
                reason: format!(
                    "gguf metadata is missing required OASR v1 key '{key}' for family adapter selection"
                ),
            }
        }
        GgmlFamilyRegistrySelectionError::InvalidMetadata(OasrV1MetadataError::EmptyValue(key)) => {
            BackendError::NativeFailClosed {
                reason: format!(
                    "gguf metadata key '{key}' must be non-empty for family adapter selection"
                ),
            }
        }
        GgmlFamilyRegistrySelectionError::Ambiguous { adapter_ids } => {
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

/// Resolves its own [`crate::ggml_runtime::ResolvedFamilyRuntimeInput`] from
/// `backend_preference`/`selected_family` rather than accepting one as a
/// parameter: the GPU-allocation-failure fallback
/// (`run_dispatch_once_with_progress_and_gpu_fallback`) retries a slice with
/// a *different* `backend_preference` (forced `CpuOnly`), so a value
/// precomputed once at the top of `transcribe_native` would be stale for
/// that retry -- recomputing here from the parameter actually in effect for
/// this attempt is what makes each attempt's resolved backend correct.
fn run_dispatch_once(
    dispatch: &GgmlAsrExecutionDispatch,
    runtime_preflight: &GgmlAsrRuntimeSourcePreflight,
    selected_family: &GgmlFamilyAdapterDescriptor,
    samples: Vec<f32>,
    request_options: GgmlAsrExecutionOptions,
    backend_preference: GgmlAsrBackendPreference,
    execution_context: &Arc<crate::RequestExecutionContext>,
) -> Result<GgmlAsrExecutionResult, BackendError> {
    let resolved_runtime = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
        backend_preference.request_backend_override(),
        crate::arch::family_auto_gpu_policy_for_model_architecture(
            selected_family.model_architecture,
        ),
    );
    let execution_request = GgmlAsrExecutionRequest {
        runtime_source_path: runtime_preflight.runtime_source.path().to_path_buf(),
        runtime_source_preflight: Some(runtime_preflight.clone()),
        selected_family: selected_family.clone(),
        prepared_audio: GgmlAsrPreparedAudio::mono_16khz(samples),
        request_options,
        backend_preference,
        resolved_runtime,
        execution_context: Arc::clone(execution_context),
    };
    let _thread_override = install_request_inference_threads_override(
        execution_request.request_options.inference_threads,
    );
    let result = dispatch
        .execute(&execution_request)
        .map_err(|error| dispatch_error_to_backend(error, execution_context))?;
    Ok(result)
}

fn execution_target_backend_preference(
    target: Option<ExecutionTarget>,
) -> Result<GgmlAsrBackendPreference, BackendError> {
    match target.unwrap_or_default() {
        ExecutionTarget::Auto => Ok(GgmlAsrBackendPreference::Auto),
        ExecutionTarget::Cpu => Ok(GgmlAsrBackendPreference::CpuOnly),
        ExecutionTarget::Accelerated => {
            let has_accelerated_device = crate::ggml_available_devices()
                .iter()
                .any(|device| device.kind.is_gpu());
            if has_accelerated_device {
                Ok(GgmlAsrBackendPreference::Accelerated)
            } else {
                Err(BackendError::NativeFailClosed {
                    reason: "execution_target=accelerated was requested, but no ggml GPU device is available."
                        .to_string(),
                })
            }
        }
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
    if !options.carry_prompt_across_slices {
        return LongformPromptCarryMode::Disabled;
    }
    resolve_builtin_decode_policy_for_architecture(model_architecture)
        .map(|policy| match policy.longform_prompt_carry_mode {
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
        .is_some_and(|descriptor| descriptor.prefer_cpu_decoder_for_multichunk_metal)
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

/// Best-effort quant tag for the `stage=request_context` log line: installed
/// packs live at `<home>/models/<model>/<quant>/<model>-<quant>.oasr` (see
/// `pull.rs`'s `InstalledPack` layout), so the runtime pack path's *parent
/// directory name* already is the quant tag -- no second GGUF/metadata read
/// needed. Falls back to the tag parsed off the request's own model ref
/// (`family:quant`) for a pack laid out outside that convention (e.g.
/// `--model-pack` pointed at an arbitrary file), and finally to `"unknown"`
/// rather than fabricating a value.
fn quant_tag_for_log(requested_model_id: &str, runtime_pack_path: &Path) -> String {
    let from_parent_dir = runtime_pack_path
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str());
    let from_request_tag = parse_model_ref(requested_model_id)
        .ok()
        .and_then(|reference| reference.tag);
    match from_parent_dir.or(from_request_tag.as_deref()) {
        Some(tag) => crate::canonical_quant_tag(tag).to_string(),
        None => "unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GgmlAsrExecutor;
    use crate::arch::DEFAULT_ENCODER_SAFE_CHUNK_SECONDS;
    use std::sync::Mutex;

    fn uncancellable_execution_context_for_test() -> Arc<crate::RequestExecutionContext> {
        Arc::new(crate::RequestExecutionContext::uncancellable(
            "test fixture",
        ))
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
            descriptor.speaker_segmentation,
            SpeakerSegmentationSource::InDecoder
        );
        let decoded = concat!(
            "[0.28][S01] And so, my fellow Americans,[2.32][3.22][S02] ask not what your ",
            "country can do for you,[7.71][8.12][S01] ask what you can do for your country.[10.59]",
        );

        let off = SpeakerPlan::resolve(false, descriptor.speaker_segmentation);
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

        let on = SpeakerPlan::resolve(true, descriptor.speaker_segmentation);
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
            let mut decode = DecodeProgress::begin(Some(id.to_string()), 1000, true);
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
        let mut decode = DecodeProgress::begin(None, 1000, false);
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
        let mut decode =
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

        let decode_thread = std::thread::spawn(move || run_native_transcription(request));

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

    /// A `ScopedSlices` family gets its declared decoder-context window in
    /// place of the shared 30s default -- widened, not clamped -- plus the
    /// three options that shape implies (a contiguous full-coverage planner
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
        assert!(target_seconds > crate::LongFormOptions::default().chunk_seconds);

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
            COHERE_LONGFORM_OVERLAP_SECONDS
        );
        assert!(
            resolution
                .provenance
                .iter()
                .any(|entry| entry.contains("core.native.longform.policy:cohere-chunk-cap="))
        );
        assert!(
            resolution
                .provenance
                .iter()
                .any(|entry| entry.contains("core.native.longform.policy:cohere-overlap="))
        );
        assert!(resolution.provenance.iter().any(|entry| {
            entry.contains("core.native.longform.policy:cohere-disable-prompt-carry")
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
            COHERE_LONGFORM_OVERLAP_SECONDS
        );
        assert!(!resolution.options.carry_prompt_across_slices);
    }

    #[test]
    fn qwen_metal_longform_policy_keeps_default_chunk_size() {
        // qwen has no `ConservativeSeq2SeqV1` decode-side profile, so
        // `chunk_seconds` (already 30.0 by default) is untouched. But qwen's
        // audio encoder IS `GlobalQuadratic` (issue #68), so the much larger
        // `max_chunk_seconds` default (120.0) -- the true ceiling the VAD/
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
        // matching architecture, so every safety cap silently no-ops and the
        // unsafe 120s default max_chunk_seconds survives untouched.
        let wrong = resolve_native_longform_policy_for_backend(
            None,
            120.0,
            selected_family.adapter_id,
            GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(wrong.options.max_chunk_seconds, 120.0);
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

        // A host that can afford more than the default chunk length keeps the
        // band the slicer needs in order to cut on a real pause.
        let mut roomy = defaults.clone();
        assert!(clamp_longform_chunks_to_encoder_memory_ceiling(
            &mut roomy, 90.0
        ));
        assert_eq!(roomy.chunk_seconds, defaults.chunk_seconds);
        assert_eq!(roomy.max_chunk_seconds, 90.0);
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
    /// `FixedWindow` (whisper) and `LocalChunked` (zipformer) architectures
    /// need no additional cap and keep whatever window their own slice shape
    /// asked for (the shared 120s default unless the family declares one). All nine
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
                descriptor.model_architecture,
                GgmlCpuGraphBackend::Cpu,
            );
            match descriptor.longform_max_safe_chunk_seconds() {
                Some(max_safe_chunk_seconds) => {
                    assert_eq!(
                        max_safe_chunk_seconds, DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
                        "'{}' GlobalQuadratic ceiling must be the shared default absent a cited \
                         upstream override",
                        descriptor.model_architecture
                    );
                    assert_eq!(
                        resolution.options.max_chunk_seconds, max_safe_chunk_seconds,
                        "'{}' must resolve max_chunk_seconds to exactly {max_safe_chunk_seconds}, got {}",
                        descriptor.model_architecture, resolution.options.max_chunk_seconds
                    );
                    assert!(
                        resolution.options.chunk_seconds <= max_safe_chunk_seconds,
                        "'{}' must cap chunk_seconds to <= {max_safe_chunk_seconds}, got {}",
                        descriptor.model_architecture,
                        resolution.options.chunk_seconds
                    );
                }
                None => {
                    // No encoder cap applies, so the window is whatever the
                    // family's own slice shape asked for: its declared
                    // decoder-context window, or the shared default when it
                    // declares none.
                    let expected = match descriptor.longform_slice_shape {
                        crate::arch::OpenAsrLongformSliceShape::ScopedSlices {
                            max_seconds,
                            ..
                        } => max_seconds,
                        crate::arch::OpenAsrLongformSliceShape::SharedWindow => 120.0,
                    };
                    assert_eq!(
                        resolution.options.max_chunk_seconds, expected,
                        "'{}' (FixedWindow/LocalChunked) must keep its declared window",
                        descriptor.model_architecture
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
    fn native_dispatch_is_process_shared() {
        let first = shared_native_ggml_execution_dispatch() as *const _;
        let second = shared_native_ggml_execution_dispatch() as *const _;
        assert_eq!(first, second);
    }

    #[test]
    fn normalize_synthesizes_single_segment_when_model_returns_none() {
        let transcription = normalize_transcription_segments(
            Transcription {
                truncated_decodes: Vec::new(),
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
        let transcription =
            run_native_transcription(request).expect("diarized transcription must succeed");

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
    fn fewer_than_two_scope_starts_is_one_scope() {
        let mut segments = vec![segment(0.0, 1.0, "a"), segment(1.0, 2.0, "b")];
        let scopes = speaker_scopes_by_start(&mut segments, &[], &[]);
        assert_eq!(scopes.len(), 1);
        assert_eq!(scopes[0].segments.len(), 2);

        let mut segments = vec![segment(0.0, 1.0, "a")];
        let scopes = speaker_scopes_by_start(&mut segments, &[0.0], &[]);
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
        let scopes = speaker_scopes_by_start(&mut segments, &[0.0, 180.0, 360.0], &[]);
        let sizes: Vec<usize> = scopes.iter().map(|scope| scope.segments.len()).collect();
        assert_eq!(sizes, vec![2, 1, 1]);
    }

    /// A segment kept from a slice's overlap re-read can start marginally
    /// before its own slice began. Scope assignment is forced non-decreasing so
    /// such a segment cannot fall back into the previous scope and split that
    /// scope's run in two.
    #[test]
    fn scope_assignment_never_moves_backwards() {
        let mut segments = vec![
            segment(0.0, 10.0, "a"),
            segment(181.0, 190.0, "b"),
            segment(179.0, 181.0, "c"),
            segment(360.0, 370.0, "d"),
        ];
        let scopes = speaker_scopes_by_start(&mut segments, &[0.0, 180.0, 360.0], &[]);
        let sizes: Vec<usize> = scopes.iter().map(|scope| scope.segments.len()).collect();
        assert_eq!(sizes, vec![1, 2, 1]);
        assert_eq!(scopes[1].segments[1].text, "c");
    }

    /// Every scope must be represented even when a slice produced no surviving
    /// segments, so scope indices keep lining up with the slices that decoded.
    #[test]
    fn a_scope_with_no_surviving_segments_is_still_a_scope() {
        let mut segments = vec![segment(0.0, 10.0, "a"), segment(360.0, 370.0, "d")];
        let scopes = speaker_scopes_by_start(&mut segments, &[0.0, 180.0, 360.0], &[]);
        let sizes: Vec<usize> = scopes.iter().map(|scope| scope.segments.len()).collect();
        assert_eq!(sizes, vec![1, 0, 1]);
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
    fn model_emits_punctuation_reads_the_architectures_capability_from_pack_metadata() {
        let dir = tempfile::tempdir().unwrap();

        let dolphin_pack = dir.path().join("dolphin.oasr");
        let mut dolphin_metadata = std::collections::BTreeMap::new();
        dolphin_metadata.insert(
            GENERAL_ARCHITECTURE_KEY.to_string(),
            crate::arch::DOLPHIN_GGML_ARCHITECTURE_ID.to_string(),
        );
        crate::testing::write_tiny_gguf_runtime_source(
            &dolphin_pack,
            &crate::testing::TinyGgufFixtureSpec::new(dolphin_metadata),
        )
        .expect("write dolphin fixture");
        // Dolphin's cn-dialect training corpus is honestly unpunctuated.
        assert_eq!(model_emits_punctuation(Some(&dolphin_pack)), Some(false));

        let whisper_pack = dir.path().join("whisper.oasr");
        let mut whisper_metadata = std::collections::BTreeMap::new();
        whisper_metadata.insert(
            GENERAL_ARCHITECTURE_KEY.to_string(),
            crate::arch::WHISPER_GGML_ARCHITECTURE_ID.to_string(),
        );
        crate::testing::write_tiny_gguf_runtime_source(
            &whisper_pack,
            &crate::testing::TinyGgufFixtureSpec::new(whisper_metadata),
        )
        .expect("write whisper fixture");
        assert_eq!(model_emits_punctuation(Some(&whisper_pack)), Some(true));

        let unknown_pack = dir.path().join("unknown.oasr");
        crate::testing::write_tiny_gguf_runtime_source(
            &unknown_pack,
            &crate::testing::TinyGgufFixtureSpec::new(std::collections::BTreeMap::new()),
        )
        .expect("write unknown fixture");
        assert_eq!(model_emits_punctuation(Some(&unknown_pack)), None);

        assert_eq!(model_emits_punctuation(None), None);
        assert_eq!(
            model_emits_punctuation(Some(Path::new("/nonexistent/pack.oasr"))),
            None
        );
    }

    #[test]
    fn apply_punctuation_stage_leaves_transcription_unchanged_when_stage_does_not_run() {
        // No model pack path at all -> `model_emits_punctuation` is `None` ->
        // the stage never runs, regardless of the FireRedPunc pack's install
        // state on this machine -- fail-closed, never fabricated punctuation.
        let transcription = Transcription {
            truncated_decodes: Vec::new(),
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

    // -- issue #158: per-slice GPU-class compute-buffer allocation fallback --

    #[test]
    fn gpu_buffer_allocation_failure_backend_extracts_name_from_nested_reason() {
        // The marker text is nested inside outer wrapping (executor id, adapter
        // id, family-specific "graph construction failed at ..." context) the
        // way it really arrives after `dispatch_error_to_backend`, not a bare
        // top-level message -- this must still find it.
        let error = BackendError::NativeFailClosed {
            reason: "ggml executor 'firered-aed-ggml-executor-v1' failed for adapter \
                     'ggml-family-firered-aed-runtime-v1': firered-aed encoder graph \
                     construction failed at 'encoder_forward': compute buffer allocation \
                     failed (backend: Vulkan0)"
                .to_string(),
        };
        assert_eq!(
            gpu_buffer_allocation_failure_backend(&error),
            Some("Vulkan0")
        );
    }

    #[test]
    fn gpu_buffer_allocation_failure_backend_ignores_unrelated_errors() {
        let unrelated = BackendError::NativeFailClosed {
            reason: "ggml executor 'whisper-ggml-executor-v1' failed for adapter 'x': \
                     tensor 'encoder.blocks.0.attn.weight' is missing from context"
                .to_string(),
        };
        assert_eq!(gpu_buffer_allocation_failure_backend(&unrelated), None);

        let other_variant = BackendError::ServeBatchUnavailable {
            reason: "compute buffer allocation failed (backend: Metal)".to_string(),
            retryable: true,
        };
        assert_eq!(
            gpu_buffer_allocation_failure_backend(&other_variant),
            None,
            "only NativeFailClosed carries a classifiable ggml-graph reason"
        );
    }

    #[test]
    fn gpu_allocation_fallback_tracker_trips_after_streak_limit_and_resets_on_success() {
        let mut tracker = GpuAllocationFallbackTracker::default();
        assert_eq!(
            tracker.effective_preference(GgmlAsrBackendPreference::Auto),
            GgmlAsrBackendPreference::Auto
        );

        // First fallback: still under the limit, GPU still tried next time.
        tracker.record(GgmlAsrBackendPreference::Auto, true);
        assert_eq!(
            tracker.effective_preference(GgmlAsrBackendPreference::Auto),
            GgmlAsrBackendPreference::Auto
        );

        // Second consecutive fallback trips the streak: forced CPU from here on.
        tracker.record(GgmlAsrBackendPreference::Auto, true);
        assert_eq!(
            tracker.effective_preference(GgmlAsrBackendPreference::Auto),
            GgmlAsrBackendPreference::CpuOnly
        );
        assert_eq!(
            tracker.effective_preference(GgmlAsrBackendPreference::Accelerated),
            GgmlAsrBackendPreference::CpuOnly
        );

        // A fresh tracker resets on a successful GPU attempt in between two
        // fallbacks -- the streak counts *consecutive* fallbacks only.
        let mut tracker = GpuAllocationFallbackTracker::default();
        tracker.record(GgmlAsrBackendPreference::Auto, true);
        tracker.record(GgmlAsrBackendPreference::Auto, false);
        tracker.record(GgmlAsrBackendPreference::Auto, true);
        assert_eq!(
            tracker.effective_preference(GgmlAsrBackendPreference::Auto),
            GgmlAsrBackendPreference::Auto,
            "a successful attempt in between resets the streak"
        );
    }

    #[test]
    fn gpu_allocation_fallback_tracker_never_forces_cpu_from_an_explicit_cpu_only_attempt() {
        // An explicit CpuOnly attempt has no lower backend to fall back to, so
        // its own failure is a real capacity error, not GPU-pressure signal --
        // it must never participate in (or be forced further by) the streak.
        let mut tracker = GpuAllocationFallbackTracker::default();
        tracker.record(GgmlAsrBackendPreference::CpuOnly, true);
        tracker.record(GgmlAsrBackendPreference::CpuOnly, true);
        tracker.record(GgmlAsrBackendPreference::CpuOnly, true);
        assert_eq!(
            tracker.effective_preference(GgmlAsrBackendPreference::Auto),
            GgmlAsrBackendPreference::Auto
        );
    }

    /// A stub `GgmlAsrExecutor` for the GPU-fallback wrapper tests: records
    /// every `backend_preference` it was invoked with, and fails with a
    /// configurable error whenever the request does *not* ask for `CpuOnly`
    /// (mimicking a GPU-class compute-buffer allocation that fails no matter
    /// how many times it is retried on the same backend, but always succeeds
    /// once actually run on CPU -- the real-world shape of issue #158).
    struct GpuAllocFallbackStubExecutor {
        calls: Mutex<Vec<GgmlAsrBackendPreference>>,
        gpu_failure_reason: String,
    }

    impl GpuAllocFallbackStubExecutor {
        fn new(gpu_failure_reason: impl Into<String>) -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                gpu_failure_reason: gpu_failure_reason.into(),
            }
        }

        fn calls_snapshot(&self) -> Vec<GgmlAsrBackendPreference> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl GgmlAsrExecutor for GpuAllocFallbackStubExecutor {
        fn executor_id(&self) -> &'static str {
            "gpu-alloc-fallback-stub"
        }

        fn supports_phrase_bias(&self) -> bool {
            true
        }

        fn execute(
            &self,
            request: &GgmlAsrExecutionRequest,
        ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
            self.calls.lock().unwrap().push(request.backend_preference);
            if matches!(
                request.backend_preference,
                GgmlAsrBackendPreference::CpuOnly
            ) {
                return Ok(GgmlAsrExecutionResult {
                    transcription: Transcription {
                        truncated_decodes: Vec::new(),
                        text: "ok-on-cpu".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                    decode_truncation: None,
                });
            }
            Err(GgmlAsrExecutionError::ExecutorFailed {
                executor_id: "gpu-alloc-fallback-stub",
                adapter_id: request.selected_family.adapter_id,
                reason: self.gpu_failure_reason.clone(),
            })
        }
    }

    /// A stub whose executor always fails, regardless of backend preference,
    /// with a non-allocation error -- for the "other errors never retry" test.
    struct AlwaysFailsUnrelatedStubExecutor {
        calls: Mutex<usize>,
    }

    impl GgmlAsrExecutor for AlwaysFailsUnrelatedStubExecutor {
        fn executor_id(&self) -> &'static str {
            "always-fails-unrelated-stub"
        }

        fn supports_phrase_bias(&self) -> bool {
            true
        }

        fn execute(
            &self,
            request: &GgmlAsrExecutionRequest,
        ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
            *self.calls.lock().unwrap() += 1;
            Err(GgmlAsrExecutionError::ExecutorFailed {
                executor_id: "always-fails-unrelated-stub",
                adapter_id: request.selected_family.adapter_id,
                reason: "tensor 'x' is missing from context".to_string(),
            })
        }
    }

    fn tiny_whisper_preflight(dir: &Path) -> GgmlAsrRuntimeSourcePreflight {
        let pack_path = dir.join("whisper.oasr");
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            GENERAL_ARCHITECTURE_KEY.to_string(),
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

    fn gpu_fallback_test_fixture(
        dir: &Path,
        executor: std::sync::Arc<dyn GgmlAsrExecutor>,
    ) -> (
        GgmlAsrExecutionDispatch,
        GgmlAsrRuntimeSourcePreflight,
        GgmlFamilyAdapterDescriptor,
    ) {
        let preflight = tiny_whisper_preflight(dir);
        let dispatch = GgmlAsrExecutionDispatch::default().with_whisper_non_streaming_cpu(executor);
        (dispatch, preflight, crate::whisper_runtime_descriptor_v1())
    }

    #[test]
    fn slice_dispatch_retries_gpu_allocation_failure_on_cpu_and_reports_it() {
        let dir = tempfile::tempdir().unwrap();
        let executor = std::sync::Arc::new(GpuAllocFallbackStubExecutor::new(
            "compute buffer allocation failed (backend: Vulkan0)",
        ));
        let (dispatch, preflight, family) = gpu_fallback_test_fixture(dir.path(), executor.clone());
        let mut decode_progress = DecodeProgress::begin(None, 1_000, false);
        let mut tracker = GpuAllocationFallbackTracker::default();

        let (result, fallback) = run_dispatch_once_with_progress_and_gpu_fallback(
            &dispatch,
            &preflight,
            &family,
            vec![0.0; 1_000],
            GgmlAsrExecutionOptions::default(),
            GgmlAsrBackendPreference::Auto,
            &uncancellable_execution_context_for_test(),
            &mut decode_progress,
            1_000,
            "index=1",
            &mut tracker,
        )
        .expect("should recover via CPU retry rather than failing the slice");

        assert_eq!(result.transcription.text, "ok-on-cpu");
        assert_eq!(
            fallback,
            Some(SliceGpuFallback::AllocationFailure {
                original_backend: "Vulkan0".to_string()
            })
        );
        assert_eq!(
            executor.calls_snapshot(),
            vec![
                GgmlAsrBackendPreference::Auto,
                GgmlAsrBackendPreference::CpuOnly
            ],
            "exactly one GPU attempt, then exactly one CPU retry"
        );
    }

    #[test]
    fn slice_dispatch_does_not_retry_a_non_allocation_error() {
        let dir = tempfile::tempdir().unwrap();
        let executor = std::sync::Arc::new(AlwaysFailsUnrelatedStubExecutor {
            calls: Mutex::new(0),
        });
        let (dispatch, preflight, family) = gpu_fallback_test_fixture(dir.path(), executor.clone());
        let mut decode_progress = DecodeProgress::begin(None, 1_000, false);
        let mut tracker = GpuAllocationFallbackTracker::default();

        let error = run_dispatch_once_with_progress_and_gpu_fallback(
            &dispatch,
            &preflight,
            &family,
            vec![0.0; 1_000],
            GgmlAsrExecutionOptions::default(),
            GgmlAsrBackendPreference::Auto,
            &uncancellable_execution_context_for_test(),
            &mut decode_progress,
            1_000,
            "index=1",
            &mut tracker,
        )
        .expect_err("a non-allocation failure must fail closed, not retry");

        assert!(
            error.to_string().contains("tensor 'x' is missing"),
            "original error must propagate unchanged: {error}"
        );
        assert_eq!(
            *executor.calls.lock().unwrap(),
            1,
            "no retry attempt for an error class other than the GPU allocation failure"
        );
    }

    #[test]
    fn slice_dispatch_does_not_recurse_when_an_explicit_cpu_only_request_fails() {
        // No lower-level backend exists below CPU: a CpuOnly request that
        // itself hits `BackendBufferAllocationFailed` must fail closed exactly
        // like any other CPU error, not loop back into itself.
        let dir = tempfile::tempdir().unwrap();
        // `GpuAllocFallbackStubExecutor` only fails when *not* CpuOnly, so a
        // dedicated stub is needed here to simulate a CPU-side allocation
        // failure specifically.
        struct AlwaysFailsCpuAllocExecutor {
            calls: Mutex<usize>,
        }
        impl GgmlAsrExecutor for AlwaysFailsCpuAllocExecutor {
            fn executor_id(&self) -> &'static str {
                "always-fails-cpu-alloc-stub"
            }
            fn supports_phrase_bias(&self) -> bool {
                true
            }
            fn execute(
                &self,
                request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                *self.calls.lock().unwrap() += 1;
                Err(GgmlAsrExecutionError::ExecutorFailed {
                    executor_id: "always-fails-cpu-alloc-stub",
                    adapter_id: request.selected_family.adapter_id,
                    reason: "compute buffer allocation failed (backend: CPU)".to_string(),
                })
            }
        }
        let cpu_executor = std::sync::Arc::new(AlwaysFailsCpuAllocExecutor {
            calls: Mutex::new(0),
        });
        let (dispatch, preflight, family) =
            gpu_fallback_test_fixture(dir.path(), cpu_executor.clone());
        let mut decode_progress = DecodeProgress::begin(None, 1_000, false);
        let mut tracker = GpuAllocationFallbackTracker::default();

        let error = run_dispatch_once_with_progress_and_gpu_fallback(
            &dispatch,
            &preflight,
            &family,
            vec![0.0; 1_000],
            GgmlAsrExecutionOptions::default(),
            GgmlAsrBackendPreference::CpuOnly,
            &uncancellable_execution_context_for_test(),
            &mut decode_progress,
            1_000,
            "index=1",
            &mut tracker,
        )
        .expect_err("an explicit CpuOnly allocation failure must fail closed");

        assert!(
            error
                .to_string()
                .contains("compute buffer allocation failed")
        );
        assert_eq!(
            *cpu_executor.calls.lock().unwrap(),
            1,
            "a CpuOnly request must never be retried -- there is no lower backend"
        );
    }

    #[test]
    fn slice_dispatch_switches_to_cpu_only_after_two_consecutive_fallbacks() {
        let dir = tempfile::tempdir().unwrap();
        let executor = std::sync::Arc::new(GpuAllocFallbackStubExecutor::new(
            "compute buffer allocation failed (backend: Vulkan0)",
        ));
        let (dispatch, preflight, family) = gpu_fallback_test_fixture(dir.path(), executor.clone());
        let mut decode_progress = DecodeProgress::begin(None, 3_000, false);
        let mut tracker = GpuAllocationFallbackTracker::default();

        for slice_index in 1..=3 {
            let (_result, fallback) = run_dispatch_once_with_progress_and_gpu_fallback(
                &dispatch,
                &preflight,
                &family,
                vec![0.0; 1_000],
                GgmlAsrExecutionOptions::default(),
                GgmlAsrBackendPreference::Auto,
                &uncancellable_execution_context_for_test(),
                &mut decode_progress,
                1_000,
                &format!("index={slice_index}"),
                &mut tracker,
            )
            .expect("every slice recovers via CPU");
            assert!(fallback.is_some(), "slice {slice_index} should be degraded");
        }

        // Slices 1 and 2 each got their own GPU attempt (and failed) before the
        // CPU retry; slice 3's streak already tripped, so it must go straight
        // to CPU with no third GPU attempt at all.
        assert_eq!(
            executor.calls_snapshot(),
            vec![
                GgmlAsrBackendPreference::Auto,
                GgmlAsrBackendPreference::CpuOnly,
                GgmlAsrBackendPreference::Auto,
                GgmlAsrBackendPreference::CpuOnly,
                GgmlAsrBackendPreference::CpuOnly,
            ]
        );
    }
}
