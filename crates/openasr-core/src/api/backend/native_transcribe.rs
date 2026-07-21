use std::{collections::BTreeMap, path::Path, sync::OnceLock, time::Instant};

use crate::NATIVE_RUNTIME_MODEL_ID_AUTO;
use crate::api::audio_io::load_wav_16khz_mono_f32_v0;
use crate::arch::{
    DEFAULT_ENCODER_SAFE_CHUNK_SECONDS, GENERAL_ARCHITECTURE_KEY, OpenAsrArchitectureRegistry,
    emits_punctuation_for_model_architecture,
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

use super::{BackendError, Transcription, TranscriptionRequest};
use crate::Segment;
use crate::WordTimestamp;
use crate::api::backend::TranscriptionLongFormMetadata;
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
/// has since been surveyed against the same industry evidence backing
/// `DEFAULT_ENCODER_SAFE_CHUNK_SECONDS` (Whisper/Moonshine/NeMo/FunASR/
/// Dolphin/Cohere all converge near 30s) and found to have no independent
/// justification, so it is unified with that default: the previous name
/// (`COHERE_LONGFORM_MAX_CHUNK_SECONDS`) was also misleading on both counts
/// (not 10s anymore, and not cohere-only -- moonshine and firered-aed carry
/// the same profile).
const CONSERVATIVE_SEQ2SEQ_LONGFORM_MAX_CHUNK_SECONDS: f32 = DEFAULT_ENCODER_SAFE_CHUNK_SECONDS;
const COHERE_LONGFORM_OVERLAP_SECONDS: f32 = 0.0;
static NATIVE_GGML_EXECUTION_DISPATCH: OnceLock<GgmlAsrExecutionDispatch> = OnceLock::new();

// Phase-aware progress for the in-flight native file transcription, published as
// a single global slot. The local desktop daemon transcribes one file at a time,
// so one slot is enough to drive the UI progress bar. The server's native path
// has no concurrency gate, though (each request's native transcription runs on
// its own `spawn_blocking` thread; see `routes/transcription.rs`), so more than
// one `run_native_transcription` can be in flight at once against this one slot.
// An owner generation (`PROGRESS_OWNER` / `NativeProgressGuard::generation`)
// keeps a second, unrelated run from clobbering the first: only the run that is
// actually reporting progress ever claims the slot, and only that run clears it
// on exit. A run whose guard is created but that fails before its first
// `publish_progress` call (e.g. model resolution errors out) never claims the
// slot and so never clears someone else's in-progress run out from under it --
// see `NativeProgressGuard` and `publish_progress` below.
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
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};

static PROGRESS_ACTIVE: AtomicBool = AtomicBool::new(false);
static PROGRESS_PHASE: AtomicU8 = AtomicU8::new(0);
static PROGRESS_FRACTION_BITS: AtomicU32 = AtomicU32::new(0);

// Owner generation of the progress slot: 0 means unclaimed. A non-zero value
// names the `NativeProgressGuard` generation currently allowed to publish to
// (and clear) the slot -- see `claim_or_check_progress_owner`.
static PROGRESS_OWNER: AtomicU64 = AtomicU64::new(0);
// Monotonically increasing counter, one draw per `NativeProgressGuard`, so
// concurrent runs never collide on the same generation number. Starts
// handing out generation 1 (0 is reserved for "unclaimed").
static PROGRESS_GENERATION_COUNTER: AtomicU64 = AtomicU64::new(0);

thread_local! {
    // The generation of the run currently executing on *this* thread, set by
    // `NativeProgressGuard::new()` and read by the `publish_*` helpers so they
    // don't need an extra parameter threaded through the whole decode/longform
    // call stack. Native transcription runs synchronously on a single thread
    // (the server's `spawn_blocking` worker, or the CLI's calling thread), so
    // this is enough to attribute a `publish_progress` call to its guard.
    static CURRENT_PROGRESS_GENERATION: Cell<u64> = const { Cell::new(0) };
}

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
    fn to_tag(self) -> u8 {
        match self {
            NativeTranscriptionPhase::Decode => 0,
            NativeTranscriptionPhase::Assemble => 1,
            NativeTranscriptionPhase::Align => 2,
        }
    }

    fn from_tag(tag: u8) -> Self {
        match tag {
            1 => NativeTranscriptionPhase::Assemble,
            2 => NativeTranscriptionPhase::Align,
            _ => NativeTranscriptionPhase::Decode,
        }
    }

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

/// Progress of the in-flight native transcription run, or `None` when no run
/// is active. Every decode call -- long-form multi-slice, forced-align
/// refine, and the short single-pass / single-slice path (a "whole file is
/// one slice" `DecodeProgress`, see `run_dispatch_once_with_progress`) --
/// reports through this slot; `None` means nothing is decoding right now, not
/// that the in-flight run is short. Only a decode that fails before its first
/// report (e.g. model resolution) leaves no signal, and the caller falls back
/// to a time-based estimate for the gap.
pub fn native_transcription_progress() -> Option<NativeTranscriptionProgress> {
    if !PROGRESS_ACTIVE.load(Ordering::Acquire) {
        return None;
    }
    let fraction = f32::from_bits(PROGRESS_FRACTION_BITS.load(Ordering::Relaxed));
    let phase = NativeTranscriptionPhase::from_tag(PROGRESS_PHASE.load(Ordering::Relaxed));
    Some(NativeTranscriptionProgress { phase, fraction })
}

/// Outcome of checking/claiming the progress slot's ownership for a generation.
enum ProgressOwnership {
    /// This generation already owns the slot; fold the report into the
    /// existing monotonic max.
    Owned,
    /// The slot was unclaimed and this generation just claimed it. The
    /// reported fraction is a fresh run's starting point, not a continuation,
    /// so it must be written directly rather than maxed against whatever a
    /// previous (already-cleared) owner left behind.
    JustAcquired,
    /// A different, still-live generation owns the slot; this generation must
    /// not touch it.
    Blocked,
}

/// Check whether `generation` owns the global progress slot, claiming it from
/// the unclaimed (`0`) state if possible. Never takes the slot away from a
/// different non-zero owner -- that owner is a live run and must not be
/// clobbered by a second, unrelated run sharing this single global slot (the
/// server has no concurrency gate on native transcription).
fn claim_or_check_progress_owner(generation: u64) -> ProgressOwnership {
    let owner = PROGRESS_OWNER.load(Ordering::Acquire);
    if owner == generation {
        return ProgressOwnership::Owned;
    }
    if owner != 0 {
        return ProgressOwnership::Blocked;
    }
    match PROGRESS_OWNER.compare_exchange(0, generation, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => ProgressOwnership::JustAcquired,
        Err(observed) if observed == generation => ProgressOwnership::Owned,
        Err(_) => ProgressOwnership::Blocked,
    }
}

/// Publish `phase` and raise the overall fraction monotonically (a later phase or a
/// further-along slice never moves the bar backward). Activates the slot on the
/// first report of a run. A no-op if a different, still-live run's generation
/// currently owns the slot: a second run sharing this one global slot must never
/// clobber another in-flight run's progress (see `NativeProgressGuard`).
fn publish_progress(phase: NativeTranscriptionPhase, fraction: f32) {
    let generation = CURRENT_PROGRESS_GENERATION.with(Cell::get);
    let clamped = fraction.clamp(0.0, 1.0);
    match claim_or_check_progress_owner(generation) {
        ProgressOwnership::Blocked => {}
        ProgressOwnership::JustAcquired => {
            // Fresh start for this run: write directly instead of maxing
            // against a stale value a previous (now-cleared) owner left.
            PROGRESS_PHASE.store(phase.to_tag(), Ordering::Relaxed);
            PROGRESS_FRACTION_BITS.store(clamped.to_bits(), Ordering::Relaxed);
            // Release so a reader that observes `active` (Acquire) also sees the phase and
            // fraction written above.
            PROGRESS_ACTIVE.store(true, Ordering::Release);
        }
        ProgressOwnership::Owned => {
            PROGRESS_PHASE.store(phase.to_tag(), Ordering::Relaxed);
            // Monotonic max on the f32 bits via a CAS loop: only ever raise the fraction.
            let mut current = PROGRESS_FRACTION_BITS.load(Ordering::Relaxed);
            loop {
                let next = f32::from_bits(current).max(clamped);
                match PROGRESS_FRACTION_BITS.compare_exchange_weak(
                    current,
                    next.to_bits(),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(observed) => current = observed,
                }
            }
            // Release so a reader that observes `active` (Acquire) also sees the phase and
            // fraction written above.
            PROGRESS_ACTIVE.store(true, Ordering::Release);
        }
    }
}

/// Enter the assembly/merge phase, raising the bar to that phase's ceiling.
fn publish_assemble_progress(with_align: bool) {
    let ceil = if with_align {
        ASSEMBLE_CEIL_WITH_ALIGN
    } else {
        ASSEMBLE_CEIL_NO_ALIGN
    };
    publish_progress(NativeTranscriptionPhase::Assemble, ceil);
}

/// Enter the forced-align refine phase, raising the bar to the align ceiling. The
/// refine is a single opaque forward pass, so the bar holds here (with the "align"
/// phase label explaining the pause) until the run completes and the slot clears.
fn publish_align_progress() {
    publish_progress(NativeTranscriptionPhase::Align, ALIGN_CEIL);
}

/// Decode-phase progress for the multi-slice long-form path. Each slice is weighted
/// by its audio sample count (not a flat per-slice tick) so the bar tracks decode
/// time -- which scales with audio duration -- rather than slice number, which makes
/// variable-length VAD slices advance the bar unevenly.
struct DecodeProgress {
    total_samples: u64,
    decoded_samples: u64,
    decode_ceil: f32,
}

impl DecodeProgress {
    fn begin(total_samples: u64, with_align: bool) -> Self {
        let decode_ceil = if with_align {
            DECODE_CEIL_WITH_ALIGN
        } else {
            DECODE_CEIL_NO_ALIGN
        };
        publish_progress(NativeTranscriptionPhase::Decode, 0.0);
        Self {
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
        publish_progress(NativeTranscriptionPhase::Decode, self.decode_ceil * ratio);
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
    decode_progress: &mut DecodeProgress,
    slice_samples: u64,
) -> Result<GgmlAsrExecutionResult, BackendError> {
    let window = decode_progress.slice_progress_window(slice_samples);
    let _token_progress_guard =
        crate::models::seq2seq_greedy_decode::install_token_step_progress_sink(
            move |step_index, max_generated_tokens| {
                if should_publish_token_step(step_index) {
                    publish_progress(
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
    )?;
    decode_progress.complete_slice(slice_samples);
    Ok(result)
}

/// RAII reset for the global progress slot: clears it on normal completion, an early
/// `?` return, or a panic, so a stale fraction never leaks into the next run. Created
/// once per `run_native_transcription` so its lifetime spans decode, assembly, and
/// the forced-align refine.
///
/// Holds a unique `generation` so this guard only clears the slot in `Drop` if it
/// is still the recognized owner (see `PROGRESS_OWNER` / `claim_or_check_progress_owner`).
/// The server has no concurrency gate on native transcription, so a second,
/// unrelated `run_native_transcription` can start and finish while a first one is
/// still decoding; without this check the second run's guard would unconditionally
/// clear the slot on both construction and drop and blank out the first run's
/// progress mid-flight. A run that never calls `publish_progress` at all (it fails
/// before reaching its first decode call, e.g. model resolution) never claims the
/// slot, so it can never steal or clear another run's ownership.
struct NativeProgressGuard {
    generation: u64,
}

impl NativeProgressGuard {
    fn new() -> Self {
        // `fetch_add` returns the pre-increment value, so the first guard gets
        // generation 1 -- 0 stays reserved for "unclaimed".
        let generation = PROGRESS_GENERATION_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        CURRENT_PROGRESS_GENERATION.with(|cell| cell.set(generation));
        Self { generation }
    }
}

impl Drop for NativeProgressGuard {
    fn drop(&mut self) {
        // Only clear this thread's attribution if it still names this guard's
        // run (defensive against any future nested-guard usage on one thread).
        CURRENT_PROGRESS_GENERATION.with(|cell| {
            if cell.get() == self.generation {
                cell.set(0);
            }
        });
        // Only clear the shared slot if this generation is still the recognized
        // owner -- i.e. this run actually published progress and no one else has
        // claimed the slot since. A run that never published (never became
        // owner) leaves the slot untouched, so it can't blank out a different,
        // still-live run sharing this global slot.
        //
        // Order matters: reset the display atomics *before* releasing
        // ownership (storing 0), not after. While `PROGRESS_OWNER` still reads
        // this generation, `claim_or_check_progress_owner` blocks every other
        // generation from claiming or publishing, so nothing can race the
        // reset below. Releasing ownership first (e.g. via a plain
        // compare-and-clear) would leave a window where a new run claims the
        // slot and publishes its own fresh fraction, only for this drop's
        // trailing `clear_progress_slot()` to immediately blank it back out.
        if PROGRESS_OWNER.load(Ordering::Acquire) == self.generation {
            clear_progress_slot();
            PROGRESS_OWNER.store(0, Ordering::Release);
        }
    }
}

fn clear_progress_slot() {
    PROGRESS_ACTIVE.store(false, Ordering::Release);
    PROGRESS_FRACTION_BITS.store(0, Ordering::Relaxed);
    PROGRESS_PHASE.store(0, Ordering::Relaxed);
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
pub(super) fn run_native_transcription(
    request: TranscriptionRequest,
) -> Result<Transcription, BackendError> {
    let refine = request.word_timestamps_refine;
    if refine && !request.word_timestamps {
        return Err(BackendError::WordTimestampAlignmentRequiresWordTimestamps);
    }
    // Spans the whole run (decode + assembly inside impl, then the punctuation
    // and forced-align post-processes below) so the progress slot is cleared
    // on every exit and the align phase advances the same monotonic bar
    // rather than running uncounted.
    let _progress = NativeProgressGuard::new();
    let input_path = request.input_path.clone();
    let language_hint = request.language.clone();
    let model_pack_path = request.model_pack_path.clone();
    let punctuate = request.punctuate;
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
    let transcription =
        apply_punctuation_stage_if_applicable(transcription, model_pack_path.as_deref(), punctuate);
    let result = if refine {
        publish_align_progress();
        refine_transcription_word_timestamps_with_forced_aligner(
            transcription,
            &input_path,
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
) -> Transcription {
    if !should_run_punctuation_stage(punctuate, model_emits_punctuation(model_pack_path)) {
        return transcription;
    }
    let Some(punc_pack_path) = resolve_firered_punc_pack_path() else {
        return transcription;
    };
    let Ok(runtime) = FireRedPuncRuntime::from_pack(&punc_pack_path) else {
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

/// Re-decodes `input_path` and calls the installed Qwen3-ForcedAligner pack
/// once over the whole finished transcript, then reassigns each segment's
/// `words` from the aligner's own per-word spans (dropping the family's
/// approximate per-word confidence -- the aligner does not produce one; never
/// inventing a value is preferred to fabricating one). Segments/text/speaker
/// attribution from the ordinary decode path are left untouched; only `words`
/// changes.
fn refine_transcription_word_timestamps_with_forced_aligner(
    mut transcription: Transcription,
    input_path: &Path,
    language_hint: Option<&str>,
) -> Result<Transcription, BackendError> {
    let pack_path = forced_aligner_pack::resolve_forced_aligner_pack_path()
        .ok_or(BackendError::WordTimestampAlignmentPackMissing { backend: "native" })?;
    let prepared_audio = load_wav_16khz_mono_f32_v0(
        input_path,
        "Native ASR Core backend",
        "Native ASR Core backend",
    )
    .map_err(|error| BackendError::NativeUnsupportedInputFormat {
        reason: error.to_string(),
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

fn run_native_transcription_impl(
    request: TranscriptionRequest,
) -> Result<Transcription, BackendError> {
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
    // Diarization is supported when the model self-diarizes (e.g. cohere) or the
    // model-agnostic neural VAD + active speaker-embedder pack is available.
    let model_self_diarizes = super::native_runtime_metadata_supports_diarization(
        &runtime_preflight.metadata,
        selected_family.self_diarizes,
    );
    let vad_diarization = request.diarize && !model_self_diarizes;
    if vad_diarization
        && (crate::diarize::embed::shared_embedder().is_none()
            || crate::diarize::vad::FireRedStreamVadProvider::shared().is_none())
    {
        // Fail closed up front rather than silently returning a speaker-less
        // transcript when the embedder or VAD model is unavailable.
        return Err(BackendError::DiarizationNotSupported { backend: "native" });
    }
    if request.diarize_speakers.is_some() {
        // Fail closed instead of silently ignoring the clustering hint: it
        // needs diarization on, and only the VAD + speaker-embedder path clusters.
        if !request.diarize {
            return Err(BackendError::DiarizeSpeakersRequiresDiarization);
        }
        if model_self_diarizes {
            return Err(BackendError::RequestOptionUnsupportedByModel {
                adapter: selected_family.adapter_id,
                option: "speakers hint",
                reason: "The model diarizes in-decoder; the exact-speaker-count hint only applies to the VAD + speaker-embedder clustering path.",
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
    let prepared_audio = load_wav_16khz_mono_f32_v0(
        &request.input_path,
        "Native ASR Core backend",
        "Native ASR Core backend",
    )
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
    let speaker_turns = if vad_diarization {
        let hint = match request.diarize_speakers {
            Some(speakers) => crate::diarize::contract::DiarizeHint::NumSpeakers(speakers),
            None => crate::diarize::contract::DiarizeHint::Auto,
        };
        compute_speaker_attribution(&prepared_audio, hint)
    } else {
        SpeakerAttribution::default()
    };

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
    request_options.word_timestamps =
        request.word_timestamps || vad_diarization || force_word_timestamps_for_segmentation;
    let strip_forced_word_timestamps =
        (vad_diarization || force_word_timestamps_for_segmentation) && !request.word_timestamps;
    request_options.word_timestamps_forced_for_diarization = strip_forced_word_timestamps;
    // OADP Phase 0: the request-level adapter path rides the execution options
    // down to the family executor (env stays the server-side fallback).
    request_options.adapter_path = request.adapter_path.clone();
    // Only the self-diarizing in-executor path (e.g. cohere) consumes this flag.
    // The VAD + speaker-embedder post-hoc path runs separately, so gating here keeps the two
    // mechanisms mutually exclusive (no future double-apply).
    request_options.diarize = request.diarize && model_self_diarizes;
    let backend_preference = execution_target_backend_preference(request.execution_target)?;
    // Installed for the whole transcribe call: the longform policy probes and
    // the provenance backend label below resolve through
    // resolve_runtime_backend(), which consults this override.
    let _backend_guard =
        install_request_backend_override(backend_preference.request_backend_override());
    // This family's Auto-mode GPU capability, so the provenance backend label
    // below resolves through the same family-aware gate the family's own
    // executor used (see `native_runtime_backend_label`'s doc comment).
    let auto_gpu_policy = crate::arch::family_auto_gpu_policy_for_model_architecture(
        selected_family.model_architecture,
    );
    let mut longform_metadata: Option<TranscriptionLongFormMetadata> = None;
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
                GgmlCpuGraphConfig::resolve_runtime_backend(),
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
                    auto_gpu_policy,
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
            // bar from the outer wrapper; the run-scoped guard clears the slot on
            // any exit. `word_timestamps_refine` reserves headroom for that phase.
            let with_align = request.word_timestamps_refine;
            let total_decode_samples: u64 = plan
                .slices
                .iter()
                .map(|slice| slice.duration_samples() as u64)
                .sum();
            let mut decode_progress = DecodeProgress::begin(total_decode_samples, with_align);
            // In-session pause/cancel control for this in-flight transcription,
            // bound to this decode thread by the caller (see
            // `install_active_transcription_control`). Checked at each slice
            // boundary: a cancel unwinds cleanly with `TranscriptionCanceled`
            // (dropping the assembler and progress guard), and a pause blocks the
            // worker here until resume or cancel. Absent (CLI / no control
            // registered) leaves the decode byte-identical to before.
            let transcription_control =
                super::transcription_control::current_transcription_control();
            let mut slice_index = 0usize;
            for slice in plan.slices {
                if let Some(control) = &transcription_control
                    && control.wait_at_slice_boundary()
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
                let result = run_dispatch_once_with_progress(
                    dispatch,
                    &runtime_preflight,
                    &selected_family,
                    chunk,
                    slice_options,
                    backend_preference,
                    &mut decode_progress,
                    slice_samples,
                )?;
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
                // both fields are consumed below and nothing needs `result`
                // as a whole afterwards, so there is nothing left to clone.
                let GgmlAsrExecutionResult {
                    transcription,
                    carry_context,
                } = result;
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
                assembler.push_slice_result(SliceTranscript {
                    slice,
                    text: transcription.text,
                    segments: transcription.segments,
                    time_domain: SegmentTimeDomain::RelativeToSliceContent,
                });
            }
            // Decode done; the merge/resegment tail below runs uncounted otherwise,
            // which is where the bar used to sit frozen at the last slice count.
            publish_assemble_progress(with_align);
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
                auto_gpu_policy,
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
                )?;
                return Ok(finalize_native_transcription(
                    fallback.into_transcription(),
                    audio_duration_seconds,
                    Some(run_metadata),
                    &speaker_turns,
                    strip_forced_word_timestamps,
                    reported_language.clone(),
                ));
            }
            return Ok(finalize_native_transcription(
                assembled,
                audio_duration_seconds,
                Some(run_metadata),
                &speaker_turns,
                strip_forced_word_timestamps,
                reported_language.clone(),
            ));
        }
        longform_metadata = Some(build_longform_metadata(
            &longform_options,
            plan_stats.chunk_count,
            plan_stats.skipped_silent_chunks,
            plan_stats.duplicate_merge_count,
            slice_kind_summary,
            timeline_kind,
            &longform_provenance,
            auto_gpu_policy,
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
    let mut single_pass_decode_progress =
        DecodeProgress::begin(single_pass_total_samples, request.word_timestamps_refine);
    let transcription = run_dispatch_once_with_progress(
        dispatch,
        &runtime_preflight,
        &selected_family,
        prepared_audio,
        request_options,
        backend_preference,
        &mut single_pass_decode_progress,
        single_pass_total_samples,
    )?;
    Ok(finalize_native_transcription(
        transcription.into_transcription(),
        audio_duration_seconds,
        longform_metadata,
        &speaker_turns,
        strip_forced_word_timestamps,
        reported_language,
    ))
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
    strip_forced_word_timestamps: bool,
    reported_language: Option<String>,
) -> Transcription {
    with_reported_language(
        apply_speaker_turns(
            with_longform_metadata(
                normalize_transcription_segments(transcription, 0.0, audio_duration_seconds),
                longform_metadata,
            ),
            speaker_turns,
            strip_forced_word_timestamps,
        ),
        reported_language,
    )
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
    let matcher = crate::diarize::enrollment::load_compatible_profile_matcher_for_active_embedder();
    let identities: BTreeMap<
        crate::diarize::contract::SpeakerId,
        crate::diarize::enrollment::SpeakerDisplayAssignment,
    > = diarization
        .centroids
        .iter()
        .filter_map(|(speaker_id, embedding)| {
            matcher.best_match(embedding).map(|profile_match| {
                (
                    *speaker_id,
                    crate::diarize::enrollment::SpeakerDisplayAssignment::from_match(
                        *speaker_id,
                        profile_match,
                    ),
                )
            })
        })
        .collect();
    if diarize_debug {
        for (speaker_id, assignment) in &identities {
            eprintln!(
                "openasr_diarize_debug stage=batch identity speaker={} display={} profile_id={}",
                speaker_id.label(),
                assignment.speaker,
                assignment.speaker_profile_id.as_deref().unwrap_or("none")
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
        GgmlCpuGraphConfig::resolve_runtime_backend(),
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
    // A self-chunking family (its dedicated executor ingests the full audio in
    // one decode with globally continuous time anchors) must never be sliced by
    // the native longform slicer -- per-window decoding would restart its time
    // anchors at zero. Force Off even for an explicit longform request, since
    // the executor never consults longform options.
    if architecture_executor_consumes_full_audio(model_architecture)
        && !matches!(options.mode, LongFormMode::Off)
    {
        provenance.push(format!(
            "native longform disabled: '{model_architecture}' executor consumes the full audio in one decode"
        ));
        options.mode = LongFormMode::Off;
    }
    if !matches!(options.mode, LongFormMode::Off) {
        apply_longform_safety_policy(model_architecture, &mut options, &mut provenance);
    }
    NativeLongformPolicyResolution {
        options,
        provenance,
    }
}

/// True when the architecture's dedicated executor ingests arbitrarily long
/// audio in a single decode (its own internal window chunking), so the shared
/// native longform slicer must stay off. Driven by the decode-policy descriptor
/// (`BuiltinDecodePolicyLongformProfile::SelfChunkingExecutorV1`) so the fact
/// lives with the family's other decode semantics, not as a magic string here.
fn architecture_executor_consumes_full_audio(model_architecture: &str) -> bool {
    resolve_builtin_decode_policy_for_architecture(model_architecture)
        .map(|policy| {
            policy.longform_profile == BuiltinDecodePolicyLongformProfile::SelfChunkingExecutorV1
        })
        .unwrap_or(false)
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
    if changed {
        provenance.push(format!(
            "core.native.longform.policy:encoder-attention-span-chunk-cap={max_safe_chunk_seconds}"
        ));
    }
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
    let Ok(runtime_ref) = parse_model_ref(runtime_source_id) else {
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

fn dispatch_error_to_backend(error: GgmlAsrExecutionError) -> BackendError {
    match error {
        GgmlAsrExecutionError::ExecutorUnavailable { .. } => BackendError::NativeFailClosed {
            reason: format!(
                "{error}. Native ggml dispatch does not fall back to non-GGUF runtime paths."
            ),
        },
        GgmlAsrExecutionError::ServeBatchUnavailable { reason, retryable } => {
            BackendError::ServeBatchUnavailable { reason, retryable }
        }
        other => BackendError::NativeFailClosed {
            reason: other.to_string(),
        },
    }
}

fn run_dispatch_once(
    dispatch: &GgmlAsrExecutionDispatch,
    runtime_preflight: &GgmlAsrRuntimeSourcePreflight,
    selected_family: &GgmlFamilyAdapterDescriptor,
    samples: Vec<f32>,
    request_options: GgmlAsrExecutionOptions,
    backend_preference: GgmlAsrBackendPreference,
) -> Result<GgmlAsrExecutionResult, BackendError> {
    let execution_request = GgmlAsrExecutionRequest {
        runtime_source_path: runtime_preflight.runtime_source.path().to_path_buf(),
        runtime_source_preflight: Some(runtime_preflight.clone()),
        selected_family: selected_family.clone(),
        prepared_audio: GgmlAsrPreparedAudio::mono_16khz(samples),
        request_options,
        backend_preference,
    };
    let _thread_override = install_request_inference_threads_override(
        execution_request.request_options.inference_threads,
    );
    let result = dispatch
        .execute(&execution_request)
        .map_err(dispatch_error_to_backend)?;
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
    auto_gpu_policy: crate::ggml_runtime::AutoGpuPolicy,
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
            native_runtime_backend_label(auto_gpu_policy)
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
            speaker_profile_id: None,
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
            speaker_profile_id: segment.speaker_profile_id,
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
            speaker_profile_id: None,
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

/// The `core.native.backend` provenance label always resolves through the
/// same family-aware gate the family's own executor used
/// (`GgmlCpuGraphConfig::resolve_family_runtime_backend`), keyed by this
/// family's `auto_gpu_policy` capability declaration. It must never call
/// `GgmlCpuGraphConfig::resolve_runtime_backend()` directly for this purpose:
/// that generic resolver reports what Auto would pick for a family with no
/// gate, which drifts from reality for any family whose policy pins (or
/// platform-scopes) Auto away from a backend -- exactly the bug that
/// produced a `core.native.backend:metal` label on a dolphin Auto request
/// that in fact ran entirely on CPU (before dolphin's own gate flipped to
/// GPU-enabled).
fn native_runtime_backend_label(
    auto_gpu_policy: crate::ggml_runtime::AutoGpuPolicy,
) -> &'static str {
    match GgmlCpuGraphConfig::resolve_family_runtime_backend(auto_gpu_policy) {
        GgmlCpuGraphBackend::Cpu => "cpu",
        GgmlCpuGraphBackend::Metal => "metal",
        GgmlCpuGraphBackend::Gpu => "gpu",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // The tests in this module that exercise `native_transcription_progress`
    // manipulate the real process-global progress statics (that is the point --
    // they are the only way to observe the slot's owner-token behavior end to
    // end), so they must not run concurrently with each other or one test's
    // writes bleed into another's assertions. `cargo test` / `cargo nextest`
    // run test functions across threads within one process, so serialize with
    // a lock rather than relying on scheduling luck.
    static PROGRESS_TEST_LOCK: Mutex<()> = Mutex::new(());

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
            AutoGpuPolicy, RequestBackendPreference, install_request_backend_override,
        };

        // Auto, family gate fully disabled (`Never` shape): must report
        // "cpu" regardless of what the generic resolver would pick.
        assert_eq!(native_runtime_backend_label(AutoGpuPolicy::Never), "cpu");

        // Auto, family gate enabled (`AllBackends` shape, every builtin
        // family but the three `ExceptMetal` ones): reports exactly what the
        // generic resolver picks -- unchanged behavior.
        let generic_auto_label = match GgmlCpuGraphConfig::resolve_runtime_backend() {
            GgmlCpuGraphBackend::Cpu => "cpu",
            GgmlCpuGraphBackend::Metal => "metal",
            GgmlCpuGraphBackend::Gpu => "gpu",
        };
        assert_eq!(
            native_runtime_backend_label(AutoGpuPolicy::AllBackends),
            generic_auto_label
        );

        // `ExceptMetal`: reports "cpu" if and only if the generic resolver
        // would have picked Metal specifically; never touches a resolved
        // Cpu or generic Gpu (CUDA/HIP/Vulkan) pick.
        let except_metal_label = native_runtime_backend_label(AutoGpuPolicy::ExceptMetal);
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
            let label = native_runtime_backend_label(AutoGpuPolicy::Never);
            assert!(label == "metal" || label == "gpu", "got {label}");
            assert_eq!(
                label,
                native_runtime_backend_label(AutoGpuPolicy::AllBackends)
            );
            assert_eq!(
                label,
                native_runtime_backend_label(AutoGpuPolicy::ExceptMetal)
            );
        }
    }

    #[test]
    fn native_progress_is_monotonic_across_phases_and_clears() {
        let _serialize = PROGRESS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // No run active -> None.
        assert_eq!(native_transcription_progress(), None);
        {
            let _guard = NativeProgressGuard::new();
            // Decode phase, weighted by sample share; a run that will forced-align
            // reserves headroom above the decode ceiling.
            let mut decode = DecodeProgress::begin(1000, true);
            let start = native_transcription_progress().expect("run is active");
            assert_eq!(start.phase, NativeTranscriptionPhase::Decode);
            assert_eq!(start.fraction, 0.0);

            decode.complete_slice(400);
            let mid = native_transcription_progress().unwrap();
            assert_eq!(mid.phase, NativeTranscriptionPhase::Decode);
            assert!(mid.fraction >= start.fraction);
            assert!((mid.fraction - DECODE_CEIL_WITH_ALIGN * 0.4).abs() < 1e-6);

            decode.complete_slice(600);
            let decoded = native_transcription_progress().unwrap();
            assert!(decoded.fraction >= mid.fraction);
            // All samples decoded -> exactly the decode ceiling.
            assert!((decoded.fraction - DECODE_CEIL_WITH_ALIGN).abs() < 1e-6);

            publish_assemble_progress(true);
            let assembled = native_transcription_progress().unwrap();
            assert_eq!(assembled.phase, NativeTranscriptionPhase::Assemble);
            assert!(assembled.fraction >= decoded.fraction);
            assert!((assembled.fraction - ASSEMBLE_CEIL_WITH_ALIGN).abs() < 1e-6);

            publish_align_progress();
            let aligning = native_transcription_progress().unwrap();
            assert_eq!(aligning.phase, NativeTranscriptionPhase::Align);
            assert!(aligning.fraction >= assembled.fraction);
            assert!(aligning.fraction <= 1.0);

            // A late lower report (e.g. an out-of-order slice) never moves the bar
            // backward; only the phase label follows the latest report.
            publish_progress(NativeTranscriptionPhase::Decode, 0.1);
            let after = native_transcription_progress().unwrap();
            assert_eq!(after.fraction, aligning.fraction);
        }
        // Guard dropped (completion / early return / panic) -> slot cleared.
        assert_eq!(native_transcription_progress(), None);
    }

    /// Regression for the owner-token fix: the server has no concurrency gate
    /// on native transcription, so a run that never calls `publish_progress`
    /// (e.g. it fails before its first decode call) can start and finish
    /// entirely while a longer, still-decoding run owns the global progress
    /// slot. Before this fix, `NativeProgressGuard::new()`/`Drop`
    /// unconditionally cleared the slot, so the second run's guard blanked
    /// out the first run's progress out from under it even though the second
    /// run never reported anything. This test uses a real background thread
    /// for the long run (a distinct generation lives in a distinct thread's
    /// `CURRENT_PROGRESS_GENERATION`) so the second run's guard, created on
    /// the test's own thread, is a genuinely different, concurrently-live
    /// generation -- not just a second guard in the same call stack.
    #[test]
    fn native_progress_concurrent_short_run_does_not_clobber_owner() {
        use std::sync::mpsc;
        use std::thread;

        let _serialize = PROGRESS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(native_transcription_progress(), None);

        // The long run claims the slot and reports partial progress, then
        // blocks (parked on `resume_rx`) until told to finish, standing in
        // for a still-decoding long file.
        let (resume_tx, resume_rx) = mpsc::channel::<()>();
        let (owner_ready_tx, owner_ready_rx) = mpsc::channel::<()>();
        let long_run = thread::spawn(move || {
            let _long_guard = NativeProgressGuard::new();
            publish_progress(NativeTranscriptionPhase::Decode, 0.4);
            owner_ready_tx.send(()).expect("test thread still waiting");
            resume_rx.recv().expect("test thread must signal resume");
        });

        owner_ready_rx
            .recv()
            .expect("long run must publish before signaling");
        let owned = native_transcription_progress().expect("long run owns the slot");
        assert_eq!(owned.phase, NativeTranscriptionPhase::Decode);
        assert!((owned.fraction - 0.4).abs() < 1e-6);

        // A second, unrelated run starts and finishes on this thread without
        // ever publishing progress (e.g. it fails before its first decode
        // call). Its guard must not touch the long run's ownership at all.
        {
            let _short_guard = NativeProgressGuard::new();
        }

        let still_owned =
            native_transcription_progress().expect("short run must not clear the long run");
        assert_eq!(still_owned, owned);

        // Only once the long run itself drops its guard does the slot clear.
        resume_tx
            .send(())
            .expect("long run still waiting to resume");
        long_run.join().expect("long run thread must not panic");
        assert_eq!(native_transcription_progress(), None);
    }

    /// Sequential (non-overlapping) runs on the owner-token slot: the second
    /// run's first report must reset the bar to its own starting point rather
    /// than being maxed against whatever the first run left behind, and each
    /// run's `Drop` must clear the slot for the next one.
    #[test]
    fn native_progress_sequential_runs_reset_start_and_clear() {
        let _serialize = PROGRESS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(native_transcription_progress(), None);

        {
            let _run1 = NativeProgressGuard::new();
            publish_progress(NativeTranscriptionPhase::Decode, 0.1);
            publish_progress(NativeTranscriptionPhase::Decode, 0.9);
            let run1_progress = native_transcription_progress().unwrap();
            assert!((run1_progress.fraction - 0.9).abs() < 1e-6);
        }
        // run1's guard dropped -> cleared before run2 starts.
        assert_eq!(native_transcription_progress(), None);

        {
            let _run2 = NativeProgressGuard::new();
            // run2's first report is lower than run1's last fraction; it must
            // become the new starting point, not be maxed against 0.9.
            publish_progress(NativeTranscriptionPhase::Decode, 0.2);
            let run2_start = native_transcription_progress().unwrap();
            assert_eq!(run2_start.phase, NativeTranscriptionPhase::Decode);
            assert!((run2_start.fraction - 0.2).abs() < 1e-6);

            // Within run2 the monotonic max still holds.
            publish_progress(NativeTranscriptionPhase::Decode, 0.05);
            let run2_after_lower = native_transcription_progress().unwrap();
            assert_eq!(run2_after_lower.fraction, run2_start.fraction);

            publish_progress(NativeTranscriptionPhase::Assemble, 0.6);
            let run2_assembled = native_transcription_progress().unwrap();
            assert_eq!(run2_assembled.phase, NativeTranscriptionPhase::Assemble);
            assert!((run2_assembled.fraction - 0.6).abs() < 1e-6);
        }
        assert_eq!(native_transcription_progress(), None);
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
        // `DecodeProgress::begin`/`complete_slice` call `publish_progress`,
        // which writes the real process-global slot, so this needs the same
        // lock + guard discipline as the other progress-slot tests above
        // (see `PROGRESS_TEST_LOCK`'s doc comment) even though the assertions
        // below only look at the pure `SliceProgressWindow` values.
        let _serialize = PROGRESS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _guard = NativeProgressGuard::new();

        let mut decode = DecodeProgress::begin(1000, false);
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
        // Same rationale as the test above: `DecodeProgress::begin` writes
        // the real global slot, so it needs the lock + guard.
        let _serialize = PROGRESS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _guard = NativeProgressGuard::new();

        // The short single-pass / single-slice path treats the whole file as
        // one slice: its window must span the entire decode phase exactly
        // like the long-form path's last slice does, not some smaller
        // fixed share -- this is what makes the two paths share one signal.
        let decode = DecodeProgress::begin(1000, true);
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
        let _serialize = PROGRESS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(native_transcription_progress(), None);

        {
            let _guard = NativeProgressGuard::new();
            let window = SliceProgressWindow {
                start_fraction: 0.0,
                span_fraction: DECODE_CEIL_NO_ALIGN,
            };
            let _sink_guard =
                crate::models::seq2seq_greedy_decode::install_token_step_progress_sink(
                    move |step_index, max_generated_tokens| {
                        if should_publish_token_step(step_index) {
                            publish_progress(
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
                    native_transcription_progress().expect("sink published at least once");
                assert!(progress.fraction >= previous);
                assert!(progress.fraction <= window.start_fraction + window.span_fraction);
                previous = progress.fraction;
            }
        }
        // Both guards dropped (sink first, then the run guard) -> slot cleared.
        assert_eq!(native_transcription_progress(), None);
    }

    /// Real-decode regression for the short-audio / single-pass progress gap
    /// this change fixes: before it, `run_native_transcription` on audio
    /// under the longform trigger (`fixtures/jfk.wav`, ~11s) never called
    /// `publish_progress` at all -- `native_transcription_progress()` stayed
    /// `None` for the whole decode, and the UI fell back to a pure time
    /// estimate with no relationship to real progress (see the recon this
    /// change is based on). Runs a real firered-aed decode on a background
    /// thread while polling the progress slot from this thread, and requires
    /// at least one snapshot strictly between 0 and the decode ceiling --
    /// proof of a genuine intermediate signal, not just an initial 0.0
    /// immediately followed by the ceiling.
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

        let _serialize = PROGRESS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(native_transcription_progress(), None);

        let pack = pack.canonicalize().expect("pack path must canonicalize");
        let wav = wav.canonicalize().expect("wav path must canonicalize");
        let request = TranscriptionRequest::new(wav, NATIVE_RUNTIME_MODEL_ID_AUTO)
            .with_model_pack_path(Some(pack));

        let decode_thread = std::thread::spawn(move || run_native_transcription(request));

        let mut saw_intermediate_signal = false;
        let mut previous_fraction = 0.0_f32;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while !decode_thread.is_finished() && std::time::Instant::now() < deadline {
            if let Some(progress) = native_transcription_progress() {
                assert_eq!(progress.phase, NativeTranscriptionPhase::Decode);
                // Monotonic even across raw polling (no lock held across
                // reads, but the CAS inside `publish_progress` guarantees a
                // reader never observes a regression).
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
        assert_eq!(native_transcription_progress(), None);
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

    #[test]
    fn self_chunking_family_forces_longform_off_even_for_long_audio() {
        // moss-transcribe-diarize ingests the full audio in one decode
        // (`SelfChunkingExecutorV1`); the native slicer must stay off so its
        // global time anchors are not restarted per VAD window.
        let implicit = resolve_native_longform_policy_for_backend(
            None,
            180.0,
            crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
            GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(implicit.options.mode, LongFormMode::Off);
        // Even an explicit longform request is overridden (the executor never
        // consults longform options).
        let explicit = resolve_native_longform_policy_for_backend(
            Some(&crate::LongFormOptions {
                mode: LongFormMode::Energy,
                ..crate::LongFormOptions::default()
            }),
            180.0,
            crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
            GgmlCpuGraphBackend::Cpu,
        );
        assert_eq!(explicit.options.mode, LongFormMode::Off);
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

    /// Data-driven production-path coverage over every builtin architecture
    /// (issue #68): a `GlobalQuadratic` encoder must never be handed a
    /// longform chunk longer than its declared safe ceiling, while
    /// `FixedWindow` (whisper) and `LocalChunked` (zipformer) architectures
    /// need no additional cap and keep the unmodified 120s default. All nine
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
            let resolution = resolve_native_longform_policy_for_backend(
                None,
                120.0,
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
                    assert_eq!(
                        resolution.options.max_chunk_seconds, 120.0,
                        "'{}' (FixedWindow/LocalChunked) must keep the unmodified default",
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
                text: "a b".to_string(),
                segments: vec![
                    Segment {
                        start: 0.8,
                        end: 1.0,
                        text: "a".to_string(),
                        speaker: None,
                        speaker_label: None,
                        speaker_profile_id: None,
                        words: Vec::new(),
                    },
                    Segment {
                        start: 0.5,
                        end: 0.7,
                        text: "b".to_string(),
                        speaker: None,
                        speaker_label: None,
                        speaker_profile_id: None,
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
                text: "long transcript".to_string(),
                segments: vec![Segment {
                    start: 0.0,
                    end: 1.0,
                    text: "long transcript".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_profile_id: None,
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
                text: "near full".to_string(),
                segments: vec![Segment {
                    start: 0.0,
                    end: 11.5,
                    text: "near full".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_profile_id: None,
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
    #[ignore = "host-local: requires the X-ASR q8_0 pack, the wespeaker diarize pack, and tmp/diar-real-case-1781172161.wav"]
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
        .with_diarization(true);
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
            speaker_profile_id: None,
            words: vec![WordTimestamp {
                word: text.to_string(),
                start,
                end,
                confidence: Some(0.9),
            }],
        }
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
            text: "hello world".to_string(),
            segments: vec![Segment {
                start: 0.0,
                end: 1.0,
                text: "hello world".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            }],
            longform: None,
            language: None,
        };
        let unchanged = apply_punctuation_stage_if_applicable(transcription.clone(), None, true);
        assert_eq!(unchanged, transcription);

        // Explicit opt-out short-circuits before any pack resolution too.
        let unchanged = apply_punctuation_stage_if_applicable(transcription.clone(), None, false);
        assert_eq!(unchanged, transcription);
    }
}
