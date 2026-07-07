use std::{collections::BTreeMap, path::Path, sync::OnceLock};

use crate::NATIVE_RUNTIME_MODEL_ID_AUTO;
use crate::api::audio_io::load_wav_16khz_mono_f32_v0;
use crate::arch::OpenAsrArchitectureRegistry;
use crate::diarize::vad::SileroVadProvider;
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, install_request_backend_override,
};
use crate::longform::{
    AudioSliceKind, EnergyLongFormVadProvider, LongFormMode, LongFormVadEngine,
    LongFormVadProvider, SegmentMergePolicy, SegmentTimeDomain, SliceTranscript,
    TranscriptAssembler, plan_longform_slices,
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
use crate::models::qwen::{
    ForcedAlignItem, forced_aligner_pack, refine_word_timestamps_with_forced_aligner,
};

const DEFAULT_NATIVE_LONGFORM_AUTO_TRIGGER_SECONDS: f32 = 30.0;
const COHERE_LONGFORM_MAX_CHUNK_SECONDS: f32 = 10.0;
const COHERE_LONGFORM_OVERLAP_SECONDS: f32 = 0.0;
static NATIVE_GGML_EXECUTION_DISPATCH: OnceLock<GgmlAsrExecutionDispatch> = OnceLock::new();

// Coarse long-form transcription progress, published as a single global slot.
// The local desktop daemon transcribes one file at a time, so one slot is enough
// to drive the UI progress bar; concurrent runs on a single daemon would share
// it. Short single-pass decodes never touch it (there are no slices to count),
// so callers see None and fall back to a time-based estimate.
static PROGRESS_SLICES_DONE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static PROGRESS_SLICES_TOTAL: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// `(slices_done, slices_total)` of the in-flight native long-form run, or `None`
/// when no multi-slice run is active.
pub fn native_transcription_progress() -> Option<(usize, usize)> {
    use std::sync::atomic::Ordering;
    let total = PROGRESS_SLICES_TOTAL.load(Ordering::Relaxed);
    if total == 0 {
        return None;
    }
    let done = PROGRESS_SLICES_DONE.load(Ordering::Relaxed).min(total);
    Some((done, total))
}

/// RAII guard: publishes the slice total on creation and resets the slot on drop,
/// so normal completion, an early `?` return, or a panic all clear it.
struct LongformProgressGuard;

impl LongformProgressGuard {
    fn begin(total: usize) -> Self {
        use std::sync::atomic::Ordering;
        PROGRESS_SLICES_DONE.store(0, Ordering::Relaxed);
        PROGRESS_SLICES_TOTAL.store(total, Ordering::Relaxed);
        Self
    }

    fn advance(&self) {
        PROGRESS_SLICES_DONE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Drop for LongformProgressGuard {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;
        PROGRESS_SLICES_DONE.store(0, Ordering::Relaxed);
        PROGRESS_SLICES_TOTAL.store(0, Ordering::Relaxed);
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
/// only when the request opted into `--word-timestamps=aligned`
/// (`word_timestamps_refine`) -- refines the finished transcript's per-word
/// timestamps with the installed Qwen3-ForcedAligner-0.6B capability pack.
/// Kept as a thin wrapper rather than threading the refinement into the
/// (already long) decode/longform function: the aligner re-reads the whole
/// file and the finished transcript text, so it has no dependency on any
/// intermediate state that function computes.
pub(super) fn run_native_transcription(
    request: TranscriptionRequest,
) -> Result<Transcription, BackendError> {
    let refine = request.word_timestamps_refine;
    if refine && !request.word_timestamps {
        return Err(BackendError::WordTimestampAlignmentRequiresWordTimestamps);
    }
    let input_path = request.input_path.clone();
    let language_hint = request.language.clone();
    let transcription = run_native_transcription_impl(request)?;
    if refine {
        refine_transcription_word_timestamps_with_forced_aligner(
            transcription,
            &input_path,
            language_hint.as_deref(),
        )
    } else {
        Ok(transcription)
    }
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
        super::native_runtime_descriptor_supports_phrase_bias(&selected_family),
        request.phrase_bias.as_ref(),
    )?;
    // Diarization is supported when the model self-diarizes (e.g. cohere) or the
    // model-agnostic neural VAD + active speaker-embedder pack is available.
    let model_self_diarizes = super::native_runtime_metadata_supports_diarization(
        &runtime_preflight.metadata,
        selected_family.adapter_id,
    );
    let vad_diarization = request.diarize && !model_self_diarizes;
    if vad_diarization
        && (crate::diarize::embed::shared_embedder().is_none()
            || crate::diarize::vad::SileroVadProvider::shared().is_none())
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
    let prepared_audio = load_wav_16khz_mono_f32_v0(
        &request.input_path,
        "Native ASR Core backend",
        "Native ASR Core backend",
    )
    .map_err(|error| BackendError::NativeUnsupportedInputFormat {
        reason: error.to_string(),
    })?;

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
        selected_family.adapter_id,
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
    let mut longform_metadata: Option<TranscriptionLongFormMetadata> = None;
    if run_longform {
        let (vad_provider, vad_engine_label) = resolve_longform_vad_provider(&longform_options);
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
            // Publish per-slice progress for the UI; the guard clears the slot on
            // any exit from this long-form path.
            let slice_progress = LongformProgressGuard::begin(plan.slices.len());
            for slice in plan.slices {
                slice_progress.advance();
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
                let result = run_dispatch_once(
                    dispatch,
                    &runtime_preflight,
                    &selected_family,
                    chunk,
                    slice_options,
                    backend_preference,
                )?;
                let transcription = result.clone().into_transcription();
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
                        if let Some(prompt_token_ids) = result
                            .carry_context
                            .and_then(|context| context.prompt_token_ids)
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
                return Ok(with_reported_language(
                    apply_speaker_turns(
                        with_longform_metadata(
                            normalize_transcription_segments(
                                fallback.into_transcription(),
                                0.0,
                                audio_duration_seconds,
                            ),
                            Some(run_metadata),
                        ),
                        &speaker_turns,
                        strip_forced_word_timestamps,
                    ),
                    reported_language.clone(),
                ));
            }
            return Ok(with_reported_language(
                apply_speaker_turns(
                    with_longform_metadata(
                        normalize_transcription_segments(assembled, 0.0, audio_duration_seconds),
                        Some(run_metadata),
                    ),
                    &speaker_turns,
                    strip_forced_word_timestamps,
                ),
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
        ));
    }

    let transcription = run_dispatch_once(
        dispatch,
        &runtime_preflight,
        &selected_family,
        prepared_audio,
        request_options,
        backend_preference,
    )?;
    let normalized = normalize_transcription_segments(
        transcription.into_transcription(),
        0.0,
        audio_duration_seconds,
    );
    Ok(with_reported_language(
        apply_speaker_turns(
            with_longform_metadata(normalized, longform_metadata),
            &speaker_turns,
            strip_forced_word_timestamps,
        ),
        reported_language,
    ))
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

/// Pick the long-form VAD provider for this request, returning the provider and
/// a label for the engine that actually ran. The neural Silero model (over the
/// process-wide shared weights) is the default; the energy gate is used when
/// selected; FireRedVAD's non-streaming and Stream-VAD checkpoints (alternative
/// neural engines, neither yet the default -- see [`LongFormVadEngine::FireRed`]
/// / [`LongFormVadEngine::FireRedStream`]) are used when explicitly selected.
/// Any neural engine falls back to `*-fallback` energy when its weights are
/// unavailable so the run metadata reflects what executed. `OPENASR_VAD`
/// overrides the option (`silero`/`neural`, `energy`/`rms`, `firered`,
/// `firered-stream`).
fn resolve_longform_vad_provider(
    options: &crate::LongFormOptions,
) -> (Box<dyn LongFormVadProvider>, &'static str) {
    match vad_engine_with_env_override(options.vad_engine) {
        LongFormVadEngine::Silero => match SileroVadProvider::shared() {
            Some(provider) => (Box::new(provider), "silero"),
            None => (Box::new(EnergyLongFormVadProvider), "energy-fallback"),
        },
        LongFormVadEngine::Energy => (Box::new(EnergyLongFormVadProvider), "energy"),
        LongFormVadEngine::FireRed => match crate::diarize::vad::FireRedVadProvider::shared() {
            Some(provider) => (Box::new(provider), "firered"),
            None => (Box::new(EnergyLongFormVadProvider), "energy-fallback"),
        },
        LongFormVadEngine::FireRedStream => {
            match crate::diarize::vad::FireRedStreamVadProvider::shared() {
                Some(provider) => (Box::new(provider), "firered-stream"),
                None => (Box::new(EnergyLongFormVadProvider), "energy-fallback"),
            }
        }
    }
}

fn vad_engine_with_env_override(default: LongFormVadEngine) -> LongFormVadEngine {
    crate::diarize::vad::longform_vad_engine_env_override().unwrap_or(default)
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
    if !matches!(options.mode, LongFormMode::Off) {
        apply_longform_safety_policy(model_architecture, &mut options, &mut provenance);
    }
    NativeLongformPolicyResolution {
        options,
        provenance,
    }
}

fn apply_longform_safety_policy(
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
    if options.chunk_seconds > COHERE_LONGFORM_MAX_CHUNK_SECONDS {
        options.chunk_seconds = COHERE_LONGFORM_MAX_CHUNK_SECONDS;
        changed = true;
    }
    if options.max_chunk_seconds > COHERE_LONGFORM_MAX_CHUNK_SECONDS {
        options.max_chunk_seconds = COHERE_LONGFORM_MAX_CHUNK_SECONDS;
        changed = true;
    }
    if options.min_chunk_seconds > COHERE_LONGFORM_MAX_CHUNK_SECONDS {
        options.min_chunk_seconds = COHERE_LONGFORM_MAX_CHUNK_SECONDS;
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
            COHERE_LONGFORM_MAX_CHUNK_SECONDS
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

fn native_runtime_model_refs_match(requested: &str, runtime_source_id: &str) -> bool {
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
        format!("core.native.backend:{}", native_runtime_backend_label()),
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

fn native_runtime_backend_label() -> &'static str {
    match GgmlCpuGraphConfig::resolve_runtime_backend() {
        GgmlCpuGraphBackend::Cpu => "cpu",
        GgmlCpuGraphBackend::Metal => "metal",
        GgmlCpuGraphBackend::Gpu => "gpu",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longform_progress_guard_publishes_and_clears() {
        // No run active -> None (short single-pass decodes report nothing).
        assert_eq!(native_transcription_progress(), None);
        {
            let guard = LongformProgressGuard::begin(3);
            assert_eq!(native_transcription_progress(), Some((0, 3)));
            guard.advance();
            guard.advance();
            assert_eq!(native_transcription_progress(), Some((2, 3)));
        }
        // Guard dropped (completion / early return / panic) -> slot cleared.
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
            COHERE_LONGFORM_MAX_CHUNK_SECONDS
        );
        assert_eq!(
            resolution.options.max_chunk_seconds,
            COHERE_LONGFORM_MAX_CHUNK_SECONDS
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
            COHERE_LONGFORM_MAX_CHUNK_SECONDS
        );
        assert_eq!(
            resolution.options.max_chunk_seconds,
            COHERE_LONGFORM_MAX_CHUNK_SECONDS
        );
        assert_eq!(
            resolution.options.min_chunk_seconds,
            COHERE_LONGFORM_MAX_CHUNK_SECONDS
        );
        assert_eq!(
            resolution.options.overlap_seconds,
            COHERE_LONGFORM_OVERLAP_SECONDS
        );
        assert!(!resolution.options.carry_prompt_across_slices);
    }

    #[test]
    fn qwen_metal_longform_policy_keeps_default_chunk_size() {
        let resolution = resolve_native_longform_policy_for_backend(
            None,
            120.0,
            crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            GgmlCpuGraphBackend::Metal,
        );
        assert_eq!(resolution.options.chunk_seconds, 30.0);
        assert!(resolution.provenance.is_empty());
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

    // --- long-form VAD engine selection (Silero / Energy / FireRed) ---

    #[test]
    fn resolve_longform_vad_provider_honors_explicit_option() {
        let mut options = crate::LongFormOptions {
            vad_engine: LongFormVadEngine::Energy,
            ..crate::LongFormOptions::default()
        };
        let (_, label) = resolve_longform_vad_provider(&options);
        assert_eq!(label, "energy");

        options.vad_engine = LongFormVadEngine::Silero;
        let (_, label) = resolve_longform_vad_provider(&options);
        assert_eq!(label, "silero");

        options.vad_engine = LongFormVadEngine::FireRed;
        let (_, label) = resolve_longform_vad_provider(&options);
        assert_eq!(label, "firered");

        options.vad_engine = LongFormVadEngine::FireRedStream;
        let (_, label) = resolve_longform_vad_provider(&options);
        assert_eq!(label, "firered-stream");
    }

    #[test]
    fn openasr_vad_env_override_selects_firered() {
        let saved = std::env::var("OPENASR_VAD").ok();
        // SAFETY: sequential mutation, restored before returning; mirrors the
        // guard already used by
        // `diarize::vad::tests::realtime_vad_prefers_neural_defaults_to_neural_with_env_precedence`.
        unsafe { std::env::set_var("OPENASR_VAD", "firered") };
        let engine = vad_engine_with_env_override(LongFormVadEngine::Silero);
        assert_eq!(engine, LongFormVadEngine::FireRed);

        unsafe { std::env::set_var("OPENASR_VAD", "energy") };
        assert_eq!(
            vad_engine_with_env_override(LongFormVadEngine::FireRed),
            LongFormVadEngine::Energy
        );

        unsafe { std::env::remove_var("OPENASR_VAD") };
        // Unrecognized values fall through to the caller's default, same as
        // the realtime alias table.
        unsafe { std::env::set_var("OPENASR_VAD", "not-an-engine") };
        assert_eq!(
            vad_engine_with_env_override(LongFormVadEngine::FireRed),
            LongFormVadEngine::FireRed
        );

        match saved {
            Some(value) => unsafe { std::env::set_var("OPENASR_VAD", value) },
            None => unsafe { std::env::remove_var("OPENASR_VAD") },
        }
    }

    // --- real-audio long-form slicing smoke test: Silero vs FireRed ---

    fn jfk_wav_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav")
    }

    fn zh_wav_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/zh_sample.wav")
    }

    fn assert_engine_slices_real_audio_without_panicking(
        engine: LongFormVadEngine,
        wav_path: std::path::PathBuf,
    ) {
        let samples = load_wav_16khz_mono_f32_v0(
            &wav_path,
            "longform VAD engine smoke test",
            "longform VAD engine smoke test",
        )
        .expect("load wav fixture");

        let mut options = crate::LongFormOptions {
            mode: LongFormMode::Vad,
            vad_engine: engine,
            ..crate::LongFormOptions::default()
        };
        // Keep the fixture (11-20s) comfortably above the min chunk size so
        // `Vad` mode actually exercises slicing rather than the `total <=
        // chunk_samples` single-slice shortcut.
        options.chunk_seconds = 2.0;
        let (provider, label) = resolve_longform_vad_provider(&options);
        assert_ne!(
            label, "energy-fallback",
            "the neural engine's vendored weights must load in tests"
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
    fn silero_and_firered_both_slice_real_jfk_audio_without_panicking() {
        assert_engine_slices_real_audio_without_panicking(
            LongFormVadEngine::Silero,
            jfk_wav_path(),
        );
        assert_engine_slices_real_audio_without_panicking(
            LongFormVadEngine::FireRed,
            jfk_wav_path(),
        );
        assert_engine_slices_real_audio_without_panicking(
            LongFormVadEngine::FireRedStream,
            jfk_wav_path(),
        );
    }

    #[test]
    fn silero_and_firered_both_slice_real_zh_audio_without_panicking() {
        assert_engine_slices_real_audio_without_panicking(LongFormVadEngine::Silero, zh_wav_path());
        assert_engine_slices_real_audio_without_panicking(
            LongFormVadEngine::FireRed,
            zh_wav_path(),
        );
        assert_engine_slices_real_audio_without_panicking(
            LongFormVadEngine::FireRedStream,
            zh_wav_path(),
        );
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
}
