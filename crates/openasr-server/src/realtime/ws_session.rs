//! WebSocket realtime session driver (`WsSession`) and utterance helpers.
//!
//! Pure code-motion from `realtime.rs`; no behavior changes.

use super::*;

pub(crate) struct WsSession {
    pub(crate) runtime: ServerRuntime,
    pub(crate) distribution: DistributionContext,
    pub(crate) session_id: RealtimeSessionId,
    /// The single connection-lifetime sequencer. Every envelope leaving this
    /// session is (re-)stamped here at the edge, so producers (controller,
    /// native sessions) never need coordinated counters.
    pub(crate) sequencer: RealtimeEventSequencer,
    /// Producer event_id -> client-visible event_id for envelopes already
    /// re-stamped at the edge; rewrites `revises_event_id` references.
    pub(crate) emitted_event_ids: std::collections::HashMap<String, String>,
    pub(crate) emitted_event_id_order: VecDeque<String>,
    pub(crate) controller: Option<RealtimeSessionController>,
    /// Dedicated-thread owner of the native streaming session, when this session
    /// is on the native streaming (Path B) route. The session itself lives on the
    /// worker thread; the WS task talks to it request/response.
    pub(crate) native_streaming: Option<NativeStreamingDecodeWorker>,
    pub(crate) native_had_speech_since_last_poll: bool,
    pub(crate) native_poll_outstanding: usize,
    pub(crate) native_command_watchdogs: VecDeque<(NativeStreamingCommandKind, Instant)>,
    /// Watchdog bound for a single native-streaming decode round-trip.
    pub(crate) native_decode_timeout: Duration,
    pub(crate) event_sender: mpsc::Sender<RealtimeEventEnvelope>,
    pub(crate) backend_jobs: Option<mpsc::Sender<RealtimeBackendWorkerMessage>>,
    pub(crate) backend_results: Option<mpsc::Receiver<BackendResult>>,
    pub(crate) backend_result_sender: Option<mpsc::Sender<BackendResult>>,
    pub(crate) backend_cancelled: Arc<AtomicBool>,
    pub(crate) pending_backend_jobs: usize,
    pub(crate) audio_frames: mpsc::Sender<RealtimeAudioFrame>,
    pub(crate) audio_frame_receiver: mpsc::Receiver<RealtimeAudioFrame>,
    pub(crate) carry: Vec<u8>,
    pub(crate) frame_duration_ms: u32,
    pub(crate) frame_byte_len: usize,
    pub(crate) next_frame_seq: u64,
    pub(crate) next_frame_start_ms: u64,
    pub(crate) language: Option<String>,
    pub(crate) task: Option<openasr_core::TranscriptionTask>,
    pub(crate) prompt: Option<String>,
    pub(crate) phrase_bias: Option<openasr_core::PhraseBiasConfig>,
    pub(crate) inference_threads: Option<u16>,
    pub(crate) execution_target: Option<openasr_core::ExecutionTarget>,
    pub(crate) word_timestamps: bool,
    pub(crate) source_name: Option<String>,
    pub(crate) history_text: Vec<String>,
    pub(crate) history_duration_ms: u64,
    pub(crate) history_recorded: bool,
    pub(crate) record_history: bool,
    pub(crate) backend_failed: bool,
    pub(crate) closed: bool,
    pub(crate) captured_audio_frames: VecDeque<RealtimeAudioFrame>,
    /// Per-session streaming diarizer, built at session.start when the client
    /// requests `diarize=true`; `None` when not requested. Capability and pack
    /// availability are checked (fail-closed) during configure.
    pub(crate) streaming_diarizer: Option<openasr_core::diarize::streaming::StreamingDiarizer>,
    /// Native true-streaming speaker-change detector. It may use a faster
    /// same-space embedder than `streaming_diarizer`, but only proposes split
    /// points; labels still come from the normal streaming diarizer.
    pub(crate) native_speaker_change_detector:
        Option<openasr_core::diarize::streaming::StreamingSpeakerChangeDetector>,
    #[cfg(test)]
    pub(crate) test_streaming_diarizer_embedder:
        Option<&'static dyn openasr_core::diarize::embed::SpeakerEmbedder>,
    /// Speaker label per utterance, computed at queue time (has the audio) and
    /// consumed when the backend transcript comes back.
    pub(crate) pending_utterance_speakers: std::collections::HashMap<
        TranscriptUtteranceId,
        openasr_core::diarize::enrollment::SpeakerDisplayAssignment,
    >,
    /// Native true-streaming diarization: speech-gated samples of the current
    /// utterance, retained only while `streaming_diarizer` is active and
    /// bounded by [`NATIVE_DIARIZE_MAX_RETAINED_SAMPLES`].
    pub(crate) native_diarize_samples: Vec<f32>,
    /// Small rolling frame buffer used to backfill the VAD start debounce for
    /// native diarization. It mirrors the fallback path's utterance pre-roll.
    pub(crate) native_diarize_preroll_frames: VecDeque<RealtimeAudioFrame>,
    /// Speaker audio retained at finalize time, waiting for the worker's
    /// terminal transcript of the matching utterance. The worker processes
    /// commands in order, so utterances complete in finalize order (FIFO).
    pub(crate) pending_native_speaker_labels: VecDeque<NativePendingSpeakerSlot>,
    /// FIFO speaker slots for native `SplitUtterance` commands. Every
    /// diarizing split command appends exactly one slot; non-empty terminal
    /// transcripts resolve the retained audio to a speaker label, while empty
    /// terminal transcripts drop the slot unlabelled. The worker returns split
    /// outcomes in command order, so this prevents max-duration and
    /// speaker-change splits from stealing labels.
    pub(crate) pending_native_split_speaker_slots: VecDeque<NativePendingSpeakerSlot>,
    /// Utterances already labelled, so every transcript event of the same
    /// utterance (post-final revisions included) reuses one label.
    pub(crate) native_speaker_by_utterance: std::collections::HashMap<
        TranscriptUtteranceId,
        Option<openasr_core::diarize::enrollment::SpeakerDisplayAssignment>,
    >,
    /// Absolute-time anchors for `native_diarize_samples`: (sample offset at
    /// frame start, frame `start_ms`). Maps a change-detector split sample
    /// index back to stream time for word-level reattribution.
    pub(crate) native_diarize_sample_spans: Vec<(usize, u64)>,
    /// Acoustic change-point estimate (absolute stream ms) per in-flight
    /// diarizing `SplitUtterance` command, in command order; `None` for
    /// splits that are not speaker-change splits (max-utterance).
    pub(crate) pending_native_split_change_points: VecDeque<Option<u64>>,
    /// Finalized mid-utterance segments that went to the client without a
    /// speaker because their utterance's label had not bound yet; once the
    /// label binds, each is revised retroactively with the speaker attached.
    pub(crate) native_speakerless_finals: Vec<PendingSpeakerRevision>,
    /// Reattributed tail pieces emitted at a speaker-change split; they
    /// acoustically belong to the NEXT utterance's speaker, so they are
    /// revised once that label binds (`utterance_id` here is the OLD
    /// utterance the text was carved from).
    pub(crate) pending_split_tail_relabels: VecDeque<PendingSpeakerRevision>,
    pub(crate) translation: Option<RealtimeTranslationLane>,
    #[cfg(test)]
    pub(crate) test_translation_worker: Option<TranslationWorkerHook>,
    /// When set together with `test_translation_worker`, the worker is built
    /// through the asynchronous thread-local init path (mirroring the real
    /// Hy-MT2 cold load); this hook runs as the worker initialization.
    #[cfg(test)]
    pub(crate) test_translation_worker_init: Option<TranslationWorkerInitHook>,
}

pub(crate) type TranslationWorkerHook = Arc<
    dyn Fn(TranslationRequest) -> Result<TranslationWorkerOutput, TranslationQueueError>
        + Send
        + Sync,
>;

pub(crate) type TranslationWorkerInitHook =
    Arc<dyn Fn() -> Result<(), TranslationQueueError> + Send + Sync>;

pub(crate) struct RealtimeTranslationLane {
    session: TranslationSession,
    segmenter: ClauseSegmenter,
    gates: HashMap<ClauseId, StabilityGate>,
    clause_meta: HashMap<ClauseId, TranslationClauseMeta>,
    retired_clause_ids: HashSet<ClauseId>,
    retired_clause_order: VecDeque<ClauseId>,
    source_segments: Vec<TranslationSourceSegmentState>,
    model_id: String,
    target_lang: TargetLang,
    provisional: bool,
    /// Whether the one-shot `translation.status` ready event was already
    /// emitted (true at build time when the worker was ready at birth, e.g.
    /// test workers).
    ready_announced: bool,
}

#[derive(Debug, Clone)]
struct TranslationSourceSegmentState {
    source_segment_id: String,
    text: String,
    finalized: bool,
    start_ms: u64,
    end_ms: u64,
}

#[derive(Debug, Clone)]
struct TranslationSourceEvent {
    source_segment_id: String,
    text: String,
    finalized: bool,
    start_ms: u64,
    end_ms: u64,
}

#[derive(Debug, Clone)]
struct TranslationClauseMeta {
    source_segment_id: String,
    source_version: u64,
    replaces_clause_id: Option<ClauseId>,
    start_ms: u64,
    end_ms: u64,
    stability: f32,
}

pub(crate) enum NativePendingSpeakerSlot {
    DeferredSamples(Vec<f32>),
    #[cfg(test)]
    Resolved(Option<openasr_core::diarize::enrollment::SpeakerDisplayAssignment>),
}

/// A finalized transcript line emitted before its speaker was known. Once a
/// label binds, a `transcript.revision` re-sends the same text with the
/// speaker attached (the desktop applies it per `(utterance, segment)` line).
#[derive(Debug, Clone)]
pub(crate) struct PendingSpeakerRevision {
    pub(crate) utterance_id: TranscriptUtteranceId,
    pub(crate) segment_id: TranscriptSegmentId,
    /// Revision of the original final; the retroactive revision uses `+1`.
    pub(crate) revision: u64,
    pub(crate) text: String,
    pub(crate) start_ms: u64,
    pub(crate) end_ms: u64,
    pub(crate) language: Option<String>,
    pub(crate) words: Vec<RealtimeTranscriptWord>,
    /// Client-visible event id of the original final (for `revises_event_id`).
    pub(crate) client_event_id: Option<String>,
}

/// Snapshot of an outgoing transcript final, captured before the envelope is
/// consumed by the emit path, for retroactive speaker revision bookkeeping.
struct FinalTranscriptSnapshot {
    utterance_id: TranscriptUtteranceId,
    segment_id: TranscriptSegmentId,
    revision: u64,
    text: String,
    start_ms: u64,
    end_ms: u64,
    language: Option<String>,
    words: Vec<RealtimeTranscriptWord>,
    speaker: Option<String>,
    speaker_label: Option<String>,
    speaker_profile_id: Option<String>,
}

/// Upper bound on retained per-utterance speech for native true-streaming
/// diarization: 30 s @ 16 kHz mono f32 (~1.9 MB). Speaker embedding
/// saturates well before that; an utterance running past the cap keeps its
/// first 30 s of speech.
const NATIVE_DIARIZE_MAX_RETAINED_SAMPLES: usize = 16_000 * 30;
const NATIVE_DIARIZE_PREROLL_MS: u64 = 500;
/// Bound on how much trailing speech a speaker-change split may reattribute
/// to the new speaker (≈ the change detector's worst-case detection lag:
/// 2.5 s analysis window + 1 s re-check cadence, conservatively capped).
const NATIVE_SPLIT_REATTRIBUTION_MAX_TAIL_MS: u64 = 3_000;
/// Cap on tracked speaker-pending finals; an utterance produces at most a
/// handful of sentence-cut segments before its terminal label binds.
const MAX_PENDING_SPEAKER_REVISIONS: usize = 16;
const MAX_TRANSLATION_SOURCE_SEGMENTS: usize = 128;
const MAX_TRANSLATION_CLAUSE_META: usize = 256;
const MAX_TRANSLATION_RETIRED_CLAUSES: usize = 256;
const TRANSLATION_TOMBSTONE_REASON_SOURCE_RETIRED: &str = "source_clause_retired";

/// The single SpeechBoundaryEvent -> client-event mapping. Both VAD paths
/// (fallback controller-routed, native WS-edge) consume it; adding a boundary
/// variant only requires updating this one match.
enum VadBoundaryEvent {
    Vad(RealtimeVadEvent),
    Error(RealtimeErrorEvent),
}

fn vad_boundary_event(boundary: &SpeechBoundaryEvent) -> VadBoundaryEvent {
    match boundary {
        SpeechBoundaryEvent::SpeechStarted {
            utterance_id,
            start_ms,
        } => VadBoundaryEvent::Vad(RealtimeVadEvent::SpeechStarted(VadSpeechStartedEvent {
            utterance_id: utterance_id.clone(),
            start_ms: *start_ms,
        })),
        SpeechBoundaryEvent::SpeechStopped {
            utterance_id,
            start_ms,
            end_ms,
        }
        | SpeechBoundaryEvent::MaxUtterance {
            utterance_id,
            start_ms,
            end_ms,
        } => VadBoundaryEvent::Vad(RealtimeVadEvent::SpeechStopped(VadSpeechStoppedEvent {
            utterance_id: utterance_id.clone(),
            start_ms: *start_ms,
            end_ms: *end_ms,
        })),
        SpeechBoundaryEvent::NoSpeechTimeout { timeout_ms, .. } => {
            VadBoundaryEvent::Error(RealtimeErrorEvent {
                code: RealtimeErrorCode::NoSpeechTimeout,
                message: format!("No speech detected within {timeout_ms} ms."),
                recoverable: true,
            })
        }
    }
}

/// The one PCM16 -> `[-1, 1]` f32 normalization both diarize retention paths
/// share (fallback utterance flattening and native speech-gated retention).
pub(crate) fn pcm16_sample_to_f32(sample: i16) -> f32 {
    sample as f32 / 32768.0
}

/// Absolute stream time of a sample index in the speech-gated diarize buffer,
/// from the per-frame anchors recorded at retention time. `None` without
/// anchors (e.g. a buffer assembled outside the frame path).
pub(crate) fn diarize_sample_abs_ms(spans: &[(usize, u64)], sample_index: usize) -> Option<u64> {
    let position = spans.partition_point(|(offset, _)| *offset <= sample_index);
    let (offset, start_ms) = spans.get(position.checked_sub(1)?)?;
    Some(start_ms.saturating_add(((sample_index - offset) / 16) as u64))
}

/// Rebases the diarize-buffer anchors after the first `split_sample` samples
/// were carved off the front, so later splits keep mapping to stream time.
pub(crate) fn rebase_diarize_sample_spans(
    spans: Vec<(usize, u64)>,
    split_sample: usize,
) -> Vec<(usize, u64)> {
    let mut rebased = Vec::with_capacity(spans.len());
    for (offset, start_ms) in spans {
        if offset >= split_sample {
            rebased.push((offset - split_sample, start_ms));
        } else {
            // Anchors are offset-ascending, so pre-cut anchors all precede
            // post-cut ones; the last anchor straddling the cut becomes the
            // new buffer head.
            rebased.clear();
            rebased.push((
                0,
                start_ms.saturating_add(((split_sample - offset) / 16) as u64),
            ));
        }
    }
    rebased
}

/// Whether this envelope is a terminal transcript for its utterance (the
/// event that binds queued speaker labels and consumes split change points).
fn is_terminal_transcript_envelope(envelope: &RealtimeEventEnvelope) -> bool {
    match &envelope.event {
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(event)) => event.is_final,
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Revision(event)) => event.is_final,
        _ => false,
    }
}

/// Snapshot of a non-empty transcript FINAL for retroactive speaker
/// attribution bookkeeping; `None` for everything else.
fn snapshot_final_transcript(envelope: &RealtimeEventEnvelope) -> Option<FinalTranscriptSnapshot> {
    let RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(event)) = &envelope.event else {
        return None;
    };
    if !event.is_final || event.text.trim().is_empty() {
        return None;
    }
    Some(FinalTranscriptSnapshot {
        utterance_id: event.utterance_id.clone(),
        segment_id: event.segment_id.clone(),
        revision: event.revision,
        text: event.text.clone(),
        start_ms: event.start_ms,
        end_ms: event.end_ms,
        language: event.language.clone(),
        words: event.words.clone(),
        speaker: event.speaker.clone(),
        speaker_label: event.speaker_label.clone(),
        speaker_profile_id: event.speaker_profile_id.clone(),
    })
}

/// Extract only the VAD speech span from a fallback buffered utterance for
/// speaker embedding. The ASR WAV still uses the original frames with pre-roll
/// and hangover; this trims those padding frames only for diarization so the
/// score space matches enrollment/native speech-gated audio.
pub(crate) fn utterance_speech_samples_f32(utterance: &BufferedUtterance) -> Vec<f32> {
    let mut samples = Vec::new();
    for frame in &utterance.frames {
        let frame_start_ms = frame.start_ms;
        let frame_end_ms = frame.end_ms();
        let overlap_start_ms = frame_start_ms.max(utterance.start_ms);
        let overlap_end_ms = frame_end_ms.min(utterance.end_ms);
        if overlap_end_ms <= overlap_start_ms {
            continue;
        }
        let sample_rate = frame.format.sample_rate_hz as usize;
        let start_offset = ((overlap_start_ms - frame_start_ms) as usize * sample_rate) / 1_000;
        let end_offset = ((overlap_end_ms - frame_start_ms) as usize * sample_rate)
            .div_ceil(1_000)
            .min(frame.samples().len());
        if end_offset <= start_offset {
            continue;
        }
        samples.extend(
            frame.samples()[start_offset..end_offset]
                .iter()
                .map(|sample| pcm16_sample_to_f32(*sample)),
        );
    }
    samples
}

fn realtime_phrase_bias_rejection_message(
    runtime: &ServerRuntime,
    capability: openasr_core::api::backend::BackendFeatureCapability,
) -> String {
    if runtime.backend == openasr_core::BackendKind::Native
        && let Some(path) = runtime.model_pack_path.as_deref()
        && let Some(adapter) = native_runtime_model_adapter_for_path(path)
    {
        return format!(
            "Realtime phrase bias / hotword boosting is not supported by the active native model family '{}' ({}); session.start was rejected instead of silently ignoring hotwords.",
            adapter.model_family(),
            adapter.adapter_id()
        );
    }

    capability
        .reason
        .unwrap_or("Realtime phrase bias / hotword boosting is not supported by this backend.")
        .to_string()
}

fn session_language_is_chinese(language: Option<&str>) -> bool {
    let Some(language) = language else {
        return false;
    };
    matches!(
        language.trim().to_ascii_lowercase().as_str(),
        "zh" | "zh-cn" | "zh-hans" | "cmn"
    )
}

impl TranslationSourceEvent {
    fn from_transcript(transcript: RealtimeTranscriptEvent) -> Option<Self> {
        match transcript {
            RealtimeTranscriptEvent::Partial(event) => Some(Self {
                source_segment_id: event.segment_id.0,
                text: event.text,
                finalized: false,
                start_ms: event.start_ms,
                end_ms: event.end_ms,
            }),
            RealtimeTranscriptEvent::Final(event) => Some(Self {
                source_segment_id: event.segment_id.0,
                text: event.text,
                finalized: true,
                start_ms: event.start_ms,
                end_ms: event.end_ms,
            }),
            RealtimeTranscriptEvent::Revision(event) => Some(Self {
                source_segment_id: event.segment_id.0,
                text: event.text,
                finalized: event.is_final,
                start_ms: event.start_ms,
                end_ms: event.end_ms,
            }),
        }
    }
}

impl RealtimeTranslationLane {
    fn observe_source(
        &mut self,
        source: TranslationSourceEvent,
    ) -> Result<Vec<RealtimeEvent>, String> {
        if source.text.trim().is_empty() && self.source_segments.is_empty() {
            return Ok(Vec::new());
        }
        self.upsert_source_segment(&source);
        let full_text = self
            .source_segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<String>();
        let update = if source.finalized {
            self.segmenter.push_final(&full_text)
        } else {
            self.segmenter.push_partial(&full_text)
        };
        let tombstones = self.apply_retired_clauses(&update)?;
        for segment in update.segments {
            self.enqueue_clause_segment(segment, &source)?;
        }
        self.prune_source_segments();
        self.prune_clause_meta();
        Ok(tombstones)
    }

    fn upsert_source_segment(&mut self, source: &TranslationSourceEvent) {
        if let Some(existing) = self
            .source_segments
            .iter_mut()
            .find(|segment| segment.source_segment_id == source.source_segment_id)
        {
            existing.text = source.text.clone();
            existing.finalized = source.finalized;
            existing.start_ms = source.start_ms;
            existing.end_ms = source.end_ms;
            return;
        }
        self.source_segments.push(TranslationSourceSegmentState {
            source_segment_id: source.source_segment_id.clone(),
            text: source.text.clone(),
            finalized: source.finalized,
            start_ms: source.start_ms,
            end_ms: source.end_ms,
        });
    }

    fn apply_retired_clauses(
        &mut self,
        update: &openasr_core::ClauseSegmentationUpdate,
    ) -> Result<Vec<RealtimeEvent>, String> {
        if update.retired_clause_ids.is_empty() {
            return Ok(Vec::new());
        }
        self.session
            .retire_clause_ids(update.retired_clause_ids.iter().copied())
            .map_err(|error| format!("Realtime translation retire failed: {error}"))?;
        self.remember_retired_clause_ids(update.retired_clause_ids.iter().copied());

        let replaced_clause_ids = update
            .segments
            .iter()
            .filter_map(|segment| segment.replaces_clause_id)
            .collect::<HashSet<_>>();
        let mut tombstones = Vec::new();
        for clause_id in &update.retired_clause_ids {
            self.gates.remove(clause_id);
            let meta = self.clause_meta.remove(clause_id);
            if replaced_clause_ids.contains(clause_id) {
                continue;
            }
            if let Some(meta) = meta {
                tombstones.push(RealtimeEvent::Translation(
                    RealtimeTranslationEvent::Tombstone(RealtimeTranslationTombstone {
                        clause_id: clause_id.to_string(),
                        source_segment_id: meta.source_segment_id,
                        source_version: meta.source_version,
                        target_lang: self.target_lang.as_str().to_string(),
                        reason: TRANSLATION_TOMBSTONE_REASON_SOURCE_RETIRED.to_string(),
                        is_final: true,
                        model: self.model_id.clone(),
                    }),
                ));
            }
        }
        Ok(tombstones)
    }

    fn remember_retired_clause_ids(&mut self, clause_ids: impl IntoIterator<Item = ClauseId>) {
        for clause_id in clause_ids {
            if self.retired_clause_ids.insert(clause_id) {
                self.retired_clause_order.push_back(clause_id);
            }
        }
        while self.retired_clause_ids.len() > MAX_TRANSLATION_RETIRED_CLAUSES {
            let Some(oldest) = self.retired_clause_order.pop_front() else {
                return;
            };
            self.retired_clause_ids.remove(&oldest);
        }
    }

    fn prune_source_segments(&mut self) {
        while self.source_segments.len() > MAX_TRANSLATION_SOURCE_SEGMENTS {
            self.source_segments.remove(0);
        }
    }

    fn prune_clause_meta(&mut self) {
        while self.clause_meta.len() > MAX_TRANSLATION_CLAUSE_META {
            let Some(oldest) = self.clause_meta.keys().min().copied() else {
                return;
            };
            self.clause_meta.remove(&oldest);
        }
    }

    fn enqueue_clause_segment(
        &mut self,
        segment: ClauseSegment,
        source: &TranslationSourceEvent,
    ) -> Result<(), String> {
        let finalized = segment.status == ClauseStatus::Finalized;
        if !finalized && !self.provisional {
            return Ok(());
        }
        // Punctuation-only clauses (e.g. a lone "。" left over after an ASR
        // final consumed the words before it) carry nothing to translate;
        // spending a worker decode on them just steals MT throughput.
        if !segment.text.chars().any(char::is_alphanumeric) {
            return Ok(());
        }
        let gate = self.gates.entry(segment.clause_id).or_default();
        let decision = gate.observe(StabilityGateInput {
            source_text: &segment.text,
            observed_at_ms: source.end_ms,
            finalized,
        });
        if !decision.should_enqueue {
            return Ok(());
        }
        self.clause_meta.insert(
            segment.clause_id,
            TranslationClauseMeta {
                source_segment_id: source.source_segment_id.clone(),
                source_version: segment.source_version,
                replaces_clause_id: segment.replaces_clause_id,
                start_ms: source.start_ms,
                end_ms: source.end_ms,
                stability: if finalized { 1.0 } else { decision.stability },
            },
        );
        self.session
            .enqueue(TranslationRequest {
                clause_id: segment.clause_id,
                replaces_clause_id: segment.replaces_clause_id,
                source_version: segment.source_version,
                source_text: segment.text,
                finalized,
                revised: segment.revised,
                target_lang: self.target_lang,
                finalized_context: Vec::new(),
            })
            .map(|_| ())
            .map_err(|error| format!("Realtime translation enqueue failed: {error}"))
    }
}

impl WsSession {
    #[cfg(test)]
    pub(crate) fn new(
        runtime: ServerRuntime,
        distribution: DistributionContext,
        event_sender: mpsc::Sender<RealtimeEventEnvelope>,
    ) -> Self {
        Self::new_with_history(runtime, distribution, event_sender, true)
    }

    pub(crate) fn new_with_history(
        runtime: ServerRuntime,
        distribution: DistributionContext,
        event_sender: mpsc::Sender<RealtimeEventEnvelope>,
        record_history: bool,
    ) -> Self {
        let (audio_frames, audio_frame_receiver) = mpsc::channel(AUDIO_FRAME_QUEUE_CAPACITY);
        let format = RealtimeAudioFormat::pcm16_mono_16khz();
        let session_id = next_session_id("rt_ws");
        Self {
            runtime,
            distribution,
            sequencer: RealtimeEventSequencer::new(session_id.clone()),
            emitted_event_ids: std::collections::HashMap::new(),
            emitted_event_id_order: VecDeque::new(),
            session_id,
            controller: None,
            native_streaming: None,
            native_had_speech_since_last_poll: false,
            native_poll_outstanding: 0,
            native_command_watchdogs: VecDeque::new(),
            native_decode_timeout: backend_result_timeout(),
            event_sender,
            backend_jobs: None,
            backend_results: None,
            backend_result_sender: None,
            backend_cancelled: Arc::new(AtomicBool::new(false)),
            pending_backend_jobs: 0,
            audio_frames,
            audio_frame_receiver,
            carry: Vec::new(),
            frame_duration_ms: DEFAULT_FRAME_DURATION_MS,
            frame_byte_len: format
                .sample_count_for_duration_ms(DEFAULT_FRAME_DURATION_MS)
                .expect("default realtime frame duration is valid")
                * 2,
            next_frame_seq: 1,
            next_frame_start_ms: 0,
            language: None,
            task: None,
            prompt: None,
            phrase_bias: None,
            inference_threads: None,
            execution_target: None,
            word_timestamps: false,
            source_name: None,
            history_text: Vec::new(),
            history_duration_ms: 0,
            history_recorded: false,
            record_history,
            backend_failed: false,
            closed: false,
            captured_audio_frames: VecDeque::new(),
            streaming_diarizer: None,
            native_speaker_change_detector: None,
            #[cfg(test)]
            test_streaming_diarizer_embedder: None,
            pending_utterance_speakers: std::collections::HashMap::new(),
            native_diarize_samples: Vec::new(),
            native_diarize_preroll_frames: VecDeque::new(),
            pending_native_speaker_labels: VecDeque::new(),
            pending_native_split_speaker_slots: VecDeque::new(),
            native_speaker_by_utterance: std::collections::HashMap::new(),
            native_diarize_sample_spans: Vec::new(),
            pending_native_split_change_points: VecDeque::new(),
            native_speakerless_finals: Vec::new(),
            pending_split_tail_relabels: VecDeque::new(),
            translation: None,
            #[cfg(test)]
            test_translation_worker: None,
            #[cfg(test)]
            test_translation_worker_init: None,
        }
    }

    pub(crate) async fn emit_capabilities(&mut self) -> Result<(), ()> {
        self.emit_event(RealtimeEvent::Lifecycle(
            RealtimeLifecycleEvent::SessionCapabilities(SessionCapabilitiesEvent {
                capabilities: self.realtime_capabilities(),
                audio_format: RealtimeAudioFormat::pcm16_mono_16khz(),
                frame_duration_ms: self.frame_duration_ms,
                frame_byte_len: self.frame_byte_len,
                max_message_bytes: MAX_WS_MESSAGE_BYTES,
            }),
        ))
        .await
    }

    fn realtime_capabilities(&self) -> RealtimeBackendCapabilities {
        #[cfg(test)]
        {
            let mut capabilities = realtime_capabilities_for_runtime_and_distribution(
                &self.runtime,
                &self.distribution,
            );
            if self.test_translation_worker.is_some() {
                capabilities.translation =
                    openasr_core::RealtimeTranslationCapability::installed_hymt2();
            }
            capabilities
        }
        #[cfg(not(test))]
        {
            realtime_capabilities_for_runtime_and_distribution(&self.runtime, &self.distribution)
        }
    }

    /// Stamps a locally-produced event onto the connection sequence and sends.
    pub(crate) async fn emit_event(&mut self, event: RealtimeEvent) -> Result<(), ()> {
        let envelope = self.sequencer.next(event, timestamp_now());
        send_event(&self.event_sender, envelope).await
    }

    /// Re-stamps an envelope produced by another sequencer (the session
    /// controller or a native streaming session) onto the connection sequence
    /// and sends it. Producer event ids are remembered so a later
    /// `transcript.revision` can have its `revises_event_id` rewritten to the
    /// id the client actually saw.
    pub(crate) async fn emit_envelope(
        &mut self,
        envelope: RealtimeEventEnvelope,
    ) -> Result<(), ()> {
        let producer_event_id = envelope.event_id.0;
        let mut event = envelope.event;
        if let RealtimeEvent::Transcript(RealtimeTranscriptEvent::Revision(revision)) = &mut event
            && let Some(revises) = revision.revises_event_id.as_mut()
            && let Some(client_visible) = self.emitted_event_ids.get(revises.0.as_str())
        {
            revises.0 = client_visible.clone();
        }
        let stamped = self.sequencer.next(event, envelope.created_at);
        self.remember_emitted_event_id(producer_event_id, stamped.event_id.0.clone());
        send_event(&self.event_sender, stamped).await
    }

    pub(crate) async fn emit_envelope_with_translation(
        &mut self,
        envelope: RealtimeEventEnvelope,
    ) -> Result<(), ()> {
        let transcript = match &envelope.event {
            RealtimeEvent::Transcript(event) => Some(event.clone()),
            _ => None,
        };
        self.emit_envelope(envelope).await?;
        if let Some(transcript) = transcript {
            self.observe_translation_source(transcript).await?;
        }
        self.drain_translation_outputs().await
    }

    pub(crate) fn remember_emitted_event_id(&mut self, producer_id: String, client_id: String) {
        const MAX_REMEMBERED_EVENT_IDS: usize = 1024;
        if self
            .emitted_event_ids
            .insert(producer_id.clone(), client_id)
            .is_none()
        {
            self.emitted_event_id_order.push_back(producer_id);
        }
        while self.emitted_event_id_order.len() > MAX_REMEMBERED_EVENT_IDS {
            if let Some(oldest) = self.emitted_event_id_order.pop_front() {
                self.emitted_event_ids.remove(&oldest);
            }
        }
    }

    pub(crate) fn has_backend_results(&self) -> bool {
        self.backend_results.is_some()
    }

    pub(crate) fn has_pending_backend_jobs(&self) -> bool {
        self.pending_backend_jobs > 0
    }

    pub(crate) fn has_active_translation(&self) -> bool {
        self.translation.is_some()
    }

    pub(crate) async fn recv_backend_result(&mut self) -> Option<BackendResult> {
        self.backend_results.as_mut()?.recv().await
    }

    pub(crate) async fn handle_incoming_message(
        &mut self,
        message: Option<Result<Message, axum::Error>>,
    ) -> bool {
        match message {
            Some(Ok(Message::Text(text))) => self.handle_text(&text).await.is_ok(),
            Some(Ok(Message::Binary(bytes))) => self.handle_binary(&bytes).await.is_ok(),
            Some(Ok(Message::Close(_))) | None => false,
            Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) => true,
            Some(Err(error)) => {
                let _ = self
                    .emit_error(
                        RealtimeErrorCode::ClientDisconnected,
                        &format!("WebSocket transport error: {error}"),
                        false,
                    )
                    .await;
                false
            }
        }
    }

    pub(crate) async fn handle_text(&mut self, text: &str) -> Result<(), ()> {
        let value = match serde_json::from_str::<serde_json::Value>(text) {
            Ok(value) => value,
            Err(error) => {
                self.emit_error(
                    RealtimeErrorCode::StartupConfigError,
                    &format!("Invalid realtime JSON control message: {error}"),
                    false,
                )
                .await?;
                return Err(());
            }
        };
        if contains_backend_field(&value) {
            self.emit_error(
                RealtimeErrorCode::StartupConfigError,
                "Realtime sessions use the server startup backend; request-level backend selection is not supported.",
                false,
            )
            .await?;
            return Err(());
        }
        let message = match serde_json::from_value::<ClientMessage>(value) {
            Ok(message) => message,
            Err(error) => {
                self.emit_error(
                    RealtimeErrorCode::StartupConfigError,
                    &format!("Unsupported realtime control message schema: {error}"),
                    false,
                )
                .await?;
                return Err(());
            }
        };

        match message {
            ClientMessage::SessionStart { session } => self.start_session(*session).await,
            ClientMessage::AudioInputConfigure {
                format,
                frame_duration_ms,
            } => self.configure_audio(format, frame_duration_ms).await,
            ClientMessage::SessionCancel { reason } => {
                self.cancel(reason.as_deref().unwrap_or("cancelled")).await
            }
            ClientMessage::SessionClose => {
                self.finish("client_closed", true).await?;
                Err(())
            }
        }
    }

    async fn configure_audio(
        &mut self,
        format: Option<ClientAudioFormat>,
        frame_duration_ms: Option<u32>,
    ) -> Result<(), ()> {
        if self.controller.is_some() {
            self.emit_error(
                RealtimeErrorCode::StartupConfigError,
                "Realtime audio.input.configure must be sent before session.start.",
                false,
            )
            .await?;
            return Err(());
        }
        let format = match format.unwrap_or_default().try_into_realtime() {
            Ok(format) => format,
            Err(message) => {
                self.emit_error(RealtimeErrorCode::UnsupportedAudioFormat, &message, false)
                    .await?;
                return Err(());
            }
        };
        if let Err(error) = format
            .sample_count_for_duration_ms(frame_duration_ms.unwrap_or(DEFAULT_FRAME_DURATION_MS))
        {
            self.emit_error(
                RealtimeErrorCode::UnsupportedAudioFormat,
                &error.to_string(),
                false,
            )
            .await?;
            return Err(());
        }
        let duration_ms = frame_duration_ms.unwrap_or(DEFAULT_FRAME_DURATION_MS);
        self.frame_duration_ms = duration_ms;
        self.frame_byte_len = format.sample_count_for_duration_ms(duration_ms).unwrap() * 2;
        Ok(())
    }

    pub(crate) async fn start_session(&mut self, session: StartSession) -> Result<(), ()> {
        if self.controller.is_some() {
            self.emit_error(
                RealtimeErrorCode::StartupConfigError,
                "Realtime session.start was received after a session was already started.",
                false,
            )
            .await?;
            return Err(());
        }
        let Some(model) = session
            .model
            .as_deref()
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .map(ToOwned::to_owned)
        else {
            self.emit_error(
                RealtimeErrorCode::StartupConfigError,
                "Realtime session.start requires session.model.",
                false,
            )
            .await?;
            return Err(());
        };
        let normalized_model = match resolve_model(&self.runtime, &self.distribution, &model) {
            Ok(model) => model,
            Err(error) => {
                self.emit_error(RealtimeErrorCode::StartupConfigError, &error, false)
                    .await?;
                return Err(());
            }
        };
        let capabilities = self.realtime_capabilities();
        if start_session_requests_phrase_bias(&session) && !capabilities.phrase_bias.supported {
            let message =
                realtime_phrase_bias_rejection_message(&self.runtime, capabilities.phrase_bias);
            self.emit_error(RealtimeErrorCode::StartupConfigError, &message, false)
                .await?;
            return Err(());
        }
        let word_timestamps = session.word_timestamps.unwrap_or(false);
        if word_timestamps && !capabilities.word_timestamps.supported {
            self.emit_error(
                RealtimeErrorCode::StartupConfigError,
                capabilities
                    .word_timestamps
                    .reason
                    .unwrap_or("Realtime word timestamps are not supported by this backend."),
                false,
            )
            .await?;
            return Err(());
        }
        let diarize = session.diarize.unwrap_or(false);
        if diarize && !capabilities.diarization.supported {
            self.emit_error(
                RealtimeErrorCode::StartupConfigError,
                capabilities
                    .diarization
                    .reason
                    .unwrap_or("Realtime diarization is not supported by this backend."),
                false,
            )
            .await?;
            return Err(());
        }
        #[cfg(test)]
        let test_translation_worker = self
            .test_translation_worker
            .clone()
            .map(|worker| (worker, self.test_translation_worker_init.clone()));
        #[cfg(not(test))]
        let test_translation_worker = None;
        let (translation, translation_summary) = match Self::build_translation_lane(
            &session,
            capabilities,
            self.distribution.clone(),
            test_translation_worker,
        )
        .await
        {
            Ok(result) => result,
            Err(message) => {
                self.emit_error(RealtimeErrorCode::StartupConfigError, &message, false)
                    .await?;
                return Err(());
            }
        };
        // Build the per-session diarizer up front so a pack that resolves but
        // fails to load rejects the session instead of silently degrading to
        // anonymous transcripts.
        self.streaming_diarizer = if diarize {
            let Some(diarizer) = self.build_streaming_diarizer(16_000) else {
                self.emit_error(
                    RealtimeErrorCode::StartupConfigError,
                    "Realtime diarization was requested but the active speaker-embedder pack could not be loaded.",
                    false,
                )
                .await?;
                return Err(());
            };
            Some(diarizer)
        } else {
            None
        };
        let phrase_bias = match build_realtime_phrase_bias_config(&session) {
            Ok(phrase_bias) => phrase_bias,
            Err(message) => {
                self.emit_error(RealtimeErrorCode::StartupConfigError, &message, false)
                    .await?;
                return Err(());
            }
        };
        if let Some(format) = session.audio_format {
            self.configure_audio(Some(format), Some(self.frame_duration_ms))
                .await?;
        }

        let vad_config = session.vad.unwrap_or_default();
        if vad_config.enabled == Some(false) {
            self.emit_error(
                RealtimeErrorCode::StartupConfigError,
                "Realtime vad.enabled=false is not implemented in M48C; omit the field or use true.",
                false,
            )
            .await?;
            return Err(());
        }
        let vad = vad_config.into_vad_config(self.frame_duration_ms);
        if let Err(error) = vad.validate() {
            self.emit_error(
                RealtimeErrorCode::StartupConfigError,
                &error.to_string(),
                false,
            )
            .await?;
            return Err(());
        }
        let buffer = match realtime_buffer_config(self.frame_duration_ms, vad) {
            Ok(buffer) => buffer,
            Err(message) => {
                self.emit_error(RealtimeErrorCode::StartupConfigError, &message, false)
                    .await?;
                return Err(());
            }
        };

        let source_name = session
            .source_name
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        let use_native_streaming =
            should_use_native_streaming_session(source_name.as_deref(), capabilities);
        self.native_speaker_change_detector = if diarize && use_native_streaming {
            self.build_streaming_speaker_change_detector(16_000)
        } else {
            None
        };
        let effective_partial_results = effective_session_partial_results(
            session.partial_results.unwrap_or(false),
            capabilities,
            use_native_streaming,
        );
        let mut config = RealtimeSessionConfig::new(
            self.session_id.0.clone(),
            normalized_model.clone(),
            timestamp_now(),
        );
        config.partial_results = effective_partial_results;
        config.word_timestamps = word_timestamps;
        config.diarize = diarize;
        config.translation = translation_summary;
        config.vad = vad;
        config.buffer = buffer;
        let mut controller = match RealtimeSessionController::new(config) {
            Ok(controller) => controller,
            Err(error) => {
                self.emit_error(
                    RealtimeErrorCode::StartupConfigError,
                    &error.to_string(),
                    false,
                )
                .await?;
                return Err(());
            }
        };

        self.language = session.language.filter(|language| language != "auto");
        self.task = session.task;
        self.prompt = session.prompt;
        self.phrase_bias = phrase_bias;
        self.inference_threads =
            match validate_realtime_inference_threads(session.inference_threads) {
                Ok(inference_threads) => inference_threads.or_else(|| {
                    self.distribution
                        .openasr_home()
                        .ok()
                        .and_then(|home| realtime_inference_threads_preference(&home))
                }),
                Err(message) => {
                    self.emit_error(RealtimeErrorCode::StartupConfigError, &message, false)
                        .await?;
                    return Err(());
                }
            };
        self.execution_target = session.execution_target.or_else(|| {
            self.distribution
                .openasr_home()
                .ok()
                .and_then(|home| realtime_execution_target_preference(&home))
        });
        self.word_timestamps = word_timestamps;
        self.source_name = source_name;
        self.translation = translation;
        if use_native_streaming {
            self.controller = Some(controller);
            let result = self
                .start_native_streaming_session(
                    normalized_model,
                    effective_partial_results,
                    word_timestamps,
                    diarize,
                )
                .await;
            if result.is_err() {
                self.controller = None;
                self.translation = None;
            }
            return result;
        }
        // FilePerUtteranceFallback is still a live path for mock/default
        // runtimes, keyless native packs, and native packs that intentionally do
        // not self-declare true streaming.
        self.spawn_backend_worker();

        let created = controller.session_created_event(timestamp_now());
        self.emit_envelope(created).await?;
        let configured =
            match controller.lifecycle(RealtimeLifecycleAction::Configure, timestamp_now()) {
                Ok(configured) => configured,
                Err(error) => {
                    self.emit_error(
                        RealtimeErrorCode::StartupConfigError,
                        &error.to_string(),
                        false,
                    )
                    .await?;
                    return Err(());
                }
            };
        self.emit_envelope(configured).await?;
        let started =
            match controller.lifecycle(RealtimeLifecycleAction::StartAudio, timestamp_now()) {
                Ok(started) => started,
                Err(error) => {
                    self.emit_error(
                        RealtimeErrorCode::StartupConfigError,
                        &error.to_string(),
                        false,
                    )
                    .await?;
                    return Err(());
                }
            };
        self.controller = Some(controller);
        if self.emit_envelope(started).await.is_err() {
            self.controller = None;
            return Err(());
        }
        Ok(())
    }

    async fn build_translation_lane(
        session: &StartSession,
        capabilities: RealtimeBackendCapabilities,
        distribution: DistributionContext,
        test_worker: Option<(TranslationWorkerHook, Option<TranslationWorkerInitHook>)>,
    ) -> Result<(Option<RealtimeTranslationLane>, SessionTranslationSummary), String> {
        let Some(options) = session.translation.as_ref() else {
            return Ok((None, SessionTranslationSummary::disabled()));
        };
        if !options.enabled.unwrap_or(false) {
            return Ok((None, SessionTranslationSummary::disabled()));
        }

        if !capabilities.translation.supported {
            let reason = capabilities
                .translation
                .reason
                .unwrap_or(openasr_core::RealtimeTranslationCapability::REASON_PACK_MISSING);
            return Err(format!(
                "Realtime translation was requested but is unavailable: {reason}."
            ));
        }
        let target_lang = options
            .target_lang
            .as_deref()
            .unwrap_or("en")
            .trim()
            .to_string();
        let Some(target_lang) = TargetLang::parse_mvp(&target_lang) else {
            return Err(
                "Realtime translation MVP only supports translation.target_lang=\"en\"."
                    .to_string(),
            );
        };
        let mode = options
            .mode
            .as_deref()
            .unwrap_or(openasr_core::RealtimeTranslationCapability::MODE_CLAUSE_RETRANSLATION);
        if mode != openasr_core::RealtimeTranslationCapability::MODE_CLAUSE_RETRANSLATION {
            return Err(
                "Realtime translation MVP only supports translation.mode=\"clause_retranslation\"."
                    .to_string(),
            );
        }
        if !session_language_is_chinese(session.language.as_deref()) {
            return Err(
                "Realtime translation MVP requires session.language=\"zh\" so zh->en is explicit."
                    .to_string(),
            );
        }
        if let Some(model) = options.model.as_deref()
            && !translation_model_ref_supported(model)
        {
            return Err(format!(
                "Realtime translation MVP only supports translation.model=\"{}\" or \"hymt2-1.8b\".",
                HYMT2_TRANSLATION_MODEL_ID
            ));
        }

        let model_id = HYMT2_TRANSLATION_MODEL_ID.to_string();
        let requested_model = options.model.clone();
        let translation_session =
            Self::build_translation_session(distribution, requested_model.as_deref(), test_worker)
                .await?;
        let summary = SessionTranslationSummary {
            enabled: true,
            target_lang: Some(target_lang.as_str().to_string()),
            model: Some(model_id.clone()),
            mode: Some(
                openasr_core::RealtimeTranslationCapability::MODE_CLAUSE_RETRANSLATION.to_string(),
            ),
        };
        let ready_announced = translation_session.worker_ready();
        Ok((
            Some(RealtimeTranslationLane {
                session: translation_session,
                segmenter: ClauseSegmenter::default(),
                gates: HashMap::new(),
                clause_meta: HashMap::new(),
                retired_clause_ids: HashSet::new(),
                retired_clause_order: VecDeque::new(),
                source_segments: Vec::new(),
                model_id,
                target_lang,
                provisional: options.provisional.unwrap_or(true),
                ready_announced,
            }),
            summary,
        ))
    }

    async fn build_translation_session(
        distribution: DistributionContext,
        requested_model: Option<&str>,
        test_worker: Option<(TranslationWorkerHook, Option<TranslationWorkerInitHook>)>,
    ) -> Result<TranslationSession, String> {
        if let Some((worker, init)) = test_worker {
            if let Some(init) = init {
                return Ok(TranslationSession::spawn_thread_local(move || {
                    init()?;
                    Ok(move |request| worker(request))
                }));
            }
            return Ok(TranslationSession::spawn(move |request| worker(request)));
        }

        let selection = resolve_translation_pack_selection(&distribution, requested_model)?;
        Ok(Self::load_hymt2_translation_session(selection))
    }

    /// Spawns the Hy-MT2 translation worker with the (multi-second) model
    /// cold load running on the worker thread, OFF the session-start critical
    /// path. `session.start` is accepted immediately; readiness is announced
    /// via a `translation.status` event from `drain_translation_outputs`, and
    /// a load failure surfaces there as a session-fatal `error` event.
    fn load_hymt2_translation_session(selection: TranslationPackSelection) -> TranslationSession {
        let path = selection.path;
        TranslationSession::spawn_thread_local(move || {
            let runtime =
                Hymt2Runtime::from_path(path).map_err(|error| TranslationQueueError::Worker {
                    reason: format!(
                        "Realtime translation Hy-MT2 runtime could not be loaded: {error}"
                    ),
                })?;
            let mut cache = Hymt2TranslationSessionCache::default();
            Ok(move |request| {
                runtime
                    .translate_request_with_cache(&mut cache, &request)
                    .map_err(|error| TranslationQueueError::Worker {
                        reason: error.to_string(),
                    })
            })
        })
    }

    async fn observe_translation_source(
        &mut self,
        transcript: RealtimeTranscriptEvent,
    ) -> Result<(), ()> {
        let Some(source) = TranslationSourceEvent::from_transcript(transcript) else {
            return Ok(());
        };
        let Some(lane) = self.translation.as_mut() else {
            return Ok(());
        };
        let tombstones = match lane.observe_source(source) {
            Ok(tombstones) => tombstones,
            Err(message) => {
                self.emit_error(RealtimeErrorCode::BackendCrashed, &message, false)
                    .await?;
                return Err(());
            }
        };
        for tombstone in tombstones {
            self.emit_event(tombstone).await?;
        }
        Ok(())
    }

    pub(crate) async fn drain_translation_outputs(&mut self) -> Result<(), ()> {
        self.announce_translation_ready().await?;
        loop {
            let next = {
                let Some(lane) = self.translation.as_ref() else {
                    return Ok(());
                };
                lane.session.try_recv()
            };
            let output = match next {
                Ok(Some(output)) => output,
                Ok(None) => return Ok(()),
                Err(error) => {
                    self.emit_error(
                        RealtimeErrorCode::BackendCrashed,
                        &format!("Realtime translation session failed: {error}"),
                        false,
                    )
                    .await?;
                    self.translation = None;
                    return Err(());
                }
            };
            if output.dropped_stale {
                if let Some(lane) = self.translation.as_mut() {
                    lane.retired_clause_ids.remove(&output.clause_id);
                }
                continue;
            }
            if self
                .translation
                .as_mut()
                .is_some_and(|lane| lane.retired_clause_ids.remove(&output.clause_id))
            {
                continue;
            }
            let accepted_output = output.clone();
            let Some(event) = self.translation_event_from_output(output) else {
                continue;
            };
            let context_error = self
                .translation
                .as_ref()
                .and_then(|lane| lane.session.record_output_context(&accepted_output).err());
            if let Some(error) = context_error {
                self.emit_error(
                    RealtimeErrorCode::BackendCrashed,
                    &format!("Realtime translation context update failed: {error}"),
                    false,
                )
                .await?;
                return Err(());
            }
            self.emit_event(event).await?;
        }
    }

    /// Emits the one-shot `translation.status` ready event once the
    /// asynchronously-initialized translation worker finishes its model load.
    async fn announce_translation_ready(&mut self) -> Result<(), ()> {
        let ready = match self.translation.as_ref() {
            Some(lane) if !lane.ready_announced => lane.session.worker_ready(),
            _ => return Ok(()),
        };
        if !ready {
            return Ok(());
        }
        let Some(lane) = self.translation.as_mut() else {
            return Ok(());
        };
        lane.ready_announced = true;
        let event = RealtimeEvent::Translation(RealtimeTranslationEvent::Status(
            RealtimeTranslationStatus {
                state: RealtimeTranslationStatus::STATE_READY.to_string(),
                model: lane.model_id.clone(),
                target_lang: lane.target_lang.as_str().to_string(),
            },
        ));
        self.emit_event(event).await
    }

    pub(crate) async fn drain_translation_until_idle(&mut self) -> Result<(), ()> {
        let deadline = Instant::now() + backend_result_timeout();
        loop {
            self.drain_translation_outputs().await?;
            let pending = self
                .translation
                .as_ref()
                .is_some_and(|lane| lane.session.has_pending_or_running());
            if !pending {
                return Ok(());
            }
            if Instant::now() >= deadline {
                self.emit_error(
                    RealtimeErrorCode::BackendTimeout,
                    "Realtime translation did not finish before the session close timeout.",
                    false,
                )
                .await?;
                return Err(());
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn translation_event_from_output(
        &mut self,
        output: TranslationOutput,
    ) -> Option<RealtimeEvent> {
        let lane = self.translation.as_mut()?;
        let meta = lane
            .clause_meta
            .get(&output.clause_id)
            .cloned()
            .unwrap_or_else(|| TranslationClauseMeta {
                source_segment_id: String::new(),
                source_version: output.source_version,
                replaces_clause_id: output.replaces_clause_id,
                start_ms: 0,
                end_ms: 0,
                stability: if output.finalized { 1.0 } else { 0.0 },
            });
        if output.finalized {
            lane.gates.remove(&output.clause_id);
        }
        let target_lang = output.target_lang.as_str().to_string();
        let model = lane.model_id.clone();
        let replaces_clause_id = meta
            .replaces_clause_id
            .or(output.replaces_clause_id)
            .map(|clause_id| clause_id.to_string());
        let revises_clause_id = replaces_clause_id.clone();
        Some(if output.finalized {
            RealtimeEvent::Translation(RealtimeTranslationEvent::Final(RealtimeTranslationFinal {
                clause_id: output.clause_id.to_string(),
                replaces_clause_id,
                revises_clause_id,
                source_segment_id: meta.source_segment_id,
                source_version: output.source_version,
                translation_version: output.translation_version,
                target_lang,
                text: output.text,
                source_text: output.source_text,
                start_ms: meta.start_ms,
                end_ms: meta.end_ms,
                is_final: true,
                model,
            }))
        } else {
            RealtimeEvent::Translation(RealtimeTranslationEvent::Partial(
                RealtimeTranslationPartial {
                    clause_id: output.clause_id.to_string(),
                    replaces_clause_id,
                    revises_clause_id,
                    source_segment_id: meta.source_segment_id,
                    source_version: output.source_version,
                    translation_version: output.translation_version,
                    target_lang,
                    text: output.text,
                    source_text: output.source_text,
                    start_ms: meta.start_ms,
                    end_ms: meta.end_ms,
                    stability: meta.stability,
                    is_final: false,
                    model,
                },
            ))
        })
    }

    fn build_streaming_diarizer(
        &self,
        sample_rate_hz: u32,
    ) -> Option<openasr_core::diarize::streaming::StreamingDiarizer> {
        #[cfg(test)]
        if let Some(embedder) = self.test_streaming_diarizer_embedder {
            return Some(
                openasr_core::diarize::streaming::StreamingDiarizer::with_embedder(
                    embedder,
                    sample_rate_hz,
                ),
            );
        }

        openasr_core::diarize::streaming::StreamingDiarizer::shared(sample_rate_hz)
    }

    fn build_streaming_speaker_change_detector(
        &self,
        sample_rate_hz: u32,
    ) -> Option<openasr_core::diarize::streaming::StreamingSpeakerChangeDetector> {
        #[cfg(test)]
        if let Some(embedder) = self.test_streaming_diarizer_embedder {
            return Some(
                openasr_core::diarize::streaming::StreamingSpeakerChangeDetector::with_embedder(
                    embedder,
                    sample_rate_hz,
                ),
            );
        }

        openasr_core::diarize::streaming::StreamingSpeakerChangeDetector::shared(sample_rate_hz)
    }

    pub(crate) async fn start_native_streaming_session(
        &mut self,
        model_id: String,
        partial_results: bool,
        word_timestamps: bool,
        diarize: bool,
    ) -> Result<(), ()> {
        let Some(model_pack_path) = self.runtime.model_pack_path.clone() else {
            self.emit_error(
                RealtimeErrorCode::StartupConfigError,
                "Native realtime streaming requires an explicit local runtime pack path.",
                false,
            )
            .await?;
            return Err(());
        };
        let Some(adapter) = native_runtime_model_adapter_for_path(&model_pack_path) else {
            self.emit_error(
                RealtimeErrorCode::StartupConfigError,
                &format!(
                    "Could not select a native streaming adapter from runtime source '{}'.",
                    model_pack_path.display()
                ),
                false,
            )
            .await?;
            return Err(());
        };
        let model_pack =
            NativeAsrModelPackRef::new(model_id, adapter.model_family(), model_pack_path);
        let context = NativeAsrSessionContext::from_realtime_session_id(self.session_id.clone());
        // With translation active, session.language is the validated zh->en
        // translation source declaration consumed by the translation lane. Only
        // families that actually honor a source-language decode hint (Whisper,
        // Cohere) may also receive it as an ASR option; xasr/qwen fail closed on
        // decode hints they ignore, which would otherwise break the translation
        // contract that REQUIRES session.language="zh".
        let asr_language = if self.translation.is_some()
            && !openasr_core::native_adapter_supports_source_language_hint(adapter.adapter_id())
        {
            None
        } else {
            self.language.clone()
        };
        let options = NativeAsrRequestOptions::new()
            .with_language(asr_language)
            .with_task(self.task)
            .with_prompt(self.prompt.clone())
            .with_phrase_bias(self.phrase_bias.clone())
            .with_inference_threads(self.inference_threads)
            .with_diarization(diarize)
            .with_partial_results(partial_results)
            .with_word_timestamps(word_timestamps);
        let session_config = NativeAsrStreamingSessionConfig::new()
            .with_audio_format(RealtimeAudioFormat::pcm16_mono_16khz())
            .with_partial_results(partial_results)
            .with_word_timestamps(word_timestamps);
        let executor = NativeBackendExecutor;
        let hardware_target = native_hardware_target_from_execution_target(self.execution_target);
        let mut session = match NativeAsrExecutor::start_streaming_session(
            &executor,
            &adapter,
            &model_pack,
            hardware_target,
            context,
            options,
            session_config,
        ) {
            Ok(session) => session,
            Err(error) => {
                self.emit_error(
                    RealtimeErrorCode::StartupConfigError,
                    &format!("Could not start native streaming session: {error}"),
                    false,
                )
                .await?;
                return Err(());
            }
        };
        let events = session.poll_events().map_err(|error| {
            eprintln!("OpenASR native streaming poll failed during startup: {error}");
        })?;
        self.forward_native_streaming_events(NativeStreamingCommandKind::Poll, events)
            .await?;
        // The session moves onto its own decode thread; the WS task queues audio
        // and drains outcomes separately so a slow partial decode never blocks
        // socket ingest for this session.
        self.attach_native_streaming_session(
            NativeStreamingWorkerKey::new(
                model_pack.root.clone(),
                hardware_target,
                self.inference_threads,
            ),
            session,
        )
        .await
        .map_err(|message| {
            eprintln!("OpenASR native streaming worker attach failed: {message}");
        })?;
        self.send_native_streaming_command(NativeStreamingCommand::Warm)
            .await?;
        eprintln!("OpenASR native streaming worker warm-up queued");
        Ok(())
    }

    pub(crate) async fn attach_native_streaming_session(
        &mut self,
        key: NativeStreamingWorkerKey,
        session: Box<dyn NativeAsrSession>,
    ) -> Result<(), String> {
        self.native_streaming = Some(NativeStreamingDecodeWorker::attach(key, session).await?);
        Ok(())
    }

    pub(crate) fn is_native_streaming(&self) -> bool {
        self.native_streaming.is_some()
    }

    pub(crate) async fn poll_native_streaming(&mut self) -> Result<(), ()> {
        if !self.native_had_speech_since_last_poll
            || self.native_poll_outstanding >= NATIVE_STREAMING_MAX_OUTSTANDING_POLLS
        {
            return Ok(());
        }
        self.native_had_speech_since_last_poll = false;
        self.native_poll_outstanding = self.native_poll_outstanding.saturating_add(1);
        self.send_native_streaming_command(NativeStreamingCommand::Poll)
            .await
    }

    pub(crate) async fn send_native_streaming_command(
        &mut self,
        command: NativeStreamingCommand,
    ) -> Result<(), ()> {
        let Some(worker) = self.native_streaming.as_mut() else {
            return Ok(());
        };
        let kind = command.kind();
        let envelope = NativeStreamingCommandEnvelope { kind, command };
        if kind == NativeStreamingCommandKind::Finalize {
            worker.finalize_requested.store(true, Ordering::Release);
        }
        if kind == NativeStreamingCommandKind::Cancel {
            worker.request_cancel();
        }
        if worker.commands.send(envelope).await.is_ok() {
            self.native_command_watchdogs
                .push_back((kind, Instant::now()));
            return Ok(());
        }
        self.fail_native_streaming_worker_stopped().await
    }

    pub(crate) async fn drain_native_streaming_outcomes(&mut self) -> Result<(), ()> {
        loop {
            let outcome = {
                let Some(worker) = self.native_streaming.as_mut() else {
                    return Ok(());
                };
                match worker.outcomes.try_recv() {
                    Ok(outcome) => outcome,
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        return self.fail_native_streaming_worker_stopped().await;
                    }
                }
            };
            let (kind, events) = self.apply_native_streaming_outcome(outcome).await?;
            self.forward_native_streaming_events(kind, events).await?;
        }
        self.enforce_native_streaming_watchdog().await
    }

    pub(crate) async fn enforce_native_streaming_watchdog(&mut self) -> Result<(), ()> {
        let Some((kind, started_at)) = self.native_command_watchdogs.front().copied() else {
            return Ok(());
        };
        if started_at.elapsed() < self.native_decode_timeout {
            return Ok(());
        }
        self.apply_native_streaming_outcome(NativeStreamingOutcome::Error {
            kind,
            message: format!(
                "decode did not return within {}s; the decode step may be hung",
                self.native_decode_timeout.as_secs()
            ),
        })
        .await
        .map(|_| ())
    }

    /// Send one lifecycle command to the decode thread and await its matching
    /// outcome, bounded by the decode watchdog. While waiting, forward any older
    /// async outcomes that complete first.
    pub(crate) async fn native_streaming_command(
        &mut self,
        command: NativeStreamingCommand,
    ) -> Result<(NativeStreamingCommandKind, Vec<RealtimeEventEnvelope>), ()> {
        let expected_kind = command.kind();
        self.send_native_streaming_command(command).await?;
        loop {
            let timeout = self.native_decode_timeout;
            let outcome = {
                let Some(worker) = self.native_streaming.as_mut() else {
                    return Ok((expected_kind, Vec::new()));
                };
                match tokio::time::timeout(timeout, worker.outcomes.recv()).await {
                    Ok(Some(outcome)) => outcome,
                    Ok(None) => return self.fail_native_streaming_worker_stopped().await,
                    Err(_elapsed) => {
                        let kind = self
                            .native_command_watchdogs
                            .front()
                            .map(|(kind, _)| *kind)
                            .unwrap_or(expected_kind);
                        NativeStreamingOutcome::Error {
                            kind,
                            message: format!(
                                "decode did not return within {}s; the decode step may be hung",
                                timeout.as_secs()
                            ),
                        }
                    }
                }
            };
            let (kind, events) = self.apply_native_streaming_outcome(outcome).await?;
            if kind == expected_kind {
                return Ok((kind, events));
            }
            self.forward_native_streaming_events(kind, events).await?;
        }
    }

    pub(crate) async fn apply_native_streaming_outcome(
        &mut self,
        outcome: NativeStreamingOutcome,
    ) -> Result<(NativeStreamingCommandKind, Vec<RealtimeEventEnvelope>), ()> {
        let kind = outcome.kind();
        let warm_started_at = self.native_command_watchdogs.pop_front().and_then(
            |(queued_kind, started_at)| {
                if queued_kind != kind {
                    eprintln!(
                        "OpenASR native streaming worker outcome order mismatch: expected {queued_kind:?}, got {kind:?}"
                    );
                }
                (kind == NativeStreamingCommandKind::Warm).then_some(started_at)
            },
        );
        if kind == NativeStreamingCommandKind::Poll {
            self.native_poll_outstanding = self.native_poll_outstanding.saturating_sub(1);
        }
        match outcome {
            NativeStreamingOutcome::Events { events, .. } => {
                if let Some(started_at) = warm_started_at {
                    eprintln!(
                        "OpenASR native streaming worker warm-up returned in {}ms",
                        started_at.elapsed().as_millis()
                    );
                }
                Ok((kind, events))
            }
            NativeStreamingOutcome::Error { message, .. } => {
                self.backend_failed = true;
                self.emit_error(
                    RealtimeErrorCode::BackendCrashed,
                    &format!("Native streaming session failed: {message}"),
                    false,
                )
                .await?;
                Err(())
            }
        }
    }

    pub(crate) async fn fail_native_streaming_worker_stopped<T>(&mut self) -> Result<T, ()> {
        self.backend_failed = true;
        self.native_poll_outstanding = 0;
        self.native_command_watchdogs.clear();
        self.emit_error(
            RealtimeErrorCode::BackendCrashed,
            "Native streaming decode worker stopped unexpectedly.",
            false,
        )
        .await?;
        Err(())
    }

    pub(crate) async fn forward_native_streaming_events(
        &mut self,
        kind: NativeStreamingCommandKind,
        events: Vec<RealtimeEventEnvelope>,
    ) -> Result<(), ()> {
        let mut consumed_split_slot = false;
        let mut consumed_split_change_point = false;
        for mut envelope in events {
            consumed_split_slot |= self
                .stamp_native_transcript_speaker(kind, &mut envelope)
                .await;
            self.remember_native_streaming_history(&envelope);
            // The change-point estimate of a speaker-change split applies to
            // the split's terminal transcript, mirroring the split-slot
            // consumption discipline (first terminal event of the command).
            let split_change_ms = if kind == NativeStreamingCommandKind::SplitUtterance
                && !consumed_split_change_point
                && is_terminal_transcript_envelope(&envelope)
            {
                consumed_split_change_point = true;
                self.pending_native_split_change_points
                    .pop_front()
                    .flatten()
            } else {
                None
            };
            let final_snapshot = (self.streaming_diarizer.is_some())
                .then(|| snapshot_final_transcript(&envelope))
                .flatten();
            let producer_event_id = envelope.event_id.0.clone();
            self.emit_envelope_with_translation(envelope).await?;
            if let Some(snapshot) = final_snapshot {
                let client_event_id = self.emitted_event_ids.get(&producer_event_id).cloned();
                self.apply_retroactive_speaker_attribution(
                    snapshot,
                    client_event_id,
                    split_change_ms,
                )
                .await?;
            }
        }
        if kind == NativeStreamingCommandKind::SplitUtterance {
            if !consumed_split_slot {
                self.pending_native_split_speaker_slots.pop_front();
            }
            if !consumed_split_change_point {
                self.pending_native_split_change_points.pop_front();
            }
        }
        Ok(())
    }

    /// Retroactive speaker attribution for native true-streaming transcripts.
    ///
    /// Two mechanisms, both via the post-final `transcript.revision` wire
    /// contract the desktop already applies per `(utterance, segment)` line:
    ///
    /// 1. Mid-utterance sentence-cut finals are emitted before the
    ///    utterance's speaker label binds (labels bind on the terminal
    ///    transcript), so they reach the client speakerless. They are
    ///    tracked here and revised with the speaker once the label binds.
    /// 2. A speaker-change split lags the acoustic change by up to the
    ///    detector's analysis window, so trailing words of the OLD
    ///    utterance's terminal final actually belong to the NEW speaker.
    ///    When the final carries word timestamps, the tail after the
    ///    estimated change point is carved off: the old final is revised
    ///    down to the kept words and the moved tail is emitted as its own
    ///    segment, relabelled once the next utterance's speaker binds.
    ///    Families without realtime word timestamps keep current behavior.
    async fn apply_retroactive_speaker_attribution(
        &mut self,
        snapshot: FinalTranscriptSnapshot,
        client_event_id: Option<String>,
        split_change_ms: Option<u64>,
    ) -> Result<(), ()> {
        if let Some(change_ms) = split_change_ms {
            self.reattribute_split_terminal_tail(&snapshot, client_event_id.clone(), change_ms)
                .await?;
        }
        if snapshot.speaker.is_none()
            && !self
                .native_speaker_by_utterance
                .contains_key(&snapshot.utterance_id)
        {
            // Records of other utterances can no longer bind (labels bind on
            // an utterance's own terminal, which precedes any later
            // utterance's events), so drop them before tracking this one.
            self.native_speakerless_finals
                .retain(|record| record.utterance_id == snapshot.utterance_id);
            if self.native_speakerless_finals.len() >= MAX_PENDING_SPEAKER_REVISIONS {
                self.native_speakerless_finals.remove(0);
            }
            self.native_speakerless_finals.push(PendingSpeakerRevision {
                utterance_id: snapshot.utterance_id.clone(),
                segment_id: snapshot.segment_id.clone(),
                revision: snapshot.revision,
                text: snapshot.text.clone(),
                start_ms: snapshot.start_ms,
                end_ms: snapshot.end_ms,
                language: snapshot.language.clone(),
                words: snapshot.words.clone(),
                client_event_id,
            });
        }
        self.flush_bound_speaker_revisions(&snapshot.utterance_id)
            .await
    }

    /// If `utterance_id`'s label has bound, emits the queued retroactive
    /// speaker revisions: its own speakerless sentence finals, plus any
    /// reattributed split tails waiting for the NEXT utterance's label.
    async fn flush_bound_speaker_revisions(
        &mut self,
        utterance_id: &TranscriptUtteranceId,
    ) -> Result<(), ()> {
        let Some(assignment) = self.native_speaker_by_utterance.get(utterance_id) else {
            return Ok(());
        };
        let assignment = assignment.clone();
        if !self.native_speakerless_finals.is_empty() {
            let records = std::mem::take(&mut self.native_speakerless_finals);
            let (own, stale): (Vec<_>, Vec<_>) = records
                .into_iter()
                .partition(|record| record.utterance_id == *utterance_id);
            drop(stale);
            if let Some(assignment) = assignment.as_ref() {
                for record in own {
                    self.emit_speaker_attribution_revision(
                        record,
                        assignment,
                        "speaker_attribution",
                    )
                    .await?;
                }
            }
        }
        while let Some(record) = self.pending_split_tail_relabels.front() {
            // A tail carved from utterance N waits for the label of a LATER
            // utterance; the bind of utterance N itself (its own terminal)
            // must not consume it.
            if record.utterance_id == *utterance_id {
                break;
            }
            let Some(record) = self.pending_split_tail_relabels.pop_front() else {
                break;
            };
            // When the next utterance resolved unlabelled the tail's speaker
            // is unknowable: leave it as emitted (current behavior).
            if let Some(assignment) = assignment.as_ref() {
                self.emit_speaker_attribution_revision(
                    record,
                    assignment,
                    "speaker_change_reattribution",
                )
                .await?;
            }
        }
        Ok(())
    }

    /// Carves the trailing words after the acoustic change point out of a
    /// speaker-change split's terminal final: revises the old final down to
    /// the kept words and emits the moved tail as its own (speakerless for
    /// now) segment, queued for relabelling at the next utterance's bind.
    async fn reattribute_split_terminal_tail(
        &mut self,
        snapshot: &FinalTranscriptSnapshot,
        client_event_id: Option<String>,
        change_ms: u64,
    ) -> Result<(), ()> {
        // No realtime word timestamps on this family: the text cannot be
        // carved faithfully — fall back to no reattribution.
        let Some(split) = openasr_core::diarize::attribution::split_transcript_tail_at_change(
            &snapshot.text,
            &snapshot.words,
            change_ms,
            NATIVE_SPLIT_REATTRIBUTION_MAX_TAIL_MS,
        ) else {
            return Ok(());
        };
        let trimmed = RealtimeTranscriptRevision {
            utterance_id: snapshot.utterance_id.clone(),
            segment_id: snapshot.segment_id.clone(),
            revises_event_id: client_event_id.map(RealtimeEventId),
            revision: snapshot.revision.saturating_add(1),
            text: split.kept_text,
            start_ms: snapshot.start_ms,
            end_ms: split.kept_end_ms,
            is_final: true,
            reason: "speaker_change_reattribution".to_string(),
            words: snapshot.words[..split.moved_from_word].to_vec(),
            language: snapshot.language.clone(),
            speaker: snapshot.speaker.clone(),
            speaker_label: snapshot.speaker_label.clone(),
            speaker_profile_id: snapshot.speaker_profile_id.clone(),
        };
        self.emit_local_transcript_event(RealtimeTranscriptEvent::Revision(trimmed))
            .await?;
        let moved_segment_id = TranscriptSegmentId(format!("{}_sw", snapshot.segment_id.0));
        let moved_words = snapshot.words[split.moved_from_word..].to_vec();
        let moved = RealtimeTranscriptFinal {
            utterance_id: snapshot.utterance_id.clone(),
            segment_id: moved_segment_id.clone(),
            revision: snapshot.revision.saturating_add(1),
            text: split.moved_text.clone(),
            start_ms: split.moved_start_ms,
            end_ms: snapshot.end_ms,
            is_final: true,
            words: moved_words.clone(),
            language: snapshot.language.clone(),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
        };
        let moved_client_id = self
            .emit_local_transcript_event(RealtimeTranscriptEvent::Final(moved))
            .await?;
        if self.pending_split_tail_relabels.len() >= MAX_PENDING_SPEAKER_REVISIONS {
            self.pending_split_tail_relabels.pop_front();
        }
        self.pending_split_tail_relabels
            .push_back(PendingSpeakerRevision {
                utterance_id: snapshot.utterance_id.clone(),
                segment_id: moved_segment_id,
                revision: snapshot.revision.saturating_add(1),
                text: split.moved_text,
                start_ms: split.moved_start_ms,
                end_ms: snapshot.end_ms,
                language: snapshot.language.clone(),
                words: moved_words,
                client_event_id: Some(moved_client_id),
            });
        Ok(())
    }

    /// Re-sends a pending line as a `transcript.revision` with the bound
    /// speaker attached. Emitted directly on the connection sequencer:
    /// history already recorded the original text and the text is unchanged,
    /// so the history and translation observers must not see it again.
    async fn emit_speaker_attribution_revision(
        &mut self,
        record: PendingSpeakerRevision,
        assignment: &openasr_core::diarize::enrollment::SpeakerDisplayAssignment,
        reason: &str,
    ) -> Result<(), ()> {
        let (speaker_label, speaker_profile_id) = if assignment.speaker_profile_id.is_some() {
            (
                Some(assignment.speaker_label.clone()),
                assignment.speaker_profile_id.clone(),
            )
        } else {
            (None, None)
        };
        let revision = RealtimeTranscriptRevision {
            utterance_id: record.utterance_id,
            segment_id: record.segment_id,
            revises_event_id: record.client_event_id.map(RealtimeEventId),
            revision: record.revision.saturating_add(1),
            text: record.text,
            start_ms: record.start_ms,
            end_ms: record.end_ms,
            is_final: true,
            reason: reason.to_string(),
            words: record.words,
            language: record.language,
            speaker: Some(assignment.speaker.clone()),
            speaker_label,
            speaker_profile_id,
        };
        self.emit_local_transcript_event(RealtimeTranscriptEvent::Revision(revision))
            .await
            .map(|_| ())
    }

    /// Emits a locally-synthesized transcript event on the connection
    /// sequencer and returns its client-visible event id.
    async fn emit_local_transcript_event(
        &mut self,
        event: RealtimeTranscriptEvent,
    ) -> Result<String, ()> {
        let envelope = self
            .sequencer
            .next(RealtimeEvent::Transcript(event), timestamp_now());
        let client_event_id = envelope.event_id.0.clone();
        send_event(&self.event_sender, envelope).await?;
        Ok(client_event_id)
    }

    /// Appends a speech frame to the bounded per-utterance retention buffer
    /// for native true-streaming diarization. An utterance running past the
    /// cap keeps its first [`NATIVE_DIARIZE_MAX_RETAINED_SAMPLES`] samples.
    fn retain_native_diarize_frame(&mut self, frame: &RealtimeAudioFrame) {
        let remaining =
            NATIVE_DIARIZE_MAX_RETAINED_SAMPLES.saturating_sub(self.native_diarize_samples.len());
        if remaining == 0 {
            return;
        }
        self.native_diarize_sample_spans
            .push((self.native_diarize_samples.len(), frame.start_ms));
        self.native_diarize_samples.extend(
            frame
                .samples()
                .iter()
                .take(remaining)
                .map(|sample| pcm16_sample_to_f32(*sample)),
        );
    }

    fn remember_native_diarize_preroll_frame(&mut self, frame: &RealtimeAudioFrame) {
        self.native_diarize_preroll_frames.push_back(frame.clone());
        let min_start_ms = frame.start_ms.saturating_sub(NATIVE_DIARIZE_PREROLL_MS);
        while self
            .native_diarize_preroll_frames
            .front()
            .is_some_and(|candidate| candidate.end_ms() <= min_start_ms)
        {
            self.native_diarize_preroll_frames.pop_front();
        }
    }

    fn backfill_native_diarize_preroll(&mut self, start_ms: u64, current_seq: u64) {
        let frames = self
            .native_diarize_preroll_frames
            .iter()
            .filter(|frame| frame.seq < current_seq && frame.end_ms() > start_ms)
            .cloned()
            .collect::<Vec<_>>();
        for frame in frames {
            self.retain_native_diarize_frame(&frame);
        }
    }

    /// Runs speaker assignment off the async executor: the embedding is
    /// CPU-heavy (rayon convolutions over up to 30 s of fbank frames) and
    /// running it inline would stall every select arm of this session's event
    /// loop at utterance boundaries. The diarizer moves into the blocking
    /// task and back, so per-session centroid state stays single-owner; if
    /// the task panics, diarization disables for the rest of the session
    /// instead of risking misaligned labels.
    async fn assign_speaker_off_loop(
        &mut self,
        samples: Vec<f32>,
        path: openasr_core::diarize::streaming::StreamingDiarizePath,
    ) -> Option<openasr_core::diarize::enrollment::SpeakerDisplayAssignment> {
        let mut diarizer = self.streaming_diarizer.take()?;
        match tokio::task::spawn_blocking(move || {
            let assignment = if samples.is_empty() {
                None
            } else {
                diarizer.assign_with_path(&samples, 16_000, path)
            };
            (diarizer, assignment)
        })
        .await
        {
            Ok((diarizer, assignment)) => {
                self.streaming_diarizer = Some(diarizer);
                assignment
            }
            Err(error) => {
                eprintln!("OpenASR realtime diarization task failed: {error}");
                None
            }
        }
    }

    async fn detect_native_speaker_change_off_loop(
        &mut self,
    ) -> Option<openasr_core::diarize::streaming::StreamingSpeakerChange> {
        let mut detector = self.native_speaker_change_detector.take()?;
        if !detector.should_analyze(self.native_diarize_samples.len()) {
            self.native_speaker_change_detector = Some(detector);
            return None;
        }
        let samples = self.native_diarize_samples.clone();
        match tokio::task::spawn_blocking(move || {
            let change = detector.analyze(&samples);
            (detector, change)
        })
        .await
        {
            Ok((detector, change)) => {
                self.native_speaker_change_detector = Some(detector);
                change
            }
            Err(error) => {
                eprintln!("OpenASR realtime speaker-change task failed: {error}");
                None
            }
        }
    }

    fn reset_native_speaker_change_detector(&mut self) {
        if let Some(detector) = self.native_speaker_change_detector.as_mut() {
            detector.reset();
        }
    }

    async fn assign_native_speaker_label_for_samples(
        &mut self,
        samples: Vec<f32>,
    ) -> Option<openasr_core::diarize::enrollment::SpeakerDisplayAssignment> {
        self.assign_speaker_off_loop(
            samples,
            openasr_core::diarize::streaming::StreamingDiarizePath::Native,
        )
        .await
    }

    async fn resolve_native_speaker_slot(
        &mut self,
        slot: NativePendingSpeakerSlot,
    ) -> Option<openasr_core::diarize::enrollment::SpeakerDisplayAssignment> {
        match slot {
            NativePendingSpeakerSlot::DeferredSamples(samples) => {
                self.assign_native_speaker_label_for_samples(samples).await
            }
            #[cfg(test)]
            NativePendingSpeakerSlot::Resolved(assignment) => assignment,
        }
    }

    /// Assigns a speaker for the utterance whose Finalize is being sent and
    /// queues the label for the worker's terminal transcript, which arrives
    /// asynchronously but in finalize order.
    pub(crate) async fn queue_native_speaker_label(&mut self) {
        let samples = std::mem::take(&mut self.native_diarize_samples);
        self.native_diarize_sample_spans.clear();
        if self.streaming_diarizer.is_none() {
            return;
        }
        self.reset_native_speaker_change_detector();
        if !samples.is_empty() {
            self.pending_native_speaker_labels
                .push_back(NativePendingSpeakerSlot::DeferredSamples(samples));
        }
    }

    async fn queue_native_split_speaker_slot(&mut self, samples: Vec<f32>) {
        if self.streaming_diarizer.is_none() {
            return;
        }
        self.pending_native_split_speaker_slots
            .push_back(NativePendingSpeakerSlot::DeferredSamples(samples));
    }

    pub(crate) async fn queue_native_max_utterance_split_speaker_slot(&mut self) {
        if self.streaming_diarizer.is_none() {
            return;
        }
        let samples = std::mem::take(&mut self.native_diarize_samples);
        self.native_diarize_sample_spans.clear();
        self.reset_native_speaker_change_detector();
        self.pending_native_split_speaker_slots
            .push_back(NativePendingSpeakerSlot::DeferredSamples(samples));
        // Not a speaker-change split: no change point to reattribute against.
        self.pending_native_split_change_points.push_back(None);
    }

    pub(crate) async fn maybe_split_native_on_speaker_change(&mut self) -> Result<bool, ()> {
        let Some(change) = self.detect_native_speaker_change_off_loop().await else {
            return Ok(false);
        };
        let retained = std::mem::take(&mut self.native_diarize_samples);
        let spans = std::mem::take(&mut self.native_diarize_sample_spans);
        let split_sample = change.split_sample.min(retained.len());
        let change_abs_ms = diarize_sample_abs_ms(&spans, split_sample);
        let before = retained[..split_sample].to_vec();
        let after = retained[split_sample..].to_vec();
        self.queue_native_split_speaker_slot(before).await;
        self.native_diarize_samples = after;
        self.native_diarize_sample_spans = rebase_diarize_sample_spans(spans, split_sample);
        self.reset_native_speaker_change_detector();
        eprintln!(
            "OpenASR realtime diarization speaker-change split: split_s={:.3} change_abs_ms={:?} ref_s={:.3} recent_s={:.3} cosine={:.4} elapsed_ms={}",
            split_sample as f32 / 16_000.0,
            change_abs_ms,
            change.reference_duration_s,
            change.recent_duration_s,
            change.cosine_similarity,
            change.elapsed_ms
        );
        self.pending_native_split_change_points
            .push_back(change_abs_ms);
        self.send_native_streaming_command(NativeStreamingCommand::SplitUtterance)
            .await?;
        Ok(true)
    }

    /// Attaches the diarized speaker to native-streaming transcript events.
    ///
    /// A label is bound to its utterance on the first terminal transcript
    /// (`is_final`) seen for the command that queued that label. Every
    /// diarizing split command queues one split slot, whether labelled or
    /// intentionally unlabelled, so split outcomes consume slots in command
    /// order.
    /// Once bound, every later event of that utterance (post-final revisions)
    /// reuses the label.
    pub(crate) async fn stamp_native_transcript_speaker(
        &mut self,
        source_kind: NativeStreamingCommandKind,
        envelope: &mut RealtimeEventEnvelope,
    ) -> bool {
        // Cheap no-op for non-diarizing sessions: labels only ever enter these
        // structures when a diarizer is active.
        if self.pending_native_speaker_labels.is_empty()
            && self.pending_native_split_speaker_slots.is_empty()
            && self.native_speaker_by_utterance.is_empty()
        {
            return false;
        }
        let RealtimeEvent::Transcript(transcript) = &envelope.event else {
            return false;
        };
        let (utterance_id, is_terminal, text_is_empty) = match transcript {
            RealtimeTranscriptEvent::Partial(event) => (
                event.utterance_id.clone(),
                false,
                event.text.trim().is_empty(),
            ),
            RealtimeTranscriptEvent::Final(event) => (
                event.utterance_id.clone(),
                event.is_final,
                event.text.trim().is_empty(),
            ),
            RealtimeTranscriptEvent::Revision(event) => (
                event.utterance_id.clone(),
                event.is_final,
                event.text.trim().is_empty(),
            ),
        };
        let can_bind_queued_label = is_terminal
            && matches!(
                source_kind,
                NativeStreamingCommandKind::Finalize | NativeStreamingCommandKind::Finish
            );
        let can_bind_split_label =
            is_terminal && matches!(source_kind, NativeStreamingCommandKind::SplitUtterance);
        if is_terminal && text_is_empty {
            return if can_bind_queued_label {
                self.pending_native_speaker_labels.pop_front().is_some()
            } else if can_bind_split_label {
                self.pending_native_split_speaker_slots
                    .pop_front()
                    .is_some()
            } else {
                false
            };
        }
        let mut consumed_queued_label = false;
        let assignment = match self.native_speaker_by_utterance.get(&utterance_id) {
            Some(assignment) => assignment.clone(),
            None if can_bind_queued_label => match self.pending_native_speaker_labels.pop_front() {
                Some(slot) => {
                    consumed_queued_label = true;
                    let assignment = self.resolve_native_speaker_slot(slot).await;
                    self.native_speaker_by_utterance
                        .insert(utterance_id.clone(), assignment.clone());
                    assignment
                }
                // No label queued yet: a terminal event without diarized audio.
                None => None,
            },
            None if can_bind_split_label => {
                match self.pending_native_split_speaker_slots.pop_front() {
                    Some(slot) => {
                        consumed_queued_label = true;
                        let assignment = self.resolve_native_speaker_slot(slot).await;
                        self.native_speaker_by_utterance
                            .insert(utterance_id.clone(), assignment.clone());
                        assignment
                    }
                    None => None,
                }
            }
            None => None,
        };
        if let Some(assignment) = assignment
            && let RealtimeEvent::Transcript(transcript) = &mut envelope.event
        {
            let (speaker_slot, label_slot, profile_slot) = match transcript {
                RealtimeTranscriptEvent::Partial(event) => (
                    &mut event.speaker,
                    &mut event.speaker_label,
                    &mut event.speaker_profile_id,
                ),
                RealtimeTranscriptEvent::Final(event) => (
                    &mut event.speaker,
                    &mut event.speaker_label,
                    &mut event.speaker_profile_id,
                ),
                RealtimeTranscriptEvent::Revision(event) => (
                    &mut event.speaker,
                    &mut event.speaker_label,
                    &mut event.speaker_profile_id,
                ),
            };
            *speaker_slot = Some(assignment.speaker);
            if assignment.speaker_profile_id.is_some() {
                *label_slot = Some(assignment.speaker_label);
                *profile_slot = assignment.speaker_profile_id;
            }
        }
        consumed_queued_label
    }

    pub(crate) fn remember_native_streaming_history(&mut self, envelope: &RealtimeEventEnvelope) {
        let (text, end_ms) = match &envelope.event {
            RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(event)) => {
                (event.text.trim(), event.end_ms)
            }
            RealtimeEvent::Transcript(RealtimeTranscriptEvent::Revision(event))
                if event.is_final =>
            {
                (event.text.trim(), event.end_ms)
            }
            _ => return,
        };
        if text.is_empty() {
            return;
        }
        self.history_text.push(text.to_string());
        self.history_duration_ms = self.history_duration_ms.max(end_ms);
    }

    /// Validates a binary websocket message's size and appends it to the
    /// PCM16LE carry buffer shared by both ingest paths.
    pub(crate) async fn ingest_binary_bytes(&mut self, bytes: &[u8]) -> Result<(), ()> {
        if bytes.len() > MAX_WS_MESSAGE_BYTES {
            self.emit_error(
                RealtimeErrorCode::UnsupportedAudioFormat,
                "Realtime binary audio message exceeded the configured WebSocket message limit.",
                false,
            )
            .await?;
            return Err(());
        }
        self.carry.extend_from_slice(bytes);
        Ok(())
    }

    /// Pops the next whole PCM16LE frame from the carry buffer, stamping and
    /// advancing the frame sequence/start time. `Ok(None)` once fewer than a
    /// frame's bytes remain buffered.
    pub(crate) async fn next_buffered_frame(&mut self) -> Result<Option<RealtimeAudioFrame>, ()> {
        if self.carry.len() < self.frame_byte_len {
            return Ok(None);
        }
        let frame_bytes = self.carry.drain(0..self.frame_byte_len).collect::<Vec<_>>();
        match RealtimeAudioFrame::from_pcm16le_bytes(
            self.next_frame_seq,
            self.next_frame_start_ms,
            RealtimeAudioFormat {
                encoding: RealtimeAudioEncoding::PcmS16Le,
                sample_rate_hz: 16_000,
                channels: 1,
            },
            &frame_bytes,
        ) {
            Ok(frame) => {
                self.next_frame_seq += 1;
                self.next_frame_start_ms += u64::from(self.frame_duration_ms);
                Ok(Some(frame))
            }
            Err(error) => {
                self.emit_error(
                    RealtimeErrorCode::UnsupportedAudioFormat,
                    &error.to_string(),
                    false,
                )
                .await?;
                Err(())
            }
        }
    }

    pub(crate) async fn handle_binary(&mut self, bytes: &[u8]) -> Result<(), ()> {
        if self.is_native_streaming() {
            return self.handle_native_streaming_binary(bytes).await;
        }
        let Some(controller) = self.controller.as_ref() else {
            self.emit_error(
                RealtimeErrorCode::StartupConfigError,
                "Realtime binary audio frames require session.start first.",
                false,
            )
            .await?;
            return Err(());
        };
        if controller.state() != RealtimeSessionState::Running {
            self.emit_error(
                RealtimeErrorCode::StartupConfigError,
                "Realtime binary audio frames require a running audio input.",
                false,
            )
            .await?;
            return Err(());
        }
        self.ingest_binary_bytes(bytes).await?;
        while let Some(frame) = self.next_buffered_frame().await? {
            if self.audio_frames.try_send(frame).is_err() {
                self.emit_error(
                    RealtimeErrorCode::AudioBufferOverflow,
                    "Realtime decoded audio frame queue is full; OpenASR stopped instead of silently dropping audio.",
                    false,
                )
                .await?;
                return Err(());
            }
            self.drain_audio_frames().await?;
        }
        self.drain_audio_frames().await
    }

    pub(crate) async fn handle_native_streaming_binary(&mut self, bytes: &[u8]) -> Result<(), ()> {
        self.ingest_binary_bytes(bytes).await?;
        while let Some(frame) = self.next_buffered_frame().await? {
            // No frame capture here: captured_audio_frames only feeds the
            // fallback path's dictation flush, which native streaming sessions
            // never reach — cloning every 20ms frame into it was pure waste.
            let (boundaries, is_speech, vad_in_speech) = self
                .controller
                .as_mut()
                .map(|controller| {
                    let (boundaries, is_speech) = controller.process_vad_frame_with_speech(&frame);
                    let vad_in_speech = controller.vad.state() == VadState::InSpeech;
                    (boundaries, is_speech, vad_in_speech)
                })
                .unwrap_or_else(|| (Vec::new(), false, false));
            if is_speech {
                self.native_had_speech_since_last_poll = true;
            }
            // Diarization needs the utterance audio after the decoder has
            // consumed it; retain a bounded copy of the active VAD span, not
            // only instantaneous high-probability frames. This matches the
            // fallback utterance path and keeps short edge turns embeddable.
            if self.streaming_diarizer.is_some() {
                let starts = boundaries.iter().filter_map(|boundary| match boundary {
                    SpeechBoundaryEvent::SpeechStarted { start_ms, .. } => Some(*start_ms),
                    _ => None,
                });
                for start_ms in starts {
                    self.backfill_native_diarize_preroll(start_ms, frame.seq);
                }
            }
            let boundary_keeps_frame = boundaries.iter().any(|boundary| {
                matches!(
                    boundary,
                    SpeechBoundaryEvent::SpeechStarted { .. }
                        | SpeechBoundaryEvent::SpeechStopped { .. }
                        | SpeechBoundaryEvent::MaxUtterance { .. }
                )
            });
            if self.streaming_diarizer.is_some()
                && (is_speech || vad_in_speech || boundary_keeps_frame)
            {
                self.retain_native_diarize_frame(&frame);
            }
            if self.streaming_diarizer.is_some() {
                self.remember_native_diarize_preroll_frame(&frame);
            }
            // Real silence finalizes (full decode reset); a forced
            // max-duration boundary only splits the transcript segment so the
            // recognition context survives the arbitrary mid-speech cut.
            let should_finalize = boundaries
                .iter()
                .any(|boundary| matches!(boundary, SpeechBoundaryEvent::SpeechStopped { .. }));
            let should_split = !should_finalize
                && boundaries
                    .iter()
                    .any(|boundary| matches!(boundary, SpeechBoundaryEvent::MaxUtterance { .. }));
            // VAD lifecycle events originate from connection-side VAD, not the
            // decoder — emit them at the WS edge so a multi-second decode (or a
            // cold first build) on the worker can never delay them.
            if !boundaries.is_empty() {
                self.emit_vad_boundary_events(&boundaries).await?;
            }
            // Decode runs on the dedicated worker thread. The WS task only queues
            // the frame here; native outcomes are drained separately so a slow
            // Poll cannot block audio ingest for this session.
            self.send_native_streaming_command(NativeStreamingCommand::PushAudio(frame))
                .await?;
            if should_finalize {
                self.native_had_speech_since_last_poll = false;
                if self.streaming_diarizer.is_some() {
                    self.queue_native_speaker_label().await;
                }
                self.send_native_streaming_command(NativeStreamingCommand::Finalize)
                    .await?;
            } else if should_split {
                if self.streaming_diarizer.is_some() {
                    self.queue_native_max_utterance_split_speaker_slot().await;
                }
                self.send_native_streaming_command(NativeStreamingCommand::SplitUtterance)
                    .await?;
            } else {
                let _ = self.maybe_split_native_on_speaker_change().await?;
            }
        }
        Ok(())
    }

    pub(crate) async fn drain_audio_frames(&mut self) -> Result<(), ()> {
        while let Ok(frame) = self.audio_frame_receiver.try_recv() {
            self.process_frame(frame).await?;
        }
        Ok(())
    }

    pub(crate) async fn process_frame(&mut self, frame: RealtimeAudioFrame) -> Result<(), ()> {
        self.remember_captured_audio_frame(&frame);
        let mut envelopes = Vec::new();
        let utterances = {
            let controller = self.controller.as_mut().expect("controller exists");
            let boundaries = controller.process_vad_frame(&frame);
            for boundary in &boundaries {
                match vad_boundary_event(boundary) {
                    VadBoundaryEvent::Vad(event) => {
                        envelopes.push(
                            controller
                                .vad_event(event, timestamp_now())
                                .map_err(|_| ())?,
                        );
                    }
                    VadBoundaryEvent::Error(error) => {
                        envelopes.push(
                            controller
                                .error_event(error, timestamp_now())
                                .map_err(|_| ())?,
                        );
                    }
                }
            }

            match controller.buffer.push_frame(frame, &boundaries) {
                Ok(utterances) => utterances,
                Err(error) => {
                    envelopes.push(
                        controller
                            .error_event(
                                RealtimeErrorEvent {
                                    code: RealtimeErrorCode::AudioBufferOverflow,
                                    message: error.to_string(),
                                    recoverable: false,
                                },
                                timestamp_now(),
                            )
                            .map_err(|_| ())?,
                    );
                    for envelope in envelopes {
                        self.emit_envelope(envelope).await?;
                    }
                    return Err(());
                }
            }
        };
        for envelope in envelopes {
            self.emit_envelope(envelope).await?;
        }
        for utterance in utterances {
            self.queue_utterance(utterance).await?;
        }
        Ok(())
    }

    pub(crate) async fn queue_utterance(&mut self, utterance: BufferedUtterance) -> Result<(), ()> {
        if utterance.reason == RealtimeUtteranceEndReason::Cancel {
            return Ok(());
        }
        let temp_wav = match write_temp_utterance_wav(&utterance) {
            Ok(temp_wav) => temp_wav,
            Err(error) => {
                self.emit_error(RealtimeErrorCode::BackendCrashed, &error.to_string(), false)
                    .await?;
                return Err(());
            }
        };
        // Diarize the utterance here, where its audio is still owned; the label is
        // attributed when the backend transcript returns (apply_backend_result).
        if self.streaming_diarizer.is_some() {
            let samples = utterance_speech_samples_f32(&utterance);
            if let Some(label) = self
                .assign_speaker_off_loop(
                    samples,
                    openasr_core::diarize::streaming::StreamingDiarizePath::Fallback,
                )
                .await
            {
                self.pending_utterance_speakers
                    .insert(utterance.utterance_id.clone(), label);
            }
        }
        let controller = self.controller.as_ref().expect("controller exists");
        let segment_id = TranscriptSegmentId(format!("{}_seg_000001", utterance.utterance_id.0));
        let job = BackendJob {
            utterance_id: utterance.utterance_id,
            start_ms: utterance.start_ms,
            end_ms: utterance.end_ms,
            segment_id,
            model_id: controller.config().model_id.clone(),
            language: self.language.clone(),
            task: self.task,
            prompt: self.prompt.clone(),
            phrase_bias: self.phrase_bias.clone(),
            inference_threads: self.inference_threads,
            execution_target: self.execution_target,
            word_timestamps: self.word_timestamps,
            display_name: "realtime-utterance.wav".to_string(),
            temp_wav,
        };
        let Some(sender) = &self.backend_jobs else {
            self.emit_error(
                RealtimeErrorCode::BackendCrashed,
                "Realtime backend worker is not running.",
                false,
            )
            .await?;
            return Err(());
        };
        let Some(result_sender) = self.backend_result_sender.as_ref().cloned() else {
            self.emit_error(
                RealtimeErrorCode::BackendCrashed,
                "Realtime backend result channel is not running.",
                false,
            )
            .await?;
            return Err(());
        };
        if self.pending_backend_jobs >= BACKEND_JOB_QUEUE_CAPACITY {
            self.emit_error(
                RealtimeErrorCode::BackpressureTimeout,
                "Realtime backend worker queue is full; OpenASR stopped instead of buffering unbounded utterances.",
                false,
            )
            .await?;
            return Err(());
        }
        let work_item = RealtimeBackendWorkItem {
            session_key: self.session_id.0.clone(),
            job,
            result_sender,
            cancelled: Arc::clone(&self.backend_cancelled),
        };
        if sender
            .try_send(RealtimeBackendWorkerMessage::Job(work_item))
            .is_err()
        {
            self.emit_error(
                RealtimeErrorCode::BackpressureTimeout,
                "Realtime backend worker queue is full; OpenASR stopped instead of buffering unbounded utterances.",
                false,
            )
            .await?;
            return Err(());
        }
        self.pending_backend_jobs += 1;
        Ok(())
    }

    pub(crate) async fn queue_dictation_fallback_utterance(&mut self) -> Result<(), ()> {
        if self.source_name.as_deref() != Some(DICTATION_SOURCE_NAME)
            || self.pending_backend_jobs > 0
            || !self.history_text.is_empty()
        {
            return Ok(());
        }
        if self.controller.is_none() {
            return Ok(());
        }
        let frames = self
            .captured_audio_frames
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        if !dictation_fallback_has_audible_audio(&frames) {
            return Ok(());
        }
        let Some(first) = frames.first() else {
            return Ok(());
        };
        let Some(last) = frames.last() else {
            return Ok(());
        };
        let utterance = BufferedUtterance {
            utterance_id: TranscriptUtteranceId("utt_dictation_000001".to_string()),
            start_ms: first.start_ms,
            end_ms: last.end_ms(),
            frames,
            reason: RealtimeUtteranceEndReason::Flush,
        };
        self.queue_utterance(utterance).await
    }

    pub(crate) async fn apply_backend_result(&mut self, result: BackendResult) -> Result<(), ()> {
        self.pending_backend_jobs = self.pending_backend_jobs.saturating_sub(1);
        match result {
            BackendResult::Final(result) => {
                let speaker_assignment =
                    self.pending_utterance_speakers.remove(&result.utterance_id);
                let controller = self.controller.as_mut().expect("controller exists");
                let history_text = result.text.trim().to_string();
                let history_end_ms = result.end_ms;
                let (speaker, speaker_label, speaker_profile_id) = speaker_assignment
                    .map(|assignment| {
                        let speaker_label = assignment
                            .speaker_profile_id
                            .is_some()
                            .then_some(assignment.speaker_label);
                        (
                            Some(assignment.speaker),
                            speaker_label,
                            assignment.speaker_profile_id,
                        )
                    })
                    .unwrap_or((None, None, None));
                let update = TranscriptUpdate {
                    utterance_id: result.utterance_id,
                    segment_id: result.segment_id,
                    revision: 1,
                    text: result.text,
                    start_ms: result.start_ms,
                    end_ms: result.end_ms,
                    language: result.language,
                    speaker,
                    speaker_label,
                    speaker_profile_id,
                    words: result.words,
                    revises_event_id: None,
                };
                if let TranscriptLifecycleResult::Event(event) =
                    controller.transcript.apply_final(update, None)
                {
                    let final_segment = match &event {
                        RealtimeTranscriptEvent::Final(final_event) => Some((
                            final_event.utterance_id.clone(),
                            final_event.segment_id.clone(),
                            final_event.revision,
                        )),
                        _ => None,
                    };
                    let envelope = controller
                        .transcript_event(event, timestamp_now())
                        .map_err(|_| ())?;
                    if let Some((utterance_id, segment_id, revision)) = final_segment {
                        controller.transcript.record_final_event_id(
                            &utterance_id,
                            &segment_id,
                            revision,
                            envelope.event_id.clone(),
                        );
                    }
                    self.emit_envelope_with_translation(envelope).await?;
                    if !history_text.is_empty() {
                        self.history_text.push(history_text);
                        self.history_duration_ms = self.history_duration_ms.max(history_end_ms);
                    }
                }
                Ok(())
            }
            BackendResult::Error(message) => {
                self.backend_failed = true;
                self.cancel_backend_jobs();
                self.emit_error(RealtimeErrorCode::BackendCrashed, &message, false)
                    .await?;
                Err(())
            }
        }
    }

    pub(crate) async fn finish(&mut self, reason: &str, close: bool) -> Result<(), ()> {
        if self.closed {
            return Ok(());
        }
        let transport_closed = reason == "transport_closed";
        if self.is_native_streaming() {
            return self
                .finish_native_streaming_session(close, transport_closed)
                .await;
        }
        if transport_closed {
            self.carry.clear();
            self.cancel_backend_jobs();
        }
        if !self.backend_failed && !transport_closed {
            if !self.carry.is_empty() {
                self.emit_error(
                    RealtimeErrorCode::UnsupportedAudioFormat,
                    "Realtime PCM16LE input ended with an incomplete frame; no audio bytes were silently dropped.",
                    false,
                )
                .await?;
                self.closed = true;
                return Err(());
            }
            self.drain_audio_frames().await?;
            if let Some(controller) = self.controller.as_mut()
                && let Some(utterance) = controller.buffer.flush(self.next_frame_start_ms)
            {
                self.queue_utterance(utterance).await?;
            }
            self.queue_dictation_fallback_utterance().await?;
        }
        if self.backend_failed {
            self.cancel_backend_jobs();
        }
        while self.pending_backend_jobs > 0 && !self.backend_failed {
            let result = self.recv_backend_result().await;
            if let Some(result) = result {
                match result {
                    final_result @ BackendResult::Final(_) => {
                        self.apply_backend_result(final_result).await?
                    }
                    BackendResult::Error(message) => {
                        self.pending_backend_jobs = self.pending_backend_jobs.saturating_sub(1);
                        if !self.backend_failed {
                            self.emit_error(RealtimeErrorCode::BackendCrashed, &message, false)
                                .await?;
                        }
                        self.backend_failed = true;
                        self.cancel_backend_jobs();
                    }
                }
            } else {
                self.emit_error(
                    RealtimeErrorCode::BackendCrashed,
                    "Realtime backend worker stopped before returning all utterance results.",
                    false,
                )
                .await?;
                return Err(());
            }
        }

        if !self.backend_failed && !transport_closed {
            self.drain_translation_until_idle().await?;
        }

        if !self.backend_failed && !transport_closed && self.record_history {
            self.record_history_entry().await?;
        }

        let stopped = match self.controller.as_mut() {
            Some(controller) if controller.state() == RealtimeSessionState::Running => Some(
                controller
                    .lifecycle(
                        RealtimeLifecycleAction::StopAudio {
                            reason: reason.to_string(),
                        },
                        timestamp_now(),
                    )
                    .map_err(|_| ())?,
            ),
            _ => None,
        };
        if let Some(stopped) = stopped {
            self.emit_envelope(stopped).await?;
        }
        let closed = match self.controller.as_mut() {
            Some(controller)
                if close
                    && !matches!(
                        controller.state(),
                        RealtimeSessionState::Closed | RealtimeSessionState::Cancelled
                    ) =>
            {
                Some(
                    controller
                        .lifecycle(
                            RealtimeLifecycleAction::Close {
                                reason: reason.to_string(),
                            },
                            timestamp_now(),
                        )
                        .map_err(|_| ())?,
                )
            }
            _ => None,
        };
        if let Some(closed) = closed {
            self.emit_envelope(closed).await?;
            self.closed = true;
        }
        if transport_closed || self.backend_failed {
            self.translation = None;
        }
        if self.backend_failed { Err(()) } else { Ok(()) }
    }

    pub(crate) async fn finish_native_streaming_session(
        &mut self,
        close: bool,
        transport_closed: bool,
    ) -> Result<(), ()> {
        if transport_closed {
            self.carry.clear();
            if let Some(worker) = self.native_streaming.take() {
                worker.detach_cancel();
            }
            self.translation = None;
            self.closed = true;
            return Ok(());
        } else if !self.carry.is_empty() {
            self.emit_error(
                RealtimeErrorCode::UnsupportedAudioFormat,
                "Realtime PCM16LE input ended with an incomplete frame; no audio bytes were silently dropped.",
                false,
            )
            .await?;
            self.closed = true;
            return Err(());
        }

        if !self.is_native_streaming() {
            return Ok(());
        }
        // A stop mid-speech never reaches the VAD SpeechStopped path, so the
        // in-flight utterance's retained audio is still undiarized: queue its
        // label now or the Finish-induced terminal transcript stays anonymous.
        if !transport_closed && !self.native_diarize_samples.is_empty() {
            self.queue_native_speaker_label().await;
        }
        let command = if transport_closed {
            NativeStreamingCommand::Cancel
        } else {
            NativeStreamingCommand::Finish { close }
        };
        let (kind, events) = self.native_streaming_command(command).await?;
        self.forward_native_streaming_events(kind, events).await?;
        if !self.backend_failed && !transport_closed {
            self.drain_translation_until_idle().await?;
        }
        // The worker exited after the terminal command; join it so the session
        // (and its decoder cache) is dropped before we report the session closed.
        if let Some(worker) = self.native_streaming.take() {
            worker.join();
        }
        if transport_closed || self.backend_failed {
            self.translation = None;
        }
        if !self.backend_failed && !transport_closed && self.record_history {
            self.record_history_entry().await?;
        }
        self.closed = true;
        Ok(())
    }

    pub(crate) async fn cancel(&mut self, reason: &str) -> Result<(), ()> {
        if self.is_native_streaming() {
            if let Some(worker) = self.native_streaming.take() {
                worker.detach_cancel();
            }
            self.translation = None;
            self.emit_error(RealtimeErrorCode::Cancelled, reason, false)
                .await?;
            if let Some(controller) = self.controller.as_mut() {
                let (closed, _) = controller
                    .cancel(self.next_frame_start_ms, timestamp_now())
                    .map_err(|_| ())?;
                self.emit_envelope(closed).await?;
            }
            self.closed = true;
            return Err(());
        }
        self.cancel_backend_jobs();
        self.emit_error(RealtimeErrorCode::Cancelled, reason, false)
            .await?;
        if let Some(controller) = self.controller.as_mut() {
            let (closed, _) = controller
                .cancel(self.next_frame_start_ms, timestamp_now())
                .map_err(|_| ())?;
            self.emit_envelope(closed).await?;
        }
        self.closed = true;
        Err(())
    }

    pub(crate) fn cancel_backend_jobs(&mut self) {
        self.backend_cancelled.store(true, Ordering::Relaxed);
        self.backend_jobs.take();
        self.backend_result_sender.take();
        self.pending_backend_jobs = 0;
        // Drop any per-utterance speaker labels whose transcript will never
        // arrive, so the map cannot grow unbounded across cancels/resets.
        self.pending_utterance_speakers.clear();
        self.native_diarize_samples = Vec::new();
        self.native_diarize_sample_spans.clear();
        self.pending_native_speaker_labels.clear();
        self.native_speaker_by_utterance.clear();
        self.pending_native_split_change_points.clear();
        self.native_speakerless_finals.clear();
        self.pending_split_tail_relabels.clear();
        self.translation = None;
    }

    pub(crate) fn remember_captured_audio_frame(&mut self, frame: &RealtimeAudioFrame) {
        self.captured_audio_frames.push_back(frame.clone());
        let max_frames = self
            .controller
            .as_ref()
            .map(|controller| controller.config().buffer.max_buffered_frames)
            .unwrap_or(DEFAULT_MAX_BUFFERED_FRAMES);
        while self.captured_audio_frames.len() > max_frames {
            self.captured_audio_frames.pop_front();
        }
    }

    pub(crate) async fn record_history_entry(&mut self) -> Result<(), ()> {
        if self.history_recorded || self.history_text.is_empty() {
            return Ok(());
        }
        let home = match self.distribution.openasr_home() {
            Ok(home) => home,
            Err(error) => {
                self.emit_error(
                    RealtimeErrorCode::BackendCrashed,
                    &format!("Could not resolve OpenASR home for realtime history: {error}"),
                    false,
                )
                .await?;
                return Err(());
            }
        };
        // History persistence is governed solely by the saved-history scope
        // (`history_retention`), matching the file-transcription path.
        // `auto_save` controls transcript-file exports and must not gate
        // history. "Off" retention is fail-fast: never write a transcript we
        // would only prune away on the next sweep.
        let document = openasr_core::config::load_config_document(&home).unwrap_or_default();
        if !document
            .preferences
            .history_retention
            .persists_new_entries()
        {
            return Ok(());
        }
        let text = self.history_text.join("\n").trim().to_string();
        if text.is_empty() {
            return Ok(());
        }
        let Some(controller) = self.controller.as_ref() else {
            return Ok(());
        };
        let model = controller.config().model_id.clone();
        let store = DaemonHistoryStore::open(&home);
        if let Err(error) = store.record(DaemonHistoryRecord {
            kind: DaemonHistoryKind::Live,
            model,
            source_name: self
                .source_name
                .clone()
                .or_else(|| Some("Live".to_string())),
            duration_seconds: Some(self.history_duration_ms as f32 / 1000.0),
            output_format: Some(ResponseFormat::Text),
            diarization_active: Some(self.streaming_diarizer.is_some()),
            provenance: Some(DaemonHistoryProvenance::Recorded),
            // Live daemon history persists only the aggregated transcript text;
            // per-segment timing lives in the realtime transcript history. No
            // segments here means the store advertises text-shaped exports only.
            segments: Vec::new(),
            text,
        }) {
            self.emit_error(
                RealtimeErrorCode::BackendCrashed,
                &format!("Could not write realtime transcription history: {error}"),
                false,
            )
            .await?;
            return Err(());
        }
        self.history_recorded = true;
        // Mirror the file-transcription path: best-effort prune after recording so live
        // history honors the retention policy on write, not only on the next
        // /v1/history read.
        if let Ok(document) = openasr_core::config::load_config_document(&home)
            && let Err(error) =
                crate::prune_history_store(&store, document.preferences.history_retention)
        {
            eprintln!("openasr-server: could not prune realtime history (continuing): {error}");
        }
        Ok(())
    }

    /// Emits VAD lifecycle events for connection-side speech boundaries
    /// directly on the connection sequence. Shares the boundary mapping with
    /// the fallback path; only the emission policy differs (edge vs
    /// controller-routed).
    async fn emit_vad_boundary_events(
        &mut self,
        boundaries: &[SpeechBoundaryEvent],
    ) -> Result<(), ()> {
        for boundary in boundaries {
            match vad_boundary_event(boundary) {
                VadBoundaryEvent::Vad(event) => {
                    self.emit_event(RealtimeEvent::Vad(event)).await?;
                }
                VadBoundaryEvent::Error(error) => {
                    self.emit_error(error.code, &error.message, error.recoverable)
                        .await?;
                }
            }
        }
        Ok(())
    }

    pub(crate) async fn emit_error(
        &mut self,
        code: RealtimeErrorCode,
        message: &str,
        recoverable: bool,
    ) -> Result<(), ()> {
        // Errors are a connection-edge concern: they must be reportable in any
        // session state (including terminal), so they bypass the controller and
        // stamp directly onto the connection sequence.
        self.emit_event(RealtimeEvent::Error(RealtimeErrorEvent {
            code,
            message: message.to_string(),
            recoverable,
        }))
        .await
    }

    pub(crate) fn spawn_backend_worker(&mut self) {
        let (result_sender, result_receiver) =
            mpsc::channel::<BackendResult>(BACKEND_JOB_QUEUE_CAPACITY);
        self.backend_jobs = Some(realtime_backend_worker_for_runtime(self.runtime.clone()));
        self.backend_results = Some(result_receiver);
        self.backend_result_sender = Some(result_sender);
    }
}
