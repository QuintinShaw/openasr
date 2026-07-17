//! In-process streaming transcription session.
//!
//! This is the library-level counterpart to the desktop server's realtime
//! WebSocket/SSE path: it takes 16 kHz mono `f32` PCM chunks, drives the same
//! native ggml streaming engine the server uses
//! ([`NativeAsrSession`]/[`NativeBackendExecutor`]), and returns incremental
//! partial/committed transcript events plus a final [`Transcription`]. It is
//! deliberately transport-free (no axum, no tokio) so an embedder that cannot
//! spawn a local HTTP server -- iOS in particular, where process spawning and
//! background servers are disallowed -- can run live captioning entirely
//! in-process by linking the open core.
//!
//! It reuses, rather than reinvents, the pieces the server already relies on:
//!
//! - Model resolution + decode: [`native_runtime_model_adapter_for_path`] and
//!   [`NativeAsrExecutor::start_streaming_session`], which return the same
//!   `Box<dyn NativeAsrSession>` the server's native worker drives. Every
//!   builtin ASR family registers a streaming executor, so packs go through
//!   the shared greedy-decode driver -- this session never hand-rolls a decode
//!   loop.
//! - Utterance segmentation: the core energy [`VadStateMachine`], the same one
//!   the server's session controller uses to turn a continuous stream into
//!   per-utterance boundaries.
//!
//! The engine is fail-closed and offline: it only ever runs the local `.oasr`
//! pack it is handed and never reaches for the network.

use std::path::Path;

use crate::realtime::{RealtimeEvent, RealtimeTranscriptEvent, RealtimeTranscriptWord};
use crate::{
    NativeAsrError, NativeAsrExecutor, NativeAsrHardwareTarget, NativeAsrModelAdapter,
    NativeAsrModelPackRef, NativeAsrRequestOptions, NativeAsrSession, NativeAsrSessionContext,
    NativeAsrStreamingSessionConfig, NativeBackendExecutor, RealtimeAudioFormat,
    RealtimeAudioFrame, RealtimeEventEnvelope, Segment, Transcription, VadConfig, VadStateMachine,
    native_runtime_model_adapter_for_path,
};

/// The kind of a [`StreamingEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingEventKind {
    /// A mutable in-progress hypothesis for the active utterance. Superseded by
    /// later `Partial`s for the same `segment_id`, then by a `Committed` event.
    Partial,
    /// A settled segment. Emitted when a VAD speech pause (or `finish`) closes
    /// an utterance; its text is stable and will not be revised except by a
    /// later `Revision` carrying the same `segment_id`.
    Committed,
    /// A post-final correction to an already-committed segment (e.g. trailing
    /// punctuation added once the segment settled). Rare; carries the same
    /// `segment_id` as the `Committed` event it revises.
    Revision,
}

/// An incremental transcript update produced by [`StreamingSession::feed`] or
/// [`StreamingSession::finish`].
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingEvent {
    pub kind: StreamingEventKind,
    pub utterance_id: String,
    pub segment_id: String,
    /// Monotonic revision number for this segment; higher supersedes lower.
    pub revision: u64,
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    /// Per-word timings, present only when `word_timestamps` was requested and
    /// the model family produces them.
    pub words: Vec<RealtimeTranscriptWord>,
    /// Detected language for this segment, when the family reports one.
    pub language: Option<String>,
}

/// Configuration for a [`StreamingSession`].
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Emit mutable `Partial` events as audio arrives. When `false`, only
    /// `Committed`/final segments surface.
    pub partial_results: bool,
    /// Attach per-word timings to events when the family supports them.
    pub word_timestamps: bool,
    /// When `Some`, an energy VAD segments the stream into utterances and a
    /// `Committed` event is emitted at each speech pause -- the shape a live
    /// caption UI wants. When `None`, the whole stream is treated as a single
    /// utterance and only finalized at [`StreamingSession::finish`].
    pub vad: Option<VadConfig>,
    /// Optional decode language hint passed through to the model.
    pub language: Option<String>,
    /// Hardware target for the decode session. `Auto` lets the runtime choose.
    pub hardware_target: NativeAsrHardwareTarget,
    /// Optional inference thread cap.
    pub inference_threads: Option<u16>,
    /// Minimum new audio (ms) between partial re-decodes; `None` uses the
    /// per-family default. See [`NativeAsrStreamingSessionConfig`].
    pub min_partial_interval_ms: Option<u32>,
    /// How much new audio (ms) to accumulate before polling the engine for a
    /// partial re-decode. Mirrors the server's time-spaced poll: a fast family
    /// whose encoder needs more than one frame must not be polled on a
    /// near-empty buffer. Also caps partial re-decode frequency. Must be >= the
    /// 20 ms frame size; smaller values are clamped up.
    pub partial_poll_interval_ms: u64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            partial_results: true,
            word_timestamps: false,
            vad: Some(VadConfig::default()),
            language: None,
            hardware_target: NativeAsrHardwareTarget::Auto,
            inference_threads: None,
            min_partial_interval_ms: None,
            partial_poll_interval_ms: DEFAULT_PARTIAL_POLL_INTERVAL_MS,
        }
    }
}

/// Default audio spacing between partial polls. Comfortably above any builtin
/// family's encoder minimum framing while still a responsive live cadence.
const DEFAULT_PARTIAL_POLL_INTERVAL_MS: u64 = 200;

/// The audio contract: 16 kHz, mono, `f32` in `[-1.0, 1.0]`.
const SAMPLE_RATE_HZ: u32 = 16_000;
/// Frame the incoming stream at 20 ms (a supported realtime frame duration).
const FRAME_DURATION_MS: u32 = 20;
const FRAME_SAMPLES: usize = (SAMPLE_RATE_HZ as usize * FRAME_DURATION_MS as usize) / 1_000;

/// A single accumulated segment, keyed by `segment_id`, tracked so `finish`
/// can assemble a [`Transcription`] from the same events the caller streamed.
#[derive(Debug, Clone)]
struct TrackedSegment {
    id: String,
    order: usize,
    start_ms: u64,
    end_ms: u64,
    text: String,
    words: Vec<RealtimeTranscriptWord>,
    language: Option<String>,
}

/// An in-process streaming transcription session over a local `.oasr` pack.
pub struct StreamingSession {
    session: Box<dyn NativeAsrSession>,
    vad: Option<VadStateMachine>,
    /// Leftover f32 samples that did not fill a whole 20 ms frame yet.
    pending: Vec<f32>,
    audio_format: RealtimeAudioFormat,
    next_seq: u64,
    /// Absolute ms offset of the next frame's first sample.
    next_start_ms: u64,
    word_timestamps: bool,
    /// New audio (samples) accumulated since the last partial poll.
    samples_since_poll: usize,
    /// Poll the engine once this many new samples have accumulated.
    poll_interval_samples: usize,
    /// Ordered accumulation of the latest text per segment, for `finish`.
    segments: Vec<TrackedSegment>,
    closed: bool,
}

impl StreamingSession {
    /// Open a streaming session over the local `.oasr` pack at `pack_path`.
    ///
    /// Fails closed with a typed [`NativeAsrError`] if the pack cannot be
    /// resolved to a known model family or the family cannot start a streaming
    /// session. Never touches the network.
    pub fn new(pack_path: &Path, cfg: StreamingConfig) -> Result<Self, NativeAsrError> {
        let adapter = native_runtime_model_adapter_for_path(pack_path).ok_or_else(|| {
            NativeAsrError::SessionFailed {
                message: format!(
                    "no native model family recognizes the pack at {}",
                    pack_path.display()
                ),
            }
        })?;
        let model_pack = NativeAsrModelPackRef::new(
            "native-streaming",
            adapter.model_family(),
            pack_path.to_path_buf(),
        );
        let audio_format = RealtimeAudioFormat::pcm16_mono_16khz();
        let options = NativeAsrRequestOptions::new()
            .with_language(cfg.language.clone())
            .with_inference_threads(cfg.inference_threads)
            .with_partial_results(cfg.partial_results)
            .with_word_timestamps(cfg.word_timestamps);
        let session_config = NativeAsrStreamingSessionConfig::new()
            .with_audio_format(audio_format)
            .with_partial_results(cfg.partial_results)
            .with_word_timestamps(cfg.word_timestamps)
            .with_min_partial_interval_ms(cfg.min_partial_interval_ms);
        let executor = NativeBackendExecutor;
        let session = NativeAsrExecutor::start_streaming_session(
            &executor,
            &adapter,
            &model_pack,
            cfg.hardware_target,
            NativeAsrSessionContext::new("in-process-streaming"),
            options,
            session_config,
        )?;
        Self::from_native_session(session, &cfg)
    }

    /// Build a session around an already-started [`NativeAsrSession`]. Used by
    /// [`Self::new`] and by tests that inject a deterministic fake session.
    fn from_native_session(
        session: Box<dyn NativeAsrSession>,
        cfg: &StreamingConfig,
    ) -> Result<Self, NativeAsrError> {
        let vad = match &cfg.vad {
            Some(vad_cfg) => {
                let vad_cfg = VadConfig {
                    frame_duration_ms: FRAME_DURATION_MS,
                    ..*vad_cfg
                };
                Some(VadStateMachine::new(vad_cfg).map_err(|error| {
                    NativeAsrError::SessionFailed {
                        message: format!("invalid streaming VAD config: {error}"),
                    }
                })?)
            }
            None => None,
        };
        let poll_interval_samples =
            ((cfg.partial_poll_interval_ms as usize) * (SAMPLE_RATE_HZ as usize) / 1_000)
                .max(FRAME_SAMPLES);
        Ok(Self {
            session,
            vad,
            pending: Vec::with_capacity(FRAME_SAMPLES),
            audio_format: RealtimeAudioFormat::pcm16_mono_16khz(),
            next_seq: 1,
            next_start_ms: 0,
            word_timestamps: cfg.word_timestamps,
            samples_since_poll: 0,
            poll_interval_samples,
            segments: Vec::new(),
            closed: false,
        })
    }

    /// Feed a chunk of 16 kHz mono `f32` PCM (any length). Returns the
    /// incremental transcript events produced by this chunk: `Partial`s for the
    /// active utterance and a `Committed` event whenever a VAD speech pause
    /// closes one.
    pub fn feed(&mut self, pcm: &[f32]) -> Result<Vec<StreamingEvent>, NativeAsrError> {
        if self.closed {
            return Err(NativeAsrError::SessionClosed);
        }
        self.pending.extend_from_slice(pcm);
        let mut events = Vec::new();
        while self.pending.len() >= FRAME_SAMPLES {
            let samples = self.pending.drain(..FRAME_SAMPLES).collect::<Vec<f32>>();
            self.process_frame(samples, &mut events)?;
        }
        Ok(events)
    }

    /// Finish the stream: drain any buffered tail audio, finalize the active
    /// utterance, and return the assembled full [`Transcription`].
    pub fn finish(mut self) -> Result<Transcription, NativeAsrError> {
        if !self.pending.is_empty() {
            // Pad the tail to a whole frame with silence so the last samples
            // still reach the decoder.
            let mut samples = std::mem::take(&mut self.pending);
            samples.resize(FRAME_SAMPLES, 0.0);
            let mut drained = Vec::new();
            self.process_frame(samples, &mut drained)?;
        }
        let finish_events = self.session.finish()?;
        self.record_events(&finish_events);
        self.closed = true;
        Ok(self.assemble_transcription())
    }

    fn process_frame(
        &mut self,
        samples: Vec<f32>,
        out: &mut Vec<StreamingEvent>,
    ) -> Result<(), NativeAsrError> {
        let start_ms = self.next_start_ms;
        let frame = self.build_frame(samples, start_ms)?;

        // Compute VAD boundaries before the frame is consumed by push_audio.
        let boundaries = self
            .vad
            .as_mut()
            .map(|vad| vad.process_energy_frame(&frame))
            .unwrap_or_default();

        // Push audio (advances the decode), then poll on a time cadence:
        // buffered re-decode families emit their cadence-driven partials from
        // `poll_events`, not from `push_audio`, so the server's session loop
        // polls on a timer. Fast families (moonshine) attempt a partial as soon
        // as audio exists, but their encoder needs more than a single 20 ms
        // frame; the server never decodes a near-empty buffer because its poll
        // is time-spaced. Match that here by only polling once at least
        // `poll_interval_samples` of new audio have accumulated. The driver's
        // own `min_partial_interval` throttles further; polling more often just
        // wastes a re-decode.
        let pushed = self.session.push_audio(frame)?;
        self.emit(&pushed, out);
        self.samples_since_poll += FRAME_SAMPLES;
        if self.samples_since_poll >= self.poll_interval_samples {
            self.samples_since_poll = 0;
            let polled = self.session.poll_events()?;
            self.emit(&polled, out);
        }

        for boundary in &boundaries {
            use crate::SpeechBoundaryEvent::*;
            match boundary {
                SpeechStopped { .. } => {
                    // A real speech pause: finalize the utterance (hard
                    // boundary) so its text commits and the next one starts.
                    let finalized = self.session.finalize_utterance()?;
                    self.emit(&finalized, out);
                    self.samples_since_poll = 0;
                }
                MaxUtterance { .. } => {
                    // A forced max-duration cut: split without treating it as a
                    // language boundary.
                    let split = self.session.split_utterance()?;
                    self.emit(&split, out);
                    self.samples_since_poll = 0;
                }
                SpeechStarted { .. } | NoSpeechTimeout { .. } => {}
            }
        }

        self.next_seq += 1;
        self.next_start_ms += FRAME_DURATION_MS as u64;
        Ok(())
    }

    /// Accumulate settled text and append the caller-facing events for one
    /// batch of engine envelopes.
    fn emit(&mut self, envelopes: &[RealtimeEventEnvelope], out: &mut Vec<StreamingEvent>) {
        self.record_events(envelopes);
        out.extend(self.map_events(envelopes));
    }

    fn build_frame(
        &self,
        samples: Vec<f32>,
        start_ms: u64,
    ) -> Result<RealtimeAudioFrame, NativeAsrError> {
        let pcm16 = samples.into_iter().map(f32_to_i16).collect::<Vec<i16>>();
        RealtimeAudioFrame::new(self.next_seq, start_ms, self.audio_format, pcm16).map_err(
            |error| NativeAsrError::SessionFailed {
                message: format!("invalid streaming audio frame: {error}"),
            },
        )
    }

    /// Translate the engine's rich wire events into the caller-facing
    /// [`StreamingEvent`]s, keeping only transcript updates.
    fn map_events(&self, envelopes: &[RealtimeEventEnvelope]) -> Vec<StreamingEvent> {
        envelopes
            .iter()
            .filter_map(|envelope| self.map_event(envelope))
            .collect()
    }

    fn map_event(&self, envelope: &RealtimeEventEnvelope) -> Option<StreamingEvent> {
        let RealtimeEvent::Transcript(transcript) = &envelope.event else {
            return None;
        };
        let event = match transcript {
            RealtimeTranscriptEvent::Partial(partial) => StreamingEvent {
                kind: StreamingEventKind::Partial,
                utterance_id: partial.utterance_id.0.clone(),
                segment_id: partial.segment_id.0.clone(),
                revision: partial.revision,
                text: partial.text.clone(),
                start_ms: partial.start_ms,
                end_ms: partial.end_ms,
                words: self.select_words(&partial.words),
                language: partial.language.clone(),
            },
            RealtimeTranscriptEvent::Final(final_event) => StreamingEvent {
                kind: StreamingEventKind::Committed,
                utterance_id: final_event.utterance_id.0.clone(),
                segment_id: final_event.segment_id.0.clone(),
                revision: final_event.revision,
                text: final_event.text.clone(),
                start_ms: final_event.start_ms,
                end_ms: final_event.end_ms,
                words: self.select_words(&final_event.words),
                language: final_event.language.clone(),
            },
            RealtimeTranscriptEvent::Revision(revision) => StreamingEvent {
                kind: StreamingEventKind::Revision,
                utterance_id: revision.utterance_id.0.clone(),
                segment_id: revision.segment_id.0.clone(),
                revision: revision.revision,
                text: revision.text.clone(),
                start_ms: revision.start_ms,
                end_ms: revision.end_ms,
                words: self.select_words(&revision.words),
                language: revision.language.clone(),
            },
        };
        Some(event)
    }

    fn select_words(&self, words: &[RealtimeTranscriptWord]) -> Vec<RealtimeTranscriptWord> {
        if self.word_timestamps {
            words.to_vec()
        } else {
            Vec::new()
        }
    }

    /// Accumulate the latest committed/final/revision text per segment so
    /// `finish` can assemble the full transcript. Partials are intentionally
    /// ignored here -- only settled text contributes to the final transcript.
    fn record_events(&mut self, envelopes: &[RealtimeEventEnvelope]) {
        for envelope in envelopes {
            let RealtimeEvent::Transcript(transcript) = &envelope.event else {
                continue;
            };
            let (segment_id, start_ms, end_ms, text, words, language) = match transcript {
                RealtimeTranscriptEvent::Final(event) => (
                    event.segment_id.0.clone(),
                    event.start_ms,
                    event.end_ms,
                    event.text.clone(),
                    event.words.clone(),
                    event.language.clone(),
                ),
                RealtimeTranscriptEvent::Revision(event) => (
                    event.segment_id.0.clone(),
                    event.start_ms,
                    event.end_ms,
                    event.text.clone(),
                    event.words.clone(),
                    event.language.clone(),
                ),
                RealtimeTranscriptEvent::Partial(_) => continue,
            };
            self.upsert_segment(segment_id, start_ms, end_ms, text, words, language);
        }
    }

    fn upsert_segment(
        &mut self,
        segment_id: String,
        start_ms: u64,
        end_ms: u64,
        text: String,
        words: Vec<RealtimeTranscriptWord>,
        language: Option<String>,
    ) {
        if let Some(existing) = self
            .segments
            .iter_mut()
            .find(|segment| segment.segment_id_matches(&segment_id))
        {
            existing.start_ms = start_ms;
            existing.end_ms = end_ms;
            existing.text = text;
            existing.words = words;
            existing.language = language;
            return;
        }
        let order = self.segments.len();
        self.segments.push(TrackedSegment {
            order,
            start_ms,
            end_ms,
            text,
            words,
            language,
            id: segment_id,
        });
    }

    fn assemble_transcription(&self) -> Transcription {
        let mut ordered = self.segments.clone();
        ordered.sort_by_key(|segment| segment.order);
        let segments = ordered
            .iter()
            .filter(|segment| !segment.text.trim().is_empty())
            .map(|segment| Segment {
                start: ms_to_seconds(segment.start_ms),
                end: ms_to_seconds(segment.end_ms),
                text: segment.text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: if self.word_timestamps {
                    segment
                        .words
                        .iter()
                        .map(realtime_word_to_timestamp)
                        .collect()
                } else {
                    Vec::new()
                },
            })
            .collect::<Vec<_>>();
        let text = join_segment_texts(segments.iter().map(|segment| segment.text.as_str()));
        let language = ordered.iter().find_map(|segment| segment.language.clone());
        Transcription {
            text,
            segments,
            longform: None,
            language,
        }
    }
}

impl TrackedSegment {
    fn segment_id_matches(&self, id: &str) -> bool {
        self.id == id
    }
}

fn f32_to_i16(sample: f32) -> i16 {
    let clamped = sample.clamp(-1.0, 1.0);
    (clamped * i16::MAX as f32).round() as i16
}

fn ms_to_seconds(ms: u64) -> f32 {
    ms as f32 / 1_000.0
}

fn realtime_word_to_timestamp(word: &RealtimeTranscriptWord) -> crate::WordTimestamp {
    crate::WordTimestamp {
        word: word.word.clone(),
        start: ms_to_seconds(word.start_ms),
        end: ms_to_seconds(word.end_ms),
        confidence: word.confidence,
    }
}

/// Join settled segment texts into one transcript. CJK families emit
/// space-free text, so segments already ending or starting with a CJK
/// character are concatenated without an inserted ASCII space; otherwise a
/// single space separates them.
fn join_segment_texts<'a>(texts: impl Iterator<Item = &'a str>) -> String {
    let mut out = String::new();
    for text in texts {
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        if !out.is_empty() {
            let prev = out.chars().last();
            let next = text.chars().next();
            if !boundary_is_cjk(prev) && !boundary_is_cjk(next) {
                out.push(' ');
            }
        }
        out.push_str(text);
    }
    out
}

fn boundary_is_cjk(ch: Option<char>) -> bool {
    matches!(ch, Some(ch) if is_cjk(ch))
}

fn is_cjk(ch: char) -> bool {
    matches!(ch as u32,
        0x3400..=0x4DBF   // CJK Ext A
        | 0x4E00..=0x9FFF // CJK Unified
        | 0xF900..=0xFAFF // CJK Compatibility Ideographs
        | 0x3000..=0x303F // CJK symbols/punctuation
        | 0xFF00..=0xFFEF // Fullwidth forms
    )
}

#[cfg(test)]
#[path = "streaming_tests.rs"]
mod tests;
