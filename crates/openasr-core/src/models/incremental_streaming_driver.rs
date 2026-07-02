//! Re-decode streaming driver, shared by all six runtime families (qwen / cohere /
//! moonshine / whisper, and the CTC families parakeet / wav2vec2). Each tick it
//! re-decodes the current (undrained) segment with full context. qwen-class seq2seq
//! models expose no real token timestamps, so it maps the decoded characters to
//! proportional absolute times to find the settled prefix; a sentence that has
//! stopped being revised (its proportional end is `COMMIT_STABILITY_MARGIN_MS`
//! behind the live edge) is finalized into its own caption segment and its audio is
//! drained — Whisper-Streaming buffer trimming. So each finalized sentence is a
//! stable line and the trailing partial is the live, revisable tail.
//!
//! Nothing is accumulated across re-decodes: every partial and segment is recomputed
//! from the latest full decode, which is what keeps boundaries from drifting or
//! duplicating. The FINAL re-decodes the whole (drained) buffer with the same prompt
//! context the trailing segment's partials used, so it stays consistent with the last
//! partial instead of jumping at finalization.

use std::time::Instant;

use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutionResult, GgmlAsrPreparedAudio,
    GgmlAsrStreamingSessionRequest,
};
use crate::models::ggml_streaming_audio::{FrameTimelineError, GgmlStreamingAudioBuffer};
use crate::models::ggml_streaming_session::{
    GgmlAsrStreamingTranscriptDriver, GgmlAsrStreamingTranscriptSession,
    GgmlAsrStreamingTranscriptUpdate,
};
use crate::models::graph_runtime_config::install_request_inference_threads_override;
use crate::models::streaming_partial_cadence::PartialDecodeCadence;
use crate::{NativeAsrSession, RealtimeAudioFrame, TranscriptUpdate, Transcription};

// Whisper-Streaming best practice: a PARTIAL re-decodes the whole current segment
// with full context, not a tiny sliding window — a small window loses context and
// produces mid-utterance garbage. Sentence cuts + the force-trim keep each segment
// bounded; this cap is only a safety ceiling (it also matches the server's
// max-utterance) so a stall can't grow the decode without bound.
const DEFAULT_TOKEN_INCREMENTAL_WINDOW_MS: u64 = 30_000;
pub(crate) const STREAMING_PARTIAL_PROMPT_TAIL_WORDS: usize = 32;
const STREAMING_WARM_UP_AUDIO_MS: usize = 1_000;
const SAMPLES_PER_MS_16KHZ: usize = 16;
// Force a segment cut after this much committed speech even without sentence
// punctuation, so a long run-on (no 。/!/?) can't grow the decode unbounded.
const FORCE_SEGMENT_TRIM_MS: u64 = 12_000;
// A character whose proportional end is at least this long before the live edge is
// treated as settled (the model has stopped revising it). qwen3-asr has no real
// token timestamps and a quantized model jitters word-to-word, so this time-based
// margin is what makes the stable prefix robust; a sentence ending inside the
// stable region is finalized into its own caption segment.
const COMMIT_STABILITY_MARGIN_MS: u64 = 1_500;
// Don't cut a segment shorter than this, so micro-pauses / stray early punctuation
// can't spawn tiny one-or-two-character caption lines.
const MIN_SEGMENT_MS: u64 = 800;
// The proportional char->time mapping has no acoustic grounding, so a sentence
// cut can land slightly inside the next word. Draining this much EARLIER than
// the estimated cut guarantees no audio is lost to a late estimate; the
// re-heard overlap text is then trimmed against the committed sentence below.
const DRAIN_SAFETY_MARGIN_MS: u64 = 240;
// Longest committed-tail/new-head overlap the dedup will strip. Bounded so a
// genuinely repeated phrase in speech cannot be eaten.
const MAX_COMMITTED_OVERLAP_TRIM_CHARS: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StreamingPartialTuning {
    min_partial_interval_ms: u32,
    first_partial_audio_ms: u32,
    window_ms: u64,
    partial_prompt_tail_words: Option<usize>,
}

impl StreamingPartialTuning {
    const fn new(
        min_partial_interval_ms: u32,
        first_partial_audio_ms: u32,
        partial_prompt_tail_words: Option<usize>,
    ) -> Self {
        Self {
            min_partial_interval_ms,
            first_partial_audio_ms,
            window_ms: DEFAULT_TOKEN_INCREMENTAL_WINDOW_MS,
            partial_prompt_tail_words,
        }
    }

    pub(crate) const fn min_partial_interval_ms(&self) -> u32 {
        self.min_partial_interval_ms
    }

    pub(crate) const fn first_partial_audio_ms(&self) -> u32 {
        self.first_partial_audio_ms
    }

    pub(crate) const fn window_ms(&self) -> u64 {
        self.window_ms
    }

    pub(crate) const fn partial_prompt_tail_words(&self) -> Option<usize> {
        self.partial_prompt_tail_words
    }
}

/// Heavy seq2seq/LLM packs need enough initial context to avoid visibly wrong
/// first captions, but 1s makes the UI feel stuck. Keep this shared profile under
/// the qwen/cohere/whisper family calls so future tuning stays model-class based
/// instead of drifting into per-executor magic numbers.
// First-partial floor: a heavy seq2seq decode over <0.5s of speech routinely
// hallucinates (measured: a 400ms first window produced a confident wrong
// sentence that was later fully rewritten). 800ms trades a little first-paint
// latency for not showing garbage as the very first caption.
pub(crate) const STREAMING_PARTIAL_TUNING_HEAVY_SEQ2SEQ: StreamingPartialTuning =
    StreamingPartialTuning::new(300, 800, Some(STREAMING_PARTIAL_PROMPT_TAIL_WORDS));

pub(crate) const STREAMING_PARTIAL_TUNING_WHISPER_SEQ2SEQ: StreamingPartialTuning =
    StreamingPartialTuning::new(250, 800, Some(STREAMING_PARTIAL_PROMPT_TAIL_WORDS));

/// Fast encoder-decoder and CTC snapshot families can attempt partials as soon as
/// audio exists. Speech-gated server Poll keeps silence/pauses at zero decode.
pub(crate) const STREAMING_PARTIAL_TUNING_FAST_SNAPSHOT: StreamingPartialTuning =
    StreamingPartialTuning::new(150, 0, None);

/// Build the streaming driver for a runtime family's `start_streaming_session`.
/// Each decode rebuilds [`GgmlAsrExecutionRequest`] from `request` plus the
/// current audio and installs the request's thread-count override on the decode
/// thread.
pub(crate) fn build_streaming_driver<E, FDecode>(
    executor: E,
    executor_id: &'static str,
    adapter_id: &'static str,
    request: &GgmlAsrStreamingSessionRequest,
    tuning: StreamingPartialTuning,
    decode: FDecode,
) -> Box<dyn GgmlAsrStreamingTranscriptDriver>
where
    E: Clone + Send + 'static,
    FDecode: Fn(&E, &GgmlAsrExecutionRequest) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError>
        + Send
        + 'static,
{
    let session_suffix = &request.session_context.session_id.0;
    let utterance_id = format!("utt_{session_suffix}");
    let segment_id = format!("seg_{session_suffix}");
    let partial_results = request.session_config.partial_results;
    let partial_floor_ms = request
        .session_config
        .partial_floor_ms(tuning.min_partial_interval_ms);

    // Clone the shared request fields once; each driver closure rebuilds the
    // per-decode request from them plus the (windowed) prepared audio.
    let runtime_source_path = request.runtime_source_path.clone();
    let runtime_source_preflight = request.runtime_source_preflight.clone();
    let selected_family = request.selected_family.clone();
    let request_options = request.request_options.clone();
    let inference_threads = request_options.inference_threads;
    let backend_preference = request.backend_preference;
    let make_request = move |audio: &GgmlAsrPreparedAudio, partial_prompt: Option<&str>| {
        let mut request_options = request_options.clone();
        if let Some(prompt) =
            merge_partial_prompt(request_options.prompt.as_deref(), partial_prompt)
        {
            request_options.prompt = Some(prompt);
            request_options.prompt_token_ids = None;
        }
        GgmlAsrExecutionRequest {
            runtime_source_path: runtime_source_path.clone(),
            runtime_source_preflight: runtime_source_preflight.clone(),
            selected_family: selected_family.clone(),
            prepared_audio: audio.clone(),
            request_options,
            backend_preference,
        }
    };

    let transcribe = Box::new(
        move |audio: &GgmlAsrPreparedAudio, partial_prompt: Option<&str>| {
            let _thread_override = install_request_inference_threads_override(inference_threads);
            decode(&executor, &make_request(audio, partial_prompt))
                .map(|result| result.transcription)
        },
    );
    Box::new(
        IncrementalStreamingTranscriptDriver::new(
            executor_id,
            adapter_id,
            utterance_id,
            segment_id,
            1,
            transcribe,
        )
        .with_partial_results(partial_results)
        .with_partial_cadence(partial_floor_ms)
        .with_first_partial_audio_ms(u64::from(tuning.first_partial_audio_ms))
        .with_partial_window_ms(tuning.window_ms)
        .with_partial_prompt_tail_words(tuning.partial_prompt_tail_words()),
    )
}

/// Shared `start_streaming_session` body for the seq2seq GGML families
/// (qwen / cohere / whisper / moonshine): validate the requested adapter, build the
/// re-decode streaming driver, and wrap it in a session. Only `family_label`, the
/// ids, the tuning profile, and the decode fn differ per family.
pub(crate) fn build_seq2seq_streaming_session<E, FDecode>(
    executor: E,
    executor_id: &'static str,
    adapter_id: &'static str,
    family_label: &str,
    request: &GgmlAsrStreamingSessionRequest,
    tuning: StreamingPartialTuning,
    decode: FDecode,
) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError>
where
    E: Clone + Send + 'static,
    FDecode: Fn(&E, &GgmlAsrExecutionRequest) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError>
        + Send
        + 'static,
{
    if request.selected_family.adapter_id != adapter_id {
        return Err(GgmlAsrExecutionError::ExecutorFailed {
            executor_id,
            adapter_id: request.selected_family.adapter_id,
            reason: format!(
                "{family_label} streaming executor requires adapter '{adapter_id}', got '{}'",
                request.selected_family.adapter_id
            ),
        });
    }
    // Gate-off → snapshot driver; gate-on → incremental/windowed driver. The FINAL
    // is byte-identical either way; only partials differ.
    let driver = build_streaming_driver(executor, executor_id, adapter_id, request, tuning, decode);
    let session = GgmlAsrStreamingTranscriptSession::new(executor_id, request, driver)?;
    Ok(Box::new(session))
}

/// Decode the accumulated audio for a streaming partial/final. Family-specific
/// streaming decode keeps live-session semantics such as serve-batch bypass, while
/// the returned FINAL remains byte-identical to offline `execute()`.
type StreamingTranscriber = dyn FnMut(&GgmlAsrPreparedAudio, Option<&str>) -> Result<Transcription, GgmlAsrExecutionError>
    + Send;

/// Whether a character is sentence-terminal (Chinese or ASCII), used to cut a
/// finalized caption segment at sentence boundaries.
fn is_sentence_end_char(ch: char) -> bool {
    matches!(ch, '。' | '！' | '？' | '!' | '?' | '；' | ';' | '…')
}

/// Punctuation that may straddle a re-heard segment join: sentence-end chars
/// plus the clause commas. Used by both strip sites in
/// `trim_committed_overlap` so they cannot drift apart.
fn is_join_punctuation(ch: char) -> bool {
    is_sentence_end_char(ch) || matches!(ch, '，' | ',')
}

/// Strips from `text`'s head the longest overlap (up to `max_chars` chars)
/// with the tail of the already-committed `committed` text. The drain safety
/// margin makes the next decode re-hear a sliver of finalized audio; with the
/// committed sentence as the decode prompt the model usually does not re-emit
/// it, but when it does, this removes the duplication at the segment join.
fn trim_committed_overlap<'a>(committed: &str, text: &'a str, max_chars: usize) -> &'a str {
    if committed.is_empty() || text.is_empty() {
        return text;
    }
    // The committed sentence usually ends in punctuation the re-decode does
    // not reproduce ("…之中啊。" vs re-heard "的预料之中啊！"), so match the
    // overlap against the committed tail with terminal punctuation stripped
    // as well.
    let committed_no_punct = committed.trim_end_matches(is_join_punctuation);
    let text_chars: Vec<(usize, char)> = text.char_indices().take(max_chars).collect();
    for overlap in (1..=text_chars.len()).rev() {
        let (last_idx, last_char) = text_chars[overlap - 1];
        let head_end = last_idx + last_char.len_utf8();
        let head = &text[..head_end];
        if committed.ends_with(head) || committed_no_punct.ends_with(head) {
            let rest = text[head_end..].trim_start_matches(is_join_punctuation);
            return rest;
        }
    }
    text
}

fn merge_partial_prompt(base_prompt: Option<&str>, partial_prompt: Option<&str>) -> Option<String> {
    let partial = partial_prompt
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())?;
    let Some(base) = base_prompt
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
    else {
        return Some(partial.to_string());
    };
    Some(format!("{base}\n{partial}"))
}

pub(crate) struct IncrementalStreamingTranscriptDriver {
    executor_id: &'static str,
    adapter_id: &'static str,
    utterance_id: String,
    segment_id: String,
    utterance_id_prefix: String,
    segment_id_prefix: String,
    utterance_index: u64,
    /// Monotonic per-segment counter. Each finalized sentence segment (and each new
    /// utterance) advances it so every emitted segment gets a unique `segment_id`.
    segment_index: u64,
    buffer: GgmlStreamingAudioBuffer,
    transcribe: Box<StreamingTranscriber>,
    partial_results: bool,
    cadence: PartialDecodeCadence,
    partial_floor_ms: u64,
    first_partial_audio_ms: u64,
    last_text: Option<String>,
    next_revision: u64,
    final_emitted: bool,
    /// Safety ceiling on the per-tick PARTIAL decode span (ms). Sentence cuts and
    /// the force-trim normally keep the current segment well below this; the FINAL
    /// still decodes the whole (drained) buffer.
    window_ms: u64,
    /// Last finalized segment text, used only as the optional decode prompt for
    /// families that opt into prompt conditioning. No live transcript state is
    /// accumulated across re-decodes (that is what caused boundary drift).
    prompt_context: String,
    partial_prompt_tail_words: Option<usize>,
}

impl IncrementalStreamingTranscriptDriver {
    pub(crate) fn new(
        executor_id: &'static str,
        adapter_id: &'static str,
        utterance_id: impl Into<String>,
        segment_id: impl Into<String>,
        first_revision: u64,
        transcribe: Box<StreamingTranscriber>,
    ) -> Self {
        let mut driver = Self {
            executor_id,
            adapter_id,
            utterance_id: utterance_id.into(),
            segment_id: segment_id.into(),
            utterance_id_prefix: String::new(),
            segment_id_prefix: String::new(),
            utterance_index: 1,
            segment_index: 1,
            buffer: GgmlStreamingAudioBuffer::default(),
            transcribe,
            partial_results: true,
            cadence: PartialDecodeCadence::every_frame(),
            partial_floor_ms: 0,
            first_partial_audio_ms: 0,
            last_text: None,
            next_revision: first_revision,
            final_emitted: false,
            window_ms: 0,
            prompt_context: String::new(),
            partial_prompt_tail_words: None,
        };
        driver.utterance_id_prefix = driver.utterance_id.clone();
        driver.segment_id_prefix = driver.segment_id.clone();
        driver
    }

    pub(crate) fn with_partial_results(mut self, partial_results: bool) -> Self {
        self.partial_results = partial_results;
        self
    }

    pub(crate) fn with_partial_cadence(mut self, min_partial_interval_ms: u64) -> Self {
        self.partial_floor_ms = min_partial_interval_ms;
        self.cadence = self.new_cadence();
        self
    }

    pub(crate) fn with_first_partial_audio_ms(mut self, first_partial_audio_ms: u64) -> Self {
        self.first_partial_audio_ms = first_partial_audio_ms;
        self.cadence = self.new_cadence();
        self
    }

    /// Bound each PARTIAL decode to the trailing `window_ms` of audio. Turns the
    /// unbounded O(buffer²)-over-an-utterance partial cost into O(window) per tick;
    /// the FINAL is unaffected (full buffer).
    pub(crate) fn with_partial_window_ms(mut self, window_ms: u64) -> Self {
        self.window_ms = window_ms.max(1);
        self
    }

    pub(crate) fn with_partial_prompt_tail_words(mut self, tail_words: Option<usize>) -> Self {
        self.partial_prompt_tail_words = tail_words.filter(|words| *words > 0);
        self
    }

    fn new_cadence(&self) -> PartialDecodeCadence {
        PartialDecodeCadence::with_floor_ms(self.partial_floor_ms)
            .with_first_decode_min_audio_ms(self.first_partial_audio_ms)
    }

    fn driver_failed(&self, reason: impl Into<String>) -> GgmlAsrExecutionError {
        GgmlAsrExecutionError::ExecutorFailed {
            executor_id: self.executor_id,
            adapter_id: self.adapter_id,
            reason: reason.into(),
        }
    }

    fn map_timeline_error(&self, error: FrameTimelineError) -> GgmlAsrExecutionError {
        self.driver_failed(error.to_string())
    }

    /// Decode whatever audio remains in the buffer (the current trailing segment).
    /// Used for the FINAL. `prompt` carries the same committed-prefix context the
    /// segment's partials used, so the FINAL stays consistent with the last partial.
    fn decode_full_buffer(
        &mut self,
        prompt: Option<&str>,
    ) -> Result<Transcription, GgmlAsrExecutionError> {
        let audio = self.buffer.prepared_audio_snapshot();
        (self.transcribe)(&audio, prompt)
    }

    fn decode_warm_up_silence(&mut self) -> Result<(), GgmlAsrExecutionError> {
        let audio = GgmlAsrPreparedAudio::mono_16khz(vec![
            0.0;
            STREAMING_WARM_UP_AUDIO_MS
                * SAMPLES_PER_MS_16KHZ
        ]);
        let _ = (self.transcribe)(&audio, None)?;
        Ok(())
    }

    fn decode_partial_if_due(
        &mut self,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        if !self.partial_results || self.final_emitted || self.buffer.is_empty() {
            return Ok(Vec::new());
        }
        let audio_end_ms = self.buffer.end_ms().unwrap_or(0);
        if !self.cadence.should_decode(audio_end_ms) {
            return Ok(Vec::new());
        }
        let started = Instant::now();
        let updates = self.decode_and_commit_window(audio_end_ms)?;
        self.cadence
            .record_decode(audio_end_ms, started.elapsed().as_millis() as u64);
        Ok(updates)
    }

    fn reset_current_utterance(&mut self) {
        self.buffer.clear();
        self.last_text = None;
        self.final_emitted = false;
        self.cadence = self.new_cadence();
        self.prompt_context.clear();
        self.utterance_index = self.utterance_index.saturating_add(1);
        self.utterance_id = format!("{}_{:06}", self.utterance_id_prefix, self.utterance_index);
        self.segment_index = self.segment_index.saturating_add(1);
        self.segment_id = format!("{}_{:06}", self.segment_id_prefix, self.segment_index);
    }

    /// Emit a settled sentence as a finalized caption segment with its own
    /// `segment_id`. Does NOT set `final_emitted` — the utterance continues; only
    /// `finish_updates` ends it. The caller drains the finalized audio from the
    /// buffer after the last cut.
    fn emit_segment_final(
        &mut self,
        text: &str,
        start_ms: u64,
        end_ms: u64,
    ) -> Option<GgmlAsrStreamingTranscriptUpdate> {
        if text.is_empty() {
            return None;
        }
        let revision = self.next_revision;
        self.next_revision = self.next_revision.saturating_add(1);
        let text = text.to_string();
        let update = TranscriptUpdate::new(
            self.utterance_id.clone(),
            self.segment_id.clone(),
            revision,
            text.clone(),
            start_ms,
            end_ms,
        );
        // The finalized sentence becomes the decode prompt context for continuity.
        self.prompt_context = text;
        self.last_text = None;
        self.segment_index = self.segment_index.saturating_add(1);
        self.segment_id = format!("{}_{:06}", self.segment_id_prefix, self.segment_index);
        Some(GgmlAsrStreamingTranscriptUpdate::final_(update))
    }

    /// Last finalized segment as the optional decode prompt (for families that opt
    /// into prompt conditioning); gives cross-sentence continuity without
    /// accumulating any live state.
    fn partial_prompt_tail(&self) -> Option<String> {
        let tail_words = self.partial_prompt_tail_words?;
        let context = self.prompt_context.trim();
        if context.is_empty() {
            return None;
        }
        // For CJK (no whitespace) this is one "word" = the whole sentence, which is
        // exactly the desired prompt; for latin it's the last `tail_words` words.
        let words: Vec<&str> = context.split_whitespace().collect();
        let start = words.len().saturating_sub(tail_words);
        Some(words[start..].join(" "))
    }

    /// Re-decode the whole current (undrained) segment, map decoded characters to
    /// proportional absolute times, finalize each settled sentence (older than
    /// `COMMIT_STABILITY_MARGIN_MS`) as its own segment and drain its audio, then
    /// emit the still-unstable remainder as the live partial. Recomputed entirely
    /// from this decode each tick — nothing is carried across re-decodes.
    fn decode_and_commit_window(
        &mut self,
        audio_end_ms: u64,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        // Decode the whole current segment for full context (accurate partials).
        // Sentence cuts + the force-trim keep each segment — and therefore this
        // window — bounded, so the per-tick decode stays fast even on long
        // continuous speech; `window_ms` is only a safety ceiling.
        let window_start_ms = self
            .buffer
            .start_ms()
            .unwrap_or(0)
            .max(audio_end_ms.saturating_sub(self.window_ms))
            .min(audio_end_ms);
        let window_dur_ms = audio_end_ms.saturating_sub(window_start_ms).max(1);
        let audio = self.buffer.prepared_audio_window(window_dur_ms);
        let prompt = self.partial_prompt_tail();
        let transcription = (self.transcribe)(&audio, prompt.as_deref())?;

        // qwen3-asr is autoregressive seq2seq with NO real token timestamps, so map
        // each decoded character to a proportional absolute time across the window
        // span. To stay drift-immune, recompute EVERYTHING from this full decode each
        // tick and never accumulate text across re-decodes — accumulating a
        // proportional split that reshuffles every tick is what duplicated/dropped
        // characters at the boundaries.
        let decoded = trim_committed_overlap(
            &self.prompt_context,
            transcription.text.trim(),
            MAX_COMMITTED_OVERLAP_TRIM_CHARS,
        );
        let chars: Vec<char> = decoded.chars().collect();
        let total = chars.len();
        let mut emitted: Vec<GgmlAsrStreamingTranscriptUpdate> = Vec::new();
        if total == 0 {
            return Ok(emitted);
        }
        let char_end_ms = |index: usize| -> u64 {
            window_start_ms
                .saturating_add((index as u64 + 1).saturating_mul(window_dur_ms) / total as u64)
        };

        // Stable region: characters whose proportional end is comfortably before the
        // live edge — settled enough to finalize. (A LocalAgreement-2 layer was tried
        // here and measured WORSE: char positions shift between re-anchored decodes,
        // so char-by-char agreement misaligns and reintroduces boundary duplication.
        // Time-based stability is the right choice for this proportional setup.)
        // A sentence ending inside the stable region is cut into its own segment; a
        // long run-on with no punctuation is force-cut at the stable edge so the
        // decode cannot grow without bound.
        let stable_until_ms = audio_end_ms.saturating_sub(COMMIT_STABILITY_MARGIN_MS);
        let stable_count = (0..total)
            .take_while(|&i| char_end_ms(i) <= stable_until_ms)
            .count();
        let mut cuts: Vec<usize> = (0..stable_count)
            .filter(|&index| is_sentence_end_char(chars[index]))
            .collect();
        if cuts.is_empty() && stable_count > 0 && window_dur_ms >= FORCE_SEGMENT_TRIM_MS {
            cuts.push(stable_count - 1);
        }

        // Finalize each settled sentence as its own segment, then drain its audio so
        // the next decode only spans the current, still-incomplete sentence. Each
        // finalized segment is a stable caption line; the trailing partial is the
        // live, revisable tail.
        let mut segment_start_char = 0usize;
        let mut segment_start_ms = window_start_ms;
        for &cut in &cuts {
            let segment_end_ms = char_end_ms(cut);
            if segment_end_ms.saturating_sub(segment_start_ms) < MIN_SEGMENT_MS {
                continue;
            }
            let segment_text: String = chars[segment_start_char..=cut].iter().collect();
            if let Some(update) =
                self.emit_segment_final(segment_text.trim(), segment_start_ms, segment_end_ms)
            {
                emitted.push(update);
                segment_start_char = cut + 1;
                segment_start_ms = segment_end_ms;
            }
        }
        if segment_start_ms > window_start_ms {
            self.buffer
                .drain_before(segment_start_ms.saturating_sub(DRAIN_SAFETY_MARGIN_MS));
        }

        // The remaining text (post-cut prefix + still-unstable tail) is the live
        // partial for the current segment.
        let partial_text: String = chars[segment_start_char..].iter().collect();
        if let Some(update) = self.emit_update(partial_text.trim(), false) {
            emitted.push(update);
        }
        Ok(emitted)
    }

    fn emit_update(
        &mut self,
        raw_text: &str,
        final_update: bool,
    ) -> Option<GgmlAsrStreamingTranscriptUpdate> {
        let text = raw_text.trim().to_string();
        if text.is_empty() {
            return None;
        }
        if !final_update && self.last_text.as_deref() == Some(text.as_str()) {
            return None;
        }
        let revision = self.next_revision;
        self.next_revision = self.next_revision.saturating_add(1);
        self.last_text = Some(text.clone());
        if final_update {
            self.final_emitted = true;
        }
        let start_ms = self.buffer.start_ms().unwrap_or(0);
        let end_ms = self
            .buffer
            .end_ms()
            .unwrap_or_else(|| start_ms.saturating_add(self.buffer.duration_ms()));
        let update = TranscriptUpdate::new(
            self.utterance_id.clone(),
            self.segment_id.clone(),
            revision,
            text,
            start_ms,
            end_ms,
        );
        Some(if final_update {
            GgmlAsrStreamingTranscriptUpdate::final_(update)
        } else {
            GgmlAsrStreamingTranscriptUpdate::partial(update)
        })
    }
}

impl GgmlAsrStreamingTranscriptDriver for IncrementalStreamingTranscriptDriver {
    fn warm_up(&mut self) -> Result<(), GgmlAsrExecutionError> {
        self.decode_warm_up_silence()
    }

    fn reset_utterance(&mut self) -> Result<(), GgmlAsrExecutionError> {
        self.reset_current_utterance();
        Ok(())
    }

    fn push_audio(
        &mut self,
        frame: RealtimeAudioFrame,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        self.buffer
            .push_frame(frame)
            .map_err(|error| self.map_timeline_error(error))?;
        Ok(Vec::new())
    }

    fn poll_updates(
        &mut self,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        self.decode_partial_if_due()
    }

    fn finish_updates(
        &mut self,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        if self.buffer.is_empty() || self.final_emitted {
            return Ok(Vec::new());
        }
        // The FINAL decodes whatever audio remains in the buffer — i.e. the current
        // (still-uncut) trailing segment, since earlier sentences were already
        // finalized and their audio drained. It reuses the same prompt context the
        // segment's partials used, so the FINAL is consistent with the last partial.
        let prompt = self.partial_prompt_tail();
        let transcription = self.decode_full_buffer(prompt.as_deref())?;
        // The drain safety margin (plus the proportional cut's acoustic
        // slack) leaves a sliver of the last committed sentence's audio in
        // the buffer, so this decode can re-hear its tail. Partials strip
        // that re-heard overlap in `decode_and_commit_window`; the FINAL must
        // strip it the same way, or a finalize/split right after a sentence
        // cut duplicates the committed tail into the next segment.
        let text = trim_committed_overlap(
            &self.prompt_context,
            transcription.text.trim(),
            MAX_COMMITTED_OVERLAP_TRIM_CHARS,
        )
        .to_string();
        Ok(self.emit_update(&text, true).into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use super::*;
    use crate::{RealtimeAudioFormat, Segment};

    fn frame(seq: u64, start_ms: u64) -> RealtimeAudioFrame {
        RealtimeAudioFrame::new(
            seq,
            start_ms,
            RealtimeAudioFormat::pcm16_mono_16khz(),
            vec![1; 320],
        )
        .unwrap()
    }

    fn transcription(text: &str) -> Transcription {
        Transcription {
            text: text.to_string(),
            segments: vec![Segment {
                start: 0.0,
                end: 0.02,
                text: text.to_string(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            }],
            longform: None,
            language: None,
        }
    }

    fn text_only_transcription(text: &str) -> Transcription {
        Transcription {
            text: text.to_string(),
            segments: Vec::new(),
            longform: None,
            language: None,
        }
    }

    /// Build a driver whose streaming decode returns scripted text per tick.
    fn driver(script: VecDeque<&'static str>) -> IncrementalStreamingTranscriptDriver {
        let mut script = script;
        IncrementalStreamingTranscriptDriver::new(
            "token-incremental-test-executor",
            crate::QWEN3_ASR_GGML_ADAPTER_ID,
            "utt_tok",
            "seg_tok",
            10,
            Box::new(move |audio, _prompt| {
                assert!(!audio.samples_f32.is_empty());
                let text = script.pop_front().unwrap_or("");
                Ok(transcription(text))
            }),
        )
    }

    fn text_of(update: &GgmlAsrStreamingTranscriptUpdate) -> (&str, u64, bool) {
        match update {
            GgmlAsrStreamingTranscriptUpdate::Partial(u) => (&u.text, u.revision, false),
            GgmlAsrStreamingTranscriptUpdate::Final(u) => (&u.text, u.revision, true),
        }
    }

    #[test]
    fn warm_up_decodes_silence_without_touching_the_live_buffer() {
        let calls = Arc::new(AtomicUsize::new(0));
        let sample_lengths = Arc::new(Mutex::new(Vec::new()));
        let calls_for_decode = Arc::clone(&calls);
        let lengths_for_decode = Arc::clone(&sample_lengths);
        let mut driver = IncrementalStreamingTranscriptDriver::new(
            "token-incremental-test-executor",
            crate::QWEN3_ASR_GGML_ADAPTER_ID,
            "utt_warm",
            "seg_warm",
            1,
            Box::new(move |audio, _prompt| {
                calls_for_decode.fetch_add(1, Ordering::SeqCst);
                lengths_for_decode
                    .lock()
                    .expect("sample length mutex poisoned")
                    .push(audio.samples_f32.len());
                Ok(transcription("ignored warm text"))
            }),
        );

        driver.warm_up().unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            *sample_lengths.lock().expect("sample length mutex poisoned"),
            vec![16_000]
        );
        assert_eq!(driver.buffer.sample_count(), 0);
        assert!(driver.last_text.is_none());
    }

    /// A windowed driver whose decode returns one word per 20 ms of the audio it is
    /// given (`w0 w1 …`), with timestamps — modelling a real windowed re-decode.
    fn windowed_transcription(audio: &GgmlAsrPreparedAudio) -> Transcription {
        let dur_ms = (audio.samples_f32.len() / 16) as u64;
        let n = (dur_ms / 20) as usize;
        let start_word = (audio.samples_f32[0] * 32768.0).round() as usize - 1;
        let words: Vec<crate::WordTimestamp> = (0..n)
            .map(|i| crate::WordTimestamp {
                word: format!("w{}", start_word + i),
                start: (i as f32) * 0.02,
                end: ((i + 1) as f32) * 0.02,
                confidence: None,
            })
            .collect();
        let text = words
            .iter()
            .map(|w| w.word.clone())
            .collect::<Vec<_>>()
            .join(" ");
        Transcription {
            text: text.clone(),
            segments: vec![Segment {
                start: 0.0,
                end: dur_ms as f32 / 1000.0,
                text,
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words,
            }],
            longform: None,
            language: None,
        }
    }

    fn windowed_driver(window_ms: u64) -> IncrementalStreamingTranscriptDriver {
        IncrementalStreamingTranscriptDriver::new(
            "token-incremental-test-executor",
            crate::QWEN3_ASR_GGML_ADAPTER_ID,
            "utt_win",
            "seg_win",
            0,
            // Model a real windowed re-decode: emit one word per 20 ms of the window
            // it is handed, named by ABSOLUTE index. `vframe` encodes the absolute
            // frame index in the sample value, so a window starting at a moved anchor
            // still produces stable absolute word names (w0 w1 …) — exactly what a
            // real model would, giving a deterministic, reproducible decode to test
            // the proportional time-stability segmentation against.
            Box::new(|audio, _prompt| Ok(windowed_transcription(audio))),
        )
        .with_partial_window_ms(window_ms)
    }

    /// A 20 ms frame whose sample value encodes its absolute index (start_ms/20 + 1),
    /// so a windowed decode can recover which absolute word the window starts at.
    fn vframe(seq: u64, start_ms: u64) -> RealtimeAudioFrame {
        let value = (start_ms / 20 + 1) as i16;
        RealtimeAudioFrame::new(
            seq,
            start_ms,
            RealtimeAudioFormat::pcm16_mono_16khz(),
            vec![value; 320],
        )
        .unwrap()
    }

    #[test]
    fn windowed_partials_commit_settled_words_monotonically() {
        // With a full-buffer window the partial is the whole current decode each
        // tick, so it grows monotonically as new audio is appended.
        let mut driver = windowed_driver(30_000);
        let texts: Vec<String> = (1..=4)
            .flat_map(|i| {
                driver.push_audio(vframe(i, (i - 1) * 20)).unwrap();
                driver
                    .poll_updates()
                    .unwrap()
                    .into_iter()
                    .map(|u| text_of(&u).0.to_string())
            })
            .collect();

        // Partials grow monotonically; older words become the committed prefix.
        assert_eq!(texts, vec!["w0", "w0 w1", "w0 w1 w2", "w0 w1 w2 w3"]);

        // The FINAL re-decodes the whole buffer (byte-identical to offline).
        let final_ = driver.finish_updates().unwrap();
        assert_eq!(text_of(&final_[0]), ("w0 w1 w2 w3", 4, true));
    }

    #[test]
    fn partial_prompt_stays_empty_until_a_segment_finalizes_and_final_is_unprompted() {
        // The decode prompt now derives from the last FINALIZED segment, not a
        // live committed tail. This mock has no sentence punctuation, so no segment
        // finalizes and every decode (partials + final) is unprompted.
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let prompts_for_decode = Arc::clone(&prompts);
        let mut driver = IncrementalStreamingTranscriptDriver::new(
            "token-incremental-test-executor",
            crate::QWEN3_ASR_GGML_ADAPTER_ID,
            "utt_prompt",
            "seg_prompt",
            0,
            Box::new(move |audio, prompt| {
                prompts_for_decode
                    .lock()
                    .expect("prompt mutex poisoned")
                    .push(prompt.map(str::to_string));
                Ok(windowed_transcription(audio))
            }),
        )
        .with_partial_window_ms(40)
        .with_partial_prompt_tail_words(Some(STREAMING_PARTIAL_PROMPT_TAIL_WORDS));

        for i in 1..=3 {
            driver.push_audio(vframe(i, (i - 1) * 20)).unwrap();
            let _ = driver.poll_updates().unwrap();
        }
        let final_ = driver.finish_updates().unwrap();
        assert_eq!(text_of(&final_[0]), ("w0 w1 w2", 3, true));

        assert_eq!(
            *prompts.lock().expect("prompt mutex poisoned"),
            vec![None, None, None, None]
        );
    }

    #[test]
    fn partial_prompt_is_disabled_unless_family_opts_in() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let prompts_for_decode = Arc::clone(&prompts);
        let mut driver = IncrementalStreamingTranscriptDriver::new(
            "token-incremental-test-executor",
            crate::QWEN3_ASR_GGML_ADAPTER_ID,
            "utt_no_prompt",
            "seg_no_prompt",
            0,
            Box::new(move |audio, prompt| {
                prompts_for_decode
                    .lock()
                    .expect("prompt mutex poisoned")
                    .push(prompt.map(str::to_string));
                Ok(windowed_transcription(audio))
            }),
        )
        .with_partial_window_ms(40);

        for i in 1..=3 {
            driver.push_audio(vframe(i, (i - 1) * 20)).unwrap();
            let _ = driver.poll_updates().unwrap();
        }

        assert_eq!(
            *prompts.lock().expect("prompt mutex poisoned"),
            vec![None, None, None]
        );
    }

    #[test]
    fn text_only_partial_transcriptions_are_segmented_for_streaming_updates() {
        let mut script = VecDeque::from(["hello", "hello world"]);
        let mut driver = IncrementalStreamingTranscriptDriver::new(
            "token-incremental-test-executor",
            crate::QWEN3_ASR_GGML_ADAPTER_ID,
            "utt_text_only",
            "seg_text_only",
            0,
            Box::new(move |_audio, _prompt| {
                let text = script.pop_front().unwrap_or("");
                Ok(text_only_transcription(text))
            }),
        )
        .with_partial_window_ms(2_000);

        driver.push_audio(frame(1, 0)).unwrap();
        let first = driver.poll_updates().unwrap();
        assert_eq!(text_of(&first[0]), ("hello", 0, false));

        driver.push_audio(frame(2, 20)).unwrap();
        let second = driver.poll_updates().unwrap();
        assert_eq!(text_of(&second[0]), ("hello world", 1, false));
    }

    #[test]
    fn cadence_throttles_windowed_partial_decodes_but_always_emits_final() {
        let mut driver = windowed_driver(30_000).with_partial_cadence(60);

        driver.push_audio(vframe(1, 0)).unwrap();
        let cold = driver.poll_updates().unwrap(); // end 20 -> cold start
        assert_eq!(text_of(&cold[0]), ("w0", 0, false));
        driver.push_audio(vframe(2, 20)).unwrap();
        assert!(driver.poll_updates().unwrap().is_empty()); // end 40
        driver.push_audio(vframe(3, 40)).unwrap();
        assert!(driver.poll_updates().unwrap().is_empty()); // end 60
        driver.push_audio(vframe(4, 60)).unwrap();
        let second = driver.poll_updates().unwrap(); // end 80 -> decode
        assert_eq!(text_of(&second[0]), ("w0 w1 w2 w3", 1, false));

        let final_ = driver.finish_updates().unwrap();
        assert_eq!(text_of(&final_[0]), ("w0 w1 w2 w3", 2, true));
    }

    #[test]
    fn reset_utterance_reopens_cold_partial_with_new_ids() {
        let mut driver = windowed_driver(30_000).with_partial_cadence(400);

        driver.push_audio(vframe(1, 0)).unwrap();
        let first_partial = driver.poll_updates().unwrap();
        let first_final = driver.finish_updates().unwrap();
        assert_eq!(text_of(&first_partial[0]), ("w0", 0, false));
        assert_eq!(text_of(&first_final[0]), ("w0", 1, true));
        let first_id = match &first_final[0] {
            GgmlAsrStreamingTranscriptUpdate::Final(update) => {
                (update.utterance_id.clone(), update.segment_id.clone())
            }
            _ => unreachable!("expected final update"),
        };

        driver.reset_utterance().unwrap();
        driver.push_audio(vframe(1, 0)).unwrap();
        let second_partial = driver.poll_updates().unwrap();

        assert_eq!(text_of(&second_partial[0]), ("w0", 2, false));
        let second_id = match &second_partial[0] {
            GgmlAsrStreamingTranscriptUpdate::Partial(update) => {
                (update.utterance_id.clone(), update.segment_id.clone())
            }
            _ => unreachable!("expected partial update"),
        };
        assert_ne!(second_id, first_id);
        assert_eq!(second_id.0.0, "utt_win_000002");
        assert_eq!(second_id.1.0, "seg_win_000002");
    }

    #[test]
    fn disabled_partials_emit_only_the_final() {
        // Partials off: push_audio never decodes, so the single scripted entry is
        // consumed by the FINAL decode in finish_updates.
        let mut driver = driver(VecDeque::from(["final text"])).with_partial_results(false);
        assert!(driver.push_audio(frame(1, 0)).unwrap().is_empty());
        let final_ = driver.finish_updates().unwrap();
        assert_eq!(text_of(&final_[0]), ("final text", 10, true));
    }

    #[test]
    fn finish_after_sentence_cut_trims_reheard_committed_tail() {
        // A sentence cut finalizes "第一句结束。" and drains its audio up to
        // the safety margin; the terminal FINAL (VAD finalize or a
        // speaker-change/max-utterance split) then re-decodes the remaining
        // buffer, which re-hears the committed tail. The FINAL must strip the
        // re-heard overlap exactly like partial decodes do, or the committed
        // tail is duplicated into the next caption segment.
        let texts = Arc::new(Mutex::new(VecDeque::from([
            // First partial decode: full window, sentence settles and is cut.
            "第一句结束。后续继续说话中",
            // Terminal decode of the drained buffer: re-hears the committed tail.
            "结束。后续继续说话中",
        ])));
        let texts_for_decode = Arc::clone(&texts);
        let mut driver = IncrementalStreamingTranscriptDriver::new(
            "token-incremental-test-executor",
            crate::QWEN3_ASR_GGML_ADAPTER_ID,
            "utt_trim_final",
            "seg_trim_final",
            0,
            Box::new(move |_audio, _prompt| {
                let text = texts_for_decode
                    .lock()
                    .expect("script mutex poisoned")
                    .pop_front()
                    .unwrap_or("");
                Ok(text_only_transcription(text))
            }),
        )
        .with_partial_window_ms(30_000);

        // 6 s of audio so the sentence end lands inside the stable region.
        for i in 0..300u64 {
            driver.push_audio(frame(i, i * 20)).unwrap();
        }
        let updates = driver.poll_updates().unwrap();
        let finals: Vec<_> = updates
            .iter()
            .map(text_of)
            .filter(|(_, _, is_final)| *is_final)
            .collect();
        assert_eq!(finals.len(), 1, "sentence cut should finalize one segment");
        assert_eq!(finals[0].0, "第一句结束。");

        let terminal = driver.finish_updates().unwrap();
        assert_eq!(terminal.len(), 1);
        let (text, _, is_final) = text_of(&terminal[0]);
        assert!(is_final);
        assert_eq!(
            text, "后续继续说话中",
            "terminal FINAL must not re-emit the committed sentence tail"
        );
    }

    #[test]
    fn trim_committed_overlap_strips_reheard_segment_tail() {
        // The drain margin makes the next decode re-hear the end of the
        // committed sentence; its decoded head duplicates that tail.
        assert_eq!(
            trim_committed_overlap(
                "可能在张总眼里，这是一种真情实感吧。",
                "实感吧。但在我眼里看",
                12
            ),
            "但在我眼里看"
        );
        // No overlap: text unchanged.
        assert_eq!(
            trim_committed_overlap("前一句。", "完全无关的下一句", 12),
            "完全无关的下一句"
        );
        // The cap bounds how much can be stripped: a long head match only
        // loses up to `max_chars`, so a genuinely repeated phrase cannot be
        // eaten wholesale.
        let committed = "重复重复重复重复重复重复重复";
        let text = "重复重复重复重复重复重复重复后续";
        assert_eq!(
            trim_committed_overlap(committed, text, 4),
            "重复重复重复重复重复后续"
        );
        // Empty committed context is a no-op.
        assert_eq!(trim_committed_overlap("", "你好", 12), "你好");
        // Terminal punctuation on the committed tail must not defeat the
        // match (measured case: '…预料之中啊。' then re-heard '的预料之中啊！').
        assert_eq!(
            trim_committed_overlap("确实是在我的预料之中啊。", "的预料之中啊！后续内容", 12),
            "后续内容"
        );
    }
}
