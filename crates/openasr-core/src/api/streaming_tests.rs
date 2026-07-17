use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::*;
use crate::realtime::{
    RealtimeEventEnvelope, RealtimeEventId, RealtimeSessionId, RealtimeTranscriptEvent,
    RealtimeTranscriptFinal, RealtimeTranscriptPartial, TranscriptSegmentId, TranscriptUtteranceId,
};
use crate::{NativeAsrError, RealtimeAudioFrame};

/// A deterministic in-memory streaming session for exercising the glue in
/// [`StreamingSession`] without any model weights. Each `push_audio` reveals
/// one more word of a fixed target sentence as a growing partial; a real VAD
/// stop (`finalize_utterance`) or `finish` settles the revealed text into a
/// FINAL and starts the next utterance.
struct FakeStreamingSession {
    words: Vec<&'static str>,
    revealed: usize,
    utterance: u64,
    finalize_calls: usize,
    finished: bool,
}

impl FakeStreamingSession {
    fn new(sentence: &'static str) -> Self {
        Self {
            words: sentence.split_whitespace().collect(),
            revealed: 0,
            utterance: 1,
            finalize_calls: 0,
            finished: false,
        }
    }

    fn current_text(&self) -> String {
        self.words[..self.revealed.min(self.words.len())].join(" ")
    }

    fn ids(&self) -> (TranscriptUtteranceId, TranscriptSegmentId) {
        (
            TranscriptUtteranceId(format!("utt_{:06}", self.utterance)),
            TranscriptSegmentId(format!("seg_{:06}", self.utterance)),
        )
    }

    fn partial_envelope(&self) -> RealtimeEventEnvelope {
        let (utterance_id, segment_id) = self.ids();
        transcript_envelope(RealtimeTranscriptEvent::Partial(
            RealtimeTranscriptPartial {
                utterance_id,
                segment_id,
                revision: self.revealed as u64,
                text: self.current_text(),
                start_ms: 0,
                end_ms: (self.revealed as u64) * 100,
                is_final: false,
                words: Vec::new(),
                language: None,
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
            },
        ))
    }

    fn final_envelope(&self) -> RealtimeEventEnvelope {
        let (utterance_id, segment_id) = self.ids();
        transcript_envelope(RealtimeTranscriptEvent::Final(RealtimeTranscriptFinal {
            utterance_id,
            segment_id,
            revision: self.revealed as u64 + 1,
            text: self.current_text(),
            start_ms: 0,
            end_ms: (self.revealed as u64) * 100,
            is_final: true,
            words: Vec::new(),
            language: Some("en".to_string()),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
        }))
    }
}

impl NativeAsrSession for FakeStreamingSession {
    fn session_id(&self) -> &str {
        "fake"
    }

    fn push_audio(
        &mut self,
        _frame: RealtimeAudioFrame,
    ) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        if self.revealed < self.words.len() {
            self.revealed += 1;
            Ok(vec![self.partial_envelope()])
        } else {
            Ok(Vec::new())
        }
    }

    fn poll_events(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        Ok(Vec::new())
    }

    fn finalize_utterance(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        self.finalize_calls += 1;
        if self.revealed == 0 {
            return Ok(Vec::new());
        }
        let event = self.final_envelope();
        // Next utterance starts fresh.
        self.utterance += 1;
        self.revealed = 0;
        Ok(vec![event])
    }

    fn finish(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        self.finished = true;
        if self.revealed == 0 {
            return Ok(Vec::new());
        }
        Ok(vec![self.final_envelope()])
    }

    fn cancel(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        Ok(Vec::new())
    }
}

fn transcript_envelope(event: RealtimeTranscriptEvent) -> RealtimeEventEnvelope {
    RealtimeEventEnvelope {
        event_type: "transcript",
        session_id: RealtimeSessionId("fake".to_string()),
        event_id: RealtimeEventId("evt".to_string()),
        seq: 0,
        created_at: "1970-01-01T00:00:00.000Z".to_string(),
        trace_id: None,
        request_id: None,
        event: RealtimeEvent::Transcript(event),
    }
}

fn wrap(session: FakeStreamingSession, cfg: &StreamingConfig) -> StreamingSession {
    StreamingSession::from_native_session(Box::new(session), cfg).unwrap()
}

/// One 20 ms frame worth of `f32` samples at `amplitude`.
fn frame_samples(amplitude: f32) -> Vec<f32> {
    vec![amplitude; FRAME_SAMPLES]
}

#[test]
fn partials_grow_then_final_no_vad() {
    let cfg = StreamingConfig {
        vad: None,
        ..Default::default()
    };
    let mut session = wrap(
        FakeStreamingSession::new("hello world this is a test"),
        &cfg,
    );

    // Feed 6 frames (one word revealed per frame in the fake).
    let mut partial_texts = Vec::new();
    for _ in 0..6 {
        let events = session.feed(&frame_samples(0.3)).unwrap();
        for event in events {
            if event.kind == StreamingEventKind::Partial {
                partial_texts.push(event.text);
            }
        }
    }

    // Partials strictly grow in length: an incremental live-caption stream.
    assert!(partial_texts.len() >= 2, "expected multiple partials");
    for window in partial_texts.windows(2) {
        assert!(
            window[1].len() >= window[0].len(),
            "partial text must not shrink: {:?} -> {:?}",
            window[0],
            window[1]
        );
    }
    assert_eq!(partial_texts.last().unwrap(), "hello world this is a test");

    let transcription = session.finish().unwrap();
    assert_eq!(transcription.text, "hello world this is a test");
    assert_eq!(transcription.segments.len(), 1);
    assert_eq!(transcription.language.as_deref(), Some("en"));
}

#[test]
fn vad_pause_emits_committed_segment() {
    let cfg = StreamingConfig::default(); // VAD on by default
    let mut session = wrap(FakeStreamingSession::new("hello world"), &cfg);

    let mut committed = Vec::new();
    // ~400 ms of speech: enough loud frames to cross speech_start_ms (200 ms).
    for _ in 0..20 {
        let events = session.feed(&frame_samples(0.5)).unwrap();
        committed.extend(
            events
                .into_iter()
                .filter(|event| event.kind == StreamingEventKind::Committed),
        );
    }
    // ~800 ms of silence: crosses speech_stop_ms (600 ms) -> utterance closes.
    for _ in 0..40 {
        let events = session.feed(&frame_samples(0.0)).unwrap();
        committed.extend(
            events
                .into_iter()
                .filter(|event| event.kind == StreamingEventKind::Committed),
        );
    }

    assert!(
        !committed.is_empty(),
        "a VAD speech pause should commit a segment mid-stream"
    );
    assert_eq!(committed[0].text, "hello world");

    let transcription = session.finish().unwrap();
    assert!(transcription.text.contains("hello world"));
}

#[test]
fn tail_audio_is_flushed_on_finish() {
    let cfg = StreamingConfig {
        vad: None,
        ..Default::default()
    };
    let mut session = wrap(FakeStreamingSession::new("tail words here"), &cfg);
    // A sub-frame chunk that never fills a 20 ms frame on its own.
    let events = session.feed(&[0.1_f32; 100]).unwrap();
    assert!(events.is_empty(), "a partial frame yields no events yet");
    // finish() pads and flushes the tail, then finalizes.
    let transcription = session.finish().unwrap();
    assert!(!transcription.text.is_empty());
}

#[test]
fn cjk_segments_join_without_inserted_space() {
    let joined = super::join_segment_texts(["你好", "世界"].into_iter());
    assert_eq!(joined, "你好世界");
    let latin = super::join_segment_texts(["hello", "world"].into_iter());
    assert_eq!(latin, "hello world");
}

#[test]
fn cancellation_token_setter_is_accepted() {
    // The fake ignores it, but the trait default must not panic when the
    // session is boxed through StreamingSession.
    let cfg = StreamingConfig {
        vad: None,
        ..Default::default()
    };
    let mut session = wrap(FakeStreamingSession::new("one two"), &cfg);
    let cancelled = Arc::new(AtomicBool::new(false));
    session.session.set_cancellation_token(cancelled.clone());
    assert!(!cancelled.load(Ordering::Relaxed));
}

/// End-to-end parity check against the batch path with a real moonshine pack.
///
/// Ignored by default (needs model weights + a compiled ggml backend, which
/// the weight-free default test suite must not require). Run manually:
///
/// ```text
/// OPENASR_TEST_STREAMING_PACK=/path/to/moonshine-tiny-q8_0.oasr \
///   cargo test -p openasr-core streaming_matches_batch_transcribe -- --ignored --nocapture
/// ```
#[test]
#[ignore = "requires a real .oasr pack via OPENASR_TEST_STREAMING_PACK"]
fn streaming_matches_batch_transcribe() {
    use crate::{
        NativeAsrExecutor, NativeAsrHardwareTarget, NativeAsrModelAdapter, NativeAsrModelPackRef,
        NativeAsrOfflineRequest, NativeAsrRequestOptions, NativeBackendExecutor,
        load_native_wav_16khz_mono_f32_v0, native_runtime_model_adapter_for_path,
    };
    use std::path::PathBuf;

    let pack = PathBuf::from(
        std::env::var("OPENASR_TEST_STREAMING_PACK")
            .expect("set OPENASR_TEST_STREAMING_PACK to a local .oasr pack"),
    );
    let wav = PathBuf::from(
        std::env::var("OPENASR_TEST_STREAMING_WAV")
            .unwrap_or_else(|_| "fixtures/jfk.wav".to_string()),
    );

    let samples = load_native_wav_16khz_mono_f32_v0(&wav, "streaming-test", "streaming-input")
        .expect("load wav as 16k mono f32");

    // Streaming: feed in 100 ms chunks with VAD off (single utterance) so the
    // final transcript is directly comparable to a whole-file batch decode.
    let cfg = StreamingConfig {
        partial_results: true,
        vad: None,
        hardware_target: NativeAsrHardwareTarget::Cpu,
        ..Default::default()
    };
    let mut session = StreamingSession::new(&pack, cfg).expect("start streaming session");
    let mut partial_count = 0usize;
    let mut last_partial_len = 0usize;
    let mut growing = 0usize;
    for chunk in samples.chunks(1_600) {
        for event in session.feed(chunk).expect("feed chunk") {
            if event.kind == StreamingEventKind::Partial {
                partial_count += 1;
                if event.text.len() >= last_partial_len {
                    growing += 1;
                }
                last_partial_len = event.text.len();
            }
        }
    }
    let streamed = session.finish().expect("finish streaming");

    // Batch reference over the whole file, same executor/adapter/pack. The
    // offline `transcribe` path validates the ref's model id against the pack's
    // own runtime id, so resolve the pack's real id rather than a placeholder.
    let adapter = native_runtime_model_adapter_for_path(&pack).expect("adapter");
    let identity = crate::resolve_local_native_runtime_model_identity(&pack, None)
        .expect("resolve pack model identity");
    let model_pack =
        NativeAsrModelPackRef::new(identity.model_id, adapter.model_family(), pack.clone());
    let batch = NativeAsrExecutor::transcribe(
        &NativeBackendExecutor,
        &adapter,
        &model_pack,
        NativeAsrHardwareTarget::Cpu,
        NativeAsrOfflineRequest::new(wav).with_options(NativeAsrRequestOptions::new()),
    )
    .expect("batch transcribe");

    let wer = crate::wer(&batch.text, &streamed.text);
    eprintln!("partials={partial_count} growing={growing}");
    eprintln!("batch    = {:?}", batch.text);
    eprintln!("streamed = {:?}", streamed.text);
    eprintln!("wer(streamed vs batch) = {wer:.3}");

    assert!(partial_count > 0, "expected incremental partials");
    assert!(
        wer <= 0.25,
        "streamed final should closely match batch (wer={wer:.3})\nbatch={:?}\nstreamed={:?}",
        batch.text,
        streamed.text
    );
}
