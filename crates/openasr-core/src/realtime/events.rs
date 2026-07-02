use serde::Serialize;
use std::time::{SystemTime, UNIX_EPOCH};

use super::{audio::RealtimeAudioFormat, backend::RealtimeBackendCapabilities};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct RealtimeSessionId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct RealtimeEventId(pub String);

pub type RealtimeEventSeq = u64;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct TranscriptUtteranceId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct TranscriptSegmentId(pub String);

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RealtimeEventEnvelope {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub session_id: RealtimeSessionId,
    pub event_id: RealtimeEventId,
    pub seq: RealtimeEventSeq,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(flatten)]
    pub event: RealtimeEvent,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum RealtimeEvent {
    Lifecycle(RealtimeLifecycleEvent),
    AudioInput(RealtimeAudioInputEvent),
    Vad(RealtimeVadEvent),
    Transcript(RealtimeTranscriptEvent),
    Translation(RealtimeTranslationEvent),
    Error(RealtimeErrorEvent),
}

impl RealtimeEvent {
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::Lifecycle(event) => event.event_type(),
            Self::AudioInput(event) => event.event_type(),
            Self::Vad(event) => event.event_type(),
            Self::Transcript(event) => event.event_type(),
            Self::Translation(event) => event.event_type(),
            Self::Error(_) => "error",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum RealtimeLifecycleEvent {
    SessionCreated(SessionCreatedEvent),
    SessionCapabilities(SessionCapabilitiesEvent),
    SessionConfigured(SessionConfiguredEvent),
    SessionClosed(SessionClosedEvent),
}

impl RealtimeLifecycleEvent {
    fn event_type(&self) -> &'static str {
        match self {
            Self::SessionCreated(_) => "session.created",
            Self::SessionCapabilities(_) => "session.capabilities",
            Self::SessionConfigured(_) => "session.configured",
            Self::SessionClosed(_) => "session.closed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SessionCreatedEvent {
    pub audio_format: RealtimeAudioFormat,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SessionCapabilitiesEvent {
    pub capabilities: RealtimeBackendCapabilities,
    pub audio_format: RealtimeAudioFormat,
    pub frame_duration_ms: u32,
    pub frame_byte_len: usize,
    pub max_message_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SessionConfiguredEvent {
    pub model: String,
    pub partial_results: bool,
    pub word_timestamps: bool,
    pub diarize: bool,
    pub translation: SessionTranslationSummary,
    pub vad: SessionVadSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SessionTranslationSummary {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_lang: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
}

impl SessionTranslationSummary {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            target_lang: None,
            model: None,
            mode: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SessionVadSummary {
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SessionClosedEvent {
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum RealtimeAudioInputEvent {
    Started(AudioInputStartedEvent),
    Stopped(AudioInputStoppedEvent),
}

impl RealtimeAudioInputEvent {
    fn event_type(&self) -> &'static str {
        match self {
            Self::Started(_) => "audio.input.started",
            Self::Stopped(_) => "audio.input.stopped",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AudioInputStartedEvent {}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AudioInputStoppedEvent {
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum RealtimeVadEvent {
    SpeechStarted(VadSpeechStartedEvent),
    SpeechStopped(VadSpeechStoppedEvent),
}

impl RealtimeVadEvent {
    fn event_type(&self) -> &'static str {
        match self {
            Self::SpeechStarted(_) => "vad.speech_started",
            Self::SpeechStopped(_) => "vad.speech_stopped",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VadSpeechStartedEvent {
    pub utterance_id: TranscriptUtteranceId,
    pub start_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VadSpeechStoppedEvent {
    pub utterance_id: TranscriptUtteranceId,
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum RealtimeTranscriptEvent {
    Partial(RealtimeTranscriptPartial),
    Final(RealtimeTranscriptFinal),
    Revision(RealtimeTranscriptRevision),
}

impl RealtimeTranscriptEvent {
    fn event_type(&self) -> &'static str {
        match self {
            Self::Partial(_) => "transcript.partial",
            Self::Final(_) => "transcript.final",
            Self::Revision(_) => "transcript.revision",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RealtimeTranscriptWord {
    pub word: String,
    pub start_ms: u64,
    pub end_ms: u64,
    /// Mean decoder softmax probability of the word's tokens (`0..=1`), when
    /// the family captures per-token scores; omitted from the wire otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RealtimeTranscriptPartial {
    pub utterance_id: TranscriptUtteranceId,
    pub segment_id: TranscriptSegmentId,
    pub revision: u64,
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub is_final: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<RealtimeTranscriptWord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Speaker label (`SPEAKER_NN`/`SPEAKER_ME`) when diarization is on; omitted
    /// otherwise, so the wire contract is identical with diarization off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    /// Stable anonymous session label (`SPEAKER_NN`) when `speaker` was replaced
    /// by an enrolled voice-match display name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_label: Option<String>,
    /// Voice-match profile id (`vp_*`) when the utterance matched an enrolled
    /// profile. This is not an authentication claim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_profile_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RealtimeTranscriptFinal {
    pub utterance_id: TranscriptUtteranceId,
    pub segment_id: TranscriptSegmentId,
    pub revision: u64,
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub is_final: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<RealtimeTranscriptWord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Speaker label (`SPEAKER_NN`/`SPEAKER_ME`) when diarization is on; omitted
    /// otherwise, so the wire contract is identical with diarization off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    /// Stable anonymous session label (`SPEAKER_NN`) when `speaker` was replaced
    /// by an enrolled voice-match display name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_label: Option<String>,
    /// Voice-match profile id (`vp_*`) when the utterance matched an enrolled
    /// profile. This is not an authentication claim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_profile_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RealtimeTranscriptRevision {
    pub utterance_id: TranscriptUtteranceId,
    pub segment_id: TranscriptSegmentId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revises_event_id: Option<RealtimeEventId>,
    pub revision: u64,
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub is_final: bool,
    pub reason: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<RealtimeTranscriptWord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Speaker label (`SPEAKER_NN`/`SPEAKER_ME`) when diarization is on; omitted
    /// otherwise, so the wire contract is identical with diarization off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    /// Stable anonymous session label (`SPEAKER_NN`) when `speaker` was replaced
    /// by an enrolled voice-match display name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_label: Option<String>,
    /// Voice-match profile id (`vp_*`) when the utterance matched an enrolled
    /// profile. This is not an authentication claim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_profile_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum RealtimeTranslationEvent {
    Status(RealtimeTranslationStatus),
    Partial(RealtimeTranslationPartial),
    Final(RealtimeTranslationFinal),
    Tombstone(RealtimeTranslationTombstone),
}

impl RealtimeTranslationEvent {
    fn event_type(&self) -> &'static str {
        match self {
            Self::Status(_) => "translation.status",
            Self::Partial(_) => "translation.partial",
            Self::Final(_) => "translation.final",
            Self::Tombstone(_) => "translation.tombstone",
        }
    }
}

/// Lifecycle status of the per-session translation lane. The translation
/// runtime cold-loads off the session-start critical path, so `session.
/// configured` reporting `translation.enabled=true` means "accepted and
/// loading"; this event reports when the runtime is actually ready. A load
/// failure is reported through the regular `error` event instead, keeping the
/// configured-truthfulness contract: enabled -> either `translation.status`
/// ready or a session-fatal error.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RealtimeTranslationStatus {
    pub state: String,
    pub model: String,
    pub target_lang: String,
}

impl RealtimeTranslationStatus {
    pub const STATE_READY: &'static str = "ready";
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RealtimeTranslationPartial {
    pub clause_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replaces_clause_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revises_clause_id: Option<String>,
    pub source_segment_id: String,
    pub source_version: u64,
    pub translation_version: u64,
    pub target_lang: String,
    pub text: String,
    pub source_text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub stability: f32,
    pub is_final: bool,
    pub model: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RealtimeTranslationFinal {
    pub clause_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replaces_clause_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revises_clause_id: Option<String>,
    pub source_segment_id: String,
    pub source_version: u64,
    pub translation_version: u64,
    pub target_lang: String,
    pub text: String,
    pub source_text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub is_final: bool,
    pub model: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RealtimeTranslationTombstone {
    pub clause_id: String,
    pub source_segment_id: String,
    pub source_version: u64,
    pub target_lang: String,
    pub reason: String,
    pub is_final: bool,
    pub model: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RealtimeErrorEvent {
    pub code: RealtimeErrorCode,
    pub message: String,
    pub recoverable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RealtimeErrorCode {
    #[serde(rename = "startup_config_error")]
    StartupConfigError,
    #[serde(rename = "backend_not_ready")]
    BackendNotReady,
    #[serde(rename = "unsupported_backend")]
    UnsupportedBackend,
    #[serde(rename = "unsupported_audio_format")]
    UnsupportedAudioFormat,
    #[serde(rename = "audio_buffer_overflow")]
    AudioBufferOverflow,
    #[serde(rename = "backpressure_timeout")]
    BackpressureTimeout,
    #[serde(rename = "client_disconnected")]
    ClientDisconnected,
    #[serde(rename = "backend_timeout")]
    BackendTimeout,
    #[serde(rename = "backend_crashed")]
    BackendCrashed,
    #[serde(rename = "vad_timeout")]
    VadTimeout,
    #[serde(rename = "no_speech_timeout")]
    NoSpeechTimeout,
    #[serde(rename = "cancelled")]
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct RealtimeEventSequencer {
    session_id: RealtimeSessionId,
    next_seq: RealtimeEventSeq,
    next_event_index: u64,
    trace_id: Option<String>,
    request_id: Option<String>,
}

impl RealtimeEventSequencer {
    pub fn new(session_id: RealtimeSessionId) -> Self {
        Self {
            session_id,
            next_seq: 1,
            next_event_index: 1,
            trace_id: None,
            request_id: None,
        }
    }

    pub fn with_trace_id(mut self, trace_id: Option<String>) -> Self {
        self.trace_id = trace_id;
        self
    }

    pub fn with_request_id(mut self, request_id: Option<String>) -> Self {
        self.request_id = request_id;
        self
    }

    pub fn next(
        &mut self,
        event: RealtimeEvent,
        created_at: impl Into<String>,
    ) -> RealtimeEventEnvelope {
        let event_type = event.event_type();
        let envelope = RealtimeEventEnvelope {
            event_type,
            session_id: self.session_id.clone(),
            event_id: RealtimeEventId(format!("evt_{:06}", self.next_event_index)),
            seq: self.next_seq,
            created_at: created_at.into(),
            trace_id: self.trace_id.clone(),
            request_id: self.request_id.clone(),
            event,
        };
        self.next_seq += 1;
        self.next_event_index += 1;
        envelope
    }
}

pub(crate) fn realtime_timestamp_now() -> String {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => format_unix_millis(duration.as_secs(), duration.subsec_millis()),
        Err(_) => "1970-01-01T00:00:00.000Z".to_string(),
    }
}

fn format_unix_millis(seconds: u64, millis: u32) -> String {
    let days = (seconds / 86_400) as i64;
    let seconds_of_day = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    (year, month as u32, day as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_sequence_is_monotonic_per_session() {
        let mut sequencer = RealtimeEventSequencer::new(RealtimeSessionId("rt_test".to_string()));
        let first = sequencer.next(
            RealtimeEvent::AudioInput(RealtimeAudioInputEvent::Started(AudioInputStartedEvent {})),
            "2026-05-09T00:00:00Z",
        );
        let second = sequencer.next(
            RealtimeEvent::AudioInput(RealtimeAudioInputEvent::Stopped(AudioInputStoppedEvent {
                reason: "user_stopped".to_string(),
            })),
            "2026-05-09T00:00:01Z",
        );

        assert_eq!(first.seq, 1);
        assert_eq!(first.event_id, RealtimeEventId("evt_000001".to_string()));
        assert_eq!(first.created_at, "2026-05-09T00:00:00Z");
        assert_eq!(second.seq, 2);
        assert_eq!(second.event_id, RealtimeEventId("evt_000002".to_string()));
        assert_eq!(second.created_at, "2026-05-09T00:00:01Z");
    }

    #[test]
    fn serializes_key_event_field_names() {
        let mut sequencer = RealtimeEventSequencer::new(RealtimeSessionId("rt_test".to_string()));
        let envelope = sequencer.next(
            RealtimeEvent::Transcript(RealtimeTranscriptEvent::Partial(
                RealtimeTranscriptPartial {
                    utterance_id: TranscriptUtteranceId("utt_1".to_string()),
                    segment_id: TranscriptSegmentId("seg_1".to_string()),
                    revision: 2,
                    text: "hello".to_string(),
                    start_ms: 0,
                    end_ms: 200,
                    is_final: false,
                    words: Vec::new(),
                    language: None,
                    speaker: None,
                    speaker_label: None,
                    speaker_profile_id: None,
                },
            )),
            "2026-05-09T00:00:00Z",
        );

        let value = serde_json::to_value(&envelope).unwrap();
        assert_eq!(value["type"], "transcript.partial");
        assert_eq!(value["session_id"], "rt_test");
        assert_eq!(value["event_id"], "evt_000001");
        assert_eq!(value["seq"], 1);
        assert_eq!(value["utterance_id"], "utt_1");
        assert_eq!(value["segment_id"], "seg_1");
        assert_eq!(value["revision"], 2);
        assert_eq!(value["is_final"], false);
        assert!(value.get("speaker").is_none());
        assert!(value.get("words").is_none());
        assert!(value.get("confidence").is_none());
        assert!(value.get("stability").is_none());
    }

    #[test]
    fn serializes_error_code_as_contract_string() {
        let event = RealtimeErrorEvent {
            code: RealtimeErrorCode::AudioBufferOverflow,
            message: "The realtime audio buffer reached its configured capacity.".to_string(),
            recoverable: false,
        };
        let value = serde_json::to_value(event).unwrap();
        assert_eq!(value["code"], "audio_buffer_overflow");
    }

    #[test]
    fn serializes_transcript_revision_with_reference_and_reason() {
        let mut sequencer = RealtimeEventSequencer::new(RealtimeSessionId("rt_test".to_string()));
        let envelope = sequencer.next(
            RealtimeEvent::Transcript(RealtimeTranscriptEvent::Revision(
                RealtimeTranscriptRevision {
                    utterance_id: TranscriptUtteranceId("utt_1".to_string()),
                    segment_id: TranscriptSegmentId("seg_1".to_string()),
                    revises_event_id: Some(RealtimeEventId("evt_000100".to_string())),
                    revision: 2,
                    text: "hello world".to_string(),
                    start_ms: 0,
                    end_ms: 200,
                    is_final: true,
                    reason: "post_final_correction".to_string(),
                    words: Vec::new(),
                    language: Some("en".to_string()),
                    speaker: None,
                    speaker_label: None,
                    speaker_profile_id: None,
                },
            )),
            "2026-05-09T00:00:02Z",
        );

        let value = serde_json::to_value(&envelope).unwrap();
        assert_eq!(value["type"], "transcript.revision");
        assert_eq!(value["revises_event_id"], "evt_000100");
        assert_eq!(value["reason"], "post_final_correction");
        assert_eq!(value["language"], "en");
        assert!(value.get("confidence").is_none());
        assert!(value.get("stability").is_none());
    }

    #[test]
    fn serializes_translation_partial_contract_fields() {
        let mut sequencer = RealtimeEventSequencer::new(RealtimeSessionId("rt_test".to_string()));
        let envelope = sequencer.next(
            RealtimeEvent::Translation(RealtimeTranslationEvent::Partial(
                RealtimeTranslationPartial {
                    clause_id: "c-12".to_string(),
                    replaces_clause_id: Some("c-10".to_string()),
                    revises_clause_id: Some("c-10".to_string()),
                    source_segment_id: "s-34".to_string(),
                    source_version: 18,
                    translation_version: 7,
                    target_lang: "en".to_string(),
                    text: "We need to keep the streaming path fast".to_string(),
                    source_text: "我们需要保持流式路径很快".to_string(),
                    start_ms: 12_340,
                    end_ms: 14_720,
                    stability: 0.82,
                    is_final: false,
                    model: "hymt2-1.8b-q4_k_m".to_string(),
                },
            )),
            "2026-05-09T00:00:02Z",
        );

        let value = serde_json::to_value(&envelope).unwrap();
        assert_eq!(value["type"], "translation.partial");
        assert_eq!(value["clause_id"], "c-12");
        assert_eq!(value["replaces_clause_id"], "c-10");
        assert_eq!(value["revises_clause_id"], "c-10");
        assert_eq!(value["source_segment_id"], "s-34");
        assert_eq!(value["source_version"], 18);
        assert_eq!(value["translation_version"], 7);
        assert_eq!(value["target_lang"], "en");
        assert!((value["stability"].as_f64().unwrap() - 0.82).abs() < 0.000_001);
        assert_eq!(value["is_final"], false);
        assert_eq!(value["model"], "hymt2-1.8b-q4_k_m");
    }

    #[test]
    fn serializes_translation_tombstone_contract_fields() {
        let mut sequencer = RealtimeEventSequencer::new(RealtimeSessionId("rt_test".to_string()));
        let envelope = sequencer.next(
            RealtimeEvent::Translation(RealtimeTranslationEvent::Tombstone(
                RealtimeTranslationTombstone {
                    clause_id: "c-12".to_string(),
                    source_segment_id: "s-34".to_string(),
                    source_version: 19,
                    target_lang: "en".to_string(),
                    reason: "source_clause_retired".to_string(),
                    is_final: true,
                    model: "hymt2-1.8b-q4_k_m".to_string(),
                },
            )),
            "2026-05-09T00:00:03Z",
        );

        let value = serde_json::to_value(&envelope).unwrap();
        assert_eq!(value["type"], "translation.tombstone");
        assert_eq!(value["clause_id"], "c-12");
        assert_eq!(value["source_segment_id"], "s-34");
        assert_eq!(value["source_version"], 19);
        assert_eq!(value["target_lang"], "en");
        assert_eq!(value["reason"], "source_clause_retired");
        assert_eq!(value["is_final"], true);
        assert_eq!(value["model"], "hymt2-1.8b-q4_k_m");
    }

    #[test]
    fn serializes_transcript_words_when_present() {
        let event = RealtimeTranscriptFinal {
            utterance_id: TranscriptUtteranceId("utt_1".to_string()),
            segment_id: TranscriptSegmentId("seg_1".to_string()),
            revision: 1,
            text: "hello world".to_string(),
            start_ms: 100,
            end_ms: 500,
            is_final: true,
            words: vec![
                RealtimeTranscriptWord {
                    word: "hello".to_string(),
                    start_ms: 100,
                    end_ms: 260,
                    confidence: Some(0.875),
                },
                RealtimeTranscriptWord {
                    word: "world".to_string(),
                    start_ms: 260,
                    end_ms: 500,
                    confidence: None,
                },
            ],
            language: None,
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
        };

        let value = serde_json::to_value(event).unwrap();

        assert_eq!(value["words"][0]["word"], "hello");
        assert_eq!(value["words"][0]["start_ms"], 100);
        assert_eq!(value["words"][1]["end_ms"], 500);
        // Word confidence is rendered when captured and omitted when not, so
        // the wire shape stays identical for families without scores.
        assert_eq!(value["words"][0]["confidence"], 0.875);
        assert!(value["words"][1].get("confidence").is_none());
    }

    #[test]
    fn formats_unix_millis_as_utc_iso8601() {
        assert_eq!(format_unix_millis(0, 0), "1970-01-01T00:00:00.000Z");
        assert_eq!(
            format_unix_millis(1_700_000_000, 42),
            "2023-11-14T22:13:20.042Z"
        );
    }
}
