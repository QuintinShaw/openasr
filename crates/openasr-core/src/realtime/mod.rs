pub mod audio;
pub mod backend;
pub mod buffer;
pub mod capture;
pub mod events;
pub mod history;
pub mod session;
pub mod transcript;
pub mod vad;
#[cfg(test)]
mod wire_bindings_test;

pub use audio::{
    DEFAULT_REALTIME_CHANNELS, DEFAULT_REALTIME_SAMPLE_RATE_HZ, RealtimeAudioEncoding,
    RealtimeAudioFormat, RealtimeAudioFrame, RealtimeFrameError,
};
pub use backend::{
    RealtimeBackendCapabilities, RealtimeBackendMode, RealtimeTranslationCapability,
    realtime_diarization_capability,
};
pub use buffer::{
    BufferedUtterance, RealtimeBuffer, RealtimeBufferConfig, RealtimeBufferError,
    RealtimeUtteranceEndReason,
};
pub use capture::{
    CaptureBackpressureQueue, CaptureEngine, CaptureEngineError, CaptureInputFormat,
    CapturePushOutcome, CaptureSample,
};
pub use events::{
    RealtimeAudioInputEvent, RealtimeErrorCode, RealtimeErrorEvent, RealtimeEvent,
    RealtimeEventEnvelope, RealtimeEventId, RealtimeEventSeq, RealtimeEventSequencer,
    RealtimeLifecycleEvent, RealtimeSessionId, RealtimeTranscriptEvent, RealtimeTranscriptFinal,
    RealtimeTranscriptPartial, RealtimeTranscriptRevision, RealtimeTranscriptWord,
    RealtimeTranslationEvent, RealtimeTranslationFinal, RealtimeTranslationPartial,
    RealtimeTranslationStatus, RealtimeTranslationTombstone, RealtimeVadEvent,
    SessionCapabilitiesEvent, SessionTranslationSummary, TranscriptSegmentId,
    TranscriptUtteranceId, VadSpeechStartedEvent, VadSpeechStoppedEvent,
};
pub use history::{
    RealtimeExportFormat, RealtimeHistoryApplyResult, RealtimeHistoryEntry,
    RealtimeHistoryExportError, RealtimeHistoryRevision, RealtimePostProcessOutput,
    RealtimePostProcessor, RealtimeTranscriptHistory,
};
pub use session::{
    RealtimeLifecycleAction, RealtimeSessionConfig, RealtimeSessionController,
    RealtimeSessionError, RealtimeSessionState,
};
pub use transcript::{
    TRANSCRIPT_REVISION_REASON_POST_FINAL_CORRECTION, TRANSCRIPT_REVISION_REASONS,
    TranscriptLifecycle, TranscriptLifecycleResult, TranscriptRevisionPolicy, TranscriptUpdate,
};
pub use vad::{
    SpeechBoundaryEvent, VadConfig, VadConfigError, VadDecision, VadFrameDecision, VadMode,
    VadState, VadStateMachine,
};
