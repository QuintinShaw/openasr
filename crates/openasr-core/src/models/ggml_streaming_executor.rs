#![cfg_attr(not(test), allow(dead_code))]

use crate::NativeAsrSession;
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest,
};
use crate::models::ggml_streaming_session::{
    GgmlAsrStreamingTranscriptDriver, GgmlAsrStreamingTranscriptSession,
};

pub(crate) trait GgmlAsrStreamingTranscriptDriverFactory: Send + Sync {
    fn executor_id(&self) -> &'static str;

    fn start_driver(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn GgmlAsrStreamingTranscriptDriver>, GgmlAsrExecutionError>;
}

pub(crate) struct GgmlAsrStreamingTranscriptExecutor<F>
where
    F: GgmlAsrStreamingTranscriptDriverFactory,
{
    factory: F,
}

impl<F> GgmlAsrStreamingTranscriptExecutor<F>
where
    F: GgmlAsrStreamingTranscriptDriverFactory,
{
    pub(crate) fn new(factory: F) -> Self {
        Self { factory }
    }
}

impl<F> GgmlAsrStreamingExecutor for GgmlAsrStreamingTranscriptExecutor<F>
where
    F: GgmlAsrStreamingTranscriptDriverFactory,
{
    fn executor_id(&self) -> &'static str {
        self.factory.executor_id()
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        let driver = self.factory.start_driver(request)?;
        let session = GgmlAsrStreamingTranscriptSession::new(self.executor_id(), request, driver)?;
        Ok(Box::new(session))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ggml_streaming_session::{
        GgmlAsrStreamingTranscriptUpdate, GgmlAsrStreamingTranscriptUpdate::Partial,
    };
    use crate::{
        GgmlAsrBackendPreference, GgmlAsrExecutionOptions, NativeAsrStreamingSessionConfig,
        RealtimeAudioFormat, RealtimeAudioFrame, RealtimeEvent, RealtimeTranscriptEvent,
        TranscriptUpdate,
    };

    struct ScriptFactory;

    struct ScriptDriver;

    impl GgmlAsrStreamingTranscriptDriverFactory for ScriptFactory {
        fn executor_id(&self) -> &'static str {
            "script-streaming-transcript-executor"
        }

        fn start_driver(
            &self,
            _request: &GgmlAsrStreamingSessionRequest,
        ) -> Result<Box<dyn GgmlAsrStreamingTranscriptDriver>, GgmlAsrExecutionError> {
            Ok(Box::new(ScriptDriver))
        }
    }

    impl GgmlAsrStreamingTranscriptDriver for ScriptDriver {
        fn push_audio(
            &mut self,
            frame: RealtimeAudioFrame,
        ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
            Ok(vec![Partial(TranscriptUpdate::new(
                "utt_executor",
                "seg_executor",
                1,
                "hello",
                frame.start_ms,
                frame.end_ms(),
            ))])
        }
    }

    fn request() -> GgmlAsrStreamingSessionRequest {
        let verified_pack = crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
            crate::models::runtime_preflight::leaked_tiny_runtime_source_preflight(),
            crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID,
        );
        GgmlAsrStreamingSessionRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack,
            selected_family: crate::arch::builtin_adapter_descriptor(
                crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            ),
            request_options: GgmlAsrExecutionOptions::default(),
            configured_diarize: false,
            backend_preference: GgmlAsrBackendPreference::Auto,
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                (GgmlAsrBackendPreference::Auto).request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            final_text_processor: None,
            session_context: crate::NativeAsrSessionContext::new("rt_executor_wrapper"),
            session_config: NativeAsrStreamingSessionConfig::new()
                .with_partial_results(true)
                .into(),
        }
    }

    fn test_frame() -> RealtimeAudioFrame {
        let format = RealtimeAudioFormat::pcm16_mono_16khz();
        let sample_count = format.sample_count_for_duration_ms(20).unwrap();
        RealtimeAudioFrame::new(1, 0, format, vec![0; sample_count]).unwrap()
    }

    #[test]
    fn transcript_executor_wraps_factory_driver_as_native_session() {
        let executor = GgmlAsrStreamingTranscriptExecutor::new(ScriptFactory);
        let mut session = executor.start_streaming_session(&request()).unwrap();

        assert_eq!(session.session_id(), "rt_executor_wrapper");
        assert_eq!(
            session
                .poll_events()
                .unwrap()
                .iter()
                .map(|event| event.event_type)
                .collect::<Vec<_>>(),
            [
                "session.created",
                "session.configured",
                "audio.input.started"
            ]
        );

        let events = session.push_audio(test_frame()).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "transcript.partial");
        assert!(matches!(
            &events[0].event,
            RealtimeEvent::Transcript(RealtimeTranscriptEvent::Partial(partial))
                if partial.text == "hello" && partial.revision == 1
        ));
    }

    #[test]
    fn transcript_executor_preserves_driver_factory_errors() {
        struct FailingFactory;

        impl GgmlAsrStreamingTranscriptDriverFactory for FailingFactory {
            fn executor_id(&self) -> &'static str {
                "failing-streaming-transcript-executor"
            }

            fn start_driver(
                &self,
                request: &GgmlAsrStreamingSessionRequest,
            ) -> Result<Box<dyn GgmlAsrStreamingTranscriptDriver>, GgmlAsrExecutionError>
            {
                Err(GgmlAsrExecutionError::ExecutorFailed {
                    executor_id: self.executor_id(),
                    adapter_id: request.selected_family.adapter_id,
                    reason: "driver unavailable".to_string(),
                })
            }
        }

        let executor = GgmlAsrStreamingTranscriptExecutor::new(FailingFactory);
        let error = match executor.start_streaming_session(&request()) {
            Ok(_) => panic!("driver factory error must fail session start"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            GgmlAsrExecutionError::ExecutorFailed {
                executor_id: "failing-streaming-transcript-executor",
                adapter_id: crate::QWEN3_ASR_GGML_ADAPTER_ID,
                ..
            }
        ));
    }
}
