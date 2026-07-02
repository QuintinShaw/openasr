use super::super::*;
use super::fixtures::test_model_pack;
use super::session::TestOnlyNativeStreamingSession;

pub(in super::super) struct TestOnlyNativeStreamingAdapter;
pub(in super::super) struct TestOnlyNativeFinalOnlyStreamingAdapter;
pub(in super::super) struct TestOnlyOfflineAdapter;
pub(in super::super) struct TestOnlyNativeStreamingExecutor;
pub(in super::super) struct TestOnlyDefaultStreamingExecutor;

fn not_implemented_offline(message: &'static str) -> Result<crate::Transcription, NativeAsrError> {
    Err(NativeAsrError::SessionFailed {
        message: message.to_string(),
    })
}

fn not_implemented_session(
    message: &'static str,
) -> Result<Box<dyn NativeAsrSession>, NativeAsrError> {
    Err(NativeAsrError::SessionFailed {
        message: message.to_string(),
    })
}

pub(in super::super) fn test_only_streaming_session(
    config: NativeAsrStreamingSessionConfig,
) -> Box<dyn NativeAsrSession> {
    test_only_streaming_session_result(config, true).unwrap()
}

pub(in super::super) fn test_only_streaming_session_result(
    config: NativeAsrStreamingSessionConfig,
    request_partial_results: bool,
) -> Result<Box<dyn NativeAsrSession>, NativeAsrError> {
    let adapter = TestOnlyNativeStreamingAdapter;
    let executor = TestOnlyNativeStreamingExecutor;
    let context = NativeAsrSessionContext::new("rt_native_test")
        .with_trace_id(Some("trace_native_test".to_string()))
        .with_request_id(Some("req_native_test".to_string()));
    let options = NativeAsrRequestOptions::new().with_partial_results(request_partial_results);
    executor.start_streaming_session(
        &adapter,
        &test_model_pack(),
        NativeAsrHardwareTarget::Cpu,
        context,
        options,
        config,
    )
}

impl NativeAsrModelAdapter for TestOnlyNativeStreamingAdapter {
    fn adapter_id(&self) -> &'static str {
        super::TEST_ONLY_STREAMING_FIXTURE_ID
    }

    fn model_family(&self) -> &'static str {
        "test-only-native-streaming"
    }

    fn capabilities(&self) -> NativeAsrCapabilities {
        NativeAsrCapabilities::native_true_streaming()
            .with_partial_results(true)
            .with_timestamps(true)
    }

    fn supports_model_pack(&self, model_pack: &NativeAsrModelPackRef) -> bool {
        model_pack.family == self.model_family()
    }
}

impl NativeAsrModelAdapter for TestOnlyNativeFinalOnlyStreamingAdapter {
    fn adapter_id(&self) -> &'static str {
        "test-only-native-final-only-streaming-adapter"
    }

    fn model_family(&self) -> &'static str {
        "test-only-native-streaming"
    }

    fn capabilities(&self) -> NativeAsrCapabilities {
        NativeAsrCapabilities::native_true_streaming().with_timestamps(true)
    }

    fn supports_model_pack(&self, model_pack: &NativeAsrModelPackRef) -> bool {
        model_pack.family == self.model_family()
    }
}

impl NativeAsrModelAdapter for TestOnlyOfflineAdapter {
    fn adapter_id(&self) -> &'static str {
        "test-only-offline-adapter"
    }

    fn model_family(&self) -> &'static str {
        "test-only-native-streaming"
    }

    fn capabilities(&self) -> NativeAsrCapabilities {
        NativeAsrCapabilities::native_offline()
    }

    fn supports_model_pack(&self, model_pack: &NativeAsrModelPackRef) -> bool {
        model_pack.family == self.model_family()
    }
}

impl NativeAsrExecutor for TestOnlyNativeStreamingExecutor {
    fn executor_id(&self) -> &'static str {
        "test-only-native-streaming-executor"
    }

    fn capabilities(&self) -> NativeAsrCapabilities {
        NativeAsrCapabilities::native_true_streaming()
            .with_partial_results(true)
            .with_timestamps(true)
    }

    fn runtime_readiness(
        &self,
        adapter: &dyn NativeAsrModelAdapter,
        model_pack: &NativeAsrModelPackRef,
        target: NativeAsrHardwareTarget,
    ) -> NativeAsrRuntimeReadiness {
        if target != NativeAsrHardwareTarget::Cpu {
            return NativeAsrRuntimeReadiness::UnsupportedHardwareTarget { target };
        }
        if !adapter.supports_model_pack(model_pack) {
            return NativeAsrRuntimeReadiness::UnsupportedModelPack {
                reason: "test fixture adapter does not support model pack".to_string(),
            };
        }
        if !adapter.capabilities().supports_true_streaming {
            return NativeAsrRuntimeReadiness::BackendDoesNotSupportTrueStreaming {
                backend: adapter.adapter_id().to_string(),
            };
        }
        NativeAsrRuntimeReadiness::Ready
    }

    fn transcribe(
        &self,
        _adapter: &dyn NativeAsrModelAdapter,
        _model_pack: &NativeAsrModelPackRef,
        _target: NativeAsrHardwareTarget,
        _request: NativeAsrOfflineRequest,
    ) -> Result<crate::Transcription, NativeAsrError> {
        not_implemented_offline("test-only native streaming fixture does not implement offline ASR")
    }

    fn start_session(
        &self,
        _adapter: &dyn NativeAsrModelAdapter,
        _model_pack: &NativeAsrModelPackRef,
        _target: NativeAsrHardwareTarget,
        _context: NativeAsrSessionContext,
        _options: NativeAsrRequestOptions,
    ) -> Result<Box<dyn NativeAsrSession>, NativeAsrError> {
        not_implemented_session("test-only fixture requires explicit streaming session config")
    }

    fn start_streaming_session(
        &self,
        adapter: &dyn NativeAsrModelAdapter,
        model_pack: &NativeAsrModelPackRef,
        target: NativeAsrHardwareTarget,
        context: NativeAsrSessionContext,
        options: NativeAsrRequestOptions,
        session_config: NativeAsrStreamingSessionConfig,
    ) -> Result<Box<dyn NativeAsrSession>, NativeAsrError> {
        let mut session_config = session_config;
        session_config.validate()?;
        if let Ok(error) =
            NativeAsrError::try_from(self.runtime_readiness(adapter, model_pack, target))
        {
            return Err(error);
        }
        session_config.partial_results = session_config.partial_results
            && adapter.capabilities().supports_partials
            && self.capabilities().supports_partials;
        session_config.word_timestamps = session_config.word_timestamps
            && options.word_timestamps
            && adapter.capabilities().supports_timestamps
            && self.capabilities().supports_timestamps;
        Ok(Box::new(TestOnlyNativeStreamingSession::new(
            context,
            model_pack.id.clone(),
            options,
            session_config,
        )?))
    }
}

impl NativeAsrExecutor for TestOnlyDefaultStreamingExecutor {
    fn executor_id(&self) -> &'static str {
        "test-only-default-streaming-executor"
    }

    fn capabilities(&self) -> NativeAsrCapabilities {
        NativeAsrCapabilities::native_true_streaming()
            .with_partial_results(true)
            .with_timestamps(true)
    }

    fn runtime_readiness(
        &self,
        _adapter: &dyn NativeAsrModelAdapter,
        _model_pack: &NativeAsrModelPackRef,
        _target: NativeAsrHardwareTarget,
    ) -> NativeAsrRuntimeReadiness {
        NativeAsrRuntimeReadiness::Ready
    }

    fn transcribe(
        &self,
        _adapter: &dyn NativeAsrModelAdapter,
        _model_pack: &NativeAsrModelPackRef,
        _target: NativeAsrHardwareTarget,
        _request: NativeAsrOfflineRequest,
    ) -> Result<crate::Transcription, NativeAsrError> {
        not_implemented_offline("test-only default executor does not implement offline ASR")
    }

    fn start_session(
        &self,
        _adapter: &dyn NativeAsrModelAdapter,
        _model_pack: &NativeAsrModelPackRef,
        _target: NativeAsrHardwareTarget,
        _context: NativeAsrSessionContext,
        _options: NativeAsrRequestOptions,
    ) -> Result<Box<dyn NativeAsrSession>, NativeAsrError> {
        not_implemented_session("default streaming test should not delegate here")
    }
}
