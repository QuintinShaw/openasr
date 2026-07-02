use std::sync::{Arc, atomic::AtomicBool};

use crate::{
    Transcription,
    realtime::{RealtimeAudioFrame, RealtimeEventEnvelope},
};

use super::{
    NativeAsrCapabilities, NativeAsrError, NativeAsrHardwareTarget, NativeAsrModelPackRef,
    NativeAsrOfflineRequest, NativeAsrRequestOptions, NativeAsrRuntimeReadiness,
    NativeAsrSessionContext, NativeAsrStreamingSessionConfig, NativeAsrTensorLayoutRef,
};

pub trait NativeAsrModelAdapter {
    fn adapter_id(&self) -> &'static str;
    fn model_family(&self) -> &'static str;
    fn capabilities(&self) -> NativeAsrCapabilities;
    fn tensor_layout(&self) -> Option<NativeAsrTensorLayoutRef> {
        None
    }
    fn supports_model_pack(&self, model_pack: &NativeAsrModelPackRef) -> bool;
    fn start_streaming_session(
        &self,
        model_pack: &NativeAsrModelPackRef,
        target: NativeAsrHardwareTarget,
        context: NativeAsrSessionContext,
        options: NativeAsrRequestOptions,
        session_config: NativeAsrStreamingSessionConfig,
    ) -> Result<Box<dyn NativeAsrSession>, NativeAsrError> {
        session_config.validate()?;
        let _ = (model_pack, target, context, options);
        Err(NativeAsrError::BackendDoesNotSupportTrueStreaming {
            backend: self.adapter_id().to_string(),
        })
    }
}

pub trait NativeAsrExecutor {
    fn executor_id(&self) -> &'static str;
    fn capabilities(&self) -> NativeAsrCapabilities;
    fn runtime_readiness(
        &self,
        adapter: &dyn NativeAsrModelAdapter,
        model_pack: &NativeAsrModelPackRef,
        target: NativeAsrHardwareTarget,
    ) -> NativeAsrRuntimeReadiness;
    fn transcribe(
        &self,
        adapter: &dyn NativeAsrModelAdapter,
        model_pack: &NativeAsrModelPackRef,
        target: NativeAsrHardwareTarget,
        request: NativeAsrOfflineRequest,
    ) -> Result<Transcription, NativeAsrError>;
    fn start_session(
        &self,
        adapter: &dyn NativeAsrModelAdapter,
        model_pack: &NativeAsrModelPackRef,
        target: NativeAsrHardwareTarget,
        context: NativeAsrSessionContext,
        options: NativeAsrRequestOptions,
    ) -> Result<Box<dyn NativeAsrSession>, NativeAsrError>;
    fn start_streaming_session(
        &self,
        adapter: &dyn NativeAsrModelAdapter,
        model_pack: &NativeAsrModelPackRef,
        target: NativeAsrHardwareTarget,
        context: NativeAsrSessionContext,
        options: NativeAsrRequestOptions,
        session_config: NativeAsrStreamingSessionConfig,
    ) -> Result<Box<dyn NativeAsrSession>, NativeAsrError> {
        session_config.validate()?;
        let _ = (adapter, model_pack, target, context, options);
        Err(NativeAsrError::BackendDoesNotSupportTrueStreaming {
            backend: self.executor_id().to_string(),
        })
    }
}

pub trait NativeAsrSession: Send {
    fn session_id(&self) -> &str;
    fn set_cancellation_token(&mut self, _cancelled: Arc<AtomicBool>) {}
    fn push_audio(
        &mut self,
        frame: RealtimeAudioFrame,
    ) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError>;
    fn poll_events(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError>;
    fn flush(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        self.poll_events()
    }
    fn warm_up(&mut self) -> Result<(), NativeAsrError> {
        Ok(())
    }
    fn finalize_utterance(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        self.finish()
    }
    /// Segment split at a forced (max-utterance-duration) boundary. Sessions
    /// that can preserve decode state across the split should override this so
    /// an arbitrary mid-speech cut cannot degrade recognition; the default
    /// falls back to a full finalize.
    fn split_utterance(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        self.finalize_utterance()
    }
    fn finish(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError>;
    fn close(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        self.cancel()
    }
    fn cancel(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError>;
}
