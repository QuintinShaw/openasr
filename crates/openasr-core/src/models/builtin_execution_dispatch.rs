use std::sync::Arc;

use thiserror::Error;

use crate::GgmlAsrExecutionDispatch;
use crate::StreamingPartialGranularity;
use crate::arch::{OpenAsrArchitectureRegistry, OpenAsrArchitectureRegistryError};

use super::cohere::CohereTranscribeGgmlExecutor;
use super::executor_component_registry::{
    BuiltinExecutorComponentRegistryError, materialize_builtin_executors_by_model_architecture,
};
use super::ggml_composed_executor::ComposedGgmlAsrExecutor;
use super::ggml_family_adapter::GgmlExecutionCapability;
use super::moonshine::MoonshineGgmlExecutor;
use super::parakeet_ctc::executor::ParakeetCtcGgmlExecutor;
use super::qwen::Qwen3AsrGgmlExecutor;
use super::sensevoice::executor::SenseVoiceGgmlExecutor;
use super::wav2vec2_ctc::executor::Wav2Vec2CtcGgmlExecutor;
use super::whisper::WhisperGgmlExecutor;
use super::xasr_zipformer::executor::XasrZipformerGgmlExecutor;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum BuiltinGgmlExecutionDispatchError {
    #[error("builtin executor materialization failed: {source}")]
    ExecutorMaterialization {
        #[source]
        source: BuiltinExecutorComponentRegistryError,
    },
    #[error(
        "builtin execution dispatch is missing a materialized executor for architecture '{model_architecture}'"
    )]
    MissingMaterializedExecutor { model_architecture: &'static str },
    #[error("builtin architecture registry failed validation: {error:?}")]
    ArchitectureRegistryInvalid {
        error: OpenAsrArchitectureRegistryError,
    },
}

pub(crate) fn build_builtin_ggml_execution_dispatch()
-> Result<GgmlAsrExecutionDispatch, BuiltinGgmlExecutionDispatchError> {
    let registry = OpenAsrArchitectureRegistry::with_builtins();
    registry.validate_references().map_err(|error| {
        BuiltinGgmlExecutionDispatchError::ArchitectureRegistryInvalid { error }
    })?;

    let mut dispatch = GgmlAsrExecutionDispatch::default();
    let executors_by_model_architecture = materialize_builtin_executors_by_model_architecture()
        .map_err(|source| BuiltinGgmlExecutionDispatchError::ExecutorMaterialization { source })?;
    let mut native_graph_lowering_executors = Vec::new();

    for descriptor in registry.descriptors() {
        let Some(executor) = executors_by_model_architecture.get(descriptor.model_architecture)
        else {
            return Err(
                BuiltinGgmlExecutionDispatchError::MissingMaterializedExecutor {
                    model_architecture: descriptor.model_architecture,
                },
            );
        };
        match descriptor.execution_capability {
            GgmlExecutionCapability::DedicatedRuntimeExecutorV1 => {
                dispatch =
                    dispatch.with_executor_for_adapter(descriptor.adapter_id, Arc::clone(executor));
            }
            GgmlExecutionCapability::NativeGraphLoweringV1 => {
                native_graph_lowering_executors
                    .push((descriptor.model_architecture, Arc::clone(executor)));
            }
        }
    }

    if !native_graph_lowering_executors.is_empty() {
        dispatch = dispatch.with_executor_for_capability(
            GgmlExecutionCapability::NativeGraphLoweringV1,
            Arc::new(
                ComposedGgmlAsrExecutor::default()
                    .with_architecture_executors(native_graph_lowering_executors),
            ),
        );
    }

    Ok(dispatch)
}

pub(crate) fn build_builtin_ggml_streaming_execution_dispatch()
-> Result<GgmlAsrExecutionDispatch, BuiltinGgmlExecutionDispatchError> {
    let registry = OpenAsrArchitectureRegistry::with_builtins();
    registry.validate_references().map_err(|error| {
        BuiltinGgmlExecutionDispatchError::ArchitectureRegistryInvalid { error }
    })?;

    // Streaming executors must be registered explicitly by a family-level
    // implementation. The offline executor registry is intentionally not reused
    // here, so metadata alone cannot turn an offline decoder into a claimed
    // realtime/partial runtime.
    //
    // Each adapter also declares its partial-result granularity here: the
    // registration site is the only place that knows whether a family's
    // streaming session is the frame-sync append-only driver (never revises
    // emitted text) or a buffered/windowed re-decode driver (may revise).
    // Only xasr-zipformer runs the frame-sync driver today; every other
    // family re-decodes a growing or windowed buffer.
    Ok(GgmlAsrExecutionDispatch::default()
        .with_streaming_executor_for_adapter(
            crate::QWEN3_ASR_GGML_ADAPTER_ID,
            Arc::new(Qwen3AsrGgmlExecutor::default()),
        )
        .with_streaming_partial_granularity_for_adapter(
            crate::QWEN3_ASR_GGML_ADAPTER_ID,
            StreamingPartialGranularity::Buffered,
        )
        .with_streaming_executor_for_adapter(
            crate::WHISPER_GGML_ADAPTER_ID,
            Arc::new(WhisperGgmlExecutor::default()),
        )
        .with_streaming_partial_granularity_for_adapter(
            crate::WHISPER_GGML_ADAPTER_ID,
            StreamingPartialGranularity::Buffered,
        )
        .with_streaming_executor_for_adapter(
            crate::COHERE_TRANSCRIBE_GGML_ADAPTER_ID,
            Arc::new(CohereTranscribeGgmlExecutor::default()),
        )
        .with_streaming_partial_granularity_for_adapter(
            crate::COHERE_TRANSCRIBE_GGML_ADAPTER_ID,
            StreamingPartialGranularity::Buffered,
        )
        .with_streaming_executor_for_adapter(
            crate::MOONSHINE_GGML_ADAPTER_ID,
            Arc::new(MoonshineGgmlExecutor::default()),
        )
        .with_streaming_partial_granularity_for_adapter(
            crate::MOONSHINE_GGML_ADAPTER_ID,
            StreamingPartialGranularity::Buffered,
        )
        .with_streaming_executor_for_adapter(
            crate::PARAKEET_CTC_GGML_ADAPTER_ID,
            Arc::new(ParakeetCtcGgmlExecutor),
        )
        .with_streaming_partial_granularity_for_adapter(
            crate::PARAKEET_CTC_GGML_ADAPTER_ID,
            StreamingPartialGranularity::Buffered,
        )
        .with_streaming_executor_for_adapter(
            crate::arch::SENSEVOICE_GGML_ADAPTER_ID,
            Arc::new(SenseVoiceGgmlExecutor),
        )
        .with_streaming_partial_granularity_for_adapter(
            crate::arch::SENSEVOICE_GGML_ADAPTER_ID,
            StreamingPartialGranularity::Buffered,
        )
        .with_streaming_executor_for_adapter(
            crate::WAV2VEC2_CTC_GGML_ADAPTER_ID,
            Arc::new(Wav2Vec2CtcGgmlExecutor),
        )
        .with_streaming_partial_granularity_for_adapter(
            crate::WAV2VEC2_CTC_GGML_ADAPTER_ID,
            StreamingPartialGranularity::Buffered,
        )
        .with_streaming_executor_for_adapter(
            crate::XASR_ZIPFORMER_GGML_ADAPTER_ID,
            Arc::new(XasrZipformerGgmlExecutor),
        )
        .with_streaming_partial_granularity_for_adapter(
            crate::XASR_ZIPFORMER_GGML_ADAPTER_ID,
            StreamingPartialGranularity::FrameSync,
        ))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::{
        GgmlAsrBackendPreference, GgmlAsrExecutionError, GgmlAsrExecutionRequest,
        GgmlAsrPreparedAudio, GgmlAsrStreamingSessionRequest, NativeAsrSessionContext,
        NativeAsrStreamingSessionConfig, parakeet_ctc_runtime_descriptor_v1,
        qwen3_asr_runtime_descriptor_v1, wav2vec2_ctc_runtime_descriptor_v1,
        whisper_runtime_descriptor_v1, xasr_zipformer_runtime_descriptor_v1,
    };

    fn missing_runtime_request() -> GgmlAsrExecutionRequest {
        GgmlAsrExecutionRequest {
            runtime_source_path: PathBuf::from("/tmp/openasr-missing-runtime.gguf"),
            runtime_source_preflight: None,
            selected_family: qwen3_asr_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(vec![0.0, 0.1]),
            request_options: crate::GgmlAsrExecutionOptions::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
        }
    }

    fn streaming_request() -> GgmlAsrStreamingSessionRequest {
        GgmlAsrStreamingSessionRequest {
            runtime_source_path: PathBuf::from("/tmp/openasr-missing-runtime.gguf"),
            runtime_source_preflight: None,
            selected_family: qwen3_asr_runtime_descriptor_v1(),
            request_options: crate::GgmlAsrExecutionOptions::default(),
            configured_diarize: false,
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
            session_context: NativeAsrSessionContext::new("rt_builtin_streaming"),
            session_config: NativeAsrStreamingSessionConfig::new()
                .with_partial_results(true)
                .into(),
        }
    }

    #[test]
    fn builtins_cover_all_dedicated_runtime_architectures() {
        let executors =
            materialize_builtin_executors_by_model_architecture().expect("executor map");
        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            if descriptor.execution_capability
                != GgmlExecutionCapability::DedicatedRuntimeExecutorV1
            {
                continue;
            }
            assert!(
                executors.contains_key(descriptor.model_architecture),
                "missing dedicated executor for {}",
                descriptor.model_architecture
            );
        }
    }

    #[test]
    fn builtins_cover_all_native_graph_lowering_architectures() {
        let executors =
            materialize_builtin_executors_by_model_architecture().expect("executor map");
        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            if descriptor.execution_capability != GgmlExecutionCapability::NativeGraphLoweringV1 {
                continue;
            }
            assert!(
                executors.contains_key(descriptor.model_architecture),
                "missing native graph lowering executor for {}",
                descriptor.model_architecture
            );
        }
    }

    #[test]
    fn builtin_dispatch_routes_qwen_native_graph_lowering_capability() {
        let dispatch = build_builtin_ggml_execution_dispatch().expect("builtin dispatch");
        let error = dispatch
            .execute(&missing_runtime_request())
            .expect_err("missing runtime should fail inside qwen executor");

        match error {
            GgmlAsrExecutionError::ExecutorFailed {
                executor_id,
                adapter_id,
                reason,
            } => {
                assert_eq!(executor_id, "qwen3-asr-ggml-executor-v1");
                assert_eq!(adapter_id, crate::QWEN3_ASR_GGML_ADAPTER_ID);
                assert!(
                    reason.contains("could not load runtime preflight"),
                    "{reason}"
                );
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn builtin_dispatch_routes_whisper_dedicated_runtime_executor() {
        let mut request = missing_runtime_request();
        request.selected_family = whisper_runtime_descriptor_v1();
        let dispatch = build_builtin_ggml_execution_dispatch().expect("builtin dispatch");
        let error = dispatch
            .execute(&request)
            .expect_err("missing runtime should fail inside whisper executor");

        match error {
            GgmlAsrExecutionError::ExecutorFailed {
                executor_id,
                adapter_id,
                reason,
            } => {
                assert_eq!(executor_id, "whisper-ggml-executor-v1");
                assert_eq!(adapter_id, crate::WHISPER_GGML_ADAPTER_ID);
                assert!(
                    reason.contains("could not load runtime preflight"),
                    "{reason}"
                );
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn builtin_dispatch_routes_xasr_zipformer_dedicated_runtime_executor() {
        let mut request = missing_runtime_request();
        request.selected_family = xasr_zipformer_runtime_descriptor_v1();
        let dispatch = build_builtin_ggml_execution_dispatch().expect("builtin dispatch");
        let error = dispatch
            .execute(&request)
            .expect_err("missing runtime should fail inside xasr executor");

        match error {
            GgmlAsrExecutionError::ExecutorFailed {
                executor_id,
                adapter_id,
                reason,
            } => {
                assert_eq!(
                    executor_id,
                    crate::arch::XASR_ZIPFORMER_EXECUTOR_COMPONENT_ID
                );
                assert_eq!(adapter_id, crate::XASR_ZIPFORMER_GGML_ADAPTER_ID);
                assert!(
                    reason.contains("could not load runtime preflight"),
                    "{reason}"
                );
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn builtin_streaming_dispatch_registers_xasr_zipformer_native_streaming() {
        let dispatch =
            build_builtin_ggml_streaming_execution_dispatch().expect("builtin streaming dispatch");
        let mut request = streaming_request();
        request.selected_family = xasr_zipformer_runtime_descriptor_v1();

        assert!(dispatch.has_streaming_executor_for(&request.selected_family));
        // X-ASR loads its runtime fail-fast at session start, so the missing
        // fixture runtime must surface here — proving the request routed into
        // the registered xasr streaming executor.
        let error = dispatch
            .start_streaming_session(&request)
            .err()
            .expect("missing runtime must fail at session start");
        let message = format!("{error:?}");
        assert!(
            message.contains(crate::arch::XASR_ZIPFORMER_STREAMING_EXECUTOR_COMPONENT_ID),
            "{message}"
        );
    }

    #[test]
    fn builtin_streaming_dispatch_declares_xasr_as_the_only_frame_sync_family() {
        let dispatch =
            build_builtin_ggml_streaming_execution_dispatch().expect("builtin streaming dispatch");
        let buffered_descriptors = [
            qwen3_asr_runtime_descriptor_v1(),
            whisper_runtime_descriptor_v1(),
            crate::cohere_transcribe_runtime_descriptor_v1(),
            crate::moonshine_runtime_descriptor_v1(),
            parakeet_ctc_runtime_descriptor_v1(),
            wav2vec2_ctc_runtime_descriptor_v1(),
        ];
        for descriptor in &buffered_descriptors {
            assert!(
                !dispatch.is_frame_sync_for(descriptor),
                "{} should be buffered, not frame-sync",
                descriptor.adapter_id
            );
        }
        assert!(dispatch.is_frame_sync_for(&xasr_zipformer_runtime_descriptor_v1()));
    }

    #[test]
    fn builtin_streaming_dispatch_registers_declared_snapshot_executors() {
        let dispatch =
            build_builtin_ggml_streaming_execution_dispatch().expect("builtin streaming dispatch");
        let cases = [
            (
                crate::qwen3_asr_runtime_descriptor_v1(),
                "qwen3-asr-ggml-snapshot-streaming-executor-v1",
            ),
            (
                whisper_runtime_descriptor_v1(),
                "whisper-ggml-snapshot-streaming-executor-v1",
            ),
            (
                crate::cohere_transcribe_runtime_descriptor_v1(),
                "cohere-transcribe-ggml-snapshot-streaming-executor-v1",
            ),
            (
                crate::moonshine_runtime_descriptor_v1(),
                "moonshine-ggml-snapshot-streaming-executor-v1",
            ),
            (
                parakeet_ctc_runtime_descriptor_v1(),
                "parakeet-ctc-ggml-snapshot-streaming-executor-v1",
            ),
            (
                wav2vec2_ctc_runtime_descriptor_v1(),
                "wav2vec2-ctc-ggml-snapshot-streaming-executor-v1",
            ),
            (
                xasr_zipformer_runtime_descriptor_v1(),
                crate::arch::XASR_ZIPFORMER_STREAMING_EXECUTOR_COMPONENT_ID,
            ),
        ];

        for (descriptor, expected_executor_id) in cases {
            let mut request = streaming_request();
            request.selected_family = descriptor;
            // Fail-fast executors (xasr) reject the missing fixture runtime at
            // session start; snapshot executors only fail once decode runs.
            let mut session = match dispatch.start_streaming_session(&request) {
                Ok(session) => session,
                Err(error) => {
                    let message = format!("{error:?}");
                    assert!(message.contains(expected_executor_id), "{message}");
                    assert!(
                        message.contains("could not load runtime preflight"),
                        "{message}"
                    );
                    continue;
                }
            };
            let _ = session.poll_events().unwrap();
            // push_audio only buffers; the decode (which loads the fixture runtime
            // and fails) runs in poll_events once enough audio passes the
            // first-decode floor. Feed ~1.2s, then poll to surface the error.
            let format = crate::realtime::RealtimeAudioFormat::pcm16_mono_16khz();
            let sample_count = format.sample_count_for_duration_ms(20).unwrap();
            let mut error = None;
            for seq in 1..=60u64 {
                match session.push_audio(
                    crate::realtime::RealtimeAudioFrame::new(
                        seq,
                        (seq - 1) * 20,
                        format,
                        vec![0; sample_count],
                    )
                    .unwrap(),
                ) {
                    Ok(_) => {}
                    Err(push_error) => {
                        error = Some(push_error);
                        break;
                    }
                }
            }
            let error = match error {
                Some(error) => error,
                None => session
                    .poll_events()
                    .expect_err("missing runtime should fail on streaming decode"),
            };
            match error {
                crate::NativeAsrError::SessionFailed { message } => {
                    assert!(message.contains(expected_executor_id), "{message}");
                    assert!(
                        message.contains("could not load runtime preflight"),
                        "{message}"
                    );
                }
                other => panic!("unexpected error {other:?}"),
            }
        }
    }
}
