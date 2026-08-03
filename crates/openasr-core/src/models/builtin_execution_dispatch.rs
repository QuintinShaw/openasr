use std::sync::Arc;

use thiserror::Error;

use crate::GgmlAsrExecutionDispatch;
use crate::StreamingPartialGranularity;
use crate::arch::{OpenAsrArchitectureRegistry, OpenAsrArchitectureRegistryError};

use super::executor_component_registry::{
    BuiltinExecutorComponentRegistryError, BuiltinStatefulExecutorScope,
    materialize_builtin_executors_by_model_architecture,
};
use super::ggml_asr_executor::GgmlAsrStreamingExecutor;
use super::ggml_composed_executor::ComposedGgmlAsrExecutor;
use super::ggml_family_adapter::GgmlExecutionCapability;

#[derive(Debug, Error, Clone, PartialEq)]
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
    #[error(
        "builtin streaming dispatch is missing a streaming executor for ASR architecture '{model_architecture}' (every registered family must declare one so realtime cadence stays descriptor-driven)"
    )]
    MissingStreamingExecutor { model_architecture: &'static str },
    #[error(
        "builtin streaming dispatch partial granularity disagrees with the architecture integration descriptor for '{model_architecture}': expected {expected:?}, actual {actual:?}"
    )]
    StreamingGranularityMismatch {
        model_architecture: &'static str,
        expected: StreamingPartialGranularity,
        actual: StreamingPartialGranularity,
    },
    #[error(
        "builtin streaming dispatch is missing partial granularity derived from the architecture integration descriptor for '{model_architecture}'"
    )]
    MissingStreamingGranularity { model_architecture: &'static str },
    #[error("native family runtime wiring validation failed: {reason}")]
    FamilyWiringInvalid { reason: String },
    #[error("builtin architecture registry failed validation: {error:?}")]
    ArchitectureRegistryInvalid {
        error: OpenAsrArchitectureRegistryError,
    },
}

pub(crate) fn build_builtin_ggml_execution_dispatch(
    stateful_executors: &BuiltinStatefulExecutorScope,
) -> Result<GgmlAsrExecutionDispatch, BuiltinGgmlExecutionDispatchError> {
    let registry = OpenAsrArchitectureRegistry::with_builtins();
    registry.validate_references().map_err(|error| {
        BuiltinGgmlExecutionDispatchError::ArchitectureRegistryInvalid { error }
    })?;
    // Force-link pack-import convert entries and run the in-memory wiring gate.
    // Never walks the source tree: release binaries have no docs/tooling checkout.
    let _pack_imports = crate::models::pack_import_surface::linked_core_pack_import_symbols();
    crate::models::family_integration_audit::validate_builtin_runtime_family_wiring().map_err(
        |error| BuiltinGgmlExecutionDispatchError::FamilyWiringInvalid {
            reason: error.to_string(),
        },
    )?;

    let mut dispatch = GgmlAsrExecutionDispatch::default();
    let executors_by_model_architecture = materialize_builtin_executors_by_model_architecture(
        stateful_executors,
    )
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
                dispatch = dispatch
                    .with_view_executor_for_adapter(descriptor.adapter_id, Arc::clone(executor));
            }
            GgmlExecutionCapability::NativeGraphLoweringV1 => {
                native_graph_lowering_executors
                    .push((descriptor.model_architecture, Arc::clone(executor)));
            }
        }
    }

    if !native_graph_lowering_executors.is_empty() {
        dispatch = dispatch.with_view_executor_for_capability(
            GgmlExecutionCapability::NativeGraphLoweringV1,
            Arc::new(
                ComposedGgmlAsrExecutor::default()
                    .with_architecture_executors(native_graph_lowering_executors),
            ),
        );
    }

    Ok(dispatch)
}

fn builtin_streaming_executor_for_architecture(
    stateful_executors: &BuiltinStatefulExecutorScope,
    model_architecture: &str,
) -> Option<Arc<dyn GgmlAsrStreamingExecutor>> {
    match model_architecture {
        crate::QWEN3_ASR_GGML_ARCHITECTURE_ID => {
            Some(stateful_executors.qwen3_asr() as Arc<dyn GgmlAsrStreamingExecutor>)
        }
        crate::WHISPER_GGML_ARCHITECTURE_ID => {
            Some(stateful_executors.whisper() as Arc<dyn GgmlAsrStreamingExecutor>)
        }
        crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID => {
            Some(stateful_executors.cohere_transcribe() as Arc<dyn GgmlAsrStreamingExecutor>)
        }
        crate::MOONSHINE_GGML_ARCHITECTURE_ID => {
            Some(stateful_executors.moonshine() as Arc<dyn GgmlAsrStreamingExecutor>)
        }
        crate::PARAKEET_CTC_GGML_ARCHITECTURE_ID => {
            Some(stateful_executors.parakeet_ctc() as Arc<dyn GgmlAsrStreamingExecutor>)
        }
        crate::PARAKEET_TDT_GGML_ARCHITECTURE_ID => {
            Some(stateful_executors.parakeet_tdt() as Arc<dyn GgmlAsrStreamingExecutor>)
        }
        crate::arch::SENSEVOICE_GGML_ARCHITECTURE_ID => {
            Some(stateful_executors.sensevoice() as Arc<dyn GgmlAsrStreamingExecutor>)
        }
        crate::WAV2VEC2_CTC_GGML_ARCHITECTURE_ID => {
            Some(stateful_executors.wav2vec2_ctc() as Arc<dyn GgmlAsrStreamingExecutor>)
        }
        crate::arch::DOLPHIN_GGML_ARCHITECTURE_ID => {
            Some(stateful_executors.dolphin() as Arc<dyn GgmlAsrStreamingExecutor>)
        }
        crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID => {
            Some(stateful_executors.firered_aed() as Arc<dyn GgmlAsrStreamingExecutor>)
        }
        crate::arch::FIRERED_LLM_GGML_ARCHITECTURE_ID => {
            Some(stateful_executors.firered_llm() as Arc<dyn GgmlAsrStreamingExecutor>)
        }
        crate::arch::FUNASR_NANO_GGML_ARCHITECTURE_ID => {
            Some(stateful_executors.funasr_nano() as Arc<dyn GgmlAsrStreamingExecutor>)
        }
        crate::arch::MIMO_ASR_GGML_ARCHITECTURE_ID => {
            Some(stateful_executors.mimo_asr() as Arc<dyn GgmlAsrStreamingExecutor>)
        }
        crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID => {
            Some(stateful_executors.moss_td() as Arc<dyn GgmlAsrStreamingExecutor>)
        }
        crate::arch::GRANITE_SPEECH_GGML_ARCHITECTURE_ID => {
            Some(stateful_executors.granite_speech() as Arc<dyn GgmlAsrStreamingExecutor>)
        }
        crate::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID => {
            Some(stateful_executors.xasr_zipformer() as Arc<dyn GgmlAsrStreamingExecutor>)
        }
        _ => None,
    }
}

pub(crate) fn build_builtin_ggml_streaming_execution_dispatch(
    stateful_executors: &BuiltinStatefulExecutorScope,
) -> Result<GgmlAsrExecutionDispatch, BuiltinGgmlExecutionDispatchError> {
    let registry = OpenAsrArchitectureRegistry::with_builtins();
    registry.validate_references().map_err(|error| {
        BuiltinGgmlExecutionDispatchError::ArchitectureRegistryInvalid { error }
    })?;
    // Same in-memory gate as offline dispatch (force-link + registry wiring).
    let _pack_imports = crate::models::pack_import_surface::linked_core_pack_import_symbols();
    crate::models::family_integration_audit::validate_builtin_runtime_family_wiring().map_err(
        |error| BuiltinGgmlExecutionDispatchError::FamilyWiringInvalid {
            reason: error.to_string(),
        },
    )?;

    // Streaming executors remain family-implemented, but partial-result
    // granularity is derived from the architecture integration descriptor
    // (single source) rather than hand-written a second time here.
    let mut dispatch = GgmlAsrExecutionDispatch::default();
    // Keep the same family-owned offline executor contracts alongside the
    // streaming executors. They are planner providers only on this dispatch;
    // `start_streaming_session` still routes exclusively through the
    // streaming maps below.
    let planning_executors = materialize_builtin_executors_by_model_architecture(
        stateful_executors,
    )
    .map_err(|source| BuiltinGgmlExecutionDispatchError::ExecutorMaterialization { source })?;
    for architecture in registry.descriptors() {
        let Some(executor) = builtin_streaming_executor_for_architecture(
            stateful_executors,
            architecture.model_architecture,
        ) else {
            return Err(
                BuiltinGgmlExecutionDispatchError::MissingStreamingExecutor {
                    model_architecture: architecture.model_architecture,
                },
            );
        };
        let planning_executor = planning_executors
            .get(architecture.model_architecture)
            .ok_or(
                BuiltinGgmlExecutionDispatchError::MissingMaterializedExecutor {
                    model_architecture: architecture.model_architecture,
                },
            )?;
        dispatch = dispatch
            .with_view_executor_for_adapter(architecture.adapter_id, Arc::clone(planning_executor))
            .with_streaming_executor_for_adapter(architecture.adapter_id, executor)
            .with_streaming_partial_granularity_for_adapter(
                architecture.adapter_id,
                architecture.integration.streaming_partial_granularity,
            );
    }

    // Fail-fast completeness gate: realtime driver selection is descriptor-driven
    // (see `native_runtime_streaming_capabilities_for_descriptor`). A registered
    // ASR family with no streaming executor would silently fall back to the
    // buffered file-per-utterance path -- the exact "no partials until a long
    // pause" defect. Reject that at startup so onboarding a new family fails
    // loudly here instead of shipping a broken live-caption cadence.
    let family_registry =
        crate::models::ggml_family_registry::GgmlFamilyRegistry::with_builtin_adapters();
    for descriptor in family_registry.descriptors() {
        if !dispatch.has_streaming_executor_for(descriptor) {
            return Err(
                BuiltinGgmlExecutionDispatchError::MissingStreamingExecutor {
                    model_architecture: descriptor.model_architecture,
                },
            );
        }
        let architecture = registry
            .find_by_model_architecture(descriptor.model_architecture)
            .expect("family registry is derived from architecture registry");
        let expected = architecture.integration.streaming_partial_granularity;
        let Some(actual) = dispatch.streaming_partial_granularity_for(descriptor) else {
            return Err(
                BuiltinGgmlExecutionDispatchError::MissingStreamingGranularity {
                    model_architecture: descriptor.model_architecture,
                },
            );
        };
        if actual != expected {
            return Err(
                BuiltinGgmlExecutionDispatchError::StreamingGranularityMismatch {
                    model_architecture: descriptor.model_architecture,
                    expected,
                    actual,
                },
            );
        }
    }

    Ok(dispatch)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::PathBuf;

    use super::*;
    use crate::{
        GgmlAsrBackendPreference, GgmlAsrExecutionError, GgmlAsrExecutionViewRequest,
        GgmlAsrPreparedAudioView, GgmlAsrStreamingSessionRequest, NativeAsrSessionContext,
        NativeAsrStreamingSessionConfig, parakeet_ctc_runtime_descriptor_v1,
        qwen3_asr_runtime_descriptor_v1, wav2vec2_ctc_runtime_descriptor_v1,
        whisper_runtime_descriptor_v1, xasr_zipformer_runtime_descriptor_v1,
    };

    fn missing_runtime_request() -> GgmlAsrExecutionViewRequest<'static> {
        GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            runtime_source_path: PathBuf::from("/tmp/openasr-missing-runtime.gguf"),
            runtime_source_preflight: None,
            selected_family: qwen3_asr_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(vec![0.0, 0.1]),
            request_options: crate::GgmlAsrExecutionOptions::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                (GgmlAsrBackendPreference::CpuOnly).request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        }
    }

    fn missing_runtime_owned_request() -> crate::GgmlAsrExecutionRequest {
        crate::GgmlAsrExecutionRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            runtime_source_path: PathBuf::from("/tmp/openasr-missing-runtime.gguf"),
            runtime_source_preflight: None,
            selected_family: qwen3_asr_runtime_descriptor_v1(),
            prepared_audio: crate::GgmlAsrPreparedAudio::mono_16khz(vec![0.0, 0.1]),
            request_options: crate::GgmlAsrExecutionOptions::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                (GgmlAsrBackendPreference::CpuOnly).request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        }
    }

    fn materialize_test_executors() -> Result<
        BTreeMap<&'static str, Arc<dyn crate::models::ggml_asr_executor::GgmlAsrViewExecutor>>,
        BuiltinExecutorComponentRegistryError,
    > {
        materialize_builtin_executors_by_model_architecture(&BuiltinStatefulExecutorScope::new())
    }

    fn build_test_offline_dispatch()
    -> Result<GgmlAsrExecutionDispatch, BuiltinGgmlExecutionDispatchError> {
        build_builtin_ggml_execution_dispatch(&BuiltinStatefulExecutorScope::new())
    }

    fn build_test_streaming_dispatch()
    -> Result<GgmlAsrExecutionDispatch, BuiltinGgmlExecutionDispatchError> {
        build_builtin_ggml_streaming_execution_dispatch(&BuiltinStatefulExecutorScope::new())
    }

    fn executor_data_ptr<T: ?Sized>(executor: &Arc<T>) -> *const () {
        Arc::as_ptr(executor) as *const ()
    }

    #[test]
    fn builtin_offline_and_streaming_maps_share_scope_executors_and_isolate_scopes() {
        let first_scope = BuiltinStatefulExecutorScope::new();
        let second_scope = BuiltinStatefulExecutorScope::new();
        let first_offline = materialize_builtin_executors_by_model_architecture(&first_scope)
            .expect("first offline executor map");
        let second_offline = materialize_builtin_executors_by_model_architecture(&second_scope)
            .expect("second offline executor map");

        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            let first_offline_executor = first_offline
                .get(descriptor.model_architecture)
                .unwrap_or_else(|| {
                    panic!(
                        "missing first offline executor for {}",
                        descriptor.model_architecture
                    )
                });
            let second_offline_executor = second_offline
                .get(descriptor.model_architecture)
                .unwrap_or_else(|| {
                    panic!(
                        "missing second offline executor for {}",
                        descriptor.model_architecture
                    )
                });
            let first_streaming_executor = builtin_streaming_executor_for_architecture(
                &first_scope,
                descriptor.model_architecture,
            )
            .unwrap_or_else(|| {
                panic!(
                    "missing first streaming executor for {}",
                    descriptor.model_architecture
                )
            });
            let second_streaming_executor = builtin_streaming_executor_for_architecture(
                &second_scope,
                descriptor.model_architecture,
            )
            .unwrap_or_else(|| {
                panic!(
                    "missing second streaming executor for {}",
                    descriptor.model_architecture
                )
            });

            assert_eq!(
                executor_data_ptr(first_offline_executor),
                executor_data_ptr(&first_streaming_executor),
                "{} must share one executor allocation between offline and streaming in one scope",
                descriptor.model_architecture
            );
            assert_ne!(
                executor_data_ptr(first_offline_executor),
                executor_data_ptr(second_offline_executor),
                "{} offline executors must be isolated between scopes",
                descriptor.model_architecture
            );
            assert_ne!(
                executor_data_ptr(&first_streaming_executor),
                executor_data_ptr(&second_streaming_executor),
                "{} streaming executors must be isolated between scopes",
                descriptor.model_architecture
            );
            assert_ne!(
                executor_data_ptr(first_offline_executor),
                executor_data_ptr(&second_streaming_executor),
                "{} must not cross-share allocations between scopes",
                descriptor.model_architecture
            );
        }
    }

    fn streaming_request() -> GgmlAsrStreamingSessionRequest {
        GgmlAsrStreamingSessionRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            runtime_source_path: PathBuf::from("/tmp/openasr-missing-runtime.gguf"),
            runtime_source_preflight: None,
            selected_family: qwen3_asr_runtime_descriptor_v1(),
            request_options: crate::GgmlAsrExecutionOptions::default(),
            configured_diarize: false,
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                (GgmlAsrBackendPreference::CpuOnly).request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            final_text_processor: None,
            session_context: NativeAsrSessionContext::new("rt_builtin_streaming"),
            session_config: NativeAsrStreamingSessionConfig::new()
                .with_partial_results(true)
                .into(),
        }
    }

    #[test]
    fn builtins_cover_all_dedicated_runtime_architectures() {
        let executors = materialize_test_executors().expect("executor map");
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
        let executors = materialize_test_executors().expect("executor map");
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
    fn builtin_dispatch_rejects_unplanned_qwen_request_before_executor() {
        let dispatch = build_test_offline_dispatch().expect("builtin dispatch");
        let error = dispatch
            .execute(&missing_runtime_owned_request())
            .expect_err("decoder family request without a plan must fail closed");
        assert!(matches!(
            error,
            GgmlAsrExecutionError::DecoderStateContractMismatch {
                executor_id: "openasr-ggml-composed-executor-v1",
                adapter_id: crate::QWEN3_ASR_GGML_ADAPTER_ID,
            }
        ));
    }

    #[test]
    fn builtin_dispatch_rejects_unplanned_whisper_request_before_executor() {
        let mut request = missing_runtime_owned_request();
        request.selected_family = whisper_runtime_descriptor_v1();
        let dispatch = build_test_offline_dispatch().expect("builtin dispatch");
        let error = dispatch
            .execute(&request)
            .expect_err("decoder family request without a plan must fail closed");
        assert!(matches!(
            error,
            GgmlAsrExecutionError::DecoderStateContractMismatch {
                executor_id: "whisper-ggml-executor-v1",
                adapter_id: crate::WHISPER_GGML_ADAPTER_ID,
            }
        ));
    }

    #[test]
    fn builtin_dispatch_routes_xasr_zipformer_dedicated_runtime_executor() {
        let mut request = missing_runtime_request();
        request.selected_family = xasr_zipformer_runtime_descriptor_v1();
        let dispatch = build_test_offline_dispatch().expect("builtin dispatch");
        let error = dispatch
            .execute_view(&request)
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
        let dispatch = build_test_streaming_dispatch().expect("builtin streaming dispatch");
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
    fn builtin_streaming_frame_sync_set_matches_architecture_integration_manifest() {
        let dispatch = build_test_streaming_dispatch().expect("builtin streaming dispatch");
        let architecture_registry = OpenAsrArchitectureRegistry::with_builtins();
        let family_registry =
            crate::models::ggml_family_registry::GgmlFamilyRegistry::with_builtin_adapters();

        let mut manifest_frame_sync = BTreeSet::new();
        let mut dispatch_frame_sync = BTreeSet::new();
        for descriptor in family_registry.descriptors() {
            let architecture = architecture_registry
                .find_by_model_architecture(descriptor.model_architecture)
                .expect("family registry is derived from architecture registry");
            if architecture.integration.streaming_partial_granularity
                == StreamingPartialGranularity::FrameSync
            {
                manifest_frame_sync.insert(descriptor.model_architecture);
            }
            if dispatch.is_frame_sync_for(descriptor) {
                dispatch_frame_sync.insert(descriptor.model_architecture);
            }
            assert_eq!(
                dispatch.streaming_partial_granularity_for(descriptor),
                Some(architecture.integration.streaming_partial_granularity),
                "family '{}' streaming granularity must be derived from its architecture integration descriptor",
                descriptor.adapter_id
            );
        }
        assert_eq!(
            manifest_frame_sync, dispatch_frame_sync,
            "FrameSync architecture set must equal the streaming dispatch FrameSync set"
        );
    }

    #[test]
    fn builtin_streaming_dispatch_covers_every_registered_asr_family() {
        // The startup completeness gate: every family the runtime can select must
        // have a streaming executor, so realtime cadence stays descriptor-driven
        // and no family silently falls back to buffered file-per-utterance.
        let dispatch = build_test_streaming_dispatch().expect("builtin streaming dispatch");
        let family_registry =
            crate::models::ggml_family_registry::GgmlFamilyRegistry::with_builtin_adapters();
        let architecture_registry = OpenAsrArchitectureRegistry::with_builtins();
        for descriptor in family_registry.descriptors() {
            assert!(
                dispatch.has_streaming_executor_for(descriptor),
                "family '{}' ({}) has no streaming executor",
                descriptor.adapter_id,
                descriptor.model_architecture,
            );
            let architecture = architecture_registry
                .find_by_model_architecture(descriptor.model_architecture)
                .expect("family registry is derived from architecture registry");
            assert_eq!(
                dispatch.streaming_partial_granularity_for(descriptor),
                Some(architecture.integration.streaming_partial_granularity),
                "family '{}' ({}) streaming partial granularity must come from its architecture integration descriptor",
                descriptor.adapter_id,
                descriptor.model_architecture,
            );
        }
    }

    #[test]
    fn builtin_streaming_dispatch_registers_declared_snapshot_executors() {
        let dispatch = build_test_streaming_dispatch().expect("builtin streaming dispatch");
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
                crate::sensevoice_runtime_descriptor_v1(),
                "sensevoice-ggml-snapshot-streaming-executor-v1",
            ),
            (
                crate::dolphin_runtime_descriptor_v1(),
                "dolphin-ggml-snapshot-streaming-executor-v1",
            ),
            (
                xasr_zipformer_runtime_descriptor_v1(),
                crate::arch::XASR_ZIPFORMER_STREAMING_EXECUTOR_COMPONENT_ID,
            ),
        ];

        for (descriptor, expected_executor_id) in cases {
            assert_eq!(
                dispatch.streaming_executor_id_for(&descriptor),
                Some(expected_executor_id),
                "family '{}' ({}) must register its declared streaming executor",
                descriptor.adapter_id,
                descriptor.model_architecture,
            );
        }
    }
}
