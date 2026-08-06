use crate::models::ggml_asr_executor::{
    GgmlAsrDecoderState, GgmlAsrDecoderStateContract, GgmlAsrDecoderStatePlanningInput,
};
use crate::models::ggml_family_adapter::GgmlAdapterBindingStrategy;
use crate::{
    GgmlAsrExecutionError, GgmlAsrExecutionResult, GgmlAsrExecutionViewRequest,
    GgmlAsrViewExecutor, GgmlFamilyAdapterDescriptor,
};
use std::{collections::BTreeMap, sync::Arc};

const COMPOSED_EXECUTOR_ID: &str = "openasr-ggml-composed-executor-v1";

#[derive(Default)]
pub(crate) struct ComposedGgmlAsrExecutor {
    executors_by_model_architecture: BTreeMap<&'static str, Arc<dyn GgmlAsrViewExecutor>>,
}

impl ComposedGgmlAsrExecutor {
    pub(crate) fn with_architecture_executors(
        mut self,
        executors_by_model_architecture: impl IntoIterator<
            Item = (&'static str, Arc<dyn GgmlAsrViewExecutor>),
        >,
    ) -> Self {
        for (model_architecture, executor) in executors_by_model_architecture {
            self = self.with_architecture_executor(model_architecture, executor);
        }
        self
    }

    pub(crate) fn with_architecture_executor(
        mut self,
        model_architecture: &'static str,
        executor: Arc<dyn GgmlAsrViewExecutor>,
    ) -> Self {
        self.executors_by_model_architecture
            .insert(model_architecture, executor);
        self
    }
}

impl GgmlAsrViewExecutor for ComposedGgmlAsrExecutor {
    fn executor_id(&self) -> &'static str {
        COMPOSED_EXECUTOR_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        !self.executors_by_model_architecture.is_empty()
            && self
                .executors_by_model_architecture
                .values()
                .all(|executor| executor.supports_phrase_bias())
    }

    fn adapter_binding_strategy_for(
        &self,
        selected_family: &GgmlFamilyAdapterDescriptor,
    ) -> Result<GgmlAdapterBindingStrategy, GgmlAsrExecutionError> {
        let executor = self
            .executors_by_model_architecture
            .get(selected_family.model_architecture)
            .ok_or(GgmlAsrExecutionError::ExecutorUnavailable {
                adapter_id: selected_family.adapter_id,
                model_family: selected_family.model_family,
                capability: "model-architecture-executor",
            })?;
        executor.adapter_binding_strategy_for(selected_family)
    }

    fn decoder_state_contract(
        &self,
        selected_family: &GgmlFamilyAdapterDescriptor,
    ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError> {
        let executor = self
            .executors_by_model_architecture
            .get(selected_family.model_architecture)
            .ok_or(GgmlAsrExecutionError::ExecutorUnavailable {
                adapter_id: selected_family.adapter_id,
                model_family: selected_family.model_family,
                capability: "model-architecture-executor",
            })?;
        executor.decoder_state_contract(selected_family)
    }

    fn replan_streaming_decoder_state(
        &self,
        selected_family: &GgmlFamilyAdapterDescriptor,
        input: &GgmlAsrDecoderStatePlanningInput<'_>,
    ) -> Result<GgmlAsrDecoderState, GgmlAsrExecutionError> {
        let executor = self
            .executors_by_model_architecture
            .get(selected_family.model_architecture)
            .ok_or(GgmlAsrExecutionError::ExecutorUnavailable {
                adapter_id: selected_family.adapter_id,
                model_family: selected_family.model_family,
                capability: "model-architecture-executor",
            })?;
        executor.replan_streaming_decoder_state(selected_family, input)
    }

    fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest<'_>,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        let Some(executor) = self
            .executors_by_model_architecture
            .get(request.selected_family.model_architecture)
        else {
            return Err(GgmlAsrExecutionError::ExecutorUnavailable {
                adapter_id: request.selected_family.adapter_id,
                model_family: request.selected_family.model_family,
                capability: "model-architecture-executor",
            });
        };
        executor.execute_view(request)
    }

    fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        for executor in self.executors_by_model_architecture.values() {
            executor.evict_prepared_runtime_content_id(pack_content_id);
        }
    }

    fn unload_idle_state(&self) {
        // The composed executor is itself registered in the dispatch maps
        // (see `builtin_execution_dispatch.rs`), so the reaper only ever
        // calls unload on this wrapper -- without forwarding, every
        // architecture behind it (e.g. qwen3-asr's NativeGraphLoweringV1
        // executor) never sees `unload_idle_state` and its cached prepared
        // runtime lives forever.
        for executor in self.executors_by_model_architecture.values() {
            executor.unload_idle_state();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        GgmlAsrBackendPreference, GgmlAsrExecutionOptions, GgmlAsrPreparedAudioView, Transcription,
    };

    use super::*;

    struct StubExecutor {
        text: &'static str,
        adapter_binding: GgmlAdapterBindingStrategy,
    }

    impl GgmlAsrViewExecutor for StubExecutor {
        fn executor_id(&self) -> &'static str {
            self.text
        }

        fn adapter_binding_strategy(&self) -> GgmlAdapterBindingStrategy {
            self.adapter_binding
        }

        fn supports_phrase_bias(&self) -> bool {
            true
        }

        fn decoder_state_contract(
            &self,
            _selected_family: &crate::GgmlFamilyAdapterDescriptor,
        ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError> {
            Ok(GgmlAsrDecoderStateContract::NoPersistentState)
        }

        fn execute_view(
            &self,
            _request: &GgmlAsrExecutionViewRequest<'_>,
        ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
            Ok(GgmlAsrExecutionResult {
                transcription: Transcription {
                    truncated_decodes: Vec::new(),
                    unnamed_speakers: Vec::new(),
                    text: self.text.to_string(),
                    segments: Vec::new(),
                    longform: None,
                    language: None,
                    ..Default::default()
                },
                carry_context: None,
                decode_truncation: None,
            })
        }

        fn evict_prepared_runtime_content_id(&self, _pack_content_id: &str) {}
    }

    fn qwen_request() -> GgmlAsrExecutionViewRequest<'static> {
        let verified_pack = crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
            crate::models::runtime_preflight::leaked_tiny_runtime_source_preflight(),
            crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID,
        );
        GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack,
            selected_family: crate::arch::builtin_adapter_descriptor(
                crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            ),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(vec![0.0, 0.1]),
            request_options: GgmlAsrExecutionOptions::default(),
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

    #[test]
    fn composed_executor_dispatches_by_model_architecture() {
        let executor = ComposedGgmlAsrExecutor::default().with_architecture_executor(
            crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            Arc::new(StubExecutor {
                text: "qwen",
                adapter_binding: GgmlAdapterBindingStrategy::Qwen3AsrLoraV1,
            }),
        );

        let result = executor
            .execute_view(&qwen_request())
            .expect("qwen should dispatch");
        assert_eq!(result.transcription.text, "qwen");
    }

    #[test]
    fn composed_executor_fails_closed_when_architecture_is_not_registered() {
        let mut request = qwen_request();
        request.selected_family =
            crate::arch::builtin_adapter_descriptor(crate::arch::WHISPER_GGML_ARCHITECTURE_ID);
        let executor = ComposedGgmlAsrExecutor::default().with_architecture_executor(
            crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            Arc::new(StubExecutor {
                text: "qwen",
                adapter_binding: GgmlAdapterBindingStrategy::Qwen3AsrLoraV1,
            }),
        );

        let error = executor
            .execute_view(&request)
            .expect_err("missing architecture executor must fail closed");
        assert!(matches!(
            error,
            GgmlAsrExecutionError::ExecutorUnavailable {
                adapter_id: crate::WHISPER_GGML_ADAPTER_ID,
                model_family: crate::WHISPER_MODEL_FAMILY,
                capability: "model-architecture-executor",
            }
        ));
    }

    #[test]
    fn adapter_binding_strategy_is_delegated_to_the_selected_architecture() {
        let executor = ComposedGgmlAsrExecutor::default()
            .with_architecture_executor(
                crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
                Arc::new(StubExecutor {
                    text: "qwen",
                    adapter_binding: GgmlAdapterBindingStrategy::Qwen3AsrLoraV1,
                }),
            )
            .with_architecture_executor(
                crate::arch::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                Arc::new(StubExecutor {
                    text: "cohere",
                    adapter_binding: GgmlAdapterBindingStrategy::Unsupported,
                }),
            );

        let qwen =
            crate::arch::builtin_adapter_descriptor(crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID);
        let cohere = crate::arch::builtin_adapter_descriptor(
            crate::arch::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
        );

        assert_eq!(
            executor
                .adapter_binding_strategy_for(&qwen)
                .expect("qwen child strategy"),
            GgmlAdapterBindingStrategy::Qwen3AsrLoraV1
        );
        assert_eq!(
            executor
                .adapter_binding_strategy_for(&cohere)
                .expect("cohere child strategy"),
            GgmlAdapterBindingStrategy::Unsupported
        );
    }

    #[test]
    fn unload_idle_state_forwards_to_every_wrapped_architecture_executor() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Reproduces the qwen3-asr idle-unload bug: the composed executor is
        // what the daemon's reaper actually holds a handle to (registered by
        // model-architecture executor set), so if it does not forward
        // `unload_idle_state` to the wrapped per-architecture executors, the
        // inner executor's cached prepared runtime is never evicted.
        struct CountingExecutor(Arc<AtomicUsize>);
        impl GgmlAsrViewExecutor for CountingExecutor {
            fn executor_id(&self) -> &'static str {
                "counting-architecture-stub"
            }
            fn supports_phrase_bias(&self) -> bool {
                true
            }
            fn decoder_state_contract(
                &self,
                _selected_family: &crate::GgmlFamilyAdapterDescriptor,
            ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError> {
                Ok(GgmlAsrDecoderStateContract::NoPersistentState)
            }
            fn execute_view(
                &self,
                _request: &GgmlAsrExecutionViewRequest<'_>,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                unreachable!("this test never executes a request")
            }
            fn evict_prepared_runtime_content_id(&self, _pack_content_id: &str) {}
            fn unload_idle_state(&self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let unload_calls = Arc::new(AtomicUsize::new(0));
        let executor = ComposedGgmlAsrExecutor::default().with_architecture_executor(
            crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            Arc::new(CountingExecutor(Arc::clone(&unload_calls))),
        );

        executor.unload_idle_state();

        assert_eq!(unload_calls.load(Ordering::SeqCst), 1);
    }
}
