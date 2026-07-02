use crate::{
    GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutionResult, GgmlAsrExecutor,
};
use std::{collections::BTreeMap, sync::Arc};

const COMPOSED_EXECUTOR_ID: &str = "openasr-ggml-composed-executor-v1";

#[derive(Default)]
pub(crate) struct ComposedGgmlAsrExecutor {
    executors_by_model_architecture: BTreeMap<&'static str, Arc<dyn GgmlAsrExecutor>>,
}

impl ComposedGgmlAsrExecutor {
    pub(crate) fn with_architecture_executors(
        mut self,
        executors_by_model_architecture: impl IntoIterator<
            Item = (&'static str, Arc<dyn GgmlAsrExecutor>),
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
        executor: Arc<dyn GgmlAsrExecutor>,
    ) -> Self {
        self.executors_by_model_architecture
            .insert(model_architecture, executor);
        self
    }
}

impl GgmlAsrExecutor for ComposedGgmlAsrExecutor {
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

    fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
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
        executor.execute(request)
    }
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use crate::{
        GgmlAsrBackendPreference, GgmlAsrExecutionOptions, GgmlAsrPreparedAudio, Transcription,
        qwen3_asr_runtime_descriptor_v1, whisper_runtime_descriptor_v1,
    };

    use super::*;

    struct StubExecutor {
        text: &'static str,
    }

    impl GgmlAsrExecutor for StubExecutor {
        fn executor_id(&self) -> &'static str {
            self.text
        }

        fn supports_phrase_bias(&self) -> bool {
            true
        }

        fn execute(
            &self,
            _request: &GgmlAsrExecutionRequest,
        ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
            Ok(GgmlAsrExecutionResult {
                transcription: Transcription {
                    text: self.text.to_string(),
                    segments: Vec::new(),
                    longform: None,
                    language: None,
                },
                carry_context: None,
            })
        }
    }

    fn qwen_request() -> GgmlAsrExecutionRequest {
        GgmlAsrExecutionRequest {
            runtime_source_path: PathBuf::from("fixtures/qwen.gguf"),
            runtime_source_preflight: None,
            selected_family: qwen3_asr_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(vec![0.0, 0.1]),
            request_options: GgmlAsrExecutionOptions::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
        }
    }

    #[test]
    fn composed_executor_dispatches_by_model_architecture() {
        let executor = ComposedGgmlAsrExecutor::default().with_architecture_executor(
            crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            Arc::new(StubExecutor { text: "qwen" }),
        );

        let result = executor
            .execute(&qwen_request())
            .expect("qwen should dispatch");
        assert_eq!(result.transcription.text, "qwen");
    }

    #[test]
    fn composed_executor_fails_closed_when_architecture_is_not_registered() {
        let mut request = qwen_request();
        request.selected_family = whisper_runtime_descriptor_v1();
        let executor = ComposedGgmlAsrExecutor::default().with_architecture_executor(
            crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            Arc::new(StubExecutor { text: "qwen" }),
        );

        let error = executor
            .execute(&request)
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
}
