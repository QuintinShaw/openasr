use std::{collections::BTreeMap, sync::Arc};

use thiserror::Error;

use crate::arch::{
    COHERE_TRANSCRIBE_EXECUTOR_COMPONENT_ID, DOLPHIN_EXECUTOR_COMPONENT_ID,
    MOONSHINE_EXECUTOR_COMPONENT_ID, OpenAsrArchitectureRegistry,
    PARAKEET_CTC_EXECUTOR_COMPONENT_ID, QWEN3_ASR_EXECUTOR_COMPONENT_ID,
    SENSEVOICE_EXECUTOR_COMPONENT_ID, WAV2VEC2_CTC_EXECUTOR_COMPONENT_ID,
    WHISPER_EXECUTOR_COMPONENT_ID, XASR_ZIPFORMER_EXECUTOR_COMPONENT_ID,
};

use super::cohere::CohereTranscribeGgmlExecutor;
use super::dolphin::executor::DolphinGgmlExecutor;
use super::ggml_asr_executor::GgmlAsrExecutor;
use super::moonshine::MoonshineGgmlExecutor;
use super::parakeet_ctc::executor::ParakeetCtcGgmlExecutor;
use super::qwen::Qwen3AsrGgmlExecutor;
use super::sensevoice::executor::SenseVoiceGgmlExecutor;
use super::wav2vec2_ctc::executor::Wav2Vec2CtcGgmlExecutor;
use super::whisper::WhisperGgmlExecutor;
use super::xasr_zipformer::executor::XasrZipformerGgmlExecutor;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum BuiltinExecutorComponentRegistryError {
    #[error(
        "unknown builtin executor component '{executor_component_id}' for architecture '{model_architecture}'"
    )]
    UnknownExecutorComponent {
        model_architecture: String,
        executor_component_id: String,
    },
}

pub(crate) fn materialize_builtin_executors_by_model_architecture()
-> Result<BTreeMap<&'static str, Arc<dyn GgmlAsrExecutor>>, BuiltinExecutorComponentRegistryError> {
    let mut executors_by_model_architecture = BTreeMap::new();

    for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
        let executor = materialize_builtin_executor_component(descriptor.executor_component_id)
            .ok_or_else(
                || BuiltinExecutorComponentRegistryError::UnknownExecutorComponent {
                    model_architecture: descriptor.model_architecture.to_string(),
                    executor_component_id: descriptor.executor_component_id.to_string(),
                },
            )?;
        executors_by_model_architecture.insert(descriptor.model_architecture, executor);
    }

    Ok(executors_by_model_architecture)
}

pub(crate) fn builtin_executor_supports_phrase_bias_for_model_architecture(
    model_architecture: &str,
) -> Option<bool> {
    materialize_builtin_executors_by_model_architecture()
        .ok()?
        .get(model_architecture)
        .map(|executor| executor.supports_phrase_bias())
}

fn materialize_builtin_executor_component(
    executor_component_id: &str,
) -> Option<Arc<dyn GgmlAsrExecutor>> {
    match executor_component_id {
        COHERE_TRANSCRIBE_EXECUTOR_COMPONENT_ID => {
            Some(Arc::new(CohereTranscribeGgmlExecutor::default()))
        }
        WHISPER_EXECUTOR_COMPONENT_ID => Some(Arc::new(WhisperGgmlExecutor::default())),
        QWEN3_ASR_EXECUTOR_COMPONENT_ID => Some(Arc::new(Qwen3AsrGgmlExecutor::default())),
        PARAKEET_CTC_EXECUTOR_COMPONENT_ID => Some(Arc::new(ParakeetCtcGgmlExecutor)),
        WAV2VEC2_CTC_EXECUTOR_COMPONENT_ID => Some(Arc::new(Wav2Vec2CtcGgmlExecutor)),
        MOONSHINE_EXECUTOR_COMPONENT_ID => Some(Arc::new(MoonshineGgmlExecutor::default())),
        XASR_ZIPFORMER_EXECUTOR_COMPONENT_ID => Some(Arc::new(XasrZipformerGgmlExecutor)),
        DOLPHIN_EXECUTOR_COMPONENT_ID => Some(Arc::new(DolphinGgmlExecutor)),
        SENSEVOICE_EXECUTOR_COMPONENT_ID => Some(Arc::new(SenseVoiceGgmlExecutor)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn materializes_builtin_executors_for_known_architectures() {
        let executors =
            materialize_builtin_executors_by_model_architecture().expect("executor map");
        let whisper = executors
            .get(crate::WHISPER_GGML_ARCHITECTURE_ID)
            .expect("whisper executor");
        let cohere = executors
            .get(crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID)
            .expect("cohere executor");
        let qwen = executors
            .get(crate::QWEN3_ASR_GGML_ARCHITECTURE_ID)
            .expect("qwen executor");

        assert_eq!(whisper.executor_id(), "whisper-ggml-executor-v1");
        assert_eq!(cohere.executor_id(), "cohere-transcribe-ggml-executor-v1");
        assert_eq!(qwen.executor_id(), "qwen3-asr-ggml-executor-v1");
    }

    #[test]
    fn materializes_builtin_executor_map_for_all_known_architectures() {
        let executors =
            materialize_builtin_executors_by_model_architecture().expect("executor map");

        for (architecture, label) in [
            (crate::WHISPER_GGML_ARCHITECTURE_ID, "whisper"),
            (
                crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                "cohere-transcribe",
            ),
            (
                crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
                crate::QWEN3_ASR_MODEL_FAMILY,
            ),
            (
                crate::arch::PARAKEET_CTC_GGML_ARCHITECTURE_ID,
                "parakeet-ctc",
            ),
            (
                crate::arch::WAV2VEC2_CTC_GGML_ARCHITECTURE_ID,
                "wav2vec2-ctc",
            ),
            (crate::arch::MOONSHINE_GGML_ARCHITECTURE_ID, "moonshine"),
            (
                crate::arch::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
                crate::arch::XASR_ZIPFORMER_MODEL_FAMILY,
            ),
        ] {
            let executor = executors.get(architecture).unwrap_or_else(|| {
                panic!("{label} executor should be materialized for {architecture}")
            });
            assert!(
                !executor.executor_id().is_empty(),
                "{label} executor id should be non-empty"
            );
        }
    }

    #[test]
    fn builtin_executor_phrase_bias_expectations_cover_all_builtins() {
        let expected_by_family = BTreeMap::from([
            ("cohere-transcribe", true),
            ("whisper", true),
            (crate::QWEN3_ASR_MODEL_FAMILY, true),
            ("parakeet-ctc", true),
            ("wav2vec2-ctc", true),
            (crate::MOONSHINE_MODEL_FAMILY, true),
            (crate::arch::XASR_ZIPFORMER_MODEL_FAMILY, false),
            (crate::arch::DOLPHIN_MODEL_FAMILY, true),
            (crate::arch::SENSEVOICE_MODEL_FAMILY, true),
        ]);
        let executors =
            materialize_builtin_executors_by_model_architecture().expect("executor map");
        let mut seen_families = std::collections::BTreeSet::new();

        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            let expected = expected_by_family
                .get(descriptor.model_family)
                .copied()
                .unwrap_or_else(|| {
                    panic!(
                        "missing phrase-bias expectation for builtin family '{}'",
                        descriptor.model_family
                    )
                });
            let executor = executors
                .get(descriptor.model_architecture)
                .unwrap_or_else(|| {
                    panic!(
                        "missing materialized executor for builtin architecture '{}'",
                        descriptor.model_architecture
                    )
                });
            seen_families.insert(descriptor.model_family);

            assert_eq!(
                executor.supports_phrase_bias(),
                expected,
                "builtin family '{}' ({}) phrase-bias capability must come from its executor",
                descriptor.model_family,
                descriptor.model_architecture
            );
            assert_eq!(
                builtin_executor_supports_phrase_bias_for_model_architecture(
                    descriptor.model_architecture
                ),
                Some(expected),
                "builtin family '{}' ({}) registry lookup must report the explicit expectation",
                descriptor.model_family,
                descriptor.model_architecture
            );
        }

        let expected_families = expected_by_family.keys().copied().collect();
        assert_eq!(
            seen_families, expected_families,
            "phrase-bias expectation table must not contain stale builtin families"
        );
    }
}
