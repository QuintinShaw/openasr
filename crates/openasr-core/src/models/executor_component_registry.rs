use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};

use thiserror::Error;

use crate::arch::{
    COHERE_TRANSCRIBE_EXECUTOR_COMPONENT_ID, DOLPHIN_EXECUTOR_COMPONENT_ID,
    FIRERED_AED_EXECUTOR_COMPONENT_ID, FIRERED_LLM_EXECUTOR_COMPONENT_ID,
    MIMO_ASR_EXECUTOR_COMPONENT_ID, MOONSHINE_EXECUTOR_COMPONENT_ID, MOSS_TD_EXECUTOR_COMPONENT_ID,
    OpenAsrArchitectureRegistry, PARAKEET_CTC_EXECUTOR_COMPONENT_ID,
    PARAKEET_TDT_EXECUTOR_COMPONENT_ID, QWEN3_ASR_EXECUTOR_COMPONENT_ID,
    SENSEVOICE_EXECUTOR_COMPONENT_ID, WAV2VEC2_CTC_EXECUTOR_COMPONENT_ID,
    WHISPER_EXECUTOR_COMPONENT_ID, XASR_ZIPFORMER_EXECUTOR_COMPONENT_ID,
};

use super::cohere::CohereTranscribeGgmlExecutor;
use super::dolphin::executor::DolphinGgmlExecutor;
use super::firered_aed::executor::FireRedAedGgmlExecutor;
use super::firered_llm::executor::FireRedLlmGgmlExecutor;
use super::ggml_asr_executor::GgmlAsrExecutor;
use super::mimo_asr::executor::MimoAsrGgmlExecutor;
use super::moonshine::MoonshineGgmlExecutor;
use super::moss_transcribe_diarize::executor::MossTdGgmlExecutor;
use super::parakeet_ctc::executor::ParakeetCtcGgmlExecutor;
use super::parakeet_tdt::executor::ParakeetTdtGgmlExecutor;
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
            Some(shared_cohere_transcribe_executor() as Arc<dyn GgmlAsrExecutor>)
        }
        WHISPER_EXECUTOR_COMPONENT_ID => {
            Some(shared_whisper_executor() as Arc<dyn GgmlAsrExecutor>)
        }
        QWEN3_ASR_EXECUTOR_COMPONENT_ID => {
            Some(shared_qwen3_asr_executor() as Arc<dyn GgmlAsrExecutor>)
        }
        PARAKEET_CTC_EXECUTOR_COMPONENT_ID => Some(Arc::new(ParakeetCtcGgmlExecutor)),
        PARAKEET_TDT_EXECUTOR_COMPONENT_ID => Some(Arc::new(ParakeetTdtGgmlExecutor)),
        WAV2VEC2_CTC_EXECUTOR_COMPONENT_ID => Some(Arc::new(Wav2Vec2CtcGgmlExecutor)),
        MOONSHINE_EXECUTOR_COMPONENT_ID => {
            Some(shared_moonshine_executor() as Arc<dyn GgmlAsrExecutor>)
        }
        XASR_ZIPFORMER_EXECUTOR_COMPONENT_ID => Some(Arc::new(XasrZipformerGgmlExecutor)),
        DOLPHIN_EXECUTOR_COMPONENT_ID => Some(Arc::new(DolphinGgmlExecutor)),
        SENSEVOICE_EXECUTOR_COMPONENT_ID => Some(Arc::new(SenseVoiceGgmlExecutor)),
        FIRERED_AED_EXECUTOR_COMPONENT_ID => Some(Arc::new(FireRedAedGgmlExecutor)),
        FIRERED_LLM_EXECUTOR_COMPONENT_ID => Some(Arc::new(FireRedLlmGgmlExecutor)),
        MIMO_ASR_EXECUTOR_COMPONENT_ID => Some(Arc::new(MimoAsrGgmlExecutor)),
        MOSS_TD_EXECUTOR_COMPONENT_ID => Some(Arc::new(MossTdGgmlExecutor)),
        _ => None,
    }
}

// Process-wide single instances for the "has-state, host-materializes-weights"
// families (qwen / cohere / whisper / moonshine). Each family's builtin
// executor struct implements *both* `GgmlAsrExecutor` (offline dispatch) and
// `GgmlAsrStreamingExecutor` (streaming dispatch) already; historically the two
// dispatch builders each called `<Family>Executor::default()` independently,
// so the offline and streaming stacks held two separate instances with two
// separate `runtime_cache_by_path` caches -- meaning a model warmed on both
// stacks paid for its host-materialized prepared runtime (weights) twice.
//
// These accessors hand out one `Arc<ConcreteExecutor>` per family per process,
// unsized-coerced independently into each dispatch's own trait-object map
// (`Arc<dyn GgmlAsrExecutor>` here, `Arc<dyn GgmlAsrStreamingExecutor>` at the
// streaming registration site in `builtin_execution_dispatch.rs`). Both
// coercions point at the same heap allocation, so both dispatches share the
// same `runtime_cache_by_path`: a cold load on either stack populates the one
// cache the other stack will also hit, and `unload_idle_state()` called from
// either dispatch's `unload_all()` clears the same cache (the other stack's
// subsequent `unload_all()` call just clears an already-empty cache, a no-op).
static SHARED_QWEN3_ASR_EXECUTOR: OnceLock<Arc<Qwen3AsrGgmlExecutor>> = OnceLock::new();
static SHARED_COHERE_TRANSCRIBE_EXECUTOR: OnceLock<Arc<CohereTranscribeGgmlExecutor>> =
    OnceLock::new();
static SHARED_WHISPER_EXECUTOR: OnceLock<Arc<WhisperGgmlExecutor>> = OnceLock::new();
static SHARED_MOONSHINE_EXECUTOR: OnceLock<Arc<MoonshineGgmlExecutor>> = OnceLock::new();

pub(crate) fn shared_qwen3_asr_executor() -> Arc<Qwen3AsrGgmlExecutor> {
    Arc::clone(SHARED_QWEN3_ASR_EXECUTOR.get_or_init(|| Arc::new(Qwen3AsrGgmlExecutor::default())))
}

pub(crate) fn shared_cohere_transcribe_executor() -> Arc<CohereTranscribeGgmlExecutor> {
    Arc::clone(
        SHARED_COHERE_TRANSCRIBE_EXECUTOR
            .get_or_init(|| Arc::new(CohereTranscribeGgmlExecutor::default())),
    )
}

pub(crate) fn shared_whisper_executor() -> Arc<WhisperGgmlExecutor> {
    Arc::clone(SHARED_WHISPER_EXECUTOR.get_or_init(|| Arc::new(WhisperGgmlExecutor::default())))
}

pub(crate) fn shared_moonshine_executor() -> Arc<MoonshineGgmlExecutor> {
    Arc::clone(SHARED_MOONSHINE_EXECUTOR.get_or_init(|| Arc::new(MoonshineGgmlExecutor::default())))
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
                crate::arch::PARAKEET_TDT_GGML_ARCHITECTURE_ID,
                "parakeet-tdt",
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
    fn builtin_executor_phrase_bias_matches_architecture_manifest() {
        let executors =
            materialize_builtin_executors_by_model_architecture().expect("executor map");

        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            let executor = executors
                .get(descriptor.model_architecture)
                .unwrap_or_else(|| {
                    panic!(
                        "missing materialized executor for builtin family '{}' ({})",
                        descriptor.model_family, descriptor.model_architecture
                    )
                });

            assert_eq!(
                executor.supports_phrase_bias(),
                descriptor.integration.supports_phrase_bias,
                "family '{}' ({}) executor phrase-bias capability disagrees with its architecture manifest",
                descriptor.model_family,
                descriptor.model_architecture
            );
            assert_eq!(
                builtin_executor_supports_phrase_bias_for_model_architecture(
                    descriptor.model_architecture
                ),
                Some(descriptor.integration.supports_phrase_bias),
                "family '{}' ({}) registry lookup must expose the manifest capability",
                descriptor.model_family,
                descriptor.model_architecture
            );
        }
    }

    // Phase 1 shared-executor regression coverage: qwen / cohere / whisper /
    // moonshine each hand out one process-wide `Arc<ConcreteExecutor>` so the
    // offline and streaming dispatches -- built independently, possibly on
    // different threads -- register the *same* instance (and therefore the
    // same `runtime_cache_by_path`) instead of two instances with two
    // independent caches. These tests share process-global `OnceLock`
    // statics with every other test in this binary, so they only assert
    // "repeated calls agree with each other", never a specific identity.

    #[test]
    fn shared_stateful_family_executors_are_process_wide_singletons() {
        assert!(Arc::ptr_eq(
            &shared_qwen3_asr_executor(),
            &shared_qwen3_asr_executor()
        ));
        assert!(Arc::ptr_eq(
            &shared_cohere_transcribe_executor(),
            &shared_cohere_transcribe_executor()
        ));
        assert!(Arc::ptr_eq(
            &shared_whisper_executor(),
            &shared_whisper_executor()
        ));
        assert!(Arc::ptr_eq(
            &shared_moonshine_executor(),
            &shared_moonshine_executor()
        ));
    }

    #[test]
    fn offline_executor_map_registers_the_same_instance_as_the_shared_accessor() {
        // `materialize_builtin_executor_component` must route these four
        // architectures through the shared accessors, not through a fresh
        // `Default::default()`. If a future edit reverts one of them to a
        // bare `Arc::new(...)`, this test catches it: the offline map's Arc
        // would stop being pointer-equal to the shared singleton the
        // streaming dispatch also draws from, silently reintroducing a
        // second resident copy of that family's host-materialized weights.
        let executors =
            materialize_builtin_executors_by_model_architecture().expect("executor map");

        let qwen_offline = executors
            .get(crate::QWEN3_ASR_GGML_ARCHITECTURE_ID)
            .expect("qwen executor");
        let qwen_shared: Arc<dyn GgmlAsrExecutor> = shared_qwen3_asr_executor();
        assert!(Arc::ptr_eq(qwen_offline, &qwen_shared));

        let cohere_offline = executors
            .get(crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID)
            .expect("cohere executor");
        let cohere_shared: Arc<dyn GgmlAsrExecutor> = shared_cohere_transcribe_executor();
        assert!(Arc::ptr_eq(cohere_offline, &cohere_shared));

        let whisper_offline = executors
            .get(crate::WHISPER_GGML_ARCHITECTURE_ID)
            .expect("whisper executor");
        let whisper_shared: Arc<dyn GgmlAsrExecutor> = shared_whisper_executor();
        assert!(Arc::ptr_eq(whisper_offline, &whisper_shared));

        let moonshine_offline = executors
            .get(crate::arch::MOONSHINE_GGML_ARCHITECTURE_ID)
            .expect("moonshine executor");
        let moonshine_shared: Arc<dyn GgmlAsrExecutor> = shared_moonshine_executor();
        assert!(Arc::ptr_eq(moonshine_offline, &moonshine_shared));
    }
}
