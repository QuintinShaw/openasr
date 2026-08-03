use std::{collections::BTreeMap, sync::Arc};

use thiserror::Error;

use crate::arch::{
    COHERE_TRANSCRIBE_EXECUTOR_COMPONENT_ID, DOLPHIN_EXECUTOR_COMPONENT_ID,
    FIRERED_AED_EXECUTOR_COMPONENT_ID, FIRERED_LLM_EXECUTOR_COMPONENT_ID,
    FUNASR_NANO_EXECUTOR_COMPONENT_ID, GRANITE_SPEECH_EXECUTOR_COMPONENT_ID,
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
use super::funasr_nano::executor::FunasrNanoGgmlExecutor;
use super::ggml_asr_executor::GgmlAsrViewExecutor;
use super::granite_speech::executor::GraniteSpeechGgmlExecutor;
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
    #[error(
        "builtin executor decoder-state contract failed for architecture '{model_architecture}': {reason}"
    )]
    DecoderStateContractFailed {
        model_architecture: String,
        reason: String,
    },
    #[error(
        "builtin executor decoder-state topology mismatch for architecture '{model_architecture}': declared={declared:?}, actual={actual:?}"
    )]
    DecoderStateTopologyMismatch {
        model_architecture: String,
        declared: crate::arch::OpenAsrDecoderStateTopology,
        actual: crate::arch::OpenAsrDecoderStateTopology,
    },
}

pub(crate) fn materialize_builtin_executors_by_model_architecture(
    stateful_executors: &BuiltinStatefulExecutorScope,
) -> Result<
    BTreeMap<&'static str, Arc<dyn GgmlAsrViewExecutor>>,
    BuiltinExecutorComponentRegistryError,
> {
    let mut executors_by_model_architecture = BTreeMap::new();

    for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
        let executor = materialize_builtin_executor_component(
            stateful_executors,
            descriptor.executor_component_id,
        )
        .ok_or_else(|| {
            BuiltinExecutorComponentRegistryError::UnknownExecutorComponent {
                model_architecture: descriptor.model_architecture.to_string(),
                executor_component_id: descriptor.executor_component_id.to_string(),
            }
        })?;
        let family_descriptor = descriptor.ggml_family_adapter_descriptor();
        let actual_state_topology = executor
            .decoder_state_contract(&family_descriptor)
            .map(decoder_state_topology)
            .map_err(
                |error| BuiltinExecutorComponentRegistryError::DecoderStateContractFailed {
                    model_architecture: descriptor.model_architecture.to_string(),
                    reason: error.to_string(),
                },
            )?;
        if actual_state_topology != descriptor.decoder_state_topology {
            return Err(
                BuiltinExecutorComponentRegistryError::DecoderStateTopologyMismatch {
                    model_architecture: descriptor.model_architecture.to_string(),
                    declared: descriptor.decoder_state_topology,
                    actual: actual_state_topology,
                },
            );
        }
        executors_by_model_architecture.insert(descriptor.model_architecture, executor);
    }

    Ok(executors_by_model_architecture)
}

fn decoder_state_topology(
    contract: super::ggml_asr_executor::GgmlAsrDecoderStateContract,
) -> crate::arch::OpenAsrDecoderStateTopology {
    use crate::arch::OpenAsrDecoderStateTopology;
    use crate::capacity::topology::StateKind;

    match contract {
        super::ggml_asr_executor::GgmlAsrDecoderStateContract::NoPersistentState => {
            OpenAsrDecoderStateTopology::None
        }
        super::ggml_asr_executor::GgmlAsrDecoderStateContract::Planned {
            streams:
                [
                    super::ggml_asr_executor::GgmlAsrDecoderStateStreamContract {
                        kind: StateKind::SelfAttentionKv,
                        ..
                    },
                ],
            ..
        } => OpenAsrDecoderStateTopology::CausalSelfAttentionKv,
        super::ggml_asr_executor::GgmlAsrDecoderStateContract::Planned { streams, .. }
            if streams.len() == 2
                && streams
                    .iter()
                    .any(|stream| stream.kind == StateKind::SelfAttentionKv)
                && streams
                    .iter()
                    .any(|stream| stream.kind == StateKind::CrossAttentionKv) =>
        {
            OpenAsrDecoderStateTopology::EncoderDecoderSelfAndCrossAttentionKv
        }
        super::ggml_asr_executor::GgmlAsrDecoderStateContract::Planned { .. } => {
            OpenAsrDecoderStateTopology::FamilyDefinedTokenScaledPersistent
        }
    }
}

/// Phrase-bias capability for a builtin architecture.
///
/// Authoritative source is the architecture integration descriptor
/// (`descriptor.integration.supports_phrase_bias`). Executor trait methods remain
/// implementation details and are audited against this value so a hand-edited
/// executor cannot silently disagree with the product capability surface.
pub(crate) fn builtin_executor_supports_phrase_bias_for_model_architecture(
    model_architecture: &str,
) -> Option<bool> {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(model_architecture)
        .map(|descriptor| descriptor.integration.supports_phrase_bias)
}

fn materialize_builtin_executor_component(
    stateful_executors: &BuiltinStatefulExecutorScope,
    executor_component_id: &str,
) -> Option<Arc<dyn GgmlAsrViewExecutor>> {
    match executor_component_id {
        COHERE_TRANSCRIBE_EXECUTOR_COMPONENT_ID => {
            Some(stateful_executors.cohere_transcribe() as Arc<dyn GgmlAsrViewExecutor>)
        }
        WHISPER_EXECUTOR_COMPONENT_ID => {
            Some(stateful_executors.whisper() as Arc<dyn GgmlAsrViewExecutor>)
        }
        QWEN3_ASR_EXECUTOR_COMPONENT_ID => {
            Some(stateful_executors.qwen3_asr() as Arc<dyn GgmlAsrViewExecutor>)
        }
        PARAKEET_CTC_EXECUTOR_COMPONENT_ID => {
            Some(stateful_executors.parakeet_ctc() as Arc<dyn GgmlAsrViewExecutor>)
        }
        PARAKEET_TDT_EXECUTOR_COMPONENT_ID => {
            Some(stateful_executors.parakeet_tdt() as Arc<dyn GgmlAsrViewExecutor>)
        }
        WAV2VEC2_CTC_EXECUTOR_COMPONENT_ID => {
            Some(stateful_executors.wav2vec2_ctc() as Arc<dyn GgmlAsrViewExecutor>)
        }
        MOONSHINE_EXECUTOR_COMPONENT_ID => {
            Some(stateful_executors.moonshine() as Arc<dyn GgmlAsrViewExecutor>)
        }
        XASR_ZIPFORMER_EXECUTOR_COMPONENT_ID => {
            Some(stateful_executors.xasr_zipformer() as Arc<dyn GgmlAsrViewExecutor>)
        }
        DOLPHIN_EXECUTOR_COMPONENT_ID => {
            Some(stateful_executors.dolphin() as Arc<dyn GgmlAsrViewExecutor>)
        }
        SENSEVOICE_EXECUTOR_COMPONENT_ID => {
            Some(stateful_executors.sensevoice() as Arc<dyn GgmlAsrViewExecutor>)
        }
        FIRERED_AED_EXECUTOR_COMPONENT_ID => {
            Some(stateful_executors.firered_aed() as Arc<dyn GgmlAsrViewExecutor>)
        }
        FIRERED_LLM_EXECUTOR_COMPONENT_ID => {
            Some(stateful_executors.firered_llm() as Arc<dyn GgmlAsrViewExecutor>)
        }
        FUNASR_NANO_EXECUTOR_COMPONENT_ID => {
            Some(stateful_executors.funasr_nano() as Arc<dyn GgmlAsrViewExecutor>)
        }
        MIMO_ASR_EXECUTOR_COMPONENT_ID => {
            Some(stateful_executors.mimo_asr() as Arc<dyn GgmlAsrViewExecutor>)
        }
        MOSS_TD_EXECUTOR_COMPONENT_ID => {
            Some(stateful_executors.moss_td() as Arc<dyn GgmlAsrViewExecutor>)
        }
        GRANITE_SPEECH_EXECUTOR_COMPONENT_ID => {
            Some(stateful_executors.granite_speech() as Arc<dyn GgmlAsrViewExecutor>)
        }
        _ => None,
    }
}

/// Stateful builtin executors owned by one [`NativeExecutionServices`](crate::NativeExecutionServices)
/// root. Offline and streaming dispatches built for that root receive clones
/// of these same allocations, while independently constructed service roots
/// never share cached prepared weights.
pub(crate) struct BuiltinStatefulExecutorScope {
    qwen3_asr: Arc<Qwen3AsrGgmlExecutor>,
    cohere_transcribe: Arc<CohereTranscribeGgmlExecutor>,
    whisper: Arc<WhisperGgmlExecutor>,
    moonshine: Arc<MoonshineGgmlExecutor>,
    xasr_zipformer: Arc<XasrZipformerGgmlExecutor>,
    parakeet_ctc: Arc<ParakeetCtcGgmlExecutor>,
    parakeet_tdt: Arc<ParakeetTdtGgmlExecutor>,
    firered_aed: Arc<FireRedAedGgmlExecutor>,
    wav2vec2_ctc: Arc<Wav2Vec2CtcGgmlExecutor>,
    dolphin: Arc<DolphinGgmlExecutor>,
    sensevoice: Arc<SenseVoiceGgmlExecutor>,
    firered_llm: Arc<FireRedLlmGgmlExecutor>,
    funasr_nano: Arc<FunasrNanoGgmlExecutor>,
    mimo_asr: Arc<MimoAsrGgmlExecutor>,
    moss_td: Arc<MossTdGgmlExecutor>,
    granite_speech: Arc<GraniteSpeechGgmlExecutor>,
}

impl BuiltinStatefulExecutorScope {
    pub(crate) fn new() -> Self {
        Self {
            qwen3_asr: Arc::new(Qwen3AsrGgmlExecutor::default()),
            cohere_transcribe: Arc::new(CohereTranscribeGgmlExecutor::default()),
            whisper: Arc::new(WhisperGgmlExecutor::default()),
            moonshine: Arc::new(MoonshineGgmlExecutor::default()),
            xasr_zipformer: Arc::new(XasrZipformerGgmlExecutor::default()),
            parakeet_ctc: Arc::new(ParakeetCtcGgmlExecutor::default()),
            parakeet_tdt: Arc::new(ParakeetTdtGgmlExecutor::default()),
            firered_aed: Arc::new(FireRedAedGgmlExecutor::default()),
            wav2vec2_ctc: Arc::new(Wav2Vec2CtcGgmlExecutor::default()),
            dolphin: Arc::new(DolphinGgmlExecutor::default()),
            sensevoice: Arc::new(SenseVoiceGgmlExecutor::default()),
            firered_llm: Arc::new(FireRedLlmGgmlExecutor::default()),
            funasr_nano: Arc::new(FunasrNanoGgmlExecutor::default()),
            mimo_asr: Arc::new(MimoAsrGgmlExecutor::default()),
            moss_td: Arc::new(MossTdGgmlExecutor::default()),
            granite_speech: Arc::new(GraniteSpeechGgmlExecutor::default()),
        }
    }

    pub(crate) fn qwen3_asr(&self) -> Arc<Qwen3AsrGgmlExecutor> {
        Arc::clone(&self.qwen3_asr)
    }

    pub(crate) fn cohere_transcribe(&self) -> Arc<CohereTranscribeGgmlExecutor> {
        Arc::clone(&self.cohere_transcribe)
    }

    pub(crate) fn whisper(&self) -> Arc<WhisperGgmlExecutor> {
        Arc::clone(&self.whisper)
    }

    pub(crate) fn moonshine(&self) -> Arc<MoonshineGgmlExecutor> {
        Arc::clone(&self.moonshine)
    }

    pub(crate) fn xasr_zipformer(&self) -> Arc<XasrZipformerGgmlExecutor> {
        Arc::clone(&self.xasr_zipformer)
    }

    pub(crate) fn parakeet_ctc(&self) -> Arc<ParakeetCtcGgmlExecutor> {
        Arc::clone(&self.parakeet_ctc)
    }

    pub(crate) fn parakeet_tdt(&self) -> Arc<ParakeetTdtGgmlExecutor> {
        Arc::clone(&self.parakeet_tdt)
    }

    pub(crate) fn firered_aed(&self) -> Arc<FireRedAedGgmlExecutor> {
        Arc::clone(&self.firered_aed)
    }

    pub(crate) fn wav2vec2_ctc(&self) -> Arc<Wav2Vec2CtcGgmlExecutor> {
        Arc::clone(&self.wav2vec2_ctc)
    }

    pub(crate) fn dolphin(&self) -> Arc<DolphinGgmlExecutor> {
        Arc::clone(&self.dolphin)
    }

    pub(crate) fn sensevoice(&self) -> Arc<SenseVoiceGgmlExecutor> {
        Arc::clone(&self.sensevoice)
    }

    pub(crate) fn firered_llm(&self) -> Arc<FireRedLlmGgmlExecutor> {
        Arc::clone(&self.firered_llm)
    }

    pub(crate) fn funasr_nano(&self) -> Arc<FunasrNanoGgmlExecutor> {
        Arc::clone(&self.funasr_nano)
    }

    pub(crate) fn mimo_asr(&self) -> Arc<MimoAsrGgmlExecutor> {
        Arc::clone(&self.mimo_asr)
    }

    pub(crate) fn moss_td(&self) -> Arc<MossTdGgmlExecutor> {
        Arc::clone(&self.moss_td)
    }

    pub(crate) fn granite_speech(&self) -> Arc<GraniteSpeechGgmlExecutor> {
        Arc::clone(&self.granite_speech)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn materialize_test_executors() -> Result<
        BTreeMap<&'static str, Arc<dyn GgmlAsrViewExecutor>>,
        BuiltinExecutorComponentRegistryError,
    > {
        materialize_builtin_executors_by_model_architecture(&BuiltinStatefulExecutorScope::new())
    }

    #[test]
    fn materializes_builtin_executors_for_known_architectures() {
        let executors = materialize_test_executors().expect("executor map");
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
        let executors = materialize_test_executors().expect("executor map");

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
            (crate::arch::DOLPHIN_GGML_ARCHITECTURE_ID, "dolphin"),
            (crate::arch::SENSEVOICE_GGML_ARCHITECTURE_ID, "sensevoice"),
            (crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID, "firered-aed"),
            (crate::arch::FIRERED_LLM_GGML_ARCHITECTURE_ID, "firered-llm"),
            (crate::arch::FUNASR_NANO_GGML_ARCHITECTURE_ID, "funasr-nano"),
            (crate::arch::MIMO_ASR_GGML_ARCHITECTURE_ID, "mimo-asr"),
            (
                crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
                "moss-transcribe-diarize",
            ),
            (
                crate::arch::GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
                "granite-speech",
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
    fn every_builtin_executor_declares_its_decoder_state_topology() {
        use crate::arch::OpenAsrDecoderStateTopology;
        use crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract;

        let executors = materialize_test_executors().expect("executor map");
        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            let executor = executors
                .get(descriptor.model_architecture)
                .unwrap_or_else(|| {
                    panic!("missing executor for {}", descriptor.model_architecture)
                });
            let family_descriptor = descriptor.ggml_family_adapter_descriptor();
            let contract = executor
                .decoder_state_contract(&family_descriptor)
                .unwrap_or_else(|error| {
                    panic!("{} contract failed: {error}", descriptor.model_family)
                });
            let actual_topology = match contract {
                GgmlAsrDecoderStateContract::NoPersistentState => OpenAsrDecoderStateTopology::None,
                GgmlAsrDecoderStateContract::Planned { planner, streams } => {
                    assert_ne!(
                        planner as usize, 0,
                        "{} returned a null decoder-state planner",
                        descriptor.model_architecture
                    );
                    assert!(
                        !streams.is_empty(),
                        "{} returned an empty decoder-state stream contract",
                        descriptor.model_architecture
                    );
                    decoder_state_topology(GgmlAsrDecoderStateContract::Planned {
                        planner,
                        streams,
                    })
                }
            };
            assert_eq!(
                actual_topology, descriptor.decoder_state_topology,
                "family '{}' ({}) executor decoder-state contract disagrees with its mandatory architecture declaration",
                descriptor.model_family, descriptor.model_architecture
            );
        }
    }

    #[test]
    fn builtin_executor_phrase_bias_matches_architecture_integration_descriptor() {
        let executors = materialize_test_executors().expect("executor map");

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
                "family '{}' ({}) executor phrase-bias implementation disagrees with its architecture integration descriptor",
                descriptor.model_family,
                descriptor.model_architecture
            );
            assert_eq!(
                builtin_executor_supports_phrase_bias_for_model_architecture(
                    descriptor.model_architecture
                ),
                Some(descriptor.integration.supports_phrase_bias),
                "family '{}' ({}) capability lookup must derive from the architecture integration descriptor",
                descriptor.model_family,
                descriptor.model_architecture
            );
        }
    }

    #[test]
    fn stateful_family_executors_are_reused_within_one_scope_and_isolated_between_scopes() {
        let first = BuiltinStatefulExecutorScope::new();
        let second = BuiltinStatefulExecutorScope::new();
        assert!(Arc::ptr_eq(&first.qwen3_asr(), &first.qwen3_asr()));
        assert!(Arc::ptr_eq(
            &first.cohere_transcribe(),
            &first.cohere_transcribe()
        ));
        assert!(Arc::ptr_eq(&first.whisper(), &first.whisper()));
        assert!(Arc::ptr_eq(&first.moonshine(), &first.moonshine()));
        assert!(Arc::ptr_eq(&first.parakeet_ctc(), &first.parakeet_ctc()));
        assert!(Arc::ptr_eq(&first.parakeet_tdt(), &first.parakeet_tdt()));
        assert!(Arc::ptr_eq(&first.firered_aed(), &first.firered_aed()));
        assert!(Arc::ptr_eq(&first.wav2vec2_ctc(), &first.wav2vec2_ctc()));
        assert!(Arc::ptr_eq(&first.dolphin(), &first.dolphin()));
        assert!(Arc::ptr_eq(&first.sensevoice(), &first.sensevoice()));
        assert!(Arc::ptr_eq(&first.firered_llm(), &first.firered_llm()));
        assert!(Arc::ptr_eq(&first.funasr_nano(), &first.funasr_nano()));
        assert!(Arc::ptr_eq(&first.mimo_asr(), &first.mimo_asr()));
        assert!(Arc::ptr_eq(&first.moss_td(), &first.moss_td()));
        assert!(Arc::ptr_eq(
            &first.granite_speech(),
            &first.granite_speech()
        ));
        assert!(!Arc::ptr_eq(&first.qwen3_asr(), &second.qwen3_asr()));
        assert!(!Arc::ptr_eq(
            &first.cohere_transcribe(),
            &second.cohere_transcribe()
        ));
        assert!(!Arc::ptr_eq(&first.parakeet_ctc(), &second.parakeet_ctc()));
        assert!(!Arc::ptr_eq(&first.parakeet_tdt(), &second.parakeet_tdt()));
        assert!(!Arc::ptr_eq(&first.firered_aed(), &second.firered_aed()));
        assert!(!Arc::ptr_eq(&first.wav2vec2_ctc(), &second.wav2vec2_ctc()));
        assert!(!Arc::ptr_eq(&first.dolphin(), &second.dolphin()));
        assert!(!Arc::ptr_eq(&first.sensevoice(), &second.sensevoice()));
        assert!(!Arc::ptr_eq(&first.firered_llm(), &second.firered_llm()));
        assert!(!Arc::ptr_eq(&first.funasr_nano(), &second.funasr_nano()));
        assert!(!Arc::ptr_eq(&first.mimo_asr(), &second.mimo_asr()));
        assert!(!Arc::ptr_eq(&first.moss_td(), &second.moss_td()));
        assert!(!Arc::ptr_eq(
            &first.granite_speech(),
            &second.granite_speech()
        ));
    }

    #[test]
    fn offline_executor_map_registers_the_same_instance_as_its_service_scope() {
        // `materialize_builtin_executor_component` must route stateful
        // architectures through the supplied service scope, not through a
        // fresh `Default::default()` or an ambient process singleton.
        let scope = BuiltinStatefulExecutorScope::new();
        let executors =
            materialize_builtin_executors_by_model_architecture(&scope).expect("executor map");

        let qwen_offline = executors
            .get(crate::QWEN3_ASR_GGML_ARCHITECTURE_ID)
            .expect("qwen executor");
        let qwen_shared: Arc<dyn GgmlAsrViewExecutor> = scope.qwen3_asr();
        assert!(Arc::ptr_eq(qwen_offline, &qwen_shared));

        let cohere_offline = executors
            .get(crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID)
            .expect("cohere executor");
        let cohere_shared: Arc<dyn GgmlAsrViewExecutor> = scope.cohere_transcribe();
        assert!(Arc::ptr_eq(cohere_offline, &cohere_shared));

        let whisper_offline = executors
            .get(crate::WHISPER_GGML_ARCHITECTURE_ID)
            .expect("whisper executor");
        let whisper_shared: Arc<dyn GgmlAsrViewExecutor> = scope.whisper();
        assert!(Arc::ptr_eq(whisper_offline, &whisper_shared));

        let moonshine_offline = executors
            .get(crate::arch::MOONSHINE_GGML_ARCHITECTURE_ID)
            .expect("moonshine executor");
        let moonshine_shared: Arc<dyn GgmlAsrViewExecutor> = scope.moonshine();
        assert!(Arc::ptr_eq(moonshine_offline, &moonshine_shared));

        let parakeet_ctc_offline = executors
            .get(crate::arch::PARAKEET_CTC_GGML_ARCHITECTURE_ID)
            .expect("parakeet-ctc executor");
        let parakeet_ctc_shared: Arc<dyn GgmlAsrViewExecutor> = scope.parakeet_ctc();
        assert!(Arc::ptr_eq(parakeet_ctc_offline, &parakeet_ctc_shared));

        let parakeet_tdt_offline = executors
            .get(crate::arch::PARAKEET_TDT_GGML_ARCHITECTURE_ID)
            .expect("parakeet-tdt executor");
        let parakeet_tdt_shared: Arc<dyn GgmlAsrViewExecutor> = scope.parakeet_tdt();
        assert!(Arc::ptr_eq(parakeet_tdt_offline, &parakeet_tdt_shared));

        let firered_aed_offline = executors
            .get(crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID)
            .expect("firered-aed executor");
        let firered_aed_shared: Arc<dyn GgmlAsrViewExecutor> = scope.firered_aed();
        assert!(Arc::ptr_eq(firered_aed_offline, &firered_aed_shared));

        let wav2vec2_ctc_offline = executors
            .get(crate::arch::WAV2VEC2_CTC_GGML_ARCHITECTURE_ID)
            .expect("wav2vec2-ctc executor");
        let wav2vec2_ctc_shared: Arc<dyn GgmlAsrViewExecutor> = scope.wav2vec2_ctc();
        assert!(Arc::ptr_eq(wav2vec2_ctc_offline, &wav2vec2_ctc_shared));

        let dolphin_offline = executors
            .get(crate::arch::DOLPHIN_GGML_ARCHITECTURE_ID)
            .expect("dolphin executor");
        let dolphin_shared: Arc<dyn GgmlAsrViewExecutor> = scope.dolphin();
        assert!(Arc::ptr_eq(dolphin_offline, &dolphin_shared));

        let sensevoice_offline = executors
            .get(crate::arch::SENSEVOICE_GGML_ARCHITECTURE_ID)
            .expect("sensevoice executor");
        let sensevoice_shared: Arc<dyn GgmlAsrViewExecutor> = scope.sensevoice();
        assert!(Arc::ptr_eq(sensevoice_offline, &sensevoice_shared));

        let firered_llm_offline = executors
            .get(crate::arch::FIRERED_LLM_GGML_ARCHITECTURE_ID)
            .expect("firered-llm executor");
        let firered_llm_shared: Arc<dyn GgmlAsrViewExecutor> = scope.firered_llm();
        assert!(Arc::ptr_eq(firered_llm_offline, &firered_llm_shared));

        let funasr_nano_offline = executors
            .get(crate::arch::FUNASR_NANO_GGML_ARCHITECTURE_ID)
            .expect("funasr-nano executor");
        let funasr_nano_shared: Arc<dyn GgmlAsrViewExecutor> = scope.funasr_nano();
        assert!(Arc::ptr_eq(funasr_nano_offline, &funasr_nano_shared));

        let mimo_asr_offline = executors
            .get(crate::arch::MIMO_ASR_GGML_ARCHITECTURE_ID)
            .expect("mimo-asr executor");
        let mimo_asr_shared: Arc<dyn GgmlAsrViewExecutor> = scope.mimo_asr();
        assert!(Arc::ptr_eq(mimo_asr_offline, &mimo_asr_shared));

        let moss_td_offline = executors
            .get(crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID)
            .expect("moss-td executor");
        let moss_td_shared: Arc<dyn GgmlAsrViewExecutor> = scope.moss_td();
        assert!(Arc::ptr_eq(moss_td_offline, &moss_td_shared));

        let granite_speech_offline = executors
            .get(crate::arch::GRANITE_SPEECH_GGML_ARCHITECTURE_ID)
            .expect("granite-speech executor");
        let granite_speech_shared: Arc<dyn GgmlAsrViewExecutor> = scope.granite_speech();
        assert!(Arc::ptr_eq(granite_speech_offline, &granite_speech_shared));
    }
}
