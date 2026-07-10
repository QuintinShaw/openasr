pub(crate) mod block_stack;
pub(crate) mod hparams;
pub(crate) mod shape_orchestrator;

use std::collections::BTreeMap;

use crate::models::ggml_family_adapter::{
    GGML_TOKENIZER_ID_KEY, GgmlExecutionCapability, GgmlFamilyAdapterDescriptor, LanguageFamilyHint,
};
use crate::models::oasr_metadata::{
    OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_DECODE_POLICY,
    OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
    OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
};
use crate::models::qwen::QWEN3_ASR_MODEL_FAMILY;
use block_stack::{
    OpenAsrBlockKind, OpenAsrBlockStackDescriptor, OpenAsrOrchestrationShape,
    OpenAsrStageDescriptor,
};
use hparams::{
    COHERE_TRANSCRIBE_DECODER_LAYERS_KEY, COHERE_TRANSCRIBE_ENCODER_LAYERS_KEY,
    COHERE_TRANSCRIBE_HPARAM_SCHEMA, DOLPHIN_HPARAM_SCHEMA, FIRERED_AED_HPARAM_SCHEMA,
    MOONSHINE_HPARAM_SCHEMA, PARAKEET_CTC_HPARAM_SCHEMA, PARAKEET_TDT_HPARAM_SCHEMA,
    QWEN3_ARCHITECTURE_VALUE, QWEN3_ASR_HPARAM_SCHEMA, QWEN3_AUDIO_LAYERS_KEY,
    QWEN3_LLM_LAYERS_KEY, SENSEVOICE_HPARAM_SCHEMA, WAV2VEC2_CTC_HPARAM_SCHEMA,
    WHISPER_HPARAM_SCHEMA, XASR_ZIPFORMER_HPARAM_SCHEMA,
};

pub(crate) const GENERAL_ARCHITECTURE_KEY: &str = "general.architecture";

pub(crate) const COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID: &str =
    "cohere-transcribe-conformer-transformer";
pub(crate) const COHERE_TRANSCRIBE_GGML_ADAPTER_ID: &str =
    "ggml-family-cohere-transcribe-runtime-v1";
pub(crate) const COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID: &str =
    "cohere-transcribe.logmel128.preemphasis.16khz.mono.v0";
pub(crate) const COHERE_TRANSCRIBE_TOKENIZER_ID: &str = "cohere-transcribe.spm.v1";
pub(crate) const COHERE_TRANSCRIBE_DECODE_POLICY_ID: &str = "cohere-transcribe.greedy.seq2seq.v1";
pub(crate) const COHERE_TRANSCRIBE_RUNTIME_TENSOR_CONTRACT_ID: &str =
    "cohere-transcribe.runtime-tensors.v1";
pub(crate) const COHERE_TRANSCRIBE_EXECUTOR_COMPONENT_ID: &str =
    "cohere-transcribe.ggml-executor.v1";

pub(crate) const WHISPER_GGML_ARCHITECTURE_ID: &str = "whisper-encoder-decoder";
pub(crate) const WHISPER_GGML_ADAPTER_ID: &str = "ggml-family-whisper-runtime-v1";
pub(crate) const WHISPER_AUDIO_FRONTEND_ID: &str = "whisper.logmel.16khz.mono.v0";
pub(crate) const WHISPER_TOKENIZER_ID: &str = "whisper.hf-bpe.v1";
pub(crate) const WHISPER_DECODE_POLICY_ID: &str = "whisper.greedy.seq2seq.v1";
pub(crate) const WHISPER_RUNTIME_TENSOR_CONTRACT_ID: &str = "whisper.runtime-tensors.v1";
pub(crate) const WHISPER_EXECUTOR_COMPONENT_ID: &str = "whisper.ggml-executor.v1";

pub(crate) const QWEN3_ASR_GGML_ARCHITECTURE_ID: &str = "qwen3-asr-encoder-decoder";
pub(crate) const QWEN3_ASR_GGML_ADAPTER_ID: &str = "ggml-family-qwen3-asr-runtime-v1";
pub(crate) const QWEN3_ASR_AUDIO_FRONTEND_ID: &str = "qwen3-asr.fbank.16khz.mono.v0";
pub(crate) const QWEN3_ASR_TOKENIZER_ID: &str = "qwen3-asr.spm.v1";
pub(crate) const QWEN3_ASR_DECODE_POLICY_ID: &str = "qwen3-asr.greedy.seq2seq.v1";
pub(crate) const QWEN3_ASR_RUNTIME_TENSOR_CONTRACT_ID: &str = "qwen3-asr.runtime-tensors.v1";
pub(crate) const QWEN3_ASR_EXECUTOR_COMPONENT_ID: &str = "qwen3-asr.ggml-executor.v1";

// parakeet-ctc (FastConformer-CTC, the goal-1 Ctc-shape onboarding).
pub(crate) const PARAKEET_CTC_GGML_ARCHITECTURE_ID: &str = "parakeet-fastconformer-ctc";
pub(crate) const PARAKEET_CTC_GGML_ADAPTER_ID: &str = "ggml-family-parakeet-ctc-runtime-v1";
pub(crate) const PARAKEET_CTC_AUDIO_FRONTEND_ID: &str = "parakeet-ctc.logmel80.16khz.mono.v0";
pub(crate) const PARAKEET_CTC_TOKENIZER_ID: &str = "parakeet-ctc.spm-bpe.v0";
pub(crate) const PARAKEET_CTC_DECODE_POLICY_ID: &str = "parakeet-ctc.greedy.ctc.v0";
pub(crate) const PARAKEET_CTC_RUNTIME_TENSOR_CONTRACT_ID: &str = "parakeet-ctc.runtime-tensors.v0";
pub(crate) const PARAKEET_CTC_EXECUTOR_COMPONENT_ID: &str = "parakeet-ctc.ggml-executor.v0";

// parakeet-tdt (FastConformer + Token-and-Duration Transducer, 25 European
// languages). Component ids are defined ahead of the full descriptor entry
// (the parakeet-ctc S2->S4 staging precedent): the importer writes them as
// pack metadata; the descriptor + executor wiring lands with the executor.
pub(crate) const PARAKEET_TDT_GGML_ARCHITECTURE_ID: &str = "parakeet-fastconformer-tdt";
pub(crate) const PARAKEET_TDT_GGML_ADAPTER_ID: &str = "ggml-family-parakeet-tdt-runtime-v1";
pub(crate) const PARAKEET_TDT_AUDIO_FRONTEND_ID: &str = "parakeet-tdt.logmel128.16khz.mono.v0";
pub(crate) const PARAKEET_TDT_TOKENIZER_ID: &str = "parakeet-tdt.spm-bpe.v0";
pub(crate) const PARAKEET_TDT_DECODE_POLICY_ID: &str = "parakeet-tdt.greedy.tdt.v0";
pub(crate) const PARAKEET_TDT_RUNTIME_TENSOR_CONTRACT_ID: &str = "parakeet-tdt.runtime-tensors.v0";
pub(crate) const PARAKEET_TDT_EXECUTOR_COMPONENT_ID: &str = "parakeet-tdt.ggml-executor.v0";

// wav2vec2-ctc (facebook/wav2vec2-base-960h, raw-waveform CTC onboarding).
pub(crate) const WAV2VEC2_CTC_GGML_ARCHITECTURE_ID: &str = "wav2vec2-ctc";
pub(crate) const WAV2VEC2_CTC_GGML_ADAPTER_ID: &str = "ggml-family-wav2vec2-ctc-runtime-v1";
pub(crate) const WAV2VEC2_CTC_AUDIO_FRONTEND_ID: &str = "wav2vec2-ctc.raw-waveform.16khz.mono.v0";
pub(crate) const WAV2VEC2_CTC_TOKENIZER_ID: &str = "wav2vec2-ctc.char.v0";
pub(crate) const WAV2VEC2_CTC_DECODE_POLICY_ID: &str = "wav2vec2-ctc.greedy.ctc.v0";
pub(crate) const WAV2VEC2_CTC_RUNTIME_TENSOR_CONTRACT_ID: &str = "wav2vec2-ctc.runtime-tensors.v0";
pub(crate) const WAV2VEC2_CTC_EXECUTOR_COMPONENT_ID: &str = "wav2vec2-ctc.ggml-executor.v0";

// X-ASR Zipformer (GilgameshWind/X-ASR-zh-en, streaming RNN-T transducer).
pub(crate) const XASR_ZIPFORMER_GGML_ARCHITECTURE_ID: &str = "xasr-zipformer-transducer";
pub(crate) const XASR_ZIPFORMER_GGML_ADAPTER_ID: &str = "ggml-family-xasr-zipformer-runtime-v1";
pub(crate) const XASR_ZIPFORMER_MODEL_FAMILY: &str = "xasr-zipformer";
pub(crate) const XASR_ZIPFORMER_AUDIO_FRONTEND_ID: &str = "xasr-zipformer.fbank80.16khz.mono.v0";
pub(crate) const XASR_ZIPFORMER_TOKENIZER_ID: &str = "xasr-zipformer.bpe.v0";
pub(crate) const XASR_ZIPFORMER_DECODE_POLICY_ID: &str = "xasr-zipformer.greedy.transducer.v0";
pub(crate) const XASR_ZIPFORMER_RUNTIME_TENSOR_CONTRACT_ID: &str =
    "xasr-zipformer.runtime-tensors.v0";
pub(crate) const XASR_ZIPFORMER_EXECUTOR_COMPONENT_ID: &str = "xasr-zipformer.ggml-executor.v0";
pub(crate) const XASR_ZIPFORMER_STREAMING_EXECUTOR_COMPONENT_ID: &str =
    "xasr-zipformer.ggml-streaming-executor.v0";

// moonshine (UsefulSensors, raw-waveform conv-stem + RoPE seq2seq encoder-decoder).
pub(crate) const MOONSHINE_GGML_ARCHITECTURE_ID: &str = "moonshine-encoder-decoder";
pub(crate) const MOONSHINE_GGML_ADAPTER_ID: &str = "ggml-family-moonshine-runtime-v1";
pub(crate) const MOONSHINE_AUDIO_FRONTEND_ID: &str = "moonshine.raw-waveform.16khz.mono.v0";
pub(crate) const MOONSHINE_TOKENIZER_ID: &str = "moonshine.spm-bpe.v0";
pub(crate) const MOONSHINE_DECODE_POLICY_ID: &str = "moonshine.greedy.seq2seq.v1";
pub(crate) const MOONSHINE_RUNTIME_TENSOR_CONTRACT_ID: &str = "moonshine.runtime-tensors.v0";
pub(crate) const MOONSHINE_EXECUTOR_COMPONENT_ID: &str = "moonshine.ggml-executor.v0";

// dolphin (WeNet E-Branchformer encoder + Transformer decoder + CTC head, char
// tokenizer, CTC/attention joint decode). Dedicated executor: the E-Branchformer
// encoder math (macaron FFN + rel-pos MHSA global branch + cgMLP/CSGU local
// branch + depthwise merge) is family-specific and not one of the composer
// block kinds, so it stays hand-written like xasr/moonshine (block_stack: None).
pub(crate) const DOLPHIN_GGML_ARCHITECTURE_ID: &str = "dolphin-ebranchformer-ctc-attention";
pub(crate) const DOLPHIN_GGML_ADAPTER_ID: &str = "ggml-family-dolphin-runtime-v1";
pub(crate) const DOLPHIN_MODEL_FAMILY: &str = "dolphin";
pub(crate) const DOLPHIN_AUDIO_FRONTEND_ID: &str = "dolphin.fbank80.16khz.mono.v0";
pub(crate) const DOLPHIN_TOKENIZER_ID: &str = "dolphin.char.v0";
pub(crate) const DOLPHIN_DECODE_POLICY_ID: &str = "dolphin.attention-rescoring.v0";
pub(crate) const DOLPHIN_RUNTIME_TENSOR_CONTRACT_ID: &str = "dolphin.runtime-tensors.v0";
pub(crate) const DOLPHIN_EXECUTOR_COMPONENT_ID: &str = "dolphin.ggml-executor.v0";

// sensevoice (FunAudioLLM/SenseVoiceSmall: SAN-M/DFSMN encoder + CTC head,
// FunASR Model License v1.1). Component ids are defined ahead of the full
// architecture-descriptor entry (the parakeet S2->S4 staging precedent): the
// importer writes them as pack metadata; the descriptor + executor wiring
// lands with the executor stage.
pub(crate) const SENSEVOICE_GGML_ARCHITECTURE_ID: &str = "sensevoice-sanm-ctc";
pub(crate) const SENSEVOICE_GGML_ADAPTER_ID: &str = "ggml-family-sensevoice-runtime-v1";
pub(crate) const SENSEVOICE_MODEL_FAMILY: &str = "sensevoice";
pub(crate) const SENSEVOICE_AUDIO_FRONTEND_ID: &str = "sensevoice.fbank80-lfr7x6.16khz.mono.v0";
pub(crate) const SENSEVOICE_TOKENIZER_ID: &str = "sensevoice.spm-bpe.v0";
pub(crate) const SENSEVOICE_DECODE_POLICY_ID: &str = "sensevoice.greedy.ctc.v0";
pub(crate) const SENSEVOICE_RUNTIME_TENSOR_CONTRACT_ID: &str = "sensevoice.runtime-tensors.v0";
pub(crate) const SENSEVOICE_EXECUTOR_COMPONENT_ID: &str = "sensevoice.ggml-executor.v0";

// firered-aed (FireRedTeam/FireRedASR-AED-L: Conformer encoder + Transformer
// decoder attention-based encoder-decoder, no CTC branch, Apache-2.0). The
// Conformer encoder math (macaron FFN + rel-pos MHSA with independent q/k/v
// LayerNorms + GLU/depthwise conv) is family-specific, so like dolphin/
// moonshine/xasr it stays on a hand-written dedicated executor
// (block_stack: None) rather than the data-driven composer.
pub(crate) const FIRERED_AED_GGML_ARCHITECTURE_ID: &str = "firered-conformer-aed";
pub(crate) const FIRERED_AED_GGML_ADAPTER_ID: &str = "ggml-family-firered-aed-runtime-v1";
pub(crate) const FIRERED_AED_MODEL_FAMILY: &str = "firered-aed";
pub(crate) const FIRERED_AED_AUDIO_FRONTEND_ID: &str = "firered-aed.fbank80.16khz.mono.v0";
pub(crate) const FIRERED_AED_TOKENIZER_ID: &str = "firered-aed.char-spm.v0";
pub(crate) const FIRERED_AED_DECODE_POLICY_ID: &str = "firered-aed.greedy.seq2seq.v0";
pub(crate) const FIRERED_AED_RUNTIME_TENSOR_CONTRACT_ID: &str = "firered-aed.runtime-tensors.v0";
pub(crate) const FIRERED_AED_EXECUTOR_COMPONENT_ID: &str = "firered-aed.ggml-executor.v0";

// hymt2 (Tencent Hunyuan-MT2 subtitle translation, hunyuan-dense decoder-only
// LLM). An auxiliary text-to-text family, NOT an ASR architecture: it is
// dispatched through `models::aux_pack_registry` / the translation routes, so
// it declares no architecture descriptor here. Its decode policy id still
// lives in this file (the single home for policy ids) and resolves directly
// through `models::decode_policy_component_registry`, keeping its greedy loop
// on the one shared decode driver.
pub(crate) const HYMT2_DECODE_POLICY_ID: &str = "hymt2.greedy.seq2seq.v0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenAsrComponentKind {
    AudioFrontend,
    DecodePolicy,
    Executor,
    RuntimeTensorContract,
    Tokenizer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OpenAsrComponentDescriptor {
    pub kind: OpenAsrComponentKind,
    pub id: &'static str,
}

/// Default `GlobalQuadratic` safe chunk-length ceiling (issue #68) -- the
/// value every new `GlobalQuadratic` builtin architecture should declare
/// unless the upstream model publishes a different explicit recommendation.
/// This is not an arbitrary number: it is where the major encoder families
/// this repo has surveyed independently converge --
///
/// - Whisper's encoder is architecture-fixed at a 30s log-mel window (see
///   `FixedWindow` below, which needs no cap at all because of this).
/// - Moonshine's model card recommends audio chunks "less than 30 seconds".
/// - NVIDIA NeMo/Parakeet's published offline/streaming guidance targets
///   20-30s chunks for FastConformer encoders.
/// - FunASR's default VAD max single-segment length is 30000ms.
/// - Dolphin (WeNet E-Branchformer) is trained and evaluated with audio
///   padded/truncated to 30s.
/// - Cohere's own longform reference decoder uses a 30s sliding window.
///
/// A new `GlobalQuadratic` architecture should use this default. Only
/// override `max_safe_chunk_seconds` with a different value when the
/// upstream model card states an explicit, different recommended chunk
/// length -- and cite that source in a comment next to the override (see
/// firered-aed's descriptor entry below, whose upstream guidance --
/// 60s-warn/200s-error -- is wider than this default; it still uses this
/// default rather than the wider figure, for RAM margin and cross-family
/// consistency, and says so in its own comment).
pub(crate) const DEFAULT_ENCODER_SAFE_CHUNK_SECONDS: f32 = 30.0;

/// How this architecture's encoder attends over time -- the single
/// declaration of the encoder memory-scaling fact that longform safety caps
/// consult (see `native_transcribe::apply_encoder_attention_span_longform_safety_policy`).
/// A pure compute/memory-footprint property, independent of the
/// `ConservativeSeq2SeqV1` decode-side longform profile
/// (`BuiltinDecodePolicyLongformProfile`, issue #60's repetition guard): a
/// family can carry both a `GlobalQuadratic` encoder cap and a tighter
/// `ConservativeSeq2SeqV1` chunk cap at once. Both constrain the same
/// `LongFormOptions` fields, so the tighter cap always wins (the policy
/// applies them in sequence and never widens a value the other narrowed).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum OpenAsrEncoderAttentionSpan {
    /// Full O(frames^2) self-attention over the whole encoder input: every
    /// additional second of audio in a single chunk adds one more row and
    /// column to every layer's attention matrix, so encoder activation memory
    /// grows quadratically with the wall-clock length of that chunk.
    /// `max_safe_chunk_seconds` is the longest chunk this repo has validated
    /// as safe on commodity RAM; longform slicing must never hand this
    /// architecture a chunk longer than that (issue #68). Use
    /// [`DEFAULT_ENCODER_SAFE_CHUNK_SECONDS`] unless the upstream model card
    /// gives an explicit different recommendation (see that constant's doc).
    GlobalQuadratic { max_safe_chunk_seconds: f32 },
    /// Architecture-fixed attention window (whisper's 30s log-mel frame): the
    /// encoder never attends beyond a fixed span regardless of the requested
    /// longform chunk length, so no additional longform safety cap applies.
    FixedWindow,
    /// Local/chunked attention with a bounded per-chunk cache (zipformer's
    /// streaming multi-scale encoder): encoder memory is bounded per chunk by
    /// construction, independent of how long the logical longform chunk is,
    /// so no additional longform safety cap applies.
    LocalChunked,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OpenAsrArchitectureDescriptor {
    pub runtime_architecture_aliases: &'static [&'static str],
    pub model_family: &'static str,
    pub model_architecture: &'static str,
    pub adapter_id: &'static str,
    /// How this family handles a source-language request (see `LanguageFamilyHint`).
    pub language_family_hint: LanguageFamilyHint,
    pub audio_frontend_id: &'static str,
    pub runtime_tensor_contract_id: &'static str,
    pub tokenizer_id: &'static str,
    pub decode_policy_id: &'static str,
    pub executor_component_id: &'static str,
    pub execution_capability: GgmlExecutionCapability,
    pub prefer_cpu_decoder_for_multichunk_metal: bool,
    /// Whether this family's own decode loop can emit diarization tokens (the
    /// cohere token-stream is the only builtin case today). The single
    /// declaration of this architecture-level capability fact -- runtime
    /// dispatch reads it via `GgmlFamilyAdapterDescriptor::self_diarizes`
    /// rather than matching on `adapter_id` (see
    /// `native_runtime_metadata_supports_diarization`, which still verifies
    /// the specific pack actually carries the tokens before trusting this).
    pub self_diarizes: bool,
    /// Whether this family's transcripts include punctuation -- an
    /// architecture/training-corpus property, not a per-release editorial
    /// choice (e.g. Dolphin's training corpus has no punctuation to learn
    /// from, so it is honestly `Some(false)`). `None` means "no fixed
    /// per-family answer" (e.g. a CTC/character family whose vocab depends on
    /// the specific imported checkpoint, not the architecture).
    ///
    /// This is the single Rust-side declaration of the fact; catalog
    /// authoring (`tooling/publish-model/scripts/_catalog.py`'s
    /// `PUNCTUATION_BY_FAMILY`) is hand-kept in lockstep with it (no
    /// Rust<->Python codegen bridge exists yet) and
    /// `registry/tests/catalog.rs`'s `embedded_catalog_emits_punctuation_matches_family`
    /// cross-checks the shipped catalog against
    /// [`emits_punctuation_for_model_architecture`] so the two cannot drift
    /// silently. `registry::CatalogModel::emits_punctuation` is a read-only
    /// wire mirror of the catalog value, not an independent declaration.
    pub emits_punctuation: Option<bool>,
    /// Canonical required GGUF/`.oasr` hparam keys for this architecture.
    /// Authoritative source of truth for the hparam schema; the per-arch
    /// runtime contract resolves aliases and optional consistency keys on top.
    pub hparam_schema: &'static [&'static str],
    /// Data-driven layer-stack declaration consumed by the per-shape composer
    /// (P4 "new model = data"). `None` for architectures that stay on a
    /// hand-written executor and are never composed (whisper, the bit-level
    /// regression gate). See [`block_stack`].
    pub block_stack: Option<OpenAsrBlockStackDescriptor>,
    /// How this architecture's encoder scales with chunk length -- the single
    /// source of truth `native_transcribe`'s longform safety policy consults
    /// to keep long, pause-free audio from handing a quadratic-attention
    /// encoder an unbounded chunk (issue #68). See
    /// [`OpenAsrEncoderAttentionSpan`]. A mandatory field (not `Option`) so a
    /// new architecture cannot compile without declaring it.
    pub encoder_attention_span: OpenAsrEncoderAttentionSpan,
}

impl OpenAsrArchitectureDescriptor {
    /// The longform chunk-length safety cap this architecture's encoder
    /// tolerates, if any (`None` when the encoder needs no additional cap --
    /// `FixedWindow`/`LocalChunked`). See [`OpenAsrEncoderAttentionSpan`].
    pub(crate) fn longform_max_safe_chunk_seconds(self) -> Option<f32> {
        match self.encoder_attention_span {
            OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                max_safe_chunk_seconds,
            } => Some(max_safe_chunk_seconds),
            OpenAsrEncoderAttentionSpan::FixedWindow
            | OpenAsrEncoderAttentionSpan::LocalChunked => None,
        }
    }

    fn matches_runtime_architecture_alias(&self, alias: &str) -> bool {
        self.runtime_architecture_aliases
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(alias))
    }

    pub(crate) fn ggml_family_adapter_descriptor(self) -> GgmlFamilyAdapterDescriptor {
        GgmlFamilyAdapterDescriptor {
            adapter_id: self.adapter_id,
            language_family_hint: self.language_family_hint,
            model_family: self.model_family,
            model_architecture: self.model_architecture,
            audio_frontend_id: self.audio_frontend_id,
            tokenizer_id: self.tokenizer_id,
            decode_policy_id: self.decode_policy_id,
            execution_capability: self.execution_capability,
            self_diarizes: self.self_diarizes,
        }
    }
}

/// Whether a builtin family's decoder ever predicts a punctuation token (see
/// [`OpenAsrArchitectureDescriptor::emits_punctuation`]), looked up by GGUF
/// `model_architecture`. The single Rust-side accessor for this fact --
/// mirrors `executor_component_registry::builtin_executor_supports_phrase_bias_for_model_architecture`'s
/// per-architecture lookup shape. Only test-consumed today (same pending-wiring
/// status as `punctuation::should_apply_punctuation`, which this is meant to
/// feed once the restoration stage is wired into a transcription path), hence
/// the explicit allow rather than `#[cfg(test)]` -- this is the intended
/// production accessor, not test-only scaffolding.
#[allow(dead_code)]
pub(crate) fn emits_punctuation_for_model_architecture(model_architecture: &str) -> Option<bool> {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(model_architecture)
        .and_then(|descriptor| descriptor.emits_punctuation)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum OpenAsrArchitectureRegistryError {
    MissingComponentReference {
        model_architecture: &'static str,
        kind: OpenAsrComponentKind,
        component_id: &'static str,
    },
    EmptyHparamSchema {
        model_architecture: &'static str,
    },
    DuplicateHparamKey {
        model_architecture: &'static str,
        key: &'static str,
    },
    /// A block-stack stage's `layer_count_hparam` is not declared in the
    /// architecture's `hparam_schema` (the composer would have no layer count).
    BlockStackLayerCountKeyNotInSchema {
        model_architecture: &'static str,
        layer_count_hparam: &'static str,
    },
    /// A block-stack stage declares an empty `tensor_name_scope` (the composer
    /// could not bind per-layer weights).
    BlockStackEmptyTensorScope {
        model_architecture: &'static str,
    },
    /// The decoder stage's `block_kind` is not the kind the declared
    /// `orchestration_shape` assembles (e.g. a `Seq2SeqDecoderLayer` under the
    /// `LlmDecoder` shape). Would route the descriptor to the wrong composer.
    DecoderBlockKindIncompatibleWithShape {
        model_architecture: &'static str,
        orchestration_shape: OpenAsrOrchestrationShape,
        block_kind: OpenAsrBlockKind,
    },
    /// The encoder stage's `block_kind` is not the kind the declared
    /// `orchestration_shape` assembles for its encoder.
    EncoderBlockKindIncompatibleWithShape {
        model_architecture: &'static str,
        orchestration_shape: OpenAsrOrchestrationShape,
        block_kind: OpenAsrBlockKind,
    },
    /// The `Ctc` shape is non-autoregressive (encoder + CTC head only) but the
    /// descriptor declared a `decoder_stage`.
    CtcShapeMustNotHaveDecoderStage {
        model_architecture: &'static str,
    },
    /// An autoregressive shape (`LlmDecoder` / `Seq2SeqEncoderDecoder`) is missing
    /// its required `decoder_stage`.
    NonCtcShapeMustHaveDecoderStage {
        model_architecture: &'static str,
        orchestration_shape: OpenAsrOrchestrationShape,
    },
    /// A `GlobalQuadratic` encoder declared a `max_safe_chunk_seconds` that is
    /// not finite and positive. Garbage data here would silently disable the
    /// longform safety cap it exists to enforce (issue #68).
    EncoderAttentionSpanNotFinitePositive {
        model_architecture: &'static str,
        max_safe_chunk_seconds: f32,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OpenAsrComponentRegistry {
    descriptors: &'static [OpenAsrComponentDescriptor],
}

impl OpenAsrComponentRegistry {
    pub(crate) fn with_builtins() -> Self {
        Self {
            descriptors: BUILTIN_COMPONENT_DESCRIPTORS,
        }
    }

    pub(crate) fn find(
        self,
        kind: OpenAsrComponentKind,
        id: &str,
    ) -> Option<OpenAsrComponentDescriptor> {
        self.descriptors
            .iter()
            .copied()
            .find(|descriptor| descriptor.kind == kind && descriptor.id == id)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OpenAsrArchitectureRegistry {
    architectures: &'static [OpenAsrArchitectureDescriptor],
    components: OpenAsrComponentRegistry,
}

impl OpenAsrArchitectureRegistry {
    pub(crate) fn with_builtins() -> Self {
        Self {
            architectures: BUILTIN_ARCHITECTURE_DESCRIPTORS,
            components: OpenAsrComponentRegistry::with_builtins(),
        }
    }

    pub(crate) fn descriptors(self) -> &'static [OpenAsrArchitectureDescriptor] {
        self.architectures
    }

    pub(crate) fn find_by_runtime_architecture_alias(
        self,
        alias: &str,
    ) -> Option<OpenAsrArchitectureDescriptor> {
        self.architectures
            .iter()
            .copied()
            .find(|descriptor| descriptor.matches_runtime_architecture_alias(alias))
    }

    pub(crate) fn find_by_model_architecture(
        self,
        architecture_id: &str,
    ) -> Option<OpenAsrArchitectureDescriptor> {
        self.architectures
            .iter()
            .copied()
            .find(|descriptor| descriptor.model_architecture == architecture_id)
    }

    pub(crate) fn validate_references(self) -> Result<(), OpenAsrArchitectureRegistryError> {
        for descriptor in self.architectures {
            self.require_component(
                *descriptor,
                OpenAsrComponentKind::AudioFrontend,
                descriptor.audio_frontend_id,
            )?;
            self.require_component(
                *descriptor,
                OpenAsrComponentKind::DecodePolicy,
                descriptor.decode_policy_id,
            )?;
            self.require_component(
                *descriptor,
                OpenAsrComponentKind::RuntimeTensorContract,
                descriptor.runtime_tensor_contract_id,
            )?;
            self.require_component(
                *descriptor,
                OpenAsrComponentKind::Tokenizer,
                descriptor.tokenizer_id,
            )?;
            self.require_component(
                *descriptor,
                OpenAsrComponentKind::Executor,
                descriptor.executor_component_id,
            )?;
            Self::validate_hparam_schema(*descriptor)?;
            Self::validate_block_stack(*descriptor)?;
            Self::validate_encoder_attention_span(*descriptor)?;
        }
        Ok(())
    }

    pub(crate) fn synthesize_selection_metadata_defaults(
        self,
        metadata: &mut BTreeMap<String, String>,
    ) {
        let Some(architecture_alias) = metadata
            .get(GENERAL_ARCHITECTURE_KEY)
            .map(String::as_str)
            .map(str::trim)
        else {
            return;
        };
        if architecture_alias.is_empty() {
            return;
        }
        let Some(descriptor) = self.find_by_runtime_architecture_alias(architecture_alias) else {
            return;
        };

        metadata
            .entry(OASR_METADATA_KEY_PACKAGE_VERSION.to_string())
            .or_insert_with(|| OASR_PACKAGE_VERSION_V1.to_string());
        metadata
            .entry(OASR_METADATA_KEY_MODEL_FAMILY.to_string())
            .or_insert_with(|| descriptor.model_family.to_string());
        metadata
            .entry(OASR_METADATA_KEY_MODEL_ARCHITECTURE.to_string())
            .or_insert_with(|| descriptor.model_architecture.to_string());
        metadata
            .entry(OASR_METADATA_KEY_AUDIO_FRONTEND.to_string())
            .or_insert_with(|| descriptor.audio_frontend_id.to_string());
        metadata
            .entry(OASR_METADATA_KEY_DECODE_POLICY.to_string())
            .or_insert_with(|| descriptor.decode_policy_id.to_string());
        metadata
            .entry(GGML_TOKENIZER_ID_KEY.to_string())
            .or_insert_with(|| descriptor.tokenizer_id.to_string());
    }

    fn require_component(
        self,
        descriptor: OpenAsrArchitectureDescriptor,
        kind: OpenAsrComponentKind,
        id: &'static str,
    ) -> Result<(), OpenAsrArchitectureRegistryError> {
        self.components.find(kind, id).map(|_| ()).ok_or(
            OpenAsrArchitectureRegistryError::MissingComponentReference {
                model_architecture: descriptor.model_architecture,
                kind,
                component_id: id,
            },
        )
    }

    fn validate_hparam_schema(
        descriptor: OpenAsrArchitectureDescriptor,
    ) -> Result<(), OpenAsrArchitectureRegistryError> {
        if descriptor.hparam_schema.is_empty() {
            return Err(OpenAsrArchitectureRegistryError::EmptyHparamSchema {
                model_architecture: descriptor.model_architecture,
            });
        }
        for (index, key) in descriptor.hparam_schema.iter().enumerate() {
            if descriptor.hparam_schema[..index].contains(key) {
                return Err(OpenAsrArchitectureRegistryError::DuplicateHparamKey {
                    model_architecture: descriptor.model_architecture,
                    key,
                });
            }
        }
        Ok(())
    }

    /// Fail-closed consistency check on the encoder-attention-span cap: a
    /// `GlobalQuadratic` architecture's `max_safe_chunk_seconds` must be
    /// finite and positive, otherwise the longform safety policy that reads
    /// it (`native_transcribe::apply_encoder_attention_span_longform_safety_policy`)
    /// would silently no-op on garbage data.
    fn validate_encoder_attention_span(
        descriptor: OpenAsrArchitectureDescriptor,
    ) -> Result<(), OpenAsrArchitectureRegistryError> {
        if let OpenAsrEncoderAttentionSpan::GlobalQuadratic {
            max_safe_chunk_seconds,
        } = descriptor.encoder_attention_span
            && !(max_safe_chunk_seconds.is_finite() && max_safe_chunk_seconds > 0.0)
        {
            return Err(
                OpenAsrArchitectureRegistryError::EncoderAttentionSpanNotFinitePositive {
                    model_architecture: descriptor.model_architecture,
                    max_safe_chunk_seconds,
                },
            );
        }
        Ok(())
    }

    /// Fail-closed consistency check on the optional block-stack descriptor: each
    /// stage's `layer_count_hparam` must be a declared hparam key, its
    /// `tensor_name_scope` must be non-empty, AND each stage's `block_kind` must
    /// be the kind its `orchestration_shape` assembles (so the descriptor can
    /// never route to the wrong composer once it becomes load-bearing in S5).
    /// Architectures with no block stack (whisper) trivially pass. Keeps the
    /// block-stack data honest before any orchestrator reads it.
    fn validate_block_stack(
        descriptor: OpenAsrArchitectureDescriptor,
    ) -> Result<(), OpenAsrArchitectureRegistryError> {
        let Some(block_stack) = descriptor.block_stack else {
            return Ok(());
        };
        for stage in block_stack.stages() {
            if stage.tensor_name_scope.is_empty() {
                return Err(
                    OpenAsrArchitectureRegistryError::BlockStackEmptyTensorScope {
                        model_architecture: descriptor.model_architecture,
                    },
                );
            }
            if !descriptor.hparam_schema.contains(&stage.layer_count_hparam) {
                return Err(
                    OpenAsrArchitectureRegistryError::BlockStackLayerCountKeyNotInSchema {
                        model_architecture: descriptor.model_architecture,
                        layer_count_hparam: stage.layer_count_hparam,
                    },
                );
            }
        }
        // block_kind <-> orchestration_shape consistency (S5a): the shape fixes
        // which nn/ block each stage assembles; a descriptor declaring a mismatch
        // would silently route to the wrong composer once load-bearing. The Ctc
        // shape (S0) is encoder-only (`decoder_stage: None`); the autoregressive
        // shapes require a decoder stage. `expected_decoder_kind` is `None` for
        // Ctc, `Some` otherwise.
        // The Ctc shape accepts more than one encoder block (parakeet's
        // FastConformer `ConformerBlock` and wav2vec2's post-norm transformer
        // layer are both valid CTC encoders), so the expected-encoder check is a
        // small allowed-set, not a single kind.
        let (expected_encoder_kinds, expected_decoder_kind): (&[OpenAsrBlockKind], _) =
            match block_stack.orchestration_shape {
                OpenAsrOrchestrationShape::LlmDecoder => (
                    &[OpenAsrBlockKind::TransformerEncoderLayer],
                    Some(OpenAsrBlockKind::LlmDecoderLayer),
                ),
                OpenAsrOrchestrationShape::Seq2SeqEncoderDecoder => (
                    &[OpenAsrBlockKind::ConformerBlock],
                    Some(OpenAsrBlockKind::Seq2SeqDecoderLayer),
                ),
                OpenAsrOrchestrationShape::Ctc => (
                    &[
                        OpenAsrBlockKind::ConformerBlock,
                        OpenAsrBlockKind::Wav2Vec2PostNormEncoderLayer,
                        OpenAsrBlockKind::SanMFsmnEncoderLayer,
                    ],
                    None,
                ),
            };
        // Shape <-> decoder-stage presence (checked before any decoder deref).
        match (expected_decoder_kind, block_stack.decoder_stage) {
            (None, Some(_)) => {
                return Err(
                    OpenAsrArchitectureRegistryError::CtcShapeMustNotHaveDecoderStage {
                        model_architecture: descriptor.model_architecture,
                    },
                );
            }
            (Some(_), None) => {
                return Err(
                    OpenAsrArchitectureRegistryError::NonCtcShapeMustHaveDecoderStage {
                        model_architecture: descriptor.model_architecture,
                        orchestration_shape: block_stack.orchestration_shape,
                    },
                );
            }
            (Some(expected_decoder_kind), Some(decoder_stage))
                if decoder_stage.block_kind != expected_decoder_kind =>
            {
                return Err(
                    OpenAsrArchitectureRegistryError::DecoderBlockKindIncompatibleWithShape {
                        model_architecture: descriptor.model_architecture,
                        orchestration_shape: block_stack.orchestration_shape,
                        block_kind: decoder_stage.block_kind,
                    },
                );
            }
            _ => {}
        }
        if let Some(encoder_stage) = block_stack.encoder_stage
            && !expected_encoder_kinds.contains(&encoder_stage.block_kind)
        {
            return Err(
                OpenAsrArchitectureRegistryError::EncoderBlockKindIncompatibleWithShape {
                    model_architecture: descriptor.model_architecture,
                    orchestration_shape: block_stack.orchestration_shape,
                    block_kind: encoder_stage.block_kind,
                },
            );
        }
        Ok(())
    }
}

const BUILTIN_COMPONENT_DESCRIPTORS: &[OpenAsrComponentDescriptor] = &[
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::AudioFrontend,
        id: COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::AudioFrontend,
        id: WHISPER_AUDIO_FRONTEND_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::AudioFrontend,
        id: QWEN3_ASR_AUDIO_FRONTEND_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::DecodePolicy,
        id: COHERE_TRANSCRIBE_DECODE_POLICY_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::DecodePolicy,
        id: WHISPER_DECODE_POLICY_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::DecodePolicy,
        id: QWEN3_ASR_DECODE_POLICY_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Executor,
        id: COHERE_TRANSCRIBE_EXECUTOR_COMPONENT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Executor,
        id: WHISPER_EXECUTOR_COMPONENT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Executor,
        id: QWEN3_ASR_EXECUTOR_COMPONENT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::RuntimeTensorContract,
        id: COHERE_TRANSCRIBE_RUNTIME_TENSOR_CONTRACT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::RuntimeTensorContract,
        id: WHISPER_RUNTIME_TENSOR_CONTRACT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::RuntimeTensorContract,
        id: QWEN3_ASR_RUNTIME_TENSOR_CONTRACT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Tokenizer,
        id: COHERE_TRANSCRIBE_TOKENIZER_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Tokenizer,
        id: WHISPER_TOKENIZER_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Tokenizer,
        id: QWEN3_ASR_TOKENIZER_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::AudioFrontend,
        id: PARAKEET_CTC_AUDIO_FRONTEND_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::DecodePolicy,
        id: PARAKEET_CTC_DECODE_POLICY_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::RuntimeTensorContract,
        id: PARAKEET_CTC_RUNTIME_TENSOR_CONTRACT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Tokenizer,
        id: PARAKEET_CTC_TOKENIZER_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Executor,
        id: PARAKEET_CTC_EXECUTOR_COMPONENT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::AudioFrontend,
        id: PARAKEET_TDT_AUDIO_FRONTEND_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::DecodePolicy,
        id: PARAKEET_TDT_DECODE_POLICY_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::RuntimeTensorContract,
        id: PARAKEET_TDT_RUNTIME_TENSOR_CONTRACT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Tokenizer,
        id: PARAKEET_TDT_TOKENIZER_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Executor,
        id: PARAKEET_TDT_EXECUTOR_COMPONENT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::AudioFrontend,
        id: WAV2VEC2_CTC_AUDIO_FRONTEND_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::DecodePolicy,
        id: WAV2VEC2_CTC_DECODE_POLICY_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::RuntimeTensorContract,
        id: WAV2VEC2_CTC_RUNTIME_TENSOR_CONTRACT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Tokenizer,
        id: WAV2VEC2_CTC_TOKENIZER_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Executor,
        id: WAV2VEC2_CTC_EXECUTOR_COMPONENT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::AudioFrontend,
        id: XASR_ZIPFORMER_AUDIO_FRONTEND_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::DecodePolicy,
        id: XASR_ZIPFORMER_DECODE_POLICY_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::RuntimeTensorContract,
        id: XASR_ZIPFORMER_RUNTIME_TENSOR_CONTRACT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Tokenizer,
        id: XASR_ZIPFORMER_TOKENIZER_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Executor,
        id: XASR_ZIPFORMER_EXECUTOR_COMPONENT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::AudioFrontend,
        id: MOONSHINE_AUDIO_FRONTEND_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::DecodePolicy,
        id: MOONSHINE_DECODE_POLICY_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::RuntimeTensorContract,
        id: MOONSHINE_RUNTIME_TENSOR_CONTRACT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Tokenizer,
        id: MOONSHINE_TOKENIZER_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Executor,
        id: MOONSHINE_EXECUTOR_COMPONENT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::AudioFrontend,
        id: DOLPHIN_AUDIO_FRONTEND_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::DecodePolicy,
        id: DOLPHIN_DECODE_POLICY_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::RuntimeTensorContract,
        id: DOLPHIN_RUNTIME_TENSOR_CONTRACT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Tokenizer,
        id: DOLPHIN_TOKENIZER_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Executor,
        id: DOLPHIN_EXECUTOR_COMPONENT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::AudioFrontend,
        id: SENSEVOICE_AUDIO_FRONTEND_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::DecodePolicy,
        id: SENSEVOICE_DECODE_POLICY_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::RuntimeTensorContract,
        id: SENSEVOICE_RUNTIME_TENSOR_CONTRACT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Tokenizer,
        id: SENSEVOICE_TOKENIZER_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Executor,
        id: SENSEVOICE_EXECUTOR_COMPONENT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::AudioFrontend,
        id: FIRERED_AED_AUDIO_FRONTEND_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::DecodePolicy,
        id: FIRERED_AED_DECODE_POLICY_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::RuntimeTensorContract,
        id: FIRERED_AED_RUNTIME_TENSOR_CONTRACT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Tokenizer,
        id: FIRERED_AED_TOKENIZER_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Executor,
        id: FIRERED_AED_EXECUTOR_COMPONENT_ID,
    },
];

const BUILTIN_ARCHITECTURE_DESCRIPTORS: &[OpenAsrArchitectureDescriptor] = &[
    OpenAsrArchitectureDescriptor {
        runtime_architecture_aliases: &["cohere-transcribe"],
        model_family: "cohere-transcribe",
        model_architecture: COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
        adapter_id: COHERE_TRANSCRIBE_GGML_ADAPTER_ID,
        language_family_hint: LanguageFamilyHint::SelectsViaPrompt {
            default_language: "en",
        },
        audio_frontend_id: COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID,
        runtime_tensor_contract_id: COHERE_TRANSCRIBE_RUNTIME_TENSOR_CONTRACT_ID,
        tokenizer_id: COHERE_TRANSCRIBE_TOKENIZER_ID,
        decode_policy_id: COHERE_TRANSCRIBE_DECODE_POLICY_ID,
        executor_component_id: COHERE_TRANSCRIBE_EXECUTOR_COMPONENT_ID,
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: true,
        self_diarizes: true,
        emits_punctuation: Some(true),
        hparam_schema: COHERE_TRANSCRIBE_HPARAM_SCHEMA,
        block_stack: Some(OpenAsrBlockStackDescriptor {
            orchestration_shape: OpenAsrOrchestrationShape::Seq2SeqEncoderDecoder,
            encoder_stage: Some(OpenAsrStageDescriptor {
                block_kind: OpenAsrBlockKind::ConformerBlock,
                layer_count_hparam: COHERE_TRANSCRIBE_ENCODER_LAYERS_KEY,
                tensor_name_scope: "enc.blk",
            }),
            decoder_stage: Some(OpenAsrStageDescriptor {
                block_kind: OpenAsrBlockKind::Seq2SeqDecoderLayer,
                layer_count_hparam: COHERE_TRANSCRIBE_DECODER_LAYERS_KEY,
                tensor_name_scope: "dec.blk",
            }),
        }),
        // Conformer encoder is full self-attention over the whole chunk:
        // quadratic in chunk length, same safe ceiling as the other
        // global-quadratic builtins (issue #68). Also carries the
        // `ConservativeSeq2SeqV1` decode-side longform profile (issue #60's
        // repetition guard); the two caps now agree at the same default, so
        // composing them (taking the min) is a no-op here.
        encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
            max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
        },
    },
    OpenAsrArchitectureDescriptor {
        runtime_architecture_aliases: &["whisper"],
        model_family: "whisper",
        model_architecture: WHISPER_GGML_ARCHITECTURE_ID,
        adapter_id: WHISPER_GGML_ADAPTER_ID,
        language_family_hint: LanguageFamilyHint::WhisperVocabGated,
        audio_frontend_id: WHISPER_AUDIO_FRONTEND_ID,
        runtime_tensor_contract_id: WHISPER_RUNTIME_TENSOR_CONTRACT_ID,
        tokenizer_id: WHISPER_TOKENIZER_ID,
        decode_policy_id: WHISPER_DECODE_POLICY_ID,
        executor_component_id: WHISPER_EXECUTOR_COMPONENT_ID,
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        self_diarizes: false,
        emits_punctuation: Some(true),
        hparam_schema: WHISPER_HPARAM_SCHEMA,
        // whisper remains the hand-written bit-level regression gate and is
        // never composed — no block-stack data until P9 sinks its optimizations
        // into the shared blocks.
        block_stack: None,
        // Architecture-fixed 30s log-mel window: the encoder never sees more
        // than a fixed span no matter how long the requested longform chunk
        // is, so it needs no additional longform safety cap.
        encoder_attention_span: OpenAsrEncoderAttentionSpan::FixedWindow,
    },
    OpenAsrArchitectureDescriptor {
        runtime_architecture_aliases: &[QWEN3_ARCHITECTURE_VALUE],
        model_family: QWEN3_ASR_MODEL_FAMILY,
        model_architecture: QWEN3_ASR_GGML_ARCHITECTURE_ID,
        adapter_id: QWEN3_ASR_GGML_ADAPTER_ID,
        language_family_hint: LanguageFamilyHint::SelfDetectsRejectsHint {
            // Qwen3-ASR conditions language via free text in the chat prompt (no
            // language tokens in its vocab) and does not expose the language it
            // auto-detects. Until that text conditioning is wired and verified
            // against a real pack, an explicit hint is rejected (not faked) and
            // the detected language is reported as null. See docs/KNOWN_LIMITATIONS.md.
            reject_reason: "Qwen3-ASR auto-detects the source language and does not accept an explicit selection; use a multilingual Whisper pack to force or report a language.",
        },
        audio_frontend_id: QWEN3_ASR_AUDIO_FRONTEND_ID,
        runtime_tensor_contract_id: QWEN3_ASR_RUNTIME_TENSOR_CONTRACT_ID,
        tokenizer_id: QWEN3_ASR_TOKENIZER_ID,
        decode_policy_id: QWEN3_ASR_DECODE_POLICY_ID,
        executor_component_id: QWEN3_ASR_EXECUTOR_COMPONENT_ID,
        execution_capability: GgmlExecutionCapability::NativeGraphLoweringV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        self_diarizes: false,
        emits_punctuation: Some(true),
        hparam_schema: QWEN3_ASR_HPARAM_SCHEMA,
        block_stack: Some(OpenAsrBlockStackDescriptor {
            orchestration_shape: OpenAsrOrchestrationShape::LlmDecoder,
            encoder_stage: Some(OpenAsrStageDescriptor {
                block_kind: OpenAsrBlockKind::TransformerEncoderLayer,
                layer_count_hparam: QWEN3_AUDIO_LAYERS_KEY,
                tensor_name_scope: "audio.blk",
            }),
            decoder_stage: Some(OpenAsrStageDescriptor {
                block_kind: OpenAsrBlockKind::LlmDecoderLayer,
                layer_count_hparam: QWEN3_LLM_LAYERS_KEY,
                tensor_name_scope: "blk",
            }),
        }),
        // The audio encoder is full self-attention over the whole chunk:
        // quadratic in chunk length (issue #68); the LLM decoder side is
        // autoregressive token generation, not chunk-length-scaled encoder
        // attention, so it does not change this classification.
        encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
            max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
        },
    },
    OpenAsrArchitectureDescriptor {
        runtime_architecture_aliases: &["parakeet-ctc", "parakeet"],
        model_family: "parakeet-ctc",
        model_architecture: PARAKEET_CTC_GGML_ARCHITECTURE_ID,
        adapter_id: PARAKEET_CTC_GGML_ADAPTER_ID,
        language_family_hint: LanguageFamilyHint::FixedMonolingual { language: "en" },
        audio_frontend_id: PARAKEET_CTC_AUDIO_FRONTEND_ID,
        runtime_tensor_contract_id: PARAKEET_CTC_RUNTIME_TENSOR_CONTRACT_ID,
        tokenizer_id: PARAKEET_CTC_TOKENIZER_ID,
        decode_policy_id: PARAKEET_CTC_DECODE_POLICY_ID,
        executor_component_id: PARAKEET_CTC_EXECUTOR_COMPONENT_ID,
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        self_diarizes: false,
        // Character/BPE CTC: whether an imported checkpoint's vocab includes
        // punctuation depends on that specific checkpoint's training corpus,
        // not the architecture, so this cannot be stated as a fixed
        // per-family fact (mirrors `_catalog.py`'s `PUNCTUATION_BY_FAMILY`
        // module docstring, which deliberately omits parakeet/wav2vec2).
        emits_punctuation: None,
        hparam_schema: PARAKEET_CTC_HPARAM_SCHEMA,
        // Non-autoregressive CTC: encoder + CTC head only, no decoder stage.
        block_stack: Some(OpenAsrBlockStackDescriptor {
            orchestration_shape: OpenAsrOrchestrationShape::Ctc,
            encoder_stage: Some(OpenAsrStageDescriptor {
                block_kind: OpenAsrBlockKind::ConformerBlock,
                layer_count_hparam: "parakeet.n_layers",
                tensor_name_scope: "enc.blk",
            }),
            decoder_stage: None,
        }),
        // FastConformer encoder is full self-attention over the whole chunk:
        // quadratic in chunk length (issue #68).
        encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
            max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
        },
    },
    OpenAsrArchitectureDescriptor {
        runtime_architecture_aliases: &["parakeet-tdt"],
        model_family: "parakeet-tdt",
        model_architecture: PARAKEET_TDT_GGML_ARCHITECTURE_ID,
        adapter_id: PARAKEET_TDT_GGML_ADAPTER_ID,
        // parakeet-tdt-0.6b-v3: 25 European languages, no per-request language
        // selection (the model decodes whatever it hears; NVIDIA's card lists
        // the fixed set).
        language_family_hint: LanguageFamilyHint::FixedMultilingual {
            languages: &[
                "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt",
                "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
            ],
        },
        audio_frontend_id: PARAKEET_TDT_AUDIO_FRONTEND_ID,
        runtime_tensor_contract_id: PARAKEET_TDT_RUNTIME_TENSOR_CONTRACT_ID,
        tokenizer_id: PARAKEET_TDT_TOKENIZER_ID,
        decode_policy_id: PARAKEET_TDT_DECODE_POLICY_ID,
        executor_component_id: PARAKEET_TDT_EXECUTOR_COMPONENT_ID,
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        self_diarizes: false,
        // Verified on the imported pack: trained on transcripts that preserve
        // punctuation and capitalization (mirrors `_catalog.py`'s
        // `PUNCTUATION_BY_FAMILY["parakeet-tdt"]`).
        emits_punctuation: Some(true),
        hparam_schema: PARAKEET_TDT_HPARAM_SCHEMA,
        // The FastConformer encoder reuses the composer conformer block, but
        // the TDT decode loop (LSTM prediction network + duration-driven
        // frame skipping) is a transducer, which is not a composer
        // orchestration shape -- dedicated executor, like xasr (block_stack:
        // None).
        block_stack: None,
        // The FastConformer encoder is full self-attention over the whole
        // chunk: quadratic in chunk length (issue #68). The TDT
        // decoder/joiner is a separate autoregressive stage and does not
        // change the encoder's scaling.
        encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
            max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
        },
    },
    OpenAsrArchitectureDescriptor {
        runtime_architecture_aliases: &["wav2vec2-ctc", "wav2vec2"],
        model_family: "wav2vec2-ctc",
        model_architecture: WAV2VEC2_CTC_GGML_ARCHITECTURE_ID,
        adapter_id: WAV2VEC2_CTC_GGML_ADAPTER_ID,
        language_family_hint: LanguageFamilyHint::FixedMonolingual { language: "en" },
        audio_frontend_id: WAV2VEC2_CTC_AUDIO_FRONTEND_ID,
        runtime_tensor_contract_id: WAV2VEC2_CTC_RUNTIME_TENSOR_CONTRACT_ID,
        tokenizer_id: WAV2VEC2_CTC_TOKENIZER_ID,
        decode_policy_id: WAV2VEC2_CTC_DECODE_POLICY_ID,
        executor_component_id: WAV2VEC2_CTC_EXECUTOR_COMPONENT_ID,
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        self_diarizes: false,
        // Character CTC: same BYO-checkpoint reasoning as parakeet-ctc above.
        emits_punctuation: None,
        hparam_schema: WAV2VEC2_CTC_HPARAM_SCHEMA,
        // Non-autoregressive CTC: raw-waveform conv extractor + post-norm
        // transformer encoder + CTC head, no decoder stage.
        block_stack: Some(OpenAsrBlockStackDescriptor {
            orchestration_shape: OpenAsrOrchestrationShape::Ctc,
            encoder_stage: Some(OpenAsrStageDescriptor {
                block_kind: OpenAsrBlockKind::Wav2Vec2PostNormEncoderLayer,
                layer_count_hparam: "wav2vec2.n_layers",
                tensor_name_scope: "enc.blk",
            }),
            decoder_stage: None,
        }),
        // Post-norm transformer encoder is full self-attention over the
        // whole chunk: quadratic in chunk length (issue #68).
        encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
            max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
        },
    },
    OpenAsrArchitectureDescriptor {
        runtime_architecture_aliases: &["xasr-zipformer", "xasr-zh-en"],
        model_family: XASR_ZIPFORMER_MODEL_FAMILY,
        model_architecture: XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
        adapter_id: XASR_ZIPFORMER_GGML_ADAPTER_ID,
        language_family_hint: LanguageFamilyHint::FixedMultilingual {
            languages: &["en", "zh"],
        },
        audio_frontend_id: XASR_ZIPFORMER_AUDIO_FRONTEND_ID,
        runtime_tensor_contract_id: XASR_ZIPFORMER_RUNTIME_TENSOR_CONTRACT_ID,
        tokenizer_id: XASR_ZIPFORMER_TOKENIZER_ID,
        decode_policy_id: XASR_ZIPFORMER_DECODE_POLICY_ID,
        executor_component_id: XASR_ZIPFORMER_EXECUTOR_COMPONENT_ID,
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        self_diarizes: false,
        emits_punctuation: Some(true),
        hparam_schema: XASR_ZIPFORMER_HPARAM_SCHEMA,
        // Zipformer2 uses multi-scale streaming cache topology plus RNN-T
        // decoder/joiner, so it stays on its dedicated executor rather than the
        // generic block-stack composer.
        block_stack: None,
        // Zipformer2's multi-scale streaming cache is local/chunked
        // attention with a bounded per-chunk cache, not global quadratic
        // attention: encoder memory is bounded independent of the logical
        // longform chunk length, so no additional longform safety cap
        // applies (issue #68).
        encoder_attention_span: OpenAsrEncoderAttentionSpan::LocalChunked,
    },
    OpenAsrArchitectureDescriptor {
        runtime_architecture_aliases: &["moonshine", "moonshine-encoder-decoder"],
        model_family: "moonshine",
        model_architecture: MOONSHINE_GGML_ARCHITECTURE_ID,
        adapter_id: MOONSHINE_GGML_ADAPTER_ID,
        language_family_hint: LanguageFamilyHint::FixedMonolingual { language: "en" },
        audio_frontend_id: MOONSHINE_AUDIO_FRONTEND_ID,
        runtime_tensor_contract_id: MOONSHINE_RUNTIME_TENSOR_CONTRACT_ID,
        tokenizer_id: MOONSHINE_TOKENIZER_ID,
        decode_policy_id: MOONSHINE_DECODE_POLICY_ID,
        executor_component_id: MOONSHINE_EXECUTOR_COMPONENT_ID,
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        self_diarizes: false,
        emits_punctuation: Some(true),
        hparam_schema: MOONSHINE_HPARAM_SCHEMA,
        // Raw-waveform conv-stem + partial-RoPE seq2seq with a self-contained
        // dedicated executor (not the data-driven block-stack composer — its
        // RoPE conv-stem encoder + cross-attn decoder are not composer blocks).
        block_stack: None,
        // The RoPE encoder is full self-attention over the whole chunk:
        // quadratic in chunk length (issue #68), matching Moonshine's own
        // model-card guidance to keep chunks under 30 seconds. Also carries
        // the `ConservativeSeq2SeqV1` decode-side longform profile (issue
        // #60's repetition guard); the two caps now agree at the same
        // default, so composing them (taking the min) is a no-op here.
        encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
            max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
        },
    },
    OpenAsrArchitectureDescriptor {
        runtime_architecture_aliases: &[DOLPHIN_GGML_ARCHITECTURE_ID, "dolphin"],
        model_family: DOLPHIN_MODEL_FAMILY,
        model_architecture: DOLPHIN_GGML_ARCHITECTURE_ID,
        adapter_id: DOLPHIN_GGML_ADAPTER_ID,
        // The dialect prefix (`<sos> <zh> <SICHUAN> <asr> <notimestamp>`) selects
        // the language/region via prompt tokens the same way OWSM/Whisper do; the
        // detected language is not surfaced yet, so treat it as prompt-selected.
        language_family_hint: LanguageFamilyHint::SelectsViaPrompt {
            default_language: "zh",
        },
        audio_frontend_id: DOLPHIN_AUDIO_FRONTEND_ID,
        runtime_tensor_contract_id: DOLPHIN_RUNTIME_TENSOR_CONTRACT_ID,
        tokenizer_id: DOLPHIN_TOKENIZER_ID,
        decode_policy_id: DOLPHIN_DECODE_POLICY_ID,
        executor_component_id: DOLPHIN_EXECUTOR_COMPONENT_ID,
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        self_diarizes: false,
        // DataoceanAI's cn-dialect-small training corpus is transcribed
        // without punctuation and the model has no punctuation-prediction
        // head/token to enable -- honestly unpunctuated, not "unknown".
        emits_punctuation: Some(false),
        hparam_schema: DOLPHIN_HPARAM_SCHEMA,
        // E-Branchformer encoder + Transformer decoder + CTC head stay on the
        // dedicated executor (the E-Branchformer block is not a composer block
        // kind), so no data-driven block-stack descriptor.
        block_stack: None,
        // The E-Branchformer's rel-pos MHSA global branch is full
        // self-attention over the whole chunk: quadratic in chunk length
        // (issue #68).
        encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
            max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
        },
    },
    OpenAsrArchitectureDescriptor {
        runtime_architecture_aliases: &[SENSEVOICE_GGML_ARCHITECTURE_ID, "sensevoice"],
        model_family: SENSEVOICE_MODEL_FAMILY,
        model_architecture: SENSEVOICE_GGML_ARCHITECTURE_ID,
        adapter_id: SENSEVOICE_GGML_ADAPTER_ID,
        // Accepts an explicit zh/yue/en/ja/ko selection via the 4-token prompt
        // and auto-detects (readable `<|lang|>` CTC tag) when unset.
        language_family_hint: LanguageFamilyHint::DetectAndSelectsViaPrompt,
        audio_frontend_id: SENSEVOICE_AUDIO_FRONTEND_ID,
        runtime_tensor_contract_id: SENSEVOICE_RUNTIME_TENSOR_CONTRACT_ID,
        tokenizer_id: SENSEVOICE_TOKENIZER_ID,
        decode_policy_id: SENSEVOICE_DECODE_POLICY_ID,
        executor_component_id: SENSEVOICE_EXECUTOR_COMPONENT_ID,
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        self_diarizes: false,
        emits_punctuation: Some(true),
        hparam_schema: SENSEVOICE_HPARAM_SCHEMA,
        // Non-autoregressive CTC: SAN-M/FSMN encoder + CTC head, no decoder
        // stage. The `tp.blk` stage rides the same dedicated executor; the
        // descriptor pins the primary `enc.blk` stack.
        block_stack: Some(OpenAsrBlockStackDescriptor {
            orchestration_shape: OpenAsrOrchestrationShape::Ctc,
            encoder_stage: Some(OpenAsrStageDescriptor {
                block_kind: OpenAsrBlockKind::SanMFsmnEncoderLayer,
                layer_count_hparam: "sensevoice.n_layers",
                tensor_name_scope: "enc.blk",
            }),
            decoder_stage: None,
        }),
        // SAN-M/FSMN encoder's self-attention memory block is full attention
        // over the whole chunk: quadratic in chunk length (issue #68).
        encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
            max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
        },
    },
    OpenAsrArchitectureDescriptor {
        runtime_architecture_aliases: &[FIRERED_AED_GGML_ARCHITECTURE_ID, "firered-aed"],
        model_family: FIRERED_AED_MODEL_FAMILY,
        model_architecture: FIRERED_AED_GGML_ARCHITECTURE_ID,
        adapter_id: FIRERED_AED_GGML_ADAPTER_ID,
        // No language-selection prompt token and no decode-time detection: the
        // char+SPM vocab is a fixed Mandarin/Chinese-dialect + English set.
        language_family_hint: LanguageFamilyHint::FixedMultilingual {
            languages: &["zh", "en"],
        },
        audio_frontend_id: FIRERED_AED_AUDIO_FRONTEND_ID,
        runtime_tensor_contract_id: FIRERED_AED_RUNTIME_TENSOR_CONTRACT_ID,
        tokenizer_id: FIRERED_AED_TOKENIZER_ID,
        decode_policy_id: FIRERED_AED_DECODE_POLICY_ID,
        executor_component_id: FIRERED_AED_EXECUTOR_COMPONENT_ID,
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        self_diarizes: false,
        // The reference tokenizer's dict.txt has no punctuation/<space>
        // entries (char + SPM vocab trained on unpunctuated Mandarin ASR
        // corpora); verified on the golden-diff fixture transcript.
        emits_punctuation: Some(false),
        hparam_schema: FIRERED_AED_HPARAM_SCHEMA,
        // Conformer encoder + Transformer decoder attention-only decode stays
        // on the dedicated executor (the Conformer block is not a composer
        // block kind), so no data-driven block-stack descriptor.
        block_stack: None,
        // Conformer encoder is full self-attention over the whole chunk:
        // quadratic in chunk length (issue #68). FireRedASR's own upstream
        // guidance is wider than the shared default -- it warns past 60s and
        // errors past 200s -- so `DEFAULT_ENCODER_SAFE_CHUNK_SECONDS` (30s)
        // is comfortably inside FireRedASR's own safe range; used here for
        // RAM margin and cross-family consistency rather than the wider
        // upstream figure. Also carries the `ConservativeSeq2SeqV1`
        // decode-side longform profile (issue #60's repetition guard, not a
        // model-accuracy limit); the two caps now agree at the same default,
        // so composing them (taking the min) is a no-op here.
        encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
            max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
        },
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_architectures_validate_component_references() {
        OpenAsrArchitectureRegistry::with_builtins()
            .validate_references()
            .expect("builtins must reference known components");
    }

    /// Pins `self_diarizes` and `emits_punctuation` per builtin architecture --
    /// the single Rust-side declaration of both capability-single-source facts
    /// this test protects against silent drift. Cohere-transcribe is the only
    /// builtin family that self-diarizes; `emits_punctuation` values mirror
    /// `tooling/publish-model/scripts/_catalog.py`'s `PUNCTUATION_BY_FAMILY`
    /// (`registry/tests/catalog.rs`'s `embedded_catalog_emits_punctuation_matches_family`
    /// cross-checks the shipped catalog against
    /// [`emits_punctuation_for_model_architecture`] so the two stay in lockstep).
    #[test]
    fn builtin_architectures_declare_self_diarizes_and_emits_punctuation() {
        let expected: &[(&str, bool, Option<bool>)] = &[
            (COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID, true, Some(true)),
            (WHISPER_GGML_ARCHITECTURE_ID, false, Some(true)),
            (QWEN3_ASR_GGML_ARCHITECTURE_ID, false, Some(true)),
            (PARAKEET_CTC_GGML_ARCHITECTURE_ID, false, None),
            (PARAKEET_TDT_GGML_ARCHITECTURE_ID, false, Some(true)),
            (WAV2VEC2_CTC_GGML_ARCHITECTURE_ID, false, None),
            (XASR_ZIPFORMER_GGML_ARCHITECTURE_ID, false, Some(true)),
            (MOONSHINE_GGML_ARCHITECTURE_ID, false, Some(true)),
            (DOLPHIN_GGML_ARCHITECTURE_ID, false, Some(false)),
            (SENSEVOICE_GGML_ARCHITECTURE_ID, false, Some(true)),
            (FIRERED_AED_GGML_ARCHITECTURE_ID, false, Some(false)),
        ];
        let registry = OpenAsrArchitectureRegistry::with_builtins();
        let mut seen = std::collections::BTreeSet::new();

        for (model_architecture, self_diarizes, emits_punctuation) in expected.iter().copied() {
            let descriptor = registry
                .find_by_model_architecture(model_architecture)
                .unwrap_or_else(|| panic!("missing builtin architecture '{model_architecture}'"));
            assert_eq!(
                descriptor.self_diarizes, self_diarizes,
                "'{model_architecture}' self_diarizes mismatch"
            );
            assert_eq!(
                descriptor.emits_punctuation, emits_punctuation,
                "'{model_architecture}' emits_punctuation mismatch"
            );
            assert_eq!(
                emits_punctuation_for_model_architecture(model_architecture),
                emits_punctuation,
                "'{model_architecture}' accessor must match the descriptor field"
            );
            seen.insert(model_architecture);
        }

        assert_eq!(
            seen.len(),
            registry.descriptors().len(),
            "expectation table must cover every builtin architecture, no more, no less"
        );
    }

    /// Pins `encoder_attention_span` per builtin architecture -- the single
    /// Rust-side declaration `native_transcribe`'s longform safety policy
    /// consults to cap chunk length for quadratic-attention encoders (issue
    /// #68). Whisper's fixed 30s window and zipformer's local/chunked
    /// streaming encoder need no additional cap; every other builtin
    /// architecture's encoder is full self-attention over the whole chunk,
    /// so all nine are `GlobalQuadratic` at `DEFAULT_ENCODER_SAFE_CHUNK_SECONDS`
    /// (none of the nine has an upstream-recommended value that overrides
    /// the shared default; see that constant's doc for the survey).
    #[test]
    fn builtin_architectures_declare_encoder_attention_span() {
        let expected: &[(&str, OpenAsrEncoderAttentionSpan)] = &[
            (
                COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                    max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
                },
            ),
            (
                WHISPER_GGML_ARCHITECTURE_ID,
                OpenAsrEncoderAttentionSpan::FixedWindow,
            ),
            (
                QWEN3_ASR_GGML_ARCHITECTURE_ID,
                OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                    max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
                },
            ),
            (
                PARAKEET_CTC_GGML_ARCHITECTURE_ID,
                OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                    max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
                },
            ),
            (
                PARAKEET_TDT_GGML_ARCHITECTURE_ID,
                OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                    max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
                },
            ),
            (
                WAV2VEC2_CTC_GGML_ARCHITECTURE_ID,
                OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                    max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
                },
            ),
            (
                XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
                OpenAsrEncoderAttentionSpan::LocalChunked,
            ),
            (
                MOONSHINE_GGML_ARCHITECTURE_ID,
                OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                    max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
                },
            ),
            (
                DOLPHIN_GGML_ARCHITECTURE_ID,
                OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                    max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
                },
            ),
            (
                SENSEVOICE_GGML_ARCHITECTURE_ID,
                OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                    max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
                },
            ),
            (
                FIRERED_AED_GGML_ARCHITECTURE_ID,
                OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                    max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
                },
            ),
        ];
        let registry = OpenAsrArchitectureRegistry::with_builtins();
        let mut seen = std::collections::BTreeSet::new();

        for (model_architecture, expected_span) in expected.iter().copied() {
            let descriptor = registry
                .find_by_model_architecture(model_architecture)
                .unwrap_or_else(|| panic!("missing builtin architecture '{model_architecture}'"));
            assert_eq!(
                descriptor.encoder_attention_span, expected_span,
                "'{model_architecture}' encoder_attention_span mismatch"
            );
            assert_eq!(
                descriptor.longform_max_safe_chunk_seconds(),
                match expected_span {
                    OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                        max_safe_chunk_seconds,
                    } => Some(max_safe_chunk_seconds),
                    OpenAsrEncoderAttentionSpan::FixedWindow
                    | OpenAsrEncoderAttentionSpan::LocalChunked => None,
                },
                "'{model_architecture}' longform_max_safe_chunk_seconds accessor mismatch"
            );
            seen.insert(model_architecture);
        }

        assert_eq!(
            seen.len(),
            registry.descriptors().len(),
            "expectation table must cover every builtin architecture, no more, no less"
        );
    }

    #[test]
    fn validate_references_rejects_non_finite_positive_encoder_attention_span_cap() {
        let base = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(FIRERED_AED_GGML_ARCHITECTURE_ID)
            .expect("firered architecture");

        for bad_value in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
            let descriptor = OpenAsrArchitectureDescriptor {
                encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                    max_safe_chunk_seconds: bad_value,
                },
                ..base
            };
            let error = OpenAsrArchitectureRegistry::validate_encoder_attention_span(descriptor)
                .expect_err("non-finite/non-positive max_safe_chunk_seconds must fail closed");
            // NaN != NaN under PartialEq, so match structurally instead of
            // asserting equality against a NaN-carrying expected value.
            match error {
                OpenAsrArchitectureRegistryError::EncoderAttentionSpanNotFinitePositive {
                    model_architecture,
                    max_safe_chunk_seconds,
                } => {
                    assert_eq!(model_architecture, FIRERED_AED_GGML_ARCHITECTURE_ID);
                    assert!(
                        max_safe_chunk_seconds == bad_value
                            || (max_safe_chunk_seconds.is_nan() && bad_value.is_nan())
                    );
                }
                other => panic!("unexpected error variant: {other:?}"),
            }
        }

        // A well-formed cap still validates.
        OpenAsrArchitectureRegistry::validate_encoder_attention_span(base)
            .expect("firered's real descriptor has a valid encoder_attention_span cap");
    }

    #[test]
    fn finds_architecture_by_runtime_alias() {
        let descriptor = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_runtime_architecture_alias("whisper")
            .expect("whisper alias");

        assert_eq!(descriptor.model_family, "whisper");
        assert_eq!(descriptor.model_architecture, WHISPER_GGML_ARCHITECTURE_ID);
        assert_eq!(descriptor.audio_frontend_id, WHISPER_AUDIO_FRONTEND_ID);
        assert_eq!(
            descriptor.runtime_tensor_contract_id,
            WHISPER_RUNTIME_TENSOR_CONTRACT_ID
        );
        assert_eq!(
            descriptor.executor_component_id,
            WHISPER_EXECUTOR_COMPONENT_ID
        );
    }

    #[test]
    fn finds_xasr_zipformer_architecture_by_runtime_alias() {
        let descriptor = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_runtime_architecture_alias("xasr-zh-en")
            .expect("xasr alias");

        assert_eq!(descriptor.model_family, XASR_ZIPFORMER_MODEL_FAMILY);
        assert_eq!(
            descriptor.model_architecture,
            XASR_ZIPFORMER_GGML_ARCHITECTURE_ID
        );
        assert_eq!(
            descriptor.runtime_tensor_contract_id,
            XASR_ZIPFORMER_RUNTIME_TENSOR_CONTRACT_ID
        );
        assert_eq!(
            descriptor.execution_capability,
            GgmlExecutionCapability::DedicatedRuntimeExecutorV1
        );
        assert!(descriptor.block_stack.is_none());
    }

    #[test]
    fn synthesizes_selection_defaults_from_runtime_architecture() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            GENERAL_ARCHITECTURE_KEY.to_string(),
            "qwen3-asr".to_string(),
        );

        OpenAsrArchitectureRegistry::with_builtins()
            .synthesize_selection_metadata_defaults(&mut metadata);

        assert_eq!(
            metadata.get(OASR_METADATA_KEY_MODEL_FAMILY),
            Some(&"qwen3-asr".to_string())
        );
        assert_eq!(
            metadata.get(OASR_METADATA_KEY_MODEL_ARCHITECTURE),
            Some(&QWEN3_ASR_GGML_ARCHITECTURE_ID.to_string())
        );
        assert_eq!(
            metadata.get(GGML_TOKENIZER_ID_KEY),
            Some(&QWEN3_ASR_TOKENIZER_ID.to_string())
        );
    }

    #[test]
    fn derives_ggml_family_adapter_descriptor() {
        let descriptor = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID)
            .expect("cohere architecture")
            .ggml_family_adapter_descriptor();

        assert_eq!(descriptor.adapter_id, COHERE_TRANSCRIBE_GGML_ADAPTER_ID);
        assert_eq!(descriptor.model_family, "cohere-transcribe");
        assert_eq!(
            descriptor.audio_frontend_id,
            COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID
        );
        assert_eq!(
            descriptor.execution_capability,
            GgmlExecutionCapability::DedicatedRuntimeExecutorV1
        );
    }

    #[test]
    fn ignores_unknown_runtime_architecture_aliases() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            GENERAL_ARCHITECTURE_KEY.to_string(),
            "unknown-runtime".to_string(),
        );

        OpenAsrArchitectureRegistry::with_builtins()
            .synthesize_selection_metadata_defaults(&mut metadata);

        assert_eq!(metadata.len(), 1);
    }

    #[test]
    fn builtin_architectures_have_non_empty_unique_hparam_schemas() {
        // validate_references walks each schema; this also exercises the
        // empty/duplicate guards that run at production dispatch build time.
        OpenAsrArchitectureRegistry::with_builtins()
            .validate_references()
            .expect("builtin hparam schemas must be non-empty and duplicate-free");

        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            for key in descriptor.hparam_schema {
                assert!(
                    !key.is_empty(),
                    "hparam key in architecture '{}' must be non-empty",
                    descriptor.model_architecture
                );
            }
        }
    }

    #[test]
    fn builtin_block_stacks_declare_expected_shapes() {
        let registry = OpenAsrArchitectureRegistry::with_builtins();

        let qwen = registry
            .find_by_model_architecture(QWEN3_ASR_GGML_ARCHITECTURE_ID)
            .expect("qwen architecture");
        let qwen_stack = qwen.block_stack.expect("qwen has a block stack");
        assert_eq!(
            qwen_stack.orchestration_shape,
            OpenAsrOrchestrationShape::LlmDecoder
        );
        let qwen_encoder = qwen_stack.encoder_stage.expect("qwen audio encoder stage");
        assert_eq!(
            qwen_encoder.block_kind,
            OpenAsrBlockKind::TransformerEncoderLayer
        );
        assert_eq!(qwen_encoder.layer_count_hparam, QWEN3_AUDIO_LAYERS_KEY);
        let qwen_decoder = qwen_stack.decoder_stage.expect("qwen llm decoder stage");
        assert_eq!(qwen_decoder.block_kind, OpenAsrBlockKind::LlmDecoderLayer);
        assert_eq!(qwen_decoder.layer_count_hparam, QWEN3_LLM_LAYERS_KEY);

        let cohere = registry
            .find_by_model_architecture(COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID)
            .expect("cohere architecture");
        let cohere_stack = cohere.block_stack.expect("cohere has a block stack");
        assert_eq!(
            cohere_stack.orchestration_shape,
            OpenAsrOrchestrationShape::Seq2SeqEncoderDecoder
        );
        assert_eq!(
            cohere_stack
                .encoder_stage
                .expect("cohere encoder")
                .block_kind,
            OpenAsrBlockKind::ConformerBlock
        );
        assert_eq!(
            cohere_stack
                .decoder_stage
                .expect("cohere decoder")
                .block_kind,
            OpenAsrBlockKind::Seq2SeqDecoderLayer
        );

        // whisper stays the hand-written gate and is never composed.
        let whisper = registry
            .find_by_model_architecture(WHISPER_GGML_ARCHITECTURE_ID)
            .expect("whisper architecture");
        assert!(whisper.block_stack.is_none());
    }

    #[test]
    fn block_stack_validation_rejects_layer_count_key_outside_schema() {
        let descriptor = OpenAsrArchitectureDescriptor {
            block_stack: Some(OpenAsrBlockStackDescriptor {
                orchestration_shape: OpenAsrOrchestrationShape::LlmDecoder,
                encoder_stage: None,
                decoder_stage: Some(OpenAsrStageDescriptor {
                    block_kind: OpenAsrBlockKind::LlmDecoderLayer,
                    // Not a member of QWEN3_ASR_HPARAM_SCHEMA.
                    layer_count_hparam: "qwen3-asr.llm.layers_typo",
                    tensor_name_scope: "blk",
                }),
            }),
            ..OpenAsrArchitectureRegistry::with_builtins()
                .find_by_model_architecture(QWEN3_ASR_GGML_ARCHITECTURE_ID)
                .expect("qwen architecture")
        };

        assert_eq!(
            OpenAsrArchitectureRegistry::validate_block_stack(descriptor),
            Err(
                OpenAsrArchitectureRegistryError::BlockStackLayerCountKeyNotInSchema {
                    model_architecture: QWEN3_ASR_GGML_ARCHITECTURE_ID,
                    layer_count_hparam: "qwen3-asr.llm.layers_typo",
                }
            )
        );
    }

    #[test]
    fn block_stack_validation_rejects_empty_tensor_scope() {
        let descriptor = OpenAsrArchitectureDescriptor {
            block_stack: Some(OpenAsrBlockStackDescriptor {
                orchestration_shape: OpenAsrOrchestrationShape::LlmDecoder,
                encoder_stage: None,
                decoder_stage: Some(OpenAsrStageDescriptor {
                    block_kind: OpenAsrBlockKind::LlmDecoderLayer,
                    layer_count_hparam: QWEN3_LLM_LAYERS_KEY,
                    tensor_name_scope: "",
                }),
            }),
            ..OpenAsrArchitectureRegistry::with_builtins()
                .find_by_model_architecture(QWEN3_ASR_GGML_ARCHITECTURE_ID)
                .expect("qwen architecture")
        };

        assert_eq!(
            OpenAsrArchitectureRegistry::validate_block_stack(descriptor),
            Err(
                OpenAsrArchitectureRegistryError::BlockStackEmptyTensorScope {
                    model_architecture: QWEN3_ASR_GGML_ARCHITECTURE_ID,
                }
            )
        );
    }

    #[test]
    fn block_stack_validation_rejects_decoder_kind_incompatible_with_shape() {
        // LlmDecoder shape with a Seq2SeqDecoderLayer decoder stage would route
        // the descriptor to the wrong composer once load-bearing (S5).
        let descriptor = OpenAsrArchitectureDescriptor {
            block_stack: Some(OpenAsrBlockStackDescriptor {
                orchestration_shape: OpenAsrOrchestrationShape::LlmDecoder,
                encoder_stage: None,
                decoder_stage: Some(OpenAsrStageDescriptor {
                    block_kind: OpenAsrBlockKind::Seq2SeqDecoderLayer,
                    layer_count_hparam: QWEN3_LLM_LAYERS_KEY,
                    tensor_name_scope: "blk",
                }),
            }),
            ..OpenAsrArchitectureRegistry::with_builtins()
                .find_by_model_architecture(QWEN3_ASR_GGML_ARCHITECTURE_ID)
                .expect("qwen architecture")
        };

        assert_eq!(
            OpenAsrArchitectureRegistry::validate_block_stack(descriptor),
            Err(
                OpenAsrArchitectureRegistryError::DecoderBlockKindIncompatibleWithShape {
                    model_architecture: QWEN3_ASR_GGML_ARCHITECTURE_ID,
                    orchestration_shape: OpenAsrOrchestrationShape::LlmDecoder,
                    block_kind: OpenAsrBlockKind::Seq2SeqDecoderLayer,
                }
            )
        );
    }

    #[test]
    fn block_stack_validation_rejects_encoder_kind_incompatible_with_shape() {
        // Seq2SeqEncoderDecoder shape with a TransformerEncoderLayer encoder
        // (should be ConformerBlock) is rejected.
        let descriptor = OpenAsrArchitectureDescriptor {
            block_stack: Some(OpenAsrBlockStackDescriptor {
                orchestration_shape: OpenAsrOrchestrationShape::Seq2SeqEncoderDecoder,
                encoder_stage: Some(OpenAsrStageDescriptor {
                    block_kind: OpenAsrBlockKind::TransformerEncoderLayer,
                    layer_count_hparam: COHERE_TRANSCRIBE_ENCODER_LAYERS_KEY,
                    tensor_name_scope: "enc.blk",
                }),
                decoder_stage: Some(OpenAsrStageDescriptor {
                    block_kind: OpenAsrBlockKind::Seq2SeqDecoderLayer,
                    layer_count_hparam: COHERE_TRANSCRIBE_DECODER_LAYERS_KEY,
                    tensor_name_scope: "dec.blk",
                }),
            }),
            ..OpenAsrArchitectureRegistry::with_builtins()
                .find_by_model_architecture(COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID)
                .expect("cohere architecture")
        };

        assert_eq!(
            OpenAsrArchitectureRegistry::validate_block_stack(descriptor),
            Err(
                OpenAsrArchitectureRegistryError::EncoderBlockKindIncompatibleWithShape {
                    model_architecture: COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                    orchestration_shape: OpenAsrOrchestrationShape::Seq2SeqEncoderDecoder,
                    block_kind: OpenAsrBlockKind::TransformerEncoderLayer,
                }
            )
        );
    }

    #[test]
    fn ctc_shape_accepts_sanm_fsmn_encoder_block() {
        // SenseVoice's SAN-M/FSMN encoder is a valid CTC encoder block kind
        // (encoder-only, no decoder stage). Reuse parakeet's Ctc descriptor and
        // swap in the FSMN encoder block: it must validate.
        let parakeet = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(PARAKEET_CTC_GGML_ARCHITECTURE_ID)
            .expect("parakeet architecture");
        let descriptor = OpenAsrArchitectureDescriptor {
            block_stack: Some(OpenAsrBlockStackDescriptor {
                orchestration_shape: OpenAsrOrchestrationShape::Ctc,
                encoder_stage: Some(OpenAsrStageDescriptor {
                    block_kind: OpenAsrBlockKind::SanMFsmnEncoderLayer,
                    layer_count_hparam: "parakeet.n_layers",
                    tensor_name_scope: "enc.blk",
                }),
                decoder_stage: None,
            }),
            ..parakeet
        };

        assert_eq!(
            OpenAsrArchitectureRegistry::validate_block_stack(descriptor),
            Ok(())
        );

        // And a decoder stage under the Ctc shape must still fail closed.
        let with_decoder = OpenAsrArchitectureDescriptor {
            block_stack: Some(OpenAsrBlockStackDescriptor {
                orchestration_shape: OpenAsrOrchestrationShape::Ctc,
                encoder_stage: Some(OpenAsrStageDescriptor {
                    block_kind: OpenAsrBlockKind::SanMFsmnEncoderLayer,
                    layer_count_hparam: "parakeet.n_layers",
                    tensor_name_scope: "enc.blk",
                }),
                decoder_stage: Some(OpenAsrStageDescriptor {
                    block_kind: OpenAsrBlockKind::Seq2SeqDecoderLayer,
                    layer_count_hparam: "parakeet.n_layers",
                    tensor_name_scope: "dec.blk",
                }),
            }),
            ..parakeet
        };
        assert_eq!(
            OpenAsrArchitectureRegistry::validate_block_stack(with_decoder),
            Err(
                OpenAsrArchitectureRegistryError::CtcShapeMustNotHaveDecoderStage {
                    model_architecture: PARAKEET_CTC_GGML_ARCHITECTURE_ID,
                }
            )
        );
    }

    #[test]
    fn builtin_block_stacks_pass_kind_shape_consistency() {
        // The two real composed builtins (qwen, cohere) must satisfy the S5a gate.
        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            OpenAsrArchitectureRegistry::validate_block_stack(*descriptor).unwrap_or_else(|err| {
                panic!(
                    "builtin '{}' block stack must pass kind/shape consistency: {err:?}",
                    descriptor.model_architecture
                )
            });
        }
    }

    /// S5 exit-signal acceptance test: a NEW model on an EXISTING orchestration
    /// shape is accepted as DATA ONLY — no new `OpenAsrOrchestrationShape`, no new
    /// `OpenAsrBlockKind`, no new error variant, no new `validate_*` code path, no
    /// new executor/orchestrator. It passes the S5a startup gate and routes
    /// through the same `validate_stage_against_descriptor` the real families use,
    /// with a count mismatch failing closed.
    #[test]
    fn exit_signal_new_llm_decoder_model_is_data_only() {
        use shape_orchestrator::{
            LayerCountResolver, OpenAsrStageRole, StageBuildPlan, validate_stage_against_descriptor,
        };

        const SYNTHETIC_ARCH: &str = "synthetic-llm-decoder-asr";

        // A stub resolver standing in for a new family's metadata read. Returns
        // the count the descriptor's hparam keys would resolve to.
        struct SyntheticResolver;
        impl LayerCountResolver for SyntheticResolver {
            fn resolve_layer_count(&self, hparam_key: &'static str) -> Option<usize> {
                match hparam_key {
                    QWEN3_AUDIO_LAYERS_KEY => Some(8),
                    QWEN3_LLM_LAYERS_KEY => Some(28),
                    _ => None,
                }
            }
        }

        // The ONLY thing that differs from a builtin is DATA: a new
        // model_architecture + new tensor-name scopes. Same shape, same block
        // kinds, same hparam keys (reusing qwen's schema for the test).
        let synthetic = OpenAsrArchitectureDescriptor {
            model_architecture: SYNTHETIC_ARCH,
            block_stack: Some(OpenAsrBlockStackDescriptor {
                orchestration_shape: OpenAsrOrchestrationShape::LlmDecoder,
                encoder_stage: Some(OpenAsrStageDescriptor {
                    block_kind: OpenAsrBlockKind::TransformerEncoderLayer,
                    layer_count_hparam: QWEN3_AUDIO_LAYERS_KEY,
                    tensor_name_scope: "synthetic.audio.blk",
                }),
                decoder_stage: Some(OpenAsrStageDescriptor {
                    block_kind: OpenAsrBlockKind::LlmDecoderLayer,
                    layer_count_hparam: QWEN3_LLM_LAYERS_KEY,
                    tensor_name_scope: "synthetic.blk",
                }),
            }),
            ..OpenAsrArchitectureRegistry::with_builtins()
                .find_by_model_architecture(QWEN3_ASR_GGML_ARCHITECTURE_ID)
                .expect("qwen architecture")
        };

        // 1. Passes the S5a startup gate with no new shape/kind/error.
        OpenAsrArchitectureRegistry::validate_block_stack(synthetic)
            .expect("a new LlmDecoder-shape model is valid as pure data");

        let block_stack = synthetic.block_stack.as_ref();
        let resolver = SyntheticResolver;

        // 2. Routes through the SAME load-bearing gate the real families use,
        //    for both stages, returning the descriptor-resolved counts.
        let decoder_count = validate_stage_against_descriptor(
            SYNTHETIC_ARCH,
            block_stack,
            OpenAsrStageRole::Decoder,
            OpenAsrOrchestrationShape::LlmDecoder,
            StageBuildPlan {
                block_kind: OpenAsrBlockKind::LlmDecoderLayer,
                tensor_name_scope: "synthetic.blk",
                family_layer_count: 28,
            },
            &resolver,
        )
        .expect("new model's decoder stack validates as data");
        assert_eq!(decoder_count, 28);

        let encoder_count = validate_stage_against_descriptor(
            SYNTHETIC_ARCH,
            block_stack,
            OpenAsrStageRole::Encoder,
            OpenAsrOrchestrationShape::LlmDecoder,
            StageBuildPlan {
                block_kind: OpenAsrBlockKind::TransformerEncoderLayer,
                tensor_name_scope: "synthetic.audio.blk",
                family_layer_count: 8,
            },
            &resolver,
        )
        .expect("new model's encoder stack validates as data");
        assert_eq!(encoder_count, 8);

        // 3. The gate still fails closed for the new model: a layer count that
        //    disagrees with the descriptor's hparam is rejected, no special-casing.
        let mismatch = validate_stage_against_descriptor(
            SYNTHETIC_ARCH,
            block_stack,
            OpenAsrStageRole::Decoder,
            OpenAsrOrchestrationShape::LlmDecoder,
            StageBuildPlan {
                block_kind: OpenAsrBlockKind::LlmDecoderLayer,
                tensor_name_scope: "synthetic.blk",
                family_layer_count: 27, // != the 28 the hparam resolves to
            },
            &resolver,
        );
        assert!(matches!(
            mismatch,
            Err(
                shape_orchestrator::ShapeOrchestratorError::LayerCountMismatch {
                    descriptor_count: 28,
                    family_count: 27,
                    ..
                }
            )
        ));
    }

    /// S0 (CTC onboarding): the new `Ctc` shape is encoder-only and every
    /// shape<->decoder-presence mismatch fails closed. Exercises the new variant
    /// (so it is not dead) without any model code.
    #[test]
    fn ctc_shape_block_stack_is_encoder_only_and_fail_closed() {
        use shape_orchestrator::{
            LayerCountResolver, OpenAsrStageRole, ShapeOrchestratorError, StageBuildPlan,
            validate_stage_against_descriptor,
        };
        const CTC_ARCH: &str = "synthetic-ctc-asr";
        // Any key present in the reused schema satisfies the in-schema check.
        const ENC_KEY: &str = QWEN3_AUDIO_LAYERS_KEY;

        struct CtcResolver;
        impl LayerCountResolver for CtcResolver {
            fn resolve_layer_count(&self, hparam_key: &'static str) -> Option<usize> {
                (hparam_key == ENC_KEY).then_some(24)
            }
        }

        let base = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(QWEN3_ASR_GGML_ARCHITECTURE_ID)
            .expect("qwen architecture");

        // Valid: encoder-only Ctc with a ConformerBlock encoder, no decoder stage.
        let ctc = OpenAsrArchitectureDescriptor {
            model_architecture: CTC_ARCH,
            block_stack: Some(OpenAsrBlockStackDescriptor {
                orchestration_shape: OpenAsrOrchestrationShape::Ctc,
                encoder_stage: Some(OpenAsrStageDescriptor {
                    block_kind: OpenAsrBlockKind::ConformerBlock,
                    layer_count_hparam: ENC_KEY,
                    tensor_name_scope: "enc.blk",
                }),
                decoder_stage: None,
            }),
            ..base
        };
        OpenAsrArchitectureRegistry::validate_block_stack(ctc)
            .expect("encoder-only Ctc stack is valid");

        let encoder_plan = StageBuildPlan {
            block_kind: OpenAsrBlockKind::ConformerBlock,
            tensor_name_scope: "enc.blk",
            family_layer_count: 24,
        };
        // The encoder stage drives through the SAME shared gate as data.
        assert_eq!(
            validate_stage_against_descriptor(
                CTC_ARCH,
                ctc.block_stack.as_ref(),
                OpenAsrStageRole::Encoder,
                OpenAsrOrchestrationShape::Ctc,
                encoder_plan,
                &CtcResolver,
            ),
            Ok(24)
        );
        // Driving the Decoder role on a Ctc stack fails closed.
        assert_eq!(
            validate_stage_against_descriptor(
                CTC_ARCH,
                ctc.block_stack.as_ref(),
                OpenAsrStageRole::Decoder,
                OpenAsrOrchestrationShape::Ctc,
                encoder_plan,
                &CtcResolver,
            ),
            Err(ShapeOrchestratorError::DecoderRequestedForCtcShape {
                model_architecture: CTC_ARCH,
            })
        );

        // A Ctc stack that wrongly declares a decoder stage is rejected.
        let ctc_with_decoder = OpenAsrArchitectureDescriptor {
            model_architecture: CTC_ARCH,
            block_stack: Some(OpenAsrBlockStackDescriptor {
                orchestration_shape: OpenAsrOrchestrationShape::Ctc,
                encoder_stage: Some(OpenAsrStageDescriptor {
                    block_kind: OpenAsrBlockKind::ConformerBlock,
                    layer_count_hparam: ENC_KEY,
                    tensor_name_scope: "enc.blk",
                }),
                decoder_stage: Some(OpenAsrStageDescriptor {
                    block_kind: OpenAsrBlockKind::LlmDecoderLayer,
                    layer_count_hparam: QWEN3_LLM_LAYERS_KEY,
                    tensor_name_scope: "blk",
                }),
            }),
            ..base
        };
        assert_eq!(
            OpenAsrArchitectureRegistry::validate_block_stack(ctc_with_decoder),
            Err(
                OpenAsrArchitectureRegistryError::CtcShapeMustNotHaveDecoderStage {
                    model_architecture: CTC_ARCH,
                }
            )
        );

        // An autoregressive shape missing its required decoder stage is rejected.
        let llm_without_decoder = OpenAsrArchitectureDescriptor {
            block_stack: Some(OpenAsrBlockStackDescriptor {
                orchestration_shape: OpenAsrOrchestrationShape::LlmDecoder,
                encoder_stage: Some(OpenAsrStageDescriptor {
                    block_kind: OpenAsrBlockKind::TransformerEncoderLayer,
                    layer_count_hparam: QWEN3_AUDIO_LAYERS_KEY,
                    tensor_name_scope: "audio.blk",
                }),
                decoder_stage: None,
            }),
            ..base
        };
        assert_eq!(
            OpenAsrArchitectureRegistry::validate_block_stack(llm_without_decoder),
            Err(
                OpenAsrArchitectureRegistryError::NonCtcShapeMustHaveDecoderStage {
                    model_architecture: QWEN3_ASR_GGML_ARCHITECTURE_ID,
                    orchestration_shape: OpenAsrOrchestrationShape::LlmDecoder,
                }
            )
        );
    }
}
