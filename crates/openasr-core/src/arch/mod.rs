pub(crate) mod block_stack;
pub(crate) mod hparams;
pub(crate) mod shape_orchestrator;

use std::collections::BTreeMap;

use crate::device::{
    execution_policy::{AcceleratedPlacementCapabilities, ExecutionCapabilities},
    execution_route::ExecutionProvider,
};
use crate::ggml_runtime::AutoGpuPolicy;
use crate::models::decode_policy_component_registry::{
    self as decode_policy, BuiltinDecodePolicyComponentDescriptor,
};
use crate::models::ggml_family_adapter::{
    GgmlAdapterBindingStrategy, GgmlExecutionCapability, GgmlFamilyAdapterDescriptor,
    GgmlFamilyAdapterSelectionError, GgmlFamilyAdapterSelectionFields,
    GgmlFamilyAdapterSelectionSpec, LanguageFamilyHint,
};
use crate::models::oasr_metadata::OASR_PACKAGE_VERSION_V1;
use crate::models::qwen::QWEN3_ASR_MODEL_FAMILY;
use block_stack::{
    OpenAsrBlockKind, OpenAsrBlockStackDescriptor, OpenAsrOrchestrationShape,
    OpenAsrStageDescriptor,
};
use hparams::{
    COHERE_TRANSCRIBE_DECODER_LAYERS_KEY, COHERE_TRANSCRIBE_ENCODER_LAYERS_KEY,
    COHERE_TRANSCRIBE_HPARAM_SCHEMA, DOLPHIN_HPARAM_SCHEMA, FIRERED_AED_HPARAM_SCHEMA,
    FIRERED_LLM_HPARAM_SCHEMA, FUNASR_NANO_HPARAM_SCHEMA, GRANITE_SPEECH_HPARAM_SCHEMA,
    MIMO_ASR_HPARAM_SCHEMA, MOONSHINE_HPARAM_SCHEMA, MOSS_TD_HPARAM_SCHEMA,
    PARAKEET_CTC_HPARAM_SCHEMA, PARAKEET_TDT_HPARAM_SCHEMA, QWEN3_ARCHITECTURE_VALUE,
    QWEN3_ASR_HPARAM_SCHEMA, QWEN3_AUDIO_LAYERS_KEY, QWEN3_LLM_LAYERS_KEY,
    SENSEVOICE_HPARAM_SCHEMA, WAV2VEC2_CTC_HPARAM_SCHEMA, WHISPER_HPARAM_SCHEMA,
    XASR_ZIPFORMER_HPARAM_SCHEMA,
};

pub(crate) const GENERAL_ARCHITECTURE_KEY: &str = "general.architecture";

/// Provider/placement rows shared by architecture descriptors. The field on
/// every descriptor remains mandatory; these constants only remove repetitive
/// provider spelling and do not infer support from a family name.
const CPU_AND_FULL_DEVICE_EXECUTION: ExecutionCapabilities = ExecutionCapabilities::new(true)
    .with_provider(
        ExecutionProvider::Metal,
        AcceleratedPlacementCapabilities::FULL_DEVICE,
    )
    .with_provider(
        ExecutionProvider::Cuda,
        AcceleratedPlacementCapabilities::FULL_DEVICE,
    )
    .with_provider(
        ExecutionProvider::Hip,
        AcceleratedPlacementCapabilities::FULL_DEVICE,
    )
    .with_provider(
        ExecutionProvider::Vulkan,
        AcceleratedPlacementCapabilities::FULL_DEVICE,
    );

const MOONSHINE_EXECUTION_CAPABILITIES: ExecutionCapabilities = ExecutionCapabilities::new(true)
    .with_provider(
        ExecutionProvider::Metal,
        AcceleratedPlacementCapabilities::FULL_DEVICE,
    )
    .with_provider(
        ExecutionProvider::Cuda,
        AcceleratedPlacementCapabilities::FULL_DEVICE,
    )
    .with_provider(
        ExecutionProvider::Hip,
        AcceleratedPlacementCapabilities::FULL_DEVICE,
    )
    .with_provider(
        ExecutionProvider::Vulkan,
        AcceleratedPlacementCapabilities::HYBRID,
    );

// Dolphin's Vulkan rescore is validated as a direct graph for every published
// quantization. Range-sensitive F16/Q4_K projections request the shared F32
// accumulation contract; Q8_0 keeps the faster backend default. Retain Hybrid
// as a same-device capacity fallback without weakening the preferred
// FullDevice candidate. Other providers expose only the all-device placement.
const DOLPHIN_EXECUTION_CAPABILITIES: ExecutionCapabilities = ExecutionCapabilities::new(true)
    .with_provider(
        ExecutionProvider::Metal,
        AcceleratedPlacementCapabilities::FULL_DEVICE,
    )
    .with_provider(
        ExecutionProvider::Cuda,
        AcceleratedPlacementCapabilities::FULL_DEVICE,
    )
    .with_provider(
        ExecutionProvider::Hip,
        AcceleratedPlacementCapabilities::FULL_DEVICE,
    )
    .with_provider(
        ExecutionProvider::Vulkan,
        AcceleratedPlacementCapabilities::FULL_DEVICE_AND_HYBRID,
    );

const FIRERED_LLM_EXECUTION_CAPABILITIES: ExecutionCapabilities = CPU_AND_FULL_DEVICE_EXECUTION;

pub const COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID: &str = "cohere-transcribe-conformer-transformer";
pub const COHERE_TRANSCRIBE_GGML_ADAPTER_ID: &str = "ggml-family-cohere-transcribe-runtime-v1";
pub const COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID: &str =
    "cohere-transcribe.logmel128.preemphasis.16khz.mono.v0";
pub const COHERE_TRANSCRIBE_TOKENIZER_ID: &str = "cohere-transcribe.spm.v1";
pub const COHERE_TRANSCRIBE_DECODE_POLICY_ID: &str = "cohere-transcribe.greedy.seq2seq.v1";
pub(crate) const COHERE_TRANSCRIBE_RUNTIME_TENSOR_CONTRACT_ID: &str =
    "cohere-transcribe.runtime-tensors.v1";
pub(crate) const COHERE_TRANSCRIBE_EXECUTOR_COMPONENT_ID: &str =
    "cohere-transcribe.ggml-executor.v1";

pub const WHISPER_GGML_ARCHITECTURE_ID: &str = "whisper-encoder-decoder";
pub const WHISPER_GGML_ADAPTER_ID: &str = "ggml-family-whisper-runtime-v1";
pub const WHISPER_AUDIO_FRONTEND_ID: &str = "whisper.logmel.16khz.mono.v0";
pub const WHISPER_TOKENIZER_ID: &str = "whisper.hf-bpe.v1";
pub const WHISPER_DECODE_POLICY_ID: &str = "whisper.greedy.seq2seq.v1";
pub(crate) const WHISPER_RUNTIME_TENSOR_CONTRACT_ID: &str = "whisper.runtime-tensors.v1";
pub(crate) const WHISPER_EXECUTOR_COMPONENT_ID: &str = "whisper.ggml-executor.v1";

pub const QWEN3_ASR_GGML_ARCHITECTURE_ID: &str = "qwen3-asr-encoder-decoder";
pub const QWEN3_ASR_GGML_ADAPTER_ID: &str = "ggml-family-qwen3-asr-runtime-v1";
pub const QWEN3_ASR_AUDIO_FRONTEND_ID: &str = "qwen3-asr.fbank.16khz.mono.v0";
pub const QWEN3_ASR_TOKENIZER_ID: &str = "qwen3-asr.spm.v1";
pub const QWEN3_ASR_DECODE_POLICY_ID: &str = "qwen3-asr.greedy.seq2seq.v1";
pub(crate) const QWEN3_ASR_RUNTIME_TENSOR_CONTRACT_ID: &str = "qwen3-asr.runtime-tensors.v1";
pub(crate) const QWEN3_ASR_EXECUTOR_COMPONENT_ID: &str = "qwen3-asr.ggml-executor.v1";

// parakeet-ctc (FastConformer-CTC, the goal-1 Ctc-shape onboarding).
pub const PARAKEET_CTC_GGML_ARCHITECTURE_ID: &str = "parakeet-fastconformer-ctc";
pub const PARAKEET_CTC_GGML_ADAPTER_ID: &str = "ggml-family-parakeet-ctc-runtime-v1";
pub const PARAKEET_CTC_AUDIO_FRONTEND_ID: &str = "parakeet-ctc.logmel80.16khz.mono.v0";
pub const PARAKEET_CTC_TOKENIZER_ID: &str = "parakeet-ctc.spm-bpe.v0";
pub const PARAKEET_CTC_DECODE_POLICY_ID: &str = "parakeet-ctc.greedy.ctc.v0";
pub(crate) const PARAKEET_CTC_RUNTIME_TENSOR_CONTRACT_ID: &str = "parakeet-ctc.runtime-tensors.v0";
pub(crate) const PARAKEET_CTC_EXECUTOR_COMPONENT_ID: &str = "parakeet-ctc.ggml-executor.v0";

// parakeet-tdt (FastConformer + Token-and-Duration Transducer, 25 European
// languages). Component ids are defined ahead of the full descriptor entry
// (the parakeet-ctc S2->S4 staging precedent): the importer writes them as
// pack metadata; the descriptor + executor wiring lands with the executor.
pub const PARAKEET_TDT_GGML_ARCHITECTURE_ID: &str = "parakeet-fastconformer-tdt";
pub const PARAKEET_TDT_GGML_ADAPTER_ID: &str = "ggml-family-parakeet-tdt-runtime-v1";
pub const PARAKEET_TDT_AUDIO_FRONTEND_ID: &str = "parakeet-tdt.logmel128.16khz.mono.v0";
pub const PARAKEET_TDT_TOKENIZER_ID: &str = "parakeet-tdt.spm-bpe.v0";
pub const PARAKEET_TDT_DECODE_POLICY_ID: &str = "parakeet-tdt.greedy.tdt.v0";
pub(crate) const PARAKEET_TDT_RUNTIME_TENSOR_CONTRACT_ID: &str = "parakeet-tdt.runtime-tensors.v0";
pub(crate) const PARAKEET_TDT_EXECUTOR_COMPONENT_ID: &str = "parakeet-tdt.ggml-executor.v0";

// wav2vec2-ctc (facebook/wav2vec2-base-960h, raw-waveform CTC onboarding).
pub const WAV2VEC2_CTC_GGML_ARCHITECTURE_ID: &str = "wav2vec2-ctc";
pub const WAV2VEC2_CTC_GGML_ADAPTER_ID: &str = "ggml-family-wav2vec2-ctc-runtime-v1";
pub const WAV2VEC2_CTC_AUDIO_FRONTEND_ID: &str = "wav2vec2-ctc.raw-waveform.16khz.mono.v0";
pub const WAV2VEC2_CTC_TOKENIZER_ID: &str = "wav2vec2-ctc.char.v0";
pub const WAV2VEC2_CTC_DECODE_POLICY_ID: &str = "wav2vec2-ctc.greedy.ctc.v0";
pub(crate) const WAV2VEC2_CTC_RUNTIME_TENSOR_CONTRACT_ID: &str = "wav2vec2-ctc.runtime-tensors.v0";
pub(crate) const WAV2VEC2_CTC_EXECUTOR_COMPONENT_ID: &str = "wav2vec2-ctc.ggml-executor.v0";

// X-ASR Zipformer (GilgameshWind/X-ASR-zh-en, streaming RNN-T transducer).
pub const XASR_ZIPFORMER_GGML_ARCHITECTURE_ID: &str = "xasr-zipformer-transducer";
pub const XASR_ZIPFORMER_GGML_ADAPTER_ID: &str = "ggml-family-xasr-zipformer-runtime-v1";
pub(crate) const XASR_ZIPFORMER_MODEL_FAMILY: &str = "xasr-zipformer";
pub const XASR_ZIPFORMER_AUDIO_FRONTEND_ID: &str = "xasr-zipformer.fbank80.16khz.mono.v0";
pub const XASR_ZIPFORMER_TOKENIZER_ID: &str = "xasr-zipformer.bpe.v0";
pub const XASR_ZIPFORMER_DECODE_POLICY_ID: &str = "xasr-zipformer.greedy.transducer.v0";
pub(crate) const XASR_ZIPFORMER_RUNTIME_TENSOR_CONTRACT_ID: &str =
    "xasr-zipformer.runtime-tensors.v0";
pub(crate) const XASR_ZIPFORMER_EXECUTOR_COMPONENT_ID: &str = "xasr-zipformer.ggml-executor.v0";
pub(crate) const XASR_ZIPFORMER_STREAMING_EXECUTOR_COMPONENT_ID: &str =
    "xasr-zipformer.ggml-streaming-executor.v0";

// moonshine (UsefulSensors, raw-waveform conv-stem + RoPE seq2seq encoder-decoder).
pub const MOONSHINE_GGML_ARCHITECTURE_ID: &str = "moonshine-encoder-decoder";
pub const MOONSHINE_GGML_ADAPTER_ID: &str = "ggml-family-moonshine-runtime-v1";
pub const MOONSHINE_AUDIO_FRONTEND_ID: &str = "moonshine.raw-waveform.16khz.mono.v0";
pub const MOONSHINE_TOKENIZER_ID: &str = "moonshine.spm-bpe.v0";
pub const MOONSHINE_DECODE_POLICY_ID: &str = "moonshine.greedy.seq2seq.v1";
pub(crate) const MOONSHINE_RUNTIME_TENSOR_CONTRACT_ID: &str = "moonshine.runtime-tensors.v0";
pub(crate) const MOONSHINE_EXECUTOR_COMPONENT_ID: &str = "moonshine.ggml-executor.v0";

// dolphin (WeNet E-Branchformer encoder + Transformer decoder + CTC head, char
// tokenizer, CTC/attention joint decode). Dedicated executor: the E-Branchformer
// encoder math (macaron FFN + rel-pos MHSA global branch + cgMLP/CSGU local
// branch + depthwise merge) is family-specific and not one of the composer
// block kinds, so it stays hand-written like xasr/moonshine (ArchitectureGraph).
pub const DOLPHIN_GGML_ARCHITECTURE_ID: &str = "dolphin-ebranchformer-ctc-attention";
pub const DOLPHIN_GGML_ADAPTER_ID: &str = "ggml-family-dolphin-runtime-v1";
pub(crate) const DOLPHIN_MODEL_FAMILY: &str = "dolphin";
pub const DOLPHIN_AUDIO_FRONTEND_ID: &str = "dolphin.fbank80.16khz.mono.v0";
pub const DOLPHIN_TOKENIZER_ID: &str = "dolphin.char.v0";
pub const DOLPHIN_DECODE_POLICY_ID: &str = "dolphin.attention-rescoring.v0";
pub(crate) const DOLPHIN_RUNTIME_TENSOR_CONTRACT_ID: &str = "dolphin.runtime-tensors.v0";
pub(crate) const DOLPHIN_EXECUTOR_COMPONENT_ID: &str = "dolphin.ggml-executor.v0";

// sensevoice (FunAudioLLM/SenseVoiceSmall: SAN-M/DFSMN encoder + CTC head,
// FunASR Model License v1.1). Component ids are defined ahead of the full
// architecture-descriptor entry (the parakeet S2->S4 staging precedent): the
// importer writes them as pack metadata; the descriptor + executor wiring
// lands with the executor stage.
pub const SENSEVOICE_GGML_ARCHITECTURE_ID: &str = "sensevoice-sanm-ctc";
pub const SENSEVOICE_GGML_ADAPTER_ID: &str = "ggml-family-sensevoice-runtime-v1";
pub(crate) const SENSEVOICE_MODEL_FAMILY: &str = "sensevoice";
pub const SENSEVOICE_AUDIO_FRONTEND_ID: &str = "sensevoice.fbank80-lfr7x6.16khz.mono.v0";
pub const SENSEVOICE_TOKENIZER_ID: &str = "sensevoice.spm-bpe.v0";
pub const SENSEVOICE_DECODE_POLICY_ID: &str = "sensevoice.greedy.ctc.v0";
pub(crate) const SENSEVOICE_RUNTIME_TENSOR_CONTRACT_ID: &str = "sensevoice.runtime-tensors.v0";
pub(crate) const SENSEVOICE_EXECUTOR_COMPONENT_ID: &str = "sensevoice.ggml-executor.v0";

// firered-aed (FireRedTeam/FireRedASR-AED-L: Conformer encoder + Transformer
// decoder attention-based encoder-decoder, no CTC branch, Apache-2.0). The
// Conformer encoder math (macaron FFN + rel-pos MHSA with independent q/k/v
// LayerNorms + GLU/depthwise conv) is family-specific, so like dolphin/
// moonshine/xasr it stays on a hand-written dedicated executor
// (ArchitectureGraph) rather than the data-driven composer.
pub(crate) const FIRERED_AED_GGML_ARCHITECTURE_ID: &str = "firered-conformer-aed";
pub(crate) const FIRERED_AED_GGML_ADAPTER_ID: &str = "ggml-family-firered-aed-runtime-v1";
pub(crate) const FIRERED_AED_MODEL_FAMILY: &str = "firered-aed";
pub(crate) const FIRERED_AED_AUDIO_FRONTEND_ID: &str = "firered-aed.fbank80.16khz.mono.v0";
pub(crate) const FIRERED_AED_TOKENIZER_ID: &str = "firered-aed.char-spm.v0";
pub(crate) const FIRERED_AED_DECODE_POLICY_ID: &str = "firered-aed.greedy.seq2seq.v0";
pub(crate) const FIRERED_AED_RUNTIME_TENSOR_CONTRACT_ID: &str = "firered-aed.runtime-tensors.v0";
pub(crate) const FIRERED_AED_EXECUTOR_COMPONENT_ID: &str = "firered-aed.ggml-executor.v0";

// firered-llm (FireRedTeam/FireRedASR2-LLM: the firered-aed Conformer encoder
// (independently-trained weights, NOT byte-identical to firered-aed-l-v2 --
// see scratchpad/fr2/T1-findings.md S3, joint finetune not frozen-encoder
// reuse) + a 2x frame-stacking Adapter (2 Linear + ReLU) + a LoRA-merged
// Qwen2-7B-Instruct decoder, Apache-2.0). Like firered-aed, decode runs on a
// hand-written dedicated executor (ArchitectureGraph) -- the Conformer
// encoder + Qwen2 decoder shapes are family-specific, not composer block
// kinds -- represented by the canonical architecture inventory below.
pub(crate) const FIRERED_LLM_GGML_ARCHITECTURE_ID: &str = "firered-llm-conformer-adapter-qwen2";
pub(crate) const FIRERED_LLM_GGML_ADAPTER_ID: &str = "ggml-family-firered-llm-runtime-v1";
pub(crate) const FIRERED_LLM_MODEL_FAMILY: &str = "firered2-llm";
pub(crate) const FIRERED_LLM_AUDIO_FRONTEND_ID: &str = "firered-llm.fbank80.16khz.mono.v0";
pub(crate) const FIRERED_LLM_TOKENIZER_ID: &str = "firered-llm.qwen2-bpe.v0";
pub(crate) const FIRERED_LLM_DECODE_POLICY_ID: &str = "firered-llm.greedy.seq2seq.v0";
pub(crate) const FIRERED_LLM_RUNTIME_TENSOR_CONTRACT_ID: &str = "firered-llm.runtime-tensors.v0";
pub(crate) const FIRERED_LLM_EXECUTOR_COMPONENT_ID: &str = "firered-llm.ggml-executor.v0";

// funasr-nano (FunAudioLLM/Fun-ASR-Nano-2512: a FunASR SAN-M/DFSMN audio encoder
// (50 enc + 20 tp blocks, LayerNorm eps 1e-5) + a 2-layer transformer adaptor
// (512->2048->1024 MLP + 2 standard transformer blocks) + a stock Qwen3-0.6B
// decoder (QK-norm, no attention bias, GQA, tied embeddings), Apache-2.0). The
// release checkpoint carries no CTC decoder (a training-only branch), so decode
// runs on a hand-written dedicated executor (ArchitectureGraph) -- represented
// by the canonical architecture inventory below.
pub(crate) const FUNASR_NANO_GGML_ARCHITECTURE_ID: &str = "funasr-nano-sanm-adapter-qwen3";
pub(crate) const FUNASR_NANO_GGML_ADAPTER_ID: &str = "ggml-family-funasr-nano-runtime-v1";
pub(crate) const FUNASR_NANO_MODEL_FAMILY: &str = "funasr-nano";
pub(crate) const FUNASR_NANO_AUDIO_FRONTEND_ID: &str = "funasr-nano.fbank80-lfr.16khz.mono.v0";
pub(crate) const FUNASR_NANO_TOKENIZER_ID: &str = "funasr-nano.qwen3-bpe.v0";
pub(crate) const FUNASR_NANO_DECODE_POLICY_ID: &str = "funasr-nano.greedy.seq2seq.v0";
pub(crate) const FUNASR_NANO_RUNTIME_TENSOR_CONTRACT_ID: &str = "funasr-nano.runtime-tensors.v0";
pub(crate) const FUNASR_NANO_EXECUTOR_COMPONENT_ID: &str = "funasr-nano.ggml-executor.v0";

// mimo-asr (XiaomiMiMo/MiMo-V2.5-ASR + XiaomiMiMo/MiMo-Audio-Tokenizer: a 32L
// rope audio-tokenizer encoder + RVQ encode + 6L bidirectional input-local
// transformer feeding a 36L Qwen2 backbone, MIT). Every stage (skip@L3 conv
// stem, RVQ residual quantization, per-group input-local batching) is
// family-specific, so like firered-aed/firered-llm it stays on a
// hand-written dedicated executor (ArchitectureGraph).
pub(crate) const MIMO_ASR_GGML_ARCHITECTURE_ID: &str = "mimo-asr";
pub(crate) const MIMO_ASR_GGML_ADAPTER_ID: &str = "ggml-family-mimo-asr-runtime-v1";
pub(crate) const MIMO_ASR_MODEL_FAMILY: &str = "mimo-asr";
pub(crate) const MIMO_ASR_AUDIO_FRONTEND_ID: &str = "mimo-tokenizer-rvq-v0";
pub(crate) const MIMO_ASR_TOKENIZER_ID: &str = "mimo-asr.gpt2-bpe.v0";
pub(crate) const MIMO_ASR_DECODE_POLICY_ID: &str = "mimo-asr.greedy.seq2seq.v0";
pub(crate) const MIMO_ASR_RUNTIME_TENSOR_CONTRACT_ID: &str = "mimo-asr.runtime-tensors.v0";
pub(crate) const MIMO_ASR_EXECUTOR_COMPONENT_ID: &str = "mimo-asr.ggml-executor.v0";

// moss-transcribe-diarize (OpenMOSS/MOSS-Transcribe-Diarize, 0.9B): a
// Whisper-Medium-architecture audio encoder (standard HF `WhisperEncoder`,
// reuses the shared `crate::nn::encoder::transformer_layer` "Whisper /
// Qwen-audio encoder shape" primitive `qwen::audio_encoder` also builds on)
// + a pure-reshape 4x time-merge + `VQAdaptor` (a plain 3-layer MLP+LayerNorm
// despite the name -- no VQ codebook) + a genuinely Qwen3-0.6B-parameterized
// decoder (QK-norm, no attention bias, GQA, tied embeddings), reusing
// `qwen`'s family-agnostic decoder machinery byte-for-byte (see
// `models::moss_transcribe_diarize::llm_decoder`'s module doc). Like
// firered-llm/mimo-asr, decode runs on a hand-written dedicated executor
// (ArchitectureGraph) -- represented by the canonical architecture inventory
// below.
pub(crate) const MOSS_TD_GGML_ARCHITECTURE_ID: &str = "moss-transcribe-diarize-whisper-qwen3";
pub(crate) const MOSS_TD_GGML_ADAPTER_ID: &str = "ggml-family-moss-transcribe-diarize-runtime-v1";
pub(crate) const MOSS_TD_MODEL_FAMILY: &str = "moss-transcribe-diarize";
pub(crate) const MOSS_TD_TARGET_INVOCATION_SECONDS: u32 = 30;
pub(crate) const MOSS_TD_MAX_INVOCATION_SECONDS: u32 = 60;
pub(crate) const MOSS_TD_AUDIO_FRONTEND_ID: &str = "moss-transcribe-diarize.fbank80.16khz.mono.v0";
pub(crate) const MOSS_TD_TOKENIZER_ID: &str = "moss-transcribe-diarize.qwen3-bpe.v0";
pub(crate) const MOSS_TD_DECODE_POLICY_ID: &str = "moss-transcribe-diarize.greedy.seq2seq.v0";
pub(crate) const MOSS_TD_RUNTIME_TENSOR_CONTRACT_ID: &str =
    "moss-transcribe-diarize.runtime-tensors.v0";
pub(crate) const MOSS_TD_EXECUTOR_COMPONENT_ID: &str = "moss-transcribe-diarize.ggml-executor.v0";

// granite-speech (ibm-granite/granite-speech-4.1-2b, Apache-2.0): a 16-layer
// Conformer CTC encoder (Shaw relative-position block-local attention,
// self-conditioned CTC mid-layer tap) + a BLIP-2 Q-Former window projector
// (new component -- see `models::granite_speech::qformer`'s module doc on why
// it stays family-local for now) + a dense Granite decoder-only LLM (GQA,
// RoPE, SwiGLU, RMSNorm, plus four Granite-specific scaling scalars --
// attention/embedding/residual multipliers and logits scaling -- modeled
// faithfully rather than folded into the shared qwen decoder stack, see
// `models::granite_speech::decoder_graph`'s module doc). Like firered-aed/
// firered-llm/mimo-asr, none of the three stages are composer block kinds,
// so this stays on a hand-written dedicated executor (ArchitectureGraph).
pub(crate) const GRANITE_SPEECH_GGML_ARCHITECTURE_ID: &str = "granite-speech";
pub(crate) const GRANITE_SPEECH_GGML_ADAPTER_ID: &str = "ggml-family-granite-speech-runtime-v1";
pub(crate) const GRANITE_SPEECH_MODEL_FAMILY: &str = "granite-speech";
/// Largest whole-second direct invocation whose exact centered-STFT,
/// frame-stack and padded Q-Former token shape fits the 4096-position decoder
/// together with the shipped prompt and decode budget. The exact integer
/// derivation is pinned in `models::granite_speech::capacity`.
pub(crate) const GRANITE_SPEECH_MAX_INVOCATION_SECONDS: u32 = 381;
pub(crate) const GRANITE_SPEECH_AUDIO_FRONTEND_ID: &str = "granite-speech.mel80x2.16khz.mono.v0";
pub(crate) const GRANITE_SPEECH_TOKENIZER_ID: &str = "granite-speech.gpt2-bpe.v0";
pub(crate) const GRANITE_SPEECH_DECODE_POLICY_ID: &str = "granite-speech.greedy.seq2seq.v0";
pub(crate) const GRANITE_SPEECH_RUNTIME_TENSOR_CONTRACT_ID: &str =
    "granite-speech.runtime-tensors.v0";
pub(crate) const GRANITE_SPEECH_EXECUTOR_COMPONENT_ID: &str = "granite-speech.ggml-executor.v0";

/// Default chunk length long-form slicing aims for: how long a slice we
/// *want*, as a transcription-quality choice.
///
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
/// **This is a decision knob, and the evidence above supports nothing else.**
/// Six model cards agreeing on a good chunk length says how these encoders
/// were trained and where they transcribe well. It says nothing about how
/// much RAM their activations need, which is a property of the host, not of
/// the corpus anyone trained on. See
/// [`DEFAULT_ENCODER_SAFE_CHUNK_SECONDS`] for the separate memory ceiling
/// that used to borrow this citation, and do not re-unify them.
pub(crate) const DEFAULT_ENCODER_CHUNK_SECONDS: f32 = 30.0;

/// Product envelope for VAD pause-seeking around the 30-second target.
/// A segment may extend to a natural boundary, but one decoder invocation
/// may never carry more than 60 seconds of content. This is semantic input to
/// decoder-state planning, not a memory-pressure fallback: an execution
/// candidate that cannot host the declared envelope is rejected or moved to
/// another approved placement without changing the slice plan.
///
/// The 30/60 target/maximum pair is a release-quality contract. Changes must
/// pass the long-form CER/DER/seam A/B gate; the capacity planner merely sizes
/// the chosen contract exactly and does not choose it.
pub(crate) const DEFAULT_ENCODER_MAX_CHUNK_SECONDS: f32 = 60.0;

/// Default `GlobalQuadratic` **memory** ceiling (issue #68) -- the longest
/// chunk a global-quadratic encoder may be handed before its attention
/// activations are a risk on commodity RAM. Every new `GlobalQuadratic`
/// builtin should declare this unless the upstream model publishes a
/// different explicit recommendation (see firered-aed's descriptor below,
/// whose upstream guidance -- 60s-warn/200s-error -- is wider; it still uses
/// this default, and says so in its own comment).
///
/// # Why this is not [`DEFAULT_ENCODER_CHUNK_SECONDS`]
///
/// It was, and that is a role confusion, not a coincidence worth preserving.
/// The two answer different questions -- "how long a slice transcribes well"
/// versus "how long a slice fits in memory" -- with different units of
/// evidence: the first is settled by model cards, the second by activation
/// footprint against the host's available RAM. Sharing one symbol had two
/// concrete costs. The clamp's `chunk_seconds` arm became unreachable on the
/// default path (the value being clamped *was* the ceiling), so the ceiling
/// was never actually exercised as a ceiling. And the arm that does fire --
/// `max_chunk_seconds` (then 120s by default) -- silently collapsed the slicer's
/// entire elasticity band onto 30s, taking away its room to hunt for a real
/// pause, on the authority of a memory argument that was never made.
///
/// **INVARIANT: this must never be defined as, or derived from,
/// [`DEFAULT_ENCODER_CHUNK_SECONDS`].** A quality convention cannot certify a
/// memory bound.
///
/// # Why the value is still 30.0
///
/// Honestly: because no better figure has been established yet, and 30s is
/// the conservative direction. The defensible derivation is from the
/// architecture itself -- `GlobalQuadratic` activation grows as
/// `frames^2 x heads x layers x dtype_width` per attention layer, so the safe
/// frame count follows from the host's available RAM -- but that needs a
/// measured per-family peak-activation coefficient, which this repo does not
/// have. Until it does, the number stays put; what changed is that it now has
/// its own name and its own justification to fail, instead of borrowing one
/// that never applied.
pub(crate) const DEFAULT_ENCODER_SAFE_CHUNK_SECONDS: f32 = 30.0;

/// Where a family's speaker structure ("which turn belongs to which of the
/// people speaking in this recording") comes from.
///
/// This is the *separation source* only. It says nothing about whether the
/// user asked for speakers (that is the request-level Voice ID switch,
/// `TranscriptionRequest::voice_id`) and nothing about identity (turning a
/// recording-local turn label into a known person is the Voice ID matching
/// stage in `crate::diarize::voice_id`, which runs on top of whichever source
/// produced the turns). Keeping the three apart is what lets a self-segmenting
/// family reuse its own stable tracks while both source types still converge
/// on the same required ReDim acoustic identity and enrolled-person matcher.
///
/// The variants are mutually exclusive by construction: exactly one source
/// produces the turns for a given transcription, so no second pass can
/// overwrite the first's labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerSegmentationSource {
    /// The family's own decode carries the speaker structure: cohere emits a
    /// `<|spltoken0|>` control-token stream, moss-transcribe-diarize writes
    /// inline `[t][Sxx]` markers as ordinary transcript characters. The family
    /// normalizes its own markup into labeled [`crate::Segment`]s (parsing
    /// stays under `models/<family>/`); the shared layer never sees the raw
    /// markup.
    InDecoder,
    /// The family emits plain transcripts, so speaker structure has to come
    /// from a separate segmenter over the same audio: today the model-agnostic
    /// neural VAD + ReDimNet2-B6 speaker-embedder clustering path, and (next)
    /// the pyannote segmenter, which plugs in at the same
    /// `crate::diarize::contract::SpeakerTimeline` boundary without any family
    /// needing to change.
    External,
}

impl SpeakerSegmentationSource {
    pub fn is_in_decoder(self) -> bool {
        matches!(self, Self::InDecoder)
    }
}

/// Where this architecture obtains the word anchors needed to project a
/// transcript onto an external speaker timeline without losing speaker
/// changes. This is an architecture capability, not a request preference:
/// executors declaring `Native` must populate `Segment.words` when requested;
/// `ForcedAligner` families require the shared alignment capability pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WordTimestampSource {
    Native,
    ForcedAligner,
}

/// How one recording is cut up for this architecture before decode -- the
/// single declaration of the family's longform *shape*, read by
/// `native_transcribe::resolve_native_longform_policy_for_backend`.
///
/// The slicing itself (VAD cut-point search, lead-in/overlap, timeline
/// mapping, overlap dedup, transcript assembly) is entirely model-agnostic and
/// lives in [`crate::longform`]; a family never implements any of it. All this
/// declares is which of those model-agnostic shapes fits, so adding a family
/// is one field, not new slicing code.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum OpenAsrLongformSliceShape {
    /// The shared slicer's generic window serves this family, and its speaker
    /// structure (if any) comes from one whole-recording external pass, so
    /// slices are never their own speaker scope. Every family whose
    /// `speaker_segmentation` is
    /// [`SpeakerSegmentationSource::External`] is this shape.
    SharedWindow,
    /// Slices are decoded independently *and* each carries its own speaker
    /// numbering, because the family diarizes in-decoder
    /// ([`SpeakerSegmentationSource::InDecoder`]). Two slices' `SPEAKER_01`
    /// are therefore unrelated labels, so every slice becomes its own
    /// [`crate::diarize::voice_id::SpeakerScope`] and cross-slice identity is
    /// re-established from voice evidence alone.
    ///
    /// Such a family also declares its own product-tested invocation envelope.
    /// `target_seconds` is the window the slicer aims for and `max_seconds` is
    /// the ceiling it may stretch a slice to while searching for a clean VAD
    /// cut. The family decoder-state topology must prove that every invocation
    /// through `max_seconds` fits its position ceilings. Memory availability
    /// may select a different execution placement, but must never rewrite this
    /// window and thereby change transcript semantics across machines.
    ///
    /// `integral_seconds` is the largest recording decoded whole before the
    /// slicer is engaged. It is a quality/product threshold inside the proven
    /// envelope, not a value reverse-engineered from free memory or the RoPE
    /// ceiling. Every seam restarts in-decoder speaker numbering; optional
    /// Voice ID can reconcile those recording-local labels across scopes.
    /// Changes to these three values therefore require CER/DER/seam-quality
    /// evidence as well as a passing decoder-state capacity proof.
    ScopedSlices {
        integral_seconds: f32,
        target_seconds: f32,
        max_seconds: f32,
    },
}

/// Semantic audio span accepted by one family executor invocation.
///
/// This is deliberately independent of both long-form product preferences
/// (`chunk_seconds` / `max_chunk_seconds`) and encoder activation-memory
/// scaling ([`OpenAsrEncoderAttentionSpan`]). A bounded family must never be
/// handed a larger buffer: some runtimes reject it, while a fixed-window
/// frontend such as Whisper would otherwise silently trim real audio. The
/// shared slicer clamps to this contract before dispatch; the family runtime
/// remains the fail-closed backstop for direct `LongFormMode::Off` requests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum OpenAsrInvocationSpan {
    /// The architecture has no duration-only semantic bound. Token/position
    /// ceilings and exact frontend shape oracles may still reject a request.
    Elastic,
    /// One invocation accepts at most this many seconds of prepared audio.
    Bounded { max_seconds: f32 },
}

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

/// Partial-result granularity of a family's streaming executor. Infrastructure
/// property (how partials are produced), not a per-model semantic: `FrameSync`
/// appends fixed low-latency chunks and never revises already-emitted text;
/// `Buffered` re-decodes a growing/windowed buffer and may revise prior
/// partials. Declared once on the architecture descriptor and
/// derived into the streaming dispatch at build time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingPartialGranularity {
    FrameSync,
    Buffered,
}

/// Which decode driver owns the family's token loop. A dedicated topology is
/// never an unlabeled escape hatch: its mathematical reason is required in
/// the same inventory row and remains subject to shared execution fences and
/// conformance gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenAsrDecodeDriverStrategy {
    SharedSeq2SeqGreedy {
        policy: BuiltinDecodePolicyComponentDescriptor,
    },
    SharedCtcGreedy {
        policy: BuiltinDecodePolicyComponentDescriptor,
    },
    Dedicated {
        decode_policy_id: &'static str,
        reason: &'static str,
    },
}

impl OpenAsrDecodeDriverStrategy {
    pub(crate) const fn decode_policy_id(self) -> &'static str {
        match self {
            Self::SharedSeq2SeqGreedy { policy } | Self::SharedCtcGreedy { policy } => {
                policy.decode_policy_id
            }
            Self::Dedicated {
                decode_policy_id, ..
            } => decode_policy_id,
        }
    }

    pub(crate) const fn shared_policy(self) -> Option<BuiltinDecodePolicyComponentDescriptor> {
        match self {
            Self::SharedSeq2SeqGreedy { policy } | Self::SharedCtcGreedy { policy } => Some(policy),
            Self::Dedicated { .. } => None,
        }
    }
}

/// Whether graph construction is data-composed from shared blocks or remains
/// architecture-specific because the current composer cannot express its
/// mathematical topology. The latter requires a reason instead of `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenAsrBlockStackStrategy {
    Shared(OpenAsrBlockStackDescriptor),
    ArchitectureGraph { reason: &'static str },
}

/// Pack-import surface for one native family. File existence alone is not
/// enough: `CoreConvert` symbols must be force-linked by the inventory
/// projection in `models::pack_import_surface`, and `ExternalTooling` paths
/// must resolve under the repo root.
#[derive(Debug, Clone, Copy)]
pub(crate) enum OpenAsrPackImportSurface {
    CoreConvert {
        symbol: &'static str,
        force_link: fn(),
    },
    ExternalTooling {
        relative_path: &'static str,
    },
}

/// Identity facts for one native family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenAsrDialectCapability {
    /// This family does not advertise region-qualified recognition codes.
    NotAdvertised,
    /// Catalog entries may advertise registered dialects the model recognizes,
    /// but callers cannot select one as a decode parameter.
    RecognizesCatalogDeclared,
    /// Callers may select exactly these dialect codes through the family's
    /// prompt protocol. The slice is the executable prompt-map domain, not a
    /// marketing coverage list.
    SelectsViaPrompt { codes: &'static [&'static str] },
}

// Canonical base-language coverage for each builtin family. These are data
// facts, not policy hints: the inventory projection exports them verbatim and
// tooling must not maintain a second family-language table.
const COHERE_RECOGNIZED_LANGUAGES: &[&str] = &[
    "ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh",
];
const WHISPER_RECOGNIZED_LANGUAGES: &[&str] = &[
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da",
    "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw", "he",
    "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "la",
    "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl",
    "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
    "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi",
    "yi", "yo", "zh",
];
const QWEN_RECOGNIZED_LANGUAGES: &[&str] = &[
    "ar", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fil", "fr", "hi", "hu", "id", "it",
    "ja", "ko", "mk", "ms", "nl", "pl", "pt", "ro", "ru", "sv", "th", "tr", "vi", "zh",
];
const ENGLISH_RECOGNIZED_LANGUAGES: &[&str] = &["en"];
const PARAKEET_TDT_RECOGNIZED_LANGUAGES: &[&str] = &[
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt",
    "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
];
const BILINGUAL_RECOGNIZED_LANGUAGES: &[&str] = &["en", "zh"];
const DOLPHIN_RECOGNIZED_LANGUAGES: &[&str] = &["zh"];
const SENSEVOICE_RECOGNIZED_LANGUAGES: &[&str] = &["en", "ja", "ko", "yue", "zh"];
const MIMO_RECOGNIZED_LANGUAGES: &[&str] = &["en", "yue", "zh"];
const MOSS_RECOGNIZED_LANGUAGES: &[&str] = &[
    "de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "th", "tl", "tr", "ur", "vi", "zh",
];
const GRANITE_SPEECH_RECOGNIZED_LANGUAGES: &[&str] = &["de", "en", "es", "fr", "ja", "pt"];

/// Identity facts for one native family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OpenAsrIdentityContract {
    pub runtime_architecture_aliases: &'static [&'static str],
    pub model_family: &'static str,
    pub model_architecture: &'static str,
    pub adapter_id: &'static str,
    pub catalog_family_id: &'static str,
    pub module_slug: &'static str,
    pub recognized_languages: &'static [&'static str],
    pub language_family_hint: LanguageFamilyHint,
    pub dialect_capability: OpenAsrDialectCapability,
}

/// Pack and importer facts for one native family.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OpenAsrPackContract {
    pub audio_frontend_id: &'static str,
    pub runtime_tensor_contract_id: &'static str,
    pub tokenizer_id: &'static str,
    pub hparam_schema: &'static [&'static str],
    pub pack_import: OpenAsrPackImportSurface,
    pub runtime_validator: fn(&crate::GgufRuntimeSourcePreflight) -> Result<(), String>,
}

/// Semantic topology of an architecture's token-scaled persistent decoder state.
///
/// This is an onboarding declaration, not a formula registry. The executor
/// still owns the family-specific position oracle and stable stream ids, while
/// startup/CI cross-checks the complete semantic shape. A new autoregressive
/// family therefore cannot accidentally omit cross-attention state, claim a
/// causal-only layout, or bypass capacity planning entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenAsrDecoderStateTopology {
    None,
    CausalSelfAttentionKv,
    EncoderDecoderSelfAndCrossAttentionKv,
    /// Escape hatch for a genuinely different, explicitly family-owned
    /// multi-stream topology. Built-in ASR families should prefer a precise
    /// shared variant whenever one describes their state.
    FamilyDefinedTokenScaledPersistent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAsrPhraseBiasStrategy {
    Unsupported,
    Always,
    RequiresTensor { tensor_name: &'static str },
}

impl OpenAsrPhraseBiasStrategy {
    pub(crate) const fn is_structurally_supported(self) -> bool {
        !matches!(self, Self::Unsupported)
    }
}

/// Whether forcing word anchors may alter a family's decode numerics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAsrWordTimestampStrategy {
    DecodeInvariant,
    DecodeSensitive,
}

/// Which reusable prepared-runtime component materializes resident model state.
/// Family identity is deliberately absent: another family with identical
/// computation may select an existing component without adding a central
/// architecture match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenAsrPreparedRuntimeStrategy {
    FamilyOwned,
    SharedCohereTranscribeV1,
    SharedQwen3AsrV1,
}

/// Runtime execution facts for one native family.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OpenAsrExecutionContract {
    pub executor_component_id: &'static str,
    pub runtime_factory: fn() -> crate::models::executor_component_registry::BuiltinExecutorHandle,
    pub execution_capability: GgmlExecutionCapability,
    pub execution_capabilities: ExecutionCapabilities,
    pub phrase_bias: OpenAsrPhraseBiasStrategy,
    pub supports_translation_task: bool,
    pub supports_source_language_hint: bool,
    /// Concrete LoRA/OADP binding the executor must implement. Dispatch
    /// cross-checks this value against the materialized executor, so a family
    /// cannot self-certify support by toggling a boolean.
    pub adapter_binding: GgmlAdapterBindingStrategy,
    pub prepared_runtime: OpenAsrPreparedRuntimeStrategy,
    pub word_timestamps: OpenAsrWordTimestampStrategy,
    pub streaming_partial_granularity: StreamingPartialGranularity,
    pub speaker_segmentation: SpeakerSegmentationSource,
    /// Source of usable word anchors for transcript/speaker attribution.
    pub word_timestamp_source: WordTimestampSource,
    pub longform_slice_shape: OpenAsrLongformSliceShape,
    pub(crate) invocation_span: OpenAsrInvocationSpan,
    pub emits_punctuation: Option<bool>,
}

/// Decoder topology and shared-driver selection for one native family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OpenAsrTopologyContract {
    pub decoder_state_topology: OpenAsrDecoderStateTopology,
    pub decode_driver: OpenAsrDecodeDriverStrategy,
    pub block_stack: OpenAsrBlockStackStrategy,
}

/// Required strategy and deep runtime optimization facts for one native family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OpenAsrOptimizationContract {
    pub prefer_cpu_decoder_for_multichunk_metal: bool,
    pub auto_gpu_policy: crate::ggml_runtime::AutoGpuPolicy,
    pub encoder_attention_span: OpenAsrEncoderAttentionSpan,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OpenAsrQuantizationContract {
    pub tensor_classification: crate::models::pack_quant::TensorQuantizationContract,
}

/// Typed skeleton-fixture policy for the production runtime-ready skeleton
/// gate. Each architecture row declares its own fixture builder here; the
/// gate iterates the inventory and consumes this facet, so no family list
/// exists at the gate. Resolving a kind into a concrete fixture spec lives
/// next to the builders in the testing module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SkeletonFixtureKind {
    CohereTranscribe,
    Whisper,
    Qwen3Asr,
    ParakeetCtc,
    ParakeetTdt,
    Wav2Vec2Ctc,
    XasrZipformer,
    Moonshine,
    Dolphin,
    SenseVoice,
    FireRedAed,
    FireRed2Llm,
    FunasrNano,
    MimoAsr,
    MossTranscribeDiarize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OpenAsrConformanceContract {
    pub profile_id: &'static str,
    pub reference_dumper_source: Option<&'static str>,
    /// Why this family is deliberately exempt from the production
    /// runtime-ready skeleton gate (`shared_runtime_ready_family_skeletons...`),
    /// which otherwise must cover every family fail-closed. `None` means the
    /// family is required to have a runtime-ready skeleton fixture; a `Some`
    /// reason is the only way out, so a new family cannot silently skip the
    /// gate. Keep the reason specific enough to audit.
    pub skeleton_exemption: Option<&'static str>,
    /// This family's runtime-ready skeleton fixture builder, consumed by the
    /// inventory-driven skeleton gate. Required unless `skeleton_exemption`
    /// carries an audit reason; a row with neither fails the gate.
    pub skeleton_fixture: Option<SkeletonFixtureKind>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OpenAsrArchitectureDescriptor {
    pub identity: OpenAsrIdentityContract,
    pub pack_contract: OpenAsrPackContract,
    pub execution_contract: OpenAsrExecutionContract,
    pub topology_contract: OpenAsrTopologyContract,
    pub optimization_contract: OpenAsrOptimizationContract,
    pub quantization_contract: OpenAsrQuantizationContract,
    pub conformance_contract: OpenAsrConformanceContract,
}

impl OpenAsrArchitectureDescriptor {
    pub(crate) fn max_single_invocation_seconds(self) -> Option<f32> {
        match self.execution_contract.invocation_span {
            OpenAsrInvocationSpan::Elastic => None,
            OpenAsrInvocationSpan::Bounded { max_seconds } => Some(max_seconds),
        }
    }

    /// The longform chunk-length safety cap this architecture's encoder
    /// tolerates, if any (`None` when the encoder needs no additional cap --
    /// `FixedWindow`/`LocalChunked`). See [`OpenAsrEncoderAttentionSpan`].
    pub(crate) fn longform_max_safe_chunk_seconds(self) -> Option<f32> {
        match self.optimization_contract.encoder_attention_span {
            OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                max_safe_chunk_seconds,
            } => Some(max_safe_chunk_seconds),
            OpenAsrEncoderAttentionSpan::FixedWindow
            | OpenAsrEncoderAttentionSpan::LocalChunked => None,
        }
    }

    #[cfg(test)]
    fn matches_runtime_architecture_alias(&self, alias: &str) -> bool {
        self.identity
            .runtime_architecture_aliases
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(alias))
    }

    pub(crate) fn ggml_family_adapter_descriptor(self) -> GgmlFamilyAdapterDescriptor {
        GgmlFamilyAdapterDescriptor {
            adapter_id: self.identity.adapter_id,
            language_family_hint: self.identity.language_family_hint,
            model_family: self.identity.model_family,
            model_architecture: self.identity.model_architecture,
            audio_frontend_id: self.pack_contract.audio_frontend_id,
            tokenizer_id: self.pack_contract.tokenizer_id,
            decode_policy_id: self.topology_contract.decode_driver.decode_policy_id(),
            execution_capability: self.execution_contract.execution_capability,
            execution_capabilities: self.execution_contract.execution_capabilities,
            adapter_binding: self.execution_contract.adapter_binding,
            speaker_segmentation: self.execution_contract.speaker_segmentation,
            phrase_bias: self.execution_contract.phrase_bias,
            word_timestamps: self.execution_contract.word_timestamps,
            word_timestamp_source: self.execution_contract.word_timestamp_source,
        }
    }
}

/// Resolve one builtin GGML adapter from the canonical architecture registry.
/// The architecture id is the only lookup key; adapter metadata is derived
/// from the same descriptor rather than a second adapter registry.
pub(crate) fn builtin_adapter_descriptor(model_architecture: &str) -> GgmlFamilyAdapterDescriptor {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(model_architecture)
        .unwrap_or_else(|| panic!("builtin architecture '{model_architecture}' must be registered"))
        .ggml_family_adapter_descriptor()
}

/// Whether a builtin family's decoder ever predicts a punctuation token (see
/// [`OpenAsrExecutionContract::emits_punctuation`]), looked up by GGUF
/// `model_architecture`. The shared offline/streaming punctuation stages
/// consume this single inventory accessor, so the capability cannot drift into
/// a second hand-maintained table.
pub(crate) fn emits_punctuation_for_model_architecture(model_architecture: &str) -> Option<bool> {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(model_architecture)
        .and_then(|descriptor| descriptor.execution_contract.emits_punctuation)
}

/// How one recording is cut up for a builtin family before decode, looked up
/// by GGUF `model_architecture`. An unrecognized architecture gets
/// [`OpenAsrLongformSliceShape::SharedWindow`], the shape that needs nothing
/// from the family beyond a plain decode.
pub(crate) fn longform_slice_shape_for_model_architecture(
    model_architecture: &str,
) -> OpenAsrLongformSliceShape {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(model_architecture)
        .map(|descriptor| descriptor.execution_contract.longform_slice_shape)
        .unwrap_or(OpenAsrLongformSliceShape::SharedWindow)
}

/// Which GPU-class backend(s) a builtin family's Auto execution may select
/// automatically (see
/// [`OpenAsrArchitectureDescriptor::auto_gpu_policy`]), looked up by GGUF
/// `model_architecture`. An unrecognized architecture defaults to
/// `AutoGpuPolicy::AllBackends` (the majority behavior: Auto uses GPU when
/// available) rather than silently pinning an unknown family to CPU. This is
/// the accessor a provenance/telemetry label must call -- with the result
/// fed into `GgmlCpuGraphConfig::resolve_family_runtime_backend` -- so the
/// reported backend can never drift from what the family's own executor
/// actually decided.
pub(crate) fn family_auto_gpu_policy_for_model_architecture(
    model_architecture: &str,
) -> crate::ggml_runtime::AutoGpuPolicy {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(model_architecture)
        .map(|descriptor| descriptor.optimization_contract.auto_gpu_policy)
        .unwrap_or(crate::ggml_runtime::AutoGpuPolicy::AllBackends)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum OpenAsrArchitectureRegistryError {
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
    /// A bounded invocation contract must be usable as a slicer ceiling.
    InvocationSpanNotFinitePositive {
        model_architecture: &'static str,
        max_seconds: f32,
    },
    /// A scoped-slice product envelope must also bound direct/off calls.
    ScopedSliceInvocationSpanMismatch {
        model_architecture: &'static str,
        required_seconds: f32,
        invocation_max_seconds: Option<f32>,
    },
    /// A family descriptor must use the quantization contract owned by the
    /// same architecture.  This prevents a copied prefix/role classifier from
    /// silently applying another family's Q8 floor policy.
    QuantizationArchitectureMismatch {
        model_architecture: &'static str,
        quantization_architecture: &'static str,
    },
    /// A family module slug is the stable join key for generated projections;
    /// it must be a non-empty snake_case identifier.
    ModuleSlugNotSnakeCase {
        model_architecture: &'static str,
        module_slug: &'static str,
    },
    /// Generated projections must have exactly one owner module per slug.
    DuplicateModuleSlug {
        module_slug: &'static str,
        first_model_architecture: &'static str,
        duplicate_model_architecture: &'static str,
    },
    /// Every canonical architecture owns exactly one GGML adapter id.
    DuplicateAdapterId {
        adapter_id: &'static str,
        first_model_architecture: &'static str,
        duplicate_model_architecture: &'static str,
    },
    /// GGUF `general.architecture` is the canonical architecture join key.
    DuplicateModelArchitecture {
        model_architecture: &'static str,
        first_adapter_id: &'static str,
        duplicate_adapter_id: &'static str,
    },
    /// A family must advertise at least one canonical base recognition language.
    RecognizedLanguagesEmpty {
        model_architecture: &'static str,
    },
    /// Canonical language facts are serialized in strict lexical order with no
    /// duplicates so every projection has deterministic bytes.
    RecognizedLanguagesNotSortedUnique {
        model_architecture: &'static str,
        language: &'static str,
    },
    /// Inventory language facts accept only lowercase ISO 639 base codes.
    RecognizedLanguageMalformed {
        model_architecture: &'static str,
        language: &'static str,
    },
    /// A fixed-monolingual hint and its canonical language fact disagree.
    FixedMonolingualLanguageMismatch {
        model_architecture: &'static str,
        language: &'static str,
    },
    /// A fixed-multilingual hint and its canonical language fact disagree.
    FixedMultilingualLanguagesMismatch {
        model_architecture: &'static str,
    },
    /// A prompt-selected family must include its default in the advertised set.
    PromptDefaultLanguageNotRecognized {
        model_architecture: &'static str,
        default_language: &'static str,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OpenAsrArchitectureRegistry {
    architectures: &'static [OpenAsrArchitectureDescriptor],
}

impl OpenAsrArchitectureRegistry {
    pub(crate) fn with_builtins() -> Self {
        Self {
            architectures: BUILTIN_ARCHITECTURE_DESCRIPTORS,
        }
    }

    pub(crate) fn descriptors(self) -> &'static [OpenAsrArchitectureDescriptor] {
        self.architectures
    }

    #[cfg(test)]
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
            .find(|descriptor| descriptor.identity.model_architecture == architecture_id)
    }

    pub(crate) fn find_by_adapter_id(
        self,
        adapter_id: &str,
    ) -> Option<OpenAsrArchitectureDescriptor> {
        self.architectures
            .iter()
            .copied()
            .find(|descriptor| descriptor.identity.adapter_id == adapter_id)
    }

    /// Select a canonical builtin GGML adapter from parsed package metadata.
    /// Selection walks architecture descriptors directly so adapter identity,
    /// component ids, and metadata matching cannot drift into a second table.
    pub(crate) fn select_ggml_adapter(
        self,
        spec: &GgmlFamilyAdapterSelectionSpec<'_>,
    ) -> Result<OpenAsrArchitectureDescriptor, GgmlFamilyAdapterSelectionError> {
        let fields = spec
            .parse_selection_fields()
            .map_err(GgmlFamilyAdapterSelectionError::InvalidMetadata)?;
        self.select_ggml_adapter_from_fields(&fields)
    }

    pub(crate) fn select_ggml_adapter_from_gguf_metadata_v1(
        self,
        metadata: &BTreeMap<String, String>,
    ) -> Result<OpenAsrArchitectureDescriptor, GgmlFamilyAdapterSelectionError> {
        let spec = GgmlFamilyAdapterSelectionSpec::from_gguf_metadata_v1(metadata);
        self.select_ggml_adapter(&spec)
    }

    pub(crate) fn select_ggml_adapter_from_fields(
        self,
        fields: &GgmlFamilyAdapterSelectionFields<'_>,
    ) -> Result<OpenAsrArchitectureDescriptor, GgmlFamilyAdapterSelectionError> {
        Self::select_ggml_adapter_from_descriptors(self.architectures, fields)
    }

    /// Shared selection core used by the builtin registry and its
    /// duplicate/ambiguity conformance tests.
    pub(crate) fn select_ggml_adapter_from_descriptors(
        descriptors: &[OpenAsrArchitectureDescriptor],
        fields: &GgmlFamilyAdapterSelectionFields<'_>,
    ) -> Result<OpenAsrArchitectureDescriptor, GgmlFamilyAdapterSelectionError> {
        if fields.package_version != OASR_PACKAGE_VERSION_V1 {
            return Err(GgmlFamilyAdapterSelectionError::UnsupportedPackageVersion {
                expected: OASR_PACKAGE_VERSION_V1,
                found: fields.package_version.to_string(),
            });
        }

        if !descriptors
            .iter()
            .any(|descriptor| descriptor.identity.model_family == fields.model_family)
        {
            return Err(GgmlFamilyAdapterSelectionError::UnknownFamily {
                model_family: fields.model_family.to_string(),
            });
        }

        let matches: Vec<_> = descriptors
            .iter()
            .filter(|descriptor| {
                descriptor
                    .ggml_family_adapter_descriptor()
                    .matches_selection_fields(fields)
            })
            .collect();

        match matches.as_slice() {
            [descriptor] => Ok(**descriptor),
            [] => Err(GgmlFamilyAdapterSelectionError::NoMatchingAdapter {
                model_family: fields.model_family.to_string(),
                model_architecture: fields.model_architecture.to_string(),
                audio_frontend_id: fields.audio_frontend_id.to_string(),
                decode_policy_id: fields.decode_policy_id.to_string(),
                tokenizer_id: fields.tokenizer_id.map(str::to_string),
            }),
            _ => Err(GgmlFamilyAdapterSelectionError::Ambiguous {
                adapter_ids: matches
                    .iter()
                    .map(|descriptor| descriptor.identity.adapter_id)
                    .collect(),
            }),
        }
    }

    pub(crate) fn validate_references(self) -> Result<(), OpenAsrArchitectureRegistryError> {
        for descriptor in self.architectures {
            Self::validate_identity(*descriptor)?;
            Self::validate_hparam_schema(*descriptor)?;
            Self::validate_block_stack(*descriptor)?;
            Self::validate_invocation_span(*descriptor)?;
            Self::validate_encoder_attention_span(*descriptor)?;
            Self::validate_quantization_contract(*descriptor)?;
        }
        Self::validate_module_slug_uniqueness(self.architectures)?;
        Self::validate_adapter_uniqueness(self.architectures)?;
        Ok(())
    }

    fn validate_adapter_uniqueness(
        descriptors: &[OpenAsrArchitectureDescriptor],
    ) -> Result<(), OpenAsrArchitectureRegistryError> {
        for (index, descriptor) in descriptors.iter().enumerate() {
            if let Some(first) = descriptors[..index]
                .iter()
                .find(|candidate| candidate.identity.adapter_id == descriptor.identity.adapter_id)
            {
                return Err(OpenAsrArchitectureRegistryError::DuplicateAdapterId {
                    adapter_id: descriptor.identity.adapter_id,
                    first_model_architecture: first.identity.model_architecture,
                    duplicate_model_architecture: descriptor.identity.model_architecture,
                });
            }
            if let Some(first) = descriptors[..index].iter().find(|candidate| {
                candidate.identity.model_architecture == descriptor.identity.model_architecture
            }) {
                return Err(
                    OpenAsrArchitectureRegistryError::DuplicateModelArchitecture {
                        model_architecture: descriptor.identity.model_architecture,
                        first_adapter_id: first.identity.adapter_id,
                        duplicate_adapter_id: descriptor.identity.adapter_id,
                    },
                );
            }
        }
        Ok(())
    }

    fn validate_module_slug_uniqueness(
        descriptors: &[OpenAsrArchitectureDescriptor],
    ) -> Result<(), OpenAsrArchitectureRegistryError> {
        for (index, descriptor) in descriptors.iter().enumerate() {
            if let Some(first) = descriptors[..index]
                .iter()
                .find(|candidate| candidate.identity.module_slug == descriptor.identity.module_slug)
            {
                return Err(OpenAsrArchitectureRegistryError::DuplicateModuleSlug {
                    module_slug: descriptor.identity.module_slug,
                    first_model_architecture: first.identity.model_architecture,
                    duplicate_model_architecture: descriptor.identity.model_architecture,
                });
            }
        }
        Ok(())
    }

    fn validate_identity(
        descriptor: OpenAsrArchitectureDescriptor,
    ) -> Result<(), OpenAsrArchitectureRegistryError> {
        let identity = descriptor.identity;
        let slug = identity.module_slug;
        let slug_bytes = slug.as_bytes();
        if slug_bytes.is_empty()
            || !slug_bytes[0].is_ascii_lowercase()
            || slug_bytes[slug_bytes.len() - 1] == b'_'
            || slug_bytes.windows(2).any(|window| window == b"__")
            || slug_bytes
                .iter()
                .any(|byte| !byte.is_ascii_lowercase() && !byte.is_ascii_digit() && *byte != b'_')
        {
            return Err(OpenAsrArchitectureRegistryError::ModuleSlugNotSnakeCase {
                model_architecture: identity.model_architecture,
                module_slug: slug,
            });
        }

        let languages = identity.recognized_languages;
        if languages.is_empty() {
            return Err(OpenAsrArchitectureRegistryError::RecognizedLanguagesEmpty {
                model_architecture: identity.model_architecture,
            });
        }
        for (index, language) in languages.iter().enumerate() {
            let bytes = language.as_bytes();
            if !(2..=3).contains(&bytes.len())
                || bytes.iter().any(|byte| !byte.is_ascii_lowercase())
            {
                return Err(
                    OpenAsrArchitectureRegistryError::RecognizedLanguageMalformed {
                        model_architecture: identity.model_architecture,
                        language,
                    },
                );
            }
            if index > 0 && languages[index - 1] >= *language {
                return Err(
                    OpenAsrArchitectureRegistryError::RecognizedLanguagesNotSortedUnique {
                        model_architecture: identity.model_architecture,
                        language,
                    },
                );
            }
        }

        match identity.language_family_hint {
            LanguageFamilyHint::FixedMonolingual { language }
                if languages.len() != 1 || languages[0] != language =>
            {
                return Err(
                    OpenAsrArchitectureRegistryError::FixedMonolingualLanguageMismatch {
                        model_architecture: identity.model_architecture,
                        language,
                    },
                );
            }
            LanguageFamilyHint::FixedMultilingual {
                languages: expected_languages,
            } if languages != expected_languages => {
                return Err(
                    OpenAsrArchitectureRegistryError::FixedMultilingualLanguagesMismatch {
                        model_architecture: identity.model_architecture,
                    },
                );
            }
            LanguageFamilyHint::SelectsViaPrompt { default_language }
                if !languages.contains(&default_language) =>
            {
                return Err(
                    OpenAsrArchitectureRegistryError::PromptDefaultLanguageNotRecognized {
                        model_architecture: identity.model_architecture,
                        default_language,
                    },
                );
            }
            _ => {}
        }
        Ok(())
    }

    fn validate_quantization_contract(
        descriptor: OpenAsrArchitectureDescriptor,
    ) -> Result<(), OpenAsrArchitectureRegistryError> {
        let model_architecture = descriptor.identity.model_architecture;
        let quantization_architecture = descriptor
            .quantization_contract
            .tensor_classification
            .model_architecture();
        if model_architecture != quantization_architecture {
            return Err(
                OpenAsrArchitectureRegistryError::QuantizationArchitectureMismatch {
                    model_architecture,
                    quantization_architecture,
                },
            );
        }
        Ok(())
    }

    fn validate_hparam_schema(
        descriptor: OpenAsrArchitectureDescriptor,
    ) -> Result<(), OpenAsrArchitectureRegistryError> {
        if descriptor.pack_contract.hparam_schema.is_empty() {
            return Err(OpenAsrArchitectureRegistryError::EmptyHparamSchema {
                model_architecture: descriptor.identity.model_architecture,
            });
        }
        for (index, key) in descriptor.pack_contract.hparam_schema.iter().enumerate() {
            if descriptor.pack_contract.hparam_schema[..index].contains(key) {
                return Err(OpenAsrArchitectureRegistryError::DuplicateHparamKey {
                    model_architecture: descriptor.identity.model_architecture,
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
        } = descriptor.optimization_contract.encoder_attention_span
            && !(max_safe_chunk_seconds.is_finite() && max_safe_chunk_seconds > 0.0)
        {
            return Err(
                OpenAsrArchitectureRegistryError::EncoderAttentionSpanNotFinitePositive {
                    model_architecture: descriptor.identity.model_architecture,
                    max_safe_chunk_seconds,
                },
            );
        }
        Ok(())
    }

    fn validate_invocation_span(
        descriptor: OpenAsrArchitectureDescriptor,
    ) -> Result<(), OpenAsrArchitectureRegistryError> {
        if let OpenAsrInvocationSpan::Bounded { max_seconds } =
            descriptor.execution_contract.invocation_span
            && !(max_seconds.is_finite() && max_seconds > 0.0)
        {
            return Err(
                OpenAsrArchitectureRegistryError::InvocationSpanNotFinitePositive {
                    model_architecture: descriptor.identity.model_architecture,
                    max_seconds,
                },
            );
        }
        if let OpenAsrLongformSliceShape::ScopedSlices {
            integral_seconds,
            max_seconds,
            ..
        } = descriptor.execution_contract.longform_slice_shape
        {
            let required_seconds = integral_seconds.max(max_seconds);
            let invocation_max_seconds = descriptor.max_single_invocation_seconds();
            if invocation_max_seconds.is_none_or(|limit| limit < required_seconds) {
                return Err(
                    OpenAsrArchitectureRegistryError::ScopedSliceInvocationSpanMismatch {
                        model_architecture: descriptor.identity.model_architecture,
                        required_seconds,
                        invocation_max_seconds,
                    },
                );
            }
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
        let OpenAsrBlockStackStrategy::Shared(block_stack) =
            descriptor.topology_contract.block_stack
        else {
            return Ok(());
        };
        for stage in block_stack.stages() {
            if stage.tensor_name_scope.is_empty() {
                return Err(
                    OpenAsrArchitectureRegistryError::BlockStackEmptyTensorScope {
                        model_architecture: descriptor.identity.model_architecture,
                    },
                );
            }
            if !descriptor
                .pack_contract
                .hparam_schema
                .contains(&stage.layer_count_hparam)
            {
                return Err(
                    OpenAsrArchitectureRegistryError::BlockStackLayerCountKeyNotInSchema {
                        model_architecture: descriptor.identity.model_architecture,
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
                        model_architecture: descriptor.identity.model_architecture,
                    },
                );
            }
            (Some(_), None) => {
                return Err(
                    OpenAsrArchitectureRegistryError::NonCtcShapeMustHaveDecoderStage {
                        model_architecture: descriptor.identity.model_architecture,
                        orchestration_shape: block_stack.orchestration_shape,
                    },
                );
            }
            (Some(expected_decoder_kind), Some(decoder_stage))
                if decoder_stage.block_kind != expected_decoder_kind =>
            {
                return Err(
                    OpenAsrArchitectureRegistryError::DecoderBlockKindIncompatibleWithShape {
                        model_architecture: descriptor.identity.model_architecture,
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
                    model_architecture: descriptor.identity.model_architecture,
                    orchestration_shape: block_stack.orchestration_shape,
                    block_kind: encoder_stage.block_kind,
                },
            );
        }
        Ok(())
    }
}

const BUILTIN_ARCHITECTURE_DESCRIPTORS: &[OpenAsrArchitectureDescriptor] = &[
    OpenAsrArchitectureDescriptor {
        identity: OpenAsrIdentityContract {
            runtime_architecture_aliases: &["cohere-transcribe"],
            model_family: "cohere-transcribe",
            model_architecture: COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            adapter_id: COHERE_TRANSCRIBE_GGML_ADAPTER_ID,
            catalog_family_id: "cohere",
            module_slug: "cohere",
            recognized_languages: COHERE_RECOGNIZED_LANGUAGES,
            language_family_hint: LanguageFamilyHint::SelectsViaPrompt {
                default_language: "en",
            },
            dialect_capability: OpenAsrDialectCapability::NotAdvertised,
        },
        pack_contract: OpenAsrPackContract {
            audio_frontend_id: COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID,
            runtime_tensor_contract_id: COHERE_TRANSCRIBE_RUNTIME_TENSOR_CONTRACT_ID,
            tokenizer_id: COHERE_TRANSCRIBE_TOKENIZER_ID,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_cohere_source_to_runtime_pack",
                force_link: || {
                    let _ = std::hint::black_box(
                        crate::models::cohere::convert_local_cohere_source_to_runtime_pack
                            as *const (),
                    );
                },
            },
            hparam_schema: COHERE_TRANSCRIBE_HPARAM_SCHEMA,
            runtime_validator:
                crate::models::cohere::runtime_contract::validate_runtime_pack_contract,
        },
        execution_contract: OpenAsrExecutionContract {
            executor_component_id: COHERE_TRANSCRIBE_EXECUTOR_COMPONENT_ID,
            runtime_factory:
                crate::models::executor_component_registry::materialize_builtin_executor::<
                    crate::models::cohere::CohereTranscribeGgmlExecutor,
                >,
            execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
            execution_capabilities: CPU_AND_FULL_DEVICE_EXECUTION,
            phrase_bias: OpenAsrPhraseBiasStrategy::Always,
            supports_translation_task: false,
            supports_source_language_hint: true,
            adapter_binding: GgmlAdapterBindingStrategy::Unsupported,
            prepared_runtime: OpenAsrPreparedRuntimeStrategy::SharedCohereTranscribeV1,
            word_timestamps: OpenAsrWordTimestampStrategy::DecodeInvariant,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            speaker_segmentation: SpeakerSegmentationSource::External,
            word_timestamp_source: WordTimestampSource::Native,
            longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
            invocation_span: OpenAsrInvocationSpan::Elastic,
            emits_punctuation: Some(true),
        },
        topology_contract: OpenAsrTopologyContract {
            decoder_state_topology:
                OpenAsrDecoderStateTopology::EncoderDecoderSelfAndCrossAttentionKv,
            decode_driver: OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy {
                policy: decode_policy::COHERE_TRANSCRIBE_DECODE_POLICY_COMPONENT,
            },
            block_stack: OpenAsrBlockStackStrategy::Shared(OpenAsrBlockStackDescriptor {
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
        },
        optimization_contract: OpenAsrOptimizationContract {
            prefer_cpu_decoder_for_multichunk_metal: true,
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
            // The decoder does have a speaker-token mode (`<|diarize|>` ->
            // `<|spltoken0|>` stream), but no published cohere pack can run it --
            // enabling it needs re-converted, re-published packs. Declaring
            // `External` is the honest state: cohere gets speakers from the
            // model-agnostic segmentation path if one is installed, and reports
            // the capability as unsupported if not, instead of advertising an
            // in-decoder mode that would fail at decode time. Flip this (and
            // restore `models::cohere::prompt`'s control-token switch) in the same
            // change that ships packs carrying the tokens.
            encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
            },
        },
        quantization_contract: OpenAsrQuantizationContract {
            tensor_classification: crate::models::cohere::TENSOR_QUANTIZATION_CONTRACT,
        },
        conformance_contract: OpenAsrConformanceContract {
            profile_id: "cohere",
            reference_dumper_source: None,
            skeleton_exemption: None,
            skeleton_fixture: Some(SkeletonFixtureKind::CohereTranscribe),
        },
    },
    OpenAsrArchitectureDescriptor {
        identity: OpenAsrIdentityContract {
            runtime_architecture_aliases: &["whisper"],
            model_family: "whisper",
            model_architecture: WHISPER_GGML_ARCHITECTURE_ID,
            adapter_id: WHISPER_GGML_ADAPTER_ID,
            catalog_family_id: "whisper",
            module_slug: "whisper",
            recognized_languages: WHISPER_RECOGNIZED_LANGUAGES,
            language_family_hint: LanguageFamilyHint::WhisperVocabGated,
            dialect_capability: OpenAsrDialectCapability::NotAdvertised,
        },
        pack_contract: OpenAsrPackContract {
            audio_frontend_id: WHISPER_AUDIO_FRONTEND_ID,
            runtime_tensor_contract_id: WHISPER_RUNTIME_TENSOR_CONTRACT_ID,
            tokenizer_id: WHISPER_TOKENIZER_ID,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_whisper_hf_source_to_runtime_pack",
                force_link: || {
                    let _ = std::hint::black_box(
                        crate::models::whisper::convert_local_whisper_hf_source_to_runtime_pack
                            as *const (),
                    );
                },
            },
            hparam_schema: WHISPER_HPARAM_SCHEMA,
            runtime_validator:
                crate::models::whisper::runtime_contract::validate_runtime_pack_contract,
            // whisper remains the hand-written bit-level regression gate and is
            // never composed — no block-stack data until P9 sinks its optimizations
            // into the shared blocks.
        },
        execution_contract: OpenAsrExecutionContract {
            executor_component_id: WHISPER_EXECUTOR_COMPONENT_ID,
            runtime_factory:
                crate::models::executor_component_registry::materialize_builtin_executor::<
                    crate::models::whisper::WhisperGgmlExecutor,
                >,
            execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
            execution_capabilities: CPU_AND_FULL_DEVICE_EXECUTION,
            phrase_bias: OpenAsrPhraseBiasStrategy::Always,
            supports_translation_task: true,
            supports_source_language_hint: true,
            adapter_binding: GgmlAdapterBindingStrategy::Unsupported,
            prepared_runtime: OpenAsrPreparedRuntimeStrategy::FamilyOwned,
            word_timestamps: OpenAsrWordTimestampStrategy::DecodeSensitive,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            speaker_segmentation: SpeakerSegmentationSource::External,
            word_timestamp_source: WordTimestampSource::Native,
            longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
            invocation_span: OpenAsrInvocationSpan::Bounded { max_seconds: 30.0 },
            emits_punctuation: Some(true),
        },
        topology_contract: OpenAsrTopologyContract {
            decoder_state_topology:
                OpenAsrDecoderStateTopology::EncoderDecoderSelfAndCrossAttentionKv,
            decode_driver: OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy {
                policy: decode_policy::WHISPER_DECODE_POLICY_COMPONENT,
            },
            block_stack: OpenAsrBlockStackStrategy::ArchitectureGraph {
                reason: "Whisper's convolutional frontend and fixed-window encoder/decoder graph are not represented by the shared block composer.",
            },
            // Architecture-fixed 30s log-mel window: the encoder never sees more
            // than a fixed span no matter how long the requested longform chunk
            // is, so it needs no additional longform safety cap.
        },
        optimization_contract: OpenAsrOptimizationContract {
            prefer_cpu_decoder_for_multichunk_metal: false,
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
            encoder_attention_span: OpenAsrEncoderAttentionSpan::FixedWindow,
        },
        quantization_contract: OpenAsrQuantizationContract {
            tensor_classification: crate::models::whisper::TENSOR_QUANTIZATION_CONTRACT,
        },
        conformance_contract: OpenAsrConformanceContract {
            profile_id: "whisper",
            reference_dumper_source: None,
            skeleton_exemption: None,
            skeleton_fixture: Some(SkeletonFixtureKind::Whisper),
        },
    },
    OpenAsrArchitectureDescriptor {
        identity: OpenAsrIdentityContract {
            runtime_architecture_aliases: &[QWEN3_ARCHITECTURE_VALUE],
            model_family: QWEN3_ASR_MODEL_FAMILY,
            model_architecture: QWEN3_ASR_GGML_ARCHITECTURE_ID,
            adapter_id: QWEN3_ASR_GGML_ADAPTER_ID,
            catalog_family_id: "qwen",
            module_slug: "qwen",
            recognized_languages: QWEN_RECOGNIZED_LANGUAGES,
            language_family_hint: LanguageFamilyHint::SelfDetectsRejectsHint {
                // Qwen3-ASR conditions language via free text in the chat prompt (no
                // language tokens in its vocab) and does not expose the language it
                // auto-detects. Until that text conditioning is wired and verified
                // against a real pack, an explicit hint is rejected (not faked) and
                // the detected language is reported as null. See docs/KNOWN_LIMITATIONS.md.
                reject_reason: "Qwen3-ASR auto-detects the source language and does not accept an explicit selection; use a multilingual Whisper pack to force or report a language.",
            },
            dialect_capability: OpenAsrDialectCapability::RecognizesCatalogDeclared,
        },
        pack_contract: OpenAsrPackContract {
            audio_frontend_id: QWEN3_ASR_AUDIO_FRONTEND_ID,
            runtime_tensor_contract_id: QWEN3_ASR_RUNTIME_TENSOR_CONTRACT_ID,
            tokenizer_id: QWEN3_ASR_TOKENIZER_ID,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_qwen_source_to_runtime_pack",
                force_link: || {
                    let _ = std::hint::black_box(
                        crate::models::qwen::convert_local_qwen_source_to_runtime_pack as *const (),
                    );
                },
            },
            hparam_schema: QWEN3_ASR_HPARAM_SCHEMA,
            runtime_validator:
                crate::models::qwen::runtime_contract::validate_runtime_pack_contract,
        },
        execution_contract: OpenAsrExecutionContract {
            executor_component_id: QWEN3_ASR_EXECUTOR_COMPONENT_ID,
            runtime_factory:
                crate::models::executor_component_registry::materialize_builtin_executor::<
                    crate::models::qwen::Qwen3AsrGgmlExecutor,
                >,
            execution_capability: GgmlExecutionCapability::NativeGraphLoweringV1,
            execution_capabilities: CPU_AND_FULL_DEVICE_EXECUTION,
            phrase_bias: OpenAsrPhraseBiasStrategy::Always,
            supports_translation_task: false,
            supports_source_language_hint: false,
            adapter_binding: GgmlAdapterBindingStrategy::Qwen3AsrLoraV1,
            prepared_runtime: OpenAsrPreparedRuntimeStrategy::SharedQwen3AsrV1,
            word_timestamps: OpenAsrWordTimestampStrategy::DecodeInvariant,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            speaker_segmentation: SpeakerSegmentationSource::External,
            word_timestamp_source: WordTimestampSource::Native,
            longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
            invocation_span: OpenAsrInvocationSpan::Elastic,
            emits_punctuation: Some(true),
        },
        topology_contract: OpenAsrTopologyContract {
            decoder_state_topology: OpenAsrDecoderStateTopology::CausalSelfAttentionKv,
            decode_driver: OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy {
                policy: decode_policy::QWEN3_ASR_DECODE_POLICY_COMPONENT,
            },
            block_stack: OpenAsrBlockStackStrategy::Shared(OpenAsrBlockStackDescriptor {
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
        },
        optimization_contract: OpenAsrOptimizationContract {
            prefer_cpu_decoder_for_multichunk_metal: false,
            // Left un-gated (`AllBackends`) for now: the measured 1.71x Metal
            // slowdown at qwen's recommended 1.7B @ q8_0 config looks like a
            // fixed size x quant platform trade-off rather than a qwen-specific
            // bug (see `models::qwen::graph_config`'s doc comment), but that
            // read is not yet confirmed by a dedicated follow-up investigation,
            // so it is deliberately not baked into the default here. Flip to
            // `ExceptMetal` once that follow-up lands (one-line change, the gate
            // machinery already exists).
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
            encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
            },
        },
        quantization_contract: OpenAsrQuantizationContract {
            tensor_classification: crate::models::qwen::TENSOR_QUANTIZATION_CONTRACT,
        },
        conformance_contract: OpenAsrConformanceContract {
            profile_id: "qwen",
            reference_dumper_source: Some("tooling/qwen-reference-dumper/dump_golden.py"),
            skeleton_exemption: None,
            skeleton_fixture: Some(SkeletonFixtureKind::Qwen3Asr),
        },
    },
    OpenAsrArchitectureDescriptor {
        identity: OpenAsrIdentityContract {
            runtime_architecture_aliases: &["parakeet-ctc", "parakeet"],
            model_family: "parakeet-ctc",
            model_architecture: PARAKEET_CTC_GGML_ARCHITECTURE_ID,
            adapter_id: PARAKEET_CTC_GGML_ADAPTER_ID,
            catalog_family_id: "parakeet",
            module_slug: "parakeet_ctc",
            recognized_languages: ENGLISH_RECOGNIZED_LANGUAGES,
            language_family_hint: LanguageFamilyHint::FixedMonolingual { language: "en" },
            dialect_capability: OpenAsrDialectCapability::NotAdvertised,
        },
        pack_contract: OpenAsrPackContract {
            audio_frontend_id: PARAKEET_CTC_AUDIO_FRONTEND_ID,
            runtime_tensor_contract_id: PARAKEET_CTC_RUNTIME_TENSOR_CONTRACT_ID,
            tokenizer_id: PARAKEET_CTC_TOKENIZER_ID,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_parakeet_ctc_source_to_runtime_pack",
                force_link: || {
                    let _ = std::hint::black_box(
                        crate::models::parakeet_ctc::convert_local_parakeet_ctc_source_to_runtime_pack
                            as *const (),
                    );
                },
            },
            hparam_schema: PARAKEET_CTC_HPARAM_SCHEMA,
            runtime_validator:
                crate::models::parakeet_ctc::runtime_contract::validate_runtime_pack_contract,
            // Non-autoregressive CTC: encoder + CTC head only, no decoder stage.
        },
        execution_contract: OpenAsrExecutionContract {
            executor_component_id: PARAKEET_CTC_EXECUTOR_COMPONENT_ID,
            runtime_factory:
                crate::models::executor_component_registry::materialize_builtin_executor::<
                    crate::models::parakeet_ctc::executor::ParakeetCtcGgmlExecutor,
                >,
            execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
            execution_capabilities: CPU_AND_FULL_DEVICE_EXECUTION,
            phrase_bias: OpenAsrPhraseBiasStrategy::Always,
            supports_translation_task: false,
            supports_source_language_hint: false,
            adapter_binding: GgmlAdapterBindingStrategy::Unsupported,
            prepared_runtime: OpenAsrPreparedRuntimeStrategy::FamilyOwned,
            word_timestamps: OpenAsrWordTimestampStrategy::DecodeInvariant,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            speaker_segmentation: SpeakerSegmentationSource::External,
            word_timestamp_source: WordTimestampSource::Native,
            longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
            invocation_span: OpenAsrInvocationSpan::Elastic,
            // Character/BPE CTC: whether an imported checkpoint's vocab includes
            // punctuation depends on that specific checkpoint's training corpus,
            // not the architecture, so this cannot be stated as a fixed
            // per-family fact. The generated inventory therefore exports an
            // unclaimed value for catalog tooling.
            emits_punctuation: None,
        },
        topology_contract: OpenAsrTopologyContract {
            decoder_state_topology: OpenAsrDecoderStateTopology::None,
            decode_driver: OpenAsrDecodeDriverStrategy::SharedCtcGreedy {
                policy: decode_policy::PARAKEET_CTC_DECODE_POLICY_COMPONENT,
            },
            block_stack: OpenAsrBlockStackStrategy::Shared(OpenAsrBlockStackDescriptor {
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
        },
        optimization_contract: OpenAsrOptimizationContract {
            prefer_cpu_decoder_for_multichunk_metal: false,
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
            encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
            },
        },
        quantization_contract: OpenAsrQuantizationContract {
            tensor_classification: crate::models::parakeet_ctc::package_import::TENSOR_QUANTIZATION_CONTRACT,
        },
        conformance_contract: OpenAsrConformanceContract {
            profile_id: "parakeet",
            reference_dumper_source: None,
            skeleton_exemption: None,
            skeleton_fixture: Some(SkeletonFixtureKind::ParakeetCtc),
        },
    },
    OpenAsrArchitectureDescriptor {
        identity: OpenAsrIdentityContract {
            runtime_architecture_aliases: &["parakeet-tdt"],
            model_family: "parakeet-tdt",
            model_architecture: PARAKEET_TDT_GGML_ARCHITECTURE_ID,
            adapter_id: PARAKEET_TDT_GGML_ADAPTER_ID,
            // parakeet-tdt-0.6b-v3: 25 European languages, no per-request language
            // selection (the model decodes whatever it hears; NVIDIA's card lists
            // the fixed set).
            catalog_family_id: "parakeet-tdt",
            module_slug: "parakeet_tdt",
            recognized_languages: PARAKEET_TDT_RECOGNIZED_LANGUAGES,
            language_family_hint: LanguageFamilyHint::FixedMultilingual {
                languages: PARAKEET_TDT_RECOGNIZED_LANGUAGES,
            },
            dialect_capability: OpenAsrDialectCapability::NotAdvertised,
        },
        pack_contract: OpenAsrPackContract {
            audio_frontend_id: PARAKEET_TDT_AUDIO_FRONTEND_ID,
            runtime_tensor_contract_id: PARAKEET_TDT_RUNTIME_TENSOR_CONTRACT_ID,
            tokenizer_id: PARAKEET_TDT_TOKENIZER_ID,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_parakeet_tdt_source_to_runtime_pack",
                force_link: || {
                    let _ = std::hint::black_box(
                        crate::models::parakeet_tdt::convert_local_parakeet_tdt_source_to_runtime_pack
                            as *const (),
                    );
                },
            },
            hparam_schema: PARAKEET_TDT_HPARAM_SCHEMA,
            runtime_validator:
                crate::models::parakeet_tdt::runtime_contract::validate_runtime_pack_contract,
            // The FastConformer encoder reuses the composer conformer block, but
            // the TDT decode loop (LSTM prediction network + duration-driven
            // frame skipping) is a transducer, which is not a composer
            // orchestration shape, so the row declares a reasoned
            // ArchitectureGraph strategy like xasr.
        },
        execution_contract: OpenAsrExecutionContract {
            executor_component_id: PARAKEET_TDT_EXECUTOR_COMPONENT_ID,
            runtime_factory:
                crate::models::executor_component_registry::materialize_builtin_executor::<
                    crate::models::parakeet_tdt::executor::ParakeetTdtGgmlExecutor,
                >,
            execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
            execution_capabilities: CPU_AND_FULL_DEVICE_EXECUTION,
            phrase_bias: OpenAsrPhraseBiasStrategy::Unsupported,
            supports_translation_task: false,
            supports_source_language_hint: false,
            adapter_binding: GgmlAdapterBindingStrategy::Unsupported,
            prepared_runtime: OpenAsrPreparedRuntimeStrategy::FamilyOwned,
            word_timestamps: OpenAsrWordTimestampStrategy::DecodeInvariant,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            speaker_segmentation: SpeakerSegmentationSource::External,
            word_timestamp_source: WordTimestampSource::Native,
            longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
            invocation_span: OpenAsrInvocationSpan::Elastic,
            // Verified on the imported pack: trained on transcripts that preserve
            // punctuation and capitalization; the generated inventory projects
            // this declaration into catalog authoring.
            emits_punctuation: Some(true),
        },
        topology_contract: OpenAsrTopologyContract {
            decoder_state_topology: OpenAsrDecoderStateTopology::None,
            decode_driver: OpenAsrDecodeDriverStrategy::Dedicated {
                decode_policy_id: PARAKEET_TDT_DECODE_POLICY_ID,
                reason: "TDT transducer decoding requires predictor and joint-network state that is neither CTC nor seq2seq greedy.",
            },
            block_stack: OpenAsrBlockStackStrategy::ArchitectureGraph {
                reason: "TDT requires encoder, predictor, and joint-network graph orchestration.",
            },
            // The FastConformer encoder is full self-attention over the whole
            // chunk: quadratic in chunk length (issue #68). The TDT
            // decoder/joiner is a separate autoregressive stage and does not
            // change the encoder's scaling.
        },
        optimization_contract: OpenAsrOptimizationContract {
            prefer_cpu_decoder_for_multichunk_metal: false,
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
            encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
            },
        },
        quantization_contract: OpenAsrQuantizationContract {
            tensor_classification: crate::models::parakeet_tdt::package_import::TENSOR_QUANTIZATION_CONTRACT,
        },
        conformance_contract: OpenAsrConformanceContract {
            profile_id: "parakeet-tdt",
            reference_dumper_source: None,
            skeleton_exemption: None,
            skeleton_fixture: Some(SkeletonFixtureKind::ParakeetTdt),
        },
    },
    OpenAsrArchitectureDescriptor {
        identity: OpenAsrIdentityContract {
            runtime_architecture_aliases: &["wav2vec2-ctc", "wav2vec2"],
            model_family: "wav2vec2-ctc",
            model_architecture: WAV2VEC2_CTC_GGML_ARCHITECTURE_ID,
            adapter_id: WAV2VEC2_CTC_GGML_ADAPTER_ID,
            catalog_family_id: "wav2vec2",
            module_slug: "wav2vec2_ctc",
            recognized_languages: ENGLISH_RECOGNIZED_LANGUAGES,
            language_family_hint: LanguageFamilyHint::FixedMonolingual { language: "en" },
            dialect_capability: OpenAsrDialectCapability::NotAdvertised,
        },
        pack_contract: OpenAsrPackContract {
            audio_frontend_id: WAV2VEC2_CTC_AUDIO_FRONTEND_ID,
            runtime_tensor_contract_id: WAV2VEC2_CTC_RUNTIME_TENSOR_CONTRACT_ID,
            tokenizer_id: WAV2VEC2_CTC_TOKENIZER_ID,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_wav2vec2_ctc_source_to_runtime_pack",
                force_link: || {
                    let _ = std::hint::black_box(
                        crate::models::wav2vec2_ctc::convert_local_wav2vec2_ctc_source_to_runtime_pack
                            as *const (),
                    );
                },
            },
            hparam_schema: WAV2VEC2_CTC_HPARAM_SCHEMA,
            runtime_validator:
                crate::models::wav2vec2_ctc::runtime_contract::validate_runtime_pack_contract,
            // Non-autoregressive CTC: raw-waveform conv extractor + post-norm
            // transformer encoder + CTC head, no decoder stage.
        },
        execution_contract: OpenAsrExecutionContract {
            executor_component_id: WAV2VEC2_CTC_EXECUTOR_COMPONENT_ID,
            runtime_factory:
                crate::models::executor_component_registry::materialize_builtin_executor::<
                    crate::models::wav2vec2_ctc::executor::Wav2Vec2CtcGgmlExecutor,
                >,
            execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
            execution_capabilities: CPU_AND_FULL_DEVICE_EXECUTION,
            phrase_bias: OpenAsrPhraseBiasStrategy::Always,
            supports_translation_task: false,
            supports_source_language_hint: false,
            adapter_binding: GgmlAdapterBindingStrategy::Unsupported,
            prepared_runtime: OpenAsrPreparedRuntimeStrategy::FamilyOwned,
            word_timestamps: OpenAsrWordTimestampStrategy::DecodeInvariant,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            speaker_segmentation: SpeakerSegmentationSource::External,
            word_timestamp_source: WordTimestampSource::Native,
            longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
            invocation_span: OpenAsrInvocationSpan::Elastic,
            // Character CTC: same BYO-checkpoint reasoning as parakeet-ctc above.
            emits_punctuation: None,
        },
        topology_contract: OpenAsrTopologyContract {
            decoder_state_topology: OpenAsrDecoderStateTopology::None,
            decode_driver: OpenAsrDecodeDriverStrategy::SharedCtcGreedy {
                policy: decode_policy::WAV2VEC2_CTC_DECODE_POLICY_COMPONENT,
            },
            block_stack: OpenAsrBlockStackStrategy::Shared(OpenAsrBlockStackDescriptor {
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
        },
        optimization_contract: OpenAsrOptimizationContract {
            prefer_cpu_decoder_for_multichunk_metal: false,
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
            encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
            },
        },
        quantization_contract: OpenAsrQuantizationContract {
            tensor_classification: crate::models::wav2vec2_ctc::package_import::TENSOR_QUANTIZATION_CONTRACT,
        },
        conformance_contract: OpenAsrConformanceContract {
            profile_id: "wav2vec2",
            reference_dumper_source: None,
            skeleton_exemption: None,
            skeleton_fixture: Some(SkeletonFixtureKind::Wav2Vec2Ctc),
        },
    },
    OpenAsrArchitectureDescriptor {
        identity: OpenAsrIdentityContract {
            runtime_architecture_aliases: &["xasr-zipformer", "xasr-zh-en"],
            model_family: XASR_ZIPFORMER_MODEL_FAMILY,
            model_architecture: XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
            adapter_id: XASR_ZIPFORMER_GGML_ADAPTER_ID,
            catalog_family_id: "xasr-zipformer",
            module_slug: "xasr_zipformer",
            recognized_languages: BILINGUAL_RECOGNIZED_LANGUAGES,
            language_family_hint: LanguageFamilyHint::FixedMultilingual {
                languages: &["en", "zh"],
            },
            dialect_capability: OpenAsrDialectCapability::NotAdvertised,
        },
        pack_contract: OpenAsrPackContract {
            audio_frontend_id: XASR_ZIPFORMER_AUDIO_FRONTEND_ID,
            runtime_tensor_contract_id: XASR_ZIPFORMER_RUNTIME_TENSOR_CONTRACT_ID,
            tokenizer_id: XASR_ZIPFORMER_TOKENIZER_ID,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_xasr_zipformer_source_to_runtime_pack",
                force_link: || {
                    let _ = std::hint::black_box(
                        crate::models::xasr_zipformer::convert_local_xasr_zipformer_source_to_runtime_pack
                            as *const (),
                    );
                },
            },
            hparam_schema: XASR_ZIPFORMER_HPARAM_SCHEMA,
            runtime_validator:
                crate::models::xasr_zipformer::runtime_contract::validate_runtime_pack_contract,
            // Zipformer2 uses multi-scale streaming cache topology plus RNN-T
            // decoder/joiner, so it stays on its dedicated executor rather than the
            // generic block-stack composer.
        },
        execution_contract: OpenAsrExecutionContract {
            executor_component_id: XASR_ZIPFORMER_EXECUTOR_COMPONENT_ID,
            runtime_factory:
                crate::models::executor_component_registry::materialize_builtin_executor::<
                    crate::models::xasr_zipformer::executor::XasrZipformerGgmlExecutor,
                >,
            execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
            execution_capabilities: CPU_AND_FULL_DEVICE_EXECUTION,
            phrase_bias: OpenAsrPhraseBiasStrategy::Unsupported,
            supports_translation_task: false,
            supports_source_language_hint: false,
            adapter_binding: GgmlAdapterBindingStrategy::Unsupported,
            prepared_runtime: OpenAsrPreparedRuntimeStrategy::FamilyOwned,
            word_timestamps: OpenAsrWordTimestampStrategy::DecodeInvariant,
            streaming_partial_granularity: StreamingPartialGranularity::FrameSync,
            speaker_segmentation: SpeakerSegmentationSource::External,
            word_timestamp_source: WordTimestampSource::Native,
            longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
            invocation_span: OpenAsrInvocationSpan::Elastic,
            emits_punctuation: Some(true),
        },
        topology_contract: OpenAsrTopologyContract {
            decoder_state_topology: OpenAsrDecoderStateTopology::None,
            decode_driver: OpenAsrDecodeDriverStrategy::Dedicated {
                decode_policy_id: XASR_ZIPFORMER_DECODE_POLICY_ID,
                reason: "Zipformer transducer decoding requires architecture-specific predictor and joiner orchestration.",
            },
            block_stack: OpenAsrBlockStackStrategy::ArchitectureGraph {
                reason: "Zipformer uses multi-resolution encoder blocks and architecture-specific transducer heads.",
            },
            // Zipformer2's multi-scale streaming cache is local/chunked
            // attention with a bounded per-chunk cache, not global quadratic
            // attention: encoder memory is bounded independent of the logical
            // longform chunk length, so no additional longform safety cap
            // applies (issue #68).
        },
        optimization_contract: OpenAsrOptimizationContract {
            prefer_cpu_decoder_for_multichunk_metal: false,
            // Was measured CPU-favored on the M1 host, but that measurement
            // predates the encoder-weight-placement fix (#139): the encoder
            // weights were pinned off the GPU buffer, so Metal never actually
            // offloaded and the per-chunk graph paid GPU dispatch overhead with
            // no offload benefit. With weights correctly placed so the encoder
            // truly resides on the GPU buffer, a first re-measurement found
            // Metal at minimum competitive with CPU end-to-end, but a later,
            // cleaner platform audit found Metal itself still net-slower
            // end-to-end on Apple Silicon specifically (dispatch-bound: a
            // 29-frame chunk graph too small to amortize per-dispatch overhead)
            // -- see `xasr_zipformer::graph_config::encoder_gpu_enabled`.
            // `auto_gpu_policy` only ever changes which backend Auto picks,
            // never correctness (output stays byte-identical), so this is
            // `ExceptMetal`: Auto still prefers the generic GPU lane
            // (CUDA/HIP/Vulkan) where it was never measured to regress, and
            // falls back to CPU on Metal specifically. An explicit `--backend
            // metal` request still gets Metal.
            auto_gpu_policy: AutoGpuPolicy::ExceptMetal,
            encoder_attention_span: OpenAsrEncoderAttentionSpan::LocalChunked,
        },
        quantization_contract: OpenAsrQuantizationContract {
            tensor_classification: crate::models::xasr_zipformer::package_import::TENSOR_QUANTIZATION_CONTRACT,
        },
        conformance_contract: OpenAsrConformanceContract {
            profile_id: "xasr-zipformer",
            reference_dumper_source: None,
            skeleton_exemption: None,
            skeleton_fixture: Some(SkeletonFixtureKind::XasrZipformer),
        },
    },
    OpenAsrArchitectureDescriptor {
        identity: OpenAsrIdentityContract {
            runtime_architecture_aliases: &["moonshine", "moonshine-encoder-decoder"],
            model_family: "moonshine",
            model_architecture: MOONSHINE_GGML_ARCHITECTURE_ID,
            adapter_id: MOONSHINE_GGML_ADAPTER_ID,
            catalog_family_id: "moonshine",
            module_slug: "moonshine",
            recognized_languages: ENGLISH_RECOGNIZED_LANGUAGES,
            language_family_hint: LanguageFamilyHint::FixedMonolingual { language: "en" },
            dialect_capability: OpenAsrDialectCapability::NotAdvertised,
        },
        pack_contract: OpenAsrPackContract {
            audio_frontend_id: MOONSHINE_AUDIO_FRONTEND_ID,
            runtime_tensor_contract_id: MOONSHINE_RUNTIME_TENSOR_CONTRACT_ID,
            tokenizer_id: MOONSHINE_TOKENIZER_ID,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_moonshine_source_to_runtime_pack",
                force_link: || {
                    let _ = std::hint::black_box(
                        crate::models::moonshine::convert_local_moonshine_source_to_runtime_pack
                            as *const (),
                    );
                },
            },
            hparam_schema: MOONSHINE_HPARAM_SCHEMA,
            runtime_validator:
                crate::models::moonshine::runtime_contract::validate_runtime_pack_contract,
            // Raw-waveform conv-stem + partial-RoPE seq2seq with a self-contained
            // dedicated executor (not the data-driven block-stack composer — its
            // RoPE conv-stem encoder + cross-attn decoder are not composer blocks).
        },
        execution_contract: OpenAsrExecutionContract {
            executor_component_id: MOONSHINE_EXECUTOR_COMPONENT_ID,
            runtime_factory:
                crate::models::executor_component_registry::materialize_builtin_executor::<
                    crate::models::moonshine::MoonshineGgmlExecutor,
                >,
            execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
            execution_capabilities: MOONSHINE_EXECUTION_CAPABILITIES,
            phrase_bias: OpenAsrPhraseBiasStrategy::Always,
            supports_translation_task: false,
            supports_source_language_hint: false,
            adapter_binding: GgmlAdapterBindingStrategy::MoonshineLoraV1,
            prepared_runtime: OpenAsrPreparedRuntimeStrategy::FamilyOwned,
            word_timestamps: OpenAsrWordTimestampStrategy::DecodeInvariant,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            speaker_segmentation: SpeakerSegmentationSource::External,
            word_timestamp_source: WordTimestampSource::Native,
            longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
            invocation_span: OpenAsrInvocationSpan::Elastic,
            emits_punctuation: Some(true),
        },
        topology_contract: OpenAsrTopologyContract {
            decoder_state_topology:
                OpenAsrDecoderStateTopology::EncoderDecoderSelfAndCrossAttentionKv,
            decode_driver: OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy {
                policy: decode_policy::MOONSHINE_DECODE_POLICY_COMPONENT,
            },
            block_stack: OpenAsrBlockStackStrategy::ArchitectureGraph {
                reason: "Moonshine's convolutional encoder and decoder graph are not represented by the shared block composer.",
            },
            // The RoPE encoder is full self-attention over the whole chunk:
            // quadratic in chunk length (issue #68), matching Moonshine's own
            // model-card guidance to keep chunks under 30 seconds. Also carries
            // the `ConservativeSeq2SeqV1` decode-side longform profile (issue
            // #60's repetition guard); the two caps now agree at the same
            // default, so composing them (taking the min) is a no-op here.
        },
        optimization_contract: OpenAsrOptimizationContract {
            prefer_cpu_decoder_for_multichunk_metal: false,
            // Host waveform/token preparation sits outside Moonshine's neural
            // graphs. Metal/CUDA/HIP run the complete encoder and decoder on
            // device; Vulkan uses the measured Hybrid split (device encoder,
            // CPU decoder). Exact Metal placement substantially narrows its
            // dispatch gap and lowers physical memory, but M1 q8 remains
            // slightly slower than CPU, so Metal stays explicit-only.
            auto_gpu_policy: AutoGpuPolicy::ExceptMetal,
            encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
            },
        },
        quantization_contract: OpenAsrQuantizationContract {
            tensor_classification: crate::models::moonshine::TENSOR_QUANTIZATION_CONTRACT,
        },
        conformance_contract: OpenAsrConformanceContract {
            profile_id: "moonshine",
            reference_dumper_source: None,
            skeleton_exemption: None,
            skeleton_fixture: Some(SkeletonFixtureKind::Moonshine),
        },
    },
    OpenAsrArchitectureDescriptor {
        identity: OpenAsrIdentityContract {
            runtime_architecture_aliases: &[DOLPHIN_GGML_ARCHITECTURE_ID, "dolphin"],
            model_family: DOLPHIN_MODEL_FAMILY,
            model_architecture: DOLPHIN_GGML_ARCHITECTURE_ID,
            adapter_id: DOLPHIN_GGML_ADAPTER_ID,
            // The dialect prefix (`<sos> <zh> <SICHUAN> <asr> <notimestamp>`) selects
            // the language/region via prompt tokens the same way OWSM/Whisper do; the
            // detected language is not surfaced yet, so treat it as prompt-selected.
            catalog_family_id: "dolphin",
            module_slug: "dolphin",
            recognized_languages: DOLPHIN_RECOGNIZED_LANGUAGES,
            language_family_hint: LanguageFamilyHint::SelectsViaPrompt {
                default_language: "zh",
            },
            dialect_capability: OpenAsrDialectCapability::SelectsViaPrompt {
                codes: crate::models::dolphin::language::DOLPHIN_CN_DIALECT_CODES,
            },
        },
        pack_contract: OpenAsrPackContract {
            audio_frontend_id: DOLPHIN_AUDIO_FRONTEND_ID,
            runtime_tensor_contract_id: DOLPHIN_RUNTIME_TENSOR_CONTRACT_ID,
            tokenizer_id: DOLPHIN_TOKENIZER_ID,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_dolphin_wenet_source_to_runtime_pack",
                force_link: || {
                    let _ = std::hint::black_box(
                        crate::models::dolphin::convert_local_dolphin_wenet_source_to_runtime_pack
                            as *const (),
                    );
                },
            },
            hparam_schema: DOLPHIN_HPARAM_SCHEMA,
            runtime_validator:
                crate::models::dolphin::runtime_contract::validate_runtime_pack_contract,
            // E-Branchformer encoder + Transformer decoder + CTC head stay on the
            // dedicated executor (the E-Branchformer block is not a composer block
            // kind), so no data-driven block-stack descriptor.
        },
        execution_contract: OpenAsrExecutionContract {
            executor_component_id: DOLPHIN_EXECUTOR_COMPONENT_ID,
            runtime_factory:
                crate::models::executor_component_registry::materialize_builtin_executor::<
                    crate::models::dolphin::executor::DolphinGgmlExecutor,
                >,
            execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
            execution_capabilities: DOLPHIN_EXECUTION_CAPABILITIES,
            phrase_bias: OpenAsrPhraseBiasStrategy::RequiresTensor {
                tensor_name: crate::models::dolphin::hotword_context::CONTEXT_MODULE_WORD_EMBEDDING_TENSOR_NAME,
            },
            supports_translation_task: false,
            supports_source_language_hint: false,
            adapter_binding: GgmlAdapterBindingStrategy::Unsupported,
            prepared_runtime: OpenAsrPreparedRuntimeStrategy::FamilyOwned,
            word_timestamps: OpenAsrWordTimestampStrategy::DecodeInvariant,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            speaker_segmentation: SpeakerSegmentationSource::External,
            word_timestamp_source: WordTimestampSource::ForcedAligner,
            longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
            invocation_span: OpenAsrInvocationSpan::Elastic,
            // DataoceanAI's cn-dialect-small training corpus is transcribed
            // without punctuation and the model has no punctuation-prediction
            // head/token to enable -- honestly unpunctuated, not "unknown".
            emits_punctuation: Some(false),
        },
        topology_contract: OpenAsrTopologyContract {
            decoder_state_topology: OpenAsrDecoderStateTopology::None,
            decode_driver: OpenAsrDecodeDriverStrategy::Dedicated {
                decode_policy_id: DOLPHIN_DECODE_POLICY_ID,
                reason: "Dolphin attention rescoring is a second-pass topology outside the shared CTC and seq2seq greedy drivers.",
            },
            block_stack: OpenAsrBlockStackStrategy::ArchitectureGraph {
                reason: "Dolphin composes a Wenet encoder with architecture-specific attention rescoring.",
            },
            // The E-Branchformer's rel-pos MHSA global branch is full
            // self-attention over the whole chunk: quadratic in chunk length
            // (issue #68).
        },
        optimization_contract: OpenAsrOptimizationContract {
            prefer_cpu_decoder_for_multichunk_metal: false,
            // Auto prefers the accelerator: once the E-Branchformer encoder + CTC
            // head weights live in a WEIGHTS-usage static arena (so the ggml
            // scheduler offloads them to Metal instead of pinning the whole encoder
            // to the CPU), Metal beats CPU end-to-end on Apple Silicon (AB-measured,
            // warm best-of-N on M1). The gate only ever picks the accelerator when
            // one is actually present (`runtime_gpu_is_available`), so non-Metal
            // hosts still resolve to CPU, and an explicit `--execution-target
            // cpu` request always wins -- see
            // `dolphin::executor::dolphin_runtime_backend`. fp16 Metal numerics
            // reproduce the golden transcript on the parity clip (CPU stays the
            // bit-exact reference gate).
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
            encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
            },
        },
        quantization_contract: OpenAsrQuantizationContract {
            tensor_classification: crate::models::dolphin::package_import::TENSOR_QUANTIZATION_CONTRACT,
        },
        conformance_contract: OpenAsrConformanceContract {
            profile_id: "dolphin",
            reference_dumper_source: None,
            skeleton_exemption: None,
            skeleton_fixture: Some(SkeletonFixtureKind::Dolphin),
        },
    },
    OpenAsrArchitectureDescriptor {
        identity: OpenAsrIdentityContract {
            runtime_architecture_aliases: &[SENSEVOICE_GGML_ARCHITECTURE_ID, "sensevoice"],
            model_family: SENSEVOICE_MODEL_FAMILY,
            model_architecture: SENSEVOICE_GGML_ARCHITECTURE_ID,
            adapter_id: SENSEVOICE_GGML_ADAPTER_ID,
            // Accepts an explicit zh/yue/en/ja/ko selection via the 4-token prompt
            // and auto-detects (readable `<|lang|>` CTC tag) when unset.
            catalog_family_id: "sensevoice",
            module_slug: "sensevoice",
            recognized_languages: SENSEVOICE_RECOGNIZED_LANGUAGES,
            language_family_hint: LanguageFamilyHint::DetectAndSelectsViaPrompt,
            dialect_capability: OpenAsrDialectCapability::NotAdvertised,
        },
        pack_contract: OpenAsrPackContract {
            audio_frontend_id: SENSEVOICE_AUDIO_FRONTEND_ID,
            runtime_tensor_contract_id: SENSEVOICE_RUNTIME_TENSOR_CONTRACT_ID,
            tokenizer_id: SENSEVOICE_TOKENIZER_ID,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_sensevoice_source_to_runtime_pack",
                force_link: || {
                    let _ = std::hint::black_box(
                        crate::models::sensevoice::convert_local_sensevoice_source_to_runtime_pack
                            as *const (),
                    );
                },
            },
            hparam_schema: SENSEVOICE_HPARAM_SCHEMA,
            runtime_validator:
                crate::models::sensevoice::runtime_contract::validate_runtime_pack_contract,
            // Non-autoregressive CTC: SAN-M/FSMN encoder + CTC head, no decoder
            // stage. The `tp.blk` stage rides the same dedicated executor; the
            // descriptor pins the primary `enc.blk` stack.
        },
        execution_contract: OpenAsrExecutionContract {
            executor_component_id: SENSEVOICE_EXECUTOR_COMPONENT_ID,
            runtime_factory:
                crate::models::executor_component_registry::materialize_builtin_executor::<
                    crate::models::sensevoice::executor::SenseVoiceGgmlExecutor,
                >,
            execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
            execution_capabilities: CPU_AND_FULL_DEVICE_EXECUTION,
            phrase_bias: OpenAsrPhraseBiasStrategy::Always,
            supports_translation_task: false,
            supports_source_language_hint: false,
            adapter_binding: GgmlAdapterBindingStrategy::Unsupported,
            prepared_runtime: OpenAsrPreparedRuntimeStrategy::FamilyOwned,
            word_timestamps: OpenAsrWordTimestampStrategy::DecodeInvariant,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            speaker_segmentation: SpeakerSegmentationSource::External,
            word_timestamp_source: WordTimestampSource::ForcedAligner,
            longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
            invocation_span: OpenAsrInvocationSpan::Elastic,
            emits_punctuation: Some(true),
        },
        topology_contract: OpenAsrTopologyContract {
            decoder_state_topology: OpenAsrDecoderStateTopology::None,
            decode_driver: OpenAsrDecodeDriverStrategy::SharedCtcGreedy {
                policy: decode_policy::SENSEVOICE_DECODE_POLICY_COMPONENT,
            },
            block_stack: OpenAsrBlockStackStrategy::Shared(OpenAsrBlockStackDescriptor {
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
        },
        optimization_contract: OpenAsrOptimizationContract {
            prefer_cpu_decoder_for_multichunk_metal: false,
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
            encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
            },
        },
        quantization_contract: OpenAsrQuantizationContract {
            tensor_classification: crate::models::sensevoice::package_import::TENSOR_QUANTIZATION_CONTRACT,
        },
        conformance_contract: OpenAsrConformanceContract {
            profile_id: "sensevoice",
            reference_dumper_source: None,
            skeleton_exemption: None,
            skeleton_fixture: Some(SkeletonFixtureKind::SenseVoice),
        },
    },
    OpenAsrArchitectureDescriptor {
        identity: OpenAsrIdentityContract {
            runtime_architecture_aliases: &[FIRERED_AED_GGML_ARCHITECTURE_ID, "firered-aed"],
            model_family: FIRERED_AED_MODEL_FAMILY,
            model_architecture: FIRERED_AED_GGML_ARCHITECTURE_ID,
            adapter_id: FIRERED_AED_GGML_ADAPTER_ID,
            // No language-selection prompt token and no decode-time detection: the
            // char+SPM vocab is a fixed Mandarin/Chinese-dialect + English set.
            catalog_family_id: "firered-aed",
            module_slug: "firered_aed",
            recognized_languages: BILINGUAL_RECOGNIZED_LANGUAGES,
            language_family_hint: LanguageFamilyHint::FixedMultilingual {
                languages: BILINGUAL_RECOGNIZED_LANGUAGES,
            },
            dialect_capability: OpenAsrDialectCapability::RecognizesCatalogDeclared,
        },
        pack_contract: OpenAsrPackContract {
            audio_frontend_id: FIRERED_AED_AUDIO_FRONTEND_ID,
            runtime_tensor_contract_id: FIRERED_AED_RUNTIME_TENSOR_CONTRACT_ID,
            tokenizer_id: FIRERED_AED_TOKENIZER_ID,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_firered_aed_source_to_runtime_pack",
                force_link: || {
                    let _ = std::hint::black_box(
                        crate::models::firered_aed::convert_local_firered_aed_source_to_runtime_pack
                            as *const (),
                    );
                },
            },
            hparam_schema: FIRERED_AED_HPARAM_SCHEMA,
            runtime_validator:
                crate::models::firered_aed::runtime_contract::validate_runtime_pack_contract,
            // Conformer encoder + Transformer decoder attention-only decode stays
            // on the dedicated executor (the Conformer block is not a composer
            // block kind), so no data-driven block-stack descriptor.
        },
        execution_contract: OpenAsrExecutionContract {
            executor_component_id: FIRERED_AED_EXECUTOR_COMPONENT_ID,
            runtime_factory:
                crate::models::executor_component_registry::materialize_builtin_executor::<
                    crate::models::firered_aed::executor::FireRedAedGgmlExecutor,
                >,
            execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
            execution_capabilities: CPU_AND_FULL_DEVICE_EXECUTION,
            phrase_bias: OpenAsrPhraseBiasStrategy::Unsupported,
            supports_translation_task: false,
            supports_source_language_hint: false,
            adapter_binding: GgmlAdapterBindingStrategy::Unsupported,
            prepared_runtime: OpenAsrPreparedRuntimeStrategy::FamilyOwned,
            word_timestamps: OpenAsrWordTimestampStrategy::DecodeInvariant,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            speaker_segmentation: SpeakerSegmentationSource::External,
            word_timestamp_source: WordTimestampSource::ForcedAligner,
            longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
            invocation_span: OpenAsrInvocationSpan::Elastic,
            // The reference tokenizer's dict.txt has no punctuation/<space>
            // entries (char + SPM vocab trained on unpunctuated Mandarin ASR
            // corpora); verified on the golden-diff fixture transcript.
            emits_punctuation: Some(false),
        },
        topology_contract: OpenAsrTopologyContract {
            decoder_state_topology:
                OpenAsrDecoderStateTopology::EncoderDecoderSelfAndCrossAttentionKv,
            decode_driver: OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy {
                policy: decode_policy::FIRERED_AED_DECODE_POLICY_COMPONENT,
            },
            block_stack: OpenAsrBlockStackStrategy::ArchitectureGraph {
                reason: "FireRed AED's Conformer encoder and attention decoder are not yet represented by the shared block composer.",
            },
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
        },
        optimization_contract: OpenAsrOptimizationContract {
            prefer_cpu_decoder_for_multichunk_metal: false,
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
            encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
            },
        },
        quantization_contract: OpenAsrQuantizationContract {
            tensor_classification: crate::models::firered_aed::package_import::TENSOR_QUANTIZATION_CONTRACT,
        },
        conformance_contract: OpenAsrConformanceContract {
            profile_id: "firered-aed",
            reference_dumper_source: Some("tooling/firered2-reference-dumper/dump_aed_encoder.py"),
            skeleton_exemption: None,
            skeleton_fixture: Some(SkeletonFixtureKind::FireRedAed),
        },
    },
    OpenAsrArchitectureDescriptor {
        identity: OpenAsrIdentityContract {
            runtime_architecture_aliases: &[FIRERED_LLM_GGML_ARCHITECTURE_ID, "firered2-llm"],
            model_family: FIRERED_LLM_MODEL_FAMILY,
            model_architecture: FIRERED_LLM_GGML_ARCHITECTURE_ID,
            adapter_id: FIRERED_LLM_GGML_ADAPTER_ID,
            // No language-selection prompt token and no decode-time detection:
            // the Qwen2 BPE vocab covers Mandarin + English (the upstream ASR
            // finetune's training languages), same shape as firered-aed.
            catalog_family_id: "firered2-llm",
            module_slug: "firered_llm",
            recognized_languages: BILINGUAL_RECOGNIZED_LANGUAGES,
            language_family_hint: LanguageFamilyHint::FixedMultilingual {
                languages: BILINGUAL_RECOGNIZED_LANGUAGES,
            },
            dialect_capability: OpenAsrDialectCapability::NotAdvertised,
        },
        pack_contract: OpenAsrPackContract {
            audio_frontend_id: FIRERED_LLM_AUDIO_FRONTEND_ID,
            runtime_tensor_contract_id: FIRERED_LLM_RUNTIME_TENSOR_CONTRACT_ID,
            tokenizer_id: FIRERED_LLM_TOKENIZER_ID,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_firered_llm_source_to_runtime_pack",
                force_link: || {
                    let _ = std::hint::black_box(
                        crate::models::firered_llm::convert_local_firered_llm_source_to_runtime_pack
                            as *const (),
                    );
                },
            },
            hparam_schema: FIRERED_LLM_HPARAM_SCHEMA,
            runtime_validator:
                crate::models::firered_llm::runtime_contract::validate_runtime_pack_contract,
            // Conformer encoder + Qwen2 decoder-only decode both stay on the
            // dedicated executor (neither shape is a composer block kind), so no
            // data-driven block-stack descriptor.
        },
        execution_contract: OpenAsrExecutionContract {
            executor_component_id: FIRERED_LLM_EXECUTOR_COMPONENT_ID,
            runtime_factory:
                crate::models::executor_component_registry::materialize_builtin_executor::<
                    crate::models::firered_llm::executor::FireRedLlmGgmlExecutor,
                >,
            execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
            execution_capabilities: FIRERED_LLM_EXECUTION_CAPABILITIES,
            phrase_bias: OpenAsrPhraseBiasStrategy::Unsupported,
            supports_translation_task: false,
            supports_source_language_hint: false,
            adapter_binding: GgmlAdapterBindingStrategy::Unsupported,
            prepared_runtime: OpenAsrPreparedRuntimeStrategy::FamilyOwned,
            word_timestamps: OpenAsrWordTimestampStrategy::DecodeInvariant,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            speaker_segmentation: SpeakerSegmentationSource::External,
            word_timestamp_source: WordTimestampSource::ForcedAligner,
            longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
            invocation_span: OpenAsrInvocationSpan::Bounded { max_seconds: 40.0 },
            // Qwen2's ChatML decode is a plain transcription completion -- no
            // learned punctuation-suppression behavior has been characterized
            // for this family yet (unlike firered-aed's punctuation-free
            // char+SPM vocab); leave unclaimed rather than assert an unverified
            // capability.
            emits_punctuation: None,
        },
        topology_contract: OpenAsrTopologyContract {
            decoder_state_topology: OpenAsrDecoderStateTopology::CausalSelfAttentionKv,
            decode_driver: OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy {
                policy: decode_policy::FIRERED_LLM_DECODE_POLICY_COMPONENT,
            },
            block_stack: OpenAsrBlockStackStrategy::ArchitectureGraph {
                reason: "FireRed LLM composes a speech encoder and adapter with a Qwen language-model backbone.",
            },
            // Same Conformer encoder shape as firered-aed (full self-attention
            // over the whole chunk, quadratic in chunk length -- issue #68), plus
            // the upstream's own explicit 40s HARD cap (not just guidance --
            // `FireRedLlmGgmlExecutor` fails closed above it). 30s stays
            // comfortably under both that hard cap and firered-aed's own
            // guidance-based ceiling.
        },
        optimization_contract: OpenAsrOptimizationContract {
            prefer_cpu_decoder_for_multichunk_metal: false,
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
            encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
            },
        },
        quantization_contract: OpenAsrQuantizationContract {
            tensor_classification: crate::models::firered_llm::package_import::TENSOR_QUANTIZATION_CONTRACT,
        },
        conformance_contract: OpenAsrConformanceContract {
            profile_id: "firered2-llm",
            reference_dumper_source: Some("tooling/firered2-reference-dumper/dump_reference.py"),
            skeleton_exemption: None,
            skeleton_fixture: Some(SkeletonFixtureKind::FireRed2Llm),
        },
    },
    OpenAsrArchitectureDescriptor {
        identity: OpenAsrIdentityContract {
            runtime_architecture_aliases: &[FUNASR_NANO_GGML_ARCHITECTURE_ID, "funasr-nano"],
            model_family: FUNASR_NANO_MODEL_FAMILY,
            model_architecture: FUNASR_NANO_GGML_ARCHITECTURE_ID,
            adapter_id: FUNASR_NANO_GGML_ADAPTER_ID,
            // No language-selection prompt token and no decode-time detection: the
            // stock Qwen3 BPE vocab covers Mandarin + English (Fun-ASR-Nano's
            // trained ASR languages).
            catalog_family_id: "funasr-nano",
            module_slug: "funasr_nano",
            recognized_languages: BILINGUAL_RECOGNIZED_LANGUAGES,
            language_family_hint: LanguageFamilyHint::FixedMultilingual {
                languages: BILINGUAL_RECOGNIZED_LANGUAGES,
            },
            dialect_capability: OpenAsrDialectCapability::NotAdvertised,
        },
        pack_contract: OpenAsrPackContract {
            audio_frontend_id: FUNASR_NANO_AUDIO_FRONTEND_ID,
            runtime_tensor_contract_id: FUNASR_NANO_RUNTIME_TENSOR_CONTRACT_ID,
            tokenizer_id: FUNASR_NANO_TOKENIZER_ID,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_funasr_nano_source_to_runtime_pack",
                force_link: || {
                    let _ = std::hint::black_box(
                        crate::models::funasr_nano::convert_local_funasr_nano_source_to_runtime_pack
                            as *const (),
                    );
                },
            },
            hparam_schema: FUNASR_NANO_HPARAM_SCHEMA,
            runtime_validator:
                crate::models::funasr_nano::runtime_contract::validate_runtime_pack_contract,
            // SAN-M encoder + Qwen3 decoder-only decode both stay on the dedicated
            // executor (neither shape is a composer block kind), so no data-driven
            // block-stack descriptor.
        },
        execution_contract: OpenAsrExecutionContract {
            executor_component_id: FUNASR_NANO_EXECUTOR_COMPONENT_ID,
            runtime_factory:
                crate::models::executor_component_registry::materialize_builtin_executor::<
                    crate::models::funasr_nano::executor::FunasrNanoGgmlExecutor,
                >,
            execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
            execution_capabilities: CPU_AND_FULL_DEVICE_EXECUTION,
            phrase_bias: OpenAsrPhraseBiasStrategy::Unsupported,
            supports_translation_task: false,
            supports_source_language_hint: false,
            adapter_binding: GgmlAdapterBindingStrategy::Unsupported,
            prepared_runtime: OpenAsrPreparedRuntimeStrategy::FamilyOwned,
            word_timestamps: OpenAsrWordTimestampStrategy::DecodeInvariant,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            speaker_segmentation: SpeakerSegmentationSource::External,
            word_timestamp_source: WordTimestampSource::ForcedAligner,
            longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
            invocation_span: OpenAsrInvocationSpan::Bounded { max_seconds: 40.0 },
            // The stock Qwen3 ChatML decode emits ordinary punctuation, but no
            // punctuation-suppression behavior has been separately characterized;
            // leave unclaimed rather than assert a capability beyond the two golden
            // clips.
            emits_punctuation: None,
        },
        topology_contract: OpenAsrTopologyContract {
            decoder_state_topology: OpenAsrDecoderStateTopology::CausalSelfAttentionKv,
            decode_driver: OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy {
                policy: decode_policy::FUNASR_NANO_DECODE_POLICY_COMPONENT,
            },
            block_stack: OpenAsrBlockStackStrategy::ArchitectureGraph {
                reason: "FunASR Nano composes a SAN-M encoder and adaptor with a Qwen language-model decoder.",
            },
            // SAN-M encoder is full self-attention over the whole chunk (quadratic
            // in chunk length), plus the upstream's own ~40s HARD cap
            // (`FunasrNanoGgmlExecutor` fails closed above it). 30s stays under both.
        },
        optimization_contract: OpenAsrOptimizationContract {
            prefer_cpu_decoder_for_multichunk_metal: false,
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
            encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
            },
        },
        quantization_contract: OpenAsrQuantizationContract {
            tensor_classification: crate::models::funasr_nano::package_import::TENSOR_QUANTIZATION_CONTRACT,
        },
        conformance_contract: OpenAsrConformanceContract {
            profile_id: "funasr-nano",
            reference_dumper_source: Some(
                "tooling/publish-model/scripts/funasr_nano_reference_oracle.py",
            ),
            skeleton_exemption: None,
            skeleton_fixture: Some(SkeletonFixtureKind::FunasrNano),
        },
    },
    OpenAsrArchitectureDescriptor {
        identity: OpenAsrIdentityContract {
            runtime_architecture_aliases: &[MIMO_ASR_GGML_ARCHITECTURE_ID],
            model_family: MIMO_ASR_MODEL_FAMILY,
            model_architecture: MIMO_ASR_GGML_ARCHITECTURE_ID,
            adapter_id: MIMO_ASR_GGML_ADAPTER_ID,
            // No language-selection prompt token and no decode-time detection:
            // the Qwen2 BPE vocab covers the upstream's trained languages
            // (Mandarin, English, Cantonese + regional dialects per its README).
            catalog_family_id: "mimo-asr",
            module_slug: "mimo_asr",
            recognized_languages: MIMO_RECOGNIZED_LANGUAGES,
            language_family_hint: LanguageFamilyHint::FixedMultilingual {
                languages: MIMO_RECOGNIZED_LANGUAGES,
            },
            dialect_capability: OpenAsrDialectCapability::NotAdvertised,
        },
        pack_contract: OpenAsrPackContract {
            audio_frontend_id: MIMO_ASR_AUDIO_FRONTEND_ID,
            runtime_tensor_contract_id: MIMO_ASR_RUNTIME_TENSOR_CONTRACT_ID,
            tokenizer_id: MIMO_ASR_TOKENIZER_ID,
            pack_import: OpenAsrPackImportSurface::ExternalTooling {
                relative_path: "tooling/mimo-asr/convert_mimo_asr.py",
            },
            hparam_schema: MIMO_ASR_HPARAM_SCHEMA,
            runtime_validator:
                crate::models::mimo_asr::runtime_contract::validate_runtime_pack_contract,
            // Audio-tokenizer encoder + RVQ + input-local + Qwen2 decode all stay
            // on the dedicated executor (none of these stages is a composer
            // block kind), so no data-driven block-stack descriptor.
        },
        execution_contract: OpenAsrExecutionContract {
            executor_component_id: MIMO_ASR_EXECUTOR_COMPONENT_ID,
            runtime_factory:
                crate::models::executor_component_registry::materialize_builtin_executor::<
                    crate::models::mimo_asr::executor::MimoAsrGgmlExecutor,
                >,
            execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
            execution_capabilities: CPU_AND_FULL_DEVICE_EXECUTION,
            phrase_bias: OpenAsrPhraseBiasStrategy::Unsupported,
            supports_translation_task: false,
            supports_source_language_hint: false,
            adapter_binding: GgmlAdapterBindingStrategy::Unsupported,
            prepared_runtime: OpenAsrPreparedRuntimeStrategy::FamilyOwned,
            word_timestamps: OpenAsrWordTimestampStrategy::DecodeInvariant,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            speaker_segmentation: SpeakerSegmentationSource::External,
            word_timestamp_source: WordTimestampSource::ForcedAligner,
            longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
            invocation_span: OpenAsrInvocationSpan::Bounded { max_seconds: 30.0 },
            // No characterized punctuation behavior for this family yet (unlike
            // firered-aed's punctuation-free vocab) -- leave unclaimed rather
            // than assert an unverified capability.
            emits_punctuation: None,
        },
        topology_contract: OpenAsrTopologyContract {
            decoder_state_topology: OpenAsrDecoderStateTopology::CausalSelfAttentionKv,
            decode_driver: OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy {
                policy: decode_policy::MIMO_ASR_DECODE_POLICY_COMPONENT,
            },
            block_stack: OpenAsrBlockStackStrategy::ArchitectureGraph {
                reason: "MiMo composes a speech tokenizer and input-local adapter with its language-model decoder.",
            },
            // The 32L rope audio-tokenizer encoder is full self-attention over
            // the whole chunk: quadratic in chunk length. The executor's own
            // 30s-per-chunk hard cap (mirroring the reference `preprocess_input`'s
            // 30s re-chunking) keeps this well inside the shared default.
        },
        optimization_contract: OpenAsrOptimizationContract {
            prefer_cpu_decoder_for_multichunk_metal: false,
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
            encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
            },
        },
        quantization_contract: OpenAsrQuantizationContract {
            tensor_classification: crate::models::mimo_asr::TENSOR_QUANTIZATION_CONTRACT,
        },
        conformance_contract: OpenAsrConformanceContract {
            profile_id: "mimo-asr",
            reference_dumper_source: None,
            skeleton_exemption: None,
            skeleton_fixture: Some(SkeletonFixtureKind::MimoAsr),
        },
    },
    OpenAsrArchitectureDescriptor {
        identity: OpenAsrIdentityContract {
            runtime_architecture_aliases: &[MOSS_TD_GGML_ARCHITECTURE_ID],
            model_family: MOSS_TD_MODEL_FAMILY,
            model_architecture: MOSS_TD_GGML_ARCHITECTURE_ID,
            adapter_id: MOSS_TD_GGML_ADAPTER_ID,
            // The Qwen3 decoder auto-detects/produces the transcript language
            // through free-text instruction-following (no dedicated language
            // token in its vocab, same shape as qwen3-asr) -- until prompt-level
            // language conditioning is wired and verified against a real pack, an
            // explicit hint is rejected (not faked) rather than silently ignored.
            catalog_family_id: "moss-transcribe-diarize",
            module_slug: "moss_transcribe_diarize",
            recognized_languages: MOSS_RECOGNIZED_LANGUAGES,
            language_family_hint: LanguageFamilyHint::SelfDetectsRejectsHint {
                reject_reason: "MOSS-Transcribe-Diarize auto-detects the source language via its Qwen3 decoder and does not accept an explicit selection; use a multilingual Whisper pack to force or report a language.",
            },
            dialect_capability: OpenAsrDialectCapability::NotAdvertised,
        },
        pack_contract: OpenAsrPackContract {
            audio_frontend_id: MOSS_TD_AUDIO_FRONTEND_ID,
            runtime_tensor_contract_id: MOSS_TD_RUNTIME_TENSOR_CONTRACT_ID,
            tokenizer_id: MOSS_TD_TOKENIZER_ID,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_moss_transcribe_diarize_source_to_runtime_pack",
                force_link: || {
                    let _ = std::hint::black_box(
                        crate::models::moss_transcribe_diarize::convert_local_moss_transcribe_diarize_source_to_runtime_pack
                            as *const (),
                    );
                },
            },
            hparam_schema: MOSS_TD_HPARAM_SCHEMA,
            runtime_validator:
                crate::models::moss_transcribe_diarize::runtime_contract::validate_runtime_pack_contract,
            // Whisper encoder + Qwen3 decoder-only decode both stay on the
            // dedicated executor (neither shape is a composer block kind), so no
            // data-driven block-stack descriptor.
        },
        execution_contract: OpenAsrExecutionContract {
            executor_component_id: MOSS_TD_EXECUTOR_COMPONENT_ID,
            runtime_factory:
                crate::models::executor_component_registry::materialize_builtin_executor::<
                    crate::models::moss_transcribe_diarize::executor::MossTdGgmlExecutor,
                >,
            execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
            execution_capabilities: CPU_AND_FULL_DEVICE_EXECUTION,
            phrase_bias: OpenAsrPhraseBiasStrategy::Unsupported,
            supports_translation_task: false,
            supports_source_language_hint: false,
            adapter_binding: GgmlAdapterBindingStrategy::Unsupported,
            prepared_runtime: OpenAsrPreparedRuntimeStrategy::FamilyOwned,
            word_timestamps: OpenAsrWordTimestampStrategy::DecodeInvariant,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            speaker_segmentation: SpeakerSegmentationSource::InDecoder,
            word_timestamp_source: WordTimestampSource::ForcedAligner,
            // Product invocation envelope: ordinary VAD slicing aims at 30s and
            // may stretch to 60s to reach a clean cut. Recordings at or below 60s
            // decode whole. This bound is machine-independent, so a recording has
            // identical slicing/transcript semantics on CPU and GPU; memory
            // pressure changes only which execution candidate is admitted.
            //
            // The family topology proves the 60s reserve exactly from the real
            // integer frontend/prompt/budget counters: 750 audio tokens + 23 time
            // marker tokens + 86 fixed tokens + 1508 generated-token allowance =
            // 2367 semantic positions. The shared greedy schedule never feeds its
            // final sampled token back, so the exact physical self-KV span is
            // 2366 rows. `MOSS_TD_MAX_KV_CACHE_POSITIONS` (8192)
            // remains a fail-closed safety ceiling, never the allocation request.
            // Every slice is still its own model speaker-label scope; optional
            // Voice ID may reconcile labels across slices at the product layer.
            longform_slice_shape: OpenAsrLongformSliceShape::ScopedSlices {
                integral_seconds: MOSS_TD_MAX_INVOCATION_SECONDS as f32,
                target_seconds: MOSS_TD_TARGET_INVOCATION_SECONDS as f32,
                max_seconds: MOSS_TD_MAX_INVOCATION_SECONDS as f32,
            },
            invocation_span: OpenAsrInvocationSpan::Bounded {
                max_seconds: MOSS_TD_MAX_INVOCATION_SECONDS as f32,
            },
            // The fixed instruction asks for full punctuation-bearing prose
            // segments; no characterized counter-example has been observed yet,
            // but this has not been verified against enough real transcripts to
            // assert as a capability -- leave unclaimed rather than guess.
            emits_punctuation: None,
        },
        topology_contract: OpenAsrTopologyContract {
            decoder_state_topology: OpenAsrDecoderStateTopology::CausalSelfAttentionKv,
            decode_driver: OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy {
                policy: decode_policy::MOSS_TD_DECODE_POLICY_COMPONENT,
            },
            block_stack: OpenAsrBlockStackStrategy::ArchitectureGraph {
                reason: "MOSS transcription/diarization uses an architecture-specific audio encoder and decoder graph.",
            },
            // Whisper's own architecture-fixed 30s log-mel window: the encoder
            // never attends past its own fixed 1500-position chunk no matter how
            // long the total requested audio is (the executor loops the encoder
            // over independent 30s windows and concatenates -- see `executor.rs`'s
            // module doc), so this needs no additional longform safety cap --
            // same classification as `whisper` itself.
        },
        optimization_contract: OpenAsrOptimizationContract {
            prefer_cpu_decoder_for_multichunk_metal: false,
            // Auto selects Metal/GPU when available. Correctness blockers that
            // once justified ExceptMetal are closed (encoder divergence falsified;
            // #180 decode graph reuse; #212 chunked resident prefill stops 3-min
            // OOM). Post-#212 quiet-window A/B on M1 (true execution_target=
            // accelerated, fp16): Metal RTF beats CPU on jfk/en_zh/aishell4
            // (~0.22-0.48x CPU RTF) with lower RSS and no OOM -- see
            // docs/model-audits/moss-transcribe-diarize.md section 6 and
            // tmp/moss-quiet-2026-07-24/. Do not re-introduce ExceptMetal without
            // a fresh quiet-window loss on true accelerated (not env-only hybrid).
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
            // The model is trained to emit `[S01]`/`[S02]`/... speaker labels
            // directly in its transcript text (see `decode_prompt`'s fixed
            // instruction), so this family diarizes itself -- there is no
            // separate diarization pass to compose.
            encoder_attention_span: OpenAsrEncoderAttentionSpan::FixedWindow,
        },
        quantization_contract: OpenAsrQuantizationContract {
            tensor_classification: crate::models::moss_transcribe_diarize::package_import::TENSOR_QUANTIZATION_CONTRACT,
        },
        conformance_contract: OpenAsrConformanceContract {
            profile_id: "moss-transcribe-diarize",
            reference_dumper_source: Some("tooling/moss-reference-dumper/dump_golden.py"),
            skeleton_exemption: None,
            skeleton_fixture: Some(SkeletonFixtureKind::MossTranscribeDiarize),
        },
    },
    OpenAsrArchitectureDescriptor {
        identity: OpenAsrIdentityContract {
            runtime_architecture_aliases: &[GRANITE_SPEECH_GGML_ARCHITECTURE_ID],
            model_family: GRANITE_SPEECH_MODEL_FAMILY,
            model_architecture: GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            adapter_id: GRANITE_SPEECH_GGML_ADAPTER_ID,
            // The Granite decoder auto-detects/produces the transcript language
            // through free-text instruction-following (no dedicated language
            // token; the model card documents multilingual prompts working
            // without a language selector) -- same shape as qwen3-asr/moss-td:
            // reject an explicit hint rather than silently ignore it.
            catalog_family_id: "granite-speech",
            module_slug: "granite_speech",
            recognized_languages: GRANITE_SPEECH_RECOGNIZED_LANGUAGES,
            language_family_hint: LanguageFamilyHint::SelfDetectsRejectsHint {
                reject_reason: "Granite Speech auto-detects the source language through free-text prompting and does not accept an explicit selection.",
            },
            dialect_capability: OpenAsrDialectCapability::NotAdvertised,
        },
        pack_contract: OpenAsrPackContract {
            audio_frontend_id: GRANITE_SPEECH_AUDIO_FRONTEND_ID,
            runtime_tensor_contract_id: GRANITE_SPEECH_RUNTIME_TENSOR_CONTRACT_ID,
            tokenizer_id: GRANITE_SPEECH_TOKENIZER_ID,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_granite_speech_source_to_runtime_pack",
                force_link: || {
                    let _ = std::hint::black_box(
                        crate::models::granite_speech::convert_local_granite_speech_source_to_runtime_pack
                            as *const (),
                    );
                },
            },
            hparam_schema: GRANITE_SPEECH_HPARAM_SCHEMA,
            runtime_validator:
                crate::models::granite_speech::runtime_contract::validate_runtime_pack_contract,
            // Conformer encoder + Q-Former projector + Granite decoder all stay
            // on the dedicated executor (none of the three is a composer block
            // kind), so no data-driven block-stack descriptor.
        },
        execution_contract: OpenAsrExecutionContract {
            executor_component_id: GRANITE_SPEECH_EXECUTOR_COMPONENT_ID,
            runtime_factory:
                crate::models::executor_component_registry::materialize_builtin_executor::<
                    crate::models::granite_speech::executor::GraniteSpeechGgmlExecutor,
                >,
            execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
            execution_capabilities: CPU_AND_FULL_DEVICE_EXECUTION,
            phrase_bias: OpenAsrPhraseBiasStrategy::Always,
            supports_translation_task: false,
            supports_source_language_hint: false,
            adapter_binding: GgmlAdapterBindingStrategy::Unsupported,
            prepared_runtime: OpenAsrPreparedRuntimeStrategy::FamilyOwned,
            word_timestamps: OpenAsrWordTimestampStrategy::DecodeInvariant,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            // Greedy decode rides the one shared seq2seq driver via the policy
            // embedded in this row (see AGENTS.md's single-driver invariant);
            // this family provides a `Seq2SeqGreedyDecodeStepExecutor` and a
            // `GRANITE_SPEECH_DECODE_POLICY_ID` descriptor rather than a
            // hand-rolled argmax loop.
            speaker_segmentation: SpeakerSegmentationSource::External,
            word_timestamp_source: WordTimestampSource::ForcedAligner,
            // External families ride the shared generic longform window (slices
            // are never their own speaker scope). This is not the whole-recording
            // single-prompt `ScopedSlices` case (only moss-transcribe-diarize is),
            // so no integral window is declared or consumed here.
            longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
            invocation_span: OpenAsrInvocationSpan::Bounded {
                max_seconds: GRANITE_SPEECH_MAX_INVOCATION_SECONDS as f32,
            },
            // The model card documents punctuation/truecasing as a real,
            // evaluated capability (a documented prompt variant + reported PER/
            // Cap-F1 metrics), and the family's own end-to-end golden samples
            // come out correctly punctuated -- unlike `MIMO_ASR`'s "not
            // characterized yet" case above, this one has been observed.
            emits_punctuation: Some(true),
        },
        topology_contract: OpenAsrTopologyContract {
            decoder_state_topology: OpenAsrDecoderStateTopology::CausalSelfAttentionKv,
            decode_driver: OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy {
                policy: decode_policy::GRANITE_SPEECH_DECODE_POLICY_COMPONENT,
            },
            block_stack: OpenAsrBlockStackStrategy::ArchitectureGraph {
                reason: "Granite Speech composes a Conformer encoder and Q-Former projector with its language-model decoder.",
            },
            // The Conformer encoder's self-attention is local to non-overlapping
            // `context_size=200`-frame blocks (Shaw relative-position attention),
            // never global over the whole utterance -- memory is bounded per
            // block regardless of total audio length, matching `LocalChunked`
            // exactly (see `encoder_graph.rs`'s module doc). This is a real,
            // verified difference from every other family's classification here:
            // it is neither `whisper`'s architecture-fixed 30s window nor a
            // quadratic-over-the-whole-chunk encoder.
        },
        optimization_contract: OpenAsrOptimizationContract {
            prefer_cpu_decoder_for_multichunk_metal: false,
            // Perf/backend tuning is out of scope for this landing (the decoder
            // is still the O(n^2) recompute-per-step prefill executor, see
            // `decode_executor.rs`'s module doc) -- start un-gated like every
            // other family's initial landing, revisit once a real measurement
            // exists.
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
            // Granite Speech emits plain transcripts with no in-decoder speaker
            // markup, so speaker structure comes from the shared external
            // segmenter pass, same as every other non-diarizing family here.
            encoder_attention_span: OpenAsrEncoderAttentionSpan::LocalChunked,
        },
        quantization_contract: OpenAsrQuantizationContract {
            tensor_classification: crate::models::granite_speech::package_import::TENSOR_QUANTIZATION_CONTRACT,
        },
        conformance_contract: OpenAsrConformanceContract {
            profile_id: "granite-speech",
            reference_dumper_source: Some("tooling/granite-speech-reference-dumper/dump_golden.py"),
            // Exempt from the runtime-ready skeleton gate: the granite-speech
            // runtime pack contract pins the EOT token id against the embedded
            // vocab, and a valid fixture therefore needs a vocab > 100257 --
            // far beyond a tiny skeleton. Coverage stays fail-closed through the
            // family metadata/tensor contract tests instead.
            skeleton_exemption: Some(
                "EOT-id contract requires a fixture vocab larger than 100257",
            ),
            skeleton_fixture: None,
        },
    },
];

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn selection_metadata(
        family: &str,
        architecture: &str,
        frontend: &str,
        decode_policy: &str,
    ) -> BTreeMap<String, String> {
        BTreeMap::from([
            (
                crate::models::oasr_metadata::OASR_METADATA_KEY_PACKAGE_VERSION.to_string(),
                OASR_PACKAGE_VERSION_V1.to_string(),
            ),
            (
                crate::models::oasr_metadata::OASR_METADATA_KEY_MODEL_FAMILY.to_string(),
                family.to_string(),
            ),
            (
                crate::models::oasr_metadata::OASR_METADATA_KEY_MODEL_ARCHITECTURE.to_string(),
                architecture.to_string(),
            ),
            (
                crate::models::oasr_metadata::OASR_METADATA_KEY_AUDIO_FRONTEND.to_string(),
                frontend.to_string(),
            ),
            (
                crate::models::oasr_metadata::OASR_METADATA_KEY_DECODE_POLICY.to_string(),
                decode_policy.to_string(),
            ),
        ])
    }

    #[test]
    fn builtin_architectures_validate_inventory_invariants() {
        OpenAsrArchitectureRegistry::with_builtins()
            .validate_references()
            .expect("builtins must satisfy inventory invariants");
    }

    #[test]
    fn builtin_accelerated_routes_match_declared_neural_topologies() {
        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            let capabilities = descriptor.execution_contract.execution_capabilities;
            assert!(capabilities.supports_cpu());
            for provider in [
                ExecutionProvider::Metal,
                ExecutionProvider::Cuda,
                ExecutionProvider::Hip,
                ExecutionProvider::Vulkan,
            ] {
                let hybrid_only = descriptor.identity.model_architecture
                    == MOONSHINE_GGML_ARCHITECTURE_ID
                    && provider == ExecutionProvider::Vulkan;
                let qualified_fallback = descriptor.identity.model_architecture
                    == DOLPHIN_GGML_ARCHITECTURE_ID
                    && provider == ExecutionProvider::Vulkan;
                assert!(
                    capabilities.supports(
                        provider,
                        if hybrid_only {
                            crate::device::execution_policy::ExecutionPlacement::Hybrid
                        } else {
                            crate::device::execution_policy::ExecutionPlacement::FullDevice
                        },
                    ),
                    "architecture '{}' lost its declared {provider} route",
                    descriptor.identity.model_architecture,
                );
                assert!(
                    qualified_fallback
                        || !capabilities.supports(
                            provider,
                            if hybrid_only {
                                crate::device::execution_policy::ExecutionPlacement::FullDevice
                            } else {
                                crate::device::execution_policy::ExecutionPlacement::Hybrid
                            },
                        ),
                    "architecture '{}' admits two unqualified topologies under {provider}",
                    descriptor.identity.model_architecture,
                );
            }
        }
    }

    #[test]
    fn firered_aed_windows_direct_routes_preserve_upstream_provider_boundaries() {
        use crate::device::execution_policy::ExecutionPlacement;

        let capabilities = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(FIRERED_AED_GGML_ARCHITECTURE_ID)
            .expect("FireRed AED descriptor")
            .execution_contract
            .execution_capabilities;
        assert!(capabilities.supports_cpu());
        for provider in [
            ExecutionProvider::Metal,
            ExecutionProvider::Cuda,
            ExecutionProvider::Hip,
            ExecutionProvider::Vulkan,
        ] {
            assert!(capabilities.supports(provider, ExecutionPlacement::FullDevice));
            assert!(!capabilities.supports(provider, ExecutionPlacement::Hybrid));
        }
    }

    #[test]
    fn firered_llm_all_gpu_providers_use_full_device() {
        use crate::device::execution_policy::ExecutionPlacement;

        let capabilities = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(FIRERED_LLM_GGML_ARCHITECTURE_ID)
            .expect("FireRed LLM descriptor")
            .execution_contract
            .execution_capabilities;
        assert!(capabilities.supports_cpu());
        for provider in [
            ExecutionProvider::Metal,
            ExecutionProvider::Cuda,
            ExecutionProvider::Hip,
            ExecutionProvider::Vulkan,
        ] {
            assert!(capabilities.supports(provider, ExecutionPlacement::FullDevice));
            assert!(!capabilities.supports(provider, ExecutionPlacement::Hybrid));
        }
    }

    #[test]
    fn canonical_registry_selects_whisper_adapter_from_oasr_metadata() {
        let metadata = selection_metadata(
            "whisper",
            WHISPER_GGML_ARCHITECTURE_ID,
            WHISPER_AUDIO_FRONTEND_ID,
            WHISPER_DECODE_POLICY_ID,
        );
        let selected = OpenAsrArchitectureRegistry::with_builtins()
            .select_ggml_adapter_from_gguf_metadata_v1(&metadata)
            .expect("whisper metadata must select one architecture");
        assert_eq!(selected.identity.adapter_id, WHISPER_GGML_ADAPTER_ID);
        assert_eq!(
            selected.execution_contract.execution_capability,
            GgmlExecutionCapability::DedicatedRuntimeExecutorV1
        );
    }

    #[test]
    fn canonical_registry_rejects_conflicting_tokenizer() {
        let mut metadata = selection_metadata(
            "whisper",
            WHISPER_GGML_ARCHITECTURE_ID,
            WHISPER_AUDIO_FRONTEND_ID,
            WHISPER_DECODE_POLICY_ID,
        );
        metadata.insert(
            crate::models::ggml_family_adapter::GGML_TOKENIZER_ID_KEY.to_string(),
            "wrong.id".to_string(),
        );
        let error = OpenAsrArchitectureRegistry::with_builtins()
            .select_ggml_adapter_from_gguf_metadata_v1(&metadata)
            .expect_err("conflicting tokenizer must fail closed");
        assert_eq!(
            error,
            GgmlFamilyAdapterSelectionError::NoMatchingAdapter {
                model_family: "whisper".to_string(),
                model_architecture: WHISPER_GGML_ARCHITECTURE_ID.to_string(),
                audio_frontend_id: WHISPER_AUDIO_FRONTEND_ID.to_string(),
                decode_policy_id: WHISPER_DECODE_POLICY_ID.to_string(),
                tokenizer_id: Some("wrong.id".to_string()),
            }
        );
    }

    #[test]
    fn canonical_registry_rejects_ambiguous_architecture_descriptors() {
        let registry = OpenAsrArchitectureRegistry::with_builtins();
        let base = registry
            .find_by_model_architecture(WHISPER_GGML_ARCHITECTURE_ID)
            .expect("whisper architecture");
        let mut duplicate = base;
        duplicate.identity.adapter_id = "ggml-family-whisper-duplicate-runtime-v1";
        let descriptors = [base, duplicate];
        let metadata = selection_metadata(
            "whisper",
            WHISPER_GGML_ARCHITECTURE_ID,
            WHISPER_AUDIO_FRONTEND_ID,
            WHISPER_DECODE_POLICY_ID,
        );
        let spec = GgmlFamilyAdapterSelectionSpec::from_gguf_metadata_v1(&metadata);
        let fields = spec.parse_selection_fields().expect("selection fields");
        let error = OpenAsrArchitectureRegistry::select_ggml_adapter_from_descriptors(
            &descriptors,
            &fields,
        )
        .expect_err("duplicate descriptors must fail closed");
        assert_eq!(
            error,
            GgmlFamilyAdapterSelectionError::Ambiguous {
                adapter_ids: vec![
                    WHISPER_GGML_ADAPTER_ID,
                    "ggml-family-whisper-duplicate-runtime-v1"
                ],
            }
        );
    }

    #[test]
    fn canonical_registry_validates_adapter_identity_uniqueness() {
        let registry = OpenAsrArchitectureRegistry::with_builtins();
        let base = registry
            .find_by_model_architecture(WHISPER_GGML_ARCHITECTURE_ID)
            .expect("whisper architecture");
        let mut duplicate = base;
        duplicate.identity.adapter_id = "ggml-family-whisper-duplicate-runtime-v1";
        let descriptors = [base, duplicate];
        assert_eq!(
            OpenAsrArchitectureRegistry::validate_adapter_uniqueness(&descriptors),
            Err(
                OpenAsrArchitectureRegistryError::DuplicateModelArchitecture {
                    model_architecture: WHISPER_GGML_ARCHITECTURE_ID,
                    first_adapter_id: WHISPER_GGML_ADAPTER_ID,
                    duplicate_adapter_id: "ggml-family-whisper-duplicate-runtime-v1",
                }
            )
        );
    }

    #[test]
    fn adapter_lookup_exposes_canonical_task_capabilities() {
        let registry = OpenAsrArchitectureRegistry::with_builtins();
        let expected = [
            (COHERE_TRANSCRIBE_GGML_ADAPTER_ID, false, true),
            (WHISPER_GGML_ADAPTER_ID, true, true),
            (QWEN3_ASR_GGML_ADAPTER_ID, false, false),
            (PARAKEET_CTC_GGML_ADAPTER_ID, false, false),
            (PARAKEET_TDT_GGML_ADAPTER_ID, false, false),
            (WAV2VEC2_CTC_GGML_ADAPTER_ID, false, false),
            (XASR_ZIPFORMER_GGML_ADAPTER_ID, false, false),
            (MOONSHINE_GGML_ADAPTER_ID, false, false),
            (DOLPHIN_GGML_ADAPTER_ID, false, false),
            (SENSEVOICE_GGML_ADAPTER_ID, false, false),
            (FIRERED_AED_GGML_ADAPTER_ID, false, false),
            (FIRERED_LLM_GGML_ADAPTER_ID, false, false),
            (FUNASR_NANO_GGML_ADAPTER_ID, false, false),
            (MIMO_ASR_GGML_ADAPTER_ID, false, false),
            (MOSS_TD_GGML_ADAPTER_ID, false, false),
            (GRANITE_SPEECH_GGML_ADAPTER_ID, false, false),
        ];

        for (adapter_id, supports_translation_task, supports_source_language_hint) in expected {
            let descriptor = registry
                .find_by_adapter_id(adapter_id)
                .expect("builtin adapter must resolve to its canonical descriptor");
            assert_eq!(
                descriptor.execution_contract.supports_translation_task, supports_translation_task,
                "translation capability mismatch for {adapter_id}"
            );
            assert_eq!(
                descriptor.execution_contract.supports_source_language_hint,
                supports_source_language_hint,
                "source-language capability mismatch for {adapter_id}"
            );
        }
        assert!(registry.find_by_adapter_id("unknown-adapter").is_none());
    }

    #[test]
    fn native_family_integration_audit_covers_builtins() {
        crate::models::family_integration_audit::source_tree_audit::audit_builtin_native_family_integrations()
            .expect("builtin native families must satisfy the integration audit");
    }

    /// Pins transcript-attribution capabilities per builtin architecture --
    /// the single Rust-side declaration of these
    /// capability-single-source facts this test protects against silent drift.
    /// moss-transcribe-diarize is the only builtin family that segments
    /// speakers in-decoder today (cohere's decoder has the mode but no
    /// publishable pack, see its descriptor). The machine-readable inventory
    /// projects `emits_punctuation` into catalog authoring; the catalog test
    /// cross-checks the shipped result against
    /// [`emits_punctuation_for_model_architecture`].
    #[test]
    fn builtin_architectures_declare_transcript_capabilities() {
        let expected: &[(&str, SpeakerSegmentationSource, Option<bool>)] = &[
            (
                COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                SpeakerSegmentationSource::External,
                Some(true),
            ),
            (
                WHISPER_GGML_ARCHITECTURE_ID,
                SpeakerSegmentationSource::External,
                Some(true),
            ),
            (
                QWEN3_ASR_GGML_ARCHITECTURE_ID,
                SpeakerSegmentationSource::External,
                Some(true),
            ),
            (
                PARAKEET_CTC_GGML_ARCHITECTURE_ID,
                SpeakerSegmentationSource::External,
                None,
            ),
            (
                PARAKEET_TDT_GGML_ARCHITECTURE_ID,
                SpeakerSegmentationSource::External,
                Some(true),
            ),
            (
                WAV2VEC2_CTC_GGML_ARCHITECTURE_ID,
                SpeakerSegmentationSource::External,
                None,
            ),
            (
                XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
                SpeakerSegmentationSource::External,
                Some(true),
            ),
            (
                MOONSHINE_GGML_ARCHITECTURE_ID,
                SpeakerSegmentationSource::External,
                Some(true),
            ),
            (
                DOLPHIN_GGML_ARCHITECTURE_ID,
                SpeakerSegmentationSource::External,
                Some(false),
            ),
            (
                SENSEVOICE_GGML_ARCHITECTURE_ID,
                SpeakerSegmentationSource::External,
                Some(true),
            ),
            (
                FIRERED_AED_GGML_ARCHITECTURE_ID,
                SpeakerSegmentationSource::External,
                Some(false),
            ),
            (
                FIRERED_LLM_GGML_ARCHITECTURE_ID,
                SpeakerSegmentationSource::External,
                None,
            ),
            (
                FUNASR_NANO_GGML_ARCHITECTURE_ID,
                SpeakerSegmentationSource::External,
                None,
            ),
            (
                MIMO_ASR_GGML_ARCHITECTURE_ID,
                SpeakerSegmentationSource::External,
                None,
            ),
            (
                MOSS_TD_GGML_ARCHITECTURE_ID,
                SpeakerSegmentationSource::InDecoder,
                None,
            ),
            (
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
                SpeakerSegmentationSource::External,
                Some(true),
            ),
        ];
        let forced_aligner_word_timestamps = std::collections::BTreeSet::from([
            DOLPHIN_GGML_ARCHITECTURE_ID,
            SENSEVOICE_GGML_ARCHITECTURE_ID,
            FIRERED_AED_GGML_ARCHITECTURE_ID,
            FIRERED_LLM_GGML_ARCHITECTURE_ID,
            FUNASR_NANO_GGML_ARCHITECTURE_ID,
            MIMO_ASR_GGML_ARCHITECTURE_ID,
            MOSS_TD_GGML_ARCHITECTURE_ID,
            GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
        ]);
        let registry = OpenAsrArchitectureRegistry::with_builtins();
        let mut seen = std::collections::BTreeSet::new();

        for (model_architecture, speaker_segmentation, emits_punctuation) in
            expected.iter().copied()
        {
            let descriptor = registry
                .find_by_model_architecture(model_architecture)
                .unwrap_or_else(|| panic!("missing builtin architecture '{model_architecture}'"));
            assert_eq!(
                descriptor.execution_contract.speaker_segmentation, speaker_segmentation,
                "'{model_architecture}' speaker_segmentation mismatch"
            );
            assert_eq!(
                descriptor.execution_contract.emits_punctuation, emits_punctuation,
                "'{model_architecture}' emits_punctuation mismatch"
            );
            assert_eq!(
                emits_punctuation_for_model_architecture(model_architecture),
                emits_punctuation,
                "'{model_architecture}' accessor must match the descriptor field"
            );
            let expected_word_source =
                if forced_aligner_word_timestamps.contains(model_architecture) {
                    WordTimestampSource::ForcedAligner
                } else {
                    WordTimestampSource::Native
                };
            assert_eq!(
                descriptor.execution_contract.word_timestamp_source, expected_word_source,
                "'{model_architecture}' word_timestamp_source mismatch"
            );
            seen.insert(model_architecture);
        }

        assert_eq!(
            seen.len(),
            registry.descriptors().len(),
            "expectation table must cover every builtin architecture, no more, no less"
        );
    }

    /// The slice shape and the speaker-segmentation source are two views of
    /// one fact and must not drift: only a family that numbers speakers inside
    /// its own decode can make a slice a speaker scope, and a family that does
    /// numbers them per slice, so it must be `ScopedSlices`. A half-connect
    /// (an `InDecoder` family left on `SharedWindow`) would silently fuse two
    /// slices' unrelated `SPEAKER_01`s into one person, which is exactly the
    /// failure `diarize::voice_id::identity`'s scope model exists to prevent.
    #[test]
    fn builtin_architectures_declare_longform_slice_shape() {
        let registry = OpenAsrArchitectureRegistry::with_builtins();
        for descriptor in registry.descriptors() {
            let scoped = matches!(
                descriptor.execution_contract.longform_slice_shape,
                OpenAsrLongformSliceShape::ScopedSlices { .. }
            );
            assert_eq!(
                scoped,
                descriptor
                    .execution_contract
                    .speaker_segmentation
                    .is_in_decoder(),
                "'{}' slice shape and speaker_segmentation disagree",
                descriptor.identity.model_architecture
            );
            assert_eq!(
                longform_slice_shape_for_model_architecture(descriptor.identity.model_architecture),
                descriptor.execution_contract.longform_slice_shape,
                "'{}' accessor must match the descriptor field",
                descriptor.identity.model_architecture
            );
            if let OpenAsrLongformSliceShape::ScopedSlices {
                integral_seconds,
                target_seconds,
                max_seconds,
            } = descriptor.execution_contract.longform_slice_shape
            {
                assert!(
                    target_seconds.is_finite() && target_seconds > 0.0,
                    "'{}' target_seconds must be positive and finite",
                    descriptor.identity.model_architecture
                );
                assert!(
                    max_seconds >= target_seconds,
                    "'{}' max_seconds must not be tighter than target_seconds",
                    descriptor.identity.model_architecture
                );
                assert!(
                    integral_seconds.is_finite() && integral_seconds > 0.0,
                    "'{}' integral_seconds must be positive and finite",
                    descriptor.identity.model_architecture
                );
                // A recording the family would decode whole must never be
                // shorter than one it would cut into pieces: that ordering is
                // what makes slicing the fallback rather than a second path
                // running in parallel with the integral one.
                assert!(
                    integral_seconds >= max_seconds,
                    "'{}' integral_seconds must not be under max_seconds, or slicing would \
                     trigger on recordings the decoder can already serve whole",
                    descriptor.identity.model_architecture
                );
            }
        }
        assert_eq!(
            longform_slice_shape_for_model_architecture("not-a-builtin-architecture"),
            OpenAsrLongformSliceShape::SharedWindow,
        );
    }

    #[test]
    fn builtin_architectures_declare_semantic_invocation_spans() {
        let expected = [
            (WHISPER_GGML_ARCHITECTURE_ID, Some(30.0)),
            (FIRERED_LLM_GGML_ARCHITECTURE_ID, Some(40.0)),
            (FUNASR_NANO_GGML_ARCHITECTURE_ID, Some(40.0)),
            (MIMO_ASR_GGML_ARCHITECTURE_ID, Some(30.0)),
            (MOSS_TD_GGML_ARCHITECTURE_ID, Some(60.0)),
            (
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
                Some(GRANITE_SPEECH_MAX_INVOCATION_SECONDS as f32),
            ),
        ];
        let registry = OpenAsrArchitectureRegistry::with_builtins();
        for descriptor in registry.descriptors() {
            let expected_max = expected
                .iter()
                .find_map(|(architecture, max)| {
                    (*architecture == descriptor.identity.model_architecture).then_some(*max)
                })
                .flatten();
            assert_eq!(
                descriptor.max_single_invocation_seconds(),
                expected_max,
                "'{}' invocation span mismatch",
                descriptor.identity.model_architecture
            );
        }
    }

    /// Pins `auto_gpu_policy` per builtin architecture. Most builtins let
    /// Auto pick any GPU-class backend automatically when available
    /// (`AllBackends`), matching how `resolve_runtime_backend` behaves
    /// generically. xasr-zipformer and moonshine are `ExceptMetal` -- Auto still
    /// prefers the generic GPU lane (CUDA/HIP/Vulkan) but falls back to CPU
    /// on Apple Silicon Metal specifically because their current graph shapes
    /// are dispatch-bound there (an explicit `--backend metal` request is
    /// unaffected). Qwen remains explicitly `AllBackends`; this table must not
    /// broaden one family's platform gate to neighboring architectures.
    /// See the field doc and each family's own executor/
    /// `graph_config` doc comment for detail. A silent flip of this table
    /// would silently deny Auto users a GPU their hardware supports (or
    /// silently regress them onto a Metal path known to be slower), for any
    /// family.
    #[test]
    fn builtin_architectures_declare_auto_gpu_policy() {
        let expected: &[(&str, AutoGpuPolicy)] = &[
            (
                COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                AutoGpuPolicy::AllBackends,
            ),
            (WHISPER_GGML_ARCHITECTURE_ID, AutoGpuPolicy::AllBackends),
            (QWEN3_ASR_GGML_ARCHITECTURE_ID, AutoGpuPolicy::AllBackends),
            (
                PARAKEET_CTC_GGML_ARCHITECTURE_ID,
                AutoGpuPolicy::AllBackends,
            ),
            (
                PARAKEET_TDT_GGML_ARCHITECTURE_ID,
                AutoGpuPolicy::AllBackends,
            ),
            (
                WAV2VEC2_CTC_GGML_ARCHITECTURE_ID,
                AutoGpuPolicy::AllBackends,
            ),
            (
                XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
                AutoGpuPolicy::ExceptMetal,
            ),
            (MOONSHINE_GGML_ARCHITECTURE_ID, AutoGpuPolicy::ExceptMetal),
            (DOLPHIN_GGML_ARCHITECTURE_ID, AutoGpuPolicy::AllBackends),
            (SENSEVOICE_GGML_ARCHITECTURE_ID, AutoGpuPolicy::AllBackends),
            (FIRERED_AED_GGML_ARCHITECTURE_ID, AutoGpuPolicy::AllBackends),
            (FIRERED_LLM_GGML_ARCHITECTURE_ID, AutoGpuPolicy::AllBackends),
            (FUNASR_NANO_GGML_ARCHITECTURE_ID, AutoGpuPolicy::AllBackends),
            (MIMO_ASR_GGML_ARCHITECTURE_ID, AutoGpuPolicy::AllBackends),
            // Post-#212 quiet-window A/B: true accelerated Metal is faster
            // than CPU; Auto may select Metal (see descriptor note).
            (MOSS_TD_GGML_ARCHITECTURE_ID, AutoGpuPolicy::AllBackends),
            (
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
                AutoGpuPolicy::AllBackends,
            ),
        ];
        let registry = OpenAsrArchitectureRegistry::with_builtins();
        let mut seen = std::collections::BTreeSet::new();

        for (model_architecture, auto_gpu_policy) in expected.iter().copied() {
            let descriptor = registry
                .find_by_model_architecture(model_architecture)
                .unwrap_or_else(|| panic!("missing builtin architecture '{model_architecture}'"));
            assert_eq!(
                descriptor.optimization_contract.auto_gpu_policy, auto_gpu_policy,
                "'{model_architecture}' auto_gpu_policy mismatch"
            );
            assert_eq!(
                family_auto_gpu_policy_for_model_architecture(model_architecture),
                auto_gpu_policy,
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

    #[test]
    fn moonshine_declares_vulkan_hybrid_without_weakening_other_accelerators() {
        use crate::device::execution_policy::ExecutionPlacement;

        let capabilities = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(MOONSHINE_GGML_ARCHITECTURE_ID)
            .expect("Moonshine descriptor")
            .execution_contract
            .execution_capabilities;
        assert!(capabilities.supports_cpu());
        for provider in [
            ExecutionProvider::Metal,
            ExecutionProvider::Cuda,
            ExecutionProvider::Hip,
        ] {
            assert!(capabilities.supports(provider, ExecutionPlacement::FullDevice));
            assert!(!capabilities.supports(provider, ExecutionPlacement::Hybrid));
        }
        assert!(capabilities.supports(ExecutionProvider::Vulkan, ExecutionPlacement::Hybrid));
        assert!(!capabilities.supports(ExecutionProvider::Vulkan, ExecutionPlacement::FullDevice,));
    }

    #[test]
    fn dolphin_declares_vulkan_full_device_and_hybrid_without_weakening_other_accelerators() {
        use crate::device::execution_policy::ExecutionPlacement;

        let capabilities = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(DOLPHIN_GGML_ARCHITECTURE_ID)
            .expect("Dolphin descriptor")
            .execution_contract
            .execution_capabilities;
        assert!(capabilities.supports_cpu());
        for provider in [
            ExecutionProvider::Metal,
            ExecutionProvider::Cuda,
            ExecutionProvider::Hip,
        ] {
            assert!(capabilities.supports(provider, ExecutionPlacement::FullDevice));
            assert!(!capabilities.supports(provider, ExecutionPlacement::Hybrid));
        }
        assert!(capabilities.supports(ExecutionProvider::Vulkan, ExecutionPlacement::FullDevice));
        assert!(capabilities.supports(ExecutionProvider::Vulkan, ExecutionPlacement::Hybrid));
    }

    /// Regression for the platform-scoping guarantee itself: `ExceptMetal`
    /// families must gate Auto to CPU on Metal while leaving a resolved
    /// generic-GPU-lane pick (CUDA/HIP/Vulkan) or CPU pick untouched, and an
    /// explicit `execution_target=accelerated`/`=cpu` request must always
    /// win regardless of the family's policy.
    #[test]
    fn except_metal_family_gates_only_apple_silicon_metal() {
        use crate::ggml_runtime::{
            GgmlCpuGraphBackend, GgmlCpuGraphConfig, RequestBackendPreference,
            ResolvedFamilyRuntimeInput,
        };

        for model_architecture in [
            XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
            MOONSHINE_GGML_ARCHITECTURE_ID,
        ] {
            let policy = family_auto_gpu_policy_for_model_architecture(model_architecture);
            assert_eq!(policy, AutoGpuPolicy::ExceptMetal);

            // Auto: gated to CPU only if the generic resolver would have picked
            // Metal specifically. `resolve` is a pure function here -- no
            // thread-local install/read round-trip.
            let resolved = GgmlCpuGraphConfig::runtime_default().backend;
            let gated = ResolvedFamilyRuntimeInput::resolve(None, policy).backend();
            if matches!(resolved, GgmlCpuGraphBackend::Metal) {
                assert_eq!(gated, GgmlCpuGraphBackend::Cpu);
            } else {
                assert_eq!(gated, resolved);
            }
            assert_ne!(gated, GgmlCpuGraphBackend::Metal);

            // An explicit accelerated request always wins, even on Metal: the
            // gate only ever pins Auto, so an explicit preference must still
            // resolve to a GPU-class backend regardless of `policy`.
            let accelerated = ResolvedFamilyRuntimeInput::resolve(
                Some(RequestBackendPreference::Accelerated),
                policy,
            )
            .backend();
            assert!(accelerated.is_gpu_class());

            // An explicit CPU-only request always wins too.
            assert_eq!(
                ResolvedFamilyRuntimeInput::resolve(
                    Some(RequestBackendPreference::CpuOnly),
                    policy,
                )
                .backend(),
                GgmlCpuGraphBackend::Cpu
            );
        }
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
            (
                FIRERED_LLM_GGML_ARCHITECTURE_ID,
                OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                    max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
                },
            ),
            (
                FUNASR_NANO_GGML_ARCHITECTURE_ID,
                OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                    max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
                },
            ),
            (
                MIMO_ASR_GGML_ARCHITECTURE_ID,
                OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                    max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
                },
            ),
            (
                MOSS_TD_GGML_ARCHITECTURE_ID,
                OpenAsrEncoderAttentionSpan::FixedWindow,
            ),
            (
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
                OpenAsrEncoderAttentionSpan::LocalChunked,
            ),
        ];
        let registry = OpenAsrArchitectureRegistry::with_builtins();
        let mut seen = std::collections::BTreeSet::new();

        for (model_architecture, expected_span) in expected.iter().copied() {
            let descriptor = registry
                .find_by_model_architecture(model_architecture)
                .unwrap_or_else(|| panic!("missing builtin architecture '{model_architecture}'"));
            assert_eq!(
                descriptor.optimization_contract.encoder_attention_span, expected_span,
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
                optimization_contract: OpenAsrOptimizationContract {
                    encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                        max_safe_chunk_seconds: bad_value,
                    },
                    ..base.optimization_contract
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

        assert_eq!(descriptor.identity.model_family, "whisper");
        assert_eq!(
            descriptor.identity.model_architecture,
            WHISPER_GGML_ARCHITECTURE_ID
        );
        assert_eq!(
            descriptor.pack_contract.audio_frontend_id,
            WHISPER_AUDIO_FRONTEND_ID
        );
        assert_eq!(
            descriptor.pack_contract.runtime_tensor_contract_id,
            WHISPER_RUNTIME_TENSOR_CONTRACT_ID
        );
        assert_eq!(
            descriptor.execution_contract.executor_component_id,
            WHISPER_EXECUTOR_COMPONENT_ID
        );
    }

    #[test]
    fn finds_xasr_zipformer_architecture_by_runtime_alias() {
        let descriptor = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_runtime_architecture_alias("xasr-zh-en")
            .expect("xasr alias");

        assert_eq!(
            descriptor.identity.model_family,
            XASR_ZIPFORMER_MODEL_FAMILY
        );
        assert_eq!(
            descriptor.identity.model_architecture,
            XASR_ZIPFORMER_GGML_ARCHITECTURE_ID
        );
        assert_eq!(
            descriptor.pack_contract.runtime_tensor_contract_id,
            XASR_ZIPFORMER_RUNTIME_TENSOR_CONTRACT_ID
        );
        assert_eq!(
            descriptor.execution_contract.execution_capability,
            GgmlExecutionCapability::DedicatedRuntimeExecutorV1
        );
        assert!(matches!(
            descriptor.topology_contract.block_stack,
            OpenAsrBlockStackStrategy::ArchitectureGraph { .. }
        ));
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
    fn builtin_architectures_have_non_empty_unique_hparam_schemas() {
        // validate_references walks each schema; this also exercises the
        // empty/duplicate guards that run at production dispatch build time.
        OpenAsrArchitectureRegistry::with_builtins()
            .validate_references()
            .expect("builtin hparam schemas must be non-empty and duplicate-free");

        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            for key in descriptor.pack_contract.hparam_schema {
                assert!(
                    !key.is_empty(),
                    "hparam key in architecture '{}' must be non-empty",
                    descriptor.identity.model_architecture
                );
            }
        }
    }

    #[test]
    fn validate_references_rejects_quantization_contract_owned_by_another_architecture() {
        let base = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(WHISPER_GGML_ARCHITECTURE_ID)
            .expect("whisper architecture");
        let descriptor = OpenAsrArchitectureDescriptor {
            quantization_contract: OpenAsrQuantizationContract {
                tensor_classification: crate::models::qwen::TENSOR_QUANTIZATION_CONTRACT,
            },
            ..base
        };

        assert!(matches!(
            OpenAsrArchitectureRegistry::validate_quantization_contract(descriptor),
            Err(
                OpenAsrArchitectureRegistryError::QuantizationArchitectureMismatch {
                    model_architecture: WHISPER_GGML_ARCHITECTURE_ID,
                    quantization_architecture: QWEN3_ASR_GGML_ARCHITECTURE_ID,
                }
            )
        ));
    }

    #[test]
    fn validate_references_rejects_invalid_identity_language_facts() {
        let base = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID)
            .expect("cohere architecture");

        let empty = OpenAsrArchitectureDescriptor {
            identity: OpenAsrIdentityContract {
                recognized_languages: &[],
                ..base.identity
            },
            ..base
        };
        assert!(matches!(
            OpenAsrArchitectureRegistry::validate_identity(empty),
            Err(OpenAsrArchitectureRegistryError::RecognizedLanguagesEmpty {
                model_architecture: COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID
            })
        ));

        let malformed = OpenAsrArchitectureDescriptor {
            identity: OpenAsrIdentityContract {
                recognized_languages: &["en", "EN"],
                ..base.identity
            },
            ..base
        };
        assert!(matches!(
            OpenAsrArchitectureRegistry::validate_identity(malformed),
            Err(
                OpenAsrArchitectureRegistryError::RecognizedLanguageMalformed {
                    model_architecture: COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                    language: "EN"
                }
            )
        ));

        let unsorted = OpenAsrArchitectureDescriptor {
            identity: OpenAsrIdentityContract {
                recognized_languages: &["zh", "en"],
                ..base.identity
            },
            ..base
        };
        assert!(matches!(
            OpenAsrArchitectureRegistry::validate_identity(unsorted),
            Err(
                OpenAsrArchitectureRegistryError::RecognizedLanguagesNotSortedUnique {
                    model_architecture: COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                    language: "en"
                }
            )
        ));

        let default_not_recognized = OpenAsrArchitectureDescriptor {
            identity: OpenAsrIdentityContract {
                recognized_languages: &["de"],
                ..base.identity
            },
            ..base
        };
        assert!(matches!(
            OpenAsrArchitectureRegistry::validate_identity(default_not_recognized),
            Err(
                OpenAsrArchitectureRegistryError::PromptDefaultLanguageNotRecognized {
                    model_architecture: COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                    default_language: "en"
                }
            )
        ));
    }

    #[test]
    fn validate_references_rejects_language_hint_mismatch_and_duplicate_module_slug() {
        let monolingual = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(PARAKEET_CTC_GGML_ARCHITECTURE_ID)
            .expect("parakeet architecture");
        let monolingual_mismatch = OpenAsrArchitectureDescriptor {
            identity: OpenAsrIdentityContract {
                recognized_languages: &["en", "zh"],
                ..monolingual.identity
            },
            ..monolingual
        };
        assert!(matches!(
            OpenAsrArchitectureRegistry::validate_identity(monolingual_mismatch),
            Err(
                OpenAsrArchitectureRegistryError::FixedMonolingualLanguageMismatch {
                    model_architecture: PARAKEET_CTC_GGML_ARCHITECTURE_ID,
                    language: "en"
                }
            )
        ));

        let multilingual = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(XASR_ZIPFORMER_GGML_ARCHITECTURE_ID)
            .expect("xasr architecture");
        let multilingual_mismatch = OpenAsrArchitectureDescriptor {
            identity: OpenAsrIdentityContract {
                recognized_languages: &["en"],
                ..multilingual.identity
            },
            ..multilingual
        };
        assert!(matches!(
            OpenAsrArchitectureRegistry::validate_identity(multilingual_mismatch),
            Err(
                OpenAsrArchitectureRegistryError::FixedMultilingualLanguagesMismatch {
                    model_architecture: XASR_ZIPFORMER_GGML_ARCHITECTURE_ID
                }
            )
        ));

        let duplicate = OpenAsrArchitectureDescriptor {
            identity: OpenAsrIdentityContract {
                model_architecture: "duplicate-module-architecture",
                ..multilingual.identity
            },
            ..multilingual
        };
        let descriptors = [multilingual, duplicate];
        assert!(matches!(
            OpenAsrArchitectureRegistry::validate_module_slug_uniqueness(&descriptors),
            Err(OpenAsrArchitectureRegistryError::DuplicateModuleSlug {
                module_slug: "xasr_zipformer",
                first_model_architecture: XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
                duplicate_model_architecture: "duplicate-module-architecture"
            })
        ));
    }

    #[test]
    fn validate_references_rejects_invalid_invocation_span_cap() {
        let base = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(WHISPER_GGML_ARCHITECTURE_ID)
            .expect("whisper architecture");

        for bad_value in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
            let descriptor = OpenAsrArchitectureDescriptor {
                execution_contract: OpenAsrExecutionContract {
                    invocation_span: OpenAsrInvocationSpan::Bounded {
                        max_seconds: bad_value,
                    },
                    ..base.execution_contract
                },
                ..base
            };
            let error = OpenAsrArchitectureRegistry::validate_invocation_span(descriptor)
                .expect_err("non-finite/non-positive invocation span must fail closed");
            match error {
                OpenAsrArchitectureRegistryError::InvocationSpanNotFinitePositive {
                    model_architecture,
                    max_seconds,
                } => {
                    assert_eq!(model_architecture, WHISPER_GGML_ARCHITECTURE_ID);
                    assert!(
                        max_seconds == bad_value || (max_seconds.is_nan() && bad_value.is_nan())
                    );
                }
                other => panic!("unexpected error variant: {other:?}"),
            }
        }
    }

    #[test]
    fn scoped_slices_require_a_direct_invocation_bound_covering_the_envelope() {
        let base = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(MOSS_TD_GGML_ARCHITECTURE_ID)
            .expect("moss architecture");

        for invocation_span in [
            OpenAsrInvocationSpan::Elastic,
            OpenAsrInvocationSpan::Bounded { max_seconds: 59.0 },
        ] {
            let descriptor = OpenAsrArchitectureDescriptor {
                execution_contract: OpenAsrExecutionContract {
                    invocation_span,
                    ..base.execution_contract
                },
                ..base
            };
            assert!(matches!(
                OpenAsrArchitectureRegistry::validate_invocation_span(descriptor),
                Err(
                    OpenAsrArchitectureRegistryError::ScopedSliceInvocationSpanMismatch {
                        model_architecture: MOSS_TD_GGML_ARCHITECTURE_ID,
                        required_seconds: 60.0,
                        ..
                    }
                )
            ));
        }
        OpenAsrArchitectureRegistry::validate_invocation_span(base)
            .expect("MOSS 60-second direct bound covers its scoped-slice envelope");
    }

    #[test]
    fn builtin_block_stacks_declare_expected_shapes() {
        let registry = OpenAsrArchitectureRegistry::with_builtins();

        let qwen = registry
            .find_by_model_architecture(QWEN3_ASR_GGML_ARCHITECTURE_ID)
            .expect("qwen architecture");
        let qwen_stack = match qwen.topology_contract.block_stack {
            OpenAsrBlockStackStrategy::Shared(stack) => stack,
            OpenAsrBlockStackStrategy::ArchitectureGraph { reason } => {
                panic!("qwen unexpectedly uses an architecture graph: {reason}")
            }
        };
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
        let cohere_stack = match cohere.topology_contract.block_stack {
            OpenAsrBlockStackStrategy::Shared(stack) => stack,
            OpenAsrBlockStackStrategy::ArchitectureGraph { reason } => {
                panic!("cohere unexpectedly uses an architecture graph: {reason}")
            }
        };
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
        assert!(matches!(
            whisper.topology_contract.block_stack,
            OpenAsrBlockStackStrategy::ArchitectureGraph { .. }
        ));
    }

    #[test]
    fn block_stack_validation_rejects_layer_count_key_outside_schema() {
        let base = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(QWEN3_ASR_GGML_ARCHITECTURE_ID)
            .expect("qwen architecture");
        let descriptor = OpenAsrArchitectureDescriptor {
            topology_contract: OpenAsrTopologyContract {
                block_stack: OpenAsrBlockStackStrategy::Shared(OpenAsrBlockStackDescriptor {
                    orchestration_shape: OpenAsrOrchestrationShape::LlmDecoder,
                    encoder_stage: None,
                    decoder_stage: Some(OpenAsrStageDescriptor {
                        block_kind: OpenAsrBlockKind::LlmDecoderLayer,
                        // Not a member of QWEN3_ASR_HPARAM_SCHEMA.
                        layer_count_hparam: "qwen3-asr.llm.layers_typo",
                        tensor_name_scope: "blk",
                    }),
                }),
                ..base.topology_contract
            },
            ..base
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
        let base = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(QWEN3_ASR_GGML_ARCHITECTURE_ID)
            .expect("qwen architecture");
        let descriptor = OpenAsrArchitectureDescriptor {
            topology_contract: OpenAsrTopologyContract {
                block_stack: OpenAsrBlockStackStrategy::Shared(OpenAsrBlockStackDescriptor {
                    orchestration_shape: OpenAsrOrchestrationShape::LlmDecoder,
                    encoder_stage: None,
                    decoder_stage: Some(OpenAsrStageDescriptor {
                        block_kind: OpenAsrBlockKind::LlmDecoderLayer,
                        layer_count_hparam: QWEN3_LLM_LAYERS_KEY,
                        tensor_name_scope: "",
                    }),
                }),
                ..base.topology_contract
            },
            ..base
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
        let base = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(QWEN3_ASR_GGML_ARCHITECTURE_ID)
            .expect("qwen architecture");
        let descriptor = OpenAsrArchitectureDescriptor {
            topology_contract: OpenAsrTopologyContract {
                block_stack: OpenAsrBlockStackStrategy::Shared(OpenAsrBlockStackDescriptor {
                    orchestration_shape: OpenAsrOrchestrationShape::LlmDecoder,
                    encoder_stage: None,
                    decoder_stage: Some(OpenAsrStageDescriptor {
                        block_kind: OpenAsrBlockKind::Seq2SeqDecoderLayer,
                        layer_count_hparam: QWEN3_LLM_LAYERS_KEY,
                        tensor_name_scope: "blk",
                    }),
                }),
                ..base.topology_contract
            },
            ..base
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
        let base = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID)
            .expect("cohere architecture");
        let descriptor = OpenAsrArchitectureDescriptor {
            topology_contract: OpenAsrTopologyContract {
                block_stack: OpenAsrBlockStackStrategy::Shared(OpenAsrBlockStackDescriptor {
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
                ..base.topology_contract
            },
            ..base
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
            topology_contract: OpenAsrTopologyContract {
                block_stack: OpenAsrBlockStackStrategy::Shared(OpenAsrBlockStackDescriptor {
                    orchestration_shape: OpenAsrOrchestrationShape::Ctc,
                    encoder_stage: Some(OpenAsrStageDescriptor {
                        block_kind: OpenAsrBlockKind::SanMFsmnEncoderLayer,
                        layer_count_hparam: "parakeet.n_layers",
                        tensor_name_scope: "enc.blk",
                    }),
                    decoder_stage: None,
                }),
                ..parakeet.topology_contract
            },
            ..parakeet
        };

        assert_eq!(
            OpenAsrArchitectureRegistry::validate_block_stack(descriptor),
            Ok(())
        );

        // And a decoder stage under the Ctc shape must still fail closed.
        let with_decoder = OpenAsrArchitectureDescriptor {
            topology_contract: OpenAsrTopologyContract {
                block_stack: OpenAsrBlockStackStrategy::Shared(OpenAsrBlockStackDescriptor {
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
                ..parakeet.topology_contract
            },
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
                    descriptor.identity.model_architecture
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
        let base = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(QWEN3_ASR_GGML_ARCHITECTURE_ID)
            .expect("qwen architecture");
        let synthetic = OpenAsrArchitectureDescriptor {
            identity: OpenAsrIdentityContract {
                model_architecture: SYNTHETIC_ARCH,
                ..base.identity
            },
            topology_contract: OpenAsrTopologyContract {
                block_stack: OpenAsrBlockStackStrategy::Shared(OpenAsrBlockStackDescriptor {
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
                ..base.topology_contract
            },
            ..base
        };

        // 1. Passes the S5a startup gate with no new shape/kind/error.
        OpenAsrArchitectureRegistry::validate_block_stack(synthetic)
            .expect("a new LlmDecoder-shape model is valid as pure data");

        let block_stack = match &synthetic.topology_contract.block_stack {
            OpenAsrBlockStackStrategy::Shared(stack) => Some(stack),
            OpenAsrBlockStackStrategy::ArchitectureGraph { .. } => None,
        };
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
            identity: OpenAsrIdentityContract {
                model_architecture: CTC_ARCH,
                ..base.identity
            },
            topology_contract: OpenAsrTopologyContract {
                decode_driver: OpenAsrDecodeDriverStrategy::SharedCtcGreedy {
                    policy: decode_policy::PARAKEET_CTC_DECODE_POLICY_COMPONENT,
                },
                block_stack: OpenAsrBlockStackStrategy::Shared(OpenAsrBlockStackDescriptor {
                    orchestration_shape: OpenAsrOrchestrationShape::Ctc,
                    encoder_stage: Some(OpenAsrStageDescriptor {
                        block_kind: OpenAsrBlockKind::ConformerBlock,
                        layer_count_hparam: ENC_KEY,
                        tensor_name_scope: "enc.blk",
                    }),
                    decoder_stage: None,
                }),
                ..base.topology_contract
            },
            ..base
        };
        OpenAsrArchitectureRegistry::validate_block_stack(ctc)
            .expect("encoder-only Ctc stack is valid");

        let ctc_block_stack = match &ctc.topology_contract.block_stack {
            OpenAsrBlockStackStrategy::Shared(stack) => Some(stack),
            OpenAsrBlockStackStrategy::ArchitectureGraph { .. } => None,
        };

        let encoder_plan = StageBuildPlan {
            block_kind: OpenAsrBlockKind::ConformerBlock,
            tensor_name_scope: "enc.blk",
            family_layer_count: 24,
        };
        // The encoder stage drives through the SAME shared gate as data.
        assert_eq!(
            validate_stage_against_descriptor(
                CTC_ARCH,
                ctc_block_stack,
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
                ctc_block_stack,
                OpenAsrStageRole::Decoder,
                OpenAsrOrchestrationShape::Ctc,
                encoder_plan,
                &CtcResolver,
            ),
            Err(ShapeOrchestratorError::DecoderRequestedForCtcShape {
                model_architecture: CTC_ARCH,
            }),
        );

        // A Ctc stack that wrongly declares a decoder stage is rejected.
        let ctc_with_decoder = OpenAsrArchitectureDescriptor {
            identity: OpenAsrIdentityContract {
                model_architecture: CTC_ARCH,
                ..base.identity
            },
            topology_contract: OpenAsrTopologyContract {
                decode_driver: OpenAsrDecodeDriverStrategy::SharedCtcGreedy {
                    policy: decode_policy::PARAKEET_CTC_DECODE_POLICY_COMPONENT,
                },
                block_stack: OpenAsrBlockStackStrategy::Shared(OpenAsrBlockStackDescriptor {
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
                ..base.topology_contract
            },
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
            topology_contract: OpenAsrTopologyContract {
                block_stack: OpenAsrBlockStackStrategy::Shared(OpenAsrBlockStackDescriptor {
                    orchestration_shape: OpenAsrOrchestrationShape::LlmDecoder,
                    encoder_stage: Some(OpenAsrStageDescriptor {
                        block_kind: OpenAsrBlockKind::TransformerEncoderLayer,
                        layer_count_hparam: QWEN3_AUDIO_LAYERS_KEY,
                        tensor_name_scope: "audio.blk",
                    }),
                    decoder_stage: None,
                }),
                ..base.topology_contract
            },
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
