pub(crate) mod block_stack;
pub(crate) mod hparams;
pub(crate) mod shape_orchestrator;

use std::collections::BTreeMap;

use crate::capacity::{CapacityAudioBound, CapacityModelDeclaration, CapacityModelDescriptor};
use crate::ggml_runtime::AutoGpuPolicy;
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
    FIRERED_LLM_HPARAM_SCHEMA, FUNASR_NANO_HPARAM_SCHEMA, GRANITE_SPEECH_HPARAM_SCHEMA,
    MIMO_ASR_HPARAM_SCHEMA, MOONSHINE_HPARAM_SCHEMA, MOSS_TD_HPARAM_SCHEMA,
    PARAKEET_CTC_HPARAM_SCHEMA, PARAKEET_TDT_HPARAM_SCHEMA, QWEN3_ARCHITECTURE_VALUE,
    QWEN3_ASR_HPARAM_SCHEMA, QWEN3_AUDIO_LAYERS_KEY, QWEN3_LLM_LAYERS_KEY,
    SENSEVOICE_HPARAM_SCHEMA, WAV2VEC2_CTC_HPARAM_SCHEMA, WHISPER_HPARAM_SCHEMA,
    XASR_ZIPFORMER_HPARAM_SCHEMA,
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

// firered-llm (FireRedTeam/FireRedASR2-LLM: the firered-aed Conformer encoder
// (independently-trained weights, NOT byte-identical to firered-aed-l-v2 --
// see scratchpad/fr2/T1-findings.md S3, joint finetune not frozen-encoder
// reuse) + a 2x frame-stacking Adapter (2 Linear + ReLU) + a LoRA-merged
// Qwen2-7B-Instruct decoder, Apache-2.0). Like firered-aed, decode runs on a
// hand-written dedicated executor (block_stack: None) -- the Conformer
// encoder + Qwen2 decoder shapes are family-specific, not composer block
// kinds -- registered in `BUILTIN_COMPONENT_DESCRIPTORS` /
// `BUILTIN_ARCHITECTURE_DESCRIPTORS` below.
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
// runs on a hand-written dedicated executor (block_stack: None) -- registered in
// `BUILTIN_COMPONENT_DESCRIPTORS` / `BUILTIN_ARCHITECTURE_DESCRIPTORS` below.
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
// hand-written dedicated executor (block_stack: None).
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
// (block_stack: None) -- registered in `BUILTIN_COMPONENT_DESCRIPTORS` /
// `BUILTIN_ARCHITECTURE_DESCRIPTORS` below.
pub(crate) const MOSS_TD_GGML_ARCHITECTURE_ID: &str = "moss-transcribe-diarize-whisper-qwen3";
pub(crate) const MOSS_TD_GGML_ADAPTER_ID: &str = "ggml-family-moss-transcribe-diarize-runtime-v1";
pub(crate) const MOSS_TD_MODEL_FAMILY: &str = "moss-transcribe-diarize";
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
// so this stays on a hand-written dedicated executor (block_stack: None).
pub(crate) const GRANITE_SPEECH_GGML_ARCHITECTURE_ID: &str = "granite-speech";
pub(crate) const GRANITE_SPEECH_GGML_ADAPTER_ID: &str = "ggml-family-granite-speech-runtime-v1";
pub(crate) const GRANITE_SPEECH_MODEL_FAMILY: &str = "granite-speech";
pub(crate) const GRANITE_SPEECH_AUDIO_FRONTEND_ID: &str = "granite-speech.mel80x2.16khz.mono.v0";
pub(crate) const GRANITE_SPEECH_TOKENIZER_ID: &str = "granite-speech.gpt2-bpe.v0";
pub(crate) const GRANITE_SPEECH_DECODE_POLICY_ID: &str = "granite-speech.greedy.seq2seq.v0";
pub(crate) const GRANITE_SPEECH_RUNTIME_TENSOR_CONTRACT_ID: &str =
    "granite-speech.runtime-tensors.v0";
pub(crate) const GRANITE_SPEECH_EXECUTOR_COMPONENT_ID: &str = "granite-speech.ggml-executor.v0";

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
/// `max_chunk_seconds`, 120s by default -- silently collapsed the slicer's
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
/// family work without a speaker-embedder pack installed, and lets an
/// embedder-equipped host name the speakers of a self-segmenting family.
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
    /// `crate::diarize::pipeline::Diarization` boundary without any family
    /// needing to change.
    External,
}

impl SpeakerSegmentationSource {
    pub fn is_in_decoder(self) -> bool {
        matches!(self, Self::InDecoder)
    }
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
    /// Such a family also pins its own slice window: an autoregressive decoder
    /// that folds the whole slice into one prompt has a hard position budget
    /// (prompt + generation), so the window is a decoder-context fact the
    /// family owns, not a generic default. `target_seconds` is the window the
    /// slicer aims for and `max_seconds` the ceiling it may stretch a slice to
    /// when no cut point is available earlier; both must leave room for the
    /// family's decode budget inside its context.
    ///
    /// `integral_seconds` is the longest recording this family can fold into a
    /// SINGLE prompt and still be granted a decode budget that covers its
    /// densest measured demand. Up to it, slicing is not applied at all.
    /// Slicing is a degradation, not the normal path: every seam restarts the
    /// in-decoder speaker numbering and forces cross-slice identity to be
    /// re-established from voice evidence alone, and the shared slicer's
    /// cut-point search can clip speech. Measured on Mandarin meeting audio,
    /// decoding whole beat slice-and-stitch by a wide margin, so the family
    /// takes the integral path whenever its context can honestly serve it and
    /// falls back to `target_seconds` slices only past that point. It is a
    /// derived quantity, not a tuning knob: it must equal the largest window
    /// whose prompt plus required budget still fits the decoder's KV capacity,
    /// and the owning family pins it against that arithmetic in a test. As of
    /// Phase 0 the owning family also derives it in parallel from pack
    /// metadata (`crate::capacity::derive_integral_seconds`) and pins derived
    /// == declared; Phase 1 moves the derived value onto the loaded pack so
    /// this literal becomes the derivation's resolved output.
    ScopedSlices {
        integral_seconds: f32,
        target_seconds: f32,
        max_seconds: f32,
    },
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
/// partials. Declared once on the architecture integration descriptor and
/// derived into the streaming dispatch at build time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingPartialGranularity {
    FrameSync,
    Buffered,
}

/// Whether this family's greedy decode must ride a shared driver registry
/// entry (`decode_policy_component_registry`) or intentionally uses a
/// dedicated non-shared loop (transducer / attention-rescoring / ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenAsrSharedDecodeDriver {
    SharedSeq2SeqGreedy,
    SharedCtcGreedy,
    Dedicated,
}

/// Pack-import surface for one native family. File existence alone is not
/// enough: `CoreConvert` symbols must be force-linked by
/// `models::pack_import_surface`, and `ExternalTooling` paths must resolve
/// under the repo root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenAsrPackImportSurface {
    CoreConvert { symbol: &'static str },
    ExternalTooling { relative_path: &'static str },
}

/// Static integration obligations for one native family.
///
/// Authoritative runtime facts (phrase-bias capability, streaming partial
/// granularity, shared-decode driver class) live here and are *derived into*
/// dispatch/capability paths. Optional tooling stays optional (`None`) rather
/// than forcing a placeholder implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OpenAsrFamilyIntegrationDescriptor {
    /// Publish-catalog family id (may differ from `model_family`, e.g. `cohere`
    /// vs `cohere-transcribe`). Used to join the shared pre-audit family list
    /// and the `docs/model-audits/<id>.md` form path.
    pub catalog_family_id: &'static str,
    pub supports_phrase_bias: bool,
    pub streaming_partial_granularity: StreamingPartialGranularity,
    pub shared_decode_driver: OpenAsrSharedDecodeDriver,
    pub pack_import: OpenAsrPackImportSurface,
    pub reference_dumper_source: Option<&'static str>,
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
    pub integration: OpenAsrFamilyIntegrationDescriptor,
    pub execution_capability: GgmlExecutionCapability,
    pub prefer_cpu_decoder_for_multichunk_metal: bool,
    /// Which GPU-class backend(s) Auto execution may select automatically
    /// when available (see [`crate::ggml_runtime::AutoGpuPolicy`]). This can
    /// only ever pin Auto to CPU -- it never overrides an explicit
    /// `execution_target=accelerated` (or `cpu`) request, which always gets
    /// exactly what it asked for via
    /// `GgmlCpuGraphConfig::resolve_family_runtime_backend`.
    ///
    /// Most builtins default `AllBackends` (the old blanket `true`). Two
    /// families flipped from CPU-pinned to `AllBackends` after their encoder
    /// weights moved into a Metal-offloadable static arena and Metal beat CPU
    /// end-to-end: Dolphin first (see
    /// `models::dolphin::executor::dolphin_runtime_backend`), then
    /// xasr-zipformer's streaming encoder once the same weight-placement fix
    /// landed for it. A later, cleaner platform audit found xasr-zipformer's
    /// streaming encoder itself dispatch-bound and net-slower on Apple
    /// Silicon Metal specifically (a 29-frame chunk graph too small to
    /// amortize Metal's per-dispatch overhead) -- see
    /// `models::xasr_zipformer::graph_config::encoder_gpu_enabled` -- so it
    /// was for a time the sole builtin `ExceptMetal`. moss-transcribe-diarize
    /// briefly shared that pin and has since returned to `AllBackends` after
    /// the post-#212 quiet-window true-accelerated Metal win (see that
    /// descriptor's `auto_gpu_policy` note). That same audit also measured a
    /// Metal slowdown for qwen and moonshine, but neither is gated: qwen's
    /// looks like a fixed size x quant platform trade-off rather than a code
    /// bug, and that read is left to a dedicated follow-up before being baked
    /// into the default (see `models::qwen::graph_config`); moonshine's had
    /// an actual architectural fix applied instead of being gated around
    /// (decoder scheduler-off activates the reusable incremental decode
    /// graph, see `models::moonshine::graph_config`). Any provenance/telemetry
    /// label reporting the backend a request actually ran on must resolve
    /// through `GgmlCpuGraphConfig::resolve_family_runtime_backend` with this
    /// same policy, not recompute generically (a generic recompute is what
    /// produced a `core.native.backend:metal` label on a gated-family Auto
    /// request that in fact ran entirely on CPU).
    pub auto_gpu_policy: crate::ggml_runtime::AutoGpuPolicy,
    /// Where this family's speaker structure comes from. The single
    /// declaration of this architecture-level fact -- runtime dispatch reads
    /// it via `GgmlFamilyAdapterDescriptor::speaker_segmentation` rather than
    /// matching on `adapter_id`. See [`SpeakerSegmentationSource`].
    pub speaker_segmentation: SpeakerSegmentationSource,
    /// How one recording is cut up for this family before decode. See
    /// [`OpenAsrLongformSliceShape`]; the shape must agree with
    /// `speaker_segmentation` (only an `InDecoder` family can be
    /// `ScopedSlices`), which
    /// `builtin_architectures_declare_longform_slice_shape` pins.
    pub(crate) longform_slice_shape: OpenAsrLongformSliceShape,
    /// How this family's single-decode capacity is established. A mandatory
    /// field (not `Option`, same reasoning as `encoder_attention_span` below)
    /// forcing every new architecture to place itself in exactly one bucket
    /// at compile time -- see [`crate::capacity::CapacityModelDeclaration`]
    /// for the three buckets and why `None`-shaped ambiguity is not on
    /// offer. `builtin_architectures_declare_capacity_model` pins the
    /// per-family declarations and
    /// `family_integration_audit` refuses a `Derived` family whose
    /// `audio_frontend_id` has no `crate::capacity` frontend-registry row.
    pub(crate) capacity_model: crate::capacity::CapacityModelDeclaration,
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
            speaker_segmentation: self.speaker_segmentation,
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

/// How one recording is cut up for a builtin family before decode, looked up
/// by GGUF `model_architecture`. An unrecognized architecture gets
/// [`OpenAsrLongformSliceShape::SharedWindow`], the shape that needs nothing
/// from the family beyond a plain decode.
pub(crate) fn longform_slice_shape_for_model_architecture(
    model_architecture: &str,
) -> OpenAsrLongformSliceShape {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(model_architecture)
        .map(|descriptor| descriptor.longform_slice_shape)
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
        .map(|descriptor| descriptor.auto_gpu_policy)
        .unwrap_or(crate::ggml_runtime::AutoGpuPolicy::AllBackends)
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
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::AudioFrontend,
        id: FIRERED_LLM_AUDIO_FRONTEND_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::DecodePolicy,
        id: FIRERED_LLM_DECODE_POLICY_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::RuntimeTensorContract,
        id: FIRERED_LLM_RUNTIME_TENSOR_CONTRACT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Tokenizer,
        id: FIRERED_LLM_TOKENIZER_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Executor,
        id: FIRERED_LLM_EXECUTOR_COMPONENT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::AudioFrontend,
        id: FUNASR_NANO_AUDIO_FRONTEND_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::DecodePolicy,
        id: FUNASR_NANO_DECODE_POLICY_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::RuntimeTensorContract,
        id: FUNASR_NANO_RUNTIME_TENSOR_CONTRACT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Tokenizer,
        id: FUNASR_NANO_TOKENIZER_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Executor,
        id: FUNASR_NANO_EXECUTOR_COMPONENT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::AudioFrontend,
        id: MIMO_ASR_AUDIO_FRONTEND_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::DecodePolicy,
        id: MIMO_ASR_DECODE_POLICY_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::RuntimeTensorContract,
        id: MIMO_ASR_RUNTIME_TENSOR_CONTRACT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Tokenizer,
        id: MIMO_ASR_TOKENIZER_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Executor,
        id: MIMO_ASR_EXECUTOR_COMPONENT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::AudioFrontend,
        id: MOSS_TD_AUDIO_FRONTEND_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::DecodePolicy,
        id: MOSS_TD_DECODE_POLICY_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::RuntimeTensorContract,
        id: MOSS_TD_RUNTIME_TENSOR_CONTRACT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Tokenizer,
        id: MOSS_TD_TOKENIZER_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Executor,
        id: MOSS_TD_EXECUTOR_COMPONENT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::AudioFrontend,
        id: GRANITE_SPEECH_AUDIO_FRONTEND_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::DecodePolicy,
        id: GRANITE_SPEECH_DECODE_POLICY_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::RuntimeTensorContract,
        id: GRANITE_SPEECH_RUNTIME_TENSOR_CONTRACT_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Tokenizer,
        id: GRANITE_SPEECH_TOKENIZER_ID,
    },
    OpenAsrComponentDescriptor {
        kind: OpenAsrComponentKind::Executor,
        id: GRANITE_SPEECH_EXECUTOR_COMPONENT_ID,
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
        integration: OpenAsrFamilyIntegrationDescriptor {
            catalog_family_id: "cohere",
            supports_phrase_bias: true,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            shared_decode_driver: OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_cohere_source_to_runtime_pack",
            },
            reference_dumper_source: None,
        },
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
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
        speaker_segmentation: SpeakerSegmentationSource::External,
        longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
        // GQA-free transformer decoder with a real self-KV cache, but audio
        // length is bounded by the ENCODER positional table, not the decoder
        // context (transcribe.cpp `LimitsBasis::audio_from_caps` bucket): the
        // two constraints resolve separately, so the capacity model derives
        // from pack metadata with the encoder named as the audio bound.
        capacity_model: CapacityModelDeclaration::Derived(CapacityModelDescriptor {
            audio_bound: CapacityAudioBound::EncoderSpan,
        }),
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
        integration: OpenAsrFamilyIntegrationDescriptor {
            catalog_family_id: "whisper",
            supports_phrase_bias: true,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            shared_decode_driver: OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_whisper_hf_source_to_runtime_pack",
            },
            reference_dumper_source: None,
        },
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        auto_gpu_policy: AutoGpuPolicy::AllBackends,
        speaker_segmentation: SpeakerSegmentationSource::External,
        longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
        // Architecture-fixed 30s log-mel window: the encoder never attends
        // past its own 1500-position chunk and longform audio is looped over
        // independent 30s windows, so audio length is bounded by that fixed
        // window, not by the (448-position) decoder context.
        capacity_model: CapacityModelDeclaration::BoundedElsewhere {
            by: "fixed 30s encoder window",
        },
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
        integration: OpenAsrFamilyIntegrationDescriptor {
            catalog_family_id: "qwen",
            supports_phrase_bias: true,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            shared_decode_driver: OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_qwen_source_to_runtime_pack",
            },
            reference_dumper_source: None,
        },
        execution_capability: GgmlExecutionCapability::NativeGraphLoweringV1,
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
        speaker_segmentation: SpeakerSegmentationSource::External,
        longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
        // GQA LLM decoder whose prompt carries the audio tokens, so the
        // decoder KV context bounds audio length -- fully derivable from pack
        // metadata (llm.n_layers / llm_kv_heads / head_dim / llm_max_positions
        // are all required keys).
        capacity_model: CapacityModelDeclaration::Derived(CapacityModelDescriptor {
            audio_bound: CapacityAudioBound::DecoderContext,
        }),
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
        integration: OpenAsrFamilyIntegrationDescriptor {
            catalog_family_id: "parakeet",
            supports_phrase_bias: true,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            shared_decode_driver: OpenAsrSharedDecodeDriver::SharedCtcGreedy,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_parakeet_ctc_source_to_runtime_pack",
            },
            reference_dumper_source: None,
        },
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        auto_gpu_policy: AutoGpuPolicy::AllBackends,
        speaker_segmentation: SpeakerSegmentationSource::External,
        longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
        // CTC: no autoregressive decoder, hence no decoder KV cache.
        capacity_model: CapacityModelDeclaration::NoDecoderKv,
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
        integration: OpenAsrFamilyIntegrationDescriptor {
            catalog_family_id: "parakeet-tdt",
            supports_phrase_bias: false,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            shared_decode_driver: OpenAsrSharedDecodeDriver::Dedicated,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_parakeet_tdt_source_to_runtime_pack",
            },
            reference_dumper_source: None,
        },
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        auto_gpu_policy: AutoGpuPolicy::AllBackends,
        speaker_segmentation: SpeakerSegmentationSource::External,
        longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
        // Transducer with constant-size prediction state: no decoder KV
        // cache to size.
        capacity_model: CapacityModelDeclaration::NoDecoderKv,
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
        integration: OpenAsrFamilyIntegrationDescriptor {
            catalog_family_id: "wav2vec2",
            supports_phrase_bias: true,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            shared_decode_driver: OpenAsrSharedDecodeDriver::SharedCtcGreedy,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_wav2vec2_ctc_source_to_runtime_pack",
            },
            reference_dumper_source: None,
        },
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        auto_gpu_policy: AutoGpuPolicy::AllBackends,
        speaker_segmentation: SpeakerSegmentationSource::External,
        longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
        // CTC: no autoregressive decoder, hence no decoder KV cache.
        capacity_model: CapacityModelDeclaration::NoDecoderKv,
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
        integration: OpenAsrFamilyIntegrationDescriptor {
            catalog_family_id: "xasr-zipformer",
            supports_phrase_bias: false,
            streaming_partial_granularity: StreamingPartialGranularity::FrameSync,
            shared_decode_driver: OpenAsrSharedDecodeDriver::Dedicated,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_xasr_zipformer_source_to_runtime_pack",
            },
            reference_dumper_source: None,
        },
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
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
        speaker_segmentation: SpeakerSegmentationSource::External,
        longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
        // Streaming transducer with constant-size state: no decoder KV cache
        // to size.
        capacity_model: CapacityModelDeclaration::NoDecoderKv,
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
        integration: OpenAsrFamilyIntegrationDescriptor {
            catalog_family_id: "moonshine",
            supports_phrase_bias: true,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            shared_decode_driver: OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_moonshine_source_to_runtime_pack",
            },
            reference_dumper_source: None,
        },
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        auto_gpu_policy: AutoGpuPolicy::AllBackends,
        speaker_segmentation: SpeakerSegmentationSource::External,
        longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
        // MHA self-KV decoder, but the RoPE encoder span is not what a single
        // decode runs into: the decoder's own output context (decoder.max_ctx,
        // a required pack key) bounds the transcript one decode can produce.
        capacity_model: CapacityModelDeclaration::BoundedElsewhere {
            by: "decoder.max_ctx output bound",
        },
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
        integration: OpenAsrFamilyIntegrationDescriptor {
            catalog_family_id: "dolphin",
            supports_phrase_bias: true,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            shared_decode_driver: OpenAsrSharedDecodeDriver::Dedicated,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_dolphin_wenet_source_to_runtime_pack",
            },
            reference_dumper_source: None,
        },
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
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
        speaker_segmentation: SpeakerSegmentationSource::External,
        longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
        // Attention decoder with self-KV, but audio length runs into the
        // sinusoidal positional table (decoder.max_ctx), not the KV budget;
        // the decoder head_dim is also not declared pack metadata, so the KV
        // figure is not derivable without a pack revision this family does
        // not need.
        capacity_model: CapacityModelDeclaration::BoundedElsewhere {
            by: "decoder.max_ctx positional span",
        },
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
        integration: OpenAsrFamilyIntegrationDescriptor {
            catalog_family_id: "sensevoice",
            supports_phrase_bias: true,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            shared_decode_driver: OpenAsrSharedDecodeDriver::SharedCtcGreedy,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_sensevoice_source_to_runtime_pack",
            },
            reference_dumper_source: None,
        },
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        auto_gpu_policy: AutoGpuPolicy::AllBackends,
        speaker_segmentation: SpeakerSegmentationSource::External,
        longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
        // CTC: no autoregressive decoder, hence no decoder KV cache.
        capacity_model: CapacityModelDeclaration::NoDecoderKv,
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
        integration: OpenAsrFamilyIntegrationDescriptor {
            catalog_family_id: "firered-aed",
            supports_phrase_bias: false,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            shared_decode_driver: OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_firered_aed_source_to_runtime_pack",
            },
            reference_dumper_source: Some("tooling/firered2-reference-dumper/dump_aed_encoder.py"),
        },
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        auto_gpu_policy: AutoGpuPolicy::AllBackends,
        speaker_segmentation: SpeakerSegmentationSource::External,
        longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
        // Attention decoder with self-KV, but audio length runs into the
        // encoder positional encoding (decoder.pe_len), not the KV budget;
        // decoder heads/head_dim are also not declared pack metadata, so the
        // KV figure is not derivable without a pack revision this family does
        // not need.
        capacity_model: CapacityModelDeclaration::BoundedElsewhere {
            by: "decoder.pe_len positional encoding",
        },
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
    OpenAsrArchitectureDescriptor {
        runtime_architecture_aliases: &[FIRERED_LLM_GGML_ARCHITECTURE_ID, "firered2-llm"],
        model_family: FIRERED_LLM_MODEL_FAMILY,
        model_architecture: FIRERED_LLM_GGML_ARCHITECTURE_ID,
        adapter_id: FIRERED_LLM_GGML_ADAPTER_ID,
        // No language-selection prompt token and no decode-time detection:
        // the Qwen2 BPE vocab covers Mandarin + English (the upstream ASR
        // finetune's training languages), same shape as firered-aed.
        language_family_hint: LanguageFamilyHint::FixedMultilingual {
            languages: &["zh", "en"],
        },
        audio_frontend_id: FIRERED_LLM_AUDIO_FRONTEND_ID,
        runtime_tensor_contract_id: FIRERED_LLM_RUNTIME_TENSOR_CONTRACT_ID,
        tokenizer_id: FIRERED_LLM_TOKENIZER_ID,
        decode_policy_id: FIRERED_LLM_DECODE_POLICY_ID,
        executor_component_id: FIRERED_LLM_EXECUTOR_COMPONENT_ID,
        integration: OpenAsrFamilyIntegrationDescriptor {
            catalog_family_id: "firered2-llm",
            supports_phrase_bias: false,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            shared_decode_driver: OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_firered_llm_source_to_runtime_pack",
            },
            reference_dumper_source: Some("tooling/firered2-reference-dumper/dump_reference.py"),
        },
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        auto_gpu_policy: AutoGpuPolicy::AllBackends,
        speaker_segmentation: SpeakerSegmentationSource::External,
        longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
        // GQA Qwen2 decoder whose prompt carries the adapted audio tokens, so
        // the decoder KV context bounds audio length -- fully derivable from
        // pack metadata (llm.n_layers / llm.n_kv_heads / llm.head_dim /
        // llm.max_positions are all required keys).
        capacity_model: CapacityModelDeclaration::Derived(CapacityModelDescriptor {
            audio_bound: CapacityAudioBound::DecoderContext,
        }),
        // Qwen2's ChatML decode is a plain transcription completion -- no
        // learned punctuation-suppression behavior has been characterized
        // for this family yet (unlike firered-aed's punctuation-free
        // char+SPM vocab); leave unclaimed rather than assert an unverified
        // capability.
        emits_punctuation: None,
        hparam_schema: FIRERED_LLM_HPARAM_SCHEMA,
        // Conformer encoder + Qwen2 decoder-only decode both stay on the
        // dedicated executor (neither shape is a composer block kind), so no
        // data-driven block-stack descriptor.
        block_stack: None,
        // Same Conformer encoder shape as firered-aed (full self-attention
        // over the whole chunk, quadratic in chunk length -- issue #68), plus
        // the upstream's own explicit 40s HARD cap (not just guidance --
        // `FireRedLlmGgmlExecutor` fails closed above it). 30s stays
        // comfortably under both that hard cap and firered-aed's own
        // guidance-based ceiling.
        encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
            max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
        },
    },
    OpenAsrArchitectureDescriptor {
        runtime_architecture_aliases: &[FUNASR_NANO_GGML_ARCHITECTURE_ID, "funasr-nano"],
        model_family: FUNASR_NANO_MODEL_FAMILY,
        model_architecture: FUNASR_NANO_GGML_ARCHITECTURE_ID,
        adapter_id: FUNASR_NANO_GGML_ADAPTER_ID,
        // No language-selection prompt token and no decode-time detection: the
        // stock Qwen3 BPE vocab covers Mandarin + English (Fun-ASR-Nano's
        // trained ASR languages).
        language_family_hint: LanguageFamilyHint::FixedMultilingual {
            languages: &["zh", "en"],
        },
        audio_frontend_id: FUNASR_NANO_AUDIO_FRONTEND_ID,
        runtime_tensor_contract_id: FUNASR_NANO_RUNTIME_TENSOR_CONTRACT_ID,
        tokenizer_id: FUNASR_NANO_TOKENIZER_ID,
        decode_policy_id: FUNASR_NANO_DECODE_POLICY_ID,
        executor_component_id: FUNASR_NANO_EXECUTOR_COMPONENT_ID,
        integration: OpenAsrFamilyIntegrationDescriptor {
            catalog_family_id: "funasr-nano",
            supports_phrase_bias: false,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            shared_decode_driver: OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy,
            pack_import: OpenAsrPackImportSurface::ExternalTooling {
                relative_path: "tooling/publish-model/scripts/funasr_nano_pt_to_safetensors.py",
            },
            reference_dumper_source: Some(
                "tooling/publish-model/scripts/funasr_nano_reference_oracle.py",
            ),
        },
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        auto_gpu_policy: AutoGpuPolicy::AllBackends,
        speaker_segmentation: SpeakerSegmentationSource::External,
        longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
        // GQA Qwen3 decoder whose prompt carries the adapted audio tokens, so
        // the decoder KV context bounds audio length -- fully derivable from
        // pack metadata (funasr.llm.n_layers / n_kv_heads / head_dim /
        // max_positions are all required keys).
        capacity_model: CapacityModelDeclaration::Derived(CapacityModelDescriptor {
            audio_bound: CapacityAudioBound::DecoderContext,
        }),
        // The stock Qwen3 ChatML decode emits ordinary punctuation, but no
        // punctuation-suppression behavior has been separately characterized;
        // leave unclaimed rather than assert a capability beyond the two golden
        // clips.
        emits_punctuation: None,
        hparam_schema: FUNASR_NANO_HPARAM_SCHEMA,
        // SAN-M encoder + Qwen3 decoder-only decode both stay on the dedicated
        // executor (neither shape is a composer block kind), so no data-driven
        // block-stack descriptor.
        block_stack: None,
        // SAN-M encoder is full self-attention over the whole chunk (quadratic
        // in chunk length), plus the upstream's own ~40s HARD cap
        // (`FunasrNanoGgmlExecutor` fails closed above it). 30s stays under both.
        encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
            max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
        },
    },
    OpenAsrArchitectureDescriptor {
        runtime_architecture_aliases: &[MIMO_ASR_GGML_ARCHITECTURE_ID],
        model_family: MIMO_ASR_MODEL_FAMILY,
        model_architecture: MIMO_ASR_GGML_ARCHITECTURE_ID,
        adapter_id: MIMO_ASR_GGML_ADAPTER_ID,
        // No language-selection prompt token and no decode-time detection:
        // the Qwen2 BPE vocab covers the upstream's trained languages
        // (Mandarin, English, Cantonese + regional dialects per its README).
        language_family_hint: LanguageFamilyHint::FixedMultilingual {
            languages: &["zh", "en", "yue"],
        },
        audio_frontend_id: MIMO_ASR_AUDIO_FRONTEND_ID,
        runtime_tensor_contract_id: MIMO_ASR_RUNTIME_TENSOR_CONTRACT_ID,
        tokenizer_id: MIMO_ASR_TOKENIZER_ID,
        decode_policy_id: MIMO_ASR_DECODE_POLICY_ID,
        executor_component_id: MIMO_ASR_EXECUTOR_COMPONENT_ID,
        integration: OpenAsrFamilyIntegrationDescriptor {
            catalog_family_id: "mimo-asr",
            supports_phrase_bias: false,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            shared_decode_driver: OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy,
            pack_import: OpenAsrPackImportSurface::ExternalTooling {
                relative_path: "tooling/mimo-asr/convert_mimo_asr.py",
            },
            reference_dumper_source: None,
        },
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
        prefer_cpu_decoder_for_multichunk_metal: false,
        auto_gpu_policy: AutoGpuPolicy::AllBackends,
        speaker_segmentation: SpeakerSegmentationSource::External,
        longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
        // GQA Qwen2 decoder whose prompt carries the audio tokens, so the
        // decoder KV context bounds audio length -- fully derivable from pack
        // metadata (mimo.llm.block_count / attention.head_count_kv /
        // attention.key_length / context_length are all required keys).
        capacity_model: CapacityModelDeclaration::Derived(CapacityModelDescriptor {
            audio_bound: CapacityAudioBound::DecoderContext,
        }),
        // No characterized punctuation behavior for this family yet (unlike
        // firered-aed's punctuation-free vocab) -- leave unclaimed rather
        // than assert an unverified capability.
        emits_punctuation: None,
        hparam_schema: MIMO_ASR_HPARAM_SCHEMA,
        // Audio-tokenizer encoder + RVQ + input-local + Qwen2 decode all stay
        // on the dedicated executor (none of these stages is a composer
        // block kind), so no data-driven block-stack descriptor.
        block_stack: None,
        // The 32L rope audio-tokenizer encoder is full self-attention over
        // the whole chunk: quadratic in chunk length. The executor's own
        // 30s-per-chunk hard cap (mirroring the reference `preprocess_input`'s
        // 30s re-chunking) keeps this well inside the shared default.
        encoder_attention_span: OpenAsrEncoderAttentionSpan::GlobalQuadratic {
            max_safe_chunk_seconds: DEFAULT_ENCODER_SAFE_CHUNK_SECONDS,
        },
    },
    OpenAsrArchitectureDescriptor {
        runtime_architecture_aliases: &[MOSS_TD_GGML_ARCHITECTURE_ID],
        model_family: MOSS_TD_MODEL_FAMILY,
        model_architecture: MOSS_TD_GGML_ARCHITECTURE_ID,
        adapter_id: MOSS_TD_GGML_ADAPTER_ID,
        // The Qwen3 decoder auto-detects/produces the transcript language
        // through free-text instruction-following (no dedicated language
        // token in its vocab, same shape as qwen3-asr) -- until prompt-level
        // language conditioning is wired and verified against a real pack, an
        // explicit hint is rejected (not faked) rather than silently ignored.
        language_family_hint: LanguageFamilyHint::SelfDetectsRejectsHint {
            reject_reason: "MOSS-Transcribe-Diarize auto-detects the source language via its Qwen3 decoder and does not accept an explicit selection; use a multilingual Whisper pack to force or report a language.",
        },
        audio_frontend_id: MOSS_TD_AUDIO_FRONTEND_ID,
        runtime_tensor_contract_id: MOSS_TD_RUNTIME_TENSOR_CONTRACT_ID,
        tokenizer_id: MOSS_TD_TOKENIZER_ID,
        decode_policy_id: MOSS_TD_DECODE_POLICY_ID,
        executor_component_id: MOSS_TD_EXECUTOR_COMPONENT_ID,
        integration: OpenAsrFamilyIntegrationDescriptor {
            catalog_family_id: "moss-transcribe-diarize",
            supports_phrase_bias: false,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            shared_decode_driver: OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_moss_transcribe_diarize_source_to_runtime_pack",
            },
            reference_dumper_source: Some("tooling/moss-reference-dumper/dump_golden.py"),
        },
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
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
        speaker_segmentation: SpeakerSegmentationSource::InDecoder,
        // Every slice is its own speaker scope (the decode restarts `[S01]`
        // numbering per slice), and the window is a decoder-context fact: the
        // Qwen3 decoder folds the whole slice into one prompt inside
        // `MOSS_TD_MAX_KV_CACHE_POSITIONS` (8192) positions. Worst case at
        // `max_seconds` = 240s: 86 fixed wrapper tokens (measured from the
        // real tokenized prompt) + 375 audio tokens per 30s encoder chunk
        // (8 chunks = 3000) + 124 time-marker digit tokens + the full
        // `MOSS_TD_MAX_GENERATED_TOKENS` decode budget (4096) = 7306
        // positions. The window is chosen so a slice-length request can always
        // be granted that entire budget: dense overlapping meeting audio
        // exhausts anything smaller, and exhausting the budget returns nothing
        // at all rather than degrading.
        //
        // Wanting the window longer runs straight into that arithmetic; wanting
        // it shorter costs identity. A slice is how much context the in-decoder
        // diarizer gets to hold one speaker together, and every seam between
        // slices is a place cross-slice identity has to be re-established from
        // voice evidence alone. 180s is the target because it leaves the
        // stretch room to 240s for finding a real pause to cut on.
        // `integral_seconds` = 300s: the longest single-prompt request whose
        // audio prompt (86 fixed wrapper tokens + 375 audio tokens per 30s
        // encoder chunk + the marker track's digit tokens -- 160 at 300s)
        // still leaves a budget covering the densest measured demand
        // (12.7 tokens/s, so 3810) inside the 8192-position decoder context:
        // 3996 + 3810 = 7806 <= 8192. 330s does not fit (4389 prompt + the
        // 4096 generation backstop = 8485), so 300s is the ceiling, not a
        // preference. Derived from the pack's own metadata by
        // `crate::capacity::derive_integral_seconds` and pinned equal to this
        // declared value by
        // `moss_transcribe_diarize::capacity::tests::derived_integral_window_equals_the_declared_constant`
        // (Phase 0: the derivation runs in parallel with zero production
        // callers; Phase 1 moves the derived value onto the loaded pack), and
        // pinned from both sides against the arithmetic by
        // `the_integral_window_is_the_largest_one_the_context_can_serve`.
        longform_slice_shape: OpenAsrLongformSliceShape::ScopedSlices {
            integral_seconds: 300.0,
            target_seconds: 180.0,
            max_seconds: 240.0,
        },
        // GQA Qwen3 decoder whose prompt carries the audio span, so the
        // decoder KV context bounds audio length -- fully derivable from pack
        // metadata (moss_td.llm.n_layers / n_kv_heads / head_dim /
        // max_positions and moss_td.adaptor.merge_size are all required
        // keys). The ONLY family whose audio is decoded whole up to its
        // integral window today, so it is also the only family that consumes
        // the derived integral window: `moss_transcribe_diarize::capacity`
        // derives it from the loaded pack's metadata and pins the result
        // equal to the declared `integral_seconds: 300.0` above (Phase 0 --
        // parallel derivation, zero production callers; Phase 1 moves the
        // derived value onto the loaded pack).
        capacity_model: CapacityModelDeclaration::Derived(CapacityModelDescriptor {
            audio_bound: CapacityAudioBound::DecoderContext,
        }),
        // The fixed instruction asks for full punctuation-bearing prose
        // segments; no characterized counter-example has been observed yet,
        // but this has not been verified against enough real transcripts to
        // assert as a capability -- leave unclaimed rather than guess.
        emits_punctuation: None,
        hparam_schema: MOSS_TD_HPARAM_SCHEMA,
        // Whisper encoder + Qwen3 decoder-only decode both stay on the
        // dedicated executor (neither shape is a composer block kind), so no
        // data-driven block-stack descriptor.
        block_stack: None,
        // Whisper's own architecture-fixed 30s log-mel window: the encoder
        // never attends past its own fixed 1500-position chunk no matter how
        // long the total requested audio is (the executor loops the encoder
        // over independent 30s windows and concatenates -- see `executor.rs`'s
        // module doc), so this needs no additional longform safety cap --
        // same classification as `whisper` itself.
        encoder_attention_span: OpenAsrEncoderAttentionSpan::FixedWindow,
    },
    OpenAsrArchitectureDescriptor {
        runtime_architecture_aliases: &[GRANITE_SPEECH_GGML_ARCHITECTURE_ID],
        model_family: GRANITE_SPEECH_MODEL_FAMILY,
        model_architecture: GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
        adapter_id: GRANITE_SPEECH_GGML_ADAPTER_ID,
        // The Granite decoder auto-detects/produces the transcript language
        // through free-text instruction-following (no dedicated language
        // token; the model card documents multilingual prompts working
        // without a language selector) -- same shape as qwen3-asr/moss-td:
        // reject an explicit hint rather than silently ignore it.
        language_family_hint: LanguageFamilyHint::SelfDetectsRejectsHint {
            reject_reason: "Granite Speech auto-detects the source language through free-text prompting and does not accept an explicit selection.",
        },
        audio_frontend_id: GRANITE_SPEECH_AUDIO_FRONTEND_ID,
        runtime_tensor_contract_id: GRANITE_SPEECH_RUNTIME_TENSOR_CONTRACT_ID,
        tokenizer_id: GRANITE_SPEECH_TOKENIZER_ID,
        decode_policy_id: GRANITE_SPEECH_DECODE_POLICY_ID,
        executor_component_id: GRANITE_SPEECH_EXECUTOR_COMPONENT_ID,
        integration: OpenAsrFamilyIntegrationDescriptor {
            catalog_family_id: "granite-speech",
            // Native keyword biasing via the prompt convention (see
            // `granite_speech::executor::supports_phrase_bias` and the
            // keyword-list end-to-end coverage), not the shared decode-time
            // logit-boost path; the family declares its own true.
            supports_phrase_bias: true,
            streaming_partial_granularity: StreamingPartialGranularity::Buffered,
            // Greedy decode rides the one shared seq2seq driver via the
            // decode-policy registry (see AGENTS.md's single-driver invariant);
            // this family provides a `Seq2SeqGreedyDecodeStepExecutor` and a
            // `GRANITE_SPEECH_DECODE_POLICY_ID` descriptor rather than a
            // hand-rolled argmax loop.
            shared_decode_driver: OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy,
            pack_import: OpenAsrPackImportSurface::CoreConvert {
                symbol: "convert_local_granite_speech_source_to_runtime_pack",
            },
            reference_dumper_source: None,
        },
        execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
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
        speaker_segmentation: SpeakerSegmentationSource::External,
        // External families ride the shared generic longform window (slices
        // are never their own speaker scope). This is not the whole-recording
        // single-prompt `ScopedSlices` case (only moss-transcribe-diarize is),
        // so no integral window is declared or consumed here.
        longform_slice_shape: OpenAsrLongformSliceShape::SharedWindow,
        // The projected audio tokens splice into the Granite decoder prompt,
        // but a single decode only ever sees one shared-window slice (this is
        // `SharedWindow`, not the whole-recording integral `ScopedSlices`
        // case), and the decoder declares no max-position/context pack key, so
        // there is no KV integral figure to derive without a pack revision
        // this landing does not need -- the shared window is what bounds the
        // audio per decode. Same shape as firered-aed/moonshine above.
        // (Promoting this to a `Derived` decoder-context model is deferred to
        // the capacity-derivation follow-up.)
        capacity_model: CapacityModelDeclaration::BoundedElsewhere {
            by: "shared-window slice; no decoder max-position pack key to derive from",
        },
        // The model card documents punctuation/truecasing as a real,
        // evaluated capability (a documented prompt variant + reported PER/
        // Cap-F1 metrics), and the family's own end-to-end golden samples
        // come out correctly punctuated -- unlike `MIMO_ASR`'s "not
        // characterized yet" case above, this one has been observed.
        emits_punctuation: Some(true),
        hparam_schema: GRANITE_SPEECH_HPARAM_SCHEMA,
        // Conformer encoder + Q-Former projector + Granite decoder all stay
        // on the dedicated executor (none of the three is a composer block
        // kind), so no data-driven block-stack descriptor.
        block_stack: None,
        // The Conformer encoder's self-attention is local to non-overlapping
        // `context_size=200`-frame blocks (Shaw relative-position attention),
        // never global over the whole utterance -- memory is bounded per
        // block regardless of total audio length, matching `LocalChunked`
        // exactly (see `encoder_graph.rs`'s module doc). This is a real,
        // verified difference from every other family's classification here:
        // it is neither `whisper`'s architecture-fixed 30s window nor a
        // quadratic-over-the-whole-chunk encoder.
        encoder_attention_span: OpenAsrEncoderAttentionSpan::LocalChunked,
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

    #[test]
    fn native_family_integration_audit_covers_builtins() {
        crate::models::family_integration_audit::source_tree_audit::audit_builtin_native_family_integrations()
            .expect("builtin native families must satisfy the integration audit");
    }

    /// Pins `speaker_segmentation` and `emits_punctuation` per builtin
    /// architecture -- the single Rust-side declaration of both
    /// capability-single-source facts this test protects against silent drift.
    /// moss-transcribe-diarize is the only builtin family that segments
    /// speakers in-decoder today (cohere's decoder has the mode but no
    /// publishable pack, see its descriptor); `emits_punctuation` values mirror
    /// `tooling/publish-model/scripts/_catalog.py`'s `PUNCTUATION_BY_FAMILY`
    /// (`registry/tests/catalog.rs`'s `embedded_catalog_emits_punctuation_matches_family`
    /// cross-checks the shipped catalog against
    /// [`emits_punctuation_for_model_architecture`] so the two stay in lockstep).
    #[test]
    fn builtin_architectures_declare_speaker_segmentation_and_emits_punctuation() {
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
        let registry = OpenAsrArchitectureRegistry::with_builtins();
        let mut seen = std::collections::BTreeSet::new();

        for (model_architecture, speaker_segmentation, emits_punctuation) in
            expected.iter().copied()
        {
            let descriptor = registry
                .find_by_model_architecture(model_architecture)
                .unwrap_or_else(|| panic!("missing builtin architecture '{model_architecture}'"));
            assert_eq!(
                descriptor.speaker_segmentation, speaker_segmentation,
                "'{model_architecture}' speaker_segmentation mismatch"
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
                descriptor.longform_slice_shape,
                OpenAsrLongformSliceShape::ScopedSlices { .. }
            );
            assert_eq!(
                scoped,
                descriptor.speaker_segmentation.is_in_decoder(),
                "'{}' slice shape and speaker_segmentation disagree",
                descriptor.model_architecture
            );
            assert_eq!(
                longform_slice_shape_for_model_architecture(descriptor.model_architecture),
                descriptor.longform_slice_shape,
                "'{}' accessor must match the descriptor field",
                descriptor.model_architecture
            );
            if let OpenAsrLongformSliceShape::ScopedSlices {
                integral_seconds,
                target_seconds,
                max_seconds,
            } = descriptor.longform_slice_shape
            {
                assert!(
                    target_seconds.is_finite() && target_seconds > 0.0,
                    "'{}' target_seconds must be positive and finite",
                    descriptor.model_architecture
                );
                assert!(
                    max_seconds >= target_seconds,
                    "'{}' max_seconds must not be tighter than target_seconds",
                    descriptor.model_architecture
                );
                assert!(
                    integral_seconds.is_finite() && integral_seconds > 0.0,
                    "'{}' integral_seconds must be positive and finite",
                    descriptor.model_architecture
                );
                // A recording the family would decode whole must never be
                // shorter than one it would cut into pieces: that ordering is
                // what makes slicing the fallback rather than a second path
                // running in parallel with the integral one.
                assert!(
                    integral_seconds >= max_seconds,
                    "'{}' integral_seconds must not be under max_seconds, or slicing would \
                     trigger on recordings the decoder can already serve whole",
                    descriptor.model_architecture
                );
            }
        }
        assert_eq!(
            longform_slice_shape_for_model_architecture("not-a-builtin-architecture"),
            OpenAsrLongformSliceShape::SharedWindow,
        );
    }

    /// Pins the capacity-model bucket of every builtin architecture -- the
    /// exhaustive companion to the field's compile-time enforcement. The
    /// field being mandatory only forces SOME declaration; this test forces
    /// the RIGHT one per family, so reclassifying a family (or onboarding a
    /// fifteenth) lands as a deliberate diff here rather than a silent
    /// `NoDecoderKv` on a family that very much has a decoder KV cache.
    /// Buckets follow the design review's taxonomy: five LLM-decoder
    /// families derive, five CTC/transducer families have no decoder KV, and
    /// four are bounded elsewhere than their decoder context.
    #[test]
    fn builtin_architectures_declare_capacity_model() {
        use crate::capacity::{
            CapacityAudioBound, CapacityModelDeclaration, CapacityModelDescriptor,
        };

        let derived = |audio_bound| {
            CapacityModelDeclaration::Derived(CapacityModelDescriptor { audio_bound })
        };
        let expected: &[(&str, CapacityModelDeclaration)] = &[
            (
                COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                derived(CapacityAudioBound::EncoderSpan),
            ),
            (
                WHISPER_GGML_ARCHITECTURE_ID,
                CapacityModelDeclaration::BoundedElsewhere {
                    by: "fixed 30s encoder window",
                },
            ),
            (
                QWEN3_ASR_GGML_ARCHITECTURE_ID,
                derived(CapacityAudioBound::DecoderContext),
            ),
            (
                PARAKEET_CTC_GGML_ARCHITECTURE_ID,
                CapacityModelDeclaration::NoDecoderKv,
            ),
            (
                PARAKEET_TDT_GGML_ARCHITECTURE_ID,
                CapacityModelDeclaration::NoDecoderKv,
            ),
            (
                WAV2VEC2_CTC_GGML_ARCHITECTURE_ID,
                CapacityModelDeclaration::NoDecoderKv,
            ),
            (
                XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
                CapacityModelDeclaration::NoDecoderKv,
            ),
            (
                MOONSHINE_GGML_ARCHITECTURE_ID,
                CapacityModelDeclaration::BoundedElsewhere {
                    by: "decoder.max_ctx output bound",
                },
            ),
            (
                DOLPHIN_GGML_ARCHITECTURE_ID,
                CapacityModelDeclaration::BoundedElsewhere {
                    by: "decoder.max_ctx positional span",
                },
            ),
            (
                SENSEVOICE_GGML_ARCHITECTURE_ID,
                CapacityModelDeclaration::NoDecoderKv,
            ),
            (
                FIRERED_AED_GGML_ARCHITECTURE_ID,
                CapacityModelDeclaration::BoundedElsewhere {
                    by: "decoder.pe_len positional encoding",
                },
            ),
            (
                FIRERED_LLM_GGML_ARCHITECTURE_ID,
                derived(CapacityAudioBound::DecoderContext),
            ),
            (
                FUNASR_NANO_GGML_ARCHITECTURE_ID,
                derived(CapacityAudioBound::DecoderContext),
            ),
            (
                MIMO_ASR_GGML_ARCHITECTURE_ID,
                derived(CapacityAudioBound::DecoderContext),
            ),
            (
                MOSS_TD_GGML_ARCHITECTURE_ID,
                derived(CapacityAudioBound::DecoderContext),
            ),
            (
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
                CapacityModelDeclaration::BoundedElsewhere {
                    by: "shared-window slice; no decoder max-position pack key to derive from",
                },
            ),
        ];
        let registry = OpenAsrArchitectureRegistry::with_builtins();
        let mut seen = std::collections::BTreeSet::new();

        for (model_architecture, capacity_model) in expected.iter().copied() {
            let descriptor = registry
                .find_by_model_architecture(model_architecture)
                .unwrap_or_else(|| panic!("missing builtin architecture '{model_architecture}'"));
            assert_eq!(
                descriptor.capacity_model, capacity_model,
                "'{model_architecture}' capacity_model mismatch"
            );
            if let CapacityModelDeclaration::BoundedElsewhere { by } = capacity_model {
                assert!(
                    !by.is_empty(),
                    "'{model_architecture}' BoundedElsewhere must name its bound"
                );
            }
            seen.insert(model_architecture);
        }

        assert_eq!(
            seen.len(),
            registry.descriptors().len(),
            "expectation table must cover every builtin architecture, no more, no less"
        );
    }

    /// Every family that declares a derived capacity model must have its
    /// versioned frontend id registered in `crate::capacity`'s frontend
    /// table, so derivation can never read a frontend fact that does not
    /// exist. (The same check runs in the release-path family integration
    /// audit; this test pins the builtin set specifically.)
    #[test]
    fn derived_capacity_families_have_registered_frontends() {
        let registry = OpenAsrArchitectureRegistry::with_builtins();
        let mut derived_count = 0usize;
        for descriptor in registry.descriptors() {
            if matches!(
                descriptor.capacity_model,
                crate::capacity::CapacityModelDeclaration::Derived(_)
            ) {
                derived_count += 1;
                assert!(
                    crate::capacity::frontend_capacity_basis(descriptor.audio_frontend_id)
                        .is_some(),
                    "'{}' declares a derived capacity model but frontend id '{}' has no \
                     capacity-frontend registry row",
                    descriptor.model_architecture,
                    descriptor.audio_frontend_id
                );
            }
        }
        // Guards the walk: the design review's taxonomy derives exactly six
        // builtin families (qwen3-asr, cohere, firered-llm, mimo-asr,
        // funasr-nano -- all DecoderContext-bound -- plus any future addition);
        // a rename that stops matching would otherwise make this test vacuously
        // pass.
        assert_eq!(
            derived_count, 6,
            "expected exactly six Derived builtin families"
        );
    }

    /// Pins `auto_gpu_policy` per builtin architecture. Most builtins let
    /// Auto pick any GPU-class backend automatically when available
    /// (`AllBackends`), matching how `resolve_runtime_backend` behaves
    /// generically. xasr-zipformer alone is `ExceptMetal` -- Auto still
    /// prefers the generic GPU lane (CUDA/HIP/Vulkan) but falls back to CPU
    /// on Apple Silicon Metal specifically, per a platform-specific
    /// performance audit that found its streaming chunk graph dispatch-bound
    /// and net-slower on Metal (an explicit `--backend metal` request is
    /// unaffected). Two other families measured a similar Metal slowdown but
    /// are deliberately NOT gated this round: qwen (the slowdown looks like a
    /// fixed size x quant platform trade-off, not a qwen-specific bug --
    /// mimo/firered-llm share qwen's exact decode driver and measure faster
    /// on Metal at their 8B @ q4_k config -- but that read awaits a dedicated
    /// follow-up before it's baked into the default; see
    /// `models::qwen::graph_config`) and moonshine (this audit found and
    /// applied an actual architectural fix -- decoder scheduler-off to
    /// activate the reusable incremental decode graph, see
    /// `models::moonshine::graph_config` -- rather than gating around the
    /// problem). See the field doc and each family's own executor/
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
            (MOONSHINE_GGML_ARCHITECTURE_ID, AutoGpuPolicy::AllBackends),
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
                descriptor.auto_gpu_policy, auto_gpu_policy,
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

        let model_architecture = XASR_ZIPFORMER_GGML_ARCHITECTURE_ID;
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
            ResolvedFamilyRuntimeInput::resolve(Some(RequestBackendPreference::CpuOnly), policy)
                .backend(),
            GgmlCpuGraphBackend::Cpu
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
