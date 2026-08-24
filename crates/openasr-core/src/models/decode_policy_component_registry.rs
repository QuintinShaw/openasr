use std::io::Write;

use thiserror::Error;

use crate::PhraseBiasConfig;
use crate::arch::OpenAsrArchitectureRegistry;
use crate::models::ctc_greedy_decode::{
    CtcGreedyDecodeConfig, CtcGreedyDecodeError, CtcGreedyDecodeResult,
    run_ctc_greedy_decode_with_progress,
};
use crate::models::phrase_bias_decode::{
    PhraseBiasBuildError, PhraseBiasTokenEncoder, build_token_phrase_biases,
};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeConfig, Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeResult,
    Seq2SeqGreedyDecodeStepExecutor, run_seq2seq_greedy_decode_loop_with_adapter_v0,
};

const SEQ2SEQ_DEBUG_TRACE_SCHEMA: &str = "openasr.seq2seq-debug-trace.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinDecodePolicyLongformPromptCarryMode {
    Disabled,
    Text,
    TokenHistory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinDecodePolicyLongformProfile {
    Default,
    ConservativeSeq2SeqV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinDecodePolicyExecutionKind {
    Seq2SeqGreedyV0,
    /// Non-autoregressive CTC greedy collapse (the `Ctc` shape). Routed through
    /// `run_builtin_ctc_decode_policy`, NOT the seq2seq loop.
    CtcGreedyV0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinDecodePolicySeq2SeqTextPostprocessKind {
    Identity,
    FunAsrNanoStripControlMarkersV0,
    Qwen3AsrStripControlPrefixV0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinDecodePolicySeq2SeqTraceKind {
    None,
    WhisperEnvV0,
    RuntimeJsonlV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinDecodePolicySeq2SeqStopTokenKind {
    None,
    Qwen3AsrAudioBoundaryV0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinDecodePolicySeq2SeqSuppressionKind {
    None,
    WhisperDefaultV0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BuiltinDecodePolicyComponentDescriptor {
    pub decode_policy_id: &'static str,
    pub execution_kind: BuiltinDecodePolicyExecutionKind,
    pub seq2seq_text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind,
    pub seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind,
    pub seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind,
    pub seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind,
    pub longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode,
    pub longform_profile: BuiltinDecodePolicyLongformProfile,
    /// CTC blank token id, `Some` only for `CtcGreedyV0` policies (read from pack
    /// metadata; `None` for seq2seq policies, which never consult it).
    pub ctc_blank_token_id: Option<u32>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum BuiltinDecodePolicyComponentRegistryError {
    #[error("unknown builtin model architecture '{model_architecture}'")]
    UnknownArchitecture { model_architecture: String },
    #[error("unknown builtin decode policy '{decode_policy_id}'")]
    UnknownDecodePolicy { decode_policy_id: String },
    #[error(
        "builtin decode policy '{decode_policy_id}' requires special token '{token_role}', but it was not available"
    )]
    MissingRequiredSpecialToken {
        decode_policy_id: String,
        token_role: &'static str,
    },
    #[error(
        "builtin decode policy '{decode_policy_id}' is CTC (CtcGreedyV0) and cannot run through the seq2seq decode loop"
    )]
    CtcPolicyRoutedThroughSeq2Seq { decode_policy_id: String },
    #[error(
        "builtin decode policy '{decode_policy_id}' is seq2seq and cannot run through the CTC decode path"
    )]
    Seq2SeqPolicyRoutedThroughCtc { decode_policy_id: String },
    #[error("builtin CTC decode policy '{decode_policy_id}' is missing ctc_blank_token_id")]
    CtcBlankTokenIdMissing { decode_policy_id: String },
    #[error("builtin decode policy '{decode_policy_id}' cannot encode phrase-bias entries")]
    PhraseBiasUnsupported { decode_policy_id: String },
    #[error("builtin decode policy '{decode_policy_id}' phrase-bias tokenization failed: {reason}")]
    PhraseBiasTokenizationFailed {
        decode_policy_id: String,
        reason: String,
    },
}

/// Execution descriptors for every built-in decode policy whose runtime shape is
/// one of the two the shared decode paths can drive: `Seq2SeqGreedyV0` (routed
/// through `run_seq2seq_greedy_decode_loop_v0` via `run_builtin_seq2seq_decode_policy`)
/// and `CtcGreedyV0` (routed through `run_builtin_ctc_decode_policy`).
///
/// Not every `*_DECODE_POLICY_ID` constant belongs here. A family is registered
/// IFF its decode runs on one of those two shared shapes. Families whose decode
/// is a dedicated non-shared loop are intentionally ABSENT, and that is the
/// registration criterion, not an oversight. Both
/// `xasr-zipformer.greedy.transducer.v0` and `parakeet-tdt.greedy.tdt.v0` are
/// RNN-T / TDT transducers (prediction network plus joiner, duration-driven frame
/// skipping): a transducer loop, not greedy seq2seq or CTC collapse. The policy
/// `dolphin.attention-rescoring.v0` is CTC-prefix with attention rescoring, its
/// own joint-decode loop. Those ride dedicated executors and never reach
/// `resolve_builtin_decode_policy`.
/// ASR ownership lives in each architecture inventory row: the row embeds one
/// of these reusable policy components in its typed decode-driver strategy.
/// There is deliberately no second family-to-policy table here.
pub(crate) const COHERE_TRANSCRIBE_DECODE_POLICY_COMPONENT: BuiltinDecodePolicyComponentDescriptor =
    BuiltinDecodePolicyComponentDescriptor {
        decode_policy_id: crate::COHERE_TRANSCRIBE_DECODE_POLICY_ID,
        execution_kind: BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0,
        seq2seq_text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1,
        seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind::None,
        seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind::None,
        longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode::TokenHistory,
        longform_profile: BuiltinDecodePolicyLongformProfile::ConservativeSeq2SeqV1,
        ctc_blank_token_id: None,
    };

pub(crate) const WHISPER_DECODE_POLICY_COMPONENT: BuiltinDecodePolicyComponentDescriptor =
    BuiltinDecodePolicyComponentDescriptor {
        decode_policy_id: crate::WHISPER_DECODE_POLICY_ID,
        execution_kind: BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0,
        seq2seq_text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1,
        seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind::None,
        seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind::WhisperDefaultV0,
        longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode::TokenHistory,
        longform_profile: BuiltinDecodePolicyLongformProfile::Default,
        ctc_blank_token_id: None,
    };

pub(crate) const QWEN3_ASR_DECODE_POLICY_COMPONENT: BuiltinDecodePolicyComponentDescriptor =
    BuiltinDecodePolicyComponentDescriptor {
        decode_policy_id: crate::QWEN3_ASR_DECODE_POLICY_ID,
        execution_kind: BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0,
        seq2seq_text_postprocess_kind:
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::Qwen3AsrStripControlPrefixV0,
        seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1,
        seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind::Qwen3AsrAudioBoundaryV0,
        seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind::None,
        longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode::Text,
        longform_profile: BuiltinDecodePolicyLongformProfile::Default,
        ctc_blank_token_id: None,
    };

pub(crate) const MOONSHINE_DECODE_POLICY_COMPONENT: BuiltinDecodePolicyComponentDescriptor =
    BuiltinDecodePolicyComponentDescriptor {
        decode_policy_id: crate::MOONSHINE_DECODE_POLICY_ID,
        execution_kind: BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0,
        seq2seq_text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1,
        seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind::None,
        seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind::None,
        // The executor has no carry_context producer; the conservative profile
        // also forces carry off. Declare the effective capability truthfully.
        longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode::Disabled,
        longform_profile: BuiltinDecodePolicyLongformProfile::ConservativeSeq2SeqV1,
        ctc_blank_token_id: None,
    };

pub(crate) const FIRERED_AED_DECODE_POLICY_COMPONENT: BuiltinDecodePolicyComponentDescriptor =
    BuiltinDecodePolicyComponentDescriptor {
        // Source of truth is `crate::arch`; firered has no crate-root re-export
        // (it is not selected through the ASR architecture constant surface).
        decode_policy_id: crate::arch::FIRERED_AED_DECODE_POLICY_ID,
        execution_kind: BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0,
        seq2seq_text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1,
        seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind::None,
        seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind::None,
        // Plain `<sos>` AED with no carry_context producer; conservative
        // slicing is the structural long-audio repetition fix (issue #60).
        longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode::Disabled,
        longform_profile: BuiltinDecodePolicyLongformProfile::ConservativeSeq2SeqV1,
        ctc_blank_token_id: None,
    };

pub(crate) const FIRERED_LLM_DECODE_POLICY_COMPONENT: BuiltinDecodePolicyComponentDescriptor =
    BuiltinDecodePolicyComponentDescriptor {
        // Source of truth is `crate::arch` (same staging precedent as firered-aed
        // above): firered-llm has no crate-root re-export.
        decode_policy_id: crate::arch::FIRERED_LLM_DECODE_POLICY_ID,
        execution_kind: BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0,
        seq2seq_text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1,
        // The eot token (ChatML `<|im_end|>`) is supplied per-request via
        // `BuiltinSeq2SeqDecodePolicyConfigInput.eot_token_id`; unlike qwen3-asr's
        // audio-boundary marker, firered-llm's prompt has no extra stop token
        // beyond eot.
        seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind::None,
        seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind::None,
        // Every chunk is a fresh ChatML turn and the executor emits no carry.
        longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode::Disabled,
        longform_profile: BuiltinDecodePolicyLongformProfile::ConservativeSeq2SeqV1,
        ctc_blank_token_id: None,
    };

pub(crate) const FUNASR_NANO_DECODE_POLICY_COMPONENT: BuiltinDecodePolicyComponentDescriptor =
    BuiltinDecodePolicyComponentDescriptor {
        // Source of truth is `crate::arch` (same staging precedent as
        // firered-aed above): funasr-nano has no crate-root re-export. The eot
        // token (ChatML `<|im_end|>`) is supplied per-request via
        // `BuiltinSeq2SeqDecodePolicyConfigInput.eot_token_id`; the audio
        // placeholder tokens only ever appear in the PROMPT (spliced in by the
        // executor). The model can still decode textual control markers used
        // by the upstream runtime for silence and stream boundaries, so strip
        // those in the shared seq2seq output path before any product consumes
        // the transcript.
        decode_policy_id: crate::arch::FUNASR_NANO_DECODE_POLICY_ID,
        execution_kind: BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0,
        seq2seq_text_postprocess_kind:
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::FunAsrNanoStripControlMarkersV0,
        seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1,
        seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind::None,
        seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind::None,
        // Every chunk is a fresh ChatML turn and the executor emits no carry.
        longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode::Disabled,
        longform_profile: BuiltinDecodePolicyLongformProfile::ConservativeSeq2SeqV1,
        ctc_blank_token_id: None,
    };

pub(crate) const MIMO_ASR_DECODE_POLICY_COMPONENT: BuiltinDecodePolicyComponentDescriptor =
    BuiltinDecodePolicyComponentDescriptor {
        // Source of truth is `crate::arch` (same staging precedent as
        // firered-aed above): mimo-asr has no crate-root re-export.
        decode_policy_id: crate::arch::MIMO_ASR_DECODE_POLICY_ID,
        execution_kind: BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0,
        // Strips the audio-boundary/speech-slot placeholder tokens
        // (`<|sosp|>`/`<|eosp|>`/`<|empty|>`/`<|eot|>`/`<|eostm|>`) inside
        // `MimoAsrTokenizer::decode_text_token_ids` itself, so the driver's
        // own postprocess stays Identity (mirrors firered-llm's shape: the
        // eot token is supplied per-request via `eot_token_id`, not a
        // registry-level stop-token-kind).
        seq2seq_text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1,
        seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind::None,
        seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind::None,
        // Every chunk is a fresh ChatML turn and the executor emits no carry.
        longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode::Disabled,
        longform_profile: BuiltinDecodePolicyLongformProfile::ConservativeSeq2SeqV1,
        ctc_blank_token_id: None,
    };

pub(crate) const MOSS_TD_DECODE_POLICY_COMPONENT: BuiltinDecodePolicyComponentDescriptor =
    BuiltinDecodePolicyComponentDescriptor {
        // Source of truth is `crate::arch` (same staging precedent as
        // firered-aed above): moss-transcribe-diarize has no crate-root
        // re-export.
        decode_policy_id: crate::arch::MOSS_TD_DECODE_POLICY_ID,
        execution_kind: BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0,
        // eot (ChatML `<|im_end|>`) is supplied per-request via
        // `BuiltinSeq2SeqDecodePolicyConfigInput.eot_token_id`; the audio-span
        // placeholder/marker tokens are never emitted by the LLM itself (they
        // only ever appear in the PROMPT, spliced in by the executor), so
        // there is nothing else to strip -- Identity postprocess, same shape
        // as firered-llm/mimo-asr.
        seq2seq_text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1,
        seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind::None,
        seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind::None,
        // This family's executor concatenates its own 30s encoder chunks into
        // ONE prompt/decode, but that prompt has to fit
        // `MOSS_TD_MAX_KV_CACHE_POSITIONS`, so it can only ever absorb a few
        // minutes -- the shared native slicer supplies the rest. How the
        // recording is cut for it is declared once on the architecture
        // descriptor (`OpenAsrLongformSliceShape::ScopedSlices`), which also
        // carries the per-slice speaker-scope consequence; the decode policy
        // holds no separate opinion. Scoped slicing disables carry and the
        // executor has no carry_context producer.
        longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode::Disabled,
        longform_profile: BuiltinDecodePolicyLongformProfile::Default,
        ctc_blank_token_id: None,
    };

pub(crate) const GRANITE_SPEECH_DECODE_POLICY_COMPONENT: BuiltinDecodePolicyComponentDescriptor =
    BuiltinDecodePolicyComponentDescriptor {
        // Source of truth is `crate::arch` (same staging precedent as
        // mimo-asr above): granite-speech has no crate-root re-export.
        decode_policy_id: crate::arch::GRANITE_SPEECH_DECODE_POLICY_ID,
        execution_kind: BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0,
        // eot (`<|end_of_text|>`) is supplied per-request via
        // `BuiltinSeq2SeqDecodePolicyConfigInput.eot_token_id`; the audio
        // placeholder token is never emitted by the LLM itself (it only ever
        // appears in the prompt, spliced in by the executor before decode
        // starts), so there is nothing else to strip -- Identity postprocess,
        // same shape as firered-llm/mimo-asr/moss-transcribe-diarize.
        seq2seq_text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1,
        seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind::None,
        seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind::None,
        // This family's executor assembles the audio splice ONCE (the whole
        // Conformer-encoded + Q-Former-projected utterance goes into a
        // single prompt, see `prompt.rs`) and decodes it in one pass -- it
        // never re-embeds audio across multiple fresh ChatML-style turns the
        // way firered-llm/mimo-asr's per-chunk hard caps do, and its encoder
        // is `LocalChunked` (Conformer block-attention), not one of the
        // small-context plain-prompted AED decoders `ConservativeSeq2SeqV1`
        // guards against repeating on long pause-free audio -- same
        // reasoning as `whisper`/`moss-transcribe-diarize` pairing
        // `FixedWindow`/`Default` above. The current executor emits no carry,
        // so the capability remains explicitly disabled.
        longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode::Disabled,
        longform_profile: BuiltinDecodePolicyLongformProfile::Default,
        ctc_blank_token_id: None,
    };

pub(crate) const PARAKEET_CTC_DECODE_POLICY_COMPONENT: BuiltinDecodePolicyComponentDescriptor =
    BuiltinDecodePolicyComponentDescriptor {
        decode_policy_id: crate::PARAKEET_CTC_DECODE_POLICY_ID,
        execution_kind: BuiltinDecodePolicyExecutionKind::CtcGreedyV0,
        // seq2seq fields are unused for CtcGreedyV0; set to no-op values.
        seq2seq_text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind::None,
        seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind::None,
        seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind::None,
        longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode::Text,
        longform_profile: BuiltinDecodePolicyLongformProfile::Default,
        // parakeet-ctc-0.6b: vocab_size 1025, pad_token_id 1024 = the CTC blank
        // (cross-checked against the pack metadata at decode time).
        ctc_blank_token_id: Some(1024),
    };

pub(crate) const WAV2VEC2_CTC_DECODE_POLICY_COMPONENT: BuiltinDecodePolicyComponentDescriptor =
    BuiltinDecodePolicyComponentDescriptor {
        decode_policy_id: crate::WAV2VEC2_CTC_DECODE_POLICY_ID,
        execution_kind: BuiltinDecodePolicyExecutionKind::CtcGreedyV0,
        seq2seq_text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind::None,
        seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind::None,
        seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind::None,
        longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode::Text,
        longform_profile: BuiltinDecodePolicyLongformProfile::Default,
        // wav2vec2-base-960h: vocab_size 32, pad_token_id 0 = the CTC blank.
        ctc_blank_token_id: Some(0),
    };

pub(crate) const SENSEVOICE_DECODE_POLICY_COMPONENT: BuiltinDecodePolicyComponentDescriptor =
    BuiltinDecodePolicyComponentDescriptor {
        decode_policy_id: crate::SENSEVOICE_DECODE_POLICY_ID,
        execution_kind: BuiltinDecodePolicyExecutionKind::CtcGreedyV0,
        seq2seq_text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind::None,
        seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind::None,
        seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind::None,
        longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode::Text,
        longform_profile: BuiltinDecodePolicyLongformProfile::Default,
        // SenseVoiceSmall: vocab_size 25055, piece 0 (`<unk>`) = the CTC blank
        // (FunASR default blank_id 0).
        ctc_blank_token_id: Some(0),
    };

pub(crate) trait BuiltinSeq2SeqDecodePolicyTokenSource: PhraseBiasTokenEncoder {
    fn audio_end_token_id(&self) -> Option<u32> {
        None
    }

    fn audio_pad_token_id(&self) -> Option<u32> {
        None
    }

    fn start_of_transcript_token_id(&self) -> Option<u32> {
        None
    }

    fn transcribe_token_id(&self) -> Option<u32> {
        None
    }

    fn no_timestamps_token_id(&self) -> Option<u32> {
        None
    }

    fn token_id_by_content(&self, _content: &str) -> Option<u32> {
        None
    }
}

impl BuiltinSeq2SeqDecodePolicyTokenSource for () {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BuiltinSeq2SeqDecodePolicyConfigInput {
    pub initial_prompt_tokens: Vec<u32>,
    pub eot_token_id: u32,
    pub vocab_size: usize,
    pub max_generated_tokens: usize,
}

pub(crate) fn resolve_builtin_decode_policy_for_architecture(
    model_architecture: &str,
) -> Result<BuiltinDecodePolicyComponentDescriptor, BuiltinDecodePolicyComponentRegistryError> {
    let descriptor = OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(model_architecture)
        .ok_or_else(
            || BuiltinDecodePolicyComponentRegistryError::UnknownArchitecture {
                model_architecture: model_architecture.to_string(),
            },
        )?;
    descriptor
        .topology_contract
        .decode_driver
        .shared_policy()
        .ok_or_else(
            || BuiltinDecodePolicyComponentRegistryError::UnknownDecodePolicy {
                decode_policy_id: descriptor
                    .topology_contract
                    .decode_driver
                    .decode_policy_id()
                    .to_string(),
            },
        )
}

pub(crate) fn resolve_builtin_decode_policy(
    decode_policy_id: &str,
) -> Result<BuiltinDecodePolicyComponentDescriptor, BuiltinDecodePolicyComponentRegistryError> {
    OpenAsrArchitectureRegistry::with_builtins()
        .descriptors()
        .iter()
        .filter_map(|architecture| architecture.topology_contract.decode_driver.shared_policy())
        .find(|descriptor| descriptor.decode_policy_id == decode_policy_id)
        .ok_or_else(
            || BuiltinDecodePolicyComponentRegistryError::UnknownDecodePolicy {
                decode_policy_id: decode_policy_id.to_string(),
            },
        )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_builtin_seq2seq_decode_policy<E>(
    decode_policy_id: &str,
    config_input: &BuiltinSeq2SeqDecodePolicyConfigInput,
    token_source: &dyn BuiltinSeq2SeqDecodePolicyTokenSource,
    phrase_bias: Option<&PhraseBiasConfig>,
    step_executor: &mut dyn Seq2SeqGreedyDecodeStepExecutor,
    decode_text_token_ids: &dyn Fn(&[u32]) -> Result<String, E>,
    map_token_decoder_error_to_shared: fn(E) -> Seq2SeqGreedyDecodeError,
    map_shared_error_to_family: fn(Seq2SeqGreedyDecodeError) -> E,
    map_registry_error: fn(BuiltinDecodePolicyComponentRegistryError) -> E,
    control: &std::sync::Arc<crate::api::backend::TranscriptionControl>,
    decode_work_progress: Option<&crate::api::backend::WorkProgressObserver>,
    unstable_decode_text: Option<&crate::api::backend::UnstableDecodeTextObserver>,
) -> Result<Seq2SeqGreedyDecodeResult, E> {
    let descriptor = resolve_builtin_decode_policy(decode_policy_id).map_err(map_registry_error)?;
    let config = build_builtin_seq2seq_decode_policy_config(
        descriptor,
        config_input,
        token_source,
        phrase_bias,
    )
    .map_err(map_registry_error)?;
    match descriptor.execution_kind {
        BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0 => {
            let normalize_text = |text: String| {
                apply_seq2seq_text_postprocess(descriptor.seq2seq_text_postprocess_kind, &text)
            };
            let mut trace_token = |step_index: usize, token_id: u32, is_eot: bool| {
                emit_seq2seq_token_trace(
                    descriptor.seq2seq_trace_kind,
                    step_index,
                    token_id,
                    is_eot,
                );
            };
            let mut on_topk = |step_index: usize, logits: &[f32]| {
                emit_seq2seq_topk_trace(descriptor.seq2seq_trace_kind, step_index, logits);
            };
            run_seq2seq_greedy_decode_loop_with_adapter_v0(
                &config,
                step_executor,
                decode_text_token_ids,
                map_token_decoder_error_to_shared,
                map_shared_error_to_family,
                &normalize_text,
                &mut trace_token,
                &mut on_topk,
                control,
                decode_work_progress,
                unstable_decode_text,
            )
        }
        // Fail closed: a CTC policy must never route through the seq2seq loop.
        BuiltinDecodePolicyExecutionKind::CtcGreedyV0 => Err(map_registry_error(
            BuiltinDecodePolicyComponentRegistryError::CtcPolicyRoutedThroughSeq2Seq {
                decode_policy_id: decode_policy_id.to_string(),
            },
        )),
    }
}

/// Non-autoregressive CTC decode entry point (the `Ctc` shape's sibling of
/// `run_builtin_seq2seq_decode_policy`). Resolves the policy descriptor, reads
/// its `ctc_blank_token_id`, and runs the frame-argmax + collapse + detokenize.
/// `frame_logits[t]` is the length-`vocab_size` logit row for frame `t`;
/// `decode_text_token_ids` maps the collapsed ids to text (its own error
/// stringified by the family). Fails closed if the policy is not `CtcGreedyV0`.
pub(crate) fn run_builtin_ctc_decode_policy<E>(
    decode_policy_id: &str,
    frame_logits: &[&[f32]],
    vocab_size: usize,
    phrase_bias: Option<&PhraseBiasConfig>,
    phrase_bias_encoder: &dyn PhraseBiasTokenEncoder,
    decode_text_token_ids: &dyn Fn(&[u32]) -> Result<String, String>,
    map_ctc_error_to_family: fn(CtcGreedyDecodeError) -> E,
    map_registry_error: fn(BuiltinDecodePolicyComponentRegistryError) -> E,
    decode_work_progress: Option<&crate::api::backend::WorkProgressObserver>,
) -> Result<CtcGreedyDecodeResult, E> {
    let descriptor = resolve_builtin_decode_policy(decode_policy_id).map_err(map_registry_error)?;
    match descriptor.execution_kind {
        BuiltinDecodePolicyExecutionKind::CtcGreedyV0 => {
            let blank_token_id = descriptor.ctc_blank_token_id.ok_or_else(|| {
                map_registry_error(
                    BuiltinDecodePolicyComponentRegistryError::CtcBlankTokenIdMissing {
                        decode_policy_id: decode_policy_id.to_string(),
                    },
                )
            })?;
            run_ctc_greedy_decode_with_progress(
                CtcGreedyDecodeConfig {
                    blank_token_id,
                    vocab_size,
                    phrase_biases: registry_phrase_biases(
                        descriptor,
                        phrase_bias,
                        phrase_bias_encoder,
                    )
                    .map_err(map_registry_error)?,
                },
                frame_logits,
                decode_text_token_ids,
                |reason| CtcGreedyDecodeError::DetokenizeFailed { reason },
                decode_work_progress,
            )
            .map_err(map_ctc_error_to_family)
        }
        // Fail closed: a seq2seq policy must never route through the CTC path.
        BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0 => Err(map_registry_error(
            BuiltinDecodePolicyComponentRegistryError::Seq2SeqPolicyRoutedThroughCtc {
                decode_policy_id: decode_policy_id.to_string(),
            },
        )),
    }
}

pub(crate) fn build_builtin_seq2seq_decode_policy_config(
    descriptor: BuiltinDecodePolicyComponentDescriptor,
    input: &BuiltinSeq2SeqDecodePolicyConfigInput,
    token_source: &dyn BuiltinSeq2SeqDecodePolicyTokenSource,
    phrase_bias: Option<&PhraseBiasConfig>,
) -> Result<Seq2SeqGreedyDecodeConfig, BuiltinDecodePolicyComponentRegistryError> {
    let stop_token_ids = match descriptor.seq2seq_stop_token_kind {
        BuiltinDecodePolicySeq2SeqStopTokenKind::None => Vec::new(),
        BuiltinDecodePolicySeq2SeqStopTokenKind::Qwen3AsrAudioBoundaryV0 => vec![
            require_special_token(
                descriptor,
                "audio_pad_token_id",
                token_source.audio_pad_token_id(),
            )?,
            require_special_token(
                descriptor,
                "audio_end_token_id",
                token_source.audio_end_token_id(),
            )?,
        ],
    };

    let (suppress_first_step_token_ids, suppress_token_ids) =
        match descriptor.seq2seq_suppression_kind {
            BuiltinDecodePolicySeq2SeqSuppressionKind::None => (Vec::new(), Vec::new()),
            BuiltinDecodePolicySeq2SeqSuppressionKind::WhisperDefaultV0 => {
                let mut suppress_token_ids = Vec::new();
                for token_id in [
                    token_source.start_of_transcript_token_id(),
                    token_source.transcribe_token_id(),
                    token_source.no_timestamps_token_id(),
                    token_source.token_id_by_content("<|startofprev|>"),
                    token_source.token_id_by_content("<|en|>"),
                ] {
                    push_unique_token_id(&mut suppress_token_ids, token_id);
                }
                // Also suppress the language/task control tokens ACTUALLY selected
                // for this request: a translate / non-English decode prompts with
                // <|xx|>/<|translate|> rather than the <|en|>/<|transcribe|>
                // defaults above, and those should not be re-emittable mid-stream.
                // Resolve them positionally relative to <|startoftranscript|> in the
                // prompt, which is robust to the longform layout where the control
                // block is preceded by `<|startofprev|> ...carry` and so does not
                // begin at index 0. The default (en+transcribe) and `.en` prefixes
                // resolve to tokens already in the set (or None), so the suppressed
                // set stays byte-identical on the WER-0-gated path.
                if let Some(sot_token_id) = token_source.start_of_transcript_token_id()
                    && let Some(sot_index) = input
                        .initial_prompt_tokens
                        .iter()
                        .position(|&token| token == sot_token_id)
                {
                    push_unique_token_id(
                        &mut suppress_token_ids,
                        input.initial_prompt_tokens.get(sot_index + 1).copied(),
                    );
                    push_unique_token_id(
                        &mut suppress_token_ids,
                        input.initial_prompt_tokens.get(sot_index + 2).copied(),
                    );
                }
                let mut suppress_first_step_token_ids = vec![input.eot_token_id];
                push_unique_token_id(
                    &mut suppress_first_step_token_ids,
                    token_source.token_id_by_content(" "),
                );
                (suppress_first_step_token_ids, suppress_token_ids)
            }
        };

    Ok(Seq2SeqGreedyDecodeConfig {
        initial_prompt_tokens: input.initial_prompt_tokens.clone(),
        eot_token_id: input.eot_token_id,
        stop_token_ids,
        vocab_size: input.vocab_size,
        max_generated_tokens: input.max_generated_tokens,
        suppress_first_step_token_ids,
        suppress_token_ids,
        phrase_biases: registry_phrase_biases(descriptor, phrase_bias, token_source)?,
    })
}

/// Build phrase-bias token sequences for a decode policy, mapping the typed
/// [`PhraseBiasBuildError`] onto the registry's fail-closed error variants. A
/// single helper for both the seq2seq and CTC paths: any `PhraseBiasTokenEncoder`
/// works (a seq2seq token source satisfies it via the supertrait bound), so the
/// encode+classify logic lives in one place instead of one copy per decode shape.
fn registry_phrase_biases<E: PhraseBiasTokenEncoder + ?Sized>(
    descriptor: BuiltinDecodePolicyComponentDescriptor,
    phrase_bias: Option<&PhraseBiasConfig>,
    encoder: &E,
) -> Result<
    Vec<crate::models::phrase_bias_decode::TokenPhraseBias>,
    BuiltinDecodePolicyComponentRegistryError,
> {
    build_token_phrase_biases(phrase_bias, encoder).map_err(|error| {
        let decode_policy_id = descriptor.decode_policy_id.to_string();
        match error {
            PhraseBiasBuildError::Unsupported => {
                BuiltinDecodePolicyComponentRegistryError::PhraseBiasUnsupported {
                    decode_policy_id,
                }
            }
            PhraseBiasBuildError::TokenizationFailed { reason } => {
                BuiltinDecodePolicyComponentRegistryError::PhraseBiasTokenizationFailed {
                    decode_policy_id,
                    reason,
                }
            }
        }
    })
}

fn require_special_token(
    descriptor: BuiltinDecodePolicyComponentDescriptor,
    token_role: &'static str,
    token_id: Option<u32>,
) -> Result<u32, BuiltinDecodePolicyComponentRegistryError> {
    token_id.ok_or_else(
        || BuiltinDecodePolicyComponentRegistryError::MissingRequiredSpecialToken {
            decode_policy_id: descriptor.decode_policy_id.to_string(),
            token_role,
        },
    )
}

fn push_unique_token_id(target: &mut Vec<u32>, token_id: Option<u32>) {
    let Some(token_id) = token_id else {
        return;
    };
    if !target.contains(&token_id) {
        target.push(token_id);
    }
}

const QWEN3_ASR_TEXT_MARKER: &str = "<asr_text>";
const FUNASR_NANO_CONTROL_MARKERS: [&str; 3] = ["/sil", "endofbreak", "FFFF"];
const FUNASR_NANO_ARTIFACT_CHARS: [char; 7] = ['Ｏ', '[', ']', '&', '＆', '|', '｜'];

pub(crate) fn apply_seq2seq_text_postprocess(
    kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind,
    decoded: &str,
) -> String {
    match kind {
        BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity => decoded.to_string(),
        BuiltinDecodePolicySeq2SeqTextPostprocessKind::FunAsrNanoStripControlMarkersV0 => {
            // Mirror the model runtime's output contract at the model boundary:
            // timestamp/control spans and vLLM artifacts are not speech. Keep
            // this family-scoped so ordinary brackets and slash text from
            // other architectures are never removed globally.
            let mut text = strip_closed_spans(decoded, '<', '>');
            text = strip_closed_spans(&text, '[', ']');
            text.retain(|character| !FUNASR_NANO_ARTIFACT_CHARS.contains(&character));
            for marker in FUNASR_NANO_CONTROL_MARKERS {
                text = text.replace(marker, " ");
            }
            text.split_whitespace().collect::<Vec<_>>().join(" ")
        }
        BuiltinDecodePolicySeq2SeqTextPostprocessKind::Qwen3AsrStripControlPrefixV0 => decoded
            [seq2seq_transcript_byte_start(kind, decoded)..]
            .trim()
            .to_string(),
    }
}

fn strip_closed_spans(input: &str, open: char, close: char) -> String {
    let mut output = String::with_capacity(input.len());
    let mut remaining = input;
    while let Some(start) = remaining.find(open) {
        output.push_str(&remaining[..start]);
        let after_open = &remaining[start + open.len_utf8()..];
        let Some(end) = after_open.find(close) else {
            output.push_str(&remaining[start..]);
            return output;
        };
        remaining = &after_open[end + close.len_utf8()..];
    }
    output.push_str(remaining);
    output
}

/// Byte offset where the spoken transcript starts inside the raw decoded
/// string for this postprocess kind. The word-timestamp path uses this to skip
/// control-prefix characters (e.g. qwen's "language English<asr_text>") so
/// words match the postprocessed transcript text.
pub(crate) fn seq2seq_transcript_byte_start(
    kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind,
    decoded: &str,
) -> usize {
    match kind {
        BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity
        | BuiltinDecodePolicySeq2SeqTextPostprocessKind::FunAsrNanoStripControlMarkersV0 => 0,
        BuiltinDecodePolicySeq2SeqTextPostprocessKind::Qwen3AsrStripControlPrefixV0 => decoded
            .find(QWEN3_ASR_TEXT_MARKER)
            .map(|index| index + QWEN3_ASR_TEXT_MARKER.len())
            .unwrap_or(0),
    }
}

fn emit_seq2seq_token_trace(
    kind: BuiltinDecodePolicySeq2SeqTraceKind,
    step_index: usize,
    token_id: u32,
    is_eot: bool,
) {
    if kind == BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1 {
        if let Some(receipt) =
            crate::models::native_execution_services::current_execution_receipt_collector()
        {
            receipt.record_token(step_index, token_id, is_eot);
        }
        append_seq2seq_debug_jsonl_trace(&format!(
            "{{\"schema\":\"{SEQ2SEQ_DEBUG_TRACE_SCHEMA}\",\"event\":\"token\",\"step_index\":{step_index},\"token_id\":{token_id},\"is_eot\":{}}}",
            usize::from(is_eot)
        ));
        return;
    }
    if kind != BuiltinDecodePolicySeq2SeqTraceKind::WhisperEnvV0
        || std::env::var_os("OPENASR_WHISPER_GGML_TRACE").is_none()
    {
        return;
    }
    eprintln!(
        "openasr_whisper_ggml_trace stage=greedy_decode event=token status=ok step_index={step_index} token_id={token_id} is_eot={}",
        usize::from(is_eot)
    );
}

fn emit_seq2seq_topk_trace(
    kind: BuiltinDecodePolicySeq2SeqTraceKind,
    step_index: usize,
    logits: &[f32],
) {
    if kind == BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1 {
        if let Some(receipt) =
            crate::models::native_execution_services::current_execution_receipt_collector()
        {
            receipt.record_top_k(step_index, logits);
        }
        let mut top = Vec::<(usize, f32)>::new();
        for (token_id, logit) in logits.iter().copied().enumerate() {
            if !logit.is_finite() {
                continue;
            }
            let insert_at = top
                .iter()
                .position(|(_, existing)| logit.total_cmp(existing).is_gt());
            if let Some(insert_at) = insert_at {
                top.insert(insert_at, (token_id, logit));
            } else if top.len() < 8 {
                top.push((token_id, logit));
            }
            if top.len() > 8 {
                top.truncate(8);
            }
        }
        let items = top
            .iter()
            .map(|(token_id, logit)| format!("{{\"token_id\":{token_id},\"value\":{logit:.6}}}"))
            .collect::<Vec<_>>()
            .join(",");
        append_seq2seq_debug_jsonl_trace(&format!(
            "{{\"schema\":\"{SEQ2SEQ_DEBUG_TRACE_SCHEMA}\",\"event\":\"top_k\",\"step_index\":{step_index},\"items\":[{items}]}}"
        ));
        return;
    }
    if kind != BuiltinDecodePolicySeq2SeqTraceKind::WhisperEnvV0
        || std::env::var_os("OPENASR_WHISPER_GGML_TRACE_TOPK").is_none()
    {
        return;
    }
    let mut top = Vec::<(usize, f32)>::new();
    for (token_id, logit) in logits.iter().copied().enumerate() {
        if !logit.is_finite() {
            continue;
        }
        let insert_at = top
            .iter()
            .position(|(_, existing)| logit.total_cmp(existing).is_gt());
        if let Some(insert_at) = insert_at {
            top.insert(insert_at, (token_id, logit));
        } else if top.len() < 8 {
            top.push((token_id, logit));
        }
        if top.len() > 8 {
            top.truncate(8);
        }
    }
    let items = top
        .iter()
        .map(|(token_id, logit)| format!("{token_id}:{logit:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "openasr_whisper_ggml_trace stage=greedy_decode event=topk status=ok step_index={step_index} topk={items}"
    );
}

fn append_seq2seq_debug_jsonl_trace(line: &str) {
    let Some(path) = std::env::var_os("OPENASR_SEQ2SEQ_TRACE_FILE") else {
        return;
    };
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    else {
        return;
    };
    if file
        .metadata()
        .map(|metadata| metadata.len() == 0)
        .unwrap_or(false)
    {
        let (Some(mode), Some(provider), Some(device)) = (
            std::env::var_os("OPENASR_SEQ2SEQ_TRACE_MODE"),
            std::env::var_os("OPENASR_SEQ2SEQ_TRACE_PROVIDER"),
            std::env::var_os("OPENASR_SEQ2SEQ_TRACE_DEVICE"),
        ) else {
            return;
        };
        let header = format!(
            "{{\"schema\":\"{SEQ2SEQ_DEBUG_TRACE_SCHEMA}\",\"event\":\"header\",\"mode\":\"{}\",\"provider\":\"{}\",\"device\":\"{}\"}}",
            mode.to_string_lossy(),
            provider.to_string_lossy(),
            device.to_string_lossy()
        );
        let _ = writeln!(file, "{header}");
    }
    let _ = writeln!(file, "{line}");
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::models::seq2seq_greedy_decode::{
        Seq2SeqGreedyDecodeStepInput, Seq2SeqGreedyDecodeStepLogitsOutput,
    };

    #[test]
    fn resolves_builtin_decode_policy_for_architecture() {
        let whisper =
            resolve_builtin_decode_policy_for_architecture(crate::WHISPER_GGML_ARCHITECTURE_ID)
                .expect("whisper decode policy");
        let cohere = resolve_builtin_decode_policy_for_architecture(
            crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
        )
        .expect("cohere decode policy");
        let qwen =
            resolve_builtin_decode_policy_for_architecture(crate::QWEN3_ASR_GGML_ARCHITECTURE_ID)
                .expect("qwen decode policy");
        let funasr = resolve_builtin_decode_policy_for_architecture(
            crate::arch::FUNASR_NANO_GGML_ARCHITECTURE_ID,
        )
        .expect("funasr-nano decode policy");

        assert_eq!(
            whisper.longform_prompt_carry_mode,
            BuiltinDecodePolicyLongformPromptCarryMode::TokenHistory
        );
        assert_eq!(
            cohere.longform_profile,
            BuiltinDecodePolicyLongformProfile::ConservativeSeq2SeqV1
        );
        assert_eq!(
            qwen.longform_prompt_carry_mode,
            BuiltinDecodePolicyLongformPromptCarryMode::Text
        );
        assert_eq!(
            whisper.seq2seq_trace_kind,
            BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1
        );
        assert_eq!(
            qwen.seq2seq_text_postprocess_kind,
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::Qwen3AsrStripControlPrefixV0
        );
        assert_eq!(
            funasr.seq2seq_text_postprocess_kind,
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::FunAsrNanoStripControlMarkersV0
        );
    }

    #[test]
    fn families_without_a_carry_producer_declare_carry_disabled() {
        for decode_policy_id in [
            crate::MOONSHINE_DECODE_POLICY_ID,
            crate::arch::FIRERED_AED_DECODE_POLICY_ID,
            crate::arch::FIRERED_LLM_DECODE_POLICY_ID,
            crate::arch::FUNASR_NANO_DECODE_POLICY_ID,
            crate::arch::MIMO_ASR_DECODE_POLICY_ID,
            crate::arch::MOSS_TD_DECODE_POLICY_ID,
            crate::arch::GRANITE_SPEECH_DECODE_POLICY_ID,
        ] {
            let policy = resolve_builtin_decode_policy(decode_policy_id)
                .unwrap_or_else(|error| panic!("{decode_policy_id}: {error}"));
            assert_eq!(
                policy.longform_prompt_carry_mode,
                BuiltinDecodePolicyLongformPromptCarryMode::Disabled,
                "{decode_policy_id} has no carry_context producer"
            );
        }
    }

    #[test]
    fn rejects_unknown_builtin_decode_policy() {
        let error = resolve_builtin_decode_policy("unknown.decode.policy.v0")
            .expect_err("unknown decode policy must fail closed");

        assert!(matches!(
            error,
            BuiltinDecodePolicyComponentRegistryError::UnknownDecodePolicy { .. }
        ));
    }

    #[test]
    fn all_decode_driver_strategy_families_resolve_from_topology_descriptor() {
        // Half-connect guard driven by the architecture registry descriptor
        // (not by decode-policy id substring matching): AED/autoregressive
        // families declare SharedSeq2SeqGreedy, CTC collapse families declare
        // SharedCtcGreedy, and dedicated loops must stay off this registry.
        use crate::arch::OpenAsrDecodeDriverStrategy;
        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            match descriptor.topology_contract.decode_driver {
                OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy { policy } => {
                    let resolved = resolve_builtin_decode_policy(policy.decode_policy_id)
                        .unwrap_or_else(|error| {
                            panic!(
                                "shared seq2seq family '{}' policy '{}' missing: {error}",
                                descriptor.identity.model_family, policy.decode_policy_id
                            )
                        });
                    assert_eq!(resolved, policy);
                    assert_eq!(
                        policy.execution_kind,
                        BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0,
                        "family '{}'",
                        descriptor.identity.model_family
                    );
                    assert_eq!(
                        policy.seq2seq_trace_kind,
                        BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1,
                        "family '{}' must emit runtime token traces into the native receipt collector",
                        descriptor.identity.model_family
                    );
                }
                OpenAsrDecodeDriverStrategy::SharedCtcGreedy { policy } => {
                    let resolved = resolve_builtin_decode_policy(policy.decode_policy_id)
                        .unwrap_or_else(|error| {
                            panic!(
                                "shared CTC family '{}' policy '{}' missing: {error}",
                                descriptor.identity.model_family, policy.decode_policy_id
                            )
                        });
                    assert_eq!(resolved, policy);
                    assert_eq!(
                        policy.execution_kind,
                        BuiltinDecodePolicyExecutionKind::CtcGreedyV0,
                        "family '{}'",
                        descriptor.identity.model_family
                    );
                    assert!(
                        policy.ctc_blank_token_id.is_some(),
                        "family '{}' CTC policy must declare blank token id",
                        descriptor.identity.model_family
                    );
                }
                OpenAsrDecodeDriverStrategy::Dedicated {
                    decode_policy_id, ..
                } => {
                    assert!(
                        resolve_builtin_decode_policy(decode_policy_id).is_err(),
                        "dedicated family '{}' must not register on the shared decode driver",
                        descriptor.identity.model_family
                    );
                }
            }
        }
    }

    struct SyntheticStepExecutor {
        vocab_size: usize,
        sequence: Vec<u32>,
    }

    impl Seq2SeqGreedyDecodeStepExecutor for SyntheticStepExecutor {
        fn decode_step_logits(
            &mut self,
            input: Seq2SeqGreedyDecodeStepInput<'_>,
        ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
            let token_id = self.sequence.get(input.step_index).copied().ok_or(
                Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: "missing token".to_string(),
                },
            )?;
            let token_idx = usize::try_from(token_id).map_err(|_| {
                Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: "token id overflow".to_string(),
                }
            })?;
            let mut logits = vec![-1000.0_f32; self.vocab_size];
            logits[token_idx] = 1000.0;
            Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits,
                greedy_token_hint: None,
            })
        }
    }

    struct SyntheticTokenSource {
        audio_end_token_id: Option<u32>,
        audio_pad_token_id: Option<u32>,
        start_of_transcript_token_id: Option<u32>,
        transcribe_token_id: Option<u32>,
        no_timestamps_token_id: Option<u32>,
        token_ids_by_content: BTreeMap<&'static str, u32>,
    }

    impl BuiltinSeq2SeqDecodePolicyTokenSource for SyntheticTokenSource {
        fn audio_end_token_id(&self) -> Option<u32> {
            self.audio_end_token_id
        }

        fn audio_pad_token_id(&self) -> Option<u32> {
            self.audio_pad_token_id
        }

        fn start_of_transcript_token_id(&self) -> Option<u32> {
            self.start_of_transcript_token_id
        }

        fn transcribe_token_id(&self) -> Option<u32> {
            self.transcribe_token_id
        }

        fn no_timestamps_token_id(&self) -> Option<u32> {
            self.no_timestamps_token_id
        }

        fn token_id_by_content(&self, content: &str) -> Option<u32> {
            self.token_ids_by_content.get(content).copied()
        }
    }

    impl PhraseBiasTokenEncoder for SyntheticTokenSource {
        fn encode_phrase_bias_tokens(&self, _phrase: &str) -> Result<Option<Vec<u32>>, String> {
            Ok(None)
        }
    }

    struct OkPhraseBiasEncoder;
    impl PhraseBiasTokenEncoder for OkPhraseBiasEncoder {
        fn encode_phrase_bias_tokens(&self, _phrase: &str) -> Result<Option<Vec<u32>>, String> {
            Ok(Some(vec![1, 2]))
        }
    }

    struct UnsupportedPhraseBiasEncoder;
    impl PhraseBiasTokenEncoder for UnsupportedPhraseBiasEncoder {
        fn encode_phrase_bias_tokens(&self, _phrase: &str) -> Result<Option<Vec<u32>>, String> {
            Ok(None)
        }
    }

    struct FailingPhraseBiasEncoder;
    impl PhraseBiasTokenEncoder for FailingPhraseBiasEncoder {
        fn encode_phrase_bias_tokens(&self, _phrase: &str) -> Result<Option<Vec<u32>>, String> {
            Err("boom: cannot encode".to_string())
        }
    }

    #[test]
    fn registry_phrase_biases_classifies_unsupported_vs_tokenization_failure() {
        let descriptor = resolve_builtin_decode_policy(crate::WHISPER_DECODE_POLICY_ID)
            .expect("whisper decode policy descriptor");
        let config = PhraseBiasConfig::from_phrases([("openasr", 5.0)]).unwrap();

        let ok = registry_phrase_biases(descriptor, Some(&config), &OkPhraseBiasEncoder)
            .expect("phrase bias builds");
        assert_eq!(ok.len(), 1);

        let unsupported =
            registry_phrase_biases(descriptor, Some(&config), &UnsupportedPhraseBiasEncoder)
                .unwrap_err();
        assert!(matches!(
            unsupported,
            BuiltinDecodePolicyComponentRegistryError::PhraseBiasUnsupported { .. }
        ));

        let failed = registry_phrase_biases(descriptor, Some(&config), &FailingPhraseBiasEncoder)
            .unwrap_err();
        assert!(matches!(
            failed,
            BuiltinDecodePolicyComponentRegistryError::PhraseBiasTokenizationFailed { reason, .. }
                if reason.contains("boom")
        ));

        // Empty/None config short-circuits to an empty bias set on the unified path.
        let empty = registry_phrase_biases(descriptor, None, &FailingPhraseBiasEncoder)
            .expect("none phrase bias is ok");
        assert!(empty.is_empty());
    }

    #[test]
    fn builtin_decode_policy_dispatch_runs_seq2seq_greedy_loop() {
        let mut step_executor = SyntheticStepExecutor {
            vocab_size: 16,
            sequence: vec![1, 2, 7],
        };
        let token_table = BTreeMap::from([(1_u32, "he"), (2_u32, "llo")]);
        let decode_text_token_ids = |token_ids: &[u32]| {
            let mut out = String::new();
            for token_id in token_ids {
                out.push_str(token_table.get(token_id).copied().unwrap_or("?"));
            }
            Ok::<String, String>(out)
        };
        let config = BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens: vec![42],
            eot_token_id: 7,
            vocab_size: 16,
            max_generated_tokens: 8,
        };
        let output = run_builtin_seq2seq_decode_policy(
            crate::WHISPER_DECODE_POLICY_ID,
            &config,
            &SyntheticTokenSource {
                audio_end_token_id: None,
                audio_pad_token_id: None,
                start_of_transcript_token_id: None,
                transcribe_token_id: None,
                no_timestamps_token_id: None,
                token_ids_by_content: BTreeMap::new(),
            },
            None,
            &mut step_executor,
            &decode_text_token_ids,
            |error| Seq2SeqGreedyDecodeError::TokenizerDecodeFailed { reason: error },
            |error| error.to_string(),
            |error| error.to_string(),
            &std::sync::Arc::new(crate::api::backend::TranscriptionControl::new()),
            None,
            None,
        )
        .expect("decode policy dispatch");

        assert_eq!(output.generated_tokens, vec![1, 2]);
        assert_eq!(output.text, "hello");
    }

    #[test]
    fn builtin_decode_policy_runs_seq2seq_decode() {
        let mut step_executor = SyntheticStepExecutor {
            vocab_size: 16,
            sequence: vec![1, 2, 7],
        };
        let token_table = BTreeMap::from([(1_u32, "he"), (2_u32, "llo")]);
        let decode_text_token_ids = |token_ids: &[u32]| {
            let mut out = String::new();
            for token_id in token_ids {
                out.push_str(token_table.get(token_id).copied().unwrap_or("?"));
            }
            Ok::<String, String>(out)
        };
        let config = BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens: vec![42],
            eot_token_id: 7,
            vocab_size: 16,
            max_generated_tokens: 8,
        };
        let output = run_builtin_seq2seq_decode_policy(
            crate::WHISPER_DECODE_POLICY_ID,
            &config,
            &SyntheticTokenSource {
                audio_end_token_id: None,
                audio_pad_token_id: None,
                start_of_transcript_token_id: None,
                transcribe_token_id: None,
                no_timestamps_token_id: None,
                token_ids_by_content: BTreeMap::new(),
            },
            None,
            &mut step_executor,
            &decode_text_token_ids,
            |error| Seq2SeqGreedyDecodeError::TokenizerDecodeFailed { reason: error },
            |error| error.to_string(),
            |error| error.to_string(),
            &std::sync::Arc::new(crate::api::backend::TranscriptionControl::new()),
            None,
            None,
        )
        .expect("decode policy dispatch");

        assert_eq!(output.generated_tokens, vec![1, 2]);
        assert_eq!(output.text, "hello");
    }

    #[test]
    fn builtin_decode_policy_dispatch_applies_qwen_text_postprocess() {
        let mut step_executor = SyntheticStepExecutor {
            vocab_size: 16,
            sequence: vec![1, 2, 3, 7],
        };
        let token_table = BTreeMap::from([
            (1_u32, "language English"),
            (2_u32, "<asr_text>"),
            (3_u32, " transcript "),
        ]);
        let decode_text_token_ids = |token_ids: &[u32]| {
            let mut out = String::new();
            for token_id in token_ids {
                out.push_str(token_table.get(token_id).copied().unwrap_or("?"));
            }
            Ok::<String, String>(out)
        };
        let config = BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens: vec![42],
            eot_token_id: 7,
            vocab_size: 16,
            max_generated_tokens: 8,
        };

        let output = run_builtin_seq2seq_decode_policy(
            crate::QWEN3_ASR_DECODE_POLICY_ID,
            &config,
            &SyntheticTokenSource {
                audio_end_token_id: Some(9),
                audio_pad_token_id: Some(8),
                start_of_transcript_token_id: None,
                transcribe_token_id: None,
                no_timestamps_token_id: None,
                token_ids_by_content: BTreeMap::new(),
            },
            None,
            &mut step_executor,
            &decode_text_token_ids,
            |error| Seq2SeqGreedyDecodeError::TokenizerDecodeFailed { reason: error },
            |error| error.to_string(),
            |error| error.to_string(),
            &std::sync::Arc::new(crate::api::backend::TranscriptionControl::new()),
            None,
            None,
        )
        .expect("decode policy dispatch");

        assert_eq!(output.text, "transcript");
    }

    #[test]
    fn funasr_nano_text_postprocess_removes_model_control_markers() {
        let kind = BuiltinDecodePolicySeq2SeqTextPostprocessKind::FunAsrNanoStripControlMarkersV0;

        assert_eq!(
            apply_seq2seq_text_postprocess(kind, "这还行哈，这塞的挺满的。/sil"),
            "这还行哈，这塞的挺满的。"
        );
        assert_eq!(
            apply_seq2seq_text_postprocess(kind, "/sil hello endofbreak world FFFF next /sil"),
            "hello world next"
        );
        assert_eq!(apply_seq2seq_text_postprocess(kind, "/sil"), "");
        assert_eq!(
            apply_seq2seq_text_postprocess(
                kind,
                "<|0.00|>你好[00:00-00:01] Ｏ＆｜ /sil <noise>world"
            ),
            "你好 world"
        );
        assert_eq!(
            apply_seq2seq_text_postprocess(kind, "保留未闭合<标签和未闭合[内容"),
            "保留未闭合<标签和未闭合内容"
        );
    }

    #[test]
    fn builds_qwen_seq2seq_config_with_policy_stop_tokens() {
        let descriptor = resolve_builtin_decode_policy(crate::QWEN3_ASR_DECODE_POLICY_ID)
            .expect("qwen descriptor");
        let config = build_builtin_seq2seq_decode_policy_config(
            descriptor,
            &BuiltinSeq2SeqDecodePolicyConfigInput {
                initial_prompt_tokens: vec![1, 2],
                eot_token_id: 7,
                vocab_size: 32,
                max_generated_tokens: 16,
            },
            &SyntheticTokenSource {
                audio_end_token_id: Some(9),
                audio_pad_token_id: Some(8),
                start_of_transcript_token_id: None,
                transcribe_token_id: None,
                no_timestamps_token_id: None,
                token_ids_by_content: BTreeMap::new(),
            },
            None,
        )
        .expect("qwen config");

        assert_eq!(config.stop_token_ids, vec![8, 9]);
    }

    #[test]
    fn builds_whisper_seq2seq_config_with_policy_suppression_lists() {
        let descriptor = resolve_builtin_decode_policy(crate::WHISPER_DECODE_POLICY_ID)
            .expect("whisper descriptor");
        let config = build_builtin_seq2seq_decode_policy_config(
            descriptor,
            &BuiltinSeq2SeqDecodePolicyConfigInput {
                initial_prompt_tokens: vec![1, 2],
                eot_token_id: 7,
                vocab_size: 32,
                max_generated_tokens: 16,
            },
            &SyntheticTokenSource {
                audio_end_token_id: None,
                audio_pad_token_id: None,
                start_of_transcript_token_id: Some(3),
                transcribe_token_id: Some(4),
                no_timestamps_token_id: Some(5),
                token_ids_by_content: BTreeMap::from([
                    ("<|startofprev|>", 6),
                    ("<|en|>", 8),
                    (" ", 9),
                ]),
            },
            None,
        )
        .expect("whisper config");

        assert_eq!(config.suppress_first_step_token_ids, vec![7, 9]);
        assert_eq!(config.suppress_token_ids, vec![3, 4, 5, 6, 8]);
    }

    #[test]
    fn whisper_suppression_adds_actual_language_and_task_tokens_from_prefix() {
        let descriptor = resolve_builtin_decode_policy(crate::WHISPER_DECODE_POLICY_ID)
            .expect("whisper descriptor");
        let token_source = SyntheticTokenSource {
            audio_end_token_id: None,
            audio_pad_token_id: None,
            start_of_transcript_token_id: Some(3),
            transcribe_token_id: Some(4),
            no_timestamps_token_id: Some(5),
            token_ids_by_content: BTreeMap::from([("<|startofprev|>", 6), ("<|en|>", 8), (" ", 9)]),
        };
        // Non-default multilingual prefix `<|sot|> <|fr|> <|translate|> <|notimestamps|>`
        // (fr=20, translate=21) must suppress the ACTUAL fr/translate tokens on top
        // of the hardcoded en/transcribe defaults.
        let config = build_builtin_seq2seq_decode_policy_config(
            descriptor,
            &BuiltinSeq2SeqDecodePolicyConfigInput {
                initial_prompt_tokens: vec![3, 20, 21, 5],
                eot_token_id: 7,
                vocab_size: 64,
                max_generated_tokens: 16,
            },
            &token_source,
            None,
        )
        .expect("whisper config");
        assert_eq!(config.suppress_token_ids, vec![3, 4, 5, 6, 8, 20, 21]);

        // Longform layout: the control block is preceded by `<|startofprev|> ...carry`,
        // so it does not start at index 0; the sot-relative read must still find
        // fr/translate (here at indices 3/4) and ignore the carry token (99).
        let longform = build_builtin_seq2seq_decode_policy_config(
            descriptor,
            &BuiltinSeq2SeqDecodePolicyConfigInput {
                initial_prompt_tokens: vec![6, 99, 3, 20, 21, 5],
                eot_token_id: 7,
                vocab_size: 64,
                max_generated_tokens: 16,
            },
            &token_source,
            None,
        )
        .expect("whisper longform config");
        assert_eq!(longform.suppress_token_ids, vec![3, 4, 5, 6, 8, 20, 21]);
    }

    #[test]
    fn qwen_seq2seq_config_fails_closed_when_required_special_tokens_are_missing() {
        let descriptor = resolve_builtin_decode_policy(crate::QWEN3_ASR_DECODE_POLICY_ID)
            .expect("qwen descriptor");

        let error = build_builtin_seq2seq_decode_policy_config(
            descriptor,
            &BuiltinSeq2SeqDecodePolicyConfigInput {
                initial_prompt_tokens: vec![1],
                eot_token_id: 7,
                vocab_size: 32,
                max_generated_tokens: 16,
            },
            &(),
            None,
        )
        .expect_err("missing qwen special tokens must fail closed");

        assert!(matches!(
            error,
            BuiltinDecodePolicyComponentRegistryError::MissingRequiredSpecialToken { .. }
        ));
    }

    #[test]
    fn cohere_seq2seq_config_leaves_policy_tokens_empty() {
        let descriptor = resolve_builtin_decode_policy(crate::COHERE_TRANSCRIBE_DECODE_POLICY_ID)
            .expect("cohere descriptor");
        let config = build_builtin_seq2seq_decode_policy_config(
            descriptor,
            &BuiltinSeq2SeqDecodePolicyConfigInput {
                initial_prompt_tokens: vec![1],
                eot_token_id: 7,
                vocab_size: 32,
                max_generated_tokens: 16,
            },
            &(),
            None,
        )
        .expect("cohere config");

        assert!(config.stop_token_ids.is_empty());
        assert!(config.suppress_first_step_token_ids.is_empty());
        assert!(config.suppress_token_ids.is_empty());
    }

    fn ctc_err_to_string(error: CtcGreedyDecodeError) -> String {
        error.to_string()
    }
    fn registry_err_to_string(error: BuiltinDecodePolicyComponentRegistryError) -> String {
        error.to_string()
    }

    /// 1025-wide one-hot logit row peaking at `id` (parakeet vocab incl. blank=1024).
    fn ctc_frame(id: usize) -> Vec<f32> {
        let mut row = vec![0.0f32; 1025];
        row[id] = 10.0;
        row
    }

    #[test]
    fn ctc_decode_policy_collapses_and_drops_blank() {
        let rows = [ctc_frame(5), ctc_frame(5), ctc_frame(1024), ctc_frame(7)];
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let detok = |ids: &[u32]| -> Result<String, String> {
            Ok(ids.iter().map(u32::to_string).collect::<Vec<_>>().join(","))
        };
        let result = run_builtin_ctc_decode_policy(
            crate::PARAKEET_CTC_DECODE_POLICY_ID,
            &refs,
            1025,
            None,
            &(),
            &detok,
            ctc_err_to_string,
            registry_err_to_string,
            None,
        )
        .expect("ctc decode");
        assert_eq!(result.token_ids, vec![5, 7]);
        assert_eq!(result.text, "5,7");
    }

    #[test]
    fn ctc_decode_policy_rejects_a_seq2seq_policy() {
        let rows = [ctc_frame(5)];
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let detok = |_: &[u32]| -> Result<String, String> { Ok(String::new()) };
        let error = run_builtin_ctc_decode_policy(
            crate::QWEN3_ASR_DECODE_POLICY_ID,
            &refs,
            1025,
            None,
            &(),
            &detok,
            ctc_err_to_string,
            registry_err_to_string,
            None,
        )
        .expect_err("seq2seq policy must not run through the CTC path");
        assert!(
            error.contains("cannot run through the CTC decode path"),
            "got: {error}"
        );
    }

    #[test]
    fn qwen_runtime_trace_producer_binds_to_request_receipt() {
        let receipt =
            crate::models::request_execution_receipt::NativeExecutionReceiptCollector::new();
        let _guard = crate::models::native_execution_services::install_execution_receipt_collector(
            Some(receipt.clone()),
        );
        emit_seq2seq_token_trace(
            BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1,
            0,
            7,
            false,
        );
        emit_seq2seq_topk_trace(
            BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1,
            0,
            &[1.0, 0.5],
        );
        let snapshot = receipt.snapshot();
        let text = snapshot.trace.jsonl;
        assert!(text.contains("\"schema\":\"openasr.gpu-correctness-trace.v1\""));
        assert!(text.contains("\"event\":\"token\""));
        assert!(text.contains("\"event\":\"top_k\""));
        assert_eq!(snapshot.token_steps.len(), 1);
        assert_eq!(snapshot.token_steps[0].token_id, 7);
        assert_eq!(snapshot.token_steps[0].top2_margin, Some(0.5));
    }

    #[test]
    fn seq2seq_greedy_policies_record_token_steps_through_shipped_emitter() {
        let policies = [
            crate::COHERE_TRANSCRIBE_DECODE_POLICY_ID,
            crate::WHISPER_DECODE_POLICY_ID,
            crate::MOONSHINE_DECODE_POLICY_ID,
            crate::arch::FIRERED_AED_DECODE_POLICY_ID,
            crate::arch::FIRERED_LLM_DECODE_POLICY_ID,
            crate::arch::FUNASR_NANO_DECODE_POLICY_ID,
            crate::arch::MIMO_ASR_DECODE_POLICY_ID,
            crate::arch::MOSS_TD_DECODE_POLICY_ID,
            crate::arch::GRANITE_SPEECH_DECODE_POLICY_ID,
            crate::QWEN3_ASR_DECODE_POLICY_ID,
        ];
        let token_table = BTreeMap::from([(1_u32, "he"), (2_u32, "llo")]);
        let decode_text_token_ids = |token_ids: &[u32]| {
            let mut out = String::new();
            for token_id in token_ids {
                out.push_str(token_table.get(token_id).copied().unwrap_or("?"));
            }
            Ok::<String, String>(out)
        };
        let config = BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens: vec![42],
            eot_token_id: 7,
            vocab_size: 16,
            max_generated_tokens: 8,
        };
        for decode_policy_id in policies {
            let receipt =
                crate::models::request_execution_receipt::NativeExecutionReceiptCollector::new();
            let _guard =
                crate::models::native_execution_services::install_execution_receipt_collector(
                    Some(receipt.clone()),
                );
            let mut step_executor = SyntheticStepExecutor {
                vocab_size: 16,
                sequence: vec![1, 2, 7],
            };
            let qwen = decode_policy_id == crate::QWEN3_ASR_DECODE_POLICY_ID;
            run_builtin_seq2seq_decode_policy(
                decode_policy_id,
                &config,
                &SyntheticTokenSource {
                    audio_end_token_id: qwen.then_some(9),
                    audio_pad_token_id: qwen.then_some(8),
                    start_of_transcript_token_id: None,
                    transcribe_token_id: None,
                    no_timestamps_token_id: None,
                    token_ids_by_content: BTreeMap::new(),
                },
                None,
                &mut step_executor,
                &decode_text_token_ids,
                |error| Seq2SeqGreedyDecodeError::TokenizerDecodeFailed { reason: error },
                |error| error.to_string(),
                |error| error.to_string(),
                &std::sync::Arc::new(crate::api::backend::TranscriptionControl::new()),
                None,
                None,
            )
            .unwrap_or_else(|error| panic!("{decode_policy_id}: {error}"));
            let snapshot = receipt.snapshot();
            assert!(
                snapshot.token_steps.len() >= 3,
                "{decode_policy_id} must record each greedy step including EOT"
            );
            assert_eq!(snapshot.token_steps[0].token_id, 1, "{decode_policy_id}");
            assert!(
                snapshot.token_steps[0].top2_margin.is_some(),
                "{decode_policy_id} host logits must record top-k margin"
            );
            assert!(
                snapshot.trace.event_count > 0,
                "{decode_policy_id} must emit receipt trace events"
            );
        }
    }

    #[test]
    fn qwen_env_trace_is_namespaced_as_non_authoritative_debug_jsonl() {
        let trace = tempfile::NamedTempFile::new().expect("trace file");
        let path = trace.path().to_string_lossy().into_owned();
        unsafe {
            std::env::set_var("OPENASR_SEQ2SEQ_TRACE_FILE", &path);
            std::env::set_var("OPENASR_SEQ2SEQ_TRACE_MODE", "cold");
            std::env::set_var("OPENASR_SEQ2SEQ_TRACE_PROVIDER", "cuda");
            std::env::set_var("OPENASR_SEQ2SEQ_TRACE_DEVICE", "cuda0");
        }
        emit_seq2seq_token_trace(
            BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1,
            0,
            7,
            false,
        );
        emit_seq2seq_topk_trace(
            BuiltinDecodePolicySeq2SeqTraceKind::RuntimeJsonlV1,
            0,
            &[1.0, 0.5],
        );
        let text = std::fs::read_to_string(&path).expect("read trace");
        assert!(text.contains("\"schema\":\"openasr.seq2seq-debug-trace.v1\""));
        assert!(!text.contains("\"schema\":\"openasr.gpu-correctness-trace.v1\""));
        assert!(text.contains("\"event\":\"token\""));
        assert!(text.contains("\"event\":\"top_k\""));
        unsafe {
            std::env::remove_var("OPENASR_SEQ2SEQ_TRACE_FILE");
            std::env::remove_var("OPENASR_SEQ2SEQ_TRACE_MODE");
            std::env::remove_var("OPENASR_SEQ2SEQ_TRACE_PROVIDER");
            std::env::remove_var("OPENASR_SEQ2SEQ_TRACE_DEVICE");
        }
    }
    #[test]
    fn parakeet_decode_policy_is_ctc_greedy_with_blank() {
        let parakeet = resolve_builtin_decode_policy(crate::PARAKEET_CTC_DECODE_POLICY_ID)
            .expect("parakeet ctc decode policy");
        assert_eq!(
            parakeet.execution_kind,
            BuiltinDecodePolicyExecutionKind::CtcGreedyV0
        );
        assert_eq!(parakeet.ctc_blank_token_id, Some(1024));
    }
}
