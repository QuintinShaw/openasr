use thiserror::Error;

use crate::PhraseBiasConfig;
use crate::arch::OpenAsrArchitectureRegistry;
use crate::models::ctc_greedy_decode::{
    CtcGreedyDecodeConfig, CtcGreedyDecodeError, CtcGreedyDecodeResult, run_ctc_greedy_decode,
};
use crate::models::phrase_bias_decode::{
    PhraseBiasBuildError, PhraseBiasTokenEncoder, build_token_phrase_biases,
};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeConfig, Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeResult,
    Seq2SeqGreedyDecodeStepExecutor, run_seq2seq_greedy_decode_loop_with_adapter_v0,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinDecodePolicyLongformPromptCarryMode {
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
    Qwen3AsrStripControlPrefixV0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinDecodePolicySeq2SeqTraceKind {
    None,
    WhisperEnvV0,
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

const BUILTIN_DECODE_POLICY_COMPONENTS: &[BuiltinDecodePolicyComponentDescriptor] = &[
    BuiltinDecodePolicyComponentDescriptor {
        decode_policy_id: crate::COHERE_TRANSCRIBE_DECODE_POLICY_ID,
        execution_kind: BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0,
        seq2seq_text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind::None,
        seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind::None,
        seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind::None,
        longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode::TokenHistory,
        longform_profile: BuiltinDecodePolicyLongformProfile::ConservativeSeq2SeqV1,
        ctc_blank_token_id: None,
    },
    BuiltinDecodePolicyComponentDescriptor {
        decode_policy_id: crate::WHISPER_DECODE_POLICY_ID,
        execution_kind: BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0,
        seq2seq_text_postprocess_kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity,
        seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind::WhisperEnvV0,
        seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind::None,
        seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind::WhisperDefaultV0,
        longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode::TokenHistory,
        longform_profile: BuiltinDecodePolicyLongformProfile::Default,
        ctc_blank_token_id: None,
    },
    BuiltinDecodePolicyComponentDescriptor {
        decode_policy_id: crate::QWEN3_ASR_DECODE_POLICY_ID,
        execution_kind: BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0,
        seq2seq_text_postprocess_kind:
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::Qwen3AsrStripControlPrefixV0,
        seq2seq_trace_kind: BuiltinDecodePolicySeq2SeqTraceKind::None,
        seq2seq_stop_token_kind: BuiltinDecodePolicySeq2SeqStopTokenKind::Qwen3AsrAudioBoundaryV0,
        seq2seq_suppression_kind: BuiltinDecodePolicySeq2SeqSuppressionKind::None,
        longform_prompt_carry_mode: BuiltinDecodePolicyLongformPromptCarryMode::Text,
        longform_profile: BuiltinDecodePolicyLongformProfile::Default,
        ctc_blank_token_id: None,
    },
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
    },
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
    },
];

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
    resolve_builtin_decode_policy(descriptor.decode_policy_id)
}

pub(crate) fn resolve_builtin_decode_policy(
    decode_policy_id: &str,
) -> Result<BuiltinDecodePolicyComponentDescriptor, BuiltinDecodePolicyComponentRegistryError> {
    BUILTIN_DECODE_POLICY_COMPONENTS
        .iter()
        .copied()
        .find(|descriptor| descriptor.decode_policy_id == decode_policy_id)
        .ok_or_else(
            || BuiltinDecodePolicyComponentRegistryError::UnknownDecodePolicy {
                decode_policy_id: decode_policy_id.to_string(),
            },
        )
}

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
            run_ctc_greedy_decode(
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

pub(crate) fn apply_seq2seq_text_postprocess(
    kind: BuiltinDecodePolicySeq2SeqTextPostprocessKind,
    decoded: &str,
) -> String {
    match kind {
        BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity => decoded.to_string(),
        BuiltinDecodePolicySeq2SeqTextPostprocessKind::Qwen3AsrStripControlPrefixV0 => decoded
            [seq2seq_transcript_byte_start(kind, decoded)..]
            .trim()
            .to_string(),
    }
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
        BuiltinDecodePolicySeq2SeqTextPostprocessKind::Identity => 0,
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
            BuiltinDecodePolicySeq2SeqTraceKind::WhisperEnvV0
        );
        assert_eq!(
            qwen.seq2seq_text_postprocess_kind,
            BuiltinDecodePolicySeq2SeqTextPostprocessKind::Qwen3AsrStripControlPrefixV0
        );
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
        )
        .expect("decode policy dispatch");

        assert_eq!(output.text, "transcript");
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
        )
        .expect_err("seq2seq policy must not run through the CTC path");
        assert!(
            error.contains("cannot run through the CTC decode path"),
            "got: {error}"
        );
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
