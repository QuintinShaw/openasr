//! Persistent decoder-state topology for Whisper's fixed-window decoder.

use crate::capacity::topology::{
    DecoderStateDemandScope, DecoderStateTopology, InvocationEnvelope, InvocationShapeInput,
    PositionBoundProof, StateBytes, StateDemand, StateKind, TopologyError,
    causal_prefix_positions_with_context_cap,
};

use super::mel::{WHISPER_HOP_LENGTH, WHISPER_SAMPLE_RATE_HZ};
use super::runtime_contract::WhisperGgmlExecutionMetadata;
use crate::models::ggml_asr_executor::{
    GgmlAsrDecoderStatePlanningError, GgmlAsrDecoderStatePlanningInput,
};
use crate::models::seq2seq_decoder_state::Seq2SeqStateIds;

const SELF_KV_STATE_ID: &str = "whisper.decoder.self_kv";
const CROSS_KV_STATE_ID: &str = "whisper.decoder.cross_kv";
pub(super) const WHISPER_MAX_GENERATED_TOKENS: usize = 256;
pub(crate) const WHISPER_DECODER_STATE_IDS: Seq2SeqStateIds = Seq2SeqStateIds {
    self_attention: SELF_KV_STATE_ID,
    cross_attention: CROSS_KV_STATE_ID,
};
pub(crate) const WHISPER_DECODER_STATE_STREAMS:
    &[crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract] = &[
    crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract::new(
        SELF_KV_STATE_ID,
        StateKind::SelfAttentionKv,
    ),
    crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract::new(
        CROSS_KV_STATE_ID,
        StateKind::CrossAttentionKv,
    ),
];

pub(crate) fn plan_whisper_decoder_state(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    let family = "whisper";
    let metadata = super::runtime_contract::validate_whisper_execution_metadata(
        input.preflight.metadata.as_ref(),
    )
    .map_err(
        |error| GgmlAsrDecoderStatePlanningError::MetadataUnavailable {
            family,
            reason: error.to_string(),
        },
    )?;
    let tokenizer =
        super::tokenizer::WhisperTokenizer::from_gguf_metadata(input.preflight.metadata.as_ref())
            .map_err(
            |error| GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
                family,
                reason: error.to_string(),
            },
        )?;
    plan_whisper_decoder_state_with_components(input, metadata, &tokenizer)
}

pub(super) fn plan_whisper_decoder_state_with_prepared_runtime(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
    prepared: &super::ggml_executor::WhisperPreparedRuntime,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    plan_whisper_decoder_state_with_components(
        input,
        prepared.execution.clone(),
        &prepared.tokenizer,
    )
}

fn plan_whisper_decoder_state_with_components(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
    metadata: WhisperGgmlExecutionMetadata,
    tokenizer: &super::tokenizer::WhisperTokenizer,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    let family = "whisper";
    let prompt_bounds = super::prompt::whisper_prompt_position_bounds(
        &metadata,
        tokenizer,
        input.request_options,
        input.envelope.max_prompt_tokens(),
    )
    .map_err(
        |error| GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
            family,
            reason: error.to_string(),
        },
    )?;
    crate::capacity::topology::DecoderStatePlan::build(
        &WhisperDecoderStateTopology::new(metadata, prompt_bounds.logical, prompt_bounds.stable),
        input.invocation,
        input.envelope,
    )
    .map_err(|source| GgmlAsrDecoderStatePlanningError::Topology { family, source })
}

const CROSS_FRAME_ALIGNMENT: usize = 256;

fn aligned_cross_frames(frames: usize) -> Result<usize, TopologyError> {
    let remainder = frames % CROSS_FRAME_ALIGNMENT;
    if remainder == 0 {
        return Ok(frames);
    }
    frames
        .checked_add(CROSS_FRAME_ALIGNMENT - remainder)
        .ok_or(TopologyError::ArithmeticOverflow {
            operation: "whisper cross-KV frame alignment",
        })
}

fn resident_kv_bytes(
    metadata: &WhisperGgmlExecutionMetadata,
    positions: usize,
    sequences: usize,
) -> Result<StateBytes, TopologyError> {
    let resident = metadata
        .decoder_layers
        .checked_mul(2)
        .and_then(|value| value.checked_mul(metadata.decoder_hidden_size))
        .and_then(|value| value.checked_mul(positions))
        .and_then(|value| value.checked_mul(sequences))
        .and_then(|value| value.checked_mul(std::mem::size_of::<u16>()))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(TopologyError::ArithmeticOverflow {
            operation: "whisper resident KV bytes",
        })?;
    Ok(StateBytes { host: 0, resident })
}

#[derive(Debug, Clone)]
pub(crate) struct WhisperDecoderStateTopology {
    metadata: WhisperGgmlExecutionMetadata,
    logical_prompt_positions: usize,
    stable_prompt_positions: usize,
}

impl WhisperDecoderStateTopology {
    pub(crate) const fn new(
        metadata: WhisperGgmlExecutionMetadata,
        logical_prompt_positions: usize,
        stable_prompt_positions: usize,
    ) -> Self {
        Self {
            metadata,
            logical_prompt_positions,
            stable_prompt_positions,
        }
    }

    fn self_kv_positions(&self, prompt_positions: usize) -> Result<usize, TopologyError> {
        let generated_positions = self
            .metadata
            .max_target_positions
            .checked_sub(prompt_positions)
            .filter(|positions| *positions > 0)
            .ok_or_else(|| TopologyError::Unavailable {
                reason: format!(
                    "whisper prompt {prompt_positions} exhausts decoder context {}",
                    self.metadata.max_target_positions
                ),
            })?
            .min(WHISPER_MAX_GENERATED_TOKENS);
        causal_prefix_positions_with_context_cap(
            SELF_KV_STATE_ID,
            prompt_positions,
            generated_positions,
            self.metadata.max_target_positions,
        )
    }
}

impl DecoderStateTopology for WhisperDecoderStateTopology {
    fn demands(
        &self,
        scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
    ) -> Result<Vec<StateDemand>, TopologyError> {
        let (invocation, prompt_positions, stable_envelope) = match scope {
            DecoderStateDemandScope::ExactInvocation(invocation) => {
                (invocation, self.logical_prompt_positions, None)
            }
            DecoderStateDemandScope::StableEnvelope(envelope) => (
                envelope.maximum_invocation(),
                self.stable_prompt_positions,
                Some(envelope),
            ),
        };
        if invocation.sample_rate_hz().get() != WHISPER_SAMPLE_RATE_HZ {
            return Err(TopologyError::UnsupportedSampleRate {
                expected_hz: WHISPER_SAMPLE_RATE_HZ,
                actual_hz: invocation.sample_rate_hz().get(),
            });
        }
        let max_samples = self
            .metadata
            .encoder_context_length
            .checked_mul(2)
            .and_then(|target_frames| target_frames.checked_mul(WHISPER_HOP_LENGTH))
            .ok_or(TopologyError::ArithmeticOverflow {
                operation: "whisper fixed frontend sample window",
            })?;
        if invocation.samples() > max_samples {
            return Err(TopologyError::InvocationSampleLimitExceeded {
                required_samples: invocation.samples(),
                max_samples,
            });
        }
        let sequences = invocation.sequences().get() as usize;
        let cross_frames = self.metadata.encoder_context_length;
        let cross_position_cap = aligned_cross_frames(cross_frames)?;
        let self_kv_positions = self.self_kv_positions(prompt_positions)?;
        let mut demands = vec![
            StateDemand::new(
                SELF_KV_STATE_ID,
                StateKind::SelfAttentionKv,
                self_kv_positions,
                self.metadata.max_target_positions,
                resident_kv_bytes(&self.metadata, self_kv_positions, sequences)?,
                PositionBoundProof::Exact,
            )?,
            StateDemand::new(
                CROSS_KV_STATE_ID,
                StateKind::CrossAttentionKv,
                cross_frames,
                cross_position_cap,
                resident_kv_bytes(&self.metadata, cross_frames, sequences)?,
                PositionBoundProof::Exact,
            )?,
        ];
        if let Some(envelope) = stable_envelope {
            let cross = demands
                .iter_mut()
                .find(|demand| demand.id == CROSS_KV_STATE_ID)
                .ok_or(TopologyError::StateIdSetMismatch {
                    id: CROSS_KV_STATE_ID,
                })?;
            let aligned = aligned_cross_frames(cross.positions)?;
            cross.positions = aligned;
            cross.hard_position_cap = aligned;
            cross.bytes = resident_kv_bytes(
                &self.metadata,
                aligned,
                envelope.max_sequences().get() as usize,
            )?;
            cross.proof = PositionBoundProof::Conservative {
                basis: "persistent cross-KV layer stride rounded to 256 frames",
            };
        }
        Ok(demands)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use crate::capacity::topology::{DecoderStatePlan, InvocationEnvelope, StateKind};

    use super::*;

    fn metadata() -> WhisperGgmlExecutionMetadata {
        WhisperGgmlExecutionMetadata {
            encoder_layers: 32,
            decoder_layers: 32,
            encoder_hidden_size: 1280,
            encoder_attention_heads: 20,
            encoder_context_length: 1500,
            decoder_attention_heads: 20,
            max_target_positions: 448,
            decoder_hidden_size: 1280,
            vocab_size: 51_865,
            decoder_start_token_id: 50_258,
            eos_token_id: 50_257,
            encoder_mels_count: 80,
        }
    }

    #[test]
    fn fixed_window_keeps_self_and_aligned_cross_axes_independent() {
        let envelope = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(30_000).unwrap(),
        )
        .unwrap();
        let plan = DecoderStatePlan::for_envelope(
            &WhisperDecoderStateTopology::new(metadata(), 4, 36),
            envelope,
        )
        .unwrap();
        assert_eq!(
            plan.reserve_positions(StateKind::SelfAttentionKv),
            Some(291)
        );
        assert_eq!(
            plan.logical_positions(StateKind::SelfAttentionKv),
            Some(259)
        );
        assert_eq!(
            plan.logical_positions(StateKind::CrossAttentionKv),
            Some(1500)
        );
        assert_eq!(
            plan.reserve_positions(StateKind::CrossAttentionKv),
            Some(1536)
        );
    }

    #[test]
    fn fixed_window_rejects_audio_the_frontend_would_silently_trim() {
        let envelope = InvocationEnvelope::new(NonZeroU32::new(16_000).unwrap(), 480_001).unwrap();
        let error = DecoderStatePlan::for_envelope(
            &WhisperDecoderStateTopology::new(metadata(), 4, 4),
            envelope,
        )
        .expect_err("more than the 30-second frontend window must fail closed");
        assert!(matches!(
            error,
            TopologyError::InvocationSampleLimitExceeded {
                required_samples: 480_001,
                max_samples: 480_000,
            }
        ));
    }

    #[test]
    fn one_and_thirty_seconds_share_the_fixed_window_arena_but_longer_calls_fail() {
        let rate = NonZeroU32::new(16_000).unwrap();
        let topology = WhisperDecoderStateTopology::new(metadata(), 4, 4);
        for seconds in [1, 30] {
            let envelope = InvocationEnvelope::new(rate, seconds * 16_000).unwrap();
            let plan = DecoderStatePlan::for_envelope(&topology, envelope).unwrap();
            assert_eq!(plan.reserve_positions_by_id(SELF_KV_STATE_ID), Some(259));
            assert_eq!(plan.reserve_positions_by_id(CROSS_KV_STATE_ID), Some(1_536));
        }
        for seconds in [60, 300] {
            let envelope = InvocationEnvelope::new(rate, seconds * 16_000).unwrap();
            assert!(matches!(
                DecoderStatePlan::for_envelope(&topology, envelope),
                Err(TopologyError::InvocationSampleLimitExceeded { .. })
            ));
        }
    }

    #[test]
    fn tiny_learned_context_is_not_treated_as_an_allocation_request() {
        let mut tiny = metadata();
        tiny.max_target_positions = 1;
        let envelope = InvocationEnvelope::new(NonZeroU32::new(16_000).unwrap(), 16_000).unwrap();
        assert!(matches!(
            DecoderStatePlan::for_envelope(&WhisperDecoderStateTopology::new(tiny, 1, 1), envelope,),
            Err(TopologyError::Unavailable { .. })
        ));
    }
}
