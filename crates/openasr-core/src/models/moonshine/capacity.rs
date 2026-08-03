//! Persistent decoder-state topology for Moonshine encoder-decoder packs.

use crate::capacity::topology::{
    DecoderStateDemandScope, DecoderStateTopology, InvocationEnvelope, InvocationShapeInput,
    PositionBoundProof, StateBytes, StateDemand, StateKind, TopologyError,
};

use super::encoder_graph::moonshine_encoder_frame_count_for_samples;
use super::runtime_contract::MoonshineExecutionMetadata;
use crate::models::ggml_asr_executor::{
    GgmlAsrDecoderStatePlanningError, GgmlAsrDecoderStatePlanningInput,
};
use crate::models::seq2seq_decoder_state::Seq2SeqStateIds;

pub(crate) const SELF_KV_STATE_ID: &str = "moonshine.decoder.self_kv";
const CROSS_KV_STATE_ID: &str = "moonshine.decoder.cross_kv";
pub(crate) const MOONSHINE_DECODER_STATE_IDS: Seq2SeqStateIds = Seq2SeqStateIds {
    self_attention: SELF_KV_STATE_ID,
    cross_attention: CROSS_KV_STATE_ID,
};
pub(crate) const MOONSHINE_DECODER_STATE_STREAMS:
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

pub(crate) fn plan_moonshine_decoder_state(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    let family = "moonshine";
    let metadata = super::runtime_contract::parse_moonshine_execution_metadata(
        input.preflight.metadata.as_ref(),
    )
    .map_err(
        |error| GgmlAsrDecoderStatePlanningError::MetadataUnavailable {
            family,
            reason: error.to_string(),
        },
    )?;
    let topology = MoonshineDecoderStateTopology::for_envelope(metadata, input.envelope)
        .map_err(|source| GgmlAsrDecoderStatePlanningError::Topology { family, source })?;
    crate::capacity::topology::DecoderStatePlan::build(&topology, input.invocation, input.envelope)
        .map_err(|source| GgmlAsrDecoderStatePlanningError::Topology { family, source })
}

fn resident_kv_bytes(
    metadata: MoonshineExecutionMetadata,
    positions: usize,
    sequences: usize,
    bytes_per_value: usize,
) -> Result<StateBytes, TopologyError> {
    let resident = metadata
        .decoder_layers
        .checked_mul(2)
        .and_then(|value| value.checked_mul(metadata.d_model))
        .and_then(|value| value.checked_mul(positions))
        .and_then(|value| value.checked_mul(sequences))
        .and_then(|value| value.checked_mul(bytes_per_value))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(TopologyError::ArithmeticOverflow {
            operation: "moonshine resident KV bytes",
        })?;
    Ok(StateBytes { host: 0, resident })
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MoonshineDecoderStateTopology {
    metadata: MoonshineExecutionMetadata,
    cross_position_cap: usize,
}

impl MoonshineDecoderStateTopology {
    /// Bind the topology to the product/session's largest legal invocation.
    /// Moonshine has no separate pack-carried encoder position ceiling; its
    /// exact convolutional output for the configured maximum chunk is the
    /// finite cross-KV safety cap.
    pub(crate) fn for_envelope(
        metadata: MoonshineExecutionMetadata,
        envelope: InvocationEnvelope,
    ) -> Result<Self, TopologyError> {
        if envelope.sample_rate_hz().get() != metadata.sample_rate_hz {
            return Err(TopologyError::UnsupportedSampleRate {
                expected_hz: metadata.sample_rate_hz,
                actual_hz: envelope.sample_rate_hz().get(),
            });
        }
        let cross_position_cap = moonshine_encoder_frame_count_for_samples(envelope.max_samples())
            .map_err(|error| TopologyError::Unavailable {
                reason: format!("moonshine encoder shape is invalid: {error}"),
            })?;
        if cross_position_cap == 0 {
            return Err(TopologyError::Unavailable {
                reason: "moonshine maximum invocation produces no encoder frames".to_string(),
            });
        }
        Ok(Self {
            metadata,
            cross_position_cap,
        })
    }
}

impl DecoderStateTopology for MoonshineDecoderStateTopology {
    fn demands(
        &self,
        scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
    ) -> Result<Vec<StateDemand>, TopologyError> {
        let invocation = match scope {
            DecoderStateDemandScope::ExactInvocation(invocation) => invocation,
            DecoderStateDemandScope::StableEnvelope(envelope) => envelope.maximum_invocation(),
        };
        if invocation.sample_rate_hz().get() != self.metadata.sample_rate_hz {
            return Err(TopologyError::UnsupportedSampleRate {
                expected_hz: self.metadata.sample_rate_hz,
                actual_hz: invocation.sample_rate_hz().get(),
            });
        }
        let cross_frames = moonshine_encoder_frame_count_for_samples(invocation.samples())
            .map_err(|error| TopologyError::Unavailable {
                reason: format!("moonshine encoder shape is invalid: {error}"),
            })?;
        if cross_frames == 0 {
            return Err(TopologyError::Unavailable {
                reason: "moonshine audio is too short to produce one encoder frame".to_string(),
            });
        }
        let sequences = invocation.sequences().get() as usize;
        // Moonshine uses a one-token BOS prompt and permits C - 1 generated
        // tokens semantically. The current greedy schedule writes the prompt
        // and only the first G - 1 generated tokens, so its exact self-KV span
        // is C - 1.
        let self_kv_positions = self
            .metadata
            .decoder_max_context
            .checked_sub(1)
            .filter(|positions| *positions > 0)
            .ok_or_else(|| TopologyError::Unavailable {
                reason: format!(
                    "moonshine decoder context {} cannot hold BOS plus a generated token",
                    self.metadata.decoder_max_context
                ),
            })?;
        Ok(vec![
            StateDemand::new(
                SELF_KV_STATE_ID,
                StateKind::SelfAttentionKv,
                self_kv_positions,
                self.metadata.decoder_max_context,
                resident_kv_bytes(
                    self.metadata,
                    self_kv_positions,
                    sequences,
                    std::mem::size_of::<u16>(),
                )?,
                PositionBoundProof::Exact,
            )?,
            StateDemand::new(
                CROSS_KV_STATE_ID,
                StateKind::CrossAttentionKv,
                cross_frames,
                self.cross_position_cap,
                resident_kv_bytes(
                    self.metadata,
                    cross_frames,
                    sequences,
                    std::mem::size_of::<f32>(),
                )?,
                PositionBoundProof::Exact,
            )?,
        ])
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use crate::capacity::topology::{DecoderStatePlan, InvocationEnvelope, StateKind};

    use super::*;

    fn metadata() -> MoonshineExecutionMetadata {
        MoonshineExecutionMetadata {
            vocab_size: 32_000,
            d_model: 288,
            encoder_layers: 6,
            decoder_layers: 6,
            n_heads: 8,
            head_dim: 36,
            rotary_dim: 32,
            encoder_ffn_dim: 1152,
            decoder_ffn_dim: 1152,
            decoder_max_context: 128,
            bos_token_id: 1,
            eos_token_id: 2,
            sample_rate_hz: 16_000,
            rope_theta: 10_000.0,
        }
    }

    #[test]
    fn waveform_counter_sizes_only_cross_axis() {
        let envelope = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(30_000).unwrap(),
        )
        .unwrap();
        let expected_cross =
            moonshine_encoder_frame_count_for_samples(envelope.max_samples()).unwrap();
        // 480000 samples through valid conv k127/s64, k7/s3, k3/s2:
        // 7499 -> 2498 -> 1248 encoder frames.
        assert_eq!(expected_cross, 1_248);
        let topology = MoonshineDecoderStateTopology::for_envelope(metadata(), envelope).unwrap();
        let plan = DecoderStatePlan::for_envelope(&topology, envelope).unwrap();
        assert_eq!(
            plan.reserve_positions(StateKind::SelfAttentionKv),
            Some(127)
        );
        assert_eq!(
            plan.reserve_positions(StateKind::CrossAttentionKv),
            Some(expected_cross)
        );
    }

    #[test]
    fn duration_boundaries_change_only_the_cross_axis() {
        let rate = NonZeroU32::new(16_000).unwrap();
        for (seconds, expected_cross) in [(1, 40), (30, 1_248), (60, 2_498), (300, 12_498)] {
            let envelope = InvocationEnvelope::new(rate, seconds * 16_000).unwrap();
            let topology =
                MoonshineDecoderStateTopology::for_envelope(metadata(), envelope).unwrap();
            let plan = DecoderStatePlan::for_envelope(&topology, envelope).unwrap();
            assert_eq!(plan.reserve_positions_by_id(SELF_KV_STATE_ID), Some(127));
            assert_eq!(
                plan.reserve_positions_by_id(CROSS_KV_STATE_ID),
                Some(expected_cross)
            );
        }
    }

    #[test]
    fn tiny_decoder_context_fails_before_graph_allocation() {
        let mut tiny = metadata();
        tiny.decoder_max_context = 1;
        let envelope = InvocationEnvelope::new(NonZeroU32::new(16_000).unwrap(), 16_000).unwrap();
        let topology = MoonshineDecoderStateTopology::for_envelope(tiny, envelope).unwrap();
        assert!(matches!(
            DecoderStatePlan::for_envelope(&topology, envelope),
            Err(TopologyError::Unavailable { .. })
        ));
    }

    #[test]
    fn product_envelope_is_a_finite_cross_position_cap() {
        let product_envelope = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(30_000).unwrap(),
        )
        .unwrap();
        let topology =
            MoonshineDecoderStateTopology::for_envelope(metadata(), product_envelope).unwrap();
        let oversized = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(60_000).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            DecoderStatePlan::for_envelope(&topology, oversized),
            Err(TopologyError::PositionCapExceeded {
                state: "moonshine.decoder.cross_kv",
                ..
            })
        ));
    }
}
