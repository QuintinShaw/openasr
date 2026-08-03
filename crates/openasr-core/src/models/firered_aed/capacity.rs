//! FireRed-AED decoder-state topology for the unified capacity planner.
//!
//! Self-attention positions and encoder cross-attention frames are separate
//! axes. Exact demand uses the frontend and encoder's integer shape oracles
//! for the current invocation; stable demand evaluates the session envelope.
//! The runtime allocates those stable spans once and activates each logical
//! invocation shape inside them without growth.
//!
use super::decode_budget::firered_aed_decode_budget;
use super::encoder_graph::predicted_encoder_time_frames;
use super::frontend::{FRAME_LENGTH_SAMPLES, FRAME_SHIFT_SAMPLES, SAMPLE_RATE_HZ};
use super::runtime_contract::FireRedAedExecutionMetadata;
use crate::capacity::topology::{
    DecoderStateDemandScope, DecoderStateTopology, InvocationEnvelope, InvocationShapeInput,
    PositionBoundProof, StateBytes, StateDemand, StateKind, TopologyError,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrDecoderStatePlanningError, GgmlAsrDecoderStatePlanningInput,
};
use crate::models::seq2seq_decoder_state::Seq2SeqStateIds;

const SELF_KV_STATE_ID: &str = "firered-aed.decoder.self_kv";
const CROSS_KV_STATE_ID: &str = "firered-aed.decoder.cross_kv";
pub(crate) const FIRERED_AED_DECODER_STATE_IDS: Seq2SeqStateIds = Seq2SeqStateIds {
    self_attention: SELF_KV_STATE_ID,
    cross_attention: CROSS_KV_STATE_ID,
};
pub(crate) const FIRERED_AED_DECODER_STATE_STREAMS:
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

pub(crate) fn plan_firered_aed_decoder_state(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    let family = "firered-aed";
    let metadata = super::runtime_contract::parse_firered_aed_execution_metadata(
        input.preflight.metadata.as_ref(),
    )
    .map_err(
        |error| GgmlAsrDecoderStatePlanningError::MetadataUnavailable {
            family,
            reason: error.to_string(),
        },
    )?;
    crate::capacity::topology::DecoderStatePlan::build(
        &FireRedAedDecoderStateTopology::new(metadata),
        input.invocation,
        input.envelope,
    )
    .map_err(|source| GgmlAsrDecoderStatePlanningError::Topology { family, source })
}

fn firered_aed_resident_kv_bytes(
    metadata: &FireRedAedExecutionMetadata,
    positions: usize,
    sequences: usize,
    bytes_per_value: usize,
) -> Result<StateBytes, TopologyError> {
    let resident = metadata
        .decoder_n_layers
        .checked_mul(2)
        .and_then(|value| value.checked_mul(metadata.d_model))
        .and_then(|value| value.checked_mul(positions))
        .and_then(|value| value.checked_mul(sequences))
        .and_then(|value| value.checked_mul(bytes_per_value))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(TopologyError::ArithmeticOverflow {
            operation: "firered-aed resident KV bytes",
        })?;
    Ok(StateBytes { host: 0, resident })
}

pub(crate) fn firered_aed_encoder_frame_count_for_samples(
    samples: usize,
    sample_rate_hz: u32,
) -> Result<usize, TopologyError> {
    if sample_rate_hz != SAMPLE_RATE_HZ {
        return Err(TopologyError::UnsupportedSampleRate {
            expected_hz: SAMPLE_RATE_HZ,
            actual_hz: sample_rate_hz,
        });
    }
    let mel_frames = samples
        .checked_sub(FRAME_LENGTH_SAMPLES)
        .map(|tail| tail / FRAME_SHIFT_SAMPLES + 1)
        .unwrap_or(0);
    if mel_frames == 0 {
        return Err(TopologyError::Unavailable {
            reason: "firered-aed audio is too short to produce one fbank frame".to_string(),
        });
    }
    predicted_encoder_time_frames(mel_frames).map_err(|error| TopologyError::Unavailable {
        reason: format!("firered-aed encoder shape is invalid: {error}"),
    })
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FireRedAedDecoderStateTopology {
    metadata: FireRedAedExecutionMetadata,
}

impl FireRedAedDecoderStateTopology {
    pub(crate) const fn new(metadata: FireRedAedExecutionMetadata) -> Self {
        Self { metadata }
    }
}

impl DecoderStateTopology for FireRedAedDecoderStateTopology {
    fn demands(
        &self,
        scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
    ) -> Result<Vec<StateDemand>, TopologyError> {
        let invocation = match scope {
            DecoderStateDemandScope::ExactInvocation(invocation) => invocation,
            DecoderStateDemandScope::StableEnvelope(envelope) => envelope.maximum_invocation(),
        };
        let sequences = invocation.sequences().get() as usize;
        let cross_frames = firered_aed_encoder_frame_count_for_samples(
            invocation.samples(),
            invocation.sample_rate_hz().get(),
        )?;
        let decode_budget = firered_aed_decode_budget(cross_frames, self.metadata.decoder_pe_len)
            .map_err(|error| TopologyError::Unavailable {
            reason: error.to_string(),
        })?;
        Ok(vec![
            StateDemand::new(
                SELF_KV_STATE_ID,
                StateKind::SelfAttentionKv,
                decode_budget.self_kv_positions,
                self.metadata.decoder_pe_len,
                firered_aed_resident_kv_bytes(
                    &self.metadata,
                    decode_budget.self_kv_positions,
                    sequences,
                    std::mem::size_of::<u16>(),
                )?,
                PositionBoundProof::Exact,
            )?,
            StateDemand::new(
                CROSS_KV_STATE_ID,
                StateKind::CrossAttentionKv,
                cross_frames,
                self.metadata.encoder_max_frames(),
                firered_aed_resident_kv_bytes(
                    &self.metadata,
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

    use super::*;
    use crate::capacity::topology::{DecoderStatePlan, InvocationEnvelope, StateKind};

    /// Real-checkpoint-shaped metadata (the same values `runtime_contract`'s
    /// test fixture parses: 16-layer MHA decoder at d_model 1280 / 20 heads /
    /// head_dim 64, decoder PE span 5000).
    fn reference_metadata() -> FireRedAedExecutionMetadata {
        FireRedAedExecutionMetadata {
            encoder_n_layers: 16,
            d_model: 1280,
            n_heads: 20,
            head_dim: 64,
            encoder_ffn_dim: 5120,
            conv_kernel: 33,
            subsample_channels: 32,
            subsample_out_dim: 608,
            feature_dim: 80,
            encoder_pe_len: 9999,
            decoder_n_layers: 16,
            decoder_ffn_dim: 5120,
            decoder_pe_len: 5000,
            vocab_size: 7832,
            sos_token_id: 3,
            eos_token_id: 4,
            pad_token_id: 2,
        }
    }

    #[test]
    fn resident_byte_oracle_matches_runtime_tensor_shapes() {
        let metadata = reference_metadata();
        // layers x 2 x d_model f32 values per cross frame:
        // 16 * 2 * 1280 * 4 B = 160 KiB.
        assert_eq!(
            firered_aed_resident_kv_bytes(&metadata, 1, 1, std::mem::size_of::<f32>()).unwrap(),
            StateBytes {
                host: 0,
                resident: 16 * 2 * 1280 * 4,
            }
        );
        // 16 layers x 2 (K+V) x 20 heads x 5000 positions x 64 head_dim x 2 B
        // (f16) = ~390 MiB.
        assert_eq!(
            firered_aed_resident_kv_bytes(
                &metadata,
                metadata.decoder_pe_len,
                1,
                std::mem::size_of::<u16>(),
            )
            .unwrap(),
            StateBytes {
                host: 0,
                resident: 16 * 2 * 20 * 5000 * 64 * 2,
            }
        );
    }

    #[test]
    fn topology_keeps_self_and_exact_cross_axes_independent() {
        let metadata = reference_metadata();
        let envelope = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(30_000).unwrap(),
        )
        .unwrap();
        let expected_cross =
            firered_aed_encoder_frame_count_for_samples(envelope.max_samples(), 16_000).unwrap();
        // 30s -> 2998 snip-edge fbank frames -> the padded 2x k3/s2
        // subsampling stem emits 1501 -> 750 encoder frames.
        assert_eq!(expected_cross, 750);
        let plan = DecoderStatePlan::for_envelope(
            &FireRedAedDecoderStateTopology::new(metadata),
            envelope,
        )
        .unwrap();
        assert_eq!(plan.reserve_positions_by_id(SELF_KV_STATE_ID), Some(750));
        assert_eq!(
            plan.reserve_positions(StateKind::CrossAttentionKv),
            Some(expected_cross)
        );
    }

    #[test]
    fn duration_boundaries_follow_encoder_and_decoder_schedules() {
        let rate = NonZeroU32::new(16_000).unwrap();
        let topology = FireRedAedDecoderStateTopology::new(reference_metadata());
        for (seconds, expected_cross, expected_self) in
            [(1, 25, 25), (30, 750, 750), (60, 1_500, 1_500)]
        {
            let envelope = InvocationEnvelope::new(rate, seconds * 16_000).unwrap();
            let plan = DecoderStatePlan::for_envelope(&topology, envelope).unwrap();
            assert_eq!(
                plan.reserve_positions_by_id(SELF_KV_STATE_ID),
                Some(expected_self)
            );
            assert_eq!(
                plan.reserve_positions_by_id(CROSS_KV_STATE_ID),
                Some(expected_cross)
            );
        }
        let five_minutes = InvocationEnvelope::new(rate, 300 * 16_000).unwrap();
        assert!(matches!(
            DecoderStatePlan::for_envelope(&topology, five_minutes),
            Err(TopologyError::PositionCapExceeded {
                state: CROSS_KV_STATE_ID,
                required: 7_500,
                hard_cap: 5_000,
            })
        ));
    }

    #[test]
    fn learned_self_and_cross_caps_fail_closed_independently() {
        let rate = NonZeroU32::new(16_000).unwrap();
        let one_second = InvocationEnvelope::new(rate, 16_000).unwrap();
        let mut no_decode_context = reference_metadata();
        no_decode_context.decoder_pe_len = 1;
        assert!(matches!(
            DecoderStatePlan::for_envelope(
                &FireRedAedDecoderStateTopology::new(no_decode_context),
                one_second,
            ),
            Err(TopologyError::Unavailable { .. })
        ));

        let mut small_encoder = reference_metadata();
        small_encoder.encoder_pe_len = 100;
        let thirty_seconds = InvocationEnvelope::new(rate, 30 * 16_000).unwrap();
        assert!(matches!(
            DecoderStatePlan::for_envelope(
                &FireRedAedDecoderStateTopology::new(small_encoder),
                thirty_seconds,
            ),
            Err(TopologyError::PositionCapExceeded {
                state: CROSS_KV_STATE_ID,
                required: 750,
                hard_cap: 50,
            })
        ));
    }
}
