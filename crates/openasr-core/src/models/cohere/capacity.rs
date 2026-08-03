//! Cohere decoder-state topology for the unified capacity planner.
//!
//! Self-attention positions and encoder cross-attention frames are separate
//! axes. Exact demand uses the frontend's integer frame-shape oracle for the
//! current invocation; stable demand uses the same oracle at the session
//! envelope. The runtime allocates those stable resident spans once and
//! activates each invocation's logical spans inside them without growth.
//!
use super::runtime_contract::CohereTranscribeExecutionMetadata;
use crate::capacity::topology::{
    DecoderStateDemandScope, DecoderStateTopology, InvocationEnvelope, InvocationShapeInput,
    PositionBoundProof, StateBytes, StateDemand, StateKind, TopologyError,
};
use crate::models::audio_frontend::{PadMode, StftFramer};
use crate::models::ggml_asr_executor::{
    GgmlAsrDecoderStatePlanningError, GgmlAsrDecoderStatePlanningInput,
};
use crate::models::seq2seq_decoder_state::Seq2SeqStateIds;

const SELF_KV_STATE_ID: &str = "cohere.decoder.self_kv";
const CROSS_KV_STATE_ID: &str = "cohere.decoder.cross_kv";
pub(crate) const COHERE_DECODER_STATE_IDS: Seq2SeqStateIds = Seq2SeqStateIds {
    self_attention: SELF_KV_STATE_ID,
    cross_attention: CROSS_KV_STATE_ID,
};
pub(crate) const COHERE_DECODER_STATE_STREAMS:
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

pub(crate) fn plan_cohere_decoder_state(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    let family = "cohere-transcribe";
    let metadata = super::runtime_contract::parse_cohere_transcribe_execution_metadata(
        input.preflight.metadata.as_ref(),
    )
    .map_err(
        |error| GgmlAsrDecoderStatePlanningError::MetadataUnavailable {
            family,
            reason: error.to_string(),
        },
    )?;
    let tokenizer = super::tokenizer::CohereTranscribeTokenizer::from_gguf_metadata(
        input.preflight.metadata.as_ref(),
    )
    .map_err(
        |error| GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
            family,
            reason: error.to_string(),
        },
    )?;
    plan_cohere_decoder_state_with_components(input, metadata, &tokenizer)
}

pub(crate) fn plan_cohere_decoder_state_with_prepared_runtime(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
    prepared: &super::prepared_runtime::CoherePreparedRuntime,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    plan_cohere_decoder_state_with_components(input, prepared.metadata, &prepared.tokenizer)
}

fn plan_cohere_decoder_state_with_components(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
    metadata: CohereTranscribeExecutionMetadata,
    tokenizer: &super::tokenizer::CohereTranscribeTokenizer,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    let family = "cohere-transcribe";
    let base_prompt = super::prompt::build_cohere_transcribe_decode_prompt(
        tokenizer,
        metadata.decoder_start_token_id,
        input.request_options.language.as_deref(),
        input.request_options,
    )
    .map_err(
        |error| GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
            family,
            reason: error.to_string(),
        },
    )?;
    let base_prompt_positions = base_prompt.token_ids.len();
    let logical_prompt_positions = super::prompt::build_cohere_initial_prompt_token_ids(
        base_prompt.token_ids,
        input.request_options,
        metadata,
    )
    .map_err(
        |error| GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
            family,
            reason: error.to_string(),
        },
    )?
    .len();
    let stable_prompt_positions = super::prompt::cohere_stable_prompt_positions(
        logical_prompt_positions,
        base_prompt_positions,
        input.envelope.max_prompt_tokens(),
        metadata.decoder_max_context,
    )
    .map_err(
        |error| GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
            family,
            reason: error.to_string(),
        },
    )?;
    let topology = CohereDecoderStateTopology::for_envelope(
        metadata,
        input.envelope,
        logical_prompt_positions,
        stable_prompt_positions,
    )
    .map_err(|source| GgmlAsrDecoderStatePlanningError::Topology { family, source })?;
    crate::capacity::topology::DecoderStatePlan::build(&topology, input.invocation, input.envelope)
        .map_err(|source| GgmlAsrDecoderStatePlanningError::Topology { family, source })
}

fn cohere_resident_kv_bytes(
    metadata: &CohereTranscribeExecutionMetadata,
    positions: usize,
    sequences: usize,
    bytes_per_value: usize,
) -> Result<StateBytes, TopologyError> {
    let resident = metadata
        .decoder_layers
        .checked_mul(2)
        .and_then(|value| value.checked_mul(metadata.decoder_d_model))
        .and_then(|value| value.checked_mul(positions))
        .and_then(|value| value.checked_mul(sequences))
        .and_then(|value| value.checked_mul(bytes_per_value))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(TopologyError::ArithmeticOverflow {
            operation: "cohere resident KV bytes",
        })?;
    Ok(StateBytes { host: 0, resident })
}

pub(crate) fn cohere_encoder_frame_count_for_samples(
    metadata: CohereTranscribeExecutionMetadata,
    samples: usize,
    sample_rate_hz: u32,
) -> Result<usize, TopologyError> {
    if sample_rate_hz != metadata.sample_rate_hz {
        return Err(TopologyError::UnsupportedSampleRate {
            expected_hz: metadata.sample_rate_hz,
            actual_hz: sample_rate_hz,
        });
    }
    let mel_frames = StftFramer::output_frame_count_for(
        metadata.n_fft,
        metadata.hop_length,
        PadMode::ZeroCenter,
        samples,
    )
    .map_err(|error| TopologyError::Unavailable {
        reason: format!("cohere STFT shape is invalid: {error}"),
    })?
    .checked_sub(1)
    .filter(|&frames| frames > 0)
    .ok_or(TopologyError::Unavailable {
        reason: "cohere audio is too short after the final mel-row drop".to_string(),
    })?;
    super::encoder_graph::predicted_encoder_time_frames(mel_frames).map_err(|error| {
        TopologyError::Unavailable {
            reason: format!("cohere encoder shape is invalid: {error}"),
        }
    })
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CohereDecoderStateTopology {
    metadata: CohereTranscribeExecutionMetadata,
    cross_position_cap: usize,
    logical_prompt_positions: usize,
    stable_prompt_positions: usize,
}

impl CohereDecoderStateTopology {
    /// Bind the topology to the product/session's largest legal invocation.
    /// Cohere has no pack-carried encoder positional ceiling, so the exact
    /// frontend count for that envelope is the only honest finite cross-KV
    /// hard cap. A later attempt to plan a larger envelope fails closed.
    pub(crate) fn for_envelope(
        metadata: CohereTranscribeExecutionMetadata,
        envelope: InvocationEnvelope,
        logical_prompt_positions: usize,
        stable_prompt_positions: usize,
    ) -> Result<Self, TopologyError> {
        let cross_position_cap = cohere_encoder_frame_count_for_samples(
            metadata,
            envelope.max_samples(),
            envelope.sample_rate_hz().get(),
        )?;
        Ok(Self {
            metadata,
            cross_position_cap,
            logical_prompt_positions,
            stable_prompt_positions,
        })
    }

    fn demands_for(
        &self,
        invocation: InvocationShapeInput,
        prompt_positions: usize,
    ) -> Result<Vec<StateDemand>, TopologyError> {
        let sequences = invocation.sequences().get() as usize;
        let cross_frames = cohere_encoder_frame_count_for_samples(
            self.metadata,
            invocation.samples(),
            invocation.sample_rate_hz().get(),
        )?;
        let decode_budget = super::decode_budget::cohere_decode_budget(
            prompt_positions,
            cross_frames,
            self.metadata.decoder_max_context,
        )
        .map_err(|error| TopologyError::Unavailable {
            reason: error.to_string(),
        })?;
        Ok(vec![
            StateDemand::new(
                SELF_KV_STATE_ID,
                StateKind::SelfAttentionKv,
                decode_budget.self_kv_positions,
                self.metadata.decoder_max_context,
                cohere_resident_kv_bytes(
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
                self.cross_position_cap,
                cohere_resident_kv_bytes(
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

impl DecoderStateTopology for CohereDecoderStateTopology {
    fn demands(
        &self,
        scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
    ) -> Result<Vec<StateDemand>, TopologyError> {
        match scope {
            DecoderStateDemandScope::ExactInvocation(invocation) => {
                self.demands_for(invocation, self.logical_prompt_positions)
            }
            DecoderStateDemandScope::StableEnvelope(envelope) => {
                self.demands_for(envelope.maximum_invocation(), self.stable_prompt_positions)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::capacity::topology::{DecoderStatePlan, InvocationEnvelope, StateKind};

    /// Real-checkpoint-shaped metadata (the same values `runtime_contract`'s
    /// `base_metadata` fixture parses: 8-layer MHA decoder at head_dim 128,
    /// 1024-position context, 16kHz/160-hop mel).
    fn reference_metadata() -> CohereTranscribeExecutionMetadata {
        CohereTranscribeExecutionMetadata {
            vocab_size: 50_000,
            encoder_layers: 48,
            encoder_d_model: 1280,
            encoder_heads: 8,
            encoder_head_dim: 160,
            encoder_ffn_dim: 5120,
            encoder_conv_kernel: 9,
            decoder_layers: 8,
            decoder_d_model: 1024,
            decoder_heads: 8,
            decoder_head_dim: 128,
            decoder_ffn_dim: 4096,
            decoder_max_context: 1024,
            decoder_start_token_id: 13_764,
            sample_rate_hz: 16_000,
            n_mels: 128,
            n_fft: 512,
            hop_length: 160,
            win_length: 400,
        }
    }

    #[test]
    fn resident_byte_oracle_matches_runtime_tensor_shapes() {
        let metadata = reference_metadata();
        // layers x 2 x d_model f32 values per cross frame:
        // 8 * 2 * 1024 * 4 B = 64 KiB.
        assert_eq!(
            cohere_resident_kv_bytes(&metadata, 1, 1, std::mem::size_of::<f32>()).unwrap(),
            StateBytes {
                host: 0,
                resident: 8 * 2 * 1024 * 4,
            }
        );
        // 8 layers x 2 (K+V) x 8 heads x 1024 positions x 128 head_dim x 2 B
        // (f16) = 32 MiB.
        assert_eq!(
            cohere_resident_kv_bytes(
                &metadata,
                metadata.decoder_max_context,
                1,
                std::mem::size_of::<u16>(),
            )
            .unwrap(),
            StateBytes {
                host: 0,
                resident: 8 * 2 * 8 * 1024 * 128 * 2,
            }
        );
    }

    #[test]
    fn frontend_count_oracle_pins_final_row_drop_boundary() {
        let metadata = reference_metadata();
        assert!(cohere_encoder_frame_count_for_samples(metadata, 159, 16_000).is_err());
        assert_eq!(
            cohere_encoder_frame_count_for_samples(metadata, 160, 16_000).unwrap(),
            1
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
            cohere_encoder_frame_count_for_samples(metadata, envelope.max_samples(), 16_000)
                .unwrap();
        // 30s -> 3000 post-drop mel frames -> three k3/s2/p1
        // subsampling stages: 1500 -> 750 -> 375 encoder frames.
        assert_eq!(expected_cross, 375);
        let topology = CohereDecoderStateTopology::for_envelope(metadata, envelope, 9, 9).unwrap();
        let plan = DecoderStatePlan::for_envelope(&topology, envelope).unwrap();
        assert_eq!(plan.reserve_positions_by_id(SELF_KV_STATE_ID), Some(520));
        assert_eq!(
            plan.reserve_positions(StateKind::CrossAttentionKv),
            Some(expected_cross)
        );
    }

    #[test]
    fn duration_boundaries_keep_self_and_cross_schedules_independent() {
        let rate = NonZeroU32::new(16_000).unwrap();
        for (seconds, expected_cross, expected_self) in [
            (1, 13, 72),
            (30, 375, 520),
            (60, 750, 520),
            (300, 3_750, 520),
        ] {
            let envelope = InvocationEnvelope::new(rate, seconds * 16_000).unwrap();
            let topology =
                CohereDecoderStateTopology::for_envelope(reference_metadata(), envelope, 9, 9)
                    .unwrap();
            let plan = DecoderStatePlan::for_envelope(&topology, envelope).unwrap();
            assert_eq!(
                plan.reserve_positions_by_id(SELF_KV_STATE_ID),
                Some(expected_self)
            );
            assert_eq!(
                plan.reserve_positions_by_id(CROSS_KV_STATE_ID),
                Some(expected_cross)
            );
            assert_ne!(expected_self, expected_self + expected_cross);
        }
    }

    #[test]
    fn small_decoder_context_fails_before_a_physical_state_is_declared() {
        let mut metadata = reference_metadata();
        metadata.decoder_max_context = 9;
        let envelope = InvocationEnvelope::new(NonZeroU32::new(16_000).unwrap(), 16_000).unwrap();
        let topology = CohereDecoderStateTopology::for_envelope(metadata, envelope, 9, 9).unwrap();
        assert!(matches!(
            DecoderStatePlan::for_envelope(&topology, envelope),
            Err(TopologyError::Unavailable { .. })
        ));
    }

    #[test]
    fn product_envelope_is_a_finite_cross_position_cap() {
        let metadata = reference_metadata();
        let product_envelope = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(30_000).unwrap(),
        )
        .unwrap();
        let topology =
            CohereDecoderStateTopology::for_envelope(metadata, product_envelope, 9, 9).unwrap();
        let oversized = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(60_000).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            DecoderStatePlan::for_envelope(&topology, oversized),
            Err(TopologyError::PositionCapExceeded {
                state: "cohere.decoder.cross_kv",
                ..
            })
        ));
    }
}
