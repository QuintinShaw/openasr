//! MOSS-TD decoder-state topology.
//!
//! The product window (30s target, 60s maximum) is a semantic envelope, not a
//! quantity inferred from available VRAM. This module proves the exact
//! logical and session-stable KV spans from the same integer frontend,
//! time-marker, and generation-budget counters used by execution. The 8192
//! family limit validates those spans but is never substituted as capacity.

use crate::capacity::KvGeometry;
use crate::capacity::topology::{
    DecoderStateDemandScope, DecoderStateTopology, InvocationEnvelope, InvocationShapeInput,
    PositionBoundProof, StateDemand, StateKind, TopologyError,
    causal_prefix_positions_with_context_cap,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrDecoderStatePlanningError, GgmlAsrDecoderStatePlanningInput,
};
use crate::nn::decoder::LlmKvCacheSpec;

use super::decode_budget::moss_td_generated_token_budget;
use super::executor::{
    CHUNK_SAMPLES, HOP_LENGTH, SAMPLE_RATE_HZ, WHISPER_ENCODER_CONV_STRIDE,
    moss_td_aligned_frame_count, moss_td_chunk_keep_frames, moss_td_chunk_token_length,
};
use super::runtime_contract::MossTdEncoderMetadata;
use super::runtime_contract::{MossTdDecoderMetadata, moss_td_kv_cache_positions};

pub(crate) const MOSS_TD_SELF_KV_STATE_ID: &str = "moss-td.decoder.self_kv";
pub(crate) const MOSS_TD_DECODER_STATE_STREAMS:
    &[crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract] = &[
    crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract::new(
        MOSS_TD_SELF_KV_STATE_ID,
        StateKind::SelfAttentionKv,
    ),
];

pub(crate) fn plan_moss_td_decoder_state(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    let family = "moss-transcribe-diarize";
    let metadata = input.preflight.metadata.as_ref();
    let encoder = super::runtime_contract::parse_encoder_metadata(metadata).map_err(|error| {
        GgmlAsrDecoderStatePlanningError::MetadataUnavailable {
            family,
            reason: error.to_string(),
        }
    })?;
    let decoder = super::runtime_contract::parse_decoder_metadata(metadata).map_err(|error| {
        GgmlAsrDecoderStatePlanningError::MetadataUnavailable {
            family,
            reason: error.to_string(),
        }
    })?;
    let adaptor = super::runtime_contract::parse_adaptor_metadata(metadata).map_err(|error| {
        GgmlAsrDecoderStatePlanningError::MetadataUnavailable {
            family,
            reason: error.to_string(),
        }
    })?;
    let tokenizer =
        super::tokenizer::MossTdTokenizer::from_gguf_metadata(metadata).map_err(|error| {
            GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
                family,
                reason: error.to_string(),
            }
        })?;
    let fixed_prompt_tokens = super::decode_prompt::moss_td_fixed_prompt_token_count(&tokenizer)
        .map_err(
            |error| GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
                family,
                reason: error.to_string(),
            },
        )?;
    let geometry = moss_td_kv_geometry(&decoder);
    let spec = crate::models::qwen::resolve_qwen_family_production_kv_cache_policy(
        input.backend,
        geometry.head_dim,
    )
    .to_spec();
    let topology = MossTdDecoderStateTopology::new(
        encoder,
        decoder,
        adaptor.merge_size,
        fixed_prompt_tokens,
        spec,
    )
    .map_err(|source| GgmlAsrDecoderStatePlanningError::Topology { family, source })?;
    crate::capacity::topology::DecoderStatePlan::build(&topology, input.invocation, input.envelope)
        .map_err(|source| GgmlAsrDecoderStatePlanningError::Topology { family, source })
}

/// Shipped-pack regression for the fixed ChatML wrapper around the audio span:
/// the 14-token
/// `<|im_start|>system...<|im_start|>user\n` prefix + the `audio_start`
/// delimiter + the `audio_end` delimiter + the 70-token instruction/ChatML
/// suffix. Measured token-for-token against the real golden fixture
/// (`tmp/moss-td/golden/jfk.json`'s `prompt_input_ids`: 227 tokens = 14
/// prefix + 1 audio-start + 141 audio span + 1 audio-end + 70 suffix, where
/// the 141-token span at 11s is 138 pad tokens + 3 marker digits -- the
/// markers are modeled separately by `marker_every_seconds`). Production does
/// not consume this number: it invokes
/// `decode_prompt::moss_td_fixed_prompt_token_count` on the loaded tokenizer,
/// while the real-pack prompt regression pins that result to 86 for the
/// shipped artifact.
#[cfg(test)]
pub(crate) const MOSS_TD_SHIPPED_FIXED_PROMPT_TOKENS: usize = 86;

/// The decoder KV geometry the loaded pack advertises.
pub(crate) fn moss_td_kv_geometry(decoder: &MossTdDecoderMetadata) -> KvGeometry {
    KvGeometry {
        n_layers: decoder.n_layers,
        kv_heads: decoder.n_kv_heads,
        head_dim: decoder.head_dim,
    }
}

/// Session-stable MOSS causal-prefix topology. Every term is the same integer
/// shape arithmetic the real frontend/prompt/decode path uses; the 8192
/// family cap remains validation only.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MossTdDecoderStateTopology {
    encoder: MossTdEncoderMetadata,
    decoder: MossTdDecoderMetadata,
    merge_size: usize,
    fixed_prompt_tokens: usize,
    kv_spec: LlmKvCacheSpec,
}

impl MossTdDecoderStateTopology {
    pub(crate) fn new(
        encoder: MossTdEncoderMetadata,
        decoder: MossTdDecoderMetadata,
        merge_size: usize,
        fixed_prompt_tokens: usize,
        kv_spec: LlmKvCacheSpec,
    ) -> Result<Self, TopologyError> {
        if merge_size == 0 {
            return Err(TopologyError::Unavailable {
                reason: "moss adaptor merge_size is zero".to_string(),
            });
        }
        Ok(Self {
            encoder,
            decoder,
            merge_size,
            fixed_prompt_tokens,
            kv_spec,
        })
    }

    fn audio_token_count(&self, sample_count: usize) -> Result<usize, TopologyError> {
        let token_stride = HOP_LENGTH
            .checked_mul(WHISPER_ENCODER_CONV_STRIDE)
            .and_then(|value| value.checked_mul(self.merge_size))
            .ok_or(TopologyError::ArithmeticOverflow {
                operation: "moss audio token stride",
            })?;
        let mut total_kept_frames = 0usize;
        let mut remaining = sample_count;
        while remaining > 0 {
            let chunk_samples = remaining.min(CHUNK_SAMPLES);
            let token_length = moss_td_chunk_token_length(chunk_samples, token_stride);
            let kept_frames = moss_td_chunk_keep_frames(
                token_length,
                self.merge_size,
                self.encoder.max_source_positions,
            );
            total_kept_frames = total_kept_frames.checked_add(kept_frames).ok_or(
                TopologyError::ArithmeticOverflow {
                    operation: "moss kept encoder frames",
                },
            )?;
            remaining -= chunk_samples;
        }
        Ok(moss_td_aligned_frame_count(total_kept_frames, self.merge_size) / self.merge_size)
    }

    fn marker_tokens(audio_token_count: usize) -> Result<usize, TopologyError> {
        let marker_seconds = super::decode_prompt::moss_td_time_marker_seconds(audio_token_count)
            .map_err(|error| TopologyError::Unavailable {
            reason: format!("moss marker topology is unavailable: {error}"),
        })?;
        let mut tokens = 0usize;
        for mut seconds in marker_seconds {
            let mut digits = 1usize;
            while seconds >= 10 {
                seconds /= 10;
                digits = digits
                    .checked_add(1)
                    .ok_or(TopologyError::ArithmeticOverflow {
                        operation: "moss marker digit count",
                    })?;
            }
            tokens = tokens
                .checked_add(digits)
                .ok_or(TopologyError::ArithmeticOverflow {
                    operation: "moss marker token sum",
                })?;
        }
        Ok(tokens)
    }

    fn positions(&self, invocation: InvocationShapeInput) -> Result<usize, TopologyError> {
        if invocation.sample_rate_hz().get() as usize != SAMPLE_RATE_HZ {
            return Err(TopologyError::UnsupportedSampleRate {
                expected_hz: SAMPLE_RATE_HZ as u32,
                actual_hz: invocation.sample_rate_hz().get(),
            });
        }
        let max_samples = SAMPLE_RATE_HZ
            .checked_mul(crate::arch::MOSS_TD_MAX_INVOCATION_SECONDS as usize)
            .ok_or(TopologyError::ArithmeticOverflow {
                operation: "moss product invocation sample ceiling",
            })?;
        if invocation.samples() > max_samples {
            return Err(TopologyError::InvocationSampleLimitExceeded {
                required_samples: invocation.samples(),
                max_samples,
            });
        }
        let audio_tokens = self.audio_token_count(invocation.samples())?;
        let marker_tokens = Self::marker_tokens(audio_tokens)?;
        let prompt_tokens = self
            .fixed_prompt_tokens
            .checked_add(audio_tokens)
            .and_then(|value| value.checked_add(marker_tokens))
            .ok_or(TopologyError::ArithmeticOverflow {
                operation: "moss prompt positions",
            })?;
        let context_cap = moss_td_kv_cache_positions(self.decoder.max_positions);
        let generated = moss_td_generated_token_budget(
            invocation.samples(),
            SAMPLE_RATE_HZ,
            prompt_tokens,
            context_cap,
        )
        .map_err(|error| TopologyError::Unavailable {
            reason: format!("moss generation budget is unavailable: {error}"),
        })?;
        causal_prefix_positions_with_context_cap(
            MOSS_TD_SELF_KV_STATE_ID,
            prompt_tokens,
            generated,
            context_cap,
        )
    }
}

impl DecoderStateTopology for MossTdDecoderStateTopology {
    fn demands(
        &self,
        scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
    ) -> Result<Vec<StateDemand>, TopologyError> {
        let invocation = match scope {
            DecoderStateDemandScope::ExactInvocation(invocation) => invocation,
            DecoderStateDemandScope::StableEnvelope(envelope) => envelope.maximum_invocation(),
        };
        let positions = self.positions(invocation)?;
        Ok(vec![StateDemand::from_llm_kv_geometry(
            MOSS_TD_SELF_KV_STATE_ID,
            StateKind::SelfAttentionKv,
            positions,
            moss_td_kv_cache_positions(self.decoder.max_positions),
            moss_td_kv_geometry(&self.decoder),
            self.kv_spec,
            invocation.sequences().get() as usize,
            PositionBoundProof::Exact,
        )?])
    }
}

/// Decoder metadata matching the real shipped checkpoint -- the same values
/// `runtime_contract::tests::full_metadata` parses (28L Qwen3 decoder, 8 KV
/// heads, head_dim 128, raw RoPE ceiling 131072). Shared by this module's
/// regression anchors and `executor.rs`'s pin tests so every capacity
/// arithmetic check argues from one checkpoint-faithful geometry.
#[cfg(test)]
pub(crate) fn shipped_pack_decoder_fixture() -> MossTdDecoderMetadata {
    MossTdDecoderMetadata {
        n_layers: 28,
        d_model: 1024,
        ffn_dim: 3072,
        n_heads: 16,
        n_kv_heads: 8,
        head_dim: 128,
        vocab_size: 151_936,
        max_positions: 131_072,
        audio_start_token_id: 151_669,
        audio_end_token_id: 151_670,
        audio_pad_token_id: 151_671,
    }
}

/// The real shipped pack's `moss_td.adaptor.merge_size`.
#[cfg(test)]
pub(crate) const SHIPPED_MERGE_SIZE: usize = 4;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacity::kv_bytes_per_position;
    use crate::capacity::topology::{
        DecoderStatePlan, InvocationEnvelope, InvocationShapeInput, StateKind,
    };
    use crate::nn::decoder::LlmKvCacheSpec;
    use std::num::NonZeroU32;

    use super::super::decode_prompt::AUDIO_TOKENS_PER_SECOND;
    use super::super::executor::AUDIO_TOKENS_PER_SECOND_FOR_LIMIT;
    use super::super::runtime_contract::MOSS_TD_MAX_KV_CACHE_POSITIONS;

    fn shipped_pack_decoder() -> MossTdDecoderMetadata {
        shipped_pack_decoder_fixture()
    }

    fn shipped_encoder() -> MossTdEncoderMetadata {
        MossTdEncoderMetadata {
            n_layers: 24,
            d_model: 1024,
            n_heads: 16,
            ffn_dim: 4096,
            n_mels: 80,
            max_source_positions: 1500,
        }
    }

    #[test]
    fn topology_separates_current_30s_demand_from_stable_60s_reserve() {
        let topology = MossTdDecoderStateTopology::new(
            shipped_encoder(),
            shipped_pack_decoder(),
            SHIPPED_MERGE_SIZE,
            MOSS_TD_SHIPPED_FIXED_PROMPT_TOKENS,
            LlmKvCacheSpec::DEFAULT,
        )
        .unwrap();
        let rate = NonZeroU32::new(SAMPLE_RATE_HZ as u32).unwrap();
        let logical = InvocationShapeInput::new(
            rate,
            crate::arch::MOSS_TD_TARGET_INVOCATION_SECONDS as usize * SAMPLE_RATE_HZ,
        )
        .unwrap();
        let envelope = InvocationEnvelope::from_milliseconds(
            rate,
            NonZeroU32::new(crate::arch::MOSS_TD_MAX_INVOCATION_SECONDS * 1_000).unwrap(),
        )
        .unwrap();
        let plan = DecoderStatePlan::build(&topology, logical, envelope).unwrap();
        assert_eq!(
            plan.logical_positions(StateKind::SelfAttentionKv),
            Some(1_289)
        );
        assert_eq!(
            plan.reserve_positions(StateKind::SelfAttentionKv),
            Some(2_366)
        );
        assert_eq!(plan.reserve_bytes().resident, 2_366 * 112 * 1024);
    }

    #[test]
    fn one_and_sixty_second_boundaries_include_marker_digits_exactly() {
        let topology = MossTdDecoderStateTopology::new(
            shipped_encoder(),
            shipped_pack_decoder(),
            SHIPPED_MERGE_SIZE,
            MOSS_TD_SHIPPED_FIXED_PROMPT_TOKENS,
            LlmKvCacheSpec::DEFAULT,
        )
        .unwrap();
        let rate = NonZeroU32::new(SAMPLE_RATE_HZ as u32).unwrap();
        for (seconds, expected_positions) in [(1, 249), (30, 1_289), (60, 2_366)] {
            let envelope = InvocationEnvelope::new(rate, seconds * SAMPLE_RATE_HZ).unwrap();
            let plan = DecoderStatePlan::for_envelope(&topology, envelope).unwrap();
            assert_eq!(
                plan.reserve_positions_by_id(MOSS_TD_SELF_KV_STATE_ID),
                Some(expected_positions),
                "unexpected MOSS capacity at {seconds}s"
            );
        }
    }

    #[test]
    fn smaller_legal_pack_context_uses_the_runtime_context_clamp() {
        let mut decoder = shipped_pack_decoder();
        decoder.max_positions = 1_024;
        let topology = MossTdDecoderStateTopology::new(
            shipped_encoder(),
            decoder,
            SHIPPED_MERGE_SIZE,
            MOSS_TD_SHIPPED_FIXED_PROMPT_TOKENS,
            LlmKvCacheSpec::DEFAULT,
        )
        .unwrap();
        let envelope = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(SAMPLE_RATE_HZ as u32).unwrap(),
            NonZeroU32::new(30_000).unwrap(),
        )
        .unwrap();
        let plan = DecoderStatePlan::for_envelope(&topology, envelope)
            .expect("runtime-valid context clamp must also be planner-valid");
        assert_eq!(
            plan.reserve_positions(StateKind::SelfAttentionKv),
            Some(1_023)
        );
    }

    #[test]
    fn topology_rejects_direct_audio_beyond_the_product_envelope() {
        let topology = MossTdDecoderStateTopology::new(
            shipped_encoder(),
            shipped_pack_decoder(),
            SHIPPED_MERGE_SIZE,
            MOSS_TD_SHIPPED_FIXED_PROMPT_TOKENS,
            LlmKvCacheSpec::DEFAULT,
        )
        .unwrap();
        let rate = NonZeroU32::new(SAMPLE_RATE_HZ as u32).unwrap();
        let max_samples = crate::arch::MOSS_TD_MAX_INVOCATION_SECONDS as usize * SAMPLE_RATE_HZ;
        let envelope = InvocationEnvelope::new(rate, max_samples + 1).unwrap();
        let error = DecoderStatePlan::for_envelope(&topology, envelope)
            .expect_err("MOSS direct invocation must not bypass the 60-second product envelope");
        assert!(matches!(
            error,
            TopologyError::InvocationSampleLimitExceeded {
                required_samples,
                max_samples: rejected_max,
            } if required_samples == max_samples + 1 && rejected_max == max_samples
        ));

        let five_minutes = InvocationEnvelope::new(rate, 300 * SAMPLE_RATE_HZ).unwrap();
        assert!(matches!(
            DecoderStatePlan::for_envelope(&topology, five_minutes),
            Err(TopologyError::InvocationSampleLimitExceeded { .. })
        ));
    }

    /// Worst-case KV bytes per position, split by copy (the figure the old
    /// `~30 GB` comment got wrong by counting only the host f32 copy).
    #[test]
    fn kv_bytes_per_position_matches_the_two_copy_reality() {
        let geometry = moss_td_kv_geometry(&shipped_pack_decoder());
        // DEFAULT (host f32 + resident f16): 448 rows/position
        // (28 layers * 2 * 8 kv-heads) at 512 B + 256 B.
        let default = kv_bytes_per_position(&geometry, LlmKvCacheSpec::DEFAULT).expect("default");
        assert_eq!(default.host, 224 * 1024);
        assert_eq!(default.resident, 112 * 1024);
        assert_eq!(default.total(), 336 * 1024);
        // Q8_0 (both copies q8_0): 136 B/row -> 2.8x under DEFAULT, not ~4x.
        let q8_0 = kv_bytes_per_position(&geometry, LlmKvCacheSpec::Q8_0).expect("q8_0");
        assert_eq!(q8_0.host, 28 * 2 * 8 * 136); // 60928 = 59.5 KiB
        assert_eq!(q8_0.resident, 28 * 2 * 8 * 136);
        assert_eq!(q8_0.total(), 2 * 28 * 2 * 8 * 136); // 119 KiB
    }

    /// The family cap is a validation ceiling, not the allocation request.
    /// Pin the concrete reduction for the product's 60s session envelope.
    #[test]
    fn stable_envelope_is_below_the_hard_cap_without_allocating_the_cap() {
        assert_eq!(MOSS_TD_MAX_KV_CACHE_POSITIONS, 8_192);
        assert_eq!(2_366 * 112 * 1024, 271_351_808);
        assert_eq!(8_192 * 112 * 1024, 939_524_096);
        const {
            assert!(2_366 < MOSS_TD_MAX_KV_CACHE_POSITIONS);
        }
    }

    /// Drift guard: the actual frontend stride and both runtime marker-rate
    /// declarations are one exact rational architectural fact.
    #[test]
    fn frontend_stride_agrees_with_the_family_constants() {
        let samples_per_audio_token = HOP_LENGTH * WHISPER_ENCODER_CONV_STRIDE * SHIPPED_MERGE_SIZE;
        // 16_000 / 1_280 = 12.5 exactly, proven without floating point.
        assert_eq!(SAMPLE_RATE_HZ * 2, samples_per_audio_token * 25);
        assert_eq!(AUDIO_TOKENS_PER_SECOND, 12.5);
        assert_eq!(AUDIO_TOKENS_PER_SECOND_FOR_LIMIT, 12.5);
        // And the stride product reproduces the chunk's audio-token count.
        assert_eq!(
            moss_td_chunk_token_length(CHUNK_SAMPLES, samples_per_audio_token),
            375
        );
    }
}
