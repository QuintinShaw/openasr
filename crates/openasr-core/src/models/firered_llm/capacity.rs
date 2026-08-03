//! FireRed-LLM decoder-state topology. Its exact position bound is assembled
//! from pack metadata, the tokenized prompt wrapper, the frontend shape
//! oracle, and the runtime generation limit.
//!
//! Reuses, never re-derives:
//! - KV geometry comes straight off the parsed pack (`firered_llm.llm.*` are
//!   required metadata keys).
//! - The audio-token count walks the exact runtime pipeline arithmetic:
//!   snip-edges fbank framing (`firered_aed::frontend`'s constants), the
//!   `2x Conv2d(k3,s2)` stem
//!   ([`crate::models::firered_aed::encoder_graph::predicted_encoder_time_frames`],
//!   the same function the real encoder path calls), and the adapter's
//!   pack-carried `downsample_rate` frame stacking.
//! - The generation budget reuses the executor's own constant, so planning
//!   and allocation share one `prompt + generation` contract.

use crate::capacity::KvGeometry;
use crate::capacity::topology::{
    DecoderStateDemandScope, DecoderStateTopology, InvocationEnvelope, InvocationShapeInput,
    PositionBoundProof, StateDemand, StateKind, TopologyError,
    causal_prefix_positions_with_context_cap,
};
use crate::models::firered_aed::encoder_graph::predicted_encoder_time_frames;
use crate::models::firered_aed::frontend::{
    FRAME_LENGTH_SAMPLES, FRAME_SHIFT_SAMPLES, SAMPLE_RATE_HZ,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrDecoderStatePlanningError, GgmlAsrDecoderStatePlanningInput,
};
use crate::nn::decoder::LlmKvCacheSpec;

use super::executor::FIRERED_LLM_MAX_GENERATED_TOKENS;
use super::runtime_contract::{FireRedLlmAdapterMetadata, FireRedLlmDecoderMetadata};

pub(crate) const FIRERED_LLM_SELF_KV_STATE_ID: &str = "firered-llm.decoder.self_kv";
pub(crate) const FIRERED_LLM_DECODER_STATE_STREAMS:
    &[crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract] = &[
    crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract::new(
        FIRERED_LLM_SELF_KV_STATE_ID,
        StateKind::SelfAttentionKv,
    ),
];

pub(crate) fn plan_firered_llm_decoder_state(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    let family = "firered-llm";
    let metadata = input.preflight.metadata.as_ref();
    let decoder =
        super::runtime_contract::parse_firered_llm_decoder_metadata(metadata).map_err(|error| {
            GgmlAsrDecoderStatePlanningError::MetadataUnavailable {
                family,
                reason: error.to_string(),
            }
        })?;
    let adapter =
        super::runtime_contract::parse_firered_llm_adapter_metadata(metadata).map_err(|error| {
            GgmlAsrDecoderStatePlanningError::MetadataUnavailable {
                family,
                reason: error.to_string(),
            }
        })?;
    let tokenizer =
        super::tokenizer::FireRedLlmTokenizer::from_gguf_metadata(metadata).map_err(|error| {
            GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
                family,
                reason: error.to_string(),
            }
        })?;
    let prompt =
        super::decode_prompt::build_firered_llm_decode_prompt(&tokenizer, 1).map_err(|error| {
            GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
                family,
                reason: error.to_string(),
            }
        })?;
    let fixed_prompt_tokens = prompt.token_ids.len().checked_sub(1).ok_or_else(|| {
        GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
            family,
            reason: "dummy speech span was absent from the tokenized prompt".to_string(),
        }
    })?;
    let geometry = firered_llm_kv_geometry(&decoder);
    let spec = crate::models::qwen::resolve_qwen_family_production_kv_cache_policy(
        input.backend,
        geometry.head_dim,
    )
    .to_spec();
    crate::capacity::topology::DecoderStatePlan::build(
        &FireRedLlmDecoderStateTopology::new(decoder, adapter, fixed_prompt_tokens, spec),
        input.invocation,
        input.envelope,
    )
    .map_err(|source| GgmlAsrDecoderStatePlanningError::Topology { family, source })
}

const FIRERED_LLM_MAX_INPUT_SAMPLES: usize = SAMPLE_RATE_HZ as usize * 40;

/// The decoder KV geometry the loaded pack advertises.
pub(crate) fn firered_llm_kv_geometry(decoder: &FireRedLlmDecoderMetadata) -> KvGeometry {
    KvGeometry {
        n_layers: decoder.n_layers,
        kv_heads: decoder.n_kv_heads,
        head_dim: decoder.head_dim,
    }
}

pub(crate) fn firered_llm_speech_token_count_for_samples(
    adapter: &FireRedLlmAdapterMetadata,
    samples: usize,
) -> Result<usize, TopologyError> {
    if adapter.downsample_rate == 0 {
        return Err(TopologyError::DivisionByZero);
    }
    let mel_frames = samples
        .checked_sub(FRAME_LENGTH_SAMPLES)
        .map(|tail| tail / FRAME_SHIFT_SAMPLES + 1)
        .unwrap_or(0);
    if mel_frames == 0 {
        return Err(TopologyError::Unavailable {
            reason: "firered-llm audio is too short to produce one fbank frame".to_string(),
        });
    }
    predicted_encoder_time_frames(mel_frames)
        .map(|frames| frames / adapter.downsample_rate)
        .map_err(|error| TopologyError::Unavailable {
            reason: format!("firered-llm encoder shape is invalid: {error}"),
        })
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FireRedLlmDecoderStateTopology {
    decoder: FireRedLlmDecoderMetadata,
    adapter: FireRedLlmAdapterMetadata,
    fixed_prompt_tokens: usize,
    kv_spec: LlmKvCacheSpec,
}

impl FireRedLlmDecoderStateTopology {
    pub(crate) const fn new(
        decoder: FireRedLlmDecoderMetadata,
        adapter: FireRedLlmAdapterMetadata,
        fixed_prompt_tokens: usize,
        kv_spec: LlmKvCacheSpec,
    ) -> Self {
        Self {
            decoder,
            adapter,
            fixed_prompt_tokens,
            kv_spec,
        }
    }
}

impl DecoderStateTopology for FireRedLlmDecoderStateTopology {
    fn demands(
        &self,
        scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
    ) -> Result<Vec<StateDemand>, TopologyError> {
        let invocation = match scope {
            DecoderStateDemandScope::ExactInvocation(invocation) => invocation,
            DecoderStateDemandScope::StableEnvelope(envelope) => envelope.maximum_invocation(),
        };
        if invocation.sample_rate_hz().get() != SAMPLE_RATE_HZ {
            return Err(TopologyError::UnsupportedSampleRate {
                expected_hz: SAMPLE_RATE_HZ,
                actual_hz: invocation.sample_rate_hz().get(),
            });
        }
        if invocation.samples() > FIRERED_LLM_MAX_INPUT_SAMPLES {
            return Err(TopologyError::InvocationSampleLimitExceeded {
                required_samples: invocation.samples(),
                max_samples: FIRERED_LLM_MAX_INPUT_SAMPLES,
            });
        }
        let speech_tokens =
            firered_llm_speech_token_count_for_samples(&self.adapter, invocation.samples())?;
        let prompt_positions = self.fixed_prompt_tokens.checked_add(speech_tokens).ok_or(
            TopologyError::ArithmeticOverflow {
                operation: "firered-llm prompt positions",
            },
        )?;
        let positions = causal_prefix_positions_with_context_cap(
            FIRERED_LLM_SELF_KV_STATE_ID,
            prompt_positions,
            FIRERED_LLM_MAX_GENERATED_TOKENS,
            self.decoder.max_positions,
        )?;
        Ok(vec![StateDemand::from_llm_kv_geometry(
            FIRERED_LLM_SELF_KV_STATE_ID,
            StateKind::SelfAttentionKv,
            positions,
            self.decoder.max_positions,
            firered_llm_kv_geometry(&self.decoder),
            self.kv_spec,
            invocation.sequences().get() as usize,
            PositionBoundProof::Exact,
        )?])
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::capacity::kv_bytes_per_position;
    use crate::capacity::topology::{DecoderStatePlan, InvocationEnvelope, StateKind};
    use crate::nn::decoder::LlmKvCacheSpec;

    /// Real-checkpoint-shaped metadata (the Qwen2-7B decoder: 28 layers, GQA
    /// with 4 KV heads, head_dim 128, 32768-position context -- the same
    /// values `runtime_contract`'s `full_metadata` fixture parses).
    fn reference_decoder() -> FireRedLlmDecoderMetadata {
        FireRedLlmDecoderMetadata {
            n_layers: 28,
            d_model: 3584,
            n_heads: 28,
            n_kv_heads: 4,
            head_dim: 128,
            ffn_dim: 18944,
            vocab_size: 152_064,
            max_positions: 32_768,
            chatml_im_start_token_id: 151_644,
            chatml_im_end_token_id: 151_645,
            endoftext_token_id: 151_643,
            speech_token_id: 151_646,
        }
    }

    fn reference_adapter() -> FireRedLlmAdapterMetadata {
        FireRedLlmAdapterMetadata {
            downsample_rate: 2,
            llm_dim: 3584,
        }
    }

    #[test]
    fn kv_geometry_reads_the_llm_decoder_metadata() {
        let geometry = firered_llm_kv_geometry(&reference_decoder());
        assert_eq!(
            geometry,
            KvGeometry {
                n_layers: 28,
                kv_heads: 4,
                head_dim: 128,
            }
        );
        // Feeds the shared KV byte model without error (224 rows/position,
        // the same shape the runtime_contract capacity anchor pins).
        let default = kv_bytes_per_position(&geometry, LlmKvCacheSpec::DEFAULT).expect("default");
        assert_eq!(default.total(), 224 * 768);
    }

    #[test]
    fn topology_matches_the_exact_runtime_shape() {
        let envelope = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(40_000).unwrap(),
        )
        .unwrap();
        let plan = DecoderStatePlan::for_envelope(
            &FireRedLlmDecoderStateTopology::new(
                reference_decoder(),
                reference_adapter(),
                32,
                LlmKvCacheSpec::DEFAULT,
            ),
            envelope,
        )
        .unwrap();
        assert_eq!(
            plan.reserve_positions(StateKind::SelfAttentionKv),
            Some(1043)
        );
    }

    #[test]
    fn duration_and_context_boundaries_match_the_runtime_schedule() {
        let rate = NonZeroU32::new(16_000).unwrap();
        let topology = FireRedLlmDecoderStateTopology::new(
            reference_decoder(),
            reference_adapter(),
            32,
            LlmKvCacheSpec::DEFAULT,
        );
        for (seconds, expected_positions) in [(1, 555), (30, 918), (40, 1_043)] {
            let envelope = InvocationEnvelope::new(rate, seconds * 16_000).unwrap();
            let plan = DecoderStatePlan::for_envelope(&topology, envelope).unwrap();
            assert_eq!(
                plan.reserve_positions_by_id(FIRERED_LLM_SELF_KV_STATE_ID),
                Some(expected_positions)
            );
        }
        for seconds in [60, 300] {
            let envelope = InvocationEnvelope::new(rate, seconds * 16_000).unwrap();
            assert!(matches!(
                DecoderStatePlan::for_envelope(&topology, envelope),
                Err(TopologyError::InvocationSampleLimitExceeded { .. })
            ));
        }

        let mut small = reference_decoder();
        // At 1s, semantic P+G=556 while physical K=555. The semantic cap
        // must reject the old one-row false positive.
        small.max_positions = 555;
        let envelope = InvocationEnvelope::new(rate, 16_000).unwrap();
        assert!(matches!(
            DecoderStatePlan::for_envelope(
                &FireRedLlmDecoderStateTopology::new(
                    small,
                    reference_adapter(),
                    32,
                    LlmKvCacheSpec::DEFAULT,
                ),
                envelope,
            ),
            Err(TopologyError::SemanticContextCapExceeded {
                required: 556,
                hard_cap: 555,
                ..
            })
        ));
    }
}
