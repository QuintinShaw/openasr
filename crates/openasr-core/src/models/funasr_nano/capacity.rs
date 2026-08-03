//! FunASR-Nano decoder-state topology. Its exact position bound is assembled
//! from pack metadata, the tokenized prompt wrapper, and the same frontend and
//! generation shape facts used by execution.
//!
//! Reuses, never re-derives:
//! - KV geometry comes straight off the parsed pack (`funasr.llm.*` are
//!   required metadata keys).
//! - The audio-token count walks the exact runtime pipeline arithmetic:
//!   snip-edges kaldi fbank framing (SenseVoice `WavFrontend` constants the
//!   executor reuses -- 16 kHz / 25 ms / 10 ms), FunASR LFR stacking
//!   (`LFR_N = 6`), and the official low-frame-rate truncation
//!   ([`super::decode_prompt::funasr_nano_audio_token_count`], the same
//!   function the real decode path calls).
//! - The generation budget reuses the executor's own constant, so planning
//!   and allocation share one `prompt + generation` contract.

use crate::capacity::KvGeometry;
use crate::capacity::topology::{
    DecoderStateDemandScope, DecoderStateTopology, InvocationEnvelope, InvocationShapeInput,
    PositionBoundProof, StateDemand, StateKind, TopologyError,
    causal_prefix_positions_with_context_cap,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrDecoderStatePlanningError, GgmlAsrDecoderStatePlanningInput,
};
use crate::models::sensevoice::frontend::{SAMPLE_RATE_HZ, sensevoice_lfr_frame_count_for_samples};
use crate::nn::decoder::LlmKvCacheSpec;

use super::decode_prompt::funasr_nano_audio_token_count;
use super::executor::FUNASR_NANO_MAX_GENERATED_TOKENS;
use super::runtime_contract::FunasrNanoDecoderMetadata;

pub(crate) const FUNASR_NANO_SELF_KV_STATE_ID: &str = "funasr-nano.decoder.self_kv";
pub(crate) const FUNASR_NANO_DECODER_STATE_STREAMS:
    &[crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract] = &[
    crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract::new(
        FUNASR_NANO_SELF_KV_STATE_ID,
        StateKind::SelfAttentionKv,
    ),
];

pub(crate) fn plan_funasr_nano_decoder_state(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    let family = "funasr-nano";
    let metadata = input.preflight.metadata.as_ref();
    let decoder =
        super::runtime_contract::parse_funasr_nano_decoder_metadata(metadata).map_err(|error| {
            GgmlAsrDecoderStatePlanningError::MetadataUnavailable {
                family,
                reason: error.to_string(),
            }
        })?;
    let tokenizer =
        super::tokenizer::FunasrNanoTokenizer::from_gguf_metadata(metadata).map_err(|error| {
            GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
                family,
                reason: error.to_string(),
            }
        })?;
    let prompt =
        super::decode_prompt::build_funasr_nano_decode_prompt(&tokenizer, 1).map_err(|error| {
            GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
                family,
                reason: error.to_string(),
            }
        })?;
    let fixed_prompt_tokens = prompt.token_ids.len().checked_sub(1).ok_or_else(|| {
        GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
            family,
            reason: "dummy audio span was absent from the tokenized prompt".to_string(),
        }
    })?;
    let geometry = funasr_nano_kv_geometry(&decoder);
    let spec = crate::models::qwen::resolve_qwen_family_production_kv_cache_policy(
        input.backend,
        geometry.head_dim,
    )
    .to_spec();
    crate::capacity::topology::DecoderStatePlan::build(
        &FunasrNanoDecoderStateTopology::new(decoder, fixed_prompt_tokens, spec),
        input.invocation,
        input.envelope,
    )
    .map_err(|source| GgmlAsrDecoderStatePlanningError::Topology { family, source })
}

const FUNASR_NANO_MAX_INPUT_SAMPLES: usize = SAMPLE_RATE_HZ as usize * 40;

/// The decoder KV geometry the loaded pack advertises.
pub(crate) fn funasr_nano_kv_geometry(decoder: &FunasrNanoDecoderMetadata) -> KvGeometry {
    KvGeometry {
        n_layers: decoder.n_layers,
        kv_heads: decoder.n_kv_heads,
        head_dim: decoder.head_dim,
    }
}

pub(crate) fn funasr_nano_audio_token_count_for_samples(
    samples: usize,
) -> Result<usize, TopologyError> {
    let lfr_frames = sensevoice_lfr_frame_count_for_samples(samples);
    if lfr_frames == 0 {
        return Err(TopologyError::Unavailable {
            reason: "funasr-nano audio is too short to produce one fbank frame".to_string(),
        });
    }
    Ok(funasr_nano_audio_token_count(lfr_frames))
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FunasrNanoDecoderStateTopology {
    decoder: FunasrNanoDecoderMetadata,
    fixed_prompt_tokens: usize,
    kv_spec: LlmKvCacheSpec,
}

impl FunasrNanoDecoderStateTopology {
    pub(crate) const fn new(
        decoder: FunasrNanoDecoderMetadata,
        fixed_prompt_tokens: usize,
        kv_spec: LlmKvCacheSpec,
    ) -> Self {
        Self {
            decoder,
            fixed_prompt_tokens,
            kv_spec,
        }
    }
}

impl DecoderStateTopology for FunasrNanoDecoderStateTopology {
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
        if invocation.samples() > FUNASR_NANO_MAX_INPUT_SAMPLES {
            return Err(TopologyError::InvocationSampleLimitExceeded {
                required_samples: invocation.samples(),
                max_samples: FUNASR_NANO_MAX_INPUT_SAMPLES,
            });
        }
        let audio_tokens = funasr_nano_audio_token_count_for_samples(invocation.samples())?;
        let prompt_positions = self.fixed_prompt_tokens.checked_add(audio_tokens).ok_or(
            TopologyError::ArithmeticOverflow {
                operation: "funasr-nano prompt positions",
            },
        )?;
        let positions = causal_prefix_positions_with_context_cap(
            FUNASR_NANO_SELF_KV_STATE_ID,
            prompt_positions,
            FUNASR_NANO_MAX_GENERATED_TOKENS,
            self.decoder.max_positions,
        )?;
        Ok(vec![StateDemand::from_llm_kv_geometry(
            FUNASR_NANO_SELF_KV_STATE_ID,
            StateKind::SelfAttentionKv,
            positions,
            self.decoder.max_positions,
            funasr_nano_kv_geometry(&self.decoder),
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

    /// Real-checkpoint-shaped metadata (Qwen3-0.6B decoder: 28 layers, GQA
    /// with 8 KV heads, head_dim 128, 40960-position context -- the same
    /// values `runtime_contract`'s `full_metadata` fixture parses).
    fn reference_decoder() -> FunasrNanoDecoderMetadata {
        FunasrNanoDecoderMetadata {
            n_layers: 28,
            d_model: 1024,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 128,
            ffn_dim: 3072,
            vocab_size: 151_936,
            max_positions: 40_960,
            chatml_im_start_token_id: 151_644,
            chatml_im_end_token_id: 151_645,
            endoftext_token_id: 151_643,
        }
    }

    #[test]
    fn kv_geometry_reads_the_llm_decoder_metadata() {
        let geometry = funasr_nano_kv_geometry(&reference_decoder());
        assert_eq!(
            geometry,
            KvGeometry {
                n_layers: 28,
                kv_heads: 8,
                head_dim: 128,
            }
        );
        // Feeds the shared KV byte model without error (448 rows/position,
        // the same shape the qwen3 / moss capacity anchors pin).
        let default = kv_bytes_per_position(&geometry, LlmKvCacheSpec::DEFAULT).expect("default");
        assert_eq!(default.total(), 448 * 768);
    }

    #[test]
    fn topology_matches_the_exact_runtime_shape() {
        let envelope = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(40_000).unwrap(),
        )
        .unwrap();
        let plan = DecoderStatePlan::for_envelope(
            &FunasrNanoDecoderStateTopology::new(reference_decoder(), 32, LlmKvCacheSpec::DEFAULT),
            envelope,
        )
        .unwrap();
        assert_eq!(
            plan.reserve_positions(StateKind::SelfAttentionKv),
            Some(627)
        );
    }

    #[test]
    fn duration_and_context_boundaries_match_lfr_and_greedy_schedule() {
        let rate = NonZeroU32::new(16_000).unwrap();
        let topology =
            FunasrNanoDecoderStateTopology::new(reference_decoder(), 32, LlmKvCacheSpec::DEFAULT);
        for (seconds, expected_positions) in [(1, 546), (30, 606), (40, 627)] {
            let envelope = InvocationEnvelope::new(rate, seconds * 16_000).unwrap();
            let plan = DecoderStatePlan::for_envelope(&topology, envelope).unwrap();
            assert_eq!(
                plan.reserve_positions_by_id(FUNASR_NANO_SELF_KV_STATE_ID),
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
        small.max_positions = 546;
        assert!(matches!(
            DecoderStatePlan::for_envelope(
                &FunasrNanoDecoderStateTopology::new(small, 32, LlmKvCacheSpec::DEFAULT),
                InvocationEnvelope::new(rate, 16_000).unwrap(),
            ),
            Err(TopologyError::SemanticContextCapExceeded {
                required: 547,
                hard_cap: 546,
                ..
            })
        ));
    }
}
