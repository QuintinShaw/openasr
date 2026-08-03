//! qwen3-asr decoder-state topology, derived from pack metadata, the exact
//! frontend shape oracle, the tokenized prompt wrapper, and the runtime decode
//! budget.
//!
//! Reuses, never re-derives:
//! - KV geometry comes straight off the parsed pack (`llm_layers` /
//!   `llm_kv_heads` / `llm_head_dim` are required metadata keys).
//! - The prompt's audio-token count is
//!   [`super::audio_encoder::qwen3_audio_token_count_for_mel_frames`], the
//!   exact row count the decode path splices into the prompt.
//! - The generation budget reuses the same `QWEN3_DECODE_*` constants and the
//!   `llm_max_positions` context clamp the runtime's
//!   `qwen3_generated_token_budget` / `required_max_positions_for_job` apply,
//!   so admission argues from the same position ceiling the real decode does.

use crate::capacity::KvGeometry;
use crate::capacity::topology::{
    DecoderStateDemandScope, DecoderStateTopology, InvocationEnvelope, InvocationShapeInput,
    PositionBoundProof, StateDemand, StateKind, TopologyError,
    causal_prefix_positions_with_context_cap,
};
use crate::models::audio_frontend::{PadMode, StftFramer};
use crate::models::decode_token_history::context_window_budget;
use crate::models::ggml_asr_executor::{
    GgmlAsrDecoderStatePlanningError, GgmlAsrDecoderStatePlanningInput,
};
use crate::nn::decoder::LlmKvCacheSpec;

use super::audio_encoder::qwen3_audio_token_count_for_mel_frames;
use super::decode_budget::qwen3_desired_generated_tokens;
use super::runtime_contract::Qwen3AsrExecutionMetadata;

pub(crate) const QWEN3_SELF_KV_STATE_ID: &str = "qwen3.decoder.self_kv";
pub(crate) const QWEN3_DECODER_STATE_STREAMS:
    &[crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract] = &[
    crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract::new(
        QWEN3_SELF_KV_STATE_ID,
        StateKind::SelfAttentionKv,
    ),
];

pub(crate) fn plan_qwen3_decoder_state(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    let family = "qwen3-asr";
    let metadata =
        super::runtime_contract::parse_qwen3_execution_metadata(input.preflight.metadata.as_ref())
            .map_err(
                |error| GgmlAsrDecoderStatePlanningError::MetadataUnavailable {
                    family,
                    reason: error.to_string(),
                },
            )?;
    let tokenizer =
        super::tokenizer::Qwen3AsrTokenizer::from_gguf_metadata(input.preflight.metadata.as_ref())
            .ok();
    plan_qwen3_decoder_state_with_components(input, metadata, tokenizer.as_ref())
}

pub(crate) fn plan_qwen3_decoder_state_with_prepared_runtime(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
    prepared: &super::prepared_runtime::Qwen3AsrPreparedRuntime,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    plan_qwen3_decoder_state_with_components(input, prepared.metadata, prepared.tokenizer.as_ref())
}

fn plan_qwen3_decoder_state_with_components(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
    metadata: Qwen3AsrExecutionMetadata,
    tokenizer: Option<&super::tokenizer::Qwen3AsrTokenizer>,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    let family = "qwen3-asr";
    let prompt = super::decode_prompt::build_qwen3_decode_prompt(
        metadata,
        tokenizer,
        1,
        input.request_options,
    )
    .map_err(
        |error| GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
            family,
            reason: error.to_string(),
        },
    )?;
    let logical_fixed_prompt_tokens = prompt.token_ids.len().checked_sub(1).ok_or_else(|| {
        GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
            family,
            reason: "dummy audio span was absent from the tokenized prompt".to_string(),
        }
    })?;
    let mut base_options = input.request_options.clone();
    base_options.prompt = None;
    base_options.prompt_token_ids = None;
    let base_prompt =
        super::decode_prompt::build_qwen3_decode_prompt(metadata, tokenizer, 1, &base_options)
            .map_err(
                |error| GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
                    family,
                    reason: error.to_string(),
                },
            )?;
    let base_fixed_prompt_tokens = base_prompt.token_ids.len().checked_sub(1).ok_or_else(|| {
        GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
            family,
            reason: "base prompt dummy audio span was absent".to_string(),
        }
    })?;
    let stable_fixed_prompt_tokens = base_fixed_prompt_tokens
        .checked_add(input.envelope.max_prompt_tokens())
        .map(|positions| positions.max(logical_fixed_prompt_tokens))
        .ok_or_else(
            || GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
                family,
                reason: "stable prompt position budget overflowed".to_string(),
            },
        )?;
    let spec =
        super::resolve_qwen_family_production_kv_cache_policy(input.backend, metadata.llm_head_dim)
            .to_spec();
    crate::capacity::topology::DecoderStatePlan::build(
        &Qwen3DecoderStateTopology::new(
            metadata,
            logical_fixed_prompt_tokens,
            stable_fixed_prompt_tokens,
            spec,
        ),
        input.invocation,
        input.envelope,
    )
    .map_err(|source| GgmlAsrDecoderStatePlanningError::Topology { family, source })
}

/// The decoder KV geometry the loaded pack advertises.
pub(crate) fn qwen3_kv_geometry(metadata: &Qwen3AsrExecutionMetadata) -> KvGeometry {
    KvGeometry {
        n_layers: metadata.llm_layers,
        kv_heads: metadata.llm_kv_heads,
        head_dim: metadata.llm_head_dim,
    }
}

/// Strict decoder-state topology for one Qwen3-ASR execution candidate.
///
/// `fixed_prompt_tokens` is measured once from the loaded tokenizer for the
/// ChatML wrapper excluding audio-pad and request/carry tokens. Keeping that
/// value outside this module avoids promoting the admission-only `32` estimate
/// above into a runtime allocation contract.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Qwen3DecoderStateTopology {
    metadata: Qwen3AsrExecutionMetadata,
    logical_fixed_prompt_tokens: usize,
    stable_fixed_prompt_tokens: usize,
    kv_spec: LlmKvCacheSpec,
}

impl Qwen3DecoderStateTopology {
    pub(crate) const fn new(
        metadata: Qwen3AsrExecutionMetadata,
        logical_fixed_prompt_tokens: usize,
        stable_fixed_prompt_tokens: usize,
        kv_spec: LlmKvCacheSpec,
    ) -> Self {
        Self {
            metadata,
            logical_fixed_prompt_tokens,
            stable_fixed_prompt_tokens,
            kv_spec,
        }
    }

    fn mel_frames(&self, invocation: InvocationShapeInput) -> Result<usize, TopologyError> {
        if invocation.sample_rate_hz().get() != self.metadata.sample_rate_hz {
            return Err(TopologyError::UnsupportedSampleRate {
                expected_hz: self.metadata.sample_rate_hz,
                actual_hz: invocation.sample_rate_hz().get(),
            });
        }
        // Reuse the actual frontend framer's allocation-free shape oracle,
        // then mirror Qwen's deliberate final-row drop.
        StftFramer::output_frame_count_for(
            self.metadata.n_fft,
            self.metadata.hop_length,
            PadMode::ZeroCenter,
            invocation.samples(),
        )
        .map_err(|error| TopologyError::Unavailable {
            reason: format!("qwen3-asr STFT shape is invalid: {error}"),
        })?
        .checked_sub(1)
        .filter(|&frames| frames > 0)
        .ok_or(TopologyError::Unavailable {
            reason: "qwen3-asr audio is too short after the final mel-row drop".to_string(),
        })
    }

    fn desired_generated_tokens(
        &self,
        invocation: InvocationShapeInput,
    ) -> Result<usize, TopologyError> {
        qwen3_desired_generated_tokens(
            invocation.samples(),
            invocation.sample_rate_hz().get() as usize,
        )
        .map_err(|error| TopologyError::Unavailable {
            reason: format!("qwen generation budget is unavailable: {error}"),
        })
    }
}

impl DecoderStateTopology for Qwen3DecoderStateTopology {
    fn demands(
        &self,
        scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
    ) -> Result<Vec<StateDemand>, TopologyError> {
        match scope {
            DecoderStateDemandScope::ExactInvocation(invocation) => {
                self.demands_for(invocation, self.logical_fixed_prompt_tokens)
            }
            DecoderStateDemandScope::StableEnvelope(envelope) => self.demands_for(
                envelope.maximum_invocation(),
                self.stable_fixed_prompt_tokens,
            ),
        }
    }
}

impl Qwen3DecoderStateTopology {
    fn demands_for(
        &self,
        invocation: InvocationShapeInput,
        fixed_prompt_tokens: usize,
    ) -> Result<Vec<StateDemand>, TopologyError> {
        let audio_tokens = qwen3_audio_token_count_for_mel_frames(self.mel_frames(invocation)?);
        let prompt_positions = fixed_prompt_tokens.checked_add(audio_tokens).ok_or(
            TopologyError::ArithmeticOverflow {
                operation: "qwen prompt positions",
            },
        )?;
        let context_remaining =
            context_window_budget(self.metadata.llm_max_positions, prompt_positions).ok_or_else(
                || TopologyError::Unavailable {
                    reason: format!(
                        "qwen prompt positions {prompt_positions} exhaust llm_max_positions {}",
                        self.metadata.llm_max_positions
                    ),
                },
            )?;
        // This is decode semantics, not a memory-pressure clamp: the runtime's
        // qwen3_generated_token_budget applies the same min before creating
        // its DecodeConfig, so the topology must reserve that legal budget
        // rather than reject a near-context request the runtime accepts.
        let generated_positions = self
            .desired_generated_tokens(invocation)?
            .min(context_remaining);
        let positions = causal_prefix_positions_with_context_cap(
            QWEN3_SELF_KV_STATE_ID,
            prompt_positions,
            generated_positions,
            self.metadata.llm_max_positions,
        )?;
        Ok(vec![StateDemand::from_llm_kv_geometry(
            QWEN3_SELF_KV_STATE_ID,
            StateKind::SelfAttentionKv,
            positions,
            self.metadata.llm_max_positions,
            qwen3_kv_geometry(&self.metadata),
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
    use crate::capacity::topology::{DecoderStatePlan, InvocationEnvelope, StateKind};
    use crate::capacity::{KvGeometry, kv_bytes_per_position};
    use crate::nn::decoder::LlmKvCacheSpec;

    /// Real-checkpoint-shaped metadata (the 1.7B config: 28-layer GQA decoder,
    /// 8 KV heads, head_dim 128, 40960-position context, 16kHz/160-hop mel).
    fn reference_metadata() -> Qwen3AsrExecutionMetadata {
        Qwen3AsrExecutionMetadata {
            sample_rate_hz: 16_000,
            n_mels: 128,
            n_fft: 400,
            win_length: 400,
            hop_length: 160,
            audio_layers: 18,
            audio_d_model: 896,
            audio_heads: 14,
            llm_layers: 28,
            llm_d_model: 1024,
            llm_heads: 16,
            llm_kv_heads: 8,
            llm_head_dim: 128,
            vocab_size: 151_936,
            llm_max_positions: 40_960,
            audio_start_token_id: 11,
            audio_end_token_id: 12,
            audio_pad_token_id: 13,
            eos_token_id: 14,
            pad_token_id: 15,
        }
    }

    #[test]
    fn kv_geometry_reads_the_llm_decoder_metadata() {
        let geometry = qwen3_kv_geometry(&reference_metadata());
        assert_eq!(
            geometry,
            KvGeometry {
                n_layers: 28,
                kv_heads: 8,
                head_dim: 128,
            }
        );
        // The geometry feeds the shared KV byte model without error (the same
        // 448-row/position shape the runtime_contract capacity anchor pins).
        let default = kv_bytes_per_position(&geometry, LlmKvCacheSpec::DEFAULT).expect("default");
        assert_eq!(default.total(), 448 * 768);
    }

    #[test]
    fn frontend_count_oracle_pins_final_row_drop_boundary() {
        let topology =
            Qwen3DecoderStateTopology::new(reference_metadata(), 32, 32, LlmKvCacheSpec::DEFAULT);
        let rate = NonZeroU32::new(16_000).unwrap();
        assert!(
            topology
                .mel_frames(InvocationShapeInput::new(rate, 159).unwrap())
                .is_err()
        );
        assert_eq!(
            topology
                .mel_frames(InvocationShapeInput::new(rate, 160).unwrap())
                .unwrap(),
            1
        );
    }

    #[test]
    fn topology_uses_exact_integer_audio_and_generation_counts() {
        let envelope = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(30_000).unwrap(),
        )
        .unwrap();
        let plan = DecoderStatePlan::for_envelope(
            &Qwen3DecoderStateTopology::new(reference_metadata(), 32, 32, LlmKvCacheSpec::DEFAULT),
            envelope,
        )
        .unwrap();
        assert_eq!(
            plan.reserve_positions(StateKind::SelfAttentionKv),
            Some(813)
        );
    }

    #[test]
    fn duration_boundaries_follow_frontend_prompt_and_greedy_schedule() {
        let rate = NonZeroU32::new(16_000).unwrap();
        let topology =
            Qwen3DecoderStateTopology::new(reference_metadata(), 32, 32, LlmKvCacheSpec::DEFAULT);
        // 100 mel frames per second, padded in 100-frame encoder chunks;
        // every chunk emits 13 decoder audio positions. Generation is the
        // family 12 token/s + 32 rule with a 128-token floor.
        for (seconds, expected_positions) in [(1, 172), (30, 813), (60, 1_563), (300, 7_563)] {
            let envelope = InvocationEnvelope::new(rate, seconds * 16_000).unwrap();
            let plan = DecoderStatePlan::for_envelope(&topology, envelope).unwrap();
            assert_eq!(
                plan.reserve_positions_by_id(QWEN3_SELF_KV_STATE_ID),
                Some(expected_positions),
                "unexpected qwen capacity at {seconds}s"
            );
            assert!(expected_positions < reference_metadata().llm_max_positions);
        }
    }

    #[test]
    fn stable_prompt_budget_is_reserved_without_inflating_logical_shape() {
        let rate = NonZeroU32::new(16_000).unwrap();
        let invocation = InvocationShapeInput::new(rate, 30 * 16_000).unwrap();
        let envelope =
            InvocationEnvelope::from_milliseconds(rate, NonZeroU32::new(30_000).unwrap())
                .unwrap()
                .with_max_prompt_tokens(512);
        let plan = DecoderStatePlan::build(
            &Qwen3DecoderStateTopology::new(
                reference_metadata(),
                32,
                32 + envelope.max_prompt_tokens(),
                LlmKvCacheSpec::DEFAULT,
            ),
            invocation,
            envelope,
        )
        .unwrap();
        let axis = plan
            .position_axis(QWEN3_SELF_KV_STATE_ID, StateKind::SelfAttentionKv)
            .unwrap();
        assert_eq!(axis.logical_positions, 813);
        assert_eq!(axis.resident_positions, 1_325);
    }

    #[test]
    fn topology_matches_runtime_context_clamp_for_generation() {
        let mut metadata = reference_metadata();
        metadata.llm_max_positions = 800;
        let envelope = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(30_000).unwrap(),
        )
        .unwrap();
        let plan = DecoderStatePlan::for_envelope(
            &Qwen3DecoderStateTopology::new(metadata, 32, 32, LlmKvCacheSpec::DEFAULT),
            envelope,
        )
        .unwrap();
        assert_eq!(
            plan.reserve_positions(StateKind::SelfAttentionKv),
            Some(799)
        );
    }

    #[test]
    fn topology_rejects_a_prompt_that_exhausts_context() {
        let mut metadata = reference_metadata();
        // 30s fixed+audio prompt is 422 positions before generation.
        metadata.llm_max_positions = 422;
        let envelope = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(30_000).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            DecoderStatePlan::for_envelope(
                &Qwen3DecoderStateTopology::new(metadata, 32, 32, LlmKvCacheSpec::DEFAULT),
                envelope,
            ),
            Err(TopologyError::Unavailable { .. })
        ));
    }

    #[test]
    fn semantic_context_rejects_the_one_row_physical_false_positive() {
        let mut metadata = reference_metadata();
        // 1s has P=45 and G=128, hence semantic=173 and physical K=172.
        // A 172-position context used to pass the physical-only cap check.
        metadata.llm_max_positions = 172;
        let envelope = InvocationEnvelope::new(NonZeroU32::new(16_000).unwrap(), 16_000).unwrap();
        // Qwen intentionally clamps G to remaining context, so it stays legal
        // and proves the runtime/planner shared clamp rather than rejecting.
        let plan = DecoderStatePlan::for_envelope(
            &Qwen3DecoderStateTopology::new(metadata, 32, 32, LlmKvCacheSpec::DEFAULT),
            envelope,
        )
        .unwrap();
        assert_eq!(
            plan.reserve_positions_by_id(QWEN3_SELF_KV_STATE_ID),
            Some(171)
        );
    }
}
