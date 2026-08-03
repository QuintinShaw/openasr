//! Hy-MT2 token-scaled persistent-state topology.
//!
//! A translation candidate owns two simultaneous self-attention histories:
//! the runtime's decode scratch and one prefix arena per streaming translation
//! session. The prefix arena is host-only; the reusable resident decode graph
//! belongs to the shared scratch path. They therefore have distinct stable
//! IDs and byte ownership even though both are causal self-KV.

use std::num::NonZeroUsize;

use crate::capacity::topology::{
    DecoderStateDemandScope, DecoderStatePlan, DecoderStateTopology, PositionBoundProof,
    StateBytes, StateDemand, StateKind, TokenInvocationContract, TokenInvocationEnvelope,
    TokenInvocationShapeInput, TopologyError, causal_prefix_positions_with_context_cap,
};
use crate::capacity::{KvGeometry, kv_bytes_per_position};
use crate::models::hymt2::prompt::{
    build_subtitle_translation_prompt_prefix, max_output_tokens_for_source_tokens,
};
use crate::models::qwen::qwen_host_kv_quoted_bytes;
use crate::nn::decoder::LlmKvCacheSpec;

use super::Hymt2ExecutionMetadata;

pub(crate) const HYMT2_DECODE_SCRATCH_STATE_ID: &str = "hymt2.decode_scratch.self_kv";
pub(crate) const HYMT2_PREFIX_CACHE_STATE_ID: &str = "hymt2.session_prefix.self_kv";

const UTF8_BYTES_PER_UNICODE_SCALAR_UPPER_BOUND: usize = 4;

#[derive(Debug, Clone)]
pub(crate) struct Hymt2DecoderCapacityContract {
    topology: Hymt2DecoderStateTopology,
    envelope: TokenInvocationEnvelope,
    max_source_clause_chars: Option<NonZeroUsize>,
    stable_plan: DecoderStatePlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Hymt2StableHostStateBytes {
    pub(crate) decode_scratch: u64,
    pub(crate) session_prefix: u64,
}

impl Hymt2StableHostStateBytes {
    pub(crate) fn total(self) -> Result<u64, TopologyError> {
        self.decode_scratch.checked_add(self.session_prefix).ok_or(
            TopologyError::ArithmeticOverflow {
                operation: "Hy-MT2 aggregate stable host state bytes",
            },
        )
    }
}

impl Hymt2DecoderCapacityContract {
    /// Plans the product clause envelope without materializing the tokenizer.
    /// The prompt topology contributes exactly BOS + USER before content and
    /// exactly ASSISTANT after it; byte-level BPE can emit no more tokens than
    /// UTF-8 bytes. This is therefore available to host admission before any
    /// large candidate-owned maps or KV buffers are allocated.
    pub(crate) fn from_clause_envelope(
        metadata: Hymt2ExecutionMetadata,
        kv_spec: LlmKvCacheSpec,
        max_source_clause_chars: usize,
    ) -> Result<Self, TopologyError> {
        let max_source_clause_chars =
            NonZeroUsize::new(max_source_clause_chars).ok_or(TopologyError::Unavailable {
                reason: "Hy-MT2 source-clause envelope must be positive".to_string(),
            })?;
        // Byte-level BPE begins with at most one symbol per UTF-8 byte and
        // merges only reduce that count. A Unicode scalar occupies at most
        // four bytes, so this is a proof, not a percentage safety margin.
        let max_source_tokens = max_source_clause_chars
            .get()
            .checked_mul(UTF8_BYTES_PER_UNICODE_SCALAR_UPPER_BOUND)
            .ok_or(TopologyError::ArithmeticOverflow {
                operation: "Hy-MT2 source character token bound",
            })?;
        let prefix_text = build_subtitle_translation_prompt_prefix(&[]);
        const PREFIX_SPECIAL_POSITIONS: usize = 2; // BOS + USER
        const GENERATION_MARKER_POSITIONS: usize = 1; // ASSISTANT
        // Tokenizing prefix+source together can alter pre-tokenizer piece
        // boundaries, so the exact prefix token count is not compositional.
        // Bound the entire fixed UTF-8 prefix by bytes instead; role markers
        // remain exact because the prompt builder inserts them as token IDs.
        let max_prefix_positions = PREFIX_SPECIAL_POSITIONS
            .checked_add(prefix_text.len())
            .and_then(|positions| positions.checked_add(max_source_tokens))
            .ok_or(TopologyError::ArithmeticOverflow {
                operation: "Hy-MT2 stable prefix positions",
            })?;
        let max_prompt_positions = max_prefix_positions
            .checked_add(GENERATION_MARKER_POSITIONS)
            .ok_or(TopologyError::ArithmeticOverflow {
                operation: "Hy-MT2 stable prompt positions",
            })?;
        let max_generated_positions = max_output_tokens_for_source_tokens(max_source_tokens);
        let envelope = TokenInvocationEnvelope::new(
            max_prompt_positions,
            max_prefix_positions,
            max_generated_positions,
        )?;
        Self::new(
            metadata,
            kv_spec,
            envelope,
            Some(max_source_clause_chars),
            PositionBoundProof::Conservative {
                basis: "UTF-8 byte upper bound for the product clause-character envelope",
            },
        )
    }

    /// Backward-compatible envelope for the public low-level runtime API,
    /// which historically accepted any prompt that fit the pack context.
    /// Product paths should use [`Self::from_tokenizer`] with their clause
    /// segmentation bound instead.
    pub(crate) fn full_context(
        metadata: Hymt2ExecutionMetadata,
        kv_spec: LlmKvCacheSpec,
    ) -> Result<Self, TopologyError> {
        let max_prompt_positions = metadata
            .runtime_context_length
            .checked_sub(1)
            .ok_or(TopologyError::InvalidTokenEnvelope)?;
        let envelope = TokenInvocationEnvelope::with_total_position_cap(
            max_prompt_positions,
            max_prompt_positions,
            max_output_tokens_for_source_tokens(max_prompt_positions),
            metadata.runtime_context_length,
        )?;
        Self::new(metadata, kv_spec, envelope, None, PositionBoundProof::Exact)
    }

    fn new(
        metadata: Hymt2ExecutionMetadata,
        kv_spec: LlmKvCacheSpec,
        envelope: TokenInvocationEnvelope,
        max_source_clause_chars: Option<NonZeroUsize>,
        stable_proof: PositionBoundProof,
    ) -> Result<Self, TopologyError> {
        let topology = Hymt2DecoderStateTopology {
            metadata,
            kv_spec,
            stable_proof,
        };
        let stable_plan = DecoderStatePlan::build::<TokenInvocationContract, _>(
            &topology,
            envelope.maximum_invocation(),
            envelope,
        )?;
        Ok(Self {
            topology,
            envelope,
            max_source_clause_chars,
            stable_plan,
        })
    }

    pub(crate) fn plan_invocation(
        &self,
        prompt_positions: usize,
        prefix_positions: usize,
        max_generated_positions: usize,
    ) -> Result<DecoderStatePlan, TopologyError> {
        DecoderStatePlan::build::<TokenInvocationContract, _>(
            &self.topology,
            TokenInvocationShapeInput::new(
                prompt_positions,
                prefix_positions,
                max_generated_positions,
            )?,
            self.envelope,
        )
    }

    pub(crate) const fn max_source_clause_chars(&self) -> Option<usize> {
        match self.max_source_clause_chars {
            Some(value) => Some(value.get()),
            None => None,
        }
    }

    pub(crate) fn stable_plan(&self) -> &DecoderStatePlan {
        &self.stable_plan
    }

    pub(crate) const fn kv_spec(&self) -> LlmKvCacheSpec {
        self.topology.kv_spec
    }

    /// Exact engine-requested capacities for the two simultaneously-live host
    /// KV owners, including each layer-table Vec in addition to K/V payload.
    pub(crate) fn stable_materialized_host_state_bytes(
        &self,
    ) -> Result<Hymt2StableHostStateBytes, TopologyError> {
        let scratch = self
            .stable_plan
            .position_axis(HYMT2_DECODE_SCRATCH_STATE_ID, StateKind::SelfAttentionKv)?;
        let prefix = self
            .stable_plan
            .position_axis(HYMT2_PREFIX_CACHE_STATE_ID, StateKind::SelfAttentionKv)?;
        let quote = |state: &'static str, positions| {
            qwen_host_kv_quoted_bytes(
                self.topology.metadata.layers,
                positions,
                self.topology.metadata.kv_heads,
                self.topology.metadata.head_dim,
                self.topology.kv_spec.host,
            )
            .map_err(|reason| TopologyError::InvalidStateLayout { state, reason })
        };
        Ok(Hymt2StableHostStateBytes {
            // Runtime scratch is built from the stable plan's invocation
            // capacity; prefix cache is explicitly built at resident span.
            decode_scratch: quote(HYMT2_DECODE_SCRATCH_STATE_ID, scratch.logical_positions)?,
            session_prefix: quote(HYMT2_PREFIX_CACHE_STATE_ID, prefix.resident_positions)?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct Hymt2DecoderStateTopology {
    metadata: Hymt2ExecutionMetadata,
    kv_spec: LlmKvCacheSpec,
    stable_proof: PositionBoundProof,
}

impl Hymt2DecoderStateTopology {
    fn demands_for(
        self,
        invocation: TokenInvocationShapeInput,
        proof: PositionBoundProof,
    ) -> Result<Vec<StateDemand>, TopologyError> {
        let scratch_positions = causal_prefix_positions_with_context_cap(
            HYMT2_DECODE_SCRATCH_STATE_ID,
            invocation.prompt_positions(),
            invocation.max_generated_positions(),
            self.metadata.runtime_context_length,
        )?;
        let geometry = KvGeometry {
            n_layers: self.metadata.layers,
            kv_heads: self.metadata.kv_heads,
            head_dim: self.metadata.head_dim,
        };
        let scratch = StateDemand::from_llm_kv_geometry(
            HYMT2_DECODE_SCRATCH_STATE_ID,
            StateKind::SelfAttentionKv,
            scratch_positions,
            self.metadata.runtime_context_length,
            geometry,
            self.kv_spec,
            invocation.sequences().get() as usize,
            proof,
        )?;

        let per_position = kv_bytes_per_position(&geometry, self.kv_spec).map_err(|reason| {
            TopologyError::InvalidStateLayout {
                state: HYMT2_PREFIX_CACHE_STATE_ID,
                reason,
            }
        })?;
        let prefix_positions = u64::try_from(invocation.prefix_positions()).map_err(|_| {
            TopologyError::ArithmeticOverflow {
                operation: "Hy-MT2 prefix position conversion",
            }
        })?;
        let sequences = u64::from(invocation.sequences().get());
        let prefix_host_bytes = per_position
            .host
            .checked_mul(prefix_positions)
            .and_then(|bytes| bytes.checked_mul(sequences))
            .ok_or(TopologyError::ArithmeticOverflow {
                operation: "Hy-MT2 prefix host bytes",
            })?;
        let prefix = StateDemand::new(
            HYMT2_PREFIX_CACHE_STATE_ID,
            StateKind::SelfAttentionKv,
            invocation.prefix_positions(),
            self.metadata.runtime_context_length,
            StateBytes {
                host: prefix_host_bytes,
                // Prefix prefill uses the host-cache graph path. There is one
                // resident reusable graph, already owned by decode scratch.
                resident: 0,
            },
            proof,
        )?;
        Ok(vec![scratch, prefix])
    }
}

impl DecoderStateTopology<TokenInvocationContract> for Hymt2DecoderStateTopology {
    fn demands(
        &self,
        scope: DecoderStateDemandScope<TokenInvocationShapeInput, TokenInvocationEnvelope>,
    ) -> Result<Vec<StateDemand>, TopologyError> {
        match scope {
            DecoderStateDemandScope::ExactInvocation(invocation) => {
                self.demands_for(invocation, PositionBoundProof::Exact)
            }
            DecoderStateDemandScope::StableEnvelope(envelope) => {
                // Correlated prompt/generation limits are not monotone in a
                // component-wise maximum. The stable scratch maximum is the
                // semantic total cap minus the one sampled-but-unwritten row;
                // prefix has its own independent maximum.
                let stable = TokenInvocationShapeInput::new_with_sequences(
                    envelope.max_total_positions().saturating_sub(1),
                    envelope.max_prefix_positions(),
                    1,
                    envelope.maximum_invocation().sequences(),
                )?;
                self.demands_for(stable, self.stable_proof)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;

    fn metadata() -> Hymt2ExecutionMetadata {
        Hymt2ExecutionMetadata {
            layers: 32,
            d_model: 2_048,
            ffn_dim: 6_144,
            heads: 16,
            kv_heads: 4,
            head_dim: 128,
            vocab_size: 120_818,
            gguf_context_length: 32_768,
            runtime_context_length: 4_096,
            rope_freq_base: 11_158_840.0,
            rms_norm_epsilon: 1.0e-5,
        }
    }

    #[test]
    fn text_topology_models_scratch_and_prefix_as_independent_live_owners() {
        let topology = Hymt2DecoderStateTopology {
            metadata: metadata(),
            kv_spec: LlmKvCacheSpec::DEFAULT,
            stable_proof: PositionBoundProof::Exact,
        };
        let envelope = TokenInvocationEnvelope::new(181, 180, 96).unwrap();
        let invocation = TokenInvocationShapeInput::new(31, 30, 40).unwrap();
        let plan =
            DecoderStatePlan::build::<TokenInvocationContract, _>(&topology, invocation, envelope)
                .unwrap();

        let scratch = plan
            .position_axis(HYMT2_DECODE_SCRATCH_STATE_ID, StateKind::SelfAttentionKv)
            .unwrap();
        let prefix = plan
            .position_axis(HYMT2_PREFIX_CACHE_STATE_ID, StateKind::SelfAttentionKv)
            .unwrap();
        assert_eq!(
            (scratch.logical_positions, scratch.resident_positions),
            (70, 276)
        );
        assert_eq!(
            (prefix.logical_positions, prefix.resident_positions),
            (30, 180)
        );
        assert!(scratch.reserve_bytes.host > 0);
        assert!(scratch.reserve_bytes.resident > 0);
        assert!(prefix.reserve_bytes.host > 0);
        assert_eq!(prefix.reserve_bytes.resident, 0);
    }

    #[test]
    fn text_invocation_outside_the_declared_token_envelope_fails_closed() {
        let topology = Hymt2DecoderStateTopology {
            metadata: metadata(),
            kv_spec: LlmKvCacheSpec::DEFAULT,
            stable_proof: PositionBoundProof::Exact,
        };
        let envelope = TokenInvocationEnvelope::new(181, 180, 96).unwrap();
        let error = DecoderStatePlan::build::<TokenInvocationContract, _>(
            &topology,
            TokenInvocationShapeInput::new(182, 180, 96).unwrap(),
            envelope,
        )
        .unwrap_err();
        assert_eq!(error, TopologyError::InvocationOutsideEnvelope);
    }

    #[test]
    fn correlated_total_cap_and_physical_kv_span_are_distinct() {
        let mut small = metadata();
        small.runtime_context_length = 70;
        let topology = Hymt2DecoderStateTopology {
            metadata: small,
            kv_spec: LlmKvCacheSpec::DEFAULT,
            stable_proof: PositionBoundProof::Exact,
        };
        let envelope = TokenInvocationEnvelope::new(40, 39, 31).unwrap();
        assert!(matches!(
            DecoderStatePlan::build::<TokenInvocationContract, _>(
                &topology,
                envelope.maximum_invocation(),
                envelope,
            ),
            Err(TopologyError::SemanticContextCapExceeded {
                required: 71,
                hard_cap: 70,
                ..
            })
        ));
    }

    #[test]
    fn text_multi_sequence_arena_scales_both_independent_owner_bytes() {
        let topology = Hymt2DecoderStateTopology {
            metadata: metadata(),
            kv_spec: LlmKvCacheSpec::DEFAULT,
            stable_proof: PositionBoundProof::Exact,
        };
        let envelope = TokenInvocationEnvelope::with_total_position_cap_and_sequences(
            181,
            180,
            96,
            277,
            NonZeroU32::new(3).unwrap(),
        )
        .unwrap();
        let invocation =
            TokenInvocationShapeInput::new_with_sequences(31, 30, 40, NonZeroU32::new(2).unwrap())
                .unwrap();
        let plan =
            DecoderStatePlan::build::<TokenInvocationContract, _>(&topology, invocation, envelope)
                .unwrap();
        let metadata = metadata();
        let per_position = kv_bytes_per_position(
            &KvGeometry {
                n_layers: metadata.layers,
                kv_heads: metadata.kv_heads,
                head_dim: metadata.head_dim,
            },
            LlmKvCacheSpec::DEFAULT,
        )
        .unwrap();
        for id in [HYMT2_DECODE_SCRATCH_STATE_ID, HYMT2_PREFIX_CACHE_STATE_ID] {
            let allocation = plan.allocation(id).unwrap();
            assert_eq!(
                allocation.reserve.bytes.host,
                per_position.host * allocation.reserve.positions as u64 * 3,
            );
            assert_eq!(
                allocation.logical.bytes.host,
                per_position.host * allocation.logical.positions as u64 * 2,
            );
        }
    }

    #[test]
    fn clause_character_bound_overflow_fails_closed() {
        assert!(matches!(
            Hymt2DecoderCapacityContract::from_clause_envelope(
                metadata(),
                LlmKvCacheSpec::DEFAULT,
                usize::MAX,
            ),
            Err(TopologyError::ArithmeticOverflow {
                operation: "Hy-MT2 source character token bound"
            })
        ));
    }
}
