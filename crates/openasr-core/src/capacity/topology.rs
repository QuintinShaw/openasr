//! Model-semantic decoder-state capacity planning.
//!
//! This module answers one question only: given the largest legal invocation
//! a session may hand to a model family, what persistent token-scaled decoder
//! state must that invocation be able to address?  It deliberately knows
//! nothing about free RAM/VRAM, device selection, or fallback policy.
//!
//! The separation is important:
//!
//! - a model's RoPE/position span is a mathematical ceiling, not an allocation
//!   request;
//! - the allocation request is the smallest proven upper bound for one legal
//!   invocation;
//! - memory pressure may reject an execution candidate, but may never change
//!   this semantic bound (and therefore never changes a transcript merely
//!   because it ran on a different machine).

use std::num::NonZeroU32;

use thiserror::Error;

use super::{KvBytesPerPosition, KvGeometry, kv_bytes_per_position};
use crate::capacity::decode_schedule::{DecodeScheduleError, greedy_self_kv_positions};
use crate::nn::decoder::LlmKvCacheSpec;

/// Product/session envelope visible before any model buffers are allocated.
///
/// `max_samples` is used instead of floating-point seconds so a frontend can
/// reuse the same integer shape arithmetic as its real encoder.  A caller
/// converting a duration configuration must round *up* to samples before
/// constructing this value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InvocationEnvelope {
    sample_rate_hz: NonZeroU32,
    max_samples: usize,
    max_sequences: NonZeroU32,
    /// Product-approved upper bound for request/carry prompt tokens that may
    /// vary across invocations in this session. Family topologies map this
    /// semantic input budget into their own wrappers and state axes.
    max_prompt_tokens: usize,
}

impl InvocationEnvelope {
    pub(crate) fn new(
        sample_rate_hz: NonZeroU32,
        max_samples: usize,
    ) -> Result<Self, TopologyError> {
        Self::new_with_sequences(sample_rate_hz, max_samples, NonZeroU32::MIN)
    }

    /// Declare sequence concurrency only when this semantic state owner owns
    /// all sequences. Per-job serve-batch plans intentionally use [`Self::new`];
    /// the shared native batch arena quotes its explicit sequence dimension.
    pub(crate) fn new_with_sequences(
        sample_rate_hz: NonZeroU32,
        max_samples: usize,
        max_sequences: NonZeroU32,
    ) -> Result<Self, TopologyError> {
        if max_samples == 0 {
            return Err(TopologyError::EmptyInvocationEnvelope);
        }
        Ok(Self {
            sample_rate_hz,
            max_samples,
            max_sequences,
            max_prompt_tokens: 0,
        })
    }

    /// Construct an exact integer-millisecond product window.  This is the
    /// preferred constructor for family defaults such as 30 s / 60 s.
    pub(crate) fn from_milliseconds(
        sample_rate_hz: NonZeroU32,
        max_duration_ms: NonZeroU32,
    ) -> Result<Self, TopologyError> {
        let max_samples = ceil_mul_div(
            u64::from(sample_rate_hz.get()),
            u64::from(max_duration_ms.get()),
            1_000,
        )?;
        Self::new(
            sample_rate_hz,
            usize::try_from(max_samples).map_err(|_| TopologyError::ArithmeticOverflow {
                operation: "invocation milliseconds to samples",
            })?,
        )
    }

    pub(crate) const fn sample_rate_hz(self) -> NonZeroU32 {
        self.sample_rate_hz
    }

    pub(crate) const fn max_samples(self) -> usize {
        self.max_samples
    }

    pub(crate) const fn max_sequences(self) -> NonZeroU32 {
        self.max_sequences
    }

    pub(crate) const fn max_prompt_tokens(self) -> usize {
        self.max_prompt_tokens
    }

    pub(crate) const fn with_max_prompt_tokens(mut self, max_prompt_tokens: usize) -> Self {
        self.max_prompt_tokens = max_prompt_tokens;
        self
    }

    pub(crate) const fn maximum_invocation(self) -> InvocationShapeInput {
        InvocationShapeInput {
            sample_rate_hz: self.sample_rate_hz,
            samples: self.max_samples,
            sequences: self.max_sequences,
        }
    }
}

/// Exact logical shape input for one invocation/chunk. This is distinct from
/// the session envelope: current decode budgets and masks derive from this
/// value, while resident arenas derive from the envelope's maximum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InvocationShapeInput {
    sample_rate_hz: NonZeroU32,
    samples: usize,
    sequences: NonZeroU32,
}

/// Modality-specific invocation coordinates consumed by the otherwise
/// modality-neutral decoder-state planner.
///
/// Audio ASR families use sample counts. Text-only auxiliary decoders use
/// already-tokenized prompt/prefix/generation axes. Keeping the contract as a
/// type parameter avoids pretending that text tokens are audio samples while
/// preserving one validation/aggregation implementation for every decoder.
pub(crate) trait DecoderInvocationContract {
    type Invocation: Copy;
    type Envelope: Copy;

    fn contains(envelope: Self::Envelope, invocation: Self::Invocation) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AudioInvocationContract {}

impl DecoderInvocationContract for AudioInvocationContract {
    type Invocation = InvocationShapeInput;
    type Envelope = InvocationEnvelope;

    fn contains(envelope: Self::Envelope, invocation: Self::Invocation) -> bool {
        invocation.sample_rate_hz == envelope.sample_rate_hz
            && invocation.samples <= envelope.max_samples
            && invocation.sequences <= envelope.max_sequences
    }
}

/// Exact token coordinates for one text-decoder invocation.
///
/// `prefix_positions` is the reusable source prefix. It may be smaller than
/// `prompt_positions` because generation markers belong to the latter only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TokenInvocationShapeInput {
    prompt_positions: usize,
    prefix_positions: usize,
    max_generated_positions: usize,
    sequences: NonZeroU32,
}

impl TokenInvocationShapeInput {
    pub(crate) fn new(
        prompt_positions: usize,
        prefix_positions: usize,
        max_generated_positions: usize,
    ) -> Result<Self, TopologyError> {
        Self::new_with_sequences(
            prompt_positions,
            prefix_positions,
            max_generated_positions,
            NonZeroU32::MIN,
        )
    }

    /// Exact invocation for an owner whose persistent state contains every
    /// listed sequence concurrently.
    pub(crate) fn new_with_sequences(
        prompt_positions: usize,
        prefix_positions: usize,
        max_generated_positions: usize,
        sequences: NonZeroU32,
    ) -> Result<Self, TopologyError> {
        if prompt_positions == 0 || prefix_positions == 0 || max_generated_positions == 0 {
            return Err(TopologyError::EmptyTokenInvocation);
        }
        if prefix_positions > prompt_positions {
            return Err(TopologyError::PrefixExceedsPrompt {
                prefix_positions,
                prompt_positions,
            });
        }
        Ok(Self {
            prompt_positions,
            prefix_positions,
            max_generated_positions,
            sequences,
        })
    }

    pub(crate) const fn prompt_positions(self) -> usize {
        self.prompt_positions
    }

    pub(crate) const fn prefix_positions(self) -> usize {
        self.prefix_positions
    }

    pub(crate) const fn max_generated_positions(self) -> usize {
        self.max_generated_positions
    }

    pub(crate) const fn sequences(self) -> NonZeroU32 {
        self.sequences
    }
}

/// Stable token bounds known before a text-decoder session is published.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TokenInvocationEnvelope {
    maximum: TokenInvocationShapeInput,
    max_generated_positions: usize,
    max_total_positions: usize,
}

impl TokenInvocationEnvelope {
    pub(crate) fn new(
        max_prompt_positions: usize,
        max_prefix_positions: usize,
        max_generated_positions: usize,
    ) -> Result<Self, TopologyError> {
        let max_total_positions = max_prompt_positions
            .checked_add(max_generated_positions)
            .ok_or(TopologyError::ArithmeticOverflow {
                operation: "text decoder stable prompt plus generation positions",
            })?;
        Self::with_total_position_cap(
            max_prompt_positions,
            max_prefix_positions,
            max_generated_positions,
            max_total_positions,
        )
    }

    /// Construct a correlated envelope where prompt and generation maxima do
    /// not have to occur simultaneously. This preserves an existing full
    /// context API without over-allocating `max_prompt + max_generation` past
    /// the model ceiling.
    pub(crate) fn with_total_position_cap(
        max_prompt_positions: usize,
        max_prefix_positions: usize,
        max_generated_positions: usize,
        max_total_positions: usize,
    ) -> Result<Self, TopologyError> {
        Self::with_total_position_cap_and_sequences(
            max_prompt_positions,
            max_prefix_positions,
            max_generated_positions,
            max_total_positions,
            NonZeroU32::MIN,
        )
    }

    /// Correlated stable bounds for one genuinely multi-sequence state owner.
    pub(crate) fn with_total_position_cap_and_sequences(
        max_prompt_positions: usize,
        max_prefix_positions: usize,
        max_generated_positions: usize,
        max_total_positions: usize,
        max_sequences: NonZeroU32,
    ) -> Result<Self, TopologyError> {
        if max_total_positions == 0
            || max_prompt_positions >= max_total_positions
            || max_generated_positions == 0
        {
            return Err(TopologyError::InvalidTokenEnvelope);
        }
        let generated_at_max_prompt =
            max_generated_positions.min(max_total_positions.saturating_sub(max_prompt_positions));
        Ok(Self {
            maximum: TokenInvocationShapeInput::new_with_sequences(
                max_prompt_positions,
                max_prefix_positions,
                generated_at_max_prompt,
                max_sequences,
            )?,
            max_generated_positions,
            max_total_positions,
        })
    }

    pub(crate) const fn maximum_invocation(self) -> TokenInvocationShapeInput {
        self.maximum
    }

    pub(crate) const fn max_prefix_positions(self) -> usize {
        self.maximum.prefix_positions
    }

    pub(crate) const fn max_total_positions(self) -> usize {
        self.max_total_positions
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenInvocationContract {}

impl DecoderInvocationContract for TokenInvocationContract {
    type Invocation = TokenInvocationShapeInput;
    type Envelope = TokenInvocationEnvelope;

    fn contains(envelope: Self::Envelope, invocation: Self::Invocation) -> bool {
        let maximum = envelope.maximum;
        invocation.prompt_positions <= maximum.prompt_positions
            && invocation.prefix_positions <= maximum.prefix_positions
            && invocation.max_generated_positions <= envelope.max_generated_positions
            && invocation
                .prompt_positions
                .checked_add(invocation.max_generated_positions)
                .is_some_and(|total| total <= envelope.max_total_positions)
            && invocation.sequences <= maximum.sequences
    }
}

impl InvocationShapeInput {
    pub(crate) fn new(sample_rate_hz: NonZeroU32, samples: usize) -> Result<Self, TopologyError> {
        Self::new_with_sequences(sample_rate_hz, samples, NonZeroU32::MIN)
    }

    /// Exact invocation for an owner whose persistent state contains every
    /// listed sequence concurrently.
    pub(crate) fn new_with_sequences(
        sample_rate_hz: NonZeroU32,
        samples: usize,
        sequences: NonZeroU32,
    ) -> Result<Self, TopologyError> {
        if samples == 0 {
            return Err(TopologyError::EmptyInvocation);
        }
        Ok(Self {
            sample_rate_hz,
            samples,
            sequences,
        })
    }

    pub(crate) const fn sample_rate_hz(self) -> NonZeroU32 {
        self.sample_rate_hz
    }

    pub(crate) const fn samples(self) -> usize {
        self.samples
    }

    pub(crate) const fn sequences(self) -> NonZeroU32 {
        self.sequences
    }
}

/// Semantic role of one independently-sized persistent decoder-state stream.
///
/// Self- and cross-attention positions are intentionally distinct.  Adding
/// them together would over-allocate encoder-decoder models: the two tensors
/// coexist, but neither tensor has a position axis equal to their sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum StateKind {
    SelfAttentionKv,
    CrossAttentionKv,
}

/// Why a position count is safe.  There is no percentage safety multiplier:
/// an exact runtime counter is already exact, while a conservative integer
/// upper bound records the architectural padding/rounding it includes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PositionBoundProof {
    Exact,
    Conservative { basis: &'static str },
}

/// Bytes occupied by one state stream, split by ownership/lifetime.
///
/// `host` covers Rust-side history retained in ordinary memory. `resident`
/// covers the backend buffer whose lifetime follows the reusable graph/arena.
/// On unified-memory systems both ultimately charge the same physical pool;
/// the physical footprint planner decides that later.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub(crate) struct StateBytes {
    pub host: u64,
    pub resident: u64,
}

impl StateBytes {
    pub(crate) fn checked_add(self, other: Self) -> Result<Self, TopologyError> {
        Ok(Self {
            host: self
                .host
                .checked_add(other.host)
                .ok_or(TopologyError::ArithmeticOverflow {
                    operation: "host state byte sum",
                })?,
            resident: self.resident.checked_add(other.resident).ok_or(
                TopologyError::ArithmeticOverflow {
                    operation: "resident state byte sum",
                },
            )?,
        })
    }
}

/// One independently-sized token-scaled persistent decoder-state allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StateDemand {
    pub id: &'static str,
    pub kind: StateKind,
    pub positions: usize,
    /// Mathematical/family safety ceiling. It validates `positions`; it is
    /// never substituted for `positions` as the allocation size.
    pub hard_position_cap: usize,
    pub bytes: StateBytes,
    pub proof: PositionBoundProof,
}

impl StateDemand {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_llm_kv_geometry(
        id: &'static str,
        kind: StateKind,
        positions: usize,
        hard_position_cap: usize,
        geometry: KvGeometry,
        spec: LlmKvCacheSpec,
        sequences: usize,
        proof: PositionBoundProof,
    ) -> Result<Self, TopologyError> {
        if sequences == 0 {
            return Err(TopologyError::ZeroSequenceCount { state: id });
        }
        let per_position = kv_bytes_per_position(&geometry, spec)
            .map_err(|reason| TopologyError::InvalidStateLayout { state: id, reason })?;
        let scale = u64::try_from(positions)
            .ok()
            .and_then(|positions| {
                u64::try_from(sequences)
                    .ok()
                    .and_then(|sequences| positions.checked_mul(sequences))
            })
            .ok_or(TopologyError::ArithmeticOverflow {
                operation: "KV positions times sequences",
            })?;
        let bytes = StateBytes {
            host: per_position.host.checked_mul(scale).ok_or(
                TopologyError::ArithmeticOverflow {
                    operation: "host KV byte count",
                },
            )?,
            resident: per_position.resident.checked_mul(scale).ok_or(
                TopologyError::ArithmeticOverflow {
                    operation: "resident KV byte count",
                },
            )?,
        };
        Self::new(id, kind, positions, hard_position_cap, bytes, proof)
    }

    pub(crate) fn new(
        id: &'static str,
        kind: StateKind,
        positions: usize,
        hard_position_cap: usize,
        bytes: StateBytes,
        proof: PositionBoundProof,
    ) -> Result<Self, TopologyError> {
        if id.trim().is_empty() {
            return Err(TopologyError::EmptyStateId);
        }
        if hard_position_cap == 0 {
            return Err(TopologyError::ZeroPositionCap { state: id });
        }
        if positions == 0 {
            return Err(TopologyError::ZeroPositions { state: id });
        }
        if positions > hard_position_cap {
            return Err(TopologyError::PositionCapExceeded {
                state: id,
                required: positions,
                hard_cap: hard_position_cap,
            });
        }
        Ok(Self {
            id,
            kind,
            positions,
            hard_position_cap,
            bytes,
            proof,
        })
    }
}

/// Which proof obligation a family demand oracle is answering.
///
/// The variants are intentionally passed through one required method. This
/// makes every implementation acknowledge both the current invocation and
/// the reusable session envelope; the shared planner never assumes that a
/// frontend, prompt topology, batching bucket, or correlated limit is
/// monotone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecoderStateDemandScope<I, E> {
    ExactInvocation(I),
    StableEnvelope(E),
}

/// Family-owned token topology. Implementations may use exact frontend shape
/// oracles, prompt templates, marker tracks, and generation-budget rules; the
/// shared planner only validates and aggregates their independent demands.
pub(crate) trait DecoderStateTopology<C = AudioInvocationContract>
where
    C: DecoderInvocationContract,
{
    fn demands(
        &self,
        scope: DecoderStateDemandScope<C::Invocation, C::Envelope>,
    ) -> Result<Vec<StateDemand>, TopologyError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StateAllocationDemand {
    pub logical: StateDemand,
    pub reserve: StateDemand,
}

/// Runtime-facing view of one independently addressed decoder-state axis.
///
/// A semantic [`StateKind`] is deliberately not an allocation key: future
/// architectures may own multiple streams of the same kind.  Consumers must
/// name the family-owned stable id and verify the expected kind when crossing
/// from the generic plan into a concrete runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct StatePositionAxis {
    pub id: &'static str,
    pub kind: StateKind,
    pub logical_positions: usize,
    pub resident_positions: usize,
    pub hard_position_cap: usize,
    pub logical_bytes: StateBytes,
    pub reserve_bytes: StateBytes,
}

impl StateAllocationDemand {
    fn new(logical: StateDemand, reserve: StateDemand) -> Result<Self, TopologyError> {
        if logical.id != reserve.id || logical.kind != reserve.kind {
            return Err(TopologyError::StateSetMismatch {
                logical: logical.id,
                reserve: reserve.id,
            });
        }
        if logical.hard_position_cap != reserve.hard_position_cap {
            return Err(TopologyError::StateCapMismatch {
                state: logical.id,
                logical: logical.hard_position_cap,
                reserve: reserve.hard_position_cap,
            });
        }
        if logical.positions > reserve.positions {
            return Err(TopologyError::ReserveDoesNotCoverLogical {
                state: logical.id,
                logical: logical.positions,
                reserve: reserve.positions,
            });
        }
        if logical.bytes.host > reserve.bytes.host
            || logical.bytes.resident > reserve.bytes.resident
        {
            return Err(TopologyError::ReserveBytesDoNotCoverLogical {
                state: logical.id,
                logical: logical.bytes,
                reserve: reserve.bytes,
            });
        }
        Ok(Self { logical, reserve })
    }
}

/// Validated logical+reserve aggregate returned by the unified planner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecoderStatePlan {
    allocations: Vec<StateAllocationDemand>,
}

impl DecoderStatePlan {
    pub(crate) fn build<C, T>(
        topology: &T,
        invocation: C::Invocation,
        envelope: C::Envelope,
    ) -> Result<Self, TopologyError>
    where
        C: DecoderInvocationContract,
        T: DecoderStateTopology<C>,
    {
        if !C::contains(envelope, invocation) {
            return Err(TopologyError::InvocationOutsideEnvelope);
        }
        let logical = topology.demands(DecoderStateDemandScope::ExactInvocation(invocation))?;
        let reserve = topology.demands(DecoderStateDemandScope::StableEnvelope(envelope))?;
        Self::from_demand_sets(logical, reserve)
    }

    fn from_demand_sets(
        logical: Vec<StateDemand>,
        reserve: Vec<StateDemand>,
    ) -> Result<Self, TopologyError> {
        if logical.is_empty() || reserve.is_empty() {
            return Err(TopologyError::EmptyStateSet);
        }
        if logical.len() != reserve.len() {
            return Err(TopologyError::StateCountMismatch {
                logical: logical.len(),
                reserve: reserve.len(),
            });
        }
        validate_unique_state_ids(&logical)?;
        validate_unique_state_ids(&reserve)?;
        let mut logical = logical;
        let mut allocations = Vec::with_capacity(logical.len());
        let mut logical_bytes = StateBytes::default();
        let mut reserve_bytes = StateBytes::default();
        for reserve in reserve {
            let logical_index = logical
                .iter()
                .position(|logical| logical.id == reserve.id)
                .ok_or(TopologyError::StateIdSetMismatch { id: reserve.id })?;
            let logical = logical.swap_remove(logical_index);
            logical_bytes = logical_bytes.checked_add(logical.bytes)?;
            reserve_bytes = reserve_bytes.checked_add(reserve.bytes)?;
            allocations.push(StateAllocationDemand::new(logical, reserve)?);
        }
        // The checked aggregates are proof obligations even though physical
        // admission consumes the independently-owned streams rather than one
        // topology-level byte bucket.
        let _validated_aggregate_bytes = (logical_bytes, reserve_bytes);
        Ok(Self { allocations })
    }

    /// Rebind an exact invocation's logical demands to an already-admitted
    /// session's resident demands.
    ///
    /// Snapshot streaming changes the logical audio shape on every decode but
    /// must keep the one resident arena allocated at session construction.
    /// Re-running a family planner gives us the exact logical demand; this
    /// operation deliberately discards that fresh plan's resident side and
    /// proves the logical side fits the original session reservation. The
    /// normal demand-set validation keeps IDs, kinds, hard caps, positions,
    /// and bytes fail-closed.
    pub(crate) fn with_resident_demands_from(
        &self,
        resident_template: &Self,
    ) -> Result<Self, TopologyError> {
        Self::from_demand_sets(
            self.allocations
                .iter()
                .map(|allocation| allocation.logical)
                .collect(),
            resident_template
                .allocations
                .iter()
                .map(|allocation| allocation.reserve)
                .collect(),
        )
    }

    /// Test convenience for inspecting an audio envelope at its largest
    /// invocation. This still calls the family's distinct exact and stable
    /// branches; it is not a production monotonicity fallback.
    #[cfg(test)]
    pub(crate) fn for_envelope<T>(
        topology: &T,
        envelope: InvocationEnvelope,
    ) -> Result<Self, TopologyError>
    where
        T: DecoderStateTopology<AudioInvocationContract>,
    {
        Self::build::<AudioInvocationContract, _>(topology, envelope.maximum_invocation(), envelope)
    }

    pub(crate) fn allocations(&self) -> &[StateAllocationDemand] {
        &self.allocations
    }

    #[cfg(test)]
    pub(crate) fn reserve_bytes(&self) -> StateBytes {
        self.allocations
            .iter()
            .try_fold(StateBytes::default(), |total, allocation| {
                total.checked_add(allocation.reserve.bytes)
            })
            .expect("plan construction already validated aggregate reserve bytes")
    }

    /// Look up one independently-sized state stream by its stable topology
    /// identity. IDs, rather than broad semantic kinds, are the planner's
    /// primary key so future models may legitimately expose multiple streams
    /// of the same kind (for example per-block recurrent states).
    pub(crate) fn allocation(&self, id: &str) -> Option<&StateAllocationDemand> {
        self.allocations
            .iter()
            .find(|allocation| allocation.logical.id == id)
    }

    pub(crate) fn position_axis(
        &self,
        id: &'static str,
        expected_kind: StateKind,
    ) -> Result<StatePositionAxis, TopologyError> {
        let allocation = self
            .allocation(id)
            .ok_or(TopologyError::StateIdMissing { id })?;
        if allocation.logical.kind != expected_kind || allocation.reserve.kind != expected_kind {
            return Err(TopologyError::StateKindMismatch {
                id,
                expected: expected_kind,
                actual: allocation.logical.kind,
            });
        }
        Ok(StatePositionAxis {
            id,
            kind: expected_kind,
            logical_positions: allocation.logical.positions,
            resident_positions: allocation.reserve.positions,
            hard_position_cap: allocation.reserve.hard_position_cap,
            logical_bytes: allocation.logical.bytes,
            reserve_bytes: allocation.reserve.bytes,
        })
    }

    #[cfg(test)]
    pub(crate) fn reserve_positions_by_id(&self, id: &str) -> Option<usize> {
        self.allocation(id)
            .map(|allocation| allocation.reserve.positions)
    }

    /// Compatibility lookup for current single-stream adapters. Returns
    /// `None` when the kind is absent *or ambiguous*; new adapters should use
    /// the stable state ID instead.
    #[cfg(test)]
    pub(crate) fn logical_positions(&self, kind: StateKind) -> Option<usize> {
        unique_allocation_by_kind(&self.allocations, kind)
            .map(|allocation| allocation.logical.positions)
    }

    /// Compatibility lookup for current single-stream adapters. Returns
    /// `None` when the kind is absent *or ambiguous*; new adapters should use
    /// the stable state ID instead.
    #[cfg(test)]
    pub(crate) fn reserve_positions(&self, kind: StateKind) -> Option<usize> {
        unique_allocation_by_kind(&self.allocations, kind)
            .map(|allocation| allocation.reserve.positions)
    }
}

#[cfg(test)]
fn unique_allocation_by_kind(
    allocations: &[StateAllocationDemand],
    kind: StateKind,
) -> Option<&StateAllocationDemand> {
    let mut matches = allocations
        .iter()
        .filter(|allocation| allocation.logical.kind == kind);
    let allocation = matches.next()?;
    matches.next().is_none().then_some(allocation)
}

fn validate_unique_state_ids(demands: &[StateDemand]) -> Result<(), TopologyError> {
    for (index, demand) in demands.iter().enumerate() {
        if demands[..index]
            .iter()
            .any(|existing| existing.id == demand.id)
        {
            return Err(TopologyError::DuplicateStateId { id: demand.id });
        }
    }
    Ok(())
}

/// The unique minimum safe self-KV span for a causal-prefix decoder using the
/// shared greedy schedule: prompt prefill plus incremental cache writes.
pub(crate) fn causal_prefix_positions(
    prompt_positions: usize,
    max_generated_positions: usize,
) -> Result<usize, TopologyError> {
    greedy_self_kv_positions(prompt_positions, max_generated_positions)
        .map_err(map_decode_schedule_error)
}

/// Validate the semantic context coordinate before deriving physical greedy
/// cache occupancy. A physical span of `P + G - 1` fitting in `C` does *not*
/// prove that the model-visible request `P + G` fits in `C`.
pub(crate) fn causal_prefix_positions_with_context_cap(
    state: &'static str,
    prompt_positions: usize,
    max_generated_positions: usize,
    context_position_cap: usize,
) -> Result<usize, TopologyError> {
    if max_generated_positions == 0 {
        return Err(TopologyError::EmptyGenerationBudget);
    }
    let semantic_positions = prompt_positions
        .checked_add(max_generated_positions)
        .ok_or(TopologyError::ArithmeticOverflow {
            operation: "causal semantic prompt plus generation positions",
        })?;
    if semantic_positions > context_position_cap {
        return Err(TopologyError::SemanticContextCapExceeded {
            state,
            prompt_positions,
            generated_positions: max_generated_positions,
            required: semantic_positions,
            hard_cap: context_position_cap,
        });
    }
    causal_prefix_positions(prompt_positions, max_generated_positions)
}

fn map_decode_schedule_error(error: DecodeScheduleError) -> TopologyError {
    match error {
        DecodeScheduleError::EmptyGenerationBudget => TopologyError::EmptyGenerationBudget,
        DecodeScheduleError::PositionOverflow => TopologyError::ArithmeticOverflow {
            operation: "causal prompt plus incremental cache writes",
        },
    }
}

/// `ceil(value * numerator / denominator)` using a wide intermediate.
///
/// Frontend counters and duration-to-sample conversion use this helper rather
/// than floating point followed by truncation.  It is monotone and either
/// returns the conservative integer upper bound or fails closed on overflow.
pub(crate) fn ceil_mul_div(
    value: u64,
    numerator: u64,
    denominator: u64,
) -> Result<u64, TopologyError> {
    if denominator == 0 {
        return Err(TopologyError::DivisionByZero);
    }
    let product = u128::from(value).checked_mul(u128::from(numerator)).ok_or(
        TopologyError::ArithmeticOverflow {
            operation: "ceil multiply",
        },
    )?;
    let denominator = u128::from(denominator);
    let quotient =
        product
            .checked_add(denominator - 1)
            .ok_or(TopologyError::ArithmeticOverflow {
                operation: "ceil numerator adjustment",
            })?
            / denominator;
    u64::try_from(quotient).map_err(|_| TopologyError::ArithmeticOverflow {
        operation: "ceil result conversion",
    })
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum TopologyError {
    #[error("decoder invocation must contain at least one sample")]
    EmptyInvocation,
    #[error("decoder invocation envelope must contain at least one sample")]
    EmptyInvocationEnvelope,
    #[error("text decoder invocation positions must all be positive")]
    EmptyTokenInvocation,
    #[error("causal decoder generation budget must contain at least one step")]
    EmptyGenerationBudget,
    #[error("text decoder invocation envelope is invalid")]
    InvalidTokenEnvelope,
    #[error("text decoder prefix span {prefix_positions} exceeds prompt span {prompt_positions}")]
    PrefixExceedsPrompt {
        prefix_positions: usize,
        prompt_positions: usize,
    },
    #[error("decoder invocation lies outside its declared session envelope")]
    InvocationOutsideEnvelope,
    #[error(
        "decoder invocation contains {required_samples} samples, exceeding the family limit {max_samples}"
    )]
    InvocationSampleLimitExceeded {
        required_samples: usize,
        max_samples: usize,
    },
    #[error("decoder topology requires {expected_hz} Hz audio, got {actual_hz} Hz")]
    UnsupportedSampleRate { expected_hz: u32, actual_hz: u32 },
    #[error("decoder state id must not be empty")]
    EmptyStateId,
    #[error("decoder state '{state}' must have a positive hard position cap")]
    ZeroPositionCap { state: &'static str },
    #[error("decoder state '{state}' must contain at least one position")]
    ZeroPositions { state: &'static str },
    #[error("decoder state '{state}' must have a positive sequence count")]
    ZeroSequenceCount { state: &'static str },
    #[error(
        "decoder state '{state}' requires {required} positions, exceeding its hard cap {hard_cap}"
    )]
    PositionCapExceeded {
        state: &'static str,
        required: usize,
        hard_cap: usize,
    },
    #[error(
        "decoder state '{state}' semantic context requires prompt {prompt_positions} + generation {generated_positions} = {required} positions, exceeding model cap {hard_cap}"
    )]
    SemanticContextCapExceeded {
        state: &'static str,
        prompt_positions: usize,
        generated_positions: usize,
        required: usize,
        hard_cap: usize,
    },
    #[error("logical/reserve decoder state counts differ: logical={logical}, reserve={reserve}")]
    StateCountMismatch { logical: usize, reserve: usize },
    #[error("a planned decoder-state topology must declare at least one state stream")]
    EmptyStateSet,
    #[error("logical/reserve decoder state sets differ: logical='{logical}', reserve='{reserve}'")]
    StateSetMismatch {
        logical: &'static str,
        reserve: &'static str,
    },
    #[error("decoder topology returned duplicate state id '{id}'")]
    DuplicateStateId { id: &'static str },
    #[error("decoder-state plan is missing required allocation '{id}'")]
    StateIdMissing { id: &'static str },
    #[error("decoder-state allocation '{id}' has kind {actual:?}, expected {expected:?}")]
    StateKindMismatch {
        id: &'static str,
        expected: StateKind,
        actual: StateKind,
    },
    #[error("logical/reserve decoder state id sets differ at '{id}'")]
    StateIdSetMismatch { id: &'static str },
    #[error("decoder state '{state}' hard caps differ: logical={logical}, reserve={reserve}")]
    StateCapMismatch {
        state: &'static str,
        logical: usize,
        reserve: usize,
    },
    #[error(
        "decoder state '{state}' reserve does not cover logical demand: logical={logical}, reserve={reserve}"
    )]
    ReserveDoesNotCoverLogical {
        state: &'static str,
        logical: usize,
        reserve: usize,
    },
    #[error(
        "decoder state '{state}' reserve bytes do not cover logical bytes: logical={logical:?}, reserve={reserve:?}"
    )]
    ReserveBytesDoNotCoverLogical {
        state: &'static str,
        logical: StateBytes,
        reserve: StateBytes,
    },
    #[error("decoder state '{state}' has an invalid storage layout: {reason}")]
    InvalidStateLayout { state: &'static str, reason: String },
    #[error("decoder-state capacity arithmetic overflowed during {operation}")]
    ArithmeticOverflow { operation: &'static str },
    #[error("decoder-state capacity arithmetic attempted division by zero")]
    DivisionByZero,
    #[error("decoder-state topology is unavailable: {reason}")]
    Unavailable { reason: String },
}

impl From<KvBytesPerPosition> for StateBytes {
    fn from(value: KvBytesPerPosition) -> Self {
        Self {
            host: value.host,
            resident: value.resident,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::GgmlKvElementType;

    struct CausalFixture;

    impl DecoderStateTopology for CausalFixture {
        fn demands(
            &self,
            scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
        ) -> Result<Vec<StateDemand>, TopologyError> {
            let invocation = match scope {
                DecoderStateDemandScope::ExactInvocation(invocation) => invocation,
                DecoderStateDemandScope::StableEnvelope(envelope) => envelope.maximum_invocation(),
            };
            let positions = causal_prefix_positions(472, 818)?;
            Ok(vec![StateDemand::from_llm_kv_geometry(
                "decoder.self_kv",
                StateKind::SelfAttentionKv,
                positions,
                8_192,
                KvGeometry {
                    n_layers: 28,
                    kv_heads: 8,
                    head_dim: 128,
                },
                LlmKvCacheSpec::DEFAULT,
                invocation.sequences().get() as usize,
                PositionBoundProof::Exact,
            )?])
        }
    }

    #[test]
    fn causal_prefix_is_the_unique_minimum_safe_span() {
        assert_eq!(causal_prefix_positions(472, 818).unwrap(), 1_289);
        assert_eq!(
            causal_prefix_positions(472, 0),
            Err(TopologyError::EmptyGenerationBudget)
        );
    }

    #[test]
    fn semantic_context_is_validated_separately_from_physical_kv_span() {
        assert_eq!(
            causal_prefix_positions_with_context_cap("decoder.self_kv", 472, 818, 1_290),
            Ok(1_289)
        );
        // The physical span is exactly 1_290 and would fit that many KV rows,
        // but the model-visible request needs 1_291 positions and is illegal.
        assert!(matches!(
            causal_prefix_positions_with_context_cap("decoder.self_kv", 472, 819, 1_290),
            Err(TopologyError::SemanticContextCapExceeded {
                required: 1_291,
                hard_cap: 1_290,
                ..
            })
        ));
        assert!(matches!(
            causal_prefix_positions_with_context_cap("decoder.self_kv", usize::MAX, 1, usize::MAX,),
            Err(TopologyError::ArithmeticOverflow {
                operation: "causal semantic prompt plus generation positions"
            })
        ));
    }

    #[test]
    fn stable_scope_is_family_owned_and_never_inferred_from_maximum_invocation() {
        struct NonMonotoneFixture;
        impl DecoderStateTopology for NonMonotoneFixture {
            fn demands(
                &self,
                scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
            ) -> Result<Vec<StateDemand>, TopologyError> {
                let positions = match scope {
                    DecoderStateDemandScope::ExactInvocation(invocation) => {
                        invocation.samples() / 1_000
                    }
                    DecoderStateDemandScope::StableEnvelope(_) => 42,
                };
                Ok(vec![StateDemand::new(
                    "decoder.non_monotone",
                    StateKind::SelfAttentionKv,
                    positions,
                    100,
                    StateBytes {
                        host: positions as u64,
                        resident: positions as u64,
                    },
                    PositionBoundProof::Exact,
                )?])
            }
        }

        let rate = NonZeroU32::new(16_000).unwrap();
        let invocation = InvocationShapeInput::new(rate, 1_000).unwrap();
        let envelope = InvocationEnvelope::new(rate, 2_000).unwrap();
        let plan = DecoderStatePlan::build(&NonMonotoneFixture, invocation, envelope).unwrap();
        let axis = plan
            .position_axis("decoder.non_monotone", StateKind::SelfAttentionKv)
            .unwrap();
        assert_eq!((axis.logical_positions, axis.resident_positions), (1, 42));
    }

    #[test]
    fn streaming_logical_rebind_keeps_one_resident_envelope_and_fails_closed() {
        struct StreamingFixture;
        impl DecoderStateTopology for StreamingFixture {
            fn demands(
                &self,
                scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
            ) -> Result<Vec<StateDemand>, TopologyError> {
                let positions = match scope {
                    DecoderStateDemandScope::ExactInvocation(invocation) => {
                        invocation.samples() / 1_000
                    }
                    DecoderStateDemandScope::StableEnvelope(envelope) => {
                        envelope.max_samples() / 1_000
                    }
                };
                Ok(vec![StateDemand::new(
                    "decoder.streaming",
                    StateKind::CrossAttentionKv,
                    positions,
                    100,
                    StateBytes {
                        host: 0,
                        resident: positions as u64,
                    },
                    PositionBoundProof::Exact,
                )?])
            }
        }

        let rate = NonZeroU32::new(16_000).unwrap();
        let session_envelope = InvocationEnvelope::new(rate, 30_000).unwrap();
        let resident = DecoderStatePlan::build::<AudioInvocationContract, _>(
            &StreamingFixture,
            session_envelope.maximum_invocation(),
            session_envelope,
        )
        .unwrap();
        let exact = DecoderStatePlan::build::<AudioInvocationContract, _>(
            &StreamingFixture,
            InvocationShapeInput::new(rate, 2_000).unwrap(),
            session_envelope,
        )
        .unwrap();
        let rebound = exact.with_resident_demands_from(&resident).unwrap();
        let axis = rebound
            .position_axis("decoder.streaming", StateKind::CrossAttentionKv)
            .unwrap();
        assert_eq!((axis.logical_positions, axis.resident_positions), (2, 30));

        let larger_envelope = InvocationEnvelope::new(rate, 40_000).unwrap();
        let too_large = DecoderStatePlan::build::<AudioInvocationContract, _>(
            &StreamingFixture,
            larger_envelope.maximum_invocation(),
            larger_envelope,
        )
        .unwrap();
        assert!(matches!(
            too_large.with_resident_demands_from(&resident),
            Err(TopologyError::ReserveDoesNotCoverLogical {
                state: "decoder.streaming",
                logical: 40,
                reserve: 30,
            })
        ));
    }

    #[test]
    fn sequence_concurrency_scales_bytes_but_not_each_sequence_position_axis() {
        let rate = NonZeroU32::new(16_000).unwrap();
        let invocation =
            InvocationShapeInput::new_with_sequences(rate, 16_000, NonZeroU32::new(2).unwrap())
                .unwrap();
        let envelope =
            InvocationEnvelope::new_with_sequences(rate, 16_000, NonZeroU32::new(4).unwrap())
                .unwrap();
        let plan = DecoderStatePlan::build(&CausalFixture, invocation, envelope).unwrap();
        let axis = plan
            .position_axis("decoder.self_kv", StateKind::SelfAttentionKv)
            .unwrap();
        assert_eq!(axis.logical_positions, axis.resident_positions);
        assert_eq!(axis.logical_bytes.host * 2, axis.reserve_bytes.host);
        assert_eq!(axis.logical_bytes.resident * 2, axis.reserve_bytes.resident);

        let outside =
            InvocationShapeInput::new_with_sequences(rate, 16_000, NonZeroU32::new(5).unwrap())
                .unwrap();
        assert_eq!(
            DecoderStatePlan::build(&CausalFixture, outside, envelope),
            Err(TopologyError::InvocationOutsideEnvelope)
        );
    }

    #[test]
    fn independent_state_streams_are_not_position_summed() {
        struct EncoderDecoderFixture;
        impl DecoderStateTopology for EncoderDecoderFixture {
            fn demands(
                &self,
                _scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
            ) -> Result<Vec<StateDemand>, TopologyError> {
                Ok(vec![
                    StateDemand::new(
                        "decoder.self_kv",
                        StateKind::SelfAttentionKv,
                        448,
                        448,
                        StateBytes {
                            host: 0,
                            resident: 4_096,
                        },
                        PositionBoundProof::Exact,
                    )?,
                    StateDemand::new(
                        "decoder.cross_kv",
                        StateKind::CrossAttentionKv,
                        1_500,
                        1_500,
                        StateBytes {
                            host: 0,
                            resident: 8_192,
                        },
                        PositionBoundProof::Exact,
                    )?,
                ])
            }
        }

        let envelope = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(30_000).unwrap(),
        )
        .unwrap();
        let plan = DecoderStatePlan::for_envelope(&EncoderDecoderFixture, envelope).unwrap();
        assert_eq!(
            plan.reserve_positions(StateKind::SelfAttentionKv),
            Some(448)
        );
        assert_eq!(
            plan.reserve_positions(StateKind::CrossAttentionKv),
            Some(1_500)
        );
        assert!(
            !plan
                .allocations()
                .iter()
                .any(|allocation| allocation.reserve.positions == 1_948)
        );
    }

    #[test]
    fn logical_and_reserve_streams_match_by_stable_id_not_vector_order() {
        struct ReorderedFixture;
        impl DecoderStateTopology for ReorderedFixture {
            fn demands(
                &self,
                scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
            ) -> Result<Vec<StateDemand>, TopologyError> {
                let invocation = match scope {
                    DecoderStateDemandScope::ExactInvocation(invocation) => invocation,
                    DecoderStateDemandScope::StableEnvelope(envelope) => {
                        envelope.maximum_invocation()
                    }
                };
                let self_kv = StateDemand::new(
                    "decoder.self_kv",
                    StateKind::SelfAttentionKv,
                    invocation.samples(),
                    64_000,
                    StateBytes {
                        host: 0,
                        resident: invocation.samples() as u64,
                    },
                    PositionBoundProof::Exact,
                )?;
                let cross_kv = StateDemand::new(
                    "decoder.cross_kv",
                    StateKind::CrossAttentionKv,
                    invocation.samples() / 2,
                    32_000,
                    StateBytes {
                        host: 0,
                        resident: (invocation.samples() / 2) as u64,
                    },
                    PositionBoundProof::Exact,
                )?;
                if invocation.samples() == 16_000 {
                    Ok(vec![self_kv, cross_kv])
                } else {
                    Ok(vec![cross_kv, self_kv])
                }
            }
        }

        let rate = NonZeroU32::new(16_000).unwrap();
        let invocation = InvocationShapeInput::new(rate, 16_000).unwrap();
        let envelope = InvocationEnvelope::new(rate, 32_000).unwrap();
        let plan = DecoderStatePlan::build(&ReorderedFixture, invocation, envelope).unwrap();
        assert_eq!(
            plan.reserve_positions(StateKind::SelfAttentionKv),
            Some(32_000)
        );
        assert_eq!(
            plan.reserve_positions(StateKind::CrossAttentionKv),
            Some(16_000)
        );
    }

    #[test]
    fn stable_ids_are_primary_keys_and_same_kind_streams_are_legal() {
        struct MultiStreamFixture;
        impl DecoderStateTopology for MultiStreamFixture {
            fn demands(
                &self,
                scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
            ) -> Result<Vec<StateDemand>, TopologyError> {
                let invocation = match scope {
                    DecoderStateDemandScope::ExactInvocation(invocation) => invocation,
                    DecoderStateDemandScope::StableEnvelope(envelope) => {
                        envelope.maximum_invocation()
                    }
                };
                let demand = |id, positions| {
                    StateDemand::new(
                        id,
                        StateKind::SelfAttentionKv,
                        positions,
                        64_000,
                        StateBytes {
                            host: 0,
                            resident: positions as u64,
                        },
                        PositionBoundProof::Exact,
                    )
                };
                Ok(vec![
                    demand("decoder.recurrent.block_0", invocation.samples())?,
                    demand("decoder.recurrent.block_1", invocation.samples() / 2)?,
                ])
            }
        }

        let rate = NonZeroU32::new(16_000).unwrap();
        let invocation = InvocationShapeInput::new(rate, 16_000).unwrap();
        let envelope = InvocationEnvelope::new(rate, 32_000).unwrap();
        let plan = DecoderStatePlan::build(&MultiStreamFixture, invocation, envelope).unwrap();

        assert_eq!(
            plan.reserve_positions_by_id("decoder.recurrent.block_0"),
            Some(32_000)
        );
        assert_eq!(
            plan.reserve_positions_by_id("decoder.recurrent.block_1"),
            Some(16_000)
        );
        assert_eq!(plan.reserve_positions(StateKind::SelfAttentionKv), None);
        let block_1 = plan
            .position_axis("decoder.recurrent.block_1", StateKind::SelfAttentionKv)
            .unwrap();
        assert_eq!(block_1.logical_positions, 8_000);
        assert_eq!(block_1.resident_positions, 16_000);
        assert!(matches!(
            plan.position_axis("decoder.recurrent.block_1", StateKind::CrossAttentionKv),
            Err(TopologyError::StateKindMismatch { .. })
        ));
        assert!(matches!(
            plan.position_axis("decoder.missing", StateKind::SelfAttentionKv),
            Err(TopologyError::StateIdMissing { .. })
        ));
    }

    #[test]
    fn duplicate_state_ids_fail_closed_even_when_kinds_differ() {
        struct DuplicateIdFixture;
        impl DecoderStateTopology for DuplicateIdFixture {
            fn demands(
                &self,
                _scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
            ) -> Result<Vec<StateDemand>, TopologyError> {
                Ok(vec![
                    StateDemand::new(
                        "decoder.state",
                        StateKind::SelfAttentionKv,
                        1,
                        1,
                        StateBytes::default(),
                        PositionBoundProof::Exact,
                    )?,
                    StateDemand::new(
                        "decoder.state",
                        StateKind::CrossAttentionKv,
                        1,
                        1,
                        StateBytes::default(),
                        PositionBoundProof::Exact,
                    )?,
                ])
            }
        }

        let envelope = InvocationEnvelope::new(NonZeroU32::new(16_000).unwrap(), 16_000).unwrap();
        assert!(matches!(
            DecoderStatePlan::for_envelope(&DuplicateIdFixture, envelope),
            Err(TopologyError::DuplicateStateId {
                id: "decoder.state"
            })
        ));
    }

    #[test]
    fn reserve_must_cover_physical_bytes_as_well_as_positions() {
        let logical = StateDemand::new(
            "decoder.self_kv",
            StateKind::SelfAttentionKv,
            10,
            20,
            StateBytes {
                host: 10,
                resident: 20,
            },
            PositionBoundProof::Exact,
        )
        .unwrap();
        let reserve = StateDemand::new(
            "decoder.self_kv",
            StateKind::SelfAttentionKv,
            20,
            20,
            StateBytes {
                host: 20,
                resident: 19,
            },
            PositionBoundProof::Exact,
        )
        .unwrap();
        assert!(matches!(
            StateAllocationDemand::new(logical, reserve),
            Err(TopologyError::ReserveBytesDoNotCoverLogical { .. })
        ));
    }

    #[test]
    fn position_cap_is_validation_not_allocation_size() {
        let demand = StateDemand::new(
            "decoder.self_kv",
            StateKind::SelfAttentionKv,
            1_290,
            131_072,
            StateBytes::default(),
            PositionBoundProof::Exact,
        )
        .unwrap();
        assert_eq!(demand.positions, 1_290);
        assert_eq!(demand.hard_position_cap, 131_072);
    }

    #[test]
    fn plan_uses_exact_geometry_bytes() {
        let envelope = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(30_000).unwrap(),
        )
        .unwrap();
        let plan = DecoderStatePlan::for_envelope(&CausalFixture, envelope).unwrap();
        // 28 * 2 * 8 rows; f32 host row=512 B, f16 resident row=256 B.
        assert_eq!(plan.reserve_bytes().host, 1_289 * 448 * 512);
        assert_eq!(plan.reserve_bytes().resident, 1_289 * 448 * 256);
    }

    #[test]
    fn q8_layout_is_accepted_without_an_empirical_margin() {
        let demand = StateDemand::from_llm_kv_geometry(
            "decoder.self_kv",
            StateKind::SelfAttentionKv,
            1_290,
            8_192,
            KvGeometry {
                n_layers: 28,
                kv_heads: 8,
                head_dim: 128,
            },
            LlmKvCacheSpec {
                host: GgmlKvElementType::Q8_0,
                resident: GgmlKvElementType::Q8_0,
            },
            1,
            PositionBoundProof::Exact,
        )
        .unwrap();
        assert_eq!(demand.positions, 1_290);
        assert_eq!(demand.bytes.resident, 1_290 * 448 * 136);
    }

    #[test]
    fn ceil_mul_div_handles_subsecond_and_large_inputs_without_float_rounding() {
        assert_eq!(ceil_mul_div(16_000, 1, 1_000).unwrap(), 16);
        assert_eq!(ceil_mul_div(16_000, 1_001, 1_000).unwrap(), 16_016);
        assert_eq!(
            ceil_mul_div(u32::MAX.into(), u32::MAX.into(), 7).unwrap(),
            2_635_249_152_159_945_290
        );
        assert_eq!(ceil_mul_div(1, 1, 0), Err(TopologyError::DivisionByZero));
    }

    #[test]
    fn one_second_and_sixty_second_envelopes_are_exact_sample_counts() {
        let rate = NonZeroU32::new(16_000).unwrap();
        let one =
            InvocationEnvelope::from_milliseconds(rate, NonZeroU32::new(1_000).unwrap()).unwrap();
        let sixty =
            InvocationEnvelope::from_milliseconds(rate, NonZeroU32::new(60_000).unwrap()).unwrap();
        assert_eq!(one.max_samples(), 16_000);
        assert_eq!(sixty.max_samples(), 960_000);
    }
}
