//! Runtime view of an encoder-decoder family's planned persistent state.
//!
//! The capacity planner deliberately keeps self-attention positions and
//! encoder cross-attention positions as independent axes. This adapter is the
//! only runtime seam that turns a generic [`GgmlAsrDecoderState`] into the two
//! axes consumed by seq2seq decoders. It therefore makes accidentally adding
//! the axes together structurally impossible.

use thiserror::Error;

use crate::capacity::topology::{StateKind, StatePositionAxis};
use crate::models::ggml_asr_executor::GgmlAsrDecoderState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Seq2SeqStateAxis {
    pub logical_positions: usize,
    pub resident_positions: usize,
    pub hard_position_cap: usize,
}

impl Seq2SeqStateAxis {
    fn validate(self, kind: StateKind) -> Result<(), Seq2SeqDecoderStateError> {
        if self.logical_positions == 0 || self.resident_positions == 0 {
            return Err(Seq2SeqDecoderStateError::EmptyAxis { kind });
        }
        if self.logical_positions > self.resident_positions {
            return Err(Seq2SeqDecoderStateError::ResidentDoesNotCoverLogical {
                kind,
                logical: self.logical_positions,
                resident: self.resident_positions,
            });
        }
        if self.logical_positions > self.hard_position_cap
            || self.resident_positions > self.hard_position_cap
        {
            return Err(Seq2SeqDecoderStateError::PlannedCapExceeded {
                kind,
                logical: self.logical_positions,
                resident: self.resident_positions,
                hard_cap: self.hard_position_cap,
            });
        }
        Ok(())
    }

    pub(crate) fn validate_exact_shape(
        self,
        kind: StateKind,
        actual_positions: usize,
    ) -> Result<(), Seq2SeqDecoderStateError> {
        if actual_positions != self.logical_positions {
            return Err(Seq2SeqDecoderStateError::LogicalShapeMismatch {
                kind,
                planned: self.logical_positions,
                actual: actual_positions,
            });
        }
        Ok(())
    }

    pub(crate) fn validate_runtime_ceiling(
        self,
        kind: StateKind,
        runtime_ceiling: usize,
    ) -> Result<(), Seq2SeqDecoderStateError> {
        if self.hard_position_cap > runtime_ceiling
            || self.resident_positions > runtime_ceiling
            || self.logical_positions > runtime_ceiling
        {
            return Err(Seq2SeqDecoderStateError::RuntimeCeilingExceeded {
                kind,
                logical: self.logical_positions,
                resident: self.resident_positions,
                planned_hard_cap: self.hard_position_cap,
                runtime_ceiling,
            });
        }
        Ok(())
    }

    pub(crate) fn validate_resident_shape(
        self,
        kind: StateKind,
        actual_positions: usize,
    ) -> Result<(), Seq2SeqDecoderStateError> {
        if actual_positions != self.resident_positions {
            return Err(Seq2SeqDecoderStateError::ResidentShapeMismatch {
                kind,
                planned: self.resident_positions,
                actual: actual_positions,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Seq2SeqDecoderState {
    pub self_attention: Seq2SeqStateAxis,
    pub cross_attention: Seq2SeqStateAxis,
}

/// Stable planner allocation ids owned by one encoder-decoder family.
///
/// `StateKind` describes semantics, but it is not an allocation key: a model
/// may legitimately own more than one stream of the same kind. Runtime wiring
/// therefore resolves the exact family-owned ids and verifies their kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Seq2SeqStateIds {
    pub self_attention: &'static str,
    pub cross_attention: &'static str,
}

/// Cache identity for a reusable seq2seq resident arena. Logical invocation
/// shapes are deliberately absent: changing them inside the same session
/// envelope must not mint a new arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Seq2SeqResidentCapacity {
    pub self_attention_positions: usize,
    pub cross_attention_positions: usize,
}

impl Seq2SeqDecoderState {
    pub(crate) fn from_request_state(
        state: &GgmlAsrDecoderState,
        ids: Seq2SeqStateIds,
    ) -> Result<Self, Seq2SeqDecoderStateError> {
        let GgmlAsrDecoderState::Planned(plan) = state else {
            return Err(Seq2SeqDecoderStateError::PlanRequired);
        };
        let self_attention = allocation_axis(plan, ids.self_attention, StateKind::SelfAttentionKv)?;
        let cross_attention =
            allocation_axis(plan, ids.cross_attention, StateKind::CrossAttentionKv)?;
        let state = Self {
            self_attention,
            cross_attention,
        };
        state.validate()?;
        Ok(state)
    }

    pub(crate) fn validate(self) -> Result<(), Seq2SeqDecoderStateError> {
        self.self_attention.validate(StateKind::SelfAttentionKv)?;
        self.cross_attention.validate(StateKind::CrossAttentionKv)
    }

    pub(crate) const fn resident_capacity(self) -> Seq2SeqResidentCapacity {
        Seq2SeqResidentCapacity {
            self_attention_positions: self.self_attention.resident_positions,
            cross_attention_positions: self.cross_attention.resident_positions,
        }
    }
}

fn allocation_axis(
    plan: &crate::capacity::topology::DecoderStatePlan,
    id: &'static str,
    kind: StateKind,
) -> Result<Seq2SeqStateAxis, Seq2SeqDecoderStateError> {
    let StatePositionAxis {
        logical_positions,
        resident_positions,
        hard_position_cap,
        ..
    } = plan.position_axis(id, kind).map_err(|error| match error {
        crate::capacity::topology::TopologyError::StateIdMissing { .. } => {
            Seq2SeqDecoderStateError::MissingAxis { id, kind }
        }
        crate::capacity::topology::TopologyError::StateKindMismatch { actual, .. } => {
            Seq2SeqDecoderStateError::AxisKindMismatch {
                id,
                expected: kind,
                actual,
            }
        }
        _ => Seq2SeqDecoderStateError::InvalidAxisPlan {
            id,
            reason: error.to_string(),
        },
    })?;
    let axis = Seq2SeqStateAxis {
        logical_positions,
        resident_positions,
        hard_position_cap,
    };
    if axis.logical_positions == 0 || axis.resident_positions == 0 {
        return Err(Seq2SeqDecoderStateError::EmptyAxis { kind });
    }
    if axis.logical_positions > axis.resident_positions {
        return Err(Seq2SeqDecoderStateError::ResidentDoesNotCoverLogical {
            kind,
            logical: axis.logical_positions,
            resident: axis.resident_positions,
        });
    }
    Ok(axis)
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum Seq2SeqDecoderStateError {
    #[error("encoder-decoder execution requires a planned decoder state")]
    PlanRequired,
    #[error("encoder-decoder plan is missing allocation '{id}' for the {kind:?} axis")]
    MissingAxis { id: &'static str, kind: StateKind },
    #[error("encoder-decoder allocation '{id}' has kind {actual:?}, expected {expected:?}")]
    AxisKindMismatch {
        id: &'static str,
        expected: StateKind,
        actual: StateKind,
    },
    #[error("encoder-decoder allocation '{id}' is invalid: {reason}")]
    InvalidAxisPlan { id: &'static str, reason: String },
    #[error("encoder-decoder plan has an empty {kind:?} axis")]
    EmptyAxis { kind: StateKind },
    #[error(
        "encoder-decoder {kind:?} resident span does not cover logical span: logical={logical}, resident={resident}"
    )]
    ResidentDoesNotCoverLogical {
        kind: StateKind,
        logical: usize,
        resident: usize,
    },
    #[error(
        "encoder-decoder {kind:?} plan exceeds hard cap {hard_cap}: logical={logical}, resident={resident}"
    )]
    PlannedCapExceeded {
        kind: StateKind,
        logical: usize,
        resident: usize,
        hard_cap: usize,
    },
    #[error("encoder-decoder {kind:?} logical shape mismatch: planned={planned}, actual={actual}")]
    LogicalShapeMismatch {
        kind: StateKind,
        planned: usize,
        actual: usize,
    },
    #[error("encoder-decoder {kind:?} resident shape mismatch: planned={planned}, actual={actual}")]
    ResidentShapeMismatch {
        kind: StateKind,
        planned: usize,
        actual: usize,
    },
    #[error(
        "encoder-decoder {kind:?} plan exceeds runtime ceiling {runtime_ceiling}: logical={logical}, resident={resident}, planned hard cap={planned_hard_cap}"
    )]
    RuntimeCeilingExceeded {
        kind: StateKind,
        logical: usize,
        resident: usize,
        planned_hard_cap: usize,
        runtime_ceiling: usize,
    },
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use crate::capacity::topology::{
        DecoderStateDemandScope, DecoderStatePlan, DecoderStateTopology, InvocationEnvelope,
        InvocationShapeInput, PositionBoundProof, StateBytes, StateDemand, TopologyError,
    };

    use super::*;

    struct Fixture;

    const IDS: Seq2SeqStateIds = Seq2SeqStateIds {
        self_attention: "test.self",
        cross_attention: "test.cross",
    };

    impl DecoderStateTopology for Fixture {
        fn demands(
            &self,
            scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
        ) -> Result<Vec<StateDemand>, TopologyError> {
            let invocation = match scope {
                DecoderStateDemandScope::ExactInvocation(invocation) => invocation,
                DecoderStateDemandScope::StableEnvelope(envelope) => envelope.maximum_invocation(),
            };
            let cross = invocation.samples() / 320;
            Ok(vec![
                StateDemand::new(
                    "test.self",
                    StateKind::SelfAttentionKv,
                    448,
                    448,
                    StateBytes::default(),
                    PositionBoundProof::Exact,
                )?,
                StateDemand::new(
                    "test.cross",
                    StateKind::CrossAttentionKv,
                    cross,
                    1_500,
                    StateBytes::default(),
                    PositionBoundProof::Exact,
                )?,
            ])
        }
    }

    fn planned_state() -> GgmlAsrDecoderState {
        let sample_rate = NonZeroU32::new(16_000).unwrap();
        let invocation = InvocationShapeInput::new(sample_rate, 160_000).unwrap();
        let envelope = InvocationEnvelope::new(sample_rate, 480_000).unwrap();
        GgmlAsrDecoderState::planned_for_test(
            DecoderStatePlan::build(&Fixture, invocation, envelope).unwrap(),
            envelope,
        )
    }

    #[test]
    fn preserves_independent_self_and_cross_axes() {
        let state = Seq2SeqDecoderState::from_request_state(&planned_state(), IDS).unwrap();
        assert_eq!(state.self_attention.logical_positions, 448);
        assert_eq!(state.self_attention.resident_positions, 448);
        assert_eq!(state.cross_attention.logical_positions, 500);
        assert_eq!(state.cross_attention.resident_positions, 1_500);
    }

    #[test]
    fn exact_shape_validation_fails_closed() {
        let state = Seq2SeqDecoderState::from_request_state(&planned_state(), IDS).unwrap();
        assert!(matches!(
            state
                .cross_attention
                .validate_exact_shape(StateKind::CrossAttentionKv, 499),
            Err(Seq2SeqDecoderStateError::LogicalShapeMismatch {
                planned: 500,
                actual: 499,
                ..
            })
        ));
    }

    #[test]
    fn resident_shape_validation_fails_closed() {
        let state = Seq2SeqDecoderState::from_request_state(&planned_state(), IDS).unwrap();
        assert!(matches!(
            state
                .cross_attention
                .validate_resident_shape(StateKind::CrossAttentionKv, 1_499),
            Err(Seq2SeqDecoderStateError::ResidentShapeMismatch {
                planned: 1_500,
                actual: 1_499,
                ..
            })
        ));
    }

    #[test]
    fn resident_cache_identity_excludes_logical_shapes() {
        let state = Seq2SeqDecoderState::from_request_state(&planned_state(), IDS).unwrap();
        let different_logical = Seq2SeqDecoderState {
            self_attention: Seq2SeqStateAxis {
                logical_positions: 224,
                ..state.self_attention
            },
            cross_attention: Seq2SeqStateAxis {
                logical_positions: 250,
                ..state.cross_attention
            },
        };
        assert_ne!(state, different_logical);
        assert_eq!(
            state.resident_capacity(),
            different_logical.resident_capacity()
        );
    }

    #[test]
    fn no_persistent_state_is_not_an_unplanned_seq2seq_request() {
        assert_eq!(
            Seq2SeqDecoderState::from_request_state(&GgmlAsrDecoderState::NoPersistentState, IDS,),
            Err(Seq2SeqDecoderStateError::PlanRequired)
        );
    }
}
