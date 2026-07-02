//! Data-driven layer-stack orchestration (P4 S5).
//!
//! [`validate_stage_against_descriptor`] is the load-bearing gate: it reads a
//! [`OpenAsrBlockStackDescriptor`] and fail-closed-validates a family's intended
//! stack (orchestration shape, block kind, tensor-name scope, layer count via
//! [`LayerCountResolver`]) against the descriptor BEFORE any ggml op is emitted.
//! A family runs it just before its existing (verbatim, byte-identical)
//! `compose_*` call. This is what makes the block-stack descriptor *load-bearing*
//! rather than merely validated at startup.
//!
//! Neither the gate nor the driver owns a ggml op or sees a per-step-derived
//! scalar (`cache_position`, `total_tokens`, `rope`, the f16 attention mask,
//! `kv_span`, `frame_count`, `position_offset`, the Metal-vs-CPU reuse branch, …).
//! Those — and the cohere deferred-`uploads` side-channel — live inside the
//! family's build closure / call site, never in the descriptor or the gate.
//!
//! Whisper has `block_stack == None` and its executor never calls the gate, so it
//! remains the hand-written bit-level reference gate, structurally untouched.
//!
//! INTERFACE NOTE: the S5b sketch had a `ShapeOrchestrator` trait whose
//! `StageCtx<'a>` GAT carried the `&mut GgmlCpuGraphBuilder`. Wiring revealed that
//! cannot compile (the builder is reused after the stack build, so the `&mut`
//! borrow is shorter-lived than the builder, and `&mut T` is invariant — a
//! single-lifetime GAT can't express it). The gate is therefore a plain
//! validation function; see [`validate_stage_against_descriptor`].

// `validate_stage_against_descriptor` is called at qwen executor construction.
// Some helpers stay test-only until more families wire in.
#![allow(dead_code)]

use crate::arch::block_stack::{
    OpenAsrBlockKind, OpenAsrBlockStackDescriptor, OpenAsrOrchestrationShape,
    OpenAsrStageDescriptor,
};

/// Which stack stage is being assembled; selects the encoder vs decoder
/// descriptor stage and is reported in validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenAsrStageRole {
    Encoder,
    Decoder,
}

/// Fail-closed mismatches between descriptor DATA and what the family hook
/// reports. These are the gates that make the descriptor authoritative; each is
/// an error, never a warning. The family lifts them into its own error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShapeOrchestratorError {
    /// `block_stack` was `None` for an architecture whose executor asked to
    /// orchestrate (whisper, the only `None`, must never reach the driver).
    MissingBlockStack { model_architecture: &'static str },
    /// The shape the executor handles (`O::SHAPE`) != the shape the descriptor
    /// declares.
    OrchestrationShapeMismatch {
        model_architecture: &'static str,
        expected: OpenAsrOrchestrationShape,
        declared: OpenAsrOrchestrationShape,
    },
    /// `Encoder` role was driven but the descriptor's `encoder_stage` is `None`.
    StageRoleAbsent {
        model_architecture: &'static str,
        role: OpenAsrStageRole,
    },
    /// Descriptor `block_kind` != the kind the family hook is wired to emit.
    BlockKindMismatch {
        model_architecture: &'static str,
        role: OpenAsrStageRole,
        descriptor_kind: OpenAsrBlockKind,
        hook_kind: OpenAsrBlockKind,
    },
    /// Family hook `tensor_name_scope` != the descriptor's.
    TensorScopeMismatch {
        model_architecture: &'static str,
        role: OpenAsrStageRole,
        descriptor_scope: &'static str,
        hook_scope: &'static str,
    },
    /// The layer count the family materialized != the count the descriptor's
    /// `layer_count_hparam` resolved to in the live metadata.
    LayerCountMismatch {
        model_architecture: &'static str,
        role: OpenAsrStageRole,
        layer_count_hparam: &'static str,
        descriptor_count: usize,
        family_count: usize,
    },
    /// `layer_count_hparam` was absent / unparseable in the live hparams.
    LayerCountHparamUnresolved {
        model_architecture: &'static str,
        layer_count_hparam: &'static str,
    },
    /// The `Decoder` role was driven on a `Ctc` (encoder-only) block stack, which
    /// has no `decoder_stage`. A Ctc family must only ever drive `Encoder`.
    DecoderRequestedForCtcShape { model_architecture: &'static str },
}

/// What a family hook DECLARES about the stack it is about to build, cross-checked
/// against the descriptor before any op is emitted. Pure `Copy` scalars +
/// `&'static str`: no ggml handle, no lifetime, no per-step-derived scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StageBuildPlan {
    /// The kind emitted for every layer. MUST equal the stage's `block_kind`.
    pub block_kind: OpenAsrBlockKind,
    /// The scope weights are bound from. MUST equal the stage's
    /// `tensor_name_scope`.
    pub tensor_name_scope: &'static str,
    /// The count the family materialized from its ACTUAL layer collection
    /// (e.g. `self.layers.len()`), NOT a literal. MUST equal what
    /// `layer_count_hparam` resolves to. Seq2Seq decoders pre-check their
    /// companion slices (cross/self-KV) align with this and otherwise report a
    /// count the driver will reject.
    pub family_layer_count: usize,
}

/// Resolves a per-stage `layer_count_hparam` against the live hparams. Implemented
/// by the family executor over its already-parsed metadata so no new parsing path
/// appears.
///
/// HONESTY CONTRACT: the impl MUST read the named hparam from the metadata map,
/// NOT return `self.layers.len()` — otherwise [`ShapeOrchestratorError::LayerCountMismatch`]
/// can never fire and the gate is vacuous. The S5g negative tests mutate the
/// DESCRIPTOR key (not the resolver) to keep this honest.
pub(crate) trait LayerCountResolver {
    fn resolve_layer_count(&self, hparam_key: &'static str) -> Option<usize>;
}

/// Fail-closed-validate a family's intended layer-stack ([`StageBuildPlan`])
/// against the block-stack descriptor BEFORE any op is emitted, returning the
/// descriptor-resolved layer count on success. This is the load-bearing gate:
/// the descriptor now governs which composer/shape/count/scope a family is
/// allowed to assemble, and a drift fails closed rather than silently building
/// the wrong thing.
///
/// `expected_shape` is the shape the *calling family* implements (e.g. the qwen
/// executor passes `LlmDecoder`); it is cross-checked against the descriptor so a
/// family can never be wired to a descriptor of the wrong shape.
///
/// INTERFACE NOTE (revised from the S5b sketch): the original design routed the
/// build itself through a `ShapeOrchestrator::build_stack(ctx)` trait method
/// whose `StageCtx<'a>` GAT carried the `&mut GgmlCpuGraphBuilder`. That cannot
/// compile: the builder is reused after the stack is assembled (`set_output`,
/// KV upload, compute), so the `&mut` borrow must end before the builder's own
/// lifetime — and `&mut T` is invariant, so a single-lifetime GAT cannot hold a
/// shorter-lived mutable borrow of a longer-lived builder. The gate is therefore
/// a plain validation function the call site invokes before its existing
/// (verbatim, byte-identical) `compose_*` call.
pub(crate) fn validate_stage_against_descriptor<R>(
    model_architecture: &'static str,
    block_stack: Option<&OpenAsrBlockStackDescriptor>,
    role: OpenAsrStageRole,
    expected_shape: OpenAsrOrchestrationShape,
    plan: StageBuildPlan,
    resolver: &R,
) -> Result<usize, ShapeOrchestratorError>
where
    R: LayerCountResolver,
{
    let stack =
        block_stack.ok_or(ShapeOrchestratorError::MissingBlockStack { model_architecture })?;
    if stack.orchestration_shape != expected_shape {
        return Err(ShapeOrchestratorError::OrchestrationShapeMismatch {
            model_architecture,
            expected: expected_shape,
            declared: stack.orchestration_shape,
        });
    }
    let stage: &OpenAsrStageDescriptor = match role {
        OpenAsrStageRole::Encoder => {
            stack
                .encoder_stage
                .as_ref()
                .ok_or(ShapeOrchestratorError::StageRoleAbsent {
                    model_architecture,
                    role,
                })?
        }
        OpenAsrStageRole::Decoder => stack
            .decoder_stage
            .as_ref()
            .ok_or(ShapeOrchestratorError::DecoderRequestedForCtcShape { model_architecture })?,
    };

    if plan.block_kind != stage.block_kind {
        return Err(ShapeOrchestratorError::BlockKindMismatch {
            model_architecture,
            role,
            descriptor_kind: stage.block_kind,
            hook_kind: plan.block_kind,
        });
    }
    if plan.tensor_name_scope != stage.tensor_name_scope {
        return Err(ShapeOrchestratorError::TensorScopeMismatch {
            model_architecture,
            role,
            descriptor_scope: stage.tensor_name_scope,
            hook_scope: plan.tensor_name_scope,
        });
    }
    let descriptor_count = resolver
        .resolve_layer_count(stage.layer_count_hparam)
        .ok_or(ShapeOrchestratorError::LayerCountHparamUnresolved {
            model_architecture,
            layer_count_hparam: stage.layer_count_hparam,
        })?;
    if descriptor_count != plan.family_layer_count {
        return Err(ShapeOrchestratorError::LayerCountMismatch {
            model_architecture,
            role,
            layer_count_hparam: stage.layer_count_hparam,
            descriptor_count,
            family_count: plan.family_layer_count,
        });
    }
    Ok(descriptor_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_ARCH: &str = "test-arch";
    const TEST_HPARAM: &str = "test.decoder.n_layers";

    struct StubResolver(Option<usize>);
    impl LayerCountResolver for StubResolver {
        fn resolve_layer_count(&self, _hparam_key: &'static str) -> Option<usize> {
            self.0
        }
    }

    fn llm_decoder_stack(
        decoder_kind: OpenAsrBlockKind,
        scope: &'static str,
    ) -> OpenAsrBlockStackDescriptor {
        OpenAsrBlockStackDescriptor {
            orchestration_shape: OpenAsrOrchestrationShape::LlmDecoder,
            encoder_stage: None,
            decoder_stage: Some(OpenAsrStageDescriptor {
                block_kind: decoder_kind,
                layer_count_hparam: TEST_HPARAM,
                tensor_name_scope: scope,
            }),
        }
    }

    fn matching_plan() -> StageBuildPlan {
        StageBuildPlan {
            block_kind: OpenAsrBlockKind::LlmDecoderLayer,
            tensor_name_scope: "blk",
            family_layer_count: 28,
        }
    }

    fn validate(
        stack: Option<&OpenAsrBlockStackDescriptor>,
        role: OpenAsrStageRole,
        plan: StageBuildPlan,
        resolver: &StubResolver,
    ) -> Result<usize, ShapeOrchestratorError> {
        validate_stage_against_descriptor(
            TEST_ARCH,
            stack,
            role,
            OpenAsrOrchestrationShape::LlmDecoder,
            plan,
            resolver,
        )
    }

    #[test]
    fn validates_and_returns_descriptor_count_when_plan_agrees() {
        let stack = llm_decoder_stack(OpenAsrBlockKind::LlmDecoderLayer, "blk");
        let resolver = StubResolver(Some(28));
        assert_eq!(
            validate(
                Some(&stack),
                OpenAsrStageRole::Decoder,
                matching_plan(),
                &resolver
            ),
            Ok(28)
        );
    }

    #[test]
    fn rejects_missing_block_stack() {
        let resolver = StubResolver(Some(28));
        assert_eq!(
            validate(None, OpenAsrStageRole::Decoder, matching_plan(), &resolver),
            Err(ShapeOrchestratorError::MissingBlockStack {
                model_architecture: TEST_ARCH
            })
        );
    }

    #[test]
    fn rejects_orchestration_shape_mismatch() {
        // A Seq2Seq descriptor validated against an LlmDecoder family.
        let stack = OpenAsrBlockStackDescriptor {
            orchestration_shape: OpenAsrOrchestrationShape::Seq2SeqEncoderDecoder,
            encoder_stage: None,
            decoder_stage: Some(OpenAsrStageDescriptor {
                block_kind: OpenAsrBlockKind::Seq2SeqDecoderLayer,
                layer_count_hparam: TEST_HPARAM,
                tensor_name_scope: "dec.blk",
            }),
        };
        let resolver = StubResolver(Some(28));
        assert_eq!(
            validate(
                Some(&stack),
                OpenAsrStageRole::Decoder,
                matching_plan(),
                &resolver
            ),
            Err(ShapeOrchestratorError::OrchestrationShapeMismatch {
                model_architecture: TEST_ARCH,
                expected: OpenAsrOrchestrationShape::LlmDecoder,
                declared: OpenAsrOrchestrationShape::Seq2SeqEncoderDecoder,
            })
        );
    }

    #[test]
    fn rejects_stage_role_absent() {
        let stack = llm_decoder_stack(OpenAsrBlockKind::LlmDecoderLayer, "blk");
        let resolver = StubResolver(Some(28));
        // encoder_stage is None on this stack.
        assert_eq!(
            validate(
                Some(&stack),
                OpenAsrStageRole::Encoder,
                matching_plan(),
                &resolver
            ),
            Err(ShapeOrchestratorError::StageRoleAbsent {
                model_architecture: TEST_ARCH,
                role: OpenAsrStageRole::Encoder,
            })
        );
    }

    #[test]
    fn rejects_block_kind_mismatch() {
        let stack = llm_decoder_stack(OpenAsrBlockKind::ConformerBlock, "blk");
        let resolver = StubResolver(Some(28));
        assert_eq!(
            validate(
                Some(&stack),
                OpenAsrStageRole::Decoder,
                matching_plan(),
                &resolver
            ),
            Err(ShapeOrchestratorError::BlockKindMismatch {
                model_architecture: TEST_ARCH,
                role: OpenAsrStageRole::Decoder,
                descriptor_kind: OpenAsrBlockKind::ConformerBlock,
                hook_kind: OpenAsrBlockKind::LlmDecoderLayer,
            })
        );
    }

    #[test]
    fn rejects_tensor_scope_mismatch() {
        let stack = llm_decoder_stack(OpenAsrBlockKind::LlmDecoderLayer, "wrong.scope");
        let resolver = StubResolver(Some(28));
        assert_eq!(
            validate(
                Some(&stack),
                OpenAsrStageRole::Decoder,
                matching_plan(),
                &resolver
            ),
            Err(ShapeOrchestratorError::TensorScopeMismatch {
                model_architecture: TEST_ARCH,
                role: OpenAsrStageRole::Decoder,
                descriptor_scope: "wrong.scope",
                hook_scope: "blk",
            })
        );
    }

    #[test]
    fn rejects_layer_count_hparam_unresolved() {
        let stack = llm_decoder_stack(OpenAsrBlockKind::LlmDecoderLayer, "blk");
        let resolver = StubResolver(None);
        assert_eq!(
            validate(
                Some(&stack),
                OpenAsrStageRole::Decoder,
                matching_plan(),
                &resolver
            ),
            Err(ShapeOrchestratorError::LayerCountHparamUnresolved {
                model_architecture: TEST_ARCH,
                layer_count_hparam: TEST_HPARAM,
            })
        );
    }

    #[test]
    fn rejects_layer_count_mismatch() {
        let stack = llm_decoder_stack(OpenAsrBlockKind::LlmDecoderLayer, "blk");
        // Descriptor hparam resolves to 24, but the family materialized 28.
        let resolver = StubResolver(Some(24));
        assert_eq!(
            validate(
                Some(&stack),
                OpenAsrStageRole::Decoder,
                matching_plan(),
                &resolver
            ),
            Err(ShapeOrchestratorError::LayerCountMismatch {
                model_architecture: TEST_ARCH,
                role: OpenAsrStageRole::Decoder,
                layer_count_hparam: TEST_HPARAM,
                descriptor_count: 24,
                family_count: 28,
            })
        );
    }
}
