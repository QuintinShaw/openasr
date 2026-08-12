//! Shared pack-quant classification, used by every model family's local-source
//! importer.
//!
//! Each family keeps its own tensor-eligibility rule (name suffix, a
//! `TensorClass`/`TensorStorage` enum, a `force_f32` override flag, a rank
//! check) and its own choice of which axis is `ne0` -- most families quantize
//! along `dims[0]`, but a reversed-dim family (dolphin) uses the last axis
//! instead. Only the truly family-agnostic tail -- 32/256 block-alignment
//! gating and which K-quant rung a request selects -- lives here, so a
//! per-family `Fp16`-mode short-circuit and eligibility check always run
//! first at the call site.

use crate::ggml_runtime::GgufWriteTensorType;

/// The pack-quant rungs a family's local-source importer can produce. `Fp16`
/// keeps the family's non-quantized representation (fp16 for rank>=2 weights,
/// f32 for 1-D vectors/CMVN/mel filterbanks, per family); `Q8_0`/`Q3_K`/`Q4_K`
/// block-quantize eligible rank-2 `.weight` matrices. Families whose rung set
/// is exactly this one alias their public `<Family>QuantizationMode` type
/// straight to `PackQuant` (see `models::cohere::CohereRuntimeQuantizationMode`
/// and friends); `Q3_K` is presently only reachable by `qwen`, and a family
/// with a materially different scheme (e.g. redimnet's single-rung `F32`)
/// keeps its own enum instead of aliasing here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[allow(non_camel_case_types)]
pub enum PackQuant {
    #[default]
    Fp16,
    Q8_0,
    Q3_K,
    Q4_K,
}

impl PackQuant {
    /// Canonical lowercase pack-quant tag (`fp16`/`q8_0`/`q3_k`/`q4_k`), used to
    /// name the output pack and report the produced rung.
    pub fn label(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Q8_0 => "q8_0",
            Self::Q3_K => "q3_k",
            Self::Q4_K => "q4_k",
        }
    }
}

/// Mathematical use of a runtime tensor. Source-name parsing ends at this
/// boundary; quantization policy consumes the role and shape, never a family
/// prefix or suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorRole {
    AcousticEncoderMatrix,
    TextDecoderMatrix,
    EmbeddingTable,
    OutputProjection,
    /// A model-specific matrix whose downstream decisions are unusually
    /// sensitive to K-quant perturbations. It carries the same Q8_0 floor as
    /// the acoustic encoder without being misclassified as acoustic.
    PrecisionCriticalMatrix,
    NonQuantizable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuantizedAxis {
    First,
    Last,
}

/// Executable classification contract used by both pack writers and the
/// post-build quant-floor audit. Registry descriptors must choose one variant;
/// there is no architecture-name fallback table.
#[derive(Debug, Clone, Copy)]
pub(crate) enum TensorQuantizationContract {
    /// Writer and audit share the same mathematical-role classifier.
    SemanticRolesV1 {
        model_architecture: &'static str,
        classify: fn(&str) -> TensorRole,
        quantized_axis: QuantizedAxis,
    },
    /// Every quantizable tensor belongs to one acoustic model.
    EntireAcousticPack { model_architecture: &'static str },
    /// The pack has no acoustic encoder to which the ASR Q8 floor applies.
    /// A reason is mandatory so `NotApplicable` cannot become a disguised
    /// backlog state in the runtime inventory.
    NotApplicable {
        model_architecture: &'static str,
        reason: &'static str,
    },
}

impl TensorQuantizationContract {
    pub(crate) const fn model_architecture(self) -> &'static str {
        match self {
            Self::SemanticRolesV1 {
                model_architecture, ..
            }
            | Self::EntireAcousticPack { model_architecture }
            | Self::NotApplicable {
                model_architecture, ..
            } => model_architecture,
        }
    }

    /// Resolve the mathematical role once at the inventory seam.
    ///
    /// Consumers that need to distinguish a safety floor from a requested
    /// storage rung (for example the post-build audit and conservative
    /// requantization) must use this projection instead of parsing tensor
    /// names beside the family classifier.
    pub(crate) fn tensor_role(self, name: &str) -> Option<TensorRole> {
        match self {
            Self::SemanticRolesV1 { classify, .. } => Some(classify(name)),
            Self::EntireAcousticPack { .. } => Some(TensorRole::AcousticEncoderMatrix),
            Self::NotApplicable { reason, .. } => {
                debug_assert!(
                    !reason.trim().is_empty(),
                    "NotApplicable quantization contracts require a reason"
                );
                None
            }
        }
    }

    /// Project one tensor through the inventory-owned semantic policy.
    /// Repack/requant tooling consumes this exact seam rather than rebuilding
    /// family-name eligibility rules beside the original importer.
    pub(crate) fn target_write_type(
        self,
        name: &str,
        dims: &[u64],
        quantization: PackQuant,
    ) -> Option<GgufWriteTensorType> {
        // K-quants are matrix storage formats in the pack contract. A name
        // alone is insufficient: norm vectors and higher-rank convolution
        // kernels can share a `.weight` suffix with decoder matrices.
        if dims.len() != 2 {
            return None;
        }
        let role = self.tensor_role(name)?;
        let quantized_axis = match self {
            Self::SemanticRolesV1 { quantized_axis, .. } => quantized_axis,
            Self::EntireAcousticPack { .. } => QuantizedAxis::First,
            Self::NotApplicable { .. } => return None,
        };
        classify_quant_tensor_role(dims, quantization, role, quantized_axis)
    }
}

impl TensorRole {
    /// Every quantizable semantic role. Consumers which derive the set of
    /// producible storage rungs use this inventory instead of maintaining a
    /// second, incomplete role table.
    pub(crate) const QUANTIZABLE: [Self; 5] = [
        Self::AcousticEncoderMatrix,
        Self::TextDecoderMatrix,
        Self::EmbeddingTable,
        Self::OutputProjection,
        Self::PrecisionCriticalMatrix,
    ];

    /// Whether this semantic role carries the shared Q8_0 safety floor.
    /// Writer classification, post-build audit, and requantization all consume
    /// this projection; family-specific consumers must not recreate it.
    pub(crate) const fn requires_q8_floor(self) -> bool {
        matches!(
            self,
            Self::AcousticEncoderMatrix | Self::PrecisionCriticalMatrix
        )
    }

    const fn is_quantizable(self) -> bool {
        !matches!(self, Self::NonQuantizable)
    }
}

/// Semantic quantization entry point. The family maps a source tensor to a
/// [`TensorRole`] and storage orientation once; this shared policy applies the
/// alignment and any semantic Q8 floor without inspecting a tensor name.
pub(crate) fn classify_quant_tensor_role(
    dims: &[u64],
    quantization: PackQuant,
    role: TensorRole,
    axis: QuantizedAxis,
) -> Option<GgufWriteTensorType> {
    if !role.is_quantizable() {
        return None;
    }
    let requires_q8_floor = role.requires_q8_floor();
    let ne0 = match axis {
        QuantizedAxis::First => dims.first(),
        QuantizedAxis::Last => dims.last(),
    }
    .copied()?;
    // Every real call site already guards this (checked ahead of the family's
    // own eligibility test), but the guard belongs on the policy itself: a
    // caller-less consumer (e.g. the quant-floor audit deriving a declared
    // tier's producible rungs) must get `None` for `Fp16`, not silently fall
    // through to the block-quant arms below.
    if quantization == PackQuant::Fp16 {
        return None;
    }
    if !ne0.is_multiple_of(32_u64) {
        return None;
    }
    if ne0.is_multiple_of(256_u64) && !requires_q8_floor {
        if quantization == PackQuant::Q3_K {
            return Some(GgufWriteTensorType::Q3_K);
        }
        if quantization == PackQuant::Q4_K {
            return Some(GgufWriteTensorType::Q4_K);
        }
    }
    Some(GgufWriteTensorType::Q8_0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_tier_never_produces_a_block_quant() {
        assert_eq!(
            classify_quant_tensor_role(
                &[256],
                PackQuant::Fp16,
                TensorRole::TextDecoderMatrix,
                QuantizedAxis::First,
            ),
            None
        );
        assert_eq!(
            classify_quant_tensor_role(
                &[256],
                PackQuant::Fp16,
                TensorRole::AcousticEncoderMatrix,
                QuantizedAxis::First,
            ),
            None
        );
    }

    #[test]
    fn inventory_projection_only_block_quantizes_rank_two_matrices() {
        let contract = TensorQuantizationContract::SemanticRolesV1 {
            model_architecture: "fixture",
            classify: |_| TensorRole::TextDecoderMatrix,
            quantized_axis: QuantizedAxis::First,
        };
        assert_eq!(
            contract.target_write_type("norm.weight", &[256], PackQuant::Q4_K),
            None
        );
        assert_eq!(
            contract.target_write_type("conv.weight", &[256, 4, 3], PackQuant::Q4_K),
            None
        );
        assert_eq!(
            contract.target_write_type("blk.weight", &[256, 4], PackQuant::Q4_K),
            Some(GgufWriteTensorType::Q4_K)
        );
        let acoustic_contract = TensorQuantizationContract::EntireAcousticPack {
            model_architecture: "fixture-acoustic",
        };
        assert_eq!(
            acoustic_contract.target_write_type("encoder.weight", &[32, 4], PackQuant::Q4_K),
            Some(GgufWriteTensorType::Q8_0)
        );
    }

    #[test]
    fn unaligned_ne0_falls_back_to_fp16_representation() {
        assert_eq!(
            classify_quant_tensor_role(
                &[31],
                PackQuant::Q8_0,
                TensorRole::TextDecoderMatrix,
                QuantizedAxis::First
            ),
            None
        );
        assert_eq!(
            classify_quant_tensor_role(
                &[32],
                PackQuant::Q8_0,
                TensorRole::TextDecoderMatrix,
                QuantizedAxis::First
            ),
            Some(GgufWriteTensorType::Q8_0)
        );
        // Encoder alignment gating is unchanged: an unaligned encoder tensor
        // still keeps its (higher-precision) fp16-mode representation.
        assert_eq!(
            classify_quant_tensor_role(
                &[31],
                PackQuant::Q4_K,
                TensorRole::AcousticEncoderMatrix,
                QuantizedAxis::First
            ),
            None
        );
    }

    #[test]
    fn q4_k_requires_256_alignment_else_falls_back_to_q8_0() {
        assert_eq!(
            classify_quant_tensor_role(
                &[32],
                PackQuant::Q4_K,
                TensorRole::TextDecoderMatrix,
                QuantizedAxis::First
            ),
            Some(GgufWriteTensorType::Q8_0)
        );
        assert_eq!(
            classify_quant_tensor_role(
                &[256],
                PackQuant::Q4_K,
                TensorRole::TextDecoderMatrix,
                QuantizedAxis::First
            ),
            Some(GgufWriteTensorType::Q4_K)
        );
    }

    #[test]
    fn q3_k_requires_256_alignment_else_falls_back_to_q8_0() {
        assert_eq!(
            classify_quant_tensor_role(
                &[32],
                PackQuant::Q3_K,
                TensorRole::TextDecoderMatrix,
                QuantizedAxis::First
            ),
            Some(GgufWriteTensorType::Q8_0)
        );
        assert_eq!(
            classify_quant_tensor_role(
                &[256],
                PackQuant::Q3_K,
                TensorRole::TextDecoderMatrix,
                QuantizedAxis::First
            ),
            Some(GgufWriteTensorType::Q3_K)
        );
    }

    #[test]
    fn encoder_carries_a_q8_0_floor_below_the_requested_rung() {
        // A 256-aligned encoder tensor would normally take the K-quant rungs,
        // but the floor clamps it to Q8_0 so long-audio greedy decode never sees
        // a sub-Q8 acoustic encoder.
        assert_eq!(
            classify_quant_tensor_role(
                &[256],
                PackQuant::Q4_K,
                TensorRole::AcousticEncoderMatrix,
                QuantizedAxis::First
            ),
            Some(GgufWriteTensorType::Q8_0)
        );
        assert_eq!(
            classify_quant_tensor_role(
                &[256],
                PackQuant::Q3_K,
                TensorRole::AcousticEncoderMatrix,
                QuantizedAxis::First
            ),
            Some(GgufWriteTensorType::Q8_0)
        );
        // Q8_0 and 32-aligned (non-256) cases are unaffected by the floor.
        assert_eq!(
            classify_quant_tensor_role(
                &[256],
                PackQuant::Q8_0,
                TensorRole::AcousticEncoderMatrix,
                QuantizedAxis::First
            ),
            Some(GgufWriteTensorType::Q8_0)
        );
        assert_eq!(
            classify_quant_tensor_role(
                &[32],
                PackQuant::Q4_K,
                TensorRole::AcousticEncoderMatrix,
                QuantizedAxis::First
            ),
            Some(GgufWriteTensorType::Q8_0)
        );
    }

    #[test]
    fn precision_critical_matrices_share_the_q8_floor_policy() {
        assert!(TensorRole::PrecisionCriticalMatrix.requires_q8_floor());
        assert_eq!(
            classify_quant_tensor_role(
                &[256, 5_000],
                PackQuant::Q4_K,
                TensorRole::PrecisionCriticalMatrix,
                QuantizedAxis::First,
            ),
            Some(GgufWriteTensorType::Q8_0)
        );
        assert_eq!(
            classify_quant_tensor_role(
                &[256, 256],
                PackQuant::Q4_K,
                TensorRole::TextDecoderMatrix,
                QuantizedAxis::First,
            ),
            Some(GgufWriteTensorType::Q4_K)
        );
    }

    #[test]
    fn decoder_keeps_the_full_requested_rung() {
        assert_eq!(
            classify_quant_tensor_role(
                &[256],
                PackQuant::Q4_K,
                TensorRole::TextDecoderMatrix,
                QuantizedAxis::First
            ),
            Some(GgufWriteTensorType::Q4_K)
        );
        assert_eq!(
            classify_quant_tensor_role(
                &[256],
                PackQuant::Q3_K,
                TensorRole::TextDecoderMatrix,
                QuantizedAxis::First
            ),
            Some(GgufWriteTensorType::Q3_K)
        );
    }

    #[test]
    fn semantic_roles_apply_policy_without_tensor_names() {
        assert_eq!(
            classify_quant_tensor_role(
                &[256, 896],
                PackQuant::Q4_K,
                TensorRole::AcousticEncoderMatrix,
                QuantizedAxis::First,
            ),
            Some(GgufWriteTensorType::Q8_0)
        );
        assert_eq!(
            classify_quant_tensor_role(
                &[896, 256],
                PackQuant::Q4_K,
                TensorRole::TextDecoderMatrix,
                QuantizedAxis::Last,
            ),
            Some(GgufWriteTensorType::Q4_K)
        );
        assert_eq!(
            classify_quant_tensor_role(
                &[256, 896],
                PackQuant::Q4_K,
                TensorRole::NonQuantizable,
                QuantizedAxis::First,
            ),
            None
        );
    }

    #[test]
    fn label_matches_canonical_pack_quant_tags() {
        assert_eq!(PackQuant::Fp16.label(), "fp16");
        assert_eq!(PackQuant::Q8_0.label(), "q8_0");
        assert_eq!(PackQuant::Q3_K.label(), "q3_k");
        assert_eq!(PackQuant::Q4_K.label(), "q4_k");
    }
}
