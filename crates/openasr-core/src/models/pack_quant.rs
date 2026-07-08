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
/// with a materially different scheme (e.g. wespeaker's single-rung `F32`)
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

/// Shared 32/256-alignment + K-quant-rung selection tail.
///
/// Callers first resolve their own family-specific tensor eligibility
/// (name/class/storage flags, rank, the `Fp16`-mode short-circuit) and the
/// correct `ne0` (the ggml-quantized axis length) before calling this; it only
/// decides, given an already-eligible rank-2 axis length, whether
/// q8_0/q3_k/q4_k applies or the tensor falls back to `None` (its fp16-mode
/// representation).
pub(crate) fn classify_quant_tensor(
    ne0: u64,
    quantization: PackQuant,
) -> Option<GgufWriteTensorType> {
    if !ne0.is_multiple_of(32_u64) {
        return None;
    }
    if ne0.is_multiple_of(256_u64) {
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
    fn unaligned_ne0_falls_back_to_fp16_representation() {
        assert_eq!(classify_quant_tensor(31, PackQuant::Q8_0), None);
        assert_eq!(
            classify_quant_tensor(32, PackQuant::Q8_0),
            Some(GgufWriteTensorType::Q8_0)
        );
    }

    #[test]
    fn q4_k_requires_256_alignment_else_falls_back_to_q8_0() {
        assert_eq!(
            classify_quant_tensor(32, PackQuant::Q4_K),
            Some(GgufWriteTensorType::Q8_0)
        );
        assert_eq!(
            classify_quant_tensor(256, PackQuant::Q4_K),
            Some(GgufWriteTensorType::Q4_K)
        );
    }

    #[test]
    fn q3_k_requires_256_alignment_else_falls_back_to_q8_0() {
        assert_eq!(
            classify_quant_tensor(32, PackQuant::Q3_K),
            Some(GgufWriteTensorType::Q8_0)
        );
        assert_eq!(
            classify_quant_tensor(256, PackQuant::Q3_K),
            Some(GgufWriteTensorType::Q3_K)
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
