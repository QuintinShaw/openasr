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

/// Which side of a model a quantizable tensor lives on, used to apply the
/// audio-encoder Q8_0 floor.
///
/// Sub-Q8 block quantization of *audio-encoder* weights is a behavioral cliff
/// rather than a gradual WER loss: long-audio greedy decode collapses into
/// repetition or empty output (e.g. the qwen3-asr 1.7b q4_k pack degrading to a
/// "Today, today" text collapse). The audio encoder therefore carries a hard
/// Q8_0 floor regardless of the requested rung, while decoder / LLM / CTC /
/// joint / embedding / output-head tensors keep the full requested rung.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuantComponent {
    /// Audio encoder / audio-tower weights: the acoustic feature extractor that
    /// feeds the decoder, including any encoder->LLM projector.
    Encoder,
    /// Everything downstream of the encoder: text decoder / LLM layers, CTC /
    /// joint heads, token embeddings, and output projections.
    Decoder,
}

/// Shared 32/256-alignment + K-quant-rung selection tail.
///
/// Callers first resolve their own family-specific tensor eligibility
/// (name/class/storage flags, rank, the `Fp16`-mode short-circuit) and the
/// correct `ne0` (the ggml-quantized axis length) before calling this; it only
/// decides, given an already-eligible rank-2 axis length, whether
/// q8_0/q3_k/q4_k applies or the tensor falls back to `None` (its fp16-mode
/// representation).
///
/// `component` carries the audio-encoder Q8_0 floor: an `Encoder` tensor never
/// takes the Q3_K/Q4_K rungs and always lands on Q8_0 once 32-aligned, while a
/// `Decoder` tensor keeps the full requested rung. Each family supplies the
/// classification (it owns the tensor-naming knowledge); the floor policy itself
/// lives here so every importer applies it identically.
pub(crate) fn classify_quant_tensor(
    ne0: u64,
    quantization: PackQuant,
    component: QuantComponent,
) -> Option<GgufWriteTensorType> {
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
    if ne0.is_multiple_of(256_u64) && component == QuantComponent::Decoder {
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
            classify_quant_tensor(256, PackQuant::Fp16, QuantComponent::Decoder),
            None
        );
        assert_eq!(
            classify_quant_tensor(256, PackQuant::Fp16, QuantComponent::Encoder),
            None
        );
    }

    #[test]
    fn unaligned_ne0_falls_back_to_fp16_representation() {
        assert_eq!(
            classify_quant_tensor(31, PackQuant::Q8_0, QuantComponent::Decoder),
            None
        );
        assert_eq!(
            classify_quant_tensor(32, PackQuant::Q8_0, QuantComponent::Decoder),
            Some(GgufWriteTensorType::Q8_0)
        );
        // Encoder alignment gating is unchanged: an unaligned encoder tensor
        // still keeps its (higher-precision) fp16-mode representation.
        assert_eq!(
            classify_quant_tensor(31, PackQuant::Q4_K, QuantComponent::Encoder),
            None
        );
    }

    #[test]
    fn q4_k_requires_256_alignment_else_falls_back_to_q8_0() {
        assert_eq!(
            classify_quant_tensor(32, PackQuant::Q4_K, QuantComponent::Decoder),
            Some(GgufWriteTensorType::Q8_0)
        );
        assert_eq!(
            classify_quant_tensor(256, PackQuant::Q4_K, QuantComponent::Decoder),
            Some(GgufWriteTensorType::Q4_K)
        );
    }

    #[test]
    fn q3_k_requires_256_alignment_else_falls_back_to_q8_0() {
        assert_eq!(
            classify_quant_tensor(32, PackQuant::Q3_K, QuantComponent::Decoder),
            Some(GgufWriteTensorType::Q8_0)
        );
        assert_eq!(
            classify_quant_tensor(256, PackQuant::Q3_K, QuantComponent::Decoder),
            Some(GgufWriteTensorType::Q3_K)
        );
    }

    #[test]
    fn encoder_carries_a_q8_0_floor_below_the_requested_rung() {
        // A 256-aligned encoder tensor would normally take the K-quant rungs,
        // but the floor clamps it to Q8_0 so long-audio greedy decode never sees
        // a sub-Q8 acoustic encoder.
        assert_eq!(
            classify_quant_tensor(256, PackQuant::Q4_K, QuantComponent::Encoder),
            Some(GgufWriteTensorType::Q8_0)
        );
        assert_eq!(
            classify_quant_tensor(256, PackQuant::Q3_K, QuantComponent::Encoder),
            Some(GgufWriteTensorType::Q8_0)
        );
        // Q8_0 and 32-aligned (non-256) cases are unaffected by the floor.
        assert_eq!(
            classify_quant_tensor(256, PackQuant::Q8_0, QuantComponent::Encoder),
            Some(GgufWriteTensorType::Q8_0)
        );
        assert_eq!(
            classify_quant_tensor(32, PackQuant::Q4_K, QuantComponent::Encoder),
            Some(GgufWriteTensorType::Q8_0)
        );
    }

    #[test]
    fn decoder_keeps_the_full_requested_rung() {
        assert_eq!(
            classify_quant_tensor(256, PackQuant::Q4_K, QuantComponent::Decoder),
            Some(GgufWriteTensorType::Q4_K)
        );
        assert_eq!(
            classify_quant_tensor(256, PackQuant::Q3_K, QuantComponent::Decoder),
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
