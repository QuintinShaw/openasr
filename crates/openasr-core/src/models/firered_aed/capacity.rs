//! firered-aed capacity derivation for the shared host-memory admission
//! check ([`crate::capacity::evaluate_host_memory_admission`]).
//!
//! Same shape as [`crate::models::cohere::capacity`] (see that module's doc
//! for the full modeling rationale): this AED decoder allocates its KV state
//! at fixed per-pack ceilings when the decoder runtime is built
//! (`super::decoder_graph::build_firered_decoder_arena_state`), independent
//! of the request's audio length --
//!
//! - the self-attention KV cache is f16 at the full `decoder_pe_len` span
//!   per layer (the family's `BoundedElsewhere` positional-encoding bound,
//!   5000 positions for the shipped pack), and
//! - the cross-attention KV cache is f32 over the pack's chunk-cap encoder
//!   frame capacity ([`super::decoder_graph::firered_decoder_cross_capacity_frames`]).
//!
//! Admission charges those exact allocations: the cross cache through the
//! shared position model (one "position" = one encoder frame at
//! `layers x 2 x d_model` f32 values, priced exactly by the two-f16-copy
//! [`FIRERED_AED_ADMISSION_KV_SPEC`]), the f16 self ceiling as exact fixed
//! bytes (a spec-based charge would overstate it 2x). A `longform mode=off`
//! request larger than the chunk cap can grow the cross cache past this
//! estimate; that under-estimate resolves to "allow" (fail open).

use crate::capacity::KvGeometry;
use crate::ggml_runtime::GgmlKvElementType;
use crate::nn::decoder::LlmKvCacheSpec;

use super::decoder_graph::firered_decoder_cross_capacity_frames;
use super::runtime_contract::FireRedAedExecutionMetadata;

/// Two f16 copies = 4 B/value total: byte-for-byte the single f32 copy the
/// cross-KV cache actually allocates. Only
/// [`crate::capacity::KvBytesPerPosition::total`] is consumed by admission;
/// the host/resident split is a modeling stand-in (see
/// `crate::models::cohere::capacity`'s identical convention).
pub(crate) const FIRERED_AED_ADMISSION_KV_SPEC: LlmKvCacheSpec = LlmKvCacheSpec {
    host: GgmlKvElementType::F16,
    resident: GgmlKvElementType::F16,
};

/// The AED decoder KV geometry the loaded pack advertises (MHA: `kv_heads`
/// equals `n_heads`, so `layers x 2 x kv_heads x head_dim` values per
/// position equals the `layers x 2 x d_model` values one cross-KV frame
/// costs).
pub(crate) fn firered_aed_kv_geometry(metadata: &FireRedAedExecutionMetadata) -> KvGeometry {
    KvGeometry {
        n_layers: metadata.decoder_n_layers,
        kv_heads: metadata.n_heads,
        head_dim: metadata.head_dim,
    }
}

/// The cross-KV positions admission charges through the shared position
/// model: the pack's chunk-cap encoder frame capacity, the same figure the
/// decoder runtime allocates its cross cache at (request-independent).
pub(crate) fn firered_aed_admission_required_positions(
    metadata: &FireRedAedExecutionMetadata,
) -> usize {
    firered_decoder_cross_capacity_frames(metadata)
}

/// Exact bytes of the f16 self-attention KV cache the decoder runtime
/// allocates at construction: K + V planes of
/// `head_dim x decoder_pe_len x n_heads` f16 values per layer (the exact
/// shape `build_firered_decoder_arena_state` allocates). `0` on arithmetic
/// failure -- an under-estimate resolves to "allow", per `crate::capacity`'s
/// fail-open invariant.
pub(crate) fn firered_aed_admission_fixed_self_kv_bytes(
    metadata: &FireRedAedExecutionMetadata,
) -> u64 {
    let plane = GgmlKvElementType::F16
        .plane_nbytes(metadata.head_dim, metadata.decoder_pe_len, metadata.n_heads)
        .unwrap_or(0) as u64;
    plane
        .saturating_mul(2) // K + V
        .saturating_mul(metadata.decoder_n_layers as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacity::kv_bytes_per_position;

    /// Real-checkpoint-shaped metadata (the same values `runtime_contract`'s
    /// test fixture parses: 16-layer MHA decoder at d_model 1280 / 20 heads /
    /// head_dim 64, decoder PE span 5000).
    fn reference_metadata() -> FireRedAedExecutionMetadata {
        FireRedAedExecutionMetadata {
            encoder_n_layers: 16,
            d_model: 1280,
            n_heads: 20,
            head_dim: 64,
            encoder_ffn_dim: 5120,
            conv_kernel: 33,
            subsample_channels: 32,
            subsample_out_dim: 608,
            feature_dim: 80,
            encoder_pe_len: 9999,
            decoder_n_layers: 16,
            decoder_ffn_dim: 5120,
            decoder_pe_len: 5000,
            vocab_size: 7832,
            sos_token_id: 3,
            eos_token_id: 4,
            pad_token_id: 2,
        }
    }

    #[test]
    fn admission_spec_prices_one_f32_copy_per_cross_frame() {
        let geometry = firered_aed_kv_geometry(&reference_metadata());
        let per_position = kv_bytes_per_position(&geometry, FIRERED_AED_ADMISSION_KV_SPEC)
            .expect("f16 spec accepts head_dim 64");
        // layers x 2 x d_model f32 values per cross frame:
        // 16 * 2 * 1280 * 4 B = 160 KiB.
        assert_eq!(per_position.total(), 16 * 2 * 1280 * 4);
    }

    #[test]
    fn fixed_self_kv_bytes_match_the_arena_allocation() {
        // 16 layers x 2 (K+V) x 20 heads x 5000 positions x 64 head_dim x 2 B
        // (f16) = ~390 MiB, the allocation `build_firered_decoder_arena_state`
        // makes at construction.
        assert_eq!(
            firered_aed_admission_fixed_self_kv_bytes(&reference_metadata()),
            16 * 2 * 20 * 5000 * 64 * 2
        );
    }

    #[test]
    fn required_positions_are_the_chunk_cap_cross_frames() {
        let metadata = reference_metadata();
        assert_eq!(
            firered_aed_admission_required_positions(&metadata),
            firered_decoder_cross_capacity_frames(&metadata)
        );
        assert!(firered_aed_admission_required_positions(&metadata) > 0);
    }
}
