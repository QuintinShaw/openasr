//! cohere-transcribe capacity derivation for the shared host-memory admission
//! check ([`crate::capacity::evaluate_host_memory_admission`]).
//!
//! Unlike the Qwen-shaped LLM families, this AED decoder's KV state is
//! allocated at FIXED per-pack ceilings the moment the decoder runtime is
//! built, independent of the request's audio length (see
//! `super::decoder_graph`):
//!
//! - the self-attention KV cache is f16 at the full `decoder_max_context`
//!   span per layer, and
//! - the cross-attention KV cache is f32 over the pack's chunk-cap encoder
//!   frame capacity ([`super::decoder_graph::cohere_decoder_cross_capacity_frames`],
//!   the shared longform safety ceiling plus margin).
//!
//! Admission therefore charges those exact allocations, not a per-request
//! position walk. The split between the two return channels keeps every byte
//! exact:
//!
//! - the cross-KV cache maps onto the shared position model byte-for-byte:
//!   one "position" = one encoder frame costing `layers x 2 x d_model` f32
//!   values, and [`COHERE_ADMISSION_KV_SPEC`] (two f16 copies, 4 B/value
//!   total) equals one f32 copy exactly;
//! - the f16 self-KV ceiling cannot be expressed at 2 B/value through the
//!   two-copy spec model, so it is returned as exact fixed bytes instead
//!   (charging it through the spec would overstate it 2x -- the false-reject
//!   direction the admission invariant forbids).
//!
//! A `longform mode=off` request larger than the chunk cap can still grow
//! the cross cache past this estimate at runtime; admission's under-estimate
//! for that opt-in path resolves to "allow" (fail open), never to a false
//! rejection of the default path.

use crate::capacity::KvGeometry;
use crate::ggml_runtime::GgmlKvElementType;
use crate::nn::decoder::LlmKvCacheSpec;

use super::decoder_graph::cohere_decoder_cross_capacity_frames;
use super::runtime_contract::CohereTranscribeExecutionMetadata;

/// Two f16 copies = 4 B/value total: byte-for-byte the single f32 copy the
/// cross-KV cache actually allocates (see the module doc). Only
/// [`crate::capacity::KvBytesPerPosition::total`] is consumed by admission,
/// so the host/resident split is a modeling stand-in, not a claim that a
/// host-side copy exists.
pub(crate) const COHERE_ADMISSION_KV_SPEC: LlmKvCacheSpec = LlmKvCacheSpec {
    host: GgmlKvElementType::F16,
    resident: GgmlKvElementType::F16,
};

/// The decoder KV geometry the loaded pack advertises (MHA: `kv_heads` equals
/// `decoder_heads`, so `layers x 2 x kv_heads x head_dim` values per position
/// equals the `layers x 2 x d_model` values one cross-KV frame costs).
pub(crate) fn cohere_kv_geometry(metadata: &CohereTranscribeExecutionMetadata) -> KvGeometry {
    KvGeometry {
        n_layers: metadata.decoder_layers,
        kv_heads: metadata.decoder_heads,
        head_dim: metadata.decoder_head_dim,
    }
}

/// The cross-KV positions admission charges through the shared position
/// model: the pack's chunk-cap encoder frame capacity, the same figure the
/// decoder runtime allocates its cross cache at (request-independent).
pub(crate) fn cohere_admission_required_positions(
    metadata: &CohereTranscribeExecutionMetadata,
) -> usize {
    cohere_decoder_cross_capacity_frames(*metadata)
}

/// Exact bytes of the f16 self-attention KV cache the decoder runtime
/// allocates at construction: K + V planes of
/// `head_dim x decoder_max_context x decoder_heads` f16 values per layer
/// (`new_persistent_self_kv_tensor_in_arena`'s exact shape). `0` on
/// arithmetic failure -- an under-estimate resolves to "allow", per
/// `crate::capacity`'s fail-open invariant.
pub(crate) fn cohere_admission_fixed_self_kv_bytes(
    metadata: &CohereTranscribeExecutionMetadata,
) -> u64 {
    let plane = GgmlKvElementType::F16
        .plane_nbytes(
            metadata.decoder_head_dim,
            metadata.decoder_max_context,
            metadata.decoder_heads,
        )
        .unwrap_or(0) as u64;
    plane
        .saturating_mul(2) // K + V
        .saturating_mul(metadata.decoder_layers as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacity::kv_bytes_per_position;

    /// Real-checkpoint-shaped metadata (the same values `runtime_contract`'s
    /// `base_metadata` fixture parses: 8-layer MHA decoder at head_dim 128,
    /// 1024-position context, 16kHz/160-hop mel).
    fn reference_metadata() -> CohereTranscribeExecutionMetadata {
        CohereTranscribeExecutionMetadata {
            vocab_size: 50_000,
            encoder_layers: 48,
            encoder_d_model: 1280,
            encoder_heads: 8,
            encoder_head_dim: 160,
            encoder_ffn_dim: 5120,
            encoder_conv_kernel: 9,
            decoder_layers: 8,
            decoder_d_model: 1024,
            decoder_heads: 8,
            decoder_head_dim: 128,
            decoder_ffn_dim: 4096,
            decoder_max_context: 1024,
            decoder_start_token_id: 13_764,
            sample_rate_hz: 16_000,
            n_mels: 128,
            n_fft: 512,
            hop_length: 160,
            win_length: 400,
        }
    }

    #[test]
    fn admission_spec_prices_one_f32_copy_per_cross_frame() {
        let geometry = cohere_kv_geometry(&reference_metadata());
        let per_position = kv_bytes_per_position(&geometry, COHERE_ADMISSION_KV_SPEC)
            .expect("f16 spec accepts head_dim 128");
        // layers x 2 x d_model f32 values per cross frame:
        // 8 * 2 * 1024 * 4 B = 64 KiB -- exactly what the shared model
        // charges per position under the two-f16-copy spec.
        assert_eq!(per_position.total(), 8 * 2 * 1024 * 4);
    }

    #[test]
    fn fixed_self_kv_bytes_match_the_arena_allocation() {
        // 8 layers x 2 (K+V) x 8 heads x 1024 positions x 128 head_dim x 2 B
        // (f16) = 32 MiB, the allocation `new_persistent_self_kv_tensor_in_arena`
        // makes at construction.
        assert_eq!(
            cohere_admission_fixed_self_kv_bytes(&reference_metadata()),
            8 * 2 * 8 * 1024 * 128 * 2
        );
    }

    #[test]
    fn required_positions_are_the_chunk_cap_cross_frames() {
        let metadata = reference_metadata();
        assert_eq!(
            cohere_admission_required_positions(&metadata),
            cohere_decoder_cross_capacity_frames(metadata)
        );
        // Request-independent, and non-degenerate for the shipped shape.
        assert!(cohere_admission_required_positions(&metadata) > 0);
    }
}
