//! moss-transcribe-diarize capacity derivation: assembles the family's
//! [`IntegralWindowDerivation`] from PACK METADATA (the loaded checkpoint's
//! decoder geometry + adaptor merge size) and family constants, so the
//! integral window is a computed quantity rather than margin-note arithmetic
//! beside `integral_seconds: 300.0`.
//!
//! Phase 0: ZERO production callers. The family still serves the declared
//! `OpenAsrLongformSliceShape::ScopedSlices` constants; the tests below run
//! the derivation in parallel and pin it equal to those constants (the
//! regression anchors that make Phase 1's switchover behavior-preserving).
//! Reject-not-degrade is already the live behavior this derivation protects:
//! an over-limit request fails closed via
//! `moss_td_request_kv_cache_positions` -> `AudioExceedsContext`, and the
//! derived window never depends on host memory (see `crate::capacity`'s
//! invariants).
//!
//! Same Phase 0 dead-code semantics as `crate::capacity` itself: no
//! production caller consumes the assembly yet, so release builds are
//! permitted to leave it uncalled. Remove the allowance when Phase 1 wires
//! the derivation into pack load.
#![cfg_attr(not(test), allow(dead_code))]

use std::num::NonZeroU32;

use crate::capacity::{IntegralWindowDerivation, KvGeometry};

use super::decode_prompt::{AUDIO_TOKENS_PER_SECOND, TIME_MARKER_EVERY_SECONDS};
use super::executor::{
    CHUNK_SAMPLES, HOP_LENGTH, MOSS_TD_MAX_GENERATED_TOKENS, SAMPLE_RATE_HZ,
    WHISPER_ENCODER_CONV_STRIDE, moss_td_chunk_token_length,
};
use super::runtime_contract::{MossTdDecoderMetadata, moss_td_kv_cache_positions};

/// Tokens of the fixed ChatML wrapper around the audio span: the 14-token
/// `<|im_start|>system...<|im_start|>user\n` prefix + the `audio_start`
/// delimiter + the `audio_end` delimiter + the 70-token instruction/ChatML
/// suffix. Measured token-for-token against the real golden fixture
/// (`tmp/moss-td/golden/jfk.json`'s `prompt_input_ids`: 227 tokens = 14
/// prefix + 1 audio-start + 141 audio span + 1 audio-end + 70 suffix, where
/// the 141-token span at 11s is 138 pad tokens + 3 marker digits -- the
/// markers are modeled separately by `marker_every_seconds`, and
/// `crate::capacity::tests::marker_digit_tokens_matches_the_real_prompt_construction`
/// pins that split against the same fixture). The flat 512-token overhead
/// this replaced was only conservatively correct by luck below ~1000s; this
/// number is the honest fixed term and the growth term is derived.
pub(crate) const MOSS_TD_FIXED_PROMPT_TOKENS: usize = 86;

/// Densest generation demand this family has actually been measured against
/// (dense overlapping Mandarin meeting audio -- AliMeeting `R8001_M8004`,
/// `R8007_M8010` -- exhausted a 12 tokens/s allowance on a 180s slice, so it
/// needs upwards of 12.7; the same figure `executor.rs`'s per-second
/// allowance doc comment cites). The integral window's required generation
/// is `ceil(window * this)`, clamped at the runaway backstop.
pub(crate) const MOSS_TD_DENSEST_MEASURED_TOKENS_PER_SECOND: f32 = 12.7;

/// The decoder KV geometry the loaded pack advertises.
pub(crate) fn moss_td_kv_geometry(decoder: &MossTdDecoderMetadata) -> KvGeometry {
    KvGeometry {
        n_layers: decoder.n_layers,
        kv_heads: decoder.n_kv_heads,
        head_dim: decoder.head_dim,
    }
}

/// The family's integral-window derivation inputs, assembled from the loaded
/// pack (`decoder` geometry + `merge_size` from `moss_td.adaptor.merge_size`)
/// and family constants. The position ceiling is the pack's advertised
/// `max_positions` clamped to the family preallocation cap -- the same clamp
/// the runtime decoder applies (`moss_td_kv_cache_positions`), so derivation
/// and runtime argue from one ceiling.
pub(crate) fn moss_td_integral_window_derivation(
    decoder: &MossTdDecoderMetadata,
    merge_size: usize,
) -> IntegralWindowDerivation {
    let token_stride = HOP_LENGTH * WHISPER_ENCODER_CONV_STRIDE * merge_size.max(1);
    IntegralWindowDerivation {
        kv_position_ceiling: moss_td_kv_cache_positions(decoder.max_positions),
        chunk_seconds: CHUNK_SAMPLES as f32 / SAMPLE_RATE_HZ as f32,
        audio_tokens_per_chunk: moss_td_chunk_token_length(CHUNK_SAMPLES, token_stride),
        audio_tokens_per_second: AUDIO_TOKENS_PER_SECOND,
        fixed_prompt_tokens: MOSS_TD_FIXED_PROMPT_TOKENS,
        marker_every_seconds: NonZeroU32::new(TIME_MARKER_EVERY_SECONDS),
        densest_generated_tokens_per_second: MOSS_TD_DENSEST_MEASURED_TOKENS_PER_SECOND,
        max_generated_tokens: MOSS_TD_MAX_GENERATED_TOKENS,
    }
}

/// Decoder metadata matching the real shipped checkpoint -- the same values
/// `runtime_contract::tests::full_metadata` parses (28L Qwen3 decoder, 8 KV
/// heads, head_dim 128, raw RoPE ceiling 131072). Shared by this module's
/// regression anchors and `executor.rs`'s pin tests so every capacity
/// arithmetic check argues from one checkpoint-faithful geometry.
#[cfg(test)]
pub(crate) fn shipped_pack_decoder_fixture() -> MossTdDecoderMetadata {
    MossTdDecoderMetadata {
        n_layers: 28,
        d_model: 1024,
        ffn_dim: 3072,
        n_heads: 16,
        n_kv_heads: 8,
        head_dim: 128,
        vocab_size: 151_936,
        max_positions: 131_072,
        audio_start_token_id: 151_669,
        audio_end_token_id: 151_670,
        audio_pad_token_id: 151_671,
    }
}

/// The real shipped pack's `moss_td.adaptor.merge_size`.
#[cfg(test)]
pub(crate) const SHIPPED_MERGE_SIZE: usize = 4;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID;
    use crate::capacity::{
        AudioFrontendCapacityBasis, FrontendGeometry, audio_tokens_per_second,
        frontend_capacity_basis, kv_bytes_at_positions, kv_bytes_per_position,
    };
    use crate::host::{MIN_SPEC_TOTAL_MEMORY_BYTES, host_memory_budget_bytes};
    use crate::nn::decoder::LlmKvCacheSpec;

    use super::super::executor::AUDIO_TOKENS_PER_SECOND_FOR_LIMIT;
    use super::super::runtime_contract::MOSS_TD_MAX_KV_CACHE_POSITIONS;

    fn shipped_pack_decoder() -> MossTdDecoderMetadata {
        shipped_pack_decoder_fixture()
    }

    fn declared_integral_seconds() -> f32 {
        let crate::arch::OpenAsrLongformSliceShape::ScopedSlices {
            integral_seconds, ..
        } = crate::arch::longform_slice_shape_for_model_architecture(MOSS_TD_GGML_ARCHITECTURE_ID)
        else {
            panic!("moss-transcribe-diarize must declare ScopedSlices");
        };
        integral_seconds
    }

    /// THE regression anchor: feeding the real pack's metadata through the
    /// derivation reproduces the declared `integral_seconds` exactly. If this
    /// ever moves, the derived window changed shipped behavior -- update it
    /// deliberately, with the reason in the commit (same discipline as the
    /// golden diffs).
    #[test]
    fn derived_integral_window_equals_the_declared_constant() {
        let derivation =
            moss_td_integral_window_derivation(&shipped_pack_decoder(), SHIPPED_MERGE_SIZE);
        let derived = crate::capacity::derive_integral_seconds(&derivation)
            .expect("the shipped pack geometry must admit an integral window");
        assert_eq!(
            derived, 300.0,
            "derived integral window moved off the declared value"
        );
        assert_eq!(
            derived,
            declared_integral_seconds(),
            "derivation and the architecture descriptor must agree"
        );
    }

    /// The derived window is maximal, pinned from both sides: 300s fits the
    /// 8192-position ceiling and the next 30s chunk up does not (the 4096
    /// generation backstop is what binds at 330s: 4389 prompt + 4096 = 8485).
    #[test]
    fn derived_window_is_the_largest_one_the_context_can_serve() {
        let derivation =
            moss_td_integral_window_derivation(&shipped_pack_decoder(), SHIPPED_MERGE_SIZE);
        assert_eq!(
            derivation.kv_position_ceiling,
            MOSS_TD_MAX_KV_CACHE_POSITIONS
        );
        assert!(derivation.required_positions_for_chunks(10) <= derivation.kv_position_ceiling);
        assert!(derivation.required_positions_for_chunks(11) > derivation.kv_position_ceiling);
    }

    /// The derivation assembles the same prompt arithmetic the pin tests in
    /// `executor.rs` hold the declared constants to (Phase 0's parallel
    /// equality): honest fixed wrapper + audio tokens + derived marker
    /// digits, not the flat 512 overhead those tests used before.
    #[test]
    fn derived_prompt_arithmetic_is_pinned() {
        let derivation =
            moss_td_integral_window_derivation(&shipped_pack_decoder(), SHIPPED_MERGE_SIZE);
        assert_eq!(derivation.chunk_seconds, 30.0);
        assert_eq!(derivation.audio_tokens_per_chunk, 375);
        assert_eq!(derivation.max_generated_tokens, 4096);
        // 300s: 86 fixed + 10 chunks * 375 audio + 160 marker digits = 3996.
        assert_eq!(derivation.prompt_tokens_for_chunks(10), 3996);
    }

    /// Worst-case KV bytes per position, split by copy (the figure the old
    /// `~30 GB` comment got wrong by counting only the host f32 copy).
    #[test]
    fn kv_bytes_per_position_matches_the_two_copy_reality() {
        let geometry = moss_td_kv_geometry(&shipped_pack_decoder());
        // DEFAULT (host f32 + resident f16): 448 rows/position
        // (28 layers * 2 * 8 kv-heads) at 512 B + 256 B.
        let default = kv_bytes_per_position(&geometry, LlmKvCacheSpec::DEFAULT).expect("default");
        assert_eq!(default.host, 224 * 1024);
        assert_eq!(default.resident, 112 * 1024);
        assert_eq!(default.total(), 336 * 1024);
        // Q8_0 (both copies q8_0): 136 B/row -> 2.8x under DEFAULT, not ~4x.
        let q8_0 = kv_bytes_per_position(&geometry, LlmKvCacheSpec::Q8_0).expect("q8_0");
        assert_eq!(q8_0.host, 28 * 2 * 8 * 136); // 60928 = 59.5 KiB
        assert_eq!(q8_0.resident, 28 * 2 * 8 * 136);
        assert_eq!(q8_0.total(), 2 * 28 * 2 * 8 * 136); // 119 KiB
    }

    /// The declared 8192-position preallocation cap fits the min-spec
    /// machine's memory budget under EVERY runtime KV policy -- the
    /// derivation-verification pin that keeps 8192 a DECLARED constant
    /// rather than a runtime-derived one (deriving it from a min-spec
    /// rationale would take worst-case DEFAULT bytes/position and could
    /// silently tighten the shipped cap; pinning the fit checks the same
    /// arithmetic without moving the number).
    #[test]
    fn declared_position_cap_fits_min_spec_budget_under_every_policy() {
        let geometry = moss_td_kv_geometry(&shipped_pack_decoder());
        let budget = host_memory_budget_bytes(MIN_SPEC_TOTAL_MEMORY_BYTES);
        assert_eq!(MIN_SPEC_TOTAL_MEMORY_BYTES, 8 * 1024 * 1024 * 1024);
        assert_eq!(budget, 6 * 1024 * 1024 * 1024); // 75% precedent

        // Worst case DEFAULT: 8192 * 336 KiB = 2.625 GiB << 6 GiB.
        let default = kv_bytes_at_positions(
            &geometry,
            LlmKvCacheSpec::DEFAULT,
            MOSS_TD_MAX_KV_CACHE_POSITIONS,
        )
        .expect("default");
        assert_eq!(default.total(), 8192 * 336 * 1024); // 2.625 GiB
        assert!(
            default.total() <= budget,
            "8192 positions at the worst-case DEFAULT policy must fit the 8 GiB min-spec budget"
        );
        // Q8_0: 8192 * 119 KiB = 952 MiB << 6 GiB.
        let q8_0 = kv_bytes_at_positions(
            &geometry,
            LlmKvCacheSpec::Q8_0,
            MOSS_TD_MAX_KV_CACHE_POSITIONS,
        )
        .expect("q8_0");
        assert_eq!(q8_0.total(), 8192 * 2 * 28 * 2 * 8 * 136);
        assert!(q8_0.total() <= budget);
    }

    /// Drift guard: the capacity frontend registry's constant geometry is the
    /// same architectural fact the executor's constants and the decode
    /// prompt's marker cadence state -- three places, one pinned number.
    #[test]
    fn frontend_registry_agrees_with_the_family_constants() {
        let AudioFrontendCapacityBasis::Constant(geometry) =
            frontend_capacity_basis(crate::arch::MOSS_TD_AUDIO_FRONTEND_ID).expect("moss row")
        else {
            panic!("moss frontend basis must be a Constant");
        };
        assert_eq!(
            geometry,
            &FrontendGeometry {
                sample_rate_hz: SAMPLE_RATE_HZ,
                hop_length: HOP_LENGTH,
                encoder_conv_stride: WHISPER_ENCODER_CONV_STRIDE,
                adaptor_merge_size: SHIPPED_MERGE_SIZE,
            }
        );
        // The derived rate equals both copies the family already states.
        let rate = audio_tokens_per_second(geometry);
        assert_eq!(rate, AUDIO_TOKENS_PER_SECOND);
        assert_eq!(rate, AUDIO_TOKENS_PER_SECOND_FOR_LIMIT);
        // And the stride product reproduces the chunk's audio-token count.
        assert_eq!(
            moss_td_chunk_token_length(
                CHUNK_SAMPLES,
                geometry.hop_length * geometry.encoder_conv_stride * geometry.adaptor_merge_size
            ),
            375
        );
    }
}
