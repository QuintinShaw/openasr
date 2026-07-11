//! Single authoritative f32 <-> f16 bit-level conversions.
//!
//! # Single source of truth -- read this before adding another copy
//!
//! A 2026-07 audit found **12 independent reimplementations** of the f32 ->
//! f16 direction scattered across `nn/decoder.rs`, `adapter_pack.rs`, and
//! half a dozen `models/*/package_import.rs` / `encoder_graph.rs` /
//! `decoder_graph.rs` files, encoding at least four different rounding
//! behaviors. One of them (`models/wav2vec2_ctc/encoder_graph.rs`) did not
//! round at all -- it truncated the mantissa unconditionally, which is a real
//! correctness bug: the exact same source f32 weight was quantized to a
//! different f16 bit pattern than every other model family. That direction
//! was consolidated into [`f32_to_f16_bits`] first.
//!
//! A follow-up 2026-07 audit found the same drift in the *reverse* direction:
//! **5 independent reimplementations** of f16 bits -> f32, in
//! `models/whisper/package_import.rs`, `ggml_runtime/gguf_tensor_data.rs`,
//! `models/whisper/ggml_executor.rs` (as `f16_to_f32_local_v0`),
//! `models/qwen/token_embedding.rs`, and `models/local_source_import.rs`.
//! Bit-exact sweep of all 65536 possible `u16` patterns against both `numpy`'s
//! `float16 -> float32` cast and ggml's own `ggml_compute_fp16_to_fp32`
//! (`third_party/openasr-ggml/src/ggml-impl.h`) showed **two of the five were
//! wrong**: the copies in `whisper/package_import.rs` and
//! `gguf_tensor_data.rs` were byte-identical to each other and shared a
//! subnormal-decode bug -- their subnormal-input loop initialized the
//! exponent accumulator to `-1` instead of `-14`, so every subnormal f16
//! input (2046 of the 65536 possible bit patterns: all nonzero mantissas
//! with a zero exponent field, both signs) decoded to exactly **half** the
//! correct magnitude. The other three copies (`ggml_executor.rs`,
//! `token_embedding.rs`, `local_source_import.rs`) were already bit-exact
//! matches for all 65536 patterns and became this shared implementation
//! unchanged. See `f16_bits_to_f32`'s test module for the full-sweep
//! regression test pinning this.
//!
//! If you are about to write `fn f32_to_f16_bits(value: f32) -> u16 { ... }`
//! or `fn f16_bits_to_f32(bits: u16) -> f32 { ... }` anywhere in this crate:
//! **stop, and call [`f32_to_f16_bits`] / [`f16_bits_to_f32`] (or their slice
//! variants) instead.** Adding another copy reintroduces the exact drift
//! this module exists to kill. If a call site is in a different
//! crate/module and importing this one is awkward, that is a sign the
//! conversion belongs in a lower shared layer, not a reason to hand-roll
//! another copy.
//!
//! # Semantics: f32 -> f16 ([`f32_to_f16_bits`])
//!
//! Round-to-nearest-even (IEEE 754 default rounding), matching what a
//! hardware `vcvtps2ph`/`FCVT` instruction or `f32::to_f16` (once stabilized)
//! would produce:
//!
//! - Ties round to the f16 value with an even mantissa LSB.
//! - NaNs preserve sign and a non-zero truncated payload (so distinct NaNs
//!   do not all collapse onto one canonical bit pattern); infinities and
//!   overflowing finite values map to the correctly-signed f16 infinity.
//! - Subnormal f16 results are produced (not flushed to zero) down to the
//!   smallest representable subnormal; values below that flush to a
//!   correctly-signed zero.
//! - Mantissa rounding that overflows into the next binade (e.g. the largest
//!   finite f16 value overflowing to infinity) ripples through the exponent
//!   field via plain integer addition, which is exact because the f16 bit
//!   layout packs sign/exponent/mantissa as one integer.
//!
//! # Semantics: f16 -> f32 ([`f16_bits_to_f32`])
//!
//! Exact (lossless): every finite f16 value, subnormal or normal, converts to
//! the identical mathematical value as an f32; signed zero, signed infinity,
//! and NaN (sign + truncated payload, left-justified into the wider f32
//! mantissa) all round-trip exactly, matching `ggml_compute_fp16_to_fp32`.

/// Convert one `f32` to its IEEE 754 binary16 bit pattern, rounding to
/// nearest with ties to even. See the module docs for full semantics.
pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 0xff {
        if mantissa == 0 {
            return sign | 0x7c00; // +/-inf
        }
        // NaN: keep a non-zero payload (truncated to the 10 f16 mantissa
        // bits) instead of collapsing every NaN onto one canonical pattern.
        let payload = ((mantissa >> 13) as u16).max(1);
        return sign | 0x7c00 | payload;
    }

    let half_exponent = exponent - 127 + 15;

    if half_exponent >= 0x1f {
        // Overflow: finite f32 magnitude too large for f16 -> infinity.
        return sign | 0x7c00;
    }

    if half_exponent <= 0 {
        // Result is subnormal in f16 (or underflows to zero).
        if half_exponent < -10 {
            return sign;
        }
        let mantissa_with_hidden = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        sign | round_to_even_shift_right(mantissa_with_hidden, shift) as u16
    } else {
        // Normal range: round the 23-bit fraction down to 10 bits. A carry
        // out of the rounded mantissa (0x400) adds cleanly into the packed
        // exponent field below it, correctly promoting to the next binade
        // (or to infinity, at the top of the range).
        let rounded_mantissa = round_to_even_shift_right(mantissa, 13) as u16;
        sign | ((half_exponent as u16) << 10).wrapping_add(rounded_mantissa)
    }
}

/// Convert a slice of `f32` values to f16 bit patterns element-wise.
pub(crate) fn f32_slice_to_f16_bits(values: &[f32]) -> Vec<u16> {
    values.iter().copied().map(f32_to_f16_bits).collect()
}

/// Convert an IEEE 754 binary16 bit pattern to its exact `f32` value. See the
/// module docs for full semantics.
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);

    let f32_bits = if exponent == 0 {
        if mantissa == 0 {
            sign // +/-0
        } else {
            // Subnormal f16: normalize the mantissa into an implicit-leading-1
            // form by shifting left until the hidden bit (0x400) appears,
            // tracking how many shifts that took. The f16 subnormal exponent
            // is -14 (bias 15, field 0), so the normalized binade starts
            // there and each shift descends one more power of two.
            let mut mant = mantissa;
            let mut exp = -14_i32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x03ff; // drop the now-explicit hidden bit
            let exponent_f32 = (exp + 127) as u32;
            sign | (exponent_f32 << 23) | (mant << 13)
        }
    } else if exponent == 0x1f {
        // Infinity (mantissa == 0) or NaN (mantissa != 0): widen the mantissa
        // into the f32 payload position, preserving a NaN's non-zero payload.
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        // Normal range: f16 bias 15 -> f32 bias 127.
        let exponent_f32 = exponent as u32 + (127 - 15);
        sign | (exponent_f32 << 23) | (mantissa << 13)
    };
    f32::from_bits(f32_bits)
}

/// Convert a slice of f16 bit patterns to `f32` values element-wise.
pub(crate) fn f16_bits_slice_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter().copied().map(f16_bits_to_f32).collect()
}

/// Round `value >> shift` to the nearest integer, ties to even, using only
/// integer arithmetic (exact for every input; no float intermediate).
fn round_to_even_shift_right(value: u32, shift: u32) -> u32 {
    if shift == 0 {
        return value;
    }
    let truncated = value >> shift;
    let round_bit = 1_u32 << (shift - 1);
    // The bits below the round bit, plus the round bit itself: this is
    // exactly the fractional remainder being discarded.
    let remainder = value & round_bit.wrapping_shl(1).wrapping_sub(1);
    if remainder > round_bit || (remainder == round_bit && (truncated & 1) != 0) {
        truncated + 1
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(value: f32) -> u16 {
        f32_to_f16_bits(value)
    }

    #[test]
    fn zero_preserves_sign() {
        assert_eq!(bits(0.0), 0x0000);
        assert_eq!(bits(-0.0), 0x8000);
    }

    #[test]
    fn common_exact_values_round_trip() {
        assert_eq!(bits(1.0), 0x3c00);
        assert_eq!(bits(-1.0), 0xbc00);
        assert_eq!(bits(2.0), 0x4000);
        assert_eq!(bits(0.5), 0x3800);
        assert_eq!(bits(65504.0), 0x7bff); // largest finite f16
    }

    #[test]
    fn infinities() {
        assert_eq!(bits(f32::INFINITY), 0x7c00);
        assert_eq!(bits(f32::NEG_INFINITY), 0xfc00);
    }

    #[test]
    fn overflow_rounds_up_to_infinity() {
        // 65520.0 is exactly halfway between the largest finite f16
        // (65504.0) and the next representable step (65536.0, which is not
        // representable as a finite f16); ties-to-even on the mantissa
        // carries into infinity here because the rounded mantissa is odd.
        assert_eq!(bits(65520.0), 0x7c00);
        // Comfortably out of range in either direction.
        assert_eq!(bits(1.0e9), 0x7c00);
        assert_eq!(bits(-1.0e9), 0xfc00);
    }

    #[test]
    fn nan_preserves_sign_and_nonzero_payload() {
        let positive_nan = bits(f32::NAN);
        assert_eq!(positive_nan & 0x8000, 0);
        assert_eq!(positive_nan & 0x7c00, 0x7c00);
        assert_ne!(positive_nan & 0x03ff, 0, "NaN payload must stay non-zero");

        let negative_nan = bits(-f32::NAN);
        assert_eq!(negative_nan & 0x8000, 0x8000);
        assert_eq!(negative_nan & 0x7c00, 0x7c00);
        assert_ne!(negative_nan & 0x03ff, 0);

        // A NaN with a payload that happens to land on zero after truncation
        // to 10 bits must still report as NaN, not accidentally as infinity.
        let payload_truncates_to_zero = f32::from_bits(0x7f80_1000); // mantissa 0x001000
        let out = bits(payload_truncates_to_zero);
        assert_eq!(out & 0x7c00, 0x7c00);
        assert_ne!(out, 0x7c00, "must not collapse to infinity");
    }

    #[test]
    fn subnormal_boundaries() {
        // Smallest positive f16 subnormal: 2^-24.
        assert_eq!(bits(f32::from_bits(0x3380_0000)), 0x0001);
        // Halfway between 0 and the smallest subnormal rounds down to zero
        // (tie-to-even: mantissa LSB of 0 is even).
        assert_eq!(bits(f32::from_bits(0x3300_0000)), 0x0000);
        // Largest f16 subnormal: 1023 * 2^-24.
        assert_eq!(bits(f32::from_bits(0x387f_c000)), 0x03ff);
        // Smallest normal f16 value: 2^-14 = 1024 * 2^-24.
        assert_eq!(bits(f32::from_bits(0x3880_0000)), 0x0400);
    }

    #[test]
    fn rounding_ties_to_even_in_normal_range() {
        // 1.0 + 2^-11 is exactly halfway between two adjacent f16 values
        // around 1.0; the lower candidate (0x3c00, even mantissa) wins.
        let tie_down = f32::from_bits(0x3f80_1000);
        assert_eq!(bits(tie_down), 0x3c00);

        // 1.0 + 3 * 2^-11 is exactly halfway between the next pair; the
        // upper candidate (0x3c02, even mantissa) wins this time.
        let tie_up = f32::from_bits(0x3f80_3000);
        assert_eq!(bits(tie_up), 0x3c02);

        // Just above a tie rounds up regardless of parity.
        let just_above_tie = f32::from_bits(0x3f80_1001);
        assert_eq!(bits(just_above_tie), 0x3c01);

        // Just below a tie rounds down regardless of parity.
        let just_below_tie = f32::from_bits(0x3f80_0fff);
        assert_eq!(bits(just_below_tie), 0x3c00);
    }

    #[test]
    fn mantissa_carry_ripples_into_exponent() {
        // A value whose mantissa rounds up to exactly 0x400 (2048 in the
        // 11-bit significand) must carry into the exponent with mantissa 0,
        // not silently truncate.
        let value = f32::from_bits(0x3fff_ffff); // just under 2.0, rounds up to 2.0
        assert_eq!(bits(value), 0x4000);
    }

    #[test]
    fn slice_conversion_matches_scalar() {
        let values = [0.0_f32, 1.0, -2.5, f32::INFINITY];
        let expected: Vec<u16> = values.iter().copied().map(f32_to_f16_bits).collect();
        assert_eq!(f32_slice_to_f16_bits(&values), expected);
    }
}

#[cfg(test)]
mod f16_bits_to_f32_tests {
    use super::*;

    #[test]
    fn zero_preserves_sign() {
        assert_eq!(f16_bits_to_f32(0x0000).to_bits(), 0f32.to_bits());
        assert_eq!(f16_bits_to_f32(0x8000).to_bits(), (-0f32).to_bits());
    }

    #[test]
    fn common_exact_values_round_trip() {
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0xbc00), -1.0);
        assert_eq!(f16_bits_to_f32(0x4000), 2.0);
        assert_eq!(f16_bits_to_f32(0x3800), 0.5);
        assert_eq!(f16_bits_to_f32(0x7bff), 65504.0); // largest finite f16
    }

    #[test]
    fn infinities() {
        assert_eq!(f16_bits_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_bits_to_f32(0xfc00), f32::NEG_INFINITY);
    }

    #[test]
    fn nan_preserves_sign_and_nonzero_payload() {
        let positive_nan = f16_bits_to_f32(0x7c01);
        assert!(positive_nan.is_nan());
        assert_eq!(positive_nan.to_bits() & 0x8000_0000, 0);
        assert_ne!(positive_nan.to_bits() & 0x007f_ffff, 0);

        let negative_nan = f16_bits_to_f32(0xfe00);
        assert!(negative_nan.is_nan());
        assert_eq!(negative_nan.to_bits() & 0x8000_0000, 0x8000_0000);
        assert_ne!(negative_nan.to_bits() & 0x007f_ffff, 0);
    }

    #[test]
    fn subnormal_boundaries() {
        // Smallest positive f16 subnormal (bits 0x0001) is exactly 2^-24.
        // This is the exact value the pre-consolidation `whisper/package_import.rs`
        // / `gguf_tensor_data.rs` copy got wrong (it returned 2^-25, half the
        // correct magnitude) -- see the module docs for the audit writeup.
        assert_eq!(f16_bits_to_f32(0x0001), f32::from_bits(0x3380_0000)); // 2^-24
        assert_eq!(f16_bits_to_f32(0x0001), 2.0f32.powi(-24));
        // Largest f16 subnormal: 1023 * 2^-24.
        assert_eq!(f16_bits_to_f32(0x03ff), f32::from_bits(0x387f_c000));
        // Smallest normal f16 value: 2^-14 = 1024 * 2^-24, exactly at the
        // subnormal/normal boundary the decode loop must land on precisely.
        assert_eq!(f16_bits_to_f32(0x0400), f32::from_bits(0x3880_0000));
        assert_eq!(f16_bits_to_f32(0x0400), 2.0f32.powi(-14));
        // Sign is preserved through the subnormal branch.
        assert_eq!(f16_bits_to_f32(0x8001), -(2.0f32.powi(-24)));
    }

    #[test]
    fn f16_to_f32_round_trips_through_f32_to_f16_bits_for_exactly_representable_values() {
        // Every value f16 can represent exactly should survive f32 -> f16 ->
        // f32 unchanged; this exercises both directions of the shared module
        // against each other for a spread of exponents/mantissas.
        let values = [
            0.0_f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            2.0,
            65504.0,
            -65504.0,
            1024.0,
            2.0f32.powi(-12), // exactly representable in f16
        ];
        for value in values {
            let bits = f32_to_f16_bits(value);
            let round_tripped = f16_bits_to_f32(bits);
            assert_eq!(
                round_tripped.to_bits(),
                value.to_bits(),
                "value {value} did not round-trip via f16 bits (got {round_tripped})"
            );
        }
    }

    /// Historical duplicate that lived in `models/whisper/package_import.rs`
    /// and byte-identically in `ggml_runtime/gguf_tensor_data.rs` before this
    /// consolidation. Kept here, inline, purely as a regression oracle: it
    /// reproduces the exact subnormal-decode bug (exponent accumulator
    /// initialized to `-1` instead of `-14`) so the test below can pin
    /// *precisely* which 2046 of the 65536 bit patterns the fix changes
    /// behavior for, and assert every other pattern is untouched.
    fn buggy_historical_f16_bits_to_f32(bits: u16) -> f32 {
        let sign = ((bits & 0x8000) as u32) << 16;
        let exponent = ((bits >> 10) & 0x1f) as u32;
        let mantissa = (bits & 0x03ff) as u32;
        let f32_bits = if exponent == 0 {
            if mantissa == 0 {
                sign
            } else {
                let mut mant = mantissa;
                let mut exp = -1_i32; // bug: should start at -14
                while (mant & 0x0400) == 0 {
                    mant <<= 1;
                    exp -= 1;
                }
                mant &= 0x03ff;
                let exponent_f32 = (127 - 15 + 1 + exp) as u32;
                sign | (exponent_f32 << 23) | (mant << 13)
            }
        } else if exponent == 0x1f {
            sign | 0x7f80_0000 | (mantissa << 13)
        } else {
            let exponent_f32 = exponent + (127 - 15);
            sign | (exponent_f32 << 23) | (mantissa << 13)
        };
        f32::from_bits(f32_bits)
    }

    /// Full 65536-value sweep, exhaustively pinning behavior against the
    /// pre-consolidation buggy copy above: the shared implementation must
    /// differ *only* on subnormal inputs (f16 exponent field == 0, mantissa
    /// != 0), and must differ from the buggy copy by exactly a factor of two
    /// there (the bug's off-by-one exponent bias). This documents, byte for
    /// byte, that the consolidation intentionally changes output only for
    /// bit patterns the old code already got wrong -- it does not "silently"
    /// unify onto a different behavior for any value real model weights
    /// would plausibly carry outside that subnormal corner.
    #[test]
    fn matches_buggy_historical_copy_everywhere_except_the_subnormal_bug_it_had() {
        let mut mismatches = 0usize;
        for bits in 0..=u16::MAX {
            let fixed = f16_bits_to_f32(bits);
            let buggy = buggy_historical_f16_bits_to_f32(bits);
            let exponent_field = (bits >> 10) & 0x1f;
            let mantissa_field = bits & 0x03ff;
            let is_subnormal_nonzero = exponent_field == 0 && mantissa_field != 0;

            if fixed.to_bits() == buggy.to_bits() {
                assert!(
                    !is_subnormal_nonzero,
                    "bits {bits:#06x} unexpectedly matched the buggy copy on a \
                     subnormal input; the bug should make these differ"
                );
                continue;
            }

            mismatches += 1;
            assert!(
                is_subnormal_nonzero,
                "bits {bits:#06x} diverged from the buggy historical copy \
                 outside the known subnormal bug -- consolidation must not \
                 change behavior anywhere else"
            );
            // The bug is exactly a missing factor of two (exponent off by one).
            assert_eq!(
                fixed,
                buggy * 2.0,
                "bits {bits:#06x}: fix should be exactly 2x the buggy value"
            );
        }
        // 1023 nonzero mantissas x 2 signs = 2046 subnormal bit patterns.
        assert_eq!(mismatches, 2046);
    }

    /// Independent oracle: ggml's own `ggml_compute_fp16_to_fp32`
    /// (`third_party/openasr-ggml/src/ggml-impl.h`), transcribed as a
    /// standalone Rust port purely for this test, using its bit-trick
    /// algorithm (magic-number float arithmetic instead of a normalize
    /// loop). Every GGUF tensor this crate reads is ultimately produced or
    /// consumed by that C code, so bit-exact agreement across all 65536
    /// patterns is the correctness bar, independent of this module's own
    /// (differently structured) implementation.
    fn ggml_reference_f16_bits_to_f32(bits: u16) -> f32 {
        let w = u32::from(bits) << 16;
        let sign = w & 0x8000_0000;
        let two_w = w.wrapping_add(w);

        let exp_offset: u32 = 0xE0 << 23;
        let exp_scale = f32::from_bits(0x7800000);
        let normalized_value = f32::from_bits((two_w >> 4).wrapping_add(exp_offset)) * exp_scale;

        let magic_mask: u32 = 126 << 23;
        let magic_bias = 0.5f32;
        let denormalized_value = f32::from_bits((two_w >> 17) | magic_mask) - magic_bias;

        let denormalized_cutoff: u32 = 1 << 27;
        let result = sign
            | if two_w < denormalized_cutoff {
                denormalized_value.to_bits()
            } else {
                normalized_value.to_bits()
            };
        f32::from_bits(result)
    }

    #[test]
    fn matches_ggml_reference_bit_exact_for_every_u16_pattern() {
        // NaN payloads are deliberately excluded from the bit-exact
        // comparison: the ggml reference here computes the NaN case via a
        // float *multiplication* against a NaN bit pattern
        // (`normalized_value = f32::from_bits(...) * exp_scale`), and IEEE
        // 754 does not mandate which payload survives arithmetic on a NaN
        // operand -- that is architecture/compiler dependent, so pinning a
        // specific payload from this Rust port would be testing an
        // unportable accident, not a real invariant. `f16_bits_to_f32` uses
        // direct bit manipulation instead (matching the 3-of-5 pre-existing
        // duplicate implementations this consolidation kept unchanged), so
        // its NaN payload is deterministic; that invariant is covered by
        // `nan_preserves_sign_and_nonzero_payload` above.
        for bits in 0..=u16::MAX {
            let ours = f16_bits_to_f32(bits);
            let theirs = ggml_reference_f16_bits_to_f32(bits);
            if ours.is_nan() && theirs.is_nan() {
                continue;
            }
            assert_eq!(
                ours.to_bits(),
                theirs.to_bits(),
                "bits {bits:#06x}: ours={ours:?} ggml={theirs:?}"
            );
        }
    }

    #[test]
    fn slice_conversion_matches_scalar() {
        let values: [u16; 5] = [0x0000, 0x3c00, 0xbc00, 0x7c00, 0x0001];
        let expected: Vec<f32> = values.iter().copied().map(f16_bits_to_f32).collect();
        assert_eq!(f16_bits_slice_to_f32(&values), expected);
    }
}
