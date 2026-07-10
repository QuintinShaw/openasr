//! Single authoritative f32 -> f16 bit-level conversion.
//!
//! # Single source of truth -- read this before adding another copy
//!
//! A 2026-07 audit found **12 independent reimplementations** of this exact
//! bit-twiddling routine scattered across `nn/decoder.rs`, `adapter_pack.rs`,
//! and half a dozen `models/*/package_import.rs` / `encoder_graph.rs` /
//! `decoder_graph.rs` files, encoding at least four different rounding
//! behaviors. One of them (`models/wav2vec2_ctc/encoder_graph.rs`) did not
//! round at all -- it truncated the mantissa unconditionally, which is a real
//! correctness bug: the exact same source f32 weight was quantized to a
//! different f16 bit pattern than every other model family.
//!
//! If you are about to write `fn f32_to_f16_bits(value: f32) -> u16 { ... }`
//! anywhere in this crate: **stop, and call [`f32_to_f16_bits`] (or
//! [`f32_slice_to_f16_bits`] for a batch) instead.** Adding a 13th copy
//! reintroduces the exact drift this module exists to kill. If a call site
//! is in a different crate/module and importing this one is awkward, that is
//! a sign the conversion belongs in a lower shared layer, not a reason to
//! hand-roll another copy.
//!
//! # Semantics
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
