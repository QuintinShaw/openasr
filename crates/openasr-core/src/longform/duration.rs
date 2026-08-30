//! Integer duration conversion for the executor-window ceiling.
//!
//! Soft longform knobs (`chunk_seconds`, overlap, padding, cut search) keep
//! [`super::slicing::seconds_to_samples`] (f32 multiply then round). The hard
//! cap that both the slicer and the decoder-state envelope consume is the
//! exact binary-rational ceil of `max_chunk_seconds` at the request rate.

use std::num::NonZeroU32;

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum ExecutorWindowLimitError {
    #[error("longform max_chunk_seconds must be a finite positive duration, got {value}")]
    InvalidDuration { value: String },
}

/// Smallest sample count that covers `max_chunk_seconds` at `sample_rate_hz`.
///
/// Decodes the finite positive f32 at the configuration boundary into its
/// exact binary rational, then performs ceil(rate * seconds) in integers.
/// No float is allowed into topology/oracle arithmetic after this point.
pub(crate) fn executor_window_limit_samples(
    max_chunk_seconds: f32,
    sample_rate_hz: NonZeroU32,
) -> Result<usize, ExecutorWindowLimitError> {
    let invalid = || ExecutorWindowLimitError::InvalidDuration {
        value: max_chunk_seconds.to_string(),
    };
    let bits = max_chunk_seconds.to_bits();
    let sign = bits >> 31;
    let exponent_bits = (bits >> 23) & 0xff;
    let fraction = bits & 0x7f_ff_ff;
    if sign != 0 || exponent_bits == 0xff || (exponent_bits == 0 && fraction == 0) {
        return Err(invalid());
    }
    let (significand, exponent_two): (u128, i32) = if exponent_bits == 0 {
        (u128::from(fraction), -149)
    } else {
        (
            u128::from((1 << 23) | fraction),
            exponent_bits as i32 - 127 - 23,
        )
    };
    let scaled = significand
        .checked_mul(u128::from(sample_rate_hz.get()))
        .ok_or_else(invalid)?;
    let samples = if exponent_two >= 0 {
        scaled
            .checked_shl(exponent_two as u32)
            .ok_or_else(invalid)?
    } else {
        let shift = exponent_two.unsigned_abs();
        if shift >= u128::BITS {
            1
        } else {
            let denominator = 1_u128 << shift;
            scaled.checked_add(denominator - 1).ok_or_else(invalid)? / denominator
        }
    };
    usize::try_from(samples).map_err(|_| invalid())
}

/// Ceiling after longform planning has already validated options and rate.
pub(crate) fn executor_window_limit_samples_checked(
    max_chunk_seconds: f32,
    sample_rate_hz: u32,
) -> usize {
    let rate = NonZeroU32::new(sample_rate_hz)
        .expect("sample_rate_hz must be non-zero after longform planning validation");
    executor_window_limit_samples(max_chunk_seconds, rate).expect(
        "max_chunk_seconds must be a finite positive duration after LongFormOptions::validate",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_boundary_ceil_uses_exact_integer_binary_rational() {
        let rate = NonZeroU32::new(16_000).unwrap();
        assert_eq!(executor_window_limit_samples(30.0, rate).unwrap(), 480_000);
        assert_eq!(executor_window_limit_samples(30.5, rate).unwrap(), 488_000);
        // The stored f32 value is slightly greater than decimal 0.1. Exact
        // integer ceil must retain that conservative final sample.
        assert_eq!(executor_window_limit_samples(0.1, rate).unwrap(), 1_601);
        assert_eq!(
            executor_window_limit_samples(f32::from_bits(1), rate).unwrap(),
            1
        );
        assert!(matches!(
            executor_window_limit_samples(0.0, rate),
            Err(ExecutorWindowLimitError::InvalidDuration { .. })
        ));
        assert!(matches!(
            executor_window_limit_samples(f32::NAN, rate),
            Err(ExecutorWindowLimitError::InvalidDuration { .. })
        ));
    }
}
