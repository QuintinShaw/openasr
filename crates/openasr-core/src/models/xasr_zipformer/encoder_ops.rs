//! Small Zipformer2 math primitives shared by the X-ASR encoder reference and
//! graph implementations.

pub(crate) const SWOOSH_LINEAR_SCALE: f32 = 0.08;
pub(crate) const SWOOSH_R_OFFSET: f32 = 1.0;
pub(crate) const SWOOSH_R_SHIFT: f32 = 0.313_261_7;
pub(crate) const SWOOSH_L_OFFSET: f32 = 4.0;
pub(crate) const SWOOSH_L_SHIFT: f32 = 0.035;

pub(crate) fn swoosh_r(value: f32) -> f32 {
    softplus(value - SWOOSH_R_OFFSET) - SWOOSH_LINEAR_SCALE * value - SWOOSH_R_SHIFT
}

pub(crate) fn swoosh_l(value: f32) -> f32 {
    softplus(value - SWOOSH_L_OFFSET) - SWOOSH_LINEAR_SCALE * value - SWOOSH_L_SHIFT
}

pub(crate) fn apply_swoosh_r(values: &mut [f32]) {
    values
        .iter_mut()
        .for_each(|value| *value = swoosh_r(*value));
}

pub(crate) fn apply_swoosh_l(values: &mut [f32]) {
    values
        .iter_mut()
        .for_each(|value| *value = swoosh_l(*value));
}

/// BiasNorm used by icefall Zipformer2:
///
/// `y = x * rsqrt(mean((x - bias)^2, axis=channel)) * exp(log_scale)`.
pub(crate) fn bias_norm_last_dim(
    values: &mut [f32],
    channels: usize,
    bias: &[f32],
    log_scale: f32,
) -> Result<(), String> {
    if channels == 0 {
        return Err("xasr BiasNorm channels must be > 0".to_string());
    }
    if bias.len() != channels {
        return Err(format!(
            "xasr BiasNorm bias has {} values, expected {channels}",
            bias.len()
        ));
    }
    if !values.len().is_multiple_of(channels) {
        return Err(format!(
            "xasr BiasNorm input has {} values, not divisible by channels {channels}",
            values.len()
        ));
    }
    let scale = log_scale.exp();
    for row in values.chunks_exact_mut(channels) {
        let mean_square = row
            .iter()
            .zip(bias.iter())
            .map(|(&value, &bias)| {
                let centered = value - bias;
                centered * centered
            })
            .sum::<f32>()
            / channels as f32;
        let multiplier = mean_square.powf(-0.5) * scale;
        for value in row {
            *value *= multiplier;
        }
    }
    Ok(())
}

fn softplus(value: f32) -> f32 {
    value.max(0.0) + (-(value.abs())).exp().ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swoosh_variants_match_onnx_expansion_constants() {
        let x = 2.5_f32;
        let r = ((x - 1.0).exp() + 1.0).ln() - 0.08 * x - 0.313_261_7;
        let l = ((x - 4.0).exp() + 1.0).ln() - 0.08 * x - 0.035;
        assert!((swoosh_r(x) - r).abs() < 1.0e-6);
        assert!((swoosh_l(x) - l).abs() < 1.0e-6);
    }

    #[test]
    fn bias_norm_reduces_last_dim_per_frame() {
        let mut values = vec![2.0, 4.0, 6.0, 3.0, 3.0, 3.0];
        let bias = vec![1.0, 2.0, 3.0];
        bias_norm_last_dim(&mut values, 3, &bias, 0.0).unwrap();

        let first_multiplier = ((1.0_f32 + 4.0 + 9.0) / 3.0).powf(-0.5);
        assert!((values[0] - 2.0 * first_multiplier).abs() < 1.0e-6);
        assert!((values[2] - 6.0 * first_multiplier).abs() < 1.0e-6);

        let second_multiplier = ((4.0_f32 + 1.0 + 0.0) / 3.0).powf(-0.5);
        assert!((values[3] - 3.0 * second_multiplier).abs() < 1.0e-6);
        assert!((values[5] - 3.0 * second_multiplier).abs() < 1.0e-6);
    }
}
