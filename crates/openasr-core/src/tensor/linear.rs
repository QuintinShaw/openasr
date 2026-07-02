use super::{TensorError, TensorViewF32};

pub fn linear_f32(
    input: &TensorViewF32<'_>,
    weight: &TensorViewF32<'_>,
    bias: Option<&[f32]>,
) -> Result<Vec<f32>, TensorError> {
    if input.rank() != 2 || weight.rank() != 2 {
        return Err(TensorError::ShapeMismatch(
            "linear expects rank-2 input and weight".to_string(),
        ));
    }

    let batch = input.layout.shape[0];
    let input_dim = input.layout.shape[1];
    let output_dim = weight.layout.shape[0];
    let weight_input_dim = weight.layout.shape[1];

    if input_dim != weight_input_dim {
        return Err(TensorError::ShapeMismatch(format!(
            "linear input dimension {input_dim} does not match weight input dimension {weight_input_dim}"
        )));
    }

    if let Some(values) = bias
        && values.len() != output_dim
    {
        return Err(TensorError::ShapeMismatch(format!(
            "linear bias length {} does not match output dimension {output_dim}",
            values.len()
        )));
    }

    let len = batch
        .checked_mul(output_dim)
        .ok_or(TensorError::ElementCountOverflow)?;
    let mut output = vec![0.0_f32; len];
    let input_row_stride = input.layout.strides[0];
    let input_col_stride = input.layout.strides[1];
    let weight_row_stride = weight.layout.strides[0];
    let weight_col_stride = weight.layout.strides[1];
    let contiguous_inner = input_col_stride == 1 && weight_col_stride == 1;

    for batch_idx in 0..batch {
        let input_row_offset = batch_idx
            .checked_mul(input_row_stride)
            .ok_or(TensorError::ElementCountOverflow)?;
        for out_idx in 0..output_dim {
            let weight_row_offset = out_idx
                .checked_mul(weight_row_stride)
                .ok_or(TensorError::ElementCountOverflow)?;
            let mut sum = if contiguous_inner {
                dot_f32_contiguous(
                    &input.data[input_row_offset..input_row_offset + input_dim],
                    &weight.data[weight_row_offset..weight_row_offset + input_dim],
                )
            } else {
                dot_f32_strided(
                    input.data,
                    input_row_offset,
                    input_col_stride,
                    weight.data,
                    weight_row_offset,
                    weight_col_stride,
                    input_dim,
                )
            };
            if let Some(bias_values) = bias {
                sum += bias_values[out_idx];
            }
            output[batch_idx * output_dim + out_idx] = sum;
        }
    }

    Ok(output)
}

fn dot_f32_contiguous(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let mut sum = 0.0_f32;
    let mut i = 0usize;
    while i + 8 <= left.len() {
        sum += left[i] * right[i]
            + left[i + 1] * right[i + 1]
            + left[i + 2] * right[i + 2]
            + left[i + 3] * right[i + 3]
            + left[i + 4] * right[i + 4]
            + left[i + 5] * right[i + 5]
            + left[i + 6] * right[i + 6]
            + left[i + 7] * right[i + 7];
        i += 8;
    }
    while i < left.len() {
        sum += left[i] * right[i];
        i += 1;
    }
    sum
}

fn dot_f32_strided(
    left: &[f32],
    left_base: usize,
    left_stride: usize,
    right: &[f32],
    right_base: usize,
    right_stride: usize,
    len: usize,
) -> f32 {
    let mut sum = 0.0_f32;
    for i in 0..len {
        sum += left[left_base + i * left_stride] * right[right_base + i * right_stride];
    }
    sum
}
