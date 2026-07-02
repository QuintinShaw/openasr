use super::{
    MAX_TENSOR_ELEMENTS, TensorError, TensorLayout, max_data_index, row_major_strides,
    validate_shape_and_strides,
};

#[derive(Debug, Clone)]
pub struct TensorViewF32<'a> {
    pub(crate) data: &'a [f32],
    pub(crate) layout: TensorLayout,
}

impl<'a> TensorViewF32<'a> {
    pub fn contiguous(data: &'a [f32], shape: &[usize]) -> Result<Self, TensorError> {
        let strides = row_major_strides(shape)?;
        Self::from_strided(data, shape, &strides)
    }

    pub fn from_strided(
        data: &'a [f32],
        shape: &[usize],
        strides: &[usize],
    ) -> Result<Self, TensorError> {
        let layout = validate_shape_and_strides(shape, strides)?;
        let max_index = max_data_index(shape, strides)?;
        let required_len = max_index
            .checked_add(1)
            .ok_or(TensorError::ElementCountOverflow)?;
        if required_len > MAX_TENSOR_ELEMENTS {
            return Err(TensorError::ElementCountExceeded {
                declared: required_len as u64,
                limit: MAX_TENSOR_ELEMENTS,
            });
        }
        if data.len() < required_len {
            return Err(TensorError::DataLengthMismatch {
                expected: required_len,
                actual: data.len(),
            });
        }

        Ok(Self { data, layout })
    }

    pub fn rank(&self) -> usize {
        self.layout.shape.len()
    }

    pub fn data(&self) -> &'a [f32] {
        self.data
    }

    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }
}
