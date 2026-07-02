use thiserror::Error;

pub const MAX_TENSOR_ELEMENTS: usize = 67_108_864;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLayout {
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) element_count: usize,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TensorError {
    #[error("invalid tensor shape: {0}")]
    InvalidShape(String),
    #[error("invalid tensor strides: {0}")]
    InvalidStrides(String),
    #[error("tensor element count overflow")]
    ElementCountOverflow,
    #[error("tensor element count {declared} exceeds safety limit {limit}")]
    ElementCountExceeded { declared: u64, limit: usize },
    #[error("tensor data length mismatch: expected {expected}, got {actual}")]
    DataLengthMismatch { expected: usize, actual: usize },
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("tensor index out of bounds: {0}")]
    IndexOutOfBounds(String),
}

pub fn checked_element_count(shape: &[usize]) -> Result<usize, TensorError> {
    let mut count = 1_u64;
    for (index, dim) in shape.iter().copied().enumerate() {
        if dim == 0 {
            return Err(TensorError::InvalidShape(format!(
                "shape dimension at index {index} must be greater than zero"
            )));
        }
        count = count
            .checked_mul(dim as u64)
            .ok_or(TensorError::ElementCountOverflow)?;
    }

    if count > MAX_TENSOR_ELEMENTS as u64 {
        return Err(TensorError::ElementCountExceeded {
            declared: count,
            limit: MAX_TENSOR_ELEMENTS,
        });
    }

    usize::try_from(count).map_err(|_| TensorError::ElementCountOverflow)
}

pub fn row_major_strides(shape: &[usize]) -> Result<Vec<usize>, TensorError> {
    if shape.is_empty() {
        return Ok(Vec::new());
    }
    let _ = checked_element_count(shape)?;
    let mut strides = vec![0_usize; shape.len()];
    let mut stride = 1_usize;
    for idx in (0..shape.len()).rev() {
        strides[idx] = stride;
        stride = stride
            .checked_mul(shape[idx])
            .ok_or(TensorError::ElementCountOverflow)?;
    }
    Ok(strides)
}

pub(crate) fn max_data_index(shape: &[usize], strides: &[usize]) -> Result<usize, TensorError> {
    if shape.len() != strides.len() {
        return Err(TensorError::InvalidStrides(format!(
            "stride rank {} must match shape rank {}",
            strides.len(),
            shape.len()
        )));
    }
    if shape.is_empty() {
        return Ok(0);
    }

    let mut max_offset = 0_usize;
    for (&dim, &stride) in shape.iter().zip(strides) {
        if dim == 0 {
            return Err(TensorError::InvalidShape(
                "tensor dimensions must be > 0".to_string(),
            ));
        }
        if stride == 0 {
            return Err(TensorError::InvalidStrides(
                "tensor strides must be > 0".to_string(),
            ));
        }
        let axis_max = (dim - 1)
            .checked_mul(stride)
            .ok_or(TensorError::ElementCountOverflow)?;
        max_offset = max_offset
            .checked_add(axis_max)
            .ok_or(TensorError::ElementCountOverflow)?;
    }
    Ok(max_offset)
}

pub(crate) fn validate_shape_and_strides(
    shape: &[usize],
    strides: &[usize],
) -> Result<TensorLayout, TensorError> {
    if shape.is_empty() {
        return Ok(TensorLayout {
            shape: Vec::new(),
            strides: Vec::new(),
            element_count: 1,
        });
    }

    let element_count = checked_element_count(shape)?;
    if strides.len() != shape.len() {
        return Err(TensorError::InvalidStrides(format!(
            "stride rank {} must match shape rank {}",
            strides.len(),
            shape.len()
        )));
    }

    for (index, stride) in strides.iter().copied().enumerate() {
        if stride == 0 {
            return Err(TensorError::InvalidStrides(format!(
                "stride at index {index} must be greater than zero"
            )));
        }
    }

    let _ = max_data_index(shape, strides)?;

    Ok(TensorLayout {
        shape: shape.to_vec(),
        strides: strides.to_vec(),
        element_count,
    })
}

impl TensorLayout {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn element_count(&self) -> usize {
        self.element_count
    }
}

pub(crate) fn contiguous_layout(shape: &[usize]) -> Result<TensorLayout, TensorError> {
    let strides = row_major_strides(shape)?;
    validate_shape_and_strides(shape, &strides)
}
