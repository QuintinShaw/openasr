use thiserror::Error;

use crate::ggml_runtime::{
    GgufOwnedWeightTensorPayload, GgufTensorDataReadError, GgufTensorDataReader,
};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct CohereOwnedGgmlWeightPayload {
    pub ggml_type: i32,
    pub dims: Vec<usize>,
    payload: GgufOwnedWeightTensorPayload,
}

impl CohereOwnedGgmlWeightPayload {
    pub(crate) fn bytes(&self) -> &[u8] {
        self.payload.bytes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CohereMatrixLayout {
    RowsByColumns,
    ColumnsByRows,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct CohereVectorWeight {
    pub name: String,
    pub len: usize,
    pub values: Vec<f32>,
    pub raw_ggml: Option<CohereOwnedGgmlWeightPayload>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct CohereMatrixWeight {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
    pub layout: CohereMatrixLayout,
    pub raw_ggml: Option<CohereOwnedGgmlWeightPayload>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct CohereTensorWeight {
    pub name: String,
    pub dims: Vec<usize>,
    pub values: Vec<f32>,
    pub raw_ggml: Option<CohereOwnedGgmlWeightPayload>,
}

#[derive(Debug, Error)]
pub(crate) enum CohereWeightLoadError {
    #[error("cohere-transcribe tensor read failed: {reason}")]
    TensorReadFailed { reason: String },
    #[error("cohere-transcribe tensor '{tensor_name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        tensor_name: String,
        shape: String,
        reason: String,
    },
    #[error("cohere-transcribe tensor '{tensor_name}' contains non-finite values")]
    NonFiniteTensorValues { tensor_name: String },
}

pub(crate) fn load_vector_weight(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    expected_len: usize,
) -> Result<CohereVectorWeight, CohereWeightLoadError> {
    load_vector_weight_impl(reader, tensor_name, expected_len, false)
}

pub(crate) fn load_vector_weight_for_runtime(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    expected_len: usize,
) -> Result<CohereVectorWeight, CohereWeightLoadError> {
    load_vector_weight_impl(reader, tensor_name, expected_len, true)
}

fn load_vector_weight_impl(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    expected_len: usize,
    allow_empty_values_when_raw_runtime_ready: bool,
) -> Result<CohereVectorWeight, CohereWeightLoadError> {
    let payload = reader
        .owned_weight_tensor_payload_by_name(tensor_name)
        .map_err(map_tensor_read_error)?;
    let raw_ggml = if payload.dims.as_slice() == [expected_len] {
        Some(CohereOwnedGgmlWeightPayload {
            ggml_type: payload.element_type.ggml_type(),
            dims: payload.dims.clone(),
            payload,
        })
    } else {
        None
    };
    let values = if allow_empty_values_when_raw_runtime_ready && raw_ggml.is_some() {
        Vec::new()
    } else {
        let values = reader
            .host_tensor_f32_copy_dequantized_by_name(tensor_name, &[expected_len as u64])
            .map_err(map_tensor_read_error)?;
        ensure_all_finite(tensor_name, &values)?;
        values
    };
    Ok(CohereVectorWeight {
        name: tensor_name.to_string(),
        len: expected_len,
        values,
        raw_ggml,
    })
}

pub(crate) fn load_matrix_weight(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    rows: usize,
    cols: usize,
) -> Result<CohereMatrixWeight, CohereWeightLoadError> {
    load_matrix_weight_with_square_layout(reader, tensor_name, rows, cols, None, false)
}

pub(crate) fn load_matrix_weight_for_runtime(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    rows: usize,
    cols: usize,
) -> Result<CohereMatrixWeight, CohereWeightLoadError> {
    load_matrix_weight_with_square_layout(reader, tensor_name, rows, cols, None, true)
}

pub(crate) fn load_embedding_weight(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    rows: usize,
    cols: usize,
) -> Result<CohereMatrixWeight, CohereWeightLoadError> {
    load_matrix_weight_with_square_layout(
        reader,
        tensor_name,
        rows,
        cols,
        Some(CohereMatrixLayout::RowsByColumns),
        false,
    )
}

pub(crate) fn load_embedding_weight_for_runtime(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    rows: usize,
    cols: usize,
) -> Result<CohereMatrixWeight, CohereWeightLoadError> {
    load_matrix_weight_with_square_layout(
        reader,
        tensor_name,
        rows,
        cols,
        Some(CohereMatrixLayout::RowsByColumns),
        true,
    )
}

fn load_matrix_weight_with_square_layout(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    rows: usize,
    cols: usize,
    square_layout_override: Option<CohereMatrixLayout>,
    allow_empty_values_when_raw_runtime_ready: bool,
) -> Result<CohereMatrixWeight, CohereWeightLoadError> {
    let tensor = require_tensor(reader, tensor_name)?;
    if tensor.dims.len() != 2 {
        return Err(CohereWeightLoadError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: render_shape_u64(&tensor.dims),
            reason: "expected rank-2 matrix".to_string(),
        });
    }
    let dim0 = tensor.dims[0] as usize;
    let dim1 = tensor.dims[1] as usize;
    let (layout, expected_shape) = if dim0 == rows && dim1 == cols {
        // When the matrix is square we cannot infer orientation from dimensions alone.
        // Most projection kernels prefer ColumnsByRows, but embeddings are row-major.
        let preferred_layout = if rows == cols {
            square_layout_override.unwrap_or(CohereMatrixLayout::ColumnsByRows)
        } else {
            CohereMatrixLayout::RowsByColumns
        };
        (preferred_layout, vec![rows as u64, cols as u64])
    } else if dim0 == cols && dim1 == rows {
        (
            CohereMatrixLayout::ColumnsByRows,
            vec![cols as u64, rows as u64],
        )
    } else {
        return Err(CohereWeightLoadError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: render_shape_u64(&tensor.dims),
            reason: format!("expected [{rows} x {cols}] or [{cols} x {rows}]"),
        });
    };
    let payload = reader
        .owned_weight_tensor_payload_by_name(tensor_name)
        .map_err(map_tensor_read_error)?;
    let raw_ggml = match layout {
        CohereMatrixLayout::RowsByColumns if payload.dims.as_slice() == [rows, cols] => {
            Some(CohereOwnedGgmlWeightPayload {
                ggml_type: payload.element_type.ggml_type(),
                dims: payload.dims.clone(),
                payload,
            })
        }
        CohereMatrixLayout::ColumnsByRows if payload.dims.as_slice() == [cols, rows] => {
            Some(CohereOwnedGgmlWeightPayload {
                ggml_type: payload.element_type.ggml_type(),
                dims: payload.dims.clone(),
                payload,
            })
        }
        _ => None,
    };
    let values = if allow_empty_values_when_raw_runtime_ready
        && matches!(layout, CohereMatrixLayout::ColumnsByRows)
        && raw_ggml.is_some()
    {
        Vec::new()
    } else {
        let values = reader
            .host_tensor_f32_copy_dequantized_by_name(tensor_name, &expected_shape)
            .map_err(map_tensor_read_error)?;
        ensure_all_finite(tensor_name, &values)?;
        values
    };
    Ok(CohereMatrixWeight {
        name: tensor_name.to_string(),
        rows,
        cols,
        values,
        layout,
        raw_ggml,
    })
}

pub(crate) fn load_tensor_weight_with_rank_for_runtime_expected_type(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    rank: usize,
    expected_ggml_type: i32,
) -> Result<CohereTensorWeight, CohereWeightLoadError> {
    let tensor = require_tensor(reader, tensor_name)?;
    let actual_dims = dims_as_usize(&tensor.dims, tensor_name)?;
    if actual_dims.len() != rank {
        return Err(CohereWeightLoadError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: render_shape_usize(&actual_dims),
            reason: format!("expected rank-{rank} tensor"),
        });
    }
    load_tensor_weight_from_actual_dims_for_runtime_expected_type(
        reader,
        tensor_name,
        &actual_dims,
        expected_ggml_type,
    )
}

pub(crate) fn load_tensor_weight_with_required_dims_and_ranks(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    accepted_ranks: &[usize],
    required_dims: &[usize],
) -> Result<CohereTensorWeight, CohereWeightLoadError> {
    load_tensor_weight_with_required_dims_and_ranks_impl(
        reader,
        tensor_name,
        accepted_ranks,
        required_dims,
        false,
    )
}

pub(crate) fn load_tensor_weight_with_required_dims_and_ranks_for_runtime(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    accepted_ranks: &[usize],
    required_dims: &[usize],
) -> Result<CohereTensorWeight, CohereWeightLoadError> {
    load_tensor_weight_with_required_dims_and_ranks_impl(
        reader,
        tensor_name,
        accepted_ranks,
        required_dims,
        true,
    )
}

fn load_tensor_weight_with_required_dims_and_ranks_impl(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    accepted_ranks: &[usize],
    required_dims: &[usize],
    allow_empty_values_when_raw_runtime_ready: bool,
) -> Result<CohereTensorWeight, CohereWeightLoadError> {
    let tensor = require_tensor(reader, tensor_name)?;
    let actual_dims = dims_as_usize(&tensor.dims, tensor_name)?;
    if !accepted_ranks.contains(&actual_dims.len()) {
        return Err(CohereWeightLoadError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: render_shape_usize(&actual_dims),
            reason: format!("expected tensor rank in {accepted_ranks:?}"),
        });
    }
    for required_dim in required_dims {
        if !actual_dims.iter().any(|value| value == required_dim) {
            return Err(CohereWeightLoadError::InvalidTensorShape {
                tensor_name: tensor_name.to_string(),
                shape: render_shape_usize(&actual_dims),
                reason: format!("expected one dimension to equal {required_dim}"),
            });
        }
    }
    load_tensor_weight_from_actual_dims(
        reader,
        tensor_name,
        &actual_dims,
        allow_empty_values_when_raw_runtime_ready,
    )
}

pub(crate) fn load_tensor_weight_with_required_dims_and_ranks_for_runtime_expected_type(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    accepted_ranks: &[usize],
    required_dims: &[usize],
    expected_ggml_type: i32,
) -> Result<CohereTensorWeight, CohereWeightLoadError> {
    let tensor = require_tensor(reader, tensor_name)?;
    let actual_dims = dims_as_usize(&tensor.dims, tensor_name)?;
    if !accepted_ranks.contains(&actual_dims.len()) {
        return Err(CohereWeightLoadError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: render_shape_usize(&actual_dims),
            reason: format!("expected tensor rank in {accepted_ranks:?}"),
        });
    }
    for required_dim in required_dims {
        if !actual_dims.iter().any(|value| value == required_dim) {
            return Err(CohereWeightLoadError::InvalidTensorShape {
                tensor_name: tensor_name.to_string(),
                shape: render_shape_usize(&actual_dims),
                reason: format!("expected one dimension to equal {required_dim}"),
            });
        }
    }
    load_tensor_weight_from_actual_dims_for_runtime_expected_type(
        reader,
        tensor_name,
        &actual_dims,
        expected_ggml_type,
    )
}

fn load_tensor_weight_from_actual_dims(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    actual_dims: &[usize],
    allow_empty_values_when_raw_runtime_ready: bool,
) -> Result<CohereTensorWeight, CohereWeightLoadError> {
    let raw_ggml = if actual_dims.len() <= 4 {
        let payload = reader
            .owned_weight_tensor_payload_by_name(tensor_name)
            .map_err(map_tensor_read_error)?;
        if payload.dims.as_slice() == actual_dims {
            Some(CohereOwnedGgmlWeightPayload {
                ggml_type: payload.element_type.ggml_type(),
                dims: payload.dims.clone(),
                payload,
            })
        } else {
            None
        }
    } else {
        None
    };
    let values = if allow_empty_values_when_raw_runtime_ready && raw_ggml.is_some() {
        Vec::new()
    } else {
        let actual_dims_u64 = actual_dims
            .iter()
            .map(|value| *value as u64)
            .collect::<Vec<_>>();
        let values = reader
            .host_tensor_f32_copy_dequantized_by_name(tensor_name, &actual_dims_u64)
            .map_err(map_tensor_read_error)?;
        ensure_all_finite(tensor_name, &values)?;
        values
    };
    Ok(CohereTensorWeight {
        name: tensor_name.to_string(),
        dims: actual_dims.to_vec(),
        values,
        raw_ggml,
    })
}

fn load_tensor_weight_from_actual_dims_for_runtime_expected_type(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    actual_dims: &[usize],
    expected_ggml_type: i32,
) -> Result<CohereTensorWeight, CohereWeightLoadError> {
    let raw_ggml = if actual_dims.len() <= 4 {
        let payload = reader
            .owned_weight_tensor_payload_by_name(tensor_name)
            .map_err(map_tensor_read_error)?;
        if payload.dims.as_slice() == actual_dims {
            Some(CohereOwnedGgmlWeightPayload {
                ggml_type: payload.element_type.ggml_type(),
                dims: payload.dims.clone(),
                payload,
            })
        } else {
            None
        }
    } else {
        None
    };
    let values = if raw_ggml
        .as_ref()
        .is_some_and(|raw| raw.ggml_type == expected_ggml_type)
    {
        Vec::new()
    } else {
        let actual_dims_u64 = actual_dims
            .iter()
            .map(|value| *value as u64)
            .collect::<Vec<_>>();
        let values = reader
            .host_tensor_f32_copy_dequantized_by_name(tensor_name, &actual_dims_u64)
            .map_err(map_tensor_read_error)?;
        ensure_all_finite(tensor_name, &values)?;
        values
    };
    Ok(CohereTensorWeight {
        name: tensor_name.to_string(),
        dims: actual_dims.to_vec(),
        values,
        raw_ggml,
    })
}

fn require_tensor<'a>(
    reader: &'a GgufTensorDataReader,
    tensor_name: &str,
) -> Result<&'a crate::GgufTensorMetadata, CohereWeightLoadError> {
    reader.tensor_index().get(tensor_name).ok_or_else(|| {
        CohereWeightLoadError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: "[]".to_string(),
            reason: "tensor is missing from GGUF tensor index".to_string(),
        }
    })
}

fn dims_as_usize(dims: &[u64], tensor_name: &str) -> Result<Vec<usize>, CohereWeightLoadError> {
    dims.iter()
        .map(|value| {
            usize::try_from(*value).map_err(|_| CohereWeightLoadError::InvalidTensorShape {
                tensor_name: tensor_name.to_string(),
                shape: render_shape_u64(dims),
                reason: "tensor dimension does not fit usize".to_string(),
            })
        })
        .collect()
}

fn ensure_all_finite(tensor_name: &str, values: &[f32]) -> Result<(), CohereWeightLoadError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CohereWeightLoadError::NonFiniteTensorValues {
            tensor_name: tensor_name.to_string(),
        });
    }
    Ok(())
}

fn map_tensor_read_error(error: GgufTensorDataReadError) -> CohereWeightLoadError {
    CohereWeightLoadError::TensorReadFailed {
        reason: error.to_string(),
    }
}

pub(crate) fn render_shape_u64(shape: &[u64]) -> String {
    let parts = shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{parts}]")
}

pub(crate) fn render_shape_usize(shape: &[usize]) -> String {
    let parts = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{parts}]")
}
