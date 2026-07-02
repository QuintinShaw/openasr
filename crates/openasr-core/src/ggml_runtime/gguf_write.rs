use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::{CString, c_void},
    path::{Path, PathBuf},
    ptr::{self, null},
};

use thiserror::Error;

use super::ffi;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GgufWriteValue {
    String(String),
    U32(u32),
    StringArray(Vec<String>),
    U32Array(Vec<u32>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub(crate) enum GgufWriteTensorType {
    F32,
    F16,
    Q8_0,
    // K-quants: all use ggml's 256-element superblock (ne0 % 256 == 0). q3_K is
    // the both-backend size/speed lever (~3.4 bpw, -24% bytes vs q4_K → ~proportional
    // decode speedup on the bandwidth-bound path, via ggml's OWN CPU+Metal K-quant
    // kernels — no GPU-specialization). q5_K/q6_K are the quality-recovery rungs.
    Q3_K,
    Q4_K,
    // Reserved quality-recovery rungs: fully wired (ggml_type + quantize allowlist)
    // and ready for per-model quant selection during onboarding, but no importer
    // currently picks them (q3_k/q4_k/q8/fp16 cover the live rungs). Not dead —
    // available; allow until an importer selects them.
    #[allow(dead_code)]
    Q5_K,
    #[allow(dead_code)]
    Q6_K,
}

impl GgufWriteTensorType {
    fn ggml_type(self) -> i32 {
        match self {
            Self::F32 => ffi::GGML_TYPE_F32,
            Self::F16 => ffi::GGML_TYPE_F16,
            Self::Q8_0 => ffi::GGML_TYPE_Q8_0,
            Self::Q3_K => ffi::GGML_TYPE_Q3_K,
            Self::Q4_K => ffi::GGML_TYPE_Q4_K,
            Self::Q5_K => ffi::GGML_TYPE_Q5_K,
            Self::Q6_K => ffi::GGML_TYPE_Q6_K,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GgufWriteTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub tensor_type: GgufWriteTensorType,
    pub data: Vec<u8>,
}

#[derive(Debug, Error)]
pub(crate) enum GgufWriteError {
    #[error("gguf output path already exists: {path}")]
    OutputExists { path: PathBuf },
    #[error("gguf output path cannot be represented as C string: {path}")]
    PathContainsNul { path: String },
    #[error("gguf string field '{field}' cannot contain NUL bytes")]
    StringContainsNul { field: &'static str },
    #[error("gguf metadata key cannot be empty")]
    EmptyMetadataKey,
    #[error("gguf metadata key '{key}' is duplicated")]
    DuplicateMetadataKey { key: String },
    #[error("gguf tensor name cannot be empty")]
    EmptyTensorName,
    #[error("gguf tensor name '{name}' is duplicated")]
    DuplicateTensorName { name: String },
    #[error("gguf tensor '{name}' rank must be 1, 2, 3, or 4; got rank={rank}")]
    UnsupportedTensorRank { name: String, rank: usize },
    #[error("gguf tensor '{name}' dimension at index {index} must be > 0")]
    NonPositiveTensorDimension { name: String, index: usize },
    #[error("gguf tensor '{name}' dimension {value} does not fit i64")]
    TensorDimensionOverflow { name: String, value: u64 },
    #[error("gguf tensor '{name}' element count overflows u64 for dims {dims:?}")]
    TensorElementCountOverflow { name: String, dims: Vec<u64> },
    #[error("gguf tensor '{name}' expected byte length overflows usize")]
    TensorByteLengthOverflow { name: String },
    #[error("gguf tensor '{name}' has invalid ggml block size {block_size}")]
    TensorInvalidBlockSize { name: String, block_size: i64 },
    #[error(
        "gguf tensor '{name}' first dimension {ne0} is not aligned to ggml block size {block_size}"
    )]
    TensorBlockAlignmentMismatch {
        name: String,
        ne0: u64,
        block_size: u64,
    },
    #[error(
        "gguf tensor '{name}' data length {actual} does not match expected {expected} for dims {dims:?}"
    )]
    TensorDataLengthMismatch {
        name: String,
        dims: Vec<u64>,
        expected: usize,
        actual: usize,
    },
    #[error("gguf quantization supports only q8_0/q4_k output, got {tensor_type:?}")]
    TensorQuantizationTypeUnsupported { tensor_type: GgufWriteTensorType },
    #[error(
        "gguf quantization source value count {actual} does not match expected {expected} for dims {dims:?}"
    )]
    TensorQuantizationSourceValueCountMismatch {
        dims: Vec<u64>,
        expected: usize,
        actual: usize,
    },
    #[error("gguf quantization source contains non-finite f32 values")]
    TensorQuantizationSourceNonFinite,
    #[error(
        "gguf quantization produced byte count {actual} but expected {expected} for dims {dims:?} and type {tensor_type:?}"
    )]
    TensorQuantizationSizeMismatch {
        dims: Vec<u64>,
        tensor_type: GgufWriteTensorType,
        expected: usize,
        actual: usize,
    },
    #[error("ggml context allocation size overflow for {tensor_count} tensor definitions")]
    GgmlContextSizeOverflow { tensor_count: usize },
    #[error(
        "ggml context initialization failed for {tensor_count} tensor definitions using {mem_size} bytes"
    )]
    GgmlContextInitFailed {
        tensor_count: usize,
        mem_size: usize,
    },
    #[error("gguf context initialization failed")]
    GgufContextInitFailed,
    #[error("ggml tensor definition for '{name}' returned null")]
    GgmlTensorInitFailed { name: String },
    #[error("ggml_set_name returned null for tensor '{name}'")]
    GgmlTensorNameFailed { name: String },
    #[error("gguf write failed for '{path}'")]
    WriteFailed { path: PathBuf },
}

pub(crate) fn write_gguf_file_v0(
    path: impl AsRef<Path>,
    metadata: &BTreeMap<String, GgufWriteValue>,
    tensors: &[GgufWriteTensor],
) -> Result<(), GgufWriteError> {
    let path = path.as_ref();
    if path.exists() {
        return Err(GgufWriteError::OutputExists {
            path: path.to_path_buf(),
        });
    }
    validate_metadata(metadata)?;
    validate_tensors(tensors)?;

    let path_cstring = path_to_cstring(path)?;
    let gguf_context = unsafe { GgufContextGuard::from_raw(ffi::gguf_init_empty()) }
        .ok_or(GgufWriteError::GgufContextInitFailed)?;
    let ggml_context = GgmlContextGuard::init_for_tensor_defs(tensors.len())?;

    for (key, value) in metadata {
        set_metadata_value(gguf_context.as_ptr(), key, value)?;
    }
    for tensor in tensors {
        add_tensor(gguf_context.as_ptr(), ggml_context.as_ptr(), tensor)?;
    }

    let success =
        unsafe { ffi::gguf_write_to_file(gguf_context.as_ptr(), path_cstring.as_ptr(), false) };
    if !success {
        return Err(GgufWriteError::WriteFailed {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

pub(crate) fn quantize_f32_to_ggml_tensor_data(
    tensor_type: GgufWriteTensorType,
    dims: &[u64],
    values: &[f32],
) -> Result<Vec<u8>, GgufWriteError> {
    if !matches!(
        tensor_type,
        GgufWriteTensorType::Q8_0
            | GgufWriteTensorType::Q3_K
            | GgufWriteTensorType::Q4_K
            | GgufWriteTensorType::Q5_K
            | GgufWriteTensorType::Q6_K
    ) {
        return Err(GgufWriteError::TensorQuantizationTypeUnsupported { tensor_type });
    }
    let expected_values = checked_element_count("quantization-source", dims).map_err(|_| {
        GgufWriteError::TensorElementCountOverflow {
            name: "quantization-source".to_string(),
            dims: dims.to_vec(),
        }
    })?;
    let expected_values =
        usize::try_from(expected_values).map_err(|_| GgufWriteError::TensorByteLengthOverflow {
            name: "quantization-source".to_string(),
        })?;
    if values.len() != expected_values {
        return Err(GgufWriteError::TensorQuantizationSourceValueCountMismatch {
            dims: dims.to_vec(),
            expected: expected_values,
            actual: values.len(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(GgufWriteError::TensorQuantizationSourceNonFinite);
    }
    let expected_bytes = expected_tensor_nbytes_for("quantization-target", dims, tensor_type)?;
    let mut bytes = vec![0_u8; expected_bytes];
    let ne0 = *dims.first().unwrap_or(&0);
    let row_count = dims
        .iter()
        .skip(1)
        .try_fold(1_u64, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| GgufWriteError::TensorElementCountOverflow {
            name: "quantization-target".to_string(),
            dims: dims.to_vec(),
        })?;
    let ne0_i64 = i64::try_from(ne0).map_err(|_| GgufWriteError::TensorDimensionOverflow {
        name: "quantization-target".to_string(),
        value: ne0,
    })?;
    let row_count_i64 =
        i64::try_from(row_count).map_err(|_| GgufWriteError::TensorDimensionOverflow {
            name: "quantization-target".to_string(),
            value: row_count,
        })?;
    let produced = unsafe {
        ffi::ggml_quantize_chunk(
            tensor_type.ggml_type(),
            values.as_ptr(),
            bytes.as_mut_ptr().cast::<c_void>(),
            0,
            row_count_i64,
            ne0_i64,
            null(),
        )
    };
    if produced != expected_bytes {
        return Err(GgufWriteError::TensorQuantizationSizeMismatch {
            dims: dims.to_vec(),
            tensor_type,
            expected: expected_bytes,
            actual: produced,
        });
    }
    Ok(bytes)
}

fn validate_metadata(metadata: &BTreeMap<String, GgufWriteValue>) -> Result<(), GgufWriteError> {
    let mut seen = BTreeSet::new();
    for key in metadata.keys() {
        if key.trim().is_empty() {
            return Err(GgufWriteError::EmptyMetadataKey);
        }
        if !seen.insert(key.as_str()) {
            return Err(GgufWriteError::DuplicateMetadataKey { key: key.clone() });
        }
    }
    Ok(())
}

fn validate_tensors(tensors: &[GgufWriteTensor]) -> Result<(), GgufWriteError> {
    let mut seen = BTreeSet::new();
    for tensor in tensors {
        if tensor.name.trim().is_empty() {
            return Err(GgufWriteError::EmptyTensorName);
        }
        if !seen.insert(tensor.name.as_str()) {
            return Err(GgufWriteError::DuplicateTensorName {
                name: tensor.name.clone(),
            });
        }
        validate_tensor_shape_and_data(tensor)?;
    }
    Ok(())
}

fn validate_tensor_shape_and_data(tensor: &GgufWriteTensor) -> Result<(), GgufWriteError> {
    if !(1..=4).contains(&tensor.dims.len()) {
        return Err(GgufWriteError::UnsupportedTensorRank {
            name: tensor.name.clone(),
            rank: tensor.dims.len(),
        });
    }
    for (index, dim) in tensor.dims.iter().enumerate() {
        if *dim == 0 {
            return Err(GgufWriteError::NonPositiveTensorDimension {
                name: tensor.name.clone(),
                index,
            });
        }
        if i64::try_from(*dim).is_err() {
            return Err(GgufWriteError::TensorDimensionOverflow {
                name: tensor.name.clone(),
                value: *dim,
            });
        }
    }

    let expected = expected_tensor_nbytes_for(&tensor.name, &tensor.dims, tensor.tensor_type)?;
    if tensor.data.len() != expected {
        return Err(GgufWriteError::TensorDataLengthMismatch {
            name: tensor.name.clone(),
            dims: tensor.dims.clone(),
            expected,
            actual: tensor.data.len(),
        });
    }
    Ok(())
}

fn checked_element_count(name: &str, dims: &[u64]) -> Result<u64, GgufWriteError> {
    dims.iter().try_fold(1_u64, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| GgufWriteError::TensorElementCountOverflow {
                name: name.to_string(),
                dims: dims.to_vec(),
            })
    })
}

fn expected_tensor_nbytes_for(
    name: &str,
    dims: &[u64],
    tensor_type: GgufWriteTensorType,
) -> Result<usize, GgufWriteError> {
    let ne0 = *dims
        .first()
        .ok_or_else(|| GgufWriteError::UnsupportedTensorRank {
            name: name.to_string(),
            rank: 0,
        })?;
    let ne0_i64 = i64::try_from(ne0).map_err(|_| GgufWriteError::TensorDimensionOverflow {
        name: name.to_string(),
        value: ne0,
    })?;
    let ggml_type = tensor_type.ggml_type();
    let block_size = unsafe { ffi::ggml_blck_size(ggml_type) };
    if block_size <= 0 {
        return Err(GgufWriteError::TensorInvalidBlockSize {
            name: name.to_string(),
            block_size,
        });
    }
    let block_size_u64 =
        u64::try_from(block_size).map_err(|_| GgufWriteError::TensorInvalidBlockSize {
            name: name.to_string(),
            block_size,
        })?;
    if !ne0.is_multiple_of(block_size_u64) {
        return Err(GgufWriteError::TensorBlockAlignmentMismatch {
            name: name.to_string(),
            ne0,
            block_size: block_size_u64,
        });
    }

    let row_size = unsafe { ffi::ggml_row_size(ggml_type, ne0_i64) };
    let rows = dims.iter().skip(1).try_fold(1_u64, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| GgufWriteError::TensorElementCountOverflow {
                name: name.to_string(),
                dims: dims.to_vec(),
            })
    })?;
    let expected_u64 = (row_size as u64).checked_mul(rows).ok_or_else(|| {
        GgufWriteError::TensorByteLengthOverflow {
            name: name.to_string(),
        }
    })?;
    usize::try_from(expected_u64).map_err(|_| GgufWriteError::TensorByteLengthOverflow {
        name: name.to_string(),
    })
}

fn set_metadata_value(
    ctx: ffi::GgufContextRaw,
    key: &str,
    value: &GgufWriteValue,
) -> Result<(), GgufWriteError> {
    let key_cstring = cstring_for_field(key, "metadata.key")?;
    match value {
        GgufWriteValue::String(value) => {
            let value_cstring = cstring_for_field(value, "metadata.string_value")?;
            unsafe {
                ffi::gguf_set_val_str(ctx, key_cstring.as_ptr(), value_cstring.as_ptr());
            }
        }
        GgufWriteValue::U32(value) => unsafe {
            ffi::gguf_set_val_u32(ctx, key_cstring.as_ptr(), *value);
        },
        GgufWriteValue::StringArray(values) => {
            let value_cstrings = values
                .iter()
                .map(|value| cstring_for_field(value, "metadata.string_array_value"))
                .collect::<Result<Vec<_>, _>>()?;
            let value_ptrs = value_cstrings
                .iter()
                .map(|value| value.as_ptr())
                .collect::<Vec<_>>();
            unsafe {
                ffi::gguf_set_arr_str(
                    ctx,
                    key_cstring.as_ptr(),
                    value_ptrs.as_ptr(),
                    value_ptrs.len(),
                );
            }
        }
        GgufWriteValue::U32Array(values) => unsafe {
            ffi::gguf_set_arr_data(
                ctx,
                key_cstring.as_ptr(),
                ffi::GGUF_TYPE_UINT32,
                values.as_ptr().cast(),
                values.len(),
            );
        },
    }
    Ok(())
}

fn add_tensor(
    gguf_ctx: ffi::GgufContextRaw,
    ggml_ctx: ffi::GgmlContextRaw,
    tensor: &GgufWriteTensor,
) -> Result<(), GgufWriteError> {
    let name_cstring = cstring_for_field(&tensor.name, "tensor.name")?;
    let ggml_type = tensor.tensor_type.ggml_type();
    let dims = tensor_dims_i64(tensor)?;
    let raw_tensor = unsafe {
        match dims.as_slice() {
            [ne0] => ffi::ggml_new_tensor_1d(ggml_ctx, ggml_type, *ne0),
            [ne0, ne1] => ffi::ggml_new_tensor_2d(ggml_ctx, ggml_type, *ne0, *ne1),
            [ne0, ne1, ne2] => ffi::ggml_new_tensor_3d(ggml_ctx, ggml_type, *ne0, *ne1, *ne2),
            [ne0, ne1, ne2, ne3] => {
                ffi::ggml_new_tensor_4d(ggml_ctx, ggml_type, *ne0, *ne1, *ne2, *ne3)
            }
            _ => unreachable!("tensor rank was validated before tensor creation"),
        }
    };
    if raw_tensor.is_null() {
        return Err(GgufWriteError::GgmlTensorInitFailed {
            name: tensor.name.clone(),
        });
    }

    let raw_tensor = unsafe { ffi::ggml_set_name(raw_tensor, name_cstring.as_ptr()) };
    if raw_tensor.is_null() {
        return Err(GgufWriteError::GgmlTensorNameFailed {
            name: tensor.name.clone(),
        });
    }

    unsafe {
        ffi::gguf_add_tensor(gguf_ctx, raw_tensor);
        ffi::gguf_set_tensor_type(gguf_ctx, name_cstring.as_ptr(), ggml_type);
        ffi::gguf_set_tensor_data(
            gguf_ctx,
            name_cstring.as_ptr(),
            tensor.data.as_ptr().cast::<c_void>(),
        );
    }
    Ok(())
}

fn tensor_dims_i64(tensor: &GgufWriteTensor) -> Result<Vec<i64>, GgufWriteError> {
    tensor
        .dims
        .iter()
        .map(|dim| {
            i64::try_from(*dim).map_err(|_| GgufWriteError::TensorDimensionOverflow {
                name: tensor.name.clone(),
                value: *dim,
            })
        })
        .collect()
}

fn path_to_cstring(path: &Path) -> Result<CString, GgufWriteError> {
    let rendered = path.as_os_str().to_string_lossy().to_string();
    CString::new(rendered.clone()).map_err(|_| GgufWriteError::PathContainsNul { path: rendered })
}

fn cstring_for_field(value: &str, field: &'static str) -> Result<CString, GgufWriteError> {
    CString::new(value).map_err(|_| GgufWriteError::StringContainsNul { field })
}

struct GgufContextGuard {
    raw: ffi::GgufContextRaw,
}

impl GgufContextGuard {
    unsafe fn from_raw(raw: ffi::GgufContextRaw) -> Option<Self> {
        (!raw.is_null()).then_some(Self { raw })
    }

    fn as_ptr(&self) -> ffi::GgufContextRaw {
        self.raw
    }
}

impl Drop for GgufContextGuard {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                ffi::gguf_free(self.raw);
            }
        }
    }
}

struct GgmlContextGuard {
    raw: ffi::GgmlContextRaw,
}

impl GgmlContextGuard {
    fn init_for_tensor_defs(tensor_count: usize) -> Result<Self, GgufWriteError> {
        let mem_size = tensor_count
            .checked_mul(4096)
            .and_then(|size| size.checked_add(1 << 20))
            .ok_or(GgufWriteError::GgmlContextSizeOverflow { tensor_count })?;
        let raw = unsafe {
            ffi::ggml_init(ffi::GgmlInitParams {
                mem_size,
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            })
        };
        if raw.is_null() {
            return Err(GgufWriteError::GgmlContextInitFailed {
                tensor_count,
                mem_size,
            });
        }
        Ok(Self { raw })
    }

    fn as_ptr(&self) -> ffi::GgmlContextRaw {
        self.raw
    }
}

impl Drop for GgmlContextGuard {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                ffi::ggml_free(self.raw);
            }
        }
    }
}
