use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    ffi::{CString, c_void},
    fs::OpenOptions,
    io::{self, Write},
    path::{Path, PathBuf},
    ptr::{self, null},
};

use thiserror::Error;

use super::ffi;

/// Build provenance recorded on every pack this codebase writes: the open-core
/// git commit whose quantization policy produced the pack's tensors.
///
/// Answering "which code version built this pack" used to require guessing
/// from file mtimes -- which once led to validating new code against stale
/// artifacts and reaching a completely backwards conclusion about a quant
/// tier's quality ceiling. The publish pipeline (`convert.sh`) always exports
/// [`BUILD_COMMIT_ENV`] from `git rev-parse HEAD`; when the variable is set,
/// [`write_gguf_file_v0`] merges this key into the pack's GGUF metadata at the
/// single write choke point, so every family's importer records it identically
/// without per-family wiring. Unset (a plain library build, a test fixture)
/// means the key is simply absent -- provenance is opt-in, but a SET value
/// must be a well-formed 40-hex commit or the write fails closed.
pub const OASR_METADATA_KEY_BUILD_COMMIT: &str = "openasr.build.commit";

/// Environment variable carrying the build commit for
/// [`OASR_METADATA_KEY_BUILD_COMMIT`]. Set by the publishing pipeline.
pub const BUILD_COMMIT_ENV: &str = "OPENASR_BUILD_COMMIT";

fn is_commit_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// The build-provenance metadata to merge into an outgoing pack, read from
/// [`BUILD_COMMIT_ENV`]. `None` when the variable is unset/empty (no
/// provenance claimed); an error when it IS set but malformed -- a builder
/// that claims provenance must claim it correctly.
pub(crate) fn build_provenance_from_env() -> Result<Option<(String, GgufWriteValue)>, GgufWriteError>
{
    match std::env::var(BUILD_COMMIT_ENV) {
        Ok(raw) => {
            let commit = raw.trim().to_ascii_lowercase();
            if commit.is_empty() {
                return Ok(None);
            }
            if !is_commit_sha(&commit) {
                return Err(GgufWriteError::InvalidBuildProvenance {
                    reason: format!(
                        "{BUILD_COMMIT_ENV} must be a 40-hex git commit sha, got {raw:?}"
                    ),
                });
            }
            Ok(Some((
                OASR_METADATA_KEY_BUILD_COMMIT.to_string(),
                GgufWriteValue::String(commit),
            )))
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(GgufWriteError::InvalidBuildProvenance {
            reason: format!("{BUILD_COMMIT_ENV} is not valid unicode"),
        }),
    }
}

// No `Eq`: `F32` carries an IEEE float. Callers compare values with `==`
// (PartialEq) when inheriting metadata from an existing pack; they never hash
// or order them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GgufWriteValue {
    String(String),
    U32(u32),
    U64(u64),
    // The reader parses native f32/bool KV entries (external pack tooling
    // bakes hparams that way), so the write choke point can spell them too.
    // No Rust importer selects them yet -- available until one does (same
    // precedent as the reserved Q5_K/Q6_K tensor types below).
    #[allow(dead_code)]
    F32(f32),
    #[allow(dead_code)]
    Bool(bool),
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
    pub(crate) fn ggml_type(self) -> i32 {
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
    #[error("gguf build provenance is invalid: {reason}")]
    InvalidBuildProvenance { reason: String },
    #[error("gguf output path cannot be represented as C string: {path}")]
    PathContainsNul { path: String },
    #[error("gguf string field '{field}' cannot contain NUL bytes")]
    StringContainsNul { field: &'static str },
    #[error("gguf metadata key '{key}' has a non-finite f32 value")]
    NonFiniteMetadataValue { key: String },
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
    #[error("gguf streaming write failed for '{path}': {source}")]
    StreamingIo {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("gguf tensor '{name}' streaming producer failed: {reason}")]
    TensorStreamingProducer { name: String, reason: String },
    #[error(
        "gguf tensor '{name}' streaming producer wrote {actual} bytes, expected exactly {expected}"
    )]
    TensorStreamingLengthMismatch {
        name: String,
        expected: u64,
        actual: u64,
    },
    #[error("gguf tensor '{name}' raw ggml type {ggml_type} has an invalid row size")]
    TensorRawTypeInvalid { name: String, ggml_type: i32 },
    #[error("gguf streaming writer supports alignment 32, got {alignment}")]
    StreamingAlignmentUnsupported { alignment: u64 },
}

/// Shape/type-only tensor declaration for a bounded-memory GGUF writer.
///
/// Unlike [`GgufWriteTensor`], this does not own the payload. The header is
/// emitted first and each payload is then produced directly into the output
/// file, so a multi-gigabyte requant never holds every transformed tensor in
/// RAM at once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GgufStreamTensorSpec {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: i32,
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
    // Merge build provenance (when the pipeline claims it) at the single write
    // choke point, so every pack this codebase emits can answer "which commit
    // built me" without per-importer wiring.
    let provenance = build_provenance_from_env()?;
    let metadata: Cow<'_, BTreeMap<String, GgufWriteValue>> = match provenance {
        Some((key, value)) => {
            let mut merged = metadata.clone();
            merged.insert(key, value);
            Cow::Owned(merged)
        }
        None => Cow::Borrowed(metadata),
    };
    validate_metadata(&metadata)?;
    validate_tensors(tensors)?;

    let path_cstring = path_to_cstring(path)?;
    let gguf_context = unsafe { GgufContextGuard::from_raw(ffi::gguf_init_empty()) }
        .ok_or(GgufWriteError::GgufContextInitFailed)?;
    let ggml_context = GgmlContextGuard::init_for_tensor_defs(tensors.len())?;

    for (key, value) in metadata.iter() {
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

/// Write a GGUF header followed by one tensor payload at a time.
///
/// `produce` must write exactly the declared byte length for each tensor. The
/// writer enforces that boundary before advancing to the next aligned tensor,
/// making a row-streaming requant both bounded-memory and structurally
/// incapable of shifting later tensor offsets.
pub(crate) fn write_gguf_file_streaming_v0<F>(
    path: impl AsRef<Path>,
    metadata: &BTreeMap<String, GgufWriteValue>,
    tensors: &[GgufStreamTensorSpec],
    mut produce: F,
) -> Result<(), GgufWriteError>
where
    F: FnMut(usize, &GgufStreamTensorSpec, &mut dyn Write) -> Result<(), GgufWriteError>,
{
    let path = path.as_ref();
    if path.exists() {
        return Err(GgufWriteError::OutputExists {
            path: path.to_path_buf(),
        });
    }
    let provenance = build_provenance_from_env()?;
    let metadata: Cow<'_, BTreeMap<String, GgufWriteValue>> = match provenance {
        Some((key, value)) => {
            let mut merged = metadata.clone();
            merged.insert(key, value);
            Cow::Owned(merged)
        }
        None => Cow::Borrowed(metadata),
    };
    validate_metadata(&metadata)?;
    validate_stream_tensor_specs(tensors)?;

    let path_cstring = path_to_cstring(path)?;
    let gguf_context = unsafe { GgufContextGuard::from_raw(ffi::gguf_init_empty()) }
        .ok_or(GgufWriteError::GgufContextInitFailed)?;
    let ggml_context = GgmlContextGuard::init_for_tensor_defs(tensors.len())?;
    for (key, value) in metadata.iter() {
        set_metadata_value(gguf_context.as_ptr(), key, value)?;
    }
    for tensor in tensors {
        add_stream_tensor_spec(gguf_context.as_ptr(), ggml_context.as_ptr(), tensor)?;
    }
    let success =
        unsafe { ffi::gguf_write_to_file(gguf_context.as_ptr(), path_cstring.as_ptr(), true) };
    if !success {
        return Err(GgufWriteError::WriteFailed {
            path: path.to_path_buf(),
        });
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(|value| match value {
            GgufWriteValue::U32(value) => Some(u64::from(*value)),
            _ => None,
        })
        .unwrap_or(32);
    // gguf's public construction API validates `general.alignment` but does
    // not update a fresh context's internal offset alignment. Until gguf
    // exposes a setter, accepting a different value would write a header
    // whose declared offsets disagree with its metadata. Fail closed rather
    // than produce a corrupt container.
    if alignment != 32 {
        return Err(GgufWriteError::StreamingAlignmentUnsupported { alignment });
    }
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(|source| GgufWriteError::StreamingIo {
            path: path.to_path_buf(),
            source,
        })?;
    let zeroes = [0_u8; 256];
    for (index, tensor) in tensors.iter().enumerate() {
        let expected = stream_tensor_nbytes(tensor)?;
        let mut exact = ExactTensorWriter {
            tensor_name: &tensor.name,
            inner: &mut file,
            expected,
            written: 0,
        };
        produce(index, tensor, &mut exact)?;
        if exact.written != expected {
            return Err(GgufWriteError::TensorStreamingLengthMismatch {
                name: tensor.name.clone(),
                expected,
                actual: exact.written,
            });
        }
        let padding = (alignment - expected % alignment) % alignment;
        let mut remaining = padding;
        while remaining > 0 {
            let count = usize::try_from(remaining.min(zeroes.len() as u64)).unwrap();
            file.write_all(&zeroes[..count])
                .map_err(|source| GgufWriteError::StreamingIo {
                    path: path.to_path_buf(),
                    source,
                })?;
            remaining -= count as u64;
        }
    }
    file.flush().map_err(|source| GgufWriteError::StreamingIo {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

struct ExactTensorWriter<'a> {
    tensor_name: &'a str,
    inner: &'a mut std::fs::File,
    expected: u64,
    written: u64,
}

impl Write for ExactTensorWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let requested = u64::try_from(bytes.len())
            .map_err(|_| io::Error::other("tensor write length does not fit u64"))?;
        let next = self
            .written
            .checked_add(requested)
            .ok_or_else(|| io::Error::other("tensor write length overflow"))?;
        if next > self.expected {
            return Err(io::Error::other(format!(
                "tensor '{}' would exceed its declared {} bytes while writing {}",
                self.tensor_name, self.expected, next
            )));
        }
        let count = self.inner.write(bytes)?;
        self.written = self
            .written
            .checked_add(count as u64)
            .ok_or_else(|| io::Error::other("tensor write count overflow"))?;
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn validate_stream_tensor_specs(tensors: &[GgufStreamTensorSpec]) -> Result<(), GgufWriteError> {
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
        if !(1..=4).contains(&tensor.dims.len()) {
            return Err(GgufWriteError::UnsupportedTensorRank {
                name: tensor.name.clone(),
                rank: tensor.dims.len(),
            });
        }
        for (index, value) in tensor.dims.iter().copied().enumerate() {
            if value == 0 {
                return Err(GgufWriteError::NonPositiveTensorDimension {
                    name: tensor.name.clone(),
                    index,
                });
            }
            i64::try_from(value).map_err(|_| GgufWriteError::TensorDimensionOverflow {
                name: tensor.name.clone(),
                value,
            })?;
        }
        stream_tensor_nbytes(tensor)?;
    }
    Ok(())
}

fn stream_tensor_nbytes(tensor: &GgufStreamTensorSpec) -> Result<u64, GgufWriteError> {
    let ne0 =
        i64::try_from(tensor.dims[0]).map_err(|_| GgufWriteError::TensorDimensionOverflow {
            name: tensor.name.clone(),
            value: tensor.dims[0],
        })?;
    let row_size = unsafe { ffi::ggml_row_size(tensor.ggml_type, ne0) };
    if row_size == 0 {
        return Err(GgufWriteError::TensorRawTypeInvalid {
            name: tensor.name.clone(),
            ggml_type: tensor.ggml_type,
        });
    }
    let rows = tensor
        .dims
        .iter()
        .skip(1)
        .try_fold(1_u64, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| GgufWriteError::TensorElementCountOverflow {
            name: tensor.name.clone(),
            dims: tensor.dims.clone(),
        })?;
    u64::try_from(row_size)
        .ok()
        .and_then(|bytes| bytes.checked_mul(rows))
        .ok_or_else(|| GgufWriteError::TensorByteLengthOverflow {
            name: tensor.name.clone(),
        })
}

pub(crate) fn quantize_f32_to_ggml_tensor_data(
    tensor_type: GgufWriteTensorType,
    dims: &[u64],
    values: &[f32],
) -> Result<Vec<u8>, GgufWriteError> {
    let mut bytes = Vec::new();
    quantize_f32_to_ggml_tensor_data_into(tensor_type, dims, values, &mut bytes)?;
    Ok(bytes)
}

/// Quantize into a caller-owned buffer so row-streaming transforms can reuse
/// one allocation for every row of a large tensor.
pub(crate) fn quantize_f32_to_ggml_tensor_data_into(
    tensor_type: GgufWriteTensorType,
    dims: &[u64],
    values: &[f32],
    bytes: &mut Vec<u8>,
) -> Result<(), GgufWriteError> {
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
    bytes.clear();
    bytes.resize(expected_bytes, 0_u8);
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
    Ok(())
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
        GgufWriteValue::U64(value) => unsafe {
            ffi::gguf_set_val_u64(ctx, key_cstring.as_ptr(), *value);
        },
        GgufWriteValue::F32(value) => {
            if !value.is_finite() {
                return Err(GgufWriteError::NonFiniteMetadataValue {
                    key: key.to_string(),
                });
            }
            unsafe {
                ffi::gguf_set_val_f32(ctx, key_cstring.as_ptr(), *value);
            }
        }
        GgufWriteValue::Bool(value) => unsafe {
            ffi::gguf_set_val_bool(ctx, key_cstring.as_ptr(), *value);
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

fn add_stream_tensor_spec(
    gguf_ctx: ffi::GgufContextRaw,
    ggml_ctx: ffi::GgmlContextRaw,
    tensor: &GgufStreamTensorSpec,
) -> Result<(), GgufWriteError> {
    let name_cstring = cstring_for_field(&tensor.name, "tensor.name")?;
    let dims = tensor
        .dims
        .iter()
        .map(|value| {
            i64::try_from(*value).map_err(|_| GgufWriteError::TensorDimensionOverflow {
                name: tensor.name.clone(),
                value: *value,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let raw_tensor = unsafe {
        match dims.as_slice() {
            [ne0] => ffi::ggml_new_tensor_1d(ggml_ctx, tensor.ggml_type, *ne0),
            [ne0, ne1] => ffi::ggml_new_tensor_2d(ggml_ctx, tensor.ggml_type, *ne0, *ne1),
            [ne0, ne1, ne2] => {
                ffi::ggml_new_tensor_3d(ggml_ctx, tensor.ggml_type, *ne0, *ne1, *ne2)
            }
            [ne0, ne1, ne2, ne3] => {
                ffi::ggml_new_tensor_4d(ggml_ctx, tensor.ggml_type, *ne0, *ne1, *ne2, *ne3)
            }
            _ => unreachable!("stream tensor rank was validated"),
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
        ffi::gguf_set_tensor_type(gguf_ctx, name_cstring.as_ptr(), tensor.ggml_type);
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

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::{
        BUILD_COMMIT_ENV, GgufStreamTensorSpec, GgufWriteError, GgufWriteTensor,
        GgufWriteTensorType, GgufWriteValue, OASR_METADATA_KEY_BUILD_COMMIT,
        quantize_f32_to_ggml_tensor_data, quantize_f32_to_ggml_tensor_data_into,
        write_gguf_file_streaming_v0, write_gguf_file_v0,
    };
    use crate::ggml_runtime::{GgufTensorDataReader, read_gguf_metadata};
    use crate::test_process_env::with_test_process_env;

    fn fixture_pack(dir: &std::path::Path) -> (std::path::PathBuf, Vec<GgufWriteTensor>) {
        let path = dir.join("provenance.oasr");
        let tensors = vec![GgufWriteTensor {
            name: "probe.weight".to_string(),
            dims: vec![1],
            tensor_type: GgufWriteTensorType::F32,
            data: 1.0_f32.to_le_bytes().to_vec(),
        }];
        (path, tensors)
    }

    const TEST_COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    #[test]
    fn build_commit_env_is_merged_into_pack_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let (path, tensors) = fixture_pack(dir.path());
        let metadata = std::collections::BTreeMap::new();

        with_test_process_env(
            [(BUILD_COMMIT_ENV, Some(OsString::from(TEST_COMMIT)))],
            || {
                write_gguf_file_v0(&path, &metadata, &tensors).expect("write pack");
            },
        );

        let read = read_gguf_metadata(&path).expect("read pack metadata");
        assert_eq!(
            read.get_string(OASR_METADATA_KEY_BUILD_COMMIT),
            Some(TEST_COMMIT)
        );
    }

    #[test]
    fn absent_build_commit_env_writes_no_provenance_key() {
        let dir = tempfile::tempdir().unwrap();
        let (path, tensors) = fixture_pack(dir.path());
        let metadata = std::collections::BTreeMap::new();

        with_test_process_env([(BUILD_COMMIT_ENV, None)], || {
            write_gguf_file_v0(&path, &metadata, &tensors).expect("write pack");
        });

        let read = read_gguf_metadata(&path).expect("read pack metadata");
        assert_eq!(read.get_string(OASR_METADATA_KEY_BUILD_COMMIT), None);
    }

    #[test]
    fn malformed_build_commit_env_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let (path, tensors) = fixture_pack(dir.path());
        let metadata = std::collections::BTreeMap::new();

        for invalid in ["not-a-sha", "0123456789abcdef", &"a".repeat(41)] {
            let path = path.with_file_name(format!("provenance-{}.oasr", invalid.len()));
            let invalid = invalid.to_string();
            let error =
                with_test_process_env([(BUILD_COMMIT_ENV, Some(OsString::from(&invalid)))], || {
                    write_gguf_file_v0(&path, &metadata, &tensors).expect_err("must fail closed")
                });
            assert!(
                matches!(error, GgufWriteError::InvalidBuildProvenance { .. }),
                "unexpected error for {invalid:?}: {error}"
            );
        }
    }

    #[test]
    fn uppercase_build_commit_is_normalized_to_lowercase() {
        let dir = tempfile::tempdir().unwrap();
        let (path, tensors) = fixture_pack(dir.path());
        let metadata = std::collections::BTreeMap::new();
        let upper = TEST_COMMIT.to_ascii_uppercase();

        with_test_process_env([(BUILD_COMMIT_ENV, Some(OsString::from(&upper)))], || {
            write_gguf_file_v0(&path, &metadata, &tensors).expect("write pack");
        });

        let read = read_gguf_metadata(&path).expect("read pack metadata");
        assert_eq!(
            read.get_string(OASR_METADATA_KEY_BUILD_COMMIT),
            Some(TEST_COMMIT)
        );
    }

    #[test]
    fn build_commit_never_clobbers_caller_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let (path, tensors) = fixture_pack(dir.path());
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            "openasr.package.version".to_string(),
            GgufWriteValue::String("1".to_string()),
        );

        with_test_process_env(
            [(BUILD_COMMIT_ENV, Some(OsString::from(TEST_COMMIT)))],
            || {
                write_gguf_file_v0(&path, &metadata, &tensors).expect("write pack");
            },
        );

        let read = read_gguf_metadata(&path).expect("read pack metadata");
        assert_eq!(read.get_string("openasr.package.version"), Some("1"));
        assert_eq!(
            read.get_string(OASR_METADATA_KEY_BUILD_COMMIT),
            Some(TEST_COMMIT)
        );
        assert_eq!(metadata.len(), 1, "caller metadata map must not be mutated");
    }

    /// The reader parses native GGUF f32/bool KV types (external pack tooling
    /// bakes hparams that way), so the single write choke point must be able
    /// to emit them too -- a fixture/pack writer that can only spell strings
    /// and u32 cannot reproduce a real pack's metadata faithfully.
    #[test]
    fn f32_and_bool_metadata_round_trip_as_native_types() {
        let dir = tempfile::tempdir().unwrap();
        let (path, tensors) = fixture_pack(dir.path());
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            "family.rope.freq_base".to_string(),
            GgufWriteValue::F32(640000.0),
        );
        metadata.insert(
            "family.attention.qkv_bias".to_string(),
            GgufWriteValue::Bool(true),
        );
        metadata.insert(
            "family.attention.qk_norm".to_string(),
            GgufWriteValue::Bool(false),
        );

        with_test_process_env([(BUILD_COMMIT_ENV, None)], || {
            write_gguf_file_v0(&path, &metadata, &tensors).expect("write pack");
        });

        let read = read_gguf_metadata(&path).expect("read pack metadata");
        assert_eq!(read.get_f32("family.rope.freq_base"), Some(640000.0));
        assert_eq!(read.get_bool("family.attention.qkv_bias"), Some(true));
        assert_eq!(read.get_bool("family.attention.qk_norm"), Some(false));
        // Native typing must not leak into the scalar-string view.
        assert_eq!(read.get_string("family.rope.freq_base"), None);
    }

    #[test]
    fn u64_metadata_round_trips_as_a_native_type() {
        let dir = tempfile::tempdir().unwrap();
        let (path, tensors) = fixture_pack(dir.path());
        let metadata = std::collections::BTreeMap::from([(
            "family.large_count".to_string(),
            GgufWriteValue::U64(u64::from(u32::MAX) + 1),
        )]);

        with_test_process_env([(BUILD_COMMIT_ENV, None)], || {
            write_gguf_file_v0(&path, &metadata, &tensors).expect("write pack");
        });

        let read = read_gguf_metadata(&path).expect("read pack metadata");
        assert_eq!(
            read.get_u64("family.large_count"),
            Some(u64::from(u32::MAX) + 1)
        );
    }

    #[test]
    fn streaming_writer_emits_exact_aligned_tensor_payloads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("streamed.gguf");
        let metadata = std::collections::BTreeMap::from([(
            "general.alignment".to_string(),
            GgufWriteValue::U32(32),
        )]);
        let specs = [
            GgufStreamTensorSpec {
                name: "first.weight".to_string(),
                dims: vec![4, 2],
                ggml_type: GgufWriteTensorType::F32.ggml_type(),
            },
            GgufStreamTensorSpec {
                name: "second.weight".to_string(),
                dims: vec![3],
                ggml_type: GgufWriteTensorType::F32.ggml_type(),
            },
        ];
        let values = [
            (0..8).map(|value| value as f32).collect::<Vec<_>>(),
            vec![10.0, 11.0, 12.0],
        ];

        with_test_process_env([(BUILD_COMMIT_ENV, None)], || {
            write_gguf_file_streaming_v0(&path, &metadata, &specs, |index, _, sink| {
                for value in &values[index] {
                    sink.write_all(&value.to_le_bytes()).map_err(|error| {
                        GgufWriteError::TensorStreamingProducer {
                            name: specs[index].name.clone(),
                            reason: error.to_string(),
                        }
                    })?;
                }
                Ok(())
            })
            .expect("stream pack");
        });

        let reader = GgufTensorDataReader::from_path(&path).expect("read streamed pack");
        assert_eq!(
            reader
                .host_tensor_f32_copy_by_name("first.weight", &[4, 2])
                .unwrap(),
            values[0]
        );
        assert_eq!(
            reader
                .host_tensor_f32_copy_by_name("second.weight", &[3])
                .unwrap(),
            values[1]
        );
        let first = reader.tensor_index().get("first.weight").unwrap();
        let second = reader.tensor_index().get("second.weight").unwrap();
        assert_eq!(second.offset_bytes - first.offset_bytes, 32);
    }

    #[test]
    fn reusable_quant_buffer_is_byte_identical_to_owned_wrapper() {
        let values = (0..256)
            .map(|index| (index as f32 - 127.5) / 64.0)
            .collect::<Vec<_>>();
        let owned = quantize_f32_to_ggml_tensor_data(GgufWriteTensorType::Q4_K, &[256], &values)
            .expect("owned q4_k quantization");
        let mut reused = vec![0xaa; 4096];
        quantize_f32_to_ggml_tensor_data_into(
            GgufWriteTensorType::Q4_K,
            &[256],
            &values,
            &mut reused,
        )
        .expect("reused q4_k quantization");
        assert_eq!(reused, owned);
    }

    #[test]
    fn non_finite_f32_metadata_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let (path, tensors) = fixture_pack(dir.path());
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert("family.bad".to_string(), GgufWriteValue::F32(f32::NAN));

        let error = with_test_process_env([(BUILD_COMMIT_ENV, None)], || {
            write_gguf_file_v0(&path, &metadata, &tensors).expect_err("must fail closed")
        });
        assert!(
            matches!(error, GgufWriteError::NonFiniteMetadataValue { ref key } if key == "family.bad"),
            "unexpected error: {error}"
        );
        assert!(!path.exists(), "a rejected write must not expose an output");
    }
}
