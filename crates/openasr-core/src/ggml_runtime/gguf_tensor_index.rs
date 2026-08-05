use std::{
    collections::BTreeMap,
    ffi::CStr,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::gguf_metadata::{GgufBoundedParseFailure, GgufContextGuard, parse_bounded_gguf_context};
use super::{
    GgmlRuntimeSource, GgmlRuntimeSourcePathError, ffi, validate_ggml_runtime_source_path,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgufTensorMetadata {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: i32,
    pub type_name: String,
    pub size_bytes: u64,
    pub offset_bytes: u64,
}

impl GgufTensorMetadata {
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn num_elements(&self) -> Option<u64> {
        self.dims
            .iter()
            .try_fold(1_u64, |acc, &dim| acc.checked_mul(dim))
    }

    pub fn has_shape(&self, shape: &[u64]) -> bool {
        self.dims == shape
    }

    pub fn has_same_shape(&self, other: &Self) -> bool {
        self.dims == other.dims
    }
}

/// One successful by-name tensor lookup recorded by an index's opt-in access
/// trace: the pack tensor name plus the stored dims the index served.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufTensorAccessRecord {
    pub name: String,
    pub dims: Vec<u64>,
}

/// Shared opt-in recorder behind [`GgufTensorIndex::enable_access_trace`].
/// Cloned indexes share one recorder, so a reader built from an index and any
/// clone of it trace into the same record list.
#[derive(Default)]
struct GgufTensorAccessTrace {
    enabled: AtomicBool,
    records: Mutex<Vec<GgufTensorAccessRecord>>,
}

impl GgufTensorAccessTrace {
    fn record(&self, tensor: &GgufTensorMetadata) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        self.records
            .lock()
            .expect("tensor access trace mutex poisoned")
            .push(GgufTensorAccessRecord {
                name: tensor.name.clone(),
                dims: tensor.dims.clone(),
            });
    }

    fn snapshot(&self) -> Vec<GgufTensorAccessRecord> {
        self.records
            .lock()
            .expect("tensor access trace mutex poisoned")
            .clone()
    }
}

#[derive(Clone)]
pub struct GgufTensorIndex {
    path: PathBuf,
    data_section_offset_bytes: u64,
    tensors: Vec<GgufTensorMetadata>,
    tensor_index_by_name: BTreeMap<String, usize>,
    /// Never part of equality or debug identity: two indexes over the same
    /// pack are the same index no matter what has been traced.
    access_trace: Arc<GgufTensorAccessTrace>,
}

impl PartialEq for GgufTensorIndex {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.data_section_offset_bytes == other.data_section_offset_bytes
            && self.tensors == other.tensors
            && self.tensor_index_by_name == other.tensor_index_by_name
    }
}

impl Eq for GgufTensorIndex {}

impl std::fmt::Debug for GgufTensorIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufTensorIndex")
            .field("path", &self.path)
            .field("data_section_offset_bytes", &self.data_section_offset_bytes)
            .field("tensors", &self.tensors)
            .field("tensor_index_by_name", &self.tensor_index_by_name)
            .finish()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GgufTensorIndexSnapshot {
    pub(crate) path: PathBuf,
    pub(crate) data_section_offset_bytes: u64,
    pub(crate) tensors: Vec<GgufTensorMetadata>,
}

impl GgufTensorIndex {
    pub(crate) fn set_display_path(&mut self, path: PathBuf) {
        self.path = path;
    }

    #[cfg(test)]
    pub(crate) fn empty_for_test(path: PathBuf) -> Self {
        Self {
            path,
            data_section_offset_bytes: 0,
            tensors: Vec::new(),
            tensor_index_by_name: BTreeMap::new(),
            access_trace: Arc::new(GgufTensorAccessTrace::default()),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn data_section_offset_bytes(&self) -> u64 {
        self.data_section_offset_bytes
    }

    pub fn tensors(&self) -> &[GgufTensorMetadata] {
        &self.tensors
    }

    pub fn get(&self, name: &str) -> Option<&GgufTensorMetadata> {
        let tensor = self
            .tensor_index_by_name
            .get(name)
            .and_then(|index| self.tensors.get(*index))?;
        self.access_trace.record(tensor);
        Some(tensor)
    }

    /// Start recording every successful by-name lookup ([`Self::get`]) this
    /// index serves, for equivalence tests that must prove a weight loader's
    /// read set matches the family's declared runtime tensor contract name
    /// for name and shape for shape. Recording is a testing affordance; it is
    /// off until enabled and costs one relaxed load per lookup while off.
    pub fn enable_access_trace(&self) {
        self.access_trace.enabled.store(true, Ordering::Relaxed);
    }

    /// Snapshot of the lookups recorded since [`Self::enable_access_trace`]
    /// (empty when tracing was never enabled). Entries keep lookup order; a
    /// tensor looked up twice is recorded twice.
    pub fn access_trace(&self) -> Vec<GgufTensorAccessRecord> {
        self.access_trace.snapshot()
    }

    pub(crate) fn to_snapshot(&self) -> GgufTensorIndexSnapshot {
        GgufTensorIndexSnapshot {
            path: self.path.clone(),
            data_section_offset_bytes: self.data_section_offset_bytes,
            tensors: self.tensors.clone(),
        }
    }

    pub(crate) fn from_snapshot(
        snapshot: GgufTensorIndexSnapshot,
    ) -> Result<Self, GgufTensorIndexReadError> {
        let mut tensor_index_by_name = BTreeMap::new();
        for (index, tensor) in snapshot.tensors.iter().enumerate() {
            if tensor_index_by_name
                .insert(tensor.name.clone(), index)
                .is_some()
            {
                return Err(GgufTensorIndexReadError::DuplicateTensorName {
                    path: snapshot.path.clone(),
                    name: tensor.name.clone(),
                });
            }
        }

        Ok(Self {
            path: snapshot.path,
            data_section_offset_bytes: snapshot.data_section_offset_bytes,
            tensors: snapshot.tensors,
            tensor_index_by_name,
            access_trace: Arc::new(GgufTensorAccessTrace::default()),
        })
    }
}

#[derive(Debug, Error)]
pub enum GgufTensorIndexReadError {
    #[error(transparent)]
    InvalidRuntimeSource(#[from] GgmlRuntimeSourcePathError),
    /// Retained for source compatibility. File size now comes from the held
    /// mapping, so buffer-backed parsing does not emit this variant.
    #[error("could not read gguf runtime source metadata for '{path}': {source}")]
    SourceMetadata {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Retained for source compatibility. Buffer-backed parsing no longer
    /// constructs a C path and therefore does not emit this variant.
    #[error("gguf tensor index path cannot be represented as C string: {path}")]
    PathContainsNul { path: String },
    #[error("gguf tensor index initialization failed for '{path}'")]
    InitFailed { path: PathBuf },
    #[error("gguf tensor index allocation failed for '{path}'")]
    AllocationFailed { path: PathBuf },
    #[error("gguf tensor count is negative for '{path}': {count}")]
    NegativeTensorCount { path: PathBuf, count: i64 },
    #[error("gguf tensor count does not fit usize for '{path}': count={count}")]
    TensorCountOverflow { path: PathBuf, count: i64 },
    #[error("gguf data section offset does not fit in u64 for '{path}': {field}={value} (usize)")]
    PlatformSizeOverflow {
        path: PathBuf,
        field: &'static str,
        value: usize,
    },
    #[error(
        "gguf data section offset exceeds file size for '{path}': offset={offset}, file_size={file_size}"
    )]
    DataSectionOutOfBounds {
        path: PathBuf,
        offset: u64,
        file_size: u64,
    },
    #[error("gguf tensor name at index {index} in '{path}' is null")]
    NullTensorName { path: PathBuf, index: i64 },
    #[error("gguf tensor name at index {index} in '{path}' is not valid utf-8: {source}")]
    InvalidTensorNameUtf8 {
        path: PathBuf,
        index: i64,
        source: std::str::Utf8Error,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' has a null dimension pointer: tensor_index={tensor_index}"
    )]
    NullTensorDimensions {
        path: PathBuf,
        tensor_name: String,
        tensor_index: i64,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' has negative dim: dim_index={dim_index}, value={value}"
    )]
    NegativeTensorDimension {
        path: PathBuf,
        tensor_name: String,
        dim_index: i32,
        value: i64,
    },
    #[error(
        "gguf tensor type name for tensor '{tensor_name}' in '{path}' is null (type={ggml_type})"
    )]
    NullTensorTypeName {
        path: PathBuf,
        tensor_name: String,
        ggml_type: i32,
    },
    #[error(
        "gguf tensor type name for tensor '{tensor_name}' in '{path}' is not valid utf-8: {source}"
    )]
    InvalidTensorTypeNameUtf8 {
        path: PathBuf,
        tensor_name: String,
        source: std::str::Utf8Error,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' has offset overflow: data_section_offset={data_section_offset}, tensor_offset={tensor_offset}"
    )]
    TensorOffsetOverflow {
        path: PathBuf,
        tensor_name: String,
        data_section_offset: u64,
        tensor_offset: u64,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' has invalid file range: offset={offset}, size={size_bytes}"
    )]
    TensorRangeOverflow {
        path: PathBuf,
        tensor_name: String,
        offset: u64,
        size_bytes: u64,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' exceeds file bounds: offset={offset}, size={size_bytes}, file_size={file_size}"
    )]
    TensorDataOutOfBounds {
        path: PathBuf,
        tensor_name: String,
        offset: u64,
        size_bytes: u64,
        file_size: u64,
    },
    #[error("gguf tensor index contains duplicate tensor name '{name}' in '{path}'")]
    DuplicateTensorName { path: PathBuf, name: String },
}

pub fn read_gguf_tensor_index(
    path: impl AsRef<Path>,
) -> Result<GgufTensorIndex, GgufTensorIndexReadError> {
    let runtime_source = validate_ggml_runtime_source_path(path)?;
    read_gguf_tensor_index_from_runtime_source(&runtime_source)
}

pub fn read_gguf_tensor_index_from_runtime_source(
    runtime_source: &GgmlRuntimeSource,
) -> Result<GgufTensorIndex, GgufTensorIndexReadError> {
    read_gguf_tensor_index_from_runtime_source_internal(
        runtime_source,
        super::runtime_gguf_parse_limits(),
    )
}

pub(crate) fn read_gguf_tensor_index_from_runtime_source_with_limits(
    runtime_source: &GgmlRuntimeSource,
    max_tensors: u64,
    max_kv: u64,
) -> Result<GgufTensorIndex, GgufTensorIndexReadError> {
    read_gguf_tensor_index_from_runtime_source_internal(
        runtime_source,
        ffi::GgufParseLimits {
            max_tensors,
            max_kv,
            ..super::runtime_gguf_parse_limits()
        },
    )
}

fn read_gguf_tensor_index_from_runtime_source_internal(
    runtime_source: &GgmlRuntimeSource,
    limits: ffi::GgufParseLimits,
) -> Result<GgufTensorIndex, GgufTensorIndexReadError> {
    let path = runtime_source.path();
    let context = parse_bounded_gguf_context(runtime_source, limits).map_err(|failure| {
        if failure == GgufBoundedParseFailure::Allocation {
            crate::models::native_execution_services::record_current_execution_candidate_failure(
                crate::device::execution_policy::ExecutionCandidateFailure::capacity(
                    "gguf-bounded-tensor-index-parse",
                    format!("allocation failed while parsing {}", path.display()),
                ),
            );
            GgufTensorIndexReadError::AllocationFailed {
                path: path.to_path_buf(),
            }
        } else {
            GgufTensorIndexReadError::InitFailed {
                path: path.to_path_buf(),
            }
        }
    })?;

    read_gguf_tensor_index_from_context(runtime_source, &context)
}

pub(crate) fn read_gguf_tensor_index_from_context(
    runtime_source: &GgmlRuntimeSource,
    context: &GgufContextGuard,
) -> Result<GgufTensorIndex, GgufTensorIndexReadError> {
    let path = runtime_source.path();
    let mmap = runtime_source.backing_mmap();
    let file_size = usize_to_u64(path, "file_size", mmap.len())?;

    let tensor_count = unsafe { ffi::gguf_get_n_tensors(context.as_ptr()) };
    if tensor_count < 0 {
        return Err(GgufTensorIndexReadError::NegativeTensorCount {
            path: path.to_path_buf(),
            count: tensor_count,
        });
    }
    let tensor_count_usize = usize::try_from(tensor_count).map_err(|_| {
        GgufTensorIndexReadError::TensorCountOverflow {
            path: path.to_path_buf(),
            count: tensor_count,
        }
    })?;

    let data_section_offset_bytes = usize_to_u64(path, "data_section_offset", unsafe {
        ffi::gguf_get_data_offset(context.as_ptr())
    })?;
    if data_section_offset_bytes > file_size {
        return Err(GgufTensorIndexReadError::DataSectionOutOfBounds {
            path: path.to_path_buf(),
            offset: data_section_offset_bytes,
            file_size,
        });
    }

    let mut tensors = Vec::with_capacity(tensor_count_usize);
    let mut tensor_index_by_name = BTreeMap::new();

    for tensor_index in 0..tensor_count {
        let name_ptr = unsafe { ffi::gguf_get_tensor_name(context.as_ptr(), tensor_index) };
        if name_ptr.is_null() {
            return Err(GgufTensorIndexReadError::NullTensorName {
                path: path.to_path_buf(),
                index: tensor_index,
            });
        }

        let name = unsafe { CStr::from_ptr(name_ptr) }
            .to_str()
            .map_err(|source| GgufTensorIndexReadError::InvalidTensorNameUtf8 {
                path: path.to_path_buf(),
                index: tensor_index,
                source,
            })?;
        let name = name.to_string();

        let dims_ptr = unsafe { ffi::gguf_get_tensor_ne(context.as_ptr(), tensor_index) };
        if dims_ptr.is_null() {
            return Err(GgufTensorIndexReadError::NullTensorDimensions {
                path: path.to_path_buf(),
                tensor_name: name.clone(),
                tensor_index,
            });
        }
        // Upstream returns a GGML_MAX_DIMS shape array and pads dimensions above
        // the stored rank with one. Trim that padding so OpenASR keeps the
        // canonical shape representation used by package validation.
        let raw_dims = unsafe { std::slice::from_raw_parts(dims_ptr, ffi::GGML_MAX_DIMS) };
        let rank = raw_dims
            .iter()
            .rposition(|&value| value != 1)
            .map_or(1, |last_non_unit| last_non_unit + 1);
        let mut dims = Vec::with_capacity(rank);
        for (dim_index, &dim_value) in raw_dims[..rank].iter().enumerate() {
            if dim_value < 0 {
                return Err(GgufTensorIndexReadError::NegativeTensorDimension {
                    path: path.to_path_buf(),
                    tensor_name: name.clone(),
                    dim_index: dim_index as i32,
                    value: dim_value,
                });
            }
            dims.push(dim_value as u64);
        }

        let ggml_type = unsafe { ffi::gguf_get_tensor_type(context.as_ptr(), tensor_index) };
        let type_name_ptr = unsafe { ffi::ggml_type_name(ggml_type) };
        if type_name_ptr.is_null() {
            return Err(GgufTensorIndexReadError::NullTensorTypeName {
                path: path.to_path_buf(),
                tensor_name: name,
                ggml_type,
            });
        }
        let type_name = unsafe { CStr::from_ptr(type_name_ptr) }
            .to_str()
            .map_err(
                |source| GgufTensorIndexReadError::InvalidTensorTypeNameUtf8 {
                    path: path.to_path_buf(),
                    tensor_name: name.clone(),
                    source,
                },
            )?
            .to_string();

        let size_bytes = usize_to_u64(path, "tensor_size", unsafe {
            ffi::gguf_get_tensor_size(context.as_ptr(), tensor_index)
        })?;
        let relative_offset = usize_to_u64(path, "tensor_offset", unsafe {
            ffi::gguf_get_tensor_offset(context.as_ptr(), tensor_index)
        })?;
        let offset_bytes = data_section_offset_bytes
            .checked_add(relative_offset)
            .ok_or_else(|| GgufTensorIndexReadError::TensorOffsetOverflow {
                path: path.to_path_buf(),
                tensor_name: name.clone(),
                data_section_offset: data_section_offset_bytes,
                tensor_offset: relative_offset,
            })?;
        let tensor_end = offset_bytes.checked_add(size_bytes).ok_or_else(|| {
            GgufTensorIndexReadError::TensorRangeOverflow {
                path: path.to_path_buf(),
                tensor_name: name.clone(),
                offset: offset_bytes,
                size_bytes,
            }
        })?;
        if tensor_end > file_size {
            return Err(GgufTensorIndexReadError::TensorDataOutOfBounds {
                path: path.to_path_buf(),
                tensor_name: name.clone(),
                offset: offset_bytes,
                size_bytes,
                file_size,
            });
        }

        let metadata = GgufTensorMetadata {
            name: name.clone(),
            dims,
            ggml_type,
            type_name,
            size_bytes,
            offset_bytes,
        };
        if tensor_index_by_name
            .insert(name.clone(), tensors.len())
            .is_some()
        {
            return Err(GgufTensorIndexReadError::DuplicateTensorName {
                path: path.to_path_buf(),
                name,
            });
        }
        tensors.push(metadata);
    }

    Ok(GgufTensorIndex {
        path: path.to_path_buf(),
        data_section_offset_bytes,
        tensors,
        tensor_index_by_name,
        access_trace: Arc::new(GgufTensorAccessTrace::default()),
    })
}

fn usize_to_u64(
    path: &Path,
    field: &'static str,
    value: usize,
) -> Result<u64, GgufTensorIndexReadError> {
    u64::try_from(value).map_err(|_| GgufTensorIndexReadError::PlatformSizeOverflow {
        path: path.to_path_buf(),
        field,
        value,
    })
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use tempfile::NamedTempFile;

    use super::{
        GgufTensorIndexReadError, GgufTensorMetadata, read_gguf_tensor_index,
        read_gguf_tensor_index_from_runtime_source,
    };
    use crate::validate_ggml_runtime_source_path;

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_i32(bytes: &mut Vec<u8>, value: i32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_i64(bytes: &mut Vec<u8>, value: i64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn write_named_single_tensor_gguf_fixture_with_dims(
        path: &Path,
        tensor_name: &str,
        dims: [i64; 2],
    ) {
        const GGUF_VERSION: u32 = 3;
        const GGML_TYPE_F32: i32 = 0;
        const DEFAULT_ALIGNMENT: usize = 32;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, GGUF_VERSION);
        push_i64(&mut bytes, 1); // n_tensors
        push_i64(&mut bytes, 0); // n_kv

        push_gguf_string(&mut bytes, tensor_name);
        push_u32(&mut bytes, 2); // n_dims
        push_i64(&mut bytes, dims[0]);
        push_i64(&mut bytes, dims[1]);
        push_i32(&mut bytes, GGML_TYPE_F32);
        push_u64(&mut bytes, 0); // first tensor starts at data blob offset 0

        while bytes.len() % DEFAULT_ALIGNMENT != 0 {
            bytes.push(0);
        }

        let elements = dims[0].max(0).saturating_mul(dims[1].max(0));
        let payload_bytes = usize::try_from(elements.saturating_mul(4)).unwrap_or(0);
        bytes.resize(bytes.len().saturating_add(payload_bytes), 0);
        fs::write(path, bytes).expect("write gguf fixture");
    }

    fn write_named_single_tensor_gguf_fixture(path: &Path, tensor_name: &str) {
        write_named_single_tensor_gguf_fixture_with_dims(path, tensor_name, [4, 2]);
    }

    fn write_single_tensor_gguf_fixture(path: &Path) {
        write_named_single_tensor_gguf_fixture(path, "encoder.weight");
    }

    #[test]
    fn reads_tensor_index_and_supports_lookup() {
        let file = NamedTempFile::new().expect("temp file");
        write_single_tensor_gguf_fixture(file.path());

        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");
        assert_eq!(index.tensors().len(), 1);

        let tensor = index
            .get("encoder.weight")
            .expect("tensor lookup by name should succeed");
        assert_eq!(tensor.name, "encoder.weight");
        assert_eq!(tensor.dims, vec![4, 2]);
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.num_elements(), Some(8));
        assert_eq!(tensor.ggml_type, 0);
        assert_eq!(tensor.type_name, "f32");
        assert_eq!(tensor.size_bytes, 32);
        assert_eq!(
            tensor.offset_bytes,
            index.data_section_offset_bytes(),
            "first tensor should start at data section base"
        );
    }

    #[test]
    fn rejects_zero_tensor_dimension_without_aborting() {
        let file = NamedTempFile::new().expect("temp file");
        write_named_single_tensor_gguf_fixture_with_dims(file.path(), "zero.weight", [0, 2]);

        let error = read_gguf_tensor_index(file.path())
            .expect_err("zero-sized GGUF tensors must be rejected by the C parser");
        assert!(matches!(error, GgufTensorIndexReadError::InitFailed { .. }));
    }

    #[test]
    fn returns_none_for_missing_tensor_lookup() {
        let file = NamedTempFile::new().expect("temp file");
        write_single_tensor_gguf_fixture(file.path());

        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");
        assert!(index.get("missing.tensor").is_none());
    }

    #[test]
    fn access_trace_is_off_until_enabled_and_records_only_hits() {
        let file = NamedTempFile::new().expect("temp file");
        write_single_tensor_gguf_fixture(file.path());

        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");
        // Off by default: lookups before enabling are not recorded.
        let _ = index.get("encoder.weight");
        index.enable_access_trace();
        let _ = index.get("missing.tensor");
        let tensor = index.get("encoder.weight").expect("tensor exists");
        assert_eq!(tensor.dims, vec![4, 2]);

        assert_eq!(
            index.access_trace(),
            vec![super::GgufTensorAccessRecord {
                name: "encoder.weight".to_string(),
                dims: vec![4, 2],
            }],
            "only successful post-enable lookups may be traced"
        );
    }

    #[test]
    fn access_trace_is_shared_with_clones() {
        let file = NamedTempFile::new().expect("temp file");
        write_single_tensor_gguf_fixture(file.path());

        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");
        index.enable_access_trace();
        let cloned = index.clone();
        let _ = cloned.get("encoder.weight");

        assert_eq!(
            index.access_trace().len(),
            1,
            "a clone traces into the same recorder as its source index"
        );
    }

    #[test]
    fn access_trace_does_not_change_index_equality() {
        let file = NamedTempFile::new().expect("temp file");
        write_single_tensor_gguf_fixture(file.path());

        let left = read_gguf_tensor_index(file.path()).expect("read tensor index");
        let right = read_gguf_tensor_index(file.path()).expect("read tensor index");
        left.enable_access_trace();
        let _ = left.get("encoder.weight");

        assert_eq!(left, right, "traced lookups are not part of index identity");
    }

    #[cfg(unix)]
    #[test]
    fn runtime_source_tensor_index_is_bound_to_the_validated_file_identity() {
        let directory = tempfile::tempdir().expect("temp dir");
        let source_path = directory.path().join("model.gguf");
        let replacement_path = directory.path().join("replacement.gguf");
        write_named_single_tensor_gguf_fixture(&source_path, "validated.weight");

        let runtime_source =
            validate_ggml_runtime_source_path(&source_path).expect("validate runtime source");
        write_named_single_tensor_gguf_fixture(&replacement_path, "replacement.weight");
        fs::rename(&replacement_path, &source_path).expect("atomically replace path");

        let held_index =
            read_gguf_tensor_index_from_runtime_source(&runtime_source).expect("read held source");
        assert!(held_index.get("validated.weight").is_some());
        assert!(held_index.get("replacement.weight").is_none());

        let replacement_index =
            read_gguf_tensor_index(&source_path).expect("read replacement source");
        assert!(replacement_index.get("replacement.weight").is_some());
        assert!(replacement_index.get("validated.weight").is_none());
    }

    #[test]
    fn shape_helpers_report_match_and_mismatch() {
        let file = NamedTempFile::new().expect("temp file");
        write_single_tensor_gguf_fixture(file.path());

        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");
        let tensor = index.get("encoder.weight").expect("tensor exists");

        assert!(tensor.has_shape(&[4, 2]));
        assert!(!tensor.has_shape(&[2, 4]));

        let other = GgufTensorMetadata {
            name: "other".to_string(),
            dims: vec![4, 2],
            ggml_type: tensor.ggml_type,
            type_name: tensor.type_name.clone(),
            size_bytes: tensor.size_bytes,
            offset_bytes: tensor.offset_bytes,
        };
        let mismatched = GgufTensorMetadata {
            name: "mismatch".to_string(),
            dims: vec![4, 1, 2],
            ggml_type: tensor.ggml_type,
            type_name: tensor.type_name.clone(),
            size_bytes: tensor.size_bytes,
            offset_bytes: tensor.offset_bytes,
        };
        assert!(tensor.has_same_shape(&other));
        assert!(!tensor.has_same_shape(&mismatched));
    }

    #[test]
    fn num_elements_fails_closed_on_overflow() {
        let tensor = GgufTensorMetadata {
            name: "overflow.tensor".to_string(),
            dims: vec![u64::MAX, 2],
            ggml_type: 0,
            type_name: "f32".to_string(),
            size_bytes: 0,
            offset_bytes: 0,
        };

        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.num_elements(), None);
    }

    #[test]
    fn fail_closed_for_reserved_oasr_magic() {
        let file = NamedTempFile::new().expect("temp file");
        fs::write(file.path(), b"OASRpayload").expect("write reserved fixture");

        let error = read_gguf_tensor_index(file.path()).expect_err("reserved magic must fail");
        assert!(matches!(
            error,
            GgufTensorIndexReadError::InvalidRuntimeSource(
                crate::ggml_runtime::GgmlRuntimeSourcePathError::ReservedOpenAsrContainer { .. }
            )
        ));
    }

    #[test]
    fn fail_closed_for_unknown_magic() {
        let file = NamedTempFile::new().expect("temp file");
        fs::write(file.path(), b"ABCDpayload").expect("write unknown magic fixture");

        let error = read_gguf_tensor_index(file.path()).expect_err("unknown magic must fail");
        assert!(matches!(
            error,
            GgufTensorIndexReadError::InvalidRuntimeSource(
                crate::ggml_runtime::GgmlRuntimeSourcePathError::Probe(
                    crate::ggml_runtime::GgmlPackageProbeError::UnknownMagic { .. }
                )
            )
        ));
    }

    #[test]
    fn fail_closed_for_short_file() {
        let file = NamedTempFile::new().expect("temp file");
        fs::write(file.path(), b"GG").expect("write short fixture");

        let error = read_gguf_tensor_index(file.path()).expect_err("short file must fail");
        assert!(matches!(
            error,
            GgufTensorIndexReadError::InvalidRuntimeSource(
                crate::ggml_runtime::GgmlRuntimeSourcePathError::Probe(
                    crate::ggml_runtime::GgmlPackageProbeError::FileTooShort { .. }
                )
            )
        ));
    }
}
