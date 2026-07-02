use std::{
    collections::BTreeMap,
    ffi::{CStr, CString},
    fs,
    os::raw::c_void,
    path::{Path, PathBuf},
    ptr,
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufTensorIndex {
    path: PathBuf,
    data_section_offset_bytes: u64,
    tensors: Vec<GgufTensorMetadata>,
    tensor_index_by_name: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GgufTensorIndexSnapshot {
    pub(crate) path: PathBuf,
    pub(crate) data_section_offset_bytes: u64,
    pub(crate) tensors: Vec<GgufTensorMetadata>,
}

impl GgufTensorIndex {
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
        self.tensor_index_by_name
            .get(name)
            .and_then(|index| self.tensors.get(*index))
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
        })
    }
}

#[derive(Debug, Error)]
pub enum GgufTensorIndexReadError {
    #[error(transparent)]
    InvalidRuntimeSource(#[from] GgmlRuntimeSourcePathError),
    #[error("could not read gguf runtime source metadata for '{path}': {source}")]
    SourceMetadata {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("gguf tensor index path cannot be represented as C string: {path}")]
    PathContainsNul { path: String },
    #[error("gguf tensor index initialization failed for '{path}'")]
    InitFailed { path: PathBuf },
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
        "gguf tensor '{tensor_name}' in '{path}' has invalid rank: rank={rank}, tensor_index={tensor_index}"
    )]
    InvalidTensorRank {
        path: PathBuf,
        tensor_name: String,
        tensor_index: i64,
        rank: u32,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' rank does not fit usize: rank={rank}, tensor_index={tensor_index}"
    )]
    TensorRankOverflow {
        path: PathBuf,
        tensor_name: String,
        tensor_index: i64,
        rank: u32,
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
    let path = runtime_source.path();
    let file_size = fs::metadata(path)
        .map_err(|source| GgufTensorIndexReadError::SourceMetadata {
            path: path.to_path_buf(),
            source,
        })?
        .len();

    let path_cstring = path_to_cstring(path)?;
    let context = unsafe {
        let raw = ffi::gguf_init_from_file(
            path_cstring.as_ptr(),
            ffi::GgufInitParams {
                no_alloc: true,
                ctx: ptr::null_mut(),
            },
        );
        GgufContextGuard::from_raw(raw)
    }
    .ok_or_else(|| GgufTensorIndexReadError::InitFailed {
        path: path.to_path_buf(),
    })?;

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

        let rank = unsafe { ffi::gguf_get_tensor_n_dims(context.as_ptr(), tensor_index) };
        if rank == 0 {
            return Err(GgufTensorIndexReadError::InvalidTensorRank {
                path: path.to_path_buf(),
                tensor_name: name.clone(),
                tensor_index,
                rank,
            });
        }
        let rank_usize =
            usize::try_from(rank).map_err(|_| GgufTensorIndexReadError::TensorRankOverflow {
                path: path.to_path_buf(),
                tensor_name: name.clone(),
                tensor_index,
                rank,
            })?;
        let mut dims = Vec::with_capacity(rank_usize);
        for dim_index in 0..rank_usize {
            let dim_index_c = i32::try_from(dim_index).map_err(|_| {
                GgufTensorIndexReadError::TensorRankOverflow {
                    path: path.to_path_buf(),
                    tensor_name: name.clone(),
                    tensor_index,
                    rank,
                }
            })?;
            let dim_value =
                unsafe { ffi::gguf_get_tensor_dim(context.as_ptr(), tensor_index, dim_index_c) };
            if dim_value < 0 {
                return Err(GgufTensorIndexReadError::NegativeTensorDimension {
                    path: path.to_path_buf(),
                    tensor_name: name.clone(),
                    dim_index: dim_index_c,
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
    })
}

fn path_to_cstring(path: &Path) -> Result<CString, GgufTensorIndexReadError> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        CString::new(path.as_os_str().as_bytes()).map_err(|_| {
            GgufTensorIndexReadError::PathContainsNul {
                path: path.display().to_string(),
            }
        })
    }

    #[cfg(not(unix))]
    {
        let rendered = path.as_os_str().to_string_lossy();
        CString::new(rendered.as_bytes()).map_err(|_| GgufTensorIndexReadError::PathContainsNul {
            path: rendered.into_owned(),
        })
    }
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

struct GgufContextGuard {
    raw: ffi::GgufContextRaw,
}

impl GgufContextGuard {
    unsafe fn from_raw(raw: ffi::GgufContextRaw) -> Option<Self> {
        (!raw.is_null()).then_some(Self { raw })
    }

    fn as_ptr(&self) -> *const c_void {
        self.raw as *const c_void
    }
}

impl Drop for GgufContextGuard {
    fn drop(&mut self) {
        unsafe { ffi::gguf_free(self.raw) };
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use tempfile::NamedTempFile;

    use super::{GgufTensorIndexReadError, GgufTensorMetadata, read_gguf_tensor_index};

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

    fn write_single_tensor_gguf_fixture(path: &Path) {
        const GGUF_VERSION: u32 = 3;
        const GGML_TYPE_F32: i32 = 0;
        const DEFAULT_ALIGNMENT: usize = 32;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, GGUF_VERSION);
        push_i64(&mut bytes, 1); // n_tensors
        push_i64(&mut bytes, 0); // n_kv

        push_gguf_string(&mut bytes, "encoder.weight");
        push_u32(&mut bytes, 2); // n_dims
        push_i64(&mut bytes, 4);
        push_i64(&mut bytes, 2);
        push_i32(&mut bytes, GGML_TYPE_F32);
        push_u64(&mut bytes, 0); // first tensor starts at data blob offset 0

        while bytes.len() % DEFAULT_ALIGNMENT != 0 {
            bytes.push(0);
        }

        // 4 * 2 elements * f32(4 bytes) = 32 bytes tensor payload
        bytes.extend_from_slice(&[0u8; 32]);
        fs::write(path, bytes).expect("write gguf fixture");
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
    fn returns_none_for_missing_tensor_lookup() {
        let file = NamedTempFile::new().expect("temp file");
        write_single_tensor_gguf_fixture(file.path());

        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");
        assert!(index.get("missing.tensor").is_none());
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
