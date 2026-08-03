use std::{
    collections::BTreeMap,
    ffi::CStr,
    os::raw::c_void,
    path::{Path, PathBuf},
    ptr,
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{
    GgmlRuntimeSource, GgmlRuntimeSourcePathError, ffi, validate_ggml_runtime_source_path,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GgufMetadataValue {
    String(String),
    U32(u32),
    U64(u64),
    Bool(bool),
    F32(f32),
    StringArray(Vec<String>),
    U32Array(Vec<u32>),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GgufMetadata {
    values: BTreeMap<String, GgufMetadataValue>,
}

impl GgufMetadata {
    pub fn values(&self) -> &BTreeMap<String, GgufMetadataValue> {
        &self.values
    }

    pub fn get(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.values.get(key)
    }

    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.values.get(key) {
            Some(GgufMetadataValue::String(value)) => Some(value.as_str()),
            _ => None,
        }
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.values.get(key) {
            Some(GgufMetadataValue::U32(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.values.get(key) {
            Some(GgufMetadataValue::U64(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.values.get(key) {
            Some(GgufMetadataValue::Bool(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.values.get(key) {
            Some(GgufMetadataValue::F32(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn get_string_array(&self, key: &str) -> Option<&[String]> {
        match self.values.get(key) {
            Some(GgufMetadataValue::StringArray(value)) => Some(value.as_slice()),
            _ => None,
        }
    }

    pub fn get_u32_array(&self, key: &str) -> Option<&[u32]> {
        match self.values.get(key) {
            Some(GgufMetadataValue::U32Array(value)) => Some(value.as_slice()),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_values_for_test(values: BTreeMap<String, GgufMetadataValue>) -> Self {
        Self { values }
    }
}

#[derive(Debug, Error)]
pub enum GgufMetadataReadError {
    #[error(transparent)]
    InvalidRuntimeSource(#[from] GgmlRuntimeSourcePathError),
    /// Retained for source compatibility. Buffer-backed parsing no longer
    /// constructs a C path and therefore does not emit this variant.
    #[error("gguf metadata path cannot be represented as C string: {path}")]
    PathContainsNul { path: String },
    #[error("gguf metadata initialization failed for '{path}'")]
    InitFailed { path: PathBuf },
    #[error("gguf metadata allocation failed for '{path}'")]
    AllocationFailed { path: PathBuf },
    #[error("gguf metadata key count is negative for '{path}': {count}")]
    NegativeKeyCount { path: PathBuf, count: i64 },
    #[error("gguf metadata key {index} in '{path}' is null")]
    NullKey { path: PathBuf, index: i64 },
    #[error("gguf metadata key {index} in '{path}' is not valid utf-8: {source}")]
    InvalidKeyUtf8 {
        path: PathBuf,
        index: i64,
        source: std::str::Utf8Error,
    },
    #[error("gguf metadata value for key '{key}' in '{path}' is null")]
    NullStringValue { path: PathBuf, key: String },
    #[error("gguf metadata value for key '{key}' in '{path}' is not valid utf-8: {source}")]
    InvalidStringValueUtf8 {
        path: PathBuf,
        key: String,
        source: std::str::Utf8Error,
    },
    #[error(
        "gguf metadata array string value for key '{key}' in '{path}' at index {index} is null"
    )]
    NullArrayStringValue {
        path: PathBuf,
        key: String,
        index: usize,
    },
    #[error(
        "gguf metadata array string value for key '{key}' in '{path}' at index {index} is not valid utf-8: {source}"
    )]
    InvalidArrayStringValueUtf8 {
        path: PathBuf,
        key: String,
        index: usize,
        source: std::str::Utf8Error,
    },
    #[error("gguf metadata array value for key '{key}' in '{path}' has null data pointer")]
    NullArrayDataPointer { path: PathBuf, key: String },
}

pub fn read_gguf_metadata(path: impl AsRef<Path>) -> Result<GgufMetadata, GgufMetadataReadError> {
    let runtime_source = validate_ggml_runtime_source_path(path)?;
    read_gguf_metadata_from_runtime_source(&runtime_source)
}

pub fn read_gguf_metadata_from_runtime_source(
    runtime_source: &GgmlRuntimeSource,
) -> Result<GgufMetadata, GgufMetadataReadError> {
    read_gguf_metadata_from_runtime_source_internal(
        runtime_source,
        super::runtime_gguf_parse_limits(),
    )
}

pub(crate) fn read_gguf_metadata_from_runtime_source_with_limits(
    runtime_source: &GgmlRuntimeSource,
    max_tensors: u64,
    max_kv: u64,
) -> Result<GgufMetadata, GgufMetadataReadError> {
    read_gguf_metadata_from_runtime_source_internal(
        runtime_source,
        ffi::GgufParseLimits {
            max_tensors,
            max_kv,
            ..super::runtime_gguf_parse_limits()
        },
    )
}

fn read_gguf_metadata_from_runtime_source_internal(
    runtime_source: &GgmlRuntimeSource,
    limits: ffi::GgufParseLimits,
) -> Result<GgufMetadata, GgufMetadataReadError> {
    let path = runtime_source.path();
    let context = parse_bounded_gguf_context(runtime_source, limits).map_err(|failure| {
        if failure == GgufBoundedParseFailure::Allocation {
            crate::models::native_execution_services::record_current_execution_candidate_failure(
                crate::device::execution_policy::ExecutionCandidateFailure::capacity(
                    "gguf-bounded-metadata-parse",
                    format!("allocation failed while parsing {}", path.display()),
                ),
            );
            GgufMetadataReadError::AllocationFailed {
                path: path.to_path_buf(),
            }
        } else {
            GgufMetadataReadError::InitFailed {
                path: path.to_path_buf(),
            }
        }
    })?;

    read_gguf_metadata_from_context(runtime_source, &context)
}

pub(crate) fn read_gguf_metadata_from_context(
    runtime_source: &GgmlRuntimeSource,
    context: &GgufContextGuard,
) -> Result<GgufMetadata, GgufMetadataReadError> {
    let path = runtime_source.path();

    let key_count = unsafe { ffi::gguf_get_n_kv(context.as_ptr()) };
    if key_count < 0 {
        return Err(GgufMetadataReadError::NegativeKeyCount {
            path: path.to_path_buf(),
            count: key_count,
        });
    }

    let mut values = BTreeMap::new();
    for key_index in 0..key_count {
        let key_ptr = unsafe { ffi::gguf_get_key(context.as_ptr(), key_index) };
        if key_ptr.is_null() {
            return Err(GgufMetadataReadError::NullKey {
                path: path.to_path_buf(),
                index: key_index,
            });
        }

        let key = unsafe { CStr::from_ptr(key_ptr) }
            .to_str()
            .map_err(|source| GgufMetadataReadError::InvalidKeyUtf8 {
                path: path.to_path_buf(),
                index: key_index,
                source,
            })?;

        let key_type = unsafe { ffi::gguf_get_kv_type(context.as_ptr(), key_index) };
        let value = match key_type {
            ffi::GGUF_TYPE_STRING => {
                let value_ptr = unsafe { ffi::gguf_get_val_str(context.as_ptr(), key_index) };
                if value_ptr.is_null() {
                    return Err(GgufMetadataReadError::NullStringValue {
                        path: path.to_path_buf(),
                        key: key.to_string(),
                    });
                }
                let value = unsafe { CStr::from_ptr(value_ptr) }
                    .to_str()
                    .map_err(|source| GgufMetadataReadError::InvalidStringValueUtf8 {
                        path: path.to_path_buf(),
                        key: key.to_string(),
                        source,
                    })?;
                Some(GgufMetadataValue::String(value.to_string()))
            }
            ffi::GGUF_TYPE_UINT32 => Some(GgufMetadataValue::U32(unsafe {
                ffi::gguf_get_val_u32(context.as_ptr(), key_index)
            })),
            ffi::GGUF_TYPE_UINT64 => Some(GgufMetadataValue::U64(unsafe {
                ffi::gguf_get_val_u64(context.as_ptr(), key_index)
            })),
            ffi::GGUF_TYPE_BOOL => Some(GgufMetadataValue::Bool(unsafe {
                ffi::gguf_get_val_bool(context.as_ptr(), key_index)
            })),
            ffi::GGUF_TYPE_FLOAT32 => Some(GgufMetadataValue::F32(unsafe {
                ffi::gguf_get_val_f32(context.as_ptr(), key_index)
            })),
            ffi::GGUF_TYPE_ARRAY => {
                let item_type = unsafe { ffi::gguf_get_arr_type(context.as_ptr(), key_index) };
                let item_count = unsafe { ffi::gguf_get_arr_n(context.as_ptr(), key_index) };
                match item_type {
                    ffi::GGUF_TYPE_STRING => {
                        let mut values = Vec::with_capacity(item_count);
                        for item_index in 0..item_count {
                            let value_ptr = unsafe {
                                ffi::gguf_get_arr_str(context.as_ptr(), key_index, item_index)
                            };
                            if value_ptr.is_null() {
                                return Err(GgufMetadataReadError::NullArrayStringValue {
                                    path: path.to_path_buf(),
                                    key: key.to_string(),
                                    index: item_index,
                                });
                            }
                            let value = unsafe { CStr::from_ptr(value_ptr) }.to_str().map_err(
                                |source| GgufMetadataReadError::InvalidArrayStringValueUtf8 {
                                    path: path.to_path_buf(),
                                    key: key.to_string(),
                                    index: item_index,
                                    source,
                                },
                            )?;
                            values.push(value.to_string());
                        }
                        Some(GgufMetadataValue::StringArray(values))
                    }
                    ffi::GGUF_TYPE_UINT32 => {
                        let data_ptr =
                            unsafe { ffi::gguf_get_arr_data(context.as_ptr(), key_index) };
                        if data_ptr.is_null() && item_count != 0 {
                            return Err(GgufMetadataReadError::NullArrayDataPointer {
                                path: path.to_path_buf(),
                                key: key.to_string(),
                            });
                        }
                        let values = if item_count == 0 {
                            Vec::new()
                        } else {
                            unsafe {
                                std::slice::from_raw_parts(data_ptr.cast::<u32>(), item_count)
                            }
                            .to_vec()
                        };
                        Some(GgufMetadataValue::U32Array(values))
                    }
                    _ => None,
                }
            }
            _ => None,
        };

        if let Some(value) = value {
            values.insert(key.to_string(), value);
        }
    }

    Ok(GgufMetadata { values })
}

pub(crate) fn bounded_gguf_parser_structural_bytes(n_kv: u64, n_tensors: u64) -> Option<u64> {
    let mut bytes = 0_usize;
    let ok = unsafe { ffi::gguf_bounded_parser_structural_bytes(n_kv, n_tensors, &mut bytes) };
    ok.then(|| u64::try_from(bytes).ok()).flatten()
}

pub(crate) fn bounded_gguf_parser_payload_wire_multiplier() -> Option<u64> {
    u64::try_from(unsafe { ffi::gguf_bounded_parser_payload_wire_multiplier() }).ok()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GgufBoundedParseFailure {
    InvalidData,
    Allocation,
}

pub(crate) fn parse_bounded_gguf_context(
    runtime_source: &GgmlRuntimeSource,
    limits: ffi::GgufParseLimits,
) -> Result<GgufContextGuard, GgufBoundedParseFailure> {
    #[cfg(test)]
    BOUNDED_PARSE_CALLS.with(|calls| calls.set(calls.get().saturating_add(1)));
    let mmap = runtime_source.backing_mmap();
    let mut parse_error = ffi::GGUF_PARSE_ERROR_NONE;
    let context = unsafe {
        let params = ffi::GgufInitParams {
            no_alloc: true,
            ctx: ptr::null_mut(),
        };
        let raw = ffi::gguf_init_from_buffer_with_limits(
            mmap.as_ptr().cast(),
            mmap.len(),
            params,
            limits,
            &mut parse_error,
        );
        GgufContextGuard::from_raw(raw)
    };
    context.ok_or_else(|| {
        if parse_error == ffi::GGUF_PARSE_ERROR_ALLOCATION {
            GgufBoundedParseFailure::Allocation
        } else {
            debug_assert!(matches!(
                parse_error,
                ffi::GGUF_PARSE_ERROR_NONE | ffi::GGUF_PARSE_ERROR_INVALID_DATA
            ));
            GgufBoundedParseFailure::InvalidData
        }
    })
}

#[cfg(test)]
thread_local! {
    static BOUNDED_PARSE_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn bounded_parse_call_count_for_current_thread() -> u64 {
    BOUNDED_PARSE_CALLS.with(std::cell::Cell::get)
}

pub(crate) struct GgufContextGuard {
    raw: ffi::GgufContextRaw,
}

impl GgufContextGuard {
    unsafe fn from_raw(raw: ffi::GgufContextRaw) -> Option<Self> {
        (!raw.is_null()).then_some(Self { raw })
    }

    pub(crate) fn as_ptr(&self) -> *const c_void {
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
    use std::{ffi::CString, fs, path::Path};

    use tempfile::NamedTempFile;

    use super::{
        GgufMetadataReadError, GgufMetadataValue, read_gguf_metadata,
        read_gguf_metadata_from_runtime_source,
    };
    use crate::validate_ggml_runtime_source_path;

    enum TestEntry<'a> {
        String(&'a str, &'a str),
        U32(&'a str, u32),
        U64(&'a str, u64),
        Bool(&'a str, bool),
        F32(&'a str, f32),
        StringArray(&'a str, Vec<&'a str>),
        U32Array(&'a str, Vec<u32>),
    }

    fn write_fixture(path: &Path, entries: &[TestEntry<'_>]) {
        let path_string = path.to_string_lossy().to_string();
        let path_c = CString::new(path_string).expect("fixture path cstring");

        let ctx = unsafe { super::ffi::gguf_init_empty() };
        assert!(!ctx.is_null(), "gguf_init_empty must produce a context");
        let guard = super::GgufContextGuard { raw: ctx };

        for entry in entries {
            match entry {
                TestEntry::String(key, value) => {
                    let key_c = CString::new(*key).expect("key cstring");
                    let value_c = CString::new(*value).expect("value cstring");
                    unsafe {
                        super::ffi::gguf_set_val_str(guard.raw, key_c.as_ptr(), value_c.as_ptr());
                    }
                }
                TestEntry::U32(key, value) => {
                    let key_c = CString::new(*key).expect("key cstring");
                    unsafe { super::ffi::gguf_set_val_u32(guard.raw, key_c.as_ptr(), *value) }
                }
                TestEntry::U64(key, value) => {
                    let key_c = CString::new(*key).expect("key cstring");
                    unsafe { super::ffi::gguf_set_val_u64(guard.raw, key_c.as_ptr(), *value) }
                }
                TestEntry::Bool(key, value) => {
                    let key_c = CString::new(*key).expect("key cstring");
                    unsafe { super::ffi::gguf_set_val_bool(guard.raw, key_c.as_ptr(), *value) }
                }
                TestEntry::F32(key, value) => {
                    let key_c = CString::new(*key).expect("key cstring");
                    unsafe { super::ffi::gguf_set_val_f32(guard.raw, key_c.as_ptr(), *value) }
                }
                TestEntry::StringArray(key, values) => {
                    let key_c = CString::new(*key).expect("key cstring");
                    let value_c = values
                        .iter()
                        .map(|value| CString::new(*value).expect("value cstring"))
                        .collect::<Vec<_>>();
                    let value_ptrs = value_c
                        .iter()
                        .map(|value| value.as_ptr())
                        .collect::<Vec<_>>();
                    unsafe {
                        super::ffi::gguf_set_arr_str(
                            guard.raw,
                            key_c.as_ptr(),
                            value_ptrs.as_ptr(),
                            value_ptrs.len(),
                        );
                    }
                }
                TestEntry::U32Array(key, values) => {
                    let key_c = CString::new(*key).expect("key cstring");
                    unsafe {
                        super::ffi::gguf_set_arr_data(
                            guard.raw,
                            key_c.as_ptr(),
                            super::ffi::GGUF_TYPE_UINT32,
                            values.as_ptr().cast(),
                            values.len(),
                        )
                    }
                }
            }
        }

        let ok = unsafe { super::ffi::gguf_write_to_file(guard.as_ptr(), path_c.as_ptr(), true) };
        assert!(ok, "gguf_write_to_file must succeed");
    }

    fn parse_raw_with_limits(
        bytes: &[u8],
        limits: super::ffi::GgufParseLimits,
    ) -> (bool, std::os::raw::c_int) {
        let mut error = super::ffi::GGUF_PARSE_ERROR_NONE;
        let raw = unsafe {
            super::ffi::gguf_init_from_buffer_with_limits(
                bytes.as_ptr().cast(),
                bytes.len(),
                super::ffi::GgufInitParams {
                    no_alloc: true,
                    ctx: std::ptr::null_mut(),
                },
                limits,
                &mut error,
            )
        };
        if !raw.is_null() {
            unsafe { super::ffi::gguf_free(raw) };
        }
        (!raw.is_null(), error)
    }

    fn empty_gguf_header(tensors: i64, kv: i64) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&tensors.to_le_bytes());
        bytes.extend_from_slice(&kv.to_le_bytes());
        bytes
    }

    #[test]
    fn bounded_parser_abi_enforces_every_header_resource_dimension() {
        let defaults = super::super::runtime_gguf_parse_limits();
        let valid = empty_gguf_header(0, 0);
        assert_eq!(parse_raw_with_limits(&valid, defaults), (true, 0));

        let (_, tensor_error) = parse_raw_with_limits(
            &empty_gguf_header(1, 0),
            super::ffi::GgufParseLimits {
                max_tensors: 0,
                ..defaults
            },
        );
        assert_eq!(tensor_error, super::ffi::GGUF_PARSE_ERROR_INVALID_DATA);

        let (_, kv_error) = parse_raw_with_limits(
            &empty_gguf_header(0, 1),
            super::ffi::GgufParseLimits {
                max_kv: 0,
                ..defaults
            },
        );
        assert_eq!(kv_error, super::ffi::GGUF_PARSE_ERROR_INVALID_DATA);

        let (_, header_error) = parse_raw_with_limits(
            &valid,
            super::ffi::GgufParseLimits {
                max_header_bytes: u64::try_from(valid.len() - 1).expect("header length"),
                ..defaults
            },
        );
        assert_eq!(header_error, super::ffi::GGUF_PARSE_ERROR_INVALID_DATA);

        let mut string_value = empty_gguf_header(0, 1);
        string_value.extend_from_slice(&4_u64.to_le_bytes());
        string_value.extend_from_slice(b"name");
        string_value.extend_from_slice(&super::ffi::GGUF_TYPE_STRING.to_le_bytes());
        string_value.extend_from_slice(&1_u64.to_le_bytes());
        string_value.extend_from_slice(b"x");
        let (_, string_error) = parse_raw_with_limits(
            &string_value,
            super::ffi::GgufParseLimits {
                max_string_bytes: 3,
                ..defaults
            },
        );
        assert_eq!(string_error, super::ffi::GGUF_PARSE_ERROR_INVALID_DATA);

        let mut array_value = empty_gguf_header(0, 1);
        array_value.extend_from_slice(&1_u64.to_le_bytes());
        array_value.extend_from_slice(b"x");
        array_value.extend_from_slice(&super::ffi::GGUF_TYPE_ARRAY.to_le_bytes());
        array_value.extend_from_slice(&super::ffi::GGUF_TYPE_UINT32.to_le_bytes());
        array_value.extend_from_slice(&2_u64.to_le_bytes());
        array_value.extend_from_slice(&1_u32.to_le_bytes());
        array_value.extend_from_slice(&2_u32.to_le_bytes());
        let (_, array_error) = parse_raw_with_limits(
            &array_value,
            super::ffi::GgufParseLimits {
                max_array_elements: 1,
                ..defaults
            },
        );
        assert_eq!(array_error, super::ffi::GGUF_PARSE_ERROR_INVALID_DATA);
    }

    #[test]
    fn reads_supported_metadata_types() {
        let file = NamedTempFile::new().expect("temp file");
        write_fixture(
            file.path(),
            &[
                TestEntry::String("openasr.model.id", "whisper-small:q4_0"),
                TestEntry::String("general.name", "Whisper Small"),
                TestEntry::U32("general.alignment", 32),
                TestEntry::U64("openasr.checkpoint.bytes", 987_654),
                TestEntry::Bool("openasr.runtime.fastpath", true),
                TestEntry::F32("openasr.runtime.temperature", 0.5),
                TestEntry::StringArray(
                    "tokenizer.ggml.tokens",
                    vec!["<|endoftext|>", "<|startoftranscript|>"],
                ),
                TestEntry::U32Array("tokenizer.ggml.special_ids", vec![50256, 50257]),
            ],
        );

        let metadata = read_gguf_metadata(file.path()).expect("read metadata");
        assert_eq!(
            metadata.get("openasr.model.id"),
            Some(&GgufMetadataValue::String("whisper-small:q4_0".to_string()))
        );
        assert_eq!(metadata.get_u32("general.alignment"), Some(32));
        assert_eq!(metadata.get_u64("openasr.checkpoint.bytes"), Some(987_654));
        assert_eq!(metadata.get_bool("openasr.runtime.fastpath"), Some(true));
        assert_eq!(metadata.get_f32("openasr.runtime.temperature"), Some(0.5));
        assert_eq!(
            metadata.get_string_array("tokenizer.ggml.tokens"),
            Some(
                &[
                    "<|endoftext|>".to_string(),
                    "<|startoftranscript|>".to_string()
                ][..]
            )
        );
        assert_eq!(
            metadata.get_u32_array("tokenizer.ggml.special_ids"),
            Some(&[50256, 50257][..])
        );
    }

    #[test]
    fn rejects_non_u32_general_alignment_without_aborting() {
        let file = NamedTempFile::new().expect("temp file");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_i64.to_le_bytes());
        bytes.extend_from_slice(&1_i64.to_le_bytes());
        let key = b"general.alignment";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&super::ffi::GGUF_TYPE_STRING.to_le_bytes());
        let value = b"32";
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value);
        std::fs::write(file.path(), bytes).expect("write malformed alignment fixture");

        let error = read_gguf_metadata(file.path())
            .expect_err("a string-valued alignment must be rejected by the C parser");
        assert!(matches!(error, GgufMetadataReadError::InitFailed { .. }));
    }

    #[test]
    fn reads_metadata_from_validated_runtime_source() {
        let file = NamedTempFile::new().expect("temp file");
        write_fixture(
            file.path(),
            &[TestEntry::String("openasr.model.id", "whisper-tiny:q8_0")],
        );

        let runtime_source =
            validate_ggml_runtime_source_path(file.path()).expect("validate runtime source");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read metadata");
        assert_eq!(
            metadata.get_string("openasr.model.id"),
            Some("whisper-tiny:q8_0")
        );
    }

    #[cfg(unix)]
    #[test]
    fn runtime_source_metadata_is_bound_to_the_validated_file_identity() {
        let directory = tempfile::tempdir().expect("temp dir");
        let source_path = directory.path().join("model.gguf");
        let replacement_path = directory.path().join("replacement.gguf");
        write_fixture(
            &source_path,
            &[TestEntry::String("openasr.model.id", "validated:model")],
        );

        let runtime_source =
            validate_ggml_runtime_source_path(&source_path).expect("validate runtime source");
        write_fixture(
            &replacement_path,
            &[TestEntry::String("openasr.model.id", "replacement:model")],
        );
        fs::rename(&replacement_path, &source_path).expect("atomically replace path");

        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read held source");
        assert_eq!(
            metadata.get_string("openasr.model.id"),
            Some("validated:model")
        );
        assert_eq!(
            read_gguf_metadata(&source_path)
                .expect("read replacement")
                .get_string("openasr.model.id"),
            Some("replacement:model")
        );
    }

    #[test]
    fn fail_closed_for_reserved_oasr_magic() {
        let file = NamedTempFile::new().expect("temp file");
        fs::write(file.path(), b"OASRpayload").expect("write reserved magic fixture");

        let error = read_gguf_metadata(file.path()).expect_err("reserved magic must fail");
        assert!(matches!(
            error,
            GgufMetadataReadError::InvalidRuntimeSource(
                crate::ggml_runtime::GgmlRuntimeSourcePathError::ReservedOpenAsrContainer { .. }
            )
        ));
    }

    #[test]
    fn fail_closed_for_short_file() {
        let file = NamedTempFile::new().expect("temp file");
        fs::write(file.path(), b"GG").expect("write short fixture");

        let error = read_gguf_metadata(file.path()).expect_err("short file must fail");
        assert!(matches!(
            error,
            GgufMetadataReadError::InvalidRuntimeSource(
                crate::ggml_runtime::GgmlRuntimeSourcePathError::Probe(
                    crate::ggml_runtime::GgmlPackageProbeError::FileTooShort { .. }
                )
            )
        ));
    }
}
