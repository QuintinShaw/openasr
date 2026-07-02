use std::{
    collections::BTreeMap,
    ffi::{CStr, CString},
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
    #[error("gguf metadata path cannot be represented as C string: {path}")]
    PathContainsNul { path: String },
    #[error("gguf metadata initialization failed for '{path}'")]
    InitFailed { path: PathBuf },
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
    let path = runtime_source.path();
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
    .ok_or_else(|| GgufMetadataReadError::InitFailed {
        path: path.to_path_buf(),
    })?;

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

fn path_to_cstring(path: &Path) -> Result<CString, GgufMetadataReadError> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        CString::new(path.as_os_str().as_bytes()).map_err(|_| {
            GgufMetadataReadError::PathContainsNul {
                path: path.display().to_string(),
            }
        })
    }

    #[cfg(not(unix))]
    {
        let rendered = path.as_os_str().to_string_lossy();
        CString::new(rendered.as_bytes()).map_err(|_| GgufMetadataReadError::PathContainsNul {
            path: rendered.into_owned(),
        })
    }
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
