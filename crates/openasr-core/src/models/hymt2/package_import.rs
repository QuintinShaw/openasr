//! Hy-MT2 fixed-revision GGUF -> `.oasr` repackaging importer.
//!
//! The Hy-MT2 release lane does not modify model weights: the upstream GGUF
//! tensor data section is preserved byte-for-byte. The importer only splices
//! `openasr.*` provenance/licensing metadata into the GGUF KV section so the
//! distributed pack carries the upstream LICENSE, the OpenASR modification
//! NOTICE, the pinned upstream revisions, and the translation-model contract
//! required by the publish tooling and the realtime server.
//!
//! Everything here is fail-closed: an unexpected source hash, architecture,
//! quantization file type, tokenizer contract, or pre-existing `openasr.*`
//! metadata aborts the import without writing the output pack.

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};
use thiserror::Error;

use super::config::{HYMT2_EXPECTED_LAYERS, HYMT2_EXPECTED_VOCAB_SIZE};
use crate::models::oasr_metadata::{OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1};

/// Pinned upstream base (safetensors) repository for provenance metadata.
pub const HYMT2_UPSTREAM_BASE_REPO: &str = "tencent/Hy-MT2-1.8B";
/// Pinned upstream base repository revision.
pub const HYMT2_UPSTREAM_BASE_REVISION: &str = "9a341cd1b679d3efd23b46e847b01745a71ed792";
/// Pinned upstream GGUF repository the source artifact must come from.
pub const HYMT2_UPSTREAM_GGUF_REPO: &str = "tencent/Hy-MT2-1.8B-GGUF";
/// Pinned upstream GGUF repository revision.
pub const HYMT2_UPSTREAM_GGUF_REVISION: &str = "1cd5208700acedef4ef93019b6cfc148b8522d45";
/// sha256 of the pinned upstream `Hy-MT2-1.8B-Q4_K_M.gguf` artifact.
pub const HYMT2_PINNED_SOURCE_GGUF_SHA256: &str =
    "dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699";
/// Upstream license URL recorded in pack metadata.
pub const HYMT2_LICENSE_SOURCE_URL: &str = "https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/blob/1cd5208700acedef4ef93019b6cfc148b8522d45/LICENSE.txt";

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const GGUF_SUPPORTED_VERSION: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
const GGUF_HEADER_LEN: usize = 24;
// llama.cpp LLAMA_FTYPE_MOSTLY_Q4_K_M.
const HYMT2_EXPECTED_GENERAL_FILE_TYPE: u32 = 15;
const HYMT2_EXPECTED_TOKENIZER_MODEL: &str = "gpt2";
const HYMT2_EXPECTED_TOKENIZER_PRE: &str = "hunyuan-dense";
const GENERAL_ARCHITECTURE_KEY: &str = "general.architecture";
const GENERAL_FILE_TYPE_KEY: &str = "general.file_type";
const GENERAL_ALIGNMENT_KEY: &str = "general.alignment";
const HUNYUAN_DENSE_ARCHITECTURE: &str = "hunyuan-dense";
const OPENASR_KEY_PREFIX: &str = "openasr.";

const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

/// One Hy-MT2 GGUF -> `.oasr` repackaging request.
#[derive(Debug, Clone)]
pub struct Hymt2ImportRequest {
    /// Pinned upstream `Hy-MT2-1.8B-Q4_K_M.gguf` source file.
    pub source_gguf: PathBuf,
    /// Output `.oasr` runtime pack path; must not exist yet.
    pub output_pack: PathBuf,
    /// Model id written to `openasr.model.id`.
    pub model_id: String,
    /// Catalog quant name written to `openasr.quantization`.
    pub quantization: String,
    /// Upstream `LICENSE.txt` contents embedded into the pack.
    pub license_text: String,
    /// OpenASR `NOTICE.openasr.txt` contents embedded into the pack.
    pub notice_text: String,
    /// Required sha256 of the source GGUF (64 lowercase hex characters).
    pub expected_source_sha256: String,
}

/// Result of a successful Hy-MT2 import.
#[derive(Debug, Clone)]
pub struct Hymt2ImportResult {
    /// Written `.oasr` pack path.
    pub output_path: PathBuf,
    /// sha256 of the source GGUF.
    pub source_sha256: String,
    /// sha256 of the written pack.
    pub pack_sha256: String,
    /// Pack size in bytes.
    pub pack_size_bytes: u64,
    /// Number of `openasr.*` metadata entries spliced into the KV section.
    pub appended_metadata_entries: usize,
    /// Tensor count carried over from the source GGUF.
    pub tensor_count: u64,
}

/// Fail-closed Hy-MT2 import errors.
#[derive(Debug, Error)]
pub enum Hymt2ImportError {
    #[error("hymt2 import i/o error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("hymt2 import output already exists: {path}")]
    OutputExists { path: PathBuf },
    #[error("hymt2 import output must end with .oasr: {path}")]
    OutputNotOasr { path: PathBuf },
    #[error(
        "hymt2 import source sha256 mismatch: expected {expected}, got {actual}; refusing to repackage an unpinned GGUF"
    )]
    SourceShaMismatch { expected: String, actual: String },
    #[error("hymt2 import expected sha256 must be 64 lowercase hex characters, got '{value}'")]
    InvalidExpectedSha { value: String },
    #[error("hymt2 import source is not a GGUF v{GGUF_SUPPORTED_VERSION} file: {reason}")]
    MalformedGguf { reason: String },
    #[error("hymt2 import source metadata mismatch for {key}: expected {expected}, got {actual}")]
    MetadataMismatch {
        key: String,
        expected: String,
        actual: String,
    },
    #[error("hymt2 import source is missing required metadata key {key}")]
    MetadataMissing { key: String },
    #[error(
        "hymt2 import source already contains reserved metadata key {key}; refusing to re-import an already-packaged file"
    )]
    ReservedMetadataPresent { key: String },
    #[error("hymt2 import {field} must not be empty")]
    EmptyField { field: &'static str },
    #[error(
        "hymt2 import NOTICE text must record the pinned upstream revision {revision}; refusing a stale notice"
    )]
    NoticeMissingRevision { revision: String },
    #[error("hymt2 import metadata value for {field} cannot contain NUL bytes")]
    FieldContainsNul { field: &'static str },
}

fn io_error(path: &Path) -> impl FnOnce(std::io::Error) -> Hymt2ImportError + '_ {
    move |source| Hymt2ImportError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Repackage the pinned upstream Hy-MT2 GGUF into an `.oasr` runtime pack,
/// preserving tensor data byte-for-byte and splicing `openasr.*` provenance,
/// translation contract, and license/notice metadata into the KV section.
pub fn import_hymt2_gguf_to_runtime_pack(
    request: &Hymt2ImportRequest,
) -> Result<Hymt2ImportResult, Hymt2ImportError> {
    validate_request(request)?;

    let source_sha256 = file_sha256(&request.source_gguf)?;
    if source_sha256 != request.expected_source_sha256 {
        return Err(Hymt2ImportError::SourceShaMismatch {
            expected: request.expected_source_sha256.clone(),
            actual: source_sha256,
        });
    }

    let source_file = File::open(&request.source_gguf).map_err(io_error(&request.source_gguf))?;
    let mut reader = BufReader::new(source_file);

    let header = read_exact_vec(&mut reader, GGUF_HEADER_LEN, &request.source_gguf)?;
    if header[0..4] != GGUF_MAGIC {
        return Err(Hymt2ImportError::MalformedGguf {
            reason: "bad magic".to_string(),
        });
    }
    let version = u32::from_le_bytes(header[4..8].try_into().expect("4-byte slice"));
    if version != GGUF_SUPPORTED_VERSION {
        return Err(Hymt2ImportError::MalformedGguf {
            reason: format!("unsupported version {version}"),
        });
    }
    let tensor_count = u64::from_le_bytes(header[8..16].try_into().expect("8-byte slice"));
    let kv_count = u64::from_le_bytes(header[16..24].try_into().expect("8-byte slice"));

    // Parse and capture the KV + tensor-info sections verbatim.
    let mut capture = CaptureReader {
        inner: &mut reader,
        captured: Vec::new(),
        path: &request.source_gguf,
    };
    let mut validator = SourceMetadataValidator::default();
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut capture)?;
        if key.starts_with(OPENASR_KEY_PREFIX) {
            return Err(Hymt2ImportError::ReservedMetadataPresent { key });
        }
        let value_type = read_u32(&mut capture)?;
        let value = read_gguf_value(&mut capture, value_type)?;
        validator.observe(&key, &value);
    }
    validator.finish()?;

    for _ in 0..tensor_count {
        let _name = read_gguf_string(&mut capture)?;
        let n_dims = read_u32(&mut capture)?;
        if n_dims == 0 || n_dims > 4 {
            return Err(Hymt2ImportError::MalformedGguf {
                reason: format!("tensor rank {n_dims} out of range"),
            });
        }
        for _ in 0..n_dims {
            let _dim = read_u64(&mut capture)?;
        }
        let _tensor_type = read_u32(&mut capture)?;
        let _offset = read_u64(&mut capture)?;
    }
    let captured = capture.captured;
    let alignment = validator.alignment.unwrap_or(GGUF_DEFAULT_ALIGNMENT);
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(Hymt2ImportError::MalformedGguf {
            reason: format!("invalid general.alignment {alignment}"),
        });
    }

    let appended = build_openasr_metadata_entries(request, &source_sha256)?;
    let appended_count = appended.len() as u64;
    let mut appended_bytes = Vec::new();
    for (key, value) in &appended {
        write_gguf_string(&mut appended_bytes, key);
        match value {
            SplicedValue::String(text) => {
                appended_bytes.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
                write_gguf_string(&mut appended_bytes, text);
            }
            SplicedValue::StringArray(items) => {
                appended_bytes.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
                appended_bytes.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
                appended_bytes.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for item in items {
                    write_gguf_string(&mut appended_bytes, item);
                }
            }
        }
    }

    // Skip the source padding between the header and the tensor data section.
    let source_header_end = GGUF_HEADER_LEN as u64 + captured.len() as u64;
    let source_padding = source_header_end.next_multiple_of(alignment) - source_header_end;
    skip_bytes(&mut reader, source_padding, &request.source_gguf)?;

    // The new `openasr.*` entries are written at the front of the KV section so
    // header-window scanners (publish preflight markers) find them without
    // walking the multi-megabyte tokenizer arrays.
    let output_header_end =
        GGUF_HEADER_LEN as u64 + appended_bytes.len() as u64 + captured.len() as u64;
    let output_padding = output_header_end.next_multiple_of(alignment) - output_header_end;

    write_output_pack(
        request,
        tensor_count,
        kv_count + appended_count,
        &appended_bytes,
        &captured,
        output_padding,
        &mut reader,
    )?;

    let pack_sha256 = file_sha256(&request.output_pack)?;
    let pack_size_bytes = std::fs::metadata(&request.output_pack)
        .map_err(io_error(&request.output_pack))?
        .len();
    Ok(Hymt2ImportResult {
        output_path: request.output_pack.clone(),
        source_sha256,
        pack_sha256,
        pack_size_bytes,
        appended_metadata_entries: appended.len(),
        tensor_count,
    })
}

fn validate_request(request: &Hymt2ImportRequest) -> Result<(), Hymt2ImportError> {
    if request.output_pack.extension().and_then(|ext| ext.to_str()) != Some("oasr") {
        return Err(Hymt2ImportError::OutputNotOasr {
            path: request.output_pack.clone(),
        });
    }
    if request.output_pack.exists() {
        return Err(Hymt2ImportError::OutputExists {
            path: request.output_pack.clone(),
        });
    }
    if request.expected_source_sha256.len() != 64
        || !request
            .expected_source_sha256
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(Hymt2ImportError::InvalidExpectedSha {
            value: request.expected_source_sha256.clone(),
        });
    }
    for (value, field) in [
        (&request.model_id, "model_id"),
        (&request.quantization, "quantization"),
        (&request.license_text, "license_text"),
        (&request.notice_text, "notice_text"),
    ] {
        if value.trim().is_empty() {
            return Err(Hymt2ImportError::EmptyField { field });
        }
        if value.contains('\0') {
            return Err(Hymt2ImportError::FieldContainsNul { field });
        }
    }
    for revision in [HYMT2_UPSTREAM_BASE_REVISION, HYMT2_UPSTREAM_GGUF_REVISION] {
        if !request.notice_text.contains(revision) {
            return Err(Hymt2ImportError::NoticeMissingRevision {
                revision: revision.to_string(),
            });
        }
    }
    Ok(())
}

enum SplicedValue {
    String(String),
    StringArray(Vec<&'static str>),
}

fn build_openasr_metadata_entries(
    request: &Hymt2ImportRequest,
    source_sha256: &str,
) -> Result<Vec<(String, SplicedValue)>, Hymt2ImportError> {
    Ok(vec![
        // `.oasr` v1 package contract (docs/format/OASR_PACKAGE_CONTRACT_V1.md):
        // `openasr.package.version = "1"` is REQUIRED by the generic pull
        // preflight; a pack without it can be built but never installed.
        (
            OASR_METADATA_KEY_PACKAGE_VERSION.to_string(),
            SplicedValue::String(OASR_PACKAGE_VERSION_V1.to_string()),
        ),
        (
            "openasr.model.kind".to_string(),
            SplicedValue::String("translation-model".to_string()),
        ),
        (
            "openasr.model.id".to_string(),
            SplicedValue::String(request.model_id.clone()),
        ),
        (
            "openasr.quantization".to_string(),
            SplicedValue::String(request.quantization.clone()),
        ),
        (
            "openasr.translation.source_langs".to_string(),
            SplicedValue::StringArray(vec!["zh"]),
        ),
        (
            "openasr.translation.target_langs".to_string(),
            SplicedValue::StringArray(vec!["en"]),
        ),
        (
            "openasr.upstream.base_repo".to_string(),
            SplicedValue::String(HYMT2_UPSTREAM_BASE_REPO.to_string()),
        ),
        (
            "openasr.upstream.base_revision".to_string(),
            SplicedValue::String(HYMT2_UPSTREAM_BASE_REVISION.to_string()),
        ),
        (
            "openasr.upstream.gguf_repo".to_string(),
            SplicedValue::String(HYMT2_UPSTREAM_GGUF_REPO.to_string()),
        ),
        (
            "openasr.upstream.gguf_revision".to_string(),
            SplicedValue::String(HYMT2_UPSTREAM_GGUF_REVISION.to_string()),
        ),
        (
            "openasr.upstream.gguf_sha256".to_string(),
            SplicedValue::String(source_sha256.to_string()),
        ),
        (
            "openasr.license.name".to_string(),
            SplicedValue::String("Apache-2.0".to_string()),
        ),
        (
            "openasr.license.source".to_string(),
            SplicedValue::String(HYMT2_LICENSE_SOURCE_URL.to_string()),
        ),
        (
            "openasr.license.files".to_string(),
            SplicedValue::StringArray(vec!["LICENSE.txt", "NOTICE.openasr.txt"]),
        ),
        (
            "openasr.license.file.LICENSE.txt".to_string(),
            SplicedValue::String(request.license_text.clone()),
        ),
        (
            "openasr.license.file.NOTICE.openasr.txt".to_string(),
            SplicedValue::String(request.notice_text.clone()),
        ),
    ])
}

#[derive(Default)]
struct SourceMetadataValidator {
    architecture: Option<String>,
    file_type: Option<u32>,
    block_count: Option<u32>,
    tokenizer_model: Option<String>,
    tokenizer_pre: Option<String>,
    token_count: Option<u64>,
    alignment: Option<u64>,
}

enum ObservedValue {
    String(String),
    U32(u32),
    ArrayLen(u64),
    Other,
}

impl SourceMetadataValidator {
    fn observe(&mut self, key: &str, value: &ObservedValue) {
        match (key, value) {
            (GENERAL_ARCHITECTURE_KEY, ObservedValue::String(text)) => {
                self.architecture = Some(text.clone());
            }
            (GENERAL_FILE_TYPE_KEY, ObservedValue::U32(value)) => self.file_type = Some(*value),
            (GENERAL_ALIGNMENT_KEY, ObservedValue::U32(value)) => {
                self.alignment = Some(u64::from(*value));
            }
            ("hunyuan-dense.block_count", ObservedValue::U32(value)) => {
                self.block_count = Some(*value);
            }
            ("tokenizer.ggml.model", ObservedValue::String(text)) => {
                self.tokenizer_model = Some(text.clone());
            }
            ("tokenizer.ggml.pre", ObservedValue::String(text)) => {
                self.tokenizer_pre = Some(text.clone());
            }
            ("tokenizer.ggml.tokens", ObservedValue::ArrayLen(count)) => {
                self.token_count = Some(*count);
            }
            _ => {}
        }
    }

    fn finish(&self) -> Result<(), Hymt2ImportError> {
        require_value(
            GENERAL_ARCHITECTURE_KEY,
            self.architecture.as_deref(),
            HUNYUAN_DENSE_ARCHITECTURE,
        )?;
        require_value(
            "tokenizer.ggml.model",
            self.tokenizer_model.as_deref(),
            HYMT2_EXPECTED_TOKENIZER_MODEL,
        )?;
        require_value(
            "tokenizer.ggml.pre",
            self.tokenizer_pre.as_deref(),
            HYMT2_EXPECTED_TOKENIZER_PRE,
        )?;
        require_numeric(
            GENERAL_FILE_TYPE_KEY,
            self.file_type.map(u64::from),
            u64::from(HYMT2_EXPECTED_GENERAL_FILE_TYPE),
        )?;
        require_numeric(
            "hunyuan-dense.block_count",
            self.block_count.map(u64::from),
            HYMT2_EXPECTED_LAYERS as u64,
        )?;
        require_numeric(
            "tokenizer.ggml.tokens",
            self.token_count,
            HYMT2_EXPECTED_VOCAB_SIZE as u64,
        )?;
        Ok(())
    }
}

fn require_value(key: &str, actual: Option<&str>, expected: &str) -> Result<(), Hymt2ImportError> {
    match actual {
        None => Err(Hymt2ImportError::MetadataMissing {
            key: key.to_string(),
        }),
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => Err(Hymt2ImportError::MetadataMismatch {
            key: key.to_string(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        }),
    }
}

fn require_numeric(key: &str, actual: Option<u64>, expected: u64) -> Result<(), Hymt2ImportError> {
    match actual {
        None => Err(Hymt2ImportError::MetadataMissing {
            key: key.to_string(),
        }),
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => Err(Hymt2ImportError::MetadataMismatch {
            key: key.to_string(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        }),
    }
}

/// Reader wrapper that captures every consumed byte so the source KV and
/// tensor-info sections can be re-emitted verbatim.
struct CaptureReader<'a, R: Read> {
    inner: &'a mut R,
    captured: Vec<u8>,
    path: &'a Path,
}

impl<R: Read> CaptureReader<'_, R> {
    fn read_exact_captured(&mut self, len: usize) -> Result<&[u8], Hymt2ImportError> {
        let start = self.captured.len();
        self.captured.resize(start + len, 0);
        self.inner
            .read_exact(&mut self.captured[start..])
            .map_err(io_error(self.path))?;
        Ok(&self.captured[start..])
    }
}

fn read_u32<R: Read>(reader: &mut CaptureReader<'_, R>) -> Result<u32, Hymt2ImportError> {
    let bytes = reader.read_exact_captured(4)?;
    Ok(u32::from_le_bytes(bytes.try_into().expect("4-byte slice")))
}

fn read_u64<R: Read>(reader: &mut CaptureReader<'_, R>) -> Result<u64, Hymt2ImportError> {
    let bytes = reader.read_exact_captured(8)?;
    Ok(u64::from_le_bytes(bytes.try_into().expect("8-byte slice")))
}

fn checked_len(len: u64) -> Result<usize, Hymt2ImportError> {
    // 256 MiB single-value ceiling keeps a corrupt length from exhausting memory.
    const MAX_SINGLE_VALUE_LEN: u64 = 256 * 1024 * 1024;
    if len > MAX_SINGLE_VALUE_LEN {
        return Err(Hymt2ImportError::MalformedGguf {
            reason: format!("metadata value length {len} exceeds sanity ceiling"),
        });
    }
    Ok(len as usize)
}

fn read_gguf_string<R: Read>(
    reader: &mut CaptureReader<'_, R>,
) -> Result<String, Hymt2ImportError> {
    let len = checked_len(read_u64(reader)?)?;
    let bytes = reader.read_exact_captured(len)?;
    String::from_utf8(bytes.to_vec()).map_err(|_| Hymt2ImportError::MalformedGguf {
        reason: "metadata string is not valid UTF-8".to_string(),
    })
}

fn scalar_type_size(value_type: u32) -> Option<u64> {
    match value_type {
        GGUF_TYPE_UINT8 | GGUF_TYPE_INT8 | GGUF_TYPE_BOOL => Some(1),
        GGUF_TYPE_UINT16 | GGUF_TYPE_INT16 => Some(2),
        GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => Some(4),
        GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => Some(8),
        _ => None,
    }
}

fn skip_captured<R: Read>(
    reader: &mut CaptureReader<'_, R>,
    len: u64,
) -> Result<(), Hymt2ImportError> {
    const CHUNK: u64 = 4 * 1024 * 1024;
    let mut remaining = len;
    while remaining > 0 {
        let step = remaining.min(CHUNK) as usize;
        reader.read_exact_captured(step)?;
        remaining -= step as u64;
    }
    Ok(())
}

fn read_gguf_value<R: Read>(
    reader: &mut CaptureReader<'_, R>,
    value_type: u32,
) -> Result<ObservedValue, Hymt2ImportError> {
    if value_type == GGUF_TYPE_STRING {
        return Ok(ObservedValue::String(read_gguf_string(reader)?));
    }
    if value_type == GGUF_TYPE_UINT32 {
        return Ok(ObservedValue::U32(read_u32(reader)?));
    }
    if value_type == GGUF_TYPE_ARRAY {
        let element_type = read_u32(reader)?;
        let count = read_u64(reader)?;
        if element_type == GGUF_TYPE_STRING {
            for _ in 0..count {
                let len = checked_len(read_u64(reader)?)?;
                skip_captured(reader, len as u64)?;
            }
        } else if let Some(size) = scalar_type_size(element_type) {
            let total = count
                .checked_mul(size)
                .ok_or_else(|| Hymt2ImportError::MalformedGguf {
                    reason: "array byte length overflow".to_string(),
                })?;
            skip_captured(reader, total)?;
        } else {
            return Err(Hymt2ImportError::MalformedGguf {
                reason: format!("unsupported array element type {element_type}"),
            });
        }
        return Ok(ObservedValue::ArrayLen(count));
    }
    if let Some(size) = scalar_type_size(value_type) {
        skip_captured(reader, size)?;
        return Ok(ObservedValue::Other);
    }
    Err(Hymt2ImportError::MalformedGguf {
        reason: format!("unsupported metadata value type {value_type}"),
    })
}

fn write_gguf_string(buffer: &mut Vec<u8>, text: &str) {
    buffer.extend_from_slice(&(text.len() as u64).to_le_bytes());
    buffer.extend_from_slice(text.as_bytes());
}

fn read_exact_vec<R: Read>(
    reader: &mut R,
    len: usize,
    path: &Path,
) -> Result<Vec<u8>, Hymt2ImportError> {
    let mut buffer = vec![0u8; len];
    reader.read_exact(&mut buffer).map_err(io_error(path))?;
    Ok(buffer)
}

fn skip_bytes<R: Read>(reader: &mut R, len: u64, path: &Path) -> Result<(), Hymt2ImportError> {
    let mut remaining = len;
    let mut buffer = [0u8; 4096];
    while remaining > 0 {
        let step = remaining.min(buffer.len() as u64) as usize;
        reader
            .read_exact(&mut buffer[..step])
            .map_err(io_error(path))?;
        remaining -= step as u64;
    }
    Ok(())
}

fn write_output_pack<R: Read>(
    request: &Hymt2ImportRequest,
    tensor_count: u64,
    kv_count: u64,
    appended_bytes: &[u8],
    captured: &[u8],
    padding: u64,
    tensor_data: &mut R,
) -> Result<(), Hymt2ImportError> {
    let output_path = &request.output_pack;
    let output_file = File::options()
        .write(true)
        .create_new(true)
        .open(output_path)
        .map_err(io_error(output_path))?;
    let mut writer = BufWriter::new(output_file);
    let write_result = (|| -> std::io::Result<()> {
        writer.write_all(&GGUF_MAGIC)?;
        writer.write_all(&GGUF_SUPPORTED_VERSION.to_le_bytes())?;
        writer.write_all(&tensor_count.to_le_bytes())?;
        writer.write_all(&kv_count.to_le_bytes())?;
        writer.write_all(appended_bytes)?;
        writer.write_all(captured)?;
        writer.write_all(&vec![0u8; padding as usize])?;
        std::io::copy(tensor_data, &mut writer)?;
        writer.flush()?;
        writer.into_inner()?.sync_all()
    })();
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(output_path);
        return Err(Hymt2ImportError::Io {
            path: output_path.to_path_buf(),
            source: error,
        });
    }
    Ok(())
}

fn file_sha256(path: &Path) -> Result<String, Hymt2ImportError> {
    let mut file = File::open(path).map_err(io_error(path))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let bytes_read = file.read(&mut buffer).map_err(io_error(path))?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_kv_string(buffer: &mut Vec<u8>, key: &str, value: &str) {
        write_gguf_string(buffer, key);
        buffer.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        write_gguf_string(buffer, value);
    }

    fn push_kv_u32(buffer: &mut Vec<u8>, key: &str, value: u32) {
        write_gguf_string(buffer, key);
        buffer.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        buffer.extend_from_slice(&value.to_le_bytes());
    }

    fn push_kv_f32(buffer: &mut Vec<u8>, key: &str, value: f32) {
        write_gguf_string(buffer, key);
        buffer.extend_from_slice(&GGUF_TYPE_FLOAT32.to_le_bytes());
        buffer.extend_from_slice(&value.to_le_bytes());
    }

    fn push_kv_string_array(buffer: &mut Vec<u8>, key: &str, count: usize) {
        write_gguf_string(buffer, key);
        buffer.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        buffer.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        buffer.extend_from_slice(&(count as u64).to_le_bytes());
        for _ in 0..count {
            write_gguf_string(buffer, "x");
        }
    }

    struct SyntheticOverrides {
        architecture: &'static str,
        file_type: u32,
        block_count: u32,
        token_count: usize,
        extra_openasr_key: bool,
    }

    impl Default for SyntheticOverrides {
        fn default() -> Self {
            Self {
                architecture: HUNYUAN_DENSE_ARCHITECTURE,
                file_type: HYMT2_EXPECTED_GENERAL_FILE_TYPE,
                block_count: HYMT2_EXPECTED_LAYERS as u32,
                token_count: HYMT2_EXPECTED_VOCAB_SIZE,
                extra_openasr_key: false,
            }
        }
    }

    const SYNTHETIC_TENSOR_DATA: &[u8] = &[0xAB; 64];

    fn synthetic_gguf(overrides: &SyntheticOverrides) -> Vec<u8> {
        let mut kv = Vec::new();
        push_kv_string(&mut kv, GENERAL_ARCHITECTURE_KEY, overrides.architecture);
        push_kv_u32(&mut kv, GENERAL_FILE_TYPE_KEY, overrides.file_type);
        push_kv_u32(&mut kv, "hunyuan-dense.block_count", overrides.block_count);
        push_kv_f32(&mut kv, "hunyuan-dense.rope.freq_base", 11_158_840.0);
        push_kv_string(&mut kv, "tokenizer.ggml.model", "gpt2");
        push_kv_string(&mut kv, "tokenizer.ggml.pre", "hunyuan-dense");
        push_kv_string_array(&mut kv, "tokenizer.ggml.tokens", overrides.token_count);
        let mut kv_count = 7u64;
        if overrides.extra_openasr_key {
            push_kv_string(&mut kv, "openasr.model.kind", "translation-model");
            kv_count += 1;
        }

        let mut tensor_info = Vec::new();
        write_gguf_string(&mut tensor_info, "token_embd.weight");
        tensor_info.extend_from_slice(&1u32.to_le_bytes());
        tensor_info.extend_from_slice(&16u64.to_le_bytes());
        tensor_info.extend_from_slice(&0i32.to_le_bytes()); // GGML_TYPE_F32
        tensor_info.extend_from_slice(&0u64.to_le_bytes());

        let mut file = Vec::new();
        file.extend_from_slice(&GGUF_MAGIC);
        file.extend_from_slice(&GGUF_SUPPORTED_VERSION.to_le_bytes());
        file.extend_from_slice(&1u64.to_le_bytes());
        file.extend_from_slice(&kv_count.to_le_bytes());
        file.extend_from_slice(&kv);
        file.extend_from_slice(&tensor_info);
        let padding = file.len().next_multiple_of(GGUF_DEFAULT_ALIGNMENT as usize) - file.len();
        file.extend(std::iter::repeat_n(0u8, padding));
        file.extend_from_slice(SYNTHETIC_TENSOR_DATA);
        file
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn request_for(source: &Path, output: &Path, source_bytes: &[u8]) -> Hymt2ImportRequest {
        Hymt2ImportRequest {
            source_gguf: source.to_path_buf(),
            output_pack: output.to_path_buf(),
            model_id: "hymt2-1.8b".to_string(),
            quantization: "q4_k_m".to_string(),
            license_text: "Apache License\nVersion 2.0".to_string(),
            notice_text: format!(
                "OpenASR repackaging notice\nbase {HYMT2_UPSTREAM_BASE_REVISION}\ngguf {HYMT2_UPSTREAM_GGUF_REVISION}\n"
            ),
            expected_source_sha256: sha256_hex(source_bytes),
        }
    }

    fn run_import(
        overrides: &SyntheticOverrides,
        mutate: impl FnOnce(&mut Hymt2ImportRequest),
    ) -> (
        tempfile::TempDir,
        Result<Hymt2ImportResult, Hymt2ImportError>,
    ) {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("source.gguf");
        let output = dir.path().join("hymt2-1.8b-q4_k_m.oasr");
        let bytes = synthetic_gguf(overrides);
        std::fs::write(&source, &bytes).expect("write synthetic gguf");
        let mut request = request_for(&source, &output, &bytes);
        mutate(&mut request);
        let result = import_hymt2_gguf_to_runtime_pack(&request);
        (dir, result)
    }

    #[test]
    fn import_preserves_tensor_data_and_prepends_openasr_metadata() {
        let (dir, result) = run_import(&SyntheticOverrides::default(), |_| {});
        let result = result.expect("import succeeds");
        assert_eq!(result.tensor_count, 1);
        assert_eq!(result.appended_metadata_entries, 16);

        let output_bytes =
            std::fs::read(dir.path().join("hymt2-1.8b-q4_k_m.oasr")).expect("read output");
        assert_eq!(&output_bytes[0..4], &GGUF_MAGIC);
        let kv_count = u64::from_le_bytes(output_bytes[16..24].try_into().expect("kv count"));
        assert_eq!(kv_count, 7 + 16);
        // Tensor data preserved byte-for-byte at the aligned tail.
        assert_eq!(
            &output_bytes[output_bytes.len() - SYNTHETIC_TENSOR_DATA.len()..],
            SYNTHETIC_TENSOR_DATA
        );
        assert_eq!(
            (output_bytes.len() - SYNTHETIC_TENSOR_DATA.len()) % GGUF_DEFAULT_ALIGNMENT as usize,
            0
        );
        // Publish preflight markers must appear in the leading header window.
        let head = &output_bytes[..output_bytes.len().min(64 * 1024)];
        for marker in [
            b"openasr.package.version".as_slice(),
            b"openasr.model.kind".as_slice(),
            b"translation-model",
            b"openasr.translation.source_langs",
            b"openasr.translation.target_langs",
            b"openasr.upstream.base_revision",
            HYMT2_UPSTREAM_BASE_REVISION.as_bytes(),
            b"openasr.upstream.gguf_revision",
            HYMT2_UPSTREAM_GGUF_REVISION.as_bytes(),
            b"openasr.license.files",
            b"LICENSE.txt",
            b"NOTICE.openasr.txt",
            b"Apache License",
        ] {
            assert!(
                head.windows(marker.len()).any(|window| window == marker),
                "marker missing from header window: {}",
                String::from_utf8_lossy(marker)
            );
        }
    }

    /// Regression test for the shipped pack that failed `openasr pull` with
    /// "missing required metadata 'openasr.package.version'": every importer
    /// output must pass the generic pull GGUF preflight gate.
    #[test]
    fn import_output_passes_generic_pull_gguf_preflight() {
        let (dir, result) = run_import(&SyntheticOverrides::default(), |_| {});
        let result = result.expect("import succeeds");
        crate::pull::preflight_gguf_package_contract(&result.output_path)
            .expect("imported pack must pass the generic pull GGUF preflight");
        drop(dir);
    }

    /// Full pull-time preflight (GGUF contract + runtime-source validation +
    /// translation runtime probe) against the real published/built pack. The
    /// synthetic fixture cannot satisfy the Hy-MT2 tensor contract, so this is
    /// the real-pack gate the release pipeline must run before upload.
    #[test]
    #[ignore = "manual real-pack gate: set OPENASR_HYMT2_REAL_PACK to the built hymt2-1.8b-q4_k_m.oasr"]
    fn hymt2_real_pack_passes_full_pull_preflight() {
        let path = std::env::var_os("OPENASR_HYMT2_REAL_PACK")
            .map(PathBuf::from)
            .expect("set OPENASR_HYMT2_REAL_PACK to the built .oasr pack");
        crate::pull::preflight_model_pack_for_install(&path)
            .expect("built Hy-MT2 pack must pass the full pull preflight");
    }

    #[test]
    fn import_rejects_source_sha_mismatch() {
        let (_dir, result) = run_import(&SyntheticOverrides::default(), |request| {
            request.expected_source_sha256 = "0".repeat(64);
        });
        assert!(matches!(
            result,
            Err(Hymt2ImportError::SourceShaMismatch { .. })
        ));
    }

    #[test]
    fn import_rejects_wrong_architecture() {
        let overrides = SyntheticOverrides {
            architecture: "qwen3-asr",
            ..SyntheticOverrides::default()
        };
        let (_dir, result) = run_import(&overrides, |_| {});
        assert!(
            matches!(result, Err(Hymt2ImportError::MetadataMismatch { key, .. }) if key == GENERAL_ARCHITECTURE_KEY)
        );
    }

    #[test]
    fn import_rejects_non_q4_k_m_file_type() {
        let overrides = SyntheticOverrides {
            file_type: 7,
            ..SyntheticOverrides::default()
        };
        let (_dir, result) = run_import(&overrides, |_| {});
        assert!(
            matches!(result, Err(Hymt2ImportError::MetadataMismatch { key, .. }) if key == GENERAL_FILE_TYPE_KEY)
        );
    }

    #[test]
    fn import_rejects_unexpected_vocab_size() {
        let overrides = SyntheticOverrides {
            token_count: 100,
            ..SyntheticOverrides::default()
        };
        let (_dir, result) = run_import(&overrides, |_| {});
        assert!(
            matches!(result, Err(Hymt2ImportError::MetadataMismatch { key, .. }) if key == "tokenizer.ggml.tokens")
        );
    }

    #[test]
    fn import_rejects_already_packaged_source() {
        let overrides = SyntheticOverrides {
            extra_openasr_key: true,
            ..SyntheticOverrides::default()
        };
        let (_dir, result) = run_import(&overrides, |_| {});
        assert!(matches!(
            result,
            Err(Hymt2ImportError::ReservedMetadataPresent { .. })
        ));
    }

    #[test]
    fn import_rejects_notice_without_pinned_revisions() {
        let (_dir, result) = run_import(&SyntheticOverrides::default(), |request| {
            request.notice_text = "stale notice".to_string();
        });
        assert!(matches!(
            result,
            Err(Hymt2ImportError::NoticeMissingRevision { .. })
        ));
    }

    #[test]
    fn import_rejects_existing_output() {
        let (_dir, result) = run_import(&SyntheticOverrides::default(), |request| {
            std::fs::write(&request.output_pack, b"existing").expect("write existing output");
        });
        assert!(matches!(result, Err(Hymt2ImportError::OutputExists { .. })));
    }

    #[test]
    fn import_rejects_non_oasr_output_suffix() {
        let (_dir, result) = run_import(&SyntheticOverrides::default(), |request| {
            request.output_pack = request.output_pack.with_extension("gguf");
        });
        assert!(matches!(
            result,
            Err(Hymt2ImportError::OutputNotOasr { .. })
        ));
    }
}
