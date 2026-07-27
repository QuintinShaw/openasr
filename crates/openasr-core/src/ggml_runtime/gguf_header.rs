//! Pure-Rust reader for the GGUF *header* (metadata KV table + tensor index)
//! over a borrowed byte window.
//!
//! The C ggml parser (`read_gguf_metadata` / `read_gguf_tensor_index`) needs a
//! seekable complete file. Auditing a pack that only exists behind a read-only
//! HTTP endpoint (a published model repo) instead fetches the leading bytes
//! with a `Range` request: the entire GGUF header -- magic, metadata, and the
//! full tensor table -- sits in a file's first kilobytes-to-megabytes, before
//! any tensor data. This reader parses exactly that prefix with zero copies of
//! tensor data, and reports [`GgufHeaderError::Truncated`] when the window
//! ends before the header does so the caller can widen the range and retry.
//!
//! Only the header surface the quant-floor audit needs is materialized:
//! string-valued metadata (all other value types are skipped in place) and the
//! tensor table's name/dims/type triples. Tensor data offsets are recorded but
//! never read.

use std::collections::BTreeMap;

use thiserror::Error;

const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// Sanity ceiling for one GGUF string (key, name, or string value): real
/// header strings are bytes-to-kilobytes, so anything larger means the parse
/// lost framing on corrupt/partial input rather than a genuine four-megabyte
/// tensor name.
const MAX_GGUF_STRING_BYTES: u64 = 4 * 1024 * 1024;

/// GGUF metadata value type tags (the stable ggml `enum gguf_type` wire
/// values). Every tag must be skippable to keep the parser framed over the
/// whole header, even for value kinds this reader does not materialize.
const GGUF_TYPE_U8: u32 = 0;
const GGUF_TYPE_I8: u32 = 1;
const GGUF_TYPE_U16: u32 = 2;
const GGUF_TYPE_I16: u32 = 3;
const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_I32: u32 = 5;
const GGUF_TYPE_F32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_U64: u32 = 10;
const GGUF_TYPE_I64: u32 = 11;
const GGUF_TYPE_F64: u32 = 12;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GgufHeaderError {
    #[error("not a GGUF container: bad magic bytes {magic:?}")]
    BadMagic { magic: [u8; 4] },
    #[error("unsupported GGUF version {version}; this reader understands versions 2 and 3")]
    UnsupportedVersion { version: u32 },
    #[error("GGUF header is internally inconsistent: {reason}")]
    InvalidEncoding { reason: String },
    #[error(
        "the byte window ends inside the GGUF header: parsed {parsed_tensors} of {tensor_count} tensor entries; widen the range and retry"
    )]
    Truncated {
        parsed_tensors: u64,
        tensor_count: u64,
    },
}

/// One tensor entry from the GGUF tensor index (no tensor data).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufHeaderTensor {
    pub name: String,
    pub dims: Vec<u64>,
    /// Raw `ggml_type` enum value (see `models::pack_quant_audit` for the
    /// quantization-tier interpretation).
    pub ggml_type: u32,
    /// Byte offset of the tensor's data relative to the start of the data
    /// section (not the file). Recorded for completeness; the header audit
    /// never reads tensor data.
    pub data_offset: u64,
}

/// The parsed header prefix of a GGUF file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufHeaderView {
    pub version: u32,
    pub tensor_count: u64,
    /// String-valued metadata only; non-string values are skipped during
    /// parsing (the audit surface needs `general.architecture`,
    /// `openasr.model.architecture`, and friends).
    pub string_metadata: BTreeMap<String, String>,
    pub tensors: Vec<GgufHeaderTensor>,
    /// Total byte length of the parsed header (end of the tensor index,
    /// before alignment padding / tensor data).
    pub header_len: usize,
}

impl GgufHeaderView {
    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        self.string_metadata.get(key).map(String::as_str)
    }
}

struct Window<'a> {
    bytes: &'a [u8],
    pos: usize,
    /// Declared tensor count from the header; 0 until that field parses.
    tensor_count: u64,
    /// Tensor entries fully parsed so far. Together with `tensor_count` this
    /// is the progress a [`GgufHeaderError::Truncated`] reports, so every
    /// window-exhaustion path (KV table included) gives the caller the same
    /// widen-and-retry signal with accurate progress.
    parsed_tensors: u64,
}

impl<'a> Window<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            tensor_count: 0,
            parsed_tensors: 0,
        }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], ()> {
        let end = self.pos.checked_add(count).ok_or(())?;
        let slice = self.bytes.get(self.pos..end).ok_or(())?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, ()> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ()> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, ()> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, ()> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i8(&mut self) -> Result<i8, ()> {
        Ok(self.take(1)?[0] as i8)
    }

    fn i16(&mut self) -> Result<i16, ()> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32, ()> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64, ()> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, ()> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f64(&mut self) -> Result<f64, ()> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn gguf_string(&mut self) -> Result<String, GgufHeaderError> {
        let len = self.u64().map_err(|()| self.truncated())?;
        if len > MAX_GGUF_STRING_BYTES {
            return Err(GgufHeaderError::InvalidEncoding {
                reason: format!(
                    "string length {len} exceeds the {MAX_GGUF_STRING_BYTES}-byte sanity ceiling"
                ),
            });
        }
        let bytes = self.take(len as usize).map_err(|()| self.truncated())?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufHeaderError::InvalidEncoding {
            reason: "header string is not valid UTF-8".to_string(),
        })
    }

    /// The single error for "the window ends mid-structure", on ANY path
    /// (magic, counts, KV key/value, tensor entry). Callers widen their byte
    /// range and retry on Truncated; misclassifying a mid-KV truncation as
    /// InvalidEncoding used to make the growing-window audit give up instead
    /// of fetching a larger prefix. Genuine framing errors (bad type tags,
    /// non-UTF-8 strings, out-of-range ranks) stay InvalidEncoding.
    fn truncated(&self) -> GgufHeaderError {
        GgufHeaderError::Truncated {
            parsed_tensors: self.parsed_tensors,
            tensor_count: self.tensor_count,
        }
    }

    /// Skip one metadata VALUE of the given type tag, materializing it only
    /// when it is a string.
    fn skip_value(&mut self, value_type: u32) -> Result<Option<String>, GgufHeaderError> {
        match value_type {
            GGUF_TYPE_U8 => {
                self.u8().map_err(|()| self.truncated())?;
                Ok(None)
            }
            GGUF_TYPE_I8 => {
                self.i8().map_err(|()| self.truncated())?;
                Ok(None)
            }
            GGUF_TYPE_U16 => {
                self.u16().map_err(|()| self.truncated())?;
                Ok(None)
            }
            GGUF_TYPE_I16 => {
                self.i16().map_err(|()| self.truncated())?;
                Ok(None)
            }
            GGUF_TYPE_U32 => {
                self.u32().map_err(|()| self.truncated())?;
                Ok(None)
            }
            GGUF_TYPE_I32 => {
                self.i32().map_err(|()| self.truncated())?;
                Ok(None)
            }
            GGUF_TYPE_F32 => {
                self.f32().map_err(|()| self.truncated())?;
                Ok(None)
            }
            GGUF_TYPE_BOOL => {
                let flag = self.u8().map_err(|()| self.truncated())?;
                if flag > 1 {
                    return Err(GgufHeaderError::InvalidEncoding {
                        reason: format!("bool metadata value must be 0 or 1, got {flag}"),
                    });
                }
                Ok(None)
            }
            GGUF_TYPE_STRING => Ok(Some(self.gguf_string()?)),
            GGUF_TYPE_ARRAY => {
                let element_type = self.u32().map_err(|()| self.truncated())?;
                let count = self.u64().map_err(|()| self.truncated())?;
                for _ in 0..count {
                    self.skip_value(element_type)?;
                }
                Ok(None)
            }
            GGUF_TYPE_U64 => {
                self.u64().map_err(|()| self.truncated())?;
                Ok(None)
            }
            GGUF_TYPE_I64 => {
                self.i64().map_err(|()| self.truncated())?;
                Ok(None)
            }
            GGUF_TYPE_F64 => {
                self.f64().map_err(|()| self.truncated())?;
                Ok(None)
            }
            unknown => Err(GgufHeaderError::InvalidEncoding {
                reason: format!("unknown GGUF metadata value type tag {unknown}"),
            }),
        }
    }
}

/// Parse the GGUF header (metadata + complete tensor index) out of a leading
/// byte window of a GGUF file.
///
/// Fails with [`GgufHeaderError::Truncated`] when the window ends before the
/// tensor index does -- the caller should widen its range and retry.
pub fn parse_gguf_header(bytes: &[u8]) -> Result<GgufHeaderView, GgufHeaderError> {
    let mut window = Window::new(bytes);

    let magic: [u8; 4] = window
        .take(4)
        .map_err(|()| window.truncated())?
        .try_into()
        .unwrap();
    if magic != GGUF_MAGIC {
        return Err(GgufHeaderError::BadMagic { magic });
    }

    let version = window.u32().map_err(|()| window.truncated())?;
    if version != 2 && version != 3 {
        return Err(GgufHeaderError::UnsupportedVersion { version });
    }

    let tensor_count = window.u64().map_err(|()| window.truncated())?;
    window.tensor_count = tensor_count;
    let kv_count = window.u64().map_err(|()| window.truncated())?;

    let mut string_metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = window.gguf_string()?;
        let value_type = window.u32().map_err(|()| window.truncated())?;
        if let Some(value) = window.skip_value(value_type)? {
            string_metadata.insert(key, value);
        }
    }

    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = window.gguf_string()?;
        let rank = window.u32().map_err(|()| window.truncated())?;
        if !(1..=4).contains(&rank) {
            return Err(GgufHeaderError::InvalidEncoding {
                reason: format!("tensor '{name}' has unsupported rank {rank}"),
            });
        }
        let mut dims = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            dims.push(window.u64().map_err(|()| window.truncated())?);
        }
        let ggml_type = window.u32().map_err(|()| window.truncated())?;
        let data_offset = window.u64().map_err(|()| window.truncated())?;
        tensors.push(GgufHeaderTensor {
            name,
            dims,
            ggml_type,
            data_offset,
        });
        window.parsed_tensors = tensors.len() as u64;
    }

    Ok(GgufHeaderView {
        version,
        tensor_count,
        string_metadata,
        tensors,
        header_len: window.pos,
    })
}

#[cfg(test)]
mod tests {
    use super::{GgufHeaderError, GgufHeaderView, parse_gguf_header};

    /// Hand-assembled GGUF v3 header: two string KVs, one u32 KV, two tensor
    /// entries. Doubles as the format spec for the reader: every byte is
    /// accounted for here.
    fn fixture_header() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes()); // version
        bytes.extend_from_slice(&2_u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&3_u64.to_le_bytes()); // kv_count

        // KV 1: general.architecture = "qwen3-asr-encoder-decoder" (string=8)
        put_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        put_string(&mut bytes, "qwen3-asr-encoder-decoder");

        // KV 2: openasr.package.version = "1" (string=8)
        put_string(&mut bytes, "openasr.package.version");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        put_string(&mut bytes, "1");

        // KV 3: dummy.block_count = 4 (u32=4)
        put_string(&mut bytes, "dummy.block_count");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&4_u32.to_le_bytes());

        // Tensor 1: audio.blk.0.attn_q.weight, [4096, 4096], Q8_0 (type 8)
        put_string(&mut bytes, "audio.blk.0.attn_q.weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&4096_u64.to_le_bytes());
        bytes.extend_from_slice(&4096_u64.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());

        // Tensor 2: blk.0.ffn_gate.weight, [11008, 4096], Q4_K (type 12)
        put_string(&mut bytes, "blk.0.ffn_gate.weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&11008_u64.to_le_bytes());
        bytes.extend_from_slice(&4096_u64.to_le_bytes());
        bytes.extend_from_slice(&12_u32.to_le_bytes());
        bytes.extend_from_slice(&16777216_u64.to_le_bytes());

        bytes
    }

    fn put_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    #[test]
    fn parses_metadata_and_tensor_index() {
        let view = parse_gguf_header(&fixture_header()).expect("parse fixture");
        assert_eq!(view.version, 3);
        assert_eq!(view.tensor_count, 2);
        assert_eq!(
            view.metadata_string("general.architecture"),
            Some("qwen3-asr-encoder-decoder")
        );
        assert_eq!(view.metadata_string("openasr.package.version"), Some("1"));
        assert_eq!(view.metadata_string("dummy.block_count"), None); // u32, skipped
        assert_eq!(view.tensors.len(), 2);
        assert_eq!(view.tensors[0].name, "audio.blk.0.attn_q.weight");
        assert_eq!(view.tensors[0].dims, vec![4096, 4096]);
        assert_eq!(view.tensors[0].ggml_type, 8);
        assert_eq!(view.tensors[1].name, "blk.0.ffn_gate.weight");
        assert_eq!(view.tensors[1].ggml_type, 12);
        assert_eq!(view.header_len, fixture_header().len());
    }

    #[test]
    fn truncated_window_reports_progress() {
        let full = fixture_header();
        // Cut inside the second tensor entry: the first tensor parsed fully.
        let cut = full.len() - 10;
        let error = parse_gguf_header(&full[..cut]).expect_err("must report truncation");
        assert_eq!(
            error,
            GgufHeaderError::Truncated {
                parsed_tensors: 1,
                tensor_count: 2,
            }
        );
    }

    #[test]
    fn kv_table_truncation_is_truncated_not_invalid_encoding() {
        // Fixture layout (byte ranges): preamble [0,24); KV1 key string
        // [32,52), value string [64,89); KV2 value string [132,133); KV3 u32
        // value [162,166); KV table ends at 166; tensors [166,292).
        let full = fixture_header();
        let expected = GgufHeaderError::Truncated {
            parsed_tensors: 0,
            tensor_count: 2,
        };

        // The old code mapped mid-KV window exhaustion to InvalidEncoding,
        // so the growing-window audit gave up with an error instead of
        // fetching a wider prefix and retrying. Each cut below lands inside
        // a different KV-table structure: a key string, a string value
        // (skip_value -> gguf_string), and a fixed-width scalar value
        // (skip_value's u32 arm).
        for cut in [40usize, 130, 164] {
            let error = parse_gguf_header(&full[..cut]).expect_err("must report truncation");
            assert_eq!(error, expected, "cut at {cut}");
        }
    }

    #[test]
    fn rejects_bad_magic_and_version() {
        let mut bytes = fixture_header();
        bytes[0] = b'X';
        assert!(matches!(
            parse_gguf_header(&bytes),
            Err(GgufHeaderError::BadMagic { .. })
        ));

        let mut bytes = fixture_header();
        bytes[4..8].copy_from_slice(&9_u32.to_le_bytes());
        assert!(matches!(
            parse_gguf_header(&bytes),
            Err(GgufHeaderError::UnsupportedVersion { version: 9 })
        ));
    }

    #[test]
    fn parses_a_real_pack_header_written_by_the_c_writer() {
        // Round-trip against the production writer: write_gguf_file_v0 (C
        // gguf_write_to_file) emits the bytes this pure-Rust reader consumes.
        use crate::ggml_runtime::{GgufWriteTensor, GgufWriteTensorType, GgufWriteValue};
        use std::collections::BTreeMap;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("header-roundtrip.oasr");
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufWriteValue::String("qwen3-asr-encoder-decoder".to_string()),
        );
        metadata.insert(
            "probe.array".to_string(),
            GgufWriteValue::U32Array(vec![1, 2, 3]),
        );
        let tensors = vec![GgufWriteTensor {
            name: "audio.weight".to_string(),
            dims: vec![64, 32],
            tensor_type: GgufWriteTensorType::F32,
            data: vec![0u8; 64 * 32 * 4],
        }];
        crate::ggml_runtime::write_gguf_file_v0(&path, &metadata, &tensors).expect("write pack");

        // Parse from ONLY the leading 4 KiB -- no full-file access.
        use std::io::Read;
        let mut window = vec![0u8; 4096];
        let mut file = std::fs::File::open(&path).expect("open pack");
        let read = file.read(&mut window).expect("read prefix");
        window.truncate(read);

        let view: GgufHeaderView = parse_gguf_header(&window).expect("parse prefix");
        assert_eq!(view.tensor_count, 1);
        assert_eq!(view.tensors[0].name, "audio.weight");
        assert_eq!(
            view.metadata_string("general.architecture"),
            Some("qwen3-asr-encoder-decoder")
        );
    }
}
