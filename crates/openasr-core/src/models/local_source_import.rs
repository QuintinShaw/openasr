use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde::Deserialize;
use thiserror::Error;

use crate::nn::half::{f16_bits_to_f32, f32_to_f16_bits};

#[derive(Debug, Error)]
pub enum LocalSourceImportError {
    #[error("could not read model source file '{path}': {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not parse model source artifact '{path}': {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("{0}")]
    Validate(String),
}

pub(crate) fn validate_error(message: impl Into<String>) -> LocalSourceImportError {
    LocalSourceImportError::Validate(message.into())
}

/// Enforce the user-facing runtime-pack extension contract on a converter's
/// output path. Every `convert_local_*_to_runtime_pack` entry point calls this so
/// a direct library caller is held to the same `.oasr`-only rule as the CLI — the
/// on-disk container stays GGUF-structured, but it is only ever *produced* as
/// `.oasr`. Shares [`crate::has_openasr_runtime_pack_extension`] as the single
/// source of truth (see also the CLI's run/import path validation).
pub(crate) fn validate_output_pack_extension(
    output_root: &Path,
) -> Result<(), LocalSourceImportError> {
    if crate::has_openasr_runtime_pack_extension(output_root) {
        return Ok(());
    }
    Err(validate_error(format!(
        "local-source converter output '{}' must end with .oasr (OpenASR native runtime pack)",
        output_root.display()
    )))
}

pub(crate) fn read_source_json_file<T: for<'de> Deserialize<'de>>(
    root: &Path,
    relative_path: &str,
) -> Result<T, LocalSourceImportError> {
    let path = root.join(relative_path);
    let bytes = read_source_file_bytes(root, relative_path)?;
    serde_json::from_slice(&bytes).map_err(|source| LocalSourceImportError::Parse { path, source })
}

pub(crate) fn read_source_file_bytes(
    root: &Path,
    relative_path: &str,
) -> Result<Vec<u8>, LocalSourceImportError> {
    let path = root.join(relative_path);
    std::fs::read(&path).map_err(|source| LocalSourceImportError::Read { path, source })
}

// --- Shared GPT-2 BPE source loading ----------------------------------------
//
// HF-style GPT-2 BPE tokenizer sources (`vocab.json` token->id map,
// `merges.txt` merge rules, `tokenizer_config.json` `added_tokens_decoder` for
// the special tokens that live past the base vocab) are shaped identically
// across the families that bake this tokenizer class into a pack (qwen,
// funasr-nano). The loaders live here -- the shared import layer -- instead of
// per-family copies, so the dense-array/padding semantics cannot drift between
// families.

const SOURCE_VOCAB_JSON: &str = "vocab.json";
const SOURCE_MERGES_TXT: &str = "merges.txt";
const SOURCE_TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";

#[derive(Debug, Deserialize)]
struct Gpt2BpeTokenizerConfigJson {
    #[serde(default)]
    added_tokens_decoder: BTreeMap<String, Gpt2BpeAddedTokenEntry>,
}

#[derive(Debug, Deserialize)]
struct Gpt2BpeAddedTokenEntry {
    content: String,
}

/// Load the dense token array for a GPT-2 BPE pack from an HF-style tokenizer
/// directory: `vocab.json` (token -> id) inverted into id order, with
/// `tokenizer_config.json`'s `added_tokens_decoder` overlaying the special
/// tokens that sit past the base vocab. Slots still empty after both sources
/// are filled with `<unused_{id}>` placeholders so the array is dense and
/// unambiguous -- the runtime builds a reverse token->id map from it, so two
/// slots sharing the empty string would collide.
pub(crate) fn load_gpt2_bpe_vocab_tokens(
    source_root: &Path,
    family: &str,
) -> Result<Vec<String>, LocalSourceImportError> {
    let vocab: BTreeMap<String, usize> = read_source_json_file(source_root, SOURCE_VOCAB_JSON)?;
    if vocab.is_empty() {
        return Err(validate_error(format!(
            "{family} vocab.json cannot be empty"
        )));
    }
    let max_id = vocab.values().copied().max().ok_or_else(|| {
        validate_error(format!("{family} vocab.json cannot determine max token id"))
    })?;
    let mut tokens = vec![String::new(); max_id + 1];
    for (token, token_id) in vocab {
        tokens[token_id] = token;
    }

    let config_path = source_root.join(SOURCE_TOKENIZER_CONFIG_JSON);
    if config_path.exists() {
        let config: Gpt2BpeTokenizerConfigJson =
            read_source_json_file(source_root, SOURCE_TOKENIZER_CONFIG_JSON)?;
        for (token_id_str, entry) in config.added_tokens_decoder {
            let token_id = token_id_str.parse::<usize>().map_err(|error| {
                validate_error(format!(
                    "invalid {family} added token id '{}' in {}: {error}",
                    token_id_str,
                    config_path.display()
                ))
            })?;
            if token_id < tokens.len() {
                tokens[token_id] = entry.content;
            } else {
                tokens.resize(token_id + 1, String::new());
                tokens[token_id] = entry.content;
            }
        }
    }

    fill_unused_gpt2_bpe_token_slots(&mut tokens);
    Ok(tokens)
}

/// Pad a dense token array to a declared vocab size (the padded embedding
/// width, which can exceed the real token count) with `<unused_{id}>`
/// placeholders. Fails closed when the source already carries more tokens than
/// the declared size -- that is a source/declared-vocab mismatch, never a
/// truncation case.
pub(crate) fn pad_gpt2_bpe_vocab_tokens(
    tokens: &mut Vec<String>,
    vocab_size: usize,
    family: &str,
) -> Result<(), LocalSourceImportError> {
    if tokens.len() > vocab_size {
        return Err(validate_error(format!(
            "{family} tokenizer source carries {} tokens, exceeding the declared vocab size {vocab_size}",
            tokens.len()
        )));
    }
    tokens.resize(vocab_size, String::new());
    fill_unused_gpt2_bpe_token_slots(tokens);
    Ok(())
}

fn fill_unused_gpt2_bpe_token_slots(tokens: &mut [String]) {
    for (index, token) in tokens.iter_mut().enumerate() {
        if token.is_empty() {
            *token = format!("<unused_{index}>");
        }
    }
}

/// Load the ordered GPT-2 BPE merge list from `merges.txt` (comment and blank
/// lines dropped). An absent file yields an empty list, matching families
/// whose checkpoints ship without one.
pub(crate) fn load_gpt2_bpe_merges(
    source_root: &Path,
    family: &str,
) -> Result<Vec<String>, LocalSourceImportError> {
    let path = source_root.join(SOURCE_MERGES_TXT);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let bytes = read_source_file_bytes(source_root, SOURCE_MERGES_TXT)?;
    let text = std::str::from_utf8(&bytes).map_err(|error| {
        validate_error(format!(
            "{family} merges.txt is not valid UTF-8 ({}): {error}",
            path.display()
        ))
    })?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToOwned::to_owned)
        .collect())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SafetensorsTensorHeader {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SafetensorsHeader {
    pub header_length_bytes: u64,
    pub data_length_bytes: u64,
    pub metadata: BTreeMap<String, String>,
    pub tensors: Vec<SafetensorsTensorHeader>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSafetensorsTensorHeader {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// Upper bound on the safetensors JSON header size, checked *before* the
/// header-length prefix is ever used to size an allocation (see [`SafetensorsFile::open`]).
/// 128 MiB comfortably covers every real header in the tree (a header holds one
/// small JSON record per tensor, not tensor payloads) while bounding the
/// allocation a hostile file can force. Mirrors
/// `whisper::local_source::safetensors::SAFETENSORS_HEADER_MAX_BYTES_V0`.
pub(crate) const SAFETENSORS_HEADER_MAX_BYTES: u64 = 128 * 1024 * 1024;

pub(crate) struct SafetensorsFile {
    path: PathBuf,
    _file: File,
    mmap: Mmap,
    data_offset_bytes: usize,
    header: SafetensorsHeader,
    by_name: BTreeMap<String, SafetensorsTensorHeader>,
}

impl SafetensorsFile {
    pub(crate) fn open(path: impl AsRef<Path>) -> Result<Self, LocalSourceImportError> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|source| LocalSourceImportError::Read {
            path: path.clone(),
            source,
        })?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|source| LocalSourceImportError::Read {
            path: path.clone(),
            source,
        })?;
        let (data_offset_bytes, header) = parse_safetensors_header(&path, &mmap)?;
        let mut by_name = BTreeMap::new();
        for tensor in &header.tensors {
            by_name.insert(tensor.name.clone(), tensor.clone());
        }
        Ok(Self {
            path,
            _file: file,
            mmap,
            data_offset_bytes,
            header,
            by_name,
        })
    }

    pub(crate) fn header(&self) -> &SafetensorsHeader {
        &self.header
    }

    pub(crate) fn tensor(&self, name: &str) -> Option<&SafetensorsTensorHeader> {
        self.by_name.get(name)
    }

    pub(crate) fn tensor_data(
        &self,
        tensor: &SafetensorsTensorHeader,
    ) -> Result<&[u8], LocalSourceImportError> {
        let start = usize::try_from(tensor.data_offsets[0]).map_err(|_| {
            validate_error(format!(
                "safetensors tensor '{}' start offset does not fit usize",
                tensor.name
            ))
        })?;
        let end = usize::try_from(tensor.data_offsets[1]).map_err(|_| {
            validate_error(format!(
                "safetensors tensor '{}' end offset does not fit usize",
                tensor.name
            ))
        })?;
        let absolute_start = self
            .data_offset_bytes
            .checked_add(start)
            .ok_or_else(|| validate_error("safetensors absolute start offset overflow"))?;
        let absolute_end = self
            .data_offset_bytes
            .checked_add(end)
            .ok_or_else(|| validate_error("safetensors absolute end offset overflow"))?;
        self.mmap.get(absolute_start..absolute_end).ok_or_else(|| {
            validate_error(format!(
                "safetensors tensor '{}' data range is out of bounds in '{}'",
                tensor.name,
                self.path.display()
            ))
        })
    }
}

/// Parse and validate one safetensors header from an already-pinned byte
/// generation. The caller owns the mapping; this helper never reopens a path
/// and never allocates from the untrusted length prefix before enforcing the
/// shared header bound.
pub(crate) fn parse_safetensors_header(
    path: &Path,
    bytes: &[u8],
) -> Result<(usize, SafetensorsHeader), LocalSourceImportError> {
    let prefix = bytes.get(..8).ok_or_else(|| {
        validate_error(format!(
            "safetensors file '{}' is too short to contain its 8-byte header length",
            path.display()
        ))
    })?;
    let header_length_bytes = u64::from_le_bytes(prefix.try_into().expect("8-byte prefix"));
    if header_length_bytes > SAFETENSORS_HEADER_MAX_BYTES {
        return Err(validate_error(format!(
            "safetensors header length {header_length_bytes} exceeds max allowed {SAFETENSORS_HEADER_MAX_BYTES} bytes"
        )));
    }
    let header_section_len = 8_u64.checked_add(header_length_bytes).ok_or_else(|| {
        validate_error(format!(
            "safetensors header length {header_length_bytes} overflows file indexing bounds"
        ))
    })?;
    let total_len = u64::try_from(bytes.len())
        .map_err(|_| validate_error("safetensors file length does not fit u64"))?;
    if total_len < header_section_len {
        return Err(validate_error(format!(
            "safetensors file '{}' is smaller than its declared header ({header_section_len} bytes needed, {total_len} available)",
            path.display()
        )));
    }
    let data_offset_bytes = usize::try_from(header_section_len)
        .map_err(|_| validate_error("safetensors data offset overflowed platform usize"))?;
    let header_bytes = bytes
        .get(8..data_offset_bytes)
        .ok_or_else(|| validate_error("safetensors header slice is out of bounds"))?;
    let header_text = std::str::from_utf8(header_bytes).map_err(|error| {
        validate_error(format!(
            "safetensors header is not valid UTF-8 JSON: {error}"
        ))
    })?;
    crate::models::safetensors_json::reject_duplicate_json_keys(header_text).map_err(|error| {
        validate_error(format!(
            "safetensors header has duplicate JSON keys: {error}"
        ))
    })?;

    let raw_value: serde_json::Value =
        serde_json::from_str(header_text).map_err(|source| LocalSourceImportError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
    let raw_object = raw_value
        .as_object()
        .ok_or_else(|| validate_error("safetensors header must be a JSON object"))?;
    let data_length_bytes = total_len - header_section_len;
    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::new();
    for (name, value) in raw_object {
        if name == "__metadata__" {
            metadata = serde_json::from_value(value.clone()).map_err(|source| {
                LocalSourceImportError::Parse {
                    path: path.to_path_buf(),
                    source,
                }
            })?;
            continue;
        }
        let raw: RawSafetensorsTensorHeader =
            serde_json::from_value(value.clone()).map_err(|source| {
                LocalSourceImportError::Parse {
                    path: path.to_path_buf(),
                    source,
                }
            })?;
        if raw.dtype.trim().is_empty() {
            return Err(validate_error(format!(
                "safetensors tensor '{name}' dtype must not be empty"
            )));
        }
        let [start, end] = raw.data_offsets;
        if end < start {
            return Err(validate_error(format!(
                "safetensors tensor '{name}' has inverted offsets {:?}",
                raw.data_offsets
            )));
        }
        if end > data_length_bytes {
            return Err(validate_error(format!(
                "safetensors tensor '{name}' data_offsets end ({end}) exceeds data section length ({data_length_bytes})"
            )));
        }
        if let Some(dtype_size) = dtype_size_bytes(&raw.dtype) {
            let element_count = tensor_element_count(name, &raw.shape)?;
            let expected_bytes = (element_count as u64).checked_mul(dtype_size).ok_or_else(
                || {
                    validate_error(format!(
                        "safetensors tensor '{name}' expected byte size overflow from shape/dtype"
                    ))
                },
            )?;
            let actual_bytes = end - start;
            if actual_bytes != expected_bytes {
                return Err(validate_error(format!(
                    "safetensors tensor '{name}' byte range ({actual_bytes}) does not match expected bytes ({expected_bytes}) from dtype '{}' and shape {:?}",
                    raw.dtype, raw.shape
                )));
            }
        }
        tensors.push(SafetensorsTensorHeader {
            name: name.clone(),
            dtype: raw.dtype,
            shape: raw.shape,
            data_offsets: raw.data_offsets,
        });
    }
    if tensors.is_empty() {
        return Err(validate_error(
            "safetensors header must include at least one tensor entry",
        ));
    }
    validate_tensor_offset_ranges(&tensors, data_length_bytes)?;
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    Ok((
        data_offset_bytes,
        SafetensorsHeader {
            header_length_bytes,
            data_length_bytes,
            metadata,
            tensors,
        },
    ))
}

/// Byte size of one element for the safetensors dtypes this shared importer
/// recognizes. `None` for anything else (family-specific/exotic dtype
/// strings a given model family may pass through) -- callers must only run
/// the shape/dtype byte cross-check for a `Some` result, per the S4/S5
/// design: unknown dtypes still get the range/overflow checks, just not the
/// cross-check against a size table entry that does not exist for them.
/// Mirrors `whisper::local_source::safetensors::dtype::safetensors_dtype_size_bytes`.
fn dtype_size_bytes(dtype: &str) -> Option<u64> {
    match dtype.trim().to_ascii_uppercase().as_str() {
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" => Some(1),
        "I16" | "U16" | "F16" | "BF16" => Some(2),
        "I32" | "U32" | "F32" => Some(4),
        "I64" | "U64" | "F64" => Some(8),
        _ => None,
    }
}

/// Validate that `tensors`' `data_offsets` ranges are sorted-contiguous,
/// non-overlapping, and exactly cover a data section of `data_length_bytes`.
/// This is the safetensors on-disk contract; mmap bounds-checking on
/// individual tensor reads (see `SafetensorsFile::tensor_data`) is a lazy
/// last line of defense and does not by itself catch a header whose tensors
/// leave gaps or overlap each other. Mirrors
/// `whisper::local_source::safetensors::dtype::validate_safetensors_tensor_offset_ranges`.
fn validate_tensor_offset_ranges(
    tensors: &[SafetensorsTensorHeader],
    data_length_bytes: u64,
) -> Result<(), LocalSourceImportError> {
    let mut ranges = tensors
        .iter()
        .map(|tensor| {
            (
                tensor.data_offsets[0],
                tensor.data_offsets[1],
                tensor.name.as_str(),
            )
        })
        .collect::<Vec<_>>();
    ranges.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

    let mut expected_offset = 0_u64;
    for (start, end, name) in ranges {
        if start > expected_offset {
            return Err(validate_error(format!(
                "safetensors tensor '{name}' data_offsets start ({start}) leaves a gap before expected offset {expected_offset}"
            )));
        }
        if start < expected_offset {
            return Err(validate_error(format!(
                "safetensors tensor '{name}' data_offsets start ({start}) overlaps previous tensor range ending at {expected_offset}"
            )));
        }
        if end > expected_offset {
            expected_offset = end;
        }
    }
    if expected_offset != data_length_bytes {
        return Err(validate_error(format!(
            "safetensors tensor ranges must fully cover data section length {data_length_bytes}; covered length is {expected_offset}"
        )));
    }
    Ok(())
}

pub(crate) fn decode_safetensors_payload_as_f32(
    tensor_name: &str,
    dtype: &str,
    data: &[u8],
) -> Result<Vec<f32>, LocalSourceImportError> {
    match dtype {
        "F32" => decode_f32_payload(tensor_name, data),
        "F16" => decode_f16_payload_as_f32(tensor_name, data),
        "BF16" => decode_bf16_payload_as_f32(tensor_name, data),
        other => Err(validate_error(format!(
            "safetensors tensor '{tensor_name}' dtype '{other}' is not supported for runtime import"
        ))),
    }
}

pub(crate) fn decode_safetensors_payload_as_f16_bits(
    tensor_name: &str,
    dtype: &str,
    data: &[u8],
) -> Result<Vec<u16>, LocalSourceImportError> {
    match dtype {
        "F16" => decode_f16_payload_bits(tensor_name, data),
        "F32" | "BF16" => {
            let values = decode_safetensors_payload_as_f32(tensor_name, dtype, data)?;
            Ok(values.into_iter().map(f32_to_f16_bits).collect())
        }
        other => Err(validate_error(format!(
            "safetensors tensor '{tensor_name}' dtype '{other}' cannot be converted to f16"
        ))),
    }
}

pub(crate) fn encode_f16_bits_le(values: Vec<u16>) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(values.len() * 2);
    for value in values {
        encoded.extend_from_slice(&value.to_le_bytes());
    }
    encoded
}

pub(crate) fn tensor_element_count(
    tensor_name: &str,
    dims: &[u64],
) -> Result<usize, LocalSourceImportError> {
    let mut count = 1_u64;
    for dim in dims {
        count = count.checked_mul(*dim).ok_or_else(|| {
            validate_error(format!(
                "tensor '{tensor_name}' element-count overflow for dims {dims:?}"
            ))
        })?;
    }
    usize::try_from(count).map_err(|_| {
        validate_error(format!(
            "tensor '{tensor_name}' element-count does not fit usize for dims {dims:?}"
        ))
    })
}

fn decode_f32_payload(tensor_name: &str, data: &[u8]) -> Result<Vec<f32>, LocalSourceImportError> {
    if !data.len().is_multiple_of(4) {
        return Err(validate_error(format!(
            "safetensors tensor '{tensor_name}' f32 payload length {} is not divisible by 4",
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn decode_f16_payload_bits(
    tensor_name: &str,
    data: &[u8],
) -> Result<Vec<u16>, LocalSourceImportError> {
    if !data.len().is_multiple_of(2) {
        return Err(validate_error(format!(
            "safetensors tensor '{tensor_name}' f16 payload length {} is not divisible by 2",
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        out.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(out)
}

fn decode_f16_payload_as_f32(
    tensor_name: &str,
    data: &[u8],
) -> Result<Vec<f32>, LocalSourceImportError> {
    let bits = decode_f16_payload_bits(tensor_name, data)?;
    Ok(bits.into_iter().map(f16_bits_to_f32).collect())
}

fn decode_bf16_payload_as_f32(
    tensor_name: &str,
    data: &[u8],
) -> Result<Vec<f32>, LocalSourceImportError> {
    if !data.len().is_multiple_of(2) {
        return Err(validate_error(format!(
            "safetensors tensor '{tensor_name}' bf16 payload length {} is not divisible by 2",
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let upper = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
        out.push(f32::from_bits(upper << 16));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        LocalSourceImportError, SAFETENSORS_HEADER_MAX_BYTES, SafetensorsFile,
        validate_output_pack_extension,
    };
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;

    #[test]
    fn output_pack_extension_accepts_oasr() {
        validate_output_pack_extension(Path::new("/tmp/whisper-small.oasr"))
            .expect(".oasr output must be accepted");
        validate_output_pack_extension(Path::new("/tmp/whisper-small.OASR"))
            .expect(".oasr output is case-insensitive");
    }

    #[test]
    fn output_pack_extension_rejects_legacy_gguf() {
        // The library boundary holds direct converter callers to the same
        // `.oasr`-only contract the CLI enforces — legacy `.gguf` is rejected.
        let error = validate_output_pack_extension(Path::new("/tmp/whisper-small.gguf"))
            .expect_err(".gguf output must be rejected");
        match error {
            LocalSourceImportError::Validate(message) => {
                assert!(
                    message.contains(".oasr"),
                    "message should cite .oasr: {message}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn output_pack_extension_rejects_missing_extension() {
        validate_output_pack_extension(Path::new("/tmp/whisper-small"))
            .expect_err("extensionless output must be rejected");
    }

    // --- SafetensorsFile::open hardening (trust-boundary negative tests) ---
    //
    // These fixtures build the raw on-disk safetensors wire format by hand
    // (8-byte little-endian header length, then the header JSON bytes, then
    // the data section) rather than through `serde_json::Map`, since several
    // cases (duplicate keys, a declared length independent of what's on
    // disk) are deliberately unrepresentable through a normal `Value`.

    /// Each test gets its own `tempfile::TempDir` (auto-unique, auto-cleaned
    /// up) rather than a hand-rolled path under `env::temp_dir()` -- tests in
    /// this module run concurrently in the same process, so anything keyed
    /// only by a short human-readable label risks two tests colliding on the
    /// same on-disk fixture file.
    fn fixture_dir() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    fn fixture_path(dir: &TempDir) -> PathBuf {
        dir.path().join("model.safetensors")
    }

    fn write_raw_safetensors_file(path: &Path, header_bytes: &[u8], data_bytes: &[u8]) {
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header_bytes).unwrap();
        file.write_all(data_bytes).unwrap();
    }

    /// Writes only the 8-byte header-length prefix (no header bytes, no data
    /// bytes) so the S1 max-length check can be exercised without ever
    /// allocating or reading the (fictitious) declared header.
    fn write_header_length_prefix_only(path: &Path, declared_header_length_bytes: u64) {
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&declared_header_length_bytes.to_le_bytes())
            .unwrap();
    }

    fn expect_validate_error(
        result: Result<SafetensorsFile, LocalSourceImportError>,
        contains: &str,
    ) -> String {
        match result {
            Ok(_) => panic!("expected a Validate error containing '{contains}', got Ok"),
            Err(LocalSourceImportError::Validate(message)) => {
                assert!(
                    message.contains(contains),
                    "error message '{message}' should contain '{contains}'"
                );
                message
            }
            Err(other) => panic!("expected LocalSourceImportError::Validate, got {other:?}"),
        }
    }

    #[test]
    fn open_rejects_header_length_far_over_the_byte_cap_without_allocating() {
        // S1: a declared header length of 2^40 (1 TiB) must be rejected by the
        // byte-cap check before any allocation or further file I/O is
        // attempted -- this file on disk is 8 bytes long, so if the importer
        // ever tried to honor the declared length it would either fail a
        // later read or (pre-hardening) attempt a terabyte-sized `vec![0; _]`.
        let dir = fixture_dir();
        let path = fixture_path(&dir);
        write_header_length_prefix_only(&path, 1_u64 << 40);
        let error = expect_validate_error(SafetensorsFile::open(&path), "exceeds max allowed");
        assert!(
            error.contains(&SAFETENSORS_HEADER_MAX_BYTES.to_string()),
            "error should cite the byte cap: {error}"
        );
    }

    #[test]
    fn open_rejects_header_length_at_u64_max_without_allocating() {
        // Same as above at the absolute extreme of the prefix's range, to
        // rule out a wraparound in the bound check itself.
        let dir = fixture_dir();
        let path = fixture_path(&dir);
        write_header_length_prefix_only(&path, u64::MAX);
        expect_validate_error(SafetensorsFile::open(&path), "exceeds max allowed");
    }

    #[test]
    fn open_rejects_header_length_exceeding_actual_file_size() {
        // A header length under the byte cap but larger than the file
        // actually on disk must still fail closed (not attempt to read past
        // EOF or read garbage).
        let dir = fixture_dir();
        let path = fixture_path(&dir);
        write_header_length_prefix_only(&path, 4096);
        expect_validate_error(
            SafetensorsFile::open(&path),
            "smaller than its declared header",
        );
    }

    #[test]
    fn open_rejects_duplicate_json_object_keys() {
        // S2: serde_json keeps the *last* value for a repeated key; the
        // shared importer must reject the header outright instead of
        // silently picking one definition for tensor "w".
        let dir = fixture_dir();
        let path = fixture_path(&dir);
        let header = br#"{"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}"#;
        write_raw_safetensors_file(&path, header, &0.0f32.to_le_bytes());
        expect_validate_error(SafetensorsFile::open(&path), "duplicate JSON keys");
    }

    #[test]
    fn open_rejects_tensor_ranges_with_a_gap() {
        // S3: tensor "a" covers [0,4), tensor "b" covers [8,12) -- bytes
        // [4,8) belong to no tensor. Must fail closed rather than silently
        // ignoring the unclaimed region.
        let dir = fixture_dir();
        let path = fixture_path(&dir);
        let header = br#"{"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "b": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]}}"#;
        write_raw_safetensors_file(&path, header, &[0_u8; 12]);
        expect_validate_error(SafetensorsFile::open(&path), "leaves a gap");
    }

    #[test]
    fn open_rejects_overlapping_tensor_ranges() {
        // S3: tensor "a" covers [0,8), tensor "b" covers [4,12) -- bytes
        // [4,8) are claimed by both.
        let dir = fixture_dir();
        let path = fixture_path(&dir);
        let header = br#"{"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}, "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}}"#;
        write_raw_safetensors_file(&path, header, &[0_u8; 12]);
        expect_validate_error(
            SafetensorsFile::open(&path),
            "overlaps previous tensor range",
        );
    }

    #[test]
    fn open_rejects_tensor_ranges_that_do_not_cover_the_full_data_section() {
        // S3: a single tensor claims only the first half of the data section;
        // the remaining bytes are unaccounted for.
        let dir = fixture_dir();
        let path = fixture_path(&dir);
        let header = br#"{"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}"#;
        write_raw_safetensors_file(&path, header, &[0_u8; 16]);
        expect_validate_error(
            SafetensorsFile::open(&path),
            "must fully cover data section length",
        );
    }

    #[test]
    fn open_rejects_tensor_range_exceeding_the_data_section() {
        // A tensor's declared `data_offsets` end reaches past the file's
        // actual data section (as computed from total file size minus the
        // header), independent of the header-vs-file-size check above.
        let dir = fixture_dir();
        let path = fixture_path(&dir);
        let header = br#"{"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}"#;
        write_raw_safetensors_file(&path, header, &[0_u8; 4]);
        expect_validate_error(SafetensorsFile::open(&path), "exceeds data section length");
    }

    #[test]
    fn open_rejects_inverted_data_offsets() {
        let dir = fixture_dir();
        let path = fixture_path(&dir);
        let header = br#"{"a": {"dtype": "F32", "shape": [1], "data_offsets": [10, 4]}}"#;
        write_raw_safetensors_file(&path, header, &[0_u8; 16]);
        expect_validate_error(SafetensorsFile::open(&path), "inverted offsets");
    }

    #[test]
    fn open_rejects_shape_dtype_byte_mismatch_for_known_dtype() {
        // S4/S5: dtype F32 with shape [3] expects 12 bytes, but the declared
        // range is only 8 bytes -- a known dtype's byte size must agree with
        // shape * data_offsets width.
        let dir = fixture_dir();
        let path = fixture_path(&dir);
        let header = br#"{"a": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}"#;
        write_raw_safetensors_file(&path, header, &[0_u8; 8]);
        expect_validate_error(
            SafetensorsFile::open(&path),
            "does not match expected bytes",
        );
    }

    #[test]
    fn open_accepts_unknown_dtype_with_only_range_checks_applied() {
        // Design intent (S4/S5): a family-specific/exotic dtype string this
        // shared parser does not recognize must NOT be rejected outright --
        // it still gets range/overflow checks, just not the byte-size
        // cross-check against a size-table entry that does not exist for it.
        let dir = fixture_dir();
        let path = fixture_path(&dir);
        let header = br#"{"a": {"dtype": "MYFAM_Q4", "shape": [999], "data_offsets": [0, 5]}}"#;
        write_raw_safetensors_file(&path, header, &[0_u8; 5]);
        let file = SafetensorsFile::open(&path)
            .expect("unrecognized dtype must pass through with only range checks");
        assert_eq!(file.header().tensors.len(), 1);
        assert_eq!(file.header().tensors[0].dtype, "MYFAM_Q4");
    }

    #[test]
    fn open_rejects_overflowing_shape_element_count() {
        // A shape whose element-count product overflows u64 (here
        // [u64::MAX, 2]) must fail closed via the checked multiplication
        // rather than wrapping.
        let dir = fixture_dir();
        let path = fixture_path(&dir);
        let header = format!(
            r#"{{"a": {{"dtype": "F32", "shape": [{}, 2], "data_offsets": [0, 8]}}}}"#,
            u64::MAX
        );
        write_raw_safetensors_file(&path, header.as_bytes(), &[0_u8; 8]);
        expect_validate_error(SafetensorsFile::open(&path), "overflow");
    }

    #[test]
    fn open_rejects_non_object_header() {
        let dir = fixture_dir();
        let path = fixture_path(&dir);
        write_raw_safetensors_file(&path, b"[]", &[]);
        expect_validate_error(SafetensorsFile::open(&path), "must be a JSON object");
    }

    #[test]
    fn open_rejects_empty_dtype() {
        let dir = fixture_dir();
        let path = fixture_path(&dir);
        let header = br#"{"a": {"dtype": "", "shape": [1], "data_offsets": [0, 4]}}"#;
        write_raw_safetensors_file(&path, header, &[0_u8; 4]);
        expect_validate_error(SafetensorsFile::open(&path), "dtype must not be empty");
    }

    #[test]
    fn open_rejects_header_with_no_tensor_entries() {
        let dir = fixture_dir();
        let path = fixture_path(&dir);
        write_raw_safetensors_file(&path, b"{}", &[]);
        expect_validate_error(
            SafetensorsFile::open(&path),
            "must include at least one tensor entry",
        );
    }

    #[test]
    fn open_accepts_a_well_formed_multi_tensor_file() {
        // Happy-path control: contiguous, non-overlapping, fully-covering
        // ranges with a byte-size-consistent known dtype must still open
        // cleanly after all of the above hardening.
        let dir = fixture_dir();
        let path = fixture_path(&dir);
        let header = br#"{"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}, "b": {"dtype": "F16", "shape": [3], "data_offsets": [8, 14]}}"#;
        write_raw_safetensors_file(&path, header, &[0_u8; 14]);
        let file = SafetensorsFile::open(&path).expect("well-formed safetensors must open");
        assert_eq!(file.header().tensors.len(), 2);
        assert_eq!(file.header().data_length_bytes, 14);
    }
}
