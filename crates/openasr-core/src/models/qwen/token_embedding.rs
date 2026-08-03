#[cfg(test)]
use std::path::Path;

use thiserror::Error;

use crate::ggml_runtime::{
    GgufOwnedWeightTensorPayload, GgufTensorDataReadError, GgufTensorDataReader,
    dequantize_ggml_row_to_f32, ggml_row_size_bytes,
};
use crate::nn::half::f16_bits_to_f32;

use super::runtime_contract::Qwen3AsrExecutionMetadata;
use super::tensor_names::TOKEN_EMBD_WEIGHT as TOKEN_EMBEDDING_TENSOR_NAME;
const GGML_TYPE_F32: i32 = 0;
const GGML_TYPE_F16: i32 = 1;

#[derive(Debug, Clone)]
pub(crate) struct Qwen3AsrTokenEmbeddingTable {
    d_model: usize,
    vocab_size: usize,
    storage: TokenEmbeddingStorage,
}

#[derive(Debug, Clone)]
enum TokenEmbeddingStorage {
    /// F32/F16 in either supported orientation, or quantized token-major
    /// storage. The payload is an owning view into the already-open GGUF mmap;
    /// cloning this table only bumps the mmap `Arc` and never duplicates the
    /// multi-hundred-megabyte vocabulary matrix.
    Mapped {
        payload: TokenEmbeddingPayload,
        layout: TokenEmbeddingLayout,
        ggml_type: i32,
        quantized_row_size: Option<usize>,
    },
    /// Rare quantized hidden-major packs cannot be gathered by ggml row. They
    /// are the only representation that requires a transposed f32 fallback.
    F32Token(Vec<f32>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenEmbeddingLayout {
    TokenMajor,
    HiddenMajor,
}

#[derive(Debug, Clone)]
enum TokenEmbeddingPayload {
    Mapped(GgufOwnedWeightTensorPayload),
    #[cfg(test)]
    TestBytes(Vec<u8>),
}

impl TokenEmbeddingPayload {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Mapped(payload) => payload.bytes(),
            #[cfg(test)]
            Self::TestBytes(bytes) => bytes,
        }
    }

    fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        match self {
            Self::Mapped(payload) => payload.retained_system_memory_bytes(),
            #[cfg(test)]
            Self::TestBytes(bytes) => {
                let mut capacity =
                    crate::models::system_memory_owner::SystemMemoryCapacity::default();
                capacity.add_vec(bytes, "qwen test token embedding payload")?;
                Ok(capacity.finish())
            }
        }
    }
}

impl Qwen3AsrTokenEmbeddingTable {
    /// Quotes the host allocation path selected by the real loader without
    /// materializing the vocabulary matrix. F32/F16 and token-major
    /// quantized tensors stay mmap-backed. Only a quantized hidden-major
    /// tensor requires a retained f32 transpose; during construction its
    /// source and destination matrices overlap, so peak is twice retained.
    pub(crate) fn quoted_system_memory_bytes_from_reader(
        reader: &GgufTensorDataReader,
        tensor_name: &'static str,
        d_model: usize,
        vocab_size: usize,
    ) -> Result<(u64, u64), String> {
        let tensor = reader
            .tensor_index()
            .get(tensor_name)
            .ok_or_else(|| format!("required tensor '{tensor_name}' is missing"))?;
        if tensor.dims.len() != 2 {
            return Err(format!(
                "tensor '{tensor_name}' must be rank 2, got {:?}",
                tensor.dims
            ));
        }
        let token_major = tensor.dims == [d_model as u64, vocab_size as u64];
        let hidden_major = tensor.dims == [vocab_size as u64, d_model as u64];
        if !token_major && !hidden_major {
            return Err(format!(
                "tensor '{tensor_name}' has shape {:?}, expected [{d_model}, {vocab_size}] or [{vocab_size}, {d_model}]",
                tensor.dims
            ));
        }
        let mapped =
            tensor.ggml_type == GGML_TYPE_F32 || tensor.ggml_type == GGML_TYPE_F16 || token_major;
        if mapped {
            let retained =
                GgufOwnedWeightTensorPayload::quoted_retained_system_memory_bytes(tensor)?;
            return Ok((retained, retained));
        }
        let retained = d_model
            .checked_mul(vocab_size)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| format!("tensor '{tensor_name}' f32 transpose quote overflowed"))?;
        let peak = retained
            .checked_mul(2)
            .ok_or_else(|| format!("tensor '{tensor_name}' transpose peak quote overflowed"))?;
        Ok((peak, retained))
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        match &self.storage {
            TokenEmbeddingStorage::Mapped { payload, .. } => {
                bytes.add(
                    payload.retained_system_memory_bytes()?,
                    "qwen mapped token embedding metadata",
                )?;
            }
            TokenEmbeddingStorage::F32Token(values) => {
                bytes.add_vec(values, "qwen token embedding f32")?;
            }
        }
        Ok(bytes.finish())
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn gather_rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, Qwen3AsrTokenEmbeddingError> {
        let out_len = token_ids
            .len()
            .checked_mul(self.d_model)
            .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
        let mut out = Vec::with_capacity(out_len);
        for &token_id in token_ids {
            let token_index = token_index_or_error(token_id, self.vocab_size)?;
            match &self.storage {
                TokenEmbeddingStorage::Mapped {
                    payload,
                    layout,
                    ggml_type,
                    quantized_row_size,
                } => {
                    let data = payload.bytes();
                    if let Some(row_size) = quantized_row_size {
                        debug_assert_eq!(*layout, TokenEmbeddingLayout::TokenMajor);
                        let start = token_index
                            .checked_mul(*row_size)
                            .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
                        let end = start
                            .checked_add(*row_size)
                            .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
                        let row_bytes = data
                            .get(start..end)
                            .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
                        let row_start = out.len();
                        dequantize_ggml_row_to_f32(*ggml_type, row_bytes, self.d_model, &mut out)
                            .map_err(|error| Qwen3AsrTokenEmbeddingError::TensorReadFailed {
                            reason: error.to_string(),
                        })?;
                        if out[row_start..].iter().any(|value| !value.is_finite()) {
                            return Err(Qwen3AsrTokenEmbeddingError::NonFiniteValues);
                        }
                    } else {
                        for hidden_idx in 0..self.d_model {
                            let element_index = match layout {
                                TokenEmbeddingLayout::TokenMajor => token_index
                                    .checked_mul(self.d_model)
                                    .and_then(|base| base.checked_add(hidden_idx)),
                                TokenEmbeddingLayout::HiddenMajor => hidden_idx
                                    .checked_mul(self.vocab_size)
                                    .and_then(|base| base.checked_add(token_index)),
                            }
                            .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
                            out.push(read_embedding_element(data, element_index, *ggml_type)?);
                        }
                    }
                }
                TokenEmbeddingStorage::F32Token(values) => {
                    let start = token_index
                        .checked_mul(self.d_model)
                        .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
                    let end = start
                        .checked_add(self.d_model)
                        .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
                    out.extend_from_slice(&values[start..end]);
                }
            }
        }
        Ok(out)
    }

    #[cfg(test)]
    pub(crate) fn mapped_payload(&self) -> Option<&GgufOwnedWeightTensorPayload> {
        match &self.storage {
            TokenEmbeddingStorage::Mapped {
                payload: TokenEmbeddingPayload::Mapped(payload),
                ..
            } => Some(payload),
            _ => None,
        }
    }
}

#[derive(Debug, Error)]
pub(crate) enum Qwen3AsrTokenEmbeddingError {
    #[error("qwen3-asr token embedding tensor read failed: {reason}")]
    TensorReadFailed { reason: String },
    #[error("qwen3-asr token embedding tensor '{tensor_name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        tensor_name: &'static str,
        shape: String,
        reason: String,
    },
    #[error(
        "qwen3-asr token embedding row gather token id {token_id} is out of vocab_size={vocab_size}"
    )]
    TokenIdOutOfRange { token_id: u32, vocab_size: usize },
    #[error("qwen3-asr token embedding row gather overflowed")]
    GatherOverflow,
    #[error("qwen3-asr token embedding tensor contains non-finite values")]
    NonFiniteValues,
}

#[cfg(test)]
pub(crate) fn load_qwen3_token_embedding_table(
    runtime_source_path: &Path,
    metadata: Qwen3AsrExecutionMetadata,
) -> Result<Qwen3AsrTokenEmbeddingTable, Qwen3AsrTokenEmbeddingError> {
    let reader =
        GgufTensorDataReader::from_path(runtime_source_path).map_err(map_tensor_read_error)?;
    load_qwen3_token_embedding_table_from_reader(&reader, metadata)
}

pub(crate) fn load_qwen3_token_embedding_table_from_reader(
    reader: &GgufTensorDataReader,
    metadata: Qwen3AsrExecutionMetadata,
) -> Result<Qwen3AsrTokenEmbeddingTable, Qwen3AsrTokenEmbeddingError> {
    load_token_embedding_table_from_reader_with_tensor_name(
        reader,
        TOKEN_EMBEDDING_TENSOR_NAME,
        metadata.llm_d_model,
        metadata.vocab_size,
    )
}

/// Like [`load_qwen3_token_embedding_table_from_reader`] but decoupled from
/// `Qwen3AsrExecutionMetadata` and qwen's own tensor name, so a sibling
/// family (e.g. firered-llm's `llm.tok_emb.weight`) can reuse the same
/// row-gather table -- looking up embedding rows by token id has no
/// Qwen2/Qwen3-specific shape.
pub(crate) fn load_token_embedding_table_from_reader_with_tensor_name(
    reader: &GgufTensorDataReader,
    tensor_name: &'static str,
    d_model: usize,
    vocab_size: usize,
) -> Result<Qwen3AsrTokenEmbeddingTable, Qwen3AsrTokenEmbeddingError> {
    let tensor = reader.tensor_index().get(tensor_name).ok_or_else(|| {
        Qwen3AsrTokenEmbeddingError::InvalidTensorShape {
            tensor_name,
            shape: "[]".to_string(),
            reason: "tensor is missing from GGUF tensor index".to_string(),
        }
    })?;
    let dims = tensor.dims.clone();
    if dims.len() != 2 {
        return Err(Qwen3AsrTokenEmbeddingError::InvalidTensorShape {
            tensor_name,
            shape: render_shape(&dims),
            reason: "expected rank-2 matrix".to_string(),
        });
    }
    let output_major_vocab_layout = dims[0] == d_model as u64 && dims[1] == vocab_size as u64;
    let input_major_hidden_layout = dims[0] == vocab_size as u64 && dims[1] == d_model as u64;
    if !output_major_vocab_layout && !input_major_hidden_layout {
        return Err(Qwen3AsrTokenEmbeddingError::InvalidTensorShape {
            tensor_name,
            shape: render_shape(&dims),
            reason: format!("expected [{d_model} x {vocab_size}] or [{vocab_size} x {d_model}]"),
        });
    }

    let ggml_type = tensor.ggml_type;
    let layout = if output_major_vocab_layout {
        TokenEmbeddingLayout::TokenMajor
    } else {
        TokenEmbeddingLayout::HiddenMajor
    };
    let storage = if ggml_type == GGML_TYPE_F32 || ggml_type == GGML_TYPE_F16 {
        let payload = reader
            .owned_weight_tensor_payload_by_name(tensor_name)
            .map_err(map_tensor_read_error)?;
        TokenEmbeddingStorage::Mapped {
            payload: TokenEmbeddingPayload::Mapped(payload),
            layout,
            ggml_type,
            quantized_row_size: None,
        }
    } else if output_major_vocab_layout {
        // Token-major quantized table: each ggml row (ne0 == d_model) is one
        // token, so keep the compact quantized bytes and dequantize a single
        // row per gathered token instead of blowing the whole vocab table up to
        // f32 at load. Non-token-major quantized layouts (rare) fall through to
        // the f32 path below, which handles the transpose.
        let row_size = ggml_row_size_bytes(ggml_type, d_model)
            .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
        let expected_len = row_size
            .checked_mul(vocab_size)
            .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
        let payload = reader
            .owned_weight_tensor_payload_by_name(tensor_name)
            .map_err(map_tensor_read_error)?;
        if payload.bytes().len() != expected_len {
            return Err(Qwen3AsrTokenEmbeddingError::InvalidTensorShape {
                tensor_name,
                shape: render_shape(&dims),
                reason: format!(
                    "quantized token-embedding payload is {} bytes, expected {expected_len}",
                    payload.bytes().len()
                ),
            });
        }
        TokenEmbeddingStorage::Mapped {
            payload: TokenEmbeddingPayload::Mapped(payload),
            layout,
            ggml_type,
            quantized_row_size: Some(row_size),
        }
    } else {
        let values = reader
            .host_tensor_f32_copy_dequantized_by_name(tensor_name, &dims)
            .map_err(map_tensor_read_error)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Qwen3AsrTokenEmbeddingError::NonFiniteValues);
        }
        let token_major_values = if output_major_vocab_layout {
            values
        } else {
            transpose_vocab_hidden_to_token_major(&values, d_model, vocab_size)?
        };
        TokenEmbeddingStorage::F32Token(token_major_values)
    };
    Ok(Qwen3AsrTokenEmbeddingTable {
        d_model,
        vocab_size,
        storage,
    })
}

fn read_embedding_element(
    data: &[u8],
    element_index: usize,
    ggml_type: i32,
) -> Result<f32, Qwen3AsrTokenEmbeddingError> {
    let element_bytes = match ggml_type {
        GGML_TYPE_F32 => 4,
        GGML_TYPE_F16 => 2,
        _ => {
            return Err(Qwen3AsrTokenEmbeddingError::TensorReadFailed {
                reason: format!(
                    "mapped scalar gather does not support ggml_type {ggml_type}; quantized tensors must be token-major"
                ),
            });
        }
    };
    let start = element_index
        .checked_mul(element_bytes)
        .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
    let end = start
        .checked_add(element_bytes)
        .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
    let bytes = data
        .get(start..end)
        .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
    let value = match ggml_type {
        GGML_TYPE_F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        GGML_TYPE_F16 => f16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])),
        _ => unreachable!("unsupported ggml type returned above"),
    };
    if !value.is_finite() {
        return Err(Qwen3AsrTokenEmbeddingError::NonFiniteValues);
    }
    Ok(value)
}

fn token_index_or_error(
    token_id: u32,
    vocab_size: usize,
) -> Result<usize, Qwen3AsrTokenEmbeddingError> {
    let token_index =
        usize::try_from(token_id).map_err(|_| Qwen3AsrTokenEmbeddingError::TokenIdOutOfRange {
            token_id,
            vocab_size,
        })?;
    if token_index >= vocab_size {
        return Err(Qwen3AsrTokenEmbeddingError::TokenIdOutOfRange {
            token_id,
            vocab_size,
        });
    }
    Ok(token_index)
}

fn transpose_vocab_hidden_to_token_major(
    source: &[f32],
    hidden_size: usize,
    vocab_size: usize,
) -> Result<Vec<f32>, Qwen3AsrTokenEmbeddingError> {
    let expected = hidden_size
        .checked_mul(vocab_size)
        .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
    if source.len() != expected {
        return Err(Qwen3AsrTokenEmbeddingError::InvalidTensorShape {
            tensor_name: TOKEN_EMBEDDING_TENSOR_NAME,
            shape: format!("[{hidden_size}, {vocab_size}]"),
            reason: format!(
                "expected {} values from shape, got {}",
                expected,
                source.len()
            ),
        });
    }
    let mut transposed = vec![0.0_f32; source.len()];
    for hidden_idx in 0..hidden_size {
        for vocab_idx in 0..vocab_size {
            let src = vocab_idx + vocab_size * hidden_idx;
            let dst = hidden_idx + hidden_size * vocab_idx;
            transposed[dst] = source[src];
        }
    }
    Ok(transposed)
}

fn map_tensor_read_error(error: GgufTensorDataReadError) -> Qwen3AsrTokenEmbeddingError {
    Qwen3AsrTokenEmbeddingError::TensorReadFailed {
        reason: error.to_string(),
    }
}

fn render_shape(shape: &[u64]) -> String {
    let parts = shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{parts}]")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::GgufTensorDataReader;
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};

    use super::*;

    fn metadata() -> Qwen3AsrExecutionMetadata {
        Qwen3AsrExecutionMetadata {
            sample_rate_hz: 16_000,
            n_mels: 80,
            n_fft: 400,
            win_length: 400,
            hop_length: 160,
            audio_layers: 2,
            audio_d_model: 16,
            audio_heads: 2,
            llm_layers: 2,
            llm_d_model: 4,
            llm_heads: 2,
            llm_kv_heads: 2,
            llm_head_dim: 2,
            vocab_size: 8,
            llm_max_positions: 256,
            audio_start_token_id: 2,
            audio_end_token_id: 3,
            audio_pad_token_id: 4,
            eos_token_id: 5,
            pad_token_id: 6,
        }
    }

    fn base_spec() -> TinyGgufFixtureSpec {
        let mut kv = BTreeMap::new();
        kv.insert("general.architecture".to_string(), "qwen3-asr".to_string());
        TinyGgufFixtureSpec::new(kv)
    }

    #[test]
    fn token_embedding_loader_accepts_hidden_vocab_layout_without_transpose() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen-token-embd-hidden-vocab.gguf");
        let spec = base_spec().with_tensor_shape(TOKEN_EMBEDDING_TENSOR_NAME, [4_u64, 8_u64]);
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

        let table = load_qwen3_token_embedding_table(&runtime_path, metadata()).expect("load");
        let rows = table.gather_rows(&[0, 1]).expect("gather");
        assert_eq!(rows.len(), 8);

        let reader = GgufTensorDataReader::from_path(&runtime_path).expect("reader");
        let raw = reader
            .host_tensor_f32_copy_by_name(TOKEN_EMBEDDING_TENSOR_NAME, &[4, 8])
            .expect("tensor");
        assert_eq!(rows, raw[0..8].to_vec());
    }

    #[test]
    fn token_embedding_loader_transposes_vocab_hidden_layout_into_token_major_rows() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen-token-embd-vocab-hidden.gguf");
        let spec = base_spec().with_tensor_shape(TOKEN_EMBEDDING_TENSOR_NAME, [8_u64, 4_u64]);
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

        let table = load_qwen3_token_embedding_table(&runtime_path, metadata()).expect("load");
        let rows = table.gather_rows(&[2]).expect("gather");
        assert_eq!(rows.len(), 4);

        let reader = GgufTensorDataReader::from_path(&runtime_path).expect("reader");
        let raw = reader
            .host_tensor_f32_copy_by_name(TOKEN_EMBEDDING_TENSOR_NAME, &[8, 4])
            .expect("tensor");
        assert_eq!(rows, vec![raw[2], raw[10], raw[18], raw[26]]);
    }

    #[test]
    fn token_embedding_gather_rejects_token_out_of_range() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen-token-embd-out-of-range.gguf");
        let spec = base_spec().with_tensor_shape(TOKEN_EMBEDDING_TENSOR_NAME, [8_u64, 4_u64]);
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

        let table = load_qwen3_token_embedding_table(&runtime_path, metadata()).expect("load");
        let error = table
            .gather_rows(&[42])
            .expect_err("out-of-range token id must fail");
        assert!(error.to_string().contains("out of vocab_size=8"));
    }

    #[test]
    fn token_embedding_f16_loader_accepts_hidden_vocab_layout_without_transpose() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen-token-embd-hidden-vocab-f16.gguf");
        let spec = base_spec()
            .with_tensor_shape(TOKEN_EMBEDDING_TENSOR_NAME, [4_u64, 8_u64])
            .with_tensor_f16(TOKEN_EMBEDDING_TENSOR_NAME);
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

        let table = load_qwen3_token_embedding_table(&runtime_path, metadata()).expect("load");
        let rows = table.gather_rows(&[0, 1]).expect("gather");
        assert_eq!(rows.len(), 8);

        let reader = GgufTensorDataReader::from_path(&runtime_path).expect("reader");
        let raw_bits = reader
            .host_tensor_f16_bits_copy_by_name(TOKEN_EMBEDDING_TENSOR_NAME, &[4, 8])
            .expect("tensor");
        let expected: Vec<f32> = raw_bits[0..8]
            .iter()
            .copied()
            .map(f16_bits_to_f32)
            .collect();
        assert_eq!(rows, expected);
    }

    #[test]
    fn token_embedding_quantized_token_major_gather_matches_full_dequant() {
        use crate::ggml_runtime::{GgmlKvElementType, dequantize_q8_0_rows};

        // q8_0 block size is 32, so d_model must be a multiple of 32.
        let d_model = 64usize;
        let vocab_size = 5usize;
        let mut values = Vec::with_capacity(d_model * vocab_size);
        for token in 0..vocab_size {
            for hidden in 0..d_model {
                values.push(((token * 7 + hidden) as f32) * 0.013 - 0.4);
            }
        }
        let data = GgmlKvElementType::Q8_0
            .quantize_rows_from_f32(&values, d_model, vocab_size)
            .expect("quantize q8_0 token rows");
        let row_size = data.len() / vocab_size;
        let reference =
            dequantize_q8_0_rows(&data, d_model, vocab_size).expect("reference dequant");

        let table = Qwen3AsrTokenEmbeddingTable {
            d_model,
            vocab_size,
            storage: TokenEmbeddingStorage::Mapped {
                payload: TokenEmbeddingPayload::TestBytes(data),
                layout: TokenEmbeddingLayout::TokenMajor,
                ggml_type: 8, // GGML_TYPE_Q8_0
                quantized_row_size: Some(row_size),
            },
        };

        let order = [3usize, 0, 4];
        let token_ids: Vec<u32> = order.iter().map(|&t| t as u32).collect();
        let gathered = table.gather_rows(&token_ids).expect("gather");
        assert_eq!(gathered.len(), order.len() * d_model);
        // Per-row lazy dequant must be byte-identical to dequantizing the whole
        // table up front (same ggml `to_float` trait), just without the f32 blow-up.
        for (out_idx, &token) in order.iter().enumerate() {
            let gathered_row = &gathered[out_idx * d_model..(out_idx + 1) * d_model];
            let reference_row = &reference[token * d_model..(token + 1) * d_model];
            assert_eq!(gathered_row, reference_row, "token {token} row must match");
        }
    }

    #[test]
    fn token_embedding_f16_loader_transposes_vocab_hidden_layout_into_token_major_rows() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen-token-embd-vocab-hidden-f16.gguf");
        let spec = base_spec()
            .with_tensor_shape(TOKEN_EMBEDDING_TENSOR_NAME, [8_u64, 4_u64])
            .with_tensor_f16(TOKEN_EMBEDDING_TENSOR_NAME);
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

        let table = load_qwen3_token_embedding_table(&runtime_path, metadata()).expect("load");
        let rows = table.gather_rows(&[2]).expect("gather");
        assert_eq!(rows.len(), 4);

        let reader = GgufTensorDataReader::from_path(&runtime_path).expect("reader");
        let raw_bits = reader
            .host_tensor_f16_bits_copy_by_name(TOKEN_EMBEDDING_TENSOR_NAME, &[8, 4])
            .expect("tensor");
        let expected = vec![
            f16_bits_to_f32(raw_bits[2]),
            f16_bits_to_f32(raw_bits[10]),
            f16_bits_to_f32(raw_bits[18]),
            f16_bits_to_f32(raw_bits[26]),
        ];
        assert_eq!(rows, expected);
    }
}
