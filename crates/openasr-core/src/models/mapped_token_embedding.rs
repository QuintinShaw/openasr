//! Family-neutral, mmap-backed token-embedding row gather.
//!
//! For a vocabulary of `V` rows, width `D`, and request for `K` rows, every
//! caller must emit `K*D` scalar values. Row-addressable F32/F16/Q8/Q4 storage
//! therefore reaches the lower bounds `Theta(K*D)` time and output space while
//! retaining only `Theta(1)` auxiliary heap metadata; the packed bytes remain
//! file-backed. The retired full-table path paid `Theta(V*D)` load time and
//! retained heap even when `K << V`. A quantized hidden-major matrix is the
//! deliberate fallback: ggml blocks do not encode independently addressable
//! token rows in that orientation, so one `Theta(V*D)` transpose is required
//! for correct random access rather than pretending zero-copy is possible.

use thiserror::Error;

use crate::ggml_runtime::{
    GgufOwnedWeightTensorPayload, GgufTensorDataReadError, GgufTensorDataReader,
    dequantize_ggml_row_to_f32, ggml_row_size_bytes,
};
use crate::nn::half::f16_bits_to_f32;

const GGML_TYPE_F32: i32 = 0;
const GGML_TYPE_F16: i32 = 1;

#[derive(Debug, Clone)]
pub(crate) struct MappedTokenEmbeddingTable {
    d_model: usize,
    vocab_size: usize,
    storage: TokenEmbeddingStorage,
}

/// Zero-copy token-embedding binding for a native decoder graph.
///
/// Only ggml's canonical `[d_model, vocab]` layout is row-addressable by
/// token id. The descriptor borrows the already-validated tensor name; the
/// graph executor resolves that name inside its own pack-wide loaded-weight
/// context, so this never creates a second mapping or device allocation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MappedTokenEmbeddingDeviceSpec<'a> {
    pub tensor_name: &'a str,
    pub d_model: usize,
    pub vocab_size: usize,
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
                capacity.add_vec(bytes, "mapped test token embedding payload")?;
                Ok(capacity.finish())
            }
        }
    }
}

impl MappedTokenEmbeddingTable {
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
                    "mapped token embedding metadata",
                )?;
            }
            TokenEmbeddingStorage::F32Token(values) => {
                bytes.add_vec(values, "mapped token embedding f32")?;
            }
        }
        Ok(bytes.finish())
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub(crate) fn device_graph_spec(&self) -> Option<MappedTokenEmbeddingDeviceSpec<'_>> {
        match &self.storage {
            TokenEmbeddingStorage::Mapped {
                payload: TokenEmbeddingPayload::Mapped(payload),
                layout: TokenEmbeddingLayout::TokenMajor,
                ..
            } => Some(MappedTokenEmbeddingDeviceSpec {
                tensor_name: &payload.metadata.name,
                d_model: self.d_model,
                vocab_size: self.vocab_size,
            }),
            #[cfg(test)]
            TokenEmbeddingStorage::Mapped {
                payload: TokenEmbeddingPayload::TestBytes(_),
                ..
            } => None,
            TokenEmbeddingStorage::Mapped {
                layout: TokenEmbeddingLayout::HiddenMajor,
                ..
            }
            | TokenEmbeddingStorage::F32Token(_) => None,
        }
    }

    pub fn gather_rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, MappedTokenEmbeddingError> {
        let out_len = token_ids
            .len()
            .checked_mul(self.d_model)
            .ok_or(MappedTokenEmbeddingError::GatherOverflow)?;
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
                            .ok_or(MappedTokenEmbeddingError::GatherOverflow)?;
                        let end = start
                            .checked_add(*row_size)
                            .ok_or(MappedTokenEmbeddingError::GatherOverflow)?;
                        let row_bytes = data
                            .get(start..end)
                            .ok_or(MappedTokenEmbeddingError::GatherOverflow)?;
                        let row_start = out.len();
                        dequantize_ggml_row_to_f32(*ggml_type, row_bytes, self.d_model, &mut out)
                            .map_err(|error| MappedTokenEmbeddingError::TensorReadFailed {
                            reason: error.to_string(),
                        })?;
                        if out[row_start..].iter().any(|value| !value.is_finite()) {
                            return Err(MappedTokenEmbeddingError::NonFiniteValues);
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
                            .ok_or(MappedTokenEmbeddingError::GatherOverflow)?;
                            out.push(read_embedding_element(data, element_index, *ggml_type)?);
                        }
                    }
                }
                TokenEmbeddingStorage::F32Token(values) => {
                    let start = token_index
                        .checked_mul(self.d_model)
                        .ok_or(MappedTokenEmbeddingError::GatherOverflow)?;
                    let end = start
                        .checked_add(self.d_model)
                        .ok_or(MappedTokenEmbeddingError::GatherOverflow)?;
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
pub(crate) enum MappedTokenEmbeddingError {
    #[error("mapped token embedding tensor read failed: {reason}")]
    TensorReadFailed { reason: String },
    #[error("mapped token embedding tensor '{tensor_name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        tensor_name: &'static str,
        shape: String,
        reason: String,
    },
    #[error(
        "mapped token embedding row gather token id {token_id} is out of vocab_size={vocab_size}"
    )]
    TokenIdOutOfRange { token_id: u32, vocab_size: usize },
    #[error("mapped token embedding row gather overflowed")]
    GatherOverflow,
    #[error("mapped token embedding tensor contains non-finite values")]
    NonFiniteValues,
}

/// Load a vocabulary table as a compact mmap-backed row gatherer. Looking up
/// embedding rows by token id has no model-family-specific shape or math.
pub(crate) fn load_mapped_token_embedding_table_from_reader(
    reader: &GgufTensorDataReader,
    tensor_name: &'static str,
    d_model: usize,
    vocab_size: usize,
) -> Result<MappedTokenEmbeddingTable, MappedTokenEmbeddingError> {
    let tensor = reader.tensor_index().get(tensor_name).ok_or_else(|| {
        MappedTokenEmbeddingError::InvalidTensorShape {
            tensor_name,
            shape: "[]".to_string(),
            reason: "tensor is missing from GGUF tensor index".to_string(),
        }
    })?;
    let dims = tensor.dims.clone();
    if dims.len() != 2 {
        return Err(MappedTokenEmbeddingError::InvalidTensorShape {
            tensor_name,
            shape: render_shape(&dims),
            reason: "expected rank-2 matrix".to_string(),
        });
    }
    let output_major_vocab_layout = dims[0] == d_model as u64 && dims[1] == vocab_size as u64;
    let input_major_hidden_layout = dims[0] == vocab_size as u64 && dims[1] == d_model as u64;
    if !output_major_vocab_layout && !input_major_hidden_layout {
        return Err(MappedTokenEmbeddingError::InvalidTensorShape {
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
            .ok_or(MappedTokenEmbeddingError::GatherOverflow)?;
        let expected_len = row_size
            .checked_mul(vocab_size)
            .ok_or(MappedTokenEmbeddingError::GatherOverflow)?;
        let payload = reader
            .owned_weight_tensor_payload_by_name(tensor_name)
            .map_err(map_tensor_read_error)?;
        if payload.bytes().len() != expected_len {
            return Err(MappedTokenEmbeddingError::InvalidTensorShape {
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
            return Err(MappedTokenEmbeddingError::NonFiniteValues);
        }
        let token_major_values = if output_major_vocab_layout {
            values
        } else {
            transpose_vocab_hidden_to_token_major(&values, tensor_name, d_model, vocab_size)?
        };
        TokenEmbeddingStorage::F32Token(token_major_values)
    };
    Ok(MappedTokenEmbeddingTable {
        d_model,
        vocab_size,
        storage,
    })
}

fn read_embedding_element(
    data: &[u8],
    element_index: usize,
    ggml_type: i32,
) -> Result<f32, MappedTokenEmbeddingError> {
    let element_bytes = match ggml_type {
        GGML_TYPE_F32 => 4,
        GGML_TYPE_F16 => 2,
        _ => {
            return Err(MappedTokenEmbeddingError::TensorReadFailed {
                reason: format!(
                    "mapped scalar gather does not support ggml_type {ggml_type}; quantized tensors must be token-major"
                ),
            });
        }
    };
    let start = element_index
        .checked_mul(element_bytes)
        .ok_or(MappedTokenEmbeddingError::GatherOverflow)?;
    let end = start
        .checked_add(element_bytes)
        .ok_or(MappedTokenEmbeddingError::GatherOverflow)?;
    let bytes = data
        .get(start..end)
        .ok_or(MappedTokenEmbeddingError::GatherOverflow)?;
    let value = match ggml_type {
        GGML_TYPE_F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        GGML_TYPE_F16 => f16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])),
        _ => unreachable!("unsupported ggml type returned above"),
    };
    if !value.is_finite() {
        return Err(MappedTokenEmbeddingError::NonFiniteValues);
    }
    Ok(value)
}

fn token_index_or_error(
    token_id: u32,
    vocab_size: usize,
) -> Result<usize, MappedTokenEmbeddingError> {
    let token_index =
        usize::try_from(token_id).map_err(|_| MappedTokenEmbeddingError::TokenIdOutOfRange {
            token_id,
            vocab_size,
        })?;
    if token_index >= vocab_size {
        return Err(MappedTokenEmbeddingError::TokenIdOutOfRange {
            token_id,
            vocab_size,
        });
    }
    Ok(token_index)
}

fn transpose_vocab_hidden_to_token_major(
    source: &[f32],
    tensor_name: &'static str,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<Vec<f32>, MappedTokenEmbeddingError> {
    let expected = hidden_size
        .checked_mul(vocab_size)
        .ok_or(MappedTokenEmbeddingError::GatherOverflow)?;
    if source.len() != expected {
        return Err(MappedTokenEmbeddingError::InvalidTensorShape {
            tensor_name,
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

fn map_tensor_read_error(error: GgufTensorDataReadError) -> MappedTokenEmbeddingError {
    MappedTokenEmbeddingError::TensorReadFailed {
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

    const TEST_TENSOR_NAME: &str = "test.token_embd.weight";
    const TEST_D_MODEL: usize = 4;
    const TEST_VOCAB_SIZE: usize = 8;

    fn load_test_table(path: &std::path::Path) -> MappedTokenEmbeddingTable {
        let reader = GgufTensorDataReader::from_path(path).expect("reader");
        load_mapped_token_embedding_table_from_reader(
            &reader,
            TEST_TENSOR_NAME,
            TEST_D_MODEL,
            TEST_VOCAB_SIZE,
        )
        .expect("load")
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
        let spec = base_spec().with_tensor_shape(TEST_TENSOR_NAME, [4_u64, 8_u64]);
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

        let table = load_test_table(&runtime_path);
        let device = table
            .device_graph_spec()
            .expect("canonical token-major embedding should be graph-bindable");
        assert_eq!(device.tensor_name, TEST_TENSOR_NAME);
        assert_eq!(device.d_model, TEST_D_MODEL);
        assert_eq!(device.vocab_size, TEST_VOCAB_SIZE);
        let rows = table.gather_rows(&[0, 1]).expect("gather");
        assert_eq!(rows.len(), 8);

        let reader = GgufTensorDataReader::from_path(&runtime_path).expect("reader");
        let raw = reader
            .host_tensor_f32_copy_by_name(TEST_TENSOR_NAME, &[4, 8])
            .expect("tensor");
        assert_eq!(rows, raw[0..8].to_vec());
    }

    #[test]
    fn token_embedding_loader_transposes_vocab_hidden_layout_into_token_major_rows() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen-token-embd-vocab-hidden.gguf");
        let spec = base_spec().with_tensor_shape(TEST_TENSOR_NAME, [8_u64, 4_u64]);
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

        let table = load_test_table(&runtime_path);
        assert!(
            table.device_graph_spec().is_none(),
            "transposed host storage must not be exposed as a canonical graph tensor"
        );
        let rows = table.gather_rows(&[2]).expect("gather");
        assert_eq!(rows.len(), 4);

        let reader = GgufTensorDataReader::from_path(&runtime_path).expect("reader");
        let raw = reader
            .host_tensor_f32_copy_by_name(TEST_TENSOR_NAME, &[8, 4])
            .expect("tensor");
        assert_eq!(rows, vec![raw[2], raw[10], raw[18], raw[26]]);
    }

    #[test]
    fn token_embedding_gather_rejects_token_out_of_range() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen-token-embd-out-of-range.gguf");
        let spec = base_spec().with_tensor_shape(TEST_TENSOR_NAME, [8_u64, 4_u64]);
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

        let table = load_test_table(&runtime_path);
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
            .with_tensor_shape(TEST_TENSOR_NAME, [4_u64, 8_u64])
            .with_tensor_f16(TEST_TENSOR_NAME);
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

        let table = load_test_table(&runtime_path);
        let rows = table.gather_rows(&[0, 1]).expect("gather");
        assert_eq!(rows.len(), 8);

        let reader = GgufTensorDataReader::from_path(&runtime_path).expect("reader");
        let raw_bits = reader
            .host_tensor_f16_bits_copy_by_name(TEST_TENSOR_NAME, &[4, 8])
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

        let table = MappedTokenEmbeddingTable {
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
    fn q4_k_token_major_gather_matches_full_row_dequantization() {
        use crate::ggml_runtime::{GgufWriteTensorType, quantize_f32_to_ggml_tensor_data};

        // Q4_K superblocks contain 256 values, so every token row uses an
        // independently addressable, block-aligned 256-wide payload.
        let d_model = 256usize;
        let vocab_size = 5usize;
        let values = (0..d_model * vocab_size)
            .map(|index| ((index % 97) as f32 - 48.0) * 0.017)
            .collect::<Vec<_>>();
        let data = quantize_f32_to_ggml_tensor_data(
            GgufWriteTensorType::Q4_K,
            &[d_model as u64, vocab_size as u64],
            &values,
        )
        .expect("quantize q4_k token rows");
        let row_size = ggml_row_size_bytes(12, d_model).expect("q4_k row size");
        assert_eq!(data.len(), row_size * vocab_size);

        let mut reference = Vec::with_capacity(values.len());
        for row in data.chunks_exact(row_size) {
            dequantize_ggml_row_to_f32(12, row, d_model, &mut reference)
                .expect("full q4_k row dequantization");
        }
        let table = MappedTokenEmbeddingTable {
            d_model,
            vocab_size,
            storage: TokenEmbeddingStorage::Mapped {
                payload: TokenEmbeddingPayload::TestBytes(data),
                layout: TokenEmbeddingLayout::TokenMajor,
                ggml_type: 12,
                quantized_row_size: Some(row_size),
            },
        };

        let order = [4usize, 1, 3];
        let gathered = table
            .gather_rows(&order.map(|token| token as u32))
            .expect("gather q4_k rows");
        for (out_index, token) in order.into_iter().enumerate() {
            assert_eq!(
                &gathered[out_index * d_model..(out_index + 1) * d_model],
                &reference[token * d_model..(token + 1) * d_model],
                "token {token} row must match full dequantization",
            );
        }
    }

    #[test]
    fn token_embedding_f16_loader_transposes_vocab_hidden_layout_into_token_major_rows() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen-token-embd-vocab-hidden-f16.gguf");
        let spec = base_spec()
            .with_tensor_shape(TEST_TENSOR_NAME, [8_u64, 4_u64])
            .with_tensor_f16(TEST_TENSOR_NAME);
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

        let table = load_test_table(&runtime_path);
        let rows = table.gather_rows(&[2]).expect("gather");
        assert_eq!(rows.len(), 4);

        let reader = GgufTensorDataReader::from_path(&runtime_path).expect("reader");
        let raw_bits = reader
            .host_tensor_f16_bits_copy_by_name(TEST_TENSOR_NAME, &[8, 4])
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
