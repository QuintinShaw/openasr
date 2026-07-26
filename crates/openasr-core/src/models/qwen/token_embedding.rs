#[cfg(test)]
use std::path::Path;

use thiserror::Error;

use crate::ggml_runtime::{
    GgufOwnedWeightTensorPayload, GgufTensorDataReadError, GgufTensorDataReader,
};

use super::runtime_contract::Qwen3AsrExecutionMetadata;
use super::tensor_names::TOKEN_EMBD_WEIGHT as TOKEN_EMBEDDING_TENSOR_NAME;

#[derive(Debug, Clone)]
pub(crate) struct Qwen3AsrTokenEmbeddingTable {
    d_model: usize,
    vocab_size: usize,
    storage: TokenEmbeddingStorage,
}

#[derive(Debug, Clone)]
enum TokenEmbeddingStorage {
    // Keep the full table in the GGUF mmap. `layout` describes how a logical
    // token row is addressed in GGUF's dimension-0-contiguous storage.
    // Cloning this is O(1): `GgufOwnedWeightTensorPayload` only clones the
    // `Arc<Mmap>` and small tensor metadata, never the vocab x hidden bytes.
    Mmap {
        payload: GgufOwnedWeightTensorPayload,
        layout: TokenEmbeddingLayout,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenEmbeddingLayout {
    // GGUF dims `[hidden, vocab]`: each token is one contiguous row.
    TokenRows,
    // GGUF dims `[vocab, hidden]`: a token is a logical column. Read just
    // that column (and, for quantized tensors, one block per hidden row).
    TokenColumns,
}

impl Qwen3AsrTokenEmbeddingTable {
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn gather_rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, Qwen3AsrTokenEmbeddingError> {
        let out_len = token_ids
            .len()
            .checked_mul(self.d_model)
            .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
        let requested_bytes = out_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
        let mut out = Vec::new();
        out.try_reserve_exact(out_len).map_err(|_| {
            Qwen3AsrTokenEmbeddingError::HostAllocationFailed {
                stage: "qwen3-asr-token-embedding-gather",
                requested_bytes,
            }
        })?;
        for &token_id in token_ids {
            let token_index = token_index_or_error(token_id, self.vocab_size)?;
            match &self.storage {
                TokenEmbeddingStorage::Mmap { payload, layout } => {
                    let start = out.len();
                    let end = start
                        .checked_add(self.d_model)
                        .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
                    out.resize(end, 0.0);
                    let destination = &mut out[start..end];
                    match layout {
                        TokenEmbeddingLayout::TokenRows => {
                            payload.dequantize_row_to_f32(token_index, destination)
                        }
                        TokenEmbeddingLayout::TokenColumns => {
                            payload.dequantize_column_to_f32(token_index, destination)
                        }
                    }
                    .map_err(map_tensor_read_error)?;
                    if destination.iter().any(|value| !value.is_finite()) {
                        return Err(Qwen3AsrTokenEmbeddingError::NonFiniteValues);
                    }
                }
            }
        }
        Ok(out)
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
    #[error(
        "qwen3-asr token embedding host allocation failed at {stage}: requested_bytes={requested_bytes}"
    )]
    HostAllocationFailed {
        stage: &'static str,
        requested_bytes: usize,
    },
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

    let payload = reader
        .owned_weight_tensor_payload_by_name(tensor_name)
        .map_err(map_tensor_read_error)?;
    if payload.dims.as_slice() != [d_model, vocab_size]
        && payload.dims.as_slice() != [vocab_size, d_model]
    {
        return Err(Qwen3AsrTokenEmbeddingError::InvalidTensorShape {
            tensor_name,
            shape: format!("{:?}", payload.dims),
            reason: "owned GGUF payload shape changed while loading".to_string(),
        });
    }
    let layout = if output_major_vocab_layout {
        TokenEmbeddingLayout::TokenRows
    } else {
        TokenEmbeddingLayout::TokenColumns
    };
    Ok(Qwen3AsrTokenEmbeddingTable {
        d_model,
        vocab_size,
        storage: TokenEmbeddingStorage::Mmap { payload, layout },
    })
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

fn map_tensor_read_error(error: GgufTensorDataReadError) -> Qwen3AsrTokenEmbeddingError {
    match error {
        GgufTensorDataReadError::TensorAllocationFailed {
            requested_bytes, ..
        } => Qwen3AsrTokenEmbeddingError::HostAllocationFailed {
            stage: "gguf-token-embedding-read",
            requested_bytes,
        },
        error => Qwen3AsrTokenEmbeddingError::TensorReadFailed {
            reason: error.to_string(),
        },
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
    use crate::ggml_runtime::{
        GgufWriteTensor, GgufWriteTensorType, quantize_f32_to_ggml_tensor_data, write_gguf_file_v0,
    };
    use crate::nn::half::f16_bits_to_f32;
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

    fn write_q4_k_token_embedding_fixture(path: &Path, dims: [u64; 2], values: Vec<f32>) {
        let data = quantize_f32_to_ggml_tensor_data(GgufWriteTensorType::Q4_K, &dims, &values)
            .expect("quantize a real q4_k fixture payload");
        write_gguf_file_v0(
            path,
            &BTreeMap::new(),
            &[GgufWriteTensor {
                name: TOKEN_EMBEDDING_TENSOR_NAME.to_string(),
                dims: dims.to_vec(),
                tensor_type: GgufWriteTensorType::Q4_K,
                data,
            }],
        )
        .expect("write a real q4_k GGUF fixture");
    }

    fn q4_k_fixture_values(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| ((index % 37) as f32 - 18.0) * 0.0625)
            .collect()
    }

    #[test]
    fn token_embedding_loader_accepts_hidden_vocab_layout_without_transpose() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen-token-embd-hidden-vocab.gguf");
        let spec = base_spec().with_tensor_shape(TOKEN_EMBEDDING_TENSOR_NAME, [4_u64, 8_u64]);
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

        let table = load_qwen3_token_embedding_table(&runtime_path, metadata()).expect("load");
        assert!(
            matches!(table.storage, TokenEmbeddingStorage::Mmap { .. }),
            "token embeddings must stay mmap-backed rather than eagerly allocating a full f32 table"
        );
        let rows = table.gather_rows(&[0, 1]).expect("gather");
        assert_eq!(rows.len(), 8);

        let reader = GgufTensorDataReader::from_path(&runtime_path).expect("reader");
        let raw = reader
            .host_tensor_f32_copy_by_name(TOKEN_EMBEDDING_TENSOR_NAME, &[4, 8])
            .expect("tensor");
        assert_eq!(rows, raw[0..8].to_vec());
    }

    #[test]
    fn token_embedding_loader_reads_vocab_hidden_layout_without_full_transpose() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen-token-embd-vocab-hidden.gguf");
        let spec = base_spec().with_tensor_shape(TOKEN_EMBEDDING_TENSOR_NAME, [8_u64, 4_u64]);
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

        let table = load_qwen3_token_embedding_table(&runtime_path, metadata()).expect("load");
        assert!(matches!(
            table.storage,
            TokenEmbeddingStorage::Mmap {
                layout: TokenEmbeddingLayout::TokenColumns,
                ..
            }
        ));
        let rows = table.gather_rows(&[2]).expect("gather");
        assert_eq!(rows.len(), 4);

        let reader = GgufTensorDataReader::from_path(&runtime_path).expect("reader");
        let raw = reader
            .host_tensor_f32_copy_by_name(TOKEN_EMBEDDING_TENSOR_NAME, &[8, 4])
            .expect("tensor");
        assert_eq!(rows, vec![raw[2], raw[10], raw[18], raw[26]]);
    }

    #[test]
    fn token_embedding_clone_reuses_the_same_mmap_payload() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen-token-embd-clone.gguf");
        let spec = base_spec().with_tensor_shape(TOKEN_EMBEDDING_TENSOR_NAME, [4_u64, 8_u64]);
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

        let table = load_qwen3_token_embedding_table(&runtime_path, metadata()).expect("load");
        let cloned = table.clone();
        let (
            TokenEmbeddingStorage::Mmap {
                payload: original, ..
            },
            TokenEmbeddingStorage::Mmap {
                payload: duplicate, ..
            },
        ) = (&table.storage, &cloned.storage);
        assert_eq!(
            original.bytes().as_ptr(),
            duplicate.bytes().as_ptr(),
            "table clone must share the same mmap region rather than copying the token table"
        );
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

    #[test]
    fn token_embedding_q4_k_rows_gather_only_requested_mmap_rows() {
        // q4_k's real ggml superblock has a 256-wide first dimension. This
        // goes through the production ggml quantizer and GGUF writer rather
        // than a zero-filled synthetic byte fixture, so the lazy row path is
        // checked against actual q4_k scale/min/bit decoding.
        let d_model = 256;
        let vocab_size = 3;
        let dims = [d_model as u64, vocab_size as u64];
        let values = q4_k_fixture_values(d_model * vocab_size);
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen-token-embd-q4-k-rows.gguf");
        write_q4_k_token_embedding_fixture(&runtime_path, dims, values);

        let reader = GgufTensorDataReader::from_path(&runtime_path).expect("reader");
        let table = load_token_embedding_table_from_reader_with_tensor_name(
            &reader,
            TOKEN_EMBEDDING_TENSOR_NAME,
            d_model,
            vocab_size,
        )
        .expect("load mmap-backed q4_k rows");
        assert!(matches!(
            table.storage,
            TokenEmbeddingStorage::Mmap {
                layout: TokenEmbeddingLayout::TokenRows,
                ..
            }
        ));

        let gathered = table.gather_rows(&[0, 2]).expect("gather requested rows");
        let dequantized = reader
            .host_tensor_f32_copy_dequantized_by_name(TOKEN_EMBEDDING_TENSOR_NAME, &dims)
            .expect("reference q4_k dequantization");
        assert_eq!(&gathered[..d_model], &dequantized[..d_model]);
        assert_eq!(&gathered[d_model..], &dequantized[d_model * 2..d_model * 3]);
    }

    #[test]
    fn token_embedding_q4_k_columns_gather_only_requested_mmap_columns() {
        // In the legacy [vocab, hidden] orientation, q4_k's block-aligned
        // first dimension is vocab. Each logical token column therefore reads
        // one value from every independently q4_k-quantized hidden row.
        let d_model = 3;
        let vocab_size = 256;
        let dims = [vocab_size as u64, d_model as u64];
        let values = q4_k_fixture_values(d_model * vocab_size);
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen-token-embd-q4-k-columns.gguf");
        write_q4_k_token_embedding_fixture(&runtime_path, dims, values);

        let reader = GgufTensorDataReader::from_path(&runtime_path).expect("reader");
        let table = load_token_embedding_table_from_reader_with_tensor_name(
            &reader,
            TOKEN_EMBEDDING_TENSOR_NAME,
            d_model,
            vocab_size,
        )
        .expect("load mmap-backed q4_k columns");
        assert!(matches!(
            table.storage,
            TokenEmbeddingStorage::Mmap {
                layout: TokenEmbeddingLayout::TokenColumns,
                ..
            }
        ));

        let gathered = table
            .gather_rows(&[2, 255])
            .expect("gather requested columns");
        let dequantized = reader
            .host_tensor_f32_copy_dequantized_by_name(TOKEN_EMBEDDING_TENSOR_NAME, &dims)
            .expect("reference q4_k dequantization");
        let expected = vec![
            dequantized[2],
            dequantized[vocab_size + 2],
            dequantized[vocab_size * 2 + 2],
            dequantized[255],
            dequantized[vocab_size + 255],
            dequantized[vocab_size * 2 + 255],
        ];
        assert_eq!(gathered, expected);
    }
}
