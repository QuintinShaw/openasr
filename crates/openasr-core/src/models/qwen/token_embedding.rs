#[cfg(test)]
use std::path::Path;

use thiserror::Error;

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};

use super::runtime_contract::Qwen3AsrExecutionMetadata;
use super::tensor_names::TOKEN_EMBD_WEIGHT as TOKEN_EMBEDDING_TENSOR_NAME;
const GGML_TYPE_F16: i32 = 1;

#[derive(Debug, Clone)]
pub(crate) struct Qwen3AsrTokenEmbeddingTable {
    d_model: usize,
    vocab_size: usize,
    storage: TokenEmbeddingStorage,
}

#[derive(Debug, Clone)]
enum TokenEmbeddingStorage {
    // Layout: token-major row-contiguous ([token][hidden]) f32.
    F32Token(Vec<f32>),
    // Layouts use raw F16 bits to avoid eager full-table f32 materialization.
    F16Token(Vec<u16>),
    F16Hidden(Vec<u16>),
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
        let mut out = Vec::with_capacity(out_len);
        for &token_id in token_ids {
            let token_index = token_index_or_error(token_id, self.vocab_size)?;
            match &self.storage {
                TokenEmbeddingStorage::F32Token(values) => {
                    let start = token_index
                        .checked_mul(self.d_model)
                        .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
                    let end = start
                        .checked_add(self.d_model)
                        .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
                    out.extend_from_slice(&values[start..end]);
                }
                TokenEmbeddingStorage::F16Token(values) => {
                    let start = token_index
                        .checked_mul(self.d_model)
                        .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
                    let end = start
                        .checked_add(self.d_model)
                        .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
                    out.extend(values[start..end].iter().copied().map(f16_bits_to_f32));
                }
                TokenEmbeddingStorage::F16Hidden(values) => {
                    for hidden_idx in 0..self.d_model {
                        let src = hidden_idx
                            .checked_mul(self.vocab_size)
                            .and_then(|base| base.checked_add(token_index))
                            .ok_or(Qwen3AsrTokenEmbeddingError::GatherOverflow)?;
                        out.push(f16_bits_to_f32(values[src]));
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
    let tensor = reader
        .tensor_index()
        .get(TOKEN_EMBEDDING_TENSOR_NAME)
        .ok_or_else(|| Qwen3AsrTokenEmbeddingError::InvalidTensorShape {
            tensor_name: TOKEN_EMBEDDING_TENSOR_NAME,
            shape: "[]".to_string(),
            reason: "tensor is missing from GGUF tensor index".to_string(),
        })?;
    let dims = tensor.dims.clone();
    if dims.len() != 2 {
        return Err(Qwen3AsrTokenEmbeddingError::InvalidTensorShape {
            tensor_name: TOKEN_EMBEDDING_TENSOR_NAME,
            shape: render_shape(&dims),
            reason: "expected rank-2 matrix".to_string(),
        });
    }
    let d_model = metadata.llm_d_model;
    let vocab_size = metadata.vocab_size;
    let output_major_vocab_layout = dims[0] == d_model as u64 && dims[1] == vocab_size as u64;
    let input_major_hidden_layout = dims[0] == vocab_size as u64 && dims[1] == d_model as u64;
    if !output_major_vocab_layout && !input_major_hidden_layout {
        return Err(Qwen3AsrTokenEmbeddingError::InvalidTensorShape {
            tensor_name: TOKEN_EMBEDDING_TENSOR_NAME,
            shape: render_shape(&dims),
            reason: format!("expected [{d_model} x {vocab_size}] or [{vocab_size} x {d_model}]"),
        });
    }

    let storage = if tensor.ggml_type == GGML_TYPE_F16 {
        let values = reader
            .host_tensor_f16_bits_copy_by_name(TOKEN_EMBEDDING_TENSOR_NAME, &dims)
            .map_err(map_tensor_read_error)?;
        if output_major_vocab_layout {
            TokenEmbeddingStorage::F16Token(values)
        } else {
            TokenEmbeddingStorage::F16Hidden(values)
        }
    } else {
        let values = reader
            .host_tensor_f32_copy_dequantized_by_name(TOKEN_EMBEDDING_TENSOR_NAME, &dims)
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

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = (bits & 0x03ff) as u32;

    let f32_bits = match exponent {
        0 => {
            if fraction == 0 {
                sign
            } else {
                let mut fraction_norm = fraction;
                let mut exponent_shift = -14_i32;
                while (fraction_norm & 0x0400) == 0 {
                    fraction_norm <<= 1;
                    exponent_shift -= 1;
                }
                fraction_norm &= 0x03ff;
                let exponent_bits = ((exponent_shift + 127) as u32) << 23;
                let fraction_bits = fraction_norm << 13;
                sign | exponent_bits | fraction_bits
            }
        }
        0x1f => {
            let exponent_bits = 0xff_u32 << 23;
            let fraction_bits = fraction << 13;
            sign | exponent_bits | fraction_bits
        }
        _ => {
            let exponent_bits = ((exponent as i32 - 15 + 127) as u32) << 23;
            let fraction_bits = fraction << 13;
            sign | exponent_bits | fraction_bits
        }
    };

    f32::from_bits(f32_bits)
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
