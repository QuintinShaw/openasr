use thiserror::Error;

use crate::arch::GENERAL_ARCHITECTURE_KEY;
use crate::models::qwen::runtime_contract::Qwen3AsrExecutionMetadata;
use crate::{GgufMetadata, GgufMetadataValue, GgufTensorIndex};

use super::tensor_names::{OUTPUT_NORM_WEIGHT, TOKEN_EMBD_WEIGHT, llm_layer_tensor_names};

pub(crate) const HUNYUAN_DENSE_ARCHITECTURE_VALUE: &str = "hunyuan-dense";
pub(crate) const HUNYUAN_DENSE_CONTEXT_LENGTH_KEY: &str = "hunyuan-dense.context_length";
pub(crate) const HUNYUAN_DENSE_BLOCK_COUNT_KEY: &str = "hunyuan-dense.block_count";
pub(crate) const HUNYUAN_DENSE_EMBEDDING_LENGTH_KEY: &str = "hunyuan-dense.embedding_length";
pub(crate) const HUNYUAN_DENSE_FEED_FORWARD_LENGTH_KEY: &str = "hunyuan-dense.feed_forward_length";
pub(crate) const HUNYUAN_DENSE_ATTENTION_HEAD_COUNT_KEY: &str =
    "hunyuan-dense.attention.head_count";
pub(crate) const HUNYUAN_DENSE_ATTENTION_HEAD_COUNT_KV_KEY: &str =
    "hunyuan-dense.attention.head_count_kv";
pub(crate) const HUNYUAN_DENSE_ATTENTION_KEY_LENGTH_KEY: &str =
    "hunyuan-dense.attention.key_length";
pub(crate) const HUNYUAN_DENSE_ATTENTION_VALUE_LENGTH_KEY: &str =
    "hunyuan-dense.attention.value_length";
pub(crate) const HUNYUAN_DENSE_ROPE_FREQ_BASE_KEY: &str = "hunyuan-dense.rope.freq_base";
pub(crate) const HUNYUAN_DENSE_ROPE_SCALING_TYPE_KEY: &str = "hunyuan-dense.rope.scaling.type";
pub(crate) const HUNYUAN_DENSE_ATTENTION_RMS_EPSILON_KEY: &str =
    "hunyuan-dense.attention.layer_norm_rms_epsilon";
pub(crate) const TOKENIZER_GGML_TOKENS_KEY: &str = "tokenizer.ggml.tokens";

pub(crate) const HYMT2_DEFAULT_RUNTIME_CONTEXT_LENGTH: usize = 4096;
pub(crate) const HYMT2_RMS_NORM_EPSILON: f32 = 1.0e-5;
pub(crate) const HYMT2_EXPECTED_LAYERS: usize = 32;
pub(crate) const HYMT2_EXPECTED_D_MODEL: usize = 2048;
pub(crate) const HYMT2_EXPECTED_FFN_DIM: usize = 6144;
pub(crate) const HYMT2_EXPECTED_HEADS: usize = 16;
pub(crate) const HYMT2_EXPECTED_KV_HEADS: usize = 4;
pub(crate) const HYMT2_EXPECTED_HEAD_DIM: usize = 128;
pub(crate) const HYMT2_EXPECTED_VOCAB_SIZE: usize = 120_818;
pub(crate) const HYMT2_EXPECTED_ROPE_FREQ_BASE: f32 = 11_158_840.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hymt2ExecutionMetadata {
    pub layers: usize,
    pub d_model: usize,
    pub ffn_dim: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub gguf_context_length: usize,
    pub runtime_context_length: usize,
    pub rope_freq_base: f32,
    pub rms_norm_epsilon: f32,
}

impl Hymt2ExecutionMetadata {
    pub(crate) fn qwen_llm_metadata(self) -> Qwen3AsrExecutionMetadata {
        Qwen3AsrExecutionMetadata {
            sample_rate_hz: 16_000,
            n_mels: 1,
            n_fft: 2,
            win_length: 2,
            hop_length: 1,
            audio_layers: 1,
            audio_d_model: self.d_model,
            audio_heads: 1,
            llm_layers: self.layers,
            llm_d_model: self.d_model,
            llm_heads: self.heads,
            llm_kv_heads: self.kv_heads,
            llm_head_dim: self.head_dim,
            vocab_size: self.vocab_size,
            llm_max_positions: self.runtime_context_length,
            audio_start_token_id: 0,
            audio_end_token_id: 0,
            audio_pad_token_id: 0,
            eos_token_id: super::tokenizer::HYMT2_EOS_TOKEN_ID,
            pad_token_id: super::tokenizer::HYMT2_PAD_TOKEN_ID,
        }
    }

    pub(crate) fn assert_expected_hymt2_1_8b(self) -> Result<(), Hymt2ConfigError> {
        let checks = [
            ("layers", self.layers, HYMT2_EXPECTED_LAYERS),
            ("d_model", self.d_model, HYMT2_EXPECTED_D_MODEL),
            ("ffn_dim", self.ffn_dim, HYMT2_EXPECTED_FFN_DIM),
            ("heads", self.heads, HYMT2_EXPECTED_HEADS),
            ("kv_heads", self.kv_heads, HYMT2_EXPECTED_KV_HEADS),
            ("head_dim", self.head_dim, HYMT2_EXPECTED_HEAD_DIM),
            ("vocab_size", self.vocab_size, HYMT2_EXPECTED_VOCAB_SIZE),
        ];
        for (field, got, expected) in checks {
            if got != expected {
                return Err(Hymt2ConfigError::UnexpectedFixedArchitecture {
                    field,
                    got,
                    expected,
                });
            }
        }
        if (self.rope_freq_base - HYMT2_EXPECTED_ROPE_FREQ_BASE).abs() > 0.5 {
            return Err(Hymt2ConfigError::InvalidMetadataValue {
                key: HUNYUAN_DENSE_ROPE_FREQ_BASE_KEY,
                reason: format!(
                    "expected {HYMT2_EXPECTED_ROPE_FREQ_BASE}, got {}",
                    self.rope_freq_base
                ),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum Hymt2ConfigError {
    #[error("hymt2 missing required GGUF metadata key '{key}'")]
    MissingRequiredMetadata { key: &'static str },
    #[error("hymt2 GGUF metadata '{key}' is invalid: {reason}")]
    InvalidMetadataValue { key: &'static str, reason: String },
    #[error("hymt2 expected general.architecture='{expected}', got '{found}'")]
    UnexpectedArchitecture {
        expected: &'static str,
        found: String,
    },
    #[error("hymt2 fixed architecture field '{field}' mismatch: got {got}, expected {expected}")]
    UnexpectedFixedArchitecture {
        field: &'static str,
        got: usize,
        expected: usize,
    },
    #[error("hymt2 missing required GGUF tensor '{name}'")]
    MissingRequiredTensor { name: String },
    #[error("hymt2 GGUF tensor '{name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        name: String,
        shape: String,
        reason: String,
    },
}

pub(crate) fn parse_hymt2_execution_metadata(
    metadata: &GgufMetadata,
) -> Result<Hymt2ExecutionMetadata, Hymt2ConfigError> {
    let architecture = required_string(metadata, GENERAL_ARCHITECTURE_KEY)?;
    if architecture != HUNYUAN_DENSE_ARCHITECTURE_VALUE {
        return Err(Hymt2ConfigError::UnexpectedArchitecture {
            expected: HUNYUAN_DENSE_ARCHITECTURE_VALUE,
            found: architecture.to_string(),
        });
    }

    let layers = required_usize(metadata, HUNYUAN_DENSE_BLOCK_COUNT_KEY)?;
    let d_model = required_usize(metadata, HUNYUAN_DENSE_EMBEDDING_LENGTH_KEY)?;
    let ffn_dim = required_usize(metadata, HUNYUAN_DENSE_FEED_FORWARD_LENGTH_KEY)?;
    let heads = required_usize(metadata, HUNYUAN_DENSE_ATTENTION_HEAD_COUNT_KEY)?;
    let kv_heads = required_usize(metadata, HUNYUAN_DENSE_ATTENTION_HEAD_COUNT_KV_KEY)?;
    let head_dim = optional_usize(metadata, HUNYUAN_DENSE_ATTENTION_KEY_LENGTH_KEY)?
        .unwrap_or_else(|| d_model / heads.max(1));
    let value_dim =
        optional_usize(metadata, HUNYUAN_DENSE_ATTENTION_VALUE_LENGTH_KEY)?.unwrap_or(head_dim);
    if value_dim != head_dim {
        return Err(Hymt2ConfigError::InvalidMetadataValue {
            key: HUNYUAN_DENSE_ATTENTION_VALUE_LENGTH_KEY,
            reason: format!("value head dim {value_dim} must equal key head dim {head_dim}"),
        });
    }
    let vocab_size = metadata
        .get_string_array(TOKENIZER_GGML_TOKENS_KEY)
        .map(|tokens| tokens.len())
        .ok_or(Hymt2ConfigError::MissingRequiredMetadata {
            key: TOKENIZER_GGML_TOKENS_KEY,
        })?;
    let gguf_context_length = required_usize(metadata, HUNYUAN_DENSE_CONTEXT_LENGTH_KEY)?;
    let runtime_context_length = gguf_context_length.min(HYMT2_DEFAULT_RUNTIME_CONTEXT_LENGTH);
    let rope_freq_base = required_f32(metadata, HUNYUAN_DENSE_ROPE_FREQ_BASE_KEY)?;
    if let Some(scaling) = metadata.get_string(HUNYUAN_DENSE_ROPE_SCALING_TYPE_KEY)
        && scaling.trim() != "none"
    {
        return Err(Hymt2ConfigError::InvalidMetadataValue {
            key: HUNYUAN_DENSE_ROPE_SCALING_TYPE_KEY,
            reason: format!("unsupported RoPE scaling type '{scaling}'"),
        });
    }
    let rms_norm_epsilon = optional_f32(metadata, HUNYUAN_DENSE_ATTENTION_RMS_EPSILON_KEY)?
        .unwrap_or(HYMT2_RMS_NORM_EPSILON);
    if !rms_norm_epsilon.is_finite() || rms_norm_epsilon <= 0.0 {
        return Err(Hymt2ConfigError::InvalidMetadataValue {
            key: HUNYUAN_DENSE_ATTENTION_RMS_EPSILON_KEY,
            reason: format!("rms_norm_epsilon={rms_norm_epsilon} must be finite and positive"),
        });
    }

    validate_positive(layers, HUNYUAN_DENSE_BLOCK_COUNT_KEY)?;
    validate_positive(d_model, HUNYUAN_DENSE_EMBEDDING_LENGTH_KEY)?;
    validate_positive(ffn_dim, HUNYUAN_DENSE_FEED_FORWARD_LENGTH_KEY)?;
    validate_positive(heads, HUNYUAN_DENSE_ATTENTION_HEAD_COUNT_KEY)?;
    validate_positive(kv_heads, HUNYUAN_DENSE_ATTENTION_HEAD_COUNT_KV_KEY)?;
    validate_positive(head_dim, HUNYUAN_DENSE_ATTENTION_KEY_LENGTH_KEY)?;
    validate_positive(vocab_size, TOKENIZER_GGML_TOKENS_KEY)?;
    validate_positive(runtime_context_length, HUNYUAN_DENSE_CONTEXT_LENGTH_KEY)?;
    if !heads.is_multiple_of(kv_heads) {
        return Err(Hymt2ConfigError::InvalidMetadataValue {
            key: HUNYUAN_DENSE_ATTENTION_HEAD_COUNT_KEY,
            reason: format!("heads={heads} must be divisible by kv_heads={kv_heads}"),
        });
    }
    if d_model != heads.saturating_mul(head_dim) {
        return Err(Hymt2ConfigError::InvalidMetadataValue {
            key: HUNYUAN_DENSE_EMBEDDING_LENGTH_KEY,
            reason: format!("d_model={d_model} must equal heads*head_dim={heads}*{head_dim}"),
        });
    }

    let parsed = Hymt2ExecutionMetadata {
        layers,
        d_model,
        ffn_dim,
        heads,
        kv_heads,
        head_dim,
        vocab_size,
        gguf_context_length,
        runtime_context_length,
        rope_freq_base,
        rms_norm_epsilon,
    };
    parsed.assert_expected_hymt2_1_8b()?;
    Ok(parsed)
}

pub(crate) fn validate_hymt2_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    metadata: Hymt2ExecutionMetadata,
) -> Result<(), Hymt2ConfigError> {
    require_rank2_either(
        index,
        TOKEN_EMBD_WEIGHT,
        metadata.d_model,
        metadata.vocab_size,
    )?;
    require_vector(index, OUTPUT_NORM_WEIGHT, metadata.d_model)?;
    let q_width = metadata
        .heads
        .checked_mul(metadata.head_dim)
        .ok_or_else(|| Hymt2ConfigError::InvalidMetadataValue {
            key: HUNYUAN_DENSE_ATTENTION_HEAD_COUNT_KEY,
            reason: "Q projection width overflow".to_string(),
        })?;
    let kv_width = metadata
        .kv_heads
        .checked_mul(metadata.head_dim)
        .ok_or_else(|| Hymt2ConfigError::InvalidMetadataValue {
            key: HUNYUAN_DENSE_ATTENTION_HEAD_COUNT_KV_KEY,
            reason: "KV projection width overflow".to_string(),
        })?;
    for layer_idx in 0..metadata.layers {
        let names = llm_layer_tensor_names(layer_idx);
        require_vector(index, &names.attn_norm_weight, metadata.d_model)?;
        require_rank2_either(index, &names.attn_q_weight, metadata.d_model, q_width)?;
        require_rank2_either(index, &names.attn_k_weight, metadata.d_model, kv_width)?;
        require_rank2_either(index, &names.attn_v_weight, metadata.d_model, kv_width)?;
        require_rank2_either(index, &names.attn_output_weight, metadata.d_model, q_width)?;
        require_vector(index, &names.attn_q_norm_weight, metadata.head_dim)?;
        require_vector(index, &names.attn_k_norm_weight, metadata.head_dim)?;
        require_vector(index, &names.ffn_norm_weight, metadata.d_model)?;
        require_rank2_either(
            index,
            &names.ffn_gate_weight,
            metadata.d_model,
            metadata.ffn_dim,
        )?;
        require_rank2_either(
            index,
            &names.ffn_up_weight,
            metadata.d_model,
            metadata.ffn_dim,
        )?;
        require_rank2_either(
            index,
            &names.ffn_down_weight,
            metadata.ffn_dim,
            metadata.d_model,
        )?;
    }
    Ok(())
}

fn required_string<'a>(
    metadata: &'a GgufMetadata,
    key: &'static str,
) -> Result<&'a str, Hymt2ConfigError> {
    let value = metadata
        .get_string(key)
        .ok_or(Hymt2ConfigError::MissingRequiredMetadata { key })?;
    let value = value.trim();
    if value.is_empty() {
        return Err(Hymt2ConfigError::InvalidMetadataValue {
            key,
            reason: "value must be non-empty".to_string(),
        });
    }
    Ok(value)
}

fn required_usize(metadata: &GgufMetadata, key: &'static str) -> Result<usize, Hymt2ConfigError> {
    optional_usize(metadata, key)?.ok_or(Hymt2ConfigError::MissingRequiredMetadata { key })
}

fn optional_usize(
    metadata: &GgufMetadata,
    key: &'static str,
) -> Result<Option<usize>, Hymt2ConfigError> {
    let Some(value) = metadata_value_to_u64(metadata, key)? else {
        return Ok(None);
    };
    usize::try_from(value)
        .map(Some)
        .map_err(|_| Hymt2ConfigError::InvalidMetadataValue {
            key,
            reason: format!("value {value} does not fit usize"),
        })
}

fn required_f32(metadata: &GgufMetadata, key: &'static str) -> Result<f32, Hymt2ConfigError> {
    optional_f32(metadata, key)?.ok_or(Hymt2ConfigError::MissingRequiredMetadata { key })
}

fn optional_f32(
    metadata: &GgufMetadata,
    key: &'static str,
) -> Result<Option<f32>, Hymt2ConfigError> {
    if let Some(value) = metadata.get_f32(key) {
        return Ok(Some(value));
    }
    let Some(value) = metadata.get_string(key) else {
        return Ok(None);
    };
    let trimmed = value.trim();
    trimmed
        .parse::<f32>()
        .map(Some)
        .map_err(|error| Hymt2ConfigError::InvalidMetadataValue {
            key,
            reason: format!("cannot parse '{trimmed}' as f32: {error}"),
        })
}

fn metadata_value_to_u64(
    metadata: &GgufMetadata,
    key: &'static str,
) -> Result<Option<u64>, Hymt2ConfigError> {
    match metadata.get(key) {
        Some(GgufMetadataValue::U32(value)) => Ok(Some(u64::from(*value))),
        Some(GgufMetadataValue::U64(value)) => Ok(Some(*value)),
        Some(GgufMetadataValue::String(value)) => {
            let trimmed = value.trim();
            trimmed.parse::<u64>().map(Some).map_err(|error| {
                Hymt2ConfigError::InvalidMetadataValue {
                    key,
                    reason: format!("cannot parse '{trimmed}' as u64: {error}"),
                }
            })
        }
        Some(_) => Err(Hymt2ConfigError::InvalidMetadataValue {
            key,
            reason: "expected integer scalar".to_string(),
        }),
        None => Ok(None),
    }
}

fn validate_positive(value: usize, key: &'static str) -> Result<(), Hymt2ConfigError> {
    if value > 0 {
        return Ok(());
    }
    Err(Hymt2ConfigError::InvalidMetadataValue {
        key,
        reason: "value must be greater than 0".to_string(),
    })
}

fn require_vector(index: &GgufTensorIndex, name: &str, len: usize) -> Result<(), Hymt2ConfigError> {
    let tensor = require_tensor(index, name)?;
    if tensor.dims == [len as u64] {
        return Ok(());
    }
    Err(invalid_tensor_shape(
        name,
        &tensor.dims,
        format!("expected vector length {len}"),
    ))
}

fn require_rank2_either(
    index: &GgufTensorIndex,
    name: &str,
    a: usize,
    b: usize,
) -> Result<(), Hymt2ConfigError> {
    let tensor = require_tensor(index, name)?;
    let expected_a = vec![a as u64, b as u64];
    let expected_b = vec![b as u64, a as u64];
    if tensor.dims == expected_a || tensor.dims == expected_b {
        return Ok(());
    }
    Err(invalid_tensor_shape(
        name,
        &tensor.dims,
        format!("expected [{a}, {b}] or [{b}, {a}]"),
    ))
}

fn require_tensor<'a>(
    index: &'a GgufTensorIndex,
    name: &str,
) -> Result<&'a crate::GgufTensorMetadata, Hymt2ConfigError> {
    index
        .get(name)
        .ok_or_else(|| Hymt2ConfigError::MissingRequiredTensor {
            name: name.to_string(),
        })
}

fn invalid_tensor_shape(name: &str, shape: &[u64], reason: String) -> Hymt2ConfigError {
    Hymt2ConfigError::InvalidTensorShape {
        name: name.to_string(),
        shape: render_shape(shape),
        reason,
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

    use super::*;
    use crate::{GgufMetadata, GgufMetadataValue};

    fn metadata(context: u64) -> GgufMetadata {
        let mut values = BTreeMap::new();
        values.insert(
            GENERAL_ARCHITECTURE_KEY.to_string(),
            GgufMetadataValue::String(HUNYUAN_DENSE_ARCHITECTURE_VALUE.to_string()),
        );
        values.insert(
            HUNYUAN_DENSE_BLOCK_COUNT_KEY.to_string(),
            GgufMetadataValue::U64(32),
        );
        values.insert(
            HUNYUAN_DENSE_EMBEDDING_LENGTH_KEY.to_string(),
            GgufMetadataValue::U64(2048),
        );
        values.insert(
            HUNYUAN_DENSE_FEED_FORWARD_LENGTH_KEY.to_string(),
            GgufMetadataValue::U64(6144),
        );
        values.insert(
            HUNYUAN_DENSE_ATTENTION_HEAD_COUNT_KEY.to_string(),
            GgufMetadataValue::U64(16),
        );
        values.insert(
            HUNYUAN_DENSE_ATTENTION_HEAD_COUNT_KV_KEY.to_string(),
            GgufMetadataValue::U64(4),
        );
        values.insert(
            HUNYUAN_DENSE_ATTENTION_KEY_LENGTH_KEY.to_string(),
            GgufMetadataValue::U64(128),
        );
        values.insert(
            HUNYUAN_DENSE_CONTEXT_LENGTH_KEY.to_string(),
            GgufMetadataValue::U64(context),
        );
        values.insert(
            HUNYUAN_DENSE_ROPE_FREQ_BASE_KEY.to_string(),
            GgufMetadataValue::F32(HYMT2_EXPECTED_ROPE_FREQ_BASE),
        );
        values.insert(
            HUNYUAN_DENSE_ROPE_SCALING_TYPE_KEY.to_string(),
            GgufMetadataValue::String("none".to_string()),
        );
        values.insert(
            TOKENIZER_GGML_TOKENS_KEY.to_string(),
            GgufMetadataValue::StringArray(vec!["x".to_string(); HYMT2_EXPECTED_VOCAB_SIZE]),
        );
        GgufMetadata::from_values_for_test(values)
    }

    #[test]
    fn parses_hymt2_metadata_and_caps_context() {
        let parsed = parse_hymt2_execution_metadata(&metadata(262_144)).expect("metadata");
        assert_eq!(parsed.gguf_context_length, 262_144);
        assert_eq!(
            parsed.runtime_context_length,
            HYMT2_DEFAULT_RUNTIME_CONTEXT_LENGTH
        );
        parsed
            .assert_expected_hymt2_1_8b()
            .expect("fixed architecture");
    }

    #[test]
    fn rejects_rope_scaling_other_than_none() {
        let mut values = metadata(4096).values().clone();
        values.insert(
            HUNYUAN_DENSE_ROPE_SCALING_TYPE_KEY.to_string(),
            GgufMetadataValue::String("dynamic".to_string()),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        let error = parse_hymt2_execution_metadata(&metadata).expect_err("must reject");
        assert!(matches!(
            error,
            Hymt2ConfigError::InvalidMetadataValue {
                key: HUNYUAN_DENSE_ROPE_SCALING_TYPE_KEY,
                ..
            }
        ));
    }
}
