use std::{collections::BTreeMap, path::PathBuf};

use serde::Deserialize;

use crate::arch::hparams::{
    WHISPER_DECODER_BLOCK_COUNT_KEY, WHISPER_DECODER_CONTEXT_LENGTH_KEY,
    WHISPER_DECODER_EMBEDDING_LENGTH_KEY, WHISPER_DECODER_HEAD_COUNT_KEY,
    WHISPER_ENCODER_BLOCK_COUNT_KEY, WHISPER_ENCODER_CONTEXT_LENGTH_KEY,
    WHISPER_ENCODER_EMBEDDING_LENGTH_KEY, WHISPER_ENCODER_HEAD_COUNT_KEY,
    WHISPER_ENCODER_MELS_COUNT_KEY, WHISPER_VOCAB_SIZE_KEY,
};
use crate::ggml_runtime::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, quantize_f32_to_ggml_tensor_data,
    read_gguf_tensor_index, validate_ggml_runtime_source_path, write_gguf_file_v0,
};
use crate::models::{
    ggml_family_adapter::GGML_TOKENIZER_ID_KEY,
    ggml_family_registry::{
        WHISPER_AUDIO_FRONTEND_ID, WHISPER_DECODE_POLICY_ID, WHISPER_GGML_ARCHITECTURE_ID,
        WHISPER_TOKENIZER_ID,
    },
    oasr_metadata::{
        OASR_FEATURE_STREAMING_GGML_TRUE_STREAMING_V1, OASR_METADATA_KEY_AUDIO_FRONTEND,
        OASR_METADATA_KEY_DECODE_POLICY, OASR_METADATA_KEY_FEATURE_STREAMING,
        OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
        OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
    },
    whisper::WHISPER_MODEL_FAMILY,
};

use super::ggml_tensor_binding::{WhisperGgufTensorBindingContext, bind_whisper_gguf_tensors};
use super::local_source::{
    SafetensorsHeaderV0, SafetensorsTensorHeaderV0, WhisperLocalSourceError,
    load_safetensors_header_v0,
    source_io::{read_source_file_bytes, read_source_json_file},
    validate_error,
};
use super::tokenizer::{
    TOKENIZER_GGML_EOT_TOKEN_ID_KEY, TOKENIZER_GGML_MERGES_KEY, TOKENIZER_GGML_MODEL_KEY,
    TOKENIZER_GGML_MODEL_VALUE_GPT2, TOKENIZER_GGML_NO_TIMESTAMPS_TOKEN_ID_KEY,
    TOKENIZER_GGML_SOT_TOKEN_ID_KEY, TOKENIZER_GGML_SPECIAL_TOKEN_IDS_KEY,
    TOKENIZER_GGML_TOKENS_KEY, TOKENIZER_GGML_TRANSCRIBE_TOKEN_ID_KEY, WhisperHfTokenizerImport,
    WhisperTokenizer, load_whisper_hf_tokenizer_import_v0,
};

const SOURCE_CONFIG_JSON: &str = "config.json";
const SOURCE_MODEL_SAFETENSORS: &str = "model.safetensors";
const DECODER_TOKEN_EMBEDDING_TENSOR_NAME: &str = "model.decoder.embed_tokens.weight";
const SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES: u64 = 8;
const OPENASR_MODEL_ID_KEY: &str = "openasr.model.id";
const GENERAL_ARCHITECTURE_KEY: &str = "general.architecture";
const GGUF_WHISPER_ARCHITECTURE_VALUE: &str = "whisper";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperLocalSourceImportRequest {
    pub source_root: PathBuf,
    pub output_root: PathBuf,
    pub package_id: String,
    pub package_variant: Option<String>,
    pub model_language: String,
    pub source_name: String,
    pub source_revision: String,
    pub license_name: String,
    pub license_source: String,
    pub quantization: WhisperRuntimeQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperLocalSourceImportRuntimeResult {
    pub output_path: PathBuf,
    pub model_id: String,
    pub tensor_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(non_camel_case_types)]
pub enum WhisperRuntimeQuantizationMode {
    #[default]
    Fp16,
    Q8_0,
    Q4_K,
}

impl WhisperRuntimeQuantizationMode {
    fn label(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Q8_0 => "q8_0",
            Self::Q4_K => "q4_k",
        }
    }
}

#[derive(Debug, Deserialize)]
struct WhisperConfigJson {
    model_type: String,
    vocab_size: u32,
    d_model: u32,
    encoder_layers: u32,
    decoder_layers: u32,
    encoder_attention_heads: u32,
    #[serde(default)]
    decoder_attention_heads: Option<u32>,
    max_source_positions: u32,
    max_target_positions: u32,
    #[serde(default)]
    num_mel_bins: Option<u32>,
    #[serde(default)]
    forced_decoder_ids: Vec<Vec<u32>>,
}

pub fn convert_local_whisper_hf_source_to_runtime_pack(
    request: &WhisperLocalSourceImportRequest,
) -> Result<WhisperLocalSourceImportRuntimeResult, WhisperLocalSourceError> {
    validate_request(request)?;
    let config: WhisperConfigJson =
        read_source_json_file(&request.source_root, SOURCE_CONFIG_JSON)?;
    validate_whisper_config(&config)?;
    let tokenizer = load_whisper_hf_tokenizer_import_v0(&request.source_root).map_err(|error| {
        validate_error(format!(
            "Whisper local-source GGUF import tokenizer load failed: {error}"
        ))
    })?;

    let safetensors_source_path = request.source_root.join(SOURCE_MODEL_SAFETENSORS);
    let model_bytes = read_source_file_bytes(&request.source_root, SOURCE_MODEL_SAFETENSORS)?;
    let safetensors_header = load_safetensors_header_v0(&safetensors_source_path)?;
    validate_whisper_tokenizer_contract(&tokenizer, &config, &safetensors_header)?;
    let tensors = gguf_tensors_from_safetensors(&safetensors_header, &model_bytes, request)?;
    let model_id = compose_model_id(&request.package_id, request.package_variant.as_deref());
    let metadata = whisper_runtime_gguf_metadata(request, &config, &tokenizer, &model_id);

    write_gguf_file_v0(&request.output_root, &metadata, &tensors).map_err(|error| {
        validate_error(format!("Whisper local-source GGUF writer failed: {error}"))
    })?;

    let index = read_gguf_tensor_index(&request.output_root).map_err(|error| {
        validate_error(format!(
            "Whisper local-source GGUF writer produced unreadable tensor index: {error}"
        ))
    })?;
    let binding_context = whisper_gguf_tensor_binding_context(&config)?;
    bind_whisper_gguf_tensors(&binding_context, &index).map_err(|error| {
        validate_error(format!(
            "Whisper local-source GGUF writer produced tensors that do not match the GGUF tensor binding contract: {error}"
        ))
    })?;
    let runtime_source = validate_ggml_runtime_source_path(&request.output_root).map_err(|error| {
        validate_error(format!(
            "Whisper local-source GGUF writer produced invalid runtime source path '{}': {error}",
            request.output_root.display()
        ))
    })?;
    WhisperTokenizer::from_ggml_runtime_source(&runtime_source).map_err(|error| {
        validate_error(format!(
            "Whisper local-source GGUF writer produced tokenizer metadata that runtime cannot load: {error}"
        ))
    })?;

    Ok(WhisperLocalSourceImportRuntimeResult {
        output_path: request.output_root.clone(),
        model_id,
        tensor_count: index.tensors().len(),
    })
}

fn gguf_tensors_from_safetensors(
    header: &SafetensorsHeaderV0,
    model_bytes: &[u8],
    request: &WhisperLocalSourceImportRequest,
) -> Result<Vec<GgufWriteTensor>, WhisperLocalSourceError> {
    if header.tensors.is_empty() {
        return Err(validate_error(
            "Whisper local-source GGUF import requires at least one safetensors tensor".to_string(),
        ));
    }

    header
        .tensors
        .iter()
        .map(|tensor| {
            let range = safetensors_payload_range(
                &tensor.name,
                header.header_length_bytes,
                tensor.data_offsets,
            )?;
            let data = model_bytes.get(range).ok_or_else(|| {
                validate_error(format!(
                    "Whisper local-source GGUF import tensor '{}' data range is out of bounds",
                    tensor.name
                ))
            })?;
            gguf_tensor_from_safetensors_tensor(tensor, data, request.quantization)
        })
        .collect()
}

fn gguf_tensor_from_safetensors_tensor(
    tensor: &SafetensorsTensorHeaderV0,
    data: &[u8],
    quantization: WhisperRuntimeQuantizationMode,
) -> Result<GgufWriteTensor, WhisperLocalSourceError> {
    if let Some(tensor_type) = quantization_tensor_type_for_whisper_tensor(tensor, quantization) {
        return gguf_quantized_tensor_from_safetensors(tensor, data, quantization, tensor_type);
    }
    if is_whisper_encoder_linear_weight(&tensor.name) {
        return gguf_runtime_encoder_linear_tensor_from_safetensors(tensor, data);
    }
    if tensor.name == DECODER_TOKEN_EMBEDDING_TENSOR_NAME {
        return gguf_runtime_decoder_token_embedding_tensor_from_safetensors(tensor, data);
    }
    if is_whisper_runtime_f16_weight(&tensor.name) {
        return gguf_runtime_f16_tensor_from_safetensors(tensor, data);
    }
    let tensor_type = gguf_tensor_type_from_safetensors_dtype(&tensor.name, &tensor.dtype)?;
    Ok(GgufWriteTensor {
        name: tensor.name.clone(),
        dims: tensor.shape.clone(),
        tensor_type,
        data: data.to_vec(),
    })
}

fn quantization_tensor_type_for_whisper_tensor(
    tensor: &SafetensorsTensorHeaderV0,
    quantization: WhisperRuntimeQuantizationMode,
) -> Option<GgufWriteTensorType> {
    if quantization == WhisperRuntimeQuantizationMode::Fp16 {
        return None;
    }
    if tensor.shape.len() != 2 {
        return None;
    }
    let name = tensor.name.as_str();
    if !is_whisper_encoder_linear_weight(name) && !is_whisper_decoder_linear_weight(name) {
        return None;
    }
    let dims = gguf_runtime_tensor_dims_from_source_tensor(tensor);
    let ne0 = dims.first().copied()?;
    if ne0.is_multiple_of(32_u64) {
        if quantization == WhisperRuntimeQuantizationMode::Q4_K && ne0.is_multiple_of(256_u64) {
            return Some(GgufWriteTensorType::Q4_K);
        }
        return Some(GgufWriteTensorType::Q8_0);
    }
    None
}

fn gguf_quantized_tensor_from_safetensors(
    tensor: &SafetensorsTensorHeaderV0,
    data: &[u8],
    quantization: WhisperRuntimeQuantizationMode,
    tensor_type: GgufWriteTensorType,
) -> Result<GgufWriteTensor, WhisperLocalSourceError> {
    let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
    let dims = gguf_runtime_tensor_dims_from_source_tensor(tensor);
    let expected = tensor_element_count(&tensor.name, &dims)?;
    if values.len() != expected {
        return Err(validate_error(format!(
            "Whisper local-source GGUF import quantization tensor '{}' decoded {} values but expected {}",
            tensor.name,
            values.len(),
            expected
        )));
    }
    let quantized = quantize_f32_to_ggml_tensor_data(tensor_type, &dims, &values)
        .map_err(|error| {
            validate_error(format!(
                "Whisper local-source GGUF import quantization failed for tensor '{}' (mode {}, tensor_type {:?}): {error}",
                tensor.name,
                quantization.label(),
                tensor_type
            ))
        })?;
    Ok(GgufWriteTensor {
        name: tensor.name.clone(),
        dims,
        tensor_type,
        data: quantized,
    })
}

fn gguf_runtime_tensor_dims_from_source_tensor(tensor: &SafetensorsTensorHeaderV0) -> Vec<u64> {
    if (is_whisper_encoder_linear_weight(&tensor.name)
        || is_whisper_decoder_linear_weight(&tensor.name))
        && tensor.shape.len() == 2
    {
        return vec![tensor.shape[1], tensor.shape[0]];
    }
    if tensor.name == DECODER_TOKEN_EMBEDDING_TENSOR_NAME && tensor.shape.len() == 2 {
        return vec![tensor.shape[1], tensor.shape[0]];
    }
    tensor.shape.clone()
}

fn is_whisper_encoder_linear_weight(name: &str) -> bool {
    name.starts_with("model.encoder.layers.")
        && matches!(
            name.rsplit_once('.').map(|(_, suffix)| suffix),
            Some("weight")
        )
        && (name.contains(".self_attn.q_proj.")
            || name.contains(".self_attn.k_proj.")
            || name.contains(".self_attn.v_proj.")
            || name.contains(".self_attn.out_proj.")
            || name.ends_with(".fc1.weight")
            || name.ends_with(".fc2.weight"))
}

fn is_whisper_decoder_linear_weight(name: &str) -> bool {
    if name == "model.decoder.output_projection.weight" {
        return true;
    }
    name.starts_with("model.decoder.layers.")
        && matches!(
            name.rsplit_once('.').map(|(_, suffix)| suffix),
            Some("weight")
        )
        && (name.contains(".self_attn.q_proj.")
            || name.contains(".self_attn.k_proj.")
            || name.contains(".self_attn.v_proj.")
            || name.contains(".self_attn.out_proj.")
            || name.contains(".encoder_attn.q_proj.")
            || name.contains(".encoder_attn.k_proj.")
            || name.contains(".encoder_attn.v_proj.")
            || name.contains(".encoder_attn.out_proj.")
            || name.contains(".cross_attn.q_proj.")
            || name.contains(".cross_attn.k_proj.")
            || name.contains(".cross_attn.v_proj.")
            || name.contains(".cross_attn.out_proj.")
            || name.ends_with(".fc1.weight")
            || name.ends_with(".fc2.weight"))
}

fn is_whisper_runtime_f16_weight(name: &str) -> bool {
    matches!(
        name,
        "model.encoder.conv1.weight"
            | "model.encoder.conv2.weight"
            | "model.decoder.embed_tokens.weight"
    ) || (name.starts_with("model.decoder.layers.")
        && matches!(
            name.rsplit_once('.').map(|(_, suffix)| suffix),
            Some("weight")
        )
        && (name.contains(".self_attn.q_proj.")
            || name.contains(".self_attn.k_proj.")
            || name.contains(".self_attn.v_proj.")
            || name.contains(".self_attn.out_proj.")
            || name.contains(".encoder_attn.q_proj.")
            || name.contains(".encoder_attn.k_proj.")
            || name.contains(".encoder_attn.v_proj.")
            || name.contains(".encoder_attn.out_proj.")
            || name.contains(".cross_attn.q_proj.")
            || name.contains(".cross_attn.k_proj.")
            || name.contains(".cross_attn.v_proj.")
            || name.contains(".cross_attn.out_proj.")
            || name.ends_with(".fc1.weight")
            || name.ends_with(".fc2.weight")))
}

fn gguf_runtime_f16_tensor_from_safetensors(
    tensor: &SafetensorsTensorHeaderV0,
    data: &[u8],
) -> Result<GgufWriteTensor, WhisperLocalSourceError> {
    let values = match tensor.dtype.as_str() {
        "F32" => decode_f32_safetensors_payload_as_f16_bits(&tensor.name, data)?,
        "F16" => decode_f16_safetensors_payload_bits(&tensor.name, data)?,
        other => {
            return Err(validate_error(format!(
                "Whisper local-source GGUF import runtime f16 tensor '{}' supports only F32/F16, got '{other}'",
                tensor.name
            )));
        }
    };
    let expected = tensor_element_count(&tensor.name, &tensor.shape)?;
    if values.len() != expected {
        return Err(validate_error(format!(
            "Whisper local-source GGUF import runtime f16 tensor '{}' decoded {} values but expected {}",
            tensor.name,
            values.len(),
            expected
        )));
    }
    Ok(GgufWriteTensor {
        name: tensor.name.clone(),
        dims: tensor.shape.clone(),
        tensor_type: GgufWriteTensorType::F16,
        data: encode_f16_bits_le(values),
    })
}

fn gguf_runtime_decoder_token_embedding_tensor_from_safetensors(
    tensor: &SafetensorsTensorHeaderV0,
    data: &[u8],
) -> Result<GgufWriteTensor, WhisperLocalSourceError> {
    let [vocab_dim, hidden_dim] = tensor.shape.as_slice() else {
        return Err(validate_error(format!(
            "Whisper local-source GGUF import decoder token embedding tensor '{}' must be rank-2, got {:?}",
            tensor.name, tensor.shape
        )));
    };
    let values = match tensor.dtype.as_str() {
        "F32" => decode_f32_safetensors_payload_as_f16_bits(&tensor.name, data)?,
        "F16" => decode_f16_safetensors_payload_bits(&tensor.name, data)?,
        other => {
            return Err(validate_error(format!(
                "Whisper local-source GGUF import decoder token embedding tensor '{}' supports only F32/F16, got '{other}'",
                tensor.name
            )));
        }
    };
    let expected = tensor_element_count(&tensor.name, &tensor.shape)?;
    if values.len() != expected {
        return Err(validate_error(format!(
            "Whisper local-source GGUF import decoder token embedding tensor '{}' decoded {} values but expected {}",
            tensor.name,
            values.len(),
            expected
        )));
    }

    Ok(GgufWriteTensor {
        name: tensor.name.clone(),
        // Safetensors [vocab, hidden] is row-major with hidden as the fastest
        // axis, which is already GGML's contiguous memory order for [hidden, vocab].
        dims: vec![*hidden_dim, *vocab_dim],
        tensor_type: GgufWriteTensorType::F16,
        data: encode_f16_bits_le(values),
    })
}

fn gguf_runtime_encoder_linear_tensor_from_safetensors(
    tensor: &SafetensorsTensorHeaderV0,
    data: &[u8],
) -> Result<GgufWriteTensor, WhisperLocalSourceError> {
    let [output_dim, input_dim] = tensor.shape.as_slice() else {
        return Err(validate_error(format!(
            "Whisper local-source GGUF import encoder linear tensor '{}' must be rank-2, got {:?}",
            tensor.name, tensor.shape
        )));
    };
    let values = match tensor.dtype.as_str() {
        "F32" => decode_f32_safetensors_payload_as_f16_bits(&tensor.name, data)?,
        "F16" => decode_f16_safetensors_payload_bits(&tensor.name, data)?,
        other => {
            return Err(validate_error(format!(
                "Whisper local-source GGUF import encoder linear tensor '{}' supports only F32/F16, got '{other}'",
                tensor.name
            )));
        }
    };
    let expected_u64 = (*input_dim).checked_mul(*output_dim).ok_or_else(|| {
        validate_error(format!(
            "Whisper local-source GGUF import encoder linear tensor '{}' shape {:?} overflows",
            tensor.name, tensor.shape
        ))
    })?;
    let expected = usize::try_from(expected_u64).map_err(|_| {
        validate_error(format!(
            "Whisper local-source GGUF import encoder linear tensor '{}' element count does not fit usize",
            tensor.name
        ))
    })?;
    if values.len() != expected {
        return Err(validate_error(format!(
            "Whisper local-source GGUF import encoder linear tensor '{}' decoded {} values but expected {}",
            tensor.name,
            values.len(),
            expected
        )));
    }
    Ok(GgufWriteTensor {
        name: tensor.name.clone(),
        dims: vec![*input_dim, *output_dim],
        tensor_type: GgufWriteTensorType::F16,
        data: encode_f16_bits_le(values),
    })
}

fn tensor_element_count(
    tensor_name: &str,
    shape: &[u64],
) -> Result<usize, WhisperLocalSourceError> {
    let count = shape.iter().try_fold(1_u64, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            validate_error(format!(
                "Whisper local-source GGUF import tensor '{tensor_name}' shape {shape:?} overflows"
            ))
        })
    })?;
    usize::try_from(count).map_err(|_| {
        validate_error(format!(
            "Whisper local-source GGUF import tensor '{tensor_name}' element count does not fit usize"
        ))
    })
}

fn encode_f16_bits_le(values: Vec<u16>) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(values.len().saturating_mul(2));
    for value in values {
        encoded.extend_from_slice(&value.to_le_bytes());
    }
    encoded
}

fn decode_f32_safetensors_payload_as_f16_bits(
    tensor_name: &str,
    data: &[u8],
) -> Result<Vec<u16>, WhisperLocalSourceError> {
    if !data.len().is_multiple_of(4) {
        return Err(validate_error(format!(
            "Whisper local-source GGUF import tensor '{tensor_name}' F32 payload byte length {} is not divisible by 4",
            data.len()
        )));
    }
    data.chunks_exact(4)
        .map(|chunk| {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if !value.is_finite() {
                return Err(validate_error(format!(
                    "Whisper local-source GGUF import tensor '{tensor_name}' contains non-finite F32 value"
                )));
            }
            Ok(f32_to_f16_bits(value))
        })
        .collect()
}

fn decode_f16_safetensors_payload_bits(
    tensor_name: &str,
    data: &[u8],
) -> Result<Vec<u16>, WhisperLocalSourceError> {
    if !data.len().is_multiple_of(2) {
        return Err(validate_error(format!(
            "Whisper local-source GGUF import tensor '{tensor_name}' F16 payload byte length {} is not divisible by 2",
            data.len()
        )));
    }
    Ok(data
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn decode_safetensors_payload_as_f32(
    tensor_name: &str,
    dtype: &str,
    data: &[u8],
) -> Result<Vec<f32>, WhisperLocalSourceError> {
    match dtype {
        "F32" => {
            if !data.len().is_multiple_of(4) {
                return Err(validate_error(format!(
                    "Whisper local-source GGUF import tensor '{tensor_name}' F32 payload byte length {} is not divisible by 4",
                    data.len()
                )));
            }
            data.chunks_exact(4)
                .map(|chunk| {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    if !value.is_finite() {
                        return Err(validate_error(format!(
                            "Whisper local-source GGUF import tensor '{tensor_name}' contains non-finite F32 value"
                        )));
                    }
                    Ok(value)
                })
                .collect()
        }
        "F16" => {
            let values = decode_f16_safetensors_payload_bits(tensor_name, data)?;
            Ok(values.into_iter().map(f16_bits_to_f32).collect())
        }
        other => Err(validate_error(format!(
            "Whisper local-source GGUF import tensor '{tensor_name}' quantization supports only F32/F16 source tensors, got '{other}'"
        ))),
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x03ff) as u32;
    let f32_bits = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut mant = mantissa;
            let mut exp = -1_i32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x03ff;
            let exponent_f32 = (127 - 15 + 1 + exp) as u32;
            sign | (exponent_f32 << 23) | (mant << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        let exponent_f32 = exponent + (127 - 15);
        sign | (exponent_f32 << 23) | (mantissa << 13)
    };
    f32::from_bits(f32_bits)
}

fn gguf_tensor_type_from_safetensors_dtype(
    name: &str,
    dtype: &str,
) -> Result<GgufWriteTensorType, WhisperLocalSourceError> {
    match dtype {
        "F32" => Ok(GgufWriteTensorType::F32),
        "F16" => Ok(GgufWriteTensorType::F16),
        other => Err(validate_error(format!(
            "Whisper local-source GGUF import supports only F32/F16 safetensors tensors this round; tensor '{name}' has dtype '{other}'"
        ))),
    }
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7fffff;
    if exponent == 255 {
        return sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exp = exponent - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x800000;
        let shift = 14 - half_exp;
        let mut half = (mantissa >> shift) as u16;
        if ((mantissa >> (shift - 1)) & 1) != 0 {
            half = half.saturating_add(1);
        }
        return sign | half;
    }
    let mut half = sign | ((half_exp as u16) << 10) | ((mantissa >> 13) as u16);
    if (mantissa & 0x1000) != 0 {
        half = half.saturating_add(1);
    }
    half
}

fn safetensors_payload_range(
    tensor_name: &str,
    header_length_bytes: u64,
    data_offsets: [u64; 2],
) -> Result<std::ops::Range<usize>, WhisperLocalSourceError> {
    let data_section_start = SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES
        .checked_add(header_length_bytes)
        .ok_or_else(|| {
            validate_error(format!(
                "Whisper local-source GGUF import safetensors header offset overflow for tensor '{tensor_name}'"
            ))
        })?;
    let start = data_section_start
        .checked_add(data_offsets[0])
        .ok_or_else(|| {
            validate_error(format!(
                "Whisper local-source GGUF import safetensors start offset overflow for tensor '{tensor_name}'"
            ))
        })?;
    let end = data_section_start
        .checked_add(data_offsets[1])
        .ok_or_else(|| {
            validate_error(format!(
                "Whisper local-source GGUF import safetensors end offset overflow for tensor '{tensor_name}'"
            ))
        })?;
    if end < start {
        return Err(validate_error(format!(
            "Whisper local-source GGUF import tensor '{tensor_name}' has inverted safetensors data offsets"
        )));
    }
    let start = usize::try_from(start).map_err(|_| {
        validate_error(format!(
            "Whisper local-source GGUF import tensor '{tensor_name}' start offset does not fit usize"
        ))
    })?;
    let end = usize::try_from(end).map_err(|_| {
        validate_error(format!(
            "Whisper local-source GGUF import tensor '{tensor_name}' end offset does not fit usize"
        ))
    })?;
    Ok(start..end)
}

fn validate_whisper_tokenizer_contract(
    tokenizer: &WhisperHfTokenizerImport,
    config: &WhisperConfigJson,
    safetensors_header: &SafetensorsHeaderV0,
) -> Result<(), WhisperLocalSourceError> {
    let tokenizer_vocab_size = u32::try_from(tokenizer.vocab_size()).map_err(|_| {
        validate_error(format!(
            "Whisper local-source GGUF tokenizer vocab size {} does not fit u32",
            tokenizer.vocab_size()
        ))
    })?;
    if tokenizer_vocab_size != config.vocab_size {
        return Err(validate_error(format!(
            "Whisper local-source GGUF tokenizer vocab size {} does not match config.vocab_size {}",
            tokenizer_vocab_size, config.vocab_size
        )));
    }

    let embedding = safetensors_header
        .tensors
        .iter()
        .find(|tensor| tensor.name == DECODER_TOKEN_EMBEDDING_TENSOR_NAME)
        .ok_or_else(|| {
            validate_error(format!(
                "Whisper local-source GGUF import requires tensor '{DECODER_TOKEN_EMBEDDING_TENSOR_NAME}' for tokenizer vocab validation"
            ))
        })?;
    if embedding.shape.len() != 2 {
        return Err(validate_error(format!(
            "Whisper local-source GGUF import tensor '{DECODER_TOKEN_EMBEDDING_TENSOR_NAME}' must be rank-2, found rank {}",
            embedding.shape.len()
        )));
    }
    let embedding_vocab = u32::try_from(embedding.shape[0]).map_err(|_| {
        validate_error(format!(
            "Whisper local-source GGUF import tensor '{DECODER_TOKEN_EMBEDDING_TENSOR_NAME}' vocab dim {} does not fit u32",
            embedding.shape[0]
        ))
    })?;
    let embedding_hidden = u32::try_from(embedding.shape[1]).map_err(|_| {
        validate_error(format!(
            "Whisper local-source GGUF import tensor '{DECODER_TOKEN_EMBEDDING_TENSOR_NAME}' hidden dim {} does not fit u32",
            embedding.shape[1]
        ))
    })?;
    if embedding_vocab != config.vocab_size {
        return Err(validate_error(format!(
            "Whisper local-source GGUF import tensor '{DECODER_TOKEN_EMBEDDING_TENSOR_NAME}' vocab dim {} does not match config.vocab_size {}",
            embedding_vocab, config.vocab_size
        )));
    }
    if embedding_hidden != config.d_model {
        return Err(validate_error(format!(
            "Whisper local-source GGUF import tensor '{DECODER_TOKEN_EMBEDDING_TENSOR_NAME}' hidden dim {} does not match config.d_model {}",
            embedding_hidden, config.d_model
        )));
    }
    if embedding_vocab != tokenizer_vocab_size {
        return Err(validate_error(format!(
            "Whisper local-source GGUF tokenizer vocab size {} does not match tensor '{}' vocab dim {}",
            tokenizer_vocab_size, DECODER_TOKEN_EMBEDDING_TENSOR_NAME, embedding_vocab
        )));
    }
    Ok(())
}

fn whisper_runtime_gguf_metadata(
    request: &WhisperLocalSourceImportRequest,
    config: &WhisperConfigJson,
    tokenizer: &WhisperHfTokenizerImport,
    model_id: &str,
) -> BTreeMap<String, GgufWriteValue> {
    let mut metadata = BTreeMap::new();
    let n_mels = whisper_num_mels(config);

    insert_metadata(&mut metadata, OPENASR_MODEL_ID_KEY, model_id);
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_PACKAGE_VERSION,
        OASR_PACKAGE_VERSION_V1,
    );
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_MODEL_FAMILY,
        WHISPER_MODEL_FAMILY,
    );
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        WHISPER_GGML_ARCHITECTURE_ID,
    );
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_AUDIO_FRONTEND,
        WHISPER_AUDIO_FRONTEND_ID,
    );
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_DECODE_POLICY,
        WHISPER_DECODE_POLICY_ID,
    );
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_FEATURE_STREAMING,
        OASR_FEATURE_STREAMING_GGML_TRUE_STREAMING_V1,
    );
    insert_metadata(&mut metadata, GGML_TOKENIZER_ID_KEY, WHISPER_TOKENIZER_ID);
    insert_metadata(
        &mut metadata,
        GENERAL_ARCHITECTURE_KEY,
        GGUF_WHISPER_ARCHITECTURE_VALUE,
    );

    insert_metadata(
        &mut metadata,
        WHISPER_ENCODER_BLOCK_COUNT_KEY,
        config.encoder_layers,
    );
    insert_metadata(
        &mut metadata,
        WHISPER_DECODER_BLOCK_COUNT_KEY,
        config.decoder_layers,
    );
    insert_metadata(
        &mut metadata,
        WHISPER_ENCODER_CONTEXT_LENGTH_KEY,
        config.max_source_positions,
    );
    insert_metadata(
        &mut metadata,
        WHISPER_ENCODER_EMBEDDING_LENGTH_KEY,
        config.d_model,
    );
    insert_metadata(
        &mut metadata,
        WHISPER_ENCODER_HEAD_COUNT_KEY,
        config.encoder_attention_heads,
    );
    insert_metadata(
        &mut metadata,
        WHISPER_DECODER_EMBEDDING_LENGTH_KEY,
        config.d_model,
    );
    insert_metadata(&mut metadata, WHISPER_ENCODER_MELS_COUNT_KEY, n_mels);
    insert_metadata(&mut metadata, WHISPER_VOCAB_SIZE_KEY, config.vocab_size);
    insert_metadata(
        &mut metadata,
        WHISPER_DECODER_HEAD_COUNT_KEY,
        whisper_decoder_attention_heads(config),
    );
    insert_metadata(
        &mut metadata,
        WHISPER_DECODER_CONTEXT_LENGTH_KEY,
        config.max_target_positions,
    );
    insert_metadata(
        &mut metadata,
        TOKENIZER_GGML_MODEL_KEY,
        TOKENIZER_GGML_MODEL_VALUE_GPT2,
    );
    insert_metadata_string_array(&mut metadata, TOKENIZER_GGML_TOKENS_KEY, &tokenizer.tokens);
    insert_metadata_string_array(&mut metadata, TOKENIZER_GGML_MERGES_KEY, &tokenizer.merges);
    insert_metadata_u32_array(
        &mut metadata,
        TOKENIZER_GGML_SPECIAL_TOKEN_IDS_KEY,
        &tokenizer.special_token_ids,
    );
    insert_metadata_u32(
        &mut metadata,
        TOKENIZER_GGML_SOT_TOKEN_ID_KEY,
        tokenizer.sot_token_id,
    );
    insert_metadata_u32(
        &mut metadata,
        TOKENIZER_GGML_EOT_TOKEN_ID_KEY,
        tokenizer.eot_token_id,
    );
    insert_metadata_u32(
        &mut metadata,
        TOKENIZER_GGML_TRANSCRIBE_TOKEN_ID_KEY,
        tokenizer.transcribe_token_id,
    );
    insert_metadata_u32(
        &mut metadata,
        TOKENIZER_GGML_NO_TIMESTAMPS_TOKEN_ID_KEY,
        tokenizer.no_timestamps_token_id,
    );

    insert_metadata(&mut metadata, "openasr.source.name", &request.source_name);
    insert_metadata(
        &mut metadata,
        "openasr.source.revision",
        &request.source_revision,
    );
    insert_metadata(&mut metadata, "openasr.license.name", &request.license_name);
    insert_metadata(
        &mut metadata,
        "openasr.license.source",
        &request.license_source,
    );

    metadata
}

fn insert_metadata(
    metadata: &mut BTreeMap<String, GgufWriteValue>,
    key: &str,
    value: impl ToString,
) {
    metadata.insert(key.to_string(), GgufWriteValue::String(value.to_string()));
}

fn insert_metadata_u32(metadata: &mut BTreeMap<String, GgufWriteValue>, key: &str, value: u32) {
    metadata.insert(key.to_string(), GgufWriteValue::U32(value));
}

fn insert_metadata_string_array(
    metadata: &mut BTreeMap<String, GgufWriteValue>,
    key: &str,
    values: &[String],
) {
    metadata.insert(
        key.to_string(),
        GgufWriteValue::StringArray(values.to_vec()),
    );
}

fn insert_metadata_u32_array(
    metadata: &mut BTreeMap<String, GgufWriteValue>,
    key: &str,
    values: &[u32],
) {
    metadata.insert(key.to_string(), GgufWriteValue::U32Array(values.to_vec()));
}

fn whisper_gguf_tensor_binding_context(
    config: &WhisperConfigJson,
) -> Result<WhisperGgufTensorBindingContext, WhisperLocalSourceError> {
    Ok(WhisperGgufTensorBindingContext {
        n_audio_layer: usize_from_config_u32("encoder_layers", config.encoder_layers)?,
        n_audio_state: usize_from_config_u32("d_model", config.d_model)?,
        n_audio_head: usize_from_config_u32(
            "encoder_attention_heads",
            config.encoder_attention_heads,
        )?,
        n_mels: usize_from_config_u32("num_mel_bins", whisper_num_mels(config))?,
        n_audio_ctx: usize_from_config_u32("max_source_positions", config.max_source_positions)?,
        n_text_layer: usize_from_config_u32("decoder_layers", config.decoder_layers)?,
        n_text_state: usize_from_config_u32("d_model", config.d_model)?,
        n_text_head: usize_from_config_u32(
            "decoder_attention_heads",
            whisper_decoder_attention_heads(config),
        )?,
        n_text_ctx: usize_from_config_u32("max_target_positions", config.max_target_positions)?,
        n_vocab: usize_from_config_u32("vocab_size", config.vocab_size)?,
    })
}

fn usize_from_config_u32(field: &str, value: u32) -> Result<usize, WhisperLocalSourceError> {
    usize::try_from(value).map_err(|_| {
        validate_error(format!(
            "Whisper local-source converter config.{field} does not fit usize"
        ))
    })
}

fn compose_model_id(package_id: &str, package_variant: Option<&str>) -> String {
    match package_variant
        .map(str::trim)
        .filter(|variant| !variant.is_empty())
    {
        Some(variant) => format!("{}:{variant}", package_id.trim()),
        None => package_id.trim().to_string(),
    }
}

fn whisper_num_mels(config: &WhisperConfigJson) -> u32 {
    config.num_mel_bins.unwrap_or(80)
}

fn whisper_decoder_attention_heads(config: &WhisperConfigJson) -> u32 {
    config
        .decoder_attention_heads
        .unwrap_or(config.encoder_attention_heads)
}

fn validate_request(
    request: &WhisperLocalSourceImportRequest,
) -> Result<(), WhisperLocalSourceError> {
    if request.package_id.trim().is_empty() {
        return Err(validate_error(
            "Whisper local-source converter requires non-empty package_id".to_string(),
        ));
    }
    if request.model_language.trim().is_empty() {
        return Err(validate_error(
            "Whisper local-source converter requires non-empty model_language".to_string(),
        ));
    }
    if !request.source_root.is_dir() {
        return Err(validate_error(format!(
            "Whisper local-source converter source root '{}' is not a directory",
            request.source_root.display()
        )));
    }
    if request.output_root.exists() {
        return Err(validate_error(format!(
            "Whisper local-source converter output root '{}' already exists",
            request.output_root.display()
        )));
    }
    // Same user-facing `.oasr`-only output contract as the CLI + the other
    // converters (the on-disk container stays GGUF-structured internally).
    if !crate::has_openasr_runtime_pack_extension(&request.output_root) {
        return Err(validate_error(format!(
            "Whisper local-source converter output '{}' must end with .oasr (OpenASR native runtime pack)",
            request.output_root.display()
        )));
    }
    Ok(())
}

fn validate_whisper_config(config: &WhisperConfigJson) -> Result<(), WhisperLocalSourceError> {
    if !config.model_type.eq_ignore_ascii_case("whisper") {
        return Err(validate_error(format!(
            "Whisper local-source converter expected config.model_type whisper, got '{}'",
            config.model_type
        )));
    }
    if config.vocab_size == 0 {
        return Err(validate_error(
            "Whisper local-source converter requires config.vocab_size > 0".to_string(),
        ));
    }
    if config.d_model == 0 {
        return Err(validate_error(
            "Whisper local-source converter requires config.d_model > 0".to_string(),
        ));
    }
    if config.encoder_layers == 0 {
        return Err(validate_error(
            "Whisper local-source converter requires config.encoder_layers > 0".to_string(),
        ));
    }
    if config.decoder_layers == 0 {
        return Err(validate_error(
            "Whisper local-source converter requires config.decoder_layers > 0".to_string(),
        ));
    }
    if config.encoder_attention_heads == 0 {
        return Err(validate_error(
            "Whisper local-source converter requires config.encoder_attention_heads > 0"
                .to_string(),
        ));
    }
    if !config
        .d_model
        .is_multiple_of(config.encoder_attention_heads)
    {
        return Err(validate_error(format!(
            "Whisper local-source converter requires config.d_model ({}) to be divisible by config.encoder_attention_heads ({})",
            config.d_model, config.encoder_attention_heads
        )));
    }
    let decoder_attention_heads = whisper_decoder_attention_heads(config);
    if decoder_attention_heads == 0 {
        return Err(validate_error(
            "Whisper local-source converter requires config.decoder_attention_heads > 0"
                .to_string(),
        ));
    }
    if !config.d_model.is_multiple_of(decoder_attention_heads) {
        return Err(validate_error(format!(
            "Whisper local-source converter requires config.d_model ({}) to be divisible by config.decoder_attention_heads ({})",
            config.d_model, decoder_attention_heads
        )));
    }
    if config.max_source_positions == 0 {
        return Err(validate_error(
            "Whisper local-source converter requires config.max_source_positions > 0".to_string(),
        ));
    }
    if config.max_target_positions == 0 {
        return Err(validate_error(
            "Whisper local-source converter requires config.max_target_positions > 0".to_string(),
        ));
    }
    if whisper_num_mels(config) == 0 {
        return Err(validate_error(
            "Whisper local-source converter requires config.num_mel_bins > 0".to_string(),
        ));
    }
    for binding in &config.forced_decoder_ids {
        if binding.len() != 2 {
            return Err(validate_error(format!(
                "Whisper local-source converter requires forced_decoder_ids entries of length 2, found {binding:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_request() -> WhisperLocalSourceImportRequest {
        WhisperLocalSourceImportRequest {
            source_root: PathBuf::from("source"),
            output_root: PathBuf::from("out.oasr"),
            package_id: "whisper-tiny.en-local".to_string(),
            package_variant: Some("hf".to_string()),
            model_language: "en".to_string(),
            source_name: "openai/whisper-tiny.en".to_string(),
            source_revision: "local-test".to_string(),
            license_name: "MIT".to_string(),
            license_source: "license".to_string(),
            quantization: WhisperRuntimeQuantizationMode::Fp16,
        }
    }

    fn fixture_config() -> WhisperConfigJson {
        WhisperConfigJson {
            model_type: "whisper".to_string(),
            vocab_size: 51_864,
            d_model: 384,
            encoder_layers: 4,
            decoder_layers: 4,
            encoder_attention_heads: 6,
            decoder_attention_heads: Some(6),
            max_source_positions: 1_500,
            max_target_positions: 448,
            num_mel_bins: Some(80),
            forced_decoder_ids: vec![vec![1, 50_362]],
        }
    }

    fn fixture_tokenizer() -> WhisperHfTokenizerImport {
        WhisperHfTokenizerImport {
            tokens: vec![
                "<|endoftext|>".to_string(),
                "<|startoftranscript|>".to_string(),
                "<|transcribe|>".to_string(),
                "<|notimestamps|>".to_string(),
            ],
            merges: vec!["a b".to_string(), "b c".to_string()],
            special_token_ids: vec![0, 1, 2, 3],
            sot_token_id: 1,
            eot_token_id: 0,
            transcribe_token_id: 2,
            no_timestamps_token_id: 3,
        }
    }

    fn string_metadata<'a>(metadata: &'a BTreeMap<String, GgufWriteValue>, key: &str) -> &'a str {
        match metadata.get(key).expect("metadata key should be present") {
            GgufWriteValue::String(value) => value,
            other => panic!("expected string metadata for '{key}', got {other:?}"),
        }
    }

    #[test]
    fn compose_model_id_uses_variant_when_present() {
        assert_eq!(
            compose_model_id("whisper-tiny.en-local", Some("hf")),
            "whisper-tiny.en-local:hf"
        );
        assert_eq!(
            compose_model_id("whisper-tiny.en-local", Some("  ")),
            "whisper-tiny.en-local"
        );
    }

    #[test]
    fn whisper_runtime_metadata_contains_oasr_and_executor_keys() {
        let request = fixture_request();
        let config = fixture_config();
        let tokenizer = fixture_tokenizer();
        let metadata = whisper_runtime_gguf_metadata(
            &request,
            &config,
            &tokenizer,
            "whisper-tiny.en-local:hf",
        );

        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_PACKAGE_VERSION),
            OASR_PACKAGE_VERSION_V1
        );
        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_MODEL_FAMILY),
            WHISPER_MODEL_FAMILY
        );
        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_MODEL_ARCHITECTURE),
            WHISPER_GGML_ARCHITECTURE_ID
        );
        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_AUDIO_FRONTEND),
            WHISPER_AUDIO_FRONTEND_ID
        );
        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_DECODE_POLICY),
            WHISPER_DECODE_POLICY_ID
        );
        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_FEATURE_STREAMING),
            OASR_FEATURE_STREAMING_GGML_TRUE_STREAMING_V1
        );
        assert_eq!(
            string_metadata(&metadata, GGML_TOKENIZER_ID_KEY),
            WHISPER_TOKENIZER_ID
        );
        assert_eq!(
            string_metadata(&metadata, GENERAL_ARCHITECTURE_KEY),
            GGUF_WHISPER_ARCHITECTURE_VALUE
        );
        assert_eq!(
            string_metadata(&metadata, WHISPER_ENCODER_BLOCK_COUNT_KEY),
            "4"
        );
        assert_eq!(
            string_metadata(&metadata, WHISPER_DECODER_BLOCK_COUNT_KEY),
            "4"
        );
        assert_eq!(
            string_metadata(&metadata, WHISPER_DECODER_HEAD_COUNT_KEY),
            "6"
        );
        assert_eq!(string_metadata(&metadata, WHISPER_VOCAB_SIZE_KEY), "51864");
        assert_eq!(
            string_metadata(&metadata, WHISPER_ENCODER_MELS_COUNT_KEY),
            "80"
        );
    }

    #[test]
    fn whisper_runtime_metadata_writes_tokenizer_kv_contract() {
        let request = fixture_request();
        let config = fixture_config();
        let tokenizer = fixture_tokenizer();
        let metadata = whisper_runtime_gguf_metadata(
            &request,
            &config,
            &tokenizer,
            "whisper-tiny.en-local:hf",
        );

        assert_eq!(
            string_metadata(&metadata, GGML_TOKENIZER_ID_KEY),
            WHISPER_TOKENIZER_ID
        );
        assert_eq!(
            string_metadata(&metadata, WHISPER_VOCAB_SIZE_KEY),
            config.vocab_size.to_string()
        );
        assert_eq!(
            metadata.get(TOKENIZER_GGML_MODEL_KEY),
            Some(&GgufWriteValue::String("gpt2".to_string()))
        );
        assert_eq!(
            metadata.get(TOKENIZER_GGML_SOT_TOKEN_ID_KEY),
            Some(&GgufWriteValue::U32(1))
        );
        assert_eq!(
            metadata.get(TOKENIZER_GGML_SPECIAL_TOKEN_IDS_KEY),
            Some(&GgufWriteValue::U32Array(vec![0, 1, 2, 3]))
        );
    }

    #[test]
    fn encoder_linear_import_writes_runtime_ready_f16_input_output_layout() {
        let tensor = SafetensorsTensorHeaderV0 {
            name: "model.encoder.layers.0.fc1.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3],
            data_offsets: [0, 24],
        };
        let values = [1.0_f32, -2.0, 3.5, 4.0, 5.0, 6.0];
        let data = values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();

        let gguf_tensor = gguf_tensor_from_safetensors_tensor(
            &tensor,
            &data,
            WhisperRuntimeQuantizationMode::Fp16,
        )
        .expect("tensor should import");

        assert_eq!(gguf_tensor.name, tensor.name);
        assert_eq!(gguf_tensor.dims, vec![3, 2]);
        assert_eq!(gguf_tensor.tensor_type, GgufWriteTensorType::F16);
        assert_eq!(gguf_tensor.data.len(), 12);
        assert_eq!(
            u16::from_le_bytes([gguf_tensor.data[0], gguf_tensor.data[1]]),
            0x3c00
        );
    }

    #[test]
    fn runtime_f16_weight_import_preserves_shape_and_halves_payload() {
        let tensor = SafetensorsTensorHeaderV0 {
            name: "model.decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 2],
            data_offsets: [0, 16],
        };
        let data = [1.0_f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();

        let gguf_tensor = gguf_tensor_from_safetensors_tensor(
            &tensor,
            &data,
            WhisperRuntimeQuantizationMode::Fp16,
        )
        .expect("tensor should import");

        assert_eq!(gguf_tensor.name, tensor.name);
        assert_eq!(gguf_tensor.dims, tensor.shape);
        assert_eq!(gguf_tensor.tensor_type, GgufWriteTensorType::F16);
        assert_eq!(gguf_tensor.data.len(), 8);
        assert_eq!(
            u16::from_le_bytes([gguf_tensor.data[0], gguf_tensor.data[1]]),
            0x3c00
        );
    }

    #[test]
    fn decoder_token_embedding_import_writes_runtime_ready_hidden_vocab_layout() {
        let tensor = SafetensorsTensorHeaderV0 {
            name: DECODER_TOKEN_EMBEDDING_TENSOR_NAME.to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3],
            data_offsets: [0, 24],
        };
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();

        let gguf_tensor = gguf_tensor_from_safetensors_tensor(
            &tensor,
            &data,
            WhisperRuntimeQuantizationMode::Fp16,
        )
        .expect("tensor should import");

        assert_eq!(gguf_tensor.name, tensor.name);
        assert_eq!(gguf_tensor.dims, vec![3, 2]);
        assert_eq!(gguf_tensor.tensor_type, GgufWriteTensorType::F16);
        assert_eq!(gguf_tensor.data.len(), 12);
        assert_eq!(
            u16::from_le_bytes([gguf_tensor.data[0], gguf_tensor.data[1]]),
            0x3c00
        );
    }

    #[test]
    fn quantized_encoder_linear_import_q8_0_writes_q8_payload() {
        let tensor = SafetensorsTensorHeaderV0 {
            name: "model.encoder.layers.0.fc1.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 256],
            data_offsets: [0, 2 * 256 * 4],
        };
        let data = (0..(2 * 256))
            .map(|index| (index as f32) * 0.001 - 0.25)
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();

        let gguf_tensor = gguf_tensor_from_safetensors_tensor(
            &tensor,
            &data,
            WhisperRuntimeQuantizationMode::Q8_0,
        )
        .expect("q8_0 tensor should import");

        assert_eq!(gguf_tensor.dims, vec![256, 2]);
        assert_eq!(gguf_tensor.tensor_type, GgufWriteTensorType::Q8_0);
        assert!(gguf_tensor.data.len() < (256 * 2 * 2));
    }

    #[test]
    fn quantized_encoder_linear_import_q4_k_is_smaller_than_q8_0() {
        let tensor = SafetensorsTensorHeaderV0 {
            name: "model.encoder.layers.0.fc1.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 256],
            data_offsets: [0, 2 * 256 * 4],
        };
        let data = (0..(2 * 256))
            .map(|index| (index as f32) * 0.002 - 0.5)
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();

        let q8 = gguf_tensor_from_safetensors_tensor(
            &tensor,
            &data,
            WhisperRuntimeQuantizationMode::Q8_0,
        )
        .expect("q8_0 tensor should import");
        let q4 = gguf_tensor_from_safetensors_tensor(
            &tensor,
            &data,
            WhisperRuntimeQuantizationMode::Q4_K,
        )
        .expect("q4_k tensor should import");

        assert_eq!(q8.tensor_type, GgufWriteTensorType::Q8_0);
        assert_eq!(q4.tensor_type, GgufWriteTensorType::Q4_K);
        assert!(q4.data.len() < q8.data.len());
        assert!(q8.data.len() < (256 * 2 * 2));
    }

    #[test]
    fn quantized_decoder_linear_import_q8_0_writes_q8_payload() {
        let tensor = SafetensorsTensorHeaderV0 {
            name: "model.decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 256],
            data_offsets: [0, 2 * 256 * 4],
        };
        let data = (0..(2 * 256))
            .map(|index| (index as f32) * 0.001 - 0.25)
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();

        let gguf_tensor = gguf_tensor_from_safetensors_tensor(
            &tensor,
            &data,
            WhisperRuntimeQuantizationMode::Q8_0,
        )
        .expect("q8_0 decoder tensor should import");

        assert_eq!(gguf_tensor.dims, vec![256, 2]);
        assert_eq!(gguf_tensor.tensor_type, GgufWriteTensorType::Q8_0);
        assert!(gguf_tensor.data.len() < (256 * 2 * 2));
    }
}
