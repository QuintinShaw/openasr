use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::Deserialize;

use crate::ggml_runtime::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, quantize_f32_to_ggml_tensor_data,
    read_gguf_tensor_index, validate_ggml_runtime_source_path, write_gguf_file_v0,
};
use crate::models::ggml_family_adapter::GGML_TOKENIZER_ID_KEY;
use crate::models::ggml_family_registry::{
    COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID, COHERE_TRANSCRIBE_DECODE_POLICY_ID,
    COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID, COHERE_TRANSCRIBE_TOKENIZER_ID,
};
use crate::models::local_source_import::{
    LocalSourceImportError, SafetensorsFile, decode_safetensors_payload_as_f16_bits,
    decode_safetensors_payload_as_f32, encode_f16_bits_le, read_source_json_file,
    tensor_element_count, validate_error, validate_output_pack_extension,
};
use crate::models::oasr_metadata::{
    OASR_FEATURE_STREAMING_GGML_TRUE_STREAMING_V1, OASR_METADATA_KEY_AUDIO_FRONTEND,
    OASR_METADATA_KEY_DECODE_POLICY, OASR_METADATA_KEY_FEATURE_STREAMING,
    OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
    OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
};
use crate::models::runtime_tensor_contract_registry::validate_builtin_runtime_tensor_contract_for_architecture;
use crate::models::{cohere::COHERE_TRANSCRIBE_MODEL_FAMILY, cohere::runtime_contract};
use crate::{read_gguf_metadata_from_runtime_source, read_gguf_tensor_index_from_runtime_source};

use super::tensor_names::{
    DEC_EMB_LN_BIAS, DEC_EMB_LN_WEIGHT, DEC_EMB_WEIGHT, DEC_HEAD_BIAS, DEC_HEAD_WEIGHT,
    DEC_OUT_LN_BIAS, DEC_OUT_LN_WEIGHT, DEC_POS_WEIGHT, ENC_PRE_OUT_BIAS, ENC_PRE_OUT_WEIGHT,
    ENC_PROJ_BIAS, ENC_PROJ_WEIGHT, FE_MEL_FB, FE_WINDOW, decoder_layer_tensor_names,
    encoder_layer_tensor_names,
};

const SOURCE_CONFIG_JSON: &str = "config.json";
const SOURCE_GENERATION_CONFIG_JSON: &str = "generation_config.json";
const SOURCE_TOKENIZER_JSON: &str = "tokenizer.json";
const SOURCE_MODEL_SAFETENSORS: &str = "model.safetensors";
const TOKENIZER_GGML_MODEL_KEY: &str = "tokenizer.ggml.model";
const TOKENIZER_GGML_MODEL_VALUE_LLAMA: &str = "llama";
const TOKENIZER_GGML_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const OPENASR_MODEL_ID_KEY: &str = "openasr.model.id";
const GENERAL_ARCHITECTURE_KEY: &str = "general.architecture";
const COHERE_ARCHITECTURE_VALUE: &str = "cohere-transcribe";

pub type CohereLocalSourceError = LocalSourceImportError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CohereLocalSourceImportRequest {
    pub source_root: PathBuf,
    pub output_root: PathBuf,
    pub package_id: String,
    pub package_variant: Option<String>,
    pub source_name: String,
    pub source_revision: String,
    pub license_name: String,
    pub license_source: String,
    pub quantization: CohereRuntimeQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CohereLocalSourceImportRuntimeResult {
    pub output_path: PathBuf,
    pub model_id: String,
    pub tensor_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(non_camel_case_types)]
pub enum CohereRuntimeQuantizationMode {
    #[default]
    Fp16,
    Q8_0,
    Q4_K,
}

#[derive(Debug, Deserialize)]
struct CohereConfigJson {
    #[serde(default)]
    vocab_size: Option<usize>,
    encoder: CohereEncoderConfigJson,
    preprocessor: CoherePreprocessorConfigJson,
    transf_decoder: CohereDecoderRootConfigJson,
}

#[derive(Debug, Deserialize)]
struct CohereEncoderConfigJson {
    n_layers: usize,
    d_model: usize,
    n_heads: usize,
    #[serde(default)]
    ff_expansion_factor: Option<usize>,
    #[serde(default)]
    conv_kernel_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct CoherePreprocessorConfigJson {
    sample_rate: u32,
    features: usize,
    n_fft: usize,
    window_size: f32,
    window_stride: f32,
}

#[derive(Debug, Deserialize)]
struct CohereDecoderRootConfigJson {
    config_dict: CohereDecoderConfigJson,
}

#[derive(Debug, Deserialize)]
struct CohereDecoderConfigJson {
    num_layers: usize,
    hidden_size: usize,
    num_attention_heads: usize,
    max_sequence_length: usize,
    #[serde(default)]
    inner_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct CohereGenerationConfigJson {
    #[serde(default)]
    decoder_start_token_id: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct TokenizerJson {
    model: TokenizerModelJson,
}

#[derive(Debug, Deserialize)]
struct TokenizerModelJson {
    vocab: BTreeMap<String, usize>,
}

pub fn convert_local_cohere_source_to_runtime_pack(
    request: &CohereLocalSourceImportRequest,
) -> Result<CohereLocalSourceImportRuntimeResult, CohereLocalSourceError> {
    validate_request(request)?;
    let config: CohereConfigJson = read_source_json_file(&request.source_root, SOURCE_CONFIG_JSON)?;
    let generation_config: CohereGenerationConfigJson =
        read_source_json_file(&request.source_root, SOURCE_GENERATION_CONFIG_JSON)?;
    let tokenizer: TokenizerJson =
        read_source_json_file(&request.source_root, SOURCE_TOKENIZER_JSON)?;
    let model_path = request.source_root.join(SOURCE_MODEL_SAFETENSORS);
    let safetensors = SafetensorsFile::open(&model_path)?;
    let vocab_tokens = build_vocab_tokens(&tokenizer.model.vocab)?;

    let metadata_fields = cohere_metadata_fields(
        &config,
        &generation_config,
        &safetensors,
        request,
        &vocab_tokens,
    )?;
    let tensors = build_cohere_runtime_tensors(&safetensors, request.quantization)?;
    let model_id = compose_model_id(&request.package_id, request.package_variant.as_deref());
    let metadata =
        cohere_runtime_gguf_metadata(&metadata_fields, request, &model_id, &vocab_tokens);

    write_gguf_file_v0(&request.output_root, &metadata, &tensors).map_err(|error| {
        validate_error(format!(
            "Cohere local-source GGUF writer failed for '{}': {error}",
            request.output_root.display()
        ))
    })?;

    // Fail closed: verify runtime contract via preflight reader/index validation.
    let runtime_source =
        validate_ggml_runtime_source_path(&request.output_root).map_err(|error| {
            validate_error(format!(
                "Cohere local-source import produced invalid runtime path '{}': {error}",
                request.output_root.display()
            ))
        })?;
    let metadata_read =
        read_gguf_metadata_from_runtime_source(&runtime_source).map_err(|error| {
            validate_error(format!(
                "Cohere import produced unreadable GGUF metadata: {error}"
            ))
        })?;
    let tensor_index =
        read_gguf_tensor_index_from_runtime_source(&runtime_source).map_err(|error| {
            validate_error(format!(
                "Cohere import produced unreadable GGUF tensor index: {error}"
            ))
        })?;
    validate_builtin_runtime_tensor_contract_for_architecture(
        COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
        &metadata_read,
        &tensor_index,
    )
    .map_err(|error| {
        validate_error(format!(
            "Cohere runtime contract validation failed: {error}"
        ))
    })?;

    let index = read_gguf_tensor_index(&request.output_root).map_err(|error| {
        validate_error(format!(
            "Cohere local-source GGUF writer produced unreadable tensor index: {error}"
        ))
    })?;
    Ok(CohereLocalSourceImportRuntimeResult {
        output_path: request.output_root.clone(),
        model_id,
        tensor_count: index.tensors().len(),
    })
}

fn build_cohere_runtime_tensors(
    safetensors: &SafetensorsFile,
    quantization: CohereRuntimeQuantizationMode,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut names = BTreeSet::new();
    let mut out = Vec::new();
    for tensor in &safetensors.header().tensors {
        if let Some((target_name, target_dims, force_f32)) =
            remap_cohere_tensor_name_and_dims(tensor.name.as_str(), tensor.shape.as_slice())?
        {
            if !names.insert(target_name.clone()) {
                return Err(validate_error(format!(
                    "Cohere import mapped duplicate destination tensor '{target_name}'"
                )));
            }
            let data = safetensors.tensor_data(tensor)?;
            let target_dims =
                normalize_cohere_weight_dims(&target_name, tensor.shape.as_slice(), target_dims)?;
            let tensor_type = quantized_tensor_type_for_cohere_tensor(
                &target_name,
                &target_dims,
                force_f32,
                quantization,
            );
            let write_tensor = match tensor_type {
                Some(qtype) => {
                    let values =
                        decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
                    let expected = tensor_element_count(&tensor.name, &target_dims)?;
                    if values.len() != expected {
                        return Err(validate_error(format!(
                            "Cohere tensor '{}' decoded {} values but expected {} for dims {:?}",
                            tensor.name,
                            values.len(),
                            expected,
                            target_dims
                        )));
                    }
                    let quantized =
                        quantize_f32_to_ggml_tensor_data(qtype, &target_dims, &values).map_err(
                            |error| {
                                validate_error(format!(
                                    "Cohere quantization failed for tensor '{}' -> '{}' ({qtype:?}): {error}",
                                    tensor.name, target_name
                                ))
                            },
                        )?;
                    GgufWriteTensor {
                        name: target_name,
                        dims: target_dims,
                        tensor_type: qtype,
                        data: quantized,
                    }
                }
                None if force_f32 => {
                    let values =
                        decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
                    let expected = tensor_element_count(&tensor.name, &target_dims)?;
                    if values.len() != expected {
                        return Err(validate_error(format!(
                            "Cohere tensor '{}' decoded {} values but expected {} for dims {:?}",
                            tensor.name,
                            values.len(),
                            expected,
                            target_dims
                        )));
                    }
                    let mut bytes = Vec::with_capacity(values.len() * 4);
                    for value in values {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                    GgufWriteTensor {
                        name: target_name,
                        dims: target_dims,
                        tensor_type: GgufWriteTensorType::F32,
                        data: bytes,
                    }
                }
                None => {
                    let bits =
                        decode_safetensors_payload_as_f16_bits(&tensor.name, &tensor.dtype, data)?;
                    GgufWriteTensor {
                        name: target_name,
                        dims: target_dims,
                        tensor_type: GgufWriteTensorType::F16,
                        data: encode_f16_bits_le(bits),
                    }
                }
            };
            out.push(write_tensor);
        }
    }
    Ok(out)
}

fn remap_cohere_tensor_name_and_dims(
    source_name: &str,
    source_shape: &[u64],
) -> Result<Option<(String, Vec<u64>, bool)>, LocalSourceImportError> {
    if source_name == "preprocessor.featurizer.fb" {
        // Source layout: [1, n_mels, n_fft_bins], runtime expects dims [n_fft_bins, n_mels]
        if source_shape.len() != 3 || source_shape[0] != 1 {
            return Err(validate_error(format!(
                "Cohere featurizer.fb expected shape [1, n_mels, n_fft_bins], got {:?}",
                source_shape
            )));
        }
        return Ok(Some((
            FE_MEL_FB.to_string(),
            vec![source_shape[2], source_shape[1]],
            false,
        )));
    }
    if source_name == "preprocessor.featurizer.window" {
        return Ok(Some((FE_WINDOW.to_string(), source_shape.to_vec(), true)));
    }
    if let Some(suffix) = source_name.strip_prefix("encoder.pre_encode.conv.") {
        return Ok(Some((
            format!("enc.pre.conv.{suffix}"),
            source_shape.to_vec(),
            false,
        )));
    }
    if source_name == "encoder.pre_encode.out.weight" {
        return Ok(Some((
            ENC_PRE_OUT_WEIGHT.to_string(),
            source_shape.to_vec(),
            false,
        )));
    }
    if source_name == "encoder.pre_encode.out.bias" {
        return Ok(Some((
            ENC_PRE_OUT_BIAS.to_string(),
            source_shape.to_vec(),
            true,
        )));
    }
    if source_name == "encoder_decoder_proj.weight" {
        return Ok(Some((
            ENC_PROJ_WEIGHT.to_string(),
            source_shape.to_vec(),
            false,
        )));
    }
    if source_name == "encoder_decoder_proj.bias" {
        return Ok(Some((
            ENC_PROJ_BIAS.to_string(),
            source_shape.to_vec(),
            true,
        )));
    }
    if source_name == "transf_decoder._embedding.token_embedding.weight" {
        return Ok(Some((
            DEC_EMB_WEIGHT.to_string(),
            source_shape.to_vec(),
            false,
        )));
    }
    if source_name == "transf_decoder._embedding.position_embedding.pos_enc" {
        return Ok(Some((
            DEC_POS_WEIGHT.to_string(),
            source_shape.to_vec(),
            false,
        )));
    }
    if source_name == "transf_decoder._embedding.layer_norm.weight" {
        return Ok(Some((
            DEC_EMB_LN_WEIGHT.to_string(),
            source_shape.to_vec(),
            true,
        )));
    }
    if source_name == "transf_decoder._embedding.layer_norm.bias" {
        return Ok(Some((
            DEC_EMB_LN_BIAS.to_string(),
            source_shape.to_vec(),
            true,
        )));
    }
    if source_name == "transf_decoder._decoder.final_layer_norm.weight" {
        return Ok(Some((
            DEC_OUT_LN_WEIGHT.to_string(),
            source_shape.to_vec(),
            true,
        )));
    }
    if source_name == "transf_decoder._decoder.final_layer_norm.bias" {
        return Ok(Some((
            DEC_OUT_LN_BIAS.to_string(),
            source_shape.to_vec(),
            true,
        )));
    }
    if source_name == "log_softmax.mlp.layer0.weight" {
        return Ok(Some((
            DEC_HEAD_WEIGHT.to_string(),
            source_shape.to_vec(),
            false,
        )));
    }
    if source_name == "log_softmax.mlp.layer0.bias" {
        return Ok(Some((
            DEC_HEAD_BIAS.to_string(),
            source_shape.to_vec(),
            true,
        )));
    }

    if let Some(rest) = source_name.strip_prefix("encoder.layers.") {
        let (layer_idx, tail) = split_layer_index(rest)?;
        let names = encoder_layer_tensor_names(layer_idx);
        let mapped = match tail {
            "norm_feed_forward1.weight" => names.ff1_norm_weight,
            "norm_feed_forward1.bias" => names.ff1_norm_bias,
            "feed_forward1.linear1.weight" => names.ff1_up_weight,
            "feed_forward1.linear1.bias" => names.ff1_up_bias,
            "feed_forward1.linear2.weight" => names.ff1_down_weight,
            "feed_forward1.linear2.bias" => names.ff1_down_bias,
            "norm_self_att.weight" => names.attn_norm_weight,
            "norm_self_att.bias" => names.attn_norm_bias,
            "self_attn.linear_q.weight" => names.attn_q_weight,
            "self_attn.linear_q.bias" => names.attn_q_bias,
            "self_attn.linear_k.weight" => names.attn_k_weight,
            "self_attn.linear_k.bias" => names.attn_k_bias,
            "self_attn.linear_v.weight" => names.attn_v_weight,
            "self_attn.linear_v.bias" => names.attn_v_bias,
            "self_attn.linear_out.weight" => names.attn_out_weight,
            "self_attn.linear_out.bias" => names.attn_out_bias,
            "self_attn.linear_pos.weight" => names.attn_pos_weight,
            "self_attn.pos_bias_u" => names.attn_pos_bias_u,
            "self_attn.pos_bias_v" => names.attn_pos_bias_v,
            "norm_conv.weight" => names.conv_norm_weight,
            "norm_conv.bias" => names.conv_norm_bias,
            "conv.pointwise_conv1.weight" => names.conv_pw1_weight,
            "conv.pointwise_conv1.bias" => names.conv_pw1_bias,
            "conv.depthwise_conv.weight" => names.conv_dw_weight,
            "conv.depthwise_conv.bias" => names.conv_dw_bias,
            "conv.batch_norm.weight" => names.conv_bn_weight,
            "conv.batch_norm.bias" => names.conv_bn_bias,
            "conv.batch_norm.running_mean" => names.conv_bn_mean,
            "conv.batch_norm.running_var" => names.conv_bn_var,
            "conv.pointwise_conv2.weight" => names.conv_pw2_weight,
            "conv.pointwise_conv2.bias" => names.conv_pw2_bias,
            "norm_feed_forward2.weight" => names.ff2_norm_weight,
            "norm_feed_forward2.bias" => names.ff2_norm_bias,
            "feed_forward2.linear1.weight" => names.ff2_up_weight,
            "feed_forward2.linear1.bias" => names.ff2_up_bias,
            "feed_forward2.linear2.weight" => names.ff2_down_weight,
            "feed_forward2.linear2.bias" => names.ff2_down_bias,
            "norm_out.weight" => names.out_norm_weight,
            "norm_out.bias" => names.out_norm_bias,
            "conv.batch_norm.num_batches_tracked" => return Ok(None),
            _ => return Ok(None),
        };
        let force_f32 = is_cohere_f32_tensor(&mapped);
        return Ok(Some((mapped, source_shape.to_vec(), force_f32)));
    }

    if let Some(rest) = source_name.strip_prefix("transf_decoder._decoder.layers.") {
        let (layer_idx, tail) = split_layer_index(rest)?;
        let names = decoder_layer_tensor_names(layer_idx);
        let mapped = match tail {
            "layer_norm_1.weight" => names.attn_ln_weight,
            "layer_norm_1.bias" => names.attn_ln_bias,
            "first_sub_layer.query_net.weight" => names.attn_q_weight,
            "first_sub_layer.query_net.bias" => names.attn_q_bias,
            "first_sub_layer.key_net.weight" => names.attn_k_weight,
            "first_sub_layer.key_net.bias" => names.attn_k_bias,
            "first_sub_layer.value_net.weight" => names.attn_v_weight,
            "first_sub_layer.value_net.bias" => names.attn_v_bias,
            "first_sub_layer.out_projection.weight" => names.attn_o_weight,
            "first_sub_layer.out_projection.bias" => names.attn_o_bias,
            "layer_norm_2.weight" => names.cross_ln_weight,
            "layer_norm_2.bias" => names.cross_ln_bias,
            "second_sub_layer.query_net.weight" => names.cross_q_weight,
            "second_sub_layer.query_net.bias" => names.cross_q_bias,
            "second_sub_layer.key_net.weight" => names.cross_k_weight,
            "second_sub_layer.key_net.bias" => names.cross_k_bias,
            "second_sub_layer.value_net.weight" => names.cross_v_weight,
            "second_sub_layer.value_net.bias" => names.cross_v_bias,
            "second_sub_layer.out_projection.weight" => names.cross_o_weight,
            "second_sub_layer.out_projection.bias" => names.cross_o_bias,
            "layer_norm_3.weight" => names.ffn_ln_weight,
            "layer_norm_3.bias" => names.ffn_ln_bias,
            "third_sub_layer.dense_in.weight" => names.ffn_up_weight,
            "third_sub_layer.dense_in.bias" => names.ffn_up_bias,
            "third_sub_layer.dense_out.weight" => names.ffn_down_weight,
            "third_sub_layer.dense_out.bias" => names.ffn_down_bias,
            _ => return Ok(None),
        };
        return Ok(Some((
            mapped.clone(),
            source_shape.to_vec(),
            is_cohere_f32_tensor(&mapped),
        )));
    }

    Ok(None)
}

fn is_cohere_f32_tensor(name: &str) -> bool {
    name == FE_WINDOW
        || name.ends_with(".bias")
        || name.contains(".norm.")
        || name.contains("_ln.")
        || name.contains("_norm.")
        || name.contains(".ln.")
        || name.contains(".bn.")
        || name.contains("pos_bias")
}

fn normalize_cohere_weight_dims(
    target_name: &str,
    source_shape: &[u64],
    fallback_dims: Vec<u64>,
) -> Result<Vec<u64>, LocalSourceImportError> {
    if !should_reverse_cohere_tensor_dims(target_name) || source_shape.len() < 2 {
        return Ok(fallback_dims);
    }
    let mut dims = source_shape.to_vec();
    dims.reverse();
    Ok(dims)
}

fn should_reverse_cohere_tensor_dims(target_name: &str) -> bool {
    if matches!(target_name, DEC_EMB_WEIGHT | DEC_POS_WEIGHT) {
        return false;
    }
    target_name.ends_with(".weight") || target_name.contains("pos_bias")
}

fn quantized_tensor_type_for_cohere_tensor(
    name: &str,
    dims: &[u64],
    force_f32: bool,
    quantization: CohereRuntimeQuantizationMode,
) -> Option<GgufWriteTensorType> {
    if force_f32 || quantization == CohereRuntimeQuantizationMode::Fp16 {
        return None;
    }
    if !name.ends_with(".weight") || dims.len() != 2 {
        return None;
    }
    let ne0 = dims.first().copied()?;
    if !ne0.is_multiple_of(32_u64) {
        return None;
    }
    if quantization == CohereRuntimeQuantizationMode::Q4_K && ne0.is_multiple_of(256_u64) {
        return Some(GgufWriteTensorType::Q4_K);
    }
    Some(GgufWriteTensorType::Q8_0)
}

#[derive(Debug, Clone)]
struct CohereMetadataFields {
    vocab_size: usize,
    encoder_layers: usize,
    encoder_d_model: usize,
    encoder_heads: usize,
    encoder_head_dim: usize,
    encoder_ffn_dim: usize,
    encoder_conv_kernel: usize,
    decoder_layers: usize,
    decoder_d_model: usize,
    decoder_heads: usize,
    decoder_head_dim: usize,
    decoder_ffn_dim: usize,
    decoder_max_context: usize,
    decoder_start_token_id: usize,
    sample_rate_hz: u32,
    n_mels: usize,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
}

fn cohere_metadata_fields(
    config: &CohereConfigJson,
    generation_config: &CohereGenerationConfigJson,
    safetensors: &SafetensorsFile,
    request: &CohereLocalSourceImportRequest,
    vocab_tokens: &[String],
) -> Result<CohereMetadataFields, LocalSourceImportError> {
    let token_embedding = safetensors
        .tensor("transf_decoder._embedding.token_embedding.weight")
        .ok_or_else(|| validate_error("missing token embedding tensor"))?;
    let pos_embedding = safetensors
        .tensor("transf_decoder._embedding.position_embedding.pos_enc")
        .ok_or_else(|| validate_error("missing positional embedding tensor"))?;
    if token_embedding.shape.len() != 2 || pos_embedding.shape.len() != 2 {
        return Err(validate_error(
            "Cohere token/position embeddings must be rank-2 tensors",
        ));
    }

    let vocab_size = config
        .vocab_size
        .unwrap_or(token_embedding.shape[0] as usize);
    if vocab_size != token_embedding.shape[0] as usize {
        return Err(validate_error(format!(
            "config.vocab_size={} does not match token embedding rows={}",
            vocab_size, token_embedding.shape[0]
        )));
    }
    if vocab_tokens.len() < vocab_size {
        return Err(validate_error(format!(
            "tokenizer vocab has {} entries but model requires {vocab_size}",
            vocab_tokens.len()
        )));
    }
    let encoder_layers = config.encoder.n_layers;
    let encoder_d_model = config.encoder.d_model;
    let encoder_heads = config.encoder.n_heads;
    let encoder_head_dim = encoder_d_model
        .checked_div(encoder_heads)
        .ok_or_else(|| validate_error("encoder head_dim division overflow"))?;
    let encoder_ffn_dim = config
        .encoder
        .ff_expansion_factor
        .unwrap_or(4)
        .checked_mul(encoder_d_model)
        .ok_or_else(|| validate_error("encoder ffn dimension overflow"))?;
    let encoder_conv_kernel = config.encoder.conv_kernel_size.unwrap_or(9);

    let decoder_layers = config.transf_decoder.config_dict.num_layers;
    let decoder_d_model = config.transf_decoder.config_dict.hidden_size;
    let decoder_heads = config.transf_decoder.config_dict.num_attention_heads;
    let decoder_head_dim = decoder_d_model
        .checked_div(decoder_heads)
        .ok_or_else(|| validate_error("decoder head_dim division overflow"))?;
    let decoder_ffn_dim = config
        .transf_decoder
        .config_dict
        .inner_size
        .unwrap_or(decoder_d_model.saturating_mul(4));
    let decoder_max_context = config.transf_decoder.config_dict.max_sequence_length;
    let decoder_start_token_id = generation_config
        .decoder_start_token_id
        .ok_or_else(|| validate_error("generation_config.decoder_start_token_id is required"))?;

    let sample_rate_hz = config.preprocessor.sample_rate;
    let n_mels = config.preprocessor.features;
    let n_fft = config.preprocessor.n_fft;
    let hop_length = (config.preprocessor.window_stride * sample_rate_hz as f32).round() as usize;
    let win_length = (config.preprocessor.window_size * sample_rate_hz as f32).round() as usize;

    if request.package_id.trim().is_empty() {
        return Err(validate_error(
            "Cohere local-source converter requires non-empty package_id",
        ));
    }
    Ok(CohereMetadataFields {
        vocab_size,
        encoder_layers,
        encoder_d_model,
        encoder_heads,
        encoder_head_dim,
        encoder_ffn_dim,
        encoder_conv_kernel,
        decoder_layers,
        decoder_d_model,
        decoder_heads,
        decoder_head_dim,
        decoder_ffn_dim,
        decoder_max_context,
        decoder_start_token_id,
        sample_rate_hz,
        n_mels,
        n_fft,
        hop_length,
        win_length,
    })
}

fn cohere_runtime_gguf_metadata(
    fields: &CohereMetadataFields,
    request: &CohereLocalSourceImportRequest,
    model_id: &str,
    vocab_tokens: &[String],
) -> BTreeMap<String, GgufWriteValue> {
    let mut metadata = BTreeMap::new();
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_PACKAGE_VERSION,
        OASR_PACKAGE_VERSION_V1,
    );
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_MODEL_FAMILY,
        COHERE_TRANSCRIBE_MODEL_FAMILY,
    );
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
    );
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_AUDIO_FRONTEND,
        COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID,
    );
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_DECODE_POLICY,
        COHERE_TRANSCRIBE_DECODE_POLICY_ID,
    );
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_FEATURE_STREAMING,
        OASR_FEATURE_STREAMING_GGML_TRUE_STREAMING_V1,
    );
    insert_metadata(
        &mut metadata,
        GGML_TOKENIZER_ID_KEY,
        COHERE_TRANSCRIBE_TOKENIZER_ID,
    );
    insert_metadata(&mut metadata, OPENASR_MODEL_ID_KEY, model_id);
    insert_metadata(
        &mut metadata,
        GENERAL_ARCHITECTURE_KEY,
        COHERE_ARCHITECTURE_VALUE,
    );

    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_VOCAB_SIZE_KEY,
        fields.vocab_size as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_ENCODER_LAYERS_KEY,
        fields.encoder_layers as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_ENCODER_D_MODEL_KEY,
        fields.encoder_d_model as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_ENCODER_HEADS_KEY,
        fields.encoder_heads as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_ENCODER_HEAD_DIM_KEY,
        fields.encoder_head_dim as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_ENCODER_FFN_DIM_KEY,
        fields.encoder_ffn_dim as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_ENCODER_CONV_KERNEL_KEY,
        fields.encoder_conv_kernel as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_DECODER_LAYERS_KEY,
        fields.decoder_layers as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_DECODER_D_MODEL_KEY,
        fields.decoder_d_model as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_DECODER_HEADS_KEY,
        fields.decoder_heads as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_DECODER_HEAD_DIM_KEY,
        fields.decoder_head_dim as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_DECODER_FFN_DIM_KEY,
        fields.decoder_ffn_dim as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_DECODER_MAX_CONTEXT_KEY,
        fields.decoder_max_context as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_DECODER_START_TOKEN_ID_KEY,
        fields.decoder_start_token_id as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_AUDIO_SAMPLE_RATE_KEY,
        fields.sample_rate_hz,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_AUDIO_MELS_COUNT_KEY,
        fields.n_mels as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_AUDIO_N_FFT_KEY,
        fields.n_fft as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_AUDIO_HOP_LENGTH_KEY,
        fields.hop_length as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::COHERE_TRANSCRIBE_AUDIO_WIN_LENGTH_KEY,
        fields.win_length as u32,
    );

    insert_metadata(
        &mut metadata,
        TOKENIZER_GGML_MODEL_KEY,
        TOKENIZER_GGML_MODEL_VALUE_LLAMA,
    );
    insert_metadata_string_array(&mut metadata, TOKENIZER_GGML_TOKENS_KEY, vocab_tokens);

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

fn build_vocab_tokens(
    vocab_map: &BTreeMap<String, usize>,
) -> Result<Vec<String>, LocalSourceImportError> {
    if vocab_map.is_empty() {
        return Err(validate_error(
            "cohere tokenizer.model.vocab cannot be empty",
        ));
    }
    let max_id = vocab_map.values().copied().max().ok_or_else(|| {
        validate_error("cohere tokenizer.model.vocab cannot determine max token id")
    })?;
    let mut tokens = vec![String::new(); max_id + 1];
    for (token, token_id) in vocab_map {
        if tokens[*token_id].is_empty() {
            tokens[*token_id] = token.clone();
        } else {
            return Err(validate_error(format!(
                "cohere tokenizer has duplicate token id {token_id}"
            )));
        }
    }
    for (token_id, token) in tokens.iter_mut().enumerate() {
        if token.is_empty() {
            *token = format!("<unused_{token_id}>");
        }
    }
    Ok(tokens)
}

fn split_layer_index(value: &str) -> Result<(usize, &str), LocalSourceImportError> {
    let (layer_str, tail) = value
        .split_once('.')
        .ok_or_else(|| validate_error(format!("invalid layer tensor suffix '{value}'")))?;
    let layer_idx = layer_str.parse::<usize>().map_err(|error| {
        validate_error(format!(
            "invalid numeric layer index in tensor suffix '{value}': {error}"
        ))
    })?;
    Ok((layer_idx, tail))
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

fn validate_request(
    request: &CohereLocalSourceImportRequest,
) -> Result<(), LocalSourceImportError> {
    if request.package_id.trim().is_empty() {
        return Err(validate_error(
            "Cohere local-source converter requires non-empty package_id",
        ));
    }
    if request.source_name.trim().is_empty() {
        return Err(validate_error(
            "Cohere local-source converter requires non-empty source_name",
        ));
    }
    if request.source_revision.trim().is_empty() {
        return Err(validate_error(
            "Cohere local-source converter requires non-empty source_revision",
        ));
    }
    if request.license_name.trim().is_empty() {
        return Err(validate_error(
            "Cohere local-source converter requires non-empty license_name",
        ));
    }
    if request.license_source.trim().is_empty() {
        return Err(validate_error(
            "Cohere local-source converter requires non-empty license_source",
        ));
    }
    validate_output_pack_extension(&request.output_root)?;
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_request() -> CohereLocalSourceImportRequest {
        CohereLocalSourceImportRequest {
            source_root: PathBuf::from("/tmp/openasr/cohere-source"),
            output_root: PathBuf::from("/tmp/openasr/cohere-runtime.gguf"),
            package_id: "cohere-transcribe-fixture".to_string(),
            package_variant: Some("q8_0".to_string()),
            source_name: "CohereLabs/cohere-transcribe".to_string(),
            source_revision: "fixture-revision".to_string(),
            license_name: "cc-by-nc".to_string(),
            license_source: "https://example.invalid/license".to_string(),
            quantization: CohereRuntimeQuantizationMode::Q8_0,
        }
    }

    fn fixture_metadata_fields() -> CohereMetadataFields {
        CohereMetadataFields {
            vocab_size: 4,
            encoder_layers: 1,
            encoder_d_model: 16,
            encoder_heads: 2,
            encoder_head_dim: 8,
            encoder_ffn_dim: 64,
            encoder_conv_kernel: 9,
            decoder_layers: 1,
            decoder_d_model: 16,
            decoder_heads: 2,
            decoder_head_dim: 8,
            decoder_ffn_dim: 64,
            decoder_max_context: 128,
            decoder_start_token_id: 1,
            sample_rate_hz: 16_000,
            n_mels: 80,
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
        }
    }

    fn string_metadata<'a>(metadata: &'a BTreeMap<String, GgufWriteValue>, key: &str) -> &'a str {
        match metadata.get(key) {
            Some(GgufWriteValue::String(value)) => value,
            other => panic!("expected string metadata for {key}, got {other:?}"),
        }
    }

    #[test]
    fn cohere_runtime_metadata_declares_snapshot_streaming_feature() {
        let tokens = vec![
            "<pad>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "fixture".to_string(),
        ];
        let request = fixture_request();
        let metadata = cohere_runtime_gguf_metadata(
            &fixture_metadata_fields(),
            &request,
            "cohere-transcribe-fixture:q8_0",
            &tokens,
        );

        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_FEATURE_STREAMING),
            OASR_FEATURE_STREAMING_GGML_TRUE_STREAMING_V1
        );
        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_MODEL_FAMILY),
            COHERE_TRANSCRIBE_MODEL_FAMILY
        );
        assert_eq!(
            string_metadata(&metadata, GGML_TOKENIZER_ID_KEY),
            COHERE_TRANSCRIBE_TOKENIZER_ID
        );
    }
}
